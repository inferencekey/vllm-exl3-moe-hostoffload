#!/usr/bin/env python3
"""Local proxy between lm-eval and a chat endpoint, used for every lean-plan quality run (E10 and E11-api).

Tasks such as humaneval_instruct end the conversation with a partial assistant message (lm-eval's gen_prefix) that
the model must CONTINUE. Plain chat completion opens a new assistant turn instead, the answer restarts the code block
and every item fails. lm-eval cannot ask for continuation itself, so the proxy does it, the same way for both backends:

- deepseek: forwards to https://api.deepseek.com/beta/chat/completions when the last message is from the assistant
  (chat prefix completion, `"prefix": true` on that message), else to /chat/completions; sets the model to
  `deepseek-flash` (served by DeepSeek-V4.1-Flash per data/static/pricing/deepseek-api-pricing-2026-10-06.html) and
  disables thinking with {"thinking": {"type": "disabled"}} (data/static/pricing/deepseek-api-thinking-mode-2026-10-06.html)
- vllm: forwards to the local server (--upstream, bearer $API_KEY) with thinking off via chat_template_kwargs; when
  the last message is from the assistant it renders the DeepSeek V4.1 prompt with vLLM's own encoder, drops the
  end-of-sentence token that encoder appends to that message (vLLM ignores continue_final_message for this
  tokenizer), and continues it through /v1/completions
- appends every response's usage (prompt, cached and completion tokens) to <usage.jsonl>, from which
  hostoffload.cost computes the real API cost

  DEEPSEEK_API_KEY=... python3 quality/api_proxy.py --usage <run_dir>/api_usage.jsonl &
  CHAT_URL=http://127.0.0.1:8100/v1/chat/completions MODEL_NAME=deepseek-flash bash quality/run_lm_eval.sh <run_dir>
The deepseek backend costs money (a few tenths of a USD for the lean subsets); run only with an approved budget.
"""
import argparse
import json
import os
import time

from aiohttp import ClientSession, ClientTimeout, web

API = "https://api.deepseek.com/chat/completions"
API_PREFIX = "https://api.deepseek.com/beta/chat/completions"


def rewrite(body: dict, backend: str, model: str | None, upstream: str) -> str:
    """Rewrite an lm-eval request in place for the backend and return the URL to forward it to."""
    cont = bool(body.get("messages")) and body["messages"][-1].get("role") == "assistant"
    body["stream"] = False
    if model or backend == "deepseek":
        body["model"] = model or "deepseek-flash"
    if backend == "deepseek":
        body["thinking"] = {"type": "disabled"}
        if cont:
            body["messages"][-1]["prefix"] = True
        return API_PREFIX if cont else API
    body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), "thinking": False}
    if cont:
        # vLLM's DeepSeek V4.1 tokenizer ignores continue_final_message and closes the last assistant message with
        # the end-of-sentence token, so the model would generate after EOS. Render the prompt with the same encoder,
        # drop that token, and continue it as a plain completion.
        msgs = body.pop("messages")
        body.pop("chat_template_kwargs")
        body["prompt"] = render_continuation(msgs)
        return upstream.replace("/chat/completions", "/completions")
    return upstream


EOS = "<\uff5cend\u2581of\u2581sentence\uff5c>"


def render_continuation(messages: list[dict]) -> str:
    """DeepSeek V4.1 chat prompt (thinking off) ending inside the last assistant message, ready to be continued."""
    from vllm.tokenizers.deepseek_v41_encoding import encode_messages
    prompt = encode_messages(messages, thinking_mode="chat", drop_thinking=True, reasoning_effort="high")
    if not prompt.endswith(EOS):
        raise ValueError("unexpected DeepSeek V4.1 encoding: the last assistant message does not end with EOS")
    return prompt[: -len(EOS)]


def as_chat(data: dict) -> dict:
    """Turn a /v1/completions response into the chat shape lm-eval reads (choices[i].message.content)."""
    for c in data.get("choices", []):
        if "text" in c and "message" not in c:
            c["message"] = {"role": "assistant", "content": c.pop("text")}
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", required=True)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--backend", choices=["deepseek", "vllm"], default="deepseek")
    ap.add_argument("--upstream", default="http://127.0.0.1:8000/v1/chat/completions", help="vllm backend only")
    ap.add_argument("--model", default=None, help="model name sent upstream (deepseek default: deepseek-flash)")
    a = ap.parse_args()
    key = os.environ["DEEPSEEK_API_KEY"] if a.backend == "deepseek" else os.environ.get("API_KEY", "none")

    async def handle(req: web.Request) -> web.Response:
        body = await req.json()
        url = rewrite(body, a.backend, a.model, a.upstream)
        async with ClientSession(timeout=ClientTimeout(total=600)) as s:
            async with s.post(url, json=body, headers={"Authorization": f"Bearer {key}"}) as r:
                text = await r.text()
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:  # upstream sent no JSON (gateway error): pass it on, lm-eval retries
                    data = {"error": {"message": text[:500]}}
                if url.endswith("/completions") and not url.endswith("/chat/completions"):
                    data = as_chat(data)
                with open(a.usage, "a") as f:
                    f.write(json.dumps({"t": time.time(), "status": r.status, "url": url,
                                        "usage": data.get("usage")}) + "\n")
                return web.json_response(data, status=r.status)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    web.run_app(app, host="127.0.0.1", port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
