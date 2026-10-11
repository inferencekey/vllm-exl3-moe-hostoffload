"""The quality proxy makes gen_prefix tasks continue the assistant message on both backends."""
import importlib.util

import pytest

from hostoffload.cost import plan_budget

pytest.importorskip("aiohttp")
spec = importlib.util.spec_from_file_location("api_proxy", plan_budget.ROOT / "quality" / "api_proxy.py")
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)

USER = {"role": "user", "content": "Complete the function"}
PREFIX = {"role": "assistant", "content": "Here is the completed function:\n```python\ndef f():\n"}


def test_deepseek_prefix_continuation():
    body = {"model": "x", "messages": [USER, dict(PREFIX)]}
    assert proxy.rewrite(body, "deepseek", None, "") == proxy.API_PREFIX
    assert body["messages"][-1]["prefix"] is True
    assert body["model"] == "deepseek-flash" and body["thinking"] == {"type": "disabled"}


def test_deepseek_plain_chat_uses_the_normal_endpoint():
    body = {"messages": [USER]}
    assert proxy.rewrite(body, "deepseek", None, "") == proxy.API
    assert "prefix" not in body["messages"][-1]


def test_vllm_continuation_uses_rendered_prompt(monkeypatch):
    rendered = "<bos><User>Complete<Assistant></think>Here is the completed function:\n```python\ndef f():\n"
    monkeypatch.setattr(proxy, "render_continuation", lambda msgs: rendered)
    body = {"model": "deepseek-v4.1-flash", "messages": [USER, dict(PREFIX)], "max_tokens": 64}
    assert proxy.rewrite(body, "vllm", None, "http://u/v1/chat/completions") == "http://u/v1/completions"
    assert body["prompt"] == rendered and "messages" not in body and body["max_tokens"] == 64
    plain = {"messages": [USER]}
    assert proxy.rewrite(plain, "vllm", None, "http://u/v1/chat/completions") == "http://u/v1/chat/completions"
    assert plain["chat_template_kwargs"]["thinking"] is False and "prompt" not in plain


def test_completion_response_becomes_chat_shape():
    d = proxy.as_chat({"choices": [{"index": 0, "text": "    return 1\n"}]})
    assert d["choices"][0]["message"] == {"role": "assistant", "content": "    return 1\n"}
