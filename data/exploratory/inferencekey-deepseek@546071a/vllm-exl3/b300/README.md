# vllm-exl3 on one B300: stopped, the pod never started (2026-10-06)

Goal of this track: DeepSeek-V4.1-Flash with vllm-exl3 on one B300 288 GB. Stopped by the user after
three pod attempts with no measurement, because the 2x RTX PRO 6000 run (../a100/) already answered
the question (usable decode, seconds of prefill, prefix cache) at 4.18 USD/h.

What happened:
- Wheels for exllamav3 1.4.9 and vllm_exl3 0.4.2 (cp312) were built on a cheap CPU pod while waiting
  for B300 capacity (`wheels/`, `pod_install.sh`).
- Attempt 1 (EUR-IS-1, 10:26 UTC): pod rented but the container never started (runtime null,
  uptime 0); deleted after 22 min (`logs/created_attempt1_stuck.txt`).
- Attempt 2 (EU-NL-1, 10:49 UTC): same symptom, deleted by a 10 min cutoff.
- Attempt 3 (8obc30jjcya44j): created by the agent shortly before the stop, deleted by the manager.

Likely cause: the pod image was `vllm/vllm-openai:deepseekv41-flash-0909` with a custom entrypoint
that installs openssh at boot (`logs/created.txt`). On a fresh B300 host the large image pull did not
finish; with no sshd from a RunPod image there was nothing to connect to. Next time use a RunPod
image (has sshd) and install vLLM on top, or check the image is cached on the host.

Spend: about 4-5 USD of pod time with no results (two B300s that never started plus a few cents of
CPU pod), against a 16 USD cap.
