# Reproducible image for one variant of the stack. Build context: the repository root, after `make stack-fetch`
# has populated third_party/ (verified against stack/recipe.lock.tsv and the commit pins in stack/pins.env).
#
#   docker build -f stack/build/Dockerfile --build-arg VARIANT=exl3-offload -t hostoffload:exl3-offload .
#
# The base is pinned by digest. The resulting image contains AGPL-3.0 code (vllm-exl3) for the exl3-* variants:
# distributing it, or serving a modified version over a network, carries AGPL-3.0 obligations (see THIRD_PARTY.md).
# The image is never pushed by this repository.
ARG BASE=vllm/vllm-openai:nightly-36768d1bfd39094681cdbc8cb37d4b31c0729c89@sha256:0ad6fa4178c34f4e04402452bbc4294df6dd2fb2785a0ecd402b2861aea2a189
FROM ${BASE}
ARG VARIANT=exl3-offload
ARG MAX_JOBS=16
WORKDIR /opt/hostoffload
COPY stack/ stack/
COPY third_party/ third_party/
RUN VARIANT=${VARIANT} MAX_JOBS=${MAX_JOBS} OUT=/opt/hostoffload/build bash stack/build/bootstrap_pod.sh \
 && rm -rf /var/lib/apt/lists/*
ENV HOB_VARIANT=${VARIANT}
ENTRYPOINT ["bash", "/opt/hostoffload/stack/serve.sh"]
