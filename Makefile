# One-command entry points. Workstation targets need no GPU; pod targets run on a prepared GPU pod.
# `make help` lists them.
SHELL := /bin/bash
PY ?= .venv/bin/python
UV ?= uv
export PYTHONPATH := $(CURDIR)/src
STRICT ?= 0

.PHONY: help env audit static static-hub static-weights stack-fetch stack-check analyze figures paper test plan \
        readmes lint release-check release clean pod-bootstrap download docker-build \
        bench-E1 bench-E2 bench-E3 bench-E4 bench-E5 bench-E6 bench-E7 bench-E8 bench-E9 quality-E10 quality-E11

help:  ## list targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | sort

env:  ## create .venv from the hash-pinned requirements.lock (Python 3.12)
	$(UV) venv -p 3.12 .venv && $(UV) pip install -p .venv/bin/python -r requirements.lock && \
	$(UV) pip install -p .venv/bin/python --no-deps -e .

# ---------------------------------------------------------------- workstation (no GPU, no spend)
audit:  ## attribution audit: fetch pinned upstreams outside the repo, check patches, write audit/evidence
	bash audit/scripts/fetch_sources.sh
	bash audit/scripts/check_patches.sh

static: static-hub static-weights  ## E0: static measurements of public files

static-hub:  ## Hub metadata snapshots
	$(PY) -m hostoffload.weights.hub_meta --out data/static/hub

static-weights: stack-fetch  ## tensor tables and placement from safetensors headers (range requests, no weights)
	$(PY) -m hostoffload.weights.split --out data/static/weights/diffbot-a1cf1bd2 \
	  --plan third_party/diffbot-recipe/plans/plan-agent30-dh.json \
	  --plan third_party/diffbot-recipe/plans/plan-gen30-dh.json \
	  --plan third_party/diffbot-recipe/plans/plan-blend30-dh.json
	$(PY) -m hostoffload.weights.split --repo deepseek-ai/DeepSeek-V4.1-Flash \
	  --revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027 --out data/static/weights/deepseek-ai-2cba9e42

stack-fetch:  ## fetch and verify every third-party input of the stack into third_party/
	@[ -f third_party/FETCHED.txt ] || bash stack/fetch_stack.sh

stack-check: stack-fetch  ## dry-run every patch against its pinned upstream (no GPU)
	git -C third_party/vllm-exl3 apply --check ../diffbot-recipe/patches/vllm-exl3/exl3-thor.patch && echo "exl3-thor.patch applies"
	cd third_party/exllamav3/exllamav3/exllamav3_ext && patch -p1 --dry-run -s < ../../../diffbot-recipe/kernels/exllamav3-5be8865-exl3_moe.patch >/dev/null && echo "exl3_moe patch applies" || \
	  (cd $(CURDIR)/third_party/exllamav3 && patch -p1 --dry-run -s < ../diffbot-recipe/kernels/exllamav3-5be8865-exl3_moe.patch >/dev/null && echo "exl3_moe patch applies")

analyze: stack-fetch  ## data/ -> paper/generated (values, tables, provenance) and figures
	$(PY) -m hostoffload.analysis.pipeline $(if $(filter 1,$(STRICT)),--strict,)

figures: analyze paper/figures/generated/stack_provenance.pdf  ## diagrams (graphviz) in addition to the pipeline's figures

# graphviz PDFs embed a unique id, so the diagram is rebuilt only when its source changes
paper/figures/generated/stack_provenance.pdf: diagrams/stack_provenance.dot
	dot -Tpdf $< -o $@

paper: figures  ## build paper/build/main.pdf (STRICT=1: fail on any missing value)
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build \
	  $(if $(filter 1,$(STRICT)),-usepretex='\def\strictbuild{}',) main.tex
	@grep -c "DATAMACROS NOT RUN" paper/build/main.log | xargs -I{} echo "NOT RUN markers in the PDF: {}"

test:  ## pytest: harness, parsers, cost model, statistics, no-fake-data guard, manifests
	$(PY) -m pytest

plan: readmes  ## regenerate experiments/PLAN.md (budget) from experiments.yaml and pricing.yaml
	$(PY) experiments/tools/make_plan_md.py

readmes:  ## create 'not run yet' READMEs for result folders without data
	$(PY) experiments/tools/make_result_readmes.py

release-check:  ## pre-release scan: em dashes, out-of-scope terms, identifiers, unlicensed files
	$(PY) scripts/release_check.py

release: release-check test paper  ## build the release tarball in dist/ (does NOT publish, tag or archive)
	bash scripts/make_release.sh

clean:
	rm -rf paper/build .pytest_cache

# ---------------------------------------------------------------- pod side (GPU, costs money: budget approval first)
VARIANT ?= exl3-offload
MODEL ?= exl3
pod-bootstrap: stack-fetch  ## build VARIANT in place inside the pinned vLLM image
	VARIANT=$(VARIANT) bash stack/build/bootstrap_pod.sh

docker-build: stack-fetch  ## build hostoffload:VARIANT with docker (no push)
	docker build -f stack/build/Dockerfile --build-arg VARIANT=$(VARIANT) -t hostoffload:$(VARIANT) .

download:  ## pinned weights to /root/models (MODEL=exl3|orig); never into the repository
	MODEL=$(MODEL) python3 experiments/tools/download.py

bench-E1:  ## build every variant with docker, record digests (CPU machine)
	bash experiments/runs/E1.sh
bench-E2:  ## feasibility boots
	bash experiments/runs/E2.sh
bench-E3:  ## host-share sweep and upstream UVA at equal bytes
	bash experiments/runs/E3.sh
bench-E4:  ## placement policy
	bash experiments/runs/E4.sh
bench-E5:  ## Engram int4 vs FP8
	bash experiments/runs/E5.sh
bench-E6:  ## side-stream SM budget
	bash experiments/runs/E6.sh
bench-E7:  ## DSpark on vs off
	bash experiments/runs/E7.sh
bench-E8:  ## main benchmark (MINIMAL=1 for the minimal subset)
	bash experiments/runs/E8.sh
bench-E9:  ## upstream all-in-VRAM reference on 2x B300
	bash experiments/runs/E9.sh
quality-E10:  ## lm-eval on the measured stack (MINIMAL=1: GSM8K, IFEval, HumanEval)
	bash experiments/runs/E10.sh
quality-E11:  ## lm-eval on the reference
	bash experiments/runs/E11.sh
