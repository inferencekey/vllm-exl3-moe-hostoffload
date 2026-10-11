# Literature search and verification protocol

Date of all searches and fetches: 2026-10-06 (single session).
Outputs: `../references.bib`, `verification.tsv` (one row per included key), `excluded.tsv` (dropped candidates and metadata decisions), `../sections/13_related.tex`.

This is a targeted, structured review for a related-work section, not a full systematic review. The description below states what was actually done, including its gaps.

## 1. Research questions

- RQ1. Which serving-system mechanisms (batching, scheduling, KV cache management, prefix caching) shape the latency and throughput of the evaluated stack (vLLM)?
- RQ2. How do prior MoE offloading and hybrid CPU-GPU systems place experts, and do they (a) copy weights to the GPU (on demand, prefetch, streaming), (b) compute experts on the CPU, or (c) let GPU kernels read host memory in place? Is placement static or dynamic?
- RQ3. Which weight quantization methods, in particular trellis and vector codes and MoE-specific low-bit methods, underlie or relate to EXL3?
- RQ4. Which speculative decoding methods relate to the DSpark drafter and to MTP?
- RQ5. What do the primary sources say about the DeepSeek architecture (DeepSeekMoE, MLA, sparse attention, Engram, MTP, DSpark, V4.1-Flash)?
- RQ6. Which benchmarks, metrics and methodology references are relevant for evaluating quantized and offloaded models (task accuracy, KL divergence, flips, serving metrics, bootstrap)?
- RQ7. Which software artifacts make up the evaluated stack, and what does their own documentation state?

## 2. Sources

Searched or fetched: arXiv (abs pages, export API, BibTeX export), ACL Anthology, USENIX (OSDI pages), proceedings.neurips.cc and neurips.cc, PMLR, proceedings.iclr.cc and iclr.cc, mlsys.org and MLSys proceedings, JMLR, Zenodo, Crossref API (for DOIs of ACM and IEEE papers), Project Euclid and publisher pages (bootstrap references), GitHub, Hugging Face (model cards, dataset cards, repository file listings), NVIDIA documentation (docs.nvidia.com), vLLM documentation (docs.vllm.ai), and general web search (WebSearch) for discovery.

Attempted but unavailable from this environment:
- DBLP (web and API): bot-check page or connection resets on every request. DBLP was therefore not used to confirm any venue.
- ACM Digital Library and IEEE Xplore: HTTP 403. ACM and IEEE venues, pages and DOIs were confirmed through the Crossref record of the DOI, the USENIX or conference accepted-papers page, the arXiv comment line, or the venue line on the paper's first page.
- OpenReview: browser check. ICLR and NeurIPS venues were confirmed on iclr.cc / proceedings.iclr.cc and neurips.cc / proceedings.neurips.cc instead.
- Semantic Scholar was not queried.

## 3. Search strategy

The search had two parts.

**Known-item search.** A seed list of candidate works per topic (serving, MoE offloading, quantization, speculative decoding, DeepSeek reports, evaluation, software) was fixed in advance by the study authors. Each item was located by title search (WebSearch and arXiv), then fetched. Exact strings for known-item lookups were the paper titles or system names (for example `"Fast Inference of Mixture-of-Experts Language Models with Offloading"`, `KTransformers SOSP 2025`, `QTIP trellis coded quantization NeurIPS 2024`); not every individual lookup string was logged.

**Open search** (logged strings), used to find work beyond the seed list:

MoE offloading and direct host access:
1. `zero-copy expert offloading MoE inference GPU host memory`
2. `unified virtual addressing MoE inference experts pinned host memory`
3. `static expert placement profiling MoE inference hybrid CPU GPU`
4. `GPU direct host memory access zero-copy LLM inference weights PCIe kernel reads host memory`
5. `arxiv MoE expert offloading "zero-copy" UVA GPU kernel reads experts from host memory`
6. `hybrid CPU-GPU MoE inference 2025 arxiv Fate DAOP HybriMoE MoE-Lens`
7. `EdgeMoE Empowering Sparse Large Language Models on Mobile Devices IEEE Transactions on Mobile Computing`
8. `ProMoE "Fast MoE-based LLM Serving using Proactive Caching" Song Zhong Chen conference`
9. `"HybriMoE" Design Automation Conference 2025 ieeexplore`
10. `"DAOP: Data-Aware Offloading and Predictive Pre-Calculation" DATE 2025 ieeexplore`
11. `"MoE-Lens" Towards the Hardware Limit High-Throughput MoE LLM Serving conference`
12. `"Klotski" expert-aware multi-batch pipeline MoE inference conference 2025`
13. `"eMoE" Task-aware Memory Efficient Mixture-of-Experts inference Tairin Shen conference`
14. `"10.1145/3806645.3807594"`
15. `"ExpertFlow" "Predictive Expert Caching and Token Scheduling" DAC 2026`
16. `"MoE-Gen" High-Throughput MoE Inference Single GPU Module-Based Batching published conference`
17. `"Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate" journal OR conference`

DeepSeek sources:
18. `DeepSeek Engram conditional memory scalable lookup arXiv`
19. `DeepSeek DSpark speculative decoding drafter`
20. `DeepSeek-V3.2 DeepSeek Sparse Attention arXiv technical report`
21. `DeepSeek-V4 technical report arXiv DeepSeek-V4-Flash Compressed Sparse Attention`
22. `Native Sparse Attention Hardware-Aligned Natively Trainable aclanthology 2025.acl-long`
23. `"Conditional Memory via Scalable Lookup" accepted conference 2026 Engram`
24. `aclanthology.org 2026.acl-long "Conditional Memory via Scalable Lookup"`
25. `"Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts" openreview`

Serving methodology (open part): searches for an LLM serving metrics paper (Metron / Etalon) and for CUDA documentation of unified virtual addressing and mapped (zero-copy) host memory.

Direct page checks: `huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash` (card and `tree/main` file listing, which lists `DeepSeek_V41_Tech_Report.pdf`), the V4-Flash card, and all software and model pages listed in the task.

## 4. Inclusion and exclusion criteria

Inclusion:
- I1. The work addresses at least one research question in Section 1.
- I2. The canonical page was fetched in this session and title, full author list and year matched.
- I3. Peer-reviewed venue confirmed on a fetched page; otherwise the work is included only as an arXiv preprint (`@misc` with `eprint`), or as software/documentation (`@misc`, `note = {Accessed 2026-10-06}`).

Exclusion:
- E1. Could not be fetched or verified (title, authors or year not confirmed).
- E2. Duplicate of an included record (same work under a new title, a mirror of a primary source, a secondary explainer of a primary paper).
- E3. Not archival (forum posts, issue trackers, blog explainers) when a primary source exists or none is citable.
- E4. Scope rule set by the study: systems whose description centers on a family of local-inference frameworks outside this study's scope are not included, and those frameworks are not mentioned. Works whose abstract does not center on them were kept and described only by their own techniques.
- E5. Low relevance or over the per-topic budget (about 15 entries for open-search MoE work), recorded with reason.

## 5. Screening and verification procedure

1. Candidates were split into seven topic groups and verified in parallel by assistant sub-agents, each with the same written rules (fetch every page, confirm title, full author order, year and venue, never invent DOIs or pages, cite arXiv when a venue is not confirmed).
2. For each candidate: fetch the canonical page, compare metadata, record the URL and the source that confirmed the venue in `verification.tsv`.
3. DOIs and page ranges were recorded only when they appeared on a fetched page or Crossref record.
4. Per-paper facts used in the related-work text (placement, CPU compute, copy vs in-place access, caching and prediction, quantization) were extracted from abstracts, and where needed from paper bodies, with the source tagged. Quantitative claims in the text come only from abstracts (or, for the evaluated software, from its own documentation).
5. Where versions disagree, the chosen version is noted in the bib `note` and in `verification.tsv` (for example: OPTQ vs GPTQ title at ICLR 2023; AWQ MLSys title; QMoE MLSys title; FlexGen PMLR author list vs arXiv; DeepSpeed-Inference SC22 author order; SpecInfer published title; TurboSpec, formerly SmartSpec; Etalon, formerly Metron; Engram arXiv version vs ACL 2026 conference version, both kept).
6. The integrator merged the group outputs, checked for duplicate keys, balanced braces, absence of en and em dashes and of out-of-scope terms, and compiled the bibliography with `bibtex` and `plainnat` (0 warnings, all keys resolved).

## 6. Counts

| Stage | Count |
|---|---|
| Records considered explicitly (included plus excluded papers and sources; open-search results not considered explicitly were not logged) | 116 |
| Included in `references.bib` | 99 |
| Excluded or merged (`excluded.tsv`, kinds "candidate dropped" and "duplicate or redundant source") | 17 |
| Included with confirmed peer-reviewed venue (conference, journal, or book) | 56 |
| Included as arXiv preprint | 28 |
| Included as software, documentation, model card, dataset, or archived software release (Zenodo) | 15 |

Of the excluded candidates, HOBBIT was fetched and verified but removed by rule E4; MoEShard, CoX-MoE, MoE-CORE, ExactMoE, BOOST, Multipath Memory Access and the LLaMA3 quantization study were seen in search results only and not verified (E1/E5). The number of raw search hits screened by title was not recorded.

## 7. Threats to validity

- **Coverage.** The seed list was fixed by the authors and the open search was limited to web search plus targeted lookups; DBLP, Semantic Scholar, ACM DL and IEEE Xplore could not be searched. Relevant work, especially 2026 preprints on direct host-memory access for MoE, may be missing. Several recent candidates were seen in search results but not verified (see `excluded.tsv`).
- **Venue confirmation.** Without DBLP, ACM DL and IEEE Xplore, some venues rest on Crossref, arXiv comments, or first-page venue lines. Entries whose venue appeared only in a search snippet (Fate, MoE-Lens) or only in an arXiv comment for an IEEE venue (HybriMoE) are cited as arXiv preprints and could be upgraded later. RULER's COLM 2024 venue rests on the arXiv comment.
- **Fact extraction.** Body-level facts were read through a summarizing fetch tool, which can omit or paraphrase details. The comparison table is therefore restricted to coarse, qualitative axes, and "n.s." marks properties not stated in the text read.
- **Scope rule.** Excluding systems centered on a family of local-inference frameworks (E4) removes some relevant hybrid CPU-GPU work (for example HOBBIT) by design, so the review is not exhaustive on consumer-hardware offloading.
- **Software sources.** The evaluated stack is documented only in GitHub READMEs and Hugging Face model cards, which can change; access date 2026-10-06 is recorded, and specific revisions should be pinned in the artifact.
- **Single reviewer.** Screening and extraction were done by assistant agents and integrated by one assistant; there was no independent second screening or inter-rater agreement.

## Post-processing of author lists (2026-10-06)

Entries with more than 20 verified authors keep their first three authors (or the corporate author `{DeepSeek-AI}`
alone) followed by `and others`, which BibTeX renders as "et al.". The full lists were checked against the source
pages before truncation; affected keys: reddi2020mlperf, deepseekai2024v2, deepseekai2024v3, deepseekai2025v32,
deepseekai2026v4, deepseekai2026v41flash, cheng2026dspark, cheng2026engram, sutawika2023lmeval,
biderman2024lessons, chen2021humaneval. A spot check of four arXiv entries (2609.19969, 2604.26074, 2607.05147,
2601.07372) against the arXiv API confirmed title, first authors and date.
