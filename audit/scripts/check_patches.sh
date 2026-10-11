#!/usr/bin/env bash
# Provenance checks over the sources fetched by fetch_sources.sh. Writes small text evidence files into audit/evidence/.
#
#   AUDIT_SRC     cache dir used by fetch_sources.sh (default ~/.cache/vllm-exl3-audit-src)
#   LOCAL_RECIPE  inferencekey's local copy of the recipe (default ~/me/inferencekey-deepseek/vllm-exl3/a100/pack/recipe)
#   IK_REPO       inferencekey-deepseek git repo, used to confirm the in-repo script copies equal commit IK_COMMIT
#   IK_COMMIT     default 546071a
#   NO_GH=1       skip the GitHub API calls (PR metadata and diffs are then read from the cache if present)
#   NO_NET=1      skip the Hugging Face commit-history calls
#
# Nothing here modifies the fetched checkouts: patches are applied to copies under $AUDIT_SRC/work.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
EV=$ROOT/audit/evidence
AUDIT_SRC=${AUDIT_SRC:-$HOME/.cache/vllm-exl3-audit-src}
LOCAL_RECIPE=${LOCAL_RECIPE:-$HOME/me/inferencekey-deepseek/vllm-exl3/a100/pack/recipe}
IK_REPO=${IK_REPO:-$HOME/me/inferencekey-deepseek}
IK_COMMIT=${IK_COMMIT:-546071a}
IK_COPY=$ROOT/data/exploratory/inferencekey-deepseek@546071a
HF_REV=a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5
HF=$AUDIT_SRC/hf-diffbot-$HF_REV
RECIPE=$HF/recipe
WORK=$AUDIT_SRC/work
VLLM_PATCHES=(pr58132-decoder-swa-replay sm120-thor-fixes pr59585-dsml-finite-args pr59894-decoder-replay-cut-trim pr58560-compressor-null-block)
PRS=(58132 58560 59585 59894 57028 56227)

mkdir -p "$EV" "$WORK" "$AUDIT_SRC/prs"
for d in vllm vllm-exl3 exllamav3 sfxnz-recipe mia-glm53; do
  [[ -d $AUDIT_SRC/$d/.git ]] || { echo "missing $AUDIT_SRC/$d: run fetch_sources.sh first" >&2; exit 1; }
done
[[ -d $RECIPE ]] || { echo "missing $RECIPE: run fetch_sources.sh first" >&2; exit 1; }
log() { printf '[check] %s\n' "$*" >&2; }
copy_tree() { tar -C "$1" --exclude=.git -cf - . | tar -C "$2" -xf -; }   # checked-out files only, no .git

# ---------------------------------------------------------------------------------------------------------------------
log "resolved sources"
{
  echo "# Sources resolved by fetch_sources.sh (checked $(date -u +%Y-%m-%dT%H:%M:%SZ))"
  for d in vllm vllm-exl3 exllamav3 sfxnz-recipe mia-glm53; do
    printf '%-14s %s %s  %s\n' "$d" "$(git -C "$AUDIT_SRC/$d" rev-parse HEAD)" "$(git -C "$AUDIT_SRC/$d" log -1 --format=%cs)" \
      "$(git -C "$AUDIT_SRC/$d" remote get-url origin)"
  done
  printf '%-14s %s\n' hf-diffbot "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["sha"])' "$HF/.revision.json")"
  echo "vllm sparse checkout:"; git -C "$AUDIT_SRC/vllm" sparse-checkout list | sed 's/^/  /'
} > "$EV/sources_resolved.txt"

# ---------------------------------------------------------------------------------------------------------------------
log "HF tree listing + recipe hashes"
python3 - "$HF" "$LOCAL_RECIPE" "$EV" <<'PY'
import hashlib, json, os, sys
hf, local, ev = sys.argv[1:4]
tree = json.load(open(os.path.join(hf, ".tree.json")))
files = sorted((e for e in tree if e["type"] == "file"), key=lambda e: e["path"])
with open(os.path.join(ev, "hf_diffbot_tree_a1cf1bd2.tsv"), "w") as f:
    f.write("# diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 @ a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5\n")
    f.write("# source: https://huggingface.co/api/models/<repo>/tree/<rev>?recursive=1\n")
    f.write("path\tsize_bytes\tlfs_sha256\tgit_blob_oid\n")
    for e in files:
        lfs = e.get("lfs") or {}
        f.write(f"{e['path']}\t{lfs.get('size', e['size'])}\t{lfs.get('oid', '')}\t{e['oid']}\n")
    tot = sum((e.get("lfs") or {}).get("size", e["size"]) for e in files)
    f.write(f"# {len(files)} files, {tot} bytes ({tot / 1e9:.1f} GB)\n")

def blob_oid(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()

rows, bad, local_diff, local_missing = [], 0, 0, 0
for e in files:
    p = e["path"]
    if not p.startswith("recipe/"):
        continue
    rel = p[len("recipe/"):]
    data = open(os.path.join(hf, p), "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    lfs = e.get("lfs")
    ok = (sha == lfs["oid"]) if lfs else (blob_oid(data) == e["oid"])
    bad += not ok
    lp = os.path.join(local, rel)
    if os.path.exists(lp):
        lsha = hashlib.sha256(open(lp, "rb").read()).hexdigest()
        lstate = "same" if lsha == sha else "DIFFERENT"
        local_diff += lstate != "same"
    else:
        lstate = "absent"; local_missing += 1
    rows.append(f"{rel}\t{len(data)}\t{sha}\t{'lfs-sha256' if lfs else 'git-blob'}:{'OK' if ok else 'MISMATCH'}\tlocal:{lstate}")
extra = []
if os.path.isdir(local):
    known = {e["path"][len("recipe/"):] for e in files if e["path"].startswith("recipe/")}
    for dp, _, fs in os.walk(local):
        for fn in fs:
            rel = os.path.relpath(os.path.join(dp, fn), local)
            if rel not in known and "/build/" not in "/" + rel and not rel.startswith(".cache"):
                extra.append(rel)
with open(os.path.join(ev, "recipe_hashes.tsv"), "w") as f:
    f.write("# recipe/ at a1cf1bd2: sha256 of the downloaded file, check against the HF tree API, and the local copy\n")
    f.write(f"# local copy: {local}\n")
    f.write("path\tsize\tsha256\tupstream_check\tlocal_copy\n")
    f.write("\n".join(rows) + "\n")
    f.write(f"# {len(rows)} files; upstream mismatches: {bad}; local differs: {local_diff}; local absent: {local_missing}; "
            f"local-only files: {len(extra)} {sorted(extra)}\n")
print(f"[check] recipe: {len(rows)} files, {bad} upstream mismatches, {local_diff} local diffs", file=sys.stderr)
PY

# ---------------------------------------------------------------------------------------------------------------------
log "patch apply checks"
APPLY=$EV/patch_apply.txt
STAT=$EV/patch_diffstats.txt
: > "$APPLY"; : > "$STAT"
{
  echo "# git apply --numstat of every recipe patch (added, removed, path)"
  for p in "${VLLM_PATCHES[@]}"; do echo "## patches/vllm/$p.diff"; git apply --numstat "$RECIPE/patches/vllm/$p.diff"; done
  echo "## patches/vllm-exl3/exl3-thor.patch"; git apply --numstat "$RECIPE/patches/vllm-exl3/exl3-thor.patch"
  echo "## kernels/exllamav3-5be8865-exl3_moe.patch"; git apply --numstat "$RECIPE/kernels/exllamav3-5be8865-exl3_moe.patch"
  echo "## quantize/quantize_mtp_exl3.diff"; git apply --numstat "$RECIPE/quantize/quantize_mtp_exl3.diff" 2>&1 || true
} > "$STAT"

# vLLM: each patch alone on the pristine tree, then all five in the order of docker/build.sh (patch -p1 --forward).
VW=$WORK/vllm-patched; rm -rf "$VW"; mkdir -p "$VW"
copy_tree "$AUDIT_SRC/vllm" "$VW"
{
  echo "# vLLM 36768d1bfd39094681cdbc8cb37d4b31c0729c89 (sparse checkout)"
  for p in "${VLLM_PATCHES[@]}"; do
    if out=$(git -C "$AUDIT_SRC/vllm" apply --check "$RECIPE/patches/vllm/$p.diff" 2>&1); then r=OK; else r="FAIL: $(echo "$out" | head -3 | tr '\n' ' ')"; fi
    echo "alone      git apply --check $p.diff: $r"
  done
  for p in "${VLLM_PATCHES[@]}"; do
    if out=$(cd "$VW" && patch -p1 --forward --no-backup-if-mismatch < "$RECIPE/patches/vllm/$p.diff" 2>&1); then
      echo "in-order   patch -p1 --forward $p.diff: OK ($(echo "$out" | grep -c '^patching file') files$(echo "$out" | grep -q -i 'offset\|fuzz' && echo ', with offset/fuzz'))"
    else echo "in-order   patch -p1 --forward $p.diff: FAIL $(echo "$out" | tail -3 | tr '\n' ' ')"; fi
  done
} >> "$APPLY"

# vllm-exl3 + exl3-thor.patch
XW=$WORK/vllm-exl3-patched; rm -rf "$XW"; mkdir -p "$XW"
copy_tree "$AUDIT_SRC/vllm-exl3" "$XW"
{
  echo "# vllm-exl3 d3cfd394920360d69f820d2dc96f8292a9e10283"
  if out=$(git -C "$AUDIT_SRC/vllm-exl3" apply --check "$RECIPE/patches/vllm-exl3/exl3-thor.patch" 2>&1); then r=OK; else r="FAIL: $out"; fi
  echo "git apply --check exl3-thor.patch: $r"
  if out=$(cd "$XW" && patch -p1 --no-backup-if-mismatch < "$RECIPE/patches/vllm-exl3/exl3-thor.patch" 2>&1); then r=OK; else r="FAIL: $out"; fi
  echo "patch -p1 exl3-thor.patch (as docker/build.sh): $r"
} >> "$APPLY"

# ExLlamaV3 exllamav3_ext + exl3_moe patch, then compare kernels/xmoe with the patched upstream files.
EXT=$AUDIT_SRC/exllamav3/exllamav3/exllamav3_ext
EW=$WORK/exl3_ext-patched; rm -rf "$EW"; mkdir -p "$EW"; cp -r "$EXT/." "$EW/"
{
  echo "# ExLlamaV3 5be886578ec80324c2c715269387be2058724b6e, directory exllamav3/exllamav3_ext"
  if out=$(cd "$EXT" && patch --dry-run -p1 < "$RECIPE/kernels/exllamav3-5be8865-exl3_moe.patch" 2>&1); then r=OK; else r="FAIL: $out"; fi
  echo "patch --dry-run -p1 exllamav3-5be8865-exl3_moe.patch: $r"
  (cd "$EW" && patch -s -p1 --no-backup-if-mismatch < "$RECIPE/kernels/exllamav3-5be8865-exl3_moe.patch")
  echo "# sitecustomize: recipe patches/vllm/sitecustomize.py vs sfxnz docker/patch/sitecustomize.py, see sitecustomize_vs_sfxnz.txt"
  echo "# quantize/quantize_mtp_exl3.diff header:"; head -4 "$RECIPE/quantize/quantize_mtp_exl3.diff" | sed 's/^/  /'
  q=$AUDIT_SRC/sfxnz-recipe/tools/quantize_experts_exl3.py
  if [[ -f $q ]]; then
    tmp=$(mktemp -d); cp "$q" "$tmp/q.py"
    if (cd "$tmp" && patch -s --dry-run q.py < "$RECIPE/quantize/quantize_mtp_exl3.diff" >/dev/null 2>&1); then r="applies to sfxnz tools/quantize_experts_exl3.py"; else r="does not apply directly to sfxnz tools/quantize_experts_exl3.py (paths or context differ)"; fi
    echo "quantize_mtp_exl3.diff: $r"; rm -rf "$tmp"
  fi
} >> "$APPLY"

{
  echo "# recipe kernels/xmoe/* vs ExLlamaV3 5be88657 exllamav3/exllamav3_ext/* (before and after the recipe's exl3_moe patch)"
  echo "# status: IDENTICAL-UPSTREAM = byte-identical to stock ExLlamaV3; MODIFIED = differs from stock (changed lines); NEW = no upstream file"
  (cd "$RECIPE/kernels/xmoe" && find . -type f | sort) | while read -r f; do
    f=${f#./}; u=$EXT/$f; pp=$EW/$f
    after=$([[ -f $pp ]] && cmp -s "$RECIPE/kernels/xmoe/$f" "$pp" && echo "=patched-upstream" || echo "!=patched-upstream")
    if [[ ! -f $u ]]; then echo "NEW                $f  ($after)"
    elif cmp -s "$RECIPE/kernels/xmoe/$f" "$u"; then echo "IDENTICAL-UPSTREAM $f"
    else echo "MODIFIED           $f  changed_lines=$(diff "$u" "$RECIPE/kernels/xmoe/$f" | grep -c '^[<>]') ($after)"; fi
  done
  echo "# kernels/xmoe_do/do_gemm.cu includes:"; grep -n '#include' "$RECIPE/kernels/xmoe_do/do_gemm.cu" | sed 's/^/  /'
  echo "# kernels/xmoe/LICENSE.exllamav3 vs upstream LICENSE: $(cmp -s "$RECIPE/kernels/xmoe/LICENSE.exllamav3" "$AUDIT_SRC/exllamav3/LICENSE" && echo identical || echo different)"
  echo "# ExLlamaV3 5be88657 already ships host/CPU-side MoE offload (different mechanism, experts computed on the CPU):"
  grep -n -E 'CPU MoE offload|EXL3_MOE_CPU_OFFLOAD|EXL3_VISION_PINNED|zero-copy device alias' "$AUDIT_SRC/exllamav3/doc/env_vars.md" | sed 's|^|  doc/env_vars.md:|'
  grep -n 'CPU-offloaded MoE experts' "$EXT/cpu/moe_handoff.h" | sed 's|^|  exllamav3/exllamav3_ext/cpu/moe_handoff.h:|'
} > "$EV/xmoe_vs_exllamav3.txt"

# ---------------------------------------------------------------------------------------------------------------------
log "HF commit history of the Diffbot repo (when each feature first appears)"
if [[ ${NO_NET:-0} != 1 ]]; then
python3 - "$EV/hf_diffbot_history.txt" <<'PY' || log "WARN: HF history not fetched"
import json, sys, urllib.request
repo = "diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000"
get = lambda u: json.load(urllib.request.urlopen(u, timeout=60))
commits = get(f"https://huggingface.co/api/models/{repo}/commits/main")
probe = {"exl3-thor.patch": "recipe/patches/vllm-exl3/exl3-thor.patch", "plans/": "recipe/plans/", "xmoe/": "recipe/kernels/xmoe/",
         "xmoe_do": "recipe/kernels/xmoe_do/", "convert_engram_int4": "recipe/quantize/convert_engram_int4.py",
         "sm120-thor-fixes": "recipe/patches/vllm/sm120-thor-fixes.diff", "quantize_mtp_exl3": "recipe/quantize/quantize_mtp_exl3.py"}
out = [f"# {repo}: commits on main (api/models/<repo>/commits/main), oldest last, and which recipe parts each tree has",
       "# columns: commit date authors | " + " ".join(probe)]
for c in commits:
    paths = [e["path"] for e in get(f"https://huggingface.co/api/models/{repo}/tree/{c['id']}?recursive=1")]
    has = ["Y" if any(p.startswith(v) for p in paths) else "." for v in probe.values()]
    who = ",".join(a.get("user") or "?" for a in c.get("authors", []))
    out.append(f"{c['id'][:10]} {c['date'][:19]} {who} | {' '.join(has)} | {c['title'][:110]}")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
PY
fi

# ---------------------------------------------------------------------------------------------------------------------
log "GitHub PRs"
if [[ ${NO_GH:-0} != 1 ]] && command -v gh >/dev/null; then
  for n in "${PRS[@]}"; do
    gh pr view "$n" -R vllm-project/vllm --json number,title,author,state,createdAt,mergedAt,mergeCommit,url,headRefOid,commits \
      > "$AUDIT_SRC/prs/pr$n.json"
    gh pr diff "$n" -R vllm-project/vllm > "$AUDIT_SRC/prs/pr$n.diff"
  done
  gh api repos/vllm-project/vllm/issues/57144 > "$AUDIT_SRC/prs/issue57144.json"
fi
python3 - "$AUDIT_SRC/prs" "$RECIPE/patches/vllm" "$EV" "${PRS[@]}" <<'PY'
import collections, json, os, re, sys
prs_dir, pdir, ev, *nums = sys.argv[1:]
def parse(path):
    files, cur = collections.defaultdict(lambda: {"+": collections.Counter(), "-": collections.Counter()}), None
    for line in open(path, errors="replace"):
        if line.startswith("+++ "):
            p = line[4:].strip().split("\t")[0]
            cur = None if p == "/dev/null" else re.sub(r"^b/", "", p); continue
        if line.startswith("--- "):
            p = line[4:].strip().split("\t")[0]
            if p != "/dev/null": cur = re.sub(r"^a/", "", p)
            continue
        if cur and line[:1] in "+-" and not line.startswith(("+++", "---")):
            files[cur][line[0]][line[1:].rstrip()] += 1
    return files
recipe_for = {"58132": "pr58132-decoder-swa-replay.diff", "58560": "pr58560-compressor-null-block.diff",
              "59585": "pr59585-dsml-finite-args.diff", "59894": "pr59894-decoder-replay-cut-trim.diff",
              "57028": "sm120-thor-fixes.diff"}
out = ["# vLLM PRs: metadata (gh pr view) and a line-level comparison of the PR diff (gh pr diff, PR head today) with the recipe file",
       "# per file: +/- = lines added/removed by the recipe patch; 'in PR' = how many of them the PR diff also has"]
for n in nums:
    jp = os.path.join(prs_dir, f"pr{n}.json")
    if not os.path.exists(jp):
        out.append(f"PR #{n}: no metadata cached"); continue
    d = json.load(open(jp))
    commits = d.get("commits") or []
    authors = sorted({a.get("login") or a.get("name") for c in commits for a in c.get("authors", [])})
    slim = {k: d.get(k) for k in ("number", "title", "state", "createdAt", "mergedAt", "url", "headRefOid")}
    slim["author"] = (d.get("author") or {}).get("login")
    slim["mergeCommit"] = (d.get("mergeCommit") or {}).get("oid")
    slim["commit_authors"] = authors
    slim["n_commits"] = len(commits)
    slim["last_commit_date"] = max((c.get("committedDate") or "" for c in commits), default=None)
    json.dump(slim, open(os.path.join(ev, f"gh_pr_{n}.json"), "w"), indent=1)
    out.append(f"\nPR #{n} {slim['state']} author={slim['author']} commit_authors={authors} merged={slim['mergedAt']} "
               f"merge_commit={slim['mergeCommit']} head={slim['headRefOid']} last_commit={slim['last_commit_date']}\n  {slim['title']}\n  {slim['url']}")
    rf = recipe_for.get(n)
    dp = os.path.join(prs_dir, f"pr{n}.diff")
    if not rf or not os.path.exists(dp):
        continue
    pr, rc = parse(dp), parse(os.path.join(pdir, rf))
    out.append(f"  recipe file: patches/vllm/{rf}")
    tot_r = tot_in = 0
    for f in sorted(set(rc) | set(pr)):
        r, p = rc.get(f), pr.get(f)
        if r is None:
            out.append(f"    {f}: only in PR (+{sum(p['+'].values())} -{sum(p['-'].values())})"); continue
        if p is None:
            out.append(f"    {f}: only in recipe (+{sum(r['+'].values())} -{sum(r['-'].values())})")
            tot_r += sum(r['+'].values()) + sum(r['-'].values()); continue
        ra, rr = sum(r["+"].values()), sum(r["-"].values())
        ia = sum(min(c, p["+"][l]) for l, c in r["+"].items()); ir = sum(min(c, p["-"][l]) for l, c in r["-"].items())
        pa, pr_ = sum(p["+"].values()), sum(p["-"].values())
        same = r["+"] == p["+"] and r["-"] == p["-"]
        tot_r += ra + rr; tot_in += ia + ir
        out.append(f"    {f}: recipe +{ra} -{rr}, in PR +{ia} -{ir}; PR +{pa} -{pr_}{'  IDENTICAL' if same else ''}")
    if tot_r:
        out.append(f"  => {tot_in}/{tot_r} changed lines of the recipe file also appear in the PR diff ({100 * tot_in / tot_r:.0f}%)")
ip = os.path.join(prs_dir, "issue57144.json")
if os.path.exists(ip):
    d = json.load(open(ip))
    out.append(f"\n#57144 is an issue, not a PR: state={d['state']} author={d['user']['login']} title={d['title']!r} {d['html_url']}")
    json.dump({k: d.get(k) for k in ("number", "title", "state", "created_at", "html_url")} | {"author": d["user"]["login"], "is_pr": "pull_request" in d},
              open(os.path.join(ev, "gh_issue_57144.json"), "w"), indent=1)
open(os.path.join(ev, "pr_vs_recipe.txt"), "w").write("\n".join(out) + "\n")
PY

{
  echo; echo "# merge commits vs the vLLM pin 36768d1 (git merge-base --is-ancestor)"
  for n in "${PRS[@]}"; do
    mc=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print((d.get("mergeCommit") or {}).get("oid") or "")' "$AUDIT_SRC/prs/pr$n.json" 2>/dev/null || true)
    [[ -n $mc ]] || { echo "PR #$n: not merged"; continue; }
    git -C "$AUDIT_SRC/vllm" cat-file -e "$mc^{commit}" 2>/dev/null || git -C "$AUDIT_SRC/vllm" fetch -q origin "$mc" 2>/dev/null || true
    if git -C "$AUDIT_SRC/vllm" merge-base --is-ancestor "$mc" 36768d1bfd39094681cdbc8cb37d4b31c0729c89 2>/dev/null; then r="already in the pin"; else r="NOT in the pin (merged later)"; fi
    echo "PR #$n merge commit $mc: $r"
  done
} >> "$EV/pr_vs_recipe.txt"

# ---------------------------------------------------------------------------------------------------------------------
log "feature greps"
FG=$EV/feature_grep.txt
{
  echo "# Presence of each feature marker per tree: count of matching lines, then up to 3 file:line hits."
  echo "# Trees: upstream = pinned commit as published; +recipe = after applying the recipe's patches (copies in \$AUDIT_SRC/work)."
} > "$FG"
grep_tree() {  # label dir pattern [pathspec-glob]
  local label=$1 dir=$2 pat=$3 hits n
  hits=$(cd "$dir" && grep -rnI -E "$pat" --include='*.py' --include='*.cu' --include='*.cuh' --include='*.h' --include='*.cpp' --include='*.sh' --include='*.md' --include='*.yaml' . 2>/dev/null | grep -v '/\.git/' || true)
  n=$(printf '%s' "$hits" | grep -c . || true)
  printf '  %-34s %4s  %s\n' "$label" "$n" "$(printf '%s\n' "$hits" | head -3 | cut -c1-150 | sed 's|^\./||' | paste -sd '|' -)"
}
declare -A FEAT=(
  [host_expert_plan]='EXL3_HOST_EXPERTS|host_experts|_HOST_PLAN'
  [cudaHostRegister]='cudaHostRegister'
  [side_stream_sm_budget]='EXL3_HOST_OVERLAP|xmoe_set_sm_budget|set_sm_budget'
  [split_launch_negative_counts]='xmoe_set_force_group|skip but keep offsets|negative expert count'
  [decode_once_prefill]='EXL3_DO_MIN_TOKENS|EXL3_DO\b|xmoe_do'
  [lean_prep]='EXL3_LEAN_PREP'
  [flat_decode_kernel]='exl3_moe_flat|flat scheduler|gemm_inner_flat'
  [engram_int4]='ENGRAM_INT4|int4_fp16s32|_engram_lookup_int4'
  [engram_host_offload]='cpu_offload|Engram CPU offload'
  [engram_disk]='DSV41_ENGRAM_DISK'
  [thor_hooks]='THOR_[A-Z_]+'
  [dspark]='[Dd][Ss]park'
  [fat_any_item]='fat\.any\(\)\.item\(\)'
  [cpu_moe_offload_exllamav3]='EXL3_MOE_CPU_OFFLOAD|moe_cpu_offload'
)
TREES=(
  "vllm@36768d1|$AUDIT_SRC/vllm"
  "vllm@36768d1+recipe|$VW"
  "vllm-exl3@d3cfd394|$AUDIT_SRC/vllm-exl3"
  "vllm-exl3@d3cfd394+exl3-thor|$XW"
  "exllamav3@5be88657|$AUDIT_SRC/exllamav3"
  "exllamav3_ext+exl3_moe.patch|$EW"
  "sfxnz@460ac89c|$AUDIT_SRC/sfxnz-recipe"
  "mia-glm53@HEAD|$AUDIT_SRC/mia-glm53"
  "diffbot-recipe@a1cf1bd2|$RECIPE"
)
for k in host_expert_plan cudaHostRegister side_stream_sm_budget split_launch_negative_counts decode_once_prefill lean_prep \
         flat_decode_kernel engram_int4 engram_host_offload engram_disk thor_hooks dspark fat_any_item cpu_moe_offload_exllamav3; do
  echo "## $k  /${FEAT[$k]}/" >> "$FG"
  for t in "${TREES[@]}"; do grep_tree "${t%%|*}" "${t#*|}" "${FEAT[$k]}" >> "$FG"; done
done

# ---------------------------------------------------------------------------------------------------------------------
# The report sections below use diff/grep/cmp, whose exit status 1 means "differs" or "no match", not failure.
set +e +o pipefail
log "fat.any().item() location"
{
  echo "# The capture-size-384 failure: fat.any().item() in the fat-expert path, reached when tokens > VLLM_EXL3_FAT_THRESHOLD (256)"
  echo "## vllm-exl3 d3cfd394 as published (src/vllm_exl3/exl3.py):"
  grep -n -E 'FAT_EXPERT_THRESHOLD = max|fat_possible = tokens|fat\.any\(\)\.item\(\)|^def apply_exl3_fused_moe' "$AUDIT_SRC/vllm-exl3/src/vllm_exl3/exl3.py" | sed 's/^/  exl3.py:/'
  echo "## vllm-exl3 d3cfd394 + recipe exl3-thor.patch (what bootstrap.sh installs):"
  grep -n -E 'FAT_EXPERT_THRESHOLD = max|fat_possible = tokens|fat\.any\(\)\.item\(\)|^def apply_exl3_fused_moe|^def apply_exl3_experts' "$XW/src/vllm_exl3/exl3.py" | sed 's/^/  exl3.py:/'
  echo "  exl3.py:1837: $(sed -n 1837p "$XW/src/vllm_exl3/exl3.py" | sed 's/^ *//')"
  echo "  exl3.py:2753: $(sed -n 2753p "$XW/src/vllm_exl3/exl3.py" | sed 's/^ *//')"
  echo "## traceback in inferencekey's log (pro6000-concurrency/logs/serve_attempt1_capture384.log):"
  log384=$IK_COPY/vllm-exl3/pro6000-concurrency/logs/serve_attempt1_capture384.log
  if [[ ! -f $log384 && -d $IK_REPO/.git ]]; then log384=<(git -C "$IK_REPO" show "$IK_COMMIT:vllm-exl3/pro6000-concurrency/logs/serve_attempt1_capture384.log"); fi
  if [[ -e $log384 ]]; then grep -m6 -o -E 'exl3.py", line [0-9]+, in [a-z_0-9]+|elif fat_possible.*|cudaErrorStreamCaptureUnsupported' "$log384" | sort -u | sed 's/^/  /'; else echo "  (log not available)"; fi
} > "$EV/fat_any_item.txt"

# ---------------------------------------------------------------------------------------------------------------------
log "sitecustomize vs sfxnz"
SC=$RECIPE/patches/vllm/sitecustomize.py
SX=$AUDIT_SRC/sfxnz-recipe/docker/patch/sitecustomize.py
{
  echo "# recipe patches/vllm/sitecustomize.py ($(wc -l < "$SC") lines) vs sfxnz@460ac89c docker/patch/sitecustomize.py ($(wc -l < "$SX") lines)"
  echo "# diffstat of sfxnz -> recipe:"; diff -u "$SX" "$SC" | { grep -c '^+[^+]' || true; } | sed 's/^/  added lines: /'; diff -u "$SX" "$SC" | { grep -c '^-[^-]' || true; } | sed 's/^/  removed lines: /'
  echo "# sfxnz hooks (comment headers in sfxnz file) and whether the recipe keeps them:"
  grep -n -E '^# [A-Z-]' "$SX" | while IFS=: read -r ln txt; do
    grep -qF -- "$txt" "$SC" && k="kept at recipe:$(grep -nF -- "$txt" "$SC" | head -1 | cut -d: -f1)" || k="NOT FOUND"
    printf '  sfxnz:%-4s %-90s %s\n' "$ln" "$(echo "$txt" | cut -c1-90)" "$k"
  done
  echo "# recipe-only hooks: 'thor' comment blocks and the env switches they read (file:line)"
  grep -n -E '^# (thor|THOR_|THOR )' "$SC" | cut -c1-140 | sed 's/^/  sitecustomize.py:/'
  echo "# env vars read by the recipe sitecustomize.py but not by sfxnz's:"
  comm -23 <(grep -o -E '"(THOR|DSV41|XMOE|DENSE)_[A-Z0-9_]+"' "$SC" | sort -u) <(grep -o -E '"(THOR|DSV41|XMOE|DENSE)_[A-Z0-9_]+"' "$SX" | sort -u) | tr '\n' ' ' | fold -w 140 | sed 's/^/  /'; echo
  echo "# hunks of the sfxnz part (lines 1-150 of the recipe file) that differ from sfxnz:"
  diff "$SX" <(head -150 "$SC") | grep -E '^[0-9]' | tr '\n' ' ' | sed 's/^/  /'; echo
} > "$EV/sitecustomize_vs_sfxnz.txt"

# ---------------------------------------------------------------------------------------------------------------------
log "licenses"
LH=$EV/license_heads.txt
{
  echo "# First lines of each license file at the pinned commit, plus declared SPDX/metadata"
  lic() { local label=$1 f=$2; echo "## $label  ($f)"; if [[ -f $f ]]; then head -4 "$f" | sed 's/^/  | /'; echo "  sha256 $(sha256sum "$f" | cut -c1-16)..."; else echo "  (absent)"; fi; }
  lic "vLLM 36768d1 LICENSE" "$AUDIT_SRC/vllm/LICENSE"
  echo "  SPDX headers in deepseek_v41 sources: $(grep -rhoE 'SPDX-License-Identifier: [A-Za-z0-9.-]+' "$AUDIT_SRC/vllm/vllm/models/deepseek_v41" | sort | uniq -c | tr -s ' ' | paste -sd ';' -)"
  lic "vllm-exl3 d3cfd394 LICENSE" "$AUDIT_SRC/vllm-exl3/LICENSE"
  lic "vllm-exl3 d3cfd394 LICENSE.APACHE-2.0 (prior releases)" "$AUDIT_SRC/vllm-exl3/LICENSE.APACHE-2.0"
  echo "  pyproject.toml: $(grep -E '^license' "$AUDIT_SRC/vllm-exl3/pyproject.toml" | paste -sd ' ' -)"
  echo "  relicense commits: $(git -C "$AUDIT_SRC/vllm-exl3" log --format='%h %cs %s' -- LICENSE | paste -sd ';' -)"
  lic "vllm-exl3 NOTICE" "$AUDIT_SRC/vllm-exl3/NOTICE"
  lic "ExLlamaV3 5be88657 LICENSE" "$AUDIT_SRC/exllamav3/LICENSE"
  lic "sfxnz 460ac89c LICENSE" "$AUDIT_SRC/sfxnz-recipe/LICENSE"
  echo "  recipe docker/LICENSE.sfxnz-recipe identical to sfxnz LICENSE: $(cmp -s "$RECIPE/docker/LICENSE.sfxnz-recipe" "$AUDIT_SRC/sfxnz-recipe/LICENSE" && echo yes || echo no)"
  echo "  sfxnz LICENSE identical to deepseek-ai/DeepSeek-V4.1-Flash LICENSE: $(cmp -s "$AUDIT_SRC/sfxnz-recipe/LICENSE" "$AUDIT_SRC/hf-meta/deepseek-ai__DeepSeek-V4.1-Flash/LICENSE" && echo yes || echo no)"
  echo "  sfxnz LICENSE history: $(git -C "$AUDIT_SRC/sfxnz-recipe" log --format='%h %cs %an' -- LICENSE | paste -sd ';' -)"
  lic "Mia's AI Lab GLM-5.3 HEAD LICENSE" "$AUDIT_SRC/mia-glm53/LICENSE"
  lic "Mia's AI Lab GLM-5.3 HEAD LICENSE.MIT" "$AUDIT_SRC/mia-glm53/LICENSE.MIT"
  echo "  license history: $(git -C "$AUDIT_SRC/mia-glm53" log --format='%h %cs %s' -- LICENSE LICENSE.MIT | paste -sd ';' -)"
  lic "Diffbot recipe/LICENSE" "$RECIPE/LICENSE"
  lic "Diffbot recipe/patches/vllm-exl3/LICENSE" "$RECIPE/patches/vllm-exl3/LICENSE"
  echo "  recipe/patches/vllm-exl3/LICENSE identical to vllm-exl3 LICENSE: $(cmp -s "$RECIPE/patches/vllm-exl3/LICENSE" "$AUDIT_SRC/vllm-exl3/LICENSE" && echo yes || echo no)"
  lic "Diffbot recipe/kernels/xmoe/LICENSE.exllamav3" "$RECIPE/kernels/xmoe/LICENSE.exllamav3"
  lic "Diffbot HF repo top-level LICENSE (weights)" "$HF/LICENSE"
  for d in "$AUDIT_SRC"/hf-meta/*; do lic "HF $(basename "$d" | sed 's/__/\//') LICENSE" "$d/LICENSE"; done
  echo "## Hugging Face repo metadata (api/models/<repo>/revision/<rev>)"
  for f in "$HF/.revision.json" "$AUDIT_SRC"/hf-meta/*/revision.json; do
    python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); c = d.get("cardData") or {}
print(f"  {d.get('id')}@{d.get('sha')}: gated={d.get('gated')} private={d.get('private')} card license={c.get('license')} "
      f"license_name={c.get('license_name')} base_model={c.get('base_model')} lastModified={d.get('lastModified')}")
PY
  done
  echo "  model card license lines (Diffbot README.md): $(grep -n -i -E '^license:|MIT license|AGPL' "$HF/README.md" | cut -c1-160 | paste -sd '|' -)"
} > "$LH"

# ---------------------------------------------------------------------------------------------------------------------
log "inferencekey scripts vs recipe"
IKV=$EV/inferencekey_vs_recipe.txt
ikfile() {  # path relative to vllm-exl3/ in the inferencekey repo; prints a readable file path
  local p=$IK_COPY/vllm-exl3/$1
  [[ -f $p ]] && { echo "$p"; return; }
  [[ -d $IK_REPO/.git ]] && { local t; t=$(mktemp); git -C "$IK_REPO" show "$IK_COMMIT:vllm-exl3/$1" > "$t" && echo "$t"; }
}
{
  echo "# inferencekey-deepseek@$IK_COMMIT scripts vs the Diffbot recipe @a1cf1bd2"
  if [[ -d $IK_REPO/.git ]]; then
    echo "## in-repo copies ($IK_COPY) vs git show $IK_COMMIT:"
    for f in $(git -C "$IK_REPO" ls-tree -r --name-only "$IK_COMMIT" vllm-exl3/a100/scripts vllm-exl3/pro6000-concurrency/scripts); do
      git -C "$IK_REPO" show "$IK_COMMIT:$f" | cmp -s - "$IK_COPY/$f" && s=same || s=DIFFERENT; echo "  $s $f"
    done
  fi
  for d in a100 pro6000-concurrency; do
    A=$(ikfile $d/scripts/serve.sh); B=$(ikfile $d/scripts/bootstrap.sh)
    echo; echo "################ $d"
    echo "## serve.sh vs recipe serve/serve.sh: env knobs and vllm flags (mechanical extraction)"
    python3 - "$RECIPE/serve/serve.sh" "$A" <<'PY'
import re, sys, shlex
def extract(path):
    s = open(path).read().replace("\\\n", " ")
    env, flags = {}, {}
    for m in re.finditer(r"\b([A-Z][A-Z0-9_]+)=\$\{\1:?-([^}]*)\}", s):          # NAME=${NAME:-v}
        env.setdefault(m.group(1), m.group(2).strip('"'))
    for line in s.splitlines():
        t = line.strip()
        if t.startswith("export "):
            for tok in shlex.split(t[7:], comments=True):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    m = re.fullmatch(r"\$\{" + k + r":?-([^}]*)\}", v)
                    env[k] = m.group(1) if m else v
    for m in re.finditer(r"-e ([A-Z][A-Z0-9_]+)=(\S+)", s):     # also inside ${X:+-e K=V}: drop the closing brace
        v = m.group(2)
        v = v[:-1] if v.endswith("}") and "{" not in v else v
        if v.strip('"') != "$" + m.group(1):                  # -e K=$K just forwards the default read above
            env[m.group(1)] = v
    for m in re.finditer(r"(?<![\w-])(--[a-z][a-z0-9-]+)(?:[ =]('[^']*'|\"[^\"]*\"|[^\s-][^\s]*))?", s):
        flags.setdefault(m.group(1), (m.group(2) or "").strip())
    return env, flags
(re_, rf), (ie, iflags) = extract(sys.argv[1]), extract(sys.argv[2])
print("  env/default            recipe                                  inferencekey")
for k in sorted(set(re_) | set(ie)):
    a, b = re_.get(k, "-"), ie.get(k, "-")
    mark = "  " if a == b else "!="
    print(f"  {mark} {k:<30} {a[:44]:<46} {b[:44]}")
print("  vllm/docker flags      recipe | inferencekey")
for k in sorted(set(rf) | set(iflags)):
    if k in rf and k in iflags: st = "both"
    elif k in rf: st = "recipe only"
    else: st = "inferencekey only"
    print(f"  {k:<36} {st}")
PY
    echo "## diff -u recipe serve/serve.sh  inferencekey $d/scripts/serve.sh  (stat)"
    diff -u "$RECIPE/serve/serve.sh" "$A" | { grep -c '^+[^+]' || true; } | sed 's/^/  added lines: /'
    diff -u "$RECIPE/serve/serve.sh" "$A" | { grep -c '^-[^-]' || true; } | sed 's/^/  removed lines: /'
    echo "## bootstrap.sh vs recipe docker/build.sh + kernels/build.sh: build steps"
    steps() { grep -o -E 'patch -p1[^&;<]*< *[^ &;]+|git clone[^&]*vllm-exl3[^ ]*|git clone[^&]*exllamav3[^ ]*|checkout [-q ]*"?\$?[A-Za-z0-9_]+"?|fix_vllm_exl3_setup.py|pip install[^&;]*|build_xmoe(_do)?\.py[^&;]*|apt-get install[^&;]*|sitecustomize.py' "$1" | sed -E 's/ +/ /g; s|/(mnt/patches\|opt/recipe/patches)/|<patches>/|' ; }
    echo "  recipe docker/build.sh + kernels/build.sh:"; { steps "$RECIPE/docker/build.sh"; steps "$RECIPE/kernels/build.sh"; } | sed 's/^/    /'
    echo "  inferencekey $d/scripts/bootstrap.sh:"; steps "$B" | sed 's/^/    /'
    echo "  vLLM patch order in bootstrap.sh: $(grep -o -E 'for p in [^;]*' "$B" | sed 's/for p in //')"
    echo "  vLLM patch order in recipe build.sh: $(grep 'patch -p1' "$RECIPE/docker/build.sh" | grep -o -E 'patches/vllm/[a-z0-9-]+\.diff' | sed 's|patches/vllm/||; s|\.diff||' | paste -sd ' ' -)"
    echo "  pins in bootstrap.sh: $(grep -o -E '[0-9a-f]{40}' "$B" | sort -u | paste -sd ' ' -)"
    echo "  pins in recipe build.sh: $(grep -o -E '[0-9a-f]{40}' "$RECIPE/docker/build.sh" | sort -u | paste -sd ' ' -)"
  done
  echo; echo "## pro6000-concurrency: settings passed at launch (from logs/serve.log, not in the committed serve.sh)"
  L=$(ikfile pro6000-concurrency/logs/serve.log)
  [[ -n $L ]] && grep -o -E "max_num_seqs': [0-9]+|cudagraph_capture_sizes': \[[0-9, ]+\]|'max_model_len': [0-9]+|enable_prefix_caching=[A-Za-z]+" "$L" | sort -u | sed 's/^/  /'
  R6=$(ikfile pro6000-concurrency/README.md)
  [[ -n $R6 ]] && { echo "## pro6000-concurrency/README.md launch line and stated changes from the recipe:"; grep -n -E 'MAX_NUM_SEQS=64|bash serve.sh|Two changes from the recipe' "$R6" | sed 's/^/  README.md:/'; }
  RA=$(ikfile a100/README.md)
  [[ -n $RA ]] && { echo "## a100/README.md statements about the recipe:"; grep -n -i -E 'diffbot|recipe' "$RA" | cut -c1-200 | sed 's/^/  README.md:/'; }
  echo "## a100/scripts/serve.sh header:"; sed -n 2p "$(ikfile a100/scripts/serve.sh)" | sed 's/^/  serve.sh:2: /'
  echo "## a100/scripts/bootstrap.sh header:"; sed -n 2,3p "$(ikfile a100/scripts/bootstrap.sh)" | sed 's/^/  bootstrap.sh: /'
  echo; echo "## inferencekey files with no recipe counterpart (scripts/ of both runs)"
  for f in a100/scripts/bench.py a100/scripts/download.py a100/scripts/pod.sh a100/scripts/rp_create_loop.py a100/scripts/rp_stock2.py \
           a100/scripts/watchdog.sh pro6000-concurrency/scripts/loadgen.py pro6000-concurrency/scripts/mem_sampler.py \
           pro6000-concurrency/scripts/summarize.py pro6000-concurrency/scripts/run_all.sh pro6000-concurrency/scripts/restart_A.sh \
           pro6000-concurrency/scripts/download.py pro6000-concurrency/scripts/rp_create_loop.py pro6000-concurrency/scripts/pod.sh \
           pro6000-concurrency/scripts/watchdog.sh; do
    p=$(ikfile "$f"); [[ -n $p ]] || continue
    printf '  %-48s %5s lines  %s\n' "$f" "$(wc -l < "$p")" "$(grep -m1 -E '^(#|""")' "$p" | grep -v '^#!' | cut -c1-110 || true)$(sed -n 2p "$p" | grep -E '^(#|""")' | cut -c1-110 || true)"
  done
  echo "  a100 vs pro6000 copies: $(for f in serve.sh bootstrap.sh download.py pod.sh rp_create_loop.py watchdog.sh; do cmp -s "$(ikfile a100/scripts/$f)" "$(ikfile pro6000-concurrency/scripts/$f)" && printf '%s=same ' $f || printf '%s=DIFF ' $f; done)"
} > "$IKV"

# Excerpts quoted from third-party files may contain en or em dashes; normalize them to ASCII so the evidence stays plain.
for f in "$EV"/*.txt "$EV"/*.tsv "$EV"/*.json; do
  python3 - "$f" <<'PY'
import sys
p = sys.argv[1]; s = open(p, encoding="utf-8", errors="replace").read()
t = s.replace("\u2014", "-").replace("\u2013", "-")
if t != s: open(p, "w", encoding="utf-8").write(t)
PY
done
log "done; evidence in $EV"
ls -la "$EV" >&2
