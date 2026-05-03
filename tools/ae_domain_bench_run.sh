#!/usr/bin/env bash
# F9 — AE-domain bench runner (always-on, fired by AOR daemon).
#
# Per AE-builder request msg_f327a6b7-9593-4607-b1e1-e6fdef14414c (2026-05-02):
# add 240-query AE bench to always-on schedule, write to bench-history,
# alert AE-builder via bridge if R@5 drops > 0.05 cycle-to-cycle.
#
# Tito META rule: "NOT NO TIME BASED THINGS" — wired as AOR (always-on
# runner with min-interval), NOT StartCalendarInterval cron.
#
# Schedule: AOR fires on git HEAD change OR 24h max-interval guarantee.
# (Earlier comment claimed substrate db mtime triggers — incorrect, the
# AOR wrapper at tools/codex_always_on_runner.sh:49-74 only checks HEAD +
# first-run + max-interval. Caught by per-commit reviewer of b0b71cf.)
# See com.ae.ae-domain-bench-aor.plist.

set -uo pipefail

REPO="/Users/tito/lWORKSPACEl/research/neural-memory"
HIST="${HOME}/.neural_memory/bench-history"
LOGS="${HOME}/.neural_memory/logs"
PY="${HOME}/.hermes/hermes-agent/venv/bin/python3"
BRIDGE_CLI="/Users/tito/lWORKSPACEl/Projects/AngelsElectric/LangGraph/plugins/hermes-skills-layer/plugins/claude-hermes-bridge/scripts/agent_bridge_mcp.mjs"

# Canonical production substrate path. Hard-coded so the bench cannot drift
# into using a copy/ablation DB. Enforced both as the --db arg passed to the
# bench script AND as the substrate-path requirement in eligibility filtering
# of prior artifacts.
CANONICAL_DB="${HOME}/.neural_memory/memory.db"

REGRESSION_PTS="${AE_BENCH_REGRESSION_PTS:-0.05}"

# Sonnet diagnostic 2026-05-02 [verified-now]: bench Spanish misses are
# pure translator-not-set issue (NM_SPANISH_TRANSLATE wires Spanish queries
# through the EN dict before retrieval; commit a0e7374 wired it but the
# bench env didn't enable it). Predicted lift: +0.09 R@5 absolute.
# Override at AOR plist EnvironmentVariables level to disable.
export NM_SPANISH_TRANSLATE="${NM_SPANISH_TRANSLATE:-1}"

# F9 shell authority (Sonnet S1 packet 2026-05-03): pick the prior bench
# artifact ONLY if it is a current-format scored production artifact. Per
# synth contract: reject stale HEAD, old schema (no provenance), copy/
# ablation tagged filenames, missing per_query rows, missing model/env/
# query provenance, null substrate counts, or db_path != canonical.
#
# Eligibility is computed in a single python heredoc so the logic can be
# unit-tested via `bash tools/ae_domain_bench_run.sh --select-eligible-prev
# <dir> [--current-head <sha>]`, which prints the chosen artifact path
# (empty if none) and exits without mutating substrate.
_select_eligible_prev() {
    local dir="$1"
    local current_head="$2"
    "$PY" - "$dir" "$CANONICAL_DB" "$current_head" <<'PY'
import json, os, re, sys
from pathlib import Path

bench_dir = Path(sys.argv[1])
canonical_db = sys.argv[2]
current_head = sys.argv[3] or ""

# Reject filenames carrying ablation/copy markers regardless of contents.
TAG_BLACKLIST = ("bge-small", "copy", "ablation", "mpnet", "clean", "scorer-")

def eligible(path: Path) -> tuple[bool, str]:
    name = path.name
    low = name.lower()
    for tag in TAG_BLACKLIST:
        if tag in low:
            return False, f"tag:{tag}"
    try:
        with path.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:
        return False, f"unreadable:{type(e).__name__}"
    if not isinstance(d, dict):
        return False, "not-dict"
    if d.get("mode") != "scored":
        return False, f"mode={d.get('mode')!r}"
    prov = d.get("provenance")
    if not isinstance(prov, dict):
        return False, "no-provenance"
    db_path = prov.get("db_path")
    if not isinstance(db_path, str) or db_path == "(default)":
        return False, f"db_path={db_path!r}"
    # Accept canonical path either by exact match or by canonical suffix.
    if not (db_path == canonical_db or db_path.endswith("/.neural_memory/memory.db")):
        return False, f"db_path-non-canonical:{db_path}"
    git_head = prov.get("git_head")
    if not isinstance(git_head, str) or not git_head or git_head == "unknown":
        return False, f"git_head={git_head!r}"
    # Stale HEAD rejection: only enforced when caller passed a current_head.
    # If current_head is empty, we skip this check (cron may pre-fetch).
    if current_head and git_head != current_head:
        return False, f"stale-head:{git_head[:8]}!={current_head[:8]}"
    sub = prov.get("substrate_counts") or {}
    if sub.get("memories") in (None, 0):
        return False, f"substrate_memories={sub.get('memories')!r}"
    if sub.get("connections_active") in (None,):
        return False, f"substrate_conn={sub.get('connections_active')!r}"
    models = prov.get("models") or {}
    if not models.get("embedding_model"):
        return False, f"embedding_model={models.get('embedding_model')!r}"
    env = prov.get("env") or {}
    # env block must exist (values may legitimately be None for unset vars
    # but the dict itself proves env was captured).
    if not isinstance(env, dict):
        return False, "env-missing"
    if not prov.get("query_file_md5"):
        return False, "query_file_md5-missing"
    pq = d.get("per_query") or []
    if not isinstance(pq, list) or not pq:
        return False, f"per_query-empty(len={len(pq) if isinstance(pq, list) else 'n/a'})"
    if "category_regression_gate" not in d:
        return False, "category_regression_gate-missing"
    return True, "ok"

candidates = sorted(bench_dir.glob("ae-domain-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
chosen = ""
for p in candidates:
    ok, reason = eligible(p)
    if ok:
        chosen = str(p)
        break

# Print the chosen path on stdout (empty if none). Decision trace on stderr.
print(chosen)
for p in candidates:
    ok, reason = eligible(p)
    print(f"{'OK ' if ok else 'NO '} {p.name}: {reason}", file=sys.stderr)
PY
}

# --- Test/dry-run mode -----------------------------------------------------
# Exposed for unit tests + manual diagnostics. Does NOT touch substrate,
# does NOT run the bench, does NOT write artifacts. Just exercises the
# eligibility filter and prints the chosen prev artifact (empty if none).
if [ "${1:-}" = "--select-eligible-prev" ]; then
    shift
    _dir="${1:-${HIST}}"
    _head=""
    if [ "${2:-}" = "--current-head" ] && [ -n "${3:-}" ]; then
        _head="$3"
    fi
    _select_eligible_prev "$_dir" "$_head"
    exit $?
fi

mkdir -p "$HIST" "$LOGS"

TS=$(date +%F-%H%M%S)
OUT="${HIST}/ae-domain-${TS}.json"
LOG="${LOGS}/ae-domain-bench.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "AE-domain bench START → ${OUT}"
cd "$REPO"

# F9 Bug B fix (rev 2 — Sonnet S1 packet 2026-05-03): replace raw `ls -t`
# with eligibility-filtered selection. Picks newest current-format scored
# production artifact (provenance + canonical db + per_query + non-null
# substrate counts + matching git HEAD). Falls back to no --prev-results
# when no eligible artifact exists, rather than crashing.
CURRENT_HEAD=$(git rev-parse HEAD 2>/dev/null || echo "")
PREV_FOR_GATE=$(_select_eligible_prev "$HIST" "$CURRENT_HEAD" 2>>"$LOG")
PREV_ARG=()
if [ -n "${PREV_FOR_GATE}" ] && [ -f "${PREV_FOR_GATE}" ]; then
    PREV_ARG=(--prev-results "${PREV_FOR_GATE}")
    log "per-category regression gate: prev=${PREV_FOR_GATE} (eligibility-filtered)"
else
    log "per-category regression gate: no eligible prior artifact (first eligible run, or all priors rejected)"
fi

# Run bench. --mode scored uses ground_truth_ids in queries.py (33 labeled
# as of HEAD f6f9193). Reranker is enabled (production retrieval profile).
# --db pinned to canonical substrate so provenance.db_path is recorded
# explicitly (was "(default)" before, which made the artifact ineligible
# for self-comparison on the next cycle).
"$PY" benchmarks/ae_domain_memory_bench/run_ae_domain_bench.py \
    --mode scored \
    --rerank \
    --db "$CANONICAL_DB" \
    --out "$OUT" \
    "${PREV_ARG[@]}" >> "$LOG" 2>&1
RC=$?

# rc=0  → all categories passed threshold
# rc=2  → some categories failed threshold but bench DID run + wrote file (per
#         run_ae_domain_bench.py:233-234). Continue regression-detect logic.
# other → real runtime error; bail.
if [ "$RC" -ne 0 ] && [ "$RC" -ne 2 ]; then
    log "BENCH FAILED rc=${RC}"
    exit "$RC"
fi
if [ ! -f "$OUT" ]; then
    log "BENCH OUT FILE MISSING after rc=${RC}; bailing"
    exit 99
fi

# Pull latest R@5 from the JSON report. run_ae_domain_bench.py:168-177
# returns top-level 'global_r@5' (with @ symbol — needs dict-key access).
# The earlier 'overall.recall_at_5' was wrong; first attempted fix to
# result.global_r@5 was ALSO wrong (matched a wrapper-format artifact
# at ae-domain-2026-05-02-070800.json, not the actual scored-mode output).
LATEST_R5=$("$PY" -c "
import json
d = json.load(open('${OUT}'))
print(d.get('global_r@5', 'NA'))
" 2>/dev/null)

# Find previous AE-domain run for delta comparison. Reuse the same eligibility
# filter so the bridge alert isn't fired against a copy/ablation/legacy artifact.
# Exclude OUT itself (just-written) by scanning eligibility, then dropping OUT
# if it shows up first.
PREV_RAW=$(_select_eligible_prev "$HIST" "$CURRENT_HEAD" 2>/dev/null)
if [ "${PREV_RAW}" = "${OUT}" ]; then
    # Re-scan against a tmp dir excluding OUT.
    _TMP_HIST=$(mktemp -d -t aedombench-prev.XXXX)
    for f in "${HIST}"/ae-domain-*.json; do
        [ "$f" = "$OUT" ] && continue
        ln -s "$f" "${_TMP_HIST}/$(basename "$f")" 2>/dev/null
    done
    PREV=$(_select_eligible_prev "$_TMP_HIST" "$CURRENT_HEAD" 2>/dev/null)
    rm -rf "$_TMP_HIST"
else
    PREV="${PREV_RAW}"
fi

if [ -n "$PREV" ] && [ "$LATEST_R5" != "NA" ]; then
    PREV_R5=$("$PY" -c "
import json
d = json.load(open('${PREV}'))
print(d.get('global_r@5', 'NA'))
" 2>/dev/null)
    log "R@5: latest=${LATEST_R5} prev=${PREV_R5} (prev=${PREV})"

    # Send bridge alert to AE-builder if regression > threshold.
    DELTA=$("$PY" -c "
try:
    print(float('${LATEST_R5}') - float('${PREV_R5}'))
except Exception:
    print('NA')
" 2>/dev/null)
    if [ "$DELTA" != "NA" ]; then
        REGRESSED=$("$PY" -c "
print('YES' if (${PREV_R5} - ${LATEST_R5}) > ${REGRESSION_PTS} else 'NO')
" 2>/dev/null)
        if [ "$REGRESSED" = "YES" ]; then
            log "ALERT REGRESSION ${DELTA} (threshold ${REGRESSION_PTS})"
            node "$BRIDGE_CLI" send \
                --from claude-code-nm-builder \
                --to claude-code-ae-builder \
                --urgency high \
                --subject "AE-domain bench REGRESSION ${DELTA} (R@5 ${PREV_R5}→${LATEST_R5})" \
                --body "AE-domain bench R@5 dropped ${DELTA} (threshold ${REGRESSION_PTS}). Latest: ${OUT}. Previous: ${PREV}. Substrate state may have shifted. Recent NM HEAD: $(cd "$REPO" && git rev-parse --short HEAD). Investigate substrate ingest or recent helper changes." \
                >> "$LOG" 2>&1 || log "bridge send failed (non-fatal)"
        fi
    fi
else
    log "R@5: latest=${LATEST_R5} (no prior run for delta)"
fi

log "AE-domain bench DONE rc=${RC} (rc=2 means category-fail, file still valid)"
# Exit 0 always when file is valid — let regression-alert fire on real drops
# rather than launchd retry-storming on each threshold-fail.
exit 0
