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

REGRESSION_PTS="${AE_BENCH_REGRESSION_PTS:-0.05}"

mkdir -p "$HIST" "$LOGS"

TS=$(date +%F-%H%M%S)
OUT="${HIST}/ae-domain-${TS}.json"
LOG="${LOGS}/ae-domain-bench.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "AE-domain bench START → ${OUT}"
cd "$REPO"

# Run bench. --mode scored uses ground_truth_ids in queries.py (33 labeled
# as of HEAD f6f9193). Reranker is enabled (production retrieval profile).
"$PY" benchmarks/ae_domain_memory_bench/run_ae_domain_bench.py \
    --mode scored \
    --rerank \
    --out "$OUT" >> "$LOG" 2>&1
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

# Find previous AE-domain run for delta comparison.
PREV=$(ls -t "${HIST}"/ae-domain-*.json 2>/dev/null | grep -v "$(basename "$OUT")" | head -1)

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
