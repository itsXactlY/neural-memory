#!/usr/bin/env bash
# Sequence: fetch -> install quadlet -> run all runners -> compare.
# This is the underlying machinery `bench.sh` wraps; exposing it
# separately makes a "re-run just the comparator" workflow trivial.

set -euo pipefail

POD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${DEMOLITION_POD_WORK:-$HOME/.bench-pod}"
PY="${PY:-python3}"

log() { printf '[run-all] %s\n' "$*"; }

mkdir -p "$WORK/results" "$WORK/datasets" "$WORK/logs"

log "1/4 fetch + verify datasets"
"$PY" "$POD_ROOT/datasets/fetch.py" --work "$WORK"

log "2/4 install quadlet manifest (best-effort)"
bash "$POD_ROOT/scripts/install_quadlet.sh" || log "  (skipped)"

log "3/4 run runners"
cd "$POD_ROOT"
for r in mazemaker hindsight letta mem0 amem cognee; do
    log "  runner: $r"
    "$PY" -m "runners.${r}_runner" --work "$WORK" >"$WORK/logs/${r}.log" 2>&1 \
        || log "    rc!=0 (check $WORK/logs/${r}.log; comparator will mark accordingly)"
done

log "4/4 compare"
"$PY" -m compare.compare --results-dir "$WORK/results" --out-dir "$WORK"

log "done — see $WORK/matrix.md  and  $WORK/verdict.md"
