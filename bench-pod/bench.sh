#!/usr/bin/env bash
# Comparison Pod — bootstrap entrypoint.
#
# Usage:
#   bash <(curl -fsSL https://mazemaker.dev/bench.sh)
#   bash bench.sh                  # full matrix
#   bash bench.sh --only=mazemaker # only the Mazemaker reference run
#   bash bench.sh --skip-fetch     # use already-fetched datasets
#
# Idempotent. Refuses to run if pre-flight fails. Never invokes sudo.
# Never invokes docker.

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate self. Two modes: piped from curl (script is /dev/fd/63 or similar,
# repo not present), or executed from a clone (script lives in the repo).
# ---------------------------------------------------------------------------

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
if [ -f "$SCRIPT_PATH" ] && [ -d "$(dirname "$SCRIPT_PATH")/scripts" ]; then
    POD_ROOT="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
    PIPED_MODE=0
else
    POD_ROOT=""
    PIPED_MODE=1
fi

VERSION="0.1.0"
WORK_DIR="${DEMOLITION_POD_WORK:-$HOME/.bench-pod}"
ONLY=""
SKIP_FETCH=0
SKIP_PREFLIGHT=0

log()  { printf '[demolish] %s\n' "$*" >&2; }
fail() { printf '[demolish][FAIL] %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

for arg in "$@"; do
    case "$arg" in
        --only=*)        ONLY="${arg#*=}" ;;
        --skip-fetch)    SKIP_FETCH=1 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --version)       printf '%s\n' "$VERSION"; exit 0 ;;
        --help|-h)
            cat <<EOF
Comparison Pod v$VERSION

  --only=<system>      Run a single runner (mazemaker|hindsight|letta|mem0|amem|cognee)
  --skip-fetch         Skip dataset fetch/verify
  --skip-preflight     Skip the runtime preflight (dangerous)
  --version            Print version and exit
  --help               Print this message and exit

Outputs under: $WORK_DIR
EOF
            exit 0
            ;;
        *) fail "unknown argument: $arg" ;;
    esac
done

# ---------------------------------------------------------------------------
# Piped-mode handling: clone the repo locally so we have the scripts/
# directory available. We never write outside $WORK_DIR.
# ---------------------------------------------------------------------------

if [ "$PIPED_MODE" -eq 1 ]; then
    log "piped mode: cloning repo into $WORK_DIR/src"
    mkdir -p "$WORK_DIR/src"
    if [ ! -d "$WORK_DIR/src/.git" ]; then
        command -v git >/dev/null 2>&1 || fail "git not found — required in piped mode"
        git clone --depth 1 "https://github.com/itsXactlY/mazemaker" "$WORK_DIR/src" \
            || fail "git clone failed"
    else
        log "repo already present, pulling"
        (cd "$WORK_DIR/src" && git pull --ff-only) || log "git pull failed; using existing checkout"
    fi
    POD_ROOT="$WORK_DIR/src/bench-pod"
fi

[ -d "$POD_ROOT" ] || fail "pod root not found: $POD_ROOT"
cd "$POD_ROOT"

log "pod root:    $POD_ROOT"
log "work dir:    $WORK_DIR"
log "version:     $VERSION"
log "only:        ${ONLY:-<all>}"

mkdir -p "$WORK_DIR/results" "$WORK_DIR/datasets" "$WORK_DIR/logs"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

if [ "$SKIP_PREFLIGHT" -eq 0 ]; then
    log "running preflight: scripts/detect_runtime.sh"
    bash "$POD_ROOT/scripts/detect_runtime.sh" \
        || fail "preflight failed — fix the reported issues and retry, or pass --skip-preflight (you have been warned)"
fi

# ---------------------------------------------------------------------------
# Fetch + verify datasets
# ---------------------------------------------------------------------------

if [ "$SKIP_FETCH" -eq 0 ]; then
    if command -v python3 >/dev/null 2>&1; then
        PY=python3
    elif command -v python >/dev/null 2>&1; then
        PY=python
    else
        fail "python3 not found — required for dataset fetch"
    fi
    log "fetching + verifying datasets"
    "$PY" "$POD_ROOT/datasets/fetch.py" --work "$WORK_DIR" \
        || fail "dataset fetch failed (see logs)"
fi

# ---------------------------------------------------------------------------
# Install Quadlet manifest (best-effort — only if systemd --user is alive)
# ---------------------------------------------------------------------------

if command -v systemctl >/dev/null 2>&1 && systemctl --user --quiet is-system-running >/dev/null 2>&1 \
   || systemctl --user list-units >/dev/null 2>&1; then
    log "installing Quadlet manifest"
    bash "$POD_ROOT/scripts/install_quadlet.sh" || log "Quadlet install non-fatal failure (will run runners directly)"
else
    log "no systemd --user; running runners directly without Quadlet"
fi

# ---------------------------------------------------------------------------
# Run runners
# ---------------------------------------------------------------------------

ALL_RUNNERS=(mazemaker hindsight letta mem0 amem cognee)
if [ -n "$ONLY" ]; then
    case " ${ALL_RUNNERS[*]} " in
        *" $ONLY "*) RUNNERS=("$ONLY") ;;
        *) fail "unknown runner: $ONLY (valid: ${ALL_RUNNERS[*]})" ;;
    esac
else
    RUNNERS=("${ALL_RUNNERS[@]}")
fi

export DEMOLITION_POD_VERSION="$VERSION"
export DEMOLITION_POD_WORK="$WORK_DIR"
# Hardfact §LLM: route all LLM extraction to in-pod ollama; disable OpenAI.
export OPENAI_API_KEY=""
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

cd "$POD_ROOT"
for r in "${RUNNERS[@]}"; do
    log "=== runner: $r ==="
    log_file="$WORK_DIR/logs/${r}.log"
    if python3 -m "runners.${r}_runner" --work "$WORK_DIR" >"$log_file" 2>&1; then
        log "  OK -> $WORK_DIR/results/${r}.json (log: $log_file)"
    else
        rc=$?
        # NotImplementedError is the expected stub status — surface as PENDING
        # to the comparator, not a hard failure.
        if grep -q "NotImplementedError" "$log_file"; then
            log "  PENDING (stub runner — see $log_file)"
        else
            log "  ERROR rc=$rc — see $log_file (continuing; comparator will mark ERROR)"
        fi
    fi
done

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

log "comparing results"
python3 -m compare.compare \
    --results-dir "$WORK_DIR/results" \
    --out-dir "$WORK_DIR" \
    || fail "comparator failed"

log "=== done ==="
log "matrix.md  : $WORK_DIR/matrix.md"
log "matrix.json: $WORK_DIR/matrix.json"
log "verdict.md : $WORK_DIR/verdict.md"
