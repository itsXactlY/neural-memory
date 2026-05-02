#!/bin/bash
# codex_apply.sh — triage + apply codex-resolver proposals.
#
# Per Tito 2026-05-02 "compound usefulness": the self-fix loop now is
# orchestrator-finds → resolver-proposes → THIS-tool-applies → reviewer-checks.
# Builder runs this to pick the right proposals + apply with `git apply`.
#
# Usage:
#   tools/codex_apply.sh                    # list all pending proposals
#   tools/codex_apply.sh --ready             # filter to READY-TO-APPLY verdicts only
#   tools/codex_apply.sh --show <file>       # show full proposal + extracted diff
#   tools/codex_apply.sh --check <file>      # extract diff + git apply --check (dry run)
#   tools/codex_apply.sh --apply <file>      # actually apply the diff (after confirmation)
#
# Safety:
#   - Default lists only; no mutation
#   - --check is read-only (git apply --check)
#   - --apply runs git apply only after y/n confirmation + working-tree-clean check
#   - Sandbox: relies on builder review; codex itself is read-only when proposing

set -uo pipefail

REPO="/Users/tito/lWORKSPACEl/research/neural-memory"
LANE="${NM_LANE:-nm-builder}"
PROPOSAL_DIR="${HOME}/.neural_memory/codex-resolutions/${LANE}"

cmd="${1:-list}"
arg="${2:-}"

list_proposals() {
    local filter="${1:-}"
    if [ ! -d "$PROPOSAL_DIR" ]; then
        echo "(no proposals dir at $PROPOSAL_DIR)"
        return
    fi
    local files=("$PROPOSAL_DIR"/*.md)
    if [ ! -e "${files[0]}" ]; then
        echo "(no proposal files in $PROPOSAL_DIR)"
        return
    fi
    printf "%-32s %-22s %s\n" "FILE" "VERDICT" "TOPIC"
    printf "%-32s %-22s %s\n" "$(printf '%.0s-' {1..32})" "$(printf '%.0s-' {1..22})" "$(printf '%.0s-' {1..40})"
    for f in "$PROPOSAL_DIR"/*.md; do
        [ -f "$f" ] || continue
        local verdict
        verdict=$(grep -oE "READY-TO-APPLY|NEEDS-REVIEW|NOT-A-CODE-FIX|INSUFFICIENT EVIDENCE" "$f" 2>/dev/null | tail -1)
        verdict="${verdict:-UNKNOWN}"
        if [ -n "$filter" ] && [ "$verdict" != "$filter" ]; then
            continue
        fi
        local base
        base=$(basename "$f" .md)
        local topic
        topic=$(echo "$base" | sed -E 's/^[0-9]+-[0-9]+-//')
        printf "%-32s %-22s %s\n" "$base.md" "$verdict" "$topic"
    done
}

extract_diff() {
    # Pull all ```diff ... ``` blocks out of a proposal markdown file
    awk '/^```diff$/{flag=1; next} /^```$/{if (flag) {flag=0; print "---NEXT---"}} flag' "$1"
}

show_proposal() {
    local f="$1"
    if [ ! -f "$f" ]; then
        echo "ERROR: file not found: $f" >&2
        exit 2
    fi
    echo "=== Full proposal: $f ==="
    cat "$f"
    echo ""
    echo "=== Extracted diff blocks ==="
    extract_diff "$f"
}

check_proposal() {
    local f="$1"
    if [ ! -f "$f" ]; then
        echo "ERROR: file not found: $f" >&2
        exit 2
    fi
    local tmp
    tmp=$(mktemp -t codex-apply-check.XXXXXX.diff)
    extract_diff "$f" | sed '/^---NEXT---$/d' > "$tmp"
    if [ ! -s "$tmp" ]; then
        echo "ERROR: no diff block found in $f" >&2
        rm -f "$tmp"
        exit 3
    fi
    echo "=== Extracted diff ($tmp) ==="
    cat "$tmp"
    echo ""
    echo "=== git apply --check ==="
    cd "$REPO" && git apply --check "$tmp"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "✓ Patch applies cleanly. Run with --apply to actually apply."
    else
        echo "✗ Patch does NOT apply cleanly (rc=$rc). Manual fix needed."
    fi
    rm -f "$tmp"
    exit $rc
}

apply_proposal() {
    local f="$1"
    if [ ! -f "$f" ]; then
        echo "ERROR: file not found: $f" >&2
        exit 2
    fi
    cd "$REPO" || exit 1
    if [ -n "$(git status --porcelain | grep -v '^?? ')" ]; then
        echo "ERROR: working tree has uncommitted changes. Stash or commit first." >&2
        git status --short
        exit 4
    fi
    local tmp
    tmp=$(mktemp -t codex-apply.XXXXXX.diff)
    extract_diff "$f" | sed '/^---NEXT---$/d' > "$tmp"
    if [ ! -s "$tmp" ]; then
        echo "ERROR: no diff block found in $f" >&2
        rm -f "$tmp"
        exit 3
    fi
    echo "=== About to apply this diff ==="
    cat "$tmp"
    echo ""
    read -r -p "Apply? [y/N] " yn
    if [ "$yn" != "y" ] && [ "$yn" != "Y" ]; then
        echo "Aborted."
        rm -f "$tmp"
        exit 0
    fi
    git apply "$tmp"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "✓ Applied. Review with 'git diff', then commit with a 'fix(codex-resolver):' prefix."
    else
        echo "✗ git apply failed (rc=$rc)"
    fi
    rm -f "$tmp"
    exit $rc
}

case "$cmd" in
    list|"")        list_proposals "" ;;
    --ready)        list_proposals "READY-TO-APPLY" ;;
    --show)         show_proposal "$arg" ;;
    --check)        check_proposal "$arg" ;;
    --apply)        apply_proposal "$arg" ;;
    -h|--help|help)
        sed -n '2,/^$/p' "$0" | head -25
        ;;
    *)
        echo "Unknown command: $cmd" >&2
        echo "Usage: $0 [list|--ready|--show <file>|--check <file>|--apply <file>]" >&2
        exit 2
        ;;
esac
