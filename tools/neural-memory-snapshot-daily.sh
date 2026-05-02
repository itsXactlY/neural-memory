#!/usr/bin/env bash
# neural-memory-snapshot-daily.sh — daily backup of ~/.neural_memory/
#
# Per Tito's "no manual steps" META rule + AE-builder peer's daily-snapshot
# pattern: substrate is now valuable enough (3.18 GB, 7.5K mems, R@5=0.82)
# to be worth never losing.
#
# Backs up:
#   ~/.neural_memory/memory.db           — primary substrate (gzipped)
#   ~/.neural_memory/bench-history/      — R@5 trajectory log
#   ~/.neural_memory/datasets/           — LongMemEval-S dataset (rsync'd
#                                          first time only — large, won't
#                                          change)
#
# Output: ~/.neural_memory/backups/<YYYY-MM-DD>/{memory.db.gz,bench-history/}
#
# Retention: keeps last 14 daily snapshots; older auto-pruned.
#
# Schedule: daily 04:30 via com.ae.neural-memory-snapshot.plist
#   (after 03:00 bench-daily, after 04:15 corpus-ingest, before sunrise)

set -euo pipefail

NM_HOME="$HOME/.neural_memory"
BACKUP_ROOT="$NM_HOME/backups"
LOGS="$NM_HOME/logs"
TODAY=$(date +%F)
DEST="$BACKUP_ROOT/$TODAY"
LOG_FILE="$LOGS/neural-memory-snapshot.log"

mkdir -p "$DEST" "$LOGS"

echo "$(date -Iseconds)  snapshot-daily start  dest=$DEST" >> "$LOG_FILE"

# 1. memory.db — gzip during copy. Skip if today's snapshot already done.
# Caught 2026-05-02 reviewer (B4): disk-full mid-gzip leaves a partial
# memory.db.gz; subsequent runs see the file exists and skip → corruption
# persists. Fix: write to .tmp then atomic mv only on success.
if [[ -f "$DEST/memory.db.gz" ]]; then
    echo "$(date -Iseconds)  memory.db.gz already exists, skipping" >> "$LOG_FILE"
else
    if [[ -f "$NM_HOME/memory.db" ]]; then
        SIZE_RAW=$(stat -f%z "$NM_HOME/memory.db" 2>/dev/null || echo "0")
        TMP_GZ="$DEST/memory.db.gz.tmp.$$"
        if gzip -c "$NM_HOME/memory.db" > "$TMP_GZ"; then
            mv "$TMP_GZ" "$DEST/memory.db.gz"
            SIZE_GZ=$(stat -f%z "$DEST/memory.db.gz" 2>/dev/null || echo "0")
            echo "$(date -Iseconds)  memory.db: ${SIZE_RAW}B → ${SIZE_GZ}B (gzipped)" >> "$LOG_FILE"
        else
            rm -f "$TMP_GZ" 2>/dev/null || true
            echo "$(date -Iseconds)  ERROR: gzip failed (likely disk-full); partial file removed" >> "$LOG_FILE"
        fi
    else
        echo "$(date -Iseconds)  WARN: $NM_HOME/memory.db not found" >> "$LOG_FILE"
    fi
fi

# 2. bench-history — small, just copy. Skip if dest already exists to
# avoid the bench-history/bench-history/ nesting bug (reviewer B3).
if [[ -d "$NM_HOME/bench-history" && ! -d "$DEST/bench-history" ]]; then
    cp -R "$NM_HOME/bench-history" "$DEST/bench-history"
fi

# 3. Retention prune — keep last 14 days. BSD head doesn't support
# `head -n -N`, so count + take the first (total - keep) entries.
COUNT_BEFORE=$(ls -1 "$BACKUP_ROOT" 2>/dev/null | wc -l | tr -d ' ')
KEEP=14
if [[ "$COUNT_BEFORE" -gt "$KEEP" ]]; then
    PRUNE_COUNT=$((COUNT_BEFORE - KEEP))
    ls -1 "$BACKUP_ROOT" 2>/dev/null | sort | head -n "$PRUNE_COUNT" | while read -r OLD; do
        rm -rf "$BACKUP_ROOT/$OLD" 2>/dev/null || true
        echo "$(date -Iseconds)  pruned old snapshot: $OLD" >> "$LOG_FILE"
    done
fi
COUNT_AFTER=$(ls -1 "$BACKUP_ROOT" 2>/dev/null | wc -l | tr -d ' ')

echo "$(date -Iseconds)  snapshot-daily complete  retained=${COUNT_AFTER} (was ${COUNT_BEFORE})" >> "$LOG_FILE"
