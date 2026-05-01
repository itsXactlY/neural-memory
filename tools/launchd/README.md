# launchd plist templates

Templates for AE main-builder lane to install via `launchctl bootstrap`.
NOT installed by neural-memory lane (cross-lane discipline: launchd ops live in AE main-builder territory).

## Available plists

### `com.ae.neural-memory-ingest.plist`
Daily 04:15 local — re-runs `tools/ingest_ae_corpus.py` + `tools/post_ingest_sanity.py`.

- **Idempotent**: ingest dedupes by content_hash; unchanged chunks skipped.
- **Schedule rationale**: 04:15 lands AFTER 03:32 memory-autocommit timer (latest edits committed first).
- **Sanity gate**: post_ingest_sanity.py exits non-zero if retrieval contracts regress; surfaces in `ingest.stderr.log`.
- **Logs**: `~/.neural_memory/logs/ingest.{stdout,stderr}.log`

## Install

```bash
cp /Users/tito/lWORKSPACEl/research/neural-memory/tools/launchd/com.ae.neural-memory-ingest.plist \
   ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ae.neural-memory-ingest.plist
launchctl print gui/$(id -u)/com.ae.neural-memory-ingest  # verify loaded
```

## Uninstall

```bash
launchctl bootout gui/$(id -u)/com.ae.neural-memory-ingest
rm ~/Library/LaunchAgents/com.ae.neural-memory-ingest.plist
```

## Coordination with AE-builder lane's existing plist git-tracking

AE-builder lane committed 38 plists to `LangGraph/launchd/` 2026-05-01 (per bridge msg_e0e6655d).
This plist lives in the neural-memory repo at `tools/launchd/` because:
- Source-of-truth for the schedule belongs with the script it invokes (ingest_ae_corpus.py)
- Cross-lane discipline: NM lane doesn't write to LangGraph/launchd/

If AE-builder lane prefers all plists in one place, copy this file to `LangGraph/launchd/` — it'll work either way as long as `launchctl bootstrap` points at the right path.
