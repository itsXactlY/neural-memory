# Neural Dream Engine

Background memory consolidation for the Mazemaker adapter. Three phases
inspired by biological sleep: NREM (replay/strengthen), REM (bridge
discovery), Insight (abstraction).

Storage backends: SQLite (default) or Postgres + pgvector
(`MM_DB_BACKEND=postgres`). The dream engine writes its session and
event tables alongside the main memory store.

## Quick start

```bash
# SQLite (default)
python python/dream_engine.py

# Postgres + pgvector
MM_DB_BACKEND=postgres python python/dream_engine.py
```

See `python/dream_engine.py` for the full configuration surface.
