# Neural Dream Engine (devops)

Background memory consolidation skill for the Mazemaker adapter.
Storage: SQLite by default, Postgres + pgvector when `MM_DB_BACKEND=postgres`.

The external store is now Postgres-only. See `docs/POSTGRES.md` in the
mazemaker-v2 repo for the operator setup recipe.
