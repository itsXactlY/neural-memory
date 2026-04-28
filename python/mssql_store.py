#!/usr/bin/env python3
"""
mssql_store.py - MSSQL storage backend for Neural Memory
Uses pyodbc with credentials from env vars or .env file.

Credentials resolution order:
  1. Environment variables: MSSQL_SERVER, MSSQL_DATABASE, MSSQL_USERNAME, MSSQL_PASSWORD
  2. .env file (~/.hermes/.env, CWD, plugin dir)
  3. Defaults (localhost, NeuralMemory, SA)
"""
import os
import struct
from pathlib import Path
from typing import Optional


def _load_dotenv(paths: list[str]) -> dict:
    env = {}
    for p in paths:
        path = Path(p).expanduser()
        if path.is_file():
            try:
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip("\"'")
                        if key and val:
                            env.setdefault(key, val)
            except Exception:
                pass
    return env


_dotenv = _load_dotenv([
    ".env",
    str(Path.home() / ".hermes" / ".env"),
    str(Path(__file__).parent / ".env"),
])


def _env(key: str, fallback: str = "") -> str:
    return os.environ.get(key) or _dotenv.get(key, fallback)

SCHEMA_SQL = """
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'NeuralMemory')
    CREATE DATABASE NeuralMemory;
GO

USE NeuralMemory;
GO

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'memories')
CREATE TABLE memories (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    label NVARCHAR(256),
    content NVARCHAR(MAX),
    embedding VARBINARY(8000),
    vector_dim INT NOT NULL,
    salience FLOAT DEFAULT 1.0,
    created_at DATETIME2(7) DEFAULT SYSUTCDATETIME(),
    last_accessed DATETIME2(7) DEFAULT SYSUTCDATETIME(),
    access_count INT DEFAULT 0
);

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'connections')
CREATE TABLE connections (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    source_id BIGINT,
    target_id BIGINT,
    weight FLOAT DEFAULT 0.5,
    edge_type NVARCHAR(50) DEFAULT 'similar',
    created_at DATETIME2(7) DEFAULT SYSUTCDATETIME(),
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conn_source')
CREATE INDEX idx_conn_source ON connections(source_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conn_target')
CREATE INDEX idx_conn_target ON connections(target_id);
"""


class MSSQLStore:
    """MSSQL-backed memory store.

    Credentials: env vars > .env > defaults.
    """

    def __init__(self, server='', database='', username='', password='',
                 driver=''):
        import pyodbc

        server = server or _env('MSSQL_SERVER', '127.0.0.1')
        if server == 'localhost':
            server = '127.0.0.1'  # MSSQL IPv4 only
        database = database or _env('MSSQL_DATABASE', 'NeuralMemory')
        username = username or _env('MSSQL_USERNAME', 'SA')
        password = password or _env('MSSQL_PASSWORD', '')
        driver = driver or _env('MSSQL_DRIVER', '{ODBC Driver 18 for SQL Server}')

        if not password:
            import logging
            logging.getLogger(__name__).warning(
                "MSSQL_PASSWORD not set — add it to ~/.hermes/.env"
            )

        self.conn_str = (
            f'DRIVER={driver};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password};'
            f'TrustServerCertificate=yes;'
        )
        self.conn = pyodbc.connect(self.conn_str, autocommit=True)
        import threading
        self._lock = threading.Lock()
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()
        # Check if memories table exists
        try:
            cursor.execute("SELECT COUNT(*) FROM memories")
        except Exception:
            # Table doesn't exist, create it
            for stmt in SCHEMA_SQL.split(';'):
                stmt = stmt.strip()
                if stmt and 'GO' not in stmt and 'CREATE DATABASE' not in stmt:
                    try:
                        cursor.execute(stmt)
                    except Exception as e:
                        pass  # Ignore if already exists
            self.conn.commit()
    
    def store(self, label: str, content: str, embedding: list[float],
              id_: Optional[int] = None) -> int:
        """Insert or upsert a memory.

        When `id_` is provided, the row is written with that exact id via
        SET IDENTITY_INSERT — used by Memory.remember() to keep the MSSQL
        mirror's IDs aligned with SQLite's source-of-truth IDs. Without
        ID alignment:

          - recall_multihop / think query MSSQL with SQLite IDs and miss
            every memory (silent zero-result graph expansion).
          - sync_bridge subsequently inserts the same memory AGAIN with
            the SQLite ID (since it sees id > max_id), producing
            duplicate rows.

        On collision (existing row with the same id), MERGE updates
        label/content/embedding so the mirror stays consistent with
        SQLite's current state — important after conflict-fusion
        rewrites in NeuralMemory.remember.
        """
        blob = struct.pack(f'{len(embedding)}f', *embedding)
        cursor = self.conn.cursor()
        with self._lock:
            if id_ is None:
                cursor.execute(
                    "INSERT INTO memories (label, content, embedding, vector_dim) "
                    "OUTPUT INSERTED.id VALUES (?, ?, ?, ?)",
                    label, content, blob, len(embedding)
                )
                row = cursor.fetchone()
                self.conn.commit()
                return row[0] if row else 0
            # Explicit-id path: SET IDENTITY_INSERT + MERGE on id so a
            # second mirror call with the same id is idempotent.
            try:
                cursor.execute("SET IDENTITY_INSERT memories ON")
                cursor.execute(
                    "MERGE memories AS tgt "
                    "USING (VALUES (?, ?, ?, ?, ?)) "
                    "  AS src (id, label, content, embedding, vector_dim) "
                    "ON tgt.id = src.id "
                    "WHEN MATCHED THEN UPDATE SET "
                    "  label = src.label, content = src.content, "
                    "  embedding = src.embedding, vector_dim = src.vector_dim "
                    "WHEN NOT MATCHED THEN INSERT (id, label, content, embedding, vector_dim) "
                    "  VALUES (src.id, src.label, src.content, src.embedding, src.vector_dim);",
                    int(id_), label, content, blob, len(embedding)
                )
                self.conn.commit()
            finally:
                # IDENTITY_INSERT is per-session; turn it off before any
                # other code on this connection runs an autoincrement INSERT.
                try:
                    cursor.execute("SET IDENTITY_INSERT memories OFF")
                    self.conn.commit()
                except Exception:
                    pass
            return int(id_)
    
    @staticmethod
    def _row_to_dict(row) -> dict:
        """Common projection used by both get() and get_all().

        Returns the same field set SQLiteStore.get returns so callers that
        run against either backend (recall_multihop, _compute_temporal_score,
        _effective_salience) get the same shape. Previously MSSQL's row dict
        omitted created_at/last_accessed, which silently defaulted those
        fields to \"now\" downstream — collapsing temporal_score to 1.0 and
        skewing salience by treating every MSSQL-fetched row as freshly
        created.
        """
        id_, label, content, blob, dim, salience, created_at, last_accessed, access = row
        embedding = list(struct.unpack(f'{dim}f', blob)) if blob else []

        # Convert MSSQL DATETIME2 to epoch seconds for cross-backend parity.
        # SQLite stores REAL epochs; consumers of this dict expect floats.
        def _epoch(dt) -> float | None:
            if dt is None:
                return None
            try:
                from datetime import timezone as _tz
                if dt.tzinfo is None:
                    # MSSQL DATETIME2 is timezone-naive; SYSUTCDATETIME()
                    # writes UTC, so attach UTC explicitly before converting.
                    return dt.replace(tzinfo=_tz.utc).timestamp()
                return dt.timestamp()
            except Exception:
                return None

        return {
            "id": id_,
            "label": label,
            "content": content,
            "embedding": embedding,
            "salience": salience,
            "created_at": _epoch(created_at),
            "last_accessed": _epoch(last_accessed),
            "access_count": access,
        }

    def get_all(self) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, label, content, embedding, vector_dim, salience, "
            "created_at, last_accessed, access_count FROM memories ORDER BY id"
        )
        return [self._row_to_dict(r) for r in cursor.fetchall()]

    def get(self, id_: int, include_embedding: bool = True) -> Optional[dict]:
        """Fetch a single memory.

        Mirrors SQLiteStore.get's `include_embedding=True` default and the
        same opt-out for label-only lookups. Pulling the embedding blob
        across the wire for every neighbour-label lookup in recall_multihop
        is wasteful (~4KB per memory at 1024-d); skipping it when the
        caller only needs label/content drops MSSQL bandwidth proportional
        to the result size.
        """
        cursor = self.conn.cursor()
        if include_embedding:
            cursor.execute(
                "SELECT id, label, content, embedding, vector_dim, salience, "
                "created_at, last_accessed, access_count FROM memories WHERE id = ?",
                id_,
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)
        # Embedding-skip path — return the same dict shape with embedding=[].
        cursor.execute(
            "SELECT id, label, content, salience, "
            "created_at, last_accessed, access_count FROM memories WHERE id = ?",
            id_,
        )
        row = cursor.fetchone()
        if not row:
            return None
        id_, label, content, salience, created_at, last_accessed, access = row

        def _epoch(dt) -> float | None:
            if dt is None:
                return None
            try:
                from datetime import timezone as _tz
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=_tz.utc).timestamp()
                return dt.timestamp()
            except Exception:
                return None

        return {
            "id": id_, "label": label, "content": content,
            "embedding": [],
            "salience": salience,
            "created_at": _epoch(created_at),
            "last_accessed": _epoch(last_accessed),
            "access_count": access,
        }
    
    def exists(self, id_: int) -> bool:
        """Cheap existence check — single-row count, no embedding pull."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM memories WHERE id = ?", id_)
        return cursor.fetchone() is not None

    def touch(self, id_: int):
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memories SET last_accessed = SYSUTCDATETIME(), access_count = access_count + 1 WHERE id = ?",
            id_
        )
        self.conn.commit()
    
    def add_connection(self, source: int, target: int, weight: float, edge_type: str = "similar"):
        """MERGE an edge into MSSQL connections.

        Canonicalises source<target so the row layout matches the SQLiteStore
        invariant (see iter 23). Mixed-orientation rows in MSSQL — the
        previous behaviour — meant any query that filters on a single
        ordering would miss half the bridge edges, breaking cross-backend
        graph traversal.
        """
        if source == target:
            return
        if source > target:
            source, target = target, source
        cursor = self.conn.cursor()
        cursor.execute(
            "MERGE connections AS target "
            "USING (VALUES (?, ?, ?, ?)) AS source (source_id, target_id, weight, edge_type) "
            "ON target.source_id = source.source_id AND target.target_id = source.target_id "
            "WHEN MATCHED THEN "
            "    UPDATE SET weight = CASE WHEN source.weight > target.weight THEN source.weight ELSE target.weight END, "
            "               edge_type = source.edge_type "
            "WHEN NOT MATCHED THEN "
            "    INSERT (source_id, target_id, weight, edge_type) "
            "    VALUES (source.source_id, source.target_id, source.weight, source.edge_type);",
            source, target, weight, edge_type
        )
        self.conn.commit()
    
    def get_connections(self, node_id: int) -> list[dict]:
        """Return edges incident to node_id.

        Output dicts carry BOTH 'type' (legacy field used by Memory.recall_multihop
        and the dashboard) and 'edge_type' (used by the mirror loop and the
        dream backends), matching SQLiteStore.get_connections's shape exactly.
        Without 'edge_type', any cross-backend caller that read it would get
        None and lose the edge classification.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT source_id, target_id, weight, edge_type "
            "FROM connections WHERE source_id = ? OR target_id = ? "
            "ORDER BY weight DESC",
            node_id, node_id,
        )
        out = []
        for r in cursor.fetchall():
            etype = r[3] or "similar"
            out.append({
                "source": r[0],
                "target": r[1],
                "weight": r[2],
                "type": etype,
                "edge_type": etype,
            })
        return out
    
    def stats(self) -> dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        mc = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM connections")
        cc = cursor.fetchone()[0]
        return {'memories': mc, 'connections': cc}
    
    def close(self):
        self.conn.close()


# Quick test
if __name__ == "__main__":
    try:
        store = MSSQLStore()
        mid = store.store("test", "Hello MSSQL", [0.1] * 1024)
        print(f"Stored: {mid}")
        m = store.get(mid)
        print(f"Retrieved: {m['label']}")
        s = store.stats()
        print(f"Stats: {s}")
        store.close()
        print("MSSQL: OK")
    except Exception as e:
        print(f"MSSQL error: {e}")
