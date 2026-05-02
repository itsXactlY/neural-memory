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

# Reconciliation-reviewer-round-1 fix 2026-05-02: removed dead
# CREATE DATABASE + USE blocks. _ensure_schema explicitly skips
# both via the `'GO' not in stmt and 'CREATE DATABASE' not in stmt`
# guard, so they were never executed. Database creation is the
# operator's responsibility (DBA-side + connection-string setup).
SCHEMA_SQL = """
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
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()
        # Check if memories table exists
        try:
            cursor.execute("SELECT COUNT(*) FROM memories")
        except Exception:
            # Reconciliation-reviewer-round-1 fix 2026-05-02: simplified
            # the dead-code filters. CREATE DATABASE + GO blocks now
            # absent from SCHEMA_SQL itself; just split + execute.
            for stmt in SCHEMA_SQL.split(';'):
                stmt = stmt.strip()
                if stmt:
                    try:
                        cursor.execute(stmt)
                    except Exception:
                        pass  # Ignore if already exists
            # autocommit=True at connection (line 121) — explicit commit
            # is a no-op but kept for readability.
    
    def store(self, label: str, content: str, embedding: list[float]) -> int:
        blob = struct.pack(f'{len(embedding)}f', *embedding)
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO memories (label, content, embedding, vector_dim) OUTPUT INSERTED.id VALUES (?, ?, ?, ?)",
            label, content, blob, len(embedding)
        )
        row = cursor.fetchone()
        self.conn.commit()
        return row[0] if row else 0
    
    def get_all(self, limit: int = 100_000) -> list[dict]:
        # Reconciliation-reviewer-round-1 fix 2026-05-02: bounded by
        # default to prevent loading entire memories table into RAM
        # for big substrates. SQLiteStore.get_all has the same shape
        # without explicit limit because it's row-iterator-based;
        # MSSQL fetchall() loads all into memory so the cap matters.
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT TOP {int(limit)} id, label, content, embedding, vector_dim, salience, access_count FROM memories ORDER BY id")
        results = []
        for row in cursor.fetchall():
            id_, label, content, blob, dim, salience, access = row
            embedding = list(struct.unpack(f'{dim}f', blob)) if blob else []
            results.append({
                'id': id_, 'label': label, 'content': content,
                'embedding': embedding, 'salience': salience, 'access_count': access
            })
        return results
    
    def get(self, id_: int) -> Optional[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, label, content, embedding, vector_dim, salience, access_count FROM memories WHERE id = ?", id_)
        row = cursor.fetchone()
        if not row:
            return None
        id_, label, content, blob, dim, salience, access = row
        embedding = list(struct.unpack(f'{dim}f', blob)) if blob else []
        return {'id': id_, 'label': label, 'content': content, 'embedding': embedding,
                'salience': salience, 'access_count': access}
    
    def touch(self, id_: int):
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memories SET last_accessed = SYSUTCDATETIME(), access_count = access_count + 1 WHERE id = ?",
            id_
        )
        self.conn.commit()
    
    def add_connection(self, source: int, target: int, weight: float, edge_type: str = "similar"):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO connections (source_id, target_id, weight, edge_type) VALUES (?, ?, ?, ?)",
            source, target, weight, edge_type
        )
        self.conn.commit()
    
    def get_connections(self, node_id: int, at_time=None,
                        include_expired: bool = False) -> list[dict]:
        # Reconciliation-reviewer-round-1 fix 2026-05-02: align return
        # shape with SQLiteStore.get_connections (8 keys, not 4).
        # Per-commit-reviewer-round-2 fix 2026-05-02: also accept + apply
        # at_time + include_expired params (memory_client.py:2175 calls
        # with include_expired=True; signature mismatch would TypeError).
        import time as _time
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT source_id, target_id, weight, edge_type, "
            "       event_time, ingestion_time, valid_from, valid_to "
            "FROM connections WHERE source_id = ? OR target_id = ? "
            "ORDER BY weight DESC",
            node_id, node_id
        )
        now = at_time if at_time is not None else _time.time()
        out = []
        for r in cursor.fetchall():
            vf, vt = r[6], r[7]
            if not include_expired:
                # Match SQLite implementation's Python-side temporal filter
                if vf is not None and now < vf:
                    continue
                if vt is not None and now > vt:
                    continue
            out.append({
                'source': r[0], 'target': r[1], 'weight': r[2], 'type': r[3],
                'event_time': r[4], 'ingestion_time': r[5],
                'valid_from': vf, 'valid_to': vt,
            })
        return out

    def get_connections_batch(
        self, node_ids, at_time=None, include_expired: bool = False,
    ) -> dict:
        """Batched edge fetch — needed by graph_search BFS for the 20×
        perf win shipped in commit b2bda67 against SQLiteStore.

        Reconciliation-reviewer-round-1 fix 2026-05-02: this method was
        missing on MSSQLStore. Per-commit-reviewer-round-2 fix 2026-05-02:
        actually APPLY the at_time + include_expired params (was accepting
        them but never applying — would have leaked dead edges into BFS).
        """
        import time as _time
        node_ids = list(node_ids)
        if not node_ids:
            return {}
        cursor = self.conn.cursor()
        placeholders = ",".join("?" * len(node_ids))
        params = list(node_ids) + list(node_ids)
        cursor.execute(
            f"SELECT source_id, target_id, weight, edge_type, "
            f"       event_time, ingestion_time, valid_from, valid_to "
            f"FROM connections "
            f"WHERE source_id IN ({placeholders}) "
            f"   OR target_id IN ({placeholders}) "
            f"ORDER BY weight DESC",
            *params
        )
        now = at_time if at_time is not None else _time.time()
        out: dict = {nid: [] for nid in node_ids}
        node_set = set(node_ids)
        for r in cursor.fetchall():
            vf, vt = r[6], r[7]
            if not include_expired:
                # Match SQLite implementation's Python-side temporal filter
                if vf is not None and now < vf:
                    continue
                if vt is not None and now > vt:
                    continue
            edge = {
                'source': r[0], 'target': r[1], 'weight': r[2], 'type': r[3],
                'event_time': r[4], 'ingestion_time': r[5],
                'valid_from': vf, 'valid_to': vt,
            }
            if r[0] in node_set:
                out[r[0]].append(edge)
            if r[1] in node_set and r[1] != r[0]:
                out[r[1]].append(edge)
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
        mid = store.store("test", "Hello MSSQL", [0.1] * 384)
        print(f"Stored: {mid}")
        m = store.get(mid)
        print(f"Retrieved: {m['label']}")
        s = store.stats()
        print(f"Stats: {s}")
        store.close()
        print("MSSQL: OK")
    except Exception as e:
        print(f"MSSQL error: {e}")
