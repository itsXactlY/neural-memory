#!/usr/bin/env python3
"""
mssql_store.py - MSSQL storage backend for Neural Memory
Uses pyodbc with creds from dontcommit.py
"""
import struct
from typing import Optional

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
    """MSSQL-backed memory store"""
    
    def __init__(self, server='localhost', database='NeuralMemory',
                 username='SA', password="q?}33YIToo:H%xue$Kr*",
                 driver='{ODBC Driver 18 for SQL Server}'):
        import pyodbc
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
            # Table doesn't exist, create it
            for stmt in SCHEMA_SQL.split(';'):
                stmt = stmt.strip()
                if stmt and 'GO' not in stmt and 'CREATE DATABASE' not in stmt:
                    try:
                        cursor.execute(stmt)
                    except Exception as e:
                        pass  # Ignore if already exists
            self.conn.commit()
    
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
    
    def get_all(self) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, label, content, embedding, vector_dim, salience, access_count FROM memories ORDER BY id")
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
    
    def get_connections(self, node_id: int) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT source_id, target_id, weight, edge_type FROM connections WHERE source_id = ? OR target_id = ? ORDER BY weight DESC",
            node_id, node_id
        )
        return [{'source': r[0], 'target': r[1], 'weight': r[2], 'type': r[3]} for r in cursor.fetchall()]
    
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
