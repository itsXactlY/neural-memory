#!/usr/bin/env python3
"""
import_honcho.py - Import Honcho Export into Neural Memory
Uses the exact neural_memory.Memory API. No message_embeddings (9524d != 384d).
"""

import json
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from neural_memory import Memory

EXPORT_DIR = Path(os.environ.get("HONCHO_EXPORT_DIR", Path.home() / "honcho_export"))
DB_PATH = Path(os.environ.get(
    "NEURAL_MEMORY_DB_PATH",
    Path.home() / ".neural_memory" / "memory.db",
)).expanduser()

def load_json(name: str):
    with open(EXPORT_DIR / name) as f:
        return json.load(f)

def import_messages(mem: Memory, batch_size: int = 256):
    """Import all messages: batch embed on GPU, bulk insert to SQLite (no auto-connect)."""
    messages = load_json("messages.json")
    print(f"\n=== Importing {len(messages)} messages (batch={batch_size}) ===")
    
    texts = []
    labels = []
    skipped = 0
    
    for msg in messages:
        content = msg.get("content", "").strip()
        if not content:
            skipped += 1
            continue
        
        peer = msg.get("peer_name", "unknown")
        session = msg.get("session_name", "unknown")
        ts = msg.get("created_at", "")[:19]
        label = f"msg:{peer}:{session}:{ts}"
        
        if len(content) > 8000:
            content = content[:8000] + "..."
        
        texts.append(content)
        labels.append(label)
    
    print(f"  {len(texts)} non-empty, {skipped} empty skipped")
    
    total = len(texts)
    t0 = time.time()
    embedder = mem._embedder
    dim = embedder.dim
    
    # Direct SQLite bulk insert (skip auto-connect for speed)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Batch embed (GPU)
        embeddings = embedder.embed_batch(batch_texts)
        
        # Bulk insert
        rows = [
            (label, text, struct.pack(f'{dim}f', *emb))
            for label, text, emb in zip(batch_labels, batch_texts, embeddings)
        ]
        conn.executemany(
            "INSERT INTO memories (label, content, embedding) VALUES (?, ?, ?)", rows
        )
        conn.commit()
        
        # In-memory graph rebuilt from DB after import completes —
        # IDs from bulk insert don't match sequential assumptions.
        
        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        
        print(f"  [{done}/{total}] {rate:.1f} msg/s, ETA: {eta:.0f}s", flush=True)
    
    conn.close()
    elapsed = time.time() - t0
    print(f"  Done: {total} messages in {elapsed:.1f}s ({total/elapsed:.1f} msg/s)")

def import_simple(mem: Memory, label_prefix: str, items: list, text_fn, label_fn):
    """Generic bulk import for small datasets (<1000 items)."""
    print(f"\n=== Importing {len(items)} {label_prefix} ===")
    embedder = mem._embedder
    dim = embedder.dim
    
    texts = [text_fn(item) for item in items]
    labels = [label_fn(item) for item in items]
    
    embeddings = embedder.embed_batch(texts)
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    rows = [
        (label, text, struct.pack(f'{dim}f', *emb))
        for label, text, emb in zip(labels, texts, embeddings)
    ]
    conn.executemany("INSERT INTO memories (label, content, embedding) VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    
    print(f"  Done: {len(items)} {label_prefix}")

def build_connections(mem: Memory, sample_size: int = 5000, threshold: float = 0.15):
    """Build graph connections. Sample-based for large DBs to avoid O(n²)."""
    print(f"\n=== Building connections (sample={sample_size}, threshold={threshold}) ===")
    import math
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # Load all memories
    rows = conn.execute("SELECT id, embedding FROM memories ORDER BY id").fetchall()
    total = len(rows)
    print(f"  {total} memories total")
    
    if total == 0:
        conn.close()
        return
    
    dim = len(rows[0][1]) // 4
    all_ids = []
    all_embs = []
    for row in rows:
        all_ids.append(row[0])
        all_embs.append(list(struct.unpack(f'{dim}f', row[1])))
    
    def cosine(a, b):
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        return dot / (na * nb) if na * nb > 0 else 0
    
    # For large DBs, connect each memory to its top-k nearest in a sliding window
    connections = []
    t0 = time.time()
    
    if total <= sample_size:
        # Full O(n²) for small DBs
        for i in range(total):
            for j in range(i+1, total):
                sim = cosine(all_embs[i], all_embs[j])
                if sim > threshold:
                    connections.append((all_ids[i], all_ids[j], sim))
            if (i+1) % 500 == 0:
                print(f"  [{i+1}/{total}] {len(connections)} connections", flush=True)
    else:
        # Windowed nearest-neighbor for large DBs
        window = 200  # compare within temporal window
        for i in range(total):
            start = max(0, i - window)
            end = min(total, i + window)
            for j in range(start, end):
                if j <= i:
                    continue
                sim = cosine(all_embs[i], all_embs[j])
                if sim > threshold:
                    connections.append((all_ids[i], all_ids[j], sim))
            if (i+1) % 2000 == 0:
                elapsed = time.time() - t0
                rate = (i+1) / elapsed
                print(f"  [{i+1}/{total}] {len(connections)} conns, {rate:.0f}/s", flush=True)
    
    # Bulk insert connections
    print(f"  Inserting {len(connections)} connections...")
    conn.executemany(
        "INSERT OR IGNORE INTO connections (source_id, target_id, weight, edge_type) VALUES (?, ?, ?, 'similar')",
        connections
    )
    conn.commit()
    
    elapsed = time.time() - t0
    print(f"  Done: {len(connections)} connections in {elapsed:.1f}s")
    conn.close()

def import_documents(mem: Memory):
    """Import documents."""
    docs = load_json("documents.json")
    import_simple(mem, "documents", docs,
        text_fn=lambda d: d.get("content", "").strip() or None,
        label_fn=lambda d: f"doc:{d.get('observer','?')}->{d.get('observed','?')}:{d.get('session_name','')}:{d.get('created_at','')[:19]}"
    )

def import_sessions(mem: Memory):
    """Import session metadata."""
    sessions = load_json("sessions.json")
    import_simple(mem, "sessions", sessions,
        text_fn=lambda s: f"Session '{s.get('name','')}' in workspace '{s.get('workspace_name','')}', created {s.get('created_at','')[:19]}, active={s.get('is_active',False)}",
        label_fn=lambda s: f"session:{s.get('name','')}:{s.get('created_at','')[:19]}"
    )

def import_peers(mem: Memory):
    """Import peer identities."""
    peers = load_json("peers.json")
    import_simple(mem, "peers", peers,
        text_fn=lambda p: f"Peer '{p.get('name','')}' in workspace '{p.get('workspace_name','')}', created {p.get('created_at','')[:19]}",
        label_fn=lambda p: f"peer:{p.get('name','')}"
    )

def import_collections(mem: Memory):
    """Import collection metadata."""
    collections = load_json("collections.json")
    import_simple(mem, "collections", collections,
        text_fn=lambda c: f"Collection '{c.get('id','?')}' in '{c.get('workspace_name','')}': observer={c.get('observer','?')}, observed={c.get('observed','?')}, created {c.get('created_at','')[:19]}",
        label_fn=lambda c: f"collection:{c.get('id','')[:20]}"
    )

def main():
    print("=" * 60)
    print("Honcho Export -> Neural Memory Import")
    print("=" * 60)
    
    t_total = time.time()
    
    # Clear existing DB
    print("\n[1/3] Clearing DB...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM memories")
    conn.execute("DELETE FROM connections")
    conn.commit()
    conn.close()
    
    mem = Memory(db_path=str(DB_PATH), use_cpp=False)  # Python backend (SQLite + embedder)
    print(f"Backend: {mem.backend}, dim={mem.dim}")
    
    # Import
    print("\n[2/3] Importing data...")
    import_peers(mem)
    import_collections(mem)
    import_sessions(mem)
    import_documents(mem)
    import_messages(mem, batch_size=256)
    
    # Build connections
    print("\n[3/3] Building graph connections...")
    build_connections(mem, sample_size=5000, threshold=0.15)
    
    # Final stats
    stats = mem.stats()
    elapsed = time.time() - t_total
    
    print(f"\n{'=' * 60}")
    print(f"IMPORT COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Final stats: {stats}")
    print(f"{'=' * 60}")
    
    mem.close()

if __name__ == "__main__":
    main()
