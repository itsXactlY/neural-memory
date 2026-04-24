#!/usr/bin/env python3
"""Skynet feature regression tests for Neural Memory.

These tests cover the Phase C/D capabilities that make memory less like a
flat vector store and more like an autonomous recall system:
- hybrid parallel retrieval (semantic + BM25/entity/temporal + salience)
- bi-temporal typed edges
- PPR spreading activation
- Louvain/derived dream insights
- conflict fusion with revision history

Run: python3 python/test_skynet_features.py
"""
from __future__ import annotations

import os
import sqlite3
import struct
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = 0
FAIL = 0


def testcase(name):
    def deco(fn):
        def wrapper():
            global PASS, FAIL
            try:
                fn()
                print(f"  PASS  {name}")
                PASS += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                FAIL += 1
        return wrapper
    return deco


def temp_db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = f.name
    f.close()
    return path


def cleanup(path):
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
        except FileNotFoundError:
            pass


@testcase("hybrid recall: BM25/entity/temporal channels surface lexical exact hits")
def test_hybrid_recall_channels():
    from memory_client import NeuralMemory

    db = temp_db()
    try:
        mem = NeuralMemory(
            db_path=db,
            embedding_backend="hash",
            use_cpp=False,
            retrieval_mode="hybrid",
            use_hnsw=False,
            lazy_graph=False,
            think_engine="ppr",
        )
        mem.remember("The ZephyrAlpha launch code is orchid-seven.", "zephyr", auto_connect=False)
        mem.remember("A dog named Lou likes long forest walks.", "pet", auto_connect=False)
        mem.remember("BTQuant runs trading experiments on crypto market data.", "work", auto_connect=False)

        results = mem.recall("ZephyrAlpha launch code", k=2, hybrid=True)
        assert results, "hybrid recall returned no results"
        top = results[0]
        assert top["label"] == "zephyr", f"expected lexical entity hit first, got {top}"
        assert "channel_scores" in top, "result missing per-channel scores"
        assert top["channel_scores"].get("bm25", 0) > 0, "BM25 channel did not contribute"
        assert top["channel_scores"].get("entity", 0) > 0, "entity channel did not contribute"
        assert "relevance" in top, "result missing final relevance score"
        assert "salience_factor" in top, "result missing salience decay factor"
        mem.close()
    finally:
        cleanup(db)


@testcase("bi-temporal typed edges: at_time filtering and edge_type roundtrip")
def test_bitemporal_typed_edges():
    from memory_client import SQLiteStore

    db = temp_db()
    try:
        store = SQLiteStore(db)
        a = store.store("a", "alpha", [1.0, 0.0, 0.0, 0.0])
        b = store.store("b", "beta", [0.0, 1.0, 0.0, 0.0])
        now = time.time()
        store.add_connection(
            a,
            b,
            0.77,
            edge_type="causal",
            event_time=now - 10,
            valid_from=now - 20,
            valid_to=now + 20,
        )
        active = store.get_connections(a, at_time=now)
        expired = store.get_connections(a, at_time=now + 3600)
        assert len(active) == 1, f"expected active edge at now, got {active}"
        assert active[0]["type"] == "causal", f"edge_type not preserved: {active}"
        assert active[0]["event_time"] is not None, "event_time missing"
        assert expired == [], f"valid_to filter failed: {expired}"
        store.close()
    finally:
        cleanup(db)


@testcase("PPR think: principled graph activation available beside BFS")
def test_ppr_think_engine():
    from memory_client import NeuralMemory

    db = temp_db()
    try:
        mem = NeuralMemory(db_path=db, embedding_backend="hash", use_cpp=False, think_engine="ppr")
        a = mem.remember("alpha root", "a", auto_connect=False)
        b = mem.remember("beta neighbor", "b", auto_connect=False)
        c = mem.remember("gamma neighbor", "c", auto_connect=False)
        mem.store.add_connection(a, b, 0.9, "similar")
        mem.store.add_connection(b, c, 0.8, "similar")
        mem._ensure_node(a)
        mem._ensure_node(b)
        mem._ensure_node(c)

        ppr = mem.think(a, depth=3, engine="ppr")
        assert ppr, "PPR returned no activation results"
        ids = [r["id"] for r in ppr]
        assert b in ids and c in ids, f"PPR failed to traverse weighted graph: {ppr}"
        assert all("activation" in r for r in ppr), f"missing activation field: {ppr}"
        mem.close()
    finally:
        cleanup(db)


@testcase("conflict fusion: same-label updates create canonical content and revision history")
def test_conflict_fusion_revision_history():
    from memory_client import NeuralMemory

    db = temp_db()
    try:
        mem = NeuralMemory(db_path=db, embedding_backend="hash", use_cpp=False)
        first = mem.remember("The deployment target is staging.", "deploy-target", detect_conflicts=True, auto_connect=False)
        second = mem.remember("The deployment target is production.", "deploy-target", detect_conflicts=True, auto_connect=False)
        assert second == first, "same-label conflict should update canonical memory, not create duplicate"
        row = mem.store.get(first)
        assert "[CANONICAL]" in row["content"], f"canonical marker missing: {row['content']}"
        revs = mem.store.conn.execute("SELECT old_content, new_content FROM memory_revisions WHERE memory_id=?", (first,)).fetchall()
        assert revs, "memory_revisions did not record superseded content"
        assert "staging" in revs[-1][0] and "production" in revs[-1][1], "revision content incorrect"
        mem.close()
    finally:
        cleanup(db)


@testcase("dream insights: Louvain splits bridged cliques and synthesizes derived memory")
def test_dream_louvain_and_derived_memory():
    from dream_engine import DreamEngine, SQLiteDreamBackend
    from memory_client import NeuralMemory

    db = temp_db()
    try:
        mem = NeuralMemory(db_path=db, embedding_backend="hash", use_cpp=False)
        ids = []
        for text in [
            "alpha cluster red apple", "alpha cluster red berry", "alpha cluster red cherry",
            "omega cluster blue river", "omega cluster blue ocean", "omega cluster blue lake",
        ]:
            ids.append(mem.remember(text, auto_connect=False, detect_conflicts=False))

        # Dense two-clique graph with one weak bridge. Connected-components sees 1;
        # Louvain-style modularity should split into 2 communities.
        for group in (ids[:3], ids[3:]):
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    mem.store.add_connection(group[i], group[j], 0.9, "similar")
        mem.store.add_connection(ids[2], ids[3], 0.08, "bridge")

        engine = DreamEngine(SQLiteDreamBackend(db), neural_memory=mem)
        stats = engine._phase_insights()
        assert stats.get("communities", 0) >= 2, f"expected Louvain split, got {stats}"
        assert stats.get("derived_facts", 0) >= 1, f"expected derived fact synthesis, got {stats}"
        conn = sqlite3.connect(db)
        try:
            derived = conn.execute("SELECT id, content FROM memories WHERE label='derived:cluster'").fetchall()
            assert derived, "no derived:cluster memory created"
            edge_count = conn.execute("SELECT COUNT(*) FROM connections WHERE edge_type='derived_from'").fetchone()[0]
            assert edge_count >= 3, f"derived_from edges missing, count={edge_count}"
        finally:
            conn.close()
        mem.close()
    finally:
        cleanup(db)


if __name__ == "__main__":
    print("=" * 60)
    print("  Neural Memory Skynet Feature Tests")
    print("=" * 60)
    tests = [
        test_hybrid_recall_channels,
        test_bitemporal_typed_edges,
        test_ppr_think_engine,
        test_conflict_fusion_revision_history,
        test_dream_louvain_and_derived_memory,
    ]
    for t in tests:
        t()
    print("=" * 60)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
