#!/usr/bin/env python3
"""
test_suite.py - Comprehensive test suite for Neural Memory Adapter
Run: python3 test_suite.py
Run specific: python3 test_suite.py --tags embed,memory,graph
"""

import sys
import os
import tempfile
import time
import threading
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = FAIL = SKIP = 0
TAGS = set()

def _testcase(name, tags=None):
    tags = tags or []
    def decorator(fn):
        def wrapper():
            global PASS, FAIL, SKIP
            try:
                fn()
                print(f"  PASS  {name}")
                PASS += 1
            except SkipTest as e:
                print(f"  SKIP  {name}: {e}")
                SKIP += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                FAIL += 1
        wrapper._name = name
        wrapper._tags = tags
        TAGS.update(tags)
        return wrapper
    return decorator

class SkipTest(Exception): pass

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = sum(x*x for x in a)**0.5
    nb = sum(x*x for x in b)**0.5
    return dot/(na*nb) if na*nb > 1e-10 else 0

# ============================================================================
# Embed Provider Tests
# ============================================================================

@_testcase("hash_embed: basic vector creation", tags=["embed"])
def test_1():
    from embed_provider import HashBackend
    b = HashBackend(dim=384)
    v = b.embed("hello world")
    assert len(v) == 384
    assert any(x != 0 for x in v), "Vector should not be all zeros"

@_testcase("hash_embed: deterministic output", tags=["embed"])
def test_2():
    from embed_provider import HashBackend
    b = HashBackend(dim=384)
    assert b.embed("test") == b.embed("test")
    assert b.embed("a") != b.embed("b")

@_testcase("hash_embed: similarity ordering", tags=["embed"])
def test_3():
    from embed_provider import HashBackend
    b = HashBackend(dim=384)
    v1 = b.embed("dog named Lou is a pet")
    v2 = b.embed("dog is a domestic animal")
    v3 = b.embed("quantum computing research paper")
    assert cosine(v1, v2) > cosine(v1, v3), "dog-dog > dog-quantum"

@_testcase("hash_embed: batch consistency", tags=["embed"])
def test_4():
    from embed_provider import HashBackend
    b = HashBackend(dim=384)
    batch = b.embed_batch(["a", "b", "c"])
    assert len(batch) == 3
    assert batch[0] == b.embed("a")

@_testcase("hash_embed: empty string", tags=["embed"])
def test_5():
    from embed_provider import HashBackend
    b = HashBackend(dim=384)
    v = b.embed("")
    assert len(v) == 384

@_testcase("hash_embed: unicode handling", tags=["embed"])
def test_21():
    from embed_provider import HashBackend
    b = HashBackend(dim=384)
    # Just verify it doesn't crash on unicode
    v = b.embed("Äöüß 中文")
    assert len(v) == 384

@_testcase("hash_embed: dimension variants", tags=["embed"])
def test_7():
    from embed_provider import HashBackend
    for dim in [64, 128, 256, 384, 512, 768]:
        b = HashBackend(dim=dim)
        v = b.embed("test")
        assert len(v) == dim

@_testcase("hash_embed: normalization", tags=["embed"])
def test_8():
    from embed_provider import HashBackend
    b = HashBackend(dim=384)
    v = b.embed("test normalization")
    norm = sum(x*x for x in v)**0.5
    assert abs(norm - 1.0) < 0.01, f"Norm should be ~1.0, got {norm}"

@_testcase("tfidf: auto-train on corpus", tags=["embed"])
def test_9():
    from embed_provider import TfidfSvdBackend
    b = TfidfSvdBackend(dim=128)
    corpus = [f"document about topic {i}" for i in range(20)]
    for text in corpus:
        b.embed(text)
    assert b._trained, "Should auto-train after 5 texts"

@_testcase("tfidf: trained embeddings are valid", tags=["embed"])
def test_10():
    from embed_provider import TfidfSvdBackend
    b = TfidfSvdBackend(dim=128)
    corpus = [f"document about topic {i}" for i in range(20)]
    for text in corpus:
        b.embed(text)
    # After training, verify embed returns correct dimension
    v = b.embed("test embedding after training")
    assert len(v) == 128, f"Expected 128, got {len(v)}"

@_testcase("sentence_transformers: singleton", tags=["embed", "slow"])
def test_11():
    from pathlib import Path
    model_dir = Path.home() / ".neural_memory" / "models" / "models--BAAI--bge-m3"
    if not model_dir.exists():
        raise SkipTest("BAAI/bge-m3 not cached (CI environment)")
    try:
        from embed_provider import SentenceTransformerBackend
    except ImportError:
        raise SkipTest("sentence-transformers not installed")
    t0 = time.time()
    b1 = SentenceTransformerBackend()
    t1 = time.time()
    b2 = SentenceTransformerBackend()
    t2 = time.time()
    assert b1.model is b2.model, "Should share same model"
    assert (t2-t1) < 0.1, f"Second init should be instant, got {t2-t1:.3f}s"

@_testcase("auto_detect: picks best backend", tags=["embed"])
def test_12():
    from embed_provider import EmbeddingProvider
    p = EmbeddingProvider(backend="auto")
    assert p.dim > 0
    v = p.embed("test")
    assert len(v) == p.dim

# ============================================================================
# SQLite Store Tests
# ============================================================================

@_testcase("sqlite: create and read", tags=["storage"])
def test_13():
    from memory_client import SQLiteStore
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        s = SQLiteStore(db)
        mid = s.store("label", "content", [0.1]*384)
        assert mid > 0
        m = s.get(mid)
        assert m['label'] == "label"
        assert len(m['embedding']) == 384
        s.close()
    finally: os.unlink(db)

@_testcase("sqlite: get_all", tags=["storage"])
def test_14():
    from memory_client import SQLiteStore
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        s = SQLiteStore(db)
        for i in range(5):
            s.store(f"item{i}", f"content{i}", [float(i)]*384)
        all_m = s.get_all()
        assert len(all_m) == 5
        s.close()
    finally: os.unlink(db)

@_testcase("sqlite: connections", tags=["storage"])
def test_15():
    from memory_client import SQLiteStore
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        s = SQLiteStore(db)
        s.store("a", "a", [0.1]*384)
        s.store("b", "b", [0.2]*384)
        s.add_connection(1, 2, 0.8, "similar")
        c = s.get_connections(1)
        assert len(c) == 1
        assert c[0]['weight'] == 0.8
        s.close()
    finally: os.unlink(db)

@_testcase("sqlite: touch updates access", tags=["storage"])
def test_16():
    from memory_client import SQLiteStore
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        s = SQLiteStore(db)
        s.store("test", "test", [0.1]*384)
        m1 = s.get(1)
        s.touch(1)
        s.touch(1)
        m2 = s.get(1)
        assert m2['access_count'] == m1['access_count'] + 2
        s.close()
    finally: os.unlink(db)

@_testcase("sqlite: stats", tags=["storage"])
def test_17():
    from memory_client import SQLiteStore
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        s = SQLiteStore(db)
        s.store("a", "a", [0.1]*384)
        s.store("b", "b", [0.2]*384)
        s.add_connection(1, 2, 0.5)
        st = s.get_stats()
        assert st['memories'] == 2
        assert st['connections'] == 1
        s.close()
    finally: os.unlink(db)

@_testcase("sqlite: thread safety (8 threads)", tags=["storage", "threading"])
def test_18():
    from memory_client import SQLiteStore
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        s = SQLiteStore(db)
        errors = []
        def writer(tid):
            try:
                for i in range(10):
                    s.store(f"t{tid}", f"d{i}", [float(i)]*384)
            except Exception as e: errors.append(str(e))
        def reader(tid):
            try:
                for i in range(10):
                    s.get_all()
            except Exception as e: errors.append(str(e))
        threads = [threading.Thread(target=writer if i%2==0 else reader, args=(i,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(errors) == 0, f"Thread errors: {errors[:3]}"
        assert s.get_stats()['memories'] == 40
        s.close()
    finally: os.unlink(db)

@_testcase("sqlite: WAL mode enabled", tags=["storage"])
def test_19():
    from memory_client import SQLiteStore
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        s = SQLiteStore(db)
        mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal", f"Expected WAL, got {mode}"
        s.close()
    finally: os.unlink(db)

@_testcase("sqlite: persistence across reopen", tags=["storage"])
def test_20():
    from memory_client import SQLiteStore
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        s1 = SQLiteStore(db)
        s1.store("persist", "test", [0.5]*384)
        s1.close()
        s2 = SQLiteStore(db)
        m = s2.get(1)
        assert m['label'] == "persist"
        s2.close()
    finally: os.unlink(db)

# ============================================================================
# Memory Client Tests
# ============================================================================

@_testcase("memory: store and recall", tags=["memory"])
def test_21():
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        m = NeuralMemory(db_path=db, embedding_backend="hash")
        m.remember("Dog named Lou", "Pet")
        m.remember("Trading platform BTQuant", "Work")
        r = m.recall("dog pet", k=2)
        assert len(r) >= 1
        assert r[0]['similarity'] > 0
        m.close()
    finally: os.unlink(db)

@_testcase("memory: auto-connections", tags=["memory"])
def test_22():
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        m = NeuralMemory(db_path=db, embedding_backend="hash")
        m.remember("Dogs are great pets")
        m.remember("Dogs need daily walks")
        m.remember("Python is a programming language")
        s = m.stats()
        assert s['connections'] >= 1, f"Expected connections, got {s['connections']}"
        m.close()
    finally: os.unlink(db)

@_testcase("memory: spreading activation", tags=["memory", "graph"])
def test_23():
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        m = NeuralMemory(db_path=db, embedding_backend="hash")
        id1 = m.remember("Topic A about dogs")
        m.remember("Topic B about cats")
        t = m.think(id1, depth=3)
        assert isinstance(t, list)
        m.close()
    finally: os.unlink(db)

@_testcase("memory: persistence", tags=["memory"])
def test_24():
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        m1 = NeuralMemory(db_path=db, embedding_backend="hash")
        m1.remember("Persistent fact about dogs")
        m1.close()
        m2 = NeuralMemory(db_path=db, embedding_backend="hash")
        r = m2.recall("dogs", k=1)
        assert len(r) >= 1
        m2.close()
    finally: os.unlink(db)

@_testcase("memory: context manager", tags=["memory"])
def test_25():
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        with NeuralMemory(db_path=db, embedding_backend="hash") as m:
            m.remember("CM test")
            assert len(m.recall("test", k=1)) >= 1
    finally: os.unlink(db)

@_testcase("memory: large batch (100 memories)", tags=["memory", "stress"])
def test_26():
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        m = NeuralMemory(db_path=db, embedding_backend="hash")
        for i in range(100):
            # Use detect_conflicts=False for batch stress test — hash backend
            # generates similar vectors for similar text, triggering conflict merge
            m.remember(f"Entry {i} about unique topic {i}", f"batch-{i}", detect_conflicts=False)
        assert m.stats()['memories'] == 100, f"Expected 100, got {m.stats()['memories']}"
        r = m.recall("unique topic 5", k=5)
        assert len(r) >= 1
        m.close()
    finally: os.unlink(db)

# ============================================================================
# Unified API Tests
# ============================================================================

@_testcase("unified: basic workflow", tags=["api"])
def test_27():
    from neural_memory import Memory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        with Memory(db_path=db, embedding_backend="hash", use_cpp=False) as m:
            m.remember("Test memory")
            r = m.recall("test", k=1)
            assert len(r) >= 1
    finally: os.unlink(db)

@_testcase("unified: stats", tags=["api"])
def test_28():
    from neural_memory import Memory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        with Memory(db_path=db, embedding_backend="hash", use_cpp=False) as m:
            m.remember("a")
            m.remember("b")
            s = m.stats()
            assert s['memories'] == 2
            assert s['embedding_dim'] > 0  # dim depends on backend (hash=384, st=1024)
    finally: os.unlink(db)

# ============================================================================
# C++ Bridge Tests
# ============================================================================

@_testcase("cpp: library symbols exist", tags=["cpp"])
def test_29():
    import subprocess
    lib = os.path.expanduser("~/projects/neural-memory-adapter/build/libneural_memory.so")
    if not os.path.exists(lib):
        raise SkipTest("C++ library not built")
    r = subprocess.run(["nm", "-D", lib], capture_output=True, text=True)
    for sym in ["neural_memory_create", "neural_memory_store", "neural_memory_retrieve_full"]:
        assert sym in r.stdout, f"Missing symbol: {sym}"

@_testcase("cpp: bridge loads", tags=["cpp"])
def test_30():
    try:
        from cpp_bridge import NeuralMemoryCpp
    except FileNotFoundError:
        raise SkipTest("C++ library not found")
    m = NeuralMemoryCpp()
    assert m is not None

# ============================================================================
# Hermes Plugin Tests
# ============================================================================

@_testcase("hermes: plugin files exist", tags=["hermes"])
def test_31():
    plugin = Path.home() / ".hermes/hermes-agent/plugins/memory/neural"
    assert (plugin / "__init__.py").exists()
    assert (plugin / "config.py").exists()
    assert (plugin / "plugin.yaml").exists()

@_testcase("hermes: plugin loads", tags=["hermes"])
def test_32():
    sys.path.insert(0, str(Path.home() / "projects/neural-memory-adapter/python"))
    sys.path.insert(0, str(Path.home() / ".hermes/hermes-agent"))
    from plugins.memory.neural import NeuralMemoryProvider
    p = NeuralMemoryProvider()
    assert p.name == "neural"

@_testcase("hermes: tool schemas", tags=["hermes"])
def test_33():
    sys.path.insert(0, str(Path.home() / "projects/neural-memory-adapter/python"))
    sys.path.insert(0, str(Path.home() / ".hermes/hermes-agent"))
    from plugins.memory.neural import ALL_TOOL_SCHEMAS
    names = [s['name'] for s in ALL_TOOL_SCHEMAS]
    assert "neural_remember" in names
    assert "neural_recall" in names
    assert "neural_think" in names
    assert "neural_graph" in names

@_testcase("hermes: config loads", tags=["hermes"])
def test_34():
    sys.path.insert(0, str(Path.home() / "projects/neural-memory-adapter/python"))
    sys.path.insert(0, str(Path.home() / ".hermes/hermes-agent"))
    from plugins.memory.neural.config import get_config
    cfg = get_config()
    assert 'db_path' in cfg
    assert 'embedding_backend' in cfg

# ============================================================================
# Performance Tests
# ============================================================================

@_testcase("perf: embed 100 texts < 1s (hash)", tags=["perf"])
def test_35():
    from embed_provider import HashBackend
    b = HashBackend(dim=384)
    t0 = time.time()
    for i in range(100):
        b.embed(f"test text number {i}")
    dt = time.time() - t0
    assert dt < 1.0, f"Too slow: {dt:.2f}s for 100 embeds"

@_testcase("perf: store 100 memories < 2s", tags=["perf"])
def test_36():
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        m = NeuralMemory(db_path=db, embedding_backend="hash")
        t0 = time.time()
        for i in range(100):
            m.remember(f"Memory {i}")
        dt = time.time() - t0
        assert dt < 2.0, f"Too slow: {dt:.2f}s for 100 stores"
        m.close()
    finally: os.unlink(db)

@_testcase("perf: recall top-5 from 100 < 0.5s", tags=["perf"])
def test_37():
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        m = NeuralMemory(db_path=db, embedding_backend="hash")
        for i in range(100):
            m.remember(f"Memory about topic {i}")
        t0 = time.time()
        r = m.recall("topic 50", k=5)
        dt = time.time() - t0
        assert dt < 0.5, f"Too slow: {dt:.2f}s for recall"
        assert len(r) >= 1
        m.close()
    finally: os.unlink(db)

# ============================================================================
# Runner
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", help="Comma-separated tags to run")
    parser.add_argument("--list-tags", action="store_true")
    args = parser.parse_args()
    
    # Collect all test functions
    tests = []
    for name, obj in list(globals().items()):
        if callable(obj) and hasattr(obj, '_tags'):
            tests.append(obj)
    
    if args.list_tags:
        all_tags = set()
        for t in tests:
            all_tags.update(t._tags)
        print("Available tags:", ", ".join(sorted(all_tags)))
        return
    
    filter_tags = set(args.tags.split(",")) if args.tags else None
    
    print("=" * 50)
    print("  Neural Memory Adapter - Test Suite")
    print("=" * 50)
    if filter_tags:
        print(f"  Tags: {', '.join(filter_tags)}")
    print()
    
    for t in tests:
        if filter_tags and not set(t._tags) & filter_tags:
            continue
        t()
    
    print()
    print("=" * 50)
    print(f"  {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 50)
    
    sys.exit(1 if FAIL > 0 else 0)

if __name__ == "__main__":
    main()
