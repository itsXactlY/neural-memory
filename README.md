# Neural Memory Adapter for Hermes Agent

Semantic memory system with knowledge graph, spreading activation, embedding-based recall, **GPU-accelerated search**, **autonomous dream consolidation**, for the Hermes Agent.

## Architecture (2026-04)

- **FastEmbed** (ONNX/CPU) - ~50ms/emb - intfloat/multilingual-e5-large 1024d
- **GPU Recall** (CUDA/RTX) - ~100ms/k=3 - torch.matmul top-k (8.8 MB VRAM)
- **Dream Engine** (Background) - NREM/REM/Insight via threading.Event
- **SQLite Store** (Source of Truth) - 34MB @ 2264 memories

### Key Design Decisions

- **FastEmbed > sentence-transformers**: ONNX Runtime, no PyTorch/CUDA conflict, ~50ms per embedding
- **GPU Recall > C++ Bridge**: CUDA cosine similarity via torch.matmul - 100ms for 2264 memories. C++ Hopfield network biased on training data.
- **SQLite = Source of Truth**: MSSQL optional mirror. SQLite is faster, simpler.
- **Raw embeddings on GPU**: FastEmbed produces ~28 magnitude vectors. GPU handles normalization.

## Features

- **Semantic Memory Storage**: FastEmbed embeddings (intfloat/multilingual-e5-large, 1024d)
- **GPU Recall Engine**: CUDA-accelerated cosine similarity search (~100ms)
- **Knowledge Graph**: Automatic connection of related memories
- **Spreading Activation**: BFS graph traversal with decay
- **Conflict Detection**: Auto-supersede conflicting memories
- **Dream Engine**: Autonomous background consolidation (NREM/REM/Insight phases)
- **MSSQL Mirror**: Optional shared database for multi-agent setups
- **C++ LSTM+kNN**: Optional (has Hopfield bias - use GPU recall instead)

## Installation

[1m
╔══════════════════════════════════════════════╗
║   Neural Memory Adapter — Installer          ║
║   Local semantic memory for hermes-agent     ║
╚══════════════════════════════════════════════╝
[0m
[0;32m✓[0m hermes-agent: /home/alca/.hermes/hermes-agent
[0;32m✓[0m Plugin target: /home/alca/.hermes/hermes-agent/plugins/memory/neural
[0;32m✓[0m Python: Python 3.14.4
[0;36m→[0m Checking dependencies...
[0;32m✓[0m pyodbc (MSSQL)
[0;32m✓[0m numpy
[0;32m✓[0m Cython
[0;32m✓[0m sentence-transformers
[0;36m→[0m Installing plugin...
[0;32m✓[0m Plugin files installed
[0;36m→[0m Building Cython fast_ops...
Compiling fast_ops.pyx because it changed.
[1/1] Cythonizing fast_ops.pyx
running build_ext
building 'fast_ops' extension
gcc -fno-strict-overflow -Wsign-compare -DNDEBUG -g -O3 -Wall -march=x86-64 -mtune=generic -O3 -pipe -fno-plt -fexceptions -Wp,-D_FORTIFY_SOURCE=3 -Wformat -Werror=format-security -fstack-clash-protection -fcf-protection -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -g -ffile-prefix-map=/build/python/src=/usr/src/debug/python -flto=auto -ffat-lto-objects -march=x86-64 -mtune=generic -O3 -pipe -fno-plt -fexceptions -Wp,-D_FORTIFY_SOURCE=3 -Wformat -Werror=format-security -fstack-clash-protection -fcf-protection -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -g -ffile-prefix-map=/build/python/src=/usr/src/debug/python -flto=auto -march=x86-64 -mtune=generic -O3 -pipe -fno-plt -fexceptions -Wp,-D_FORTIFY_SOURCE=3 -Wformat -Werror=format-security -fstack-clash-protection -fcf-protection -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -g -ffile-prefix-map=/build/python/src=/usr/src/debug/python -flto=auto -fPIC -I/usr/lib/python3.14/site-packages/numpy/_core/include -I/usr/include/python3.14 -c fast_ops.c -o build/temp.linux-x86_64-cpython-314/fast_ops.o -O3 -march=native
gcc -shared -Wl,-O1 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,-z,pack-relative-relocs -flto=auto -Wl,-O1 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,-z,pack-relative-relocs -flto=auto build/temp.linux-x86_64-cpython-314/fast_ops.o -L/usr/lib -o build/lib.linux-x86_64-cpython-314/fast_ops.cpython-314-x86_64-linux-gnu.so
copying build/lib.linux-x86_64-cpython-314/fast_ops.cpython-314-x86_64-linux-gnu.so -> 
[0;32m✓[0m fast_ops compiled and installed
[0;36m→[0m Checking C++ library...
[0;32m✓[0m C++ bridge available: /home/alca/projects/neural-memory-adapter/build/libneural_memory.so
[1;33m⚠[0m Database exists but may be corrupted
[0;36m→[0m Verifying installation...
Embedding backend: HashBackend (1024d)
  NeuralMemory: 2264 memories
[0;32m✓[0m NeuralMemory importable

[1m═══════════════════════════════════════════[0m
[0;32m Installation complete![0m
[1m═══════════════════════════════════════════[0m

  To activate, add to config.yaml:

    memory:
      provider: neural

  Or run: hermes memory setup
          → select 'neural' from the list

  Optional: MSSQL backend
    Add to config.yaml:
      memory:
        neural:
          dream:
            mssql:
              server: 127.0.0.1
              database: NeuralMemory
              username: SA
              password: <your-password>

  Restart hermes to load the plugin.

### What the installer does

1. Detects hermes-agent location
2. Installs Python deps: fastembed, sentence-transformers, torch, numpy
3. Copies plugin files to hermes-agent/plugins/memory/neural/
4. Builds Cython fast_ops (optional)
5. Builds C++ library (optional, has Hopfield bias)
6. Initializes SQLite database
7. Configures Hermes (memory.provider: neural)

### Dependencies (auto-installed)

| Package | Purpose | Size |
|---------|---------|------|
| fastembed | Embedding (ONNX, CPU) | ~200MB |
| sentence-transformers | GPU batch embedding | ~1.2GB |
| torch | GPU recall (matmul) | ~2GB |
| numpy | Array ops | ~30MB |
| pyodbc | MSSQL (optional) | ~1MB |

## Configuration

All settings in ~/.hermes/config.yaml:



### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| NEURAL_MEMORY_DB_PATH | ~/.neural_memory/memory.db | SQLite path |
| NEURAL_EMBEDDING_BACKEND | auto | Embedding backend |
| EMBED_MODEL | intfloat/multilingual-e5-large | Model name |

## GPU Recall Engine

gpu_recall.py loads all embeddings onto GPU for sub-100ms cosine similarity search.

### How it works
1. Loads ~/.neural_memory/gpu_cache/embeddings.npy (8.8 MB for 2264 memories)
2. Moves tensor to CUDA device
3. On query: embeds with FastEmbed, runs torch.matmul on GPU
4. Returns top-k via torch.topk

### Rebuild cache
rm -rf ~/.neural_memory/gpu_cache/  # then restart

## Database Management

### SQLite (Source of Truth)
2264

### MSSQL (Optional Mirror)
=== Neural Memory SQLite -> MSSQL Sync ===

[1/3] SQLite: /home/alca/.neural_memory/memory.db

============================================================
  Neural Memory MSSQL Production Migration
============================================================
  Config: ~/.hermes/config.yaml
  SQLite: ~/.neural_memory/memory.db
  Dry-run: False
  History retention: 3 days


============================================================
  DIAGNOSIS
============================================================

--- Tables ---
  memories                              2,264 rows      18 MB
  connection_history                    3,716 rows       0 MB
  GraphEdges_v2                             0 rows       0 MB
  GraphNodes_v2                             0 rows       0 MB
  NeuralMemory                              0 rows       0 MB
  connections                             747 rows       0 MB
  dream_insights                        1,041 rows       0 MB
  dream_sessions                        1,392 rows       0 MB

  DB total: 37 MB

--- Unique Constraints ---
  ✓ connection_history.PK__connecti__3213E83F9DA98505 (CLUSTERED)
  ✓ connection_history.UX_connection_history_unique (NONCLUSTERED)
  ✓ connections.PK__connecti__3213E83F61D8F24F (CLUSTERED)
  ✓ connections.UX_connections_unique (NONCLUSTERED)
  ✓ dream_insights.PK__dream_in__3213E83F58EA27F7 (CLUSTERED)
  ✓ dream_sessions.PK__dream_se__3213E83F175B7806 (CLUSTERED)
  ✓ GraphEdges_v2.PK__GraphEdg__50F0513B469A45F8 (CLUSTERED)
  ✓ GraphNodes_v2.PK__GraphNod__5F19EF164CF44BB6 (CLUSTERED)
  ✓ memories.PK__memories__3213E83F2CA6FD17 (CLUSTERED)

--- No issues found ---

============================================================
  Functional Verification
============================================================
  ✓ memories > 0: 2264
  ✓ connections > 0: 747
  ✓ connection_history > 0: 3716
  ✗ GraphNodes_v2 > 0: 0
  ✗ GraphEdges_v2 > 0: 0
  ✓ No connection dupes: 0
  ✓ No history dupes: 0
  ✓ UX_connections_unique exists: 1
  ✓ UX_connection_history_unique exists: 1
  ✓ No orphan connections: 0
  ✓ No NeuralMemory_old: 0
  ✓ No GraphNodes (V1): 0
  ✓ No GraphEdges (V1): 0
  ✓ MERGE on connections works: 1
    (MERGE test on (24162,16507) — updated and restored)

  Score: 12/14 passed

============================================================
  Code Verification (MERGE/UPSERT)
============================================================
  ✗ mssql_store.py: 1 raw INSERT(s) found
      mssql_store.py:131: raw INSERT INTO connections

  3 issue(s) found ✗

## Troubleshooting

### Recall returns wrong results
- C++ bridge has Hopfield bias. Set use_cpp=False in __init__.py
- GPU cache stale: rm -rf ~/.neural_memory/gpu_cache/ and restart

### Embedding too slow
- Use FastEmbed (ONNX, ~50ms) not sentence-transformers
- For batch: sentence-transformers with CUDA (108 mem/s on RTX 4060 Ti)

## File Structure



## License

MIT
