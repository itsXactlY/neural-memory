# Neural Memory Adapter for Hermes Agent

Semantic memory system with knowledge graph, spreading activation, embedding-based recall, and **autonomous dream consolidation** for the Hermes Agent.


[![Demo](assets/cover.png)](https://github.com/user-attachments/assets/2d938624-cc39-4f8b-b35b-485b23e93355)

## Features

- **Semantic Memory Storage**: Store memories with automatic embedding generation
- **Knowledge Graph**: Automatic connection of related memories based on similarity
- **Spreading Activation**: Explore connected ideas through graph traversal
- **Conflict Detection**: Automatically detect and update conflicting memories
- **Dream Engine**: Autonomous background consolidation (NREM/REM/Insight phases)
- **MSSQL Backend**: Optional shared database for multi-agent setups
- **CUDA Acceleration**: GPU-accelerated embeddings via sentence-transformers

## Installation

```bash
cd ~/projects/neural-memory-adapter
bash install_database.sh   # Setup database
bash install.sh            # Install plugin
```

### Installation Modes

| Mode | Command | RAM | Backend | Embeddings |
|------|---------|-----|---------|------------|
| **Lite** | `bash install.sh --lite` | ~50MB | SQLite | hash/tfidf (auto) |
| **Full Stack** | `bash install.sh --full` | ~500MB | SQLite + MSSQL | sentence-transformers |

**Lite** — Budget VPS friendly. No GPU, no Docker, no external services. Perfect for small setups.

**Full Stack** — Production. MSSQL shared database, sentence-transformers embeddings, optional GPU. Supports multi-agent dream consolidation.

The installer will:
1. Check/install Python dependencies
2. Build the C++ library (optional)
3. Create databases (SQLite + optionally MSSQL)
4. Install the Hermes plugin
5. Install the neural-dream-engine skill
6. Configure Hermes

### Prerequisites for Full Stack

- MSSQL Server running (`sudo systemctl start mssql-server`)
- ODBC Driver 18 (`yay -S msodbcsql18`)
- `pyodbc` (`pip install pyodbc`)

## Configuration

### Credentials (`.env`)

MSSQL credentials go in `~/.hermes/.env` (never hardcoded):

```bash
MSSQL_SERVER=localhost
MSSQL_DATABASE=NeuralMemory
MSSQL_USERNAME=SA
MSSQL_PASSWORD=your_password_here
MSSQL_DRIVER={ODBC Driver 18 for SQL Server}
```

Resolution order: env vars > `.env` > config.yaml > defaults.

### Config (`config.yaml`)

```yaml
memory:
  provider: neural
  neural:
    db_path: ~/.neural_memory/memory.db
    embedding_backend: sentence-transformers  # or: auto
    prefetch_limit: 10
    search_limit: 10
    dream:
      enabled: true
      idle_threshold: 300        # seconds before dream cycle
      memory_threshold: 50       # dream after N new memories
      mssql:                     # optional: MSSQL backend for dreams
        server: localhost
        database: NeuralMemory
```

## Tools

When active, the following tools are available:

| Tool | Description |
|------|-------------|
| `neural_remember` | Store a memory (with conflict detection) |
| `neural_recall` | Search memories by semantic similarity |
| `neural_think` | Spreading activation from a memory (engine: `bfs` or `ppr`) |
| `neural_graph` | View knowledge graph statistics |
| `neural_dream` | Force a dream cycle (all/nrem/rem/insight) |
| `neural_dream_stats` | Dream engine statistics |
| `neural_dashboard` | Generate Plotly HTML dashboard from current DB state (H10) |

## Phase B upgrades (2026-04-18 and later)

Seven retrieval + consolidation upgrades on top of the original design. All
additive — defaults preserve prior behavior; new capabilities activate via
constructor params or when optional deps are installed.

### Salience-aware recall (default on)

Every recall multiplies `combined = (1-tw)*sim + tw*temporal` by an effective
salience factor:

```
base * exp(-k*age_days) + log1p(access_count) * α    clamped to [0.1, 2.0]
```

Stale memories decay gently; frequently-accessed memories boost. Opt out with
`NeuralMemory(salience_multiply=False)`.

### Bi-temporal edges (backward compat)

The `connections` table gains nullable `event_time`, `ingestion_time`,
`valid_from`, `valid_to`. Pre-existing edges (all NULL) are always-valid.
Query historical graph state with `store.get_connections(mem_id, at_time=ts)`.
Follows the [Graphiti](https://github.com/getzep/graphiti) design.

### Cross-encoder reranker (opt-in)

```python
NeuralMemory(rerank=True, rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
```

Lazy-loaded; silent no-op if `sentence-transformers` absent. Reranks top-k*3
candidates before slicing to k.

### Personalized PageRank think() engine (opt-in)

```python
mem.think(start_id, engine='ppr', alpha=0.15)
```

BFS remains default. PPR is the HippoRAG-2 approach — principled convergence,
handles hubs correctly. Uses `networkx.pagerank` if available; pure-numpy
power-iteration fallback otherwise.

### HNSW index + lazy graph (opt-in)

```python
NeuralMemory(use_hnsw=True, lazy_graph=True, hnsw_ef=100)
```

`use_hnsw=True` (default when hnswlib installed) adds ANN retrieval for the
Python-only recall path. `lazy_graph=True` defers eager full-DB load —
scales past ~10k memories. `hnsw_ef` tunes the accuracy/speed tradeoff.

### Louvain community detection in Insight phase

Automatic when `networkx` installed. Replaces BFS connected-components
(which degenerated on dense graphs). Deterministic seed=42. BFS fallback
if networkx is absent.

### LongMemEval benchmarks

```bash
python3 benchmarks/lme_eval.py                           # synthetic smoke
python3 benchmarks/lme_real.py --dataset /path/to/lme    # real dataset
```

Reports R@{1,5,10}, MRR, p50/p95 latency. Flags for `--rerank --use-hnsw
--engine`. Real dataset: `huggingface.co/datasets/xiaowu0162/longmemeval`
(note: one underscore, not two).

## Optional dependencies

All optional; probed by `install.sh` with warn-if-absent. Each unlocks a
specific feature without breaking anything if absent.

| Dep | Enables |
|-----|---------|
| `sentence-transformers` | MiniLM-L6-v2 semantic embeddings + cross-encoder reranker |
| `networkx` | Louvain community detection + faster PPR |
| `hnswlib` | HNSW ANN fast-path in the Python-only recall path |
| `pyodbc` | MSSQL backend for shared-agent dream consolidation |

Install all three Phase-B deps:
```bash
pip install sentence-transformers networkx hnswlib
```

## Feature-state introspection

```python
mem.stats()
# returns {
#   memories, connections, embedding_dim, embedding_backend,
#   cpp_available, hnsw_active, hnsw_count, hnsw_ef,
#   lazy_graph, louvain_available, reranker_loaded,
#   rerank_enabled, salience_multiply
# }
```

## Maintenance tools

- `tools/compact.py` — weekly compaction sweep (H11). Hard-deletes memories
  matching all of: `age > 180 days`, `effective_salience < 0.15`, `0 valid edges`,
  label not in sticky-prefix whitelist. Dry-run default; audit log per deletion.
- `tools/observer.py` — continuous ingest daemon (A6). Watches git repos +
  Obsidian vaults, writes changes as memories via the `remember` CLI. Launchd
  plist at `~/Library/LaunchAgents/com.ae.neural-observer.plist`.
- `tools/obsidian_sync.py` — mirror the memory graph into an Obsidian vault
  as one `.md` per memory with wikilinks matching SQL edges. Obsidian's graph
  view then renders your live memory graph.
- `tools/dashboard/generate.py` — Plotly HTML dashboard (memory categories,
  connection strength, knowledge graph ring, degree distribution, flow).

## Dream Engine

Autonomous background memory consolidation inspired by biological sleep:

**Phase 1 — NREM (Replay & Consolidation)**
Replays recent memories via spreading activation. Active connections get strengthened (+0.05), inactive ones weakened. Dead connections pruned.

**Phase 2 — REM (Exploration & Bridge Discovery)**
Finds isolated memories, discovers semantically similar unconnected memories, creates tentative bridge connections.

**Phase 3 — Insight (Community Detection)**
Finds connected components (communities), identifies bridge nodes, creates abstract insight entries.

### Triggers

- Automatic: after 5 min idle (configurable)
- Automatic: every 50 new memories (configurable)
- Manual: `neural_dream` tool
- Cron: every 6 hours (default)

### Standalone Worker

```bash
# One-shot cycle
python hermes-plugin/dream_worker.py

# Specific phase
python hermes-plugin/dream_worker.py --phase nrem

# Daemon mode
python hermes-plugin/dream_worker.py --daemon --idle 300
```

## Architecture

### Python Components

- `memory_client.py`: Main NeuralMemory class (remember/recall/think/graph)
- `embed_provider.py`: Embedding backends (sentence-transformers, tfidf, hash)
- `neural_memory.py`: Lower-level memory operations
- `dream_engine.py`: Dream engine core + SQLite backend
- `dream_mssql_store.py`: MSSQL backend for dream engine
- `dream_worker.py`: Standalone full-stack dream worker
- `cpp_bridge.py`: Optional C++ acceleration bridge

### C++ Components (Optional)

- `libneural_memory.so`: SIMD-accelerated vector operations
- `knowledge_graph.cpp`: Graph operations
- `hopfield.cpp`: Hopfield network for pattern completion

## Testing

```bash
cd ~/projects/neural-memory-adapter/python
python3 demo.py
```

## File Structure

```
neural-memory-adapter/
├── install.sh                    # Plugin installer (Lite/Full picker)
├── install_database.sh           # Database setup (SQLite/MSSQL)
├── .env.example                  # Credential template
├── hermes-plugin/                # Plugin files (deployed to Hermes)
│   ├── __init__.py               # MemoryProvider + tools
│   ├── config.py                 # Configuration loader
│   ├── plugin.yaml               # Plugin metadata
│   ├── memory_client.py          # Main client
│   ├── embed_provider.py         # Embedding backends
│   ├── dream_engine.py           # Dream engine (SQLite backend)
│   ├── dream_mssql_store.py      # Dream engine (MSSQL backend)
│   ├── dream_worker.py           # Standalone dream worker
│   ├── mssql_store.py            # MSSQL storage backend
│   └── skills/                   # Bundled skills
│       └── neural-dream-engine/
│           └── SKILL.md
├── python/                       # Python source files
│   ├── memory_client.py          # Main client (source of truth)
│   ├── dream_mssql_store.py      # MSSQL backend
│   ├── dream_worker.py           # Standalone worker
│   ├── import_honcho.py          # Honcho migration tool
│   └── demo.py                   # Demo script
├── skills/                       # Skills for installer
│   └── neural-dream-engine/
│       └── SKILL.md
├── src/                          # C++ source files
├── build/                        # Build artifacts
├── tools/
│   └── dashboard/                # Interactive HTML dashboard
│       ├── generate.py
│       └── template.html
└── README.md
```

## Memory Storage

- **Database**: SQLite at `~/.neural_memory/memory.db`
- **MSSQL**: `localhost/NeuralMemory` (Full Stack only)
- **Embeddings**: Cached in `~/.neural_memory/models/` (~87MB, once)
- **Graph**: In-memory graph loaded on startup

## Conflict Detection

When storing a memory with similar content to an existing one:
- High similarity (>0.7) + different content → updates existing memory
- Marks old memory as `[SUPERSEDED]` and adds `[UPDATED TO]`
- Returns the existing memory ID instead of creating duplicate

## Dashboard

Interactive HTML dashboard with knowledge graph visualization, category breakdowns, and connection analysis.

```bash
# From SQLite (default)
python tools/dashboard/generate.py

# From MSSQL
python tools/dashboard/generate.py --mssql --mssql-password 'yourpass'

# Custom output path
python tools/dashboard/generate.py -o /tmp/dashboard.html
```

Opens a self-contained HTML file with Plotly charts:
- **Category donut** -- memory type distribution
- **Connection strength** -- weight histogram
- **Knowledge graph** -- top 50 hub nodes, force-layout colored by category
- **Degree scatter** -- node degree vs avg connection weight
- **Category heatmap** -- connection flow between memory types

## Troubleshooting

### Plugin not loading

Check if `tool_error` function exists in `tools/registry.py`:
```bash
grep -n "def tool_error" ~/.hermes/hermes-agent/tools/registry.py
```

### Dependencies missing

```bash
# Lite
pip install numpy

# Full Stack
pip install sentence-transformers numpy pyodbc
```

### Database issues

Delete the database to start fresh:
```bash
rm ~/.neural_memory/memory.db
```

### MSSQL connection

Verify credentials in `~/.hermes/.env`:
```bash
grep MSSQL ~/.hermes/.env
```

## License

See LICENSE file.
