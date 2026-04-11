[![Demo](assets/cover.png)](https://github.com/user-attachments/assets/2d938624-cc39-4f8b-b35b-485b23e93355)

# Neural Memory Adapter for Hermes Agent

Semantic memory system with knowledge graph, spreading activation, embedding-based recall, and **autonomous dream consolidation** for the Hermes Agent.

> **Integration Status:** The upstream PR ([NousResearch/hermes-agent#7185](https://github.com/NousResearch/hermes-agent/pull/7185)) is currently open. Until it is merged, use the fork for full integration:
> ```
> https://github.com/itsXactlY/hermes-agent/tree/neural-memory-clean
> ```

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
| `neural_think` | Spreading activation from a memory |
| `neural_graph` | View knowledge graph statistics |
| `neural_dream` | Force a dream cycle (all/nrem/rem/insight) |
| `neural_dream_stats` | Dream engine statistics |

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
