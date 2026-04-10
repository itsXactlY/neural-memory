# Neural Memory Adapter for Hermes Agent

Semantic memory system with knowledge graph, spreading activation, and embedding-based recall for the Hermes Agent.

## Features

- **Semantic Memory Storage**: Store memories with automatic embedding generation
- **Knowledge Graph**: Automatic connection of related memories based on similarity
- **Spreading Activation**: Explore connected ideas through graph traversal
- **Conflict Detection**: Automatically detect and update conflicting memories
- **CUDA Acceleration**: GPU-accelerated embeddings via sentence-transformers

## Installation

```bash
cd ~/projects/neural-memory-adapter
bash install.sh
```

The installer will:
1. Check/install Python dependencies (sentence-transformers, numpy)
2. Build the C++ library (optional, for performance)
3. Create necessary directories (~/.neural_memory/)
4. Install the Hermes plugin
5. Configure Hermes to use neural memory

## Configuration

Add to `~/.hermes/config.yaml`:

```yaml
memory:
  provider: neural
  neural:
    db_path: ~/.neural_memory/memory.db
    embedding_backend: auto
    consolidation_interval: 0
    max_episodic: 0
    prefetch_limit: 10
    search_limit: 10
```

## Tools

When active, the following tools are available:

- `neural_remember`: Store a memory (with conflict detection)
- `neural_recall`: Search memories by semantic similarity
- `neural_think`: Spreading activation from a memory
- `neural_graph`: View knowledge graph statistics

## Architecture

### Python Components

- `memory_client.py`: Main NeuralMemory class with remember/recall/think/graph
- `embed_provider.py`: Embedding backends (sentence-transformers, tfidf, hash)
- `neural_memory.py`: Lower-level memory operations
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
├── install.sh              # Installation script
├── hermes-plugin/          # Plugin files (deployed to Hermes)
│   ├── __init__.py         # MemoryProvider implementation
│   ├── config.py           # Configuration loader
│   ├── plugin.yaml         # Plugin metadata
│   ├── memory_client.py    # Main client
│   ├── embed_provider.py   # Embedding backends
│   └── ...
├── python/                 # Python source files
│   ├── memory_client.py    # Main client (source of truth)
│   ├── demo.py             # Demo script
│   └── ...
├── src/                    # C++ source files
├── build/                  # Build artifacts
├── tools/
│   └── dashboard/          # Interactive HTML dashboard
│       ├── generate.py     # Generate dashboard from SQLite or MSSQL
│       └── template.html   # Plotly dashboard template
└── README.md
```

## Memory Storage

- **Database**: SQLite at `~/.neural_memory/memory.db`
- **Embeddings**: Cached in `~/.neural_memory/models/`
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
pip install sentence-transformers numpy
```

### Database issues

Delete the database to start fresh:
```bash
rm ~/.neural_memory/memory.db
```

## License

See LICENSE file.
