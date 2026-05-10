![Demo](https://github.com/user-attachments/assets/2d938624-cc39-4f8b-b35b-485b23e93355)

# Mazemaker Plugin

Local semantic memory with knowledge graph, spreading activation, and auto-connections.
Runs entirely offline — no API keys, no cloud.

## Features

- **Semantic search** via vector embeddings (FastEmbed ONNX, 1024d) with multi-channel fusion (semantic + BM25 + entity + temporal + PPR)
- **Optional ColBERT late-interaction rerank** (opt-in via `MM_COLBERT_ENABLED=1`) — verified +5.10 pp R@1 / +3.81 pp MRR on LongMemEval-S 500q
- **Knowledge graph** with automatic connection discovery between related memories
- **Spreading activation** for exploring connected ideas beyond direct similarity
- **Conflict detection** — detect and supersede conflicting memories
- **Dream Engine** — autonomous background consolidation (NREM/REM/Insight)
- **GPU recall** — CUDA-accelerated cosine similarity (~100ms for 10K memories)
- **Postgres + pgvector** primary backend optional for shared multi-agent / Pro deployments
- **Fully offline-capable** — SQLite primary always works; license heartbeat is the only outbound traffic

## Configuration

```yaml
# ~/.hermes/config.yaml
memory:
  provider: neural
  neural:
    db_path: ~/.neural_memory/memory.db
    embedding_backend: fastembed  # fastembed|hash|tfidf|sentence-transformers|auto
```

Or via environment variables:
- `NEURAL_MEMORY_DB_PATH` — SQLite database path
- `NEURAL_EMBEDDING_BACKEND` — Embedding backend selection

## Tools

The plugin surfaces the four core tools as Hermes MCP schemas. The full nine-tool set (dream control + telemetry, stats, prune, quota) is callable through the standalone `Memory` class and the daemon path — see the root [README](../README.md).

| Tool | Description |
|------|-------------|
| `mazemaker_remember` | Store a memory (auto-embedded, auto-connected, with conflict detection) |
| `mazemaker_recall` | Multi-channel search (semantic + BM25 + entity + temporal + PPR + optional ColBERT) |
| `mazemaker_think` | Spreading activation — BFS or Personalized PageRank over the connection graph |
| `mazemaker_graph` | Knowledge graph statistics — memory count, density, strongest associations |

## Embedding Backends (auto-priority)

| Priority | Backend | Model | Speed | Requirements |
|----------|---------|-------|-------|--------------|
| 1st | FastEmbed | intfloat/multilingual-e5-large | ~50ms | `pip install fastembed` |
| 2nd | sentence-transformers | BAAI/bge-m3 1024d | ~200ms | GPU recommended |
| 3rd | tfidf | — | varies | numpy only |
| 4th | hash | — | instant | nothing |

## Dependencies

```bash
pip install numpy fastembed sqlite-utils
```

For GPU recall:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Clean VM Verified

Tested on fresh Debian 12 QEMU/KVM (4GB RAM) — all 12 MemoryProvider integration tests pass.
See main README for full test results.

4GB RAM minimum for FastEmbed model download. Use `--hash-backend` or `install.sh --hash-backend` for constrained environments.
