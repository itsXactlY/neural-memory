# Mazemaker Adapter for Hermes Agents and any MCP-compatible system

> **Give your AI a memory that actually sticks.**
> One install, and your assistant remembers across sessions — not just what you said, but how the pieces fit together.

![Neural Brain Hero](assets/neural_brain_hero.png)

---

## In one minute

You know how AI assistants forget everything between chats? This fixes that.

Plug it in, and your assistant gets a brain that:

- 📌 **Remembers what you tell it** — preferences, decisions, fixes, the path of that file you mentioned three weeks ago.
- 🧠 **Connects related ideas** — like a real notebook with cross-references, not a search box.
- 😴 **Reflects while you sleep** — overnight, it strengthens what matters and notices new connections.
- ✏️ **Updates itself when you change your mind** — old facts get superseded, not duplicated.

That's it. The rest of this README goes deeper the further you scroll.

> **🌀 Don't want to install anything?** A managed hosted version is in the works:
> **[mazemaker.dev](https://mazemaker.dev)** / **[mazemaker.online](https://mazemaker.online)**
> — *Build the maze. Your agent finds the way.*  Sign up, point your agent at an MCP endpoint, done. (Both domains are placeholders until launch.)

---

## Get it running

```bash
cd ~/projects/mazemaker-adapter
bash install.sh          # auto-detect hermes-agent
bash install.sh /path    # explicit path
```

The installer figures out:

1. Python deps (FastEmbed, torch, numpy)
2. Whether you have a GPU
3. Where to deploy the plugin
4. Database init (SQLite at `~/.neural_memory/memory.db`)
5. Config setup

Restart hermes after install: `hermes gateway restart`

That's the beginner path. If anything goes wrong, the [Production Lessons](#production-lessons) section near the bottom has every gotcha I've seen on a clean VM.

**Live Dashboard — Knowledge Graph**

[![Dashboard](assets/neural_memory.png)](https://raw.githubusercontent.com/itsXactlY/mazemaker/refs/heads/master/assets/neural_memory.png)

---

## Why it's different from "ChatGPT memory"

Most AI memory systems are search engines: type a query, get back the few documents that contain the same words. That works for simple recall. It falls apart for everything else an agent actually needs.

| Need                                                           | Search-engine memory | This system |
|----------------------------------------------------------------|----------------------|-------------|
| Find a fact you told it once                                  | ✅                   | ✅          |
| Follow a chain of reasoning across multiple memories          | ❌                   | ✅          |
| Notice that two related facts should be connected             | ❌                   | ✅          |
| Replace a stale fact when you tell it the new one             | ❌                   | ✅          |
| Hold its ground when irrelevant noise piles up                 | ❌                   | ✅          |
| Tell you *why* it surfaced a particular memory                 | ❌                   | ✅          |

The right mental model isn't "vector database." It's **a small brain that lives next to your agent.**

If you want the numbers behind those check marks, keep scrolling. If you don't, the install above is enough — defaults are good.

---

## The numbers

A vector database with cosine similarity will do the first row of that table well and fail every other row. We measured that explicitly.

| Capability                                                          | Vanilla cosine          | Neural-Memory       | Lift               |
|---------------------------------------------------------------------|-------------------------|---------------------|--------------------|
| Hop-2 graph reasoning (answer reachable only via A→B→C edges)      | **0.00** R@10           | **1.00** R@10       | **+1.00**          |
| Real edges vs shuffled control (proves traversal, not embedding)    | n/a                     | 1.00 → 0.27         | **+0.73 collapse** |
| Post-dream synthesis (facts inferable only after consolidation)     | structurally **0.00**   | **0.43** at scale   | **+0.43 lift**     |
| Conflict supersession (winner@1 with `detect_conflicts=False`)      | 0.03 control            | **0.33**            | **+0.30**          |
| Cross-session continuity under concept-mode distractors             | **0.06**                | **0.62**            | **+0.56**          |
| Lean retrieval mode (real prose, n=200) vs default skynet           | n/a                     | **0.60** vs 0.42    | **+0.18 R@5**      |

R@10 = "the right answer is in the top-10 results", scored 0..1. Higher is better. Every cell measured by a suite that *cannot* be solved by token overlap, with negative controls (shuffled edges, supersession off, pre-dream zero) that *must* fail when the relevant mechanism is disabled.

Full numbers, the JSON dumps, and the suite catalog: [`benchmarks/README.md`](benchmarks/README.md).

---

## How we proved it (the audit story)

A peer-review-grade benchmark for this kind of system **didn't exist**. Existing semantic-memory evaluations measure either retrieval (BEIR, MS MARCO) or QA (NaturalQuestions) — none of them test graph traversal, dream consolidation, or supersession.

So we built one, and had it independently audited by **GPT-5.5** (via [codex CLI](https://openai.com/codex)). It pushed back hard. Eight rounds:

| Round | Verdict                                | Headline reason                                                                  |
|-------|----------------------------------------|----------------------------------------------------------------------------------|
| v2    | **no**                                 | Lexical leakage in queries; broken dream suite; no baseline                      |
| v3    | **no**                                 | Topic-word leakage; cross-instance anchor collisions; wrong-class import         |
| v4    | qualified-y                            | Source-level fixes pending verification                                          |
| v5    | **YES** + 4 caveats                    | Every condition empirically satisfied                                            |
| v6    | qualified-y w/ 4 caveats               | Real-text mode + lean preset shipped; 4 follow-ups                               |
| v7    | qualified-y w/ 1 caveat                | n=200 real-prose: lean **beats** default skynet by +0.18 R@5                     |
| **v8**| **UNCONDITIONAL YES — no residual caveat** | Dream lift +0.43 at scale; the +0.04 at v7 was a sample-size artifact            |

Every prompt and every verdict, from "no, this is just lexical retrieval" to "unconditional yes — accept it as evidence", is committed verbatim under [`benchmarks/audit/`](benchmarks/audit/). Open `codex-v2-audit-2026-04-28.md` and `codex-v8-verdict-2026-04-28.md` side by side to see the journey end-to-end.

---

## What the benchmark *gave back* to the production code

Running the benchmark wasn't just measurement. It surfaced real engineering wins. Each one is now a documented, opt-in option in `~/.hermes/config.yaml`:

- **`retrieval_mode: lean`** — channel ablation proved that on real prose, BM25 / temporal / salience are dead-weight (or actively *harmful*). Lean drops them. Result: **4× faster than skynet on synthetic; +0.18 R@5 better than skynet on real prose**. The benchmark told the production code which channels to remove.
- **`recall_score_percentile`** — the legacy `score_floor` operates on a badly-scaled internal score (~0..0.05); a sensible-looking value like 0.2 silently nukes everything. The new percentile knob is calibrated [0,1] by *rank*, so `0.5` keeps top half regardless of corpus or model.
- **PPR is the load-bearing channel for ranking** (-0.13 MRR if removed); semantic is the load-bearing channel for recall (-0.26 if removed). Surface this in your config tuning.

Run the benchmark yourself:

```bash
# Full v8 run on real-text corpus (200 chunks from the project's own docs):
python -m benchmarks.neural_memory_benchmark.runner \
  --realistic --suite baseline --suite lean_skynet \
  --suite graph_reasoning --suite dream_derived_fact \
  --suite conflict_quality --suite continuity_controls \
  --suite channel_ablation \
  --output-dir benchmarks/results/my-run --seed 42

# Single-suite quick check (graph reasoning is the headline):
python -m benchmarks.neural_memory_benchmark.runner \
  --paraphrase --suite graph_reasoning
```

A full run takes ~12 minutes on a workstation. Every suite produces a JSON file under `benchmarks/results/<your-dir>/results/`.

---

## Features (technical bullets)

If the install + the cheat sheet above is enough for you, you can stop reading here. Below this line everything gets progressively more technical.

- **Semantic memory storage** — auto-embed via FastEmbed ONNX (intfloat/multilingual-e5-large, 1024d). Falls back to sentence-transformers, then TF-IDF, then hash.
- **Knowledge graph** — auto-connect related memories by cosine threshold, plus explicit `add_connection()` for typed edges. Canonical (source<target) orientation enforced everywhere.
- **Spreading activation** — BFS or Personalized PageRank for `think(start_id)`. The only path that solves hop-2 retrieval; vanilla cosine literally cannot.
- **Dream Engine** — three-phase autonomous consolidation: NREM (strengthen activated edges + prune weak), REM (bridge isolated memories), Insight (Louvain communities + materialise `derived:cluster` summary memories).
- **Conflict detection + supersession** — fuse-or-mark with revision history. `detect_conflicts=False` control arm proves the algorithm is doing real work, not just relying on recency.
- **Multi-channel retrieval** — semantic + BM25 + entity + temporal + PPR, fused via Reciprocal Rank Fusion. Six presets (`semantic`, `hybrid`, `advanced`, `skynet`, `lean`, `trim`).
- **GPU recall** — CUDA-accelerated cosine over an in-memory matrix (~100ms for 10k memories). CPU fallback automatic.
- **SQLite-first** — always works, no external DB needed. WAL mode + bg checkpointing. **Postgres + pgvector optional** for shared multi-agent / Pro-tier deployments (set `MM_DB_BACKEND=postgres`).
- **Hermes plugin / MCP server / standalone library** — one core, three integration shapes.

---

## Architecture

### Embedding Backends (auto-priority)

| Priority | Backend | Model | Speed | Requirements |
|----------|---------|-------|-------|--------------|
| 1st | FastEmbed | intfloat/multilingual-e5-large | ~50ms | `pip install fastembed` |
| 2nd | sentence-transformers | BAAI/bge-m3 1024d | ~200ms | GPU recommended |
| 3rd | tfidf | — | varies | numpy only |
| 4th | hash | — | instant | nothing |

FastEmbed uses ONNX runtime — no PyTorch conflict, works on CPU. Falls back automatically.

### GPU Recall Engine

```python
# gpu_recall.py — CUDA cosine similarity
# Loads all embeddings into GPU, does torch.matmul for batch similarity
# ~100ms for 10K memories vs ~500ms CPU

from gpu_recall import GPURecall
engine = GPURecall()
results = engine.recall(query_embedding, all_embeddings, top_k=10)
```

Auto-detects CUDA. Falls back to Python/numpy if no GPU.

### Data Flow

```mermaid
flowchart TD
    subgraph Store["mazemaker_remember"]
        A[User content] --> B[FastEmbed encode]
        B --> C[1024d vector]
        C --> D[SQLite INSERT]
        C --> E[Cosine similarity search]
        E --> F[Create connections]
    end

    subgraph Recall["mazemaker_recall"]
        G[User query] --> H[FastEmbed encode]
        H --> I{CUDA available?}
        I -->|Yes| J[GPU torch.matmul]
        I -->|No| K[CPU numpy dot]
        J --> L[Top-k results]
        K --> L
    end

    subgraph Think["mazemaker_think"]
        M[Source memory] --> N[BFS on connections]
        N --> O[Apply decay factor]
        O --> P[Ranked activation]
    end

    subgraph Dream["mazemaker_dream"]
        Q[Idle trigger] --> R[NREM replay]
        R --> S[REM bridge discovery]
        S --> T[Insight communities]
        T --> U[Consolidated graph]
    end
```

### Storage

- **SQLite (always)**: `~/.neural_memory/memory.db` — source of truth
- **Embeddings cache**: `~/.neural_memory/models/` (auto-downloaded, ~2.2 GB)
- **GPU cache**: `~/.neural_memory/gpu_cache/` (embeddings.npy + metadata.pkl)
- **Access logs**: `~/.neural_memory/access_logs/` (JSON Lines)
- **Postgres + pgvector (optional)**: enabled via `MM_DB_BACKEND=postgres` — graph/cold-storage mirror for shared multi-agent deployments

### SQLite Schema

```sql
-- Core tables
memories (id, content, embedding, category, salience, ...)
connections (source_id, target_id, weight, edge_type)
connection_history (source_id, target_id, last_weight, last_updated)

-- Dream engine
dream_sessions (id, phase, started_at, completed_at, stats)
dream_insights (id, session_id, type, data)

-- Indexes
idx_memories_category ON memories(category)
idx_connections_source ON connections(source_id)
idx_connections_target ON connections(target_id)
```

---

## Configuration (every knob)

All settings in `~/.hermes/config.yaml`. The defaults below are the recommended preset based on the v8 benchmark.

```yaml
memory:
  provider: neural
  neural:
    db_path: ~/.neural_memory/memory.db
    embedding_backend: fastembed       # auto | fastembed | sentence-transformers | tfidf | hash

    # 2026-04-28 benchmark recommended preset.
    # `lean` beat `skynet` by +0.18 R@5 / +0.16 MRR on real prose at n=200,
    # and is 4× faster on synthetic at -0.02 recall. Drops the channels
    # (BM25, temporal, salience) that channel_ablation proved actively
    # hurt recall on real text.
    retrieval_mode: lean               # semantic | hybrid | advanced | skynet | lean | trim
    retrieval_candidates: 128
    use_hnsw: auto                     # ANN index above ~1k memories
    think_engine: ppr                  # bfs | ppr — PPR is the load-bearing channel for ranking

    # Calibrated [0,1] noise floor — drops the bottom X fraction of
    # ranked candidates by RANK. Calibrated alternative to the legacy
    # recall_score_floor (which lived on the badly-scaled raw RRF
    # score ~0..0.05; values >= 0.2 silently nuke everything).
    recall_score_percentile: 0.3

    # Optional: MMR diversity in result set (0.0=pure relevance,
    # 0.7=balanced). Off by default.
    mmr_lambda: 0.0

    # Hermes session knobs
    prefetch_limit: 10
    search_limit: 50
    consolidation_interval: 0
    session_extract_facts: true
    session_fact_limit: 5

    dream:
      enabled: true
      idle_threshold: 600              # seconds before dream cycle
      memory_threshold: 50             # dream after N new memories
    # To enable the Postgres + pgvector mirror, set MM_DB_BACKEND=postgres
    # and supply MM_POSTGRES_DSN (or the discrete MM_POSTGRES_* vars).
```

### Retrieval-mode cheat sheet

| Mode       | Channels active                            | Use when                                     |
|------------|--------------------------------------------|----------------------------------------------|
| `semantic` | semantic only                              | Lowest latency, no hybrid fusion needed      |
| `hybrid`   | semantic + BM25                            | Add lexical recall                           |
| `advanced` | semantic + BM25 + entity                   | + named-entity grounding                     |
| `skynet`   | all six channels                           | Default; over-channeled per benchmark        |
| **`lean`** | semantic + entity + PPR                    | **Recommended** — drops dead-weight channels |
| `trim`     | semantic + BM25 + entity + temporal + PPR  | Conservative middle-ground (drops only salience) |

---

## Tools (LLM-callable surface)

When the plugin is active, these tools appear in Hermes:

| Tool | Description |
|------|-------------|
| `neural_remember` | Store a memory (with conflict detection) |
| `neural_recall` | Search memories by semantic similarity |
| `neural_think` | Spreading activation from a memory |
| `neural_graph` | View knowledge graph statistics |
| `neural_dream` | Force a dream cycle (all/nrem/rem/insight) |
| `neural_dream_stats` | Dream engine statistics |

---

## Dream Engine (deep dive)

Autonomous background memory consolidation, biological-sleep inspired:

```mermaid
flowchart LR
    subgraph Trigger
        T1[Idle 600s] --> D
        T2[50 new memories] --> D
        T3[Manual / Cron] --> D
    end

    D[Dream Cycle] --> NREM
    D --> REM
    D --> INSIGHT

    subgraph NREM["Phase 1 — NREM"]
        direction TB
        N1[Replay 100 recent memories] --> N2[Spreading activation]
        N2 --> N3{Connection active?}
        N3 -->|Yes| N4[Strengthen +0.05]
        N3 -->|No| N5[Weaken -0.01]
        N3 -->|Dead <0.05| N6[Prune]
    end

    subgraph REM["Phase 2 — REM"]
        direction TB
        R1[Find 50 isolated memories] --> R2[Search similar unconnected]
        R2 --> R3[Create bridge connections]
        R3 --> R4[weight = similarity × 0.3]
    end

    subgraph INSIGHT["Phase 3 — Insight"]
        direction TB
        I1[BFS connected components] --> I2[Identify communities]
        I2 --> I3[Find bridge nodes]
        I3 --> I4[Store dream_insights]
    end
```

### Triggers

- Automatic: after 600s idle (configurable)
- Automatic: every 50 new memories (configurable)
- Manual: `neural_dream` tool
- Standalone: `python python/dream_worker.py --daemon`

---

## Testing

### Smoke Test (Quick)

```bash
cd ~/projects/mazemaker-adapter/python
python3 demo.py
```

### Full Test Suite

```bash
# Plugin test suite
cd ~/.hermes/hermes-agent/plugins/memory/neural
python3 test_suite.py

# Upside-Down Test Suite — edge cases, corruption, concurrency, SQL injection
cd ~/projects/mazemaker-adapter
python3 tests/test_upside_down.py
```

### Clean Smoke Test (Any Machine)

```bash
cd ~/projects/mazemaker-adapter
python3 -c "
import sys; sys.path.insert(0, 'python')
from mazemaker import Mazemaker
nm = Mazemaker(db_path='/tmp/test.db', embedding_backend='cpu', use_cpp=False)
mid = nm.remember('test memory', label='smoke')
results = nm.recall('test')
assert len(results) > 0, 'recall failed'
print(f'SMOKE TEST PASS: {len(results)} results')
"
```

### Verified: Clean VM — Debian 12 (2026-04-21)

Tested on a fresh Debian 12 QEMU/KVM VM — hermes-agent + mazemaker only, no jack-in-a-box.

| Property | Value |
|----------|-------|
| VM | Debian 12, 4 GB RAM, KVM enabled |
| hermes-agent | git clone (itsXactlY fork) |
| mazemaker | git clone + FastEmbed ONNX |
| Embedding | intfloat/multilingual-e5-large (1024d) |
| C++ bridge | Not built (Python fallback) |

**All 12 integration tests passed:**

| # | Test | Result |
|---|------|--------|
| 1 | Mazemaker standalone (remember/recall/graph) | PASS |
| 2 | Memory Provider (FastEmbed 1024d) | PASS |
| 3 | NeuralMemoryProvider.__init__ | PASS |
| 4 | is_available() | PASS |
| 5 | initialize(session_id) | PASS |
| 6 | get_tool_schemas() → 4 tools | PASS |
| 7 | system_prompt_block() (250 chars) | PASS |
| 8 | handle_tool_call — neural_remember | PASS |
| 9 | handle_tool_call — neural_recall | PASS |
| 10 | handle_tool_call — neural_graph | PASS |
| 11 | prefetch() | PASS |
| 12 | shutdown() | PASS |

### VM / Constrained Environment Notes

- **4 GB RAM minimum** — FastEmbed model download (~500 MB). 2 GB = OOM killed.
- **HashBackend** works as fallback on low-RAM systems (1024d, instant, no deps).
- **C++ bridge optional** — Python fallback covers all functionality.
- **FastEmbed >= 0.5.1** — earlier versions default to CLS embedding (deprecated).
- **`python3-venv` required** on Debian — `apt install python3.11-venv` if missing.
- **PEP 668 (Debian)** — `pip install` needs venv or `--break-system-packages`.
- **Cloud-init delay** — 60–90 s on first boot. Don't assume SSH is ready immediately.
- **prefetch() returns empty** on fresh DB — expected, no prior memories to pre-load.

---

## File Structure

```
mazemaker-adapter/
├── install.sh                    # Installer
├── hermes-plugin/                # Plugin (deployed to hermes-agent)
│   ├── __init__.py               # MemoryProvider + tools
│   ├── config.py                 # Config loader
│   ├── plugin.yaml               # Plugin metadata
│   ├── neural_memory.py          # Unified Memory class
│   ├── memory_client.py          # Main client (Mazemaker, SQLiteStore)
│   ├── embed_provider.py         # Embedding backends (FastEmbed, st, tfidf, hash)
│   ├── gpu_recall.py             # CUDA cosine similarity engine
│   ├── dream_engine.py           # Dream engine (NREM/REM/Insight)
│   ├── dream_worker.py           # Standalone daemon
│   ├── access_logger.py          # Recall event logger
│   └── ...
├── python/                       # Python source (mirrors hermes-plugin)
│   └── ...
├── src/                          # C++ source (optional, legacy)
│   ├── memory/lstm.cpp           # LSTM predictor
│   ├── memory/knn.cpp            # kNN engine
│   └── memory/hopfield.cpp       # Hopfield network
├── benchmarks/                   # The eight-round audit story
│   ├── README.md                 # Suite catalog + headline numbers
│   ├── audit/                    # codex-v2..v8 prompts + verdicts (verbatim)
│   └── neural_memory_benchmark/  # Suites + dataset generators
└── README.md
```

---

## Production Lessons

### Embedding & Runtime

- **FastEmbed > sentence-transformers** — ONNX runtime, no PyTorch conflict, fast on CPU.
- **FastEmbed >= 0.5.1** — earlier versions default to CLS embedding. Pin version or set `add_custom_model`.
- **GPU recall > C++ Bridge** — C++ Hopfield had bias issues; GPU matmul is clean.
- **numpy before FastEmbed** — FastEmbed imports numpy at load time; install order matters.
- **Don't force PyTorch** — let FastEmbed handle CPU. torch only needed for GPU recall.

### Storage & Architecture

- **SQLite = Source of Truth** — Postgres + pgvector is an optional mirror. SQLite always works.
- **Auto-detect everything** — CUDA, backends, venv paths. Minimize config burden.
- **4 tool schemas** exposed by NeuralMemoryProvider: `neural_remember`, `neural_recall`, `neural_think`, `neural_graph`. (`neural_dream` / `neural_dream_stats` are standalone Memory class only.)

### Benchmark-driven defaults

- **`retrieval_mode: lean` is the new recommended default** — channel_ablation at n=200 on real prose proved BM25/temporal/salience are dead-weight or actively harmful. Lean drops them. +0.18 R@5 vs skynet.
- **`recall_score_percentile` over `recall_score_floor`** — the legacy floor lives on a 0..0.05 scale and is silently broken for any reasonable user input. Percentile is calibrated [0,1] by rank.
- **`think_engine: ppr` over `bfs`** for ranking-quality runs — channel_ablation proved PPR is the biggest MRR contributor (-0.13 if removed).

---

## License

See [LICENSE](LICENSE).
