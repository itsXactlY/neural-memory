# External benchmarks

External, third-party memory benchmarks run against the Mazemaker engine.
Internal benchmarks live in `benchmarks/neural_memory_benchmark/`; the harnesses
here only call the public Mazemaker API (`memory_client.Mazemaker`).

## Verified results — at a glance

Result JSONs in [`results/`](results/) are tracked in-tree (whitelisted in this
directory's `.gitignore`) so anyone reproducing the benches can diff their
numbers against ours without a separate artifact download.

### LongMemEval-S — 500q retrieval, 470 gradeable

Same harness, same dataset (sha256 `d6f21ea9…`), same config
(`recall_mode=hybrid, k=10, granularity=session`); only ColBERT@1.5 differs.

| Metric | hybrid baseline | hybrid + ColBERT@1.5 | Δ |
|---|---|---|---|
| **R@1** | 0.8064 | **0.8574** | **+5.10 pp** |
| **R@5** | 0.9596 | **0.9787** | **+1.91 pp** |
| **R@10** | 0.9830 | **0.9894** | +0.64 pp |
| **MRR** | 0.8733 | **0.9114** | **+3.81 pp** |
| p50 latency | 41.1 ms | 56.9 ms | +15.8 ms |

Per-question-type R@5 with ColBERT@1.5 reaches **1.0000** on
`knowledge-update`, `multi-session`, `single-session-assistant`. The
biggest single-category swing is `single-session-user`: R@5 +7.8 pp,
MRR +10.4 pp.

Result files:
- [`results/longmemeval_s_master-baseline_20260509T214714Z.json`](results/longmemeval_s_master-baseline_20260509T214714Z.json) — no-CB
- [`results/longmemeval_s_colbert-on-master_20260510T034308Z.json`](results/longmemeval_s_colbert-on-master_20260510T034308Z.json) — CB@1.5

### Demolition Bench — 10 Hindsight-failed models, 20 questions

| Run | Aggregate | Errors | Notes |
|---|---|---|---|
| no-ColBERT canonical | 186/200 = 93.0% | 2 | hybrid + rerank + advanced |
| ColBERT@1.5 broken | 168/200 = 84.0% | 24 | superseded — GPU contention bug |
| **ColBERT@1.5 fixed** | **188/200 = 94.0%** | **0** | reproducibility-fix verified |

`gemma3:270m` (270M params, runs on a Raspberry Pi) scores 18/20 = 90% in
both conditions. JSON leaks: 0 / 200 across both runs. The "broken"
canonical is intentionally kept in tree as the before-state of the
reproducibility fix — anyone reading the diff or re-running the harness
can see the 24-error baseline and verify their own run drops to 0.

Result files:
- [`results/demolition_canonical-synthetic20_20260509T213339Z.json`](results/demolition_canonical-synthetic20_20260509T213339Z.json) — no-CB
- [`results/demolition_colbert-w15-canonical_20260509T223543Z.json`](results/demolition_colbert-w15-canonical_20260509T223543Z.json) — CB@1.5 *broken*
- [`results/demolition_colbert-w15-clean_20260510T010504Z.json`](results/demolition_colbert-w15-clean_20260510T010504Z.json) — CB@1.5 *fixed*

### Reproducibility fix (Demolition Bench, 2026-05-10)

The ColBERT-on canonical regressed from 186/200 to 168/200 with 24 HTTP-500
errors before the fix landed. Root cause was GPU contention, not ColBERT
logic: the in-process ColBERT helper (BGE-M3 ~1.4 GB VRAM), the bench's
torch CUDA init, and ollama's keep-alive holding the previous LLM in VRAM
combined to OOM the GPU. ollama returned 500.

Two surgical fixes at the top of `demolition_bench.py`:

1. Hide CUDA from the bench python (`CUDA_VISIBLE_DEVICES=""`,
   `MM_COLBERT_DEVICE=cpu`, `MM_FORCE_CPU=1`). Operator override:
   `MM_BENCH_ALLOW_CUDA=1`.
2. New `ollama_evict(model)` helper sending `keep_alive: 0` between
   models. No more stacked-VRAM OOM.

Operator pre-flight: stop `mazemaker-dream-worker.service` (5-min
consolidation cycle holds ~1 GB GPU and breaks reproducibility). After
the fix: 0 / 200 errors deterministic, 188/200 correct, identical
distribution to no-CB.

## LongMemEval-S

Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
Memory* (ICLR 2025). https://arxiv.org/abs/2410.10813 ·
https://github.com/xiaowu0162/LongMemEval

500 evaluation instances; each instance ships with ~50 user/assistant
chat sessions and a question whose evidence lives in 1–N specific sessions
(`answer_session_ids`). Mazemaker is judged on whether it surfaces a gold
session in its top-k recall.

### Dataset

The harness expects:

    benchmarks/external/data/longmemeval_s/longmemeval_s.json

Download the cleaned 2025-09 release directly from Hugging Face:

    mkdir -p benchmarks/external/data/longmemeval_s
    curl -L -o benchmarks/external/data/longmemeval_s/longmemeval_s.json \
      https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

The harness records the SHA-256 of the file in every result JSON so two runs
on different dataset versions can be told apart.

### Running

Default — full benchmark, BGE-M3 embeddings, hybrid recall:

    python -m benchmarks.external.longmemeval_s

Common variants:

    # skynet retrieval mode + cross-encoder rerank
    python -m benchmarks.external.longmemeval_s \
        --recall-mode skynet --rerank --tag skynet-rerank

    # quick smoke / CI mode (hash backend, 5 stratified questions)
    python -m benchmarks.external.longmemeval_s \
        --backend hash --limit 5 --stratified --tag smoke

    # turn-level granularity (each chat turn is one memory)
    python -m benchmarks.external.longmemeval_s --granularity turn

CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--recall-mode` | `hybrid` | `semantic`, `hybrid`, `advanced`, `skynet`, `lean`, `trim` |
| `--rerank` | off | Enable cross-encoder reranker on the head |
| `--backend` | `auto` | Embedding backend; `auto` selects BGE-M3 via shared server / GPU / CPU |
| `--granularity` | `session` | `session` (one memory per chat session) or `turn` |
| `-k`, `--k` | 10 | Top-k cutoff for recall metrics |
| `--limit N` | 0 | Only run the first N questions (or stratified-N with `--stratified`) |
| `--stratified` | off | When `--limit` < 500, sample proportionally by `question_type` |
| `--tag` | `""` | Adds a tag to the output filename |

### Output

Each run writes to `benchmarks/external/results/longmemeval_s_<tag>_<UTC-ts>.json`:

```json
{
  "timestamp": "20260509T194109Z",
  "git_sha": "<engine HEAD>",
  "git_dirty": false,
  "system_config": {
    "recall_mode": "hybrid",
    "rerank": false,
    "embedding_backend": "auto",
    "granularity": "session",
    "k": 10,
    "limit": 0,
    "use_hnsw": false,
    "use_cpp": false
  },
  "dataset": {
    "name": "longmemeval_s",
    "path": "...",
    "sha256": "..."
  },
  "metrics": {
    "n_total": 500,
    "n_gradeable": 470,
    "recall@1": 0.xx,
    "recall@5": 0.xx,
    "recall@10": 0.xx,
    "MRR": 0.xx,
    "p50_ms": ...,
    "p95_ms": ...
  },
  "metrics_by_question_type": { ... },
  "per_question": [{"qid": ..., "rank_of_gold": ..., "latency_ms": ...}, ...]
}
```

`n_gradeable` excludes the 30 abstention questions (qids ending in `_abs`),
which have empty `answer_session_ids` and so cannot contribute to recall.

### Methodology

For each question:

1. A fresh in-memory Mazemaker is created in `tempfile.mkdtemp()` (no
   cross-question state, no shared FAISS / HNSW state).
2. Each `haystack_session` is rendered to a single text blob and ingested via
   `nm.remember(text, label="session:<sid>", auto_connect=False,
   detect_conflicts=False)`. `detect_conflicts=False` keeps ingestion linear
   and avoids merging unrelated sessions that happen to share a label prefix;
   `auto_connect=False` keeps ingestion latency out of the way of the recall
   measurement. Mazemaker's graph still hydrates lazily on the first recall.
3. `nm.recall(question, k=10, hybrid=...)` is timed with
   `time.perf_counter()`. Only the recall call counts toward `p50_ms` /
   `p95_ms`; ingest is reported separately as `mean_ingest_ms`.
4. The first surfaced result whose label parses to a session id in
   `answer_session_ids` is the "gold hit"; its 1-indexed position is
   `rank_of_gold` (None means no gold in top-k).
5. `recall@N` = fraction of gradeable questions with `rank_of_gold ≤ N`.
   `MRR` = mean of `1/rank_of_gold` (or 0 if no gold in top-k) across
   gradeable questions.

### Reproducibility

Every result file embeds:

- `git_sha` — the Mazemaker repo HEAD.
- `git_dirty` — true if there were uncommitted changes when the run started.
- `dataset.sha256` — full hash of the LongMemEval-S JSON file.
- `system_config` — every flag that affects engine behaviour in the harness.

Two runs on the same `git_sha` + same `dataset.sha256` + same `system_config`
should reproduce the same metrics modulo embedding-server non-determinism
(BGE-M3 batching is deterministic; hybrid retrieval has no random tie
breakers in the engine).

### Comparing runs

`compare_runs.py` produces a markdown table from any two result files:

    python -m benchmarks.external.compare_runs \
        results/longmemeval_s_master-baseline_<ts>.json \
        results/longmemeval_s_pr5-merged_<ts>.json

It shows per-metric deltas, per-question-type recall@5 deltas, and which
specific questions had their gold rank promoted/demoted.

### Notes

- `recall_mode=skynet` is the canonical "all channels active" Mazemaker
  configuration. `lean` and `trim` zero certain channels per the
  2026-04-28 codex benchmark; `hybrid` is the safe parallel-channel default.
- The internal `benchmarks/lme_eval.py` harness is *similar* but pre-dates
  the official LongMemEval dataset and uses a synthetic smoke corpus by
  default. This `external/longmemeval_s.py` harness exists specifically to
  produce numbers that are directly comparable to the published
  LongMemEval-S leaderboard.

## Demolition Bench

`demolition_bench.py` — head-to-head harness against the 10 small/medium
open-source models that Hindsight's public benchmark page lists as scoring
0/N. Hindsight's pipeline gates each answer on a strict JSON schema; if
the model can't emit that exact shape, the answer is thrown out. The
demolition harness asks every model in plain English for a one-sentence
answer, then substring-matches the gold answer.

The 10 Hindsight-fail models are the default `--models`:

```
gemma3:1b   gemma3:12b   gemma3:270m
qwen2.5:0.5b   qwen2.5:3b
smollm2:1.7b   deepseek-r1:1.5b
granite3.1-dense:2b   llama3.2:latest   ministral-3:3b
```

### Pre-reqs

- `ollama` running on `http://localhost:11434` (or override via
  `OLLAMA_URL`). Auto-pulls missing models via `ollama pull` unless
  `--no-auto-pull` is set.
- The Mazemaker engine import works the same way as the LongMemEval
  harness — picks up whatever's currently in `python/`.

### Running

```bash
# Smoke (1 model, 3 synthetic questions, hash backend — fast and offline-safe)
python -u -m benchmarks.external.demolition_bench \
    --models gemma3:270m --n 3 --dataset synthetic \
    --backend hash --no-rerank --tag smoke

# Default config: full 10-model grid, BGE-M3 + cross-encoder rerank,
# `advanced` retrieval mode, 20 stratified questions
python -u -m benchmarks.external.demolition_bench --n 20

# Custom subset
python -u -m benchmarks.external.demolition_bench \
    --models qwen2.5:3b llama3.2:latest --n 50

# ColBERT-on second pass to compare
python -u -m benchmarks.external.demolition_bench \
    --n 20 --enable-colbert --tag colbert-on
```

### Datasets

- `--dataset longmemeval_s` (default) — uses the cached
  `data/longmemeval_s/longmemeval_s.json` and stratifies a subset by
  `question_type`. Gold substring = the dataset's `answer` field.
- `--dataset synthetic` — 20 hand-built two-session questions across
  factual recall, multi-session, temporal, entity-tracking, and
  multi-step categories. Useful when LongMemEval isn't available, when
  you want a fast deterministic check, or when running on a machine
  that can't do BGE-M3.

### Output

Each run writes `demolition_<tag>_<UTC-ts>.json` AND
`demolition_<tag>_<UTC-ts>.md` under `results/`. The markdown is a
ready-to-quote table with the Hindsight `0/N` claim next to the
Mazemaker accuracy.

### Methodology

For every (model, question) pair:

1. Spin up an isolated Mazemaker on a temp DB.
2. Ingest each haystack session as one memory (`auto_connect=False`,
   `detect_conflicts=False`).
3. `nm.recall(question, k=10, hybrid=True, rerank=True)`.
4. Build a plain-English prompt:

   ```
   Use the following memories from a chat history to answer the question.

   [memory 1]
   ...

   Question: <question>
   Answer in one sentence.
   ```

5. Send via `POST /api/chat` with a system prompt that explicitly says
   *no JSON, no markdown*. `temperature=0` for repeatability.
6. Score: `gold_substring.lower() in answer.lower()`. Also tracks
   `json_leaks` (answers starting with `{` or `[` despite the prompt)
   so we can spot models that hallucinate a schema we never asked for.

The harness logs `wall_seconds`, `avg_model_latency_ms`,
`avg_recall_latency_ms`, and per-question rows so we can audit any
suspicious score after the fact.
