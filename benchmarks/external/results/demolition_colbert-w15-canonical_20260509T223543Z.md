# Demolition Bench — 20260509T223543Z

**Headline:** Mazemaker measures memory. Hindsight measures whether the LLM can output JSON.

Engine `git_sha`: `443011cba04b7dcd66766a590adbe0b9894b96e7` (dirty)
Dataset: `synthetic_v1` (n=20, hash=b1e2de77fabe79a9)
Config: backend=`auto`, recall_mode=`advanced`, rerank=True, enable_colbert=True, k_retrieval=10, judge=`substring_match`

| Model | Hindsight Result | Mazemaker Accuracy | Avg model latency | Avg recall latency | JSON leaks | Errors |
|---|---|---|---|---|---|---|
| gemma3:1b | 0/N (couldn't follow JSON schema) | 19/20 = 95.0% | 0.55s | 0.31s | 0 | 0 |
| gemma3:12b | 0/N (couldn't follow JSON schema) | 20/20 = 100.0% | 3.25s | 0.15s | 0 | 0 |
| gemma3:270m | 0/N (couldn't follow JSON schema) | 18/20 = 90.0% | 0.34s | 0.21s | 0 | 0 |
| qwen2.5:0.5b | 0/N (couldn't follow JSON schema) | 18/20 = 90.0% | 0.32s | 0.31s | 0 | 0 |
| qwen2.5:3b | 0/N (couldn't follow JSON schema) | 19/20 = 95.0% | 0.87s | 0.13s | 0 | 0 |
| smollm2:1.7b | 0/N (couldn't follow JSON schema) | 14/20 = 70.0% | 0.35s | 0.14s | 0 | 6 |
| deepseek-r1:1.5b | 0/N (couldn't follow JSON schema) | 1/20 = 5.0% | 7.74s | 0.13s | 0 | 18 |
| granite3.1-dense:2b | 0/N (couldn't follow JSON schema) | 20/20 = 100.0% | 0.36s | 0.13s | 0 | 0 |
| llama3.2:latest | 0/N (couldn't follow JSON schema) | 19/20 = 95.0% | 0.72s | 0.18s | 0 | 0 |
| ministral-3:3b | 0/N (couldn't follow JSON schema) | 20/20 = 100.0% | 0.95s | 0.13s | 0 | 0 |

**Methodology:** for every question we spin up an isolated Mazemaker, ingest the haystack sessions, retrieve top-10 memories, build a plain-English prompt from the top 5, and ask the model in natural language for a one-sentence answer. Scoring = the dataset's gold substring appears in the model's response (case-insensitive).

`JSON leaks` counts answers that began with `{` or `[` despite the plain-text prompt — indicating the model insisted on a structured shape we never asked for. These were still scored on substring match.
