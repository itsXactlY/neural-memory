# Demolition Bench — 20260510T010504Z

**Headline:** Mazemaker measures memory. Hindsight measures whether the LLM can output JSON.

Engine `git_sha`: `443011cba04b7dcd66766a590adbe0b9894b96e7` (dirty)
Dataset: `synthetic_v1` (n=20, hash=b1e2de77fabe79a9)
Config: backend=`auto`, recall_mode=`advanced`, rerank=True, enable_colbert=True, k_retrieval=10, judge=`substring_match`

| Model | Hindsight Result | Mazemaker Accuracy | Avg model latency | Avg recall latency | JSON leaks | Errors |
|---|---|---|---|---|---|---|
| gemma3:1b | 0/N (couldn't follow JSON schema) | 19/20 = 95.0% | 0.53s | 0.45s | 0 | 0 |
| gemma3:12b | 0/N (couldn't follow JSON schema) | 20/20 = 100.0% | 1.46s | 0.38s | 0 | 0 |
| gemma3:270m | 0/N (couldn't follow JSON schema) | 18/20 = 90.0% | 0.29s | 0.40s | 0 | 0 |
| qwen2.5:0.5b | 0/N (couldn't follow JSON schema) | 18/20 = 90.0% | 0.28s | 0.37s | 0 | 0 |
| qwen2.5:3b | 0/N (couldn't follow JSON schema) | 19/20 = 95.0% | 0.33s | 0.38s | 0 | 0 |
| smollm2:1.7b | 0/N (couldn't follow JSON schema) | 20/20 = 100.0% | 0.39s | 0.38s | 0 | 0 |
| deepseek-r1:1.5b | 0/N (couldn't follow JSON schema) | 15/20 = 75.0% | 1.52s | 0.37s | 0 | 0 |
| granite3.1-dense:2b | 0/N (couldn't follow JSON schema) | 20/20 = 100.0% | 0.44s | 0.36s | 0 | 0 |
| llama3.2:latest | 0/N (couldn't follow JSON schema) | 19/20 = 95.0% | 0.58s | 0.41s | 0 | 0 |
| ministral-3:3b | 0/N (couldn't follow JSON schema) | 20/20 = 100.0% | 0.63s | 0.36s | 0 | 0 |

**Methodology:** for every question we spin up an isolated Mazemaker, ingest the haystack sessions, retrieve top-10 memories, build a plain-English prompt from the top 5, and ask the model in natural language for a one-sentence answer. Scoring = the dataset's gold substring appears in the model's response (case-insensitive).

`JSON leaks` counts answers that began with `{` or `[` despite the plain-text prompt — indicating the model insisted on a structured shape we never asked for. These were still scored on substring match.
