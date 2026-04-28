# GPT-5.3 final verdict on the *executed* benchmark

You audited this benchmark three times (v2 source, v3 source, v4 source). Your v4 verdict was **qualified-y**: the source-level fixes were sound but you could only accept it if the actual run showed graph_lift + shuffle_collapse, strict post-dream lift with pre-dream zero, and clean ablation/HNSW controls.

The benchmark just ran end-to-end on `--paraphrase --seed 42`. Real numbers below. Read the saved JSON if you want the full detail (paths at the bottom). Decide whether the qualified-y now upgrades to **yes**, stays at **qualified-y** (with new caveats), or downgrades.

## Live results from this run

### graph_reasoning (the headline test)
```
raw_cosine    : R@10=0.0000   MRR=0.0000
nm_semantic   : R@10=0.1333   MRR=0.0191
nm_skynet     : R@10=0.9333   MRR=0.2863
nm_multihop   : R@10=1.0000   MRR=0.2867   ← perfect
nm_think      : R@10=0.2333   MRR=0.2333
multihop[ctrl]: R@10=0.2667   MRR=0.0258   ← shuffled-edge control
chain edges   : 60 (expected 60, exactly 2 per chain, no A↔C shortcut)
```
graph_lift_vs_raw (multihop) = +1.0000.
shuffle_collapse (real - shuffled) = 1.0000 - 0.2667 = +0.7333.

### dream_derived_fact
```
pre-dream:  derived_fact_hit_rate_multihop = 0.00  (forbidden by construction; no derived:* exists yet)
            single_doc_both_tokens_rate    = 0.00  (forbidden by template — no premise carries both tokens)
            connections=1050  derived=0
post-dream: derived_fact_hit_rate_multihop = 0.32
            derived_fact_hit_rate_semantic = 0.04
            single_doc_both_tokens_rate    = 0.00  (Insight didn't synthesise both into one doc)
            connections=1301  derived=12  dream_elapsed=0.43s
lift.derived_fact_hit_rate_multihop = +0.32   ← unambiguous, was structurally 0 pre-dream
lift.semantic_both_tokens_legacy    = -0.28   ← legacy metric flagged inflated, do not interpret
```

### channel_ablation
```
all channels resolved defaults (read from live NeuralMemory):
  {semantic:1.0, bm25:0.9, entity:1.0, temporal:0.35, ppr:0.55, salience:0.25}
all channels: R@5=0.90  MRR=0.8483
no_semantic : R@5=0.88  MRR=0.8067   Δrecall=-0.0200  Δmrr=-0.0416
no_bm25     : R@5=0.90  MRR=0.8483   Δrecall= 0.0000  Δmrr= 0.0000
no_entity   : R@5=0.86  MRR=0.8267   Δrecall=-0.0400  Δmrr=-0.0216
no_temporal : R@5=0.90  MRR=0.8457   Δrecall= 0.0000  Δmrr=-0.0026
no_ppr      : R@5=0.86  MRR=0.7267   Δrecall=-0.0400  Δmrr=-0.1216  ← largest MRR contributor
no_salience : R@5=0.90  MRR=0.8547   Δrecall= 0.0000  Δmrr=+0.0064  ← null-or-slightly-harmful
```

### continuity_controls (concept-mode, your v3 fix request)
Each tier injects 2 near-distractors per target with concept vocabulary lifted from the query but a fresh anchor.
```
tier_0_noise_0_dist_0:    nm=0.66  raw=0.46  recency=0.10
tier_1_noise_200_dist_100:nm=0.62  raw=0.06  recency=0.00
tier_2_noise_1200_dist_200:nm=0.58  raw=0.06  recency=0.00
tier_3_noise_6200_dist_300:nm=0.20  raw=0.06  recency=0.00
```
Raw cosine drops 0.46 → 0.06 (the design forces this). nm beats raw by ≥0.14 across every tier.

### conflict_quality (supersession lift)
```
with_supersession:        winner@1=0.3333  loser>winner=0.1333
control_no_supersession:  winner@1=0.0333  loser>winner=0.6000
supersession_lift:        winner_rank_1=+0.3000  loser_drop=+0.4667
```
Without supersession the loser dominates (0.60); with it, the winner climbs to top-1 in 33% of cases.

### baseline (raw cosine vs nm)
```
raw     : R@5=1.0000  MRR=1.0000  p50= 1.5ms
semantic: R@5=0.4200  MRR=0.2213  p50=32.4ms
skynet  : R@5=0.9000  MRR=0.7667  p50=339.9ms   ← 200x latency, recovers most recall
```
Note: this is the *anchor-based* paraphrase task (queries share the unique anchor with target). Raw cosine wins because the anchor itself is an orthogonal direction in embedding space. The continuity_controls test (above) is the concept-mode follow-up where this advantage is removed.

### lstm_knn ablation
```
delta: recall=+0.040  MRR=+0.076  p50_overhead=+38.8ms  p95_overhead=+36.9ms
```
Small recall lift, real latency cost.

### diversity
MMR sweep produces a real recall/entropy tradeoff (R@5 0.42→0.32 as λ goes 0→0.7, entropy 2.08→3.35).
score_floor calibration bug confirmed: floor≥0.2 nukes results because relevance is RRF-scaled ~0.05.

## What you said in v4

> qualified-y. I would accept this as evidence if the actual reported results show graph lift plus shuffle collapse, strict post-dream lift with pre-dream zero, and HNSW/ablation controls behaving as claimed.

## Your task — final verdict

Read the JSON files if you need detail beyond the summaries above:
- `benchmarks/results/run-2026-04-28-codex-judge/results/graph_reasoning_results.json`
- `benchmarks/results/run-2026-04-28-codex-judge/results/dream_derived_fact_results.json`
- `benchmarks/results/run-2026-04-28-codex-judge/results/continuity_controls_results.json`
- `benchmarks/results/run-2026-04-28-codex-judge/results/channel_ablation_results.json`
- `benchmarks/results/run-2026-04-28-codex-judge/results/conflict_quality_results.json`

You don't need to run anything; the JSON has the actual top-k id lists, etc. if you want to spot-check.

Output a final report under 400 words with sections:

`## Did the run satisfy the v4 criteria` — y/n on each of: graph_lift, shuffle_collapse, strict_post_dream_lift, channel_ablation_clean. Cite the specific number.

`## Anything inconsistent` — does any number contradict another? E.g. `single_doc_both_tokens_rate=0.0` post-dream while `derived_fact_hit_rate_multihop=0.32` — is that surprising? What does that imply about how the dream engine surfaces derived facts?

`## Final verdict` — **yes / qualified-y / no**, one paragraph. The bar: would a peer reviewer reading this benchmark accept it as proof neural-memory-adapter does something a vanilla vector store cannot?
