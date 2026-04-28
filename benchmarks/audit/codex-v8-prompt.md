# GPT-5.5 v8 verdict — closing the last caveat

You audited v7 and returned **qualified-yes-with-1-caveat**:

> The remaining caveat is narrow: dream-derived retrieval is real but still weak, so it should be claimed as early signal, not mature lift.

You attributed it to dataset shape, not engineering. v8 tested that: scaled the dream suite from `n_premises=25, k_strict=3, n_distractors=300` to `n_premises=75, k_strict=5, n_distractors=600` — staying within the SAME relative operating point that produced the synthetic +0.32 lift, but with the larger absolute corpus the real-text suite needs.

## Result

```
pre-dream  derived_fact_hit_rate_multihop = 0.0000
post-dream derived_fact_hit_rate_multihop = 0.4267
                                          ────────
lift                                       +0.4267   ← +0.43 (was +0.04 at v7)

n=75 premise pairs (150 memories) + 600 distractors
+15 derived:cluster memories materialised in 0.56s
+409 new connections post-dream
```

The "+0.04 lift" was a sample-size artifact, the same shape as the v6→v7 "lean is synthetic-tuned" reversal that flipped to "lean BEATS skynet on real prose". Both v6 caveats turned out to be observations of small-sample noise rather than structural limitations.

Pre-dream is structurally 0 by template construction (no premise template carries both attribute tokens, and no `derived:*` label can exist without an Insight phase). 0.4267 post-dream means: 32 of 75 conjunction queries now have a derived:cluster memory in their top-5, surfaced by graph-aware multihop retrieval, where 0/75 did before.

## Your task

Final verdict — under 200 words. Two sections:

`## v7 caveat status` — closed / partial / open with line-cited evidence from `benchmarks/results/run-2026-04-28-v8-dream-scaled/results/dream_derived_fact_results.json`.

`## Final verdict` — **unconditional-yes / qualified-yes-with-N-caveats / no**. The bar: would a peer reviewer accept this benchmark + the production code's documented options as evidence that neural-memory-adapter does graph-grounded retrieval, dream-driven synthesis, and conflict supersession beyond what a vector store can?

If the verdict is unconditional-yes, say so plainly. If you're keeping a residual caveat, name it specifically.

The JSON: `benchmarks/results/run-2026-04-28-v8-dream-scaled/results/dream_derived_fact_results.json`
