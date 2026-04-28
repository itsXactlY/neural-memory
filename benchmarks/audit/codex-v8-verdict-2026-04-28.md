## v7 caveat status

Closed. The strict pre-dream baseline is 0 derived facts and 0.0000 multihop derived hits ([JSON lines 4, 12](/home/alca/projects/neural-memory-adapter/benchmarks/results/run-2026-04-28-v8-dream-scaled/results/dream_derived_fact_results.json:4)). Post-dream materialises 15 derived facts in 0.56s and raises `derived_fact_hit_rate_multihop` to 0.4267 ([lines 20-21, 29](/home/alca/projects/neural-memory-adapter/benchmarks/results/run-2026-04-28-v8-dream-scaled/results/dream_derived_fact_results.json:20)). The recorded lift is +0.4267 with 409 new connections ([lines 41-42](/home/alca/projects/neural-memory-adapter/benchmarks/results/run-2026-04-28-v8-dream-scaled/results/dream_derived_fact_results.json:41)), and the JSON explicitly frames this as the strict unambiguous Insight-phase signal ([line 44](/home/alca/projects/neural-memory-adapter/benchmarks/results/run-2026-04-28-v8-dream-scaled/results/dream_derived_fact_results.json:44)).

## Final verdict

**unconditional-yes**.

A peer reviewer should accept this benchmark plus the documented production options as evidence that `neural-memory-adapter` does graph-grounded retrieval, dream-driven synthesis, and conflict supersession beyond what a vector store can. No residual caveat.