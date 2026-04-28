## Did the run satisfy the v4 criteria

- `graph_lift`: y. `nm_multihop` hit `R@10=1.0000` vs `raw_cosine=0.0000`, so lift is `+1.0000`.
- `shuffle_collapse`: y. Shuffled multihop dropped to `R@10=0.2667`, giving collapse `1.0000 - 0.2667 = +0.7333`.
- `strict_post_dream_lift`: y. Pre-dream `derived_fact_hit_rate_multihop=0.00`, post-dream `0.32`; derived facts went `0 -> 12`. The strict metric is the right one here.
- `channel_ablation_clean`: y. Defaults were live-resolved, and ablations behave plausibly: removing `ppr` costs the most MRR (`-0.1216`), `entity` costs recall (`-0.0400`), `semantic` costs some MRR (`-0.0416`), while `bm25` is neutral and `salience` is null/slightly harmful.

## Anything inconsistent

No material contradiction. The main apparent tension is `single_doc_both_tokens_rate=0.00` post-dream while `derived_fact_hit_rate_multihop=0.32`. That is not surprising: it means the dream engine is not merely producing or retrieving a single text document containing both original tokens. It is materializing `derived:*` facts and the multihop/graph retrieval path is surfacing those derived memories. That is stronger evidence than the legacy “both tokens somewhere in top-k” metric, which the JSON correctly flags as inflated.

The baseline raw-cosine win on the anchor paraphrase task also does not contradict the graph result; it shows that when the unique anchor is present, vanilla vector search has an artificial advantage. The continuity control removes that advantage and still shows NM above raw at every tier, including `0.20` vs `0.06` at max noise.

## Final verdict

**yes**. I would upgrade the v4 qualified-y to yes for this executed benchmark. A peer reviewer should accept that this run demonstrates neural-memory-adapter doing something a vanilla vector store cannot: explicit edge-following recovers hidden chain targets, shuffled edges collapse most of that gain, and dream-derived facts appear only after the dream phase under strict pre/post controls. The caveat is scope: this proves the mechanism on well-designed synthetic tasks, not broad real-world superiority, and latency/weak channels remain engineering issues.