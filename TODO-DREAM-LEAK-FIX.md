# Neural Memory Dream Engine — Leak Fix & C++26 Rewrite Master Plan

## Status: AUDIT COMPLETE — IMPLEMENTATION IN PROGRESS

---

## PART 0: FORENSIC SUMMARY (Root Causes Identified)

### Critical Memory Leaks — WHERE + WHY + HOW

| # | Location | File:Line | Mechanism | Severity |
|---|----------|-----------|-----------|----------|
| L1 | AccessLogger._buffer | `access_logger.py:88-93` | List replacement creates new object on every 1000-item cycle; unbounded if flush lags | CRITICAL |
| L2 | AccessLogger JSONL | `access_logger.py:138` | `access_log.jsonl` grows forever — no rotation, no cap | CRITICAL |
| L3 | connection_history | `dream_engine.py:786-787` | Prune only every 50 cycles; MSSQL path (`CppDreamBackend`) returns 0 — skips entirely | CRITICAL |
| L4 | dream_sessions/insights | `dream_engine.py:867` | 3 sessions per cycle; prune only every 50 cycles; insights accumulate linearly | HIGH |
| L5 | KnowledgeGraph spreading_activation | `knowledge_graph.cpp:193-240` | Per-node `vector<uint64_t>` path allocation; unbounded maps | HIGH |
| L6 | KnowledgeGraph edge duplication | `knowledge_graph.cpp:93-99` | Each edge stored twice (A→B + B→A); 2x memory waste | MEDIUM |
| L7 | SharedEmbedServer torch state | `embed_provider.py:246-293` | Repeated GPU reload retries fragment VRAM; tokenization state accumulates | MEDIUM |
| L8 | DreamWorker embedding cache | `dream_worker.py:175-190` | FIFO eviction only runs AFTER adding — can exceed 2048 during batch REM | MEDIUM |
| L9 | networkx import per call | `dream_engine.py:954` | Full module import on every `_phase_insights()` call | LOW |
| L10 | SIMD unaligned loads | `simd.h:37-38,66-67` | `_mm256_loadu_ps` vs aligned — 10-20% perf loss; no AVX-512 path | MEDIUM |
| L11 | EpisodicMemory evict index rebuild | `memory_manager.cpp:115-118` | O(n) full-map rebuild on every remove() | MEDIUM |
| L12 | SemanticMemory centroid lookups | `memory_manager.cpp:237-240` | Repeated hash lookups per cluster member in hot path | LOW |

---

## PART I: PYTHON LEAK FIXES (Tracks 1-2)

### Track 1 — P0 Critical Leak Fixes

#### T1.1: AccessLogger._buffer — Fix Unbounded List Growth
**File:** `python/access_logger.py`
**Problem:** `self._buffer = self._buffer[-self._max_buffer:]` creates a new list object while holding `_file_lock`, causing memory pressure and potential deadlock under burst load.
**Fix:** Replace circular buffer with `collections.deque(maxlen=1000)` — O(1) append and eviction, no allocation.
**PREREQUISITE:** None
**VERIFICATION:** `python tests/test_upside_down.py` — test_access_logger

#### T1.2: AccessLogger JSONL — Add Log Rotation
**File:** `python/access_logger.py`
**Problem:** `access_log.jsonl` grows forever. No cleanup, no size cap, no rotation.
**Fix:** 
- Rotate when file exceeds 100MB
- Keep max 5 rotated files
- Add `time-based` rotation option (daily)
- Implement `clean_old_logs()` that deletes files older than 30 days
**PREREQUISITE:** T1.1
**VERIFICATION:** Write test that generates 200MB of logs, verify rotation, verify old files deleted

#### T1.3: Connection History — Prune After Every NREM
**File:** `python/dream_engine.py`
**Problem:** `prune_connection_history(keep_days=7)` only runs `if self._dream_count % 50 == 0`. NREM fires `log_connection_change()` for every strengthened edge. With 100 memories × 50 connections = 5000 history rows inserted before one prune.
**Fix:** Prune connection history after EVERY NREM phase, not every 50 cycles. Change `% 50` check to always run.
**PREREQUISITE:** None
**VERIFICATION:** Insert 10000 history rows, run one NREM cycle, verify rows < threshold

#### T1.4: CppDreamBackend — Implement prune_connection_history
**File:** `python/cpp_dream_backend.py` line 362-364
**Problem:** MSSQL path literally returns `0` with docstring "Skip — C++/MSSQL handles history internally" — but it doesn't.
**Fix:** Implement actual MSSQL DELETE from `connection_history` where `changed_at < cutoff`.
**PREREQUISITE:** None
**VERIFICATION:** Direct MSSQL query before/after

---

### Track 2 — P1-P2 Performance Fixes

#### T2.1: DreamWorker Embedding Cache — True FIFO via OrderedDict
**File:** `python/dream_worker.py` line 153-190
**Problem:** FIFO eviction only triggers on next `get_embedding()` call. During REM batch, `iso_embeds` and `comp_embeds` fill up before eviction runs.
**Fix:** Use `collections.OrderedDict` with `move_to_end()`. Evict oldest BEFORE adding when at limit, not after.
**PREREQUISITE:** T1.1 (deque already imported)
**VERIFICATION:** `DREAM_EMBED_CACHE_MAX=128` stress test with REM phase

#### T2.2: SIMD — AVX-512 Path + Aligned Loads
**File:** `include/neural/simd.h`, `src/simd/simd_engine.cpp`
**Problem:** `_mm256_loadu_ps` unaligned — 10-20% slower. No AVX-512 detection.
**Fix:**
- Add `SIMD_HAS_AVX512` compile-time check
- Add `std::assume_aligned` hints for hot path vectors
- Add `batch_cosine_avx512` for the 16-elements-per-cycle path
- Add `batch_cosine_similarity_openmp` with proper `schedule(static)` chunking
**PREREQUISITE:** None
**VERIFICATION:** `cd build && ctest -R simd` + benchmark before/after

#### T2.3: networkx Import — Top-Level or Lazy
**File:** `python/dream_engine.py` line 954
**Problem:** `import networkx as nx` inside `_detect_communities()` runs on every Insight phase call.
**Fix:** Move import to top of file. If networkx unavailable, fall back to BFS (already implemented).
**PREREQUISITE:** None
**VERIFICATION:** `python -c "import sys; sys.setprofile(lambda *a: None)"` — measure import overhead

#### T2.4: KnowledgeGraph — CSR Adjacency + Sorted Edges
**File:** `src/graph/knowledge_graph.cpp`
**Problem:** `unordered_map<uint64, vector<Edge>>` — O(1) average but with high constant. Edge duplication doubles memory. Spreading activation allocates per-node path vectors.
**Fix:**
- Replace with CSR (Compressed Sparse Row) representation
- Store undirected edges once (not twice)
- Add `compute_spreading_activation_csr()` that uses CSR
- Thread-local path vector pool to eliminate allocation storm
**PREREQUISITE:** T2.2 (SIMD batch needed for CSR batch cosine)
**VERIFICATION:** Memory usage before/after with 100K edges

#### T2.5: EpisodicMemory — Index Rebuild Optimization
**File:** `src/memory/memory_manager.cpp` line 115-118
**Problem:** `evict_oldest_internal()` clears and rebuilds entire `id_to_index_` map on every eviction.
**Fix:** On remove(), splice out the entry and only update affected indices (entries after the removed one shift by 1). Use `entries_.erase(begin() + idx)` which is O(1) for deque.
**PREREQUISITE:** None
**VERIFICATION:** Benchmark episodic memory eviction 10000 times

---

## PART II: C++26 DREAM ENGINE ARCHITECTURE (Track 3)

### Track 3 — C++26 Full Rewrite Specification

#### T3.1: CSR Graph Specification
**Output:** `docs/cpp26/CSR-GRAPH-SPEC.md`
**Content:**
- Data structure design (row_offsets, col_indices, edge_weights)
- CSR construction from SQLite/MSSQL edges
- Bimodal search: dense scan for hot nodes, CSR skip for cold
- Lock-free atomic weight update via compare-and-swap
- AVX-512 batch cosine similarity on CSR node vectors
- Memory estimate: 100K edges → ~2.4MB vs ~25MB for adjacency list

#### T3.2: Slab Allocator Specification
**Output:** `docs/cpp26/SLAB-ALLOCATOR-SPEC.md`
**Content:**
- `SlabAllocator<T, SLAB_SIZE=4096>` template
- Zero-fragmentation design
- O(1) alloc/dealloc
- Per-phase slab pools (NREM, REM, Insight each get own slab)
- Memory budget enforcement with mmap spill

#### T3.3: Lock-Free Pipeline Specification
**Output:** `docs/cpp26/PIPELINE-SPEC.md`
**Content:**
- Lock-free MPSC ring buffer (single producer, multi consumer)
- C++26 coroutine-based phase scheduling
- Phase result channels with move semantics
- Thread pool work-stealing algorithm
- Memory budget per phase (oversize → mmap spill)
- Signal-safe shutdown

#### T3.4: ASM64 SIMD Specification
**Output:** `docs/cpp26/ASM64-SPEC.md`
**Content:**
- `batch_cosine_avx512` — 16 floats per cycle (AVX-512)
- `hebbian_update_avx512` — FMA chain weight update
- `spreading_activation_avx512` — priority queue + SIMD cosine
- Register allocation for Zen4 / Skylake-X
- Fallback chain: AVX-512 → AVX2 → SSE4 → scalar

#### T3.5: Per-Layer Maze Architecture
**Output:** `docs/cpp26/PER-LAYER-MAZE-SPEC.md`
**Content:**
- Each layer (NREM/REM/Insight) gets own `DreamLayer` class
- Per-layer memory slab allocator (no cross-phase fragmentation)
- Per-layer CSR subgraph (not full graph)
- Per-layer thread pool (configurable concurrency)
- Phase communication via lock-free ring buffers
- Layer-specific algorithms as pluggable strategy objects

---

## PART III: C++ IMPLEMENTATION (Track 4)

### Track 4 — C++ Implementation (After Tracks 1-3 Complete)

#### T4.1: CSR Graph Implementation
**Files:** `src/graph/csr_graph.cpp`, `include/neural/csr_graph.h`
**Implements:** T3.1 specification

#### T4.2: Slab Allocator Implementation
**Files:** `src/memory/slab_allocator.cpp`, `include/neural/slab_allocator.h`
**Implements:** T3.2 specification

#### T4.3: Lock-Free Pipeline Implementation
**Files:** `src/pipeline/dream_pipeline.cpp`, `include/neural/dream_pipeline.h`
**Implements:** T3.3 specification

#### T4.4: ASM64 SIMD Implementation
**Files:** `src/simd/asm64_batch_cosine.S`, `src/simd/asm64_hebbian.S`
**Implements:** T3.4 specification

#### T4.5: Per-Layer Maze Integration
**Files:** `src/dream/dream_layer.cpp`, `include/neural/dream_layer.h`, `src/dream/dream_scheduler.cpp`
**Implements:** T3.5 specification

---

## COORDINATION PROTOCOL

### Crew Communication
All subagents write progress to `CREW-PROGRESS.md` in their working directory.
Subagents check `CREW-PROGRESS.md` of dependencies before starting.
Coordinate via file-based signaling:

```
Signal: <subagent>-STARTED:<task>
Signal: <subagent>-BLOCKED:<reason>
Signal: <subagent>-COMPLETE:<files_changed>
Signal: <subagent>-FAILED:<reason>
```

### Subagent Assignments

| Subagent | Tasks | Depends On |
|----------|-------|------------|
| `crew-leakfix-1` | T1.1, T1.2, T1.3 | — |
| `crew-leakfix-2` | T1.4, T2.1, T2.3 | — |
| `crew-perf-1` | T2.2 (SIMD) | — |
| `crew-arch-docs` | T3.1, T3.2, T3.3, T3.4, T3.5 | — |
| `crew-cpp-impl` | T4.1, T4.2, T4.3, T4.4, T4.5 | T3.1-T3.5 |

### Testing Strategy

| Test | Target | Run |
|------|--------|-----|
| `test_access_logger.py` | T1.1, T1.2 | `python tests/test_upside_down.py::test_access_logger*` |
| `test_dream_leak_fixes.py` | T1.3, T1.4, T2.1 | `python tests/test_dream_leak_fixes.py` (new) |
| `test_csr_graph.py` | T4.1 | `cd build && ctest -R csr` (new) |
| `test_slab_alloc.py` | T4.2 | `cd build && ctest -R slab` (new) |
| `test_asm64_simd.py` | T4.4 | `cd build && ctest -R asm64` (new) |
| `test_dream_pipeline.py` | T4.3 | `cd build && ctest -R pipeline` (new) |

---

## SUCCESS CRITERIA

- [ ] Zero unbounded growth in AccessLogger (verified 10M+ recall events)
- [ ] connection_history stays below 50K rows under sustained NREM load
- [ ] dream_sessions/dream_insights pruned to <1000 rows via 30-day retention
- [ ] C++ CSR graph: 100K edges in <5MB RSS (vs ~25MB current)
- [ ] AVX-512 SIMD: batch cosine 16x throughput vs scalar
- [ ] Lock-free pipeline: <1ms latency per phase handoff
- [ ] Per-layer maze: each phase isolated, configurable memory budgets
- [ ] Full test suite passing before and after every change
