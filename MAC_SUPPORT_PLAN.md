# Apple Silicon Support Plan — Neural Memory

**Branch:** `feat/apple-silicon-support`
**Goal:** Neural Memory runs natively on macOS with Apple Silicon (M1-M4), using Metal GPU acceleration for embeddings.

## Problem

Neural Memory is CUDA/Linux-only. On Mac, it falls back to weak TF-IDF+SVD (384d) instead of proper bge-m3 (1024d) embeddings. Three root causes:

1. **embed_provider.py** — Only checks `torch.cuda.is_available()`, no MPS (Metal) detection
2. **cpp_bridge.py** — Hardcoded `.so` extension, no `.dylib` for macOS
3. **C++ SIMD** — x86 SSE4.1/AVX2/AVX512 only, no ARM NEON
4. **neural_memory.py** — Hardcodes `dim=384` for C++ init

## Workstreams

### WS1: MPS GPU Backend in embed_provider.py
**File:** `embed_provider.py` (plugin copy + project copy)

- Add `torch.backends.mps.is_available()` check alongside CUDA
- Device priority: CUDA > MPS > CPU
- MPS uses unified memory — no memory check needed (same pool as CPU)
- Update `_auto_detect()` to show "MPS" device string on Mac
- Update print messages to say "Metal (MPS)" not just "CUDA"

### WS2: Cross-Platform Library Loading in cpp_bridge.py
**File:** `cpp_bridge.py`

- Add `.dylib` candidates alongside `.so` for macOS
- Use `ctypes.util.find_library` with platform detection
- Check for `libneural_memory.dylib` on Darwin
- If C++ lib not available on Mac, gracefully skip (Python fallback is fine)

### WS3: ARM NEON SIMD or Pure Python Fallback
**File:** `src/simd/simd_engine.cpp` + `include/simd/simd.h`

- Add `#ifdef __ARM_NEON` or `__aarch64__` guards
- ARM NEON intrinsics for cosine_similarity (float32x4_t)
- OR: detect ARM at compile time and skip SIMD (let Python fast_ops handle it)
- Cython fast_ops.pyx works on any platform — that's the real fallback
- CMakeLists.txt: detect Apple Silicon, skip AVX/SSE flags

### WS4: Dynamic Dimension Fix
**File:** `neural_memory.py`

- `self._cpp.initialize(dim=384)` → `self._cpp.initialize(dim=self._dim)`
- Use embedder's actual dimension, not hardcoded 384

### WS5: Integration & Testing
- Test on Mac: `python3 -c "from neural_memory import Memory; m = Memory(); print(m)"`
- Verify bge-m3 loads via MPS on M4 Pro
- Benchmark: tok/s for embed_batch with Metal vs CPU
- Verify C++ fallback gracefully skips on Mac

## Files to Modify

| File | Changes |
|------|---------|
| `plugins/memory/neural/embed_provider.py` | MPS detection, device priority |
| `plugins/memory/neural/cpp_bridge.py` | .dylib support, graceful Mac fallback |
| `plugins/memory/neural/neural_memory.py` | Dynamic dim=384 → self._dim |
| `src/simd/simd_engine.cpp` | NEON path or ARM guard |
| `include/simd/simd.h` | NEON intrinsics |
| `CMakeLists.txt` | Apple Silicon detection |

## Non-Goals (v1)

- Full Cython fast_ops build on Mac (Python fallback is fine for v1)
- MSSQL on Mac (SQLite only)
- Dream Engine on Mac (consolidation is optional)
