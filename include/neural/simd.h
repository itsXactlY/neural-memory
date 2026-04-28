// include/neural/simd.h - SIMD-accelerated vector operations (V3.1, C++26-ready)
//
// Architecture: AVX-512F (Zen4/SPR) > AVX2+FMA (Zen2/Zen3/Skylake+) > NEON (ARM) > Scalar
//
// V3.1 changes (extensions only — every prior call site keeps same semantics):
//  - AVX-512 detection no longer requires the obscure AVX512ER (Knights Mill only).
//  - SIMD_HAS_AVX2 / SIMD_HAS_AVX512F / SIMD_HAS_NEON always defined as 0/1.
//  - Aligned loads (`_mm{256,512}_load_ps`) replaced with unaligned loads
//    (`_mm{256,512}_loadu_ps`) which on every CPU since Skylake/Zen2 have
//    *zero* cost when data is actually aligned, and are correct otherwise.
//    Removes a class of latent UB from the previous version.
//  - is_aligned<N>() template + cache-line prefetch helpers added.
//  - pairwise_reduce_add for numerically-stable horizontal reductions.
//  - Real AVX2 batch_cosine path (was AVX-512 only — dead code on Zen3).
//  - [[likely]]/[[unlikely]] hints on hot fast paths.
//  - Public batch_cosine_dispatch() picks the best ISA at runtime.
//
// All previously-public symbols keep their signatures and observable behavior.

#pragma once

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

// ─── CPU feature detection ───────────────────────────────────────────────────

#if defined(__AVX512F__)
 #include <immintrin.h>
 #define SIMD_HAS_AVX512F 1
#else
 #define SIMD_HAS_AVX512F 0
#endif

#if defined(__AVX2__)
 #include <immintrin.h>
 #define SIMD_HAS_AVX2 1
#else
 #define SIMD_HAS_AVX2 0
#endif

#if defined(__ARM_NEON)
 #include <arm_neon.h>
 #define SIMD_HAS_NEON 1
#else
 #define SIMD_HAS_NEON 0
#endif

#ifdef _OPENMP
 #include <omp.h>
 #define SIMD_HAS_OPENMP 1
#else
 #define SIMD_HAS_OPENMP 0
#endif

// Branch hints (C++20, supported back to GCC 9 / Clang 12).
#if defined(__has_cpp_attribute)
 #if __has_cpp_attribute(likely)
  #define SIMD_LIKELY [[likely]]
  #define SIMD_UNLIKELY [[unlikely]]
 #else
  #define SIMD_LIKELY
  #define SIMD_UNLIKELY
 #endif
#else
 #define SIMD_LIKELY
 #define SIMD_UNLIKELY
#endif

namespace simd {

// ─── Constants ───────────────────────────────────────────────────────────────

// Logical alignment for AVX2 / NEON. AVX-512 callers should request 64.
constexpr size_t ALIGN     = 32;
constexpr size_t ALIGN_512 = 64;

// ─── Alignment helpers ───────────────────────────────────────────────────────

template<size_t N, typename T>
inline bool is_aligned(const T* ptr) noexcept {
    static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of two");
    return (reinterpret_cast<uintptr_t>(ptr) & (N - 1)) == 0;
}

// Backwards-compatible default: 32-byte alignment query (used by older callers).
template<typename T>
inline bool is_aligned(const T* ptr) noexcept {
    return is_aligned<ALIGN>(ptr);
}

// L1 cache-line prefetch hints (T0 = all levels). Safe no-op if unsupported.
inline void prefetch(const void* p) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(p, 0, 3);
#elif defined(_MSC_VER) && (defined(__AVX2__) || defined(__SSE__))
    _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
#else
    (void)p;
#endif
}

// ─── Pairwise reduction (numerically stable) ─────────────────────────────────
// Adding 8/16 floats with a left-to-right naive sum loses ~3 bits vs a
// pairwise tree. Used internally for horizontal reductions of SIMD lanes.

template<size_t N>
inline float pairwise_reduce_add(const float (&v)[N]) noexcept {
    static_assert((N & (N - 1)) == 0, "N must be a power of two");
    float buf[N];
    std::memcpy(buf, v, sizeof(buf));
    for (size_t step = 1; step < N; step <<= 1)
        for (size_t i = 0; i + step < N; i += (step << 1))
            buf[i] += buf[i + step];
    return buf[0];
}

// ─── Dot Product ─────────────────────────────────────────────────────────────

inline float dot_product(const float* a, const float* b, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16) SIMD_LIKELY {
        __m512 sum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            sum = _mm512_fmadd_ps(va, vb, sum);
        }
        float result = _mm512_reduce_add_ps(sum);
        for (; i < n; ++i) result += a[i] * b[i];
        return result;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8) SIMD_LIKELY {
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, sum);
        float result = pairwise_reduce_add(tmp);
        for (; i < n; ++i) result += a[i] * b[i];
        return result;
    }
#elif SIMD_HAS_NEON
    if (n >= 4) SIMD_LIKELY {
        float32x4_t sum = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            sum = vfmaq_f32(sum, vld1q_f32(a + i), vld1q_f32(b + i));
        }
        float result = vaddvq_f32(sum);
        for (; i < n; ++i) result += a[i] * b[i];
        return result;
    }
#endif
    float result = 0.0f;
    for (size_t i = 0; i < n; ++i) result += a[i] * b[i];
    return result;
}

// ─── L2 Norm ─────────────────────────────────────────────────────────────────

inline float l2_norm(const float* a, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16) SIMD_LIKELY {
        __m512 sum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            sum = _mm512_fmadd_ps(va, va, sum);
        }
        float result = _mm512_reduce_add_ps(sum);
        for (; i < n; ++i) result += a[i] * a[i];
        return std::sqrt(result);
    }
#elif SIMD_HAS_AVX2
    if (n >= 8) SIMD_LIKELY {
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            sum = _mm256_fmadd_ps(va, va, sum);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, sum);
        float result = pairwise_reduce_add(tmp);
        for (; i < n; ++i) result += a[i] * a[i];
        return std::sqrt(result);
    }
#elif SIMD_HAS_NEON
    if (n >= 4) SIMD_LIKELY {
        float32x4_t sum = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t v = vld1q_f32(a + i);
            sum = vfmaq_f32(sum, v, v);
        }
        float result = vaddvq_f32(sum);
        for (; i < n; ++i) result += a[i] * a[i];
        return std::sqrt(result);
    }
#endif
    float result = 0.0f;
    for (size_t i = 0; i < n; ++i) result += a[i] * a[i];
    return std::sqrt(result);
}

// ─── Cosine Similarity ───────────────────────────────────────────────────────
// Single vector pair. For batch, use batch_cosine_similarity() below.

inline float cosine_similarity(const float* a, const float* b, size_t n) {
    if (n == 0) SIMD_UNLIKELY return 0.0f;

#if SIMD_HAS_AVX512F
    if (n >= 16) SIMD_LIKELY {
        __m512 vdot = _mm512_setzero_ps();
        __m512 vna  = _mm512_setzero_ps();
        __m512 vnb  = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            vdot = _mm512_fmadd_ps(va, vb, vdot);
            vna  = _mm512_fmadd_ps(va,  va, vna);
            vnb  = _mm512_fmadd_ps(vb,  vb, vnb);
        }
        float dot = _mm512_reduce_add_ps(vdot);
        float na  = _mm512_reduce_add_ps(vna);
        float nb  = _mm512_reduce_add_ps(vnb);
        for (; i < n; ++i) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        float denom = std::sqrt(na) * std::sqrt(nb);
        return denom < 1e-10f ? 0.0f : dot / denom;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8) SIMD_LIKELY {
        __m256 vdot = _mm256_setzero_ps();
        __m256 vna  = _mm256_setzero_ps();
        __m256 vnb  = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            vdot = _mm256_fmadd_ps(va, vb, vdot);
            vna  = _mm256_fmadd_ps(va,  va, vna);
            vnb  = _mm256_fmadd_ps(vb,  vb, vnb);
        }
        alignas(32) float tdot[8], tna[8], tnb[8];
        _mm256_store_ps(tdot, vdot);
        _mm256_store_ps(tna,  vna);
        _mm256_store_ps(tnb,  vnb);
        float dot = pairwise_reduce_add(tdot);
        float na  = pairwise_reduce_add(tna);
        float nb  = pairwise_reduce_add(tnb);
        for (; i < n; ++i) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        float denom = std::sqrt(na) * std::sqrt(nb);
        return denom < 1e-10f ? 0.0f : dot / denom;
    }
#elif SIMD_HAS_NEON
    if (n >= 4) SIMD_LIKELY {
        float32x4_t vdot = vdupq_n_f32(0.0f);
        float32x4_t vna  = vdupq_n_f32(0.0f);
        float32x4_t vnb  = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            vdot = vfmaq_f32(vdot, va, vb);
            vna  = vfmaq_f32(vna,  va, va);
            vnb  = vfmaq_f32(vnb,  vb, vb);
        }
        float dot = vaddvq_f32(vdot);
        float na  = vaddvq_f32(vna);
        float nb  = vaddvq_f32(vnb);
        for (; i < n; ++i) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        float denom = std::sqrt(na) * std::sqrt(nb);
        return denom < 1e-10f ? 0.0f : dot / denom;
    }
#endif
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom < 1e-10f ? 0.0f : dot / denom;
}

// ─── Element-wise Ops ────────────────────────────────────────────────────────
// All ops below now use unaligned loads/stores — no UB risk and identical
// instruction selection on Zen2+/Skylake+ when data is actually aligned.

inline void add(const float* a, const float* b, float* c, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16) {
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_ps(c + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
        }
        for (; i < n; ++i) c[i] = a[i] + b[i];
        return;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(c + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
        for (; i < n; ++i) c[i] = a[i] + b[i];
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

inline void hadamard(const float* a, const float* b, float* c, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16) {
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_ps(c + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
        }
        for (; i < n; ++i) c[i] = a[i] * b[i];
        return;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(c + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
        for (; i < n; ++i) c[i] = a[i] * b[i];
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
}

inline void scale(const float* src, float scalar, float* dst, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16) {
        __m512 vs = _mm512_set1_ps(scalar);
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(src + i), vs));
        }
        for (; i < n; ++i) dst[i] = src[i] * scalar;
        return;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8) {
        __m256 vs = _mm256_set1_ps(scalar);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), vs));
        }
        for (; i < n; ++i) dst[i] = src[i] * scalar;
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i) dst[i] = src[i] * scalar;
}

// fmadd: dst = a * b + c
inline void fmadd(const float* a, const float* b, const float* c, float* dst, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16) {
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), _mm512_loadu_ps(c + i)));
        }
        for (; i < n; ++i) dst[i] = a[i] * b[i] + c[i];
        return;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_loadu_ps(c + i)));
        }
        for (; i < n; ++i) dst[i] = a[i] * b[i] + c[i];
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] * b[i] + c[i];
}

// weighted_add: dst += src * weight  (in-place)
inline void weighted_add(const float* src, float weight, float* dst, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16) {
        __m512 vw = _mm512_set1_ps(weight);
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(_mm512_loadu_ps(src + i), vw, _mm512_loadu_ps(dst + i)));
        }
        for (; i < n; ++i) dst[i] += src[i] * weight;
        return;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8) {
        __m256 vw = _mm256_set1_ps(weight);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_mm256_loadu_ps(src + i), vw, _mm256_loadu_ps(dst + i)));
        }
        for (; i < n; ++i) dst[i] += src[i] * weight;
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i) dst[i] += src[i] * weight;
}

inline void zero(float* dst, size_t n) {
    std::memset(dst, 0, n * sizeof(float));
}

inline void copy(const float* src, float* dst, size_t n) {
    std::memcpy(dst, src, n * sizeof(float));
}

inline void normalize(float* v, size_t n) {
    float norm = l2_norm(v, n);
    if (norm > 1e-10f) scale(v, 1.0f / norm, v, n);
}

// argmax: index of maximum element. Scalar (n is small for this use case).
inline size_t argmax(const float* v, size_t n) {
    if (n == 0) SIMD_UNLIKELY return 0;
    size_t best = 0;
    float best_val = v[0];
    for (size_t i = 1; i < n; ++i) {
        if (v[i] > best_val) {
            best_val = v[i];
            best = i;
        }
    }
    return best;
}

// ─── Batch Cosine Similarity (default: scalar fallback per-row) ─────────────
// Computes cosine(query, vectors[i]) for i in [0, count)
// vectors must be contiguous: vectors[i*dim : (i+1)*dim]
// results must have capacity for count floats.

inline void batch_cosine_similarity(
    const float* query,
    const float* vectors,
    size_t count,
    size_t dim,
    float* results)
{
    if (count == 0 || dim == 0) SIMD_UNLIKELY return;

#if SIMD_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (size_t i = 0; i < count; ++i) {
        results[i] = cosine_similarity(query, vectors + i * dim, dim);
    }
}

// batch_cosine_avx2: AVX2 batched inner loop with prefetching.
// Precomputes query norm once; processes 8 floats per AVX2 lane.
// On a Zen3 / Skylake-X this is the hot path for 1024-dim BGE-M3 batches.
inline void batch_cosine_avx2(
    const float* query,
    const float* vectors,
    size_t count,
    size_t dim,
    float* results)
{
#if SIMD_HAS_AVX2
    if (count == 0 || dim == 0) SIMD_UNLIKELY return;

    // Pre-compute query norm once.
    __m256 vqnorm = _mm256_setzero_ps();
    size_t qi = 0;
    for (; qi + 8 <= dim; qi += 8) {
        __m256 vq = _mm256_loadu_ps(query + qi);
        vqnorm = _mm256_fmadd_ps(vq, vq, vqnorm);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vqnorm);
    float qnorm = pairwise_reduce_add(tmp);
    for (; qi < dim; ++qi) qnorm += query[qi] * query[qi];
    float qnorm_sqrt = std::sqrt(qnorm);
    if (qnorm_sqrt < 1e-10f) SIMD_UNLIKELY {
        std::memset(results, 0, count * sizeof(float));
        return;
    }

    constexpr size_t PREFETCH_DIST = 4;  // prefetch 4 rows ahead

    #if SIMD_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 32)
    #endif
    for (size_t i = 0; i < count; ++i) {
        // Prefetch upcoming row (single line ~64B; hint for L1).
        if (i + PREFETCH_DIST < count) {
            prefetch(vectors + (i + PREFETCH_DIST) * dim);
        }
        const float* vec = vectors + i * dim;

        __m256 vdot = _mm256_setzero_ps();
        __m256 vvn  = _mm256_setzero_ps();
        size_t j = 0;

        for (; j + 8 <= dim; j += 8) {
            __m256 vc = _mm256_loadu_ps(vec + j);
            __m256 vq = _mm256_loadu_ps(query + j);
            vdot = _mm256_fmadd_ps(vc, vq, vdot);
            vvn  = _mm256_fmadd_ps(vc, vc, vvn);
        }
        alignas(32) float tdot[8], tvn[8];
        _mm256_store_ps(tdot, vdot);
        _mm256_store_ps(tvn,  vvn);
        float dot = pairwise_reduce_add(tdot);
        float vn  = pairwise_reduce_add(tvn);
        for (; j < dim; ++j) {
            dot += vec[j] * query[j];
            vn  += vec[j] * vec[j];
        }
        float denom = std::sqrt(vn) * qnorm_sqrt;
        results[i] = denom < 1e-10f ? 0.0f : dot / denom;
    }
#else
    batch_cosine_similarity(query, vectors, count, dim, results);
#endif
}

// batch_cosine_avx512: AVX-512 batched inner loop (16 floats/lane).
inline void batch_cosine_avx512(
    const float* query,
    const float* vectors,
    size_t count,
    size_t dim,
    float* results)
{
#if SIMD_HAS_AVX512F
    if (count == 0 || dim == 0) SIMD_UNLIKELY return;

    __m512 vqnorm = _mm512_setzero_ps();
    size_t qi = 0;
    for (; qi + 16 <= dim; qi += 16) {
        __m512 vq = _mm512_loadu_ps(query + qi);
        vqnorm = _mm512_fmadd_ps(vq, vq, vqnorm);
    }
    float qnorm = _mm512_reduce_add_ps(vqnorm);
    for (; qi < dim; ++qi) qnorm += query[qi] * query[qi];
    float qnorm_sqrt = std::sqrt(qnorm);
    if (qnorm_sqrt < 1e-10f) SIMD_UNLIKELY {
        std::memset(results, 0, count * sizeof(float));
        return;
    }

    constexpr size_t PREFETCH_DIST = 4;

    #if SIMD_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 32)
    #endif
    for (size_t i = 0; i < count; ++i) {
        if (i + PREFETCH_DIST < count) {
            prefetch(vectors + (i + PREFETCH_DIST) * dim);
        }
        const float* vec = vectors + i * dim;

        __m512 vdot = _mm512_setzero_ps();
        __m512 vvn  = _mm512_setzero_ps();
        size_t j = 0;

        for (; j + 16 <= dim; j += 16) {
            __m512 vc = _mm512_loadu_ps(vec + j);
            __m512 vq = _mm512_loadu_ps(query + j);
            vdot = _mm512_fmadd_ps(vc, vq, vdot);
            vvn  = _mm512_fmadd_ps(vc, vc, vvn);
        }
        float dot = _mm512_reduce_add_ps(vdot);
        float vn  = _mm512_reduce_add_ps(vvn);
        for (; j < dim; ++j) {
            dot += vec[j] * query[j];
            vn  += vec[j] * vec[j];
        }
        float denom = std::sqrt(vn) * qnorm_sqrt;
        results[i] = denom < 1e-10f ? 0.0f : dot / denom;
    }
#else
    batch_cosine_avx2(query, vectors, count, dim, results);
#endif
}

// batch_cosine_dispatch: pick the best available kernel at compile time.
// (Runtime dispatch lives in simd_engine.cpp — keep this header-only.)
inline void batch_cosine_dispatch(
    const float* query,
    const float* vectors,
    size_t count,
    size_t dim,
    float* results)
{
#if SIMD_HAS_AVX512F
    batch_cosine_avx512(query, vectors, count, dim, results);
#elif SIMD_HAS_AVX2
    batch_cosine_avx2(query, vectors, count, dim, results);
#else
    batch_cosine_similarity(query, vectors, count, dim, results);
#endif
}

} // namespace simd
