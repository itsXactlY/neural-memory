// include/neural/simd.h - SIMD-accelerated vector operations
// Architecture: AVX-512 (Zen4/Xeon) > AVX2 (Zen2/Zen3) > NEON (ARM) > Scalar
// Aligned loads throughout — std::assume_aligned tells the compiler
// that input pointers are 32-byte (AVX2) or 64-byte (AVX-512) aligned.

#pragma once

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>

// ─── CPU feature detection ────────────────────────────────────────────────────

#if defined(__AVX512F__) && defined(__AVX512ER__)  // AVX-512 CD/F/BW/VL/4VNNIW
 #include <immintrin.h>
 #define SIMD_HAS_AVX512F 1
 #define SIMD_AVX512_WIDTH 16
#elif defined(__AVX2__)
 #include <immintrin.h>
 #define SIMD_HAS_AVX2 1
 #define SIMD_AVX2_WIDTH 8
#else
 #define SIMD_HAS_AVX2 0
#endif

#ifdef __ARM_NEON
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

namespace simd {

constexpr size_t ALIGN = 32;   // AVX2/AVX-512 alignment boundary

// ─── Alignment helpers ────────────────────────────────────────────────────────

template<typename T>
inline bool is_aligned(const T* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & (ALIGN - 1)) == 0;
}

// ─── Dot Product ─────────────────────────────────────────────────────────────

inline float dot_product(const float* a, const float* b, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16 && is_aligned(a) && is_aligned(b)) {
        __m512 sum = _mm512_setzero_ps();
        size_t i = 0;
        const float* aa = std::assume_aligned<64>(a);
        const float* bb = std::assume_aligned<64>(b);
        for (; i + 16 <= n; i += 16) {
            __m512 va = _mm512_load_ps(aa + i);
            __m512 vb = _mm512_load_ps(bb + i);
            sum = _mm512_fmadd_ps(va, vb, sum);
        }
        float result = _mm512_reduce_add_ps(sum);
        for (; i < n; ++i) result += a[i] * b[i];
        return result;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8 && is_aligned(a) && is_aligned(b)) {
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;
        const float* aa = std::assume_aligned<32>(a);
        const float* bb = std::assume_aligned<32>(b);
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_load_ps(aa + i);
            __m256 vb = _mm256_load_ps(bb + i);
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, sum);
        float result = tmp[0] + tmp[1] + tmp[2] + tmp[3]
                     + tmp[4] + tmp[5] + tmp[6] + tmp[7];
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
    if (n >= 16 && is_aligned(a)) {
        __m512 sum = _mm512_setzero_ps();
        size_t i = 0;
        const float* aa = std::assume_aligned<64>(a);
        for (; i + 16 <= n; i += 16) {
            __m512 va = _mm512_load_ps(aa + i);
            sum = _mm512_fmadd_ps(va, va, sum);
        }
        float result = _mm512_reduce_add_ps(sum);
        for (; i < n; ++i) result += a[i] * a[i];
        return std::sqrt(result);
    }
#elif SIMD_HAS_AVX2
    if (n >= 8 && is_aligned(a)) {
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;
        const float* aa = std::assume_aligned<32>(a);
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_load_ps(aa + i);
            sum = _mm256_fmadd_ps(va, va, sum);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, sum);
        float result = tmp[0] + tmp[1] + tmp[2] + tmp[3]
                     + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; i < n; ++i) result += a[i] * a[i];
        return std::sqrt(result);
    }
#endif
    float result = 0.0f;
    for (size_t i = 0; i < n; ++i) result += a[i] * a[i];
    return std::sqrt(result);
}

// ─── Cosine Similarity ──────────────────────────────────────────────────────
// Single vector pair. For batch, use batch_cosine_similarity() below.

inline float cosine_similarity(const float* a, const float* b, size_t n) {
    if (n == 0) return 0.0f;

#if SIMD_HAS_AVX512F
    if (n >= 16 && is_aligned(a) && is_aligned(b)) {
        __m512 vdot = _mm512_setzero_ps();
        __m512 vna  = _mm512_setzero_ps();
        __m512 vnb  = _mm512_setzero_ps();
        size_t i = 0;
        const float* aa = std::assume_aligned<64>(a);
        const float* bb = std::assume_aligned<64>(b);
        for (; i + 16 <= n; i += 16) {
            __m512 va = _mm512_load_ps(aa + i);
            __m512 vb = _mm512_load_ps(bb + i);
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
    if (n >= 8 && is_aligned(a) && is_aligned(b)) {
        __m256 vdot = _mm256_setzero_ps();
        __m256 vna  = _mm256_setzero_ps();
        __m256 vnb  = _mm256_setzero_ps();
        size_t i = 0;
        const float* aa = std::assume_aligned<32>(a);
        const float* bb = std::assume_aligned<32>(b);
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_load_ps(aa + i);
            __m256 vb = _mm256_load_ps(bb + i);
            vdot = _mm256_fmadd_ps(va, vb, vdot);
            vna  = _mm256_fmadd_ps(va,  va, vna);
            vnb  = _mm256_fmadd_ps(vb,  vb, vnb);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, vdot);
        float dot = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        _mm256_store_ps(tmp, vna);
        float na = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        _mm256_store_ps(tmp, vnb);
        float nb = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
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

// ─── Element-wise Ops ───────────────────────────────────────────────────────

inline void add(const float* a, const float* b, float* c, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16 && is_aligned(c)) {
        size_t i = 0;
        const float* aa = std::assume_aligned<64>(a);
        const float* bb = std::assume_aligned<64>(b);
        float* cc = std::assume_aligned<64>(c);
        for (; i + 16 <= n; i += 16) {
            _mm512_store_ps(cc + i, _mm512_add_ps(_mm512_load_ps(aa + i), _mm512_load_ps(bb + i)));
        }
    }
#elif SIMD_HAS_AVX2
    if (n >= 8 && is_aligned(c)) {
        size_t i = 0;
        const float* aa = std::assume_aligned<32>(a);
        const float* bb = std::assume_aligned<32>(b);
        float* cc = std::assume_aligned<32>(c);
        for (; i + 8 <= n; i += 8) {
            _mm256_store_ps(cc + i, _mm256_add_ps(_mm256_load_ps(aa + i), _mm256_load_ps(bb + i)));
        }
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

inline void hadamard(const float* a, const float* b, float* c, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16 && is_aligned(c)) {
        size_t i = 0;
        const float* aa = std::assume_aligned<64>(a);
        const float* bb = std::assume_aligned<64>(b);
        float* cc = std::assume_aligned<64>(c);
        for (; i + 16 <= n; i += 16) {
            _mm512_store_ps(cc + i, _mm512_mul_ps(_mm512_load_ps(aa + i), _mm512_load_ps(bb + i)));
        }
    }
#elif SIMD_HAS_AVX2
    if (n >= 8 && is_aligned(c)) {
        size_t i = 0;
        const float* aa = std::assume_aligned<32>(a);
        const float* bb = std::assume_aligned<32>(b);
        float* cc = std::assume_aligned<32>(c);
        for (; i + 8 <= n; i += 8) {
            _mm256_store_ps(cc + i, _mm256_mul_ps(_mm256_load_ps(aa + i), _mm256_load_ps(bb + i)));
        }
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
}

inline void scale(const float* src, float scalar, float* dst, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16 && is_aligned(dst)) {
        __m512 vs = _mm512_set1_ps(scalar);
        size_t i = 0;
        const float* ss = std::assume_aligned<64>(src);
        float* dd = std::assume_aligned<64>(dst);
        for (; i + 16 <= n; i += 16) {
            _mm512_store_ps(dd + i, _mm512_mul_ps(_mm512_load_ps(ss + i), vs));
        }
        for (; i < n; ++i) dst[i] = src[i] * scalar;
        return;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8 && is_aligned(dst)) {
        __m256 vs = _mm256_set1_ps(scalar);
        size_t i = 0;
        const float* ss = std::assume_aligned<32>(src);
        float* dd = std::assume_aligned<32>(dst);
        for (; i + 8 <= n; i += 8) {
            _mm256_store_ps(dd + i, _mm256_mul_ps(_mm256_load_ps(ss + i), vs));
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
    if (n >= 16 && is_aligned(dst)) {
        size_t i = 0;
        const float* aa = std::assume_aligned<64>(a);
        const float* bb = std::assume_aligned<64>(b);
        const float* cc = std::assume_aligned<64>(c);
        float* dd = std::assume_aligned<64>(dst);
        for (; i + 16 <= n; i += 16) {
            _mm512_store_ps(dd + i, _mm512_fmadd_ps(_mm512_load_ps(aa + i), _mm512_load_ps(bb + i), _mm512_load_ps(cc + i)));
        }
        for (; i < n; ++i) dst[i] = a[i] * b[i] + c[i];
        return;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8 && is_aligned(dst)) {
        size_t i = 0;
        const float* aa = std::assume_aligned<32>(a);
        const float* bb = std::assume_aligned<32>(b);
        const float* cc = std::assume_aligned<32>(c);
        float* dd = std::assume_aligned<32>(dst);
        for (; i + 8 <= n; i += 8) {
            _mm256_store_ps(dd + i, _mm256_fmadd_ps(_mm256_load_ps(aa + i), _mm256_load_ps(bb + i), _mm256_load_ps(cc + i)));
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
        const float* ss = std::assume_aligned<64>(src);
        float* dd = std::assume_aligned<64>(dst);
        for (; i + 16 <= n; i += 16) {
            _mm512_store_ps(dd + i, _mm512_fmadd_ps(_mm512_load_ps(ss + i), vw, _mm512_load_ps(dd + i)));
        }
        for (; i < n; ++i) dst[i] += src[i] * weight;
        return;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8) {
        __m256 vw = _mm256_set1_ps(weight);
        size_t i = 0;
        const float* ss = std::assume_aligned<32>(src);
        float* dd = std::assume_aligned<32>(dst);
        for (; i + 8 <= n; i += 8) {
            _mm256_store_ps(dd + i, _mm256_fmadd_ps(_mm256_load_ps(ss + i), vw, _mm256_load_ps(dd + i)));
        }
        for (; i < n; ++i) dst[i] += src[i] * weight;
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i) dst[i] += src[i] * weight;
}

inline void zero(float* dst, size_t n) {
#if SIMD_HAS_AVX512F
    if (n >= 16 && is_aligned(dst)) {
        __m512 vz = _mm512_setzero_ps();
        size_t i = 0;
        float* dd = std::assume_aligned<64>(dst);
        for (; i + 16 <= n; i += 16) _mm512_store_ps(dd + i, vz);
        for (; i < n; ++i) dst[i] = 0.0f;
        return;
    }
#elif SIMD_HAS_AVX2
    if (n >= 8 && is_aligned(dst)) {
        __m256 vz = _mm256_setzero_ps();
        size_t i = 0;
        float* dd = std::assume_aligned<32>(dst);
        for (; i + 8 <= n; i += 8) _mm256_store_ps(dd + i, vz);
        for (; i < n; ++i) dst[i] = 0.0f;
        return;
    }
#endif
    std::memset(dst, 0, n * sizeof(float));
}

inline void copy(const float* src, float* dst, size_t n) {
    std::memcpy(dst, src, n * sizeof(float));
}

inline void normalize(float* v, size_t n) {
    float norm = l2_norm(v, n);
    if (norm > 1e-10f) scale(v, 1.0f / norm, v, n);
}

// argmax: index of maximum element
inline size_t argmax(const float* v, size_t n) {
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

// ─── Batch Cosine Similarity ───────────────────────────────────────────────
// Computes cosine(query, vectors[i]) for i in [0, count)
// vectors must be dim-aligned and contiguous: vectors[i*dim : (i+1)*dim]
// results must have capacity for count floats.

inline void batch_cosine_similarity(
    const float* query,
    const float* vectors,
    size_t count,
    size_t dim,
    float* results)
{
    if (count == 0 || dim == 0) return;

#if SIMD_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (size_t i = 0; i < count; ++i) {
        results[i] = cosine_similarity(query, vectors + i * dim, dim);
    }
}

// batch_cosine_avx512: AVX-512 batched inner loop
// Precomputes query norm; processes 16 vectors per iteration using AVX-512
// for the dot product and norm accumulation.
inline void batch_cosine_avx512(
    const float* query,
    const float* vectors,
    size_t count,
    size_t dim,
    float* results)
{
#if SIMD_HAS_AVX512F
    if (!is_aligned(vectors) || !is_aligned(results)) {
        batch_cosine_similarity(query, vectors, count, dim, results);
        return;
    }

    // Pre-compute query norm components once
    __m512 vqnorm = _mm512_setzero_ps();
    size_t qi = 0;
    const float* qq = std::assume_aligned<64>(query);
    for (; qi + 16 <= dim; qi += 16) {
        __m512 vq = _mm512_load_ps(qq + qi);
        vqnorm = _mm512_fmadd_ps(vq, vq, vqnorm);
    }
    float qnorm = _mm512_reduce_add_ps(vqnorm);
    for (; qi < dim; ++qi) qnorm += query[qi] * query[qi];
    float qnorm_sqrt = std::sqrt(qnorm);
    if (qnorm_sqrt < 1e-10f) {
        for (size_t i = 0; i < count; ++i) results[i] = 0.0f;
        return;
    }
    __m512 vqnorm_sqrt = _mm512_set1_ps(qnorm_sqrt);

    const float* vv = std::assume_aligned<64>(vectors);
    for (size_t i = 0; i < count; ++i) {
        const float* vec = vv + i * dim;

        __m512 vdot = _mm512_setzero_ps();
        __m512 vvn  = _mm512_setzero_ps();
        size_t j = 0;

        for (; j + 16 <= dim; j += 16) {
            __m512 vc = _mm512_load_ps(vec + j);
            __m512 vq = _mm512_load_ps(qq + j);
            vdot = _mm512_fmadd_ps(vc, vq, vdot);
            vvn  = _mm512_fmadd_ps(vc, vc,  vvn);
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
    batch_cosine_similarity(query, vectors, count, dim, results);
#endif
}

} // namespace simd
