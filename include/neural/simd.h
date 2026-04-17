// include/neural/simd.h - SIMD-accelerated vector operations with scalar fallback
#pragma once

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstddef>

#ifdef __AVX2__
#include <immintrin.h>
#define SIMD_HAS_AVX2 1
#else
#define SIMD_HAS_AVX2 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
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

constexpr size_t ALIGN = 32;

// ============================================================================
// Dot Product
// ============================================================================

inline float dot_product(const float* a, const float* b, size_t n) {
#if SIMD_HAS_AVX2
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sum);
    float result = 0.0f;
    for (int j = 0; j < 8; ++j) result += tmp[j];
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
#elif SIMD_HAS_NEON
    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }
    float result = vaddvq_f32(sum);
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
#else
    float result = 0.0f;
    for (size_t i = 0; i < n; ++i) result += a[i] * b[i];
    return result;
#endif
}

// ============================================================================
// Cosine Similarity
// ============================================================================

inline float cosine_similarity(const float* a, const float* b, size_t n) {
#if SIMD_HAS_AVX2
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    __m256 vdot = _mm256_setzero_ps();
    __m256 vna = _mm256_setzero_ps();
    __m256 vnb = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vdot = _mm256_fmadd_ps(va, vb, vdot);
        vna = _mm256_fmadd_ps(va, va, vna);
        vnb = _mm256_fmadd_ps(vb, vb, vnb);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vdot);
    for (int j = 0; j < 8; ++j) dot += tmp[j];
    _mm256_store_ps(tmp, vna);
    for (int j = 0; j < 8; ++j) norm_a += tmp[j];
    _mm256_store_ps(tmp, vnb);
    for (int j = 0; j < 8; ++j) norm_b += tmp[j];
    for (; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
#elif SIMD_HAS_NEON
    float32x4_t vdot = vdupq_n_f32(0.0f);
    float32x4_t vna = vdupq_n_f32(0.0f);
    float32x4_t vnb = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vdot = vfmaq_f32(vdot, va, vb);
        vna = vfmaq_f32(vna, va, va);
        vnb = vfmaq_f32(vnb, vb, vb);
    }
    float dot = vaddvq_f32(vdot);
    float norm_a = vaddvq_f32(vna);
    float norm_b = vaddvq_f32(vnb);
    for (; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
#else
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
#endif
}

// ============================================================================
// L2 Norm
// ============================================================================

inline float l2_norm(const float* a, size_t n) {
#if SIMD_HAS_AVX2
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        sum = _mm256_fmadd_ps(va, va, sum);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sum);
    float result = 0.0f;
    for (int j = 0; j < 8; ++j) result += tmp[j];
    for (; i < n; ++i) result += a[i] * a[i];
    return std::sqrt(result);
#elif SIMD_HAS_NEON
    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        sum = vfmaq_f32(sum, va, va);
    }
    float result = vaddvq_f32(sum);
    for (; i < n; ++i) result += a[i] * a[i];
    return std::sqrt(result);
#else
    float result = 0.0f;
    for (size_t i = 0; i < n; ++i) result += a[i] * a[i];
    return std::sqrt(result);
#endif
}

// ============================================================================
// Element-wise Operations
// ============================================================================

inline void add(const float* a, const float* b, float* c, size_t n) {
#if SIMD_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
#elif SIMD_HAS_NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(c + i, vaddq_f32(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
#endif
}

inline void hadamard(const float* a, const float* b, float* c, size_t n) {
#if SIMD_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] * b[i];
#elif SIMD_HAS_NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(c + i, vmulq_f32(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] * b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
#endif
}

inline void scale(const float* src, float scalar, float* dst, size_t n) {
#if SIMD_HAS_AVX2
    __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(va, vs));
    }
    for (; i < n; ++i) dst[i] = src[i] * scalar;
#elif SIMD_HAS_NEON
    float32x4_t vs = vdupq_n_f32(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(src + i);
        vst1q_f32(dst + i, vmulq_f32(va, vs));
    }
    for (; i < n; ++i) dst[i] = src[i] * scalar;
#else
    for (size_t i = 0; i < n; ++i) dst[i] = src[i] * scalar;
#endif
}

inline void fmadd(const float* a, const float* b, const float* c, float* dst, size_t n) {
#if SIMD_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(va, vb, vc));
    }
    for (; i < n; ++i) dst[i] = a[i] * b[i] + c[i];
#elif SIMD_HAS_NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        vst1q_f32(dst + i, vfmaq_f32(vc, va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] * b[i] + c[i];
#else
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] * b[i] + c[i];
#endif
}

inline void weighted_add(const float* src, float weight, float* dst, size_t n) {
#if SIMD_HAS_AVX2
    __m256 vw = _mm256_set1_ps(weight);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(src + i);
        __m256 vd = _mm256_loadu_ps(dst + i);
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(va, vw, vd));
    }
    for (; i < n; ++i) dst[i] += weight * src[i];
#elif SIMD_HAS_NEON
    float32x4_t vw = vdupq_n_f32(weight);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(src + i);
        float32x4_t vd = vld1q_f32(dst + i);
        vst1q_f32(dst + i, vfmaq_f32(vd, va, vw));
    }
    for (; i < n; ++i) dst[i] += weight * src[i];
#else
    for (size_t i = 0; i < n; ++i) dst[i] += weight * src[i];
#endif
}

inline void zero(float* dst, size_t n) {
#if SIMD_HAS_AVX2
    size_t i = 0;
    __m256 vz = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, vz);
    }
    for (; i < n; ++i) dst[i] = 0.0f;
#elif SIMD_HAS_NEON
    float32x4_t vz = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(dst + i, vz);
    }
    for (; i < n; ++i) dst[i] = 0.0f;
#else
    std::memset(dst, 0, n * sizeof(float));
#endif
}

inline void copy(const float* src, float* dst, size_t n) {
    std::memcpy(dst, src, n * sizeof(float));
}

inline void normalize(float* v, size_t n) {
    float norm = l2_norm(v, n);
    if (norm > 1e-10f) scale(v, 1.0f / norm, v, n);
}

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

// ============================================================================
// Batch Operations
// ============================================================================

inline void batch_cosine_similarity(
    const float* query,
    const float* vectors,
    size_t count,
    size_t dim,
    float* results)
{
#if SIMD_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < count; ++i) {
        results[i] = cosine_similarity(query, vectors + i * dim, dim);
    }
}

} // namespace simd
