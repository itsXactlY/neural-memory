// include/neural/vector.h - Vector types using actual SIMD API
#pragma once

#include "neural/simd.h"
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <new>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <utility>

namespace neural {

// ============================================================================
// Aligned allocation helpers
// ============================================================================
// V3.1: alignment bumped from 32 → 64 so AVX-512 paths can use aligned loads
// safely on hardware that has them. AVX2 / NEON paths are unaffected (the
// stricter alignment is a superset of the previous 32-byte requirement).

template<typename T>
T* aligned_alloc_t(size_t count) {
    if (count == 0) return nullptr;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, simd::ALIGN_512, count * sizeof(T)) != 0) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
}

template<typename T>
void aligned_free_t(T* ptr) {
    free(ptr);
}

// ============================================================================
// Vector32f - Float32 vector with SIMD operations
// ============================================================================
// V3.1 extensions:
//  - Adds move constructor + move assignment (rule of 5). Previously every
//    "return Vector32f" through `operator+` etc. did a full malloc+memcpy;
//    moves now steal the buffer. Pure perf win, no observable behavior change.
//  - Underlying storage is 64-byte aligned (was 32) → AVX-512 ready.
//  - aligned_alloc_t throws std::bad_alloc on failure, no more silent nullptr.

class Vector32f {
public:
    Vector32f() noexcept : data_(nullptr), dim_(0), owns_(false) {}

    explicit Vector32f(size_t dim) : dim_(dim), owns_(true) {
        data_ = aligned_alloc_t<float>(dim);
        if (data_) std::memset(data_, 0, dim * sizeof(float));
    }

    Vector32f(size_t dim, float fill) : dim_(dim), owns_(true) {
        data_ = aligned_alloc_t<float>(dim);
        if (data_) std::fill_n(data_, dim, fill);
    }

    Vector32f(std::initializer_list<float> init) : dim_(init.size()), owns_(true) {
        data_ = aligned_alloc_t<float>(dim_);
        if (data_) std::copy(init.begin(), init.end(), data_);
    }

    Vector32f(const float* src, size_t dim) : dim_(dim), owns_(true) {
        data_ = aligned_alloc_t<float>(dim);
        if (data_) std::memcpy(data_, src, dim * sizeof(float));
    }

    ~Vector32f() { if (owns_ && data_) aligned_free_t(data_); }

    Vector32f(const Vector32f& o) : dim_(o.dim_), owns_(true) {
        data_ = aligned_alloc_t<float>(dim_);
        if (data_ && o.data_) std::memcpy(data_, o.data_, dim_ * sizeof(float));
    }

    Vector32f& operator=(const Vector32f& o) {
        if (this != &o) {
            if (owns_ && data_) aligned_free_t(data_);
            dim_ = o.dim_;
            owns_ = true;
            data_ = aligned_alloc_t<float>(dim_);
            if (data_ && o.data_) std::memcpy(data_, o.data_, dim_ * sizeof(float));
        }
        return *this;
    }

    // V3.1: move semantics — eliminates malloc+memcpy on returned-by-value.
    Vector32f(Vector32f&& o) noexcept
        : data_(o.data_), dim_(o.dim_), owns_(o.owns_) {
        o.data_ = nullptr;
        o.dim_  = 0;
        o.owns_ = false;
    }

    Vector32f& operator=(Vector32f&& o) noexcept {
        if (this != &o) {
            if (owns_ && data_) aligned_free_t(data_);
            data_ = o.data_;
            dim_  = o.dim_;
            owns_ = o.owns_;
            o.data_ = nullptr;
            o.dim_  = 0;
            o.owns_ = false;
        }
        return *this;
    }

    // Access
    size_t dim() const { return dim_; }
    float* data() { return data_; }
    const float* data() const { return data_; }

    float operator[](size_t i) const { return data_[i]; }
    float& operator[](size_t i) { return data_[i]; }

    // SIMD operations
    float dot(const Vector32f& o) const {
        return simd::dot_product(data_, o.data_, dim_);
    }

    float cosine_similarity(const Vector32f& o) const {
        return simd::cosine_similarity(data_, o.data_, dim_);
    }

    float norm() const {
        return simd::l2_norm(data_, dim_);
    }

    void normalize() {
        simd::normalize(data_, dim_);
    }

    // Arithmetic
    Vector32f operator+(const Vector32f& o) const {
        Vector32f r(dim_);
        simd::add(data_, o.data_, r.data_, dim_);
        return r;
    }

    Vector32f operator*(const Vector32f& o) const {  // Hadamard
        Vector32f r(dim_);
        simd::hadamard(data_, o.data_, r.data_, dim_);
        return r;
    }

    Vector32f operator*(float s) const {
        Vector32f r(dim_);
        simd::scale(data_, s, r.data_, dim_);
        return r;
    }

    void zero() { simd::zero(data_, dim_); }

    bool empty() const { return dim_ == 0; }

    friend std::ostream& operator<<(std::ostream& os, const Vector32f& v) {
        os << "[";
        size_t show = std::min(v.dim_, (size_t)8);
        for (size_t i = 0; i < show; ++i) {
            if (i > 0) os << ", ";
            os << v.data_[i];
        }
        if (v.dim_ > show) os << ", ...";
        os << "] (" << v.dim_ << "d)";
        return os;
    }

private:
    float* data_;
    size_t dim_;
    bool owns_;
};

// ============================================================================
// Batch operations
// ============================================================================

inline std::vector<float> batch_cosine_similarity(
    const Vector32f& query,
    const std::vector<Vector32f>& vectors)
{
    size_t n = vectors.size();
    std::vector<float> results(n);
    for (size_t i = 0; i < n; ++i) {
        results[i] = query.cosine_similarity(vectors[i]);
    }
    return results;
}

inline std::vector<float> batch_cosine_similarity_contiguous(
    const float* query,
    const float* vectors,
    size_t count,
    size_t dim)
{
    std::vector<float> results(count);
    simd::batch_cosine_similarity(query, vectors, count, dim, results.data());
    return results;
}

} // namespace neural
