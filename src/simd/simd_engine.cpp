// simd_engine.cpp - SIMD compilation unit
// All SIMD functions are inline in simd.h - this file provides:
// 1. Runtime CPU feature detection
// 2. Explicit template instantiations if needed

#include "neural/simd.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <cpuid.h>

namespace neural {
namespace simd {

// ============================================================================
// Runtime CPU detection
// ============================================================================

struct CpuFeatures {
    bool sse41 = false;
    bool avx2 = false;
    bool avx512f = false;
    bool fma = false;
};

static CpuFeatures g_features;
static bool g_detected = false;

const CpuFeatures& detect_cpu() {
    if (g_detected) return g_features;
    
    unsigned int eax, ebx, ecx, edx;
    
    // Check SSE4.1 (CPUID leaf 1)
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        g_features.sse41 = (ecx >> 19) & 1;
        g_features.fma = (ecx >> 12) & 1;
    }
    
    // Check AVX2 and AVX-512F (CPUID leaf 7, subleaf 0)
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        g_features.avx2 = (ebx >> 5) & 1;
        g_features.avx512f = (ebx >> 16) & 1;  // AVX-512 Foundation
    }
    
    g_detected = true;
    return g_features;
}

void print_simd_info() {
    auto& f = detect_cpu();
    std::cout << "CPU Features:\n";
    std::cout << "  SSE4.1:  " << (f.sse41 ? "YES" : "NO") << "\n";
    std::cout << "  AVX2:    " << (f.avx2 ? "YES" : "NO") << "\n";
    std::cout << "  AVX-512F: " << (f.avx512f ? "YES" : "NO") << "\n";
    std::cout << "  FMA:     " << (f.fma ? "YES" : "NO") << "\n";
    #ifdef _OPENMP
    std::cout << "  OpenMP: YES (max threads: " << omp_get_max_threads() << ")\n";
    #else
    std::cout << "  OpenMP: NO\n";
    #endif
}

} // namespace simd
} // namespace neural
