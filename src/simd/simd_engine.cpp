// simd_engine.cpp - SIMD compilation unit
// All SIMD functions are inline in simd.h - this file provides:
// 1. Runtime CPU feature detection (cached, thread-safe via std::call_once)
// 2. Build-time standard / ISA banner

#include "neural/simd.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <mutex>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
 #include <cpuid.h>
 #define SIMD_ENGINE_X86 1
#else
 #define SIMD_ENGINE_X86 0
#endif

namespace neural {
namespace simd {

// ============================================================================
// Runtime CPU detection
// ============================================================================

struct CpuFeatures {
    bool sse41    = false;
    bool avx      = false;
    bool avx2     = false;
    bool avx512f  = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool fma      = false;
    bool f16c     = false;
    bool bmi2     = false;
    bool neon     = false;
};

static CpuFeatures g_features;
static std::once_flag g_detect_once;

static void detect_once() {
#if SIMD_ENGINE_X86
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        g_features.sse41 = (ecx >> 19) & 1;
        g_features.fma   = (ecx >> 12) & 1;
        g_features.f16c  = (ecx >> 29) & 1;
        g_features.avx   = (ecx >> 28) & 1;
    }

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        g_features.avx2     = (ebx >>  5) & 1;
        g_features.bmi2     = (ebx >>  8) & 1;
        g_features.avx512f  = (ebx >> 16) & 1;
        g_features.avx512bw = (ebx >> 30) & 1;
        g_features.avx512vl = (ebx >> 31) & 1;
    }
#elif SIMD_HAS_NEON
    g_features.neon = true;
#endif
}

const CpuFeatures& detect_cpu() {
    std::call_once(g_detect_once, &detect_once);
    return g_features;
}

void print_simd_info() {
    auto& f = detect_cpu();
    std::cout << "Build:\n"
              << "  C++ standard: " << __cplusplus << "L\n"
              << "  SIMD compile-time: "
              << (SIMD_HAS_AVX512F ? "AVX-512F"
                  : SIMD_HAS_AVX2  ? "AVX2"
                  : SIMD_HAS_NEON  ? "NEON"
                  : "scalar")
              << "\n";
    std::cout << "CPU Features (runtime):\n"
              << "  SSE4.1:    " << (f.sse41    ? "YES" : "NO") << "\n"
              << "  AVX:       " << (f.avx      ? "YES" : "NO") << "\n"
              << "  AVX2:      " << (f.avx2     ? "YES" : "NO") << "\n"
              << "  AVX-512F:  " << (f.avx512f  ? "YES" : "NO") << "\n"
              << "  AVX-512BW: " << (f.avx512bw ? "YES" : "NO") << "\n"
              << "  AVX-512VL: " << (f.avx512vl ? "YES" : "NO") << "\n"
              << "  FMA:       " << (f.fma      ? "YES" : "NO") << "\n"
              << "  F16C:      " << (f.f16c     ? "YES" : "NO") << "\n"
              << "  BMI2:      " << (f.bmi2     ? "YES" : "NO") << "\n"
              << "  NEON:      " << (f.neon     ? "YES" : "NO") << "\n";
    #ifdef _OPENMP
    std::cout << "  OpenMP:    YES (max threads: " << omp_get_max_threads() << ")\n";
    #else
    std::cout << "  OpenMP:    NO\n";
    #endif
}

} // namespace simd
} // namespace neural
