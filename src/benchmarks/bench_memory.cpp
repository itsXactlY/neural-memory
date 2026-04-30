// benchmarks/bench_memory.cpp - Memory System Benchmark
#include "mazemaker/memory_adapter.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>

using namespace neural;
using Clock = std::chrono::high_resolution_clock;

std::vector<float> random_embedding(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    // Normalize
    float norm = 0;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 0) for (float& x : v) x /= norm;
    return v;
}

void bench_store(NeuralMemoryAdapter& adapter, size_t count, size_t dim) {
    std::mt19937 rng(42);
    
    std::cout << "\n--- STORE " << count << " memories (" << dim << "d) ---\n";
    
    auto start = Clock::now();
    for (size_t i = 0; i < count; ++i) {
        auto emb = random_embedding(dim, rng);
        adapter.store(emb, "mem_" + std::to_string(i), "Content for memory " + std::to_string(i));
    }
    auto end = Clock::now();
    
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double us_per = (ms * 1000.0) / count;
    
    std::cout << "  Total: " << std::fixed << std::setprecision(1) << ms << " ms\n";
    std::cout << "  Per op: " << std::setprecision(1) << us_per << " us\n";
    std::cout << "  Throughput: " << std::setprecision(0) << count / (ms / 1000.0) << " ops/sec\n";
}

void bench_retrieve(NeuralMemoryAdapter& adapter, size_t count, size_t dim) {
    std::mt19937 rng(123);
    
    std::cout << "\n--- RETRIEVE " << count << " queries (top-10) ---\n";
    
    auto start = Clock::now();
    size_t total_results = 0;
    for (size_t i = 0; i < count; ++i) {
        auto cue = random_embedding(dim, rng);
        auto results = adapter.retrieve(cue, 10);
        total_results += results.size();
    }
    auto end = Clock::now();
    
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double us_per = (ms * 1000.0) / count;
    
    std::cout << "  Total: " << std::fixed << std::setprecision(1) << ms << " ms\n";
    std::cout << "  Per op: " << std::setprecision(1) << us_per << " us\n";
    std::cout << "  Avg results: " << std::setprecision(1) << (double)total_results / count << "\n";
}

void bench_consolidation(NeuralMemoryAdapter& adapter) {
    std::cout << "\n--- CONSOLIDATION ---\n";
    
    auto start = Clock::now();
    size_t consolidated = adapter.consolidate();
    auto end = Clock::now();
    
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    std::cout << "  Time: " << std::fixed << std::setprecision(1) << ms << " ms\n";
    std::cout << "  Consolidated: " << consolidated << " memories\n";
}

void bench_spreading_activation(NeuralMemoryAdapter& adapter, size_t count) {
    std::cout << "\n--- SPREADING ACTIVATION (" << count << " seeds) ---\n";
    
    std::mt19937 rng(456);
    std::uniform_int_distribution<uint64_t> dist(1, 1000);
    
    auto start = Clock::now();
    size_t total_activated = 0;
    for (size_t i = 0; i < count; ++i) {
        uint64_t seed = dist(rng);
        auto results = adapter.think(seed, 3);
        total_activated += results.size();
    }
    auto end = Clock::now();
    
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double us_per = (ms * 1000.0) / count;
    
    std::cout << "  Total: " << std::fixed << std::setprecision(1) << ms << " ms\n";
    std::cout << "  Per op: " << std::setprecision(1) << us_per << " us\n";
    std::cout << "  Avg activated: " << std::setprecision(1) << (double)total_activated / count << "\n";
}

void bench_link_prediction(NeuralMemoryAdapter& adapter) {
    std::cout << "\n--- LINK PREDICTION ---\n";
    
    auto start = Clock::now();
    size_t predicted = adapter.predict_links();
    auto end = Clock::now();
    
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    std::cout << "  Time: " << std::fixed << std::setprecision(1) << ms << " ms\n";
    std::cout << "  Predicted edges: " << predicted << "\n";
}

void print_stats(const StatsSnapshot& stats) {
    std::cout << "\n=== STATS ===\n";
    std::cout << "  Hopfield: " << stats.hopfield_patterns << " patterns ("
              << std::setprecision(1) << stats.hopfield_occupancy * 100 << "%)\n";
    std::cout << "  Graph: " << stats.graph_nodes << " nodes, "
              << stats.graph_edges << " edges (density: "
              << std::setprecision(3) << stats.graph_density << ")\n";
    std::cout << "  Operations:\n";
    std::cout << "    Stores: " << stats.total_stores << " (avg "
              << stats.avg_store_us << " us)\n";
    std::cout << "    Retrieves: " << stats.total_retrieves << " (avg "
              << stats.avg_retrieve_us << " us)\n";
    std::cout << "    Consolidations: " << stats.total_consolidations
              << " (last " << stats.last_consolidation_us << " us)\n";
}

int main() {
    std::cout << "==============================================\n";
    std::cout << "  Neural Memory Adapter - Memory Benchmark\n";
    std::cout << "==============================================\n";
    
    // Initialize adapter (no DB connection needed for in-memory benchmark)
    NeuralMemoryAdapter adapter;
    AdapterConfig config = AdapterConfig::balanced();
    config.enable_consolidation_thread = false;
    config.enable_decay_thread = false;
    config.enable_link_prediction = false;
    
    if (!adapter.initialize(config)) {
        std::cerr << "Failed to initialize adapter\n";
        return 1;
    }
    
    std::cout << "\nAdapter initialized (768-dim, in-memory only)\n";
    
    // Benchmark different workloads
    bench_store(adapter, 1000, 768);
    bench_store(adapter, 10000, 768);
    bench_retrieve(adapter, 100, 768);
    bench_retrieve(adapter, 1000, 768);
    bench_consolidation(adapter);
    bench_spreading_activation(adapter, 100);
    bench_link_prediction(adapter);
    
    print_stats(adapter.get_stats());
    
    adapter.shutdown();
    std::cout << "\nDone.\n";
    return 0;
}
