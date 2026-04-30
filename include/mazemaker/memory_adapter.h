#pragma once
// neural/memory_adapter.h - THE Public API
// One header to rule them all

#include "mazemaker/vector.h"
#include "mazemaker/simd.h"
#include "mazemaker/hopfield.h"
#include "mazemaker/memory.h"
#include "mazemaker/graph.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <optional>
#include <chrono>
#include <thread>
#include <atomic>
#include <shared_mutex>

namespace neural {

// ============================================================================
// Configuration
// ============================================================================

struct AdapterConfig {
    // --- Vector dimensions ---
    size_t vector_dim = 768;
    
    // --- Hopfield Memory ---
    float hopfield_beta = 20.0f;
    size_t hopfield_capacity = 1024;
    float hopfield_decay = 0.999f;
    
    // --- Memory Tiers ---
    size_t episodic_capacity = 10000;
    size_t semantic_capacity = 100000;
    float consolidation_threshold = 0.8f;  // Trigger at this occupancy
    float consolidation_merge_threshold = 0.85f;
    
    // --- Graph ---
    float graph_decay = 0.85f;
    float graph_threshold = 0.01f;
    int graph_max_depth = 5;
    size_t link_prediction_interval_sec = 300;  // Every 5 min
    float edge_prune_threshold = 0.01f;
    
    // --- Background Threads ---
    bool enable_consolidation_thread = true;
    bool enable_decay_thread = true;
    bool enable_link_prediction = true;
    size_t consolidation_interval_sec = 60;
    size_t decay_interval_sec = 300;
    
    // --- Performance ---
    size_t search_batch_size = 1000;
    bool enable_simd = true;
    bool enable_openmp = true;
    
    // --- Presets ---
    static AdapterConfig fast() {
        AdapterConfig cfg;
        cfg.vector_dim = 512;
        cfg.hopfield_capacity = 512;
        cfg.episodic_capacity = 5000;
        cfg.consolidation_interval_sec = 120;
        cfg.enable_link_prediction = false;
        return cfg;
    }
    
    static AdapterConfig balanced() {
        return AdapterConfig{};  // defaults
    }
    
    static AdapterConfig accurate() {
        AdapterConfig cfg;
        cfg.hopfield_capacity = 2048;
        cfg.episodic_capacity = 50000;
        cfg.semantic_capacity = 500000;
        cfg.graph_max_depth = 8;
        cfg.consolidation_interval_sec = 30;
        cfg.link_prediction_interval_sec = 120;
        return cfg;
    }
};

// ============================================================================
// Memory Result
// ============================================================================

struct MemoryResult {
    uint64_t id;
    std::vector<float> embedding;
    std::string label;
    std::string content;
    float similarity;
    float salience;
    std::vector<uint64_t> connected_ids;
};

struct StatsSnapshot {
    // Memory
    size_t episodic_count;
    size_t semantic_count;
    float episodic_occupancy;
    float semantic_occupancy;
    
    // Hopfield
    size_t hopfield_patterns;
    float hopfield_occupancy;
    
    // Graph
    size_t graph_nodes;
    size_t graph_edges;
    float graph_density;
    
    // Performance (microseconds)
    uint64_t avg_store_us;
    uint64_t avg_retrieve_us;
    uint64_t avg_search_us;
    uint64_t last_consolidation_us;
    
    // Operations
    uint64_t total_stores;
    uint64_t total_retrieves;
    uint64_t total_searches;
    uint64_t total_consolidations;
};

// ============================================================================
// Neural Memory Adapter
// ============================================================================

class NeuralMemoryAdapter {
public:
    NeuralMemoryAdapter() = default;
    ~NeuralMemoryAdapter();
    
    // Non-copyable, non-movable
    NeuralMemoryAdapter(const NeuralMemoryAdapter&) = delete;
    NeuralMemoryAdapter& operator=(const NeuralMemoryAdapter&) = delete;
    
    // --- Lifecycle ---
    
    bool initialize(const AdapterConfig& config);
    void shutdown();
    bool is_initialized() const { return initialized_; }
    
    // --- Core Operations ---
    
    // Store a memory (embedding + metadata)
    // Returns memory ID
    uint64_t store(const std::vector<float>& embedding,
                   const std::string& label = "",
                   const std::string& content = "",
                   const std::string& source = "user");
    
    // Store with automatic embedding (requires text content)
    // Uses simple TF-IDF-like encoding if no model available
    uint64_t store_text(const std::string& text,
                        const std::string& label = "");
    
    // Retrieve memories by partial cue (pattern completion)
    std::vector<MemoryResult> retrieve(const std::vector<float>& cue,
                                        size_t k = 10) const;
    
    // Retrieve by text query
    std::vector<MemoryResult> retrieve_text(const std::string& query,
                                             size_t k = 10) const;
    
    // Full-text search across stored content
    std::vector<MemoryResult> search(const std::string& query,
                                      size_t k = 10) const;
    
    // Read a specific memory by ID
    std::optional<MemoryResult> read(uint64_t id) const;
    
    // --- Graph Operations ---
    
    // Find connections for a memory
    std::vector<std::pair<uint64_t, float>> connections(uint64_t id) const;
    
    // Spreading activation from a memory
    std::vector<graph::TraversalResult> think(uint64_t start_id,
                                               int depth = 3) const;
    
    // Find shortest path between two memories
    std::optional<std::vector<uint64_t>> path(uint64_t from, uint64_t to) const;
    
    // Get predicted links
    std::vector<graph::ConnectionPrediction> predicted_connections(
        size_t max_results = 20) const;
    
    // --- Consolidation ---
    
    // Force consolidation run
    size_t consolidate();
    
    // Force edge decay
    void decay();
    
    // Force link prediction
    size_t predict_links();
    
    // --- Statistics ---
    
    StatsSnapshot get_stats() const;
    
    // --- Configuration ---
    
    void set_beta(float beta);
    float get_beta() const;
    
    void set_consolidation_threshold(float threshold);
    
    const AdapterConfig& config() const { return config_; }

    // --- Graph Edge Operations (in-memory graph) ---

    // Add edge. Returns true on success.
    bool add_edge(uint64_t from_id, uint64_t to_id, float weight,
                  const std::string& edge_type = "similar");

    // Batch add edges. Returns count added.
    int batch_add_edges(const std::vector<uint64_t>& from_ids,
                        const std::vector<uint64_t>& to_ids,
                        const std::vector<float>& weights,
                        const std::string& edge_type = "similar");

    // Batch strengthen edges by delta (clamped to 1.0).
    int batch_strengthen_edges(const std::vector<uint64_t>& from_ids,
                               const std::vector<uint64_t>& to_ids,
                               float delta);

    // Bulk weaken edges by delta and prune those below threshold.
    int bulk_weaken_prune(float delta, float threshold);

    // Get all edges for a node
    struct EdgeInfo {
        uint64_t from_id;
        uint64_t to_id;
        float weight;
    };
    std::vector<EdgeInfo> get_edges(uint64_t node_id) const;

    // Count edges
    int64_t count_edges() const;

private:
    AdapterConfig config_;
    bool initialized_ = false;
    
    // Components
    std::unique_ptr<HopfieldLayer> hopfield_;
    std::unique_ptr<memory::MemoryManager> memory_manager_;
    std::unique_ptr<graph::KnowledgeGraph> graph_;

    // Background threads
    std::thread consolidation_thread_;
    std::thread decay_thread_;
    std::thread link_prediction_thread_;
    std::atomic<bool> running_{false};
    
    // Performance tracking
    mutable std::shared_mutex stats_mutex_;
    mutable uint64_t total_stores_ = 0;
    mutable uint64_t total_retrieves_ = 0;
    mutable uint64_t total_searches_ = 0;
    mutable uint64_t total_consolidations_ = 0;
    mutable uint64_t total_store_us_ = 0;
    mutable uint64_t total_retrieve_us_ = 0;
    mutable uint64_t total_search_us_ = 0;
    mutable uint64_t last_consolidation_us_ = 0;
    
    // Background worker functions
    void consolidation_worker();
    void decay_worker();
    void link_prediction_worker();
    
    // Helpers
    std::vector<float> text_to_embedding(const std::string& text) const;
    MemoryResult to_result(const memory::MemoryEntry& entry, float similarity = 1.0f) const;
};

} // namespace neural
