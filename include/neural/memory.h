// include/neural/memory.h - 3-Tier Memory Interface
#pragma once

#include <vector>
#include <string>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <optional>
#include <atomic>
#include "hopfield.h"

namespace neural::memory {

struct MemoryEntry {
    uint64_t id = 0;
    std::vector<float> embedding;
    std::string label;
    std::string content;            // Raw content / description
    std::string source;             // "perception", "inference", "consolidated"
    uint64_t timestamp = 0;         // Creation time (us)
    uint64_t last_accessed = 0;     // Last access time
    uint64_t access_count = 0;      // Total retrievals
    float salience = 1.0f;          // Importance score
    float decay_factor = 1.0f;      // Exponential decay
    std::vector<uint64_t> linked;   // Connected memory IDs

    // Age in seconds
    double age_seconds(uint64_t now_us) const;
    // Time since last access
    double recency_seconds(uint64_t now_us) const;
};

// ============================================================
// EpisodicMemory - Fast-write FIFO buffer
// ============================================================
class EpisodicMemory {
public:
    explicit EpisodicMemory(size_t capacity = 10000, size_t dimensions = 512);

    // Write new memory (returns assigned ID)
    uint64_t write(const std::vector<float>& embedding, const std::string& label = "",
                   const std::string& content = "");

    // Read by ID
    const MemoryEntry* read(uint64_t id) const;

    // Search by similarity (cosine)
    std::vector<std::pair<uint64_t, float>> search(const std::vector<float>& query, size_t k = 10) const;

    // Get entries for consolidation (sorted by access frequency, then age)
    std::vector<const MemoryEntry*> candidates_for_consolidation(size_t max_count) const;

    // Remove oldest entry (FIFO eviction)
    std::optional<MemoryEntry> evict_oldest();

    // Remove by ID
    bool remove(uint64_t id);

    // Touch (update access metadata)
    void touch(uint64_t id);
    
    // Set the next ID sequence (used to sync with external databases)
    void set_next_id(uint64_t next_id) { next_id_.store(next_id); }

    size_t size() const { return entries_.size(); }
    size_t capacity() const { return capacity_; }
    bool is_full() const { return entries_.size() >= capacity_; }
    float occupancy() const { return static_cast<float>(entries_.size()) / capacity_; }

private:
    std::optional<MemoryEntry> evict_oldest_internal();    // Internal (no lock)

    size_t capacity_;
    size_t dimensions_;
    std::deque<MemoryEntry> entries_;                      // FIFO ordering
    std::unordered_map<uint64_t, size_t> id_to_index_;     // ID -> deque index
    std::atomic<uint64_t> next_id_{1};
    mutable std::shared_mutex mutex_;
};

// ============================================================
// SemanticMemory - Consolidated, clustered long-term storage
// ============================================================
struct Cluster {
    uint64_t id = 0;
    std::vector<float> centroid;
    std::vector<uint64_t> member_ids;
    float coherence = 0.0f;         // Avg intra-cluster similarity
    uint64_t created = 0;
    uint64_t last_updated = 0;
};

class SemanticMemory {
public:
    explicit SemanticMemory(size_t dimensions = 512, size_t max_clusters = 256);

    // Store consolidated memory
    uint64_t store(const MemoryEntry& entry);

    // Read by ID
    const MemoryEntry* read(uint64_t id) const;

    // Search by similarity
    std::vector<std::pair<uint64_t, float>> search(const std::vector<float>& query, size_t k = 10) const;

    // Cluster management
    void rebuild_clusters();
    const std::vector<Cluster>& clusters() const { return clusters_; }

    // Merge similar memories
    uint64_t merge(uint64_t id_a, uint64_t id_b);

    // Decay all salience values
    void decay_all(float decay_factor = 0.999f);

    size_t size() const { return entries_.size(); }

private:
    size_t dimensions_;
    size_t max_clusters_;
    std::unordered_map<uint64_t, MemoryEntry> entries_;
    std::vector<Cluster> clusters_;
    std::atomic<uint64_t> next_id_{1000000};  // Offset from episodic IDs
    mutable std::shared_mutex mutex_;

    // Assign entry to nearest cluster (or create new)
    void assign_to_cluster(uint64_t entry_id, const std::vector<float>& embedding);
    // K-means-like update for a cluster
    void update_cluster_centroid(Cluster& cluster);
};

// ============================================================
// Consolidation Event - record of a consolidation action
// ============================================================
struct ConsolidationEvent {
    uint64_t timestamp = 0;
    std::vector<uint64_t> episodic_ids;     // Source episodic memories
    uint64_t semantic_id = 0;               // Resulting semantic memory
    float strength = 0.0f;                  // Connection strength
    std::string operation;                  // "transfer", "merge", "cluster"
};

// ============================================================
// MemoryManager - Coordinates episodic and semantic memory
// ============================================================
class MemoryManager {
public:
    explicit MemoryManager(size_t dimensions = 512,
                           size_t episodic_capacity = 10000,
                           size_t semantic_max_clusters = 256);

    // --- Unified interface ---

    // Store new experience
    uint64_t remember(const std::vector<float>& embedding, const std::string& label = "",
                      const std::string& content = "");

    // Recall - searches both tiers
    std::vector<std::pair<uint64_t, float>> recall(const std::vector<float>& query, size_t k = 10) const;

    // Read a memory (auto-detects tier)
    const MemoryEntry* read(uint64_t id) const;
    
    // Sync ID counter with database
    void set_next_id(uint64_t id) { episodic_.set_next_id(id); }

    // --- Consolidation ---

    // Run one consolidation pass
    size_t consolidate(size_t batch_size = 64);

    // Set consolidation callback (called when new semantic memory is created)
    using ConsolidationCallback = std::function<void(const ConsolidationEvent&)>;
    void on_consolidation(ConsolidationCallback cb) { consolidation_cb_ = cb; }

    // Auto-consolidation: trigger when episodic memory occupancy exceeds threshold
    void set_auto_consolidate_threshold(float threshold) { auto_consolidate_threshold_ = threshold; }

    // --- Connection tracking ---

    // Get connection strength between two memories
    float connection_strength(uint64_t id_a, uint64_t id_b) const;

    // Get all connections for a memory
    std::vector<std::pair<uint64_t, float>> connections(uint64_t id) const;

    // --- Decay ---

    // Apply exponential forgetting decay
    void apply_decay(float decay_factor = 0.999f);

    // --- Queries ---

    size_t episodic_size() const { return episodic_.size(); }
    size_t semantic_size() const { return semantic_.size(); }
    float episodic_occupancy() const { return episodic_.occupancy(); }

    // Access the underlying Hopfield layer
    HopfieldLayer& hopfield() { return hopfield_; }
    const HopfieldLayer& hopfield() const { return hopfield_; }

private:
    EpisodicMemory episodic_;
    SemanticMemory semantic_;
    HopfieldLayer hopfield_;

    // Connection graph: (id_a, id_b) -> strength
    std::unordered_map<uint64_t,
        std::unordered_map<uint64_t, float>> connections_;
    mutable std::shared_mutex conn_mutex_;

    // Consolidation state
    ConsolidationCallback consolidation_cb_;
    float auto_consolidate_threshold_ = 0.8f;
    uint64_t last_consolidation_time_ = 0;

    size_t dimensions_;

    // Auto-consolidate if needed
    void maybe_auto_consolidate();
};

// Utility: current time in microseconds
uint64_t now_us();

} // namespace neural::memory
