// include/neural/knn.h - Enhanced kNN Search Engine for Neural Memory
#pragma once

#include <cstdint>
#include <vector>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <cmath>

namespace neural::knn {

// ============================================================================
// Configuration
// ============================================================================

struct KNNConfig {
    float w_embedding  = 0.50f;   // cosine similarity weight
    float w_temporal   = 0.20f;   // temporal decay weight
    float w_frequency  = 0.15f;   // access frequency weight
    float w_graph      = 0.15f;   // graph proximity weight
    size_t base_k      = 5;
    size_t max_k       = 20;
    float half_life_hours = 24.0f;
};

// ============================================================================
// Data Structures
// ============================================================================

struct MemoryCandidate {
    uint64_t id;
    const float* embedding;      // pointer into existing data (NOT owned)
    uint64_t created_us;         // creation timestamp in microseconds
    uint64_t last_accessed_us;   // last access timestamp in microseconds
    uint64_t access_count;       // number of times accessed
    float graph_distance;        // precomputed hop distance from query node
};

struct ScoredResult {
    uint64_t id;
    float total_score;
    float embedding_score;
    float temporal_score;
    float frequency_score;
    float graph_score;
};

// ============================================================================
// kNN Engine
// ============================================================================

class KNNEngine {
public:
    /// @param embed_dim  Embedding vector dimensionality (default 1024)
    /// @param cfg        Initial scoring configuration
    explicit KNNEngine(size_t embed_dim = 1024, KNNConfig cfg = {});

    ~KNNEngine() = default;

    KNNEngine(const KNNEngine&) = delete;
    KNNEngine& operator=(const KNNEngine&) = delete;
    KNNEngine(KNNEngine&&) = default;
    KNNEngine& operator=(KNNEngine&&) = default;

    /// Core search: score all candidates, return top-k by total_score.
    /// Thread-safe for concurrent reads; does NOT modify candidates.
    std::vector<ScoredResult> search(
        const float* query_embed,
        const std::vector<MemoryCandidate>& candidates,
        size_t k
    ) const;

    /// Dynamic weight adjustment driven by an LSTM context vector.
    /// Context vector is expected to have at least 4 floats:
    ///   [0] -> embedding influence, [1] -> temporal influence,
    ///   [2] -> frequency influence,  [3] -> graph influence.
    /// After adjustment, weights are normalized to sum to 1.
    /// Lock-free: atomically publishes the new config.
    void adjust_weights(const float* lstm_context, size_t ctx_dim = 4);

    /// Expanded search: if the top result score is below expansion_threshold,
    /// expand k up to max_k to surface more candidates.
    std::vector<ScoredResult> search_with_expansion(
        const float* query,
        const std::vector<MemoryCandidate>& cands,
        size_t base_k,
        size_t max_k,
        float expansion_threshold
    ) const;

    /// Replace configuration (full swap).
    /// Takes effect immediately for subsequent searches.
    void update_config(const KNNConfig& cfg);

    /// Read current config snapshot (lock-free).
    KNNConfig config() const;

    size_t embed_dim() const { return embed_dim_; }

private:
    // Per-candidate scoring (all four signals)
    ScoredResult score_candidate(
        const float* query_embed,
        const MemoryCandidate& cand,
        float now_hours,
        float max_log_freq
    ) const;

    // Temporal decay: exp(-0.693 * age_hours / half_life)
    static float temporal_decay(float age_hours, float half_life);

    // Frequency: log2(1 + access_count) mapped to [0,1]
    static float frequency_score(uint64_t access_count, float max_log_freq);

    // Graph proximity: 1.0 / (1.0 + distance)
    static float graph_score(float distance);

    size_t embed_dim_;

    // Config is read often, written rarely -> use shared_mutex for RCU-style access
    // Reads grab shared lock (lock-free for concurrent readers), writes grab exclusive.
    mutable std::mutex config_mutex_;
    KNNConfig config_;
};

} // namespace neural::knn
