// src/memory/knn.cpp - Enhanced kNN Search Engine implementation
#include "neural/knn.h"
#include "neural/simd.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <cassert>

namespace neural::knn {

// ============================================================================
// Helpers
// ============================================================================

namespace {

// Convert microseconds-since-epoch to fractional hours.
inline float us_to_hours(uint64_t us) {
    return static_cast<float>(us) / (1000.0f * 1000.0f * 3600.0f);
}

// Normalized frequency: log2(1+x) divided by max_log_freq, clamped to [0,1].
inline float normalized_log_freq(uint64_t count, float max_log_freq) {
    if (max_log_freq < 1e-10f) return 0.0f;
    float v = std::log2(1.0f + static_cast<float>(count)) / max_log_freq;
    return std::clamp(v, 0.0f, 1.0f);
}

} // anonymous namespace

// ============================================================================
// Constructor
// ============================================================================

KNNEngine::KNNEngine(size_t embed_dim, KNNConfig cfg)
    : embed_dim_(embed_dim)
    , config_(cfg)
{
    assert(embed_dim_ > 0 && "Embedding dimension must be > 0");
}

// ============================================================================
// Temporal decay
// ============================================================================

float KNNEngine::temporal_decay(float age_hours, float half_life) {
    if (half_life < 1e-10f) return 0.0f;
    // exp(-ln2 * age / half_life) == 2^(-age/half_life)
    return std::exp(-0.69314718f * age_hours / half_life);
}

// ============================================================================
// Frequency score
// ============================================================================

float KNNEngine::frequency_score(uint64_t access_count, float max_log_freq) {
    return normalized_log_freq(access_count, max_log_freq);
}

// ============================================================================
// Graph score
// ============================================================================

float KNNEngine::graph_score(float distance) {
    return 1.0f / (1.0f + distance);
}

// ============================================================================
// Score a single candidate across all signals
// ============================================================================

ScoredResult KNNEngine::score_candidate(
    const float* query_embed,
    const MemoryCandidate& cand,
    float now_hours,
    float max_log_freq
) const {
    // We hold shared lock on config_mutex_ (caller ensures)
    const float w_emb   = config_.w_embedding;
    const float w_temp  = config_.w_temporal;
    const float w_freq  = config_.w_frequency;
    const float w_graph = config_.w_graph;
    const float half_life = config_.half_life_hours;

    ScoredResult r{};
    r.id = cand.id;

    // 1. Embedding cosine similarity (SIMD-accelerated, scalar fallback)
    r.embedding_score = simd::cosine_similarity(query_embed, cand.embedding, embed_dim_);

    // 2. Temporal decay
    float age_hours = now_hours - us_to_hours(cand.last_accessed_us);
    r.temporal_score = temporal_decay(age_hours, half_life);

    // 3. Frequency
    r.frequency_score = frequency_score(cand.access_count, max_log_freq);

    // 4. Graph proximity
    r.graph_score = graph_score(cand.graph_distance);

    // Weighted combination
    r.total_score = w_emb   * r.embedding_score
                  + w_temp  * r.temporal_score
                  + w_freq  * r.frequency_score
                  + w_graph * r.graph_score;

    return r;
}

// ============================================================================
// Core search
// ============================================================================

std::vector<ScoredResult> KNNEngine::search(
    const float* query_embed,
    const std::vector<MemoryCandidate>& candidates,
    size_t k
) const {
    if (candidates.empty() || k == 0) return {};

    const size_t n = candidates.size();
    const size_t eff_k = std::min(k, n);

    // Snapshot config under lock (concurrent writers via update_config/adjust_weights)
    KNNConfig cfg;
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        cfg = config_;
    }

    // Determine time reference: use max last_accessed as "now" for relative scoring.
    uint64_t max_ts = 0;
    for (const auto& c : candidates) {
        max_ts = std::max(max_ts, c.last_accessed_us);
    }
    float now_hours = us_to_hours(max_ts);

    // Determine max log-freq for normalization
    float max_lf = 0.0f;
    for (const auto& c : candidates) {
        max_lf = std::max(max_lf, std::log2(1.0f + static_cast<float>(c.access_count)));
    }
    if (max_lf < 1e-10f) max_lf = 1.0f;  // avoid division by zero

    // Score all candidates
    std::vector<ScoredResult> scored;
    scored.reserve(n);

    // Use batch_cosine_similarity for the embedding component
    std::vector<float> cos_sims(n);
    {
        // Flatten candidate embeddings into contiguous buffer for batch op.
        // NOTE: candidates may point to non-contiguous memory. We must copy.
        std::vector<float> flat_embeddings(n * embed_dim_);
        for (size_t i = 0; i < n; ++i) {
            std::copy(candidates[i].embedding,
                      candidates[i].embedding + embed_dim_,
                      flat_embeddings.data() + i * embed_dim_);
        }
        simd::batch_cosine_similarity(query_embed, flat_embeddings.data(),
                                       n, embed_dim_, cos_sims.data());
    }

    // Compute remaining signals and combine
    for (size_t i = 0; i < n; ++i) {
        const auto& cand = candidates[i];
        ScoredResult r{};
        r.id = cand.id;

        r.embedding_score = cos_sims[i];

        float age_hours = now_hours - us_to_hours(cand.last_accessed_us);
        r.temporal_score = temporal_decay(age_hours, cfg.half_life_hours);

        r.frequency_score = frequency_score(cand.access_count, max_lf);
        r.graph_score = graph_score(cand.graph_distance);

        r.total_score = cfg.w_embedding  * r.embedding_score
                      + cfg.w_temporal   * r.temporal_score
                      + cfg.w_frequency  * r.frequency_score
                      + cfg.w_graph      * r.graph_score;

        scored.push_back(r);
    }

    // Partial sort for top-k
    if (eff_k < n) {
        std::partial_sort(scored.begin(), scored.begin() + eff_k, scored.end(),
            [](const ScoredResult& a, const ScoredResult& b) {
                return a.total_score > b.total_score;  // descending
            });
        scored.resize(eff_k);
    } else {
        std::sort(scored.begin(), scored.end(),
            [](const ScoredResult& a, const ScoredResult& b) {
                return a.total_score > b.total_score;
            });
    }

    return scored;
}

// ============================================================================
// Dynamic weight adjustment
// ============================================================================

void KNNEngine::adjust_weights(const float* lstm_context, size_t ctx_dim) {
    if (!lstm_context || ctx_dim < 4) return;

    // Extract raw influences from LSTM context.
    // Use sigmoid to map arbitrary context values to (0,1).
    auto sigmoid = [](float x) -> float {
        return 1.0f / (1.0f + std::exp(-x));
    };

    float raw_emb   = sigmoid(lstm_context[0]);
    float raw_temp  = sigmoid(lstm_context[1]);
    float raw_freq  = sigmoid(lstm_context[2]);
    float raw_graph = sigmoid(lstm_context[3]);

    float total = raw_emb + raw_temp + raw_freq + raw_graph;
    if (total < 1e-10f) {
        // Fallback: reset to defaults
        std::lock_guard<std::mutex> lock(config_mutex_);
        config_.w_embedding = 0.50f;
        config_.w_temporal  = 0.20f;
        config_.w_frequency = 0.15f;
        config_.w_graph     = 0.15f;
        return;
    }

    float inv = 1.0f / total;

    std::lock_guard<std::mutex> lock(config_mutex_);
    config_.w_embedding  = raw_emb   * inv;
    config_.w_temporal   = raw_temp  * inv;
    config_.w_frequency  = raw_freq  * inv;
    config_.w_graph      = raw_graph * inv;
}

// ============================================================================
// Search with expansion
// ============================================================================

std::vector<ScoredResult> KNNEngine::search_with_expansion(
    const float* query,
    const std::vector<MemoryCandidate>& cands,
    size_t base_k,
    size_t max_k,
    float expansion_threshold
) const {
    auto results = search(query, cands, base_k);

    // If top result is below threshold, expand k to surface more candidates
    if (!results.empty() && results.front().total_score < expansion_threshold) {
        size_t expanded_k = std::min(max_k, cands.size());
        if (expanded_k > base_k) {
            results = search(query, cands, expanded_k);
        }
    }

    return results;
}

// ============================================================================
// Config access
// ============================================================================

void KNNEngine::update_config(const KNNConfig& cfg) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_ = cfg;
}

KNNConfig KNNEngine::config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

} // namespace neural::knn
