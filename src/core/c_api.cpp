// neural/core/c_api.cpp - C-compatible API implementation
// Wraps NeuralMemoryAdapter for use via ctypes / FFI.
#include "mazemaker/c_api.h"
#include "mazemaker/memory_adapter.h"
#include "mazemaker/lstm.h"
#include "mazemaker/knn.h"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <new>
#include <vector>

using namespace neural;

// Helper: convert handle to typed pointer
static inline NeuralMemoryAdapter* to_adapter(MazemakerHandle h) {
    return static_cast<NeuralMemoryAdapter*>(h);
}

// ============================================================================
// Lifecycle
// ============================================================================

MAZEMAKER_API MazemakerHandle mazemaker_create(void) {
    return mazemaker_create_dim(1024);
}

MAZEMAKER_API MazemakerHandle mazemaker_create_dim(int vector_dim) {
    if (vector_dim <= 0) vector_dim = 1024;

    auto* adapter = new (std::nothrow) NeuralMemoryAdapter();
    if (!adapter) return nullptr;

    AdapterConfig config;
    config.vector_dim = static_cast<size_t>(vector_dim);
    // Disable background threads for Python use (Python manages its own lifecycle)
    config.enable_consolidation_thread = false;
    config.enable_decay_thread = false;
    config.enable_link_prediction = false;
    // C API binds to in-memory adapter. Persistence is handled by Python
    // backends (SQLite or Postgres+pgvector).

    if (!adapter->initialize(config)) {
        delete adapter;
        return nullptr;
    }

    return static_cast<MazemakerHandle>(adapter);
}

MAZEMAKER_API void mazemaker_destroy(MazemakerHandle handle) {
    if (!handle) return;
    auto* adapter = to_adapter(handle);
    adapter->shutdown();
    delete adapter;
}

// ============================================================================
// Core operations
// ============================================================================

MAZEMAKER_API uint64_t mazemaker_store(
    MazemakerHandle handle,
    const float* vec,
    int dim,
    const char* label,
    const char* content
) {
    if (!handle || !vec || dim <= 0) return 0;
    auto* adapter = to_adapter(handle);

    std::vector<float> embedding(vec, vec + dim);
    std::string lbl = label ? label : "";
    std::string cnt = content ? content : "";

    return adapter->store(embedding, lbl, cnt, "api");
}

MAZEMAKER_API uint64_t mazemaker_store_text(
    MazemakerHandle handle,
    const char* text,
    const char* label
) {
    if (!handle || !text) return 0;
    auto* adapter = to_adapter(handle);

    std::string txt = text;
    std::string lbl = label ? label : "";

    return adapter->store_text(txt, lbl);
}

MAZEMAKER_API int mazemaker_retrieve(
    MazemakerHandle handle,
    const float* vec,
    int dim,
    int k,
    uint64_t* ids,
    float* scores
) {
    if (!handle || !vec || dim <= 0 || k <= 0 || !ids || !scores) return 0;
    auto* adapter = to_adapter(handle);

    std::vector<float> cue(vec, vec + dim);
    auto results = adapter->retrieve(cue, static_cast<size_t>(k));

    int count = 0;
    for (const auto& r : results) {
        if (count >= k) break;
        ids[count] = r.id;
        scores[count] = r.similarity;
        count++;
    }
    return count;
}

MAZEMAKER_API int mazemaker_retrieve_full(
    MazemakerHandle handle,
    const float* vec,
    int dim,
    int k,
    MazemakerResult* results
) {
    if (!handle || !vec || dim <= 0 || k <= 0 || !results) return 0;
    auto* adapter = to_adapter(handle);

    std::vector<float> cue(vec, vec + dim);
    auto mem_results = adapter->retrieve(cue, static_cast<size_t>(k));

    int count = 0;
    for (const auto& r : mem_results) {
        if (count >= k) break;
        auto& out = results[count];
        out.id = r.id;
        out.embedding = const_cast<float*>(r.embedding.data());
        out.embedding_dim = static_cast<int>(r.embedding.size());

        // Safe string copy
        std::strncpy(out.label, r.label.c_str(), sizeof(out.label) - 1);
        out.label[sizeof(out.label) - 1] = '\0';
        std::strncpy(out.content, r.content.c_str(), sizeof(out.content) - 1);
        out.content[sizeof(out.content) - 1] = '\0';

        out.similarity = r.similarity;
        out.salience = r.salience;
        count++;
    }
    return count;
}

MAZEMAKER_API int mazemaker_search(
    MazemakerHandle handle,
    const char* query,
    int k,
    uint64_t* ids,
    float* scores
) {
    if (!handle || !query || k <= 0 || !ids || !scores) return 0;
    auto* adapter = to_adapter(handle);

    std::string q = query;
    auto results = adapter->search(q, static_cast<size_t>(k));

    int count = 0;
    for (const auto& r : results) {
        if (count >= k) break;
        ids[count] = r.id;
        scores[count] = r.similarity;
        count++;
    }
    return count;
}

MAZEMAKER_API int mazemaker_read(
    MazemakerHandle handle,
    uint64_t id,
    MazemakerResult* result
) {
    if (!handle || !result) return 0;
    auto* adapter = to_adapter(handle);

    auto opt = adapter->read(id);
    if (!opt) return 0;

    const auto& r = *opt;
    result->id = r.id;
    result->embedding = const_cast<float*>(r.embedding.data());
    result->embedding_dim = static_cast<int>(r.embedding.size());

    std::strncpy(result->label, r.label.c_str(), sizeof(result->label) - 1);
    result->label[sizeof(result->label) - 1] = '\0';
    std::strncpy(result->content, r.content.c_str(), sizeof(result->content) - 1);
    result->content[sizeof(result->content) - 1] = '\0';

    result->similarity = r.similarity;
    result->salience = r.salience;
    return 1;
}

// ============================================================================
// Graph / Spreading Activation
// ============================================================================

MAZEMAKER_API int mazemaker_think_c(
    MazemakerHandle handle,
    uint64_t start_id,
    int depth,
    uint64_t* node_ids,
    float* activations,
    int max_results
) {
    if (!handle || !node_ids || !activations || max_results <= 0) return 0;
    auto* adapter = to_adapter(handle);

    auto traversal = adapter->think(start_id, depth);

    int count = 0;
    for (const auto& tr : traversal) {
        if (count >= max_results) break;
        // Convert node_id back to memory_id (subtract 1 offset)
        node_ids[count] = tr.node_id > 0 ? tr.node_id - 1 : tr.node_id;
        activations[count] = tr.activation;
        count++;
    }
    return count;
}

// ============================================================================
// Consolidation & Decay
// ============================================================================

MAZEMAKER_API size_t mazemaker_consolidate(MazemakerHandle handle) {
    if (!handle) return 0;
    return to_adapter(handle)->consolidate();
}

MAZEMAKER_API void mazemaker_decay(MazemakerHandle handle) {
    if (!handle) return;
    to_adapter(handle)->decay();
}

MAZEMAKER_API size_t mazemaker_predict_links(MazemakerHandle handle) {
    if (!handle) return 0;
    return to_adapter(handle)->predict_links();
}

// ============================================================================
// Configuration
// ============================================================================

MAZEMAKER_API void mazemaker_set_beta(MazemakerHandle handle, float beta) {
    if (!handle) return;
    to_adapter(handle)->set_beta(beta);
}

MAZEMAKER_API float mazemaker_get_beta(MazemakerHandle handle) {
    if (!handle) return 0.0f;
    return to_adapter(handle)->get_beta();
}

MAZEMAKER_API void mazemaker_set_consolidation_threshold(MazemakerHandle handle, float threshold) {
    if (!handle) return;
    to_adapter(handle)->set_consolidation_threshold(threshold);
}

// ============================================================================
// Statistics
// ============================================================================

MAZEMAKER_API void mazemaker_stats_c(MazemakerHandle handle, MazemakerStats* stats) {
    if (!handle || !stats) return;
    auto* adapter = to_adapter(handle);

    auto s = adapter->get_stats();

    stats->episodic_count = s.episodic_count;
    stats->semantic_count = s.semantic_count;
    stats->episodic_occupancy = s.episodic_occupancy;
    stats->semantic_occupancy = s.semantic_occupancy;
    stats->hopfield_patterns = s.hopfield_patterns;
    stats->hopfield_occupancy = s.hopfield_occupancy;
    stats->graph_nodes = s.graph_nodes;
    stats->graph_edges = s.graph_edges;
    stats->graph_density = s.graph_density;
    stats->avg_store_us = s.avg_store_us;
    stats->avg_retrieve_us = s.avg_retrieve_us;
    stats->avg_search_us = s.avg_search_us;
    stats->total_stores = s.total_stores;
    stats->total_retrieves = s.total_retrieves;
    stats->total_searches = s.total_searches;
    stats->total_consolidations = s.total_consolidations;
}

// ============================================================================
// Graph / Edge Operations (in-memory adapter)
// ============================================================================
// Edge persistence is handled by the Python layer (SQLite or Postgres). The
// C API exposes the in-memory graph operations only.

MAZEMAKER_API int mazemaker_add_edge(
    MazemakerHandle handle,
    uint64_t from_id,
    uint64_t to_id,
    float weight,
    const char* edge_type
) {
    if (!handle) return 0;
    auto* adapter = to_adapter(handle);

    std::string etype = edge_type ? edge_type : "similar";
    return adapter->add_edge(from_id, to_id, weight, etype) ? 1 : 0;
}

MAZEMAKER_API int mazemaker_batch_add_edges(
    MazemakerHandle handle,
    const uint64_t* from_ids,
    const uint64_t* to_ids,
    const float* weights,
    int count,
    const char* edge_type
) {
    if (!handle || !from_ids || !to_ids || !weights || count <= 0) return 0;
    auto* adapter = to_adapter(handle);

    std::vector<uint64_t> fids(from_ids, from_ids + count);
    std::vector<uint64_t> tids(to_ids, to_ids + count);
    std::vector<float> wts(weights, weights + count);
    std::string etype = edge_type ? edge_type : "similar";

    return adapter->batch_add_edges(fids, tids, wts, etype);
}

MAZEMAKER_API int mazemaker_batch_strengthen_edges(
    MazemakerHandle handle,
    const uint64_t* from_ids,
    const uint64_t* to_ids,
    int count,
    float delta
) {
    if (!handle || !from_ids || !to_ids || count <= 0) return 0;
    auto* adapter = to_adapter(handle);

    std::vector<uint64_t> fids(from_ids, from_ids + count);
    std::vector<uint64_t> tids(to_ids, to_ids + count);

    return adapter->batch_strengthen_edges(fids, tids, delta);
}

MAZEMAKER_API int mazemaker_bulk_weaken_prune(
    MazemakerHandle handle,
    float delta,
    float threshold
) {
    if (!handle) return 0;
    return to_adapter(handle)->bulk_weaken_prune(delta, threshold);
}

MAZEMAKER_API int mazemaker_get_edges(
    MazemakerHandle handle,
    uint64_t node_id,
    uint64_t* edge_ids,
    float* weights,
    int max_edges
) {
    if (!handle || !edge_ids || !weights || max_edges <= 0) return 0;
    auto* adapter = to_adapter(handle);

    auto edges = adapter->get_edges(node_id);
    int count = 0;
    for (const auto& e : edges) {
        if (count >= max_edges) break;
        edge_ids[2 * count] = e.from_id;
        edge_ids[2 * count + 1] = e.to_id;
        weights[count] = e.weight;
        count++;
    }
    return count;
}

MAZEMAKER_API int64_t mazemaker_count_edges(MazemakerHandle handle) {
    if (!handle) return 0;
    return to_adapter(handle)->count_edges();
}

// ============================================================================
// LSTM Predictor C API
// ============================================================================

static inline neural::lstm::LSTMPredictor* to_lstm(LSTMPredictorHandle h) {
    return static_cast<neural::lstm::LSTMPredictor*>(h);
}

MAZEMAKER_API LSTMPredictorHandle nm_lstm_create(int input_dim, int hidden_dim) {
    if (input_dim <= 0 || hidden_dim <= 0) return nullptr;
    try {
        auto* lstm = new (std::nothrow) neural::lstm::LSTMPredictor(
            static_cast<size_t>(input_dim),
            static_cast<size_t>(hidden_dim),
            static_cast<size_t>(input_dim));
        return static_cast<LSTMPredictorHandle>(lstm);
    } catch (...) {
        return nullptr;
    }
}

// Helper: flatten C array into vector<vector<float>>
static std::vector<std::vector<float>> flatten_sequence(const float* seq, int seq_len, int dim) {
    std::vector<std::vector<float>> result;
    result.reserve(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        result.emplace_back(seq + i * dim, seq + (i + 1) * dim);
    }
    return result;
}

MAZEMAKER_API int nm_lstm_forward(LSTMPredictorHandle handle,
                                const float* sequence, int seq_len,
                                float* output) {
    if (!handle || !sequence || seq_len <= 0 || !output) return -1;
    auto* lstm = to_lstm(handle);
    int d = lstm->input_dim();
    try {
        auto seq_vec = flatten_sequence(sequence, seq_len, d);
        auto result = lstm->forward(seq_vec);
        size_t out_dim = lstm->output_dim();
        std::memcpy(output, result.data(), out_dim * sizeof(float));
        return 0;
    } catch (...) {
        return -1;
    }
}

MAZEMAKER_API float nm_lstm_train(LSTMPredictorHandle handle,
                                const float* sequence, int seq_len,
                                const float* target, float lr) {
    if (!handle || !sequence || seq_len <= 0 || !target) return -1.0f;
    auto* lstm = to_lstm(handle);
    int d = lstm->input_dim();
    try {
        auto seq_vec = flatten_sequence(sequence, seq_len, d);
        std::vector<float> tgt_vec(target, target + d);
        return lstm->train_step(seq_vec, tgt_vec);
    } catch (...) {
        return -1.0f;
    }
}

MAZEMAKER_API int nm_lstm_save(LSTMPredictorHandle handle, const char* path) {
    if (!handle || !path) return -1;
    auto* lstm = to_lstm(handle);
    try {
        lstm->save(std::string(path));
        return 0;
    } catch (...) {
        return -1;
    }
}

MAZEMAKER_API LSTMPredictorHandle nm_lstm_load(const char* path, int input_dim, int hidden_dim) {
    if (!path || input_dim <= 0 || hidden_dim <= 0) return nullptr;
    try {
        auto* lstm = new (std::nothrow) neural::lstm::LSTMPredictor(
            static_cast<size_t>(input_dim),
            static_cast<size_t>(hidden_dim),
            static_cast<size_t>(input_dim));
        if (!lstm) return nullptr;
        try {
            lstm->load(std::string(path));
        } catch (...) {
            delete lstm;
            return nullptr;
        }
        return static_cast<LSTMPredictorHandle>(lstm);
    } catch (...) {
        return nullptr;
    }
}

MAZEMAKER_API void nm_lstm_destroy(LSTMPredictorHandle handle) {
    if (!handle) return;
    delete to_lstm(handle);
}

// ============================================================================
// kNN Engine C API
// ============================================================================

static inline neural::knn::KNNEngine* to_knn(KNNEngineHandle h) {
    return static_cast<neural::knn::KNNEngine*>(h);
}

MAZEMAKER_API KNNEngineHandle nm_knn_create(int embed_dim) {
    if (embed_dim <= 0) return nullptr;
    try {
        auto* knn = new (std::nothrow) neural::knn::KNNEngine(
            static_cast<size_t>(embed_dim));
        return static_cast<KNNEngineHandle>(knn);
    } catch (...) {
        return nullptr;
    }
}

MAZEMAKER_API int nm_knn_search(KNNEngineHandle handle,
                              const float* query, int embed_dim,
                              const float* candidates,
                              const uint64_t* candidate_ids,
                              int count, int k,
                              const float* timestamps,
                              const float* access_counts,
                              const float* graph_scores,
                              const float* lstm_context,
                              KNNCResult* results) {
    if (!handle || !query || embed_dim <= 0 || !candidates ||
        !candidate_ids || count <= 0 || k <= 0 ||
        !timestamps || !access_counts || !graph_scores || !results) {
        return -1;
    }
    auto* knn = to_knn(handle);
    try {
        // Build MemoryCandidate vector from flat arrays
        std::vector<neural::knn::MemoryCandidate> cands;
        cands.reserve(count);
        for (int i = 0; i < count; ++i) {
            neural::knn::MemoryCandidate mc;
            mc.id = candidate_ids[i];
            mc.embedding = candidates + i * embed_dim;
            mc.created_us = static_cast<uint64_t>(timestamps[i] * 1e6f);     // seconds -> microseconds
            mc.last_accessed_us = static_cast<uint64_t>(timestamps[i] * 1e6f);
            mc.access_count = static_cast<uint64_t>(access_counts[i]);
            mc.graph_distance = graph_scores[i];
            cands.push_back(mc);
        }

        auto res = knn->search(query, cands, static_cast<size_t>(k));
        int n = std::min(static_cast<int>(res.size()), k);
        for (int i = 0; i < n; ++i) {
            results[i].id = res[i].id;
            results[i].score = res[i].total_score;
            results[i].embed_similarity = res[i].embedding_score;
            results[i].temporal_score = res[i].temporal_score;
            results[i].freq_score = res[i].frequency_score;
            results[i].graph_score = res[i].graph_score;
        }
        return n;
    } catch (...) {
        return -1;
    }
}

MAZEMAKER_API void nm_knn_adjust_weights(KNNEngineHandle handle,
                                       const float* lstm_context) {
    if (!handle) return;
    to_knn(handle)->adjust_weights(lstm_context);
}

MAZEMAKER_API void nm_knn_destroy(KNNEngineHandle handle) {
    if (!handle) return;
    delete to_knn(handle);
}
