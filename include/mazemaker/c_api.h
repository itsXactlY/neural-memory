// neural/c_api.h - C-compatible API for Python/ctypes integration
// All functions use extern "C" linkage for ABI stability.
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
    #define MAZEMAKER_API __declspec(dllexport)
#else
    #define MAZEMAKER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Opaque handle
// ============================================================================
typedef void* MazemakerHandle;

// ============================================================================
// Result structures
// ============================================================================
typedef struct {
    uint64_t id;
    float*   embedding;      // Caller must NOT free (valid until next call)
    int      embedding_dim;
    char     label[256];
    char     content[4096];
    float    similarity;
    float    salience;
} MazemakerResult;

typedef struct {
    size_t   episodic_count;
    size_t   semantic_count;
    float    episodic_occupancy;
    float    semantic_occupancy;
    size_t   hopfield_patterns;
    float    hopfield_occupancy;
    size_t   graph_nodes;
    size_t   graph_edges;
    float    graph_density;
    uint64_t avg_store_us;
    uint64_t avg_retrieve_us;
    uint64_t avg_search_us;
    uint64_t total_stores;
    uint64_t total_retrieves;
    uint64_t total_searches;
    uint64_t total_consolidations;
} MazemakerStats;

// ============================================================================
// Lifecycle
// ============================================================================

// Create a new adapter with default config (1024-dim vectors).
// Returns NULL on failure.
MAZEMAKER_API MazemakerHandle mazemaker_create(void);

// Create with explicit vector dimension.
MAZEMAKER_API MazemakerHandle mazemaker_create_dim(int vector_dim);

// Destroy adapter and free resources.
MAZEMAKER_API void mazemaker_destroy(MazemakerHandle handle);

// ============================================================================
// Core operations
// ============================================================================

// Store a vector. Returns memory ID (0 on failure).
MAZEMAKER_API uint64_t mazemaker_store(
    MazemakerHandle handle,
    const float* vec,
    int dim,
    const char* label,
    const char* content
);

// Store text (uses internal text->embedding).
MAZEMAKER_API uint64_t mazemaker_store_text(
    MazemakerHandle handle,
    const char* text,
    const char* label
);

// Retrieve top-k memories by vector similarity.
// Writes up to k results into ids[] and scores[].
// Returns actual number of results written.
MAZEMAKER_API int mazemaker_retrieve(
    MazemakerHandle handle,
    const float* vec,
    int dim,
    int k,
    uint64_t* ids,
    float* scores
);

// Retrieve top-k with full result detail.
// results must point to an array of at least k MazemakerResult.
// Returns actual number of results written.
MAZEMAKER_API int mazemaker_retrieve_full(
    MazemakerHandle handle,
    const float* vec,
    int dim,
    int k,
    MazemakerResult* results
);

// Text-based search.
MAZEMAKER_API int mazemaker_search(
    MazemakerHandle handle,
    const char* query,
    int k,
    uint64_t* ids,
    float* scores
);

// Read a specific memory by ID. Returns 1 on success, 0 if not found.
MAZEMAKER_API int mazemaker_read(
    MazemakerHandle handle,
    uint64_t id,
    MazemakerResult* result
);

// ============================================================================
// Graph / Edge Operations (in-memory graph)
// ============================================================================

// Add an edge. Returns 1 on success.
MAZEMAKER_API int mazemaker_add_edge(
    MazemakerHandle handle,
    uint64_t from_id,
    uint64_t to_id,
    float weight,
    const char* edge_type
);

// Batch add edges. Returns number of edges added.
MAZEMAKER_API int mazemaker_batch_add_edges(
    MazemakerHandle handle,
    const uint64_t* from_ids,
    const uint64_t* to_ids,
    const float* weights,
    int count,
    const char* edge_type
);

// Batch strengthen edges: update weight = min(weight + delta, 1.0).
// from_ids/to_ids are parallel arrays of length count.
MAZEMAKER_API int mazemaker_batch_strengthen_edges(
    MazemakerHandle handle,
    const uint64_t* from_ids,
    const uint64_t* to_ids,
    int count,
    float delta
);

// Bulk weaken all edges by delta and prune those below threshold.
// Returns number of edges pruned.
MAZEMAKER_API int mazemaker_bulk_weaken_prune(
    MazemakerHandle handle,
    float delta,
    float threshold
);

// Get all edges for a node (from OR to). Writes up to max_edges into buffer.
// Returns actual number of edges written.
// edge_ids[2*i] = from_id, edge_ids[2*i+1] = to_id, weights[i] = weight
MAZEMAKER_API int mazemaker_get_edges(
    MazemakerHandle handle,
    uint64_t node_id,
    uint64_t* edge_ids,     // Must hold max_edges * 2 uint64_t
    float* weights,         // Must hold max_edges float
    int max_edges
);

// Count edges in the in-memory graph.
MAZEMAKER_API int64_t mazemaker_count_edges(MazemakerHandle handle);

// Spreading activation via C++ (runs on the in-memory graph).
// Returns number of activated nodes. Writes to node_ids[] and activations[].
MAZEMAKER_API int mazemaker_think_c(
    MazemakerHandle handle,
    uint64_t start_id,
    int depth,
    uint64_t* node_ids,
    float* activations,
    int max_results
);

// ============================================================================
// Consolidation & Decay
// ============================================================================

// Force consolidation pass. Returns number of memories consolidated.
MAZEMAKER_API size_t mazemaker_consolidate(MazemakerHandle handle);

// Apply decay to all memories and edges.
MAZEMAKER_API void mazemaker_decay(MazemakerHandle handle);

// Run link prediction. Returns number of new edges added.
MAZEMAKER_API size_t mazemaker_predict_links(MazemakerHandle handle);

// ============================================================================
// Configuration
// ============================================================================

MAZEMAKER_API void mazemaker_set_beta(MazemakerHandle handle, float beta);
MAZEMAKER_API float mazemaker_get_beta(MazemakerHandle handle);
MAZEMAKER_API void mazemaker_set_consolidation_threshold(MazemakerHandle handle, float threshold);

// ============================================================================
// Statistics
// ============================================================================

MAZEMAKER_API void mazemaker_stats_c(MazemakerHandle handle, MazemakerStats* stats);

// ============================================================================
// LSTM Predictor (opaque handle)
// ============================================================================

typedef void* LSTMPredictorHandle;

// Create LSTM predictor. Returns NULL on failure.
MAZEMAKER_API LSTMPredictorHandle nm_lstm_create(int input_dim, int hidden_dim);

// Forward pass: predict next embedding from sequence.
// sequence: float array of shape (seq_len * input_dim), row-major.
// output: float array of length input_dim (written by function).
// Returns 0 on success, -1 on error.
MAZEMAKER_API int nm_lstm_forward(LSTMPredictorHandle handle,
                                const float* sequence, int seq_len,
                                float* output);

// Train on one (sequence, target) pair. Returns MSE loss.
// Returns -1.0f on error.
MAZEMAKER_API float nm_lstm_train(LSTMPredictorHandle handle,
                                const float* sequence, int seq_len,
                                const float* target, float lr);

// Save/load weights. Returns 0 on success, -1 on error.
MAZEMAKER_API int nm_lstm_save(LSTMPredictorHandle handle, const char* path);
MAZEMAKER_API LSTMPredictorHandle nm_lstm_load(const char* path, int input_dim, int hidden_dim);

// Destroy LSTM predictor.
MAZEMAKER_API void nm_lstm_destroy(LSTMPredictorHandle handle);

// ============================================================================
// kNN Engine (opaque handle)
// ============================================================================

typedef void* KNNEngineHandle;

// kNN search result
typedef struct {
    uint64_t id;
    float score;
    float embed_similarity;
    float temporal_score;
    float freq_score;
    float graph_score;
} KNNCResult;

// Create kNN engine. Returns NULL on failure.
MAZEMAKER_API KNNEngineHandle nm_knn_create(int embed_dim);

// Search candidates with multi-signal scoring.
// query: float array of length embed_dim
// candidates: float array of shape (count * embed_dim), row-major
// candidate_ids: uint64_t array of length count
// timestamps: float array of length count (epoch seconds)
// access_counts: float array of length count
// graph_scores: float array of length count (0-1)
// lstm_context: optional float array of length embed_dim (NULL to skip)
// results: output KNNCResult array of length k (must be pre-allocated)
// Returns actual number of results, -1 on error.
MAZEMAKER_API int nm_knn_search(KNNEngineHandle handle,
                              const float* query, int embed_dim,
                              const float* candidates,
                              const uint64_t* candidate_ids,
                              int count, int k,
                              const float* timestamps,
                              const float* access_counts,
                              const float* graph_scores,
                              const float* lstm_context,
                              KNNCResult* results);

// Adjust weights based on LSTM context.
// lstm_context: optional (NULL to reset to defaults).
MAZEMAKER_API void nm_knn_adjust_weights(KNNEngineHandle handle,
                                       const float* lstm_context);

// Destroy kNN engine.
MAZEMAKER_API void nm_knn_destroy(KNNEngineHandle handle);

#ifdef __cplusplus
}
#endif
