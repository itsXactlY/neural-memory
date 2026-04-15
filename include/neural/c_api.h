// neural/c_api.h - C-compatible API for Python/ctypes integration
// All functions use extern "C" linkage for ABI stability.
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
    #define NEURAL_API __declspec(dllexport)
#else
    #define NEURAL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Opaque handle
// ============================================================================
typedef void* NeuralMemoryHandle;

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
} NeuralMemoryResult;

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
} NeuralMemoryStats;

// ============================================================================
// Lifecycle
// ============================================================================

// Create a new adapter with default config (384-dim vectors).
// Returns NULL on failure.
NEURAL_API NeuralMemoryHandle neural_memory_create(void);

// Create with explicit vector dimension.
NEURAL_API NeuralMemoryHandle neural_memory_create_dim(int vector_dim);

// Destroy adapter and free resources.
NEURAL_API void neural_memory_destroy(NeuralMemoryHandle handle);

// ============================================================================
// Core operations
// ============================================================================

// Store a vector. Returns memory ID (0 on failure).
NEURAL_API uint64_t neural_memory_store(
    NeuralMemoryHandle handle,
    const float* vec,
    int dim,
    const char* label,
    const char* content
);

// Store text (uses internal text->embedding).
NEURAL_API uint64_t neural_memory_store_text(
    NeuralMemoryHandle handle,
    const char* text,
    const char* label
);

// Retrieve top-k memories by vector similarity.
// Writes up to k results into ids[] and scores[].
// Returns actual number of results written.
NEURAL_API int neural_memory_retrieve(
    NeuralMemoryHandle handle,
    const float* vec,
    int dim,
    int k,
    uint64_t* ids,
    float* scores
);

// Retrieve top-k with full result detail.
// results must point to an array of at least k NeuralMemoryResult.
// Returns actual number of results written.
NEURAL_API int neural_memory_retrieve_full(
    NeuralMemoryHandle handle,
    const float* vec,
    int dim,
    int k,
    NeuralMemoryResult* results
);

// Text-based search.
NEURAL_API int neural_memory_search(
    NeuralMemoryHandle handle,
    const char* query,
    int k,
    uint64_t* ids,
    float* scores
);

// Read a specific memory by ID. Returns 1 on success, 0 if not found.
NEURAL_API int neural_memory_read(
    NeuralMemoryHandle handle,
    uint64_t id,
    NeuralMemoryResult* result
);

// ============================================================================
// Graph / Edge Operations (MSSQL-backed)
// ============================================================================

// Store a vector into NeuralMemory table + create GraphNode. Returns node ID.
NEURAL_API uint64_t neural_memory_store_mssql(
    NeuralMemoryHandle handle,
    const float* vec,
    int dim,
    const char* label,
    const char* content
);

// Add an edge to GraphEdges. Returns 1 on success.
NEURAL_API int neural_memory_add_edge(
    NeuralMemoryHandle handle,
    uint64_t from_id,
    uint64_t to_id,
    float weight,
    const char* edge_type
);

// Batch add edges. Returns number of edges added.
NEURAL_API int neural_memory_batch_add_edges(
    NeuralMemoryHandle handle,
    const uint64_t* from_ids,
    const uint64_t* to_ids,
    const float* weights,
    int count,
    const char* edge_type
);

// Batch strengthen edges: update weight = min(weight + delta, 1.0).
// from_ids/to_ids are parallel arrays of length count.
NEURAL_API int neural_memory_batch_strengthen_edges(
    NeuralMemoryHandle handle,
    const uint64_t* from_ids,
    const uint64_t* to_ids,
    int count,
    float delta
);

// Bulk weaken all edges: UPDATE GraphEdges SET weight = MAX(weight - delta, 0.0) WHERE weight > threshold.
// Then prunes edges below threshold. Returns number of edges pruned.
NEURAL_API int neural_memory_bulk_weaken_prune(
    NeuralMemoryHandle handle,
    float delta,
    float threshold
);

// Get all edges for a node (from OR to). Writes up to max_edges into buffer.
// Returns actual number of edges written.
// edge_ids[2*i] = from_id, edge_ids[2*i+1] = to_id, weights[i] = weight
NEURAL_API int neural_memory_get_edges(
    NeuralMemoryHandle handle,
    uint64_t node_id,
    uint64_t* edge_ids,     // Must hold max_edges * 2 uint64_t
    float* weights,         // Must hold max_edges float
    int max_edges
);

// Count edges in GraphEdges table.
NEURAL_API int64_t neural_memory_count_edges(NeuralMemoryHandle handle);

// Spreading activation via C++ (runs on GraphNodes/GraphEdges from MSSQL).
// Returns number of activated nodes. Writes to node_ids[] and activations[].
NEURAL_API int neural_memory_think(
    NeuralMemoryHandle handle,
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
NEURAL_API size_t neural_memory_consolidate(NeuralMemoryHandle handle);

// Apply decay to all memories and edges.
NEURAL_API void neural_memory_decay(NeuralMemoryHandle handle);

// Run link prediction. Returns number of new edges added.
NEURAL_API size_t neural_memory_predict_links(NeuralMemoryHandle handle);

// ============================================================================
// Configuration
// ============================================================================

NEURAL_API void neural_memory_set_beta(NeuralMemoryHandle handle, float beta);
NEURAL_API float neural_memory_get_beta(NeuralMemoryHandle handle);
NEURAL_API void neural_memory_set_consolidation_threshold(NeuralMemoryHandle handle, float threshold);

// ============================================================================
// Statistics
// ============================================================================

NEURAL_API void neural_memory_stats(NeuralMemoryHandle handle, NeuralMemoryStats* stats);

// ============================================================================
// LSTM Predictor (opaque handle)
// ============================================================================

typedef void* LSTMPredictorHandle;

// Create LSTM predictor. Returns NULL on failure.
NEURAL_API LSTMPredictorHandle nm_lstm_create(int input_dim, int hidden_dim);

// Forward pass: predict next embedding from sequence.
// sequence: float array of shape (seq_len * input_dim), row-major.
// output: float array of length input_dim (written by function).
// Returns 0 on success, -1 on error.
NEURAL_API int nm_lstm_forward(LSTMPredictorHandle handle,
                                const float* sequence, int seq_len,
                                float* output);

// Train on one (sequence, target) pair. Returns MSE loss.
// Returns -1.0f on error.
NEURAL_API float nm_lstm_train(LSTMPredictorHandle handle,
                                const float* sequence, int seq_len,
                                const float* target, float lr);

// Save/load weights. Returns 0 on success, -1 on error.
NEURAL_API int nm_lstm_save(LSTMPredictorHandle handle, const char* path);
NEURAL_API LSTMPredictorHandle nm_lstm_load(const char* path, int input_dim, int hidden_dim);

// Destroy LSTM predictor.
NEURAL_API void nm_lstm_destroy(LSTMPredictorHandle handle);

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
NEURAL_API KNNEngineHandle nm_knn_create(int embed_dim);

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
NEURAL_API int nm_knn_search(KNNEngineHandle handle,
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
NEURAL_API void nm_knn_adjust_weights(KNNEngineHandle handle,
                                       const float* lstm_context);

// Destroy kNN engine.
NEURAL_API void nm_knn_destroy(KNNEngineHandle handle);

#ifdef __cplusplus
}
#endif
