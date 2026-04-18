// vector_store.cpp - Core vector storage operations
// Insert/retrieve/search vectors stored as VARBINARY in SQL Server.
// Cosine similarity computed in C++ after fetching raw bytes.
// Metadata stored as NVARCHAR(MAX) JSON.

#include "neural/mssql.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>

#ifdef __AVX2__
#ifdef __AVX2__
#include <immintrin.h>
#endif
#endif

namespace neural::mssql {

// ============================================================================
// MSSQLVectorAdapter - Construction & Lifecycle
// ============================================================================

MSSQLVectorAdapter::MSSQLVectorAdapter(const ConnectionConfig& config)
    : config_(config) {}

MSSQLVectorAdapter::~MSSQLVectorAdapter() {
    shutdown();
}

bool MSSQLVectorAdapter::initialize() {
    if (initialized_) return true;

    pool_ = std::make_unique<ConnectionPool>(config_);
    if (!pool_->initialize()) {
        std::cerr << "[MSSQLVectorAdapter] Failed to initialize connection pool" << std::endl;
        return false;
    }

    // Validate schema exists
    auto conn = pool_->acquire();
    if (!conn) {
        std::cerr << "[MSSQLVectorAdapter] Failed to acquire connection for validation" << std::endl;
        return false;
    }

    Statement stmt(conn->dbc());
    std::string check_sql =
        "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_NAME IN ('NeuralMemory', 'GraphNodes_v2', 'GraphEdges_v2')";

    if (!stmt.prepare(check_sql) || !stmt.execute() || !stmt.fetch()) {
        std::cerr << "[MSSQLVectorAdapter] Schema validation failed" << std::endl;
        pool_->release(std::move(conn));
        return false;
    }

    auto count = stmt.get_int64(1);
    if (!count || *count < 3) {
        std::cerr << "[MSSQLVectorAdapter] WARNING: Expected 3 tables, found "
                  << (count ? std::to_string(*count) : "0") << std::endl;
    }

    stmt.reset();
    pool_->release(std::move(conn));

    initialized_ = true;
    return true;
}

void MSSQLVectorAdapter::shutdown() {
    if (pool_) {
        pool_->shutdown();
        pool_.reset();
    }
    initialized_ = false;
}

// ============================================================================
// Vector Conversion Utilities
// ============================================================================

std::vector<uint8_t> MSSQLVectorAdapter::vector_to_binary(const std::vector<float>& vec) {
    std::vector<uint8_t> bytes(vec.size() * sizeof(float));
    std::memcpy(bytes.data(), vec.data(), bytes.size());
    return bytes;
}

std::vector<float> MSSQLVectorAdapter::binary_to_vector(const std::vector<uint8_t>& data) {
    size_t float_count = data.size() / sizeof(float);
    std::vector<float> vec(float_count);
    std::memcpy(vec.data(), data.data(), data.size());
    return vec;
}

// ============================================================================
// Cosine Similarity - AVX2-optimized when available
// ============================================================================

float MSSQLVectorAdapter::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;

    const size_t n = a.size();
    const float* pa = a.data();
    const float* pb = b.data();

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

#ifdef __AVX2__
    // Process 8 floats at a time with AVX2
    const size_t simd_end = n - (n % 8);
    __m256 vdot = _mm256_setzero_ps();
    __m256 vna = _mm256_setzero_ps();
    __m256 vnb = _mm256_setzero_ps();

    for (size_t i = 0; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(pa + i);
        __m256 vb = _mm256_loadu_ps(pb + i);
        vdot = _mm256_fmadd_ps(va, vb, vdot);
        vna = _mm256_fmadd_ps(va, va, vna);
        vnb = _mm256_fmadd_ps(vb, vb, vnb);
    }

    // Horizontal sum
    alignas(32) float dot_arr[8], na_arr[8], nb_arr[8];
    _mm256_store_ps(dot_arr, vdot);
    _mm256_store_ps(na_arr, vna);
    _mm256_store_ps(nb_arr, vnb);

    for (int i = 0; i < 8; ++i) {
        dot += dot_arr[i];
        norm_a += na_arr[i];
        norm_b += nb_arr[i];
    }

    // Handle remaining elements
    for (size_t i = simd_end; i < n; ++i) {
        dot += pa[i] * pb[i];
        norm_a += pa[i] * pa[i];
        norm_b += pb[i] * pb[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dot += pa[i] * pb[i];
        norm_a += pa[i] * pa[i];
        norm_b += pb[i] * pb[i];
    }
#endif

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

// ============================================================================
// Single Vector Insert
// ============================================================================

bool MSSQLVectorAdapter::insert_vector(uint64_t id, std::span<const float> vector,
                                        const std::string& metadata_json) {
    if (!initialized_ || !pool_) return false;

    auto conn = pool_->acquire();
    if (!conn) return false;

    Statement stmt(conn->dbc());
    std::string sql =
        "INSERT INTO NeuralMemory (id, legacy_id, vector_data, vector_dim, metadata_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, GETUTCDATE(), GETUTCDATE())";

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return false;
    }

    // Convert vector to binary
    std::vector<uint8_t> vec_bytes(vector.size() * sizeof(float));
    std::memcpy(vec_bytes.data(), vector.data(), vec_bytes.size());

    int64_t id_val = static_cast<int64_t>(id);
    int32_t dim_val = static_cast<int32_t>(vector.size());
    SQLLEN id_ind = 0;
    SQLLEN leg_ind = 0;
    SQLLEN vec_ind = static_cast<SQLLEN>(vec_bytes.size());
    SQLLEN dim_ind = 0;
    SQLLEN meta_ind = static_cast<SQLLEN>(metadata_json.size());

    // Bind parameters
    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0,
                     &id_val, 0, &id_ind);

    // Bind legacy_id (= id for new rows)
    SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0,
                     &id_val, 0, &leg_ind);

    SQLBindParameter(stmt.stmt(), 3, SQL_PARAM_INPUT,
                     SQL_C_BINARY, SQL_VARBINARY,
                     static_cast<SQLULEN>(vec_bytes.size()), 0,
                     vec_bytes.data(), static_cast<SQLLEN>(vec_bytes.size()), &vec_ind);

    SQLBindParameter(stmt.stmt(), 4, SQL_PARAM_INPUT,
                     SQL_C_SLONG, SQL_INTEGER, 10, 0,
                     &dim_val, 0, &dim_ind);

    SQLBindParameter(stmt.stmt(), 5, SQL_PARAM_INPUT,
                     SQL_C_CHAR, SQL_VARCHAR,
                     static_cast<SQLULEN>(metadata_json.size()), 0,
                     const_cast<char*>(metadata_json.c_str()),
                     static_cast<SQLLEN>(metadata_json.size()), &meta_ind);

    bool ok = stmt.execute();
    if (!ok) {
        auto errors = stmt.get_errors();
        std::cerr << "[MSSQL] insert_vector failed for id " << id << std::endl;
        for (const auto& err : errors) {
            std::cerr << "  State: " << err.state << " Native: " << err.native_error << " Msg: " << err.message << std::endl;
        }
    }
    pool_->release(std::move(conn));
    return ok;
}

// ============================================================================
// Batch Vector Insert
// ============================================================================

bool MSSQLVectorAdapter::insert_vectors(const std::vector<uint64_t>& ids,
                                         const std::vector<std::vector<float>>& vectors,
                                         const std::vector<std::string>& metadata_jsons) {
    if (!initialized_ || !pool_) return false;
    if (ids.empty() || ids.size() != vectors.size() || ids.size() != metadata_jsons.size()) {
        return false;
    }

    std::string sql =
        "INSERT INTO NeuralMemory (id, legacy_id, vector_data, vector_dim, metadata_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, GETUTCDATE(), GETUTCDATE())";

    // Direct transactional batch insert
    auto conn = pool_->acquire();
    if (!conn) return false;

    conn->begin_transaction();

    Statement stmt(conn->dbc());

    if (!stmt.prepare(sql)) {
        conn->rollback_transaction();
        pool_->release(std::move(conn));
        return false;
    }

    bool success = true;
    for (size_t i = 0; i < ids.size(); ++i) {
        int64_t id_val = static_cast<int64_t>(ids[i]);
        auto vec_bytes = vector_to_binary(vectors[i]);
        const auto& meta = metadata_jsons[i];
        int32_t dim_val = static_cast<int32_t>(vectors[i].size());

        SQLLEN id_ind = 0;
        SQLLEN leg_ind = 0;
        SQLLEN vec_ind = static_cast<SQLLEN>(vec_bytes.size());
        SQLLEN dim_ind = 0;
        SQLLEN meta_ind = static_cast<SQLLEN>(meta.size());

        SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                         SQL_C_SBIGINT, SQL_BIGINT, 0, 0,
                         &id_val, 0, &id_ind);

        SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                         SQL_C_SBIGINT, SQL_BIGINT, 0, 0,
                         &id_val, 0, &leg_ind);

        SQLBindParameter(stmt.stmt(), 3, SQL_PARAM_INPUT,
                         SQL_C_BINARY, SQL_VARBINARY,
                         static_cast<SQLULEN>(vec_bytes.size()), 0,
                         vec_bytes.data(), static_cast<SQLLEN>(vec_bytes.size()), &vec_ind);

        SQLBindParameter(stmt.stmt(), 4, SQL_PARAM_INPUT,
                         SQL_C_SLONG, SQL_INTEGER, 10, 0,
                         &dim_val, 0, &dim_ind);

        SQLBindParameter(stmt.stmt(), 5, SQL_PARAM_INPUT,
                         SQL_C_CHAR, SQL_VARCHAR,
                         static_cast<SQLULEN>(meta.size()), 0,
                         const_cast<char*>(meta.c_str()),
                         static_cast<SQLLEN>(meta.size()), &meta_ind);

        if (!stmt.execute()) {
            success = false;
            break;
        }
        stmt.reset();
    }

    if (success) {
        conn->commit_transaction();
    } else {
        conn->rollback_transaction();
    }

    pool_->release(std::move(conn));
    return success;
}

// ============================================================================
// Retrieve Single Vector by ID
// ============================================================================

std::optional<SearchResult> MSSQLVectorAdapter::get_vector(uint64_t id) {
    if (!initialized_ || !pool_) return std::nullopt;

    auto conn = pool_->acquire();
    if (!conn) return std::nullopt;

    Statement stmt(conn->dbc());
    std::string sql =
        "SELECT id, vector_data, metadata_json FROM NeuralMemory WHERE id = ?";

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return std::nullopt;
    }

    int64_t id_val = static_cast<int64_t>(id);
    SQLLEN id_ind = 0;
    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0,
                     &id_val, 0, &id_ind);

    if (!stmt.execute() || !stmt.fetch()) {
        pool_->release(std::move(conn));
        return std::nullopt;
    }

    SearchResult result;
    auto id_opt = stmt.get_int64(1);
    auto vec_opt = stmt.get_binary(2);
    auto meta_opt = stmt.get_string(3);

    result.id = static_cast<uint64_t>(id_opt.value_or(0));
    result.vector = vec_opt ? binary_to_vector(*vec_opt) : std::vector<float>{};
    result.metadata_json = meta_opt.value_or("{}");
    result.similarity = 1.0f; // exact match

    pool_->release(std::move(conn));
    return result;
}

// ============================================================================
// Retrieve Multiple Vectors by IDs
// ============================================================================

std::vector<SearchResult> MSSQLVectorAdapter::get_vectors(const std::vector<uint64_t>& ids) {
    std::vector<SearchResult> results;
    if (!initialized_ || !pool_ || ids.empty()) return results;

    auto conn = pool_->acquire();
    if (!conn) return results;

    // Build IN clause with parameterized values
    // Since we can't parameterize IN directly, we use a temp table approach
    // or build a batch query. For simplicity, we'll query one at a time for small batches
    // and use a table-valued approach for large batches.

    if (ids.size() <= 100) {
        // Individual queries for small batches
        for (uint64_t id : ids) {
            auto result = get_vector(id);
            if (result) results.push_back(*result);
        }
    } else {
        // Use OPENJSON for large ID batches
        Statement stmt(conn->dbc());

        std::ostringstream json_arr;
        json_arr << "[";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) json_arr << ",";
            json_arr << ids[i];
        }
        json_arr << "]";

        std::string sql =
            "SELECT nm.id, nm.vector_data, nm.metadata_json "
            "FROM NeuralMemory nm "
            "INNER JOIN OPENJSON(?) WITH (id BIGINT '$') AS ids ON nm.id = ids.id";

        if (!stmt.prepare(sql)) {
            pool_->release(std::move(conn));
            return results;
        }

        std::string json_str = json_arr.str();
        SQLLEN json_ind = static_cast<SQLLEN>(json_str.size());
        SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                         SQL_C_CHAR, SQL_VARCHAR,
                         static_cast<SQLULEN>(json_str.size()), 0,
                         const_cast<char*>(json_str.c_str()),
                         static_cast<SQLLEN>(json_str.size()), &json_ind);

        if (!stmt.execute()) {
            pool_->release(std::move(conn));
            return results;
        }

        while (stmt.fetch()) {
            SearchResult result;
            auto id_opt = stmt.get_int64(1);
            auto vec_opt = stmt.get_binary(2);
            auto meta_opt = stmt.get_string(3);

            result.id = static_cast<uint64_t>(id_opt.value_or(0));
            result.vector = vec_opt ? binary_to_vector(*vec_opt) : std::vector<float>{};
            result.metadata_json = meta_opt.value_or("{}");
            result.similarity = 1.0f;
            results.push_back(std::move(result));
        }
    }

    pool_->release(std::move(conn));
    return results;
}

// ============================================================================
// Similarity Search - Top-K Nearest Neighbors
// ============================================================================

std::vector<SearchResult> MSSQLVectorAdapter::search_similar(
    std::span<const float> query_vector, int top_k, float threshold) {

    std::vector<SearchResult> results;
    if (!initialized_ || !pool_ || query_vector.empty()) return results;

    auto conn = pool_->acquire();
    if (!conn) return results;

    // Fetch all vectors (for datasets that fit in memory) and compute cosine similarity
    // For production with millions of vectors, consider:
    // 1. Pre-filtering with approximate methods (LSH, HNSW)
    // 2. Using SQL Server CLR for in-database similarity
    // 3. Using a vector index (FAISS, etc.) as a pre-filter

    Statement stmt(conn->dbc());

    // Retrieve all vectors - in production, add WHERE clause for filtering
    std::string sql =
        "SELECT id, vector_data, metadata_json FROM NeuralMemory";

    if (!stmt.prepare(sql) || !stmt.execute()) {
        pool_->release(std::move(conn));
        return results;
    }

    // Compute similarity for each vector
    while (stmt.fetch()) {
        auto id_opt = stmt.get_int64(1);
        auto vec_opt = stmt.get_binary(2);
        auto meta_opt = stmt.get_string(3);

        if (!id_opt || !vec_opt) continue;

        auto candidate_vec = binary_to_vector(*vec_opt);
        float sim = cosine_similarity(
            std::vector<float>(query_vector.begin(), query_vector.end()),
            candidate_vec);

        if (sim >= threshold) {
            SearchResult result;
            result.id = static_cast<uint64_t>(*id_opt);
            result.vector = std::move(candidate_vec);
            result.metadata_json = meta_opt.value_or("{}");
            result.similarity = sim;
            results.push_back(std::move(result));
        }
    }

    // Sort by similarity descending and take top-K
    std::sort(results.begin(), results.end());
    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }

    pool_->release(std::move(conn));
    return results;
}

// ============================================================================
// Similarity Search with Metadata Filter
// ============================================================================

std::vector<SearchResult> MSSQLVectorAdapter::search_similar_filtered(
    std::span<const float> query_vector,
    const std::string& metadata_filter_json,
    int top_k, float threshold) {

    std::vector<SearchResult> results;
    if (!initialized_ || !pool_ || query_vector.empty()) return results;

    auto conn = pool_->acquire();
    if (!conn) return results;

    // Parse filter JSON for common patterns and build WHERE clause
    // This is a simplified implementation - production would use JSON_VALUE queries
    Statement stmt(conn->dbc());

    std::string sql;
    // Use OPENJSON to filter by metadata properties
    // Example: filter = {"category": "image"} -> WHERE JSON_VALUE(metadata_json, '$.category') = 'image'
    // For generality, we fetch all and filter in-memory if JSON parsing is complex

    sql = "SELECT id, vector_data, metadata_json FROM NeuralMemory "
          "WHERE metadata_json LIKE ?";  // Simple LIKE-based filter

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return results;
    }

    // Build a LIKE pattern from the filter (simplified)
    std::string like_pattern = "%" + metadata_filter_json.substr(1, metadata_filter_json.size() - 2) + "%";
    SQLLEN filter_ind = static_cast<SQLLEN>(like_pattern.size());
    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_CHAR, SQL_VARCHAR,
                     static_cast<SQLULEN>(like_pattern.size()), 0,
                     const_cast<char*>(like_pattern.c_str()),
                     static_cast<SQLLEN>(like_pattern.size()), &filter_ind);

    if (!stmt.execute()) {
        pool_->release(std::move(conn));
        return results;
    }

    while (stmt.fetch()) {
        auto id_opt = stmt.get_int64(1);
        auto vec_opt = stmt.get_binary(2);
        auto meta_opt = stmt.get_string(3);

        if (!id_opt || !vec_opt) continue;

        auto candidate_vec = binary_to_vector(*vec_opt);
        float sim = cosine_similarity(
            std::vector<float>(query_vector.begin(), query_vector.end()),
            candidate_vec);

        if (sim >= threshold) {
            SearchResult result;
            result.id = static_cast<uint64_t>(*id_opt);
            result.vector = std::move(candidate_vec);
            result.metadata_json = meta_opt.value_or("{}");
            result.similarity = sim;
            results.push_back(std::move(result));
        }
    }

    std::sort(results.begin(), results.end());
    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }

    pool_->release(std::move(conn));
    return results;
}

// ============================================================================
// Delete Vector
// ============================================================================

bool MSSQLVectorAdapter::delete_vector(uint64_t id) {
    if (!initialized_ || !pool_) return false;

    auto conn = pool_->acquire();
    if (!conn) return false;

    Statement stmt(conn->dbc());
    std::string sql = "DELETE FROM NeuralMemory WHERE id = ?";

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return false;
    }

    int64_t id_val = static_cast<int64_t>(id);
    SQLLEN id_ind = 0;
    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0,
                     &id_val, 0, &id_ind);

    bool ok = stmt.execute();
    pool_->release(std::move(conn));
    return ok;
}

// ============================================================================
// Update Metadata
// ============================================================================

bool MSSQLVectorAdapter::update_metadata(uint64_t id, const std::string& metadata_json) {
    if (!initialized_ || !pool_) return false;

    auto conn = pool_->acquire();
    if (!conn) return false;

    Statement stmt(conn->dbc());
    std::string sql =
        "UPDATE NeuralMemory SET metadata_json = ?, updated_at = GETUTCDATE() WHERE id = ?";

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return false;
    }

    SQLLEN meta_ind = static_cast<SQLLEN>(metadata_json.size());
    int64_t id_val = static_cast<int64_t>(id);
    SQLLEN id_ind = 0;

    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_CHAR, SQL_VARCHAR,
                     static_cast<SQLULEN>(metadata_json.size()), 0,
                     const_cast<char*>(metadata_json.c_str()),
                     static_cast<SQLLEN>(metadata_json.size()), &meta_ind);

    SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0,
                     &id_val, 0, &id_ind);

    bool ok = stmt.execute();
    pool_->release(std::move(conn));
    return ok;
}

// ============================================================================
// Graph Operations
// ============================================================================

bool MSSQLVectorAdapter::insert_graph_node(uint64_t node_id, const std::string& node_type,
                                            const std::string& properties_json) {
    if (!initialized_ || !pool_) return false;

    auto conn = pool_->acquire();
    if (!conn) return false;

    Statement stmt(conn->dbc());
    std::string sql =
        "INSERT INTO GraphNodes_v2 (node_id, node_type, properties_json, created_at, updated_at) "
        "VALUES (?, ?, ?, GETUTCDATE(), GETUTCDATE())";

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return false;
    }

    int64_t nid = static_cast<int64_t>(node_id);
    SQLLEN nid_ind = 0;
    SQLLEN type_ind = static_cast<SQLLEN>(node_type.size());
    SQLLEN props_ind = static_cast<SQLLEN>(properties_json.size());

    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &nid, 0, &nid_ind);
    SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                     SQL_C_CHAR, SQL_VARCHAR, static_cast<SQLULEN>(node_type.size()), 0,
                     const_cast<char*>(node_type.c_str()), static_cast<SQLLEN>(node_type.size()), &type_ind);
    SQLBindParameter(stmt.stmt(), 3, SQL_PARAM_INPUT,
                     SQL_C_CHAR, SQL_VARCHAR, static_cast<SQLULEN>(properties_json.size()), 0,
                     const_cast<char*>(properties_json.c_str()), static_cast<SQLLEN>(properties_json.size()), &props_ind);

    bool ok = stmt.execute();
    pool_->release(std::move(conn));
    return ok;
}

bool MSSQLVectorAdapter::insert_graph_edge(uint64_t from_id, uint64_t to_id,
                                            const std::string& edge_type, float weight) {
    if (!initialized_ || !pool_) return false;

    auto conn = pool_->acquire();
    if (!conn) return false;

    Statement stmt(conn->dbc());
    std::string sql =
        "INSERT INTO GraphEdges_v2 (from_node_id, to_node_id, edge_type, weight, created_at) "
        "VALUES (?, ?, ?, ?, GETUTCDATE())";

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return false;
    }

    int64_t fid = static_cast<int64_t>(from_id);
    int64_t tid = static_cast<int64_t>(to_id);
    SQLLEN fid_ind = 0, tid_ind = 0, weight_ind = 0;
    SQLLEN type_ind = static_cast<SQLLEN>(edge_type.size());

    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &fid, 0, &fid_ind);
    SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &tid, 0, &tid_ind);
    SQLBindParameter(stmt.stmt(), 3, SQL_PARAM_INPUT,
                     SQL_C_CHAR, SQL_VARCHAR, static_cast<SQLULEN>(edge_type.size()), 0,
                     const_cast<char*>(edge_type.c_str()), static_cast<SQLLEN>(edge_type.size()), &type_ind);
    SQLBindParameter(stmt.stmt(), 4, SQL_PARAM_INPUT,
                     SQL_C_FLOAT, SQL_REAL, 0, 0, &weight, 0, &weight_ind);

    bool ok = stmt.execute();
    pool_->release(std::move(conn));
    return ok;
}

bool MSSQLVectorAdapter::delete_graph_node(uint64_t node_id) {
    if (!initialized_ || !pool_) return false;

    auto conn = pool_->acquire();
    if (!conn) return false;

    conn->begin_transaction();
    Statement stmt(conn->dbc());

    // Delete edges first
    std::string del_edges =
        "DELETE FROM GraphEdges_v2 WHERE from_node_id = ? OR to_node_id = ?";
    if (!stmt.prepare(del_edges)) {
        conn->rollback_transaction();
        pool_->release(std::move(conn));
        return false;
    }

    int64_t nid = static_cast<int64_t>(node_id);
    SQLLEN nid_ind = 0;
    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &nid, 0, &nid_ind);
    SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &nid, 0, &nid_ind);
    stmt.execute();
    stmt.reset();

    // Delete node
    std::string del_node = "DELETE FROM GraphNodes_v2 WHERE node_id = ?";
    if (!stmt.prepare(del_node)) {
        conn->rollback_transaction();
        pool_->release(std::move(conn));
        return false;
    }

    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &nid, 0, &nid_ind);

    bool ok = stmt.execute();
    if (ok) conn->commit_transaction();
    else conn->rollback_transaction();

    pool_->release(std::move(conn));
    return ok;
}

bool MSSQLVectorAdapter::delete_graph_edge(uint64_t from_id, uint64_t to_id,
                                            const std::string& edge_type) {
    if (!initialized_ || !pool_) return false;

    auto conn = pool_->acquire();
    if (!conn) return false;

    Statement stmt(conn->dbc());
    std::string sql =
        "DELETE FROM GraphEdges_v2 WHERE from_node_id = ? AND to_node_id = ? AND edge_type = ?";

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return false;
    }

    int64_t fid = static_cast<int64_t>(from_id);
    int64_t tid = static_cast<int64_t>(to_id);
    SQLLEN fid_ind = 0, tid_ind = 0;
    SQLLEN type_ind = static_cast<SQLLEN>(edge_type.size());

    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &fid, 0, &fid_ind);
    SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &tid, 0, &tid_ind);
    SQLBindParameter(stmt.stmt(), 3, SQL_PARAM_INPUT,
                     SQL_C_CHAR, SQL_VARCHAR, static_cast<SQLULEN>(edge_type.size()), 0,
                     const_cast<char*>(edge_type.c_str()), static_cast<SQLLEN>(edge_type.size()), &type_ind);

    bool ok = stmt.execute();
    pool_->release(std::move(conn));
    return ok;
}

// ============================================================================
// Spreading Activation
// ============================================================================

std::vector<MSSQLVectorAdapter::ActivationResult> MSSQLVectorAdapter::spreading_activation(
    uint64_t start_node, float decay_factor, float threshold, int max_depth) {

    std::vector<ActivationResult> results;
    if (!initialized_ || !pool_) return results;

    auto conn = pool_->acquire();
    if (!conn) return results;

    // Call the stored procedure
    Statement stmt(conn->dbc());
    std::string sql = "{CALL SpreadingActivation(?, ?, ?, ?)}";

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return results;
    }

    int64_t start_id = static_cast<int64_t>(start_node);
    SQLLEN start_ind = 0;
    SQLLEN decay_ind = 0;
    SQLLEN thresh_ind = 0;
    SQLLEN depth_ind = 0;

    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &start_id, 0, &start_ind);
    SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                     SQL_C_FLOAT, SQL_REAL, 0, 0,
                     const_cast<float*>(&decay_factor), 0, &decay_ind);
    SQLBindParameter(stmt.stmt(), 3, SQL_PARAM_INPUT,
                     SQL_C_FLOAT, SQL_REAL, 0, 0,
                     const_cast<float*>(&threshold), 0, &thresh_ind);
    SQLBindParameter(stmt.stmt(), 4, SQL_PARAM_INPUT,
                     SQL_C_SLONG, SQL_INTEGER, 0, 0,
                     const_cast<int*>(&max_depth), 0, &depth_ind);

    if (!stmt.execute()) {
        pool_->release(std::move(conn));
        return results;
    }

    while (stmt.fetch()) {
        ActivationResult result;
        auto nid = stmt.get_int64(1);
        auto ntype = stmt.get_string(2);
        auto act = stmt.get_float(3);
        auto dep = stmt.get_int64(4);

        result.node_id = static_cast<uint64_t>(nid.value_or(0));
        result.node_type = ntype.value_or("");
        result.activation = act.value_or(0.0f);
        result.depth = static_cast<int>(dep.value_or(0));

        if (result.activation >= threshold) {
            results.push_back(std::move(result));
        }
    }

    pool_->release(std::move(conn));
    return results;
}

// ============================================================================
// Consolidation
// ============================================================================

bool MSSQLVectorAdapter::run_consolidation(float merge_threshold) {
    if (!initialized_ || !pool_) return false;

    auto conn = pool_->acquire();
    if (!conn) return false;

    Statement stmt(conn->dbc());
    std::string sql = "{CALL Consolidation(?)}";

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return false;
    }

    SQLLEN thresh_ind = 0;
    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_FLOAT, SQL_REAL, 0, 0,
                     const_cast<float*>(&merge_threshold), 0, &thresh_ind);

    bool ok = stmt.execute();
    pool_->release(std::move(conn));
    return ok;
}

// ============================================================================
// Utility
// ============================================================================

int64_t MSSQLVectorAdapter::count_vectors() {
    if (!initialized_ || !pool_) return 0;

    auto conn = pool_->acquire();
    if (!conn) return 0;

    Statement stmt(conn->dbc());
    std::string sql = "SELECT COUNT(*) FROM NeuralMemory";

    if (!stmt.prepare(sql) || !stmt.execute() || !stmt.fetch()) {
        pool_->release(std::move(conn));
        return 0;
    }

    auto count = stmt.get_int64(1);
    pool_->release(std::move(conn));
    return count.value_or(0);
}

int64_t MSSQLVectorAdapter::get_max_vector_id() {
    if (!initialized_) return 0;
    auto conn = pool_->acquire();
    if (!conn) return 0;

    Statement stmt(conn->dbc());
    if (!stmt.prepare("SELECT ISNULL(MAX(id), 0) FROM NeuralMemory") || !stmt.execute() || !stmt.fetch()) {
        pool_->release(std::move(conn));
        return 0;
    }

    int64_t max_id = stmt.get_int64(1).value_or(0);
    pool_->release(std::move(conn));
    return max_id;
}

size_t MSSQLVectorAdapter::pool_available() const {
    return pool_ ? pool_->available_connections() : 0;
}

size_t MSSQLVectorAdapter::pool_in_use() const {
    return pool_ ? pool_->in_use_connections() : 0;
}

// ============================================================================
// Batch Edge Operations (for NREM deadlock prevention)
// ============================================================================

int MSSQLVectorAdapter::batch_strengthen_edges(
    const std::vector<uint64_t>& from_ids,
    const std::vector<uint64_t>& to_ids,
    float delta)
{
    if (!initialized_ || !pool_ || from_ids.empty() || from_ids.size() != to_ids.size())
        return 0;

    auto conn = pool_->acquire();
    if (!conn) return 0;

    conn->begin_transaction();
    Statement stmt(conn->dbc());

    std::string sql =
        "UPDATE GraphEdges_v2 SET weight = CASE "
        "WHEN weight + ? > 1.0 THEN 1.0 ELSE weight + ? END "
        "WHERE from_node_id = ? AND to_node_id = ?";

    if (!stmt.prepare(sql)) {
        conn->rollback_transaction();
        pool_->release(std::move(conn));
        return 0;
    }

    int updated = 0;
    for (size_t i = 0; i < from_ids.size(); ++i) {
        int64_t from_val = static_cast<int64_t>(from_ids[i]);
        int64_t to_val = static_cast<int64_t>(to_ids[i]);
        SQLLEN ind = 0;

        SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                         SQL_C_FLOAT, SQL_REAL, 0, 0, const_cast<float*>(&delta), 0, &ind);
        SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                         SQL_C_FLOAT, SQL_REAL, 0, 0, const_cast<float*>(&delta), 0, &ind);
        SQLBindParameter(stmt.stmt(), 3, SQL_PARAM_INPUT,
                         SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &from_val, 0, &ind);
        SQLBindParameter(stmt.stmt(), 4, SQL_PARAM_INPUT,
                         SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &to_val, 0, &ind);

        if (stmt.execute()) {
            updated++;
        }
        stmt.reset();
    }

    conn->commit_transaction();
    pool_->release(std::move(conn));
    return updated;
}

int MSSQLVectorAdapter::bulk_weaken_prune(float delta, float threshold) {
    if (!initialized_ || !pool_) return 0;

    auto conn = pool_->acquire();
    if (!conn) return 0;

    conn->begin_transaction();

    // Step 1: Bulk weaken
    {
        Statement stmt(conn->dbc());
        std::string sql =
            "UPDATE GraphEdges_v2 SET weight = CASE "
            "WHEN weight - ? < 0.0 THEN 0.0 ELSE weight - ? END "
            "WHERE weight > ?";

        if (!stmt.prepare(sql)) {
            conn->rollback_transaction();
            pool_->release(std::move(conn));
            return 0;
        }

        SQLLEN ind = 0;
        SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                         SQL_C_FLOAT, SQL_REAL, 0, 0, const_cast<float*>(&delta), 0, &ind);
        SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                         SQL_C_FLOAT, SQL_REAL, 0, 0, const_cast<float*>(&delta), 0, &ind);
        SQLBindParameter(stmt.stmt(), 3, SQL_PARAM_INPUT,
                         SQL_C_FLOAT, SQL_REAL, 0, 0, const_cast<float*>(&threshold), 0, &ind);

        stmt.execute();
    }

    // Step 2: Prune
    int pruned = 0;
    {
        Statement stmt(conn->dbc());
        std::string sql = "DELETE FROM GraphEdges_v2 WHERE weight < ?";

        if (stmt.prepare(sql)) {
            SQLLEN ind = 0;
            SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                             SQL_C_FLOAT, SQL_REAL, 0, 0, const_cast<float*>(&threshold), 0, &ind);
            if (stmt.execute()) {
                Statement count_stmt(conn->dbc());
                if (count_stmt.prepare("SELECT @@ROWCOUNT") &&
                    count_stmt.execute() && count_stmt.fetch()) {
                    pruned = static_cast<int>(count_stmt.get_int64(1).value_or(0));
                }
            }
        }
    }

    conn->commit_transaction();
    pool_->release(std::move(conn));
    return pruned;
}

std::vector<MSSQLVectorAdapter::EdgeInfo> MSSQLVectorAdapter::get_edges(uint64_t node_id) const {
    std::vector<EdgeInfo> results;
    if (!initialized_ || !pool_) return results;

    auto conn = pool_->acquire();
    if (!conn) return results;

    Statement stmt(conn->dbc());
    std::string sql =
        "SELECT from_node_id, to_node_id, weight FROM GraphEdges_v2 "
        "WHERE from_node_id = ? OR to_node_id = ?";

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return results;
    }

    int64_t id_val = static_cast<int64_t>(node_id);
    SQLLEN ind = 0;
    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &id_val, 0, &ind);
    SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &id_val, 0, &ind);

    if (!stmt.execute()) {
        pool_->release(std::move(conn));
        return results;
    }

    while (stmt.fetch()) {
        EdgeInfo edge;
        edge.from_id = static_cast<uint64_t>(stmt.get_int64(1).value_or(0));
        edge.to_id = static_cast<uint64_t>(stmt.get_int64(2).value_or(0));
        edge.weight = stmt.get_float(3).value_or(0.0f);
        results.push_back(edge);
    }

    pool_->release(std::move(conn));
    return results;
}

int64_t MSSQLVectorAdapter::count_edges() const {
    if (!initialized_ || !pool_) return 0;

    auto conn = pool_->acquire();
    if (!conn) return 0;

    Statement stmt(conn->dbc());
    std::string sql = "SELECT COUNT(*) FROM GraphEdges_v2";

    if (!stmt.prepare(sql) || !stmt.execute() || !stmt.fetch()) {
        pool_->release(std::move(conn));
        return 0;
    }

    auto count = stmt.get_int64(1);
    pool_->release(std::move(conn));
    return count.value_or(0);
}

// ============================================================================
// MERGE/UPSERT Edge
// ============================================================================

bool MSSQLVectorAdapter::add_graph_edge_or_update(uint64_t from_id, uint64_t to_id,
                                                   const std::string& edge_type, float weight) {
    if (!initialized_ || !pool_) return false;

    auto conn = pool_->acquire();
    if (!conn) return false;

    Statement stmt(conn->dbc());
    std::string sql =
        "MERGE GraphEdges_v2 AS target "
        "USING (SELECT ? AS from_node_id, ? AS to_node_id, ? AS edge_type) AS source "
        "ON target.from_node_id = source.from_node_id "
        "AND target.to_node_id = source.to_node_id "
        "AND target.edge_type = source.edge_type "
        "WHEN MATCHED THEN "
        "    UPDATE SET target.weight = ? "
        "WHEN NOT MATCHED THEN "
        "    INSERT (from_node_id, to_node_id, edge_type, weight, created_at) "
        "    VALUES (?, ?, ?, ?, GETUTCDATE());";

    if (!stmt.prepare(sql)) {
        pool_->release(std::move(conn));
        return false;
    }

    int64_t fid = static_cast<int64_t>(from_id);
    int64_t tid = static_cast<int64_t>(to_id);
    SQLLEN fid_ind = 0, tid_ind = 0, weight_ind = 0;
    SQLLEN type_ind = static_cast<SQLLEN>(edge_type.size());

    // source from_node_id
    SQLBindParameter(stmt.stmt(), 1, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &fid, 0, &fid_ind);
    // source to_node_id
    SQLBindParameter(stmt.stmt(), 2, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &tid, 0, &tid_ind);
    // source edge_type
    SQLBindParameter(stmt.stmt(), 3, SQL_PARAM_INPUT,
                     SQL_C_CHAR, SQL_VARCHAR, static_cast<SQLULEN>(edge_type.size()), 0,
                     const_cast<char*>(edge_type.c_str()), static_cast<SQLLEN>(edge_type.size()), &type_ind);
    // MATCHED update weight
    SQLBindParameter(stmt.stmt(), 4, SQL_PARAM_INPUT,
                     SQL_C_FLOAT, SQL_REAL, 0, 0, &weight, 0, &weight_ind);
    // NOT MATCHED INSERT from_node_id
    SQLBindParameter(stmt.stmt(), 5, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &fid, 0, &fid_ind);
    // NOT MATCHED INSERT to_node_id
    SQLBindParameter(stmt.stmt(), 6, SQL_PARAM_INPUT,
                     SQL_C_SBIGINT, SQL_BIGINT, 0, 0, &tid, 0, &tid_ind);
    // NOT MATCHED INSERT edge_type
    SQLBindParameter(stmt.stmt(), 7, SQL_PARAM_INPUT,
                     SQL_C_CHAR, SQL_VARCHAR, static_cast<SQLULEN>(edge_type.size()), 0,
                     const_cast<char*>(edge_type.c_str()), static_cast<SQLLEN>(edge_type.size()), &type_ind);
    // NOT MATCHED INSERT weight
    SQLBindParameter(stmt.stmt(), 8, SQL_PARAM_INPUT,
                     SQL_C_FLOAT, SQL_REAL, 0, 0, &weight, 0, &weight_ind);

    bool ok = stmt.execute();
    pool_->release(std::move(conn));
    return ok;
}

} // namespace neural::mssql
