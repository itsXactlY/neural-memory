// mssql.h - MSSQL Vector Storage Adapter
// Production-grade ODBC-based connection to Microsoft SQL Server
// Requires unixODBC: <sql.h>, <sqlext.h>

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <optional>
#include <atomic>
#include <queue>
#include <unordered_map>
#include <span>

// unixODBC headers
#include <sql.h>
#include <sqlext.h>

namespace neural::mssql {

// ============================================================================
// Constants
// ============================================================================

constexpr int DEFAULT_PORT = 1433;
constexpr int DEFAULT_MIN_CONNECTIONS = 2;
constexpr int DEFAULT_MAX_CONNECTIONS = 16;
constexpr int DEFAULT_IDLE_TIMEOUT_SEC = 300;
constexpr int DEFAULT_MAX_LIFETIME_SEC = 3600;
constexpr int DEFAULT_CONNECT_TIMEOUT_SEC = 10;
constexpr int DEFAULT_QUERY_TIMEOUT_SEC = 30;
constexpr int DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5;
constexpr int DEFAULT_CIRCUIT_BREAKER_RESET_SEC = 30;
constexpr size_t DEFAULT_BATCH_SIZE = 1000;

// ============================================================================
// ConnectionConfig
// ============================================================================

struct ConnectionConfig {
    std::string server = "localhost";
    std::string database = "NeuralMemory";
    std::string username;
    std::string password;
    int port = DEFAULT_PORT;
    bool trusted_connection = false;
    std::string driver = "ODBC Driver 18 for SQL Server";
    bool encrypt = true;
    bool trust_server_certificate = false;
    int connect_timeout_sec = DEFAULT_CONNECT_TIMEOUT_SEC;
    int query_timeout_sec = DEFAULT_QUERY_TIMEOUT_SEC;

    // Pool settings
    int min_connections = DEFAULT_MIN_CONNECTIONS;
    int max_connections = DEFAULT_MAX_CONNECTIONS;
    int idle_timeout_sec = DEFAULT_IDLE_TIMEOUT_SEC;
    int max_lifetime_sec = DEFAULT_MAX_LIFETIME_SEC;

    // Circuit breaker
    int circuit_breaker_threshold = DEFAULT_CIRCUIT_BREAKER_THRESHOLD;
    int circuit_breaker_reset_sec = DEFAULT_CIRCUIT_BREAKER_RESET_SEC;

    // Build ODBC connection string
    std::string to_connection_string() const;
};

// ============================================================================
// SearchResult
// ============================================================================

struct SearchResult {
    uint64_t id;
    std::vector<float> vector;
    std::string metadata_json;
    float similarity; // cosine similarity score, 0.0 to 1.0

    bool operator<(const SearchResult& other) const {
        return similarity > other.similarity; // descending order
    }
};

// ============================================================================
// ODBCHandle RAII wrapper
// ============================================================================

class ODBCHandle {
public:
    ODBCHandle() = default;
    ODBCHandle(SQLSMALLINT handle_type, SQLHANDLE input_handle);
    ~ODBCHandle();

    ODBCHandle(const ODBCHandle&) = delete;
    ODBCHandle& operator=(const ODBCHandle&) = delete;
    ODBCHandle(ODBCHandle&& other) noexcept;
    ODBCHandle& operator=(ODBCHandle&& other) noexcept;

    SQLHANDLE get() const { return handle_; }
    operator SQLHANDLE() const { return handle_; }
    explicit operator bool() const { return handle_ != SQL_NULL_HANDLE; }

    void release();

private:
    SQLHANDLE handle_ = SQL_NULL_HANDLE;
    SQLSMALLINT type_ = 0;
};

// ============================================================================
// SQL error extraction
// ============================================================================

struct SQLError {
    std::string state;
    int native_error = 0;
    std::string message;
};

std::vector<SQLError> get_odbc_errors(SQLSMALLINT handle_type, SQLHANDLE handle);

// ============================================================================
// Connection - single ODBC connection
// ============================================================================

class Connection {
public:
    Connection() = default;
    explicit Connection(const ConnectionConfig& config);
    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&& other) noexcept;
    Connection& operator=(Connection&& other) noexcept;

    bool connect();
    void disconnect();
    bool is_connected() const { return connected_; }
    bool is_valid() const;
    bool test_query();

    SQLHDBC dbc() const { return dbc_.get(); }
    SQLHSTMT alloc_statement();

    void begin_transaction();
    void commit_transaction();
    void rollback_transaction();

    std::chrono::steady_clock::time_point created_at() const { return created_at_; }
    std::chrono::steady_clock::time_point last_used_at() const { return last_used_at_; }
    void touch() { last_used_at_ = std::chrono::steady_clock::now(); }

    int failure_count() const { return failure_count_; }
    void record_failure() { ++failure_count_; }
    void reset_failures() { failure_count_ = 0; }

private:
    const ConnectionConfig* config_ = nullptr;
    ODBCHandle env_;
    ODBCHandle dbc_;
    bool connected_ = false;
    std::chrono::steady_clock::time_point created_at_;
    std::chrono::steady_clock::time_point last_used_at_;
    int failure_count_ = 0;
};

// ============================================================================
// ConnectionPool
// ============================================================================

class ConnectionPool {
public:
    explicit ConnectionPool(const ConnectionConfig& config);
    ~ConnectionPool();

    ConnectionPool(const ConnectionPool&) = delete;
    ConnectionPool& operator=(const ConnectionPool&) = delete;

    // Initialize pool with min_connections
    bool initialize();

    // Acquire a connection (blocks if none available, up to timeout)
    std::unique_ptr<Connection> acquire(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

    // Return a connection to the pool
    void release(std::unique_ptr<Connection> conn);

    // Pool stats
    size_t total_connections() const;
    size_t available_connections() const;
    size_t in_use_connections() const;

    // Health check: removes stale connections
    void health_check();

    // Circuit breaker
    bool is_circuit_open() const;
    void record_global_failure();
    void record_global_success();

    // Shutdown: close all connections
    void shutdown();

private:
    std::unique_ptr<Connection> create_connection();
    void evict_stale_connections();
    bool is_connection_expired(const Connection& conn) const;

    ConnectionConfig config_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::unique_ptr<Connection>> available_;
    std::vector<std::unique_ptr<Connection>> all_conns_; // for cleanup tracking
    size_t in_use_count_ = 0;
    bool shutdown_ = false;

    // Circuit breaker
    std::atomic<int> consecutive_failures_{0};
    std::atomic<bool> circuit_open_{false};
    std::chrono::steady_clock::time_point circuit_opened_at_;
};

// ============================================================================
// Statement - RAII SQL statement
// ============================================================================

class Statement {
public:
    explicit Statement(SQLHDBC dbc);
    ~Statement();

    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;
    Statement(Statement&& other) noexcept;
    Statement& operator=(Statement&& other) noexcept;

    SQLHSTMT stmt() const { return stmt_.get(); }
    operator SQLHSTMT() const { return stmt_.get(); }

    bool prepare(const std::string& sql);
    bool execute();
    bool fetch();
    void close_cursor();
    void reset();

    // Parameter binding
    bool bind_param(SQLUSMALLINT index, SQLSMALLINT c_type, SQLSMALLINT sql_type,
                    SQLULEN col_size, SQLLEN* indicator, void* data, SQLLEN data_len);
    bool bind_int64(SQLUSMALLINT index, int64_t& value, SQLLEN& indicator);
    bool bind_string(SQLUSMALLINT index, const std::string& value, SQLLEN& indicator);
    bool bind_binary(SQLUSMALLINT index, const void* data, SQLLEN data_len, SQLLEN& indicator);
    bool bind_float(SQLUSMALLINT index, float& value, SQLLEN& indicator);

    // Column binding for result sets
    bool bind_col(SQLUSMALLINT index, SQLSMALLINT c_type, void* buffer,
                  SQLLEN buf_len, SQLLEN* indicator);

    // Retrieve column data
    std::optional<int64_t> get_int64(SQLUSMALLINT col);
    std::optional<std::string> get_string(SQLUSMALLINT col);
    std::optional<std::vector<uint8_t>> get_binary(SQLUSMALLINT col);
    std::optional<float> get_float(SQLUSMALLINT col);

    // Diagnostics
    std::vector<SQLError> get_errors() const;

private:
    ODBCHandle stmt_;
};

// ============================================================================
// BulkInserter
// ============================================================================

class BulkInserter {
public:
    struct Config {
        size_t batch_size = DEFAULT_BATCH_SIZE;
        bool use_transactions = true;
        bool use_tvp = false;          // Table-Valued Parameters
        std::string tvp_type_name;     // Name of TVP type in SQL Server
    };

    explicit BulkInserter(ConnectionPool& pool);
    explicit BulkInserter(ConnectionPool& pool, const Config& config);
    ~BulkInserter();

    BulkInserter(const BulkInserter&) = delete;
    BulkInserter& operator=(const BulkInserter&) = delete;

    // Initialize prepared statement
    bool prepare(const std::string& insert_sql);

    // Bind parameters for a single row (call before each add_row)
    bool bind_row_params(size_t param_count,
                         const std::vector<SQLSMALLINT>& c_types,
                         const std::vector<SQLSMALLINT>& sql_types,
                         const std::vector<SQLULEN>& col_sizes,
                         const std::vector<void*>& data_ptrs,
                         const std::vector<SQLLEN>& data_lens,
                         std::vector<SQLLEN>& indicators);

    // Add a row and auto-flush when batch is full
    bool add_row();

    // Flush remaining rows
    bool flush();

    // Transaction control
    bool begin_transaction();
    bool commit_transaction();
    bool rollback_transaction();

    // Stats
    size_t rows_inserted() const { return rows_inserted_; }
    size_t batches_sent() const { return batches_sent_; }

private:
    ConnectionPool& pool_;
    Config config_;
    std::unique_ptr<Connection> conn_;
    std::unique_ptr<Statement> stmt_;
    size_t current_batch_ = 0;
    size_t rows_inserted_ = 0;
    size_t batches_sent_ = 0;
    bool in_transaction_ = false;
};

// ============================================================================
// StreamingCursor - for large result sets
// ============================================================================

class StreamingCursor {
public:
    struct Config {
        size_t fetch_buffer_size = 1000; // rows to buffer at a time
    };

    explicit StreamingCursor(ConnectionPool& pool);
    explicit StreamingCursor(ConnectionPool& pool, const Config& config);
    ~StreamingCursor();

    StreamingCursor(const StreamingCursor&) = delete;
    StreamingCursor& operator=(const StreamingCursor&) = delete;

    // Open a cursor with parameterized query
    bool open(const std::string& query_sql,
              const std::vector<SQLSMALLINT>& param_c_types,
              const std::vector<void*>& param_ptrs,
              const std::vector<SQLLEN>& param_indicators);

    // Bind result columns (call after open, before next)
    bool bind_result_col(SQLUSMALLINT index, SQLSMALLINT c_type,
                         void* buffer, SQLLEN buf_len, SQLLEN* indicator);

    // Fetch next row
    bool next();

    // Close cursor
    void close();

    bool is_open() const { return open_; }

private:
    ConnectionPool& pool_;
    Config config_;
    std::unique_ptr<Connection> conn_;
    std::unique_ptr<Statement> stmt_;
    bool open_ = false;
};

// ============================================================================
// MSSQLVectorAdapter - main facade
// ============================================================================

class MSSQLVectorAdapter {
public:
    explicit MSSQLVectorAdapter(const ConnectionConfig& config);
    ~MSSQLVectorAdapter();

    MSSQLVectorAdapter(const MSSQLVectorAdapter&) = delete;
    MSSQLVectorAdapter& operator=(const MSSQLVectorAdapter&) = delete;

    // Initialize: create pool, validate schema
    bool initialize();
    void shutdown();

    // ---- Vector Operations ----

    // Insert a single vector with metadata
    bool insert_vector(uint64_t id, std::span<const float> vector,
                       const std::string& metadata_json);

    // Insert a batch of vectors (uses BulkInserter internally)
    bool insert_vectors(const std::vector<uint64_t>& ids,
                        const std::vector<std::vector<float>>& vectors,
                        const std::vector<std::string>& metadata_jsons);

    // Retrieve a single vector by ID
    std::optional<SearchResult> get_vector(uint64_t id);

    // Retrieve multiple vectors by IDs
    std::vector<SearchResult> get_vectors(const std::vector<uint64_t>& ids);

    // Similarity search: top-K nearest neighbors
    std::vector<SearchResult> search_similar(std::span<const float> query_vector,
                                              int top_k = 10,
                                              float threshold = 0.0f);

    // Search with metadata filter
    std::vector<SearchResult> search_similar_filtered(std::span<const float> query_vector,
                                                       const std::string& metadata_filter_json,
                                                       int top_k = 10,
                                                       float threshold = 0.0f);

    // Delete vector by ID
    bool delete_vector(uint64_t id);

    // Update vector metadata
    bool update_metadata(uint64_t id, const std::string& metadata_json);

    // ---- Graph Operations ----

    bool insert_graph_node(uint64_t node_id, const std::string& node_type,
                           const std::string& properties_json);
    bool insert_graph_edge(uint64_t from_id, uint64_t to_id,
                           const std::string& edge_type, float weight);
    bool delete_graph_node(uint64_t node_id);
    bool delete_graph_edge(uint64_t from_id, uint64_t to_id, const std::string& edge_type);

    // Spreading activation: returns activated nodes with activation levels
    struct ActivationResult {
        uint64_t node_id;
        std::string node_type;
        float activation;
        int depth;
    };
    std::vector<ActivationResult> spreading_activation(
        uint64_t start_node, float decay_factor, float threshold, int max_depth);

    // Consolidation: merge weakly activated memories
    bool run_consolidation(float merge_threshold);

    // ---- Utility ----

    // Count vectors in store
    int64_t count_vectors();
    
    // Get maximum ID (useful for syncing auto-increment sequences)
    int64_t get_max_vector_id();

    // Get pool stats
    size_t pool_available() const;
    size_t pool_in_use() const;

    // ---- Batch Edge Operations ----

    // Batch strengthen edges: UPDATE weight = MIN(weight + delta, 1.0)
    int batch_strengthen_edges(const std::vector<uint64_t>& from_ids,
                               const std::vector<uint64_t>& to_ids,
                               float delta);

    // Bulk weaken + prune: weaken all edges, then delete below threshold
    int bulk_weaken_prune(float delta, float threshold);

    // Edge info struct for get_edges
    struct EdgeInfo {
        uint64_t from_id;
        uint64_t to_id;
        float weight;
    };

    // Get all edges for a node (from OR to)
    std::vector<EdgeInfo> get_edges(uint64_t node_id) const;

    // Count total edges
    int64_t count_edges() const;

    // MERGE/UPSERT: insert edge if not exists, update weight if exists
    bool add_graph_edge_or_update(uint64_t from_id, uint64_t to_id,
                                  const std::string& edge_type, float weight);

    // Compute cosine similarity in C++ (used after fetching VARBINARY)
    static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

    // Convert float vector to VARBINARY bytes
    static std::vector<uint8_t> vector_to_binary(const std::vector<float>& vec);

    // Convert VARBINARY bytes to float vector
    static std::vector<float> binary_to_vector(const std::vector<uint8_t>& data);

private:
    ConnectionConfig config_;
    std::unique_ptr<ConnectionPool> pool_;
    bool initialized_ = false;
};

} // namespace neural::mssql
