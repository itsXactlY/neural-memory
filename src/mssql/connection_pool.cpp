// connection_pool.cpp - Thread-safe ODBC connection pool for MSSQL
// Implements: ODBC connection pooling, health checks, circuit breaker,
// idle timeout, max lifetime eviction.

#include "neural/mssql.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

namespace neural::mssql {

// ============================================================================
// Helpers
// ============================================================================

std::vector<SQLError> get_odbc_errors(SQLSMALLINT handle_type, SQLHANDLE handle) {
    std::vector<SQLError> errors;
    SQLINTEGER i = 1;
    SQLCHAR state[6];
    SQLINTEGER native_error;
    SQLCHAR message[1024];
    SQLSMALLINT msg_len;

    while (SQLGetDiagRecA(handle_type, handle, i, state, &native_error,
                          message, sizeof(message), &msg_len) == SQL_SUCCESS) {
        SQLError err;
        err.state = reinterpret_cast<char*>(state);
        err.native_error = static_cast<int>(native_error);
        err.message = std::string(reinterpret_cast<char*>(message), msg_len);
        errors.push_back(std::move(err));
        ++i;
    }
    return errors;
}

// ============================================================================
// ODBCHandle RAII
// ============================================================================

ODBCHandle::ODBCHandle(SQLSMALLINT handle_type, SQLHANDLE input_handle)
    : type_(handle_type) {
    SQLRETURN rc = SQLAllocHandle(handle_type, input_handle, &handle_);
    if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
        handle_ = SQL_NULL_HANDLE;
    }
}

ODBCHandle::~ODBCHandle() {
    release();
}

ODBCHandle::ODBCHandle(ODBCHandle&& other) noexcept
    : handle_(other.handle_), type_(other.type_) {
    other.handle_ = SQL_NULL_HANDLE;
    other.type_ = 0;
}

ODBCHandle& ODBCHandle::operator=(ODBCHandle&& other) noexcept {
    if (this != &other) {
        release();
        handle_ = other.handle_;
        type_ = other.type_;
        other.handle_ = SQL_NULL_HANDLE;
        other.type_ = 0;
    }
    return *this;
}

void ODBCHandle::release() {
    if (handle_ != SQL_NULL_HANDLE) {
        SQLFreeHandle(type_, handle_);
        handle_ = SQL_NULL_HANDLE;
    }
}

// ============================================================================
// ConnectionConfig
// ============================================================================

std::string ConnectionConfig::to_connection_string() const {
    std::ostringstream oss;
    
    // Clean driver string (remove existing braces if present)
    std::string clean_driver = driver;
    if (!clean_driver.empty() && clean_driver.front() == '{' && clean_driver.back() == '}') {
        clean_driver = clean_driver.substr(1, clean_driver.length() - 2);
    }
    
    oss << "DRIVER={" << clean_driver << "};"
        << "SERVER=" << server << "," << port << ";"
        << "DATABASE=" << database << ";";

    if (trusted_connection) {
        oss << "Trusted_Connection=yes;";
    } else {
        oss << "UID=" << username << ";";
        
        // Escape password if it contains special characters
        std::string clean_pwd = password;
        if (!clean_pwd.empty()) {
            // If the user's dotenv somehow bypassed quotes and left literal single quotes at the ends
            if (clean_pwd.front() == '\'' && clean_pwd.back() == '\'') {
                clean_pwd = clean_pwd.substr(1, clean_pwd.length() - 2);
            }
            
            // ODBC escaping: wrap in {} and double any inner }
            std::string escaped_pwd;
            for (char c : clean_pwd) {
                if (c == '}') escaped_pwd += "}}";
                else escaped_pwd += c;
            }
            oss << "PWD={" << escaped_pwd << "};";
        } else {
            oss << "PWD=;";
        }
    }

    if (encrypt) {
        oss << "Encrypt=yes;";
    }
    if (trust_server_certificate) {
        oss << "TrustServerCertificate=yes;";
    }

    oss << "Connection Timeout=" << connect_timeout_sec << ";";
    return oss.str();
}

// ============================================================================
// Connection
// ============================================================================

Connection::Connection(const ConnectionConfig& config)
    : config_(&config),
      created_at_(std::chrono::steady_clock::now()),
      last_used_at_(std::chrono::steady_clock::now()) {}

Connection::~Connection() {
    disconnect();
}

Connection::Connection(Connection&& other) noexcept
    : config_(other.config_),
      env_(std::move(other.env_)),
      dbc_(std::move(other.dbc_)),
      connected_(other.connected_),
      created_at_(other.created_at_),
      last_used_at_(other.last_used_at_),
      failure_count_(other.failure_count_) {
    other.connected_ = false;
    other.config_ = nullptr;
}

Connection& Connection::operator=(Connection&& other) noexcept {
    if (this != &other) {
        disconnect();
        config_ = other.config_;
        env_ = std::move(other.env_);
        dbc_ = std::move(other.dbc_);
        connected_ = other.connected_;
        created_at_ = other.created_at_;
        last_used_at_ = other.last_used_at_;
        failure_count_ = other.failure_count_;
        other.connected_ = false;
        other.config_ = nullptr;
    }
    return *this;
}

bool Connection::connect() {
    if (!config_) return false;

    // Allocate environment
    env_ = ODBCHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE);
    if (!env_) return false;

    SQLRETURN rc = SQLSetEnvAttr(env_.get(), SQL_ATTR_ODBC_VERSION,
                                  (SQLPOINTER)SQL_OV_ODBC3, 0);
    if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) return false;

    // Enable ODBC connection pooling on the environment
    SQLSetEnvAttr(env_.get(), SQL_ATTR_CONNECTION_POOLING,
                  (SQLPOINTER)SQL_CP_ONE_PER_HENV, 0);
    SQLSetEnvAttr(env_.get(), SQL_ATTR_CP_MATCH,
                  (SQLPOINTER)SQL_CP_STRICT_MATCH, 0);

    // Allocate connection handle
    dbc_ = ODBCHandle(SQL_HANDLE_DBC, env_.get());
    if (!dbc_) return false;

    // Set login timeout
    SQLSetConnectAttr(dbc_.get(), SQL_ATTR_LOGIN_TIMEOUT,
                      (SQLPOINTER)(intptr_t)config_->connect_timeout_sec, 0);

    // Set query timeout on connection level
    SQLSetConnectAttr(dbc_.get(), SQL_ATTR_QUERY_TIMEOUT,
                      (SQLPOINTER)(intptr_t)config_->query_timeout_sec, 0);

    // Connect
    std::string conn_str = config_->to_connection_string();
    SQLCHAR out_str[1024];
    SQLSMALLINT out_len;

    rc = SQLDriverConnectA(dbc_.get(), NULL,
                           (SQLCHAR*)conn_str.c_str(), SQL_NTS,
                           out_str, sizeof(out_str), &out_len,
                           SQL_DRIVER_NOPROMPT);

    if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
        std::cerr << "[ODBC ERROR] Connection failed for string: " << conn_str << std::endl;
        auto errors = get_odbc_errors(SQL_HANDLE_DBC, dbc_.get());
        for (const auto& err : errors) {
            std::cerr << "State: " << err.state << ", Native: " << err.native_error 
                      << ", Message: " << err.message << std::endl;
        }
        dbc_.release();
        env_.release();
        return false;
    }

    connected_ = true;
    failure_count_ = 0;
    return true;
}

void Connection::disconnect() {
    if (connected_ && dbc_) {
        SQLDisconnect(dbc_.get());
    }
    dbc_.release();
    env_.release();
    connected_ = false;
}

bool Connection::is_valid() const {
    if (!connected_ || !dbc_) return false;
    SQLUINTEGER dead_flag = 0;
    SQLRETURN rc = SQLGetConnectAttr(dbc_.get(), SQL_ATTR_CONNECTION_DEAD,
                                      &dead_flag, 0, nullptr);
    return (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO) && dead_flag == SQL_CD_FALSE;
}

bool Connection::test_query() {
    if (!connected_ || !dbc_) return false;

    ODBCHandle stmt(SQL_HANDLE_STMT, dbc_.get());
    if (!stmt) return false;

    SQLRETURN rc = SQLExecDirectA(stmt.get(),
                                   (SQLCHAR*)"SELECT 1", SQL_NTS);
    return (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO);
}

SQLHSTMT Connection::alloc_statement() {
    SQLHSTMT stmt = SQL_NULL_HSTMT;
    SQLRETURN rc = SQLAllocHandle(SQL_HANDLE_STMT, dbc_.get(), &stmt);
    if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
        return SQL_NULL_HSTMT;
    }
    return stmt;
}

void Connection::begin_transaction() {
    if (!dbc_) return;
    // Set autocommit off
    SQLSetConnectAttr(dbc_.get(), SQL_ATTR_AUTOCOMMIT,
                      (SQLPOINTER)SQL_AUTOCOMMIT_OFF, 0);
}

void Connection::commit_transaction() {
    if (!dbc_) return;
    SQLEndTran(SQL_HANDLE_DBC, dbc_.get(), SQL_COMMIT);
    // Restore autocommit
    SQLSetConnectAttr(dbc_.get(), SQL_ATTR_AUTOCOMMIT,
                      (SQLPOINTER)SQL_AUTOCOMMIT_ON, 0);
}

void Connection::rollback_transaction() {
    if (!dbc_) return;
    SQLEndTran(SQL_HANDLE_DBC, dbc_.get(), SQL_ROLLBACK);
    // Restore autocommit
    SQLSetConnectAttr(dbc_.get(), SQL_ATTR_AUTOCOMMIT,
                      (SQLPOINTER)SQL_AUTOCOMMIT_ON, 0);
}

// ============================================================================
// Statement
// ============================================================================

Statement::Statement(SQLHDBC dbc) {
    SQLHSTMT handle = SQL_NULL_HSTMT;
    SQLRETURN rc = SQLAllocHandle(SQL_HANDLE_STMT, dbc, &handle);
    if (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO) {
        stmt_ = ODBCHandle(SQL_HANDLE_STMT, SQL_NULL_HANDLE);
        // Manually move the handle into our ODBCHandle
        // We need to bypass the allocate in ODBCHandle constructor
        // Since we already have the handle, we'll use a different approach
    }
    // Actually, let's do this properly
    stmt_ = ODBCHandle(SQL_HANDLE_STMT, dbc);
}

Statement::~Statement() {}

Statement::Statement(Statement&& other) noexcept
    : stmt_(std::move(other.stmt_)) {}

Statement& Statement::operator=(Statement&& other) noexcept {
    if (this != &other) {
        stmt_ = std::move(other.stmt_);
    }
    return *this;
}

bool Statement::prepare(const std::string& sql) {
    if (!stmt_) return false;
    SQLRETURN rc = SQLPrepareA(stmt_.get(), (SQLCHAR*)sql.c_str(), SQL_NTS);
    return (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO);
}

bool Statement::execute() {
    if (!stmt_) return false;
    SQLRETURN rc = SQLExecute(stmt_.get());
    return (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO ||
            rc == SQL_NO_DATA);
}

bool Statement::fetch() {
    if (!stmt_) return false;
    SQLRETURN rc = SQLFetch(stmt_.get());
    return (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO);
}

void Statement::close_cursor() {
    if (stmt_) SQLCloseCursor(stmt_.get());
}

void Statement::reset() {
    if (stmt_) {
        SQLFreeStmt(stmt_.get(), SQL_CLOSE);
        SQLFreeStmt(stmt_.get(), SQL_RESET_PARAMS);
    }
}

bool Statement::bind_param(SQLUSMALLINT index, SQLSMALLINT c_type, SQLSMALLINT sql_type,
                           SQLULEN col_size, SQLLEN* indicator, void* data, SQLLEN data_len) {
    if (!stmt_) return false;
    SQLRETURN rc = SQLBindParameter(stmt_.get(), index, SQL_PARAM_INPUT,
                                     c_type, sql_type, col_size, 0,
                                     data, data_len, indicator);
    return (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO);
}

bool Statement::bind_int64(SQLUSMALLINT index, int64_t& value, SQLLEN& indicator) {
    return bind_param(index, SQL_C_SBIGINT, SQL_BIGINT, 0, &indicator, &value, 0);
}

bool Statement::bind_string(SQLUSMALLINT index, const std::string& value, SQLLEN& indicator) {
    // String binding - note: we need a non-const pointer for ODBC
    // The caller must ensure the data persists
    indicator = static_cast<SQLLEN>(value.size());
    // For SQL_NTS, we can pass SQL_NTS as indicator instead
    // But for parameterized queries we need exact length
    return bind_param(index, SQL_C_CHAR, SQL_VARCHAR,
                      static_cast<SQLULEN>(value.size()),
                      &indicator,
                      const_cast<char*>(value.c_str()),
                      static_cast<SQLLEN>(value.size()));
}

bool Statement::bind_binary(SQLUSMALLINT index, const void* data, SQLLEN data_len, SQLLEN& indicator) {
    indicator = data_len;
    return bind_param(index, SQL_C_BINARY, SQL_VARBINARY,
                      static_cast<SQLULEN>(data_len),
                      &indicator,
                      const_cast<void*>(data),
                      data_len);
}

bool Statement::bind_float(SQLUSMALLINT index, float& value, SQLLEN& indicator) {
    return bind_param(index, SQL_C_FLOAT, SQL_REAL, 0, &indicator, &value, 0);
}

bool Statement::bind_col(SQLUSMALLINT index, SQLSMALLINT c_type, void* buffer,
                          SQLLEN buf_len, SQLLEN* indicator) {
    if (!stmt_) return false;
    SQLRETURN rc = SQLBindCol(stmt_.get(), index, c_type, buffer, buf_len, indicator);
    return (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO);
}

std::optional<int64_t> Statement::get_int64(SQLUSMALLINT col) {
    int64_t value = 0;
    SQLLEN indicator = 0;
    SQLRETURN rc = SQLGetData(stmt_.get(), col, SQL_C_SBIGINT, &value, 0, &indicator);
    if (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO) {
        if (indicator == SQL_NULL_DATA) return std::nullopt;
        return value;
    }
    return std::nullopt;
}

std::optional<std::string> Statement::get_string(SQLUSMALLINT col) {
    char buf[4096];
    SQLLEN indicator = 0;
    std::string result;

    while (true) {
        SQLRETURN rc = SQLGetData(stmt_.get(), col, SQL_C_CHAR, buf, sizeof(buf), &indicator);
        if (indicator == SQL_NULL_DATA) return std::nullopt;
        if (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO) {
            result += buf;
            if (rc == SQL_SUCCESS) break;
        } else {
            break;
        }
    }
    return result;
}

std::optional<std::vector<uint8_t>> Statement::get_binary(SQLUSMALLINT col) {
    uint8_t buf[8192];
    SQLLEN indicator = 0;
    std::vector<uint8_t> result;

    while (true) {
        SQLRETURN rc = SQLGetData(stmt_.get(), col, SQL_C_BINARY, buf, sizeof(buf), &indicator);
        if (indicator == SQL_NULL_DATA) return std::nullopt;
        if (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO) {
            SQLLEN bytes_read = (indicator > (SQLLEN)sizeof(buf)) ? (SQLLEN)sizeof(buf) : indicator;
            if (indicator <= (SQLLEN)sizeof(buf)) {
                result.insert(result.end(), buf, buf + indicator);
            } else {
                // First chunk returned is (buf_size - 1) bytes for some drivers
                size_t chunk = std::strlen(reinterpret_cast<char*>(buf));
                if (chunk == 0 && result.empty()) {
                    result.insert(result.end(), buf, buf + sizeof(buf) - 1);
                } else {
                    result.insert(result.end(), buf, buf + sizeof(buf) - 1);
                }
            }
            if (rc == SQL_SUCCESS) break;
        } else {
            break;
        }
    }
    return result;
}

std::optional<float> Statement::get_float(SQLUSMALLINT col) {
    float value = 0.0f;
    SQLLEN indicator = 0;
    SQLRETURN rc = SQLGetData(stmt_.get(), col, SQL_C_FLOAT, &value, 0, &indicator);
    if (rc == SQL_SUCCESS || rc == SQL_SUCCESS_WITH_INFO) {
        if (indicator == SQL_NULL_DATA) return std::nullopt;
        return value;
    }
    return std::nullopt;
}

std::vector<SQLError> Statement::get_errors() const {
    return get_odbc_errors(SQL_HANDLE_STMT, stmt_.get());
}

// ============================================================================
// ConnectionPool
// ============================================================================

ConnectionPool::ConnectionPool(const ConnectionConfig& config)
    : config_(config) {}

ConnectionPool::~ConnectionPool() {
    shutdown();
}

bool ConnectionPool::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_) return false;

    for (int i = 0; i < config_.min_connections; ++i) {
        auto conn = create_connection();
        if (!conn) return false;
        available_.push(std::move(conn));
    }
    return true;
}

std::unique_ptr<Connection> ConnectionPool::acquire(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (is_circuit_open()) {
        return nullptr;
    }

    auto deadline = std::chrono::steady_clock::now() + timeout;

    while (true) {
        if (shutdown_) return nullptr;

        // Evict stale connections
        evict_stale_connections();

        if (!available_.empty()) {
            auto conn = std::move(available_.front());
            available_.pop();

            // Validate connection
            if (!conn->is_connected() || !conn->test_query()) {
                // This connection is dead, try creating a new one
                conn = create_connection();
                if (!conn) {
                    cv_.notify_one();
                    return nullptr;
                }
            }

            ++in_use_count_;
            conn->touch();
            return conn;
        }

        // Try to create a new connection if under max
        size_t total = available_.size() + in_use_count_;
        if (total < static_cast<size_t>(config_.max_connections)) {
            auto conn = create_connection();
            if (conn) {
                ++in_use_count_;
                conn->touch();
                return conn;
            }
        }

        // Wait for a connection to be released
        if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            return nullptr;
        }
    }
}

void ConnectionPool::release(std::unique_ptr<Connection> conn) {
    if (!conn) return;

    std::lock_guard<std::mutex> lock(mutex_);

    if (shutdown_) {
        conn->disconnect();
        return;
    }

    if (conn->failure_count() > 0) {
        // Too many failures, discard this connection
        conn->disconnect();
    } else if (!conn->is_connected() || !conn->is_valid()) {
        conn->disconnect();
    } else {
        conn->touch();
        available_.push(std::move(conn));
    }

    if (in_use_count_ > 0) --in_use_count_;
    cv_.notify_one();
}

size_t ConnectionPool::total_connections() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_.size() + in_use_count_;
}

size_t ConnectionPool::available_connections() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_.size();
}

size_t ConnectionPool::in_use_connections() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return in_use_count_;
}

void ConnectionPool::health_check() {
    std::lock_guard<std::mutex> lock(mutex_);
    evict_stale_connections();

    // Ensure minimum connections
    while (available_.size() + in_use_count_ < static_cast<size_t>(config_.min_connections)) {
        auto conn = create_connection();
        if (!conn) break;
        available_.push(std::move(conn));
    }
}

bool ConnectionPool::is_circuit_open() const {
    if (!circuit_open_.load()) return false;

    auto elapsed = std::chrono::steady_clock::now() - circuit_opened_at_;
    if (elapsed > std::chrono::seconds(config_.circuit_breaker_reset_sec)) {
        // Reset circuit breaker
        const_cast<std::atomic<bool>&>(circuit_open_).store(false);
        const_cast<std::atomic<int>&>(consecutive_failures_).store(0);
        return false;
    }
    return true;
}

void ConnectionPool::record_global_failure() {
    int failures = consecutive_failures_.fetch_add(1) + 1;
    if (failures >= config_.circuit_breaker_threshold) {
        circuit_open_.store(true);
        circuit_opened_at_ = std::chrono::steady_clock::now();
    }
}

void ConnectionPool::record_global_success() {
    consecutive_failures_.store(0);
    circuit_open_.store(false);
}

void ConnectionPool::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;

    while (!available_.empty()) {
        auto& conn = available_.front();
        if (conn) conn->disconnect();
        available_.pop();
    }
    in_use_count_ = 0;
    cv_.notify_all();
}

std::unique_ptr<Connection> ConnectionPool::create_connection() {
    auto conn = std::make_unique<Connection>(config_);
    if (!conn->connect()) {
        record_global_failure();
        return nullptr;
    }
    record_global_success();
    return conn;
}

void ConnectionPool::evict_stale_connections() {
    auto now = std::chrono::steady_clock::now();
    std::queue<std::unique_ptr<Connection>> kept;

    while (!available_.empty()) {
        auto conn = std::move(available_.front());
        available_.pop();

        if (is_connection_expired(*conn)) {
            conn->disconnect();
        } else {
            kept.push(std::move(conn));
        }
    }
    available_ = std::move(kept);
}

bool ConnectionPool::is_connection_expired(const Connection& conn) const {
    auto now = std::chrono::steady_clock::now();

    // Check max lifetime
    auto age = now - conn.created_at();
    if (age > std::chrono::seconds(config_.max_lifetime_sec)) return true;

    // Check idle timeout
    auto idle = now - conn.last_used_at();
    if (idle > std::chrono::seconds(config_.idle_timeout_sec)) return true;

    return false;
}

} // namespace neural::mssql
