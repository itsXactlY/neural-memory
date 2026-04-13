// neural/core/memory_adapter.cpp - Integration Implementation
#include "neural/memory_adapter.h"
#include <chrono>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace neural {

using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;

// ============================================================================
// Lifecycle
// ============================================================================

NeuralMemoryAdapter::~NeuralMemoryAdapter() {
    shutdown();
}

bool NeuralMemoryAdapter::initialize(const AdapterConfig& config) {
    if (initialized_) return false;
    
    config_ = config;
    
    // Initialize Hopfield layer
    HopfieldConfig hcfg;
    hcfg.dimensions = config.vector_dim;
    hcfg.capacity = config.hopfield_capacity;
    hcfg.beta = config.hopfield_beta;
    hcfg.decay_rate = config.hopfield_decay;
    hopfield_ = std::make_unique<HopfieldLayer>(hcfg);
    
    // Initialize Memory Manager
    memory_manager_ = std::make_unique<memory::MemoryManager>(
        config.vector_dim,
        config.episodic_capacity,
        config.semantic_capacity
    );
    
    // Initialize Knowledge Graph
    graph_ = std::make_unique<graph::KnowledgeGraph>();
    
    // Initialize MSSQL (optional - skip if no server configured)
#ifdef USE_MSSQL
    if (!config.db_config.server.empty()) {
        db_ = std::make_unique<mssql::MSSQLVectorAdapter>(config.db_config);
        db_->initialize();
        
        // Sync ID sequence with the database max ID
        int64_t max_id = db_->get_max_vector_id();
        if (max_id > 0) {
            memory_manager_->set_next_id(max_id + 1);
        }
    }
#endif
    
    // Start background threads
    running_ = true;
    
    if (config.enable_consolidation_thread) {
        consolidation_thread_ = std::thread(&NeuralMemoryAdapter::consolidation_worker, this);
    }
    if (config.enable_decay_thread) {
        decay_thread_ = std::thread(&NeuralMemoryAdapter::decay_worker, this);
    }
    if (config.enable_link_prediction) {
        link_prediction_thread_ = std::thread(&NeuralMemoryAdapter::link_prediction_worker, this);
    }
    
    initialized_ = true;
    return true;
}

void NeuralMemoryAdapter::shutdown() {
    if (!initialized_) return;
    
    running_ = false;
    
    if (consolidation_thread_.joinable()) consolidation_thread_.join();
    if (decay_thread_.joinable()) decay_thread_.join();
    if (link_prediction_thread_.joinable()) link_prediction_thread_.join();
    
    // Final consolidation
    consolidate();
    
#ifdef USE_MSSQL
    if (db_) db_->shutdown();
#endif

    hopfield_.reset();
    memory_manager_.reset();
    graph_.reset();
#ifdef USE_MSSQL
    db_.reset();
#endif
    
    initialized_ = false;
}

// ============================================================================
// Core Operations
// ============================================================================

uint64_t NeuralMemoryAdapter::store(const std::vector<float>& embedding,
                                     const std::string& label,
                                     const std::string& content,
                                     const std::string& source) {
    auto start = Clock::now();
    
    // 1. Store in episodic memory (fast write)
    uint64_t id = memory_manager_->remember(embedding, label, content);
    
    // 2. Store in Hopfield layer
    hopfield_->store(embedding, label, source);
    
    // 3. Create graph node
    uint64_t node_id = graph_->add_node(
        graph::NodeType::Memory, label, id
    );
    graph_->set_embedding(node_id, embedding);
    
    // 4. Find and create edges to similar existing memories
    auto similar = memory_manager_->recall(embedding, 5);
    for (const auto& [sim_id, sim_score] : similar) {
        if (sim_id == id) continue;
        if (sim_score > 0.15f) {
            // Find or create node for this memory
            // For simplicity, use heuristic mapping
            graph_->add_edge(node_id, sim_id + 1,  // +1 for node_id offset
                graph::EdgeType::Similar, sim_score);
        }
    }
    
    // 5. Store in MSSQL (if available)
#ifdef USE_MSSQL
    if (db_) {
        std::string metadata = "{\"label\":\"" + label + "\",\"source\":\"" + source + "\"}";
        db_->insert_vector(id, embedding, metadata);
    }
#endif
    
    auto end = Clock::now();
    auto us = std::chrono::duration_cast<Microseconds>(end - start).count();
    
    {
        std::unique_lock lock(stats_mutex_);
        total_stores_++;
        total_store_us_ += us;
    }
    
    return id;
}

uint64_t NeuralMemoryAdapter::store_text(const std::string& text,
                                          const std::string& label) {
    auto embedding = text_to_embedding(text);
    return store(embedding, label, text, "text");
}

std::vector<MemoryResult> NeuralMemoryAdapter::retrieve(
    const std::vector<float>& cue, size_t k) const
{
    auto start = Clock::now();
    
    std::vector<MemoryResult> results;
    
    // 1. Hopfield pattern completion
    auto hopfield_result = hopfield_->retrieve(cue);
    if (hopfield_result.converged) {
        // Use completed pattern for semantic search
        auto semantic = memory_manager_->recall(hopfield_result.pattern, k);
        for (const auto& [id, score] : semantic) {
            auto entry = memory_manager_->read(id);
            if (entry) {
                results.push_back(to_result(*entry, score));
            }
        }
    }
    
    // 2. Direct memory search (fallback / supplement)
    if (results.size() < k) {
        auto direct = memory_manager_->recall(cue, k - results.size());
        for (const auto& [id, score] : direct) {
            // Skip duplicates
            bool found = false;
            for (const auto& r : results) {
                if (r.id == id) { found = true; break; }
            }
            if (found) continue;
            
            auto entry = memory_manager_->read(id);
            if (entry) {
                results.push_back(to_result(*entry, score));
            }
        }
    }
    
    // 3. Spreading activation enrichment
    if (!results.empty() && results[0].similarity > 0.5f) {
        // Find connected memories via graph
        auto activated = graph_->spread_activation(
            results[0].id + 1,  // +1 for node offset
            config_.graph_decay,
            config_.graph_threshold,
            2  // shallow
        );
        
        for (const auto& tr : activated) {
            if (tr.node_id > 1) {  // Skip node 0
                uint64_t mem_id = tr.node_id - 1;
                auto entry = memory_manager_->read(mem_id);
                if (entry) {
                    // Check not already in results
                    bool found = false;
                    for (const auto& r : results) {
                        if (r.id == mem_id) { found = true; break; }
                    }
                    if (!found) {
                        auto mr = to_result(*entry, tr.activation);
                        results.push_back(mr);
                    }
                }
            }
        }
    }
    
    // Sort by similarity and limit
    std::sort(results.begin(), results.end(),
        [](const MemoryResult& a, const MemoryResult& b) {
            return a.similarity > b.similarity;
        });
    if (results.size() > k) results.resize(k);
    
    auto end = Clock::now();
    auto us = std::chrono::duration_cast<Microseconds>(end - start).count();
    
    {
        std::unique_lock lock(stats_mutex_);
        total_retrieves_++;
        total_retrieve_us_ += us;
    }
    
    return results;
}

std::vector<MemoryResult> NeuralMemoryAdapter::retrieve_text(
    const std::string& query, size_t k) const
{
    auto embedding = text_to_embedding(query);
    return retrieve(embedding, k);
}

std::vector<MemoryResult> NeuralMemoryAdapter::search(
    const std::string& query, size_t k) const
{
    auto start = Clock::now();
    
    // Simple text search: linear scan over stored content
    std::vector<MemoryResult> results;
    
    // Convert query to embedding for semantic search
    auto query_emb = text_to_embedding(query);
    auto semantic = memory_manager_->recall(query_emb, k * 2);
    
    for (const auto& [id, score] : semantic) {
        auto entry = memory_manager_->read(id);
        if (entry) {
            // Boost score if query text appears in content
            float text_boost = 0.0f;
            if (!entry->content.empty() && entry->content.find(query) != std::string::npos) {
                text_boost = 0.2f;
            }
            if (!entry->label.empty() && entry->label.find(query) != std::string::npos) {
                text_boost = 0.3f;
            }
            
            auto mr = to_result(*entry, score + text_boost);
            results.push_back(mr);
        }
    }
    
    std::sort(results.begin(), results.end(),
        [](const MemoryResult& a, const MemoryResult& b) {
            return a.similarity > b.similarity;
        });
    if (results.size() > k) results.resize(k);
    
    auto end = Clock::now();
    auto us = std::chrono::duration_cast<Microseconds>(end - start).count();
    
    {
        std::unique_lock lock(stats_mutex_);
        total_searches_++;
        total_search_us_ += us;
    }
    
    return results;
}

std::optional<MemoryResult> NeuralMemoryAdapter::read(uint64_t id) const {
    auto entry = memory_manager_->read(id);
    if (!entry) return std::nullopt;
    return to_result(*entry, 1.0f);
}

// ============================================================================
// Graph Operations
// ============================================================================

std::vector<std::pair<uint64_t, float>> NeuralMemoryAdapter::connections(
    uint64_t id) const
{
    return graph_->neighbors(id + 1);  // +1 for node offset
}

std::vector<graph::TraversalResult> NeuralMemoryAdapter::think(
    uint64_t start_id, int depth) const
{
    return graph_->spread_activation(
        start_id + 1,
        config_.graph_decay,
        config_.graph_threshold,
        depth
    );
}

std::optional<std::vector<uint64_t>> NeuralMemoryAdapter::path(
    uint64_t from, uint64_t to) const
{
    auto raw_path = graph_->shortest_path(from + 1, to + 1);
    if (!raw_path) return std::nullopt;
    
    // Convert node IDs back to memory IDs
    std::vector<uint64_t> result;
    for (uint64_t node_id : *raw_path) {
        if (node_id > 0) result.push_back(node_id - 1);
    }
    return result;
}

std::vector<graph::ConnectionPrediction> NeuralMemoryAdapter::predicted_connections(
    size_t max_results) const
{
    return graph_->predict_links(max_results);
}

// ============================================================================
// Consolidation
// ============================================================================

size_t NeuralMemoryAdapter::consolidate() {
    auto start = Clock::now();
    
    // 1. Run memory consolidation (episodic -> semantic)
    // Note: consolidate(batch_size) takes size_t, not float threshold.
    // The threshold is used for auto-trigger check, not as batch parameter.
    size_t consolidated = memory_manager_->consolidate(64);
    
    // 2. Update graph centrality
    graph_->update_centrality();
    
    auto end = Clock::now();
    auto us = std::chrono::duration_cast<Microseconds>(end - start).count();
    
    {
        std::unique_lock lock(stats_mutex_);
        total_consolidations_++;
        last_consolidation_us_ = us;
    }
    
    return consolidated;
}

void NeuralMemoryAdapter::decay() {
    // Decay memory salience
    memory_manager_->apply_decay(config_.hopfield_decay);
    
    // Decay graph edges
    graph_->decay_edges(config_.hopfield_decay);
    
    // Prune very weak edges
    graph_->prune_weak_edges(config_.edge_prune_threshold);
}

size_t NeuralMemoryAdapter::predict_links() {
    auto predictions = graph_->predict_links(100);
    
    size_t added = 0;
    for (const auto& pred : predictions) {
        if (pred.confidence > 0.3f) {
            graph_->add_edge(pred.source_id, pred.target_id,
                graph::EdgeType::Inferred, pred.confidence);
            added++;
        }
    }
    
    return added;
}

// ============================================================================
// Statistics
// ============================================================================

StatsSnapshot NeuralMemoryAdapter::get_stats() const {
    std::shared_lock lock(stats_mutex_);
    
    StatsSnapshot stats;
    
    // Memory
    stats.episodic_count = 0;  // Would need to expose from memory_manager_
    stats.semantic_count = 0;
    stats.episodic_occupancy = memory_manager_->episodic_occupancy();
    stats.semantic_occupancy = 0;  // Would need expose
    
    // Hopfield
    stats.hopfield_patterns = hopfield_->pattern_count();
    stats.hopfield_occupancy = hopfield_->occupancy();
    
    // Graph
    auto gs = graph_->get_stats();
    stats.graph_nodes = gs.nodes;
    stats.graph_edges = gs.edges;
    stats.graph_density = gs.density;
    
    // Performance
    stats.avg_store_us = total_stores_ > 0 ? total_store_us_ / total_stores_ : 0;
    stats.avg_retrieve_us = total_retrieves_ > 0 ? total_retrieve_us_ / total_retrieves_ : 0;
    stats.avg_search_us = total_searches_ > 0 ? total_search_us_ / total_searches_ : 0;
    stats.last_consolidation_us = last_consolidation_us_;
    
    // Operations
    stats.total_stores = total_stores_;
    stats.total_retrieves = total_retrieves_;
    stats.total_searches = total_searches_;
    stats.total_consolidations = total_consolidations_;
    
    return stats;
}

// ============================================================================
// Configuration
// ============================================================================

void NeuralMemoryAdapter::set_beta(float beta) {
    hopfield_->set_beta(beta);
    config_.hopfield_beta = beta;
}

float NeuralMemoryAdapter::get_beta() const {
    return hopfield_->get_beta();
}

void NeuralMemoryAdapter::set_consolidation_threshold(float threshold) {
    config_.consolidation_threshold = threshold;
}

// ============================================================================
// Background Workers
// ============================================================================

void NeuralMemoryAdapter::consolidation_worker() {
    while (running_) {
        std::this_thread::sleep_for(
            std::chrono::seconds(config_.consolidation_interval_sec)
        );
        if (!running_) break;
        
        // Check if consolidation is needed
        if (memory_manager_->episodic_occupancy() > config_.consolidation_threshold) {
            consolidate();
        }
    }
}

void NeuralMemoryAdapter::decay_worker() {
    while (running_) {
        std::this_thread::sleep_for(
            std::chrono::seconds(config_.decay_interval_sec)
        );
        if (!running_) break;
        
        decay();
    }
}

void NeuralMemoryAdapter::link_prediction_worker() {
    while (running_) {
        std::this_thread::sleep_for(
            std::chrono::seconds(config_.link_prediction_interval_sec)
        );
        if (!running_) break;
        
        predict_links();
    }
}

// ============================================================================
// MSSQL Graph Edge Operations (delegate to db_)
// ============================================================================

#ifdef USE_MSSQL

uint64_t NeuralMemoryAdapter::store_mssql(const std::vector<float>& embedding,
                                           const std::string& label,
                                           const std::string& content) {
    if (!db_) return 0;

    // Generate a unique ID
    uint64_t id = memory_manager_->remember(embedding, label, content);

    // Store vector in MSSQL
    std::string metadata = "{\"label\":\"" + label + "\",\"source\":\"mssql\"}";
    if (!db_->insert_vector(id, embedding, metadata)) return 0;

    // Create graph node
    if (!db_->insert_graph_node(id, "memory",
                                 "{\"label\":\"" + label + "\"}")) return 0;

    return id;
}

bool NeuralMemoryAdapter::add_edge(uint64_t from_id, uint64_t to_id, float weight,
                                    const std::string& edge_type) {
    if (!db_) return false;
    return db_->insert_graph_edge(from_id, to_id, edge_type, weight);
}

int NeuralMemoryAdapter::batch_add_edges(const std::vector<uint64_t>& from_ids,
                                          const std::vector<uint64_t>& to_ids,
                                          const std::vector<float>& weights,
                                          const std::string& edge_type) {
    if (!db_ || from_ids.empty() || from_ids.size() != to_ids.size() ||
        from_ids.size() != weights.size())
        return 0;

    int added = 0;
    for (size_t i = 0; i < from_ids.size(); ++i) {
        if (db_->insert_graph_edge(from_ids[i], to_ids[i], edge_type, weights[i])) {
            added++;
        }
    }
    return added;
}

int NeuralMemoryAdapter::batch_strengthen_edges(const std::vector<uint64_t>& from_ids,
                                                 const std::vector<uint64_t>& to_ids,
                                                 float delta) {
    if (!db_) return 0;
    return db_->batch_strengthen_edges(from_ids, to_ids, delta);
}

int NeuralMemoryAdapter::bulk_weaken_prune(float delta, float threshold) {
    if (!db_) return 0;
    return db_->bulk_weaken_prune(delta, threshold);
}

std::vector<NeuralMemoryAdapter::EdgeInfo> NeuralMemoryAdapter::get_edges(uint64_t node_id) const {
    if (!db_) return {};
    auto db_edges = db_->get_edges(node_id);
    std::vector<EdgeInfo> result;
    result.reserve(db_edges.size());
    for (const auto& e : db_edges) {
        result.push_back({e.from_id, e.to_id, e.weight});
    }
    return result;
}

int64_t NeuralMemoryAdapter::count_edges() const {
    if (!db_) return 0;
    return db_->count_edges();
}

#endif

// ============================================================================
// Helpers
// ============================================================================

std::vector<float> NeuralMemoryAdapter::text_to_embedding(const std::string& text) const {
    // Simple embedding: character-level hashing into vector space
    // This is a PLACEHOLDER - in production, use a real embedding model
    std::vector<float> embedding(config_.vector_dim, 0.0f);
    
    if (text.empty()) return embedding;
    
    // Hash each character and distribute across dimensions
    for (size_t i = 0; i < text.size(); ++i) {
        uint32_t h = std::hash<char>{}(text[i]);
        h ^= (static_cast<uint32_t>(i) * 2654435761U);
        
        size_t idx = h % config_.vector_dim;
        embedding[idx] += 1.0f;
        
        // Also add to nearby dimensions for locality
        size_t idx2 = (h ^ 0x9e3779b9) % config_.vector_dim;
        embedding[idx2] += 0.5f;
    }
    
    // Normalize
    float norm = 0;
    for (float v : embedding) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (float& v : embedding) v /= norm;
    }
    
    return embedding;
}

MemoryResult NeuralMemoryAdapter::to_result(
    const memory::MemoryEntry& entry, float similarity) const
{
    MemoryResult result;
    result.id = entry.id;
    result.embedding = entry.embedding;
    result.label = entry.label;
    result.content = entry.content;
    result.similarity = similarity;
    result.salience = entry.salience;
    result.connected_ids = entry.linked;
    return result;
}

} // namespace neural
