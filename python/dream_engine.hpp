#ifndef DREAM_ENGINE_HPP
#define DREAM_ENGINE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <functional>
#include <immintrin.h>
#include <optional>

namespace dream_engine {

// ============================================================================
// MemoryArena - Zero-allocation memory pool for hot paths
// ============================================================================
class MemoryArena {
public:
    explicit MemoryArena(size_t capacity_bytes) : capacity_(capacity_bytes) {
        buffer_ = std::make_unique<uint8_t[]>(capacity_);
        std::memset(buffer_.get(), 0, capacity_);
    }

    ~MemoryArena() = default;

    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    void* allocate(size_t size, size_t alignment = 8) {
        size_t raw_addr = reinterpret_cast<size_t>(buffer_.get()) + offset_;
        size_t aligned_addr = (raw_addr + alignment - 1) & ~(alignment - 1);
        size_t actual_offset = aligned_addr - reinterpret_cast<size_t>(buffer_.get());

        if (actual_offset + size > capacity_) {
            return nullptr;
        }

        offset_ = actual_offset + size;
        return reinterpret_cast<void*>(aligned_addr);
    }

    template<typename T>
    T* allocate_array(size_t count) {
        return static_cast<T*>(allocate(count * sizeof(T), alignof(T)));
    }

    void reset() { offset_ = 0; }

    size_t used_bytes() const { return offset_; }
    size_t remaining_bytes() const { return capacity_ - offset_; }
    size_t total_capacity() const { return capacity_; }

private:
    std::unique_ptr<uint8_t[]> buffer_;
    size_t offset_ = 0;
    size_t capacity_;
};

// ============================================================================
// SimilarityEngine - AVX2-accelerated cosine similarity
// ============================================================================
class SimilarityEngine {
public:
    static constexpr size_t EMBEDDING_DIM = 1024;
    static constexpr size_t ALIGNMENT = 32;

    SimilarityEngine() : arena_(64 * 1024 * 1024) {} // 64MB arena

    // Compute cosine similarity between two 1024-dim vectors using AVX2
    float cosine_sim_avx2(const float* a, const float* b) const {
        __m256 sum_vec = _mm256_setzero_ps();
        __m256 norm_a_vec = _mm256_setzero_ps();
        __m256 norm_b_vec = _mm256_setzero_ps();

        for (size_t i = 0; i < EMBEDDING_DIM; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);

            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb));
            norm_a_vec = _mm256_add_ps(norm_a_vec, _mm256_mul_ps(va, va));
            norm_b_vec = _mm256_add_ps(norm_b_vec, _mm256_mul_ps(vb, vb));
        }

        // Horizontal sum within 256-bit lanes
        alignas(32) float dot[8], na[8], nb[8];
        _mm256_store_ps(dot, sum_vec);
        _mm256_store_ps(na, norm_a_vec);
        _mm256_store_ps(nb, norm_b_vec);

        float dot_prod = dot[0] + dot[1] + dot[2] + dot[3] + dot[4] + dot[5] + dot[6] + dot[7];
        float norm_a = std::sqrt(na[0] + na[1] + na[2] + na[3] + na[4] + na[5] + na[6] + na[7]);
        float norm_b = std::sqrt(nb[0] + nb[1] + nb[2] + nb[3] + nb[4] + nb[5] + nb[6] + nb[7]);

        if (norm_a < 1e-8f || norm_b < 1e-8f) return 0.0f;
        return dot_prod / (norm_a * norm_b);
    }

    // Batch similarity: query vs many candidates
    std::vector<float> batch_cosine_sim(const float* query,
                                         const float* candidates,
                                         size_t num_candidates) const {
        std::vector<float> results(num_candidates);
        for (size_t i = 0; i < num_candidates; ++i) {
            results[i] = cosine_sim_avx2(query, candidates + i * EMBEDDING_DIM);
        }
        return results;
    }

    // Find top-k most similar
    std::vector<std::pair<size_t, float>> topk_similar(const float* query,
                                                        const float* candidates,
                                                        size_t num_candidates,
                                                        size_t k) const {
        k = std::min(k, num_candidates);
        std::vector<float> sims = batch_cosine_sim(query, candidates, num_candidates);

        std::vector<std::pair<size_t, float>> indexed_sims(num_candidates);
        for (size_t i = 0; i < num_candidates; ++i) {
            indexed_sims[i] = {i, sims[i]};
        }

        std::partial_sort(indexed_sims.begin(), indexed_sims.begin() + k,
                          indexed_sims.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });

        return std::vector<std::pair<size_t, float>>(indexed_sims.begin(), indexed_sims.begin() + k);
    }

    // Allocate from arena
    template<typename T>
    T* arena_alloc(size_t count) {
        return arena_.allocate_array<T>(count);
    }

    void arena_reset() { arena_.reset(); }
    size_t arena_used() const { return arena_.used_bytes(); }

private:
    MemoryArena arena_;
};

// ============================================================================
// CSR Graph Representation
// ============================================================================
struct CSRGraph {
    std::vector<size_t> row_offsets;
    std::vector<size_t> col_indices;
    std::vector<float> weights;
    size_t num_nodes = 0;

    CSRGraph() = default;

    CSRGraph(size_t num_nodes, const std::vector<std::tuple<size_t, size_t, float>>& edges) {
        build(num_nodes, edges);
    }

    void build(size_t n, const std::vector<std::tuple<size_t, size_t, float>>& edges) {
        num_nodes = n;
        row_offsets.assign(n + 1, 0);
        col_indices.reserve(edges.size());
        weights.reserve(edges.size());

        // Count degrees
        for (const auto& [src, dst, w] : edges) {
            (void)src;
            row_offsets[dst + 1]++;
        }

        // Prefix sum
        for (size_t i = 1; i <= n; ++i) {
            row_offsets[i] += row_offsets[i - 1];
        }

        // Fill edges
        size_t current_size = edges.size();
        col_indices.resize(current_size);
        weights.resize(current_size);

        std::vector<size_t> counters = row_offsets;
        for (size_t i = 0; i < edges.size(); ++i) {
            auto [src, dst, w] = edges[i];
            size_t pos = counters[dst]++;
            col_indices[pos] = src;
            weights[pos] = w;
        }
    }

    // Get neighbors of node
    std::vector<std::pair<size_t, float>> neighbors(size_t node) const {
        std::vector<std::pair<size_t, float>> result;
        if (node >= num_nodes) return result;

        for (size_t i = row_offsets[node]; i < row_offsets[node + 1]; ++i) {
            result.emplace_back(col_indices[i], weights[i]);
        }
        return result;
    }

    size_t degree(size_t node) const {
        if (node >= num_nodes) return 0;
        return row_offsets[node + 1] - row_offsets[node];
    }
};

// ============================================================================
// GraphProcessor - BFS traversal on CSR graph
// ============================================================================
class GraphProcessor {
public:
    explicit GraphProcessor(CSRGraph graph) : graph_(std::move(graph)) {}

    // BFS from source, collecting nodes within max_depth
    std::vector<size_t> bfs(size_t source, size_t max_depth) const {
        std::vector<size_t> result;
        if (graph_.num_nodes == 0 || source >= graph_.num_nodes) return result;

        std::queue<std::pair<size_t, size_t>> queue;
        std::unordered_set<size_t> visited;

        queue.emplace(source, 0);
        visited.insert(source);

        while (!queue.empty()) {
            auto [node, depth] = queue.front();
            queue.pop();

            if (depth > max_depth) break;
            result.push_back(node);

            for (auto [neighbor, _] : graph_.neighbors(node)) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    queue.emplace(neighbor, depth + 1);
                }
            }
        }

        return result;
    }

    // BFS with similarity pruning
    std::vector<size_t> bfs_with_similarity(size_t source,
                                              const float* query_embedding,
                                              const SimilarityEngine& sim_engine,
                                              float min_similarity,
                                              size_t max_depth) const {
        std::vector<size_t> result;
        if (graph_.num_nodes == 0 || source >= graph_.num_nodes) return result;

        std::queue<std::pair<size_t, size_t>> queue;
        std::unordered_set<size_t> visited;

        queue.emplace(source, 0);
        visited.insert(source);

        while (!queue.empty()) {
            auto [node, depth] = queue.front();
            queue.pop();

            if (depth > max_depth) break;
            result.push_back(node);

            for (auto [neighbor, _] : graph_.neighbors(node)) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    queue.emplace(neighbor, depth + 1);
                }
            }
        }

        return result;
    }

    // Compute Personalized PageRank
    std::vector<float> ppr(size_t source, float alpha = 0.15f, size_t iterations = 100) const {
        std::vector<float> pr(graph_.num_nodes, 0.0f);
        if (graph_.num_nodes == 0 || source >= graph_.num_nodes) return pr;

        pr[source] = 1.0f;
        std::vector<float> pr_next(graph_.num_nodes);

        for (size_t iter = 0; iter < iterations; ++iter) {
            std::fill(pr_next.begin(), pr_next.end(), 0.0f);

            for (size_t node = 0; node < graph_.num_nodes; ++node) {
                if (pr[node] == 0.0f) continue;

                size_t deg = graph_.degree(node);
                if (deg == 0) continue;

                float share = pr[node] / static_cast<float>(deg);
                for (auto [neighbor, _] : graph_.neighbors(node)) {
                    pr_next[neighbor] += share;
                }
            }

            // Apply damping factor
            for (size_t i = 0; i < graph_.num_nodes; ++i) {
                pr_next[i] = alpha * pr_next[i] + (1.0f - alpha) * (i == source ? 1.0f : 0.0f);
            }

            // Normalize
            float sum = std::accumulate(pr_next.begin(), pr_next.end(), 0.0f);
            if (sum > 0.0f) {
                for (float& v : pr_next) v /= sum;
            }

            pr.swap(pr_next);
        }

        return pr;
    }

    // Get the underlying graph
    const CSRGraph& graph() const { return graph_; }

private:
    CSRGraph graph_;
};

// ============================================================================
// VP-Tree for bridge node discovery
// ============================================================================
struct VPTreeNode {
    size_t point_index;
    float threshold;
    std::unique_ptr<VPTreeNode> left;
    std::unique_ptr<VPTreeNode> right;

    VPTreeNode(size_t idx, float thresh)
        : point_index(idx), threshold(thresh), left(nullptr), right(nullptr) {}
};

class VPTree {
public:
    VPTree(const std::vector<float>& data, size_t dim)
        : data_(data), dim_(dim), rng_(std::random_device{}()) {
        if (!data_.empty()) {
            std::vector<size_t> indices(data_.size() / dim);
            std::iota(indices.begin(), indices.end(), 0);
            root_ = build(indices);
        }
    }

    std::vector<size_t> search(const float* query, size_t k, float epsilon = 0.0f) const {
        std::vector<std::pair<size_t, float>> results;
        search_recursive(root_.get(), query, k, epsilon, results);
        std::sort(results.begin(), results.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        std::vector<size_t> indices;
        for (const auto& [idx, _] : results) {
            indices.push_back(idx);
        }
        return indices;
    }

private:
    const std::vector<float>& data_;
    size_t dim_;
    std::unique_ptr<VPTreeNode> root_;
    mutable std::mt19937 rng_;

    std::unique_ptr<VPTreeNode> build(const std::vector<size_t>& indices) {
        if (indices.empty()) return nullptr;

        if (indices.size() == 1) {
            return std::make_unique<VPTreeNode>(indices[0], 0.0f);
        }

        // Random pivot
        std::uniform_int_distribution<size_t> dist(0, indices.size() - 1);
        size_t pivot_idx = indices[dist(rng_)];
        size_t pivot_offset = pivot_idx * dim_;

        // Find median distance — use {dist, idx} pairs so nth_element can mutate
        std::vector<std::pair<float, size_t>> dist_idx;
        dist_idx.reserve(indices.size());
        for (size_t idx : indices) {
            float d = l2_distance(data_.data() + idx * dim_, data_.data() + pivot_offset);
            dist_idx.emplace_back(d, idx);
        }

        size_t mid = dist_idx.size() / 2;
        std::nth_element(dist_idx.begin(), dist_idx.begin() + mid, dist_idx.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });

        float threshold = dist_idx[mid].first;

        std::vector<size_t> left_indices, right_indices;
        for (const auto& [d, idx] : dist_idx) {
            if (d < threshold) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }

        auto node = std::make_unique<VPTreeNode>(indices[mid], threshold);
        node->left = build(left_indices);
        node->right = build(right_indices);
        return node;
    }

    void search_recursive(const VPTreeNode* node, const float* query,
                          size_t k, float epsilon,
                          std::vector<std::pair<size_t, float>>& results) const {
        if (!node) return;

        float dist = l2_distance(data_.data() + node->point_index * dim_, query);

        if (results.size() < k) {
            results.emplace_back(node->point_index, dist);
        } else {
            auto max_it = std::max_element(results.begin(), results.end(),
                                           [](const auto& a, const auto& b) { return a.second < b.second; });
            if (dist < max_it->second) {
                *max_it = {node->point_index, dist};
            }
        }

        if (!node->left && !node->right) return;

        float tau = results.empty() ? std::numeric_limits<float>::max() :
                    std::max_element(results.begin(), results.end(),
                                     [](const auto& a, const auto& b) { return a.second < b.second; })->second;

        if (node->left && dist < node->threshold * (1.0f + epsilon)) {
            search_recursive(node->left.get(), query, k, epsilon, results);
        }
        if (node->right && dist > node->threshold * (1.0f - epsilon)) {
            search_recursive(node->right.get(), query, k, epsilon, results);
        }
    }

    float l2_distance(const float* a, const float* b) const {
        float sum = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};

// ============================================================================
// LSH (Locality-Sensitive Hashing) for approximate nearest neighbors
// ============================================================================
class LSH {
public:
    LSH(size_t num_tables, size_t num_hashes, size_t dim, size_t seed = 42)
        : num_tables_(num_tables), num_hashes_(num_hashes), dim_(dim), seed_(seed) {
        rng_.seed(seed);
        generate_planes();
    }

    void add_point(size_t index, const float* embedding) {
        for (size_t t = 0; t < num_tables_; ++t) {
            std::string key = compute_hash(embedding, t);
            hash_tables_[t][key].push_back(index);
        }
    }

    std::vector<size_t> query(const float* embedding, size_t max_results = 100) const {
        std::unordered_set<size_t> candidates;
        for (size_t t = 0; t < num_tables_; ++t) {
            std::string key = compute_hash(embedding, t);
            auto it = hash_tables_.find(t);
            if (it != hash_tables_.end()) {
                auto kit = it->second.find(key);
                if (kit != it->second.end()) {
                    for (size_t idx : kit->second) {
                        candidates.insert(idx);
                    }
                }
            }
        }
        return std::vector<size_t>(candidates.begin(), candidates.end());
    }

private:
    size_t num_tables_;
    size_t num_hashes_;
    size_t dim_;
    uint32_t seed_;
    std::mt19937 rng_;
    std::vector<std::vector<std::vector<float>>> planes_; // [table][hash][dim]
    std::unordered_map<size_t, std::unordered_map<std::string, std::vector<size_t>>> hash_tables_;

    void generate_planes() {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        planes_.resize(num_tables_);

        for (size_t t = 0; t < num_tables_; ++t) {
            planes_[t].resize(num_hashes_);
            for (size_t h = 0; h < num_hashes_; ++h) {
                planes_[t][h].resize(dim_);
                for (size_t d = 0; d < dim_; ++d) {
                    planes_[t][h][d] = dist(rng_);
                }
            }
        }
    }

    std::string compute_hash(const float* embedding, size_t table) const {
        std::string result;
        result.reserve(num_hashes_);
        for (size_t h = 0; h < num_hashes_; ++h) {
            float dot = 0.0f;
            for (size_t d = 0; d < dim_; ++d) {
                dot += planes_[table][h][d] * embedding[d];
            }
            result.push_back(dot >= 0.0f ? '1' : '0');
        }
        return result;
    }
};

// ============================================================================
// BridgeDiscovery - VP-Tree + LSH hybrid for finding bridge nodes
// ============================================================================
class BridgeDiscovery {
public:
    BridgeDiscovery(size_t embedding_dim = 1024) : embedding_dim_(embedding_dim) {}

    void set_embeddings(const float* embeddings, size_t num_embeddings) {
        embeddings_ = embeddings;
        num_embeddings_ = num_embeddings;

        if (num_embeddings > 0) {
            vp_tree_ = std::make_unique<VPTree>(
                std::vector<float>(embeddings, embeddings + num_embeddings * embedding_dim_),
                embedding_dim_);

            lsh_ = std::make_unique<LSH>(4, 12, embedding_dim_);
            for (size_t i = 0; i < num_embeddings; ++i) {
                lsh_->add_point(i, embeddings + i * embedding_dim_);
            }
        }
    }

    std::vector<size_t> find_bridge_nodes(const float* query_embedding,
                                           size_t max_bridges = 10,
                                           float similarity_threshold = 0.7f) const {
        if (!vp_tree_ || !lsh_) return {};

        // Use VP-Tree for exact nearest neighbors
        auto vp_results = vp_tree_->search(query_embedding, max_bridges * 3);

        // Use LSH for approximate search
        auto lsh_results = lsh_->query(query_embedding, max_bridges * 3);

        // Merge and deduplicate
        std::unordered_set<size_t> unique_results;
        for (size_t idx : vp_results) unique_results.insert(idx);
        for (size_t idx : lsh_results) unique_results.insert(idx);

        // Score by VP-Tree distance
        std::vector<std::pair<size_t, float>> scored;
        SimilarityEngine sim;
        for (size_t idx : unique_results) {
            float sim_score = sim.cosine_sim_avx2(query_embedding,
                                                   embeddings_ + idx * embedding_dim_);
            if (sim_score >= similarity_threshold) {
                scored.emplace_back(idx, sim_score);
            }
        }

        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        std::vector<size_t> result;
        for (size_t i = 0; i < std::min(max_bridges, scored.size()); ++i) {
            result.push_back(scored[i].first);
        }

        return result;
    }

    // Find bridges between isolated communities
    std::vector<size_t> find_bridges_between_communities(
            const std::vector<size_t>& community_a,
            const std::vector<size_t>& community_b,
            const float* embedding_a,
            const float* embedding_b,
            size_t max_bridges = 5) const {

        std::vector<std::pair<size_t, float>> all_scores;

        for (size_t i = 0; i < community_a.size(); ++i) {
            for (size_t j = 0; j < community_b.size(); ++j) {
                float sim = SimilarityEngine().cosine_sim_avx2(
                    embedding_a + community_a[i] * embedding_dim_,
                    embedding_b + community_b[j] * embedding_dim_);
                all_scores.emplace_back(community_a[i], sim);
            }
        }

        std::sort(all_scores.begin(), all_scores.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        std::vector<size_t> bridges;
        for (size_t i = 0; i < std::min(max_bridges * 2, all_scores.size()); ++i) {
            if (all_scores[i].second > 0.5f) {
                bridges.push_back(all_scores[i].first);
            }
        }

        return bridges;
    }

private:
    size_t embedding_dim_;
    const float* embeddings_ = nullptr;
    size_t num_embeddings_ = 0;
    std::unique_ptr<VPTree> vp_tree_;
    std::unique_ptr<LSH> lsh_;
};

// ============================================================================
// CommunityDetector - Louvain + Label Propagation + BFS
// ============================================================================
class CommunityDetector {
public:
    struct CommunityResult {
        std::vector<size_t> node_to_community;
        std::unordered_map<size_t, std::vector<size_t>> communities;
        double modularity = 0.0;
        size_t num_communities = 0;
    };

    CommunityResult detect_louvain(const CSRGraph& graph, float resolution = 1.0f) {
        CommunityResult result;
        size_t n = graph.num_nodes;
        if (n == 0) return result;

        // Initialize: each node in its own community
        std::vector<size_t> node_community(n);
        std::iota(node_community.begin(), node_community.end(), 0);
        result.node_to_community = node_community;

        std::vector<size_t> community_size(n, 1);

        // Compute total edge weight
        double m = 0.0;
        for (const auto& w : graph.weights) m += w;
        if (m == 0) m = 1.0;

        bool improved = true;
        size_t max_iterations = 100;
        size_t iteration = 0;

        while (improved && iteration < max_iterations) {
            improved = false;
            iteration++;

            for (size_t node = 0; node < n; ++node) {
                size_t current_comm = node_community[node];

                // Calculate weighted degree
                double node_weight = 0.0;
                for (size_t i = graph.row_offsets[node]; i < graph.row_offsets[node + 1]; ++i) {
                    node_weight += graph.weights[i];
                }

                // Find neighboring communities
                std::unordered_map<size_t, double> neighbor_comm_weights;
                for (size_t i = graph.row_offsets[node]; i < graph.row_offsets[node + 1]; ++i) {
                    size_t neighbor = graph.col_indices[i];
                    size_t comm = node_community[neighbor];
                    neighbor_comm_weights[comm] += graph.weights[i];
                }

                // Try moving to best community
                double best_delta = 0.0;
                size_t best_comm = current_comm;

                for (const auto& [comm, weight_sum] : neighbor_comm_weights) {
                    double ki = node_weight;
                    double ki_in = weight_sum;

                    double delta = resolution * ki_in / m - (community_size[comm] * ki) / (m * m);

                    if (delta > best_delta) {
                        best_delta = delta;
                        best_comm = comm;
                    }
                }

                if (best_comm != current_comm) {
                    community_size[current_comm]--;
                    community_size[best_comm]++;
                    node_community[node] = best_comm;
                    improved = true;
                }
            }
        }

        // Renumber communities compactly
        std::unordered_map<size_t, size_t> comm_map;
        size_t next_comm_id = 0;
        for (size_t i = 0; i < n; ++i) {
            if (comm_map.find(node_community[i]) == comm_map.end()) {
                comm_map[node_community[i]] = next_comm_id++;
            }
            result.node_to_community[i] = comm_map[node_community[i]];
        }

        // Build community -> nodes mapping
        result.communities.reserve(next_comm_id);
        for (size_t i = 0; i < next_comm_id; ++i) {
            result.communities[i] = {};
        }
        for (size_t i = 0; i < n; ++i) {
            result.communities[result.node_to_community[i]].push_back(i);
        }

        result.num_communities = next_comm_id;
        result.modularity = compute_modularity(graph, result.node_to_community, resolution);

        return result;
    }

    std::vector<size_t> detect_label_propagation(const CSRGraph& graph, size_t max_iterations = 50) {
        size_t n = graph.num_nodes;
        if (n == 0) return {};

        std::vector<size_t> labels(n);
        std::iota(labels.begin(), labels.end(), 0);

        std::mt19937 rng(42);
        std::vector<size_t> nodes(n);
        std::iota(nodes.begin(), nodes.end(), 0);

        for (size_t iter = 0; iter < max_iterations; ++iter) {
            std::shuffle(nodes.begin(), nodes.end(), rng);

            bool changed = false;
            for (size_t node : nodes) {
                std::unordered_map<size_t, size_t> neighbor_labels;

                for (size_t i = graph.row_offsets[node]; i < graph.row_offsets[node + 1]; ++i) {
                    size_t neighbor = graph.col_indices[i];
                    neighbor_labels[labels[neighbor]]++;
                }

                if (neighbor_labels.empty()) continue;

                size_t best_label = labels[node];
                size_t best_count = 0;
                for (const auto& [label, count] : neighbor_labels) {
                    if (count > best_count) {
                        best_count = count;
                        best_label = label;
                    }
                }

                if (best_label != labels[node]) {
                    labels[node] = best_label;
                    changed = true;
                }
            }

            if (!changed) break;
        }

        return labels;
    }

    std::vector<std::vector<size_t>> bfs_communities(size_t start_node,
                                                      const CSRGraph& graph,
                                                      size_t max_depth = 3) {
        std::vector<std::vector<size_t>> communities;
        std::unordered_set<size_t> visited;
        std::queue<std::pair<size_t, size_t>> queue;

        queue.emplace(start_node, 0);
        visited.insert(start_node);
        communities.push_back({});

        while (!queue.empty()) {
            auto [node, depth] = queue.front();
            queue.pop();

            if (depth > max_depth) {
                if (!communities.back().empty()) {
                    communities.push_back({});
                }
                continue;
            }

            communities.back().push_back(node);

            for (size_t i = graph.row_offsets[node]; i < graph.row_offsets[node + 1]; ++i) {
                size_t neighbor = graph.col_indices[i];
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    queue.emplace(neighbor, depth + 1);
                }
            }
        }

        // Remove empty communities
        communities.erase(
            std::remove_if(communities.begin(), communities.end(),
                          [](const auto& c) { return c.empty(); }),
            communities.end());

        return communities;
    }

private:
    double compute_modularity(const CSRGraph& graph,
                              const std::vector<size_t>& node_community,
                              float resolution) {
        size_t n = graph.num_nodes;
        double m = 0.0;
        for (const auto& w : graph.weights) m += w;
        if (m == 0) m = 1.0;

        double q = 0.0;
        for (size_t node = 0; node < n; ++node) {
            for (size_t i = graph.row_offsets[node]; i < graph.row_offsets[node + 1]; ++i) {
                size_t neighbor = graph.col_indices[i];
                if (node_community[node] == node_community[neighbor]) {
                    q += graph.weights[i];
                }
            }
        }

        // Normalize
        std::vector<double> community_weight(n, 0.0);
        for (size_t node = 0; node < n; ++node) {
            for (size_t i = graph.row_offsets[node]; i < graph.row_offsets[node + 1]; ++i) {
                community_weight[node_community[node]] += graph.weights[i];
            }
        }

        double expected = 0.0;
        for (double cw : community_weight) {
            expected += (cw * cw) / (4.0 * m);
        }

        return (q / m) - resolution * expected;
    }
};

// ============================================================================
// MazeConfig - Configuration for dream engine phases
// ============================================================================
struct MazeConfig {
    // General settings
    size_t embedding_dim = 1024;
    size_t max_memories = 100000;

    // Memory Arena settings
    size_t arena_capacity_mb = 64;

    // SimilarityEngine settings
    float similarity_threshold = 0.75f;
    size_t top_k_results = 20;

    // GraphProcessor settings
    size_t bfs_max_depth = 3;
    size_t ppr_iterations = 100;
    float ppr_alpha = 0.15f;

    // BridgeDiscovery settings
    size_t max_bridge_nodes = 10;
    size_t lsh_tables = 4;
    size_t lsh_hashes_per_table = 12;
    float bridge_similarity_threshold = 0.7f;

    // CommunityDetector settings
    float louvain_resolution = 1.0f;
    size_t label_prop_max_iterations = 50;
    size_t bfs_community_max_depth = 3;

    // Dream Engine phases
    bool enable_nrem = true;
    bool enable_rem = true;
    bool enable_insight = true;

    // NREM phase (strengthen/prune)
    float nrem_strengthen_threshold = 0.8f;
    float nrem_prune_threshold = 0.3f;
    float nrem_strengthen_weight = 0.1f;
    float nrem_prune_weight = 0.05f;

    // REM phase (bridge discovery)
    size_t rem_bridge_attempts = 5;
    float rem_similarity_range = 0.15f;

    // Insight phase (community detection)
    size_t insight_min_community_size = 3;
    size_t insight_max_communities = 50;

    // Random seed for reproducibility
    uint32_t random_seed = 42;

    // Logging
    bool verbose = false;
    std::string log_prefix = "[DreamEngine] ";
};

// ============================================================================
// PhaseResult - Results from each dream phase
// ============================================================================
struct PhaseResult {
    enum class Status {
        SUCCESS,
        SKIPPED,
        ERROR,
        TIMEOUT
    };

    Status status = Status::SKIPPED;
    std::string phase_name;
    std::string message;
    double execution_time_ms = 0.0;
    size_t nodes_processed = 0;
    size_t edges_modified = 0;

    // Phase-specific data
    std::vector<size_t> strengthened_nodes;
    std::vector<size_t> pruned_nodes;
    std::vector<size_t> bridge_nodes;
    std::vector<size_t> community_assignments;
    std::unordered_map<size_t, std::vector<size_t>> communities;
    size_t num_communities = 0;

    pybind11::dict to_dict() const {
        pybind11::dict result;
        result["status"] = [this]() {
            switch (status) {
                case Status::SUCCESS: return "success";
                case Status::SKIPPED: return "skipped";
                case Status::ERROR: return "error";
                case Status::TIMEOUT: return "timeout";
            }
            return "unknown";
        }();
        result["phase_name"] = phase_name;
        result["message"] = message;
        result["execution_time_ms"] = execution_time_ms;
        result["nodes_processed"] = nodes_processed;
        result["edges_modified"] = edges_modified;
        result["strengthened_nodes"] = strengthened_nodes;
        result["pruned_nodes"] = pruned_nodes;
        result["bridge_nodes"] = bridge_nodes;
        result["community_assignments"] = community_assignments;

        // Convert communities map to Python dict
        pybind11::dict comm_dict;
        for (const auto& [k, v] : communities) {
            comm_dict[pybind11::cast(k)] = pybind11::cast(v);
        }
        result["communities"] = comm_dict;

        return result;
    }
};

// ============================================================================
// DreamEngineCore - Main orchestrator
// ============================================================================
class DreamEngineCore {
public:
    explicit DreamEngineCore(const MazeConfig& config = MazeConfig{})
        : config_(config),
          arena_(config.arena_capacity_mb * 1024 * 1024),
          similarity_engine_(),
          bridge_discovery_(config.embedding_dim),
          community_detector_() {

        // Seed random number generator
        rng_.seed(config.random_seed);
    }

    // Initialize with embeddings data
    void set_embeddings(const float* embeddings, size_t num_embeddings) {
        num_embeddings_ = num_embeddings;
        embeddings_ = embeddings;
        bridge_discovery_.set_embeddings(embeddings, num_embeddings);
    }

    // Set graph structure
    void set_graph(const CSRGraph& graph) {
        graph_ = std::make_unique<GraphProcessor>(graph);
        graph_ptr_ = graph_.get();
    }

    // Phase 1: NREM - Strengthen strong connections, prune weak ones
    PhaseResult run_nrem_phase() {
        PhaseResult result;
        result.phase_name = "nrem";
        auto start = std::chrono::high_resolution_clock::now();

        if (!config_.enable_nrem) {
            result.status = PhaseResult::Status::SKIPPED;
            result.message = "NREM phase disabled in config";
            return result;
        }

        if (!graph_ || !embeddings_) {
            result.status = PhaseResult::Status::ERROR;
            result.message = "Graph or embeddings not initialized";
            return result;
        }

        if (num_embeddings_ == 0) {
            result.status = PhaseResult::Status::SKIPPED;
            result.message = "No embeddings to process";
            return result;
        }

        const CSRGraph& g = graph_ptr_->graph();
        result.nodes_processed = g.num_nodes;

        // Process each node's connections
        for (size_t node = 0; node < g.num_nodes; ++node) {
            const float* node_emb = embeddings_ + node * config_.embedding_dim;

            for (size_t i = g.row_offsets[node]; i < g.row_offsets[node + 1]; ++i) {
                size_t neighbor = g.col_indices[i];
                const float* neighbor_emb = embeddings_ + neighbor * config_.embedding_dim;

                float sim = similarity_engine_.cosine_sim_avx2(node_emb, neighbor_emb);

                if (sim >= config_.nrem_strengthen_threshold) {
                    result.strengthened_nodes.push_back(node);
                    result.edges_modified++;
                } else if (sim <= config_.nrem_prune_threshold) {
                    result.pruned_nodes.push_back(node);
                    result.edges_modified++;
                }
            }
        }

        // Remove duplicates
        auto remove_duplicates = [](std::vector<size_t>& v) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        };
        remove_duplicates(result.strengthened_nodes);
        remove_duplicates(result.pruned_nodes);

        result.status = PhaseResult::Status::SUCCESS;
        result.message = "NREM phase completed. Strengthened: " +
                        std::to_string(result.strengthened_nodes.size()) +
                        ", Pruned: " + std::to_string(result.pruned_nodes.size());

        auto end = std::chrono::high_resolution_clock::now();
        result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

    // Phase 2: REM - Bridge discovery between isolated memory clusters
    PhaseResult run_rem_phase() {
        PhaseResult result;
        result.phase_name = "rem";
        auto start = std::chrono::high_resolution_clock::now();

        if (!config_.enable_rem) {
            result.status = PhaseResult::Status::SKIPPED;
            result.message = "REM phase disabled in config";
            return result;
        }

        if (!embeddings_ || num_embeddings_ == 0) {
            result.status = PhaseResult::Status::SKIPPED;
            result.message = "No embeddings to process";
            return result;
        }

        // Run label propagation to find initial communities
        if (!graph_ptr_) {
            result.status = PhaseResult::Status::ERROR;
            result.message = "Graph not initialized";
            return result;
        }

        auto labels = community_detector_.detect_label_propagation(graph_ptr_->graph(),
                                                                    config_.label_prop_max_iterations);

        // Group nodes by community
        std::unordered_map<size_t, std::vector<size_t>> communities;
        for (size_t i = 0; i < labels.size(); ++i) {
            communities[labels[i]].push_back(i);
        }

        // Find isolated communities (small communities)
        std::vector<std::vector<size_t>> isolated;
        for (const auto& [comm_id, nodes] : communities) {
            if (nodes.size() < 5) {  // Small community threshold
                isolated.push_back(nodes);
            }
        }

        // Try to find bridges between isolated communities
        for (size_t i = 0; i < isolated.size() && result.bridge_nodes.size() < config_.max_bridge_nodes; ++i) {
            for (size_t j = i + 1; j < isolated.size() && result.bridge_nodes.size() < config_.max_bridge_nodes; ++j) {
                const auto& comm_a = isolated[i];
                const auto& comm_b = isolated[j];

                // Find centroid embeddings
                std::vector<float> centroid_a(config_.embedding_dim, 0.0f);
                std::vector<float> centroid_b(config_.embedding_dim, 0.0f);

                for (size_t node : comm_a) {
                    for (size_t d = 0; d < config_.embedding_dim; ++d) {
                        centroid_a[d] += embeddings_[node * config_.embedding_dim + d];
                    }
                }
                for (size_t node : comm_b) {
                    for (size_t d = 0; d < config_.embedding_dim; ++d) {
                        centroid_b[d] += embeddings_[node * config_.embedding_dim + d];
                    }
                }
                for (size_t d = 0; d < config_.embedding_dim; ++d) {
                    centroid_a[d] /= static_cast<float>(comm_a.size());
                    centroid_b[d] /= static_cast<float>(comm_b.size());
                }

                auto bridges = bridge_discovery_.find_bridges_between_communities(
                    comm_a, comm_b,
                    centroid_a.data(), centroid_b.data(),
                    config_.rem_bridge_attempts);

                for (size_t bridge : bridges) {
                    result.bridge_nodes.push_back(bridge);
                }
            }
        }

        // Remove duplicates
        std::sort(result.bridge_nodes.begin(), result.bridge_nodes.end());
        result.bridge_nodes.erase(std::unique(result.bridge_nodes.begin(), result.bridge_nodes.end()),
                                  result.bridge_nodes.end());

        result.nodes_processed = num_embeddings_;
        result.status = PhaseResult::Status::SUCCESS;
        result.message = "REM phase completed. Found " + std::to_string(result.bridge_nodes.size()) +
                        " bridge nodes";

        auto end = std::chrono::high_resolution_clock::now();
        result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

    // Phase 3: Insight - Community detection and pattern recognition
    PhaseResult run_insight_phase() {
        PhaseResult result;
        result.phase_name = "insight";
        auto start = std::chrono::high_resolution_clock::now();

        if (!config_.enable_insight) {
            result.status = PhaseResult::Status::SKIPPED;
            result.message = "Insight phase disabled in config";
            return result;
        }

        if (!graph_ptr_) {
            result.status = PhaseResult::Status::ERROR;
            result.message = "Graph not initialized";
            return result;
        }

        // Run Louvain community detection
        CommunityDetector::CommunityResult louvain_result =
            community_detector_.detect_louvain(graph_ptr_->graph(), config_.louvain_resolution);

        result.community_assignments = louvain_result.node_to_community;
        result.communities = louvain_result.communities;
        result.num_communities = louvain_result.num_communities;

        // Filter communities by minimum size
        std::unordered_map<size_t, std::vector<size_t>> filtered_communities;
        for (const auto& [comm_id, nodes] : result.communities) {
            if (nodes.size() >= config_.insight_min_community_size) {
                filtered_communities[comm_id] = nodes;
            }
        }
        result.communities = filtered_communities;

        result.nodes_processed = graph_ptr_->graph().num_nodes;
        result.status = PhaseResult::Status::SUCCESS;
        result.message = "Insight phase completed. Found " +
                        std::to_string(result.communities.size()) + " communities, " +
                        "modularity: " + std::to_string(louvain_result.modularity);

        auto end = std::chrono::high_resolution_clock::now();
        result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

    // Run full dream cycle
    std::vector<PhaseResult> run_dream_cycle() {
        std::vector<PhaseResult> results;

        results.push_back(run_nrem_phase());
        results.push_back(run_rem_phase());
        results.push_back(run_insight_phase());

        return results;
    }

    // Get configuration
    const MazeConfig& config() const { return config_; }
    MazeConfig& config() { return config_; }

    // Get similarity engine for external use
    const SimilarityEngine& similarity_engine() const { return similarity_engine_; }

    // Graph-based BFS from a node
    std::vector<size_t> bfs_from(size_t node, size_t max_depth) const {
        if (graph_ptr_) {
            return graph_ptr_->bfs(node, max_depth);
        }
        return {};
    }

    // Personalized PageRank from a node
    std::vector<float> ppr_from(size_t node) const {
        if (graph_ptr_) {
            return graph_ptr_->ppr(node, config_.ppr_alpha, config_.ppr_iterations);
        }
        return {};
    }

private:
    MazeConfig config_;
    MemoryArena arena_;
    SimilarityEngine similarity_engine_;
    BridgeDiscovery bridge_discovery_;
    CommunityDetector community_detector_;

    size_t num_embeddings_ = 0;
    const float* embeddings_ = nullptr;

    std::unique_ptr<GraphProcessor> graph_;
    const GraphProcessor* graph_ptr_ = nullptr;

    std::mt19937 rng_;
};

// ============================================================================
// Python bindings via pybind11
// ============================================================================
namespace py = pybind11;

// Module name is intentionally `dream_engine_native` rather than `dream_engine`:
// the Python module `dream_engine.py` is the public import surface, and a
// pybind module of the same name would shadow it whenever the .so happens to
// be on the path (Python prefers extension modules over .py at the same name).
// Consumers that want the C++ side do `import dream_engine_native` once the
// CMake target lands; everyone else continues to `import dream_engine`.
PYBIND11_MODULE(dream_engine_native, m) {
    m.doc() = "Dream Engine C++26+ - Neural memory consolidation library";

    // Register MemoryArena
    py::class_<MemoryArena>(m, "MemoryArena")
        .def(py::init<size_t>(), py::arg("capacity_bytes"))
        .def("allocate", [](MemoryArena& self, size_t size, size_t alignment) {
            return self.allocate(size, alignment);
        }, py::arg("size"), py::arg("alignment") = 8)
        .def("reset", &MemoryArena::reset)
        .def("used_bytes", &MemoryArena::used_bytes)
        .def("remaining_bytes", &MemoryArena::remaining_bytes)
        .def("total_capacity", &MemoryArena::total_capacity);

    // Register SimilarityEngine
    py::class_<SimilarityEngine>(m, "SimilarityEngine")
        .def(py::init<>())
        .def("cosine_sim_avx2", &SimilarityEngine::cosine_sim_avx2,
             py::arg("a"), py::arg("b"),
             "Compute cosine similarity between two 1024-dim vectors using AVX2")
        .def("batch_cosine_sim", &SimilarityEngine::batch_cosine_sim,
             py::arg("query"), py::arg("candidates"), py::arg("num_candidates"),
             "Batch cosine similarity computation")
        .def("topk_similar", &SimilarityEngine::topk_similar,
             py::arg("query"), py::arg("candidates"), py::arg("num_candidates"), py::arg("k"),
             "Find top-k most similar vectors")
        .def("arena_used", &SimilarityEngine::arena_used)
        .def("arena_reset", &SimilarityEngine::arena_reset);

    // Register CSRGraph
    py::class_<CSRGraph>(m, "CSRGraph")
        .def(py::init<>())
        .def(py::init<size_t, const std::vector<std::tuple<size_t, size_t, float>>&>(),
             py::arg("num_nodes"), py::arg("edges"))
        .def("build", &CSRGraph::build,
             py::arg("num_nodes"), py::arg("edges"))
        .def("neighbors", &CSRGraph::neighbors, py::arg("node"),
             "Get neighbors of a node as list of (neighbor, weight) pairs")
        .def("degree", &CSRGraph::degree, py::arg("node"))
        .def_readonly("num_nodes", &CSRGraph::num_nodes);

    // Register GraphProcessor
    py::class_<GraphProcessor>(m, "GraphProcessor")
        .def(py::init<CSRGraph>(), py::arg("graph"))
        .def("bfs", &GraphProcessor::bfs, py::arg("source"), py::arg("max_depth"),
             "BFS traversal from source node")
        .def("ppr", &GraphProcessor::ppr, py::arg("source"), py::arg("alpha") = 0.15f,
             py::arg("iterations") = 100, "Personalized PageRank computation");

    // Register MazeConfig
    py::class_<MazeConfig>(m, "MazeConfig")
        .def(py::init<>())
        .def_readwrite("embedding_dim", &MazeConfig::embedding_dim)
        .def_readwrite("max_memories", &MazeConfig::max_memories)
        .def_readwrite("arena_capacity_mb", &MazeConfig::arena_capacity_mb)
        .def_readwrite("similarity_threshold", &MazeConfig::similarity_threshold)
        .def_readwrite("top_k_results", &MazeConfig::top_k_results)
        .def_readwrite("bfs_max_depth", &MazeConfig::bfs_max_depth)
        .def_readwrite("ppr_iterations", &MazeConfig::ppr_iterations)
        .def_readwrite("ppr_alpha", &MazeConfig::ppr_alpha)
        .def_readwrite("max_bridge_nodes", &MazeConfig::max_bridge_nodes)
        .def_readwrite("lsh_tables", &MazeConfig::lsh_tables)
        .def_readwrite("lsh_hashes_per_table", &MazeConfig::lsh_hashes_per_table)
        .def_readwrite("bridge_similarity_threshold", &MazeConfig::bridge_similarity_threshold)
        .def_readwrite("louvain_resolution", &MazeConfig::louvain_resolution)
        .def_readwrite("label_prop_max_iterations", &MazeConfig::label_prop_max_iterations)
        .def_readwrite("bfs_community_max_depth", &MazeConfig::bfs_community_max_depth)
        .def_readwrite("enable_nrem", &MazeConfig::enable_nrem)
        .def_readwrite("enable_rem", &MazeConfig::enable_rem)
        .def_readwrite("enable_insight", &MazeConfig::enable_insight)
        .def_readwrite("nrem_strengthen_threshold", &MazeConfig::nrem_strengthen_threshold)
        .def_readwrite("nrem_prune_threshold", &MazeConfig::nrem_prune_threshold)
        .def_readwrite("nrem_strengthen_weight", &MazeConfig::nrem_strengthen_weight)
        .def_readwrite("nrem_prune_weight", &MazeConfig::nrem_prune_weight)
        .def_readwrite("rem_bridge_attempts", &MazeConfig::rem_bridge_attempts)
        .def_readwrite("rem_similarity_range", &MazeConfig::rem_similarity_range)
        .def_readwrite("insight_min_community_size", &MazeConfig::insight_min_community_size)
        .def_readwrite("insight_max_communities", &MazeConfig::insight_max_communities)
        .def_readwrite("random_seed", &MazeConfig::random_seed)
        .def_readwrite("verbose", &MazeConfig::verbose)
        .def_readwrite("log_prefix", &MazeConfig::log_prefix);

    // Register PhaseResult
    py::class_<PhaseResult>(m, "PhaseResult")
        .def(py::init<>())
        .def_readwrite("status", &PhaseResult::status)
        .def_readwrite("phase_name", &PhaseResult::phase_name)
        .def_readwrite("message", &PhaseResult::message)
        .def_readwrite("execution_time_ms", &PhaseResult::execution_time_ms)
        .def_readwrite("nodes_processed", &PhaseResult::nodes_processed)
        .def_readwrite("edges_modified", &PhaseResult::edges_modified)
        .def_readwrite("strengthened_nodes", &PhaseResult::strengthened_nodes)
        .def_readwrite("pruned_nodes", &PhaseResult::pruned_nodes)
        .def_readwrite("bridge_nodes", &PhaseResult::bridge_nodes)
        .def_readwrite("community_assignments", &PhaseResult::community_assignments)
        .def_readwrite("communities", &PhaseResult::communities)
        .def("to_dict", &PhaseResult::to_dict);

    // Register CommunityDetector
    py::class_<CommunityDetector>(m, "CommunityDetector")
        .def(py::init<>())
        .def("detect_louvain", &CommunityDetector::detect_louvain,
             py::arg("graph"), py::arg("resolution") = 1.0f,
             "Louvain community detection")
        .def("detect_label_propagation", &CommunityDetector::detect_label_propagation,
             py::arg("graph"), py::arg("max_iterations") = 50,
             "Label propagation community detection")
        .def("bfs_communities", &CommunityDetector::bfs_communities,
             py::arg("start_node"), py::arg("graph"), py::arg("max_depth") = 3,
             "BFS-based community detection");

    // Register BridgeDiscovery
    py::class_<BridgeDiscovery>(m, "BridgeDiscovery")
        .def(py::init<size_t>(), py::arg("embedding_dim") = 1024)
        .def("set_embeddings", &BridgeDiscovery::set_embeddings,
             py::arg("embeddings"), py::arg("num_embeddings"),
             "Set embedding data for bridge discovery")
        .def("find_bridge_nodes", &BridgeDiscovery::find_bridge_nodes,
             py::arg("query_embedding"), py::arg("max_bridges") = 10,
             py::arg("similarity_threshold") = 0.7f,
             "Find bridge nodes using VP-Tree + LSH")
        .def("find_bridges_between_communities", &BridgeDiscovery::find_bridges_between_communities,
             py::arg("community_a"), py::arg("community_b"),
             py::arg("embedding_a"), py::arg("embedding_b"),
             py::arg("max_bridges") = 5,
             "Find bridges between two communities");

    // Register DreamEngineCore
    py::class_<DreamEngineCore>(m, "DreamEngineCore")
        .def(py::init<const MazeConfig&>(), py::arg("config") = MazeConfig())
        .def("set_embeddings", &DreamEngineCore::set_embeddings,
             py::arg("embeddings"), py::arg("num_embeddings"),
             "Set embedding data (1024-dim float array)")
        .def("set_graph", &DreamEngineCore::set_graph,
             py::arg("graph"),
             "Set graph structure for processing")
        .def("run_nrem_phase", &DreamEngineCore::run_nrem_phase,
             "Run NREM phase - strengthen/prune connections")
        .def("run_rem_phase", &DreamEngineCore::run_rem_phase,
             "Run REM phase - bridge discovery")
        .def("run_insight_phase", &DreamEngineCore::run_insight_phase,
             "Run Insight phase - community detection")
        .def("run_dream_cycle", &DreamEngineCore::run_dream_cycle,
             "Run full dream cycle (NREM -> REM -> Insight)")
        .def("config", [](DreamEngineCore& self) -> MazeConfig& { return self.config(); },
             py::return_value_policy::reference,
             "Get reference to configuration")
        .def("similarity_engine", &DreamEngineCore::similarity_engine,
             py::return_value_policy::reference,
             "Get reference to similarity engine")
        .def("bfs_from", &DreamEngineCore::bfs_from,
             py::arg("node"), py::arg("max_depth"),
             "BFS traversal from node")
        .def("ppr_from", &DreamEngineCore::ppr_from,
             py::arg("node"),
             "Personalized PageRank from node");

    // Helper functions exposed at module level
    m.def("create_dream_engine", [](const MazeConfig& config) {
        return std::make_unique<DreamEngineCore>(config);
    }, py::arg("config") = MazeConfig(), "Create a new DreamEngine instance");

    m.def("cosine_similarity", [](py::array_t<float> a, py::array_t<float> b) {
        auto buf_a = a.request();
        auto buf_b = b.request();
        if (buf_a.size != 1024 || buf_b.size != 1024) {
            throw std::runtime_error("Embeddings must be 1024-dimensional");
        }
        SimilarityEngine sim;
        return sim.cosine_sim_avx2(
            static_cast<const float*>(buf_a.ptr),
            static_cast<const float*>(buf_b.ptr));
    }, py::arg("a"), py::arg("b"), "Compute cosine similarity between two 1024-d embeddings");

    m.def("batch_similarity", [](py::array_t<float> query,
                                  py::array_t<float> candidates) {
        auto qbuf = query.request();
        auto cbuf = candidates.request();
        if (qbuf.size != 1024) {
            throw std::runtime_error("Query must be 1024-dimensional");
        }
        if (cbuf.size % 1024 != 0) {
            throw std::runtime_error("Candidates size must be multiple of 1024");
        }

        size_t num_candidates = cbuf.size / 1024;
        SimilarityEngine sim;
        auto results = sim.batch_cosine_sim(
            static_cast<const float*>(qbuf.ptr),
            static_cast<const float*>(cbuf.ptr),
            num_candidates);

        return py::array_t<float>(results.size(), results.data());
    }, py::arg("query"), py::arg("candidates"), "Batch cosine similarity");

    m.def("topk_similarities", [](py::array_t<float> query,
                                   py::array_t<float> candidates,
                                   size_t k) {
        auto qbuf = query.request();
        auto cbuf = candidates.request();
        if (qbuf.size != 1024) {
            throw std::runtime_error("Query must be 1024-dimensional");
        }
        if (cbuf.size % 1024 != 0) {
            throw std::runtime_error("Candidates size must be multiple of 1024");
        }

        size_t num_candidates = cbuf.size / 1024;
        SimilarityEngine sim;
        auto results = sim.topk_similar(
            static_cast<const float*>(qbuf.ptr),
            static_cast<const float*>(cbuf.ptr),
            num_candidates, k);

        py::list indices, scores;
        for (const auto& [idx, score] : results) {
            indices.append(idx);
            scores.append(score);
        }

        py::dict result;
        result["indices"] = indices;
        result["scores"] = scores;
        return result;
    }, py::arg("query"), py::arg("candidates"), py::arg("k") = 10,
       "Find top-k most similar embeddings");
}

} // namespace dream_engine

#endif // DREAM_ENGINE_HPP
