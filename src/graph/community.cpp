// src/graph/community.cpp - Louvain modularity-optimising community detection.
//
// Runs the classic Louvain "phase 1" sweep: each node tries to join the
// neighbour community that most increases modularity, repeated until a full
// sweep produces no moves. We deliberately skip "phase 2" (graph aggregation
// + recursion) because the dream loop only needs flat communities for
// cluster/bridge insight emission, and one phase is already enough to break
// up connected components into meaningful clusters at this scale.
//
// Modularity gain when moving u from {} into community c (after u has been
// removed from its previous community) is:
//
//   ΔQ = (k_u→c / m) − (k_u · Σtot[c]) / (2 m²)
//
// where k_u→c is the sum of edge weights from u into nodes already in c,
// k_u is u's weighted degree, Σtot[c] is the sum of weighted degrees of
// nodes currently in c, and m is the total edge weight.

#include "mazemaker/community.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace neural::graph {

namespace {

// Internal compact representation. uint32_t indices keep the per-node hot
// path cache-friendly even on graphs with millions of edges.
struct Adjacency {
    std::vector<std::vector<std::pair<uint32_t, double>>> nbrs;
    std::vector<double> degree;
    double total_weight = 0.0;
};

// Build a dense-index adjacency map. Returns a parallel vector mapping
// dense index → original uint64 node id, plus a hash for the reverse
// lookup which the detector itself doesn't need but the caller might.
struct BuildOut {
    Adjacency adj;
    std::vector<uint64_t> idx_to_id;
};

[[nodiscard]] BuildOut build_dense_(std::span<const LouvainEdge> edges) {
    std::unordered_map<uint64_t, uint32_t> id_to_idx;
    id_to_idx.reserve(edges.size() * 2);
    std::vector<uint64_t> idx_to_id;
    idx_to_id.reserve(edges.size() * 2);

    auto get_or_assign = [&](uint64_t id) -> uint32_t {
        auto [it, inserted] = id_to_idx.try_emplace(
            id, static_cast<uint32_t>(idx_to_id.size())
        );
        if (inserted) idx_to_id.push_back(id);
        return it->second;
    };

    // First pass: assign indices, count degrees so we can reserve neighbour vecs.
    std::vector<size_t> deg_count;
    for (const auto& e : edges) {
        if (e.src == e.dst || e.weight <= 0.0f) continue;
        const uint32_t u = get_or_assign(e.src);
        const uint32_t v = get_or_assign(e.dst);
        if (deg_count.size() < idx_to_id.size()) deg_count.resize(idx_to_id.size(), 0);
        deg_count[u]++;
        deg_count[v]++;
    }

    const uint32_t n = static_cast<uint32_t>(idx_to_id.size());
    Adjacency a;
    a.nbrs.resize(n);
    a.degree.assign(n, 0.0);
    for (uint32_t i = 0; i < n; ++i) a.nbrs[i].reserve(deg_count[i]);

    // Second pass: fill in
    for (const auto& e : edges) {
        if (e.src == e.dst || e.weight <= 0.0f) continue;
        const uint32_t u = id_to_idx.at(e.src);
        const uint32_t v = id_to_idx.at(e.dst);
        const double w  = static_cast<double>(e.weight);
        a.nbrs[u].emplace_back(v, w);
        a.nbrs[v].emplace_back(u, w);
        a.degree[u] += w;
        a.degree[v] += w;
        a.total_weight += w;
    }

    return BuildOut{ .adj = std::move(a), .idx_to_id = std::move(idx_to_id) };
}

// Newman-Girvan modularity for the partition implied by `comm`.
// Q = Σ_c [ Σin(c) / (2m) − (Σtot(c) / (2m))² ]
[[nodiscard]] double compute_modularity_(
    const Adjacency& adj,
    const std::vector<uint32_t>& comm
) {
    const double m  = adj.total_weight;
    if (m <= 0.0) return 0.0;
    const double m2 = 2.0 * m;

    std::unordered_map<uint32_t, double> sum_in;
    std::unordered_map<uint32_t, double> sum_tot;
    sum_in.reserve(64);
    sum_tot.reserve(64);

    const auto n = static_cast<uint32_t>(adj.nbrs.size());
    for (uint32_t u = 0; u < n; ++u) {
        sum_tot[comm[u]] += adj.degree[u];
        for (const auto& [v, w] : adj.nbrs[u]) {
            // Each undirected edge appears in both u's and v's neighbour
            // lists. Counting only when u <= v gives a single pass per
            // edge; we double the contribution for u != v to recover the
            // 2 · w per intra-community edge that the formula expects.
            if (comm[u] != comm[v]) continue;
            if (v < u) continue;
            sum_in[comm[u]] += (u == v) ? w : 2.0 * w;
        }
    }

    double Q = 0.0;
    for (const auto& [c, sin] : sum_in) {
        const double tot = sum_tot[c];
        Q += sin / m2 - (tot / m2) * (tot / m2);
    }
    return Q;
}

} // anonymous namespace

CommunityDetector::CommunityDetector() noexcept : cfg_{} {}
CommunityDetector::CommunityDetector(Config cfg) noexcept : cfg_(cfg) {}
CommunityDetector::~CommunityDetector() = default;

CommunityDetector::Result
CommunityDetector::detect(std::span<const LouvainEdge> edges) const {
    Result result{};
    if (edges.empty()) return result;

    auto [adj, idx_to_id] = build_dense_(edges);
    const auto n = static_cast<uint32_t>(adj.nbrs.size());
    if (n == 0 || adj.total_weight <= 0.0) return result;

    const double m  = adj.total_weight;
    const double m2 = 2.0 * m;

    // Initialise: every node in its own community.
    std::vector<uint32_t> comm(n);
    std::iota(comm.begin(), comm.end(), 0u);

    // Σtot[c] = sum of degrees of nodes currently in community c.
    std::vector<double> sum_tot(n);
    std::ranges::copy(adj.degree, sum_tot.begin());

    std::mt19937 rng(cfg_.seed);
    std::vector<uint32_t> visit_order(n);
    std::iota(visit_order.begin(), visit_order.end(), 0u);

    // Reuse the per-node neighbour-community map across iterations to
    // avoid hammering the allocator. clear() preserves capacity.
    std::unordered_map<uint32_t, double> w_to_comm;
    w_to_comm.reserve(64);

    int it = 0;
    for (; it < cfg_.max_iterations; ++it) {
        bool moved_any = false;
        std::ranges::shuffle(visit_order, rng);

        for (uint32_t u : visit_order) {
            const uint32_t cu  = comm[u];
            const double   k_u = adj.degree[u];
            if (k_u <= 0.0) continue;

            // Sum edge weights from u to each neighbour community.
            w_to_comm.clear();
            for (const auto& [v, w] : adj.nbrs[u]) {
                if (v == u) continue;
                w_to_comm[comm[v]] += w;
            }

            // Tentatively remove u from its current community before
            // computing gains, so the gain formula compares "u outside"
            // → "u inside cv" symmetrically across all candidates
            // including its own original community.
            sum_tot[cu] -= k_u;

            uint32_t best_c    = cu;
            double   best_gain = 0.0;

            // Self-stay candidate: gain of re-joining cu after removal.
            double w_u_to_cu = 0.0;
            if (auto found = w_to_comm.find(cu); found != w_to_comm.end()) {
                w_u_to_cu = found->second;
            }
            const double stay_gain = w_u_to_cu / m - (k_u * sum_tot[cu]) / (m2 * m);
            best_gain = stay_gain;

            for (const auto& [cv, w_uv] : w_to_comm) {
                if (cv == cu) continue;
                const double gain = w_uv / m - (k_u * sum_tot[cv]) / (m2 * m);
                if (gain > best_gain + cfg_.min_gain) {
                    best_gain = gain;
                    best_c    = cv;
                }
            }

            // Insert u into best community.
            sum_tot[best_c] += k_u;
            comm[u] = best_c;
            if (best_c != cu) moved_any = true;
        }

        if (!moved_any) {
            ++it; // count the no-op sweep that proved convergence
            break;
        }
    }
    result.iterations_used = it;

    // Group nodes by final community.
    std::unordered_map<uint32_t, std::vector<uint64_t>> groups;
    groups.reserve(64);
    for (uint32_t i = 0; i < n; ++i) {
        groups[comm[i]].push_back(idx_to_id[i]);
    }

    result.communities.reserve(groups.size());
    for (auto& [_, members] : groups) {
        std::ranges::sort(members);
        result.communities.emplace_back(std::move(members));
    }
    std::ranges::sort(result.communities, [](const auto& a, const auto& b) {
        if (a.size() != b.size()) return a.size() > b.size();
        return a.front() < b.front();
    });

    result.modularity = compute_modularity_(adj, comm);
    return result;
}

} // namespace neural::graph
