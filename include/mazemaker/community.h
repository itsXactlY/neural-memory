// mazemaker/community.h - Louvain community detection
//
// Modularity-optimising community discovery for the Insight phase of the
// dream cycle. Pure-Python NetworkX Louvain takes minutes on a 1M-edge
// graph; this lives in libmazemaker and uses C++26 idioms (std::span,
// std::ranges, designated init) on top of plain hash-based adjacency.
//
// Runtime: O(I · E · α) where I is iteration count (capped, ~3-10 in
// practice) and α is the average neighbour-community fanout per move.
// On a 1M-edge / 14k-node graph this returns in ~1-2s on a single core.
//
// Usage:
//   neural::graph::CommunityDetector det{};
//   auto result = det.detect(edges);
//   for (const auto& comm : result.communities) { ... }
//   double Q = result.modularity;

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace neural::graph {

// External edge representation. The detector treats edges as undirected:
// each (src, dst, weight) record contributes to both endpoints' adjacency
// lists. Self-loops and non-positive weights are ignored.
struct LouvainEdge {
    uint64_t src;
    uint64_t dst;
    float weight;
};

class CommunityDetector {
public:
    struct Config {
        // Maximum sweeps over all nodes. Each sweep visits every node and
        // moves it to the modularity-maximising neighbour community. We
        // bail early once a sweep produces no moves.
        int max_iterations = 10;

        // Modularity gain below this threshold is treated as no-improvement,
        // so floating-point noise doesn't keep the loop alive forever near
        // a fixed point.
        double min_gain = 1e-9;

        // Deterministic node-visit order. Same input → same partition.
        unsigned seed = 42;
    };

    struct Result {
        // One inner vector per community; nodes inside are sorted by id
        // ascending. The outer vector is sorted by community size DESC so
        // callers that only inspect the top-N communities don't need a
        // separate sort.
        std::vector<std::vector<uint64_t>> communities;

        // Newman-Girvan modularity Q in [-0.5, 1.0]. Higher = better
        // partition. Random graphs cluster near 0; well-clustered graphs
        // typically score 0.3-0.7.
        double modularity = 0.0;

        // Number of sweeps actually performed before convergence (or hit
        // max_iterations). Useful telemetry — knob tuning later.
        int iterations_used = 0;
    };

    CommunityDetector() noexcept;
    explicit CommunityDetector(Config cfg) noexcept;
    ~CommunityDetector();

    CommunityDetector(const CommunityDetector&)            = delete;
    CommunityDetector& operator=(const CommunityDetector&) = delete;
    CommunityDetector(CommunityDetector&&)                 = default;
    CommunityDetector& operator=(CommunityDetector&&)      = default;

    // Run detection. Stateless across calls — callable repeatedly with
    // different edge sets, no internal cache. Reads `edges` once, builds
    // a dense-index undirected adjacency map, runs Louvain, returns the
    // partition over the original uint64 node IDs.
    [[nodiscard]] Result detect(std::span<const LouvainEdge> edges) const;

private:
    Config cfg_;
};

} // namespace neural::graph
