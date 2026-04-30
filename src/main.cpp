// main.cpp - Neural Memory Adapter CLI Demo
#include "mazemaker/memory_adapter.h"
#include <iostream>
#include <string>
#include <sstream>
#include <random>
#include <iomanip>

using namespace neural;

void print_usage() {
    std::cout << "Usage: mazemaker_demo [mode]\n"
              << "Modes:\n"
              << "  demo        - Interactive demo\n"
              << "  benchmark   - Run benchmarks\n"
              << "  store <text> - Store a memory\n"
              << "  search <q>  - Search memories\n"
              << "  stats       - Show statistics\n";
}

std::vector<float> random_embedding(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    float norm = 0;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 0) for (float& x : v) x /= norm;
    return v;
}

void run_demo() {
    std::cout << "=== Neural Memory Adapter Demo ===\n\n";
    
    // Initialize
    NeuralMemoryAdapter adapter;
    AdapterConfig config = AdapterConfig::fast();

    if (!adapter.initialize(config)) {
        std::cerr << "Failed to initialize\n";
        return;
    }
    std::cout << "Adapter initialized (512-dim, fast mode)\n\n";
    
    std::mt19937 rng(42);
    
    // Store some memories
    std::cout << "Storing 100 random memories...\n";
    for (int i = 0; i < 100; ++i) {
        auto emb = random_embedding(512, rng);
        std::string label = "memory_" + std::to_string(i);
        std::string content = "This is the content of memory number " + std::to_string(i);
        adapter.store(emb, label, content);
    }
    std::cout << "Done.\n\n";
    
    // Retrieve
    std::cout << "Retrieving with random cue (top 5)...\n";
    auto cue = random_embedding(512, rng);
    auto results = adapter.retrieve(cue, 5);
    
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "  " << (i+1) << ". [" << results[i].id << "] "
                  << results[i].label << " (sim: "
                  << std::fixed << std::setprecision(3) << results[i].similarity << ")\n";
        if (!results[i].content.empty()) {
            std::cout << "     " << results[i].content.substr(0, 60) << "...\n";
        }
    }
    
    // Think (spreading activation)
    if (!results.empty()) {
        std::cout << "\nThinking from memory " << results[0].id << "...\n";
        auto thoughts = adapter.think(results[0].id, 3);
        for (size_t i = 0; i < std::min(thoughts.size(), (size_t)5); ++i) {
            std::cout << "  -> Node " << thoughts[i].node_id
                      << " (activation: " << std::setprecision(3) << thoughts[i].activation
                      << ", depth: " << thoughts[i].depth << ")\n";
        }
    }
    
    // Consolidate
    std::cout << "\nRunning consolidation...\n";
    size_t consolidated = adapter.consolidate();
    std::cout << "Consolidated " << consolidated << " memories.\n";
    
    // Stats
    auto stats = adapter.get_stats();
    std::cout << "\n=== Final Stats ===\n";
    std::cout << "  Stores: " << stats.total_stores << " (avg " << stats.avg_store_us << " us)\n";
    std::cout << "  Retrieves: " << stats.total_retrieves << " (avg " << stats.avg_retrieve_us << " us)\n";
    std::cout << "  Graph: " << stats.graph_nodes << " nodes, " << stats.graph_edges << " edges\n";
    std::cout << "  Hopfield: " << stats.hopfield_patterns << " patterns\n";
    
    adapter.shutdown();
    std::cout << "\nDone.\n";
}

void run_interactive() {
    std::cout << "=== Neural Memory Adapter - Interactive ===\n\n";
    
    NeuralMemoryAdapter adapter;
    AdapterConfig config = AdapterConfig::balanced();

    if (!adapter.initialize(config)) {
        std::cerr << "Failed to initialize\n";
        return;
    }
    
    std::cout << "Commands: store <text>, search <query>, think <id>, stats, quit\n\n";
    
    std::string line;
    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) break;
        
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        
        if (cmd == "quit" || cmd == "exit") break;
        
        if (cmd == "store") {
            std::string text;
            std::getline(iss >> std::ws, text);
            if (text.empty()) {
                std::cout << "Usage: store <text>\n";
                continue;
            }
            uint64_t id = adapter.store_text(text, "user_input");
            std::cout << "Stored as ID " << id << "\n";
        }
        else if (cmd == "search") {
            std::string query;
            std::getline(iss >> std::ws, query);
            if (query.empty()) {
                std::cout << "Usage: search <query>\n";
                continue;
            }
            auto results = adapter.retrieve_text(query, 5);
            for (size_t i = 0; i < results.size(); ++i) {
                std::cout << "  " << (i+1) << ". [" << results[i].id << "] "
                          << results[i].label << " ("
                          << std::setprecision(3) << results[i].similarity << ")\n";
                if (!results[i].content.empty()) {
                    std::cout << "     " << results[i].content.substr(0, 80) << "\n";
                }
            }
        }
        else if (cmd == "think") {
            uint64_t id;
            if (!(iss >> id)) {
                std::cout << "Usage: think <id>\n";
                continue;
            }
            auto thoughts = adapter.think(id, 3);
            for (const auto& t : thoughts) {
                std::cout << "  Node " << t.node_id
                          << " (act: " << std::setprecision(3) << t.activation
                          << ", depth: " << t.depth << ")\n";
            }
        }
        else if (cmd == "stats") {
            auto stats = adapter.get_stats();
            std::cout << "  Stores: " << stats.total_stores << "\n";
            std::cout << "  Retrieves: " << stats.total_retrieves << "\n";
            std::cout << "  Graph: " << stats.graph_nodes << " nodes, "
                      << stats.graph_edges << " edges\n";
            std::cout << "  Hopfield: " << stats.hopfield_patterns << " patterns\n";
        }
        else if (cmd == "consolidate") {
            size_t n = adapter.consolidate();
            std::cout << "Consolidated " << n << " memories.\n";
        }
        else {
            std::cout << "Unknown command: " << cmd << "\n";
        }
    }
    
    adapter.shutdown();
}

int main(int argc, char* argv[]) {
    std::string mode = "demo";
    if (argc > 1) mode = argv[1];
    
    if (mode == "demo") {
        run_demo();
    }
    else if (mode == "interactive") {
        run_interactive();
    }
    else if (mode == "help" || mode == "--help") {
        print_usage();
    }
    else {
        std::cerr << "Unknown mode: " << mode << "\n";
        print_usage();
        return 1;
    }
    
    return 0;
}
