// include/neural/lstm.h - LSTM Access Pattern Predictor for Neural Memory
#pragma once

#include "neural/simd.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace neural::lstm {

// ============================================================================
// Xavier uniform initialization (free function)
// ============================================================================

inline float xavier_uniform(size_t fan_in, size_t fan_out, std::mt19937& rng) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    return dist(rng);
}

// ============================================================================
// LSTM Weight Matrices
// ============================================================================
// Gate weight layout for one layer:
//   gate_pre = W * x + U * h_prev + b
//   Each gate row has (input_dim + hidden_dim) concatenated weights,
//   so W_gate is (hidden_dim x input_dim), U_gate is (hidden_dim x hidden_dim)
//   We store flattened row-major vectors.

struct LSTMWeights {
    size_t input_dim = 0;
    size_t hidden_dim = 0;

    // Layer 1: input_to_hidden (hidden_dim x input_dim each)
    std::vector<float> Wf, Wi, Wo, Wc;

    // Layer 1: hidden_to_hidden (hidden_dim x hidden_dim each)
    std::vector<float> Uf, Ui, Uo, Uc;

    // Layer 1: biases (hidden_dim each)
    std::vector<float> bf, bi, bo, bc;

    // Layer 2: hidden_to_hidden (hidden_dim x hidden_dim each)
    std::vector<float> W2f, W2i, W2o, W2c;
    std::vector<float> U2f, U2i, U2o, U2c;
    std::vector<float> b2f, b2i, b2o, b2c;

    // Projection: hidden_dim -> output_dim (1024)
    std::vector<float> W_proj;  // (output_dim x hidden_dim)
    std::vector<float> b_proj;  // (output_dim)

    size_t output_dim = 0;

    // Initialize with Xavier uniform
    void init(size_t in_dim, size_t h_dim, size_t out_dim, unsigned seed = 42);
    void init_layer1(size_t in_dim, size_t h_dim, std::mt19937& rng);
    void init_layer2(size_t h_dim, std::mt19937& rng);
    void init_projection(size_t h_dim, size_t out_dim, std::mt19937& rng);
};

// ============================================================================
// LSTM State (per-layer)
// ============================================================================

struct LSTMState {
    std::vector<float> h;  // hidden state
    std::vector<float> c;  // cell state

    LSTMState() = default;
    explicit LSTMState(size_t hidden_dim);

    void reset();
};

// ============================================================================
// Adam Optimizer State (for momentum-based SGD)
// ============================================================================

struct AdamState {
    std::vector<float> m;  // first moment
    std::vector<float> v;  // second moment
    uint64_t t = 0;        // time step

    AdamState() = default;
    explicit AdamState(size_t param_count);
    void resize(size_t param_count);
};

// ============================================================================
// LSTM Predictior
// ============================================================================

class LSTMPredictor {
public:
    LSTMPredictor(size_t input_dim  = 1024,
                   size_t hidden_dim = 256,
                   size_t output_dim = 1024);

    // Forward pass: process a sequence of embeddings, predict next embedding.
    // sequence: list of (input_dim) embeddings (last N queries).
    // Returns: predicted (output_dim) embedding for next access.
    std::vector<float> forward(const std::vector<std::vector<float>>& sequence) const;

    // Single-step forward (returns both hidden state and output embedding).
    // Useful for incremental processing.
    // Returns output embedding; updates state in-place.
    std::vector<float> forward_step(const std::vector<float>& input,
                                     LSTMState& state1,
                                     LSTMState& state2) const;

    // Online training step: given input sequence and target embedding,
    // computes MSE loss and updates weights with Adam optimizer.
    // Returns: loss value.
    float train_step(const std::vector<std::vector<float>>& sequence,
                     const std::vector<float>& target);

    // Save / load weights to binary format
    void save(const std::string& path) const;
    void load(const std::string& path);

    // Access dimensions
    size_t input_dim() const { return input_dim_; }
    size_t hidden_dim() const { return hidden_dim_; }
    size_t output_dim() const { return output_dim_; }

    // Read-only weight access (locked with shared_mutex)
    const LSTMWeights& weights() const { return weights_; }

private:
    size_t input_dim_;
    size_t hidden_dim_;
    size_t output_dim_;

    LSTMWeights weights_;
    AdamState adam_state_;

    // Hyperparameters
    float learning_rate_ = 1e-3f;
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float epsilon_ = 1e-8f;

    // Thread safety
    mutable std::shared_mutex rw_mutex_;

    // --- Internal forward helpers (assume lock is held) ---

    // One LSTM layer step: given input vec, prev state, layer weights -> new state + gates
    // gate_pre[g] = Wg_row * x + Ug_row * h_prev + b[g]  (one row per hidden unit)
    // Outputs: new h, c; also fills gate vectors for backprop.
    void lstm_cell_forward(
        const float* x, size_t x_dim,
        const LSTMState& prev,
        LSTMState& next,
        // Weight pointers for this layer
        const float* Wf, const float* Wi, const float* Wo, const float* Wc,
        const float* Uf, const float* Ui, const float* Uo, const float* Uc,
        const float* bf, const float* bi, const float* bo, const float* bc,
        // Scratch buffers for gates (hidden_dim each)
        float* f_gate, float* i_gate, float* o_gate, float* c_tilde
    ) const;

    // Layer 2 step (hidden_dim input -> hidden_dim output)
    void lstm_cell_forward_2(
        const float* x,              // input from layer 1 (hidden_dim)
        const LSTMState& prev,       // layer 2's own previous state
        LSTMState& next,
        const float* Wf, const float* Wi, const float* Wo, const float* Wc,
        const float* Uf, const float* Ui, const float* Uo, const float* Uc,
        const float* bf, const float* bi, const float* bo, const float* bc,
        float* f_gate, float* i_gate, float* o_gate, float* c_tilde
    ) const;

    // Linear projection: out = W_proj * h + b_proj
    void project(const float* h, float* out) const;

    // --- Activation functions ---
    static void sigmoid_vec(const float* src, float* dst, size_t n);
    static void tanh_vec(const float* src, float* dst, size_t n);
    static float sigmoid_scalar(float x);
    static float tanh_scalar(float x);

    // --- Weight file helpers ---
    static void write_vec(std::ofstream& f, const std::vector<float>& v);
    static void read_vec(std::ifstream& f, std::vector<float>& v);
};

} // namespace neural::lstm
