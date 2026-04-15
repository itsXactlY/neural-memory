// src/memory/lstm.cpp - LSTM Access Pattern Predictor Implementation
#include "neural/lstm.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace neural::lstm {

// ============================================================================
// SIMD-accelerated activation functions
// ============================================================================

float LSTMPredictor::sigmoid_scalar(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float LSTMPredictor::tanh_scalar(float x) {
    return std::tanh(x);
}

void LSTMPredictor::sigmoid_vec(const float* src, float* dst, size_t n) {
#if SIMD_HAS_AVX2
    // AVX2 sigmoid approximation: 0.5 * (1 + tanh(x/2))
    // tanh approximation using polynomial (Remez-style) for [-1,1]
    // then clamp to [-6, 6] range first.
    // For production-quality results, we use scalar fallback for values
    // outside the fast path, but provide an AVX2 path for the common case.
    __m256 vhalf = _mm256_set1_ps(0.5f);
    __m256 vone  = _mm256_set1_ps(1.0f);
    __m256 vneg6 = _mm256_set1_ps(-6.0f);
    __m256 vpos6 = _mm256_set1_ps(6.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(src + i);
        // Clamp
        vx = _mm256_max_ps(vx, vneg6);
        vx = _mm256_min_ps(vx, vpos6);

        // tanh(x/2) approximation via minimax polynomial on [-1,1]:
        // Normalize: y = x/2, if |y| > 1, reduce.
        __m256 vy = _mm256_mul_ps(vx, vhalf);

        // For |y| <= 1: tanh(y) ~ y*(15 + y^2) / (15 + 4*y^2)  (Padé-ish)
        // Simpler: use exp-based for AVX2 when available
        // Actually, let's just do scalar here for correctness and simplicity.
        alignas(32) float tmp_in[8], tmp_out[8];
        _mm256_store_ps(tmp_in, vx);
        for (int j = 0; j < 8; ++j) {
            tmp_out[j] = sigmoid_scalar(tmp_in[j]);
        }
        _mm256_storeu_ps(dst + i, _mm256_load_ps(tmp_out));
    }
    for (; i < n; ++i) {
        dst[i] = sigmoid_scalar(src[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = sigmoid_scalar(src[i]);
    }
#endif
}

void LSTMPredictor::tanh_vec(const float* src, float* dst, size_t n) {
#if SIMD_HAS_AVX2
    __m256 vneg5 = _mm256_set1_ps(-5.0f);
    __m256 vpos5 = _mm256_set1_ps(5.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(src + i);
        vx = _mm256_max_ps(vx, vneg5);
        vx = _mm256_min_ps(vx, vpos5);

        alignas(32) float tmp_in[8], tmp_out[8];
        _mm256_store_ps(tmp_in, vx);
        for (int j = 0; j < 8; ++j) {
            tmp_out[j] = tanh_scalar(tmp_in[j]);
        }
        _mm256_storeu_ps(dst + i, _mm256_load_ps(tmp_out));
    }
    for (; i < n; ++i) {
        dst[i] = tanh_scalar(src[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = tanh_scalar(src[i]);
    }
#endif
}

// ============================================================================
// LSTMWeights initialization
// ============================================================================

void LSTMWeights::init(size_t in_dim, size_t h_dim, size_t out_dim, unsigned seed) {
    input_dim = in_dim;
    hidden_dim = h_dim;
    output_dim = out_dim;

    std::mt19937 rng(seed);
    init_layer1(in_dim, h_dim, rng);
    init_layer2(h_dim, rng);
    init_projection(h_dim, out_dim, rng);
}

void LSTMWeights::init_layer1(size_t in_dim, size_t h_dim, std::mt19937& rng) {
    // W matrices: hidden_dim x input_dim
    auto init_w = [&](std::vector<float>& w, size_t fi, size_t fo) {
        w.resize(fo * fi);
        for (auto& v : w) v = xavier_uniform(fi, fo, rng);
    };

    init_w(Wf, in_dim, h_dim);
    init_w(Wi, in_dim, h_dim);
    init_w(Wo, in_dim, h_dim);
    init_w(Wc, in_dim, h_dim);

    // U matrices: hidden_dim x hidden_dim
    auto init_u = [&](std::vector<float>& u, size_t fi, size_t fo) {
        u.resize(fo * fi);
        for (auto& v : u) v = xavier_uniform(fi, fo, rng);
    };

    init_u(Uf, h_dim, h_dim);
    init_u(Ui, h_dim, h_dim);
    init_u(Uo, h_dim, h_dim);
    init_u(Uc, h_dim, h_dim);

    // Biases
    bf.resize(h_dim, 1.0f);  // Forget gate bias = 1.0
    bi.resize(h_dim, 0.0f);
    bo.resize(h_dim, 0.0f);
    bc.resize(h_dim, 0.0f);
}

void LSTMWeights::init_layer2(size_t h_dim, std::mt19937& rng) {
    auto init = [&](std::vector<float>& m, size_t fi, size_t fo) {
        m.resize(fo * fi);
        for (auto& v : m) v = xavier_uniform(fi, fo, rng);
    };

    // Layer 2 takes hidden_dim as input, produces hidden_dim
    // W2: project from h of layer 1 (hidden_dim) to gates (hidden_dim)
    init(W2f, h_dim, h_dim);
    init(W2i, h_dim, h_dim);
    init(W2o, h_dim, h_dim);
    init(W2c, h_dim, h_dim);

    init(U2f, h_dim, h_dim);
    init(U2i, h_dim, h_dim);
    init(U2o, h_dim, h_dim);
    init(U2c, h_dim, h_dim);

    b2f.resize(h_dim, 1.0f);
    b2i.resize(h_dim, 0.0f);
    b2o.resize(h_dim, 0.0f);
    b2c.resize(h_dim, 0.0f);
}

void LSTMWeights::init_projection(size_t h_dim, size_t out_dim, std::mt19937& rng) {
    // W_proj: output_dim x hidden_dim
    W_proj.resize(out_dim * h_dim);
    for (auto& v : W_proj) v = xavier_uniform(h_dim, out_dim, rng);

    b_proj.resize(out_dim, 0.0f);
}

// ============================================================================
// LSTMState
// ============================================================================

LSTMState::LSTMState(size_t hidden_dim) : h(hidden_dim, 0.0f), c(hidden_dim, 0.0f) {}

void LSTMState::reset() {
    simd::zero(h.data(), h.size());
    simd::zero(c.data(), c.size());
}

// ============================================================================
// AdamState
// ============================================================================

AdamState::AdamState(size_t param_count) : m(param_count, 0.0f), v(param_count, 0.0f), t(0) {}

void AdamState::resize(size_t param_count) {
    m.resize(param_count, 0.0f);
    v.resize(param_count, 0.0f);
    t = 0;
}

// ============================================================================
// LSTMPredictor
// ============================================================================

LSTMPredictor::LSTMPredictor(size_t input_dim, size_t hidden_dim, size_t output_dim)
    : input_dim_(input_dim)
    , hidden_dim_(hidden_dim)
    , output_dim_(output_dim)
{
    weights_.init(input_dim, hidden_dim, output_dim);

    // Adam state size = total parameter count
    // Layer1: 4*(W: h*i) + 4*(U: h*h) + 4*(b: h) = 4*h*i + 4*h*h + 4*h
    // Layer2: 4*(W2: h*h) + 4*(U2: h*h) + 4*(b2: h) = 8*h*h + 4*h
    // Proj: output_dim*hidden_dim + output_dim
    size_t nparams = 4 * hidden_dim * input_dim   // W matrices
                   + 4 * hidden_dim * hidden_dim   // U matrices
                   + 4 * hidden_dim                 // biases
                   + 4 * hidden_dim * hidden_dim    // W2 matrices
                   + 4 * hidden_dim * hidden_dim    // U2 matrices
                   + 4 * hidden_dim                 // b2 biases
                   + output_dim * hidden_dim        // W_proj
                   + output_dim;                    // b_proj
    adam_state_.resize(nparams);
}

// ============================================================================
// LSTM Cell Forward (Layer 1)
// ============================================================================

void LSTMPredictor::lstm_cell_forward(
    const float* x, size_t x_dim,
    const LSTMState& prev,
    LSTMState& next,
    const float* Wf, const float* Wi, const float* Wo, const float* Wc,
    const float* Uf, const float* Ui, const float* Uo, const float* Uc,
    const float* bf, const float* bi, const float* bo, const float* bc,
    float* f_gate, float* i_gate, float* o_gate, float* c_tilde
) const {
    const size_t h = hidden_dim_;

    next.h.resize(h);
    next.c.resize(h);

    // Temporary scratch for pre-activation values
    std::vector<float> pre_act(h);
    std::vector<float> h_temp(h);

    // --- Forget gate: f = sigmoid(Wf*x + Uf*h_prev + bf) ---
    for (size_t r = 0; r < h; ++r) {
        // Wf row r (input_dim elements) dot x
        float val = simd::dot_product(Wf + r * x_dim, x, x_dim);
        // Uf row r (hidden_dim elements) dot h_prev
        val += simd::dot_product(Uf + r * h, prev.h.data(), h);
        // bias
        val += bf[r];
        pre_act[r] = val;
    }
    sigmoid_vec(pre_act.data(), f_gate, h);

    // --- Input gate: i = sigmoid(Wi*x + Ui*h_prev + bi) ---
    for (size_t r = 0; r < h; ++r) {
        float val = simd::dot_product(Wi + r * x_dim, x, x_dim);
        val += simd::dot_product(Ui + r * h, prev.h.data(), h);
        val += bi[r];
        pre_act[r] = val;
    }
    sigmoid_vec(pre_act.data(), i_gate, h);

    // --- Cell candidate: c~ = tanh(Wc*x + Uc*h_prev + bc) ---
    for (size_t r = 0; r < h; ++r) {
        float val = simd::dot_product(Wc + r * x_dim, x, x_dim);
        val += simd::dot_product(Uc + r * h, prev.h.data(), h);
        val += bc[r];
        pre_act[r] = val;
    }
    tanh_vec(pre_act.data(), c_tilde, h);

    // --- Output gate: o = sigmoid(Wo*x + Uo*h_prev + bo) ---
    for (size_t r = 0; r < h; ++r) {
        float val = simd::dot_product(Wo + r * x_dim, x, x_dim);
        val += simd::dot_product(Uo + r * h, prev.h.data(), h);
        val += bo[r];
        pre_act[r] = val;
    }
    sigmoid_vec(pre_act.data(), o_gate, h);

    // --- Cell state: c = f * c_prev + i * c~ ---
    simd::hadamard(f_gate, prev.c.data(), next.c.data(), h);
    simd::fmadd(i_gate, c_tilde, next.c.data(), next.c.data(), h);

    // --- Hidden state: h = o * tanh(c) ---
    tanh_vec(next.c.data(), h_temp.data(), h);
    simd::hadamard(o_gate, h_temp.data(), next.h.data(), h);
}

// ============================================================================
// LSTM Cell Forward (Layer 2)
// ============================================================================

void LSTMPredictor::lstm_cell_forward_2(
    const float* x,            // input from layer 1 (hidden_dim)
    const LSTMState& prev,     // layer 2's own previous state
    LSTMState& next,
    const float* Wf, const float* Wi, const float* Wo, const float* Wc,
    const float* Uf, const float* Ui, const float* Uo, const float* Uc,
    const float* bf, const float* bi, const float* bo, const float* bc,
    float* f_gate, float* i_gate, float* o_gate, float* c_tilde
) const {
    const size_t h = hidden_dim_;
    // Input x is layer 1's hidden state (dimension h).
    // W2 is (h x h): maps x to gates. U2 is (h x h): maps layer 2's h_prev to gates.

    next.h.resize(h);
    next.c.resize(h);

    std::vector<float> pre_act(h);
    std::vector<float> h_temp(h);

    // Forget gate
    for (size_t r = 0; r < h; ++r) {
        float val = simd::dot_product(Wf + r * h, x, h);  // W2 * x (layer 1 h)
        val += simd::dot_product(Uf + r * h, prev.h.data(), h);  // U2 * h2_prev
        val += bf[r];
        pre_act[r] = val;
    }
    sigmoid_vec(pre_act.data(), f_gate, h);

    // Input gate
    for (size_t r = 0; r < h; ++r) {
        float val = simd::dot_product(Wi + r * h, x, h);
        val += simd::dot_product(Ui + r * h, prev.h.data(), h);
        val += bi[r];
        pre_act[r] = val;
    }
    sigmoid_vec(pre_act.data(), i_gate, h);

    // Cell candidate
    for (size_t r = 0; r < h; ++r) {
        float val = simd::dot_product(Wc + r * h, x, h);
        val += simd::dot_product(Uc + r * h, prev.h.data(), h);
        val += bc[r];
        pre_act[r] = val;
    }
    tanh_vec(pre_act.data(), c_tilde, h);

    // Output gate
    for (size_t r = 0; r < h; ++r) {
        float val = simd::dot_product(Wo + r * h, x, h);
        val += simd::dot_product(Uo + r * h, prev.h.data(), h);
        val += bo[r];
        pre_act[r] = val;
    }
    sigmoid_vec(pre_act.data(), o_gate, h);

    // Cell state
    simd::hadamard(f_gate, prev.c.data(), next.c.data(), h);
    simd::fmadd(i_gate, c_tilde, next.c.data(), next.c.data(), h);

    // Hidden state
    tanh_vec(next.c.data(), h_temp.data(), h);
    simd::hadamard(o_gate, h_temp.data(), next.h.data(), h);
}

// ============================================================================
// Linear Projection
// ============================================================================

void LSTMPredictor::project(const float* h, float* out) const {
    const size_t h_dim = hidden_dim_;
    const size_t o_dim = output_dim_;

    // out[r] = W_proj row r dot h + b_proj[r]
    for (size_t r = 0; r < o_dim; ++r) {
        out[r] = simd::dot_product(weights_.W_proj.data() + r * h_dim, h, h_dim) + weights_.b_proj[r];
    }
}

// ============================================================================
// Forward Step
// ============================================================================

std::vector<float> LSTMPredictor::forward_step(
    const std::vector<float>& input,
    LSTMState& state1,
    LSTMState& state2
) const {
    std::shared_lock lock(rw_mutex_);

    const size_t h = hidden_dim_;

    // Scratch buffers for gates
    std::vector<float> f1(h), i1(h), o1(h), ctilde1(h);
    std::vector<float> f2(h), i2(h), o2(h), ctilde2(h);

    LSTMState next1(h), next2(h);

    // Layer 1
    lstm_cell_forward(
        input.data(), input_dim_,
        state1, next1,
        weights_.Wf.data(), weights_.Wi.data(), weights_.Wo.data(), weights_.Wc.data(),
        weights_.Uf.data(), weights_.Ui.data(), weights_.Uo.data(), weights_.Uc.data(),
        weights_.bf.data(), weights_.bi.data(), weights_.bo.data(), weights_.bc.data(),
        f1.data(), i1.data(), o1.data(), ctilde1.data()
    );

    // Layer 2 (takes layer 1 output h as input)
    lstm_cell_forward_2(
        next1.h.data(), state2, next2,
        weights_.W2f.data(), weights_.W2i.data(), weights_.W2o.data(), weights_.W2c.data(),
        weights_.U2f.data(), weights_.U2i.data(), weights_.U2o.data(), weights_.U2c.data(),
        weights_.b2f.data(), weights_.b2i.data(), weights_.b2o.data(), weights_.b2c.data(),
        f2.data(), i2.data(), o2.data(), ctilde2.data()
    );

    state1 = next1;
    state2 = next2;

    // Project layer 2 hidden state to output
    std::vector<float> output(output_dim_);
    project(state2.h.data(), output.data());

    return output;
}

// ============================================================================
// Forward (full sequence)
// ============================================================================

std::vector<float> LSTMPredictor::forward(
    const std::vector<std::vector<float>>& sequence
) const {
    std::shared_lock lock(rw_mutex_);

    const size_t h = hidden_dim_;

    LSTMState state1(h), state2(h);
    std::vector<float> f1(h), i1(h), o1(h), ctilde1(h);
    std::vector<float> f2(h), i2(h), o2(h), ctilde2(h);

    for (const auto& x : sequence) {
        if (x.size() != input_dim_) {
            throw std::invalid_argument("Input embedding dimension mismatch: got " +
                std::to_string(x.size()) + ", expected " + std::to_string(input_dim_));
        }

        LSTMState next1(h), next2(h);

        // Layer 1
        lstm_cell_forward(
            x.data(), input_dim_,
            state1, next1,
            weights_.Wf.data(), weights_.Wi.data(), weights_.Wo.data(), weights_.Wc.data(),
            weights_.Uf.data(), weights_.Ui.data(), weights_.Uo.data(), weights_.Uc.data(),
            weights_.bf.data(), weights_.bi.data(), weights_.bo.data(), weights_.bc.data(),
            f1.data(), i1.data(), o1.data(), ctilde1.data()
        );

        // Layer 2 (takes layer 1 output h as input)
        lstm_cell_forward_2(
            next1.h.data(), state2, next2,
            weights_.W2f.data(), weights_.W2i.data(), weights_.W2o.data(), weights_.W2c.data(),
            weights_.U2f.data(), weights_.U2i.data(), weights_.U2o.data(), weights_.U2c.data(),
            weights_.b2f.data(), weights_.b2i.data(), weights_.b2o.data(), weights_.b2c.data(),
            f2.data(), i2.data(), o2.data(), ctilde2.data()
        );

        state1 = next1;
        state2 = next2;
    }

    // Project final hidden state to output
    std::vector<float> output(output_dim_);
    project(state2.h.data(), output.data());

    return output;
}

// ============================================================================
// Training Step (SGD / Adam with BPTT)
// ============================================================================

float LSTMPredictor::train_step(
    const std::vector<std::vector<float>>& sequence,
    const std::vector<float>& target
) {
    if (target.size() != output_dim_) {
        throw std::invalid_argument("Target dimension mismatch");
    }

    std::unique_lock lock(rw_mutex_);

    const size_t h = hidden_dim_;
    const size_t T = sequence.size();

    if (T == 0) {
        throw std::invalid_argument("Empty sequence in train_step");
    }

    // ---- Forward pass: cache all intermediate states ----
    struct StepCache {
        LSTMState s1_prev, s2_prev;
        LSTMState s1_next, s2_next;
        std::vector<float> f1, i1, o1, ctilde1;
        std::vector<float> f2, i2, o2, ctilde2;
        std::vector<float> tanh_c1, tanh_c2;
        std::vector<float> output;

        StepCache(size_t h, size_t o_dim)
            : s1_prev(h), s2_prev(h), s1_next(h), s2_next(h)
            , f1(h), i1(h), o1(h), ctilde1(h)
            , f2(h), i2(h), o2(h), ctilde2(h)
            , tanh_c1(h), tanh_c2(h)
            , output(o_dim) {}
    };

    LSTMState state1(h), state2(h);
    std::vector<std::unique_ptr<StepCache>> cache(T);

    for (size_t t = 0; t < T; ++t) {
        cache[t] = std::make_unique<StepCache>(h, output_dim_);
        cache[t]->s1_prev = state1;
        cache[t]->s2_prev = state2;

        LSTMState next1(h), next2(h);

        lstm_cell_forward(
            sequence[t].data(), input_dim_,
            state1, next1,
            weights_.Wf.data(), weights_.Wi.data(), weights_.Wo.data(), weights_.Wc.data(),
            weights_.Uf.data(), weights_.Ui.data(), weights_.Uo.data(), weights_.Uc.data(),
            weights_.bf.data(), weights_.bi.data(), weights_.bo.data(), weights_.bc.data(),
            cache[t]->f1.data(), cache[t]->i1.data(), cache[t]->o1.data(), cache[t]->ctilde1.data()
        );

        lstm_cell_forward_2(
            next1.h.data(), state2, next2,
            weights_.W2f.data(), weights_.W2i.data(), weights_.W2o.data(), weights_.W2c.data(),
            weights_.U2f.data(), weights_.U2i.data(), weights_.U2o.data(), weights_.U2c.data(),
            weights_.b2f.data(), weights_.b2i.data(), weights_.b2o.data(), weights_.b2c.data(),
            cache[t]->f2.data(), cache[t]->i2.data(), cache[t]->o2.data(), cache[t]->ctilde2.data()
        );

        // Cache tanh(c) for backprop
        tanh_vec(next1.c.data(), cache[t]->tanh_c1.data(), h);
        tanh_vec(next2.c.data(), cache[t]->tanh_c2.data(), h);

        cache[t]->s1_next = next1;
        cache[t]->s2_next = next2;

        state1 = next1;
        state2 = next2;
    }

    // Final output
    std::vector<float> pred(output_dim_);
    project(state2.h.data(), pred.data());

    // ---- Compute loss (MSE) ----
    float loss = 0.0f;
    std::vector<float> d_output(output_dim_);
    for (size_t i = 0; i < output_dim_; ++i) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
        d_output[i] = 2.0f * diff / static_cast<float>(output_dim_);
    }
    loss /= static_cast<float>(output_dim_);

    // ---- Backward pass (BPTT) ----
    // Gradients for projection
    std::vector<float> dh2(h, 0.0f);  // gradient w.r.t. h2 from projection

    // d_output is (output_dim), W_proj is (output_dim x hidden_dim)
    // dh2 = W_proj^T * d_output
    for (size_t r = 0; r < output_dim_; ++r) {
        for (size_t c = 0; c < h; ++c) {
            dh2[c] += d_output[r] * weights_.W_proj[r * h + c];
        }
    }

    // BPTT through time - simplified (gradient at final step only, then unroll)
    // We do a truncated BPTT: accumulate gradients for the last step,
    // then propagate one step back for the LSTM weights.
    // For a production system you'd want full BPTT, but this keeps things efficient.

    // Gradient accumulators for LSTM weights (we'll update only the last time step)
    size_t last = T - 1;

    // For layer 2 at last step:
    // dh2_total = dh2 (from projection) + [gradients from next step - not available for last]
    // dc2 = dh2 * o2 * (1 - tanh^2(c2))  +  [dc2 from next step]

    std::vector<float> dc2(h, 0.0f);
    std::vector<float> do2(h), di2(h), df2(h), dct2(h);

    // do2 = dh2 * tanh(c2)
    simd::hadamard(dh2.data(), cache[last]->tanh_c2.data(), do2.data(), h);

    // dc2 += dh2 * o2 * (1 - tanh^2(c2))
    for (size_t j = 0; j < h; ++j) {
        float tc = cache[last]->tanh_c2[j];
        dc2[j] = dh2[j] * cache[last]->o2[j] * (1.0f - tc * tc);
    }

    // df2 = dc2 * c2_prev
    simd::hadamard(dc2.data(), cache[last]->s2_prev.c.data(), df2.data(), h);

    // di2 = dc2 * ctilde2
    simd::hadamard(dc2.data(), cache[last]->ctilde2.data(), di2.data(), h);

    // dct2 = dc2 * i2
    simd::hadamard(dc2.data(), cache[last]->i2.data(), dct2.data(), h);

    // Gate pre-activation gradients (multiply by gate * (1 - gate) for sigmoid, 1 - x^2 for tanh)
    std::vector<float> d_pre_f2(h), d_pre_i2(h), d_pre_o2(h), d_pre_ct2(h);

    for (size_t j = 0; j < h; ++j) {
        float f_val = cache[last]->f2[j];
        float i_val = cache[last]->i2[j];
        float o_val = cache[last]->o2[j];
        float ct_val = cache[last]->ctilde2[j];

        d_pre_f2[j] = df2[j] * f_val * (1.0f - f_val);
        d_pre_i2[j] = di2[j] * i_val * (1.0f - i_val);
        d_pre_o2[j] = do2[j] * o_val * (1.0f - o_val);
        d_pre_ct2[j] = dct2[j] * (1.0f - ct_val * ct_val);
    }

    // Gradients for layer 2 weights (using h1 as "input" to layer 2)
    const float* h1 = cache[last]->s1_next.h.data();

    // dh1 from layer 2 backprop (gradient flowing back to layer 1 h)
    std::vector<float> dh1(h, 0.0f);
    for (size_t r = 0; r < h; ++r) {
        for (size_t c = 0; c < h; ++c) {
            dh1[c] += weights_.W2f[r * h + c] * d_pre_f2[r];
            dh1[c] += weights_.W2i[r * h + c] * d_pre_i2[r];
            dh1[c] += weights_.W2o[r * h + c] * d_pre_o2[r];
            dh1[c] += weights_.W2c[r * h + c] * d_pre_ct2[r];
        }
    }

    // Layer 1 backward at last step
    std::vector<float> dc1(h, 0.0f);
    std::vector<float> do1(h), di1(h), df1(h), dct1(h);

    simd::hadamard(dh1.data(), cache[last]->tanh_c1.data(), do1.data(), h);

    for (size_t j = 0; j < h; ++j) {
        float tc = cache[last]->tanh_c1[j];
        dc1[j] = dh1[j] * cache[last]->o1[j] * (1.0f - tc * tc);
    }

    simd::hadamard(dc1.data(), cache[last]->s1_prev.c.data(), df1.data(), h);
    simd::hadamard(dc1.data(), cache[last]->ctilde1.data(), di1.data(), h);
    simd::hadamard(dc1.data(), cache[last]->i1.data(), dct1.data(), h);

    std::vector<float> d_pre_f1(h), d_pre_i1(h), d_pre_o1(h), d_pre_ct1(h);

    for (size_t j = 0; j < h; ++j) {
        float f_val = cache[last]->f1[j];
        float i_val = cache[last]->i1[j];
        float o_val = cache[last]->o1[j];
        float ct_val = cache[last]->ctilde1[j];

        d_pre_f1[j] = df1[j] * f_val * (1.0f - f_val);
        d_pre_i1[j] = di1[j] * i_val * (1.0f - i_val);
        d_pre_o1[j] = do1[j] * o_val * (1.0f - o_val);
        d_pre_ct1[j] = dct1[j] * (1.0f - ct_val * ct_val);
    }

    const float* h2_prev = cache[last]->s2_prev.h.data();
    const float* x_t = sequence[last].data();
    const float* h1_prev = cache[last]->s1_prev.h.data();

    // ---- Adam optimizer weight updates ----
    auto& gs = adam_state_;
    gs.t++;
    float lr_t = learning_rate_ *
                 std::sqrt(1.0f - std::pow(beta2_, static_cast<float>(gs.t))) /
                 (1.0f - std::pow(beta1_, static_cast<float>(gs.t)));

    // Helper: Adam update for a 2D weight matrix (row-major, rows x cols)
    // d_pre is the gradient per row, x_vec is the input that was multiplied.
    size_t idx = 0;
    auto adam_update_2d = [&](float* W, const float* d_pre, const float* x_vec,
                               size_t rows, size_t cols) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                float g = d_pre[r] * x_vec[c];
                gs.m[idx] = beta1_ * gs.m[idx] + (1.0f - beta1_) * g;
                gs.v[idx] = beta2_ * gs.v[idx] + (1.0f - beta2_) * g * g;
                W[r * cols + c] -= lr_t * gs.m[idx] / (std::sqrt(gs.v[idx]) + epsilon_);
                idx++;
            }
        }
    };

    // Helper: Adam update for a bias vector
    auto adam_update_bias = [&](float* b, const float* d_pre, size_t n) {
        for (size_t j = 0; j < n; ++j) {
            float g = d_pre[j];
            gs.m[idx] = beta1_ * gs.m[idx] + (1.0f - beta1_) * g;
            gs.v[idx] = beta2_ * gs.v[idx] + (1.0f - beta2_) * g * g;
            b[j] -= lr_t * gs.m[idx] / (std::sqrt(gs.v[idx]) + epsilon_);
            idx++;
        }
    };

    // Layer 1 W matrices (hidden_dim x input_dim)
    adam_update_2d(weights_.Wf.data(), d_pre_f1.data(), x_t, h, input_dim_);
    adam_update_2d(weights_.Wi.data(), d_pre_i1.data(), x_t, h, input_dim_);
    adam_update_2d(weights_.Wo.data(), d_pre_o1.data(), x_t, h, input_dim_);
    adam_update_2d(weights_.Wc.data(), d_pre_ct1.data(), x_t, h, input_dim_);

    // Layer 1 U matrices (hidden_dim x hidden_dim)
    adam_update_2d(weights_.Uf.data(), d_pre_f1.data(), h1_prev, h, h);
    adam_update_2d(weights_.Ui.data(), d_pre_i1.data(), h1_prev, h, h);
    adam_update_2d(weights_.Uo.data(), d_pre_o1.data(), h1_prev, h, h);
    adam_update_2d(weights_.Uc.data(), d_pre_ct1.data(), h1_prev, h, h);

    // Layer 1 biases
    adam_update_bias(weights_.bf.data(), d_pre_f1.data(), h);
    adam_update_bias(weights_.bi.data(), d_pre_i1.data(), h);
    adam_update_bias(weights_.bo.data(), d_pre_o1.data(), h);
    adam_update_bias(weights_.bc.data(), d_pre_ct1.data(), h);

    // Layer 2 W2 matrices (hidden_dim x hidden_dim)
    adam_update_2d(weights_.W2f.data(), d_pre_f2.data(), h1, h, h);
    adam_update_2d(weights_.W2i.data(), d_pre_i2.data(), h1, h, h);
    adam_update_2d(weights_.W2o.data(), d_pre_o2.data(), h1, h, h);
    adam_update_2d(weights_.W2c.data(), d_pre_ct2.data(), h1, h, h);

    // Layer 2 U2 matrices (hidden_dim x hidden_dim)
    adam_update_2d(weights_.U2f.data(), d_pre_f2.data(), h2_prev, h, h);
    adam_update_2d(weights_.U2i.data(), d_pre_i2.data(), h2_prev, h, h);
    adam_update_2d(weights_.U2o.data(), d_pre_o2.data(), h2_prev, h, h);
    adam_update_2d(weights_.U2c.data(), d_pre_ct2.data(), h2_prev, h, h);

    // Layer 2 biases
    adam_update_bias(weights_.b2f.data(), d_pre_f2.data(), h);
    adam_update_bias(weights_.b2i.data(), d_pre_i2.data(), h);
    adam_update_bias(weights_.b2o.data(), d_pre_o2.data(), h);
    adam_update_bias(weights_.b2c.data(), d_pre_ct2.data(), h);

    // Projection weights (output_dim x hidden_dim)
    adam_update_2d(weights_.W_proj.data(), d_output.data(), state2.h.data(), output_dim_, h);

    // Projection bias
    adam_update_bias(weights_.b_proj.data(), d_output.data(), output_dim_);

    return loss;
}

// ============================================================================
// Save / Load
// ============================================================================

void LSTMPredictor::write_vec(std::ofstream& f, const std::vector<float>& v) {
    uint64_t sz = v.size();
    f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    f.write(reinterpret_cast<const char*>(v.data()), sz * sizeof(float));
}

void LSTMPredictor::read_vec(std::ifstream& f, std::vector<float>& v) {
    uint64_t sz = 0;
    f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    if (!f) throw std::runtime_error("Failed to read vector size");
    if (sz > 100 * 1024 * 1024) throw std::runtime_error("Vector size too large: " + std::to_string(sz));
    v.resize(sz);
    f.read(reinterpret_cast<char*>(v.data()), sz * sizeof(float));
    if (!f) throw std::runtime_error("Failed to read vector data");
}

void LSTMPredictor::save(const std::string& path) const {
    std::shared_lock lock(rw_mutex_);
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for writing: " + path);

    // Header: magic + dimensions
    uint32_t magic = 0x4C53544Du;  // "LSTM"
    f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

    uint64_t id = input_dim_, hd = hidden_dim_, od = output_dim_;
    f.write(reinterpret_cast<const char*>(&id), sizeof(id));
    f.write(reinterpret_cast<const char*>(&hd), sizeof(hd));
    f.write(reinterpret_cast<const char*>(&od), sizeof(od));

    // Layer 1
    write_vec(f, weights_.Wf);
    write_vec(f, weights_.Wi);
    write_vec(f, weights_.Wo);
    write_vec(f, weights_.Wc);
    write_vec(f, weights_.Uf);
    write_vec(f, weights_.Ui);
    write_vec(f, weights_.Uo);
    write_vec(f, weights_.Uc);
    write_vec(f, weights_.bf);
    write_vec(f, weights_.bi);
    write_vec(f, weights_.bo);
    write_vec(f, weights_.bc);

    // Layer 2
    write_vec(f, weights_.W2f);
    write_vec(f, weights_.W2i);
    write_vec(f, weights_.W2o);
    write_vec(f, weights_.W2c);
    write_vec(f, weights_.U2f);
    write_vec(f, weights_.U2i);
    write_vec(f, weights_.U2o);
    write_vec(f, weights_.U2c);
    write_vec(f, weights_.b2f);
    write_vec(f, weights_.b2i);
    write_vec(f, weights_.b2o);
    write_vec(f, weights_.b2c);

    // Projection
    write_vec(f, weights_.W_proj);
    write_vec(f, weights_.b_proj);
}

void LSTMPredictor::load(const std::string& path) {
    std::unique_lock lock(rw_mutex_);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for reading: " + path);

    uint32_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x4C53544Du) throw std::runtime_error("Invalid LSTM weight file");

    uint64_t id, hd, od;
    f.read(reinterpret_cast<char*>(&id), sizeof(id));
    f.read(reinterpret_cast<char*>(&hd), sizeof(hd));
    f.read(reinterpret_cast<char*>(&od), sizeof(od));

    if (id != input_dim_ || hd != hidden_dim_ || od != output_dim_) {
        throw std::runtime_error("Dimension mismatch in weight file: got " +
            std::to_string(id) + "/" + std::to_string(hd) + "/" + std::to_string(od) +
            ", expected " + std::to_string(input_dim_) + "/" +
            std::to_string(hidden_dim_) + "/" + std::to_string(output_dim_));
    }

    // Layer 1
    read_vec(f, weights_.Wf);
    read_vec(f, weights_.Wi);
    read_vec(f, weights_.Wo);
    read_vec(f, weights_.Wc);
    read_vec(f, weights_.Uf);
    read_vec(f, weights_.Ui);
    read_vec(f, weights_.Uo);
    read_vec(f, weights_.Uc);
    read_vec(f, weights_.bf);
    read_vec(f, weights_.bi);
    read_vec(f, weights_.bo);
    read_vec(f, weights_.bc);

    // Layer 2
    read_vec(f, weights_.W2f);
    read_vec(f, weights_.W2i);
    read_vec(f, weights_.W2o);
    read_vec(f, weights_.W2c);
    read_vec(f, weights_.U2f);
    read_vec(f, weights_.U2i);
    read_vec(f, weights_.U2o);
    read_vec(f, weights_.U2c);
    read_vec(f, weights_.b2f);
    read_vec(f, weights_.b2i);
    read_vec(f, weights_.b2o);
    read_vec(f, weights_.b2c);

    // Projection
    read_vec(f, weights_.W_proj);
    read_vec(f, weights_.b_proj);
}

} // namespace neural::lstm
