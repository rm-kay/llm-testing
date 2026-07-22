#include "llm/model.hpp"

#include <cassert>
#include <random>

#include "llm/ops.hpp"

namespace llm {

namespace {

// Deterministic Gaussian fill for a freshly-shaped tensor.
Tensor randn(std::vector<int> shape, std::mt19937_64& rng, float std_dev) {
    Tensor t(std::move(shape));
    std::normal_distribution<float> dist(0.0f, std_dev);
    for (float& v : t.data) v = dist(rng);
    return t;
}

// Layer-norm weights are conventionally initialized to gamma=1, beta=0.
Tensor ones(int n)  { return Tensor({n}, std::vector<float>(n, 1.0f)); }
Tensor zeros(int n) { return Tensor({n}, std::vector<float>(n, 0.0f)); }

}  // namespace

void GPT::random_init(uint64_t seed) {
    std::mt19937_64 rng(seed);
    const int C = cfg_.n_embd;
    const float s = 0.02f;

    w_.wte = randn({cfg_.vocab_size, C}, rng, s);
    w_.wpe = randn({cfg_.n_ctx, C}, rng, s);

    w_.blocks.clear();
    w_.blocks.reserve(cfg_.n_layer);
    for (int l = 0; l < cfg_.n_layer; ++l) {
        BlockWeights b;
        b.ln1_g = ones(C);              b.ln1_b = zeros(C);
        b.attn_w = randn({3 * C, C}, rng, s);      b.attn_b = zeros(3 * C);
        b.attn_proj_w = randn({C, C}, rng, s);     b.attn_proj_b = zeros(C);
        b.ln2_g = ones(C);              b.ln2_b = zeros(C);
        b.fc_w = randn({4 * C, C}, rng, s);        b.fc_b = zeros(4 * C);
        b.proj_w = randn({C, 4 * C}, rng, s);      b.proj_b = zeros(C);
        w_.blocks.push_back(std::move(b));
    }

    w_.lnf_g = ones(C);
    w_.lnf_b = zeros(C);
    w_.lm_head = randn({cfg_.vocab_size, C}, rng, s);
}

Tensor GPT::forward(const std::vector<int>& tokens) const {
    const int T = static_cast<int>(tokens.size());
    const int C = cfg_.n_embd;
    assert(T > 0 && T <= cfg_.n_ctx);

    // --- Embedding: token embedding + positional embedding ---
    Tensor x({T, C});
    for (int t = 0; t < T; ++t) {
        const int id = tokens[t];
        assert(id >= 0 && id < cfg_.vocab_size);
        const float* tok = w_.wte.row_ptr(id);
        const float* pos = w_.wpe.row_ptr(t);
        float* dst = x.row_ptr(t);
        for (int c = 0; c < C; ++c) dst[c] = tok[c] + pos[c];
    }

    // --- Transformer blocks ---
    for (const BlockWeights& b : w_.blocks) {
        // Attention sub-layer with residual connection.
        Tensor a = x;  // copy so we can normalize without touching the residual
        ops::layernorm_inplace(a, b.ln1_g, b.ln1_b, cfg_.ln_eps);
        Tensor qkv  = ops::linear(a, b.attn_w, b.attn_b);      // [T, 3C]
        Tensor attn = ops::causal_attention(qkv, cfg_.n_head); // [T, C]
        attn = ops::linear(attn, b.attn_proj_w, b.attn_proj_b);
        for (int i = 0; i < x.size(); ++i) x.data[i] += attn.data[i];

        // MLP sub-layer with residual connection.
        Tensor m = x;
        ops::layernorm_inplace(m, b.ln2_g, b.ln2_b, cfg_.ln_eps);
        m = ops::linear(m, b.fc_w, b.fc_b);   // [T, 4C]
        ops::gelu_inplace(m);
        m = ops::linear(m, b.proj_w, b.proj_b);  // [T, C]
        for (int i = 0; i < x.size(); ++i) x.data[i] += m.data[i];
    }

    // --- Final norm + unembedding to vocab logits ---
    ops::layernorm_inplace(x, w_.lnf_g, w_.lnf_b, cfg_.ln_eps);
    return ops::linear(x, w_.lm_head, Tensor{});  // [T, vocab], no bias
}

}  // namespace llm
