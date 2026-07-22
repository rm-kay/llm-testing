#pragma once

#include <cstdint>
#include <vector>

#include "llm/config.hpp"
#include "llm/tensor.hpp"

namespace llm {

// Weights for a single transformer block. Shapes are annotated in terms of
// C = n_embd. Linear weights follow the [out, in] convention (see ops::linear).
struct BlockWeights {
    // Pre-attention layer norm.
    Tensor ln1_g, ln1_b;              // [C], [C]
    // Fused QKV projection and the output projection.
    Tensor attn_w, attn_b;            // [3C, C], [3C]
    Tensor attn_proj_w, attn_proj_b;  // [C, C],  [C]
    // Pre-MLP layer norm.
    Tensor ln2_g, ln2_b;              // [C], [C]
    // MLP: up-projection (x4), GELU, down-projection.
    Tensor fc_w, fc_b;                // [4C, C], [4C]
    Tensor proj_w, proj_b;            // [C, 4C], [C]
};

// Full set of model weights.
struct GPTWeights {
    Tensor wte;                        // token embedding    [vocab, C]
    Tensor wpe;                        // position embedding [n_ctx, C]
    std::vector<BlockWeights> blocks;  // n_layer transformer blocks
    Tensor lnf_g, lnf_b;               // final layer norm   [C], [C]
    Tensor lm_head;                    // output projection  [vocab, C] (no bias)
};

// A small GPT-style decoder-only transformer.
class GPT {
public:
    explicit GPT(const Config& cfg) : cfg_(cfg) {}

    // Fill all weights with deterministic pseudo-random values (Gaussian,
    // std=0.02, GPT-2 style init). Same seed => same weights => reproducible
    // tests and benchmarks.
    void random_init(uint64_t seed = 42);

    // Run the forward pass over a sequence of token ids.
    //   tokens: length T, each in [0, vocab_size)
    //   returns logits [T, vocab_size]
    Tensor forward(const std::vector<int>& tokens) const;

    const Config&     config()  const { return cfg_; }
    const GPTWeights& weights() const { return w_; }

private:
    Config     cfg_;
    GPTWeights w_;
};

}  // namespace llm
