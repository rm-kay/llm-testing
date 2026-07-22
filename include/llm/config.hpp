#pragma once

namespace llm {

// Hyperparameters for a small GPT-style decoder-only transformer.
//
// Defaults describe a tiny model that runs quickly on CPU. They are big enough
// that the naive kernels are measurable in the benchmarks, but small enough
// that the unit tests are instant.
struct Config {
    int   vocab_size = 96;     // size of the token vocabulary
    int   n_ctx      = 128;    // maximum sequence length (context window)
    int   n_embd     = 128;    // embedding / residual stream width (a.k.a. C)
    int   n_head     = 4;      // number of attention heads (must divide n_embd)
    int   n_layer    = 4;      // number of transformer blocks
    float ln_eps     = 1e-5f;  // epsilon for layer norm

    int head_dim() const { return n_embd / n_head; }
};

}  // namespace llm
