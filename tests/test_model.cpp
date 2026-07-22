#include <cmath>
#include <vector>

#include "llm/config.hpp"
#include "llm/model.hpp"
#include "test_framework.hpp"

using namespace llm;

static Config tiny_config() {
    Config c;
    c.vocab_size = 32;
    c.n_ctx      = 16;
    c.n_embd     = 16;
    c.n_head     = 2;
    c.n_layer    = 2;
    return c;
}

TEST(forward_output_shape) {
    GPT model(tiny_config());
    model.random_init(123);
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    Tensor logits = model.forward(tokens);
    ASSERT_TRUE(logits.rows() == (int)tokens.size());
    ASSERT_TRUE(logits.cols() == tiny_config().vocab_size);
}

TEST(forward_is_finite) {
    GPT model(tiny_config());
    model.random_init(7);
    std::vector<int> tokens = {0, 5, 10, 15, 20, 31};
    Tensor logits = model.forward(tokens);
    for (float v : logits.data) ASSERT_TRUE(std::isfinite(v));
}

TEST(forward_is_deterministic) {
    // Same seed + same tokens => bit-for-bit identical logits.
    GPT a(tiny_config());
    a.random_init(99);
    GPT b(tiny_config());
    b.random_init(99);

    std::vector<int> tokens = {3, 1, 4, 1, 5, 9, 2};
    Tensor la = a.forward(tokens);
    Tensor lb = b.forward(tokens);
    ASSERT_TRUE(la.size() == lb.size());
    for (int i = 0; i < la.size(); ++i) ASSERT_NEAR(la.data[i], lb.data[i], 0.0);
}

TEST(forward_causality_prefix_stability) {
    // Because attention is causal, the logits at position t depend only on
    // tokens[0..t]. Extending the sequence must not change earlier rows.
    GPT model(tiny_config());
    model.random_init(2024);

    std::vector<int> shortseq = {7, 8, 9};
    std::vector<int> longseq  = {7, 8, 9, 10, 11};

    Tensor ls = model.forward(shortseq);
    Tensor ll = model.forward(longseq);

    const int V = tiny_config().vocab_size;
    for (int t = 0; t < (int)shortseq.size(); ++t) {
        for (int v = 0; v < V; ++v) {
            ASSERT_NEAR(ls.at(t, v), ll.at(t, v), 1e-4);
        }
    }
}

TEST(single_token_forward_runs) {
    GPT model(tiny_config());
    model.random_init(1);
    Tensor logits = model.forward({5});
    ASSERT_TRUE(logits.rows() == 1);
    for (float v : logits.data) ASSERT_TRUE(std::isfinite(v));
}
