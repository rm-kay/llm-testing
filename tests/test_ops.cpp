#include <cmath>
#include <vector>

#include "llm/ops.hpp"
#include "test_framework.hpp"

using namespace llm;

// ---- matmul -------------------------------------------------------------

TEST(matmul_small_known_values) {
    // [2x3] * [3x2] with hand-computed result.
    Tensor a({2, 3}, {1, 2, 3,
                      4, 5, 6});
    Tensor b({3, 2}, {7, 8,
                      9, 10,
                      11, 12});
    Tensor c = ops::matmul(a, b);
    ASSERT_TRUE(c.rows() == 2 && c.cols() == 2);
    ASSERT_NEAR(c.at(0, 0), 58, 1e-5);   // 1*7 + 2*9 + 3*11
    ASSERT_NEAR(c.at(0, 1), 64, 1e-5);   // 1*8 + 2*10 + 3*12
    ASSERT_NEAR(c.at(1, 0), 139, 1e-5);  // 4*7 + 5*9 + 6*11
    ASSERT_NEAR(c.at(1, 1), 154, 1e-5);  // 4*8 + 5*10 + 6*12
}

TEST(matmul_identity_is_noop) {
    Tensor a({2, 2}, {3, 5, 7, 9});
    Tensor id({2, 2}, {1, 0, 0, 1});
    Tensor c = ops::matmul(a, id);
    for (int i = 0; i < a.size(); ++i) ASSERT_NEAR(c.data[i], a.data[i], 1e-6);
}

// ---- linear -------------------------------------------------------------

TEST(linear_matches_manual) {
    // x[1x2], w[3x2] (out=3, in=2), bias[3]
    Tensor x({1, 2}, {1, 2});
    Tensor w({3, 2}, {1, 0,
                      0, 1,
                      1, 1});
    Tensor b({3}, {10, 20, 30});
    Tensor y = ops::linear(x, w, b);
    ASSERT_TRUE(y.rows() == 1 && y.cols() == 3);
    ASSERT_NEAR(y.at(0, 0), 1 * 1 + 2 * 0 + 10, 1e-5);
    ASSERT_NEAR(y.at(0, 1), 1 * 0 + 2 * 1 + 20, 1e-5);
    ASSERT_NEAR(y.at(0, 2), 1 * 1 + 2 * 1 + 30, 1e-5);
}

TEST(linear_no_bias) {
    Tensor x({1, 2}, {3, 4});
    Tensor w({1, 2}, {1, 1});
    Tensor y = ops::linear(x, w, Tensor{});
    ASSERT_NEAR(y.at(0, 0), 7, 1e-6);
}

// ---- layernorm ----------------------------------------------------------

TEST(layernorm_gives_zero_mean_unit_var) {
    Tensor x({2, 4}, {1, 2, 3, 4,
                      10, 10, 10, 14});
    Tensor g({4}, {1, 1, 1, 1});
    Tensor b({4}, {0, 0, 0, 0});
    ops::layernorm_inplace(x, g, b, 1e-5f);

    for (int r = 0; r < 2; ++r) {
        float mean = 0, var = 0;
        for (int c = 0; c < 4; ++c) mean += x.at(r, c);
        mean /= 4;
        for (int c = 0; c < 4; ++c) var += (x.at(r, c) - mean) * (x.at(r, c) - mean);
        var /= 4;
        ASSERT_NEAR(mean, 0.0, 1e-5);
        ASSERT_NEAR(var, 1.0, 1e-3);  // slightly < 1 due to eps
    }
}

TEST(layernorm_applies_affine) {
    Tensor x({1, 3}, {1, 2, 3});
    Tensor g({3}, {2, 2, 2});
    Tensor b({3}, {5, 5, 5});
    ops::layernorm_inplace(x, g, b, 1e-5f);
    // Symmetric input => middle element normalizes to ~0 => value ~= beta.
    ASSERT_NEAR(x.at(0, 1), 5.0, 1e-3);
}

// ---- gelu ---------------------------------------------------------------

TEST(gelu_known_points) {
    Tensor x({1, 5}, {-3, -1, 0, 1, 3});
    ops::gelu_inplace(x);
    ASSERT_NEAR(x.at(0, 2), 0.0, 1e-6);          // gelu(0) == 0
    ASSERT_NEAR(x.at(0, 3), 0.841192, 1e-3);     // gelu(1) ~= 0.8412
    ASSERT_NEAR(x.at(0, 1), -0.158808, 1e-3);    // gelu(-1) ~= -0.1588
    // GELU is NOT globally monotonic (it dips below 0 for negative x), but it
    // is monotonic on the positive side and approaches identity as x grows.
    ASSERT_TRUE(x.at(0, 4) > x.at(0, 3));        // gelu(3) > gelu(1)
    ASSERT_NEAR(x.at(0, 4), 2.9964, 1e-2);       // gelu(3) ~= 3
}

// ---- softmax ------------------------------------------------------------

TEST(softmax_sums_to_one_and_orders) {
    std::vector<float> v = {1.0f, 2.0f, 3.0f};
    ops::softmax_inplace(v.data(), 3);
    float sum = v[0] + v[1] + v[2];
    ASSERT_NEAR(sum, 1.0, 1e-6);
    ASSERT_TRUE(v[2] > v[1] && v[1] > v[0]);  // preserves ordering
}

TEST(softmax_stable_for_large_inputs) {
    std::vector<float> v = {1000.0f, 1000.0f};  // would overflow without the max-shift
    ops::softmax_inplace(v.data(), 2);
    ASSERT_NEAR(v[0], 0.5, 1e-6);
    ASSERT_NEAR(v[1], 0.5, 1e-6);
}

// ---- attention ----------------------------------------------------------

TEST(attention_is_causal) {
    // 3 tokens, C=2, single head. Give distinct value vectors so any leakage
    // from future positions would change token 0's output.
    const int T = 3, C = 2;
    Tensor qkv({T, 3 * C});
    for (int t = 0; t < T; ++t) {
        // q and k all zero => uniform attention over the *allowed* keys.
        qkv.at(t, 0) = 0; qkv.at(t, 1) = 0;                 // q
        qkv.at(t, 2) = 0; qkv.at(t, 3) = 0;                 // k
        qkv.at(t, 4) = (float)(t + 1); qkv.at(t, 5) = 0;    // v = [t+1, 0]
    }
    Tensor out = ops::causal_attention(qkv, /*n_head=*/1);
    // Token 0 can only see itself => output value == its own v == 1.
    ASSERT_NEAR(out.at(0, 0), 1.0, 1e-5);
    // Token 2 sees tokens 0,1,2 uniformly => mean of {1,2,3} == 2.
    ASSERT_NEAR(out.at(2, 0), 2.0, 1e-5);
}
