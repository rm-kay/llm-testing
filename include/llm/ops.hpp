#pragma once

#include "llm/tensor.hpp"

// Core numerical building blocks used by the model.
//
// Every kernel here is written in the most straightforward way possible
// (plain nested loops, one output allocation per call). This is where most of
// the hand-optimization opportunity lives: blocking/tiling the matmul, fusing
// the bias/activation, reusing scratch buffers, SIMD, threading, etc.
namespace llm::ops {

// Plain matrix multiply: a[M, K] x b[K, N] -> [M, N].
Tensor matmul(const Tensor& a, const Tensor& b);

// Affine layer, PyTorch nn.Linear convention: weight is stored [out, in].
//   x: [T, in], w: [out, in], b: [out] (may be empty for no bias)
//   returns [T, out], where out[t, o] = sum_i x[t, i] * w[o, i] + b[o]
Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b);

// Row-wise layer norm, in place. Each row of x (length C) is normalized to
// zero mean / unit variance, then scaled by gamma and shifted by beta.
//   x: [T, C], gamma: [C], beta: [C]
void layernorm_inplace(Tensor& x, const Tensor& gamma, const Tensor& beta, float eps);

// GELU activation (tanh approximation, GPT-2 style), applied element-wise.
void gelu_inplace(Tensor& x);

// Numerically-stable softmax over the first `n` elements of `row`, in place.
void softmax_inplace(float* row, int n);

// Causal (masked) multi-head self-attention core.
//   qkv: [T, 3C], columns laid out as [q(C) | k(C) | v(C)]
//   returns [T, C]
// Query position t may only attend to key positions s <= t.
Tensor causal_attention(const Tensor& qkv, int n_head);

}  // namespace llm::ops
