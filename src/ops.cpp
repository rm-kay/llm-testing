#include "llm/ops.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace llm::ops {

Tensor transpose(const Tensor &a) {
  assert(a.shape.size() == 2);
  const int M = a.shape[0];
  const int N = a.shape[1];
  Tensor out({N, M});
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      out.at(j, i) = a.at(i, j);
    }
  }
  return out;
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  assert(a.shape.size() == 2 && b.shape.size() == 2);
  const int M = a.shape[0];
  const int K = a.shape[1];
  const int N = b.shape[1];
  assert(b.shape[0] == K);

  Tensor out({M, N});
  constexpr int TILE_WIDTH = 128;
  // triple loop with tiling: we load only a square for each at a time
  for (int ii = 0; ii < M; ii += TILE_WIDTH) {
    for (int kk = 0; kk < K; kk += TILE_WIDTH) {
      for (int jj = 0; jj < N; jj += TILE_WIDTH) {

        // loop ordering here auto vecs to simd
        for (int i = ii; i < ii + TILE_WIDTH && i < M; ++i) {
          for (int k = kk; k < kk + TILE_WIDTH && k < K; ++k) {
            for (int j = jj; j < jj + TILE_WIDTH && j < N; ++j) {
              out.at(i, j) += a.at(i, k) * b.at(k, j);
            }
          }
        }
      }
    }
  }
  return out;
}

Tensor linear(const Tensor &x, const Tensor &w, const Tensor &b) {
  assert(x.shape.size() == 2 && w.shape.size() == 2);
  const int T = x.shape[0];
  const int in = x.shape[1];
  const int out = w.shape[0];
  assert(w.shape[1] == in);
  const bool has_bias = !b.data.empty();
  assert(!has_bias || b.size() == out);

  // y has shape T x out
  // x is T x in, w is out x in, b is 1 x out
  // y is x @ w_t + b
  // y(t, o) = b(o) + dot(x_t, w_o)
  auto w_t = transpose(w);
  auto y = matmul(x, w_t);
  if (has_bias) {
    for (int t = 0; t < T; ++t) {
      for (int o = 0; o < out; ++o) {
        y.at(t, o) += b[o];
      }
    }
  }
  return y;
}

void layernorm_inplace(Tensor &x, const Tensor &gamma, const Tensor &beta,
                       float eps) {
  assert(x.shape.size() == 2);
  const int T = x.shape[0];
  const int C = x.shape[1];
  assert(gamma.size() == C && beta.size() == C);

  for (int t = 0; t < T; ++t) {
    float *row = x.row_ptr(t);

    float mean = 0.0f;
    for (int c = 0; c < C; ++c)
      mean += row[c];
    mean /= C;

    float var = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float d = row[c] - mean;
      var += d * d;
    }
    var /= C;

    const float inv_std = 1.0f / std::sqrt(var + eps);
    for (int c = 0; c < C; ++c) {
      row[c] = (row[c] - mean) * inv_std * gamma[c] + beta[c];
    }
  }
}

void gelu_inplace(Tensor &x) {
  // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  constexpr float kCoeff = 0.044715f;
  for (float &v : x.data) {
    const float x3 = v * v * v;
    const float inner = kSqrt2OverPi * (v + kCoeff * x3);
    v = 0.5f * v * (1.0f + std::tanh(inner));
  }
}

void softmax_inplace(float *row, int n) {
  float max_v = row[0];
  for (int i = 1; i < n; ++i)
    max_v = std::fmax(max_v, row[i]);

  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    row[i] = std::exp(row[i] - max_v);
    sum += row[i];
  }

  const float inv_sum = 1.0f / sum;
  for (int i = 0; i < n; ++i)
    row[i] *= inv_sum;
}

Tensor causal_attention(const Tensor &qkv, int n_head) {
  assert(qkv.shape.size() == 2);
  const int T = qkv.shape[0];
  const int C3 = qkv.shape[1];
  assert(C3 % 3 == 0);
  const int C = C3 / 3;
  assert(C % n_head == 0);
  const int hd = C / n_head; // per-head dimension
  const float scale = 1.0f / std::sqrt((float)hd);

  // Column offsets of q/k/v inside each row of `qkv`.
  const int q_off = 0;
  const int k_off = C;
  const int v_off = 2 * C;

  Tensor out({T, C});
  std::vector<float> scores(
      T); // reused per (head, query) — but reallocated per call

  for (int h = 0; h < n_head; ++h) {
    const int hoff = h * hd; // this head's slice within each C-wide block
    for (int t = 0; t < T; ++t) {
      const float *q = qkv.row_ptr(t) + q_off + hoff;

      // Attention logits against all allowed keys (s <= t).
      for (int s = 0; s <= t; ++s) {
        const float *k = qkv.row_ptr(s) + k_off + hoff;
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d)
          dot += q[d] * k[d];
        scores[s] = dot * scale;
      }

      softmax_inplace(scores.data(), t + 1);

      // Weighted sum of value vectors -> this head's output for token t.
      float *o = out.row_ptr(t) + hoff;
      for (int d = 0; d < hd; ++d)
        o[d] = 0.0f;
      for (int s = 0; s <= t; ++s) {
        const float *v = qkv.row_ptr(s) + v_off + hoff;
        const float p = scores[s];
        for (int d = 0; d < hd; ++d)
          o[d] += p * v[d];
      }
    }
  }
  return out;
}

} // namespace llm::ops
