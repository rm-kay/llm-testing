#include <cstdio>
#include <vector>

#include "bench_framework.hpp"
#include "llm/config.hpp"
#include "llm/model.hpp"
#include "llm/ops.hpp"

using namespace llm;

// Keep results observable so the optimizer can't delete the work we're timing.
static volatile float g_sink = 0.0f;
static void keep(const Tensor& t) { g_sink += t.data.empty() ? 0.0f : t.data[0]; }

static Tensor random_tensor(std::vector<int> shape, uint64_t seed) {
    Tensor t(std::move(shape));
    uint64_t s = seed;
    for (float& v : t.data) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;  // LCG
        v = ((s >> 33) / (float)(1ULL << 31)) - 1.0f;             // ~[-1, 1]
    }
    return t;
}

int main() {
    bench::print_header();

    // ---- matmul: [M,K] x [K,N], FLOPs = 2*M*N*K ----
    {
        const int M = 256, K = 256, N = 256;
        Tensor a = random_tensor({M, K}, 1);
        Tensor b = random_tensor({K, N}, 2);
        auto r = bench::measure("matmul 256x256x256",
                                [&] { keep(ops::matmul(a, b)); },
                                20, 3, 2.0 * M * N * K);
        bench::print(r);
    }

    // ---- linear: x[T,in] * w[out,in]^T, FLOPs = 2*T*in*out ----
    {
        const int T = 128, in = 512, out = 512;
        Tensor x = random_tensor({T, in}, 3);
        Tensor w = random_tensor({out, in}, 4);
        Tensor bias = random_tensor({out}, 5);
        auto r = bench::measure("linear 128x512x512",
                                [&] { keep(ops::linear(x, w, bias)); },
                                20, 3, 2.0 * T * in * out);
        bench::print(r);
    }

    // ---- layernorm over [T, C] ----
    {
        const int T = 512, C = 768;
        Tensor x = random_tensor({T, C}, 6);
        Tensor g = random_tensor({C}, 7);
        Tensor bb = random_tensor({C}, 8);
        auto r = bench::measure("layernorm 512x768",
                                [&] { Tensor y = x; ops::layernorm_inplace(y, g, bb, 1e-5f); keep(y); });
        bench::print(r);
    }

    // ---- gelu over [T, C] ----
    {
        const int T = 512, C = 3072;
        Tensor x = random_tensor({T, C}, 9);
        auto r = bench::measure("gelu 512x3072",
                                [&] { Tensor y = x; ops::gelu_inplace(y); keep(y); });
        bench::print(r);
    }

    // ---- causal attention core, qkv[T, 3C] ----
    {
        const int T = 256, C = 256, n_head = 8;
        Tensor qkv = random_tensor({T, 3 * C}, 10);
        auto r = bench::measure("attention T=256 C=256 h=8",
                                [&] { keep(ops::causal_attention(qkv, n_head)); },
                                20, 3);
        bench::print(r);
    }

    // ---- full model forward pass (default config) ----
    {
        Config cfg;  // defaults from config.hpp
        GPT model(cfg);
        model.random_init(42);
        std::vector<int> tokens;
        const int T = 64;
        for (int i = 0; i < T; ++i) tokens.push_back(i % cfg.vocab_size);
        auto r = bench::measure("forward T=64 (default cfg)",
                                [&] { keep(model.forward(tokens)); },
                                20, 3);
        bench::print(r);
    }

    std::printf("\n(sink=%g — ignore, exists only to defeat dead-code elimination)\n",
                (float)g_sink);
    return 0;
}
