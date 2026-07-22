// Google Benchmark suite for the naive kernels.
//
// Sizes are chosen to *sweep across the cache hierarchy* so cache-blocking and
// access-pattern optimizations show a visible cliff. For this CPU (per-core,
// single-thread): L1d = 48 KiB, L2 = 1.25 MiB, L3 = 25 MiB (shared).
//
// Each row prints a `WS_MiB` counter = working-set size, so you can see which
// tier a given case lives in, plus a throughput counter (FLOP/s or bytes/s).
//
// Measurement tips (see README): pin to one P-core and avoid frequency noise:
//   taskset -c 0 ./build/benchmarks --benchmark_repetitions=10 \
//           --benchmark_report_aggregates_only=true
// Filter to one kernel with e.g. --benchmark_filter='BM_matmul'.

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include "llm/config.hpp"
#include "llm/model.hpp"
#include "llm/ops.hpp"

using namespace llm;

namespace {

// Deterministic pseudo-random fill (cheap LCG) so benchmarks are reproducible.
Tensor random_tensor(std::vector<int> shape, uint64_t seed) {
    Tensor t(std::move(shape));
    uint64_t s = seed;
    for (float& v : t.data) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        v = ((s >> 33) / static_cast<float>(1ULL << 31)) - 1.0f;  // ~[-1, 1]
    }
    return t;
}

// A per-iteration FLOP count reported as a rate (FLOP/s, shown with SI prefix).
benchmark::Counter flops(double per_iter) {
    return benchmark::Counter(per_iter, benchmark::Counter::kIsIterationInvariantRate,
                              benchmark::Counter::OneK::kIs1000);
}

// Report the working-set size (MiB) so the cache tier is obvious per row.
void set_working_set(benchmark::State& state, double bytes) {
    state.counters["WS_MiB"] = bytes / (1024.0 * 1024.0);
}

}  // namespace

// ---- matmul: square NxN x NxN, sweeps L1 -> L2 -> L3 -----------------------
// Working set = 3*N^2*4 bytes. N=64:48KiB(L1) ... 256:768KiB(L2) ... 1024:12MiB(L3).
// Add ->Arg(1536)/->Arg(2048) to reach DRAM (slow: 7-17 GFLOP/iter naive).
static void BM_matmul(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    Tensor a = random_tensor({N, N}, 1);
    Tensor b = random_tensor({N, N}, 2);
    for (auto _ : state) {
        Tensor c = ops::matmul(a, b);
        benchmark::DoNotOptimize(c.data.data());
        benchmark::ClobberMemory();
    }
    state.counters["FLOP/s"] = flops(2.0 * N * N * N);
    set_working_set(state, 3.0 * N * N * 4);
    state.SetComplexityN(N);
}
BENCHMARK(BM_matmul)->RangeMultiplier(2)->Range(64, 1024)->ArgName("N")
    ->Complexity()->Unit(benchmark::kMillisecond);

// ---- linear: x[T,in] * w[out,in]^T, square in=out=N -----------------------
// Dominant working set is w = N^2*4 bytes (re-streamed for each of T rows).
static void BM_linear(benchmark::State& state) {
    const int T = 128;
    const int N = static_cast<int>(state.range(0));
    Tensor x = random_tensor({T, N}, 3);
    Tensor w = random_tensor({N, N}, 4);
    Tensor bias = random_tensor({N}, 5);
    for (auto _ : state) {
        Tensor y = ops::linear(x, w, bias);
        benchmark::DoNotOptimize(y.data.data());
        benchmark::ClobberMemory();
    }
    state.counters["FLOP/s"] = flops(2.0 * T * N * N);
    set_working_set(state, (2.0 * T * N + N * N) * 4);
    state.SetComplexityN(N);
}
BENCHMARK(BM_linear)->RangeMultiplier(2)->Range(128, 2048)->ArgName("in=out")
    ->Complexity()->Unit(benchmark::kMillisecond);

// ---- layernorm: [R, 1024], sweep rows L1 -> DRAM (memory-bandwidth bound) --
static void BM_layernorm(benchmark::State& state) {
    const int C = 1024;
    const int R = static_cast<int>(state.range(0));
    Tensor x = random_tensor({R, C}, 6);
    Tensor g = random_tensor({C}, 7);
    Tensor b = random_tensor({C}, 8);
    for (auto _ : state) {
        Tensor y = x;  // copy so each iteration normalizes fresh data
        ops::layernorm_inplace(y, g, b, 1e-5f);
        benchmark::DoNotOptimize(y.data.data());
        benchmark::ClobberMemory();
    }
    // Reads x + writes x (the copy dominates); ~2 passes over R*C floats.
    state.SetBytesProcessed(int64_t(state.iterations()) * 2 * R * C * 4);
    set_working_set(state, 1.0 * R * C * 4);
}
BENCHMARK(BM_layernorm)->RangeMultiplier(4)->Range(16, 16384)->ArgName("rows")
    ->Unit(benchmark::kMicrosecond);

// ---- gelu: [R, 1024] element-wise (tanh: compute bound) -------------------
static void BM_gelu(benchmark::State& state) {
    const int C = 1024;
    const int R = static_cast<int>(state.range(0));
    Tensor base = random_tensor({R, C}, 9);
    for (auto _ : state) {
        Tensor y = base;
        ops::gelu_inplace(y);
        benchmark::DoNotOptimize(y.data.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(int64_t(state.iterations()) * R * C);
    set_working_set(state, 1.0 * R * C * 4);
}
BENCHMARK(BM_gelu)->RangeMultiplier(4)->Range(16, 16384)->ArgName("rows")
    ->Unit(benchmark::kMicrosecond);

// ---- causal attention: qkv[T, 3C], C=512, 8 heads; O(T^2) in sequence len --
static void BM_attention(benchmark::State& state) {
    const int C = 512, n_head = 8;
    const int T = static_cast<int>(state.range(0));
    Tensor qkv = random_tensor({T, 3 * C}, 10);
    for (auto _ : state) {
        Tensor out = ops::causal_attention(qkv, n_head);
        benchmark::DoNotOptimize(out.data.data());
        benchmark::ClobberMemory();
    }
    // ~2 flops per (query, key, dim) for scores + weighted sum, over T^2/2 pairs.
    state.counters["FLOP/s"] = flops(2.0 * T * T * C);
    set_working_set(state, 3.0 * T * C * 4);
    state.SetComplexityN(T);
}
BENCHMARK(BM_attention)->RangeMultiplier(2)->Range(64, 1024)->ArgName("T")
    ->Complexity()->Unit(benchmark::kMillisecond);

// ---- full forward pass at the default config, swept over context length ---
static void BM_forward(benchmark::State& state) {
    Config cfg;  // defaults from config.hpp (n_ctx = 128)
    const int T = static_cast<int>(state.range(0));
    GPT model(cfg);
    model.random_init(42);
    std::vector<int> tokens;
    for (int i = 0; i < T; ++i) tokens.push_back(i % cfg.vocab_size);
    for (auto _ : state) {
        Tensor logits = model.forward(tokens);
        benchmark::DoNotOptimize(logits.data.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(int64_t(state.iterations()) * T);  // tokens/s
}
BENCHMARK(BM_forward)->RangeMultiplier(2)->Range(16, 128)->ArgName("T")
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
