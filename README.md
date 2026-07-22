# llm-cpp

A small, **deliberately naive** GPT-style (decoder-only) transformer for
inference, written in readable C++17. The goal is to give you a clear codebase
to read, understand, and then **optimize by hand** — with unit tests and
benchmarks so you can verify correctness and measure speedups as you go.

This is **not** high-performance code. It uses textbook loops, allocates a fresh
tensor for every op, and has no threading or SIMD. That headroom is the point.

## Layout

```
include/llm/
  tensor.hpp     # flat row-major float32 tensor (owns its data)
  config.hpp     # model hyperparameters
  ops.hpp        # kernel declarations
  model.hpp      # weights + GPT class
src/
  ops.cpp        # matmul, linear, layernorm, gelu, softmax, attention  <- hot loops
  model.cpp      # weight init + forward pass (embedding -> blocks -> logits)
tests/
  test_framework.hpp   # ~50-line header-only test framework
  test_ops.cpp         # per-kernel correctness (known values, invariants)
  test_model.cpp       # shapes, finiteness, determinism, causality
bench/
  bench_main.cpp       # Google Benchmark suite, size-swept across cache tiers
examples/
  generate.cpp   # char-level tokenize -> forward -> greedy decode demo
```

## Model

A standard GPT-2-style decoder block, pre-norm:

```
x = wte[token] + wpe[position]
for each block:
    x = x + attn_proj( causal_multi_head_attention( ln1(x) ) )
    x = x + mlp_down( gelu( mlp_up( ln2(x) ) ) )
logits = lm_head( ln_f(x) )
```

Defaults (see `config.hpp`): vocab 96, ctx 128, n_embd 128, 4 heads, 4 layers.

## Build & run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # first run fetches Google Benchmark
cmake --build build -j

./build/unit_tests    # correctness
./build/benchmarks    # timings
./build/generate      # end-to-end demo (random weights => gibberish text)
ctest --test-dir build   # tests via CTest
```

Options:
- `-DLLM_NATIVE=ON` — build with `-march=native` (leave OFF for reproducible
  numbers; turn on when you start chasing SIMD/auto-vectorization).
- `-DLLM_BUILD_BENCHMARKS=OFF` — skip Google Benchmark (builds tests offline).
- `-DCMAKE_BUILD_TYPE=Debug` — enables `assert()` bounds checks in tensors.

### Benchmarks & the cache hierarchy

Benchmarks use [Google Benchmark](https://github.com/google/benchmark), fetched
automatically at configure time. Each kernel is **swept across sizes that cross
the CPU cache tiers**, so cache-blocking / access-pattern wins show a visible
cliff. Every row prints a `WS_MiB` counter (working-set size) plus a throughput
counter (`FLOP/s` or `bytes_per_second`) — read `WS_MiB` against your per-core
caches (`lscpu -C`) to see which tier a case lives in.

For example, the naive `matmul` collapses from ~9 GFLOP/s in L1 to <1 GFLOP/s
once the matrices spill to L3 — that gap is what blocking should recover.

Useful flags:
```bash
# Pin to one core + average repetitions => low-noise, comparable numbers.
taskset -c 0 ./build/benchmarks \
    --benchmark_repetitions=10 --benchmark_report_aggregates_only=true

./build/benchmarks --benchmark_filter='BM_matmul'   # one kernel (suite is slow)
./build/benchmarks --benchmark_out=baseline.json --benchmark_out_format=json
```
Compare two runs with `build/_deps/benchmark-src/tools/compare.py baseline.json new.json`.

Note: the biggest cases (e.g. `BM_matmul/N:1024`) take seconds per iteration
with the naive kernels, so a full run is tens of seconds — filter while
iterating. Google Benchmark will warn if CPU frequency scaling is enabled; for
the steadiest numbers set the governor to `performance` first.

## Suggested optimization workflow

1. Record a baseline:
   `taskset -c 0 ./build/benchmarks --benchmark_out=baseline.json --benchmark_out_format=json`.
2. Pick a kernel in `src/ops.cpp` (the `matmul`/`linear` triple loops are the
   juiciest targets).
3. Change it, then:
   - `./build/unit_tests` must stay green (correctness), and
   - re-run the benchmark and diff:
     `build/_deps/benchmark-src/tools/compare.py baseline.json new.json` (or eyeball `FLOP/s`
     across the size sweep — watch how the win grows as you cross into L3/DRAM).
4. `git commit` per optimization so you can bisect wins/regressions.

### Easy wins to try
- Reorder the `matmul` loops to `i, k, j` for a contiguous inner loop.
- Fuse bias + activation into `linear` instead of a separate pass.
- Reuse scratch buffers instead of allocating a new `Tensor` per op.
- Block/tile `matmul`; then add `#pragma omp parallel for` or SIMD.
- Cache K/V across generation steps (currently recomputed every forward).

## Notes / non-goals

- Inference only — no training, no autograd.
- float32, single precision throughout.
- Weights are randomly initialized, so `generate` output is meaningless; it
  exists to exercise the full pipeline, not to produce real text.
