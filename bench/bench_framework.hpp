#pragma once

// A tiny benchmarking helper: run a callable repeatedly, discard warmup
// iterations, and report min / median / mean wall-clock time. `min` is usually
// the number to compare across your optimizations (least noise). Optionally
// pass a FLOP count to also print GFLOP/s.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace bench {

struct Result {
    std::string name;
    double      min_ms;
    double      median_ms;
    double      mean_ms;
    double      gflops;  // 0 if not provided
};

// Run `fn` `iters` times (after `warmup` untimed runs) and collect timings.
// `flops_per_iter` is optional; pass 0 to skip the GFLOP/s column.
template <typename Fn>
Result measure(const std::string& name, Fn&& fn, int iters = 30, int warmup = 3,
               double flops_per_iter = 0.0) {
    using clock = std::chrono::steady_clock;

    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = clock::now();
        fn();
        const auto t1 = clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(samples.begin(), samples.end());
    double sum = 0;
    for (double s : samples) sum += s;

    Result r;
    r.name      = name;
    r.min_ms    = samples.front();
    r.median_ms = samples[samples.size() / 2];
    r.mean_ms   = sum / samples.size();
    r.gflops    = flops_per_iter > 0.0 ? (flops_per_iter / (r.min_ms * 1e-3)) / 1e9 : 0.0;
    return r;
}

inline void print_header() {
    std::printf("%-28s %12s %12s %12s %12s\n",
                "benchmark", "min (ms)", "median (ms)", "mean (ms)", "GFLOP/s");
    std::printf("%-28s %12s %12s %12s %12s\n",
                "----------------------------", "--------", "-----------",
                "---------", "-------");
}

inline void print(const Result& r) {
    if (r.gflops > 0.0) {
        std::printf("%-28s %12.4f %12.4f %12.4f %12.2f\n",
                    r.name.c_str(), r.min_ms, r.median_ms, r.mean_ms, r.gflops);
    } else {
        std::printf("%-28s %12.4f %12.4f %12.4f %12s\n",
                    r.name.c_str(), r.min_ms, r.median_ms, r.mean_ms, "-");
    }
}

}  // namespace bench
