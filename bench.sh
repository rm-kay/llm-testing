taskset -c 0 ./build/benchmarks --benchmark_out=new.json --benchmark_out_format=json && build/_deps/benchmark-src/tools/compare.py --dump_to_json compared.json benchmarks baseline.json new.json
