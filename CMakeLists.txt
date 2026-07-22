cmake_minimum_required(VERSION 3.16)
project(llm_cpp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimized build so benchmark numbers are meaningful. Override
# with -DCMAKE_BUILD_TYPE=Debug (e.g. to run tests under sanitizers/asserts).
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optional: tune for the host CPU. Off by default for portable/reproducible
# numbers; turn on with -DLLM_NATIVE=ON once you start optimizing.
option(LLM_NATIVE "Build with -march=native" OFF)

# Core library: the model and its numerical kernels.
add_library(llm_core
    src/ops.cpp
    src/model.cpp
)
target_include_directories(llm_core PUBLIC include)
target_compile_options(llm_core PRIVATE -Wall -Wextra)
if(LLM_NATIVE)
    target_compile_options(llm_core PRIVATE -march=native)
endif()

# Unit tests.
add_executable(unit_tests
    tests/test_main.cpp
    tests/test_ops.cpp
    tests/test_model.cpp
)
target_include_directories(unit_tests PRIVATE tests)
target_link_libraries(unit_tests PRIVATE llm_core)

# Benchmarks (Google Benchmark, fetched at configure time). Turn OFF to build
# without network access: -DLLM_BUILD_BENCHMARKS=OFF.
option(LLM_BUILD_BENCHMARKS "Build benchmarks (requires fetching Google Benchmark)" ON)
if(LLM_BUILD_BENCHMARKS)
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL     OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR      OFF CACHE BOOL "" FORCE)  # tolerate new-compiler warnings
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(benchmark)

    add_executable(benchmarks bench/bench_main.cpp)
    target_link_libraries(benchmarks PRIVATE llm_core benchmark::benchmark)
    if(LLM_NATIVE)
        target_compile_options(benchmarks PRIVATE -march=native)
    endif()
endif()

# End-to-end example.
add_executable(generate examples/generate.cpp)
target_link_libraries(generate PRIVATE llm_core)

enable_testing()
add_test(NAME unit_tests COMMAND unit_tests)
