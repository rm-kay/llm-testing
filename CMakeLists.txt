cmake_minimum_required(VERSION 3.16)
project(llm_cpp LANGUAGES CXX)

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

# Benchmarks.
add_executable(benchmarks bench/bench_main.cpp)
target_include_directories(benchmarks PRIVATE bench)
target_link_libraries(benchmarks PRIVATE llm_core)
if(LLM_NATIVE)
    target_compile_options(benchmarks PRIVATE -march=native)
endif()

# End-to-end example.
add_executable(generate examples/generate.cpp)
target_link_libraries(generate PRIVATE llm_core)

enable_testing()
add_test(NAME unit_tests COMMAND unit_tests)
