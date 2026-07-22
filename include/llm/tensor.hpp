#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <vector>

namespace llm {

// A minimal, intentionally-simple row-major dense float32 tensor.
//
// The whole point of this repo is readability, so this is deliberately naive:
//   * it always owns its data in a single flat std::vector<float>,
//   * every op allocates fresh output tensors (no buffer reuse / arenas),
//   * there is no broadcasting, no strides, no views.
//
// Those are exactly the kinds of things you might optimize by hand later.
struct Tensor {
    std::vector<int>   shape;
    std::vector<float> data;

    Tensor() = default;

    explicit Tensor(std::vector<int> shape_)
        : shape(std::move(shape_)), data(static_cast<size_t>(numel(shape)), 0.0f) {}

    Tensor(std::vector<int> shape_, std::vector<float> data_)
        : shape(std::move(shape_)), data(std::move(data_)) {
        assert(static_cast<int>(data.size()) == numel(shape));
    }

    // Total number of elements.
    int size() const { return numel(shape); }

    // Convenience accessors for the common 2D [rows, cols] case.
    int rows() const { assert(shape.size() == 2); return shape[0]; }
    int cols() const { assert(shape.size() == 2); return shape[1]; }

    // 1D element access.
    float& operator[](int i)             { return data[static_cast<size_t>(i)]; }
    float  operator[](int i) const       { return data[static_cast<size_t>(i)]; }

    // 2D element access (row-major): element (r, c) lives at r * cols + c.
    float& at(int r, int c) {
        assert(shape.size() == 2 && r >= 0 && r < shape[0] && c >= 0 && c < shape[1]);
        return data[static_cast<size_t>(r) * shape[1] + c];
    }
    float at(int r, int c) const {
        assert(shape.size() == 2 && r >= 0 && r < shape[0] && c >= 0 && c < shape[1]);
        return data[static_cast<size_t>(r) * shape[1] + c];
    }

    // Pointer to the start of row r (2D only).
    float*       row_ptr(int r)       { return data.data() + static_cast<size_t>(r) * shape[1]; }
    const float* row_ptr(int r) const { return data.data() + static_cast<size_t>(r) * shape[1]; }

    static int numel(const std::vector<int>& s) {
        return std::accumulate(s.begin(), s.end(), 1, [](int a, int b) { return a * b; });
    }
};

}  // namespace llm
