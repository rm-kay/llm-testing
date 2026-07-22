#pragma once

// A ~50-line zero-dependency test framework. Tests self-register at static
// init time via the TEST() macro; test_main.cpp runs them all and reports.
//
//   TEST(my_case) {
//       ASSERT_TRUE(1 + 1 == 2);
//       ASSERT_NEAR(0.1 + 0.2, 0.3, 1e-9);
//   }

#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace testing {

struct AssertionError {
    std::string msg;
};

struct TestCase {
    std::string           name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

}  // namespace testing

#define TEST(name)                                                       \
    static void name();                                                  \
    static ::testing::Registrar test_registrar_##name(#name, name);      \
    static void name()

#define ASSERT_TRUE(cond)                                                \
    do {                                                                 \
        if (!(cond)) {                                                   \
            throw ::testing::AssertionError{                             \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                "  ASSERT_TRUE(" #cond ")"};                             \
        }                                                                \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                           \
    do {                                                                 \
        const double _a = (a), _b = (b), _t = (tol);                     \
        if (std::fabs(_a - _b) > _t) {                                   \
            throw ::testing::AssertionError{                             \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                "  ASSERT_NEAR(" #a ", " #b "): " +                      \
                std::to_string(_a) + " vs " + std::to_string(_b)};       \
        }                                                                \
    } while (0)
