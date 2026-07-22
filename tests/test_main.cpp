#include <cstdio>

#include "test_framework.hpp"

// Runs every registered test case and prints a summary. Returns non-zero if
// any test failed, so it doubles as a CTest / CI gate.
int main() {
    int passed = 0;
    int failed = 0;

    for (const auto& tc : testing::registry()) {
        try {
            tc.fn();
            std::printf("[ PASS ] %s\n", tc.name.c_str());
            ++passed;
        } catch (const testing::AssertionError& e) {
            std::printf("[ FAIL ] %s\n         %s\n", tc.name.c_str(), e.msg.c_str());
            ++failed;
        } catch (const std::exception& e) {
            std::printf("[ FAIL ] %s\n         unexpected exception: %s\n",
                        tc.name.c_str(), e.what());
            ++failed;
        }
    }

    std::printf("\n%d passed, %d failed, %d total\n",
                passed, failed, passed + failed);
    return failed == 0 ? 0 : 1;
}
