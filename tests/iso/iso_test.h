// Shared harness of the per-module isolation test programs (ARCH-1 8.5, arch025-isoall). Each program is built by
// cmake/iso.cmake (and by hand in tools/gates/iso_check.py) from a copy of the files its tests/iso/<module>/isolation.txt
// lists, with that copy as the ONLY include path and the module objects linked directly (nothing else), so a test here
// sees exactly the module and its declared dependencies. Header-only, no WHIRL include.
//   <program> [--only NAME[,NAME...]] [--list]      last line "<module> tests: N passed, M failed"
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdio>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace iso {

inline int g_pass = 0, g_fail = 0;

struct Test {
    const char* name;
    std::function<void()> fn;
};

inline int run(const char* module, const std::vector<Test>& tests, int argc, char** argv) {
    std::set<std::string> only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--list") {
            for (const Test& t : tests) std::printf("%s\n", t.name);
            return 0;
        }
        if (a == "--only" && i + 1 < argc) {
            const std::string s = argv[++i];
            std::size_t p = 0;
            while (p <= s.size()) {
                std::size_t e = s.find(',', p);
                if (e == std::string::npos) e = s.size();
                only.insert(s.substr(p, e - p));
                p = e + 1;
            }
        } else {
            std::fprintf(stderr, "usage: %s [--only NAME[,NAME...]] [--list]\n", argv[0]);
            return 2;
        }
    }
    for (const std::string& n : only)
        if (std::none_of(tests.begin(), tests.end(), [&](const Test& t) { return n == t.name; })) {
            std::fprintf(stderr, "unknown test '%s' (--list)\n", n.c_str());
            return 2;
        }
    for (const Test& t : tests) {
        if (!only.empty() && !only.count(t.name)) continue;
        const int f0 = g_fail;
        try {
            t.fn();
        } catch (const std::exception& ex) {
            ++g_fail;
            std::fprintf(stderr, "  FAIL %s: exception %s\n", t.name, ex.what());
        }
        std::printf("%-24s %s\n", t.name, g_fail == f0 ? "ok" : "FAIL");
    }
    std::printf("%s tests: %d passed, %d failed\n", module, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

}  // namespace iso

#define ISO_CHECK(cond)                                                            \
    do {                                                                           \
        if (cond) {                                                                \
            ++iso::g_pass;                                                         \
        } else {                                                                   \
            ++iso::g_fail;                                                         \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                          \
    } while (0)

#define ISO_THROWS(expr, Ex)          \
    do {                              \
        bool thrown_ = false;         \
        try {                         \
            (void)(expr);             \
        } catch (const Ex&) {         \
            thrown_ = true;           \
        }                             \
        ISO_CHECK(thrown_);           \
    } while (0)
