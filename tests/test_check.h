// The one test-assert header of the WHIRL host test programs (arch025 N16; replaces the per-file CHECK copies).
// Header-only, no WHIRL include, so an isolated test tree (cmake/iso.cmake) can copy it next to its tests.
//
//   CHECK(cond)          counts a pass, or a fail with "FAIL <file>:<line>: <cond>" on stdout (WHIRL_TEST_FAIL_STDERR:
//                        "  FAIL ..." on stderr)
//   CHECK_EQ(a, b)       the same for a == b (each side evaluated once); CHECK_DETAIL(cond, text) adds a detail text
//   whirl::test::check(ok, what)   the message form: "FAIL: <what>"
//   whirl::test::g_pass / g_fail   the counters (a program prints its own summary line from them)
//   whirl::test::throws(f)   true when f() throws
//   whirl::test::Suite   one module's tests: {module, {{"testName", fn}...}, child}; child (optional) handles the
//                        program's helper modes (the test runs this program as a child) and returns -1 otherwise
//   whirl::test::runSuites(suites, argc, argv, prepare)   [--list] [--only NAME[,NAME...]]; per suite a line
//                        "<module> tests: N passed, M failed"; returns 0 / 1 / 2 (usage, unknown name)
// Pattern: refsrc llama.cpp tests/testing.h (one shared assert + filter header) + llama_build_and_test (one program
// per test file); see progress\arch025-lane3.PRIOR_ART.md section 5.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdio>
#include <exception>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace whirl::test {

inline int g_pass = 0, g_fail = 0;

template <class F>
bool throws(F f) {
    try {
        f();
    } catch (...) {
        return true;
    }
    return false;
}

struct Case {
    const char* name;
    std::function<void()> fn;
};

struct Suite {
    const char* module;
    std::vector<Case> cases;
    int (*child)(int argc, char** argv) = nullptr;  // helper modes of this suite's tests; -1 = not a helper call
};

// --list / --only parsing shared by the per-module programs and whirl-tests. Returns false on a usage error.
inline bool parseArgs(int argc, char** argv, bool* list, std::set<std::string>* only) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--list") {
            *list = true;
        } else if (a == "--only" && i + 1 < argc) {
            const std::string s = argv[++i];
            std::size_t p = 0;
            while (p <= s.size()) {
                std::size_t e = s.find(',', p);
                if (e == std::string::npos) e = s.size();
                if (e > p) only->insert(s.substr(p, e - p));
                p = e + 1;
            }
        } else {
            std::fprintf(stderr, "usage: %s [--list] [--only NAME[,NAME...]]\n", argv[0]);
            return false;
        }
    }
    return true;
}

// Runs one case, counting an escaped exception as a failure (the old single main had no guard: an exception ended
// the program, which also counted as a failed run).
inline void runCase(const Case& c) {
    try {
        c.fn();
    } catch (const std::exception& ex) {
        ++g_fail;
        std::printf("FAIL %s: exception %s\n", c.name, ex.what());
    } catch (...) {
        ++g_fail;
        std::printf("FAIL %s: unknown exception\n", c.name);
    }
}

// The suites' helper modes: the exit code of the one that handles argv, else -1.
inline int runChildModes(const std::vector<const Suite*>& suites, int argc, char** argv) {
    for (const Suite* s : suites)
        if (s->child) {
            const int rc = s->child(argc, argv);
            if (rc >= 0) return rc;
        }
    return -1;
}

// The per-module programs: every case of the given suites in their order. prepare (e.g. clearing WHIRL_* variables)
// runs after the helper modes, before the first case, as in the old single main().
inline int runSuites(const std::vector<const Suite*>& suites, int argc, char** argv, void (*prepare)() = nullptr) {
    const int child_rc = runChildModes(suites, argc, argv);
    if (child_rc >= 0) return child_rc;
    bool list = false;
    std::set<std::string> only;
    if (!parseArgs(argc, argv, &list, &only)) return 2;
    for (const std::string& n : only) {
        bool found = false;
        for (const Suite* s : suites)
            for (const Case& c : s->cases) found = found || n == c.name;
        if (!found) {
            std::fprintf(stderr, "unknown test '%s' (--list)\n", n.c_str());
            return 2;
        }
    }
    if (list) {
        for (const Suite* s : suites)
            for (const Case& c : s->cases) std::printf("%s %s\n", s->module, c.name);
        return 0;
    }
    if (prepare) prepare();
    for (const Suite* s : suites) {
        const int p0 = g_pass, f0 = g_fail;
        for (const Case& c : s->cases)
            if (only.empty() || only.count(c.name)) runCase(c);
        std::printf("%s tests: %d passed, %d failed\n", s->module, g_pass - p0, g_fail - f0);
    }
    return g_fail ? 1 : 0;
}

}  // namespace whirl::test

// Where a failed check is printed. Default: stdout "FAIL <file>:<line>: <what>" (unit, cache, tier, vision, device, ops,
// hiprt). A program that printed its failures on stderr as "  FAIL ..." (sampling, server) defines
// WHIRL_TEST_FAIL_STDERR before including this header. Each program is one choice (both functions always exist).
namespace whirl::test {
inline void failOut(const char* file, int line, const char* what, const char* detail = "") {
    std::printf("FAIL %s:%d: %s%s%s\n", file, line, what, *detail ? " " : "", detail);
}
inline void failErr(const char* file, int line, const char* what, const char* detail = "") {
    std::fprintf(stderr, "  FAIL %s:%d: %s%s%s\n", file, line, what, *detail ? " " : "", detail);
}
// counts one check (the message form of model / device / ops / hiprt: check(ok, what) prints "FAIL: <what>")
inline bool count(bool ok) {
    ++(ok ? g_pass : g_fail);
    return ok;
}
inline void check(bool ok, const char* what) {
    if (!count(ok)) std::printf("FAIL: %s\n", what);
}
inline void check(bool ok, const std::string& what) { check(ok, what.c_str()); }
}  // namespace whirl::test

#ifdef WHIRL_TEST_FAIL_STDERR
#define WHIRL_TEST_FAIL ::whirl::test::failErr
#else
#define WHIRL_TEST_FAIL ::whirl::test::failOut
#endif

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (cond) {                                                              \
            ++::whirl::test::g_pass;                                             \
        } else {                                                                 \
            ++::whirl::test::g_fail;                                             \
            WHIRL_TEST_FAIL(__FILE__, __LINE__, #cond);                          \
        }                                                                        \
    } while (0)

// CHECK with a detail string printed after the condition (hiprt's CHECKD)
#define CHECK_DETAIL(cond, detail)                                               \
    do {                                                                         \
        if (cond) {                                                              \
            ++::whirl::test::g_pass;                                             \
        } else {                                                                 \
            ++::whirl::test::g_fail;                                             \
            WHIRL_TEST_FAIL(__FILE__, __LINE__, #cond, std::string(detail).c_str()); \
        }                                                                        \
    } while (0)

// both sides are copied (not bound by reference): a side such as parse(d).find(k)->asString() refers into a temporary
// that dies at the end of its declaration (the old unit_tests.cpp copy; a reference here read freed memory)
#define CHECK_EQ(a, b)                                                           \
    do {                                                                         \
        const auto va_ = (a);                                                    \
        const auto vb_ = (b);                                                    \
        if (va_ == vb_) {                                                        \
            ++::whirl::test::g_pass;                                             \
        } else {                                                                 \
            ++::whirl::test::g_fail;                                             \
            WHIRL_TEST_FAIL(__FILE__, __LINE__, #a " == " #b);                   \
        }                                                                        \
    } while (0)
