// WHIRL unit tests of module hip.leak (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-hip-leak-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "platform/env.h"
#include "whirl/leak_count.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>  // K32GetProcessMemoryInfo (arch025-surv testPlatformProcSurvivors)
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
// arch025-Q2: the GPU-side leak count. A unit that frees what it made passes; a missing device free, a missing pinned
// free, a stream or an event left alive each fail and are named; what was alive before the unit is not blamed on it;
// a free of a block made before counting started is ignored; off counts nothing and never reports.
void testLeakCount() {
    namespace lc = whirl::hip::leakcount;
    const bool was_on = lc::g_on;
    lc::testEnable(true);
    lc::onDevAlloc(0x5000, 4096);  // alive before the units (e.g. the kernel-test context)
    const lc::Snapshot base = lc::snapshot();
    CHECK_EQ(base.dev, 1);
    CHECK_EQ(base.dev_bytes, 4096);
    // clean unit
    lc::onDevAlloc(0x10000, 1u << 20);
    lc::onHostAlloc(0x20000, 8192);
    lc::onStream(+1);
    lc::onEvent(+1);
    lc::onEvent(-1);
    lc::onStream(-1);
    lc::onHostFree(0x20000);
    lc::onDevFree(0x10000);
    CHECK(!lc::check("clean", base));
    // injected missing device free
    lc::onDevAlloc(0x30000, 768);
    lc::onDevAlloc(0x40000, 256);
    lc::onDevFree(0x40000);
    const lc::Snapshot b1 = lc::snapshot();
    CHECK(lc::leaked(base, b1) == "1 device allocation (768 bytes)");
    CHECK(lc::check("missing_free", base));
    lc::onDevFree(0x30000);
    // pinned host, stream, event
    const lc::Snapshot b2 = lc::snapshot();
    lc::onHostAlloc(0x50000, 64);
    lc::onStream(+1);
    lc::onEvent(+1);
    lc::onEvent(+1);
    CHECK(lc::leaked(b2, lc::snapshot()) == "1 pinned host allocation (64 bytes), 1 stream, 2 events");
    CHECK(lc::check("handles", b2));
    lc::onHostFree(0x50000);
    lc::onStream(-1);
    lc::onEvent(-2);
    // a free of an unknown block changes nothing
    lc::onDevFree(0x777);
    lc::onHostFree(0x888);
    CHECK(lc::leaked(base, lc::snapshot()).empty());
    CHECK(!lc::check("after", base));
    CHECK_EQ(lc::report(), 2);  // missing_free + handles of 5 units
    // off: nothing counted, check never fires, report silent
    lc::testEnable(false);
    const lc::Snapshot z = lc::snapshot();
    lc::onDevFree(0x5000);
    CHECK(!lc::check("off", z));
    CHECK_EQ(lc::report(), 0);
    lc::testEnable(was_on);
}

// arch025-env61: g_on's WHIRL_LEAK_CHECK rule (leakcount::switchFrom, leak_count.cpp passes platform::env::get).
// One read, of exactly "WHIRL_LEAK_CHECK"; on = set, not empty and not "0", as the v0.2.5 std::getenv rule
// (v != nullptr && *v != 0 && std::strcmp(v, "0") != 0), which each row also computes as the oracle.
// the process's WHIRL_LEAK_CHECK at static initialization (before the harness clears WHIRL_*), by the old rule
const bool k_leak_env_at_start = [] {
    const char* g = std::getenv("WHIRL_LEAK_CHECK");
    return g != nullptr && *g != 0 && std::strcmp(g, "0") != 0;
}();

void testLeakSwitch() {
    namespace lc = whirl::hip::leakcount;
    // g_on was set once, at static initialization, by the rule (testLeakCount restored it)
    CHECK_EQ(lc::g_on, k_leak_env_at_start);
    struct Row {
        std::optional<std::string> v;
        bool on;
    };
    const Row rows[] = {{std::nullopt, false}, {std::string(), false}, {std::string("0"), false}, {std::string("1"), true},
                        {std::string("00"), true}, {std::string(" 0"), true}, {std::string("0 "), true},
                        {std::string("false"), true}, {std::string("yes"), true}};
    for (const Row& r : rows) {
        int calls = 0;
        std::string asked;
        const bool on = lc::switchFrom([&](const char* name) {
            ++calls;
            asked = name;
            return r.v;
        });
        const bool old_rule = r.v.has_value() && !r.v->empty() && std::strcmp(r.v->c_str(), "0") != 0;
        CHECK_EQ(on, r.on);
        CHECK_EQ(on, old_rule);
        CHECK_EQ(calls, 1);
        CHECK(asked == "WHIRL_LEAK_CHECK");
    }
#ifdef _WIN32
    // the real reader gives the bytes std::getenv gave (the same CRT narrow table): unset, "1", "0", "yes", restored
    const char* had = std::getenv("WHIRL_LEAK_CHECK");
    const std::optional<std::string> saved = had ? std::optional<std::string>(had) : std::nullopt;
    for (const char* v : {"", "1", "0", "yes"}) {
        _putenv_s("WHIRL_LEAK_CHECK", v);  // "" removes the variable
        const char* g = std::getenv("WHIRL_LEAK_CHECK");
        const bool old_rule = g != nullptr && *g != 0 && std::strcmp(g, "0") != 0;
        CHECK_EQ(lc::switchFrom([](const char* name) { return whirl::platform::env::get(name); }), old_rule);
    }
    _putenv_s("WHIRL_LEAK_CHECK", saved ? saved->c_str() : "");
#endif
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_hip_leak() {
    static const whirl::test::Suite s{"hip.leak",
                                      {
                                          {"testLeakCount", [] { testLeakCount(); }},
                                          {"testLeakSwitch", [] { testLeakSwitch(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_hip_leak()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
