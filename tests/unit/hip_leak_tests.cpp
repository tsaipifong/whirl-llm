// WHIRL unit tests of module hip.leak (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-hip-leak-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
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

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_hip_leak() {
    static const whirl::test::Suite s{"hip.leak",
                                      {
                                          {"testLeakCount", [] { testLeakCount(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_hip_leak()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
