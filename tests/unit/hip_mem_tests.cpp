// WHIRL unit tests of module hip.mem (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-hip-mem-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/device_mem.h"
#include "whirl/uma_mem.h"

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
// VIS-8S-2: device-local memory taken, per device class (numbers from the 8060S / R9700 probes:
// pinned 887 MiB moves hipMemGetInfo by 2 x 887 on the integrated GPU, 0 on the discrete one).
void testLocalMemTaken() {
    using whirl::hip::LocalMemSnap;
    using whirl::hip::MemClass;
    constexpr std::uint64_t M = 1ull << 20;
    auto snap = [&](MemClass c, std::uint64_t fr, bool pdh, std::uint64_t ded, std::uint64_t sh) {
        LocalMemSnap s;
        s.cls = c;
        s.hip_free = fr * M;
        s.pdh_ok = pdh;
        s.dedicated = ded * M;
        s.shared = sh * M;
        return s;
    };
    // integrated: pinned mmproj copy (system RAM) -> 0 taken although hipMemGetInfo dropped 1774 MiB
    auto t = whirl::hip::localMemTaken(snap(MemClass::integrated, 50000, true, 166, 10), snap(MemClass::integrated, 48226, true, 166, 897));
    CHECK_EQ(t.mib, 0.0);
    CHECK_EQ(t.hip_free_mib, 1774.0);
    CHECK_EQ(t.shared_mib, 887.0);
    // integrated: a 16 MiB device allocation (carve-out) is caught
    t = whirl::hip::localMemTaken(snap(MemClass::integrated, 50000, true, 166, 10), snap(MemClass::integrated, 47210, true, 182, 897));
    CHECK_EQ(t.mib, 16.0);
    // integrated, counters unreadable: falls back to hipMemGetInfo (over-reports: fails, never passes wrongly)
    t = whirl::hip::localMemTaken(snap(MemClass::integrated, 50000, false, 0, 0), snap(MemClass::integrated, 48226, false, 0, 0));
    CHECK_EQ(t.mib, 1774.0);
    t = whirl::hip::localMemTaken(snap(MemClass::integrated, 50000, true, 166, 0), snap(MemClass::integrated, 48226, false, 0, 0));
    CHECK_EQ(t.mib, 1774.0);
    // discrete: hipMemGetInfo, whatever the counters say
    t = whirl::hip::localMemTaken(snap(MemClass::discrete, 9000, true, 185, 7), snap(MemClass::discrete, 9000, true, 185, 894));
    CHECK_EQ(t.mib, 0.0);
    CHECK_EQ(t.shared_mib, 887.0);
    t = whirl::hip::localMemTaken(snap(MemClass::discrete, 9000, true, 185, 7), snap(MemClass::discrete, 8984, true, 185, 7));
    CHECK_EQ(t.mib, 16.0);
    CHECK(std::string(t.how).find("discrete") != std::string::npos);
    // given back: negative
    t = whirl::hip::localMemTaken(snap(MemClass::discrete, 8984, false, 0, 0), snap(MemClass::discrete, 9000, false, 0, 0));
    CHECK_EQ(t.mib, -16.0);
}

// UMA-1: the integrated-GPU carve-out check (pure parts of whirl/uma_mem.h)
void testUmaCheck() {
    using whirl::hip::UmaMem;
    using whirl::hip::UmaNeed;
    constexpr std::uint64_t GiB = 1ull << 30;
    UmaMem u;
    u.integrated = true;
    u.known = true;
    u.dedicated = 64 * GiB;
    u.shared = 48 * GiB;
    CHECK_EQ(whirl::hip::umaSizeText(u), std::string("dedicated 64.0 GiB (+ shared 48.0 GiB)"));
    UmaNeed n;
    n.weights = 16 * GiB;
    n.fixed = 20 * GiB;
    n.kv_per_token = 64 * 1024;
    n.min_tokens = 131072;   // 8 GiB of KV
    n.want_tokens = 262144;  // 16 GiB of KV
    // 64 GiB: fits the full pool, no refusal, no cap
    CHECK(whirl::hip::umaRefusal(u, n, "AMD Radeon(TM) 8060S Graphics").empty());
    CHECK(!whirl::hip::umaPoolCap(u, n, 256).has_value());
    // 32 GiB: one request fits (20 + 8 + 1.5), the full pool does not -> pool capped to the room
    u.dedicated = 32 * GiB;
    CHECK(whirl::hip::umaRefusal(u, n, "x").empty());
    const auto cap = whirl::hip::umaPoolCap(u, n, 256);
    CHECK(cap.has_value());
    if (cap) {
        CHECK_EQ(*cap % 256, 0u);
        CHECK(*cap >= n.min_tokens && *cap < n.want_tokens);
        CHECK(n.fixed + static_cast<std::uint64_t>(*cap) * n.kv_per_token <= whirl::hip::umaUsable(u.dedicated));
    }
    // 0.5 GiB (issue #7): refused, the message names the setting and the size to set
    u.dedicated = GiB / 2;
    const std::string why = whirl::hip::umaRefusal(u, n, "AMD Radeon(TM) 8060S Graphics");
    CHECK(!why.empty());
    CHECK(why.find("0.5 GB of dedicated GPU memory") != std::string::npos);
    CHECK(why.find("Armoury Crate") != std::string::npos);
    CHECK(why.find("BIOS") != std::string::npos);
    CHECK(why.find("at least 30 GB") != std::string::npos);   // 20 + 8 + 1.5 = 29.5 -> 30
    CHECK(why.find("38 GB for the full 262144-token") != std::string::npos);  // 20 + 16 + 1.5 = 37.5 -> 38
    CHECK(why.find("WHIRL_UMA_CHECK=0") != std::string::npos);
    CHECK(why.find("simulated") == std::string::npos);
    u.fake = true;
    CHECK(whirl::hip::umaRefusal(u, n, "x").find("simulated: WHIRL_FAKE_DEDICATED_GB") != std::string::npos);
    // a discrete GPU or an unknown carve-out: never refused, never capped, no 719 hint
    UmaMem d;
    d.dedicated = 0;
    CHECK(whirl::hip::umaRefusal(d, n, "x").empty());
    CHECK(!whirl::hip::umaPoolCap(d, n, 256).has_value());
    CHECK(whirl::hip::uma719Hint(&d).empty());
    CHECK(whirl::hip::uma719Hint(nullptr).empty());
    UmaMem unk;
    unk.integrated = true;
    CHECK(whirl::hip::umaRefusal(unk, n, "x").empty());
    // hipError 719 on the iGPU: the hint names the carve-out size and the setting
    const std::string h = whirl::hip::uma719Hint(&u);
    CHECK(h.find("0.5 GB") != std::string::npos);
    CHECK(h.find("Armoury Crate") != std::string::npos);
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_hip_mem() {
    static const whirl::test::Suite s{"hip.mem",
                                      {
                                          {"testLocalMemTaken", [] { testLocalMemTaken(); }},
                                          {"testUmaCheck", [] { testUmaCheck(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_hip_mem()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
