// WHIRL unit tests of module mem.wddm_watch (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-mem-wddm_watch-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/wddm_watch.h"

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
// ---------------------------------------------------------------------------
// WDDM-LOG: runtime video memory watch (formatting, spill thresholds, rate limit, recovery, decode drop)
void testWddmWatch() {
    using namespace whirl::wddmwatch;
    constexpr std::uint64_t MiB = 1ull << 20;
    auto smp = [&](std::uint64_t used_mib, std::uint64_t budget_mib, std::uint64_t shared_mib, std::uint64_t pinned_mib = 0) {
        Sample s;
        s.ok = true;
        s.local_usage = used_mib * MiB;
        s.local_budget = budget_mib * MiB;
        s.nonlocal_usage = shared_mib * MiB;
        s.pinned = pinned_mib * MiB;
        return s;
    };
    auto has = [](const std::optional<Event>& e, const char* t) { return e && e->text.find(t) != std::string::npos; };
    const Sample base = smp(29000, 31232, 400);
    // formatting
    CHECK_EQ(format(smp(29798, 31232, 412), base), std::string("vram local 29.10 / budget 30.50 GiB, shared 412 MiB (+12 MiB since start)"));
    CHECK_EQ(format(base, base), std::string("vram local 28.32 / budget 30.50 GiB, shared 400 MiB (+0 MiB since start)"));
    CHECK_EQ(format(smp(29000, 31232, 397), base), std::string("vram local 28.32 / budget 30.50 GiB, shared 397 MiB (-3 MiB since start)"));
    CHECK_EQ(format(smp(29000, 31232, 400), Sample{}), std::string("vram local 28.32 / budget 30.50 GiB, shared 400 MiB"));
    CHECK_EQ(format(Sample{}, base), std::string());
    {
        Sample r = smp(29000, 31232, 400, 9318);
        r.rss = 18636ull * MiB;
        CHECK_EQ(format(r, base), std::string("vram local 28.32 / budget 30.50 GiB, shared 400 MiB (+0 MiB since start), ram 18.20 GiB (pinned 9.10 GiB)"));
        r.rss = 0;
        CHECK_EQ(format(r, Sample{}), std::string("vram local 28.32 / budget 30.50 GiB, shared 400 MiB, pinned ram 9.10 GiB"));
    }

    // no WDDM numbers: never warns
    {
        Watch w(Sample{});
        CHECK(!w.check(smp(40000, 31232, 9000), 0.0));
        Watch w2(base);
        CHECK(!w2.check(Sample{}, 0.0));
    }
    // under the budget, growth up to 256 MiB: quiet
    {
        Watch w(base);
        CHECK(!w.check(base, 0.0));
        CHECK(!w.check(smp(31232, 31232, 400 + 256), 10.0));  // at the budget, exactly +256: no spill
        CHECK(!w.spilling());
        CHECK_EQ(w.warnings(), 0u);
    }
    // pinned host memory WHIRL allocated later (vision encoder, checkpoints) is not a spill
    {
        Watch w(smp(29000, 31232, 400, 2000));
        CHECK(!w.check(smp(29000, 31232, 400 + 1000, 3000), 0.0));
        CHECK(w.check(smp(29000, 31232, 400 + 1300, 3000), 0.0).has_value());  // 300 MiB beyond the new pinned memory
    }
    // shared growth: warning at the start (all four parts), repeated at most once a minute while it lasts,
    // INFO when it is over
    {
        Watch w(base);
        const auto a = w.check(smp(29000, 31232, 400 + 300), 100.0);
        CHECK(a && a->warn);
        CHECK(a && a->text.rfind("GPU memory spill: WHIRL's shared GPU memory grew by 300 MiB since startup: Windows has likely "
                                 "moved part of WHIRL's GPU buffers to system RAM", 0) == 0);
        CHECK(has(a, "Generation slows down if those buffers are used at every step: decode can become several times slower."));
        CHECK(has(a, "What you can do: close other programs that use the GPU (browsers, games, other AI tools); or restart the "
                     "server, which sizes the KV pool from the budget at startup, and set WHIRL_POOL_BUDGET_MARGIN_MB=4096 (or "
                     "more) to keep more room free for other programs."));
        // review P2-1 / P3-1: never the variable that turns the budget rule off, never --parallel 1
        CHECK(!has(a, "WHIRL_POOL_RESERVE_MB"));
        CHECK(!has(a, "--parallel"));
        CHECK(has(a, "Now: vram local 28.32 / budget 30.50 GiB, shared 700 MiB (+300 MiB since start)"));
        CHECK(!has(a, "over the Windows"));
        CHECK(w.spilling());
        CHECK(!w.check(smp(29000, 31232, 400 + 300), 130.0));  // within the minute
        CHECK(!w.check(smp(29000, 31232, 400 + 900), 150.0));  // grew, still within the minute
        const auto b = w.check(smp(29000, 31232, 400 + 900), 160.0);
        CHECK(b && b->warn);
        CHECK(b && b->text.rfind("GPU memory spill continues (60 s so far): WHIRL's shared GPU memory grew by 900 MiB", 0) == 0);
        const auto c = w.check(smp(29000, 31232, 400 + 900), 400.0);
        CHECK(c && c->text.rfind("GPU memory spill continues (5 min so far): ", 0) == 0);
        const auto d = w.check(smp(29000, 31232, 400 + 100), 410.0);
        CHECK(d && !d->warn);
        CHECK(d && d->text.rfind("GPU memory spill is over (it lasted 5 min): WHIRL's GPU memory is back under the Windows video "
                                 "memory budget and its shared GPU memory is back within 256 MiB of startup, so decode is no "
                                 "longer slowed by it. Now: vram local 28.32",
                                 0) == 0);
        CHECK(!w.spilling());
        CHECK(!w.check(smp(29000, 31232, 400 + 100), 420.0));  // one INFO only
        CHECK_EQ(w.warnings(), 3u);
    }
    // local over the budget: named with the amount; a new spill within the minute after the last warning
    // waits for the minute; one that ends before any warning gets no INFO line
    {
        Watch w(base);
        const auto a = w.check(smp(31300, 31232, 400), 0.0);
        CHECK(a && a->warn);
        CHECK(a && a->text.rfind("GPU memory spill: WHIRL's GPU memory is 0.07 GiB over the Windows video memory (WDDM) budget, "
                                 "so Windows can page part of WHIRL's GPU buffers out of VRAM (this can happen before the shared "
                                 "GPU memory number grows).",
                                 0) == 0);
        CHECK(has(a, "Generation is likely slower now: decode can become several times slower."));
        CHECK(has(w.check(smp(30000, 31232, 400), 10.0), "GPU memory spill is over (it lasted 10 s)"));
        CHECK(!w.check(smp(31300, 31232, 400), 20.0));  // new spill, 20 s after the last warning: held
        CHECK(!w.check(smp(30000, 31232, 400), 30.0));  // over before it was reported: no INFO
        CHECK(!w.check(smp(31300, 31232, 400), 40.0));
        const auto b = w.check(smp(31300, 31232, 400), 61.0);
        CHECK(b && b->text.rfind("GPU memory spill: ", 0) == 0);  // a new spill, not "continues"
        CHECK_EQ(w.warnings(), 2u);
    }
    // over the budget and grown: both in one sentence
    {
        Watch w(base);
        const auto a = w.check(smp(31300, 31232, 1000), 0.0);
        CHECK(has(a, "GPU memory spill: WHIRL's GPU memory is 0.07 GiB over the Windows video memory (WDDM) budget, and Windows "
                     "has likely moved part of WHIRL's GPU buffers to system RAM (shared GPU memory +600 MiB since startup)."));
        CHECK(has(a, "Generation is likely slower now"));
    }
    // the KV pool size in the advice (review P3-3), other programs' VRAM note
    {
        Config c;
        c.pool_tokens = 334080;
        Watch w(base, c);
        const auto a = w.check(smp(31300, 31232, 400), 0.0);
        CHECK(has(a, "or restart the server, which sizes the KV pool (the KV pool is now 334080 tokens) from the budget at startup"));
        CHECK_EQ(othersNote(Others{3328 * MiB, 205 * MiB}),
                 std::string(" Other programs use 3.25 GiB of VRAM on this GPU now (0.20 GiB when the server started)."));
        CHECK_EQ(othersNote(Others{1024 * MiB, std::nullopt}), std::string(" Other programs use 1.00 GiB of VRAM on this GPU now."));
    }
    // a spill already present at startup is seen against the sizing-time baseline (review P3-4): the server passes
    // the read the pool was sized from, not a sample taken after everything was allocated
    {
        const Sample sized = smp(17640, 31760, 13692, 13576);       // before the pool: embedding + RAM tier pinned
        Watch w(sized);
        CHECK(w.check(smp(30260, 31760, 13692 + 600, 13576), 0.0).has_value());  // 600 MiB demoted while starting
        Watch ok(sized);
        CHECK(!ok.check(smp(30260, 31760, 13692 + 300, 13576 + 300), 0.0));     // 300 MiB pinned checkpoints (WHIRL_CKPT_HOST=1)
    }
    // config defaults (the server uses them: 256 MiB, 60 s)
    {
        Config c;
        CHECK_EQ(c.grow_warn, 256 * MiB);
        CHECK(c.min_gap_s == 60.0);
    }

    // decode speed drop within a request
    {
        DecodeWatch d;
        CHECK(!d.report(100, 1000.0, 25, false));  // before the early window (200 tokens)
        CHECK(!d.report(200, 2000.0, 50, false));  // early: 100 tok/s, 40 ms per cycle
        CHECK(!d.report(300, 3000.0, 75, false));  // 1 s window: too short to judge, keeps growing
        CHECK(!d.report(400, 4500.0, 100, false));  // 2.5 s window at 80 tok/s: -20%, fine
        const auto a = d.report(500, 7000.0, 125, true);  // 100 tok in 2.5 s = 40 tok/s, 100 ms per cycle, spill seen
        CHECK(a && a->warn);
        CHECK(a && a->text == "decode slowed from 100.0 to 40.0 tok/s (60% slower than early in this request; 40.0 -> 100.0 ms "
                              "per decode cycle), likely because of the GPU memory spill above");
        CHECK(!d.report(600, 10000.0, 150, true));  // once per request
        d.reset();
        CHECK(!d.report(200, 2000.0, 50, false));
        const auto b = d.report(300, 6000.0, 150, false);  // 25 tok/s at the same 40 ms per cycle: fewer tokens per cycle
        CHECK(b && !b->warn);
        CHECK(b && b->text == "decode slowed from 100.0 to 25.0 tok/s (75% slower than early in this request; 40.0 -> 40.0 ms per "
                              "decode cycle); each cycle is not slower: fewer tokens per cycle (fewer drafts accepted)");
        DecodeWatch e;
        CHECK(!e.report(200, 2000.0, 0, false));
        const auto c = e.report(300, 5000.0, 0, false);  // no cycle numbers
        CHECK(c && c->text == "decode slowed from 100.0 to 33.3 tok/s (67% slower than early in this request); no GPU memory spill is seen");
        DecodeWatch f;
        CHECK(!f.report(200, 2000.0, 50, false));
        CHECK(!f.report(300, 3600.0, 75, false));  // 62.5 tok/s: -37.5%, under the 40% line
        // review P3-5: a spill while each cycle is NOT slower is not blamed (draft acceptance fell)
        DecodeWatch g;
        CHECK(!g.report(200, 2000.0, 50, true));
        const auto gs = g.report(300, 6000.0, 150, true);
        CHECK(gs && !gs->warn);
        CHECK(gs && gs->text == "decode slowed from 100.0 to 25.0 tok/s (75% slower than early in this request; 40.0 -> 40.0 ms per "
                                "decode cycle); each cycle is not slower: fewer tokens per cycle (fewer drafts accepted), not the "
                                "GPU memory spill");
        // another request prefilled in the window (--parallel >= 2), no spill: named as the likely cause
        DecodeWatch q;
        CHECK(!q.report(200, 2000.0, 50, false, 1000));
        const auto qs = q.report(300, 7000.0, 75, false, 9192);  // 100 tok in 5 s, 200 ms per cycle; 8192 prefill tokens since
        CHECK(qs && !qs->warn);
        CHECK(qs && qs->text == "decode slowed from 100.0 to 20.0 tok/s (80% slower than early in this request; 40.0 -> 200.0 ms per "
                                "decode cycle); no GPU memory spill is seen; another request's prefill shared the GPU in this window "
                                "(with --parallel 2 or more, prefill chunks run between decode cycles)");
        // both: the spill is named, the prefill too
        DecodeWatch r;
        CHECK(!r.report(200, 2000.0, 50, true, 0));
        const auto rs = r.report(300, 7000.0, 75, true, 4096);
        CHECK(rs && rs->warn);
        CHECK(rs && rs->text.find(", likely because of the GPU memory spill above; also, another request's prefill shared the GPU") !=
                        std::string::npos);
    }
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_mem_wddm_watch() {
    static const whirl::test::Suite s{"mem.wddm_watch",
                                      {
                                          {"testWddmWatch", [] { testWddmWatch(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_mem_wddm_watch()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
