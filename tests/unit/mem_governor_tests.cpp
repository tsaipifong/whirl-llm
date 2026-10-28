// WHIRL unit tests of module mem.governor (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-mem-governor-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/mem_governor.h"

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
// v0.2.2.2 runtime memory governor: decisions with injected budgets (R9700-like numbers: Swift balance 36966 B/token,
// pool 330240 tokens = 11.37 GiB, margin 2048 MiB)
void testMemGovernor() {
    using namespace whirl::memgov;
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    Config c;
    c.per_tok = 36966;
    c.page = 256;
    c.pool_start = 330240;
    c.pool_min = 4096;
    c.ctx = 131072;
    c.recover_s = 30;
    c.hold_s = 60;
    const std::uint64_t M = 2048 * MiB;
    const std::uint64_t B0 = 31764 * MiB;          // budget at startup
    const std::uint64_t L0 = B0 - M - 300 * MiB;   // sized with the margin (+ rounding)
    auto in = [&](double t, std::uint64_t local, std::uint64_t budget, std::uint32_t pool, bool idle, std::uint64_t parked = 0,
                  std::uint64_t freeable = 0) {
        Input x;
        x.now_s = t;
        x.local = local;
        x.budget = budget;
        x.margin = M;
        x.pool = pool;
        x.all_idle = idle;
        x.parked = parked;
        x.freeable = freeable;
        return x;
    };
    // normal runs: the budget wiggles by up to 1 GiB: never an action
    {
        Governor g(c);
        for (int i = 0; i < 200; ++i) {
            const std::uint64_t b = B0 - static_cast<std::uint64_t>((i * 37) % 1000) * MiB;
            const Plan p = g.tick(in(i * 5.0, L0, b, 330240, i % 3 == 0));
            CHECK(!p.free_cache && !p.pool_target && !p.hold_admission && !p.warn_slow && !p.episode_start);
            CHECK_EQ(p.ctx_new, 131072u);
        }
        CHECK(!g.reduced());
    }
    // drop mid-run: another program takes 4 GiB while a request runs
    Governor g(c);
    {
        const std::uint64_t B1 = B0 - 4 * GiB;
        Plan p = g.tick(in(100, L0, B1, 330240, false, 0, 600 * MiB));
        CHECK(p.episode_start && p.free_cache && p.ram_tier_note && p.hold_admission && p.shrink && p.warn_slow);
        CHECK(p.pool_target.has_value());
        // the deficit (local + margin - budget = 4096 - 300 MiB) in whole pages
        const std::uint64_t def = L0 + M - B1;
        const std::uint32_t pages = static_cast<std::uint32_t>((def + 36966ull * 256 - 1) / (36966ull * 256));
        CHECK_EQ(*p.pool_target, 330240u - pages * 256);
        CHECK_EQ(p.ctx_new, 131072u);  // the smaller pool still holds 131072
        // still running, 600 MiB of checkpoint buffers parked by step 1: same target (the parked buffers come back after)
        p = g.tick(in(105, L0 - 600 * MiB, B1, 330240, false, 600 * MiB));
        CHECK(!p.episode_start && !p.ram_tier_note && p.hold_admission);
        CHECK_EQ(*p.pool_target, 330240u - pages * 256);
        // the request ends: every slot idle -> the target is applied by the engine
        p = g.tick(in(110, L0 - 600 * MiB, B1, 330240, true, 600 * MiB));
        CHECK(p.shrink && !p.hold_admission && !p.warn_slow);
        const std::uint32_t small = *p.pool_target;
        g.resized(small, false, 110);
        CHECK_EQ(g.shrinks(), 1u);
        // after the re-allocation and the parked buffers' return: the full margin free, no further action
        const std::uint64_t L1 = L0 - static_cast<std::uint64_t>(330240 - small) * 36966;
        for (int i = 0; i < 20; ++i) {
            p = g.tick(in(115 + i * 5.0, L1, B1, small, i % 2 == 0));
            CHECK(!p.pool_target && !p.free_cache && !p.hold_admission);
        }
        CHECK(B1 - L1 >= M);
        CHECK(g.reduced());
        // recovery: the other program exits; nothing before recover_s (30 s) of room and hold (60 s after the last trigger)
        const std::uint64_t B2 = B0;
        bool grew = false;
        double t_grow = 0;
        for (double t = 220; t < 400 && !grew; t += 5) {
            p = g.tick(in(t, L1, B2, small, true));
            if (p.pool_target) {
                grew = true;
                t_grow = t;
                CHECK(*p.pool_target > small);
                CHECK(*p.pool_target <= 330240u);
                g.resized(*p.pool_target, true, t);
            }
        }
        CHECK(grew);
        CHECK(t_grow >= 220 + 30);
        // grown back to the startup pool: the episode ends at the next sample
        p = g.tick(in(t_grow + 5, L0, B2, 330240, true));
        CHECK(p.episode_end);
        CHECK(!g.reduced());
        CHECK_EQ(g.grows(), 1u);
        // no grow while a request runs
        Governor h(c);
        h.tick(in(0, L0, B1, 330240, true));
        h.resized(small, false, 0);
        for (double t = 5; t < 300; t += 5) CHECK(!h.tick(in(t, L1, B2, small, false)).pool_target.has_value());
    }
    // the context for new requests follows a pool smaller than ctx (8 GiB taken from a 5 GiB pool's server)
    {
        Config c2 = c;
        c2.pool_start = 147456;  // 5.08 GiB
        Governor g2(c2);
        const std::uint64_t L = B0 - M - 100 * MiB;
        const Plan p = g2.tick(in(0, L, B0 - 4 * GiB, 147456, false));
        CHECK(p.shrink && p.ctx_new == *p.pool_target && p.ctx_new < 131072u);
        CHECK(p.ctx_new >= 4096u);
        // more than the pool can give: the smallest pool, never below pool_min
        const Plan q = g2.tick(in(5, L, B0 - 12 * GiB, 147456, true));
        CHECK(q.shrink && *q.pool_target == 4096u);
    }
    // repeated oscillation (another program starts and exits every 40 s for an hour): the hysteresis and the doubling
    // hold keep re-allocations rare
    {
        Governor g3(c);
        std::uint32_t pool = 330240;
        std::uint32_t resizes = 0;
        for (double t = 0; t < 3600; t += 5) {
            const bool hog = static_cast<int>(t / 40) % 2 == 1;
            const std::uint64_t b = hog ? B0 - 4 * GiB : B0;
            const std::uint64_t local = L0 - static_cast<std::uint64_t>(330240 - pool) * 36966;
            const Plan p = g3.tick(in(t, local, b, pool, true));
            if (p.pool_target) {
                g3.resized(*p.pool_target, *p.pool_target > pool, t);
                pool = *p.pool_target;
                ++resizes;
            }
        }
        // every 40 s an episode would be 90 changes; the hold (60 s, doubling up to 15 min) caps it
        CHECK(resizes <= 20u);
        CHECK(g3.hold() > 60.0);
        CHECK(g3.shrinks() >= 1u);
        // REVIEW-GOV-1 P3-1: after quiet_reset_s (30 min) without an over-budget sample the doubled hold goes back to hold_s
        for (double t = 3600; t < 3600 + 1900; t += 5) {
            const Plan p = g3.tick(in(t, L0 - static_cast<std::uint64_t>(330240 - pool) * 36966, B0, pool, true));
            if (p.pool_target) {
                g3.resized(*p.pool_target, *p.pool_target > pool, t);
                pool = *p.pool_target;
            }
        }
        CHECK_EQ(pool, 330240u);
        g3.tick(in(5600, L0, B0 - 4 * GiB, 330240, true));
        CHECK(g3.hold() <= 60.0);
    }
    // lagging WDDM counters after a re-allocation (R9700 S1, 2026-10-21: a 4.8 GiB grow first read +8.1 GiB, so the governor
    // shrank right back): within trust_s the usage counts as at most the expected one; afterwards the measured one
    {
        Governor g5(c);
        const std::uint64_t B1 = B0 - 4 * GiB;
        Plan p = g5.tick(in(0, L0, B1, 330240, true));
        CHECK(p.shrink);
        const std::uint32_t small = *p.pool_target;
        const std::uint64_t L1 = L0 - static_cast<std::uint64_t>(330240 - small) * 36966;
        g5.resized(small, false, 0, L1);
        // the free shows only partly (1.8 GiB still counted): no second shrink while the counters catch up
        for (double t = 5; t < 120; t += 5) CHECK(!g5.tick(in(t, L1 + 1843 * MiB, B1, small, true)).pool_target.has_value());
        // a real further drop after the window (trust_s, 120 s) is still acted on
        p = g5.tick(in(125, L1 + 1843 * MiB, B1, small, true));
        CHECK(p.shrink);
        // a grow that first reads too high (stale old pool) but leaves more than half the margin free is not shrunk back
        // within the window (REVIEW-GOV-2 C-1 / REVIEW-GOV-3 C-2: a reading inside the trigger band after a grow is a real
        // overshoot and is acted on, see testMemGovernorGrowOvershoot; this case used L0 + 3 GiB, 0.7 GiB over the budget,
        // before the C-1 fix, and L0 + 2 GiB, inside the trigger band, before the C-2 fix)
        Governor g6(c);
        g6.tick(in(0, L0, B1, 330240, true));
        g6.resized(small, false, 0, L1);
        g6.resized(330240, true, 200, L0);
        for (double t = 205; t < 220; t += 5) CHECK(!g6.tick(in(t, L0 + 1 * GiB, B0, 330240, true)).pool_target.has_value());
        CHECK(L0 + 1 * GiB + M / 2 < B0);  // 1 GiB high, but more than half the margin still free (outside the trigger band)
        // growDenied restarts the recovery time
        Governor g7(c);
        g7.tick(in(0, L0, B1, 330240, true));
        g7.resized(small, false, 0, L1);
        bool asked = false;
        double t_ask = 0;
        for (double t = 100; t < 300 && !asked; t += 5)
            if (g7.tick(in(t, L1, B0, small, true)).pool_target) {
                asked = true;
                t_ask = t;
            }
        CHECK(asked);
        g7.growDenied(t_ask);
        CHECK(!g7.tick(in(t_ask + 5, L1, B0, small, true)).pool_target.has_value());
        CHECK(g7.tick(in(t_ask + 35, L1, B0, small, true)).pool_target.has_value());
    }
    // a grow within trust_s is sized from the larger of measured and expected usage (R9700 S1c: sized from the expected
    // one, 0.86 GiB under the measured, it overshot and was shrunk back)
    {
        const std::uint64_t B1 = B0 - 4 * GiB;
        auto growTarget = [&](std::uint64_t measured_extra) {
            Governor g8(c);
            const Plan p0 = g8.tick(in(0, L0, B1, 330240, true));
            const std::uint32_t small = *p0.pool_target;
            const std::uint64_t L1 = L0 - static_cast<std::uint64_t>(330240 - small) * 36966;
            g8.resized(small, false, 0, L1);
            for (double t = 5; t < 110; t += 5) {
                const Plan p = g8.tick(in(t, L1 + measured_extra, B0, small, true));
                if (p.pool_target) return *p.pool_target - small;
            }
            return 0u;
        };
        const std::uint32_t a = growTarget(0), b = growTarget(880 * MiB);
        CHECK(a > 0u && b > 0u && b < a);
        CHECK(static_cast<std::uint64_t>(a - b) * 36966 >= 800 * MiB);
    }
    // REVIEW-GOV-1 P2-1: with the budget back at its startup value the episode really ends: the pool grows back to its startup
    // size and new requests get the startup context again. Driven by the governor's own targets (no hand-set pool), with
    // the R9700 S2 numbers (Ornith, budget 31.02 GiB, margin 2068 MiB, startup sized with exactly the margin free) and the
    // 0.19 GiB that usage outside the pool grew by after the episode in those runs
    {
        Config r;
        r.per_tok = 22528;
        r.page = 256;
        r.pool_start = 393984;
        r.pool_min = 4096;
        r.ctx = 393984;  // --parallel 1 on a card where the context per request is the whole pool
        const std::uint64_t Mr = 2068 * MiB, Br = 31764 * MiB, Lr0 = Br - Mr, drift = 195 * MiB;
        auto inr = [&](double t, std::uint64_t local, std::uint64_t budget, std::uint32_t pool) {
            Input x;
            x.now_s = t;
            x.local = local;
            x.budget = budget;
            x.margin = Mr;
            x.pool = pool;
            x.all_idle = true;
            return x;
        };
        auto usage = [&](std::uint32_t pool, std::uint64_t extra) {
            return Lr0 - static_cast<std::uint64_t>(393984 - pool) * 22528 + extra;
        };
        // run the episode: budget drop of 4 GiB at t=0, back to Br at t_back; returns the pool at the end
        auto episode = [&](Governor& g, std::uint64_t b_back, double t_back, double t_end, bool& ended) {
            std::uint32_t pool = 393984;
            ended = false;
            for (double t = 0; t < t_end; t += 5) {
                const std::uint64_t b = t < t_back ? Br - 4 * GiB : b_back;
                const Plan p = g.tick(inr(t, usage(pool, t >= 5 ? drift : 0), b, pool));
                if (p.episode_end) ended = true;
                if (p.pool_target) {
                    const std::uint32_t np = *p.pool_target;
                    g.resized(np, np > pool, t, usage(np, 0));
                    pool = np;
                }
            }
            return pool;
        };
        Governor g(r);
        bool ended = false;
        const std::uint32_t pool = episode(g, Br, 300, 1200, ended);
        CHECK_EQ(g.shrinks(), 1u);
        CHECK_EQ(g.grows(), 1u);  // one grow, straight back
        CHECK_EQ(pool, 393984u);
        CHECK(ended);
        CHECK(!g.reduced());
        CHECK_EQ(g.ctxNew(), 393984u);
        // stable afterwards: an hour of samples at the startup budget with the drift: no action
        for (double t = 1200; t < 4800; t += 5) {
            const Plan p = g.tick(inr(t, usage(393984, drift), Br, 393984));
            CHECK(!p.pool_target && !p.free_cache && !p.episode_start);
        }
        // the budget comes back only partly (1 GiB still taken): the startup pool would leave less than endMargin free, so
        // the pool grows only part way and new requests keep the smaller context (no spill risk taken)
        Governor h(r);
        bool ended_h = false;
        const std::uint32_t pool_h = episode(h, Br - 1 * GiB, 300, 1200, ended_h);
        CHECK(!ended_h);
        CHECK(h.reduced());
        CHECK(pool_h < 393984u);
        CHECK_EQ(h.ctxNew(), pool_h);
        CHECK(Br - 1 * GiB >= usage(pool_h, drift) + Mr);  // the full margin stays free
    }
    // unknown budget: nothing
    {
        Governor g4(c);
        CHECK(!g4.tick(in(0, L0, 0, 330240, true)).free_cache);
    }
    // the status object
    {
        Status st;
        st.on = true;
        st.reduced = true;
        st.pool_tokens = 218112;
        st.pool_tokens_start = 330240;
        st.ctx_per_request = 131072;
        st.slots = 1;
        st.slots_admitting = 1;
        st.last_action = "pool 330240 -> \"218112\"";
        const std::string j = statusJson(st);
        CHECK(j.find("{\"on\":true,\"reduced\":true,\"pool_tokens\":218112,\"pool_tokens_start\":330240,\"ctx_per_request\":131072,") == 0);
        CHECK(j.find("\"last_action\":\"pool 330240 -> \\\"218112\\\"\"}") != std::string::npos);
    }
}

// REVIEW-GOV-2 C-1: a grow that costs more than expected (R9700 Swift R2b_nolim1 19:14:48: WHIRL 25.87 -> 32.50 GiB with
// the budget at 31.02 GiB, the HIP resource cache kept the freed pool) leaves WHIRL over the budget. The governor must not
// report the end of the episode ("back to the startup settings") while the measured usage is over the budget, and must act
// on it at once instead of trusting the expected usage for trust_s (120 s: the next episode started 115 s later).
// Written from the review's description, before the fix.
void testMemGovernorGrowOvershoot() {
    using namespace whirl::memgov;
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    Config r;
    r.per_tok = 22528;
    r.page = 256;
    r.pool_start = 393984;
    r.pool_min = 4096;
    r.ctx = 393984;
    const std::uint64_t Mr = 2068 * MiB, Br = 31764 * MiB, Lr0 = Br - Mr;
    auto usage = [&](std::uint32_t pool) { return Lr0 - static_cast<std::uint64_t>(393984 - pool) * 22528; };
    auto inr = [&](double t, std::uint64_t local, std::uint64_t budget, std::uint32_t pool) {
        Input x;
        x.now_s = t;
        x.local = local;
        x.budget = budget;
        x.margin = Mr;
        x.pool = pool;
        x.all_idle = true;
        return x;
    };
    // overshoot: what the grow costs above the expected usage, kept until the next re-allocation
    double t_band_end = -1;  // episode_end while inside the trigger band (less than margin/2 free), REVIEW-GOV-3 C-2
    auto run = [&](std::uint64_t overshoot, double& t_grow, double& t_act, double& t_bad_end, std::uint64_t& measured_after) {
        Governor g(r);
        std::uint32_t pool = 393984;
        bool grown = false;
        t_grow = t_act = t_bad_end = t_band_end = -1;
        measured_after = 0;
        for (double t = 0; t < 1200; t += 5) {
            const std::uint64_t b = t < 300 ? Br - 4 * GiB : Br;
            const std::uint64_t measured = usage(pool) + (grown ? overshoot : 0);
            const Plan p = g.tick(inr(t, measured, b, pool));
            if (grown) {
                if (p.episode_end && measured > b && t_bad_end < 0) t_bad_end = t;
                if (p.episode_end && measured + Mr / 2 > b && t_band_end < 0) t_band_end = t;
                if ((p.free_cache || p.shrink || p.episode_start) && t_act < 0) t_act = t;
                if (t_act >= 0) break;
            }
            if (p.pool_target) {
                const std::uint32_t np = *p.pool_target;
                g.resized(np, np > pool, t, usage(np));
                if (np > pool && !grown) {
                    grown = true;
                    t_grow = t;
                    measured_after = usage(np) + overshoot;
                }
                pool = np;
            }
        }
    };
    // the measured R2b_nolim1 case: after the grow WHIRL is 1.48 GiB over the budget
    {
        const std::uint64_t over = Mr + 1515 * MiB;  // expected leaves the margin free; measured = budget + 1.48 GiB
        double t_grow, t_act, t_bad_end;
        std::uint64_t m;
        run(over, t_grow, t_act, t_bad_end, m);
        std::printf("C-1 overshoot: grow at t=%.0f, measured after it %.2f GiB vs budget %.2f GiB; episode_end while over the "
                    "budget at t=%.0f; first action after the grow at t=%.0f (%.0f s after it)\n",
                    t_grow, m / double(GiB), Br / double(GiB), t_bad_end, t_act, t_act >= 0 ? t_act - t_grow : -1.0);
        CHECK(t_grow > 0);
        CHECK(m > Br);
        CHECK(t_bad_end < 0);                                   // no "back to the startup settings" while over the budget
        CHECK(t_act > 0 && t_act - t_grow <= r.settle_s + 10);  // acted on within the next samples, not after trust_s
    }
    // REVIEW-GOV-3 C-2: a grow that costs 1.52 GiB more than expected lands under the budget but inside the trigger band
    // (500 MiB free, the trigger is at margin/2 = 1034 MiB): it must be acted on at once too, not min()-masked for trust_s
    // with the episode end logged and a new episode ~117 s later (the review's probe: episode_end t=233, episode_start t=350)
    {
        const std::uint64_t over = Mr - 500 * MiB;  // measured 500 MiB under the budget
        double t_grow, t_act, t_bad_end;
        std::uint64_t m;
        run(over, t_grow, t_act, t_bad_end, m);
        std::printf("C-2 overshoot in the trigger band: grow at t=%.0f, measured after it %.2f GiB vs budget %.2f GiB "
                    "(trigger at %.2f GiB); episode_end inside the band at t=%.0f; first action after the grow at t=%.0f "
                    "(%.0f s after it)\n",
                    t_grow, m / double(GiB), Br / double(GiB), (Br - Mr / 2) / double(GiB), t_band_end, t_act,
                    t_act >= 0 ? t_act - t_grow : -1.0);
        CHECK(t_grow > 0);
        CHECK(m < Br && m + Mr / 2 > Br);                       // under the budget, inside the trigger band
        CHECK(t_band_end < 0);                                  // no "back to the startup settings" inside the band
        CHECK(t_act > 0 && t_act - t_grow <= r.settle_s + 10);  // acted on within the next samples, not after trust_s
    }
    // control: a grow whose reading is high but leaves more than half the margin free (counters catching up) is not acted
    // on within trust_s (this control read 300 MiB under the budget, inside the trigger band, before the C-2 fix)
    {
        const std::uint64_t over = Mr - 1300 * MiB;  // measured 1300 MiB under the budget, 266 MiB outside the band
        double t_grow, t_act, t_bad_end;
        std::uint64_t m;
        run(over, t_grow, t_act, t_bad_end, m);
        CHECK(t_grow > 0);
        CHECK(m + Mr / 2 < Br);
        CHECK(t_bad_end < 0);
        CHECK(t_act < 0 || t_act - t_grow >= r.trust_s - 5);  // not shrunk back while the counters may lag
    }
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_mem_governor() {
    static const whirl::test::Suite s{"mem.governor",
                                      {
                                          {"testMemGovernor", [] { testMemGovernor(); }},
                                          {"testMemGovernorGrowOvershoot", [] { testMemGovernorGrowOvershoot(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_mem_governor()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
