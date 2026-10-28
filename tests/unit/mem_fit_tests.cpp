// WHIRL unit tests of module mem.fit (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-mem-fit-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/json.h"
#include "whirl/numerics.h"
#include "whirl/vram_limit.h"
#include "whirl/pool_budget.h"
#include "whirl/startup_fit.h"
#include "whirl/gpu_select.h"
#include "whirl/prefill_schedule.h"  // sched::prefillExec (arch025-N15c)
#include "whirl/model_types.h"  // qwen35::max_batch_default / max_batch_limit (arch025-N15c pins)
#include "whirl/device_profile.h"  // device::profile (arch025-N15c pins)

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
void testVramLimit() {
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    // WHIRL_VRAM_LIMIT_MB parsing
    CHECK_EQ(vram::parseLimitMb("16384"), 16384ull);
    CHECK_EQ(vram::parseLimitMb(" 15360 "), 15360ull);
    CHECK_EQ(vram::parseLimitMb(""), 0ull);
    CHECK_EQ(vram::parseLimitMb("0"), 0ull);
    CHECK_EQ(vram::parseLimitMb("-1"), 0ull);
    CHECK_EQ(vram::parseLimitMb("16G"), 0ull);
    CHECK_EQ(vram::parseLimitMb("99999999999999"), 0ull);
    CHECK_EQ(vram::parseLimitMb(nullptr), 0ull);

    // clamp: the simulated card keeps other programs' use, loses the missing VRAM
    const std::uint64_t total = 32 * GiB - 300 * MiB, free = total - 1 * GiB;
    const auto c = vram::clampMemInfo(free, total, 16 * GiB);
    CHECK_EQ(c.total, 16 * GiB);
    CHECK_EQ(c.free, 15 * GiB);
    CHECK_EQ(c.total - c.free, total - free);  // used stays the same
    const auto n = vram::clampMemInfo(free, total, 0);
    CHECK(n.free == free && n.total == total);  // no limit: unchanged
    const auto big = vram::clampMemInfo(free, total, 64 * GiB);
    CHECK(big.free == free && big.total == total);  // limit above the card: unchanged
    CHECK_EQ(vram::clampMemInfo(2 * GiB, total, 1 * GiB).free, 0ull);  // floor at 0
    // WDDM budget: lowered by the VRAM the simulated card lacks
    const std::uint64_t budget = 31 * GiB;
    CHECK_EQ(vram::clampBudget(budget, total, 16 * GiB), budget - (total - 16 * GiB));
    CHECK_EQ(vram::clampBudget(budget, total, 0), budget);
    CHECK_EQ(vram::clampBudget(1 * GiB, total, 1 * GiB), 0ull);
    // allocation cap of the process: the simulated card's free VRAM at the first allocation
    CHECK_EQ(vram::processCap(free, total, 16 * GiB), 15 * GiB);

    // counter: over-cap allocations are refused and leave the count unchanged
    vram::Counter k;
    k.setCap(10 * MiB);
    CHECK(k.tryReserve(4 * MiB));
    CHECK(k.tryReserve(6 * MiB));
    CHECK_EQ(k.live(), 10 * MiB);
    CHECK(!k.tryReserve(1));
    CHECK_EQ(k.live(), 10 * MiB);
    k.release(6 * MiB);
    CHECK(!k.tryReserve(7 * MiB));
    CHECK(k.tryReserve(6 * MiB));
    CHECK(!k.tryReserve(~0ull));  // no wrap-around
    k.release(100 * MiB);         // over-release clamps at 0
    CHECK_EQ(k.live(), 0ull);
    vram::Counter u;  // cap 0 = unlimited
    CHECK(u.tryReserve(64 * GiB) && u.live() == 64 * GiB);

    // limit note / override
    vram::setLimitMb(16384);
    CHECK_EQ(vram::limitBytes(), 16384 * MiB);
    CHECK(vram::limitNote() == "VRAM limit 16384 MiB (WHIRL_VRAM_LIMIT_MB)");
    vram::setLimitMb(0);
    CHECK(vram::limitNote().empty());

    // auto-shrink decision. 27B Q3_K_S-like on 16 GB: weights 11.35 GiB, prefill buffers
    // ~0.56 MiB per row (2.3 GiB at 4096), checkpoints 150.6 MiB, q8h 36.1 KiB/token
    vram::FitInput fi;
    fi.avail = 13 * GiB + 700 * MiB;
    fi.weights = 11 * GiB + 358 * MiB;
    fi.fixed = 512 * MiB;
    fi.buf_per_row = 576 * 1024;
    fi.buf_fixed = 64 * MiB;
    fi.ckpt_bytes = 150 * MiB;
    fi.parallel = 1;
    fi.kv_per_token = 36 * 1024;
    fi.pool_min_tokens = 16384;
    fi.batch = 4096;
    fi.n_ck = 4;
    fi.n_spe = 2;
    CHECK_EQ(vram::fixedNeed(fi, 4096, 4, 2), fi.weights + fi.fixed + fi.buf_fixed + fi.buf_per_row * 4096 + 6 * fi.ckpt_bytes);
    const vram::FitPlan p = vram::planFit(fi);
    CHECK(p.reduced && p.fits);
    CHECK_EQ(p.batch, 1280u);  // VRAM-01: never below one 1024 chunk + the 256-row merged tail
    CHECK_EQ(p.n_ck, 2u);
    CHECK_EQ(p.n_spe, 1u);
    CHECK(p.pool_tokens >= 16384);
    CHECK(p.note == "prefill batch 4096 -> 1280, checkpoints per slot 4 -> 2, shared checkpoints 2 -> 1");
    // the same model on 32 GB: nothing changes
    vram::FitInput roomy = fi;
    roomy.avail = 29 * GiB;
    const vram::FitPlan pr = vram::planFit(roomy);
    CHECK(!pr.reduced && pr.fits && pr.batch == 4096 && pr.n_ck == 4 && pr.n_spe == 2 && pr.note.empty());
    // just short of the wanted pool: only the first step (batch 2048)
    vram::FitInput step1 = fi;
    step1.avail = vram::fixedNeed(fi, 4096, 4, 2) + 16000ull * fi.kv_per_token;
    const vram::FitPlan p1 = vram::planFit(step1);
    CHECK(p1.reduced && p1.batch == 2048 && p1.n_ck == 4 && p1.n_spe == 2);
    // user-set batch: only the checkpoints move
    vram::FitInput fixb = fi;
    fixb.batch_fixed = true;
    const vram::FitPlan pb = vram::planFit(fixb);
    CHECK(pb.batch == 4096 && pb.n_ck == 1 && pb.n_spe == 0);
    // both fixed: no change, and it does not fit
    vram::FitInput fixall = fixb;
    fixall.ck_fixed = true;
    fixall.avail = 12 * GiB;
    const vram::FitPlan pf = vram::planFit(fixall);
    CHECK(!pf.reduced && !pf.fits && pf.pool_tokens < 4096);
    // hopeless (weights alone over the budget): every step taken, still does not fit
    vram::FitInput hopeless = fi;
    hopeless.avail = 10 * GiB;
    const vram::FitPlan ph = vram::planFit(hopeless);
    CHECK(ph.reduced && !ph.fits && ph.batch == 1280 && ph.n_ck == 1 && ph.n_spe == 0 && ph.pool_tokens == 0);
    // prefix cache off (0 checkpoints): checkpoint steps change nothing
    vram::FitInput nopc = fi;
    nopc.n_ck = 0;
    nopc.n_spe = 0;
    const vram::FitPlan pn = vram::planFit(nopc);
    CHECK(pn.n_ck == 0 && pn.n_spe == 0 && pn.batch <= 2048);
    // VRAM-01 sweep: whatever the budget, the planned batch is never below the scheduler's largest
    // forward (1024-row chunk + 256-row tail), so a long prompt cannot hit BatchTooLarge
    for (std::uint32_t req : {4096u, 2048u, 1536u, 1280u}) {
        for (std::uint64_t av = 8 * GiB; av <= 30 * GiB; av += 256 * MiB) {
            vram::FitInput sw = fi;
            sw.batch = req;
            sw.avail = av;
            const vram::FitPlan ps = vram::planFit(sw);
            CHECK(ps.batch >= 1280u && ps.batch <= req);
        }
    }
    CHECK_EQ(vram::min_prefill_batch, 1280u);
    // the same floor for a user-set WHIRL_PREFILL_BATCH (REVIEW-RC4-C P2-3)
    CHECK_EQ(vram::prefillBatchFloor(1u), 1280u);
    CHECK_EQ(vram::prefillBatchFloor(512u), 1280u);
    CHECK_EQ(vram::prefillBatchFloor(1279u), 1280u);
    CHECK_EQ(vram::prefillBatchFloor(1280u), 1280u);
    CHECK_EQ(vram::prefillBatchFloor(4096u), 4096u);
    // arch025-N15c pins: WHIRL_PREFILL_BATCH, one rule for whirl.exe and whirl-server == the 44895be server code
    // (server_main.cpp: clamp to [1, max_batch_limit] then the floor; after the load min(loaded, floor(v))). The
    // 44895be CLI (build_model.cpp max(max_batch_default, min(limit, v)); cmd_chat.cpp max(1280, min(loaded, v)))
    // differed only for v < 4096 at load and for a loaded batch < 1280 after it.
    {
        const std::uint32_t lim = qwen35::max_batch_limit;
        std::size_t cli_load_diff = 0, cli_cap_diff = 0;
        for (std::uint32_t v : {0u, 1u, 2u, 512u, 1024u, 1279u, 1280u, 1281u, 2048u, 3000u, 4095u, 4096u, 4097u, 8192u, 16383u, 16384u,
                                16385u, 100000u, 0xffffffffu}) {
            CHECK_EQ(vram::prefillBatchAtLoad(v, lim), vram::prefillBatchFloor(std::clamp(v, 1u, lim)));
            cli_load_diff += vram::prefillBatchAtLoad(v, lim) != std::max(qwen35::max_batch_default, std::min(lim, v));
            for (std::uint32_t loaded : {1024u, 1280u, 2048u, 4096u, 16384u}) {
                CHECK_EQ(vram::prefillBatchCap(loaded, v), std::min(loaded, vram::prefillBatchFloor(v)));
                cli_cap_diff += vram::prefillBatchCap(loaded, v) != std::max<std::uint32_t>(1280u, std::min(loaded, v));
            }
        }
        CHECK_EQ(cli_load_diff, std::size_t(11));  // the 11 listed values below 4096 (the CLI floor was 4096)
        CHECK_EQ(vram::prefillBatchAtLoad(2048u, lim), 2048u);  // the CLI loaded 4096 rows for this value
        CHECK_EQ(cli_cap_diff, std::size_t(19));  // loaded 1024, every v: the CLI raised it to 1280 (above the loaded batch)
    }
    // arch025-N15c: sched::prefillExec == the formula both front ends wrote out (44895be common.cpp / server_main.cpp)
    for (hip::Arch a : {hip::Arch::gfx1201, hip::Arch::gfx1151})
        for (bool moe : {false, true})
            for (std::uint32_t mb : {1280u, 2048u, 2304u, 4096u, 8192u, 16384u})
                for (std::optional<std::size_t> req : {std::optional<std::size_t>{}, std::optional<std::size_t>{1024}, std::optional<std::size_t>{3000},
                                                       std::optional<std::size_t>{100000}}) {
                    const device::Profile& dprof = device::profile(a);
                    CHECK_EQ(sched::prefillExec(req, dprof, moe, mb),
                             sched::prefillExecFor(req.value_or(sched::prefillExecDefault(dprof, moe)), mb, dprof.prefill_fill_batch));
                }
}

// server defaults by card size (vram::cardDefaults)
void testCardDefaults() {
    const std::uint64_t GiB = 1ull << 30, MiB = 1ull << 20;
    // R9700 32 GB / RX 7900 XTX 24 GB: unchanged (4 slots, f16 when it fits, no headroom, no log line)
    for (std::uint64_t t : {32 * GiB - 140 * MiB, 24 * GiB, 20 * GiB}) {
        vram::CardInput in;
        in.total = t;
        const vram::CardDefaults d = vram::cardDefaults(in);
        CHECK(!d.small && d.parallel == 4 && !d.parallel_auto && d.headroom == 0 && d.note.empty());
    }
    // 16 GB card (or WHIRL_VRAM_LIMIT_MB=16384 / 14336): 1 slot; no KV preference (BAL-Q8); no extra headroom since
    // 0.2.2.1 (the WDDM margin, 3 GiB with a display attached, covers the desktop)
    for (std::uint64_t t : {16 * GiB, 14336 * MiB, 20 * GiB - 1}) {
        vram::CardInput in;
        in.total = t;
        const vram::CardDefaults d = vram::cardDefaults(in);
        CHECK(d.small && d.parallel == 1 && d.parallel_auto);
        CHECK_EQ(d.headroom, 0ull);
        CHECK(d.note.find("small card") != std::string::npos && d.note.find("--parallel 1") != std::string::npos &&
              d.note.find("KV") == std::string::npos && d.note.find("WDDM budget margin") != std::string::npos);
    }
    vram::CardInput s16;
    s16.total = 16 * GiB;
    // --parallel given: kept (also --parallel 4 explicitly)
    {
        vram::CardInput in = s16;
        in.parallel_arg = 4;
        const vram::CardDefaults d = vram::cardDefaults(in);
        CHECK(d.small && d.parallel == 4 && !d.parallel_auto && d.note.find("--parallel 4 (given)") != std::string::npos);
        in.parallel_arg = 2;
        CHECK_EQ(vram::cardDefaults(in).parallel, 2u);
    }
    // WHIRL_VRAM_HEADROOM_MB overrides, on small and large cards; 0 = none
    {
        vram::CardInput in = s16;
        in.headroom_mb_env = 0;
        CHECK_EQ(vram::cardDefaults(in).headroom, 0ull);
        in.headroom_mb_env = 3072;
        CHECK_EQ(vram::cardDefaults(in).headroom, 3072 * MiB);
        vram::CardInput big;
        big.total = 32 * GiB;
        big.headroom_mb_env = 1024;
        const vram::CardDefaults d = vram::cardDefaults(big);
        CHECK(!d.small && d.parallel == 4 && d.headroom == 1024 * MiB && d.note.find("1024 MiB") != std::string::npos);
    }
    // explicit WHIRL_POOL_RESERVE_MB: no default headroom (WHIRL_VRAM_HEADROOM_MB still adds one)
    {
        vram::CardInput in = s16;
        in.reserve_explicit = true;
        CHECK_EQ(vram::cardDefaults(in).headroom, 0ull);
        in.headroom_mb_env = 512;
        CHECK_EQ(vram::cardDefaults(in).headroom, 512 * MiB);
    }
    // UMA iGPU (8060S) and an unknown size are never "small"
    {
        vram::CardInput in = s16;
        in.uma = true;
        const vram::CardDefaults d = vram::cardDefaults(in);
        CHECK(!d.small && d.parallel == 4 && d.headroom == 0);
        vram::CardInput z;
        CHECK(!vram::cardDefaults(z).small);
    }
}


// ---------------------------------------------------------------------------
// issue #5: KV pool re-fit to the WDDM budget read after the pool is allocated
void testPoolBudget() {
    using whirl::poolbudget::fitPool;
    constexpr std::uint64_t MiB = 1ull << 20;
    constexpr double GiB = 1073741824.0;
    auto gib = [&](double g) { return static_cast<std::uint64_t>(g * GiB); };
    const std::uint64_t per_tok = 36966;  // 36.1 KiB/token (q8h KV, Swift-1.5 27B), as in the issue #5 log
    const std::uint32_t page = 256;
    const std::uint64_t margin = 768 * MiB;
    // the user's numbers: sized with budget 31.02 GiB, used 17.64 GiB -> room 12.63 GiB -> pool 366592 tokens
    const std::uint64_t room = gib(31.02) - gib(17.64) - margin;
    const std::uint32_t pool0 = static_cast<std::uint32_t>(room / per_tok / page * page);
    CHECK(pool0 >= 366080 && pool0 <= 367104);  // within two pages of the log's 366592 (rounded GiB figures)
    // with the pool allocated: local 29.92 GiB, budget 29.55 GiB (after pinning) -> must shrink
    {
        const auto t = fitPool(pool0, gib(29.92), gib(29.55), margin, per_tok, page, 4096);
        CHECK(t.shrink);
        CHECK(!t.at_floor);
        CHECK_EQ(t.pool_tokens % page, 0u);
        CHECK(t.over == gib(29.92) + margin - gib(29.55));
        // after freeing the dropped pages, local + margin fits under the post-pin budget
        const std::uint64_t freed = static_cast<std::uint64_t>(pool0 - t.pool_tokens) * per_tok;
        CHECK(gib(29.92) - freed + margin <= gib(29.55));
        // and not more than one page beyond what is needed
        CHECK(gib(29.92) - freed + margin + page * per_tok > gib(29.55));
        std::printf("pool budget: issue #5 numbers: pool %u -> %u tokens (%.2f -> %.2f GiB)\n", pool0, t.pool_tokens,
                    pool0 * static_cast<double>(per_tok) / GiB, t.pool_tokens * static_cast<double>(per_tok) / GiB);
    }
    // fits: no change (also exactly at the line)
    {
        const auto t = fitPool(pool0, gib(29.00), gib(31.02), margin, per_tok, page, 4096);
        CHECK(!t.shrink);
        CHECK_EQ(t.over, 0u);
        CHECK_EQ(t.pool_tokens, pool0);
        const auto e = fitPool(pool0, gib(29.00), gib(29.00) + margin, margin, per_tok, page, 4096);
        CHECK(!e.shrink);
    }
    // budget unknown: no change
    CHECK(!fitPool(pool0, gib(40.0), 0, margin, per_tok, page, 4096).shrink);
    // far over: kept at the smallest pool, flagged
    {
        const auto t = fitPool(65536, gib(40.0), gib(20.0), margin, per_tok, page, 4096);
        CHECK(t.shrink);
        CHECK(t.at_floor);
        CHECK_EQ(t.pool_tokens, 4096u);
        const auto u = fitPool(4096, gib(40.0), gib(20.0), margin, per_tok, page, 4096);  // already there
        CHECK(!u.shrink);
        CHECK(u.at_floor);
        CHECK(u.over > 0);
    }
    // one byte over: one page less
    {
        const auto t = fitPool(8192, gib(10.0) + 1, gib(10.0) + margin, margin, per_tok, page, 4096);
        CHECK_EQ(t.pool_tokens, 8192u - page);
    }
    // the shrink loop guard: another round only when the last free really lowered the local usage
    {
        using whirl::poolbudget::usageFell;
        const std::uint64_t freed = gib(1.0);
        CHECK(usageFell(gib(30.0), gib(29.0), freed));
        CHECK(usageFell(gib(30.0), gib(29.5), freed));      // half: enough (WDDM counts in its own granularity)
        CHECK(!usageFell(gib(30.0), gib(29.6), freed));     // stale counter: stop
        CHECK(!usageFell(gib(30.0), gib(30.0), freed));
        CHECK(!usageFell(gib(30.0), gib(30.2), freed));
    }
}

// v0.2.2.1 default WDDM budget margin (owner 2026-10-21): other processes' VRAM the budget does not show yet + room
void testPoolMargin() {
    using namespace whirl::poolbudget;
    constexpr std::uint64_t MiB = 1ull << 20;
    auto m = [&](std::uint64_t budget_mib, std::uint64_t start_mib, bool display, bool known, std::uint64_t others_mib) {
        MarginIn in;
        in.local_budget = budget_mib * MiB;
        in.budget_start = start_mib * MiB;
        in.display = display;
        in.others_known = known;
        in.others = others_mib * MiB;
        return defaultMargin(in);
    };
    // R9700 headless, nothing else on it: budget 31.02 GiB, 6% = 1906 MiB < the 2048 floor
    {
        const Margin r = m(31764, 31764, false, true, 0);
        CHECK_EQ(r.bytes, 2048 * MiB);
        CHECK_EQ(r.room, 2048 * MiB);
        CHECK_EQ(r.pending, 0ull);
        CHECK(r.why.rfind("2048 MiB room (no display on this GPU)", 0) == 0);
    }
    // the hog test: another process holds 3307 MiB, the budget does not show it yet -> counted, plus the room
    {
        const Margin r = m(31764, 31764, false, true, 3307);
        CHECK_EQ(r.pending, 3307 * MiB);
        CHECK_EQ(r.bytes, (3307 + 2048) * MiB);
        CHECK(r.why.find("+ 3307 MiB of the 3307 MiB other processes use that the budget does not show yet") != std::string::npos);
    }
    // ... after the pool was allocated the budget dropped 2.94 GiB (31.02 -> 28.08): only the rest is still pending
    {
        const Margin r = m(28754, 31764, false, true, 3307);
        CHECK_EQ(r.pending, (3307 - (31764 - 28754)) * MiB);
        CHECK(r.why.find("(it dropped 3010 MiB since startup)") != std::string::npos);
        // dropped more than the others used: never negative, never counted twice
        CHECK_EQ(m(24000, 31764, false, true, 3307).pending, 0ull);
    }
    // a monitor on the card: 3072 MiB room
    CHECK_EQ(m(31764, 31764, true, true, 0).room, 3072 * MiB);
    CHECK(m(31764, 31764, true, true, 0).why.find("display is attached") != std::string::npos);
    // a large card: 6% of the budget (48 GiB budget: 2949.12 -> 2950 MiB)
    {
        const Margin r = m(49152, 49152, false, true, 0);
        CHECK_EQ(r.room, 2950 * MiB);
        CHECK(r.why.rfind("2950 MiB room (6% of the 49152 MiB budget)", 0) == 0);
    }
    // counters unreadable: the budget-only rule, said so
    {
        const Margin r = m(31764, 31764, false, false, 99999);
        CHECK_EQ(r.bytes, 2048 * MiB);
        CHECK(r.why.find("other processes' VRAM unknown") != std::string::npos);
    }
    // budget at start unknown: treated as now (nothing dropped)
    CHECK_EQ(m(31764, 0, false, true, 1000).pending, 1000 * MiB);
    // WHIRL_POOL_BUDGET_MARGIN_MB sets the room; pending is still counted
    {
        MarginIn in;
        in.local_budget = 31764 * MiB;
        in.budget_start = 31764 * MiB;
        in.others_known = true;
        in.others = 500 * MiB;
        in.room_override = 4096 * MiB;
        const Margin r = defaultMargin(in);
        CHECK_EQ(r.room, 4096 * MiB);
        CHECK_EQ(r.bytes, (4096 + 500) * MiB);
        CHECK(r.why.rfind("4096 MiB room (WHIRL_POOL_BUDGET_MARGIN_MB)", 0) == 0);
    }
    // whole MiB
    CHECK_EQ(m(31764, 31764, false, true, 1).bytes % MiB, 0ull);
}

// REVIEW-GOV-3 question 4 (R9700 Ornith --mmproj, gov3 orn_new3): startup left exactly the margin free (29696 MiB local,
// budget 31767 MiB, margin 2068 MiB); the first image added 873 MiB (encoder weights, 860.5 MiB pinned, made resident) and a
// 1920x1080 image 134 MiB more (activations that did not fit the lent prefill scratch), so 30 MiB were left before the
// governor's trigger (margin/2 free) and a false episode could never end. With --mmproj the startup KV pool must leave room
// for what the encoder adds; without it (text only) the pool must be exactly what it was. Written before the fix.
void testVisionPoolReserve() {
    using namespace whirl;
    namespace nu = whirl::numerics;
    constexpr std::uint64_t KiB = 1ull << 10, MiB = 1ull << 20;
    const std::uint64_t per_tok = 22528;                   // Ornith MXFP4 KV bytes per token
    const std::uint64_t avail = 8468 * MiB;                // orn_new3: room 8.27 GiB (budget - used - margin)
    const std::uint64_t added = 873 * MiB + 134 * MiB;     // measured local usage the encoder added (orn_new3 health)
    poolbudget::VisionVram vv;
    vv.weights = 860 * MiB + 512 * KiB;                    // the mmproj's weights (the startup log's pinned MiB)
    vv.act_extra = 134 * MiB;                              // beyond the lent scratch (the 1920x1080 step)
    const std::uint64_t r = poolbudget::visionReserve(vv);
    const nu::PoolFit text = nu::serverPoolFit(avail, per_tok, std::nullopt, false, 131072, false, 256, 8 * 262144);
    const nu::PoolFit text2 =
        nu::serverPoolFit(poolbudget::kvAvail(avail, poolbudget::visionReserve({})), per_tok, std::nullopt, false, 131072, false, 256, 8 * 262144);
    const nu::PoolFit vis =
        nu::serverPoolFit(poolbudget::kvAvail(avail, r), per_tok, std::nullopt, false, 131072, false, 256, 8 * 262144);
    std::printf("vision pool reserve: reserve %llu MiB (encoder added %llu MiB); KV pool text-only %u tokens, no-mmproj via "
                "kvAvail %u, --mmproj %u tokens (%lld tokens less)\n",
                static_cast<unsigned long long>(r / MiB), static_cast<unsigned long long>(added / MiB), text.pool, text2.pool,
                vis.pool, static_cast<long long>(text.pool) - static_cast<long long>(vis.pool));
    // text only: exactly the same pool (no reserve, kvAvail is the identity)
    CHECK_EQ(poolbudget::visionReserve({}), 0ull);
    CHECK_EQ(text2.pool, text.pool);
    for (std::uint64_t a : {0ull, 1ull, 4096ull * MiB, 8468ull * MiB, 31767ull * MiB}) CHECK_EQ(poolbudget::kvAvail(a, 0), a);
    // --mmproj: the reserve covers what the encoder really added, without reserving much more
    CHECK(r >= added);
    CHECK(r <= added + 64 * MiB);
    // and the pool is smaller by at least that much, so after the encoder and the 1920x1080 image the margin is still free
    CHECK(vis.pool < text.pool);
    CHECK((text.pool - vis.pool) * per_tok >= added);
    CHECK(vis.pool >= 131072);  // one 131072-token request still fits
}

// v0.2.2.1 startup fit of explicit settings (owner 2026-10-21: adjust, log, refuse only below the minimum)
void testStartupFit() {
    using namespace whirl;
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    // 1. RAM tier (ramTierFit)
    // explicit: kept when within half of the available RAM and within the 20480 MiB cap
    CHECK(!startfit::ramTierFit(12288, 48 * GiB, 16384, true).has_value());
    CHECK(!startfit::ramTierFit(20480, 48 * GiB, 16384, true).has_value());
    // the hipError 719 repro (REVIEW-0221 C1): 24064 MiB on a box with 49247 MiB available is within half, but capped
    {
        const auto r = startfit::ramTierFit(24064, 49247 * MiB, 16384, true);
        CHECK(r.has_value() && r->mb == 20480ull && r->safe && !r->half);
    }
    // v0.2.2.2: the cap first, then half of the available RAM on the capped size: 65536 with 48955 MiB available -> the cap
    // (20480 is within half), no longer the default size
    {
        const auto r = startfit::ramTierFit(65536, 48955 * MiB, 16384, true);
        CHECK(r.has_value() && r->mb == 20480ull && !r->half && r->safe);
    }
    {
        const auto r = startfit::ramTierFit(65536, 64 * GiB, 0, true);
        CHECK(r.has_value() && r->mb == 20480ull && !r->half && r->safe);
    }
    // the cap, then half: 30 GiB available -> 15360
    {
        const auto r = startfit::ramTierFit(65536, 30 * GiB, 8192, true);
        CHECK(r.has_value() && r->mb == 15360ull && r->half && r->safe);
    }
    CHECK_EQ(startfit::ramTierFit(9000, 17000 * MiB, 0, true)->mb, 8192ull);  // half = 8500 MiB -> 8192 (512 steps)
    CHECK_EQ(startfit::ramTierFit(9000, 12 * GiB, 8192, true)->mb, 6144ull);  // half
    // monotonic (0.2.2.2 item 1; 0.2.2.1 gave 22528 -> 20480 but 24064 -> 16384 with 47348 MiB available): asking for more
    // never gives less, for any available RAM
    for (const std::uint64_t avail_mb : {8192ull, 17000ull, 30720ull, 47348ull, 49247ull, 65536ull, 131072ull}) {
        std::uint64_t prev = 0;
        bool mono = true;
        for (std::uint64_t ask = 512; ask <= 65536; ask += 512) {
            const auto r = startfit::ramTierFit(ask, avail_mb * MiB, 16384, true);
            const std::uint64_t got = r ? r->mb : ask;
            mono = mono && got >= prev;
            prev = got;
        }
        CHECK(mono);
    }
    CHECK_EQ(startfit::ramTierFit(22528, 47348 * MiB, 16384, true)->mb, 20480ull);
    CHECK_EQ(startfit::ramTierFit(24064, 47348 * MiB, 16384, true)->mb, 20480ull);
    // the automatic size (a PC with 128 GB RAM: 32 GiB by the 1/4 rule) is capped too, never the half rule
    {
        const auto r = startfit::ramTierFit(32768, 100 * GiB, 32768, false);
        CHECK(r.has_value() && r->mb == 20480ull && r->safe && !r->half);
        CHECK(!startfit::ramTierFit(16384, 20 * GiB, 16384, false).has_value());  // auto within the cap: kept (half rule not applied)
    }
    CHECK(!startfit::ramTierFit(0, 8 * GiB, 8192, true).has_value());  // 0 = tiers off: kept
    CHECK(!startfit::ramTierFit(4096, 0, 8192, true).has_value());     // RAM unknown, small: kept
    CHECK_EQ(startfit::ramTierFit(30000, 0, 0, true)->mb, 20480ull);    // RAM unknown, large: capped
    // s8-719 repro (scratch\s8_719\results.txt, 2026-10-22 23:34-23:43): the 8060S at a 96 GB carve-out (32407 MiB RAM,
    // SharedSystemMemory 16203 MiB on every adapter), the R9700 also a HIP GPU (DXGI non-local budget 15435 MiB), the
    // embedding (1288 MiB) pinned, Swift precise server. --kv-ram-mb 4096 answered, 8192 failed with hipError 719 on the
    // first request (0.2.3 and v0.2.2.2); with only the 8060S visible to HIP, 8192 answered.
    {
        startfit::PinnedState s;
        s.total_phys = 32407 * MiB;
        s.pinned_max = 1441 * MiB;  // 8060S adapter Shared Usage with the embedding pinned (case s23_ram0)
        s.gpus = {{0, 0, 16203 * MiB}, {15435 * MiB, 1288 * MiB, 16203 * MiB}};  // 8060S (integrated: budget 0), R9700
        const std::uint64_t room = startfit::pinnedRoomMb(s);
        CHECK(room >= 4096ull && room < 8192ull);
        const auto r = startfit::ramTierFit(8192, 19374 * MiB, 0, true, room);  // 19374 MiB available (case g23_swift)
        CHECK(r.has_value() && r->mb < 8192ull && r->mb >= 4096ull && r->pinned);
        CHECK(!startfit::ramTierFit(4096, 19374 * MiB, 0, true, room).has_value());  // 4096 answered: kept
        // only the 8060S visible to HIP (HIP_VISIBLE_DEVICES): 8192 answered -> kept
        s.gpus = {{0, 0, 16203 * MiB}};
        CHECK(!startfit::ramTierFit(8192, 19430 * MiB, 0, true, startfit::pinnedRoomMb(s)).has_value());
    }
    // kvram719 (the same PC at a 64/64 carve-out: 65175 MiB RAM, R9700 non-local budget 48026 MiB, the 8060S's
    // SharedSystemMemory 32587 MiB, both HIP GPUs, process on the R9700): 24064 failed, 20480 worked
    {
        startfit::PinnedState s;
        s.total_phys = 65175 * MiB;
        s.pinned_max = 1288 * MiB;
        s.gpus = {{48026 * MiB, 1288 * MiB, 32587 * MiB}, {0, 0, 32587 * MiB}};
        const std::uint64_t room = startfit::pinnedRoomMb(s);
        CHECK(room >= 20480ull && room < 24064ull);
        CHECK(!startfit::ramTierFit(20480, 55 * GiB, 16384, true, room).has_value());
        s.pinned_max = (10890 + 1288) * MiB;  // another server had 10890 MiB pinned: 20480 failed (kvram719 10:41)
        CHECK(startfit::pinnedRoomMb(s) < 20480ull);
    }
    CHECK_EQ(startfit::pinnedRoomMb(startfit::PinnedState{}), startfit::pinned_room_unknown);  // nothing known
    // everything pinned by other programs: room 0 -> the tier is off rather than crashing
    {
        startfit::PinnedState s;
        s.total_phys = 64 * GiB;
        s.pinned_max = 30 * GiB;
        const auto r = startfit::ramTierFit(16384, 30 * GiB, 16384, false, startfit::pinnedRoomMb(s));
        CHECK(r.has_value() && r->mb == 0ull && r->pinned);
    }
    // REVIEW-S8-719 A1: the reason text of an adjusted (healthy) server must match none of the error patterns the gates and
    // repro scripts scan server logs for (kvram719_tier.ps1:50 "hipError|\| failed:", golden.ps1:377/393/397, fix719.ps1:190;
    // PowerShell -match / Select-String are case-insensitive), and it still names the error the user would have seen (719)
    {
        const std::regex bad(R"(hipError|hip.*\b719\b|\b719\b.*hip|\| failed:|PAL failed|Failed PAL|result:-5|unspecified launch)",
                             std::regex::icase);
        for (const std::uint64_t room : {0ull, 512ull, 4096ull, 8192ull, 12288ull, 20480ull})
            for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
                const std::string why = startfit::pinnedWhy(room, n);
                const std::string line = "2026-10-23 00:05:55.955 W adjusted: --kv-ram-mb 8192 -> " + std::to_string(room) + " MiB (" + why + ")";
                CHECK(!std::regex_search(line, bad));
                CHECK(why.find("719") != std::string::npos);
            }
    }
    // REVIEW-S8-719 A2: every measured working size stays admitted, and single-GPU defaults with no failure on record keep
    // their tier (the machine reserve 6144 MiB cut them)
    {
        // 8060S only visible (HIP_VISIBLE_DEVICES), 96 GB carve-out, case s23_ram12288_vis (scratch\s8_719\results.txt 23:39):
        // --kv-ram-mb 12288 was lowered to 8704 by the half rule (17643 MiB available) and answered (8060S Shared Usage 10161 MiB)
        startfit::PinnedState s;
        s.total_phys = 32407 * MiB;
        s.pinned_max = 1441 * MiB;
        for (const std::uint64_t own : {0ull, 1441ull}) {  // DXGI reports 0 for the iGPU; PDH 1441 (REVIEW C4)
            s.gpus = {{0, own * MiB, 16203 * MiB}};
            const auto r = startfit::ramTierFit(12288, 17643 * MiB, 0, true, startfit::pinnedRoomMb(s));
            CHECK(r.has_value() && r->mb == 8704ull && r->half && !r->pinned);
            // golden's 8192 on the 8060S alone with other programs holding 1000 MiB more shared GPU memory: kept
            startfit::PinnedState s2 = s;
            s2.pinned_max = (1441 + 1000) * MiB;
            CHECK(!startfit::ramTierFit(8192, 18930 * MiB, 0, true, startfit::pinnedRoomMb(s2)).has_value());
        }
        // single R9700 on a 32 GB-RAM host (DXGI non-local budget as measured, 15435 MiB), other programs 700 MiB of shared
        // GPU memory: the automatic tier (1/4 of RAM, 8192) and one max-context entry (9216) are kept
        startfit::PinnedState r9;
        r9.total_phys = 32407 * MiB;
        r9.pinned_max = (1294 + 700) * MiB;
        r9.gpus = {{15435 * MiB, 1294 * MiB, 16203 * MiB}};
        CHECK(!startfit::ramTierFit(8192, 20000 * MiB, 8192, false, startfit::pinnedRoomMb(r9), 9216).has_value());
        CHECK(!startfit::ramTierFit(9216, 20000 * MiB, 9216, false, startfit::pinnedRoomMb(r9), 9216).has_value());
        // single R9700 at 64 GB (kvram719 numbers), automatic 16384: kept
        r9.total_phys = 65175 * MiB;
        r9.pinned_max = 1300 * MiB;
        r9.gpus = {{48026 * MiB, 1294 * MiB, 32587 * MiB}};
        CHECK(!startfit::ramTierFit(16384, 50000 * MiB, 16384, false, startfit::pinnedRoomMb(r9), 9216).has_value());
        // single 8060S at 64 GB, automatic 8192: kept
        startfit::PinnedState i8;
        i8.total_phys = 65175 * MiB;
        i8.pinned_max = 1441 * MiB;
        i8.gpus = {{0, 1441 * MiB, 32587 * MiB}};
        CHECK(!startfit::ramTierFit(8192, 50000 * MiB, 8192, false, startfit::pinnedRoomMb(i8), 9216).has_value());
        // R9700 + 8060S at 64/64: process on the 8060S (golden 8192), on the R9700 (automatic 16384, explicit 20480): kept
        startfit::PinnedState both;
        both.total_phys = 65175 * MiB;
        both.pinned_max = 1441 * MiB;
        both.gpus = {{0, 1441 * MiB, 32587 * MiB}, {48026 * MiB, 1294 * MiB, 32587 * MiB}};
        const std::uint64_t rb = startfit::pinnedRoomMb(both);
        CHECK(!startfit::ramTierFit(8192, 50000 * MiB, 0, true, rb, 9216).has_value());
        CHECK(!startfit::ramTierFit(16384, 50000 * MiB, 16384, false, rb, 9216).has_value());
        CHECK(!startfit::ramTierFit(20480, 50000 * MiB, 16384, true, rb, 9216).has_value());
    }
    // REVIEW-S8-719 C2 (FR-CACHE P3-4, src/tier/ram_size.h): a RAM tier the pinned room lowers below one max-context entry
    // is off (the longest session could never be kept), not a tier smaller than one entry
    {
        // REVIEW-S8-719-2 X1/X2: only the automatic size; an explicit size is clamped to the room and used as given
        // (ram_size.h), so golden's 8192 on two GPUs at 96 GB gets 4096 (round 1, 709d575: gate 8/8), not off
        const auto e = startfit::ramTierFit(8192, 19374 * MiB, 0, true, 4096, 9216);  // explicit, room 4096 < entry 9216
        CHECK(e.has_value() && e->mb == 4096ull && e->pinned && !e->below_entry);
        const auto a = startfit::ramTierFit(9216, 19374 * MiB, 9216, false, 4096, 9216);  // automatic = one entry
        CHECK(a.has_value() && a->mb == 0ull && a->below_entry);
        const auto k = startfit::ramTierFit(8192, 19374 * MiB, 0, true, 4096, 4096);  // room holds one entry: 4096
        CHECK(k.has_value() && k->mb == 4096ull && k->pinned && !k->below_entry);
        CHECK(!startfit::ramTierFit(4096, 19374 * MiB, 0, true, startfit::pinned_room_unknown, 9216).has_value());  // user's own size
    }
    // REVIEW-S8-719-2 X2: an explicit size is monotonic in the request (asking for more never gives a smaller tier), and a
    // room > 0 never turns an explicit tier off. Sweep 0..20480 step 512 over room / entry / available-RAM configurations,
    // including the measured two-GPU room 4096 with entry 10240 (8060S 96 GB + R9700, golden slot).
    {
        struct Cfg { std::uint64_t room, entry, avail_mb; };
        const Cfg cfgs[] = {{4096, 10240, 19032}, {4096, 9216, 19374}, {4096, 4096, 19374}, {0, 10240, 19032},
                            {12288, 10240, 17643}, {12288, 0, 17643}, {20480, 9216, 50000}, {19456, 10240, 50000},
                            {8192, 16384, 12000}, {startfit::pinned_room_unknown, 10240, 30000}, {512, 9216, 40000}};
        for (const Cfg& c : cfgs) {
            std::uint64_t prev = 0;
            bool mono = true, never_off = true, within = true;
            for (std::uint64_t ask = 0; ask <= 20480; ask += 512) {
                const auto r = startfit::ramTierFit(ask, c.avail_mb * MiB, 0, true, c.room, c.entry);
                const std::uint64_t got = r ? r->mb : ask;
                if (got < prev) mono = false;
                if (ask > 0 && c.room != startfit::pinned_room_unknown && c.room > 0 && got == 0) never_off = false;
                if (got > ask || (c.room != startfit::pinned_room_unknown && got > c.room)) within = false;
                if (r && r->below_entry) never_off = false;  // the entry rule is for the automatic size only
                prev = got;
            }
            CHECK(mono);
            CHECK(never_off);
            CHECK(within);
        }
    }
    // REVIEW-S8-719-2 C-1: when the pinned room turns the tier off, the text says the SSD tier is off with it (both forms),
    // and still matches none of the log-scan error patterns
    {
        const std::regex bad(R"(hipError|hip.*\b719\b|\b719\b.*hip|\| failed:|PAL failed|Failed PAL|result:-5|unspecified launch)",
                             std::regex::icase);
        for (const bool be : {true, false}) {
            const std::string off = startfit::pinnedOffWhy(be, 10240);
            CHECK(off.find("SSD") != std::string::npos);
            CHECK(off.find("RAM") != std::string::npos);
            const std::string line = "2026-10-23 00:21:26.686 W kv tier: RAM tier 0 MiB (the automatic size would be 10240 MiB): " +
                                     startfit::pinnedWhy(4096, 2) + off;
            CHECK(!std::regex_search(line, bad));
        }
    }
    // REVIEW-S8-719-2 independent tests (scratch\s8_719\review2\indep_test.cpp, written from the problem statement), ported:
    // measured inputs of fixb_case_g23_swift 00:21:26 (8060S 96 GB + R9700) and the 64 GB / 64/64 configurations
    {
        const auto two96 = [&](std::uint64_t own_r, std::uint64_t own_i, std::uint64_t pmax) {
            startfit::PinnedState s;
            s.total_phys = 33982058496ull;
            s.pinned_max = pmax * MiB;
            s.gpus = {{0, own_i * MiB, 16203 * MiB}, {15435 * MiB, own_r * MiB, 0}};
            return s;
        };
        const auto fit = [&](std::uint64_t mb, std::uint64_t avail_mb, bool expl, std::uint64_t room, std::uint64_t entry) {
            const auto r = startfit::ramTierFit(mb, avail_mb * MiB, 0, expl, room, entry);
            return r ? r->mb : mb;
        };
        const std::uint64_t room2 = startfit::pinnedRoomMb(two96(1294, 1381, 1453));
        // T1/T1b/T1c: explicit 8192 lowered, > 0, at most the measured-working 4096 (also with own usage reported doubled)
        CHECK(fit(8192, 19032, true, room2, 0) > 0ull && fit(8192, 19032, true, room2, 0) <= 4096ull);
        CHECK(fit(8192, 19032, true, startfit::pinnedRoomMb(two96(2588, 1381, 1453)), 0) <= 4096ull);
        // T2 (lead judgment): explicit 8192 with entry 10240 -> 4096, not off; T2b: automatic one entry (10240) -> off
        CHECK_EQ(fit(8192, 19032, true, room2, 10240), 4096ull);
        {
            const auto a = startfit::ramTierFit(10240, 19032 * MiB, 0, false, room2, 10240);
            CHECK(a.has_value() && a->mb == 0ull && a->below_entry);
        }
        // T3: explicit 4096 on two GPUs kept
        CHECK_EQ(fit(4096, 19032, true, room2, 0), 4096ull);
        // T4/T5/T5b: 8060S only visible: 8192 kept, 12288 -> 8704 by the half rule (not the pinned room), also with entry 10240
        startfit::PinnedState one;
        one.total_phys = 33982058496ull;
        one.pinned_max = 1453 * MiB;
        one.gpus = {{0, 1381 * MiB, 16203 * MiB}};
        const std::uint64_t room1 = startfit::pinnedRoomMb(one);
        CHECK_EQ(fit(8192, 17643, true, room1, 0), 8192ull);
        {
            const auto r = startfit::ramTierFit(12288, 17643 * MiB, 0, true, room1, 0);
            CHECK(r.has_value() && r->mb == 8704ull && r->half && !r->pinned);
        }
        CHECK_EQ(fit(8192, 17643, true, room1, 10240), 8192ull);
        // T6/T6b/T6c: single R9700 (64 GB host: auto 16384, explicit 20480; 32 GB desktop: auto 10240) kept
        startfit::PinnedState r64;
        r64.total_phys = 65174 * MiB;
        r64.pinned_max = 1500 * MiB;
        r64.gpus = {{48026 * MiB, 1294 * MiB, 0}};
        CHECK_EQ(fit(16384, 40000, false, startfit::pinnedRoomMb(r64), 9216), 16384ull);
        CHECK_EQ(fit(20480, 50000, true, startfit::pinnedRoomMb(r64), 9216), 20480ull);
        startfit::PinnedState d32;
        d32.total_phys = 32407 * MiB;
        d32.pinned_max = (1294 + 600) * MiB;
        d32.gpus = {{15435 * MiB, 1294 * MiB, 0}};
        CHECK_EQ(fit(10240, 22000, false, startfit::pinnedRoomMb(d32), 10240), 10240ull);
        // T7/T7b: single 8060S 64 GB: auto 10240 and golden 8192 kept
        startfit::PinnedState i64;
        i64.total_phys = 65174 * MiB;
        i64.pinned_max = 1453 * MiB;
        i64.gpus = {{0, 1381 * MiB, 32587 * MiB}};
        CHECK_EQ(fit(10240, 40000, false, startfit::pinnedRoomMb(i64), 10240), 10240ull);
        CHECK_EQ(fit(8192, 40000, true, startfit::pinnedRoomMb(i64), 10240), 8192ull);
        // T8/T8b/T8c: 64/64 two GPUs: golden 8192, auto 16384, explicit 20480 kept (T8d, the 197 MiB margin, is noted only)
        startfit::PinnedState t64 = i64;
        t64.gpus.push_back({48026 * MiB, 1294 * MiB, 0});
        const std::uint64_t rt = startfit::pinnedRoomMb(t64);
        CHECK_EQ(fit(8192, 40000, true, rt, 10240), 8192ull);
        CHECK_EQ(fit(16384, 40000, false, rt, 9216), 16384ull);
        CHECK_EQ(fit(20480, 50000, true, rt, 9216), 20480ull);
        // T10: explicit sizes monotonic at the measured two-GPU room (512..24576)
        std::uint64_t prev = 0;
        bool mono = true;
        for (std::uint64_t ask = 512; ask <= 24576; ask += 512) {
            const std::uint64_t got = fit(ask, 19032, true, room2, 10240);
            if (got < prev) mono = false;
            prev = got;
        }
        CHECK(mono);
        // T11/T11b: nothing known -> room unknown, explicit 8192 unchanged
        CHECK_EQ(startfit::pinnedRoomMb(startfit::PinnedState{}), startfit::pinned_room_unknown);
        CHECK(!startfit::ramTierFit(8192, 40000 * MiB, 0, true, startfit::pinned_room_unknown, 10240).has_value());
    }
    // 2b. slots after the weights: the C5 case (15 slots left 13312 tokens for a 16384-token request) -> fewer slots
    {
        const std::uint64_t per_tok = 36966, per_slot = 380 * MiB, shared = 556 * MiB;
        const std::uint64_t avail = shared + 15 * per_slot + 13312 * per_tok;  // 15 slots: exactly 13312 tokens
        const std::uint32_t p = startfit::slotsThatFit(15, avail, per_slot, shared, per_tok, 16384);
        CHECK(p < 15u && p >= 1u);
        CHECK(avail - shared - per_slot * p >= 16384 * per_tok);
        CHECK(avail - shared - per_slot * (p + 1) < 16384 * per_tok);  // the least reduction
        CHECK_EQ(startfit::slotsThatFit(4, 40 * GiB, per_slot, shared, per_tok, 16384), 4u);  // fits: unchanged
        CHECK_EQ(startfit::slotsThatFit(8, 1 * GiB, per_slot, shared, per_tok, 16384), 1u);   // never below 1
    }
    // the startup JSON (REVIEW-0221 A1: kv_ram_mb is the RAM tier in use)
    {
        startfit::StartupInfo si;
        si.ctx_per_request = 13312;
        si.pool_tokens = 13312;
        si.parallel = 15;
        si.kv_ram_mb = 18432;
        si.wddm_margin_mib = 9468;
        si.other_processes_mib = 7412;
        si.display_attached = false;
        si.adjusted = {"--parallel 16 -> 15", "say \"hi\" \\ ok"};
        CHECK_EQ(startfit::startupJson(si),
                 std::string("{\"ctx_per_request\":13312,\"pool_tokens\":13312,\"parallel\":15,\"kv_ram_mb\":18432,\"wddm_margin_mib\":9468,"
                             "\"other_processes_mib\":7412,\"display_attached\":false,\"adjusted\":[\"--parallel 16 -> 15\",\"say \\\"hi\\\" "
                             "\\\\ ok\"]}"));
        CHECK_EQ(startfit::startupJson(startfit::StartupInfo{}),
                 std::string("{\"ctx_per_request\":0,\"pool_tokens\":0,\"parallel\":0,\"kv_ram_mb\":0,\"wddm_margin_mib\":0,"
                             "\"other_processes_mib\":0,\"display_attached\":false,\"adjusted\":[]}"));
        // bug-gatenul: control bytes in an "adjusted" entry (a newline, a tab, a NUL) must be escaped
        // (RFC 8259 §7) so the JSON stays valid and parses back to the same text
        startfit::StartupInfo sc;
        sc.adjusted = {std::string("a\nb\tc\0d\x1f", 8)};
        const std::string jc = startfit::startupJson(sc);
        CHECK(jc.find("\"adjusted\":[\"a\\nb\\tc\\u0000d\\u001f\"]") != std::string::npos);
        bool parsed = false;
        try {
            const json::Value pv = json::parse(jc);
            parsed = pv.get("adjusted") && pv.get("adjusted")->asArray().size() == 1 &&
                     pv.get("adjusted")->asArray()[0].asString() == sc.adjusted[0];
        } catch (const std::exception&) {
        }
        CHECK(parsed);
    }
    // bug-gatenul guard: whirl-server-gate's results.json joins reasoning / content / tool calls with NUL
    // (ChatResult::text()); the writer escapes it as \u0000 and the --compare reader gets the same bytes back
    {
        const std::string t = std::string("think") + '\0' + "answer" + '\0' + "f(x);";
        json::Object o;
        o.set("basic/en_nothink", json::Value(t));
        const std::string d = json::dumpPython(json::Value(o));
        CHECK(d.find("think\\u0000answer\\u0000f(x);") != std::string::npos);
        CHECK_EQ(json::parse(d).asObject().find("basic/en_nothink")->asString(), t);
    }
    // a Swift-like card: 16 GiB weights, 0.75 GiB fixed per slot, 36 KiB / token, ~11 GiB left with 1 slot
    vram::FitInput fi;
    fi.avail = 28 * GiB;
    fi.weights = 16 * GiB;
    fi.fixed = 512 * MiB;
    fi.buf_per_row = 64 * 1024;
    fi.buf_fixed = 256 * MiB;
    fi.ckpt_bytes = 150 * MiB;
    fi.kv_per_token = 36966;
    fi.pool_min_tokens = 16384;
    fi.batch = 4096;
    fi.n_ck = 4;
    fi.n_spe = 2;
    const std::uint64_t per_slot = 150 * MiB;
    // 2. --parallel 4 fits: unchanged
    {
        vram::FitInput a = fi;
        a.parallel = 4;
        a.fixed += 4 * per_slot;
        const auto r = startfit::fitParallel(a, per_slot, true);
        CHECK_EQ(r.parallel, 4u);
        CHECK(r.plan.pool_tokens >= startfit::wantTokens(a));
    }
    // --parallel 16 on a card with less room (another program holds VRAM): lowered until one 16k request fits
    {
        vram::FitInput a = fi;
        a.avail = 24 * GiB;
        a.parallel = 16;
        a.fixed += 16 * per_slot;
        a.ck_fixed = true;  // keep the checkpoint plan out of it
        a.batch_fixed = true;
        const auto r = startfit::fitParallel(a, per_slot, true);
        CHECK(r.parallel < 16u && r.parallel >= 1u);
        CHECK(r.pool_requested < startfit::wantTokens(a));
        CHECK(r.plan.pool_tokens >= startfit::wantTokens(r.fi));
        // one slot more would not fit (the least reduction)
        vram::FitInput b = r.fi;
        b.parallel = r.parallel + 1;
        b.fixed += per_slot;
        CHECK(vram::planFit(b).pool_tokens < startfit::wantTokens(b));
        // not given (a default): never lowered here
        const auto d = startfit::fitParallel(a, per_slot, false);
        CHECK_EQ(d.parallel, 16u);
    }
    // even one slot does not fit: kept at 1, still short (the server then refuses at sizing)
    {
        vram::FitInput a = fi;
        a.avail = 17 * GiB;
        a.parallel = 4;
        a.fixed += 4 * per_slot;
        a.batch_fixed = a.ck_fixed = true;
        const auto r = startfit::fitParallel(a, per_slot, true);
        CHECK_EQ(r.parallel, 1u);
        CHECK(r.plan.pool_tokens < startfit::wantTokens(r.fi));
    }
    // 3. user-set prefill batch / checkpoints: lowered only when short, and only if it helps
    {
        vram::FitInput a = fi;
        a.avail = 16 * GiB + 1600 * MiB;  // tight
        a.parallel = 1;
        a.fixed += per_slot;
        a.batch_fixed = a.ck_fixed = true;
        const vram::FitPlan p = vram::planFit(a);
        CHECK(p.pool_tokens < startfit::wantTokens(a));
        const auto r = startfit::fitFixed(a, p);
        CHECK(r.changed);
        CHECK(r.batch_changed || r.ck_changed);
        CHECK(r.plan.pool_tokens > p.pool_tokens);
        CHECK(!r.fi.batch_fixed && !r.fi.ck_fixed);
        // with room: nothing changes
        vram::FitInput roomy = a;
        roomy.avail = 30 * GiB;
        const auto q = startfit::fitFixed(roomy, vram::planFit(roomy));
        CHECK(!q.changed);
        // nothing user-set: nothing to lower
        vram::FitInput autoset = a;
        autoset.batch_fixed = autoset.ck_fixed = false;
        CHECK(!startfit::fitFixed(autoset, vram::planFit(autoset)).changed);
    }
    // 4. / 5. -c and --ctx-per-slot: serverPoolFit as "not explicit" lowers instead of refusing
    {
        const std::uint64_t per_tok = 36966;
        const std::uint64_t avail = 11 * GiB;
        const auto refused = numerics::serverPoolFit(avail, per_tok, 400000u, true, 131072, true, 256, 8 * 262144);
        CHECK(refused.refuse);  // the old contract (still used where nothing is adjusted)
        const auto r = numerics::serverPoolFit(avail, per_tok, 400000u, false, 131072, false, 256, 8 * 262144);
        CHECK(!r.refuse);
        CHECK(r.pool < 400000u && r.pool >= 4096u && r.pool % 256 == 0);
        CHECK(static_cast<std::uint64_t>(r.pool) * per_tok <= avail);
        CHECK_EQ(r.slot_ctx, 131072u);  // still fits one 131072-token request
        const auto s = numerics::serverPoolFit(3 * GiB, per_tok, 400000u, false, 131072, false, 256, 8 * 262144);
        CHECK(!s.refuse);
        CHECK(s.slot_ctx == s.pool && s.slot_ctx < 131072u);  // context per request capped to the pool
        // below the minimum pool: refused with a clear message
        const auto z = numerics::serverPoolFit(100 * MiB, per_tok, 400000u, false, 131072, false, 256, 8 * 262144);
        CHECK(z.refuse);
        CHECK(z.msg.find("at least 4096 needed") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// arch024-B0 (0.2.4 bug-fix line): REVIEW-S8-719 C3 / C5, REVIEW-S8-719-3 N3-1 / N3-2 / N3-4 / N3-6 (+ the N3-5 blind spot).
// Each block is written from the review finding, before its fix (progress\arch024-B0.log has the before/after runs).
void testB0SizingFollowUps() {
    using namespace whirl;
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    const std::regex bad(R"(hipError|hip.*\b719\b|\b719\b.*hip|\| failed:|PAL failed|Failed PAL|result:-5|unspecified launch)",
                         std::regex::icase);
    // C5 (REVIEW-S8-719:77-80): a desktop R9700 host whose Ryzen iGPU (gfx1036, no WHIRL kernels) is a HIP GPU. What makes
    // n = 1 is restrictToSelected (gpu_restrict.cpp, since 0.2.3): HIP_VISIBLE_DEVICES names the chosen GPU alone, whichever
    // GPU that is (REVIEW-B0 K2); on this host the chosen one happens to be the R9700 because the iGPU has no kernels. So
    // the pinned room sees n = 1 and the 32 GB host keeps v0.2.2.2's automatic 8192 (C5 feared n = 2 -> about 4096). Either order.
    {
        using namespace whirl::gpusel;
        auto rec = [](int i, const char* name, const char* arch, int bus, bool integ, bool sup, std::uint64_t mib) {
            GpuRecord r;
            r.index = i;
            r.name = name;
            r.arch = arch;
            r.pci_bus = bus;
            r.luid = std::format("{:016x}", 0x2a000 + 0x1000 * i);
            r.integrated = integ;
            r.supported = sup;
            r.total_mib = mib;
            r.compute_units = integ ? 2 : 64;
            r.uuid = std::format("{:08x}-0000-0000-0000-{:012x}", 0xc0 + i, 0xfeed00 + bus);
            return r;
        };
        for (const bool igpu_first : {false, true}) {
            const int ri = igpu_first ? 1 : 0;
            std::vector<GpuRecord> g = {rec(ri, "AMD Radeon AI PRO R9700", "gfx1201", 0x03, false, true, 32624),
                                        rec(1 - ri, "AMD Radeon(TM) Graphics", "gfx1036", 0x7c, true, false, 16300)};
            if (igpu_first) std::swap(g[0], g[1]);
            CHECK_EQ(defaultGpu(g).index, ri);
            const std::vector<int> sel = selectGpus(g, "");
            CHECK(sel.size() == 1 && g[sel[0]].arch == "gfx1201");
            CHECK_EQ(visibleDevicesValue({g[sel[0]].index}), std::to_string(ri));  // one ordinal: HIP sees one GPU
        }
        startfit::PinnedState one;
        one.total_phys = 32407 * MiB;
        one.pinned_max = (1294 + 300) * MiB;
        one.gpus = {{15435 * MiB, 1294 * MiB, 16203 * MiB}};
        CHECK(!startfit::ramTierFit(8192, 20000 * MiB, 8192, false, startfit::pinnedRoomMb(one), 8192).has_value());
    }
    // REVIEW-B0 K2: the owner's host, 8060S + R9700 (scratch\s8_719\devices_all.txt: device 0 8060S gfx1151 integrated,
    // 95.8 GiB carve-out, pci:196:0; device 1 R9700 gfx1201 31.9 GiB, pci:71:0). The default is the 8060S (the most GPU
    // memory, owner 2026-10-23), --device pci:71:0 the R9700; either way ONE ordinal is made visible, and the enumeration
    // HIP then shows (the chosen GPU alone, as ordinal 0) passes restrictToSelected's identity check with n = 1. (The
    // runtime call hipGetDeviceCount itself is GPU-only: s8_719_root.ps1's "1 HIP GPU(s)" and release_gate srv_gpu_select.)
    {
        using namespace whirl::gpusel;
        auto rec = [](int i, const char* name, const char* arch, int bus, bool integ, std::uint64_t total, std::uint64_t fit,
                      int cu) {
            GpuRecord r;
            r.index = i;
            r.name = name;
            r.arch = arch;
            r.pci_bus = bus;
            r.luid = std::format("{:016x}", i == 0 ? 0x182faull : 0x1aa96ull);
            r.integrated = integ;
            r.supported = true;
            r.total_mib = total;
            r.fit_mib = fit;
            r.compute_units = cu;
            r.uuid = std::format("{:08x}-0000-0000-0000-{:012x}", 0xd0 + i, 0xbeef00 + bus);
            return r;
        };
        const std::vector<GpuRecord> host = {
            rec(0, "AMD Radeon(TM) 8060S Graphics", "gfx1151", 196, true, (958 + 158) * 1024 / 10, 958 * 1024 / 10, 20),
            rec(1, "AMD Radeon AI PRO R9700", "gfx1201", 71, false, 319 * 1024 / 10, 319 * 1024 / 10, 32)};
        CHECK_EQ(defaultGpu(host).index, 0);  // the 8060S, not the R9700
        for (const char* spec : {"", "pci:196:0", "pci:71:0"}) {
            const std::vector<int> sel = selectGpus(host, spec);
            CHECK_EQ(sel.size(), 1u);
            const GpuRecord& chosen = host[sel[0]];
            CHECK_EQ(chosen.arch, std::string(std::string_view(spec) == "pci:71:0" ? "gfx1201" : "gfx1151"));
            CHECK_EQ(visibleDevicesValue({chosen.index}), std::to_string(chosen.index));  // one ordinal
            GpuRecord seen = chosen;  // what HIP enumerates after HIP_VISIBLE_DEVICES: that GPU alone, renumbered 0
            seen.index = 0;
            const std::vector<GpuRecord> now = {seen};
            CHECK(now.size() == sel.size() && sameGpu(now[0], chosen));
            CHECK(!sameGpu(now[0], host[1 - sel[0]]));  // and the other GPU would not pass for it
        }
    }
    // C3 (REVIEW-S8-719:65-69): the per-GPU term must count every process's pinned memory on that adapter, not only this
    // process's. 8060S at 96 GB (32407 MiB RAM), this server on the R9700 alone (n = 1, non-local budget 15435), another
    // server holding 6000 MiB pinned on the R9700 adapter: the room + all of that adapter's Shared Usage + the reserve must
    // stay within the budget (the machine term, half of the RAM, is looser here by 768 MiB).
    {
        startfit::PinnedState s;
        s.total_phys = 32407 * MiB;
        s.pinned_max = (1294 + 6000) * MiB;
        s.gpus = {{15435 * MiB, 1294 * MiB, 0, (1294 + 6000) * MiB}};
        const std::uint64_t room = startfit::pinnedRoomMb(s);
        CHECK(room * MiB + (1294 + 6000) * MiB + startfit::pinned_reserve_mb * MiB <= 15435 * MiB);
        CHECK(room > 0ull);
        // the adapter's usage unknown (0): this process only, as in 0.2.3
        s.gpus = {{15435 * MiB, 1294 * MiB, 0, 0}};
        s.pinned_max = 1294 * MiB;
        CHECK_EQ(startfit::pinnedRoomMb(s), 11776ull);  // 15435 - 1294 - 2048 = 12093 -> 11776
        // integrated GPU (C4): a SharedSystemMemory below half of the RAM, other programs 6000 MiB on that adapter
        startfit::PinnedState i;
        i.total_phys = 65536 * MiB;
        i.pinned_max = (1441 + 6000) * MiB;
        i.gpus = {{0, 1441 * MiB, 16384 * MiB, (1441 + 6000) * MiB}};
        const std::uint64_t ri = startfit::pinnedRoomMb(i);
        CHECK(ri * MiB + (1441 + 6000) * MiB + startfit::pinned_reserve_mb * MiB <= 16384 * MiB);
    }
    // N3-1 (REVIEW-S8-719-3:55-57): an explicit size is monotonic in the request at every MiB, not only at multiples of 512
    // (fixb s23_ram12288_vis: 17643 MiB available, ask 8800 -> 8800 but 8822 -> 8704), and never above the ask
    for (const std::uint64_t avail_mb : {17643ull, 17000ull, 19032ull, 8191ull, 40001ull})
        for (const std::uint64_t room : {startfit::pinned_room_unknown, 4096ull, 12288ull}) {
            std::uint64_t prev = 0;
            bool mono = true, within = true;
            for (std::uint64_t ask = 1; ask <= 20480; ++ask) {
                const auto r = startfit::ramTierFit(ask, avail_mb * MiB, 0, true, room, 0);
                const std::uint64_t got = r ? r->mb : ask;
                if (got < prev) mono = false;
                if (got > ask) within = false;
                prev = got;
            }
            CHECK(mono);
            CHECK(within);
        }
    {
        const auto r = startfit::ramTierFit(12288, 17643 * MiB, 0, true);  // the measured case keeps its 8704
        CHECK(r.has_value() && r->mb == 8704ull && r->half);
    }
    // N3-4 (REVIEW-S8-719-3:61, REVIEW-S8-719 N2): pinned memory this process allocates after the measurement (the
    // discrete-GPU mmproj, 887 MiB in s23_ram8192_mm) must come out of the room, the same as if it were pinned already
    {
        startfit::PinnedState s;
        s.total_phys = 32407 * MiB;
        s.pinned_max = 1294 * MiB;
        s.gpus = {{15435 * MiB, 1294 * MiB, 0, 1294 * MiB}};
        startfit::PinnedState l = s;
        l.later = 887 * MiB;
        startfit::PinnedState now = s;
        now.pinned_max += 887 * MiB;
        now.gpus[0].own_nonlocal += 887 * MiB;
        now.gpus[0].adapter_shared += 887 * MiB;
        CHECK_EQ(startfit::pinnedRoomMb(l), startfit::pinnedRoomMb(now));
        CHECK(startfit::pinnedRoomMb(l) < startfit::pinnedRoomMb(s));
        startfit::PinnedState ig;  // the integrated GPU and the machine term too
        ig.total_phys = 32407 * MiB;
        ig.pinned_max = 1441 * MiB;
        ig.gpus = {{0, 1441 * MiB, 16203 * MiB, 1441 * MiB}};
        startfit::PinnedState igl = ig;
        igl.later = 887 * MiB;
        CHECK(startfit::pinnedRoomMb(igl) * MiB + (1441 + 887) * MiB + startfit::pinned_reserve_mb * MiB <= 16203 * MiB);
        // REVIEW-B0 N1: the mmproj file size is not an upper bound of what Vision::load pins. MEASURED: Swift's
        // mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf is 927607584 B (884.6 MiB) and pinned 887 MiB (REVIEW-S8-719-3 N3-4)
        constexpr std::uint64_t swift_mmproj = 927607584;
        CHECK(startfit::mmprojPinnedBytes(swift_mmproj) >= 887 * MiB);
        CHECK(startfit::mmprojPinnedBytes(swift_mmproj) <= swift_mmproj + 128 * MiB);  // a bound, not a guess of 2x
        CHECK(startfit::mmprojPinnedBytes(0) == 0);  // no mmproj: nothing later
    }
    // N3-2 (REVIEW-S8-719-3:59): when the half-of-available-RAM rule (not the pinned room) turns an explicit tier off, the
    // texts must not say "no room left"; they must name the half rule. A room of 0 still says "no room left".
    {
        const std::string pw = startfit::pinnedWhy(4096, 1);
        const auto h = startfit::ramTierFit(8192, 700 * MiB, 0, true, 4096, 0);  // room 4096, then half of 700 MiB -> 0
        CHECK(h.has_value() && h->mb == 0ull && h->half);
        const auto th = startfit::ramTierText(h, 8192, true, "--kv-ram-mb", "--kv-ram-mb 8192", pw, 0, 700);
        CHECK(th.adjusted_why.find("no room left") == std::string::npos);
        CHECK(th.off_reason.find("no room left") == std::string::npos);
        CHECK(th.off_reason.find("half") != std::string::npos);
        CHECK(th.off_reason.find("SSD") != std::string::npos);
        const auto h2 = startfit::ramTierFit(4096, 700 * MiB, 0, true);  // the half rule alone
        const auto th2 = startfit::ramTierText(h2, 4096, true, "--kv-ram-mb", "--kv-ram-mb 4096", "", 0, 700);
        CHECK(th2.off_reason.find("half") != std::string::npos);
        const auto z = startfit::ramTierFit(8192, 40000 * MiB, 0, true, 0, 0);  // room 0
        CHECK(z.has_value() && z->mb == 0ull && z->pinned);
        const auto tz = startfit::ramTierText(z, 8192, true, "--kv-ram-mb", "--kv-ram-mb 8192", startfit::pinnedWhy(0, 1), 0, 40000);
        CHECK(tz.off_reason.find("no room left") != std::string::npos);
    }
    // N3-6 (REVIEW-S8-719-3:63): "kv tier: RAM tier size 4096 MiB (WHIRL_KV_RAM_MB 8192)" shows the asked size next to the
    // clamped one; a lowered size must say it was lowered (explicit and automatic). Unchanged: the reason as before.
    {
        const auto e = startfit::ramTierFit(8192, 19374 * MiB, 0, true, 4096, 9216);
        const auto te = startfit::ramTierText(e, 8192, true, "WHIRL_KV_RAM_MB", "WHIRL_KV_RAM_MB 8192", startfit::pinnedWhy(4096, 1), 9216, 19374);
        CHECK(te.size_reason.find("lowered") != std::string::npos);
        const auto a = startfit::ramTierFit(16384, 30 * GiB, 16384, false, 12288, 9216);
        CHECK(a.has_value() && a->mb == 12288ull);
        const auto ta = startfit::ramTierText(a, 16384, false, "--kv-ram-mb", "auto: 1/4 of 64 GiB RAM", startfit::pinnedWhy(12288, 1), 9216, 30720);
        CHECK(ta.size_reason.find("lowered") != std::string::npos);
        const auto tu = startfit::ramTierText(std::nullopt, 8192, true, "--kv-ram-mb", "--kv-ram-mb 8192", "", 9216, 30720);
        CHECK_EQ(tu.size_reason, std::string("--kv-ram-mb 8192"));
        CHECK_EQ(tu.off_reason, std::string("--kv-ram-mb 8192"));
    }
    // N3-5 blind spot: every line form the server builds from these texts ("adjusted:", "RAM tier off (...)", "RAM tier
    // size N MiB (...)", the automatic W line) matches none of the log-scan error patterns, over a grid of fits
    {
        int lines = 0;
        bool clean = true;
        for (const std::uint64_t ask : {0ull, 4096ull, 8192ull, 8822ull, 24064ull})
            for (const std::uint64_t avail : {700ull, 17643ull, 50000ull})
                for (const std::uint64_t room : {startfit::pinned_room_unknown, 0ull, 4096ull})
                    for (const bool ex : {true, false}) {
                        const auto f = startfit::ramTierFit(ask, avail * MiB, ask, ex, room, 9216);
                        const auto t = startfit::ramTierText(f, ask, ex, "--kv-ram-mb", ex ? "--kv-ram-mb N" : "auto: 1/4 of 64 GiB RAM",
                                                             room == startfit::pinned_room_unknown ? "" : startfit::pinnedWhy(room, 1), 9216, avail);
                        const std::uint64_t mb = f ? f->mb : ask;
                        for (const std::string& l :
                             {std::format("2026-10-23 18:00:00.000 W adjusted: --kv-ram-mb {} -> {} MiB ({})", ask, mb, t.adjusted_why),
                              std::format("2026-10-23 18:00:00.000 I kv tier: RAM tier off ({})", t.off_reason),
                              std::format("2026-10-23 18:00:00.000 I kv tier: RAM tier size {} MiB ({})", mb, t.size_reason),
                              std::format("2026-10-23 18:00:00.000 W kv tier: RAM tier {} MiB (the automatic size would be {} MiB): {}", mb, ask,
                                          t.pinned_why)}) {
                            ++lines;
                            if (std::regex_search(l, bad)) clean = false;
                        }
                    }
        CHECK(clean);
        CHECK(lines > 0);
    }
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
// arch025-N10c (lane 1 leg 5): ServerRun::planSlots' slot re-check arithmetic (2d166f4 server_main.cpp:954-969: the
// fixed VRAM per slot and shared, and the pool a slot count leaves) -> startfit::slotFixed / poolTokensWith; the
// replay rows are model::slotStateBytes' pend1 (the same expression; whirl-server-tests slot_replay_rows). References = the old text. Written before the
// module functions existed.
void testSlotFixed() {
    using namespace whirl;
    constexpr std::uint64_t MiB = 1ull << 20;
    std::size_t n = 0, bad = 0;
    for (std::uint64_t st : {0ull, 37 * MiB + 5, 150 * MiB})
        for (std::uint64_t pend : {0ull, 9 * MiB + 77})
            for (std::uint64_t n_ck : {0ull, 1ull, 3ull})
                for (std::uint64_t ck : {0ull, 39 * MiB + 1})
                    for (bool use_dfl : {false, true})
                        for (std::uint64_t dfl : {0ull, 300 * MiB + 3})
                            for (std::uint32_t n_spe : {0u, 2u})
                                for (std::uint32_t snap : {0u, 1u, 4u}) {
                                    startfit::SlotFixedIn in;
                                    in.state = st;
                                    in.replay = pend;
                                    in.n_ck = n_ck;
                                    in.ck = ck;
                                    in.dflash_slot = use_dfl ? dfl : 0;
                                    in.n_spe = n_spe;
                                    in.snap_sets = snap;
                                    const startfit::SlotFixed r = startfit::slotFixed(in);
                                    const std::uint64_t per_slot = (st + pend + n_ck * ck + (use_dfl ? dfl : 0)) * 104 / 100;
                                    const std::uint64_t shared =
                                        (n_spe * ck + static_cast<std::uint64_t>(snap) * st) * 104 / 100 + (256ull << 20);
                                    ++n;
                                    if (r.per_slot != per_slot || r.shared != shared) ++bad;
                                }
    std::size_t m = 0;
    for (std::uint64_t av : {0ull, 300 * MiB, 9000 * MiB, 31000 * MiB})
        for (std::uint64_t per_slot : {0ull, 200 * MiB + 9})
            for (std::uint64_t shared : {256 * MiB, 2000 * MiB})
                for (std::uint64_t per_tok : {0ull, 22528ull, 65536ull})
                    for (std::uint32_t p : {1u, 2u, 7u, 16u}) {
                        const std::uint64_t fixed = shared + per_slot * p;
                        const std::uint64_t old = av > fixed && per_tok ? (av - fixed) / per_tok : 0;
                        ++m;
                        if (startfit::poolTokensWith(p, av, per_slot, shared, per_tok) != old) ++bad;
                    }
    std::printf("slot fixed: %zu + %zu cases vs the old planSlots text, %zu differ\n", n, m, bad);
    CHECK(n == 864 && m == 192);
    CHECK_EQ(bad, std::size_t{0});
}

// arch025-N10b (lane 1 leg 5): the VRAM the KV pool may use was computed three times in server_main.cpp (buildModel's
// pre-load fit, ServerRun::planSlots' slot re-check, ServerRun::planPool's pool sizing with its "kv pool sizing:" note).
// poolbudget::kvRoom / kvRoomNote / defaultReserveMb are the one copy; the references below are the three old texts
// (2d166f4 server_main.cpp:512-517, :949-953, :1293-1327, values passed in instead of read), compared over a grid.
// Written before the module function existed.
void testKvRoom() {
    using namespace whirl;
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    // the old reserve default (the three copies wrote the same expression)
    for (bool moe : {false, true})
        for (bool ok : {false, true}) CHECK_EQ(poolbudget::defaultReserveMb(moe, ok), moe ? 1536u : (ok ? 768u : 3072u));
    // the old pre-load / slot re-check rule (copies 1 and 2: no vision reserve, no note)
    const auto old12 = [](std::uint64_t free, std::uint64_t res, bool res_explicit, bool w_ok, std::uint64_t budget,
                          std::uint64_t usage, std::uint64_t margin, std::uint64_t headroom) {
        std::uint64_t av = free > res ? free - res : 0;
        if (!res_explicit && w_ok) av = std::min(av, budget > usage + margin ? budget - usage - margin : 0);
        av = av > headroom ? av - headroom : 0;
        return av;
    };
    // the old pool-sizing lambda with its note (copy 3)
    const auto old3 = [](std::uint64_t free, std::uint64_t reserve, bool reserve_explicit, bool w_ok, std::uint64_t budget,
                         std::uint64_t usage, std::uint64_t budget_margin, std::uint64_t headroom, std::uint64_t vis_reserve,
                         const std::string& why, const std::string& vis_reserve_note, std::string& budget_note) {
        std::uint64_t a = free > reserve ? free - reserve : 0;
        if (!reserve_explicit && w_ok) {
            const std::uint64_t room = budget > usage + budget_margin ? budget - usage - budget_margin : 0;
            budget_note = std::format("WDDM budget {:.2f} GiB, used {:.2f} GiB, margin {} MiB ({}) -> room {:.2f} GiB; free "
                                      "VRAM {:.2f} GiB minus {} MiB -> {:.2f} GiB",
                                      budget / 1073741824.0, usage / 1073741824.0, budget_margin >> 20, why, room / 1073741824.0,
                                      free / 1073741824.0, reserve >> 20, a / 1073741824.0);
            a = std::min(a, room);
        } else if (reserve_explicit) {
            budget_note = std::format("WHIRL_POOL_RESERVE_MB={}: free VRAM {:.2f} GiB minus the reserve, WDDM budget not applied",
                                      reserve >> 20, free / 1073741824.0);
        } else {
            budget_note = std::format("WDDM budget unavailable: free VRAM {:.2f} GiB minus {} MiB", free / 1073741824.0, reserve >> 20);
        }
        if (headroom > 0) {
            budget_note += std::format("; minus {} MiB desktop headroom (WHIRL_VRAM_HEADROOM_MB) -> {:.2f} GiB", headroom >> 20,
                                       (a > headroom ? a - headroom : 0) / 1073741824.0);
            a = a > headroom ? a - headroom : 0;
        }
        if (vis_reserve > 0 && !reserve_explicit) {
            a = poolbudget::kvAvail(a, vis_reserve);
            budget_note += std::format("; minus {} -> {:.2f} GiB", vis_reserve_note, a / 1073741824.0);
        }
        return a;
    };
    const std::string why = "pending 0 MiB + room 2048 MiB (floor)";
    const std::string vnote = "1007 MiB for the vision encoder (--mmproj)";
    std::size_t n = 0, bad = 0;
    for (std::uint64_t free : {0ull, 1 * GiB, 20 * GiB + 123 * MiB, 31 * GiB})
        for (std::uint64_t res : {768 * MiB, 3072 * MiB, 0ull})
            for (bool res_explicit : {false, true})
                for (bool w_ok : {false, true})
                    for (std::uint64_t budget : {0ull, 31767 * MiB})
                        for (std::uint64_t usage : {0ull, 29696 * MiB, 40 * GiB})
                            for (std::uint64_t margin : {0ull, 2068 * MiB})
                                for (std::uint64_t headroom : {0ull, 512 * MiB, 40 * GiB})
                                    for (std::uint64_t vis : {0ull, 1007 * MiB}) {
                                        poolbudget::RoomIn in;
                                        in.free = free;
                                        in.reserve = res;
                                        in.reserve_explicit = res_explicit;
                                        in.wddm_ok = w_ok;
                                        in.local_budget = budget;
                                        in.local_usage = usage;
                                        in.margin = margin;
                                        in.headroom = headroom;
                                        in.vis_reserve = vis;
                                        const poolbudget::Room r = poolbudget::kvRoom(in);
                                        std::string note_old;
                                        const std::uint64_t a3 =
                                            old3(free, res, res_explicit, w_ok, budget, usage, margin, headroom, vis, why, vnote, note_old);
                                        const std::string note_new = poolbudget::kvRoomNote(in, r, why, vnote);
                                        ++n;
                                        if (r.bytes != a3 || note_new != note_old) ++bad;
                                        if (vis == 0) {
                                            ++n;
                                            if (r.bytes != old12(free, res, res_explicit, w_ok, budget, usage, margin, headroom)) ++bad;
                                        }
                                    }
    std::printf("kv room: %zu cases vs the three old server_main copies, %zu differ\n", n, bad);
    CHECK(n == 3456 + 1728);
    CHECK_EQ(bad, std::size_t{0});
    // control: the comparison is live (a changed headroom rule would be seen)
    {
        poolbudget::RoomIn in;
        in.free = 20 * GiB;
        in.reserve = 768 * MiB;
        in.headroom = 512 * MiB;
        CHECK_EQ(poolbudget::kvRoom(in).bytes, old12(20 * GiB, 768 * MiB, false, false, 0, 0, 0, 512 * MiB));
        CHECK(poolbudget::kvRoom(in).bytes != old12(20 * GiB, 768 * MiB, false, false, 0, 0, 0, 0));
    }
}

// arch025-N10 (lane 1 leg 6): the pre-load estimate's adjust / log section of buildModel (536d470 server_main.cpp:560-629:
// the -c re-plan, fitParallel, fitFixed, the three "adjusted:" texts, the "VRAM is tight" warning, the VramLimit refusal
// text and the "may not fit" warning) -> startfit::preloadAdjust. The reference below is that old text with the
// measured values passed in, adjust() pushing into a list and logW / throw replaced by strings. Written before the
// module function existed.
namespace leg6 {
struct OldPreload {
    std::uint32_t parallel = 1;
    whirl::vram::FitInput fi, fitp, fitf;
    whirl::vram::FitPlan plan, fitf_plan;
    std::uint64_t per_slot = 0;
    std::vector<std::pair<std::string, std::string>> adjusted;
    std::string tight, refusal, short_warn;
    bool refuse = false;
};
inline OldPreload oldPreload(whirl::vram::FitInput fi, whirl::vram::FitPlan plan, std::uint64_t st1, std::uint64_t pend1,
                             std::uint32_t parallel, bool parallel_given, bool ctx_given, std::uint32_t slot_ctx,
                             std::uint32_t vram_tight_pool, std::uint64_t av, std::uint64_t margin, std::uint64_t pending,
                             std::uint64_t room, std::uint64_t weights, std::uint64_t total) {
    using namespace whirl;
    OldPreload o;
    const auto adjust = [&o](const std::string& what, const std::string& why) { o.adjusted.emplace_back(what, why); };
    const double g = 1024.0 * 1024.0 * 1024.0;
    {
        const auto want = [&] { return startfit::wantTokens(fi); };
        const std::string budget_why =
            std::format("{:.2f} GiB usable under the Windows video memory budget, keeping {:.2f} GiB free for other programs "
                        "({:.2f} GiB they use that the budget does not show yet + {:.2f} GiB room)",
                        av / g, margin / g, pending / g, room / g);
        if (ctx_given && plan.pool_tokens < want()) {
            fi.pool_min_tokens = std::min<std::uint32_t>(slot_ctx, vram_tight_pool);
            plan = vram::planFit(fi);
        }
        const std::uint32_t par0 = parallel;
        o.fitp = fi;
        o.per_slot = (st1 + pend1) + (st1 + pend1) / 25;
        const startfit::ParallelFit pfit = startfit::fitParallel(fi, (st1 + pend1) + (st1 + pend1) / 25, parallel_given);
        if (pfit.parallel != par0) {
            parallel = pfit.parallel;
            fi = pfit.fi;
            plan = pfit.plan;
            adjust(std::format("--parallel {} -> {}", par0, parallel),
                   std::format("with {} slots, their recurrent state and prefix checkpoints leave a KV pool of about {} tokens, "
                               "less than one {}-token request; {}",
                               par0, pfit.pool_requested, want(), budget_why));
        }
        o.fitf = fi;
        o.fitf_plan = plan;
        const startfit::FixedFit ffit = startfit::fitFixed(fi, plan);
        if (ffit.changed) {
            if (ffit.batch_changed)
                adjust(std::format("WHIRL_PREFILL_BATCH {} -> {}", plan.batch, ffit.plan.batch),
                       std::format("the prefill buffers left a KV pool of about {} tokens, less than {}; {}", plan.pool_tokens,
                                   want(), budget_why));
            if (ffit.ck_changed)
                adjust(std::format("prefix checkpoints {} per slot + {} shared -> {} + {} (WHIRL_SERVE_CKPTS / WHIRL_SYS_CKPTS)",
                                   plan.n_ck, plan.n_spe, ffit.plan.n_ck, ffit.plan.n_spe),
                       std::format("they left a KV pool of about {} tokens, less than {}; {}", plan.pool_tokens, want(),
                                   budget_why));
            fi = ffit.fi;
            plan = ffit.plan;
        }
    }
    if (plan.reduced)
        o.tight = std::format("VRAM is tight (about {:.2f} GiB usable for weights {:.2f} GiB + buffers + checkpoints + KV): reduced {} so the KV "
                              "pool gets about {} tokens (wanted {}); WHIRL_PREFILL_BATCH (at least 1280) / WHIRL_SERVE_CKPTS / WHIRL_SYS_CKPTS "
                              "choose these yourself, --ctx a smaller pool",
                              av / g, weights / g, plan.note, plan.pool_tokens, fi.pool_min_tokens);
    if (!plan.fits && plan.pool_tokens == 0) {
        o.refuse = true;
        o.refusal = std::format("needs about {:.2f} GiB of VRAM before any KV cache (weights {:.2f} GiB + prefill batch {}, "
                                "state, checkpoints), about {:.2f} GiB usable of {:.2f} GiB{}",
                                plan.need_fixed / g, weights / g, plan.batch, av / g, total / g,
                                vram::limitBytes() ? " (" + vram::limitNote() + ")" : std::string());
    }
    if (!plan.fits)
        o.short_warn = std::format("VRAM estimate: weights + buffers + checkpoints {:.2f} GiB leave about {} KV tokens of {:.2f} GiB usable "
                                   "(at least {} needed); the model may not fit this GPU",
                                   plan.need_fixed / g, plan.pool_tokens, av / g, fi.pool_hard_min);
    o.parallel = parallel;
    o.fi = fi;
    o.plan = plan;
    return o;
}
inline bool sameFi(const whirl::vram::FitInput& a, const whirl::vram::FitInput& b) {
    return a.avail == b.avail && a.weights == b.weights && a.fixed == b.fixed && a.buf_per_row == b.buf_per_row &&
           a.buf_fixed == b.buf_fixed && a.ckpt_bytes == b.ckpt_bytes && a.parallel == b.parallel && a.kv_per_token == b.kv_per_token &&
           a.pool_min_tokens == b.pool_min_tokens && a.pool_hard_min == b.pool_hard_min && a.batch == b.batch && a.n_ck == b.n_ck &&
           a.n_spe == b.n_spe && a.batch_fixed == b.batch_fixed && a.ck_fixed == b.ck_fixed;
}
inline bool samePlan(const whirl::vram::FitPlan& a, const whirl::vram::FitPlan& b) {
    return a.batch == b.batch && a.n_ck == b.n_ck && a.n_spe == b.n_spe && a.need_fixed == b.need_fixed && a.pool_tokens == b.pool_tokens &&
           a.reduced == b.reduced && a.fits == b.fits && a.note == b.note;
}
}  // namespace leg6

void testPreloadAdjust() {
    using namespace whirl;
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    std::size_t n = 0, bad = 0, n_adj = 0, n_tight = 0, n_refuse = 0, n_short = 0;
    for (std::uint64_t av : {6 * GiB, 14 * GiB + 300 * MiB, 21 * GiB, 29 * GiB})
        for (std::uint64_t weights : {7 * GiB, 15 * GiB + 77 * MiB, 18 * GiB})
            for (std::uint32_t parallel : {1u, 4u})
                for (bool given : {false, true})
                    for (bool ctx_given : {false, true})
                        for (std::uint32_t pool_min : {16384u, 262144u})
                            for (bool batch_fixed : {false, true})
                                for (bool ck_fixed : {false, true})
                                    for (std::uint32_t n_ck : {0u, 3u}) {
                                        vram::FitInput fi;
                                        fi.avail = av;
                                        fi.weights = weights;
                                        fi.fixed = 600 * MiB + 13;
                                        fi.buf_per_row = 120 * 1024 + 7;
                                        fi.buf_fixed = 300 * MiB;
                                        fi.ckpt_bytes = 150 * MiB + 600 * 1024;
                                        fi.parallel = parallel;
                                        fi.kv_per_token = 32768;
                                        fi.pool_min_tokens = pool_min;
                                        fi.batch = 4096;
                                        fi.n_ck = n_ck;
                                        fi.n_spe = n_ck ? 2 : 0;
                                        fi.batch_fixed = batch_fixed;
                                        fi.ck_fixed = ck_fixed;
                                        const vram::FitPlan plan = vram::planFit(fi);
                                        const std::uint64_t st1 = 149 * MiB + 5, pend1 = 37 * MiB + 1;
                                        const leg6::OldPreload o = leg6::oldPreload(fi, plan, st1, pend1, parallel, given, ctx_given, 131072,
                                                                                    16384, av, 2068 * MiB, 300 * MiB, 1768 * MiB, weights,
                                                                                    32 * GiB);
                                        startfit::PreloadIn in;
                                        in.fi = fi;
                                        in.plan = plan;
                                        in.st1 = st1;
                                        in.pend1 = pend1;
                                        in.parallel = parallel;
                                        in.parallel_given = given;
                                        in.ctx_given = ctx_given;
                                        in.slot_ctx = 131072;
                                        in.tight_pool = 16384;
                                        in.avail = av;
                                        in.margin = 2068 * MiB;
                                        in.pending = 300 * MiB;
                                        in.room = 1768 * MiB;
                                        in.weights = weights;
                                        in.total = 32 * GiB;
                                        const startfit::PreloadAdjust r = startfit::preloadAdjust(in);
                                        bool same = r.parallel == o.parallel && leg6::sameFi(r.fi, o.fi) && leg6::samePlan(r.plan, o.plan) &&
                                                    leg6::sameFi(r.fitp, o.fitp) && r.per_slot == o.per_slot && leg6::sameFi(r.fitf, o.fitf) &&
                                                    leg6::samePlan(r.fitf_plan, o.fitf_plan) && r.tight == o.tight && r.refuse == o.refuse &&
                                                    r.refusal == o.refusal && r.short_warn == o.short_warn && r.adjusted.size() == o.adjusted.size();
                                        for (std::size_t i = 0; same && i < r.adjusted.size(); ++i)
                                            same = r.adjusted[i].what == o.adjusted[i].first && r.adjusted[i].why == o.adjusted[i].second;
                                        ++n;
                                        if (!same) ++bad;
                                        n_adj += o.adjusted.empty() ? 0 : 1;
                                        n_tight += o.tight.empty() ? 0 : 1;
                                        n_refuse += o.refuse ? 1 : 0;
                                        n_short += o.short_warn.empty() ? 0 : 1;
                                    }
    std::printf("preload adjust: %zu cases vs the old buildModel text (%zu adjusted, %zu tight, %zu refused, %zu short), %zu differ\n", n,
                n_adj, n_tight, n_refuse, n_short, bad);
    CHECK_EQ(n, std::size_t{1536});
    CHECK(n_adj > 0 && n_tight > 0 && n_refuse > 0 && n_short > 0);  // every branch is exercised
    CHECK_EQ(bad, std::size_t{0});
}

// arch025-N10 (lane 1 leg 6): the UMA carve-out check's fixed-memory formula (536d470 server_main.cpp:643-654) ->
// startfit::umaFixed; the reference is the old text with the sizing results passed in.
void testUmaFixed() {
    using namespace whirl;
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    std::size_t n = 0, bad = 0;
    for (std::uint64_t w : {7 * GiB + 3, 20 * GiB})
        for (std::uint64_t state : {0ull, 900 * MiB + 11})
            for (std::uint64_t bufs : {300 * MiB, 2 * GiB + 5})
                for (std::uint64_t st1 : {0ull, 149 * MiB + 5})
                    for (std::uint64_t pend1 : {0ull, 37 * MiB + 1})
                        for (std::uint32_t drafts : {0u, 1u, 8u})
                            for (std::uint32_t par : {1u, 4u})
                                for (std::size_t n_ck : {std::size_t{0}, std::size_t{3}})
                                    for (std::uint32_t n_spe : {0u, 2u})
                                        for (bool has_mtp : {false, true})
                                            for (std::uint32_t n_embd : {2048u, 5120u}) {
                                                const std::uint32_t n_vocab = 248320;
                                                // old text
                                                const std::uint64_t ck = st1 + static_cast<std::uint64_t>(n_embd) * 4 + static_cast<std::uint64_t>(n_vocab) * 4;
                                                const std::uint64_t snaps = std::max<std::uint32_t>(1, drafts) * st1;
                                                const std::uint64_t slots_b = par * (st1 + pend1) + snaps;
                                                const std::uint64_t ck_b = (par * n_ck + n_spe) * ck;
                                                std::uint64_t fixed = w + state + bufs + slots_b + ck_b + (has_mtp ? 65536ull * n_embd * 5 / 16 : 0) + (256ull << 20);
                                                fixed += fixed / 12;
                                                startfit::UmaFixedIn in;
                                                in.weights = w;
                                                in.state = state;
                                                in.bufs = bufs;
                                                in.st1 = st1;
                                                in.pend1 = pend1;
                                                in.drafts = drafts;
                                                in.parallel = par;
                                                in.n_ck = n_ck;
                                                in.n_spe = n_spe;
                                                in.has_mtp = has_mtp;
                                                in.n_embd = n_embd;
                                                in.n_vocab = n_vocab;
                                                const startfit::UmaFixed r = startfit::umaFixed(in);
                                                ++n;
                                                if (r.slots != slots_b || r.ck != ck_b || r.fixed != fixed) ++bad;
                                            }
    std::printf("uma fixed: %zu cases vs the old buildModel text, %zu differ\n", n, bad);
    CHECK_EQ(n, std::size_t{3072});
    CHECK_EQ(bad, std::size_t{0});
}

const whirl::test::Suite& unitSuite_mem_fit() {
    static const whirl::test::Suite s{"mem.fit",
                                      {
                                          {"testVramLimit", [] { testVramLimit(); }},
                                          {"testCardDefaults", [] { testCardDefaults(); }},
                                          {"testPoolBudget", [] { testPoolBudget(); }},
                                          {"testPoolMargin", [] { testPoolMargin(); }},
                                          {"testStartupFit", [] { testStartupFit(); }},
                                          {"testB0SizingFollowUps", [] { testB0SizingFollowUps(); }},
                                          {"testVisionPoolReserve", [] { testVisionPoolReserve(); }},
                                          {"testKvRoom", [] { testKvRoom(); }},
                                          {"testSlotFixed", [] { testSlotFixed(); }},
                                          {"testPreloadAdjust", [] { testPreloadAdjust(); }},
                                          {"testUmaFixed", [] { testUmaFixed(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_mem_fit()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
