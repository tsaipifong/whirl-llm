// WHIRL unit tests of module mem.fit (vram_limit.*) (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-mem-fit-tests and together in whirl-tests). arch025-iso4: the tests of the other modules that used to
// share this file (server.kv_fit: pool budget, startup fit, KV room, slot fixed ...) are tests/unit/server_kv_fit_tests.cpp, unchanged.

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/vram_limit.h"
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


}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_mem_fit() {
    static const whirl::test::Suite s{"mem.fit",
                                      {
                                          {"testVramLimit", [] { testVramLimit(); }},
                                          {"testCardDefaults", [] { testCardDefaults(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_mem_fit()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
