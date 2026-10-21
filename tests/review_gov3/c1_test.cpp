// REVIEW-GOV-3 independent test for REVIEW-GOV-2 C-1 (reviewer; written from the bug description only, before
// reading 719b81d's test or 2a64b86's fix):
//   "after a grow-back re-allocation the governor uses min(measured, expected) for 120 s and checks the episode end
//    first, so it can report back-to-startup while measured VRAM is over the budget and only act ~115-120 s later".
// CPU only: drives whirl::memgov::Governor through tick / resized like Engine::govApply (every slot idle, a planned
// re-allocation happens at once and is reported with expected = usage before + bytes added or freed).
// R9700 Swift numbers: budget 31767 MiB (31.02 GiB), margin 2068 MiB, startup local 29699 MiB (= budget - margin),
// pool 340480 tokens, 4 GiB hog. The grow-back costs `overshoot` more than expected (REVIEW-GOV-2 C-1: R2b_nolim1
// 19:14:48 read 32.50 GiB against 31.02 after a grow, then "back to the startup settings").
// Checks (C-1): after the grow, no episode_end while the measured usage is over the budget, and the governor acts
// (episode start / free cache / shrink plan) within 10 s of the grow, not ~115-120 s later.
// Controls (must pass on both builds): an exact grow ends the episode with no new episode.
// Probe (printed, not counted): a read lag after a normal grow (R9700: a 4.8 GiB grow first read +8.1 GiB).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/mem_governor.h"

#include <cstdint>
#include <cstdio>

using namespace whirl::memgov;

namespace {

int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...)                                           \
    do {                                                           \
        if (cond) {                                                \
            ++g_pass;                                              \
        } else {                                                   \
            ++g_fail;                                              \
            std::printf("  FAIL line %d: %s | ", __LINE__, #cond); \
            std::printf(__VA_ARGS__);                              \
            std::printf("\n");                                     \
        }                                                          \
    } while (0)

constexpr std::uint64_t MiB = 1ull << 20;
constexpr double GiBd = 1024.0 * 1024.0 * 1024.0;

struct Result {
    double t_grow = -1;          // first grow after the hog exits
    std::uint32_t grow_to = 0;
    double end_over_t = -1;      // an episode_end tick with measured > budget (after t_grow)
    double end_t = -1;           // first episode_end after t_grow
    double act_t = -1;           // first action after t_grow (episode_start / free_cache / shrink plan)
    double max_over_mib = 0;     // measured - budget at the episode_end tick
    bool reduced_at_end = false;
    int over_s = 0;              // seconds with measured usage over the budget after t_grow
};

// overshoot: bytes the grow costs over the expected; lag/lag_s: extra read for lag_s seconds after a grow, then gone
// release_next: the overshoot is given back at the next re-allocation (R9700 S1, no HIP cache limit: a grow read +3.33 GiB over
// the expected, the next re-allocation 4 s later freed it)
int g_osc_grows = 0, g_osc_shrinks = 0;
Result run(std::uint64_t overshoot, std::uint64_t lag, double lag_s, bool verbose, bool release_next = false, double t_end = 800) {
    Config cfg;
    cfg.per_tok = 35840;
    cfg.page = 256;
    cfg.pool_start = 340480;
    cfg.ctx = 131072;
    const std::uint64_t budget0 = 31767 * MiB, margin = 2068 * MiB, hog = 4096 * MiB;
    const std::uint64_t fixed = budget0 - margin - static_cast<std::uint64_t>(cfg.pool_start) * cfg.per_tok;
    Governor g(cfg);
    std::uint32_t pool = cfg.pool_start;
    std::uint64_t extra = 0;     // persistent extra usage from the overshoot
    double t_last_resize = -1e9;
    std::uint64_t lag_now = 0;
    Result r;
    const double t_hog_on = 20, t_hog_off = 200;
    g_osc_grows = g_osc_shrinks = 0;
    for (double t = 0; t <= t_end; t += 1.0) {
        const std::uint64_t budget = (t >= t_hog_on && t < t_hog_off) ? budget0 - hog : budget0;
        const std::uint64_t lagv = (t - t_last_resize < lag_s) ? lag_now : 0;
        const std::uint64_t local = fixed + static_cast<std::uint64_t>(pool) * cfg.per_tok + extra + lagv;
        Input in;
        in.now_s = t;
        in.local = local;
        in.budget = budget;
        in.margin = margin;
        in.pool = pool;
        in.all_idle = true;
        const Plan p = g.tick(in);
        if (r.t_grow >= 0 && t > r.t_grow && local > budget) ++r.over_s;
        const bool after = r.t_grow >= 0 && t > r.t_grow;
        if (after) {
            if (r.act_t < 0 && (p.episode_start || p.free_cache || p.shrink)) r.act_t = t;
            if (p.episode_end && r.end_t < 0) r.end_t = t;
            if (p.episode_end && local > budget && r.end_over_t < 0) {
                r.end_over_t = t;
                r.max_over_mib = (static_cast<double>(local) - static_cast<double>(budget)) / MiB;
            }
        }
        if (verbose && (p.episode_start || p.episode_end || p.pool_target))
            std::printf("    t=%4.0f local %.2f GiB budget %.2f GiB%s%s%s\n", t, local / GiBd, budget / GiBd,
                        p.episode_start ? " episode_start" : "", p.episode_end ? " episode_end" : "",
                        p.pool_target ? (p.shrink ? " shrink" : " grow") : "");
        if (p.pool_target && *p.pool_target != pool) {
            const std::uint32_t to = *p.pool_target;
            const bool grew = to > pool;
            const std::int64_t delta = (static_cast<std::int64_t>(to) - static_cast<std::int64_t>(pool)) *
                                       static_cast<std::int64_t>(cfg.per_tok);
            const std::uint64_t expected = static_cast<std::uint64_t>(static_cast<std::int64_t>(local - lagv) + delta);
            if (verbose)
                std::printf("    t=%4.0f re-allocate %u -> %u (expected %.2f GiB)\n", t, pool, to, expected / GiBd);
            pool = to;
            if (release_next) extra = 0;
            if (grew) ++g_osc_grows; else ++g_osc_shrinks;
            if (grew) {
                extra += overshoot;
                lag_now = lag;
                if (t >= t_hog_off && r.t_grow < 0) {
                    r.t_grow = t;
                    r.grow_to = to;
                }
            } else {
                lag_now = 0;
            }
            t_last_resize = t;
            g.resized(pool, grew, t, expected);
        }
    }
    r.reduced_at_end = g.reduced();
    return r;
}

void c1() {
    const std::uint64_t over = 3011 * MiB;  // 2.94 GiB: the measured grow cost over the expected one in R2b_nolim1
    std::printf("C-1 [grow costs 2.94 GiB more than expected]\n");
    const Result r = run(over, 0, 0, true);
    CHECK(r.t_grow >= 0, "a grow after the hog exited");
    CHECK(r.end_over_t < 0, "no episode_end while measured usage is over the budget (episode_end at t=%.0f, %.0f MiB over)",
          r.end_over_t, r.max_over_mib);
    CHECK(r.act_t >= 0 && r.act_t - r.t_grow <= 10.0,
          "the governor acts within 10 s of the grow at t=%.0f (first action t=%.0f, %.0f s later)", r.t_grow, r.act_t,
          r.act_t >= 0 ? r.act_t - r.t_grow : -1.0);
    std::printf("  grow t=%.0f to %u | episode_end t=%.0f (over budget: %s) | first action t=%.0f | %d s over the budget after the grow\n", r.t_grow, r.grow_to,
                r.end_t, r.end_over_t >= 0 ? "yes" : "no", r.act_t, r.over_s);
}

void control() {
    std::printf("control [exact grow]\n");
    const Result r = run(0, 0, 0, false);
    CHECK(r.t_grow >= 0 && r.grow_to == 340480, "grow straight back to pool_start (to %u)", r.grow_to);
    CHECK(r.end_t >= 0, "episode_end reported after the grow");
    CHECK(r.act_t < 0, "no new episode after an exact grow (first action t=%.0f)", r.act_t);
    CHECK(!r.reduced_at_end, "reduced() false at the end");
    std::printf("  grow t=%.0f | episode_end t=%.0f | action after the grow: %s\n", r.t_grow, r.end_t, r.act_t >= 0 ? "yes" : "no");
}

void probe_lag() {
    // R9700 (mem_governor.h trust_s comment): a grow of 4.8 GiB first read +8.1 GiB, i.e. 3.3 GiB extra, "over several seconds"
    std::printf("probe [read lag after an exact grow, not counted]\n");
    for (double s : {2.0, 5.0, 10.0, 30.0}) {
        const Result r = run(0, 3379 * MiB, s, false);
        std::printf("  lag 3.30 GiB for %4.0f s: episode_end t=%.0f, new episode/action after the grow: %s (t=%.0f), reduced at end %d\n",
                    s, r.end_t, r.act_t >= 0 ? "YES" : "no", r.act_t, r.reduced_at_end ? 1 : 0);
    }
}

void probe_band() {
    // a grow that costs 1.27 GiB more than expected (S1c read 0.86 GiB over the expected on a partial grow) on the last step to the startup pool:
    // measured stays under the budget but inside the trigger band (less than margin/2 free)
    std::printf("probe [grow overshoot 1.27 GiB: under the budget, inside the trigger band; not counted]\n");
    const Result r = run(1300 * MiB, 0, 0, true);
    std::printf("  grow t=%.0f | episode_end t=%.0f | first action after the grow t=%.0f (%.0f s later)\n", r.t_grow, r.end_t, r.act_t,
                r.act_t >= 0 ? r.act_t - r.t_grow : -1.0);
}

void probe_release() {
    // no HIP cache limit, R9700 S1 pattern: each grow reads 3.33 GiB over the expected until the next re-allocation frees it
    std::printf("probe [overshoot 3.33 GiB given back at the next re-allocation, 3600 s; not counted]\n");
    const Result r = run(3410 * MiB, 0, 0, false, true, 3600);
    std::printf("  first grow after the hog t=%.0f | re-allocations: %d grows, %d shrinks | reduced at the end %d | %d s over the budget after the first grow\n", r.t_grow, g_osc_grows,
                g_osc_shrinks, r.reduced_at_end ? 1 : 0, r.over_s);
}

}  // namespace

int main() {
    c1();
    control();
    probe_lag();
    probe_band();
    probe_release();
    std::printf("review_gov3 C-1 tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
