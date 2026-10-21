// REVIEW-GOV-4 independent test for REVIEW-GOV-3 C-2 (reviewer; written from the one-sentence description only, before
// reading cbaf376's test or 9129932's fix):
//   "after a grow-back, a measured value below the budget but inside the trigger zone (free < margin/2) was masked by
//    min(measured, expected) for 120 s and the episode end was reported".
// Harness: the REVIEW-GOV-3 C-1 driver (tests/review_gov3/c1_test.cpp; R9700 Swift numbers, budget 31767 MiB, margin
// 2068 MiB, 4 GiB hog from t=20 to t=200, every slot idle, expected = usage before + bytes added or freed).
// The grow straight back to the startup pool happens when endMargin (1551 MiB) stays free with the expected usage; the
// startup pool then leaves margin - residual free (residual: what the other program left behind after it exited). The
// measured value lands in the trigger band (free < margin/2 = 1034 MiB) when overshoot + residual > 1034 MiB.
// Checks (C-2), cases (residual, overshoot) = (0, 1100), (0, 1300), (400, 700) MiB, all under the budget, all in the band:
//   a. no episode_end while less than margin/2 is free under the measured usage;
//   b. the governor acts (episode start / free cache / shrink plan) within 10 s of the grow.
// Controls (must pass on both builds; outside the band): (0, 0), (0, 900), (400, 0) end the episode with no new action.
// Probes (printed, not counted): a read lag inside the band after an exact grow, which the fix no longer masks.
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
    double end_band_t = -1;      // an episode_end tick with measured + margin/2 > budget (after t_grow)
    double band_free_mib = 0;    // budget - measured at that tick
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
Result run(std::uint64_t overshoot, std::uint64_t lag, double lag_s, bool verbose, bool release_next = false, double t_end = 800, std::uint64_t residual = 0) {
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
        const std::uint64_t budget = (t >= t_hog_on && t < t_hog_off) ? budget0 - hog : budget0 - (t >= t_hog_off ? residual : 0);
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
            if (p.episode_end && local + margin / 2 > budget && r.end_band_t < 0) {
                r.end_band_t = t;
                r.band_free_mib = (static_cast<double>(budget) - static_cast<double>(local)) / MiB;
            }
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

void c2(std::uint64_t res_mib, std::uint64_t over_mib) {
    std::printf("C-2 [residual %llu MiB, grow costs %llu MiB more than expected: under the budget, in the band]\n",
                static_cast<unsigned long long>(res_mib), static_cast<unsigned long long>(over_mib));
    const Result r = run(over_mib * MiB, 0, 0, false, false, 800, res_mib * MiB);
    CHECK(r.t_grow >= 0 && r.grow_to == 340480, "a grow straight back to the startup pool after the hog exited (to %u)", r.grow_to);
    CHECK(r.end_band_t < 0, "no episode_end with less than margin/2 free (episode_end at t=%.0f with %.0f MiB free, trigger 1034)",
          r.end_band_t, r.band_free_mib);
    CHECK(r.act_t >= 0 && r.act_t - r.t_grow <= 10.0, "acts within 10 s of the grow at t=%.0f (first action t=%.0f, %.0f s later)",
          r.t_grow, r.act_t, r.act_t >= 0 ? r.act_t - r.t_grow : -1.0);
    std::printf("  grow t=%.0f to %u | first episode_end t=%.0f (in band: %s, %.0f MiB free) | first action t=%.0f | reduced at end %d\n",
                r.t_grow, r.grow_to, r.end_t, r.end_band_t >= 0 ? "yes" : "no", r.band_free_mib, r.act_t, r.reduced_at_end ? 1 : 0);
}

void control(std::uint64_t res_mib, std::uint64_t over_mib) {
    std::printf("control [residual %llu MiB, grow overshoot %llu MiB, outside the band]\n", static_cast<unsigned long long>(res_mib),
                static_cast<unsigned long long>(over_mib));
    const Result r = run(over_mib * MiB, 0, 0, false, false, 800, res_mib * MiB);
    CHECK(r.t_grow >= 0 && r.grow_to == 340480, "grow straight back to pool_start (to %u)", r.grow_to);
    CHECK(r.end_t >= 0 && r.end_t - r.t_grow <= 10.0, "episode_end soon after the grow (t=%.0f)", r.end_t);
    CHECK(r.act_t < 0, "no action after the grow (first action t=%.0f)", r.act_t);
    CHECK(!r.reduced_at_end, "reduced() false at the end");
    std::printf("  grow t=%.0f | episode_end t=%.0f | action after the grow: %s\n", r.t_grow, r.end_t, r.act_t >= 0 ? "yes" : "no");
}

void probe_band_lag() {
    std::printf("probe [read lag inside the band after an exact grow; not counted]\n");
    for (std::uint64_t lag : {600ull, 1300ull}) {
        for (double s : {2.0, 5.0, 10.0}) {
            const Result r = run(0, lag * MiB, s, false);
            std::printf("  lag %4llu MiB for %4.0f s: episode_end t=%.0f, action after the grow: %s (t=%.0f), grows %d shrinks %d, reduced at end %d\n",
                        static_cast<unsigned long long>(lag), s, r.end_t, r.act_t >= 0 ? "YES" : "no", r.act_t, g_osc_grows, g_osc_shrinks,
                        r.reduced_at_end ? 1 : 0);
        }
    }
}

}  // namespace

int main() {
    c2(0, 1100);
    c2(0, 1300);
    c2(400, 700);
    control(0, 0);
    control(0, 900);
    control(400, 0);
    probe_band_lag();
    std::printf("review_gov4 C-2 tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
