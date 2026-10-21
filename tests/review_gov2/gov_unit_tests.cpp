// REVIEW-GOV-2 independent tests (reviewer, written from REVIEW-GOV-1's bug descriptions only; the fixer's tests in
// 04462e8 were not read). CPU only: drives whirl::memgov::Governor through its public API (tick / resized) the way the
// engine does, with a simulated WDDM budget and local usage.
//   P2-1: after the other program exits (startup budget back, the drift seen on the R9700 Ornith run), the pool must
//         grow straight to pool_start in one step and the episode must end (reduced=false, ctx back to startup).
//   P3-1: a doubled hold must go back to hold_s after quiet_reset_s (1800 s default) without an over-budget sample.
// Only Config fields that exist in both 3a4febd and the fixed tree are set (quiet_reset_s keeps its default).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/mem_governor.h"

#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

using namespace whirl::memgov;

namespace {

int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (cond) {                                                            \
            ++g_pass;                                                          \
        } else {                                                               \
            ++g_fail;                                                          \
            std::printf("  FAIL line %d: %s | ", __LINE__, #cond);             \
            std::printf(__VA_ARGS__);                                          \
            std::printf("\n");                                                 \
        }                                                                      \
    } while (0)

constexpr std::uint64_t MiB = 1ull << 20;
constexpr std::uint64_t GiB = 1ull << 30;

// The engine side, simplified: every slot idle, a pool re-allocation happens at once and is reported to the governor
// with the expected usage (usage before + bytes added / freed), as Engine::govApply does.
struct Sim {
    Config cfg;
    Governor g;
    std::uint64_t fixed = 0;   // weights etc.
    std::uint64_t drift = 0;   // extra local usage (applied after the first re-allocation)
    bool drift_on = false;
    std::uint32_t pool = 0;
    std::uint64_t budget = 0, margin = 0;
    struct Ev {
        double t;
        std::uint32_t from, to;
        bool grew;
    };
    std::vector<Ev> resizes;
    bool ended = false;
    double t_end = -1;

    std::uint64_t local() const { return fixed + static_cast<std::uint64_t>(pool) * cfg.per_tok + (drift_on ? drift : 0); }
    Plan step(double t) {
        Input in;
        in.now_s = t;
        in.local = local();
        in.budget = budget;
        in.margin = margin;
        in.pool = pool;
        in.all_idle = true;
        Plan p = g.tick(in);
        if (p.pool_target && *p.pool_target != pool) {
            const std::uint64_t before = local();
            const std::uint32_t old = pool;
            pool = *p.pool_target;
            drift_on = true;
            const std::int64_t delta = (static_cast<std::int64_t>(pool) - static_cast<std::int64_t>(old)) *
                                       static_cast<std::int64_t>(cfg.per_tok);
            const std::uint64_t exp = static_cast<std::uint64_t>(static_cast<std::int64_t>(before) + delta);
            g.resized(pool, !p.shrink, t, exp);
            resizes.push_back({t, old, pool, !p.shrink});
        }
        if (p.episode_end && !ended) {
            ended = true;
            t_end = t;
        }
        return p;
    }
};

// P2-1 with the R9700 Ornith numbers: 393984-token pool (8.27 GiB), margin 2068 MiB, budget 31.02 GiB, a 4 GiB other
// program, and after it exits WHIRL's usage drift (R2a: 29.00 GiB at startup, 29.19 GiB after the grow back).
void testP21(std::uint64_t drift, const char* label) {
    std::printf("P2-1 [%s]\n", label);
    Sim s;
    s.cfg.per_tok = 22528;  // 393984 tokens x 22528 B = 8.27 GiB (R2a log: "393984 -> 200960 tokens (8.27 -> 4.22 GiB)")
    s.cfg.page = 256;
    s.cfg.pool_start = 393984;
    s.cfg.pool_min = 4096;
    s.cfg.ctx = 131072;
    s.g = Governor(s.cfg);
    s.margin = 2068 * MiB;
    const std::uint64_t b0 = static_cast<std::uint64_t>(31.02 * GiB);
    s.budget = b0;
    s.pool = s.cfg.pool_start;
    // startup sizing: exactly the margin free under the budget
    s.fixed = b0 - s.margin - static_cast<std::uint64_t>(s.pool) * s.cfg.per_tok;
    s.drift = drift;
    double t = 0;
    for (; t < 60; t += 5) s.step(t);
    CHECK(s.resizes.empty(), "no action at startup (resizes %zu)", s.resizes.size());
    // the other program takes 4 GiB: the budget drops
    s.budget = b0 - 4 * GiB;
    for (; t < 600; t += 5) s.step(t);
    CHECK(s.resizes.size() == 1 && !s.resizes[0].grew, "one shrink while the other program holds VRAM (resizes %zu)", s.resizes.size());
    CHECK(s.g.reduced(), "episode active while shrunk");
    const std::size_t n_before = s.resizes.size();
    // the other program exits: the startup budget is back
    s.budget = b0;
    for (; t < 3000; t += 5) s.step(t);
    const std::size_t n_grow = s.resizes.size() - n_before;
    CHECK(n_grow == 1, "exactly one grow after the exit (got %zu)", n_grow);
    if (n_grow >= 1) {
        const auto& e = s.resizes[n_before];
        CHECK(e.grew && e.to == s.cfg.pool_start, "the first grow goes straight to pool_start (%u -> %u, pool_start %u)", e.from, e.to,
              s.cfg.pool_start);
    }
    CHECK(s.pool == s.cfg.pool_start, "pool back at startup size (pool %u)", s.pool);
    CHECK(s.ended, "episode_end reported");
    CHECK(!s.g.reduced(), "reduced() false after the episode (reduced %d)", s.g.reduced() ? 1 : 0);
    CHECK(s.g.ctxNew() == s.cfg.ctx, "context for new requests back to startup (%u vs %u)", s.g.ctxNew(), s.cfg.ctx);
    // after the end: the free space is the margin minus the drift, above the trigger, so nothing more happens
    const double free_mib = (static_cast<double>(s.budget) - static_cast<double>(s.local())) / MiB;
    std::printf("  resizes:");
    for (const auto& e : s.resizes) std::printf(" t=%.0f %u->%u", e.t, e.from, e.to);
    std::printf(" | end t=%.0f | free under budget at the end %.0f MiB\n", s.t_end, free_mib);
}

// P3-1: shrink, grow, shrink again within osc_window -> hold doubles; then a quiet period (no over-budget sample) of
// more than quiet_reset_s (default 1800 s) -> hold back to hold_s, and the next episode's grow waits hold_s, not 2x.
void testP31() {
    std::printf("P3-1\n");
    Sim s;
    s.cfg.per_tok = 22528;
    s.cfg.page = 256;
    s.cfg.pool_start = 393984;
    s.cfg.ctx = 131072;
    s.cfg.hold_s = 60;
    s.cfg.recover_s = 30;
    s.g = Governor(s.cfg);
    s.margin = 2068 * MiB;
    const std::uint64_t b0 = static_cast<std::uint64_t>(31.02 * GiB);
    s.budget = b0;
    s.pool = s.cfg.pool_start;
    s.fixed = b0 - s.margin - static_cast<std::uint64_t>(s.pool) * s.cfg.per_tok;
    auto eff = [&] { return s.g.hold() < 0 ? s.cfg.hold_s : s.g.hold(); };
    double t = 0;
    auto run_until = [&](double t1) {
        for (; t < t1; t += 5) s.step(t);
    };
    // episode 1: drop, shrink
    s.budget = b0 - 4 * GiB;
    run_until(100);
    // the budget comes back with plenty of room (the grow happens after the hold)
    s.budget = b0 + 4 * GiB;
    run_until(400);
    std::size_t grows = 0;
    for (const auto& e : s.resizes) grows += e.grew ? 1 : 0;
    CHECK(grows >= 1, "grew back after episode 1 (grows %zu)", grows);
    // the budget drops again within osc_window after the grow: oscillation -> hold doubles
    s.budget = b0 - 4 * GiB;
    run_until(450);
    const double doubled = eff();
    CHECK(doubled >= 2 * s.cfg.hold_s, "hold doubled after the oscillation (hold %.0f)", doubled);
    const double t_last_over = t - 5;
    // quiet: plenty of room, no over-budget sample, for more than 1800 s
    s.budget = b0 + 4 * GiB;
    run_until(t_last_over + 1800 + 60);
    const double after_quiet = eff();
    CHECK(after_quiet == s.cfg.hold_s, "hold back to hold_s after 1800 s quiet (hold %.0f, hold_s %.0f)", after_quiet, s.cfg.hold_s);
    // behaviour: a new episode; the grow after it must come about hold_s after the last over-budget sample, not 2x
    const std::size_t n0 = s.resizes.size();
    s.budget = b0 - 4 * GiB;
    const double T = t;
    run_until(T + 10);
    const double t_over = t - 5;
    s.budget = b0 + 4 * GiB;
    run_until(T + 600);
    double t_grow = -1;
    for (std::size_t i = n0; i < s.resizes.size(); ++i)
        if (s.resizes[i].grew) {
            t_grow = s.resizes[i].t;
            break;
        }
    CHECK(t_grow > 0 && t_grow - t_over < 1.5 * s.cfg.hold_s,
          "grow after the new episode %.0f s after the last over-budget sample (must be < %.0f; doubled hold would give >= %.0f)",
          t_grow - t_over, 1.5 * s.cfg.hold_s, doubled);
    std::printf("  hold: doubled %.0f, after quiet %.0f; new-episode grow %.0f s after the last over-budget sample\n", doubled,
                after_quiet, t_grow - t_over);
}


// Finding probe (REVIEW-GOV-2 C-1, not one of the four bugs; not counted in pass/fail): a grow whose real cost is larger
// than the governor's expected usage (R2b_nolim: +6.63 GiB for a +3.55 GiB pool) leaves WHIRL over the budget. Within
// trust_s of the grow the trigger uses min(measured, expected), so the episode end is reported while measured usage is
// over the budget, and nothing acts until trust_s has passed.
void probeOvershoot() {
    std::printf("PROBE C-1 (grow overshoot)\n");
    Sim s;
    s.cfg.per_tok = 36966;  // Swift-1.5 27B: 11.73 GiB / 340480 tokens
    s.cfg.page = 256;
    s.cfg.pool_start = 329728;
    s.cfg.ctx = 131072;
    s.g = Governor(s.cfg);
    s.margin = 2068 * MiB;
    const std::uint64_t b0 = static_cast<std::uint64_t>(31.02 * GiB);
    s.budget = b0;
    s.pool = s.cfg.pool_start;
    s.fixed = b0 - s.margin - static_cast<std::uint64_t>(s.pool) * s.cfg.per_tok;
    double t = 0;
    for (; t < 30; t += 5) s.step(t);
    s.budget = b0 - 4 * GiB;
    for (; t < 300; t += 5) s.step(t);
    s.budget = b0;
    // from the next re-allocation on, the allocator keeps 3 GiB more than the pool (the default HIP cache, R2b_nolim)
    const std::size_t n0 = s.resizes.size();
    double t_end = -1, over_at_end = 0, t_trigger = -1;
    for (; t < 900; t += 5) {
        if (s.resizes.size() > n0 && s.drift != 3 * GiB) s.drift = 3 * GiB;
        Plan p = s.step(t);
        const double over = (static_cast<double>(s.local()) - static_cast<double>(s.budget)) / GiB;
        if (p.episode_end && t_end < 0) { t_end = t; over_at_end = over; }
        if (p.episode_start && t_end >= 0 && t_trigger < 0) t_trigger = t;
    }
    std::printf("  resizes:");
    for (const auto& e : s.resizes) std::printf(" t=%.0f %u->%u", e.t, e.from, e.to);
    std::printf("\n  episode_end at t=%.0f with measured usage %.2f GiB over the budget; next episode start t=%.0f (%.0f s later)\n",
                t_end, over_at_end, t_trigger, t_trigger - t_end);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "--probe") {
        probeOvershoot();
        return 0;
    }
    testP21(static_cast<std::uint64_t>(0.19 * GiB), "drift 0.19 GiB (R9700 Ornith R2a)");
    testP21(0, "no drift");
    testP31();
    std::printf("review_gov2 governor tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
