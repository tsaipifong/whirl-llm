// whirl-iso-server-decode-floor-tests: module server.decode_floor (src/server/decode_floor.h, class DecodeFloor,
// header-only; registered by arch025-reg SC-14, manifest arch025-regb2 / PLAN N4-reg b2): the test program is the only
// translation unit and includes nothing but the header (ARCH-1 8.5 isolation, header-only mode).
// The checks pin the bookkeeping the scheduler and the prefill rely on, with expected values worked out from the
// header's formulas (pure arithmetic on caller times, no clock):
// - off (N <= 0): prefill always allowed, plan = the cap;
// - plan: the first chunk always runs; with a rows/ms EMA the budget is rows_per_ms * 1000 * gap_tokens / N (capped);
//   the cycles k after a forward solve k * tpc >= N * (T_pf + k * T_c); a floor above tpc / T_c is unreachable and
//   gets max_decode_ratio * T_pf / T_c cycles;
// - prefillAllowed: false while a decoding slot is under N tok/s over the period, true once it is above, and true once
//   the decode-only time reaches the cap (max(min_decode_ms, max_decode_ratio * T_pf); unknown_pf_decode_ms while T_pf
//   is not measured); idle() ends the period;
// - EMAs: the first sample is taken as is, then cur + 0.25 * (x - cur); a cycle with a prefill in it is not a cycle time.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/decode_floor.h"

#include <cmath>
#include <vector>

namespace {

using F = whirl::server::DecodeFloor;
using T = F::SlotTok;

bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

void testOff() {
    F f;
    ISO_CHECK(!f.on() && f.minTps() == 0);
    F g(-3);
    ISO_CHECK(!g.on());
    const std::vector<T> dec{{0, 1, 5}};
    g.startPeriod(0, dec);
    ISO_CHECK(g.prefillAllowed(10, dec));
    const F::Plan p = g.plan(256, 4096);
    ISO_CHECK(p.rows == 4096 && p.cycles == 0 && p.reachable);
}

void testEma() {
    F f(10);
    f.noteCycle(40, false, {});
    ISO_CHECK(near(f.cycleMs(), 40));
    f.noteCycle(80, false, {});
    ISO_CHECK(near(f.cycleMs(), 50));
    f.noteCycle(500, true, {});  // a prefill ran in it: not a cycle time
    ISO_CHECK(near(f.cycleMs(), 50));
    f.notePrefill(1000, 100);
    ISO_CHECK(near(f.rowsPerMs(), 10));
    f.notePrefill(0, 100);  // ignored
    f.notePrefill(100, 0);  // ignored
    ISO_CHECK(near(f.rowsPerMs(), 10));
}

void testPlan() {
    F f(20);  // 20 tok/s
    ISO_CHECK(f.plan(300, 200).rows == 200);  // first chunk capped by cap
    ISO_CHECK(f.plan(100, 4096).rows == 100);  // no rows/ms yet: the first chunk
    f.notePrefill(1000, 100);                // 10 rows/ms
    // gap = 1000 * 10 / 20 = 500 ms -> 5000 rows, capped at 4096
    ISO_CHECK(f.plan(100, 4096).rows == 4096);
    ISO_CHECK(f.plan(100, 8192).rows == 5000);
    // tokens per cycle: one slot, 2 tokens per cycle, T_c = 50 ms
    std::vector<T> dec{{0, 9, 0}};
    f.startPeriod(0, dec);
    f.noteCycle(50, false, dec);  // first: has_last only
    dec[0].n_gen = 2;
    f.noteCycle(50, false, dec);  // tpc = 2
    const F::Plan p = f.plan(100, 1000);  // rows 1000 -> T_pf 100 ms; per cycle 2 - 20 * 0.05 = 1; k = ceil(20 * 0.1 / 1) = 2
    ISO_CHECK(p.rows == 1000 && p.cycles == 2 && p.reachable);
    F g(100);  // 100 tok/s: 2 - 100 * 0.05 < 0 -> unreachable
    g.notePrefill(1000, 100);
    std::vector<T> d2{{0, 9, 0}};
    g.startPeriod(0, d2);
    g.noteCycle(50, false, d2);
    d2[0].n_gen = 2;
    g.noteCycle(50, false, d2);
    const F::Plan q = g.plan(100, 1000);  // T_pf = 100 ms; cycles = ceil(4 * 100 / 50) = 8
    ISO_CHECK(!q.reachable && q.cycles == 8);
}

void testPrefillAllowed() {
    F f(10);  // 10 tok/s
    std::vector<T> dec{{0, 1, 100}};
    ISO_CHECK(f.prefillAllowed(0, dec));  // no period yet
    f.startPeriod(1000, dec);
    f.notePrefill(500, 200);  // T_pf = 200 ms: cap = max(50, 800) = 800 ms decode-only
    // at 1500 ms: need 10 * 0.5 = 5 tokens; 3 produced -> decode first
    dec[0].n_gen = 103;
    ISO_CHECK(!f.prefillAllowed(1500, dec));
    dec[0].n_gen = 105;
    ISO_CHECK(f.prefillAllowed(1500, dec));
    // starvation guard: decode-only time 2000 - 1000 - 200 = 800 >= cap even when the floor is not met
    dec[0].n_gen = 106;
    ISO_CHECK(f.prefillAllowed(2000, dec));
    ISO_CHECK(!f.prefillAllowed(1999, dec));
    // T_pf not measured in a new period: cap = unknown_pf_decode_ms
    f.startPeriod(5000, dec);
    ISO_CHECK(!f.prefillAllowed(5999, dec) && f.prefillAllowed(6000, dec));
    f.idle();
    ISO_CHECK(f.prefillAllowed(6001, dec));  // no period: unrestricted
    ISO_CHECK(f.prefillAllowed(6001, {}));
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.decode_floor",
                    {{"off", testOff}, {"ema", testEma}, {"plan", testPlan}, {"prefill_allowed", testPrefillAllowed}},
                    argc, argv);
}
