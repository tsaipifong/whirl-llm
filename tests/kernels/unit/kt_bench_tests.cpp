// Unit tests of kt::Bench (tests/kernels/kt_bench.{h,cpp}), the shared kernel timer of whirl-kernel-test --bench
// (measure-tools P2). CPU only: a fake clock and scripted block timers stand in for the GPU.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Usage: whirl-kt-bench-tests [--emit FILE]  (--emit also writes sample rows for the importer check,
// `db_record.py jsonl FILE --dry-run`). Exit 0 = all pass.

#include "../kt_bench.h"
#include "../kt_dq4s.h"  // v0252-8sq5: the dq4s split-count bench table
#include "whirl/ops/attention.h"  // ops::dq4Splits (the production split rule the dq4s table pairs against)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <span>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace kt;

static int g_fail = 0, g_pass = 0;

static void check(const std::string& name, bool ok, const std::string& info = "") {
    std::printf("%s  %s", ok ? "PASS" : "FAIL", name.c_str());
    if (!ok && !info.empty()) std::printf("  (%s)", info.c_str());
    std::printf("\n");
    (ok ? g_pass : g_fail)++;
}

static std::string num(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.6g", v);
    return b;
}

// Fake wall clock: blocks advance it by their GPU time; sleeps are recorded.
class FakeClock : public Clock {
public:
    double t = 1000.0;
    std::vector<double> sleeps;
    double now() override { return t; }
    void sleep(double s) override {
        sleeps.push_back(s);
        t += s;
    }
};

// A scripted kernel: per-launch time from a function of the sample counter; records every call.
struct FakeKernel {
    FakeClock& clk;
    std::function<double(int call)> us_per_launch;  // call = 0, 1, ... (every BlockFn call, warm-up included)
    std::vector<int> launches;
    std::vector<std::uint64_t> rots;
    std::string tag;
    std::vector<std::string>* order = nullptr;
    int calls = 0;
    BlockFn fn() {
        return [this](int n, std::uint64_t rot0) {
            launches.push_back(n);
            rots.push_back(rot0);
            if (order) order->push_back(tag);
            const double us = us_per_launch(calls++);
            const double ms = us * n / 1000.0;
            clk.t += ms / 1000.0 + 20e-6;  // GPU time + 20 us host overhead
            return ms;
        };
    }
};

static BenchCfg quietCfg() {
    BenchCfg c;
    c.burst_idle_s = 0;
    c.min_block_us = 200;  // the rule tests below use 200 us blocks; the default (1000) is checked separately
    return c;
}

// --------------------------------------------------------------------------------------------- stop rule + stats
static void test_constant_kernel() {
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    FakeKernel k{clk, [](int) { return 10.0; }};
    const CellStats cs = b.run(k.fn());
    check("stop: constant 10 us kernel stops on rel<=1% at min_samples (10)", cs.stop_reason == "rel<=1%" && cs.st.n == 10,
          cs.stop_reason + " n=" + std::to_string(cs.st.n));
    check("stats: median 10 us, MAD 0, p10 = p90 = 10", cs.st.median == 10.0 && cs.st.mad == 0.0 && cs.st.p10 == 10.0 && cs.st.p90 == 10.0,
          num(cs.st.median));
    // calibration: 10 us per launch, block >= 200 us -> 32 launches (1, 2, 4, 8, 16, 32 doubling)
    check("calibration: block doubles to the first size >= 200 us (32 launches of 10 us)", cs.block == 32, std::to_string(cs.block));
    check("warm-up: >= 100 ms of GPU time before the first timed sample (and < 100 ms + one block)",
          cs.warm_ms >= 100.0 && cs.warm_ms < 100.0 + 32 * 10.0 / 1000.0 + 1e-9, num(cs.warm_ms));
    check("ok() true; no error", cs.ok() && cs.error.empty());
}

static void test_calibration_slow_kernel() {
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    FakeKernel k{clk, [](int) { return 500.0; }};
    const CellStats cs = b.run(k.fn());
    check("calibration: a 500 us kernel keeps block 1", cs.block == 1, std::to_string(cs.block));
    FakeKernel k2{clk, [](int) { return 0.5; }};
    BenchCfg c = quietCfg();
    c.max_block = 64;
    Bench b2(c, &clk);
    const CellStats cs2 = b2.run(k2.fn());
    check("calibration: capped at max_block (0.5 us kernel, cap 64)", cs2.block == 64, std::to_string(cs2.block));
}

static void test_time_stop() {
    FakeClock clk;
    BenchCfg c = quietCfg();
    c.max_cell_s = 0.5;
    Bench b(c, &clk);
    // alternating 10 / 12 us: IQR / median ~ 18 %, never <= 1 % -> time budget
    FakeKernel k{clk, [](int i) { return i % 2 ? 12.0 : 10.0; }};
    const CellStats cs = b.run(k.fn());
    check("stop: noisy samples (10/12 us) stop on the time budget", cs.stop_reason == "time", cs.stop_reason);
    check("stop: measured wall time within budget + one sample", cs.measure_s >= 0.5 && cs.measure_s < 0.5 + 0.01, num(cs.measure_s));
    check("stats: IQR/median of 10/12 alternation > 1 %", cs.rel() > 0.01, num(cs.rel()));
}

static void test_n_stop() {
    FakeClock clk;
    BenchCfg c = quietCfg();
    c.max_samples = 25;
    c.max_cell_s = 1e9;
    Bench b(c, &clk);
    FakeKernel k{clk, [](int i) { return i % 2 ? 12.0 : 10.0; }};
    const CellStats cs = b.run(k.fn());
    check("stop: max_samples reached -> \"n\" with exactly 25 samples", cs.stop_reason == "n" && cs.st.n == 25,
          cs.stop_reason + " " + std::to_string(cs.st.n));
}

static void test_rel_after_settle() {
    // first 15 samples noisy, then constant: stops once the window of all kept samples has IQR/median <= 1 %
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    int timed = 0;
    FakeKernel k{clk, nullptr};
    k.us_per_launch = [&](int) { return 10.0; };
    const CellStats cs0 = b.run(k.fn());
    (void)cs0;
    check("relIqr: {10,10,10,11} quartiles", std::fabs(relIqr({10, 10, 10, 11}) - 0.025) < 1e-12, num(relIqr({10, 10, 10, 11})));
    check("relIqr: < 2 samples -> 0", relIqr({5}) == 0.0);
    (void)timed;
}

static void test_error() {
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    int calls = 0;
    BlockFn zero = [&](int, std::uint64_t) {
        ++calls;
        return 0.0;
    };
    const CellStats cs = b.run(zero);
    check("error: zero-duration block -> stop_reason error, no samples, not ok()", cs.stop_reason == "error" && cs.st.n == 0 && !cs.ok(),
          cs.stop_reason);
    check("error: reason names the zero duration", cs.error.find("zero") != std::string::npos, cs.error);
    // an invalid reading during calibration doubles the block (too short to time); error only at max_block (1..65536)
    check("error: an always-zero timer fails after the block reached max_block (17 sizes)", calls == 17, std::to_string(calls));
    BlockFn nan = [&](int, std::uint64_t) { return std::nan(""); };
    check("error: NaN block -> error", b.run(nan).stop_reason == "error");
    // an error during sampling (after calibration) also stops
    int n2 = 0;
    BlockFn late = [&](int n, std::uint64_t) {
        clk.t += 1e-3;
        return ++n2 > 320 ? -1.0 : 0.01 * n;  // invalid from the 3rd timed sample on (317 calibration + warm-up calls)
    };
    const CellStats cs2 = b.run(late);
    check("error: a timer that stays invalid during sampling -> error, samples discarded", cs2.stop_reason == "error" && cs2.st.n == 0,
          cs2.stop_reason + " " + cs2.error);
    check("error: the error names the invalid count", cs2.error.find("readings invalid") != std::string::npos, cs2.error);
}

// --------------------------------------------------------------------------------------------- event timer (fix 1)
// Fake GPU runtime: a stream is an id; launches add their duration to the stream's GPU clock; an event recorded on a
// stream takes that stream's clock at its position. `zero_below_us`: an interval shorter than this reads 0 (what the
// R9700 smoke showed for short blocks); `stale_reuse`: a re-recorded event keeps its first timestamp (a reused pair
// then reads zero / negative intervals).
struct FakeRt {
    FakeClock& clk;
    double zero_below_us = 0;
    bool stale_reuse = false;
    std::map<void*, double> gpu;        // stream -> GPU clock (us)
    struct Ev {
        double t = -1;
        void* stream = nullptr;
        int records = 0;
        bool synced = false;
    };
    std::map<void*, Ev> ev;
    std::vector<std::string> log;
    int next = 1, live = 0, created = 0;
    EventApi api() {
        EventApi a;
        a.create = [this] {
            void* e = reinterpret_cast<void*>(static_cast<std::uintptr_t>(next++));
            ev[e] = {};
            ++live;
            ++created;
            log.push_back("create");
            return e;
        };
        a.destroy = [this](void* e) {
            ev.erase(e);
            --live;
            log.push_back("destroy");
        };
        a.record = [this](void* e, void* s) {
            Ev& x = ev[e];
            if (!(stale_reuse && x.records > 0)) x.t = gpu[s];
            x.stream = s;
            ++x.records;
            x.synced = false;
            log.push_back("record");
        };
        a.sync = [this](void* e) {
            ev[e].synced = true;
            log.push_back("sync");
        };
        a.stream_sync = [this](void*) { log.push_back("stream_sync"); };
        a.elapsed_ms = [this](void* e0, void* e1) {
            log.push_back(ev[e1].synced ? "elapsed" : "elapsed-unsynced");
            const double us = ev[e1].t - ev[e0].t;
            return us < zero_below_us ? 0.0 : us / 1000.0;
        };
        return a;
    }
    // one launch of `us` on stream s (also advances the host clock: the host waits for it at the sync)
    void launch(void* s, double us) {
        gpu[s] += us;
        clk.t += us / 1e6;
    }
};

static void test_event_protocol() {
    FakeClock clk;
    FakeRt rt{clk};
    void* const s = reinterpret_cast<void*>(0x5);
    int wrong_stream = 0;
    auto st = std::make_shared<TimerStats>();
    BlockFn f = eventBlock(rt.api(), s,
                           [&](std::uint64_t, void* stream) {
                               if (stream != s) ++wrong_stream;
                               rt.launch(stream, 10.0);
                           },
                           &clk, st);
    const double ms = f(4, 0);
    check("timer: 4 launches of 10 us -> 0.04 ms", std::fabs(ms - 0.04) < 1e-12, num(ms));
    std::string seq;
    for (const auto& x : rt.log) seq += x + " ";
    check("timer: protocol create create record record sync stream_sync elapsed destroy destroy",
          seq == "create create record record sync stream_sync elapsed destroy destroy ", seq);
    check("timer: every call gets the stream the events are recorded on", wrong_stream == 0);
    check("timer: no event leaks", rt.live == 0, std::to_string(rt.live));
    f(4, 4);
    check("timer: a fresh event pair per block (4 created after two blocks)", rt.created == 4, std::to_string(rt.created));
    check("timer: valid readings are not retried", st->blocks == 2 && st->retries == 0 && st->invalid == 0);
}

static void test_event_reuse_would_fail() {
    // the first version's pattern: one pair re-recorded for every block, on a runtime that keeps stale timestamps
    FakeClock clk;
    FakeRt rt{clk};
    rt.stale_reuse = true;
    void* const s = reinterpret_cast<void*>(0x5);
    EventApi a = rt.api();
    void* e0 = a.create();
    void* e1 = a.create();
    auto reused = [&](int n) {
        a.record(e0, s);
        for (int j = 0; j < n; ++j) rt.launch(s, 10.0);
        a.record(e1, s);
        a.sync(e1);
        return a.elapsed_ms(e0, e1);
    };
    reused(1);
    const double second = reused(2);
    check("timer (model of the smoke defect): a re-recorded pair reads a stale interval", std::fabs(second - 0.01) < 1e-12, num(second));
    // eventBlock on the same runtime: fresh events -> correct
    BlockFn f = eventBlock(rt.api(), s, [&](std::uint64_t, void* stream) { rt.launch(stream, 10.0); }, &clk);
    check("timer: fresh events per block read the right interval on that runtime", std::fabs(f(2, 0) - 0.02) < 1e-12);
}

static void test_event_invalid_short_blocks() {
    // short intervals read 0 (< 300 us): the bench must still calibrate to a valid block and time correctly
    FakeClock clk;
    FakeRt rt{clk};
    rt.zero_below_us = 300;
    void* const s = reinterpret_cast<void*>(0x7);
    auto st = std::make_shared<TimerStats>();
    BlockFn f = eventBlock(rt.api(), s, [&](std::uint64_t, void* stream) { rt.launch(stream, 10.0); }, &clk, st);
    BenchCfg c = quietCfg();
    c.min_block_us = 200;  // even with a 200 us target the zero readings below 300 us must not end the cell
    Bench b(c, &clk);
    const CellStats cs = b.run(f);
    check("timer: zero readings of short blocks -> calibration doubles past them, cell ok", cs.ok() && cs.stop_reason == "rel<=1%",
          cs.stop_reason + " " + cs.error);
    check("timer: calibrated block 32 (320 us, the first valid size)", cs.block == 32, std::to_string(cs.block));
    check("timer: median 10 us per launch", cs.st.median == 10.0, num(cs.st.median));
    check("timer: the zero readings were retried and counted", st->retries > 0 && st->invalid > 0 && !st->last_invalid.empty(),
          std::to_string(st->retries) + " " + st->last_invalid);
    BenchCfg d;
    check("timer: default min_block_us is 1000", d.min_block_us == 1000.0);
}

static void test_event_longer_than_host() {
    // a reading longer than the host interval around it is impossible -> invalid (retried, then rejected)
    FakeClock clk;
    FakeRt rt{clk};
    void* const s = reinterpret_cast<void*>(0x9);
    auto st = std::make_shared<TimerStats>();
    BlockFn f = eventBlock(rt.api(), s, [&](std::uint64_t, void* stream) { rt.gpu[stream] += 5000.0; /* host clock not advanced */ }, &clk, st);
    const double ms = f(1, 0);
    check("timer: GPU interval > host wall + 0.5 ms -> invalid (<= 0), counted", ms <= 0 && st->invalid == 1 && st->retries == 1,
          num(ms) + " " + st->last_invalid);
}

static void test_occasional_invalid_dropped() {
    // one invalid reading in 50 during sampling is dropped and counted, the cell still succeeds
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    int calls = 0;
    BlockFn f = [&](int n, std::uint64_t) {
        ++calls;
        clk.t += 10e-6 * n;
        return calls == 320 ? 0.0 : 0.01 * n;  // call 320 = 3rd timed sample (6 calibration + 311 warm-up calls before)
    };
    const CellStats cs = b.run(f);
    check("sampling: an occasional invalid reading is dropped (invalid 1), cell ok", cs.ok() && cs.invalid == 1, std::to_string(cs.invalid) + " " + cs.stop_reason);
}

// --------------------------------------------------------------------------------------------- rotation
static void test_rotation() {
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    FakeKernel k{clk, [](int) { return 10.0; }};
    b.run(k.fn());
    bool contiguous = true;
    for (std::size_t i = 1; i < k.rots.size(); ++i) contiguous = contiguous && k.rots[i] == k.rots[i - 1] + static_cast<std::uint64_t>(k.launches[i - 1]);
    check("rotation: every block starts where the previous ended (rot0 contiguous from 0)", !k.rots.empty() && k.rots[0] == 0 && contiguous);
    // a second cell continues the counter of its own (fresh FakeKernel), not reset to the same copy as warm-up
    check("rotationCopies: 8 MB set, 512 MB target -> 64 copies", rotationCopies(8u << 20, 512u << 20, 4ull << 30) == 64,
          std::to_string(rotationCopies(8u << 20, 512u << 20, 4ull << 30)));
    check("rotationCopies: rounds up (100 MB, 512 MB -> 6)", rotationCopies(100u << 20, 512u << 20, 4ull << 30) == 6);
    check("rotationCopies: capped by the memory budget (8 MB, 2 GB target, 1 GB cap -> 128)",
          rotationCopies(8u << 20, 2ull << 30, 1ull << 30) == 128);
    check("rotationCopies: set larger than the target -> 1", rotationCopies(1ull << 30, 512u << 20, 4ull << 30) == 1);
    check("rotationCopies: set larger than the cap -> still 1", rotationCopies(8ull << 30, 512u << 20, 4ull << 30) == 1);
    check("rotationCopies: max_copies cap", rotationCopies(1024, 512u << 20, 4ull << 30, 48) == 48);
    check("rotationCopies: ws 0 -> 1", rotationCopies(0, 512u << 20, 4ull << 30) == 1);
    check("rotationTarget: gfx1201 512 MB, gfx1151 2 GB",
          rotationTarget("gfx1201") == (512ull << 20) && rotationTarget("gfx1151") == (2ull << 30) && rotationTarget("gfx1151:xnack-") == (2ull << 30));
}

// --------------------------------------------------------------------------------------------- anchor + cooldown
static void test_anchor_drift_once() {
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    // anchor 20 us; after setAnchor (warm-up, calibration, reference) every call is a check:
    // 1 = start of the cell, 2 = after 4 samples (+10 % -> cooldown), 3 = re-check after the sleep (back to 20)
    int checks = 0;
    bool armed = false;
    BlockFn afn = [&](int n, std::uint64_t) {
        double us = 20.0;
        if (armed && ++checks == 2) us = 22.0;
        clk.t += us * n / 1e6;
        return us * n / 1000.0;
    };
    b.setAnchor(afn, "fake.anchor");
    armed = true;
    FakeKernel k{clk, [](int) { return 10.0; }};
    const CellStats cs = b.run(k.fn());
    check("anchor: one +10 % drift -> exactly one cooldown", cs.cooldowns == 1, std::to_string(cs.cooldowns));
    check("anchor: first cooldown sleeps cool_first_s (2 s) and the anchor recovered", clk.sleeps.size() == 1 && clk.sleeps[0] == 2.0,
          std::to_string(clk.sleeps.size()));
    check("anchor: the samples since the last clean anchor (4) were discarded", cs.discarded == 4, std::to_string(cs.discarded));
    check("anchor: still stops on rel<=1% with >= 10 kept samples", cs.stop_reason == "rel<=1%" && cs.st.n >= 10,
          cs.stop_reason + " " + std::to_string(cs.st.n));
    check("anchor: range recorded (min 20, max 22 us per launch)", cs.anchor_min_us == 20.0 && cs.anchor_max_us == 22.0,
          num(cs.anchor_min_us) + " " + num(cs.anchor_max_us));
    check("anchor: cooldown seconds excluded from the measuring time", cs.cooldown_s == 2.0 && cs.measure_s < 1.0, num(cs.measure_s));
}

static void test_anchor_noisy() {
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    // after the reference the anchor alternates hot/cool: every periodic check drifts, every re-check recovers
    int checks = 0;
    bool armed = false;
    BlockFn afn = [&](int n, std::uint64_t) {
        double us = 20.0;
        if (armed && ++checks % 2 == 0) us = 21.0;  // +5 %
        clk.t += us * n / 1e6;
        return us * n / 1000.0;
    };
    b.setAnchor(afn, "fake.anchor");
    armed = true;
    FakeKernel k{clk, [](int) { return 10.0; }};
    const CellStats cs = b.run(k.fn());
    check("anchor: > 2 cooldowns in a cell -> stop_reason noisy", cs.stop_reason == "noisy" && cs.cooldowns == 3,
          cs.stop_reason + " " + std::to_string(cs.cooldowns));
}

static void test_anchor_cooldown_cap() {
    FakeClock clk;
    BenchCfg c = quietCfg();
    Bench b(c, &clk);
    // the anchor never comes back after the reference: sleeps 2, 4, 8, 16, 30 (cap 60 s total), then the cell goes on
    int checks = 0;
    bool armed = false;
    BlockFn afn = [&](int n, std::uint64_t) {
        double us = 20.0;
        if (armed && ++checks >= 1) us = 25.0;
        clk.t += us * n / 1e6;
        return us * n / 1000.0;
    };
    b.setAnchor(afn, "fake.anchor");
    armed = true;
    FakeKernel k{clk, [](int) { return 10.0; }};
    const CellStats cs = b.run(k.fn());
    double total = 0;
    for (std::size_t i = 0; i < clk.sleeps.size() && i < 5; ++i) total += clk.sleeps[i];
    check("cooldown: exponential 2, 4, 8, 16 then capped to 60 s total per cooldown",
          clk.sleeps.size() >= 5 && clk.sleeps[0] == 2 && clk.sleeps[1] == 4 && clk.sleeps[2] == 8 && clk.sleeps[3] == 16 &&
              std::fabs(total - 60.0) < 1e-9,
          std::to_string(clk.sleeps.size()) + " total " + num(total));
    check("cooldown: a never-recovering anchor ends noisy", cs.stop_reason == "noisy", cs.stop_reason);
}

static void test_anchor_faster_becomes_reference() {
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    int checks = 0;
    bool armed = false;
    BlockFn afn = [&](int n, std::uint64_t) {
        double us = 20.0;
        if (armed && ++checks >= 1) us = 19.0;  // cooler machine later
        clk.t += us * n / 1e6;
        return us * n / 1000.0;
    };
    b.setAnchor(afn, "fake.anchor");
    armed = true;
    FakeKernel k{clk, [](int) { return 10.0; }};
    const CellStats cs = b.run(k.fn());
    check("anchor: a faster anchor becomes the new reference (no cooldown)", cs.cooldowns == 0 && b.anchorRefUs() == 19.0,
          num(b.anchorRefUs()));
    check("anchor: name kept", b.anchorName() == "fake.anchor");
}

// --------------------------------------------------------------------------------------------- fix 2 (smoke 2)
static void test_rewarm_after_cooldown() {
    // after the cooldown sleep the anchor runs >= warmup_ms before it is read (an idled GPU clock reads slow), and the
    // cell kernel is re-warmed before sampling goes on
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    std::vector<char> ev;  // 'a' anchor block, 'k' cell block, 'D' the drifting anchor reading
    int checks = 0;
    bool armed = false;
    BlockFn afn = [&](int n, std::uint64_t) {
        double us = 20.0;
        if (armed) {
            if (++checks == 2) {
                us = 22.0;  // the 2nd check drifts once
                ev.push_back('D');
            } else {
                ev.push_back('a');
            }
        }
        clk.t += us * n / 1e6;
        return us * n / 1000.0;
    };
    b.setAnchor(afn, "fake.anchor");
    armed = true;
    FakeKernel k{clk, [](int) { return 10.0; }};
    BlockFn kf = k.fn();
    const CellStats cs = b.run([&](int n, std::uint64_t r) {
        ev.push_back('k');
        return kf(n, r);
    });
    std::size_t d = 0;
    while (d < ev.size() && ev[d] != 'D') ++d;
    int anchor_after = 0, cell_after = 0;
    std::size_t j = d + 1;
    for (; j < ev.size() && ev[j] == 'a'; ++j) ++anchor_after;
    for (; j < ev.size() && ev[j] == 'k'; ++j) ++cell_after;
    // 100 ms of 20 us anchor launches in blocks of 16 (320 us): 313 blocks of warm-up + 1 read
    check("cooldown: the anchor is re-warmed (>= 100 ms) after the sleep before it is read", anchor_after >= 314, std::to_string(anchor_after));
    // cell: 100 ms of 10 us launches in blocks of 32 (320 us) = 313 warm-up blocks, then the samples
    check("cooldown: the cell kernel is re-warmed (>= 100 ms) before sampling resumes", cell_after >= 313, std::to_string(cell_after));
    check("cooldown: one cooldown (2 s), cell still ok", cs.cooldowns == 1 && cs.ok() && clk.sleeps.size() == 1,
          std::to_string(cs.cooldowns) + " " + cs.stop_reason);
}

static void test_cache_state_anchor_switched_off() {
    // the smoke-2 defect: the anchor reads 33 us right after itself (cache-hot) and 135 us after any cell block (cold).
    // Sleeping never recovers it. The bench must bound the damage: noisy_streak_off (3) noisy cells -> anchor off.
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    bool last_was_anchor = false;
    BlockFn afn = [&](int n, std::uint64_t) {
        // a block of n launches: the first launch is cold when a cell ran before, the rest hot
        const double us = (last_was_anchor ? 33.0 * n : 135.0 + 33.0 * (n - 1));
        last_was_anchor = true;
        clk.t += us / 1e6;
        return us / 1000.0;
    };
    b.setAnchor(afn, "fake.cache_anchor");
    int noisy = 0;
    for (int cell = 0; cell < 6; ++cell) {
        BlockFn kf = [&](int n, std::uint64_t) {
            last_was_anchor = false;
            clk.t += 10e-6 * n;
            return 0.01 * n;
        };
        b.beforeCell();
        const CellStats cs = b.run(kf);
        noisy += cs.stop_reason == "noisy";
    }
    check("cache-state anchor: switched off after 3 noisy cells", b.anchorSwitchedOff() && !b.hasAnchor(), b.anchorError());
    check("cache-state anchor: exactly 3 noisy cells, the later ones run without cooldowns", noisy == 3, std::to_string(noisy));
    double slept = 0;
    for (double x : clk.sleeps) slept += x;
    check("cache-state anchor: total cooldown sleep bounded (<= 3 cells x 2 x 60 s)", slept <= 360.0 + 1e-9, num(slept));
}

// --------------------------------------------------------------------------------------------- A/B
static void test_ab_order_and_ratio() {
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    std::vector<std::string> order;
    FakeKernel ka{clk, [](int) { return 10.0; }}, kb{clk, [](int) { return 9.0; }};
    ka.tag = "A";
    kb.tag = "B";
    ka.order = kb.order = &order;
    const AbStats ab = b.runAB(ka.fn(), kb.fn());
    // drop warm-up / calibration calls: the timed part starts after both calibrated; take the last 8 entries of the pattern
    std::string seq;
    for (const auto& s : order) seq += s;
    const std::size_t p = seq.size() >= 8 ? seq.size() - static_cast<std::size_t>(ab.pairs) * 2 : 0;
    const std::string timed = seq.substr(p, 8);
    check("A/B: timed launches alternate ABBA (A B B A A B B A)", timed == "ABBAABBA", timed);
    check("A/B: paired ratio B/A = 0.9", std::fabs(ab.ratio.st.median - 0.9) < 1e-12, num(ab.ratio.st.median));
    check("A/B: stops on the ratio's rel<=1% at min_samples pairs (10)", ab.ratio.stop_reason == "rel<=1%" && ab.pairs == 10,
          ab.ratio.stop_reason + " " + std::to_string(ab.pairs));
    check("A/B: per-side medians 10 / 9 us", ab.a.st.median == 10.0 && ab.b.st.median == 9.0);
    check("A/B: order effect AB pairs == BA pairs for a drift-free kernel", ab.order_ab == ab.order_ba && ab.order_ab == 0.9);
}

static void test_ab_order_bias() {
    // whichever runs first in a pair is 5 % faster (8060S order bias): the ratio median stays at the true 1.0 only because
    // both orders are equally represented; the order effect is visible
    FakeClock clk;
    Bench b(quietCfg(), &clk);
    std::vector<std::string> order;
    bool first = true;
    auto t = [&](const std::string&) {
        const double us = first ? 9.5 : 10.0;
        first = !first;
        return us;
    };
    FakeKernel ka{clk, nullptr}, kb{clk, nullptr};
    ka.us_per_launch = [&](int) { return t("A"); };
    kb.us_per_launch = [&](int) { return t("B"); };
    // warm-up and calibration call the sides separately: reset the parity right before pairs start is not possible from
    // outside, so check the summary only
    const AbStats ab = b.runAB(ka.fn(), kb.fn());
    check("A/B order bias: AB and BA pair medians differ (order effect visible)", std::fabs(ab.order_ab - ab.order_ba) > 0.05,
          num(ab.order_ab) + " " + num(ab.order_ba));
    check("A/B order bias: even number of pairs", ab.pairs % 2 == 0, std::to_string(ab.pairs));
}

// --------------------------------------------------------------------------------------------- burst rule
static void test_burst() {
    FakeClock clk;
    BenchCfg c;
    c.burst_s = 8;
    c.burst_idle_s = 60;
    c.max_samples = 1000000;  // stop on the 3 s time budget
    Bench b(c, &clk);
    // cells of ~3 s each (time budget, noisy kernel): 1st and 2nd fit in 8 s, the 3rd would not -> idle 60 s first
    FakeKernel k{clk, [](int i) { return i % 2 ? 12.0 : 10.0; }};
    b.beforeCell();
    b.run(k.fn());
    b.beforeCell();
    b.run(k.fn());
    const std::size_t sleeps_before = clk.sleeps.size();
    b.beforeCell();
    check("burst: no idle before cells 1 and 2 (busy ~6 s)", sleeps_before == 0, std::to_string(sleeps_before));
    check("burst: idle 60 s before a cell that would end past 8 s", clk.sleeps.size() == 1 && clk.sleeps[0] == 60.0 && b.bursts() == 1);
    check("burst: busy time reset after the idle", b.groupBusy() == 0.0, num(b.groupBusy()));
    BenchCfg off;
    off.burst_idle_s = 0;
    off.max_samples = 1000000;
    FakeClock clk2;
    Bench b2(off, &clk2);
    FakeKernel k2{clk2, [](int i) { return i % 2 ? 12.0 : 10.0; }};
    for (int i = 0; i < 4; ++i) {
        b2.beforeCell();
        b2.run(k2.fn());
    }
    check("burst: burst_idle_s 0 never sleeps, overruns counted", clk2.sleeps.empty() && b2.overruns() >= 1, std::to_string(b2.overruns()));
}

// --------------------------------------------------------------------------------------------- sha256, roof, rows
static void test_sha256() {
    check("sha256(\"\")", sha256Hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", sha256Hex("", 0));
    check("sha256(\"abc\")", sha256Hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";  // 56 bytes: two-block padding
    check("sha256(56-byte NIST vector)", sha256Hex(m.data(), m.size()) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          sha256Hex(m.data(), m.size()));
    std::string mil(1000000, 'a');
    check("sha256(10^6 x 'a')", sha256Hex(mil.data(), mil.size()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

static void test_roof() {
    const Roof r = roofFor("gfx1201"), s = roofFor("gfx1151");
    check("roof: R9700 640 GB/s, 130 TF/s practical f16", r.bw == 640e9 && r.f16_prac == 130e12);
    check("roof: 8060S 256 GB/s, 22 TF/s practical f16 (KLIB-1)", s.bw == 256e9 && s.f16_prac == 22e12);
    check("roof: unknown arch -> zeros", roofFor("gfx90a").bw == 0);
}

static std::string rowJson() {
    CellStats cs;
    cs.samples = {81.0, 82.0, 81.5, 80.5, 83.0};
    cs.st = whirl::bench::summarize(cs.samples);
    cs.stop_reason = "rel<=1%";
    cs.block = 4;
    cs.cooldowns = 1;
    cs.discarded = 4;
    cs.anchor_ref_us = 20;
    cs.anchor_min_us = 20;
    cs.anchor_max_us = 20.5;
    CellInfo ci;
    ci.family = "gemv";
    ci.kernel = "gemvq_mxfp4";
    ci.cell = "swift.blk.0.ffn_down.n1";
    ci.shape = "5120x17408";
    ci.settings = "state=burst;rot_bytes=536870912";
    ci.code_hash = "c0de";
    ci.gpu_arch = "gfx1151";
    ci.gpu_pci = "pci:196:0";
    ci.bytes = 47349760;  // 81.5 us -> 581 GB/s
    return whirl::bench::to_json(cellRow(ci, cs, "0.2.5-dev"));
}

static void test_rows() {
    const std::string j = rowJson();
    auto has = [&](const std::string& s) { return j.find(s) != std::string::npos; };
    check("row: metric kern.gemv.gemvq_mxfp4, unit us, value = median",
          has("\"metric\": \"kern.gemv.gemvq_mxfp4\"") && has("\"unit\": \"us\"") && has("\"value\": 81.5"), j);
    check("row: tool, component K-gemv, cell; block is an extra key, not in settings (adaptive)",
          has("\"tool\": \"whirl-kernel-test\"") && has("\"component\": \"K-gemv\"") && has("\"cell\": \"swift.blk.0.ffn_down.n1\"") &&
              has("\"block\": 4") && !has("block=") && has("\"settings\": \"state=burst;rot_bytes=536870912\""),
          j);
    check("row: stats n / median / p10 / p90 / mad and stop_reason", has("\"n\": 5") && has("\"median\": 81.5") && has("\"mad\": 0.5") &&
                                                                         has("\"stop_reason\": \"rel<=1%\""));
    check("row: anchor {ref, min, max, drift_pct, cooldowns}", has("\"anchor\": {\"ref\"") && has("\"cooldowns\": 1") && has("\"drift_pct\": 2.5"), j);
    check("row: gbps and pct_of_roof (bandwidth) as extra keys", has("\"gbps\": ") && has("\"pct_of_roof\": ") && has("\"roof\": \"bw\""), j);
    check("row: raw samples kept", has("\"samples\": [81, 82, 81.5, 80.5, 83]"), j);
    check("row: work trace launches per call", has("\"work\": {\"launches\": 1"), j);
    // gbps = 47349760 / 81.5e-6 / 1e9 = 580.97...
    const std::size_t p = j.find("\"gbps\": ");
    const double g = p == std::string::npos ? 0 : std::atof(j.c_str() + p + 8);
    check("row: gbps = bytes / median", std::fabs(g - 47349760.0 / 81.5e-6 / 1e9) < 1e-6, num(g));
    // a failed cell has no value -> null (the importer refuses it)
    CellStats bad;
    bad.stop_reason = "error";
    CellInfo ci;
    ci.family = "gemv";
    ci.kernel = "x";
    const std::string jb = whirl::bench::to_json(cellRow(ci, bad, "v"));
    check("row: error cell -> value null (importer refuses)", jb.find("\"value\": null") != std::string::npos, jb);
    // A/B row
    AbStats ab;
    ab.ratio.samples = {0.9, 0.91, 0.9, 0.89};
    ab.ratio.st = whirl::bench::summarize(ab.ratio.samples);
    ab.ratio.stop_reason = "rel<=1%";
    ab.pairs = 4;
    ab.order_ab = 0.9;
    ab.order_ba = 0.905;
    CellInfo ca;
    ca.family = "gemv";
    ca.kernel = "gemvq_mxfp4";
    const std::string ja = whirl::bench::to_json(abRow(ca, ab, "v", "b0b0"));
    check("ab row: metric .ab_ratio, unit ratio, code_hash_b, order effect", ja.find("kern.gemv.gemvq_mxfp4.ab_ratio") != std::string::npos &&
                                                                              ja.find("\"unit\": \"ratio\"") != std::string::npos &&
                                                                              ja.find("\"code_hash_b\": \"b0b0\"") != std::string::npos &&
                                                                              ja.find("\"order_ab\": 0.9") != std::string::npos,
          ja);
}

static void test_jsonl_file(const std::string& emit) {
    namespace fs = std::filesystem;
    const fs::path p = emit.empty() ? fs::temp_directory_path() / "kt_bench_tests.jsonl" : fs::path(emit);
    std::error_code ec;
    fs::remove(p, ec);
    {
        whirl::bench::JsonlWriter w(p.string(), false);
        CellStats cs;
        cs.samples = {10, 10.1, 9.9};
        cs.st = whirl::bench::summarize(cs.samples);
        cs.stop_reason = "time";
        cs.block = 32;
        CellInfo ci;
        ci.family = "attn";
        ci.kernel = "attn_wsplit1";
        ci.cell = "swift.L32768";
        ci.gpu_arch = "gfx1201";
        ci.gpu_pci = "pci:71:0";
        ci.bytes = 32768.0 * 4096;
        ci.settings = "state=burst";
        ci.started = "2026-10-25T14:00:00";
        check("jsonl: row written", w.write(cellRow(ci, cs, "0.2.5-dev")));
        check("jsonl: second row written", w.write(whirl::bench::Row{[] {
                  whirl::bench::Row r;
                  r.tool = "whirl-kernel-test";
                  r.metric = "kern.attn.attn_wsplit1";
                  r.unit = "us";
                  r.value = 1.0;
                  return r;
              }()}));
    }
    std::ifstream in(p);
    std::string line;
    int lines = 0;
    bool all_obj = true;
    while (std::getline(in, line)) {
        ++lines;
        all_obj = all_obj && !line.empty() && line.front() == '{' && line.back() == '}';
    }
    check("jsonl: two lines, each one JSON object", lines == 2 && all_obj, std::to_string(lines));
    in.close();
    if (emit.empty()) fs::remove(p, ec);
}

// ---------------------------------------------------------------- dq4s (v0252-8sq5): attn_dq4o split count S vs S_ship
static int opsSplits(const std::vector<int>& pos, int win) {
    std::vector<std::int32_t> p(pos.begin(), pos.end());
    return static_cast<int>(whirl::ops::dq4Splits(std::span<const std::int32_t>(p.data(), p.size()), win, 256));
}

static std::function<const char*(const char*)> fakeEnv(std::map<std::string, std::string> m) {
    auto mm = std::make_shared<std::map<std::string, std::string>>(std::move(m));
    return [mm](const char* n) -> const char* {
        const auto it = mm->find(n);
        return it == mm->end() ? nullptr : it->second.c_str();
    };
}

static void test_dq4s_mirror_is_ops_rule() {
    // the table's partition mirror reproduces ops::dq4Splits (so its same-partition / grid checks are the kernel's)
    std::size_t n = 0, bad = 0;
    std::string first;
    for (int L = 1; L <= 270000; L += (L < 4096 ? 37 : L < 70000 ? 997 : 4099))
        for (const int R : {1, 2, 3, 5, 9})
            for (const int win : {0, dq4s::kDraftWin}) {
                if (R > L) continue;
                std::vector<int> pos;
                for (int r = L - R; r < L; ++r) pos.push_back(r);
                const int a = dq4s::splitsMirror(pos, win, 256), b = opsSplits(pos, win);
                ++n;
                if (a != b) {
                    ++bad;
                    if (first.empty()) first = "L " + std::to_string(L) + " R " + std::to_string(R) + " win " + std::to_string(win) + ": " + std::to_string(a) + " vs " + std::to_string(b);
                }
            }
    check("dq4s: splitsMirror == ops::dq4Splits (" + std::to_string(n) + " cases)", n > 1000 && bad == 0, first);
    // the mirror's per / part at known points (src/ops/attention.cpp dq4Per: max(256, ceil(n / (min(ns,128) 64)) 64))
    check("dq4s: per(32768, 128) == 256", dq4s::per(32768, 128) == 256, num(dq4s::per(32768, 128)));
    check("dq4s: per(131072, 256) == per(131072, 128) == 1024", dq4s::per(131072, 256) == 1024 && dq4s::per(131072, 128) == 1024);
    check("dq4s: per(32768, 16) == 2048", dq4s::per(32768, 16) == 2048);
    const auto p = dq4s::part(131072, 0, 32);
    check("dq4s: part(131072, 0, 32) == {0, 4096, 32}", p[0] == 0 && p[1] == 4096 && p[2] == 32);
    const auto w = dq4s::part(131072, dq4s::kDraftWin, 32);
    check("dq4s: part with the draft window takes the window branch at 128k", w[0] == 1 && w[2] <= 32);
}

static void test_dq4s_grid_ok() {
    const std::vector<int> r2 = {131070, 131071};
    check("dq4s: gridOk 8..128 at 128k, 2 rows", dq4s::gridOk(r2, 0, 8) && dq4s::gridOk(r2, 0, 128));
    check("dq4s: gridOk rejects 0 and > 256", !dq4s::gridOk(r2, 0, 0) && !dq4s::gridOk(r2, 0, 257));
    check("dq4s: gridOk rejects < 8 with the window on", !dq4s::gridOk({131071}, dq4s::kDraftWin, 4) && dq4s::gridOk({131071}, dq4s::kDraftWin, 8));
    std::vector<int> r33;
    for (int r = 0; r < 33; ++r) r33.push_back(100000 + r);
    check("dq4s: gridOk rejects 33 rows", !dq4s::gridOk(r33, 0, 64));
    check("dq4s: samePartition(8192: 32 / 64 / 128) and not 16", dq4s::samePartition({8191}, 0, 32, 64) && dq4s::samePartition({8191}, 0, 32, 128) &&
                                                                     !dq4s::samePartition({8191}, 0, 32, 16));
}

static void test_dq4s_default_table() {
    std::string err;
    const dq4s::Opts o = dq4s::optsFromEnv(fakeEnv({}), &err);
    check("dq4s: default opts parse", err.empty(), err);
    check("dq4s: default models ornith, swift", o.models == std::vector<std::string>{"ornith", "swift"});
    check("dq4s: default keys 8192 .. 131072", o.keys == std::vector<int>{8192, 16384, 32768, 49152, 65536, 98304, 131072});
    check("dq4s: default splits 8 .. 128", o.splits == std::vector<int>{8, 16, 32, 64, 128});
    std::vector<std::string> tags;
    for (const auto& rs : o.rowsets) tags.push_back(rs.tag);
    check("dq4s: default row sets 1 2 3 d1", tags == std::vector<std::string>{"1", "2", "3", "d1"});
    check("dq4s: default no A/A cells", !o.aa);
    const std::vector<dq4s::Cell> cs = dq4s::cells(o);
    std::set<std::string> names;
    std::size_t bad_ship = 0, bad_grid = 0, bad_pos = 0, bad_name = 0, bad_aa = 0, bad_shape = 0, bad_draft = 0;
    for (const dq4s::Cell& c : cs) {
        names.insert(c.name);
        bad_ship += c.s_ship != opsSplits(c.pos, c.win);
        bad_grid += !dq4s::gridOk(c.pos, c.win, c.s) || !dq4s::gridOk(c.pos, c.win, c.s_ship);
        bool pos_ok = static_cast<int>(c.pos.size()) == c.rows;
        for (int r = 0; r < c.rows && pos_ok; ++r) pos_ok = c.pos[static_cast<std::size_t>(r)] == c.keys - c.rows + r;
        bad_pos += !pos_ok;
        const std::string want = c.model + ".q4kv.L" + std::to_string(c.keys) + ".rows" + c.rows_tag + ".s" + std::to_string(c.s) + "_vs_s" + std::to_string(c.s_ship);
        bad_name += c.name != want;
        bad_aa += c.s == c.s_ship || c.same_partition || dq4s::samePartition(c.pos, c.win, c.s, c.s_ship);
        bad_shape += !((c.model == "ornith" && c.heads == 16 && c.nkv == 2) || (c.model == "swift" && c.heads == 24 && c.nkv == 4));
        bad_draft += c.rows_tag == "d1" ? !(c.win == dq4s::kDraftWin && c.rows == 1 && c.keys >= dq4s::kDraftMinKeys) : c.win != 0;
    }
    check("dq4s: default table has cells", cs.size() > 50, num(static_cast<double>(cs.size())));
    check("dq4s: cell names unique", names.size() == cs.size());
    check("dq4s: S_ship of every cell == ops::dq4Splits(rows, win, 256)", bad_ship == 0, num(static_cast<double>(bad_ship)));
    check("dq4s: no cell with a grid the kernel rejects", bad_grid == 0);
    check("dq4s: rows at L - R .. L - 1", bad_pos == 0);
    check("dq4s: cell name <model>.q4kv.L<L>.rows<tag>.s<S>_vs_s<S_ship>", bad_name == 0);
    check("dq4s: default cells all differ from S_ship's partition (no A/A)", bad_aa == 0);
    check("dq4s: shapes Ornith 16/2, Swift 24/4", bad_shape == 0);
    check("dq4s: d1 = one row, the default draft window, keys >= 65536; others no window", bad_draft == 0);
    // pins: today's grid at the operating points (ops rule) and the cells made there
    auto find = [&](const std::string& n) { return std::any_of(cs.begin(), cs.end(), [&](const dq4s::Cell& c) { return c.name == n; }); };
    check("dq4s: ornith 32768 rows1 S_ship 128, S 8..64 timed", find("ornith.q4kv.L32768.rows1.s8_vs_s128") && find("ornith.q4kv.L32768.rows1.s64_vs_s128") &&
                                                                    !find("ornith.q4kv.L32768.rows1.s128_vs_s128"));
    check("dq4s: ornith 8192 rows1 S_ship 32: S 8 / 16 only (64 / 128 same partition)", find("ornith.q4kv.L8192.rows1.s8_vs_s32") &&
                                                                                          find("ornith.q4kv.L8192.rows1.s16_vs_s32") &&
                                                                                          !find("ornith.q4kv.L8192.rows1.s64_vs_s32"));
    check("dq4s: ornith 131072 rows2 and the draft row", find("ornith.q4kv.L131072.rows2.s32_vs_s128") && find("ornith.q4kv.L131072.rowsd1.s32_vs_s128"));
    check("dq4s: no draft row below 65536", !find("ornith.q4kv.L32768.rowsd1.s8_vs_s128"));
    // run order: model, keys, row set, S ascending
    bool ordered = true;
    for (std::size_t i = 1; i < cs.size(); ++i) {
        const auto& a = cs[i - 1];
        const auto& b = cs[i];
        if (a.model == b.model && a.keys == b.keys && a.rows_tag == b.rows_tag) ordered = ordered && a.s < b.s;
        if (a.model == b.model && a.keys > b.keys) ordered = false;
    }
    check("dq4s: run order model / keys / rows / S", ordered);
}

static void test_dq4s_env_filters() {
    std::string err;
    dq4s::Opts o = dq4s::optsFromEnv(fakeEnv({{"WHIRL_KT_DQ4S_MODELS", "ornith"}, {"WHIRL_KT_DQ4S_L", "32768,131072"}, {"WHIRL_KT_DQ4S_ROWS", "2 d1"},
                                              {"WHIRL_KT_DQ4S_S", "16,64"}}),
                                     &err);
    check("dq4s env: filters parse", err.empty(), err);
    const auto cs = dq4s::cells(o);
    std::set<std::string> got;
    for (const auto& c : cs) got.insert(c.name);
    const std::set<std::string> want = {"ornith.q4kv.L32768.rows2.s16_vs_s128", "ornith.q4kv.L32768.rows2.s64_vs_s128", "ornith.q4kv.L131072.rows2.s16_vs_s128",
                                        "ornith.q4kv.L131072.rows2.s64_vs_s128", "ornith.q4kv.L131072.rowsd1.s16_vs_s128", "ornith.q4kv.L131072.rowsd1.s64_vs_s128"};
    std::string g;
    for (const auto& n : got) g += n + " ";
    check("dq4s env: exactly the filtered cells", got == want, g);
    o = dq4s::optsFromEnv(fakeEnv({{"WHIRL_KT_DQ4S_MODELS", "swift"}, {"WHIRL_KT_DQ4S_L", "65536"}, {"WHIRL_KT_DQ4S_ROWS", "1"}, {"WHIRL_KT_DQ4S_AA", "1"}}), &err);
    const auto ca = dq4s::cells(o);
    const bool has_aa = std::any_of(ca.begin(), ca.end(), [](const dq4s::Cell& c) { return c.s == c.s_ship && c.same_partition; });
    const bool only_swift = std::all_of(ca.begin(), ca.end(), [](const dq4s::Cell& c) { return c.model == "swift" && c.heads == 24 && c.nkv == 4; });
    check("dq4s env: AA=1 adds the S == S_ship control cell", err.empty() && o.aa && has_aa && only_swift);
    for (const auto& kv : std::vector<std::pair<std::string, std::string>>{{"WHIRL_KT_DQ4S_MODELS", "llama"},
                                                                         {"WHIRL_KT_DQ4S_L", "abc"},
                                                                         {"WHIRL_KT_DQ4S_L", "0"},
                                                                         {"WHIRL_KT_DQ4S_ROWS", "x"},
                                                                         {"WHIRL_KT_DQ4S_ROWS", "33"},
                                                                         {"WHIRL_KT_DQ4S_S", "-8"},
                                                                         {"WHIRL_KT_DQ4S_S", "512"}}) {
        std::string e2;
        (void)dq4s::optsFromEnv(fakeEnv({{kv.first, kv.second}}), &e2);
        check("dq4s env: " + kv.first + "=" + kv.second + " is an error", !e2.empty());
    }
}

// v0252-8sq5 round 2: the reference stays today's rule (ops::dq4Splits) when the 8060S default becomes the profile table;
// WHIRL_KT_DQ4S_TABLE=1 times the table's count (ops::dq4TableSplits on the gfx1151 row) against it, one cell per (keys, rows)
static void test_dq4s_table_cells() {
    std::string err;
    const dq4s::Opts o = dq4s::optsFromEnv(fakeEnv({{"WHIRL_KT_DQ4S_TABLE", "1"}, {"WHIRL_KT_DQ4S_MODELS", "ornith"}}), &err);
    check("dq4s table: opts parse", err.empty() && o.table, err);
    const auto cs = dq4s::cells(o);
    const auto& tab = whirl::device::k_profile_gfx1151.dq4_split_table;
    std::size_t bad = 0, n = 0;
    std::set<std::string> keyrows;
    for (const auto& c : cs) {
        std::vector<std::int32_t> p(c.pos.begin(), c.pos.end()), z(p.size(), 0);
        const int t = static_cast<int>(whirl::ops::dq4TableSplits(std::span<const std::int32_t>(p), std::span<const std::int32_t>(z), c.win, 256,
                                                                   whirl::ops::kDq4SplitsAuto, tab));
        bad += c.s != t || c.s_ship != opsSplits(c.pos, c.win) || c.name.size() < 6 || c.name.substr(c.name.size() - 6) != ".table";
        keyrows.insert(std::to_string(c.keys) + "/" + c.rows_tag);
        ++n;
    }
    // 7 keys x rows 1 / 2 / 3 + 3 draft-row keys = 24 (Ornith), every one with a table count other than today's partition
    check("dq4s table: one cell per (keys, rows), s = the gfx1151 table, S_ship = ops::dq4Splits", n == 24 && keyrows.size() == 24 && bad == 0,
          num(static_cast<double>(n)) + " cells, " + num(static_cast<double>(bad)) + " bad");
    check("dq4s table: ornith 32768 rows2 -> s8 vs s128", std::any_of(cs.begin(), cs.end(), [](const dq4s::Cell& c) { return c.name == "ornith.q4kv.L32768.rows2.s8_vs_s128.table"; }));
    check("dq4s: gridOk is ops::dq4GridOk (+ rows bound)", dq4s::gridOk({32767}, 0, 16) == whirl::ops::dq4GridOk(std::span<const std::int32_t>(std::vector<std::int32_t>{32767}), 0, 16, 256));
}

static void test_dq4s_geo_ci() {
    check("dq4s: t975(1) 12.706, (10) 2.228, (30) 2.042, large -> 1.96", std::fabs(dq4s::t975(1) - 12.706) < 1e-3 && std::fabs(dq4s::t975(10) - 2.228) < 1e-3 &&
                                                                       std::fabs(dq4s::t975(30) - 2.042) < 1e-3 && std::fabs(dq4s::t975(100000) - 1.96) < 1e-3);
    const dq4s::GeoCi c = dq4s::geoCi({0.9, 0.9, 0.9, 0.9});
    check("dq4s: geoCi of constant 0.9", std::fabs(c.g - 0.9) < 1e-12 && std::fabs(c.lo - 0.9) < 1e-12 && std::fabs(c.hi - 0.9) < 1e-12 && c.n == 4);
    const dq4s::GeoCi d = dq4s::geoCi({0.9, 1.1});
    const double m = 0.5 * (std::log(0.9) + std::log(1.1)), sd = std::fabs(std::log(1.1) - std::log(0.9)) / std::sqrt(2.0), h = 12.706 * sd / std::sqrt(2.0);
    check("dq4s: geoCi of {0.9, 1.1}", std::fabs(d.g - std::exp(m)) < 1e-9 && std::fabs(d.lo - std::exp(m - h)) < 1e-3 && std::fabs(d.hi - std::exp(m + h)) < 1e-2,
          num(d.lo) + " " + num(d.hi));
    const dq4s::GeoCi one = dq4s::geoCi({1.2});
    check("dq4s: geoCi of one ratio: lo = hi = g", one.n == 1 && one.lo == one.g && one.hi == one.g && std::fabs(one.g - 1.2) < 1e-12);
    const std::string line = dq4s::geoLine(dq4s::geoCi({0.95, 0.96, 0.94, 0.95}));
    check("dq4s: judge line", line.find("B/A geometric mean 0.95") != std::string::npos && line.find("95% interval [") != std::string::npos &&
                                  line.find("B time -5.0") != std::string::npos && line.find("4 pairs") != std::string::npos,
          line);
}

int main(int argc, char** argv) {
    std::string emit;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--emit") == 0 && i + 1 < argc) emit = argv[++i];
    try {
        test_constant_kernel();
        test_calibration_slow_kernel();
        test_time_stop();
        test_n_stop();
        test_rel_after_settle();
        test_error();
        test_event_protocol();
        test_event_reuse_would_fail();
        test_event_invalid_short_blocks();
        test_event_longer_than_host();
        test_occasional_invalid_dropped();
        test_rotation();
        test_anchor_drift_once();
        test_anchor_noisy();
        test_anchor_cooldown_cap();
        test_anchor_faster_becomes_reference();
        test_rewarm_after_cooldown();
        test_cache_state_anchor_switched_off();
        test_ab_order_and_ratio();
        test_ab_order_bias();
        test_burst();
        test_sha256();
        test_roof();
        test_rows();
        test_jsonl_file(emit);
        test_dq4s_mirror_is_ops_rule();
        test_dq4s_grid_ok();
        test_dq4s_default_table();
        test_dq4s_env_filters();
        test_dq4s_geo_ci();
        test_dq4s_table_cells();
    } catch (const std::exception& e) {
        check(std::string("exception: ") + e.what(), false);
    }
    std::printf("whirl-kt-bench-tests: %d PASS, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
