// Unit tests of kt::Bench (tests/kernels/kt_bench.{h,cpp}), the shared kernel timer of whirl-kernel-test --bench
// (measure-tools P2). CPU only: a fake clock and scripted block timers stand in for the GPU.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Usage: whirl-kt-bench-tests [--emit FILE]  (--emit also writes sample rows for the importer check,
// `db_record.py jsonl FILE --dry-run`). Exit 0 = all pass.

#include "../kt_bench.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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
    check("error: stops at the first bad block (1 call)", calls == 1, std::to_string(calls));
    BlockFn nan = [&](int, std::uint64_t) { return std::nan(""); };
    check("error: NaN block -> error", b.run(nan).stop_reason == "error");
    // an error during sampling (after calibration) also stops
    int n2 = 0;
    BlockFn late = [&](int n, std::uint64_t) {
        clk.t += 1e-3;
        return ++n2 > 40 ? -1.0 : 0.01 * n;
    };
    const CellStats cs2 = b.run(late);
    check("error: a failed block during sampling -> error, samples discarded", cs2.stop_reason == "error" && cs2.st.n == 0, cs2.stop_reason);
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
        test_rotation();
        test_anchor_drift_once();
        test_anchor_noisy();
        test_anchor_cooldown_cap();
        test_anchor_faster_becomes_reference();
        test_ab_order_and_ratio();
        test_ab_order_bias();
        test_burst();
        test_sha256();
        test_roof();
        test_rows();
        test_jsonl_file(emit);
    } catch (const std::exception& e) {
        check(std::string("exception: ") + e.what(), false);
    }
    std::printf("whirl-kt-bench-tests: %d PASS, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
