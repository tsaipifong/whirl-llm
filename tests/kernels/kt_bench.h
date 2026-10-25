// whirl-kernel-test --bench: the shared kernel timer kt::Bench (measure-tools P2; progress\measure-tools.DESIGN.md A1, (d)).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. No HIP in this header or in kt_bench.cpp: the GPU side is a BlockFn (kt_bench_hip.cpp times a
// block of launches between two hipEvents) and the wall clock / sleep is a Clock, so the rules below are unit-tested on
// the CPU with a fake timer (tests\kernels\unit\kt_bench_tests.cpp, target whirl-kt-bench-tests).
//
// One cell = one kernel at one shape:
//   1. warm-up: >= warmup_ms of GPU time in single launches (clock ramp, lazy allocations), untimed;
//   2. calibration: the block (launches per timed sample) doubles until one block takes >= min_block_us
//      (torch blocked_autorange; one launch per event pair would time launch overhead);
//   3. samples: one sample = one block / launches (us per launch). Stop when n >= min_samples and IQR / median
//      <= rel_target ("rel<=1%"), or after max_cell_s of measuring ("time"), or at max_samples ("n");
//   4. anchor (optional): every anchor_every samples a fixed anchor block is re-timed; slower than the coolest anchor
//      seen by > drift_pct -> cooldown (sleep cool_first_s, doubling to cool_max_step_s, <= cool_total_s, until the
//      anchor is back) and the samples since the last clean anchor are discarded; more than max_cooldowns cooldowns
//      in one cell -> stop_reason "noisy" (the row is written but never used as a baseline). A faster anchor becomes
//      the new reference (llama.cpp tools/tuning: what was measured before it ran on a hotter machine);
//   5. rotation: the BlockFn gets the index of the first launch; launch i uses input copy i % copies, so the rotated
//      working set exceeds the GPU's last-level cache (rotationCopies);
//   6. A/B (runAB): A and B alternate per sample pair in ABBA order (pairs AB, BA, AB, BA: launches A B B A A B B A); the judged number is the
//      paired ratio B / A with the same stop rule; the order effect (median ratio of AB pairs vs BA pairs) is kept.
// Burst state (DESIGN (d), Z13 FPPT 10 s): beforeCell() idles burst_idle_s before a cell that would push the busy time
// since the last idle past burst_s (the runner idles 60-90 s before the process starts).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../bench/bench_out.h"

namespace kt {

struct BenchCfg {
    double warmup_ms = 100;      // untimed warm-up, GPU time
    // calibrated block >= this. 1000, not torch's 200: the R9700 smoke (2026-10-25 14:15) returned zero / negative
    // hipEvent intervals for blocks of tens to ~200 us of work, while every block >= ~870 us was valid
    double min_block_us = 1000;
    int max_block = 1 << 16;
    int min_samples = 10, max_samples = 2000;
    double rel_target = 0.01;    // IQR / median
    double max_cell_s = 3.0;     // measuring wall time per cell (cooldown sleeps excluded)
    int anchor_every = 4;        // samples (A/B: pairs) between anchor re-timings
    double drift_pct = 3.0;
    double cool_first_s = 2, cool_max_step_s = 32, cool_total_s = 60;
    int max_cooldowns = 2;       // more -> "noisy"
    int max_invalid_pct = 5;
    int noisy_streak_off = 3;    // this many noisy cells in a row -> the anchor is switched off (no more cooldowns)     // invalid timer readings tolerated while sampling (dropped, counted); more -> "error"
    double burst_s = 8, burst_idle_s = 0;  // burst_idle_s 0 = off
};

// wall clock and sleep (fake in the unit tests)
class Clock {
public:
    virtual ~Clock() = default;
    virtual double now() = 0;  // seconds, monotonic
    virtual void sleep(double s) = 0;
};
Clock& realClock();

// Times `launches` consecutive launches whose first launch has rotation index `rot0` (launch j uses rot0 + j); returns
// the GPU time of the whole block in ms. <= 0 or non-finite = nothing was timed (an error, no row).
using BlockFn = std::function<double(int launches, std::uint64_t rot0)>;

struct CellStats {
    whirl::bench::Stats st;       // of `samples`, us per launch
    std::vector<double> samples;  // kept samples (after discards), us per launch
    std::string stop_reason;      // "rel<=1%", "time", "n", "noisy", "error"
    std::string error;            // stop_reason "error": why
    int block = 0;                // launches per sample
    int discarded = 0;            // samples dropped after an anchor drift
    int invalid = 0;              // timer readings rejected (zero / negative / longer than the host wall time) and dropped
    int cooldowns = 0;
    double cooldown_s = 0;        // slept
    double measure_s = 0;         // wall time of warm-up + calibration + samples (no cooldowns)
    double warm_ms = 0;           // GPU time of the untimed warm-up
    double anchor_ref_us = 0, anchor_min_us = 0, anchor_max_us = 0;  // per launch; 0 = no anchor
    std::string anchor_name;
    double rel() const;           // IQR / median of samples
    bool ok() const { return stop_reason != "error" && st.n > 0; }
};

struct AbStats {
    CellStats a, b;               // per side (samples in pair order)
    CellStats ratio;              // samples = B / A per pair; its stop_reason is the run's stop reason
    double order_ab = 0, order_ba = 0;  // median ratio of pairs run A first / B first
    int pairs = 0;
};

class Bench {
public:
    explicit Bench(BenchCfg cfg = {}, Clock* clock = nullptr);
    const BenchCfg& cfg() const { return cfg_; }
    BenchCfg& cfg() { return cfg_; }

    // optional anchor kernel (DESIGN (d): one GEMV of a real Swift matrix); calibrated on first use
    void setAnchor(BlockFn f, std::string name);
    const std::string& anchorName() const { return anchor_name_; }
    bool hasAnchor() const { return static_cast<bool>(anchor_); }
    bool anchorSwitchedOff() const { return anchor_off_; }  // after cfg.noisy_streak_off noisy cells in a row
    const std::string& anchorError() const { return anchor_error_; }  // setAnchor failed: why
    double anchorRefUs() const { return anchor_ref_us_; }

    void beforeCell();  // burst rule (idle when needed)
    CellStats run(const BlockFn& f);
    AbStats runAB(const BlockFn& a, const BlockFn& b);

    int bursts() const { return bursts_; }        // idles taken
    int overruns() const { return overruns_; }    // groups that still ran past burst_s
    double groupBusy() const { return busy_; }

private:
    struct Cal {
        int block = 0;
        std::string error;
    };
    Cal warmCalibrate(const BlockFn& f, std::uint64_t& rot);
    double sample(const BlockFn& f, int block, std::uint64_t& rot, std::string& err);  // us per launch, <= 0 error
    // anchor check: false = drifted (a cooldown was taken; caller discards); sets noisy when over the limit
    bool anchorCheck(CellStats& cs, bool& noisy);
    double anchorTime();
    void account(double busy_s);
    void warmFor(const BlockFn& f, int block, std::uint64_t& rot, double ms);  // re-warm after a sleep (clock ramp)
    void cellEnd(const std::string& stop_reason);

    BenchCfg cfg_;
    Clock* clock_;
    BlockFn anchor_;
    std::string anchor_name_, anchor_error_;
    double last_warm_ms_ = 0;
    int anchor_block_ = 0;
    std::uint64_t anchor_rot_ = 0;
    double anchor_ref_us_ = 0;
    double busy_ = 0, last_cell_s_ = 0;
    int bursts_ = 0, overruns_ = 0;
    bool overrun_counted_ = false;
    int noisy_streak_ = 0;
    bool anchor_off_ = false;
};

// ---------------------------------------------------------------- event timing of one block (HIP-free protocol)
// The GPU timer behind a BlockFn, written against an abstract event API so its protocol is unit-tested on the CPU:
//   per block a FRESH event pair (never re-recorded events), e0 and e1 recorded on the same stream the calls launch on
//   (the stream is passed to every call), e1 synchronized and then the stream, elapsed(e0, e1) read, both destroyed.
// A reading is valid when 0 < ms <= host wall time of the block + 0.5 ms (a GPU interval cannot be longer than the host
// interval that encloses its submission and completion); an invalid reading is retried once with new events.
struct EventApi {
    std::function<void*()> create;
    std::function<void(void*)> destroy;
    std::function<void(void* ev, void* stream)> record;
    std::function<void(void* ev)> sync;
    std::function<void(void* stream)> stream_sync;
    std::function<double(void* e0, void* e1)> elapsed_ms;
};
struct TimerStats {
    std::int64_t blocks = 0, retries = 0, invalid = 0;  // invalid = still invalid after the retry
    std::string last_invalid;                            // the last rejected reading, for the error text
};
using StreamCall = std::function<void(std::uint64_t rot, void* stream)>;
BlockFn eventBlock(EventApi api, void* stream, StreamCall call, Clock* clock = nullptr, std::shared_ptr<TimerStats> stats = nullptr);

// IQR / median of a sample set (0 for < 2 samples or median 0)
double relIqr(std::vector<double> v);

// copies of a working set so copies * ws >= target_bytes, at most cap_bytes in total and max_copies; >= 1
int rotationCopies(std::uint64_t ws_bytes, std::uint64_t target_bytes, std::uint64_t cap_bytes, int max_copies = 4096);

// DESIGN A1 rotation target: > 512 MB on the R9700 (64 MB IC), > 2 GB on the 8060S (32 MB MALL)
std::uint64_t rotationTarget(const std::string& gcn_arch);

// Hardware ceilings (tools\quick\sol.py HW, audit\KLIB-1.md): bandwidth peak / practical, f16 peak / practical.
struct Roof {
    double bw = 0, bw_prac = 0, f16 = 0, f16_prac = 0;
};
Roof roofFor(const std::string& gcn_arch);

std::string sha256Hex(const void* data, std::size_t n);

// One JSONL row for a cell (metric kern.<family>.<kernel>, unit us, value = median). bytes / flops = work per launch
// (0 = unknown); they add gbps / tflops and pct_of_roof (bandwidth for bytes, practical f16 for flops) as extra keys.
struct CellInfo {
    std::string family, kernel, cell, shape, settings, code_hash, gpu_arch, gpu_pci, started, variant;
    double bytes = 0, flops = 0;
    std::int64_t launches_per_call = 1;
};
whirl::bench::Row cellRow(const CellInfo& ci, const CellStats& cs, const std::string& tool_ver);
// the paired ratio row of an A/B (metric kern.<family>.<kernel>.ab_ratio, unit ratio)
whirl::bench::Row abRow(const CellInfo& ci, const AbStats& ab, const std::string& tool_ver, const std::string& code_hash_b);

// ---------------------------------------------------------------- whirl-kernel-test --bench (bench_kernels.cpp)
struct BenchOpts {
    std::set<std::string> families;      // gemv gemvh gemm attn moe gdn
    std::string jsonl;                   // --jsonl FILE (appended)
    std::string ab_hsaco;                // --ab HSACO
    BenchCfg cfg;
    std::uint64_t rot_target = 0;        // --rot-mb (0: rotationTarget(arch))
    std::uint64_t rot_cap = 4ull << 30;  // --rot-cap-mb: device memory per rotated set
    int moe_layers = 2;                  // --moe-layers: real expert layers rotated (with 64 routing sets)
    bool no_anchor = false;              // --no-anchor
    std::string state = "burst";         // declared power state (DESIGN (d))
    std::string tool_ver, code_hash, code_hash_b, gpu_arch, gpu_pci;
};
extern const char* const kBenchFamilies[7];  // feat026-argmax: + "argmax" (keep in step with bench_kernels.cpp)
bool benchFamilyKnown(const std::string& f);
struct Ctx;
void runBench(Ctx& c, const BenchOpts& o);

}  // namespace kt
