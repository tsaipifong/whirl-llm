// Runtime watch of this process's WDDM video memory (issues #5 / #6: decode slows down in long
// agent sessions; the suspect is Windows moving GPU buffers to system RAM). The server samples
// hip::wddmMemInfo() for the periodic "batch |" status line and at each request end (never per decode
// cycle), warns in plain words while the GPU memory spills, says when the spill is over, and reports
// a sharp drop of a request's decode speed, naming the spill as the likely cause only while one is seen.
// SPDX-License-Identifier: Apache-2.0
//
// Header free of HIP: formatting, thresholds and the rate limits are unit-tested on the CPU with
// injected values.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace whirl::wddmwatch {

// one sample: the fields of hip::WddmMemInfo the watch needs, plus this process's RAM
// (ok = false: no WDDM numbers; rss / pinned 0 = not known, left out of the text)
struct Sample {
    bool ok = false;
    std::uint64_t local_budget = 0, local_usage = 0, nonlocal_usage = 0;
    std::uint64_t rss = 0;     // process working set
    std::uint64_t pinned = 0;  // pinned host memory WHIRL allocated (hip::hostMalloc)
};

// "vram local 29.10 / budget 30.50 GiB, shared 412 MiB (+12 MiB since start), ram 18.20 GiB (pinned 9.10 GiB)";
// the growth is the non-local usage against `base`. "" when s is not ok.
std::string format(const Sample& s, const Sample& base);

struct Config {
    std::uint64_t grow_warn = 256ull << 20;  // non-local growth since start that counts as a spill
    double min_gap_s = 60.0;                 // at most one spill warning per this many seconds
};

struct Event {
    bool warn = false;  // true: a WARNING (spill), false: INFO (spill over)
    std::string text;
};

// Spill = local usage over the budget, or non-local usage grown by more than grow_warn since the
// baseline. While it lasts a WARNING is printed when it starts and again at most once per min_gap_s
// (with how long it has lasted); when it ends after a warning, one INFO line says so.
class Watch {
public:
    explicit Watch(Sample base = {}, Config c = {}) : base_(base), cfg_(c) {}
    // a line to log, or nullopt; now_s is a monotonic clock in seconds
    std::optional<Event> check(const Sample& s, double now_s);
    const Sample& baseline() const { return base_; }
    // the last sample checked was a spill
    bool spilling() const { return spilling_; }
    std::uint32_t warnings() const { return n_warn_; }

private:
    Sample base_;
    Config cfg_;
    bool spilling_ = false;
    bool warned_ = false;       // a warning was printed for the current spill
    bool ever_warned_ = false;
    double warn_t_ = 0.0;       // time of the last warning (any spill)
    double spill_t0_ = 0.0;     // start of the current spill
    std::uint32_t n_warn_ = 0;
};

// Decode speed of one request, from the cumulative numbers of its periodic "gen:" report. The early
// average is the speed over the first early_tok tokens; each later report's window (since the
// previous report) is compared with it, and a window slower by more than drop_frac produces one line
// per request. cycles 0: no per-cycle figure.
class DecodeWatch {
public:
    struct Config {
        std::uint32_t early_tok = 200;
        double drop_frac = 0.40;
        double min_window_ms = 2000.0;  // shorter windows are too noisy to judge
    };
    DecodeWatch() = default;
    explicit DecodeWatch(Config c) : cfg_(c) {}
    void reset() { *this = DecodeWatch(cfg_); }
    // n_gen, ms (since decode start) and cycles are cumulative; spill: a GPU memory spill is seen now.
    // Returns the line (without the "req / slot" prefix) and whether it is a warning.
    std::optional<Event> report(std::uint32_t n_gen, double ms, std::uint64_t cycles, bool spill);

private:
    Config cfg_;
    bool have_early_ = false, fired_ = false;
    double early_tps_ = 0.0, early_ms_cyc_ = 0.0;
    std::uint32_t prev_n_ = 0;
    double prev_ms_ = 0.0;
    std::uint64_t prev_cyc_ = 0;
};

}  // namespace whirl::wddmwatch
