// Runtime WDDM video memory watch (see include/whirl/wddm_watch.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/wddm_watch.h"

#include <format>

namespace whirl::wddmwatch {

namespace {

constexpr double gib = 1073741824.0;
constexpr double mib = 1048576.0;

// non-local growth since the baseline that is not pinned host memory WHIRL allocated since then
// (pinned host buffers count as non-local too; a vision encoder or a checkpoint made later is not a spill)
std::uint64_t spillGrowth(const Sample& s, const Sample& base) {
    const std::uint64_t g = s.nonlocal_usage > base.nonlocal_usage ? s.nonlocal_usage - base.nonlocal_usage : 0;
    const std::uint64_t p = s.pinned > base.pinned ? s.pinned - base.pinned : 0;
    return g > p ? g - p : 0;
}

std::string duration(double s) {
    return s < 120.0 ? std::format("{:.0f} s", s) : std::format("{:.0f} min", s / 60.0);
}

}  // namespace

std::string format(const Sample& s, const Sample& base) {
    if (!s.ok) return {};
    std::string r = std::format("vram local {:.2f} / budget {:.2f} GiB, shared {:.0f} MiB", s.local_usage / gib,
                                s.local_budget / gib, s.nonlocal_usage / mib);
    if (base.ok) {
        const double d = (static_cast<double>(s.nonlocal_usage) - static_cast<double>(base.nonlocal_usage)) / mib;
        // "-0" would read as a change
        r += std::format(" ({}{:.0f} MiB since start)", d < -0.5 ? "-" : "+", d < -0.5 ? -d : (d < 0.5 ? 0.0 : d));
    }
    if (s.rss > 0) {
        r += std::format(", ram {:.2f} GiB", s.rss / gib);
        if (s.pinned > 0) r += std::format(" (pinned {:.2f} GiB)", s.pinned / gib);
    } else if (s.pinned > 0) {
        r += std::format(", pinned ram {:.2f} GiB", s.pinned / gib);
    }
    return r;
}

std::optional<Event> Watch::check(const Sample& s, double now_s) {
    if (!s.ok || !base_.ok) return std::nullopt;
    const bool over = s.local_usage > s.local_budget;
    const std::uint64_t grow = spillGrowth(s, base_);
    const bool grown = grow > cfg_.grow_warn;
    const bool spill = over || grown;
    const bool was = spilling_;
    spilling_ = spill;
    if (!spill) {
        if (!was || !warned_) return std::nullopt;
        warned_ = false;
        return Event{false, std::format("GPU memory spill is over (it lasted {}): WHIRL's GPU memory is back under the Windows "
                                        "video memory budget and its shared GPU memory is back within {:.0f} MiB of startup, so "
                                        "decode is no longer slowed by it. Now: {}",
                                        duration(now_s - spill_t0_), cfg_.grow_warn / mib, format(s, base_))};
    }
    if (!was) {
        spill_t0_ = now_s;
        warned_ = false;
    }
    if (ever_warned_ && now_s - warn_t_ < cfg_.min_gap_s) return std::nullopt;
    const bool repeat = warned_;
    warned_ = ever_warned_ = true;
    warn_t_ = now_s;
    ++n_warn_;
    // (1) what happened
    std::string what;
    if (over && grown)
        what = std::format("WHIRL's GPU memory is {:.2f} GiB over the Windows video memory (WDDM) budget, and Windows has moved "
                           "part of WHIRL's GPU buffers to system RAM (shared GPU memory +{:.0f} MiB since startup).",
                           (s.local_usage - s.local_budget) / gib, grow / mib);
    else if (over)
        what = std::format("WHIRL's GPU memory is {:.2f} GiB over the Windows video memory (WDDM) budget, so Windows can move "
                           "part of WHIRL's GPU buffers to system RAM at any time.",
                           (s.local_usage - s.local_budget) / gib);
    else
        what = std::format("WHIRL's shared GPU memory grew by {:.0f} MiB since startup: Windows has moved part of WHIRL's GPU "
                           "buffers to system RAM (Windows may also have lowered the video memory budget because other "
                           "programs use the GPU).",
                           grow / mib);
    // (2) the consequence, (3) what to do, (4) the numbers
    const char* effect = grown ? "Generation is getting slower now: decode can become several times slower."
                               : "Generation gets slower as soon as it does: decode can become several times slower.";
    std::string head = repeat ? std::format("GPU memory spill continues ({} so far): ", duration(now_s - spill_t0_))
                              : std::string("GPU memory spill: ");
    return Event{true, std::format("{}{} {} What you can do: close other programs that use the GPU (browsers, games, other AI "
                                   "tools); or restart the server with a smaller -c, with --parallel 1, or with "
                                   "WHIRL_POOL_RESERVE_MB=3072 (or larger). Now: {}",
                                   head, what, effect, format(s, base_))};
}

std::optional<Event> DecodeWatch::report(std::uint32_t n_gen, double ms, std::uint64_t cycles, bool spill) {
    if (!have_early_) {
        if (n_gen < cfg_.early_tok || ms <= 0.0) return std::nullopt;
        have_early_ = true;
        early_tps_ = n_gen * 1000.0 / ms;
        early_ms_cyc_ = cycles > 0 ? ms / static_cast<double>(cycles) : 0.0;
        prev_n_ = n_gen;
        prev_ms_ = ms;
        prev_cyc_ = cycles;
        return std::nullopt;
    }
    const double dms = ms - prev_ms_;
    if (dms < cfg_.min_window_ms || n_gen < prev_n_) return std::nullopt;  // the window keeps growing
    const double tps = (n_gen - prev_n_) * 1000.0 / dms;
    const double ms_cyc = cycles > prev_cyc_ && early_ms_cyc_ > 0.0 ? dms / static_cast<double>(cycles - prev_cyc_) : 0.0;
    prev_n_ = n_gen;
    prev_ms_ = ms;
    prev_cyc_ = cycles;
    if (fired_ || tps >= early_tps_ * (1.0 - cfg_.drop_frac)) return std::nullopt;
    fired_ = true;
    std::string t = std::format("decode slowed from {:.1f} to {:.1f} tok/s ({:.0f}% slower than early in this request", early_tps_,
                                tps, 100.0 * (1.0 - tps / early_tps_));
    if (ms_cyc > 0.0) t += std::format("; {:.1f} -> {:.1f} ms per decode cycle", early_ms_cyc_, ms_cyc);
    t += ")";
    if (spill)
        t += ", likely because of the GPU memory spill above";
    else if (ms_cyc > 0.0 && ms_cyc < early_ms_cyc_ * 1.2)
        t += "; no GPU memory spill is seen, and each cycle is not slower: fewer tokens per cycle (fewer drafts accepted)";
    else
        t += "; no GPU memory spill is seen";
    return Event{spill, t};
}

}  // namespace whirl::wddmwatch
