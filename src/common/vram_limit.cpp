// Simulated VRAM cap and VRAM-fit arithmetic (see include/whirl/vram_limit.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/vram_limit.h"

#include <algorithm>
#include <cstdlib>
#include <format>

namespace whirl::vram {

namespace {

std::uint64_t readEnvLimit() {
    char* buf = nullptr;
    std::size_t len = 0;
    std::uint64_t mb = 0;
    // _dupenv_s keeps MSVC quiet about getenv
    if (_dupenv_s(&buf, &len, "WHIRL_VRAM_LIMIT_MB") == 0 && buf != nullptr) {
        mb = parseLimitMb(buf);
        std::free(buf);
    }
    return mb;
}

std::atomic<std::uint64_t>& limitSlot() {
    static std::atomic<std::uint64_t> v{readEnvLimit() << 20};
    return v;
}

std::uint64_t satSub(std::uint64_t a, std::uint64_t b) { return a > b ? a - b : 0; }

}  // namespace

std::uint64_t parseLimitMb(const char* s) {
    if (s == nullptr) return 0;
    while (*s == ' ' || *s == '\t') ++s;
    if (*s < '0' || *s > '9') return 0;
    std::uint64_t v = 0;
    for (; *s >= '0' && *s <= '9'; ++s) {
        v = v * 10 + static_cast<std::uint64_t>(*s - '0');
        if (v > (1ull << 30)) return 0;  // > 1 PiB: not a VRAM size
    }
    while (*s == ' ' || *s == '\t') ++s;
    return *s == '\0' ? v : 0;
}

std::uint64_t limitBytes() { return limitSlot().load(std::memory_order_relaxed); }

void setLimitMb(std::uint64_t mb) { limitSlot().store(mb << 20, std::memory_order_relaxed); }

std::string limitNote() {
    const std::uint64_t l = limitBytes();
    if (l == 0) return {};
    return std::format("VRAM limit {} MiB (WHIRL_VRAM_LIMIT_MB)", l >> 20);
}

Clamped clampMemInfo(std::uint64_t free, std::uint64_t total, std::uint64_t limit) {
    if (limit == 0 || limit >= total) return {free, total};
    return {satSub(free, total - limit), limit};
}

std::uint64_t clampBudget(std::uint64_t budget, std::uint64_t total, std::uint64_t limit) {
    if (limit == 0 || limit >= total) return budget;
    return satSub(budget, total - limit);
}

std::uint64_t Counter::live() const { return live_.load(std::memory_order_relaxed); }

bool Counter::tryReserve(std::uint64_t bytes) {
    if (cap_ == 0) {
        live_.fetch_add(bytes, std::memory_order_relaxed);
        return true;
    }
    std::uint64_t cur = live_.load(std::memory_order_relaxed);
    do {
        if (bytes > cap_ || cur > cap_ - bytes) return false;
    } while (!live_.compare_exchange_weak(cur, cur + bytes, std::memory_order_relaxed));
    return true;
}

void Counter::release(std::uint64_t bytes) {
    std::uint64_t cur = live_.load(std::memory_order_relaxed);
    while (!live_.compare_exchange_weak(cur, cur > bytes ? cur - bytes : 0, std::memory_order_relaxed)) {
    }
}

std::uint64_t processCap(std::uint64_t free_at_first, std::uint64_t total, std::uint64_t limit) {
    return clampMemInfo(free_at_first, total, limit).free;
}

std::uint64_t fixedNeed(const FitInput& in, std::uint32_t batch, std::uint32_t n_ck, std::uint32_t n_spe) {
    return in.weights + in.fixed + in.buf_fixed + in.buf_per_row * batch +
           in.ckpt_bytes * (static_cast<std::uint64_t>(n_ck) * std::max(in.parallel, 1u) + n_spe);
}

FitPlan planFit(const FitInput& in) {
    FitPlan p;
    p.batch = in.batch;
    p.n_ck = in.n_ck;
    p.n_spe = in.n_spe;
    auto eval = [&] {
        p.need_fixed = fixedNeed(in, p.batch, p.n_ck, p.n_spe);
        p.pool_tokens = in.kv_per_token ? satSub(in.avail, p.need_fixed) / in.kv_per_token : 0;
    };
    eval();
    const std::uint64_t want = std::max(in.pool_min_tokens, in.pool_hard_min);
    // steps in order; each applies only if it lowers something and is allowed
    struct Step {
        std::uint32_t batch, n_ck, n_spe;
    };
    const Step steps[] = {
        {2048, 0xffffffffu, 0xffffffffu},  // batch 2048
        {0, 2, 1},                         // checkpoints: 2 per slot, 1 shared
        {1024, 0xffffffffu, 0xffffffffu},  // batch 1024
        {0, 1, 0},                         // checkpoints: 1 per slot, no shared
        {512, 0xffffffffu, 0xffffffffu},   // batch 512
    };
    const std::uint32_t b0 = p.batch, c0 = p.n_ck, s0 = p.n_spe;
    for (const Step& s : steps) {
        if (p.pool_tokens >= want) break;
        bool changed = false;
        if (s.batch != 0 && !in.batch_fixed && p.batch > s.batch) {
            p.batch = s.batch;
            changed = true;
        }
        if (s.batch == 0 && !in.ck_fixed) {
            if (p.n_ck > s.n_ck) {
                p.n_ck = s.n_ck;
                changed = true;
            }
            if (p.n_spe > s.n_spe) {
                p.n_spe = s.n_spe;
                changed = true;
            }
        }
        if (changed) eval();
    }
    p.reduced = p.batch != b0 || p.n_ck != c0 || p.n_spe != s0;
    p.fits = p.pool_tokens >= in.pool_hard_min;
    if (p.reduced) {
        std::string n;
        auto add = [&](const std::string& s) { n += (n.empty() ? "" : ", ") + s; };
        if (p.batch != b0) add(std::format("prefill batch {} -> {}", b0, p.batch));
        if (p.n_ck != c0) add(std::format("checkpoints per slot {} -> {}", c0, p.n_ck));
        if (p.n_spe != s0) add(std::format("shared checkpoints {} -> {}", s0, p.n_spe));
        p.note = n;
    }
    return p;
}

SmallKv smallCardKv(std::uint64_t avail, std::uint64_t floor_tokens, std::uint64_t q8v_bytes_per_token) {
    if (q8v_bytes_per_token == 0) return SmallKv::q8h;
    if (floor_tokens * q8v_bytes_per_token <= avail) return SmallKv::q8v;
    if (avail / q8v_bytes_per_token >= small_card_q8v_min_tokens) return SmallKv::q8v_short;
    return SmallKv::q8h;
}

CardDefaults cardDefaults(const CardInput& in) {
    CardDefaults d;
    d.small = !in.uma && in.total > 0 && in.total < small_card_below;
    d.parallel = in.parallel_arg ? *in.parallel_arg : in.parallel_default;
    if (d.small && !in.parallel_arg && d.parallel > 1) {
        d.parallel = 1;
        d.parallel_auto = true;
    }
    d.prefer_q8v = d.small && in.kv_auto;
    std::uint64_t mb = 0;
    if (in.headroom_mb_env) mb = *in.headroom_mb_env;
    else if (d.small && !in.reserve_explicit) mb = small_card_headroom_mb;
    d.headroom = mb << 20;
    const double gib = static_cast<double>(in.total) / (1024.0 * 1024.0 * 1024.0);
    std::string hr;
    if (in.headroom_mb_env) hr = std::format("{} MiB of VRAM kept free for the desktop and other apps (WHIRL_VRAM_HEADROOM_MB)", mb);
    else if (d.small && in.reserve_explicit) hr = "no desktop headroom (WHIRL_POOL_RESERVE_MB given; WHIRL_VRAM_HEADROOM_MB adds one)";
    else if (d.small) hr = std::format("{} MiB of VRAM kept free for the desktop and other apps (WHIRL_VRAM_HEADROOM_MB to change, 0 = none)", mb);
    if (d.small) {
        const std::string par = d.parallel_auto ? std::format("--parallel 1 (default {} on cards >= 20 GiB; pass --parallel N for more)", in.parallel_default)
                                                : std::format("--parallel {} (given)", d.parallel);
        const std::string kv = in.kv_auto ? "KV auto prefers q8v (K f16, V int8; WHIRL_KV=f16 forces f16)" : "KV format as set by WHIRL_KV";
        d.note = std::format("small card ({:.1f} GiB < 20 GiB, likely also driving the desktop): {}, {}, {}", gib, par, kv, hr);
    } else if (!hr.empty()) {
        d.note = hr;
    }
    return d;
}

}  // namespace whirl::vram
