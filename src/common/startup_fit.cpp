// Startup fit of explicit settings (see include/whirl/startup_fit.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/startup_fit.h"

#include <algorithm>
#include <format>

namespace whirl::startfit {

std::uint64_t pinnedRoomMb(const PinnedState& s) {
    constexpr std::uint64_t MiB = 1ull << 20;
    std::uint64_t room = pinned_room_unknown;
    const auto take = [&](std::uint64_t limit_b, std::uint64_t used_b, std::uint64_t reserve_mb) {
        const std::uint64_t free_mb = limit_b > used_b ? (limit_b - used_b) / MiB : 0;
        room = std::min(room, free_mb > reserve_mb ? (free_mb - reserve_mb) / 512 * 512 : 0);
    };
    // per HIP GPU: a hipHostMalloc buffer is mapped to every HIP GPU and charged on each. A discrete GPU's DXGI non-local
    // budget holds this process's pinned memory times the number of HIP GPUs (kvram719, R9700 box: 22528 MiB worked,
    // 23552 failed at 48026 / 2). An integrated GPU reports a non-local budget of 0 (the 8060S); its shared GPU memory
    // (SharedSystemMemory) is the limit there, without the division (kvram719: 22528 MiB fit the 8060S's 32587).
    // REVIEW-S8-719 C3: what is pinned on that adapter by every process (PDH GPU Adapter Memory Shared Usage), not only
    // this one; this process alone when the adapter's counter is unknown (0)
    const std::uint64_t n = std::max<std::uint64_t>(s.gpus.size(), 1);
    for (const PinnedGpu& g : s.gpus) {
        const std::uint64_t used = std::max(g.own_nonlocal, g.adapter_shared) + s.later;  // + what is pinned later (N3-4)
        if (g.nonlocal_budget > 0) take(g.nonlocal_budget / n, used, pinned_reserve_mb);
        else if (g.shared_system > 0) take(g.shared_system, used, pinned_reserve_mb);
    }
    // machine: every program's GPU-pinned memory (largest adapter) well under half of the RAM (kvram719)
    if (s.total_phys > 0) take(s.total_phys / 2, s.pinned_max + s.later, pinned_machine_reserve_mb);
    return room;
}

std::uint64_t mmprojPinnedBytes(std::uint64_t file_bytes) { return file_bytes; }

std::string pinnedWhy(std::uint64_t room_mb, std::size_t n_gpus) {
    // no "hip" anywhere, no "failed:" (REVIEW-S8-719 A1): log scanners match hipError / hip...719 case-insensitively
    return std::format("lowered to {} MiB: the GPUs can keep only so much pinned host memory resident ({} GPU(s) map it; "
                       "a larger RAM tier makes the first request fail with GPU error 719)",
                       room_mb, n_gpus);
}

std::string pinnedOffWhy(bool below_entry, std::uint64_t entry_mb) {
    return below_entry ? std::format("; that is below one max-context entry ({} MiB), so the host tiers are off (RAM and SSD)",
                                     entry_mb)
                       : std::string("; no room left, so the host tiers are off (RAM and SSD)");
}

std::optional<RamFit> ramTierFit(std::uint64_t mb, std::uint64_t avail_phys_bytes, std::uint64_t auto_mb, bool explicit_size,
                                 std::uint64_t pinned_room_mb, std::uint64_t entry_mb) {
    (void)auto_mb;
    if (mb == 0) return std::nullopt;
    RamFit r;
    r.mb = mb;
    // v0.2.2.2: the cap first, then the half-of-available-RAM rule on the capped size, which lowers to half (whole
    // 512 MiB): asking for more never gives less (0.2.2.1: 22528 -> 20480 but 24064 -> 16384). s8-719: the GPU-pinned
    // room after the cap (explicit and automatic sizes alike; each step is a min, so an explicit size stays monotonic).
    if (r.mb > ram_tier_safe_mb) {
        r.mb = ram_tier_safe_mb;
        r.safe = true;
    }
    if (pinned_room_mb != pinned_room_unknown && r.mb > pinned_room_mb) {
        r.mb = pinned_room_mb;
        r.pinned = true;
        // FR-CACHE P3-4 (src/tier/ram_size.h): the automatic size holds one max-context entry, else off (REVIEW-S8-719
        // C2). Not an explicit size: it is clamped to the room and used as given (ram_size.h), so asking for more never
        // gives less (REVIEW-S8-719-2 X1/X2: golden's 8192 on two GPUs at 96 GB -> 4096, not off).
        if (!explicit_size && r.mb > 0 && r.mb < entry_mb) {
            r.mb = 0;
            r.below_entry = true;
        }
    }
    // the half rule is a min with half of the available RAM in whole 512 MiB, so it is monotonic at every MiB too
    // (REVIEW-S8-719-3 N3-1: comparing the ask with the unrounded half gave 8800 -> 8800 but 8822 -> 8704)
    if (explicit_size && avail_phys_bytes > 0) {
        const std::uint64_t half_mb = (avail_phys_bytes / 2 >> 20) / 512 * 512;
        if (r.mb > half_mb) {
            r.mb = half_mb;
            r.half = true;
        }
    }
    if (!r.half && !r.safe && !r.pinned) return std::nullopt;
    return r;
}

RamTierText ramTierText(const std::optional<RamFit>& rf, std::uint64_t asked_mb, bool explicit_size, std::string_view src,
                        std::string_view size_reason, std::string pinned_why, std::uint64_t entry_mb, std::uint64_t avail_mb) {
    RamTierText t;
    t.pinned_why = std::move(pinned_why);
    t.off_reason = std::string(size_reason);  // what the "RAM tier off" line names (REVIEW-S8-719 N3)
    t.size_reason = std::string(size_reason);
    if (!rf) return t;
    // the pinned room itself turned the tier off (room 0, or below one entry for the automatic size); when the half rule
    // did it after the room (REVIEW-S8-719-3 N3-2) the room was > 0, so "no room left" would be wrong: the half rule is named
    const bool room_off = rf->mb == 0 && rf->pinned && !rf->half;
    if (room_off) {
        t.pinned_why += pinnedOffWhy(rf->below_entry, entry_mb);
        t.off_reason = std::format("{} {} MiB asked: {}", explicit_size ? std::string(src) : std::string("auto"), asked_mb, t.pinned_why);
    }
    // REVIEW-S8-719-3 N3-6: the size line names the reason of the size asked; say it was lowered, and to what
    if (rf->mb > 0 && rf->mb != asked_mb)
        t.size_reason = explicit_size ? std::format("{} asked, lowered to {} MiB: see the adjusted: line", size_reason, rf->mb)
                                      : std::format("{}; lowered to {} MiB: see the line above", size_reason, rf->mb);
    if (explicit_size) {
        std::string why;
        if (rf->safe) why = std::format("capped at {} MiB while a crash with larger RAM tiers is investigated", ram_tier_safe_mb);
        if (rf->pinned) why += std::format("{}{}", why.empty() ? "" : "; ", t.pinned_why);
        if (rf->half)
            why += std::format("{}the RAM tier is pinned host memory; more than half of the {} MiB of RAM available now "
                               "would make the system page, so it is lowered to half",
                               why.empty() ? "" : "; ", avail_mb);
        if (rf->mb == 0 && rf->half) {
            why += "; that leaves 0 MiB, so the host tiers are off (RAM and SSD)";
            t.off_reason = std::format("{} {} MiB asked: {}", std::string(src), asked_mb, why);
        }
        t.adjusted_why = why;
    }
    return t;
}

std::uint64_t wantTokens(const vram::FitInput& fi) { return std::max<std::uint64_t>(fi.pool_min_tokens, fi.pool_hard_min); }

ParallelFit fitParallel(const vram::FitInput& fi, std::uint64_t per_slot_bytes, bool given) {
    ParallelFit r;
    r.fi = fi;
    r.plan = vram::planFit(fi);
    r.parallel = fi.parallel;
    r.pool_requested = r.plan.pool_tokens;
    while (given && r.parallel > 1 && r.plan.pool_tokens < wantTokens(r.fi)) {
        --r.parallel;
        r.fi.parallel = r.parallel;
        r.fi.fixed = r.fi.fixed > per_slot_bytes ? r.fi.fixed - per_slot_bytes : 0;
        r.plan = vram::planFit(r.fi);
    }
    return r;
}

FixedFit fitFixed(const vram::FitInput& fi, const vram::FitPlan& plan) {
    FixedFit r;
    r.fi = fi;
    r.plan = plan;
    if (plan.pool_tokens >= wantTokens(fi) || !(fi.batch_fixed || fi.ck_fixed)) return r;
    vram::FitInput f2 = fi;
    f2.batch_fixed = f2.ck_fixed = false;
    const vram::FitPlan p2 = vram::planFit(f2);
    if (p2.pool_tokens <= plan.pool_tokens) return r;
    r.changed = true;
    r.batch_changed = fi.batch_fixed && p2.batch != plan.batch;
    r.ck_changed = fi.ck_fixed && (p2.n_ck != plan.n_ck || p2.n_spe != plan.n_spe);
    r.fi = f2;
    r.plan = p2;
    return r;
}

std::uint32_t slotsThatFit(std::uint32_t parallel, std::uint64_t avail, std::uint64_t per_slot, std::uint64_t shared,
                           std::uint64_t per_token, std::uint64_t target_tokens) {
    if (per_token == 0) return parallel;
    std::uint32_t p = std::max<std::uint32_t>(parallel, 1);
    const auto pool = [&](std::uint32_t n) {
        const std::uint64_t fixed = shared + per_slot * n;
        return avail > fixed ? (avail - fixed) / per_token : 0;
    };
    while (p > 1 && pool(p) < target_tokens) --p;
    return p;
}

std::string startupJson(const StartupInfo& s) {
    std::string j = std::format("{{\"ctx_per_request\":{},\"pool_tokens\":{},\"parallel\":{},\"kv_ram_mb\":{},\"wddm_margin_mib\":{},"
                                "\"other_processes_mib\":{},\"display_attached\":{},\"adjusted\":[",
                                s.ctx_per_request, s.pool_tokens, s.parallel, s.kv_ram_mb, s.wddm_margin_mib, s.other_processes_mib,
                                s.display_attached ? "true" : "false");
    for (std::size_t k = 0; k < s.adjusted.size(); ++k) {
        j += k ? ",\"" : "\"";
        for (const char ch : s.adjusted[k]) {
            if (ch == '"' || ch == '\\') j += '\\';
            j += ch;
        }
        j += "\"";
    }
    return j + "]}";
}

}  // namespace whirl::startfit
