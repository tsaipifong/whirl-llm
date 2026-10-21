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
    if (s.nonlocal_budget > 0) take(s.nonlocal_budget / static_cast<std::uint64_t>(std::max(s.hip_devices, 1)), s.own_nonlocal, pinned_reserve_mb);
    if (s.total_phys > 0) take(s.total_phys / 2, s.pinned_max, pinned_machine_reserve_mb);
    return room;
}

std::optional<RamFit> ramTierFit(std::uint64_t mb, std::uint64_t avail_phys_bytes, std::uint64_t auto_mb, bool explicit_size,
                                 std::uint64_t pinned_room_mb) {
    if (mb == 0) return std::nullopt;
    RamFit r;
    r.mb = mb;
    if (explicit_size && avail_phys_bytes > 0 && (mb << 20) > avail_phys_bytes / 2) {
        const std::uint64_t half = (avail_phys_bytes / 2 >> 20) / 512 * 512;
        r.mb = auto_mb > 0 ? std::min(half, auto_mb) : half;
        r.half = true;
    }
    if (pinned_room_mb != pinned_room_unknown && r.mb > pinned_room_mb) {
        r.mb = pinned_room_mb;
        r.pinned = true;
    }
    if (!r.half && !r.pinned) return std::nullopt;
    return r;
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
