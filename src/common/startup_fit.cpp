// Startup fit of explicit settings (see include/whirl/startup_fit.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/startup_fit.h"

#include <algorithm>

namespace whirl::startfit {

std::optional<std::uint64_t> ramTierCap(std::uint64_t mb, std::uint64_t avail_phys_bytes, std::uint64_t auto_mb) {
    if (mb == 0 || avail_phys_bytes == 0) return std::nullopt;
    if ((mb << 20) <= avail_phys_bytes / 2) return std::nullopt;
    const std::uint64_t half = (avail_phys_bytes / 2 >> 20) / 512 * 512;
    return auto_mb > 0 ? std::min(half, auto_mb) : half;
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

}  // namespace whirl::startfit
