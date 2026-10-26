// feat026-cascade: kernel choice of the cascade prefix pass (whirl/ops/cascade_kernels.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/cascade_kernels.h"

#include "../../kernels/gfx1151/attn_ws4_index.h"  // ws4::rows_per_block, ws4::kD

#include <algorithm>

namespace whirl::ops {

namespace {
using u32 = std::uint32_t;
constexpr hip::Dim3 D(u32 x, u32 y = 1, u32 z = 1) { return hip::Dim3{x, y, z}; }
bool ws4gSet(const kernels::AttnKernels& k) { return k.attn_ws4g[0] != nullptr && k.attn_ws4g[1] != nullptr && k.attn_ws4g[2] != nullptr; }
}  // namespace

u32 cascadeGroupCap(const CascadePrefixProblem& p, const kernels::AttnKernels& k, const kernels::Caps& caps) {
    if (p.n_head_kv == 0 || p.n_head % p.n_head_kv != 0 || p.head_dim != 256) return 0;
    const u32 grp = p.n_head / p.n_head_kv;
    if (caps.attn_group1) {
        // gfx1151: attn_wsplit1 groups of one query only (its P.V is not exact with masked keys); attn_ws4g is
        // the multi-row kernel (f16 / int8 KV; no q4 variant)
        if (p.kv_q4 || !ws4gSet(k) || grp > 16 || p.head_dim != static_cast<u32>(ws4::kD)) return 0;
        return static_cast<u32>(kernels::kMaxVerifyRows);
    }
    // feat026-cascade-wide: 8 waves x 16 columns share one K / V tile
    if (k.attn_cascade_wide != nullptr && 128 / grp >= 2) return std::min<u32>(128 / grp, kCascadeMaxRows);
    if (k.attn_wsplit2 != nullptr && 32 / grp >= 2) return std::min<u32>(32 / grp, kCascadeMaxRows);
    if (k.attn_wsplit1 != nullptr && 16 / grp >= 2) return 16 / grp;
    return 0;
}

CascadePrefixPlan chooseCascadePrefix(const CascadePlan& plan, const CascadePrefixProblem& p, const kernels::AttnKernels& k,
                                      const kernels::Caps& caps) {
    CascadePrefixPlan pl;
    const u32 cap = cascadeGroupCap(p, k, caps);
    if (!plan.any() || cap == 0 || p.n_split == 0 || plan.groups.size() > static_cast<std::size_t>(kernels::kMaxVerifyRows)) return pl;
    const u32 grp = p.n_head / p.n_head_kv;
    u32 widest = 0, ng = 0;
    for (const CascadeGroup& g : plan.groups) {
        if (g.count > cap || g.count < 2) return CascadePrefixPlan{};
        pl.groups.first[ng] = static_cast<std::int32_t>(g.first);
        pl.groups.count[ng] = static_cast<std::int32_t>(g.count);
        widest = std::max(widest, g.count);
        ng += 1;
    }
    if (caps.attn_group1) {
        // attn_ws4g: CG 16-column tiles per block (<= 3), more rows in column-group blocks on grid.x
        const u32 cg = std::min<u32>(3, (widest * grp + 15) / 16);
        const u32 rpb = static_cast<u32>(ws4::rows_per_block(static_cast<int>(cg), static_cast<int>(grp)));
        if (rpb == 0) return CascadePrefixPlan{};
        pl.kind = CascadePrefixKind::ws4g;
        pl.f = k.attn_ws4g[cg - 1];
        pl.grid = D(p.n_head_kv * ((widest + rpb - 1) / rpb), p.n_split, ng);
        pl.block = D(256);
        return pl;
    }
    if (widest * grp > 32 || (widest * grp > 16 && k.attn_wsplit2 == nullptr)) {
        // wider than attn_wsplit2 (or no attn_wsplit2): the wide kernel (cap 128 / grp checked above)
        if (k.attn_cascade_wide == nullptr || widest * grp > 128) return CascadePrefixPlan{};
        pl.kind = CascadePrefixKind::wide;
        pl.f = k.attn_cascade_wide;
        pl.grid = D(p.n_head_kv, p.n_split, ng);
        pl.block = D(256);
        return pl;
    }
    const bool two = k.attn_wsplit2 != nullptr && widest * grp > 16;
    pl.kind = two ? CascadePrefixKind::wsplit2 : CascadePrefixKind::wsplit1;
    pl.f = two ? k.attn_wsplit2 : k.attn_wsplit1;
    if (pl.f == nullptr) return CascadePrefixPlan{};
    pl.grid = D(p.n_head_kv, p.n_split, ng);
    pl.block = D(128);
    return pl;
}

}  // namespace whirl::ops
