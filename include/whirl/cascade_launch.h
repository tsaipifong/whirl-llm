// feat026-cascade: the launches of the cascade (shared-prefix) decode attention, shared by Model::cascadeBegin /
// Model::cascadeDecode (src/model/cascade.cpp) and the kernel test family "cascade" (tests/kernels/test_cascade.cpp),
// so the test runs the production launch sequence and arguments, not a copy (feat026-cascade-r1, review C1).
// The plans come from the pure planners in whirl/ops/cascade.h and whirl/ops/cascade_kernels.h; only this header
// launches. Inline, so the kernel test needs no model library.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "whirl/device_profile.h"
#include "whirl/hip.h"
#include "whirl/kernels_abi.h"
#include "whirl/ops/attention.h"
#include "whirl/ops/cascade.h"
#include "whirl/ops/cascade_kernels.h"

#include <array>
#include <cstdint>
#include <span>

namespace whirl::cascade {

// device buffers of one attention layer's decode / verify (the model's members of the same names)
struct Bufs {
    hip::DevPtr qf = 0;        // q then gate per head (q_stride 2 * head_dim)
    hip::DevPtr part_ml = 0;   // suffix (and plain) partials: rows x n_split x n_head x (m, l)
    hip::DevPtr part_acc = 0;  // rows x n_split x n_head x head_dim
    hip::DevPtr cas_ml = 0, cas_acc = 0;  // prefix partials, same shape
    hip::DevPtr cas_rows = 0;  // attn_cascade_rows output: [0, 32) suffix positions, [32, 64) suffix page-table
                               // offsets, [64, 96) prefix positions (-1: row not in a group)
    hip::DevPtr gate = 0;      // the split kernels' skip argument (int*, nonzero value: the kernel returns)
    hip::Stream stream = nullptr;
};

// cas_rows for a plan's rows (one block of 32 threads; the kernel uses threadIdx only)
inline void launchRows(const kernels::AttnKernels& k, const ops::CascadeRows& tab, std::uint32_t n, const Bufs& b) {
    hip::launch(k.attn_cascade_rows, hip::Dim3{1, 1, 1}, hip::Dim3{32, 1, 1}, 0, b.stream, b.cas_rows, tab, static_cast<std::int32_t>(n));
}

// suffix pass (normal kernel choice on the moved rows) + prefix pass (one multi-row group launch per shared prefix)
// + merge (prefix partials folded into suffix split 0). On true, `out` is the suffix plan: the caller runs its
// combine (attn_combine / attn_combine_q8 over out.n_split splits) as on the plain path. On false nothing was
// launched and the caller takes the plain path. cas_rows must hold launchRows' output for this plan.
inline bool launchDecode(const ops::CascadePlan& plan, const ops::AttnDecodeProblem& ap, const ops::AttnDecodeFlags& af,
                         const kernels::AttnKernels& k, const kernels::Caps& caps, const device::Profile& prof, const kernels::KvArgs& kva,
                         float scale, const Bufs& b, ops::AttnDecodePlan& out) {
    using i32 = std::int32_t;
    // the MTP draft window changes which keys a row sees (lossy draft-only path): keep the plain path there
    if (!plan.any() || kva.win != 0 || kva.kvbase == 0 || !ap.rows_known || ap.n > ops::kCascadeMaxRows) return false;
    if (k.attn_cascade_merge == nullptr) return false;
    const std::uint32_t n = ap.n, hd = ap.head_dim;
    // suffix rows: positions - P, page-table offsets + P / 256 (rows outside every group unchanged)
    std::array<i32, ops::kCascadeMaxRows> spos{}, sbase{};
    for (std::uint32_t r = 0; r < n; ++r) {
        const i32 p = plan.keys[r];
        spos[r] = ap.row_pos[r] - p;
        sbase[r] = ap.row_base[r] + p / static_cast<i32>(ops::kCascadePage);
    }
    ops::AttnDecodeProblem sp = ap;
    sp.row_pos = std::span<const i32>(spos.data(), n);
    sp.row_base = std::span<const i32>(sbase.data(), n);
    const ops::AttnDecodePlan pl = ops::chooseAttnDecode(sp, af, k, caps, prof);
    ops::CascadePrefixProblem cp;
    cp.head_dim = hd;
    cp.n_head = ap.n_head;
    cp.n_head_kv = ap.n_head_kv;
    cp.kv_q4 = ap.kv_q4;
    cp.n_split = pl.n_split;
    const ops::CascadePrefixPlan pp = ops::chooseCascadePrefix(plan, cp, k, caps);
    if (pp.kind == ops::CascadePrefixKind::none || pp.f == nullptr) return false;
    const hip::DevPtr spos_d = b.cas_rows, sbase_d = b.cas_rows + 32 * 4, ppos_d = b.cas_rows + 64 * 4;
    kernels::KvArgs kvs = kva;
    kvs.kvbase = sbase_d;
    const i32 H = static_cast<i32>(ap.n_head), HKV = static_cast<i32>(ap.n_head_kv);
    // suffix pass: the normal kernel choice on the moved rows, into part_ml / part_acc
    if (pl.kind == ops::AttnDecodeKind::split)
        hip::launch(pl.f, pl.grid, pl.block, 0, b.stream, b.qf, kvs, b.part_ml, b.part_acc, H, HKV, static_cast<i32>(hd), static_cast<i32>(2 * hd),
                    spos_d, scale, i32(1), b.gate);
    else
        hip::launch(pl.f, pl.grid, pl.block, 0, b.stream, b.qf, kvs, b.part_ml, b.part_acc, H, HKV, static_cast<i32>(2 * hd), spos_d, scale, b.gate,
                    pl.groups);
    // prefix pass: one multi-row group per shared prefix (the rows' own page tables map the same pages there,
    // so the group's first row's table serves all of them), every row at position P - 1, into cas_ml / cas_acc
    hip::launch(pp.f, pp.grid, pp.block, 0, b.stream, b.qf, kva, b.cas_ml, b.cas_acc, H, HKV, static_cast<i32>(2 * hd), ppos_d, scale, b.gate,
                pp.groups);
    // merge: prefix partials folded into split 0 of the suffix partials (rows with ppos >= 0)
    hip::launch(k.attn_cascade_merge, hip::Dim3{ap.n_head, n, 1}, hip::Dim3{256, 1, 1}, 0, b.stream, b.part_ml, b.part_acc, b.cas_ml, b.cas_acc,
                b.cas_rows, H, static_cast<i32>(hd), static_cast<i32>(pl.n_split));
    out = pl;
    return true;
}

}  // namespace whirl::cascade
