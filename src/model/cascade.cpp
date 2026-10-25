// feat026-cascade: shared-prefix (cascade) decode / verify attention in the model (whirl/ops/cascade.h).
// The only engine-facing hooks are Model::cascadeBegin (Model::verifyBatchEnqueue, after the row tables) and
// Model::cascadeDecode (Model::attnBlock's decode branch); Model::mapPages keeps host_ptab.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/model.h"

#include "whirl/ops/attention.h"
#include "whirl/ops/cascade.h"
#include "whirl/ops/cascade_kernels.h"

#include <algorithm>
#include <cstdio>

namespace whirl::qwen35 {

namespace {
using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;
inline i32 I(u64 v) { return static_cast<i32>(v); }
}  // namespace

void Model::cascadeBegin(u32 n) {
    cas_plan = ops::CascadePlan{};
    if (!cascade_cfg.on || cas_ml == 0 || cas_acc == 0 || cas_rows == 0) return;
    // never in precise (the split ranges / merge order differ from the plain path: not bitwise)
    if (prec_dec || num_plan.mode == numerics::Mode::precise) return;
    if (k.attn_cascade_rows == nullptr || k.attn_cascade_merge == nullptr || row_n != n || n < 2 || n > ops::kCascadeMaxRows) return;
    ops::CascadePrefixProblem cp;
    cp.head_dim = cfg.head_dim;
    cp.n_head = cfg.n_head;
    cp.n_head_kv = cfg.n_head_kv;
    cp.kv_q4 = kv_q4;
    const u32 cap = ops::cascadeGroupCap(cp, k, k.caps);
    if (cap < 2) return;
    // page table of the sequence at offset `base` (as Model::mapPages last wrote it)
    const ops::CascadePages pages = [this](i32 base) -> std::span<const i32> {
        for (std::size_t s = 0; s < seqs.size() && s < host_ptab.size(); ++s)
            if (static_cast<i32>(seqs[s].kv_base) == base) return std::span<const i32>(host_ptab[s]);
        return {};
    };
    const std::span<const i32> rp(row_pos.data(), n), rb(row_base.data(), n);
    ops::CascadePlan pl = ops::planCascade(rp, rb, pages, cascade_cfg, cap);
    if (!pl.any()) return;
    const ops::CascadeRows tab = ops::cascadeRows(pl, rp, rb);
    hip::launch(k.attn_cascade_rows, hip::Dim3{32, 1, 1}, hip::Dim3{32, 1, 1}, 0, stream, cas_rows, tab, I(n));
    cas_plan = std::move(pl);
    cas_steps += 1;
    for (const ops::CascadeGroup& g : cas_plan.groups) cas_rows_n += g.count;
    // evidence for the GPU runner (stderr, verifies 1, 2, 4, 8, ...): the cascade really ran
    if ((cas_steps & (cas_steps - 1)) == 0)
        std::fprintf(stderr, "whirl cascade: verify %llu: %zu group(s), first %u rows x %u shared keys (rows so far %llu)\n",
                     static_cast<unsigned long long>(cas_steps), cas_plan.groups.size(), cas_plan.groups[0].count, cas_plan.groups[0].keys,
                     static_cast<unsigned long long>(cas_rows_n));
}

bool Model::cascadeDecode(const ops::AttnDecodeProblem& ap, const ops::AttnDecodeFlags& af, const KvArgs& kva, float scale,
                          ops::AttnDecodePlan& out) {
    // the MTP draft window changes which keys a row sees (lossy draft-only path): keep the plain path there
    if (!cas_plan.any() || kva.win != 0 || kva.kvbase == 0 || !ap.rows_known || ap.n > ops::kCascadeMaxRows) return false;
    const u32 n = ap.n, hd = ap.head_dim;
    // suffix rows: positions - P, page-table offsets + P / 256 (rows outside every group unchanged)
    std::array<i32, ops::kCascadeMaxRows> spos{}, sbase{};
    for (u32 r = 0; r < n; ++r) {
        const i32 p = cas_plan.keys[r];
        spos[r] = ap.row_pos[r] - p;
        sbase[r] = ap.row_base[r] + p / static_cast<i32>(ops::kCascadePage);
    }
    ops::AttnDecodeProblem sp = ap;
    sp.row_pos = std::span<const i32>(spos.data(), n);
    sp.row_base = std::span<const i32>(sbase.data(), n);
    const ops::AttnDecodePlan pl = ops::chooseAttnDecode(sp, af, k, k.caps, device::profile(arch));
    ops::CascadePrefixProblem cp;
    cp.head_dim = hd;
    cp.n_head = ap.n_head;
    cp.n_head_kv = ap.n_head_kv;
    cp.kv_q4 = ap.kv_q4;
    cp.n_split = pl.n_split;
    const ops::CascadePrefixPlan pp = ops::chooseCascadePrefix(cas_plan, cp, k, k.caps);
    if (pp.kind == ops::CascadePrefixKind::none || pp.f == nullptr) return false;
    const DevPtr spos_d = cas_rows, sbase_d = cas_rows + 32 * 4, ppos_d = cas_rows + 64 * 4;
    KvArgs kvs = kva;
    kvs.kvbase = sbase_d;
    // suffix pass: the normal kernel choice on the moved rows, into part_ml / part_acc
    if (pl.kind == ops::AttnDecodeKind::split)
        hip::launch(pl.f, pl.grid, pl.block, 0, stream, qf, kvs, part_ml, part_acc, I(ap.n_head), I(ap.n_head_kv), I(hd), I(2 * hd), spos_d, scale,
                    i32(1), gate);
    else
        hip::launch(pl.f, pl.grid, pl.block, 0, stream, qf, kvs, part_ml, part_acc, I(ap.n_head), I(ap.n_head_kv), I(2 * hd), spos_d, scale, gate,
                    pl.groups);
    // prefix pass: one multi-row group per shared prefix (the rows' own page tables map the same pages there,
    // so the group's first row's table serves all of them), every row at position P - 1, into cas_ml / cas_acc
    hip::launch(pp.f, pp.grid, pp.block, 0, stream, qf, kva, cas_ml, cas_acc, I(ap.n_head), I(ap.n_head_kv), I(2 * hd), ppos_d, scale, gate,
                pp.groups);
    // merge: prefix partials folded into split 0 of the suffix partials (rows with ppos >= 0)
    hip::launch(k.attn_cascade_merge, hip::Dim3{ap.n_head, n, 1}, hip::Dim3{256, 1, 1}, 0, stream, part_ml, part_acc, cas_ml, cas_acc, cas_rows,
                I(ap.n_head), I(hd), I(pl.n_split));
    out = pl;
    return true;
}

}  // namespace whirl::qwen35
