// feat026-cascade: shared-prefix (cascade) decode / verify attention in the model (whirl/ops/cascade.h).
// The only engine-facing hooks are Model::cascadeBegin (Model::verifyBatchEnqueue, after the row tables) and
// Model::cascadeDecode (Model::attnBlock's decode branch); Model::mapPages keeps host_ptab.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/model.h"

#include "whirl/cascade_launch.h"  // the launches (shared with the kernel test family "cascade")
#include "whirl/ops/attention.h"
#include "whirl/ops/cascade.h"
#include "whirl/ops/cascade_kernels.h"

#include <algorithm>
#include <cstdio>

namespace whirl::qwen35 {

namespace {
using u32 = std::uint32_t;
using i32 = std::int32_t;
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
    if (!pl.any()) {
        // feat026-cascade-r1 C2: rows that share a prefix, but no two of their sequences fit one prefix group of the
        // real cap (gfx1201 with MTP rows): counted and logged so a 0% gain is not read as "no shared prefix"
        if (cap < ops::kCascadeMaxRows && ops::planCascade(rp, rb, pages, cascade_cfg, ops::kCascadeMaxRows).any()) {
            cas_nofit += 1;
            if ((cas_nofit & (cas_nofit - 1)) == 0)
                std::fprintf(stderr, "whirl cascade: nofit %llu: rows share a prefix but exceed the prefix group cap %u (n %u)\n",
                             static_cast<unsigned long long>(cas_nofit), cap, n);
        }
        return;
    }
    const ops::CascadeRows tab = ops::cascadeRows(pl, rp, rb);
    cascade::Bufs rb_;
    rb_.cas_rows = cas_rows;
    rb_.stream = stream;
    cascade::launchRows(k, tab, n, rb_);
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
    // a gated MTP draft (gate set by mtpEnqueue) never takes the cascade: the merge kernel has no skip argument
    // (cas_plan is empty there anyway: it lives only inside verifyBatchEnqueue)
    if (!cas_plan.any() || gate != 0) return false;
    cascade::Bufs b;
    b.qf = qf;
    b.part_ml = part_ml;
    b.part_acc = part_acc;
    b.cas_ml = cas_ml;
    b.cas_acc = cas_acc;
    b.cas_rows = cas_rows;
    b.gate = gate;
    b.stream = stream;
    return cascade::launchDecode(cas_plan, ap, af, k, k.caps, device::profile(arch), kva, scale, b, out);
}

}  // namespace whirl::qwen35
