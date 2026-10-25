// feat026-cascade: kernel choice of the cascade prefix pass (see whirl/ops/cascade.h). Pure function of the
// problem and the kernel table, like whirl/ops/attention.h; src/model/cascade.cpp launches it.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "whirl/hip.h"
#include "whirl/kernels_abi.h"
#include "whirl/ops/cascade.h"

#include <cstdint>

namespace whirl::ops {

struct CascadePrefixProblem {
    std::uint32_t head_dim = 256, n_head = 0, n_head_kv = 0;
    bool kv_q4 = false;
    std::uint32_t n_split = 0;  // the suffix plan's split count (the partial layout both passes share)
};
enum class CascadePrefixKind : std::uint8_t { none, wsplit1, wsplit2, ws4g };
struct CascadePrefixPlan {
    CascadePrefixKind kind = CascadePrefixKind::none;
    hip::Function f = nullptr;
    hip::Dim3 grid{}, block{128, 1, 1};
    kernels::AwGroups groups{};
};

// rows one prefix group may hold with this kernel set (0: no multi-row prefix kernel: no cascade)
//   gfx1151 (caps.attn_group1): attn_ws4g (f16 / int8 KV, head_dim 256, GQA group <= 16): kMaxVerifyRows
//     rows (more than one CG-3 tile uses column-group blocks on grid.x, like a verify group);
//   gfx1201: attn_wsplit2 (32 columns) or attn_wsplit1 (16 columns): 32 / grp or 16 / grp rows.
std::uint32_t cascadeGroupCap(const CascadePrefixProblem& p, const kernels::AttnKernels& k, const kernels::Caps& caps);
// the prefix launch for a plan's groups (kind none when the plan has none or no kernel fits)
CascadePrefixPlan chooseCascadePrefix(const CascadePlan& plan, const CascadePrefixProblem& p, const kernels::AttnKernels& k,
                                      const kernels::Caps& caps);

}  // namespace whirl::ops
