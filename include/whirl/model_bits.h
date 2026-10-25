// feat026-o6 (1), owner decision O6 (0.2.6): the bits key of a loaded Model, for the SSD tier fingerprint.
// SPDX-License-Identifier: Apache-2.0
//
// numerics::Plan::bitsKey() (arch025-N2a) names everything in the numerics plan that decides the output bits: the mode,
// every item's enabled state, q8dec and every execution flag. The fingerprint takes it in place of logLine(num_plan),
// whose descriptions (skip reasons, item notes) changed the key whenever their wording changed.
//
// The flags come from the Model members the forward reads, not from Model::num_plan.flags: until N2b nothing fills the
// plan's flags from the real run (num_plan.flags are the defaults there), and a key built from them would change a second
// time when N2b wires them. N2b makes num_plan.flags the source of these members, so num_plan.flags == flagsOf(m) then
// (model_tests testPlanVsLoadOrTune checks plan.bitsKey() == tierBitsKey() already), and the key stays the same.
// Prior art: progress\feat026-o6.PRIOR_ART.md (vLLM compute_hash factors; llama.cpp session file version).
#pragma once

#include "whirl/model.h"
#include "whirl/numerics.h"

#include <string>

namespace whirl::qwen35 {

// Every numerics::Flags field from the Model member that holds it (the "D exec:" dump key of the same name,
// src/model/exec_dump.cpp; model_tests testModelBitsKey checks each field against the dump). gdn_v0: the front end
// swapped in gdn_step_norm_v0 (WHIRL_GDN_V0; the Model keeps only the kernel pointer).
inline numerics::Flags flagsOf(const Model& m, bool gdn_v0) {
    numerics::Flags f;
    f.use_graph = m.use_graph;
    f.no_fuse = m.no_fuse;
    f.dbg_flags = m.dbg_flags;
    f.naive_attn = m.naive_attn;
    f.attn_wide = m.attn_wide;
    f.attn_dq4 = m.attn_dq4;
    f.dq4_dyn = m.dq4_dyn;
    f.attn_ws4 = m.attn_ws4;
    f.wide_verify = m.wide_verify;
    f.moe_wide = m.moe_wide;
    f.head_chunk = m.head_chunk;
    f.float_gemv = m.float_gemv;
    f.prec_dec = m.prec_dec;
    f.gemv_max = m.gemv_max;
    f.fp8_prefill = m.fp8_prefill;
    f.fp8_mask = m.fp8_mask;
    f.relax_on = m.relax.on;
    f.relax_k = m.relax.k;
    f.relax_alpha = m.relax.alpha;
    f.relax_eps = m.relax.eps;
    f.relax_delta = m.relax.delta;
    f.gdn_chunked = m.gdn_chunked;
    f.gdn_wmma = m.gdn_wmma;
    f.gdn_strix = m.gdn_strix;
    f.gdn_strix_min = m.gdn_strix_min;
    f.act_fuse = m.act_fuse;
    f.attn_kx_on = m.attn_kx_on;
    f.gdn_ba_on = m.gdn_ba_on;
    f.gdn_in2_on = m.gdn_in2_on;
    f.attn_dqf_on = m.attn_dqf_on;
    f.attn_seg = m.attn_seg;
    f.attn_kg_on = m.attn_kg_on;
    f.ffn_h16 = m.ffn_h16;
    f.h16_dq = m.h16_dq;
    f.g8t = m.g8t;
    f.gemmh_on = m.gemmh_on;
    f.gemmhq_on = m.gemmhq_on;
    f.draft_window = m.draft_window;
    f.draft_window_min = m.draft_window_min;
    f.moe_bn_force = m.moe_bn_force;
    f.moe_fp8 = m.moe_fp8;
    f.moe_mxw = m.moe_mxw;
    f.moe_rbf = m.moe_rbf;
    f.gv_group = gv_group;
    f.gv_nmax = gv_nmax;
    f.gdn_v0 = gdn_v0;
    return f;
}

// The model's plan (items, q8dec: what loadOrTune and the front end left in num_plan) with the flags above, as
// numerics::Plan::bitsKey(). The tier fingerprint's "numerics" part (feat026-o6; before: logLine(num_plan)).
inline std::string tierBitsKey(const Model& m, bool gdn_v0) {
    numerics::Plan p = m.num_plan;
    p.flags = flagsOf(m, gdn_v0);
    return p.bitsKey();
}

}  // namespace whirl::qwen35
