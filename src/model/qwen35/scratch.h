// qwen35::Scratch: the prefill / decode activation buffers of a loaded Model (arch025-M3, R-8c) and the caches of
// what they hold. Allocated once by Model::load (Model::allocScratch, src/model/qwen35/scratch.cpp, in this order;
// bufferBytes there is the pre-load estimate of the same list), freed with the Model's allocation ledger.
// SPDX-License-Identifier: Apache-2.0
//
// A private base of qwen35::Model: the forward (Model's member functions) reads the members by name as before;
// outside the model they are read through Model::scratch() (read-only). The declarations moved verbatim from
// include/whirl/model.h (same order and initial values).

#pragma once

#include "whirl/hip.h"

#include <cstdint>

namespace whirl::qwen35 {

using hip::DevPtr;

struct Scratch {
    DevPtr x = 0, h = 0, qf = 0, kv_k = 0, kv_v = 0, attn_out = 0, qkv = 0, conv_out = 0, z = 0, beta = 0, alpha = 0, ba_buf = 0,
           gdn_out = 0, ffn_g = 0, ffn_u = 0, logits = 0, scores = 0, ids = 0, pos_buf = 0, out_tok = 0;
    DevPtr part_ml = 0, part_acc = 0;
    DevPtr x16 = 0, xq = 0, xd = 0;
    DevPtr xq_src = 0;
    std::uint32_t xq_n = 0;
    DevPtr x16_src = 0;
    std::uint32_t x16_count = 0;
    bool x16_fp8 = false;
    DevPtr sx8 = 0;
    std::uint8_t mm_class = 1;
    DevPtr w16 = 0;
    DevPtr gc_w = 0, gc_u = 0, gc_m = 0, gc_g = 0;
    std::uint64_t gc_w_bytes = 0, gc_u_bytes = 0, gc_m_bytes = 0;  // (GDN-8S scratch bound checks)
    std::uint64_t x16_bytes = 0;  // x16 allocation size (x16b sub-buffer bound check)
    DevPtr dq_ptab = 0;
    std::uint32_t dq_rows = 0;  // rows ffn_g / ffn_u hold
    // FC-1c: attn_kgs softmax state (m, l per query row and head) between key ranges of a prefill
    // longer than dq_rows (and for q4: every prefill dequantizes its keys into ffn_g / ffn_u)
    DevPtr dq_st_ml = 0;
    DevPtr hn = 0, mtp_cat = 0, mtp_h = 0, mtp_ids = 0;
    std::uint32_t ff_scratch = 0;
    DevPtr moe_logits = 0, moe_ids = 0, moe_w = 0, moe_sg = 0, moe_g = 0, moe_u = 0, moe_xq = 0, moe_xd = 0, moe_ysh = 0,
           moe_perm = 0, moe_inv = 0, moe_tiles = 0, moe_ntiles = 0, moe_tiles64 = 0, moe_ntiles64 = 0, moe_x16 = 0, moe_yg = 0, moe_yu = 0, moe_yd = 0;
    // MXFP4 routed experts (Ornith MXFP4): per-pos fp8 activation scales [max_batch * K];
    // prefill experts on the MXFP4 x fp8 grouped GEMM (WHIRL_MOE_FP8=0 -> f16 gemm_moe);
    // decode experts on the whole-block kernels (WHIRL_MOE_MXW=0 -> generic moe_gu / moe_down).
    DevPtr moe_sx = 0;
};

}  // namespace whirl::qwen35
