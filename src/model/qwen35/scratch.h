// qwen35::Scratch: the prefill / decode activation buffers of a loaded Model (arch025-M3, R-8c) and the caches of
// what they hold. Allocated once by Model::load (Model::allocScratch, src/model/qwen35/model_scratch.cpp, in this order;
// bufferBytes in scratch.cpp is the pre-load estimate of the same list), freed with the Model's allocation ledger.
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

struct Config;  // whirl/model_types.h

// v0252-f1arg3: bytes of the multi-block verify argmax workspace (AmWs, kernels/argmax_key.h AM_WS_BYTES; static_assert in
// forward/mtp_verify.cpp) that Context::allocCtl places directly before the control words when the switch is on
inline constexpr std::uint64_t am_ws_bytes = 512;

// Prefill / decode buffers of Model::load for prefill batch B (moved from include/whirl/model.h, arch025-lane2 N9 S2c).
// fd_splits: attention split-K partials (resolveFdSplits: 64, 256 with q4 KV, WHIRL_FD_SPLITS).
std::uint64_t bufferBytes(const Config& cfg, std::uint32_t B, std::uint64_t ffs, std::uint64_t max_elems, std::uint32_t fd_splits = 64);

// v0252-kvlong2 (O1, PLAN V1-137): the f16 copy of a sequence's K / V that q8 / q8h / q4 prefill attention reads lives in
// ffn_g / ffn_u (Model::allocScratch). rows: keys it holds (dq_rows); bytes_each: the size of ffn_g and of ffn_u; extra:
// what both add over the FFN scratch (B * ffs * 4 each). ctx: keys one sequence may reach (CLI --ctx, the server's
// per-request context; 0 = the FFN scratch size, the plan before O1). Capped at ops::kDqMaxKeys (the page table).
struct DqCopy {
    std::uint32_t rows = 0;
    std::uint64_t bytes_each = 0;
    std::uint64_t extra = 0;
};
DqCopy dqCopyPlan(const Config& cfg, std::uint32_t B, std::uint64_t ffs, std::uint32_t ctx);
// the grown copy is taken only when its extra bytes, what the load still allocates (still) and a 1 GiB margin fit in
// the free VRAM; else the FFN-sized copy (key ranges past it, attn_kgs) as before O1
bool dqCopyFits(std::uint64_t extra, std::uint64_t free, std::uint64_t still);

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
    // v0252-w4a4k2 (PLAN V1-144 K2): the W4A4 fake-quant arm's activation copy (x16_bytes, allocated by Context::matmul at its
    // first use; 0 = never: the arm is a fast-only research switch, numerics flag w4a4_sim, default off)
    DevPtr x16w = 0;
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
    // feat026-moep: the fused combine's (token, row block) counters (zeroed at allocation, left zeroed by the kernel).
    // v0251-a7port: not allocated (0): the gfx1151 default combine (moe_combine4) needs none, and WHIRL_MOEP_COMB=2 / 4
    // fall back to the plain down twin + combine (ops::chooseMoe cnt_ok), so no buffer size or estimate changes
    DevPtr moe_cnt = 0;
    // v0252-8sdet round 4 (COMB 6): how many leading 32-bit words of moe_yd are known to be zero at the stream's current
    // point (host bookkeeping, no device read). A gemm_moep[fq]di launch leaves the sums + counters it used zero (last
    // arriver: atomic swap + counter reset), so ffn.cpp launches moe_fix_zero only when a COMB 6 block needs more words
    // than this; any other MoE prefill plan (it may write moe_yd) and a throwing COMB 6 launch set it to 0; allocation too.
    std::uint64_t moe_fix_clean = 0;
    // v0251-a11port (feat026-router): the fused router's block counter, one zeroed u32 the kernel resets (graph-capture
    // safe). Not an allocation of its own: bytes 8..11 of the zeroed 16-byte moe_ntiles (moe_route writes bytes 0..3
    // only), so no buffer size or estimate changes on either card
    DevPtr moe_rcnt = 0;
};

}  // namespace whirl::qwen35
