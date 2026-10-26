// qwen35::Scratch (src/model/qwen35/scratch.h): Model::load's prefill / decode buffer allocations
// (Model::allocScratch) and their pre-load estimate (bufferBytes), next to each other (arch025-M3, R-8c).
// SPDX-License-Identifier: Apache-2.0
// Moved verbatim from src/model/loader.cpp (bufferBytes; the allocation block of Model::load). The only change in the
// block: fd_splits is the caller's fdSplitsOf(kv_fmt, env), computed right before the call (it reads the load's
// environment snapshot only).

#include "model/qwen35/scratch.h"

#include "whirl/model.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;

}  // namespace

u64 bufferBytes(const Config& cfg, u32 Bq, u64 ffs, u64 max_elems, u32 fd_splits) {
    // mirrors the buffer allocations of Model::load below (keep in step)
    const u64 f4 = 4;
    const u64 B = Bq;
    u64 n = 0;
    n += B * cfg.n_embd * f4 * 2;                         // x, h
    n += B * cfg.n_head * cfg.head_dim * 2 * f4;          // qf
    n += B * cfg.n_head_kv * cfg.head_dim * f4 * 2;       // kv_k, kv_v
    n += B * cfg.n_head * cfg.head_dim * f4;              // attn_out
    n += B * cfg.convCh() * f4 * 2;                       // qkv, conv_out
    n += B * cfg.d_inner * f4;                            // z
    n += B * cfg.n_v_heads * f4 * 2;                      // beta, alpha
    n += B * 2 * cfg.n_v_heads * f4;                      // ba_buf
    n += B * cfg.d_inner * f4;                            // gdn_out
    n += B * ffs * f4 * 2;                                // ffn_g, ffn_u
    n += static_cast<u64>(max_verify_rows) * cfg.n_vocab * f4;  // logits
    n += B * cfg.n_embd * f4;                             // hn
    n += B * 2 * cfg.n_embd * f4;                         // mtp_cat
    n += static_cast<u64>(max_small_batch) * cfg.n_embd * f4;  // mtp_h
    n += B * 4 * 3;                                       // mtp_ids, ids, pos_buf
    n += ctl_words * 4;                                   // out_tok
    const u64 n_split = fd_splits;
    n += max_verify_rows * n_split * cfg.n_head * 2 * f4;  // part_ml
    n += max_verify_rows * n_split * cfg.n_head * cfg.head_dim * f4;  // part_acc
    n += B * cfg.n_head * 2 * f4;                         // dq_st_ml (q8 / q4 / KG-3 prefill attention softmax state)
    n += (static_cast<u64>(Model::kDqMaxKeys) / 256 + 1) * 4;  // dq_ptab (identity pages for up to kDqMaxKeys keys)
    n += max_elems * 2;                                   // w16
    {
        const u64 n_chunks = (B + 63) / 64;
        const u64 heads = cfg.n_v_heads;
        n += n_chunks * heads * (64 * 128 * 4 * 2 + 64 * 64 * 4 + 64 * 4);  // gc_w, gc_u, gc_m, gc_g
    }
    const u64 in_max = std::max<u64>(std::max<u64>(cfg.n_ff, 2ull * cfg.n_embd), std::max<u64>(cfg.d_inner, static_cast<u64>(cfg.n_head) * cfg.head_dim));
    n += max_verify_rows * in_max + 256;                  // xq
    n += (max_verify_rows * in_max / 32 + 8) * 4;         // xd
    n += B * std::max<u64>(in_max, cfg.n_embd) * 2;       // x16
    n += B * 4;                                           // sx8
    if (cfg.moe) {
        const u64 Kx = cfg.n_expert_used;
        const u64 F = cfg.n_ff_exp;
        const u64 E = cfg.n_embd;
        n += B * cfg.n_expert * f4;                       // moe_logits
        n += B * Kx * 4 + B * Kx * f4 + B * f4;           // moe_ids, moe_w, moe_sg
        n += max_verify_rows * Kx * F * f4 * 2;           // moe_g, moe_u
        n += max_verify_rows * Kx * F + 256;              // moe_xq
        n += (max_verify_rows * Kx * F / 32 + 8) * 4;     // moe_xd
        n += B * E * f4;                                  // moe_ysh
        n += B * Kx * 4 * 2;                              // moe_perm, moe_inv
        n += (B * Kx / 32 + cfg.n_expert + 1) * 16;       // moe_tiles
        n += 16;                                          // moe_ntiles
        n += (B * Kx / 64 + cfg.n_expert + 1) * 16;       // moe_tiles64
        n += 16;                                          // moe_ntiles64
        n += B * Kx * std::max(E, F) * 2;                 // moe_x16
        n += B * Kx * F * f4 * 2;                         // moe_yg, moe_yu
        n += B * Kx * E * f4;                             // moe_yd
        n += B * Kx * f4;                                 // moe_sx
    }
    return n;
}

void Model::allocScratch(u32 fd_splits_) {
    Model& m = *this;
    const Config& cfg = m.cfg;
    const u64 f4 = 4;
    const u64 B = m.max_batch;
    const u64 buffers_from = m.alloc_total;
    u64 max_elems = 0;
    m.x = m.alloc(B * cfg.n_embd * f4);
    m.h = m.alloc(B * cfg.n_embd * f4);
    m.qf = m.alloc(B * cfg.n_head * cfg.head_dim * 2 * f4);
    m.kv_k = m.alloc(B * cfg.n_head_kv * cfg.head_dim * f4);
    m.kv_v = m.alloc(B * cfg.n_head_kv * cfg.head_dim * f4);
    m.attn_out = m.alloc(B * cfg.n_head * cfg.head_dim * f4);
    m.qkv = m.alloc(B * cfg.convCh() * f4);
    m.conv_out = m.alloc(B * cfg.convCh() * f4);
    m.z = m.alloc(B * cfg.d_inner * f4);
    m.beta = m.alloc(B * cfg.n_v_heads * f4);
    m.alpha = m.alloc(B * cfg.n_v_heads * f4);
    m.ba_buf = m.alloc(B * 2 * cfg.n_v_heads * f4);
    m.gdn_out = m.alloc(B * cfg.d_inner * f4);
    // ffn_g / ffn_u double as autotune / bench scratch for every matrix
    u32 ffs = cfg.n_ff;
    for (const Layer& L : m.layers)
        for (const Mat& w : m.layerMats(L).slice()) ffs = std::max(ffs, std::max(w.nrows, w.ncols));
    m.ff_scratch = ffs;
    m.ffn_g = m.alloc(B * ffs * f4);
    m.ffn_u = m.alloc(B * ffs * f4);
    m.logits = m.alloc(static_cast<u64>(max_verify_rows) * cfg.n_vocab * f4);
    m.hn = m.alloc(B * cfg.n_embd * f4);
    m.mtp_cat = m.alloc(B * 2 * cfg.n_embd * f4);
    m.mtp_h = m.alloc(static_cast<u64>(max_small_batch) * cfg.n_embd * f4);
    m.mtp_ids = m.alloc(B * 4);
    m.ids = m.alloc(B * 4);
    m.host_ids.assign(static_cast<std::size_t>(2 * B), 0);
    m.pos_buf = m.alloc(B * 4);
    m.out_tok = m.allocZero(ctl_words * 4);
    m.fd_splits = fd_splits_;
    const u64 n_split = m.fd_splits;
    m.part_ml = m.alloc(max_verify_rows * n_split * cfg.n_head * 2 * f4);
    m.part_acc = m.alloc(max_verify_rows * n_split * cfg.n_head * cfg.head_dim * f4);
    if (kvQuant(m.kv_fmt) && !kvKf16(m.kv_fmt) && (kvIs4bit(m.kv_fmt) ? m.k.kv_dq_rows_r != nullptr : m.k.kv_dq_rows != nullptr)) {
        // q8 / q8h / q4 prefill attention: f16 copies of a sequence's K / V in ffn_g / ffn_u (Model::prefillAttn)
        m.dq_rows = static_cast<u32>(std::min<u64>(B * ffs * f4 / (static_cast<u64>(cfg.n_head_kv) * cfg.head_dim * 2), 1u << 30)) / 256 * 256;
        if (m.k.kv_dq_rows_r != nullptr) m.dq_st_ml = m.alloc(B * cfg.n_head * 2 * f4);
        // identity pages for every key position (the key ranges of attn_kgs index it by sequence position)
        std::vector<std::int32_t> id(std::max<u64>(m.dq_rows, kDqMaxKeys) / 256 + 1);
        for (std::size_t i = 0; i < id.size(); ++i) id[i] = static_cast<std::int32_t>(i);
        m.dq_ptab = m.alloc(id.size() * 4);
        hip::upload(m.dq_ptab, id.data(), id.size() * 4);
    }
    // KG-3: attn_prefill_wmma_s softmax state (also f16 KV, for WHIRL_ATTN_SEG key ranges)
    if (m.dq_st_ml == 0 && m.k.attn_pws != nullptr) m.dq_st_ml = m.alloc(B * cfg.n_head * 2 * f4);
    {
        for (const Layer& L : m.layers)
            for (const Mat& w : m.layerMats(L).slice()) max_elems = std::max(max_elems, static_cast<u64>(w.ncols) * w.nrows);
        m.w16 = m.alloc(max_elems * 2);
    }
    {
        const u64 n_chunks = (B + 63) / 64;
        const u64 heads = cfg.n_v_heads;
        m.gc_w = m.alloc(n_chunks * heads * 64 * 128 * 4);
        m.gc_u = m.alloc(n_chunks * heads * 64 * 128 * 4);
        m.gc_m = m.alloc(n_chunks * heads * 64 * 64 * 4);
        m.gc_w_bytes = n_chunks * heads * 64 * 128 * 4;
        m.gc_u_bytes = m.gc_w_bytes;
        m.gc_m_bytes = n_chunks * heads * 64 * 64 * 4;
        m.gc_g = m.alloc(n_chunks * heads * 64 * 4);
    }
    // widest decode matmul input: FFN, DeltaNet / attention output, MTP eh_proj
    const u64 in_max = std::max<u64>(std::max<u64>(cfg.n_ff, 2ull * cfg.n_embd), std::max<u64>(cfg.d_inner, static_cast<u64>(cfg.n_head) * cfg.head_dim));
    m.xq = m.alloc(max_verify_rows * in_max + 256);
    m.xd = m.alloc((max_verify_rows * in_max / 32 + 8) * 4);
    m.x16_bytes = B * std::max<u64>(in_max, cfg.n_embd) * 2;
    m.x16 = m.alloc(m.x16_bytes);
    // GDN prefill x16b (fp8 rows + f16 rows from one norm) must fit in x16 for any batch > 16
    if (B > 16 && m.x16bOff() + B * cfg.n_embd * 2 > m.x16_bytes) throw ModelError("X16bOverflow");
    m.sx8 = m.alloc(B * 4);
    // (arch025-N2b: fp8_prefill is numerics::plan's, in loadOrTune: the item fp8, WHIRL_FP8 overrides)
    if (cfg.moe) {
        const u64 Kx = cfg.n_expert_used;
        const u64 F = cfg.n_ff_exp;
        const u64 E = cfg.n_embd;
        m.moe_logits = m.alloc(B * cfg.n_expert * f4);
        m.moe_ids = m.allocZero(B * Kx * 4);
        m.moe_w = m.allocZero(B * Kx * f4);
        m.moe_sg = m.allocZero(B * f4);
        // verify batches up to max_verify_rows rows run on the decode experts (precise, and the
        // balance / fast wide verify)
        m.moe_g = m.alloc(max_verify_rows * Kx * F * f4);
        m.moe_u = m.alloc(max_verify_rows * Kx * F * f4);
        m.moe_xq = m.alloc(max_verify_rows * Kx * F + 256);
        m.moe_xd = m.alloc((max_verify_rows * Kx * F / 32 + 8) * 4);
        m.moe_ysh = m.allocZero(B * E * f4);
        m.moe_perm = m.alloc(B * Kx * 4);
        m.moe_inv = m.alloc(B * Kx * 4);
        m.moe_tiles = m.alloc((B * Kx / 32 + cfg.n_expert + 1) * 16);
        m.moe_ntiles = m.allocZero(16);
        m.moe_tiles64 = m.alloc((B * Kx / 64 + cfg.n_expert + 1) * 16);
        m.moe_ntiles64 = m.allocZero(16);
        m.moe_x16 = m.alloc(B * Kx * std::max(E, F) * 2);
        m.moe_yg = m.alloc(B * Kx * F * f4);
        m.moe_yu = m.alloc(B * Kx * F * f4);
        m.moe_yd = m.alloc(B * Kx * E * f4);
        m.moe_sx = m.alloc(B * Kx * f4);
    }
    // sizing self-check: the pre-load estimate (bufferBytes, the server's auto-shrink) must cover what was just
    // allocated; a shortfall means an allocation was added without updating the estimate
    if (const u64 used = m.alloc_total - buffers_from, est = bufferBytes(cfg, m.max_batch, ffs, max_elems, m.fd_splits); used > est)
        m.load_note += "warning: the VRAM estimate for the prefill buffers (" + std::to_string(est >> 20) + " MiB) is below what was allocated (" +
                       std::to_string(used >> 20) + " MiB); please report\n";
}

}  // namespace whirl::qwen35
