// qwen35::Scratch (src/model/qwen35/scratch.h): the pre-load estimate (bufferBytes) of Model::load's prefill / decode
// buffer allocations (arch025-M3, R-8c). Model::allocScratch, the allocations themselves, moved verbatim to
// src/model/qwen35/model_scratch.cpp (the model core; arch025-lane2 N9 S2c) so this file does not include the Model
// class; Model::kDqMaxKeys is ops::kDqMaxKeys (its alias in model.h).
// SPDX-License-Identifier: Apache-2.0
// Moved verbatim from src/model/loader.cpp (bufferBytes; the allocation block of Model::load). The only change in the
// block: fd_splits is the caller's fdSplitsOf(kv_fmt, env), computed right before the call (it reads the load's
// environment snapshot only).

#include "model/qwen35/scratch.h"

#include "whirl/model_types.h"     // Config, max_verify_rows, max_small_batch, ctl_words
#include "whirl/ops/attention.h"  // ops::kDqMaxKeys (= Model::kDqMaxKeys)

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
    n += (static_cast<u64>(ops::kDqMaxKeys) / 256 + 1) * 4;  // dq_ptab (identity pages for up to kDqMaxKeys keys)
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

}  // namespace whirl::qwen35
