// Model forward, attention block (arch025-F2, moved from src/model/forward.cpp): the MTP draft attention
// (draftAttn, draft window OVH-C1), the gated full-attention sublayer attnBlock and the WMMA prefill attention.
// The kernel choices are in src/ops/attention.cpp (chooseAttnPrep / chooseAttnCore, arch025-F6; chooseAttnDecode /
// chooseAttnPrefill, K-3); here the problems are built and the plans launched.
// SPDX-License-Identifier: Apache-2.0

#include "../forward_internal.h"

namespace whirl::qwen35 {

// Gated full-attention sublayer on h (already attn-normed) for n tokens at
// pos_buf, adding into x.
void Model::draftAttn(const AttnW& a, u32 n) {
    // restored on every exit, also when attnBlock throws (OVH-C1)
    ValueScope<std::int32_t> draft_win(attn_win, draftWindowArg(nf().draft_window, nf().draft_window_min));
    attnBlock(a, mtpKv(), n);
}

void Model::attnBlock(const AttnW& a, const KvLayer& lkv, u32 n) {
    if (attn_test_hook) attn_test_hook(attn_test_ctx, attn_win);
    const u32 hd = cfg.head_dim;
    const bool batched = bplan != nullptr || mtp_batch;
    const DevPtr kvb = batched ? kvbase_buf : 0;
    KvArgs kva = kvArgs(lkv, kvb, kv_off);
    kva.win = attn_win;  // 0 except for the MTP drafts (only attn_wsplit reads it)
    {
        const Mat ws[3] = {a.q, a.k, a.v};
        const DevPtr ys[3] = {qf, kv_k, kv_v};
        matmulGroup(ws, h, ys, n);
    }
    const float theta_scale = std::pow(cfg.rope_base, -2.0f / static_cast<float>(cfg.n_rot));
    if (!psegs.empty()) {
        // segmented prefill: each sequence's rows against its own cache region
        const float scale_s = 1.0f / std::sqrt(static_cast<float>(hd));
        for (const PRows& ps : psegs) {
            selectSeq(ps.seq);
            const KvArgs kvs = kvArgs(lkv, 0, kv_off);
            const u64 r0 = ps.r0;
            const DevPtr qf_s = qf + r0 * cfg.n_head * 2 * hd * 4;
            const DevPtr pos_s = pos_buf + r0 * 4;
            if (ops::chooseAttnPrepSegmented(mrope_on) == ops::AttnPrep::mrope) {
                const auto& sec = cfg.rope_sections;
                hip::launch(k.attn_prep_m, D(cfg.n_head + cfg.n_head_kv, ps.n), D(256), 0, stream, qf_s, kv_k + r0 * cfg.n_head_kv * hd * 4,
                            kv_v + r0 * cfg.n_head_kv * hd * 4, a.q_norm, a.k_norm, kvs, pos_s, I(cfg.n_head), I(cfg.n_head_kv), I(hd),
                            I(cfg.n_rot), theta_scale, cfg.eps, rpos_rows ? rpos_buf + r0 * 12 : u64(0), rdelta, I(sec[0]), I(sec[1]), I(sec[2]));
            } else
                hip::launch(k.attn_prep, D(cfg.n_head + cfg.n_head_kv, ps.n), D(256), 0, stream, qf_s, kv_k + r0 * cfg.n_head_kv * hd * 4,
                            kv_v + r0 * cfg.n_head_kv * hd * 4, a.q_norm, a.k_norm, kvs, pos_s, I(cfg.n_head), I(cfg.n_head_kv), I(hd), I(cfg.n_rot),
                            theta_scale, cfg.eps);
            prefillAttn(qf_s, kvs, attn_out + r0 * cfg.n_head * hd * 4, pos_s, ps.n, ps.pos0, scale_s);
        }
        xq_src = 0;
        x16_src = 0;
        mark(OpClass::attn);
        matmul(a.o, attn_out, x, n, true);
        return;
    }
    // (q8 KV: always the fused prep - it holds the quantizer and the rotation)
    const ops::AttnPrep prep = ops::chooseAttnPrep(mrope_on, hd, kvQuant(kv_fmt), nf().no_fuse, k);
    if (prep == ops::AttnPrep::mrope) {
        // vision: image prompt / decode after one: multi-section RoPE positions
        const auto& sec = cfg.rope_sections;
        hip::launch(k.attn_prep_m, D(cfg.n_head + cfg.n_head_kv, n), D(256), 0, stream, qf, kv_k, kv_v, a.q_norm, a.k_norm, kva, pos_buf,
                    I(cfg.n_head), I(cfg.n_head_kv), I(hd), I(cfg.n_rot), theta_scale, cfg.eps, rpos_rows ? rpos_buf : u64(0), rdelta,
                    I(sec[0]), I(sec[1]), I(sec[2]));
        xq_src = 0;
        x16_src = 0;
    } else if (prep == ops::AttnPrep::fused) {
        // q/k norm + RoPE + K/V store in one launch (same arithmetic)
        hip::launch(k.attn_prep, D(cfg.n_head + cfg.n_head_kv, n), D(256), 0, stream, qf, kv_k, kv_v, a.q_norm, a.k_norm, kva, pos_buf,
                    I(cfg.n_head), I(cfg.n_head_kv), I(hd), I(cfg.n_rot), theta_scale, cfg.eps);
        xq_src = 0;
        x16_src = 0;
    } else {
        rmsnorm(qf, a.q_norm, qf, hd, n * cfg.n_head, 2 * hd, 2 * hd);
        rmsnorm(kv_k, a.k_norm, kv_k, hd, n * cfg.n_head_kv, hd, hd);
        hip::launch(k.rope_neox, D(cfg.n_head, n), D(64), 0, stream, qf, pos_buf, I(2 * hd), I(2 * hd * cfg.n_head), I(cfg.n_rot), theta_scale);
        hip::launch(k.rope_neox, D(cfg.n_head_kv, n), D(64), 0, stream, kv_k, pos_buf, I(hd), I(hd * cfg.n_head_kv), I(cfg.n_rot), theta_scale);
        const u32 row = cfg.n_head_kv * hd;
        hip::launch(k.kv_store, D(n), D(256), 0, stream, kv_k, kv_v, kva, pos_buf, I(row));
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const ops::AttnCore core = ops::chooseAttnCore(n, small_max, hd, nf().naive_attn);
    if (core == ops::AttnCore::decode) {
        // flash-decoding; grid.z = query (speculative verify), query t sees positions up to pos_buf[t]. The kernel,
        // its query groups, the split count and the combine come from ops::chooseAttnDecode (K-3).
        ops::AttnDecodeProblem ap;
        ap.n = n;
        ap.head_dim = hd;
        ap.n_head = cfg.n_head;
        ap.n_head_kv = cfg.n_head_kv;
        ap.max_ctx = max_ctx;
        ap.kv_q4 = kvIs4bit(kv_fmt);
        ap.rows_known = row_n == n;
        if (ap.rows_known) {
            ap.row_pos = std::span<const i32>(row_pos.data(), n);
            ap.row_base = std::span<const i32>(row_base.data(), n);
        }
        ap.win = kva.win;
        ap.o_on_gemv = fused(n) && k.gemvq[ti(a.o.ty)] != nullptr && n <= gvMax();
        ops::AttnDecodeFlags af;
        af.attn_ws4 = nf().attn_ws4;
        af.attn_dq4 = nf().attn_dq4;
        af.dq4_dyn = nf().dq4_dyn;
        af.attn_wide = nf().attn_wide;
        af.fd_splits = fd_splits;
        af.dbg_flags = nf().dbg_flags;
        af.prec_dec = nf().prec_dec;
        const ops::AttnDecodePlan pl = ops::chooseAttnDecode(ap, af, k, k.caps, device::profile(arch));
        if (pl.kind == ops::AttnDecodeKind::split)
            hip::launch(pl.f, pl.grid, pl.block, 0, stream, qf, kva, part_ml, part_acc, I(cfg.n_head), I(cfg.n_head_kv), I(hd), I(2 * hd), pos_buf,
                        scale, i32(1), gate);
        else
            hip::launch(pl.f, pl.grid, pl.block, 0, stream, qf, kva, part_ml, part_acc, I(cfg.n_head), I(cfg.n_head_kv), I(2 * hd), pos_buf, scale,
                        gate, pl.groups);
        if (pl.combine_q8) {
            // the o-projection's int8 input comes out of the combine
            hip::launch(k.attn_combine_q8, pl.combine_grid, pl.combine_block, 0, stream, part_ml, part_acc, qf, attn_out, I(cfg.n_head), I(hd),
                        I(2 * hd), I(pl.n_split), xq, xd);
            mark(OpClass::attn);
            xq_src = attn_out;
            xq_n = n;
            x16_src = 0;
            matmul(a.o, attn_out, x, n, true);
            return;
        }
        hip::launch(k.attn_combine, pl.combine_grid, pl.combine_block, 0, stream, part_ml, part_acc, qf, attn_out, I(cfg.n_head), I(hd), I(2 * hd),
                    I(pl.n_split));
    } else {
        if (core == ops::AttnCore::prefill) {
            prefillAttn(qf, kva, attn_out, pos_buf, n, fwd_pos0, scale);
        } else {
            if (scores == 0) scores = alloc(static_cast<u64>(max_batch) * cfg.n_head * max_ctx * 4);
            hip::launch(k.attn_decode, D(cfg.n_head, n), D(256), 0, stream, qf, kva, attn_out, scores, I(cfg.n_head), I(cfg.n_head_kv), I(hd),
                        I(2 * hd), I(max_ctx), pos_buf, scale);
        }
    }
    xq_src = 0;
    x16_src = 0;
    mark(OpClass::attn);
    matmul(a.o, attn_out, x, n, true);
}

// WMMA prefill attention of n queries at positions pos0 .. (keys 0 .. pos0 + n - 1): the launches (head ranges,
// key ranges, K / V dequant copies into ffn_g / ffn_u) come from ops::chooseAttnPrefill (K-3); here the KV
// arguments are built (the cache, or the copy whose row r0 is row 0 of the range).
// The prefill attention selector flags. `prec` is the numerics MODE (precise), not prec_dec: --precise with
// WHIRL_Q8DEC=1 has prec_dec false, and attn_kgw (WHIRL_ATTN_KGVAR=2, not bit-identical) must still be refused.
ops::AttnPrefillFlags Model::prefillAttnFlags() const {
    ops::AttnPrefillFlags pf;
    pf.attn_kg_on = nf().attn_kg_on;
    pf.attn_kg_var = nf().attn_kg_var;
    pf.prec = num_plan.mode == numerics::Mode::precise;
    pf.attn_kx_on = nf().attn_kx_on;
    pf.attn_dqf_on = nf().attn_dqf_on;
    pf.attn_seg = nf().attn_seg;
    return pf;
}

void Model::prefillAttn(DevPtr q, const KvArgs& kva, DevPtr out, DevPtr pos, u32 n, u32 pos0, float scale) {
    ops::AttnPrefillProblem pp;
    pp.n = n;
    pp.pos0 = pos0;
    pp.head_dim = cfg.head_dim;
    pp.n_head = cfg.n_head;
    pp.n_head_kv = cfg.n_head_kv;
    pp.scale = scale;
    pp.kv_q8 = kvQuant(kv_fmt);
    pp.dq_rows = dq_rows;
    pp.dq_ptab = dq_ptab != 0;
    pp.dq_st_ml = dq_st_ml != 0;
    const ops::AttnPrefillFlags pf = prefillAttnFlags();
    const ops::AttnPrefillPlan pl = ops::chooseAttnPrefill(pp, pf, k);
    const i32 H = I(cfg.n_head), HKV = I(cfg.n_head_kv), QS = I(2 * cfg.head_dim);
    for (const ops::AttnStep& s : pl.steps) {
        switch (s.op) {
            case ops::AttnStep::Op::dq_rows:
                hip::launch(s.f, s.grid, s.block, 0, stream, kva, ffn_g, ffn_u, I(s.nrows), I(s.row_el));
                break;
            case ops::AttnStep::Op::dq_rows_r:
                hip::launch(s.f, s.grid, s.block, 0, stream, kva, ffn_g, ffn_u, I(s.r0), I(s.nrows), I(s.row_el));
                break;
            case ops::AttnStep::Op::attn: {
                KvArgs kv = kva;
                if (s.from_copy) {
                    const u64 row_el = static_cast<u64>(cfg.n_head_kv) * cfg.head_dim;
                    kv = KvArgs{};
                    kv.k = ffn_g - static_cast<u64>(s.r0) * row_el * 2;  // row r0 of the sequence = row 0 of the copy
                    kv.v = ffn_u - static_cast<u64>(s.r0) * row_el * 2;
                    kv.ptab = dq_ptab;
                }
                if (s.ranged)
                    hip::launch(s.f, s.grid, s.block, 0, stream, q, kv, out, H, HKV, QS, pos, I(n), scale, I(s.head0), I(s.r0), I(s.r1), dq_st_ml);
                else
                    hip::launch(s.f, s.grid, s.block, 0, stream, q, kv, out, H, HKV, QS, pos, I(n), scale, I(s.head0));
                break;
            }
        }
    }
}

}  // namespace whirl::qwen35
