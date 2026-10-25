// Model forward: matmul dispatch, attention / DeltaNet / FFN / MoE blocks,
// prefill, decode, MTP drafting and speculative verify.
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (Model: gemvLaunch .. readLogits).

#include "forward_internal.h"

namespace whirl::qwen35 {

void Model::forward(std::span<const u32> tokens, u32 pos0) {
    const u32 n = static_cast<u32>(tokens.size());
    if (n == 0 || n > max_batch) throw ModelError("BatchTooLarge");
    xq_src = 0;
    x16_src = 0;
    if (pos0 + n > max_ctx) throw ModelError("ContextTooLong");
    setTokens(ids, tokens, pos0);
    fwd_pos0 = pos0;
    run(n);
}

// Decode one token entirely from device state: the input token is ids[0] and
// its position pos_buf[0] (both written by the previous argmax), so the launch
// sequence is identical every step and never waits on the host.
void Model::decodeStep() {
    if (!use_graph || prof != nullptr) return decodeStepLaunches();
    if (!decode_graph) {
        hip::Graph::beginCapture(stream);
        try {
            decodeStepLaunches();
        } catch (...) {
            try {
                (void)hip::Graph::endCapture(stream);
            } catch (...) {
            }
            throw;
        }
        decode_graph = std::make_unique<hip::Graph>(hip::Graph::endCapture(stream));
    }
    decode_graph->launch(stream);
}

void Model::decodeStepLaunches() {
    // device positions (argmax advances pos_buf): the host row positions are not known here
    row_n = 0;
    // vision: device positions (argmax advances pos_buf): RoPE position = pos - delta
    visBegin();
    if (const VisMap* vm = visOf(cur_seq)) {
        if (k.attn_prep_m == nullptr) throw ModelError("MropeUnsupported");
        mrope_on = true;
        rdelta = vm->deltaEnd();
    }
    run(1);
    hip::launch(k.argmax, D(1), D(1024), 0, stream, logits, I(cfg.n_vocab), out_tok, ids, pos_buf, i32(1));
}

// Seed device state for decodeStep after a prefill of n_past tokens.
u32 Model::beginDecode(u32 n_past) {
    const i32 p = static_cast<i32>(n_past);
    hip::upload(pos_buf, &p, 4);
    hip::launch(k.argmax, D(1), D(1024), 0, stream, logits, I(cfg.n_vocab), out_tok, ids, pos_buf, i32(0));
    i32 out = 0;
    hip::download(&out, out_tok, 4);
    return static_cast<u32>(out);
}

// Gated full-attention sublayer on h (already attn-normed) for n tokens at
// pos_buf, adding into x.
void Model::draftAttn(const AttnW& a, u32 n) {
    // restored on every exit, also when attnBlock throws (OVH-C1)
    ValueScope<std::int32_t> draft_win(attn_win, draftWindowArg(draft_window, draft_window_min));
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
            if (mrope_on) {
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
    if (mrope_on) {
        // vision: image prompt / decode after one: multi-section RoPE positions
        const auto& sec = cfg.rope_sections;
        hip::launch(k.attn_prep_m, D(cfg.n_head + cfg.n_head_kv, n), D(256), 0, stream, qf, kv_k, kv_v, a.q_norm, a.k_norm, kva, pos_buf,
                    I(cfg.n_head), I(cfg.n_head_kv), I(hd), I(cfg.n_rot), theta_scale, cfg.eps, rpos_rows ? rpos_buf : u64(0), rdelta,
                    I(sec[0]), I(sec[1]), I(sec[2]));
        xq_src = 0;
        x16_src = 0;
    } else if (k.attn_prep != nullptr && (!no_fuse || kv_q8) && hd <= 512) {
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
    if (n <= small_max) {
        // flash-decoding; grid.z = query (speculative verify), query t sees positions up to pos_buf[t]. The kernel,
        // its query groups, the split count and the combine come from ops::chooseAttnDecode (K-3).
        ops::AttnDecodeProblem ap;
        ap.n = n;
        ap.head_dim = hd;
        ap.n_head = cfg.n_head;
        ap.n_head_kv = cfg.n_head_kv;
        ap.max_ctx = max_ctx;
        ap.kv_q4 = kv_q4;
        ap.rows_known = row_n == n;
        if (ap.rows_known) {
            ap.row_pos = std::span<const i32>(row_pos.data(), n);
            ap.row_base = std::span<const i32>(row_base.data(), n);
        }
        ap.win = kva.win;
        ap.o_on_gemv = fused(n) && k.gemvq[ti(a.o.ty)] != nullptr && n <= gvMax();
        ops::AttnDecodeFlags af;
        af.attn_ws4 = attn_ws4;
        af.attn_dq4 = attn_dq4;
        af.dq4_dyn = dq4_dyn;
        af.attn_wide = attn_wide;
        af.fd_splits = fd_splits;
        af.dbg_flags = dbg_flags;
        af.prec_dec = prec_dec;
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
        if (hd == 256 && !naive_attn) {
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
void Model::prefillAttn(DevPtr q, const KvArgs& kva, DevPtr out, DevPtr pos, u32 n, u32 pos0, float scale) {
    ops::AttnPrefillProblem pp;
    pp.n = n;
    pp.pos0 = pos0;
    pp.head_dim = cfg.head_dim;
    pp.n_head = cfg.n_head;
    pp.n_head_kv = cfg.n_head_kv;
    pp.scale = scale;
    pp.kv_q8 = kv_q8;
    pp.dq_rows = dq_rows;
    pp.dq_ptab = dq_ptab != 0;
    pp.dq_st_ml = dq_st_ml != 0;
    ops::AttnPrefillFlags pf;
    pf.attn_kg_on = attn_kg_on;
    pf.attn_kx_on = attn_kx_on;
    pf.attn_dqf_on = attn_dqf_on;
    pf.attn_seg = attn_seg;
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

// Post-attention norm + SwiGLU FFN on x, residual added into x.
void Model::ffnBlock(DevPtr post_norm, const Mat& gate_w, const Mat& up, const Mat& down, u32 n) {
    const Mat gu[2] = {gate_w, up};
    if (fused(n))
        rmsnormQ8(x, post_norm, h, n);
    else
        rmsnormIn(x, post_norm, h, n, gu, 2);
    struct ClassReset {
        std::uint8_t& c;
        ~ClassReset() { c = 1; }
    } class_reset{mm_class};
    mm_class = 2;
    std::optional<ActIn> down_in;
    if (!fused(n) && cfg.n_ff % 4 == 0 && cfg.n_ff <= 1024 * 24) down_in = actCommon(std::span<const Mat>(&down, 1), n, 4);
    // f16 gate / up outputs straight into silu -> fp8 (MXFP4 speed mode)
    const ActIn gu_in = actCommon(gu, n, 2).value_or(ActIn::gemv);
    const bool h16 = ffn_h16 && down_in.has_value() && cfg.n_ff % 8 == 0 && h16Out(gate_w, n, 2) && h16Out(up, n, 2) &&
                     ((gu_in == ActIn::fp8 && *down_in == ActIn::fp8 && k.silu_mul_x8h != nullptr) ||
                      (gu_in == ActIn::f16 && *down_in == ActIn::f16 && k.silu_mul_x16h != nullptr));
    out_h16 = h16;
    {
        const DevPtr ys[2] = {ffn_g, ffn_u};
        matmulGroup(gu, h, ys, n);
    }
    out_h16 = false;
    mm_class = 4;
    hip::Function silu_x = nullptr;
    if (h16)
        silu_x = *down_in == ActIn::fp8 ? x8(k.silu_mul_x8h, k.silu_mul_x8ht) : k.silu_mul_x16h;
    else if (down_in)
        silu_x = *down_in == ActIn::fp8 ? x8(k.silu_mul_x8, k.silu_mul_x8t) : k.silu_mul_x16;
    if (fused(n) && ops::preciseAllows(ops::BitClass::q8_row, prec_dec, n, small_max)) {  // fused(n): n <= small_max
        hip::launch(k.silu_mul_q8, D(n * cfg.n_ff / 256), D(256), 0, stream, ffn_g, ffn_u, xq, xd, I(n * cfg.n_ff));
        xq_src = ffn_g;
        xq_n = n;
        x16_src = 0;
        mark(OpClass::misc);
    } else if (silu_x != nullptr) {
        // silu(g) * u straight into the down GEMM's input (x16 as ffn_g's conversion)
        hip::launch(silu_x, D(n), D(256), 0, stream, ffn_g, ffn_u, x16, sx8, I(cfg.n_ff));
        xq_src = 0;
        x16_src = ffn_g;
        x16_count = n * cfg.n_ff;
        x16_fp8 = *down_in == ActIn::fp8;
        mark(OpClass::misc);
    } else {
        elementwise(k.silu_mul, ffn_g, ffn_u, n * cfg.n_ff);
    }
    matmul(down, ffn_g, x, n, true);
}

// Post-attention norm + MoE FFN (qwen35moe) on x, residual added into x:
// softmax router -> top-K experts (weights renormalized) plus the shared
// expert scaled by sigmoid(sh_gate . h). Small n: int8 GEMVs over the selected
// experts' rows; larger n: pairs grouped by expert, grouped WMMA GEMM.
void Model::moeBlock(DevPtr post_norm, const Mat& sg, const Mat& su, const Mat& sd, const MoeW& mo, u32 n) {
    const u32 E = cfg.n_embd;
    const u32 K = cfg.n_expert_used;
    const u32 F = cfg.n_ff_exp;
    const u32 R = cfg.n_expert;
    // precise mode: every verify / decode batch (up to max_verify_rows rows, e.g. a long
    // n-gram draft) takes the decode experts, so a generated token's MoE output never depends
    // on whether its batch had more than 16 rows (MTP + n-gram == plain). balance / fast: a wide
    // verify / MTP batch (small_max raised to max_verify_rows) takes the int8 decode experts too,
    // never the prefill grouped GEMM (fp8 / f16) - every row as its own 1-token decode
    // the experts' kernels, grids and tiles: ops::chooseMoe (K-4)
    ops::MoeProblem mp;
    mp.n = n;
    mp.E = E;
    mp.K = K;
    mp.F = F;
    mp.R = R;
    mp.gate_ty = mo.gate.ty;
    mp.up_ty = mo.up.ty;
    mp.down_ty = mo.down.ty;
    mp.gate_rb = mo.gate.row_bytes;
    mp.up_rb = mo.up.row_bytes;
    mp.gate_ncols = mo.gate.ncols;
    mp.down_ncols = mo.down.ncols;
    mp.refs = mo.gate.ref != 0 && mo.up.ref != 0 && mo.down.ref != 0;
    mp.sx_buf = moe_sx != 0;
    mp.tiles64_buf = moe_tiles64 != 0;
    ops::MoeFlags mf;
    mf.prec_dec = prec_dec;
    mf.small_max = small_max;
    mf.max_verify_rows = max_verify_rows;
    mf.max_small_batch = max_small_batch;
    mf.moe_mxw = moe_mxw;
    mf.moe_rbf = moe_rbf;
    mf.moe_fp8 = moe_fp8;
    mf.moe_bn_force = moe_bn_force;
    const ops::MoePlan pl = ops::chooseMoe(mp, mf, k, k.silu_mul_x8h);
    const bool small = pl.small;
    // precise decode: f32-activation experts (moe_gu_f / moe_down_f) and the shared expert on
    // the precise matmul path (no int8 activations); a token's output is the same alone or in
    // a verify / concurrent batch
    const bool small_q8 = pl.small_q8;
    if (small_q8)
        rmsnormQ8(x, post_norm, h, n);
    else
        rmsnorm(x, post_norm, h, E, n, E, E);
    // router logits (f32 activations for small batches, also in precise mode)
    if (small) {
        // 16 tokens per launch (the kernel's tile); each token's logits are the same in any piece
        for (u32 t0 = 0; t0 < n; t0 += max_small_batch)
            hip::launch(k.moe_logits_f32, pl.logits_grid, D(256), 0, stream, mo.router.ptr, h + static_cast<u64>(t0) * E * 4,
                        moe_logits + static_cast<u64>(t0) * R * 4, I(R), I(E), I(std::min(max_small_batch, n - t0)), gate);
        mark(OpClass::matmul);
    } else {
        matmul(mo.router, h, moe_logits, n, false);
    }
    hip::launch(k.moe_topk, D(n), D(256), 0, stream, moe_logits, I(R), h, mo.sh_gate, moe_ids, moe_w, moe_sg, I(R), I(K), I(E));
    mark(OpClass::misc);
    if (moe_dump) {
        const std::size_t old = moe_dump->size();
        moe_dump->resize(old + static_cast<std::size_t>(n) * K);
        hip::sync();
        hip::download(moe_dump->data() + old, moe_ids, static_cast<std::size_t>(n) * K * 4);
    }
    const DevPtr ysh = mo.sh_gate != 0 ? moe_ysh : 0;
    if (small && prec_dec) {
        // precise decode experts: f32 h (no int8 / f16 rounding), f32 accumulation, every
        // pair on its own waves (a token's result does not depend on the batch)
        if (!pl.gu_split) {
            hip::launch(pl.gu, pl.gu_grid, D(256), 0, stream, mo.gate.ptr, mo.up.ptr, mo.gate.row_bytes, I(F), I(2 * F), h, moe_ids, moe_g, moe_u, I(E),
                        I(K));
        } else {
            hip::launch(pl.gu, pl.gu_grid, D(256), 0, stream, mo.gate.ptr, mo.gate.ptr, mo.gate.row_bytes, I(F), I(F), h, moe_ids, moe_g, moe_g, I(E),
                        I(K));
            hip::launch(pl.gu_up, pl.gu_grid, D(256), 0, stream, mo.up.ptr, mo.up.ptr, mo.up.row_bytes, I(F), I(F), h, moe_ids, moe_u, moe_u, I(E),
                        I(K));
        }
        if (prof) prof->matmul_bytes += (mo.gate.row_bytes + mo.up.row_bytes) * F * n * K;
        mark(OpClass::matmul);
        if (ysh != 0) {
            const Mat ws[2] = {sg, su};
            const DevPtr ys[2] = {ffn_g, ffn_u};
            matmulGroup(ws, h, ys, n);
            elementwise(k.silu_mul, ffn_g, ffn_u, n * cfg.n_ff);
            matmul(sd, ffn_g, ysh, n, false);
        }
        elementwise(k.silu_mul, moe_g, moe_u, n * K * F);
        hip::launch(pl.down, pl.down_grid, D(256), 0, stream, mo.down.ptr, mo.down.row_bytes, I(E), moe_g, moe_ids, moe_w, moe_sg, ysh, x, I(F),
                    I(K));
        if (prof) prof->matmul_bytes += mo.down.row_bytes * E * n * K;
        xq_src = 0;
        x16_src = 0;
        mark(OpClass::matmul);
        return;
    }
    if (small_q8) {
        // experts: gate + up from the int8 h that rmsnormQ8 left in xq
        const u64 gu_rb = mo.gate.row_bytes;
        if (!pl.gu_split) {
            hip::launch(pl.gu, pl.gu_grid, D(256), 0, stream, mo.gate.ptr, mo.up.ptr, gu_rb, I(F), I(2 * F), xq, xd, moe_ids, moe_g, moe_u, I(E), I(K),
                        gate);
        } else {
            hip::launch(pl.gu, pl.gu_grid, D(256), 0, stream, mo.gate.ptr, mo.gate.ptr, gu_rb, I(F), I(F), xq, xd, moe_ids, moe_g, moe_g, I(E), I(K),
                        gate);
            hip::launch(pl.gu_up, pl.gu_grid, D(256), 0, stream, mo.up.ptr, mo.up.ptr, mo.up.row_bytes, I(F), I(F), xq, xd, moe_ids, moe_u, moe_u,
                        I(E), I(K), gate);
        }
        if (prof) prof->matmul_bytes += (mo.gate.row_bytes + mo.up.row_bytes) * F * n * K;
        mark(OpClass::matmul);
        // shared expert (xq still holds h)
        if (ysh != 0) {
            const Mat ws[2] = {sg, su};
            const DevPtr ys[2] = {ffn_g, ffn_u};
            matmulGroup(ws, h, ys, n);
            hip::launch(k.silu_mul_q8, D(n * cfg.n_ff / 256), D(256), 0, stream, ffn_g, ffn_u, xq, xd, I(n * cfg.n_ff));
            xq_src = ffn_g;
            xq_n = n;
            x16_src = 0;
            mark(OpClass::misc);
            matmul(sd, ffn_g, ysh, n, false);
        }
        hip::launch(k.silu_mul_q8, D(n * K * F / 256), D(256), 0, stream, moe_g, moe_u, moe_xq, moe_xd, I(n * K * F));
        mark(OpClass::misc);
        hip::launch(pl.down, pl.down_grid, D(256), 0, stream, mo.down.ptr, mo.down.row_bytes, I(E), moe_xq, moe_xd, moe_ids, moe_w, moe_sg, ysh, x,
                    I(F), I(K), gate);
        if (prof) prof->matmul_bytes += mo.down.row_bytes * E * n * K;
        xq_src = 0;
        x16_src = 0;
        mark(OpClass::matmul);
        return;
    }
    // prefill: group (token, slot) pairs by expert, grouped GEMMs, combine
    const u32 pairs = n * K;
    const u32 bn = pl.bn;
    hip::launch(k.moe_route, D(1), D(1024), 0, stream, moe_ids, I(pairs), I(R), I(bn), moe_perm, moe_inv, moe_tiles, moe_ntiles);
    if (pl.kind == ops::MoePrefillKind::fp8) {
        // MXFP4 experts x fp8 activations (per-pos scale, folded E8M0 as gemm8): gate + up in
        // one launch with f16 outputs, silu(g) * u -> fp8 rows, down with f32 output
        hip::launch(k.moe_gather_fp8, D(pairs), D(256), 0, stream, h, moe_perm, moe_x16, moe_sx, I(E), I(K));
        mark(OpClass::misc);
        // row-block-fast variants (same outputs): 1-D grid, the blocks sharing a tile's rows together
        hip::launch(pl.fgu, pl.gu_pgrid, D(256), 0, stream, mo.gate.ptr, mo.up.ptr, mo.gate.row_bytes, mo.gate.ref, mo.up.ref, I(F), moe_x16, moe_sx,
                    moe_yg, moe_yu, I(E), moe_tiles, moe_ntiles);
        mark(OpClass::matmul);
        hip::launch(k.silu_mul_x8h, D(pairs), D(256), 0, stream, moe_yg, moe_yu, moe_x16, moe_sx, I(F));
        mark(OpClass::misc);
        hip::launch(pl.fdn, pl.dn_pgrid, D(256), 0, stream, mo.down.ptr, DevPtr{0}, mo.down.row_bytes, mo.down.ref, DevPtr{0}, I(E), moe_x16, moe_sx,
                    moe_yd, DevPtr{0}, I(F), moe_tiles, moe_ntiles);
        mark(OpClass::matmul);
    } else if (pl.kind == ops::MoePrefillKind::fused_gu) {
        // Radeon 8060S: gate + up + SwiGLU in one grouped GEMM over 64-token tiles (A, f16,
        // into the f32 gate buffer: X is still being read), down over 32-token tiles; the
        // same perm positions for both tile lists, every row bit-identical to the path below
        hip::launch(k.moe_tiles, D(1), D(1024), 0, stream, moe_ids, I(pairs), I(R), I(64), moe_tiles64, moe_ntiles64);
        hip::launch(k.moe_gather_f16, D(pairs), D(256), 0, stream, h, moe_perm, moe_x16, I(E), I(K));
        mark(OpClass::misc);
        hip::launch(pl.fgu, pl.gu_pgrid, D(256), 0, stream, mo.gate.ptr, mo.up.ptr, mo.gate.row_bytes, I(F), moe_x16, moe_yg, I(E), moe_tiles64,
                    moe_ntiles64);
        hip::launch(pl.fdn, pl.dn_pgrid, D(256), 0, stream, mo.down.ptr, mo.down.row_bytes, I(E), moe_yg, moe_yd, I(F), moe_tiles, moe_ntiles);
        mark(OpClass::matmul);
    } else if (pl.kind == ops::MoePrefillKind::gemmr) {
        // MXFP4 experts, f16 activations (precise): gate + up + SwiGLU in one grouped GEMM
        // into the f32 gate buffer as f16 (X is still being read), then down; row block
        // fastest grid, empty 16-token fragments skipped - every value as the path below
        hip::launch(k.moe_gather_f16, D(pairs), D(256), 0, stream, h, moe_perm, moe_x16, I(E), I(K));
        mark(OpClass::misc);
        hip::launch(pl.fgu, pl.gu_pgrid, D(256), 0, stream, mo.gate.ptr, mo.up.ptr, mo.gate.row_bytes, I(F), moe_x16, moe_yg, I(E), moe_tiles,
                    moe_ntiles);
        hip::launch(pl.fdn, pl.dn_pgrid, D(256), 0, stream, mo.down.ptr, DevPtr{0}, mo.down.row_bytes, I(E), moe_yg, moe_yd, I(F), moe_tiles,
                    moe_ntiles);
        mark(OpClass::matmul);
    } else {
        hip::launch(k.moe_gather_f16, D(pairs), D(256), 0, stream, h, moe_perm, moe_x16, I(E), I(K));
        mark(OpClass::misc);
        hip::launch(pl.g_gate, pl.gu_pgrid, D(256), 0, stream, mo.gate.ptr, mo.gate.row_bytes, I(F), moe_x16, moe_yg, I(E), moe_tiles, moe_ntiles);
        hip::launch(pl.g_up, pl.gu_pgrid, D(256), 0, stream, mo.up.ptr, mo.up.row_bytes, I(F), moe_x16, moe_yu, I(E), moe_tiles, moe_ntiles);
        hip::launch(k.moe_act_f16, D((pairs * F + 255) / 256), D(256), 0, stream, moe_yg, moe_yu, moe_x16, I(pairs * F));
        hip::launch(pl.g_down, pl.dn_pgrid, D(256), 0, stream, mo.down.ptr, mo.down.row_bytes, I(E), moe_x16, moe_yd, I(F), moe_tiles, moe_ntiles);
        mark(OpClass::matmul);
    }
    if (ysh != 0) {
        matmul(sg, h, ffn_g, n, false);
        matmul(su, h, ffn_u, n, false);
        elementwise(k.silu_mul, ffn_g, ffn_u, n * cfg.n_ff);
        matmul(sd, ffn_g, ysh, n, false);
    }
    hip::launch(k.moe_combine, D((E + 255) / 256, n), D(256), 0, stream, x, ysh, moe_sg, moe_yd, moe_inv, moe_w, I(E), I(K));
    xq_src = 0;
    x16_src = 0;
    mark(OpClass::misc);
}

// Rows [0, host.size()) take host tokens, the next n_dev rows the i32 tokens at
// dev_src; positions pos0 + r. Kernel args only: never blocks.
void Model::setTokensDev(DevPtr ids_dev, std::span<const u32> host, DevPtr dev_src, u32 n_dev, u32 pos0) {
    const u32 n = static_cast<u32>(host.size()) + n_dev;
    visRowsCur(n, pos0);
    Tok16 t;
    for (std::size_t i = 0; i < host.size(); ++i) t.t[i] = lookupId(host[i]);
    row_n = n;
    for (u32 r = 0; r < n; ++r) {
        row_pos[r] = static_cast<i32>(pos0 + r);
        row_base[r] = static_cast<i32>(kv_off);
    }
    hip::launch(k.set_tokens, D(1), D(32), 0, stream, ids_dev, pos_buf, dev_src, t, I(n), I(host.size()), I(pos0));
}

DevPtr Model::ctlSlot(u32 word) const { return out_tok + 4 * static_cast<u64>(ctl_off + word); }

void Model::setTokens(DevPtr ids_dev, std::span<const u32> tokens, u32 pos0) {
    if (tokens.size() <= max_small_batch) return setTokensDev(ids_dev, tokens, 0, 0, pos0);
    const std::size_t cap = host_ids.size() / 2;
    const std::size_t n = tokens.size();
    visRowsCur(static_cast<u32>(n), pos0);
    for (std::size_t i = 0; i < n; ++i) {
        host_ids[i] = lookupId(tokens[i]);
        host_ids[cap + i] = static_cast<i32>(pos0 + i);
    }
    hip::upload(ids_dev, host_ids.data(), n * 4);
    hip::upload(pos_buf, host_ids.data() + cap, n * 4);
}

// MTP block over n rows: row r pairs hidden + r*E with token tokens[r] at
// position pos0 + r. Fills the MTP KV cache; if want_draft, returns the argmax
// of the last row.
u32 Model::mtpForward(DevPtr hidden, std::span<const u32> tokens, u32 pos0, bool want_draft) {
    mtpEnqueue(hidden, tokens, 0, pos0, want_draft ? std::optional<u32>(0) : std::nullopt);
    if (!want_draft) return 0;
    i32 r = 0;
    hip::download(&r, draftSlot(0), 4);
    return static_cast<u32>(r);
}

// mtpForward without a host round trip: the last row's token may come from
// the device (dev_tok) and the draft lands in draftSlot(slot).
void Model::mtpEnqueue(DevPtr hidden, std::span<const u32> tokens, DevPtr dev_tok, u32 pos0, std::optional<u32> slot) {
    if (!mtp) throw ModelError("NoMtp");
    const MtpW& mw = *mtp;
    const u32 E = cfg.n_embd;
    const u32 n_dev = dev_tok != 0 ? 1 : 0;
    const u32 n = static_cast<u32>(tokens.size()) + n_dev;
    xq_src = 0;
    x16_src = 0;
    // chained draft after a possible device-side cutoff: gate its big kernels
    const bool gated = n_dev > 0 && tokens.empty() && slot.has_value() && *slot > 0 && draft_p_min > 0;
    if (gated) gate = ctlSlot(ctl_stop);
    struct GateReset {
        DevPtr& g;
        ~GateReset() { g = 0; }
    } gate_reset{gate};
    if (n_dev > 0)
        setTokensDev(mtp_ids, tokens, dev_tok, n_dev, pos0);
    else
        setTokens(mtp_ids, tokens, pos0);
    hip::launch(k.get_rows[ti(tok_embd.ty)], D(8, n), D(128), 0, stream, tok_embd.ptr, tok_embd.row_bytes, mtp_ids, x, I(E));
    if (n_inj > 0) injectRows();
    rmsnorm(x, mw.enorm, mtp_cat, E, n, E, 2 * E);
    rmsnorm(hidden, mw.hnorm, mtp_cat + static_cast<u64>(E) * 4, E, n, E, 2 * E);
    matmul(mw.eh_proj, mtp_cat, x, n, false);
    if (fused(n))
        rmsnormQ8(x, mw.attn_norm, h, n);
    else
        rmsnorm(x, mw.attn_norm, h, E, n, E, E);
    draftAttn(mw.attn, n);  // drafts only: sink + last W positions (OVH-C1)
    if (mw.moe)
        moeBlock(mw.post_norm, mw.ffn_gate, mw.ffn_up, mw.ffn_down, *mw.moe, n);
    else
        ffnBlock(mw.post_norm, mw.ffn_gate, mw.ffn_up, mw.ffn_down, n);
    if (!slot) return;
    const u32 draft_slot = *slot;
    const DevPtr last = x + static_cast<u64>(n - 1) * E * 4;
    if (fused(1))
        rmsnormQ8(last, mw.head_norm, h, 1);
    else
        rmsnorm(last, mw.head_norm, h, E, 1, E, E);
    // draft head over the vocabulary subset (draft_map) or the first draft_vocab rows only
    Mat head = draft_head ? *draft_head : output;
    if (draft_rows > 0 && draft_d2) head.nrows = draft_rows;
    else if (draft_vocab > 0 && draft_vocab < head.nrows) head.nrows = draft_vocab;
    draftHeadMatmul(head, 1);
    const DevPtr map = draft_rows > 0 && draft_d2 && draft_map ? *draft_map : DevPtr{0};
    hip::launch(k.draft_pick, D(1), D(1024), 0, stream, logits, I(head.nrows), ctlSlot(ctl_drafts), ctlSlot(ctl_probs), ctlSlot(ctl_nd),
                I(draft_slot), I(draft_n_min), draft_p_min, map);
}

// DFlash2 capture: x (the input of layer li) for the last min(n, cap_rows) rows into li's slot.
void Model::captureLayerInput(std::size_t li, u32 n) {
    if (!cap_buf || li >= cap_slot.size() || cap_slot[li] < 0 || n == 0) return;
    const u64 E = cfg.n_embd;
    const u32 c = std::min(n, cap_rows);
    hip::copyAsync(cap_buf + static_cast<u64>(cap_slot[li]) * cap_rows * E * 4, x + static_cast<u64>(n - c) * E * 4,
                   static_cast<std::size_t>(c) * E * 4, stream);
    cap_n = c;
}

// Apply sequence s's kept verify rows to its recurrent state (replay mode).
void Model::commitSeq(u32 s) {
    if (!gdn_replay || s >= seqs.size()) return;
    const Seq& sq = seqs[s];
    if (sq.npend == 0) return;
    const u32 ch = cfg.convCh();
    const float scale_g = 1.0f / std::sqrt(static_cast<float>(cfg.d_state));
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const Layer& L = layers[i];
        if (L.kind != LayerKind::gdn) continue;
        const GdnW& g = L.gdn;
        GdnSegs sc;
        sc.s[0].state = sq.conv[i];
        sc.s[0].pend = pendPtr(sq, i, true);
        sc.s[0].npend = static_cast<i32>(sq.npend);
        hip::launch(k.gdn_conv_l2, D(ch / 128), D(128), 0, stream, qkv, g.conv, conv_out, I(ch), I(2 * cfg.n_k_heads), cfg.eps, sc);
        GdnSegs ss;
        ss.s[0].state = sq.ssm[i];
        ss.s[0].pend = pendPtr(sq, i, false);
        ss.s[0].npend = static_cast<i32>(sq.npend);
        hip::launch(k.gdn_step_norm, D(cfg.n_v_heads), D(512), 0, stream, conv_out, alpha, beta, z, g.norm, gdn_out, xq, xd, I(cfg.n_k_heads),
                    scale_g, cfg.eps, I(cfg.n_v_heads), I(ch), ss);
    }
    xq_src = 0;
    seqs[s].npend = 0;
}

GdnSegs Model::gdnSegsBatch(const BatchPlan& bp, std::size_t i, bool conv) const {
    GdnSegs segs;
    i32 row = 0;
    for (std::size_t kk = 0; kk < bp.segs.size(); ++kk) {
        const VSeg& sg = bp.segs[kk];
        const Seq& sq = seqs[sg.seq];
        segs.s[kk].state = conv ? sq.conv[i] : sq.ssm[i];
        segs.s[kk].row0 = row;
        segs.s[kk].nrows = static_cast<i32>(sg.nd + 1);
        if (gdn_replay) {
            segs.s[kk].pend = pendPtr(sq, i, conv);
            segs.s[kk].npend = static_cast<i32>(sq.npend);
            row += static_cast<i32>(sg.nd + 1);
            continue;
        }
        for (u32 r = 0; r < sg.nd; ++r) {
            const u32 set = bp.snap_base[kk] + r;
            segs.s[kk].snap[r] = conv ? snap_conv[set][i] : snap_ssm[set][i];
        }
        row += static_cast<i32>(sg.nd + 1);
    }
    return segs;
}

// Batched verify over sequences (see the header).
u32 Model::verifyBatchEnqueue(std::span<const VSeg> segs) {
    if (segs.empty() || segs.size() > gdn_max_seg) throw ModelError("BadBatch");
    // several sequences need the segment-aware fused DeltaNet kernels
    const bool unfused = !fusedDecode();
    if (unfused) {
        if (segs.size() > 1) throw ModelError("BatchNeedsFusedDecode");
        for (const VSeg& sg : segs)
            if (sg.nd > 1) throw ModelError("BatchNeedsFusedDecode");
        selectSeq(segs[0].seq);
        snap_rows = segs[0].nd;
    }
    struct SnapReset {
        u32& r;
        ~SnapReset() { r = 0; }
    } snap_reset{snap_rows};
    BatchPlan plan;
    plan.segs = segs;
    RowTab tab;
    RowIdx dst;
    u32 n = 0;
    u32 snaps = 0;
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const VSeg& sg = segs[kk];
        if (sg.pos + sg.nd + 1 > max_ctx) throw ModelError("ContextTooLong");
        if (sg.nd + 1 > max_small_batch || n + sg.nd + 1 > verifyRows()) throw ModelError("BatchTooLarge");
        plan.snap_base[kk] = snaps;
        snaps += sg.nd;
        for (u32 r = 0; r < sg.nd + 1; ++r) {
            // drafts: the sequence's control area (MTP), or literal tokens (n-gram)
            if (r == 0)
                tab.tok[n] = lookupId(sg.next);
            else if (!sg.drafts.empty())
                tab.tok[n] = lookupId(sg.drafts[r - 1]);
            else
                tab.tok[n] = -static_cast<i32>(sg.seq * ctl_words + ctl_drafts + r - 1) - 1;
            tab.pos[n] = static_cast<i32>(sg.pos + r);
            tab.base[n] = static_cast<i32>(seqs[sg.seq].kv_base);
            dst.v[n] = static_cast<i32>(sg.seq * ctl_words + ctl_rows + r);
            n += 1;
        }
    }
    if (!gdn_replay) {
        if (snaps > gdn_max_snap) throw ModelError("BatchTooLarge");
        ensureSnapshots(snaps);
    }
    row_n = n;
    for (u32 ri = 0; ri < n; ++ri) {
        row_pos[ri] = tab.pos[ri];
        row_base[ri] = tab.base[ri];
    }
    // vision: RoPE rows of a batched verify (seg k: nd + 1 rows from sg.pos)
    visBegin();
    if (visAny()) {
        bool any = false;
        for (const VSeg& sg : segs)
            if (visOf(sg.seq) != nullptr) any = true;
        if (any) {
            visEnsure();
            u32 r = 0;
            for (const VSeg& sg : segs)
                for (u32 j = 0; j < sg.nd + 1; ++j) visRow(visOf(sg.seq), r++, sg.pos + j);
            n_inj = 0;  // decode rows are text
            visCommit(n);
        }
    }
    hip::launch(k.set_rows, D(1), D(32), 0, stream, ids, pos_buf, kvbase_buf, out_tok, tab, I(n));
    keep_hidden = true;
    all_logits = true;
    bplan = &plan;
    small_max = n > max_small_batch ? max_verify_rows : max_small_batch;
    struct Reset {
        Model& m;
        ~Reset() {
            m.keep_hidden = false;
            m.all_logits = false;
            m.bplan = nullptr;
            m.small_max = max_small_batch;
        }
    } reset_flags{*this};
    xq_src = 0;
    x16_src = 0;
    run(n);
    // the kept rows of the previous verify are in the state now
    if (gdn_replay)
        for (const VSeg& sg : segs) seqs[sg.seq].npend = 0;
    hip::launch(k.argmax_rows_to, D(n), D(1024), 0, stream, logits, I(cfg.n_vocab), out_tok, dst);
    return n;
}

// Batched MTP draft step r over sequences (see the header).
void Model::mtpBatchStepEx(std::span<const MSeg> segs, u32 r, bool draft) {
    if (!mtp) throw ModelError("NoMtp");
    const MtpW& mw = *mtp;
    const u64 E = cfg.n_embd;
    if (segs.empty() || segs.size() > max_small_batch) throw ModelError("BadBatch");
    RowTab tab;
    RowIdx last;
    RowIdx area;
    u32 n = 0;
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const MSeg& sg = segs[kk];
        const Seq& sq = seqs[sg.seq];
        if (r == 0) {
            if (sg.pend.size() > max_small_batch || n + sg.pend.size() > verifyRows()) throw ModelError("BatchTooLarge");
            hip::copyAsync(mtp_in + static_cast<u64>(n) * E * 4, sq.hid, sg.pend.size() * E * 4, stream);
            for (std::size_t j = 0; j < sg.pend.size(); ++j) {
                tab.tok[n] = lookupId(sg.pend[j]);
                tab.pos[n] = static_cast<i32>(sg.pend_pos + j);
                tab.base[n] = static_cast<i32>(sq.kv_base);
                n += 1;
            }
        } else {
            tab.tok[n] = -static_cast<i32>(sg.seq * ctl_words + ctl_drafts + r - 1) - 1;
            tab.pos[n] = static_cast<i32>(sg.pos + r);
            tab.base[n] = static_cast<i32>(sq.kv_base);
            n += 1;
        }
        last.v[kk] = static_cast<i32>(n - 1);
        area.v[kk] = static_cast<i32>(sg.seq * ctl_words);
    }
    const DevPtr hidden = r == 0 ? mtp_in : h;
    xq_src = 0;
    x16_src = 0;
    mtp_batch = true;
    small_max = n > max_small_batch ? max_verify_rows : max_small_batch;
    struct BatchReset {
        Model& m;
        ~BatchReset() {
            m.mtp_batch = false;
            m.small_max = max_small_batch;
        }
    } batch_reset{*this};
    row_n = n;
    for (u32 ri = 0; ri < n; ++ri) {
        row_pos[ri] = tab.pos[ri];
        row_base[ri] = tab.base[ri];
    }
    // vision: RoPE rows of the batched MTP step
    visBegin();
    if (visAny()) {
        bool any = false;
        for (const MSeg& sg : segs)
            if (visOf(sg.seq) != nullptr) any = true;
        if (any) {
            visEnsure();
            u32 ri = 0;
            for (const MSeg& sg : segs) {
                const std::size_t cnt = r == 0 ? sg.pend.size() : 1;
                for (std::size_t j = 0; j < cnt; ++j) {
                    const u32 p = r == 0 ? sg.pend_pos + static_cast<u32>(j) : sg.pos + r;
                    visRow(visOf(sg.seq), ri++, p);
                }
            }
            n_inj = 0;  // decode rows are text
            visCommit(n);
        }
    }
    hip::launch(k.set_rows, D(1), D(32), 0, stream, mtp_ids, pos_buf, kvbase_buf, out_tok, tab, I(n));
    hip::launch(k.get_rows[ti(tok_embd.ty)], D(8, n), D(128), 0, stream, tok_embd.ptr, tok_embd.row_bytes, mtp_ids, x, I(E));
    rmsnorm(x, mw.enorm, mtp_cat, static_cast<u32>(E), n, static_cast<u32>(E), static_cast<u32>(2 * E));
    rmsnorm(hidden, mw.hnorm, mtp_cat + E * 4, static_cast<u32>(E), n, static_cast<u32>(E), static_cast<u32>(2 * E));
    matmul(mw.eh_proj, mtp_cat, x, n, false);
    rmsnormQ8(x, mw.attn_norm, h, n);
    draftAttn(mw.attn, n);  // drafts only: sink + last W positions (OVH-C1)
    if (mw.moe)
        moeBlock(mw.post_norm, mw.ffn_gate, mw.ffn_up, mw.ffn_down, *mw.moe, n);
    else
        ffnBlock(mw.post_norm, mw.ffn_gate, mw.ffn_up, mw.ffn_down, n);
    if (!draft) return;
    // head over each seg's last row
    const u32 ns = static_cast<u32>(segs.size());
    hip::launch(k.rmsnorm_q8_rows, D(ns), D(1024), 0, stream, x, mw.head_norm, h, xq, xd, I(E), cfg.eps, last);
    xq_src = h;
    xq_n = ns;
    x16_src = 0;
    Mat head = draft_head ? *draft_head : output;
    if (draft_rows > 0 && draft_d2) head.nrows = draft_rows;
    else if (draft_vocab > 0 && draft_vocab < head.nrows) head.nrows = draft_vocab;
    draftHeadMatmul(head, ns);
    const DevPtr map = draft_rows > 0 && draft_d2 && draft_map ? *draft_map : DevPtr{0};
    hip::launch(k.draft_pick_rows, D(ns), D(1024), 0, stream, logits, I(head.nrows), out_tok, area, I(r), I(draft_n_min), draft_p_min,
                map);
    if (draftSampleOk()) {
        // specsample: rows of sampling requests redraw draft r from the filtered draft distribution  by the race
        // coupled to the target sampler for output index pos0 + r (whirl/spec_sample.h)
        kernels::DSampArgs da;
        bool any = false;
        for (u32 kk = 0; kk < ns; ++kk) {
            const u32 sq = segs[kk].seq;
            if (sq >= draft_samp.size() || !draft_samp[sq].on) continue;
            const spec::DraftSample& ds = draft_samp[sq];
            any = true;
            da.on[kk] = 1;
            da.top_k[kk] = static_cast<i32>(std::min<std::uint32_t>(ds.top_k, 1u << 30));
            da.qoff[kk] = static_cast<i32>((static_cast<u64>(sq) * max_drafts + r) * spec::q_words);
            da.inv_t[kk] = ds.inv_t;
            da.top_p[kk] = ds.top_p;
            da.min_p[kk] = ds.min_p;
            const std::uint64_t key = spec::raceBase(ds.seed, ds.pos0 + r);
            da.key_lo[kk] = static_cast<std::uint32_t>(key);
            da.key_hi[kk] = static_cast<std::uint32_t>(key >> 32);
        }
        if (any)
            hip::launch(k.draft_sample_rows, D(ns), D(1024), 0, stream, logits, I(head.nrows), out_tok, area, I(r), map, da, draft_qbuf);
    }
}

void Model::readCtl(std::span<i32> dst) { hip::download(dst.data(), out_tok, dst.size() * 4); }

// Draft-head logits for the n rows of h into logits: the 2-bit head when
// built, else the Q4_K / full head.
void Model::draftHeadMatmul(const Mat& head, u32 n) {
    if (draft_d2) {
        if (xq_src != h || xq_n != n) {
            const u32 cnt = n * head.ncols;
            hip::launch(k.quantize_q8, D((cnt + 255) / 256), D(256), 0, stream, h, xq, xd, I(cnt));
            xq_src = h;
            xq_n = n;
        }
        hip::launch(k.gemv_d2[n - 1], D((head.nrows + 7) / 8), D(256), 0, stream, *draft_d2, xq, xd, logits, I(head.ncols), I(head.nrows), gate);
        mark(OpClass::head);
        return;
    }
    matmul(head, h, logits, n, false);
}

void Model::argmaxRows(u32 n, std::span<u32> out) {
    argmaxRowsEnqueue(n);
    i32 buf[max_small_batch];
    hip::download(buf, out_tok, n * 4);
    for (u32 r = 0; r < n; ++r) out[r] = static_cast<u32>(buf[r]);
}

void Model::argmaxRowsEnqueue(u32 n) {
    hip::launch(k.argmax_rows, D(n), D(1024), 0, stream, logits, I(cfg.n_vocab), ctlSlot(ctl_rows));
}

void Model::verifyEnqueueEx(u32 next, u32 n_draft, std::optional<std::span<const u32>> host_drafts, u32 pos0) {
    const u32 n = n_draft + 1;
    if (pos0 + n > max_ctx) throw ModelError("ContextTooLong");
    ensureSnapshots(n_draft);
    keep_hidden = true;
    all_logits = true;
    snap_rows = n_draft;
    struct Reset {
        Model& m;
        ~Reset() {
            m.keep_hidden = false;
            m.all_logits = false;
            m.snap_rows = 0;
        }
    } reset_flags{*this};
    xq_src = 0;
    x16_src = 0;
    if (host_drafts) {
        u32 tv[max_small_batch];
        tv[0] = next;
        for (u32 i = 0; i < n_draft; ++i) tv[1 + i] = (*host_drafts)[i];
        setTokensDev(ids, std::span<const u32>(tv, n), 0, 0, pos0);
    } else {
        const u32 one[1] = {next};
        setTokensDev(ids, one, draftSlot(0), n_draft, pos0);
    }
    run(n);
    argmaxRowsEnqueue(n);
}

Model::Cycle Model::readCycle(std::span<u32> out, std::span<u32> drafts) {
    i32 buf[ctl_words];
    hip::download(buf, out_tok, sizeof buf);
    for (std::size_t r = 0; r < out.size(); ++r) out[r] = static_cast<u32>(buf[ctl_rows + r]);
    for (std::size_t r = 0; r < drafts.size(); ++r) drafts[r] = static_cast<u32>(buf[ctl_drafts + r]);
    Cycle c;
    c.nd = static_cast<u32>(std::max<i32>(0, buf[ctl_nd]));
    for (u32 r = 0; r < max_small_batch; ++r) std::memcpy(&c.probs[r], &buf[ctl_probs + r], 4);
    c.nd = std::min<u32>(c.nd, static_cast<u32>(drafts.size()));
    return c;
}

// Main-model forward over tokens at pos0 keeping every row's logits and normed
// hidden; the recurrent state after rows 0 .. len-2 is snapshotted.
void Model::verify(std::span<const u32> tokens, u32 pos0, std::span<u32> out) {
    const u32 snaps = static_cast<u32>(tokens.size() - 1);
    ensureSnapshots(snaps);
    keep_hidden = true;
    all_logits = true;
    snap_rows = snaps;
    struct Reset {
        Model& m;
        ~Reset() {
            m.keep_hidden = false;
            m.all_logits = false;
            m.snap_rows = 0;
        }
    } reset_flags{*this};
    forward(tokens, pos0);
    argmaxRows(static_cast<u32>(tokens.size()), out);
}

// Roll the recurrent state back to the snapshot taken after row keep - 1.
void Model::restoreSnapshot(u32 keep) {
    // a captured decode graph holds the old state pointers
    decode_graph.reset();
    // the unfused decode path writes only snapshot 0: any later set would be stale
    if (keep == 0 || keep > snap_sets || (keep > 1 && !fusedDecode())) throw ModelError("SnapshotNotWritten");
    const u32 kk = keep - 1;
    for (u32 i = 0; i < cfg.n_layer; ++i) {
        if (ssm_state[i] == 0) continue;
        std::swap(ssm_state[i], snap_ssm[kk][i]);
        std::swap(conv_state[i], snap_conv[kk][i]);
    }
}

GdnSegs Model::gdnSegsSingle(std::size_t i, bool conv, u32 n) const {
    GdnSegs segs;
    segs.s[0].state = conv ? conv_state[i] : ssm_state[i];
    segs.s[0].row0 = 0;
    segs.s[0].nrows = static_cast<i32>(n);
    for (u32 kk = 0; kk < std::min(snap_rows, gdn_max_snap); ++kk) segs.s[0].snap[kk] = conv ? snap_conv[kk][i] : snap_ssm[kk][i];
    return segs;
}

// Chunked (C = 64) DeltaNet prefill recurrence over n >= 64 rows: the f32 prep
// + scan pair, or (gdn_wmma) the f16-WMMA pair.
void Model::gdnChunked(std::size_t i, DevPtr conv_o, DevPtr alpha_b, DevPtr beta_b, DevPtr out, u32 n, float scale) {
    const u32 ch = cfg.convCh();
    const u32 n_chunks = (n + 63) / 64;
    if (gdn_strix && n >= gdn_strix_min) {
        // GDN-8S: strix-llama.cpp chunks (CL = 32) on the existing scratch: T' | P in gc_m, Vh in gc_w,
        // Qh | Kh | gate factors in gc_u (each fits when n <= the batch the gc_* buffers were sized for)
        const u64 H = cfg.n_v_heads, Hk = cfg.n_k_heads, nch = (n + 31) / 32;
        const u64 tp = H * nch * 32 * 32 * 2, qk = Hk * n * 128 * 2, vh = H * n * 128 * 2, gb = H * nch * 32 * 8;
        if (2 * tp <= gc_m_bytes && vh <= gc_w_bytes && 2 * qk + gb <= gc_u_bytes) {
            hip::launch(k.gdn_sx_prep, D(Hk, nch), D(256), 0, stream, conv_o, alpha_b, beta_b, gc_m, gc_m + tp, gc_u + 2 * qk, gc_u, gc_u + qk,
                        gc_w, I(H), I(Hk), I(n), I(nch), I(ch), scale);
            // state in place (curr_state == state); no fused output norm (on_* = null)
            hip::launch(k.gdn_sx_scan, D(H), D(256), 0, stream, gc_u, gc_u + qk, gc_w, ssm_state[i], out, ssm_state[i], gc_m, gc_m + tp,
                        gc_u + 2 * qk, I(H), I(Hk), I(n), I(nch), scale, u64(0), u64(0), u64(0), u64(0), 0.0f, false);
            return;
        }
    }
    if (gdn_wmma && k.gdn_wprep != nullptr && k.gdn_wscan8 != nullptr) {
        hip::launch(k.gdn_wprep, D(cfg.n_v_heads, n_chunks), D(256), 0, stream, conv_o, alpha_b, beta_b, gc_w, gc_g, I(n), I(cfg.n_k_heads),
                    I(cfg.n_v_heads), I(ch));
        hip::launch(k.gdn_wscan8, D(cfg.n_v_heads, cfg.headV() / 128), D(256), 0, stream, conv_o, gc_w, gc_g, beta_b, ssm_state[i], out, I(n),
                    I(cfg.n_k_heads), I(cfg.n_v_heads), I(ch), scale);
        return;
    }
    hip::launch(k.gdn_chunk_prep, D(n_chunks, cfg.n_v_heads), D(256), 0, stream, conv_o, alpha_b, beta_b, gc_w, gc_u, gc_m, gc_g, I(n),
                I(cfg.n_k_heads), I(cfg.n_v_heads), I(ch));
    hip::launch(k.gdn_chunk_scan, D(cfg.n_v_heads, cfg.headV() / 32), D(256), 0, stream, conv_o, gc_w, gc_u, gc_m, gc_g, ssm_state[i], out, I(n),
                I(cfg.n_k_heads), I(cfg.n_v_heads), I(ch), scale);
}

void Model::gdnScan(std::size_t i, DevPtr conv_o, DevPtr alpha_b, DevPtr beta_b, DevPtr out, u32 n, float scale) {
    const u32 ch = cfg.convCh();
    if (gdn_chunked && n >= 64 && cfg.d_state == 128 && cfg.headV() == 128) {
        gdnChunked(i, conv_o, alpha_b, beta_b, out, n, scale);
    } else {
        hip::launch(k.gdn_seq_128, D(cfg.n_v_heads, cfg.headV() / 32), D(128), 0, stream, conv_o, alpha_b, beta_b, ssm_state[i], out, I(n),
                    I(cfg.n_k_heads), I(cfg.n_v_heads), I(cfg.headV()), I(ch), scale, snap_rows > 0 ? snap_ssm[0][i] : u64(0), i32(0));
    }
}

void Model::run(u32 n) {
    if (gdn_replay && bplan == nullptr) {
        if (!psegs.empty()) {
            for (const PRows& ps : psegs) commitSeq(ps.seq);
        } else {
            commitSeq(cur_seq);
        }
    }
    xq_src = 0;
    x16_src = 0;
    hip::launch(k.get_rows[ti(tok_embd.ty)], D(8, n), D(128), 0, stream, tok_embd.ptr, tok_embd.row_bytes, ids, x, I(cfg.n_embd));
    if (n_inj > 0) injectRows();
    mark(OpClass::embed);

    const u32 E = cfg.n_embd;
    // DFlash2 drafter features: x at the input of its target layers (a copy on the stream; the
    // trunk's arithmetic is untouched)
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const Layer& L = layers[i];
        bool gdn_in2 = false;  // x16b holds this layer's normed input as f16 for [beta; alpha]
        if (cap_buf) captureLayerInput(i, n);
        if (fused(n)) {
            rmsnormQ8(x, L.attn_norm, h, n);
        } else if (L.kind == LayerKind::attn) {
            const Mat c[3] = {L.attn.q, L.attn.k, L.attn.v};
            rmsnormIn(x, L.attn_norm, h, n, c, 1);
        } else if (gdnIn2(L.gdn, n)) {
            // one norm: fp8 (+ sx8) into x16 for qkv / gate (registered as h's
            // conversion; h itself is left unwritten) and f16 into x16b for [beta; alpha]
            if (x16bOff() + static_cast<u64>(n) * E * 2 > x16_bytes) throw ModelError("X16bOverflow");
            hip::launch(x8(k.rmsnorm_x8h16, k.rmsnorm_x8h16t), D(n), D(256), 0, stream, x, L.attn_norm, x16, sx8, x16 + x16bOff(), I(E),
                        cfg.eps);
            xq_src = 0;
            x16_src = h;
            x16_count = n * E;
            x16_fp8 = true;
            mark(OpClass::norm);
            gdn_in2 = true;
        } else {
            const Mat c[4] = {L.gdn.qkv, L.gdn.gate, L.gdn.beta, L.gdn.alpha};
            rmsnormIn(x, L.attn_norm, h, n, c, 1);
        }
        if (L.kind == LayerKind::attn) {
            attnBlock(L.attn, kvLayer(i), n);
        } else {
            const GdnW& g = L.gdn;
            const u32 ch = cfg.convCh();
            // MXFP4 speed mode: the qkv / z GEMMs write f16 for the conv and the gated norm
            bool gdn_h16 = ffn_h16 && act_fuse && !fused(n) && cfg.d_conv == 4 && n > 1 && cfg.d_state == 128 && ch % 128 == 0 &&
                           cfg.headV() == 128 && cfg.n_v_heads <= 64 && k.gdn_conv_l2n_h != nullptr && k.gdn_conv_state_h != nullptr &&
                           h16Out(g.qkv, n, 1) && h16Out(g.gate, n, 1);
            if (gdn_h16) {
                const Mat pc[2] = {g.qkv, g.gate};
                const ActIn pin = actCommon(pc, n, 1).value_or(ActIn::gemv);
                const ActIn oin = actCommon(std::span<const Mat>(&g.out, 1), n, 1).value_or(ActIn::gemv);
                gdn_h16 = (pin == ActIn::fp8 && oin == ActIn::fp8 && k.gated_norm_x8h != nullptr) ||
                          (pin == ActIn::f16 && oin == ActIn::f16 && k.gated_norm_x16h != nullptr);
            }
            out_h16 = gdn_h16;
            {
                const Mat ws[2] = {g.qkv, g.gate};
                const DevPtr ys[2] = {qkv, z};
                matmulGroup(ws, h, ys, n);
            }
            out_h16 = false;
            const bool gdn_fusable = fused(n) && gdnAbFusable(g) && cfg.d_conv == 4 && cfg.d_state == 128 && cfg.headV() == 128 && ch % 128 == 0;
            if (gdn_fusable) {
                const u32 nh = cfg.n_v_heads;
                const u32 n_seg = bplan ? static_cast<u32>(bplan->segs.size()) : 1;
                const GdnSegs segs_c = bplan ? gdnSegsBatch(*bplan, i, true) : gdnSegsSingle(i, true, n);
                const hip::Function abconv = (dbg_flags & 1) != 0 ? nullptr : k.gdn_abconv[ti(g.beta.ty)];
                if (!ops::preciseAllows(ops::BitClass::q8_row, prec_dec, n, small_max)) {  // gdn_fusable: n <= small_max
                    // precise decode: [beta; alpha] on the f16 GEMM from h (f32, written by
                    // rmsnormQ8), the gates, then the same conv step as below
                    const u32 nhr = n * nh;
                    if (gdn_ba_on && k.gdn_gates_ba != nullptr && ba_buf != 0 && rowsContiguous(g.beta, g.alpha)) {
                        matmul(concatRows(g.beta, g.alpha), h, ba_buf, n, false);
                        hip::launch(k.gdn_gates_ba, D((nhr + 63) / 64), D(64), 0, stream, ba_buf, beta, alpha, g.dt, g.a, I(nhr), I(nh));
                    } else {
                        matmul(g.beta, h, beta, n, false);
                        matmul(g.alpha, h, alpha, n, false);
                        hip::launch(k.gdn_gates, D((nhr + 63) / 64), D(64), 0, stream, beta, alpha, g.dt, g.a, I(nhr), I(nh));
                    }
                    hip::launch(k.gdn_conv_l2, D(ch / 128, n_seg), D(128), 0, stream, qkv, g.conv, conv_out, I(ch), I(2 * cfg.n_k_heads), cfg.eps,
                                segs_c);
                } else if (abconv != nullptr) {
                    // beta/alpha projections and the conv step share one launch
                    const u32 ab_blocks = (2 * nh + 7) / 8;
                    hip::launch(abconv, D(ab_blocks * n + (ch / 128) * n_seg), D(256), 0, stream, g.beta.ptr, g.alpha.ptr, g.beta.row_bytes, xq, xd,
                                beta, alpha, g.dt, g.a, I(g.beta.ncols), I(nh), I(n), qkv, g.conv, conv_out, I(ch), I(2 * cfg.n_k_heads), cfg.eps,
                                segs_c);
                } else {
                    hip::launch(k.gdn_ab[ti(g.beta.ty)], D((2 * nh + 7) / 8, n), D(256), 0, stream, g.beta.ptr, g.alpha.ptr, g.beta.row_bytes, xq,
                                xd, beta, alpha, g.dt, g.a, I(g.beta.ncols), I(nh));
                    hip::launch(k.gdn_conv_l2, D(ch / 128, n_seg), D(128), 0, stream, qkv, g.conv, conv_out, I(ch), I(2 * cfg.n_k_heads), cfg.eps,
                                segs_c);
                }
                mark(OpClass::gdn_prep);
                const float scale_g = 1.0f / std::sqrt(static_cast<float>(cfg.d_state));
                const GdnSegs segs_s = bplan ? gdnSegsBatch(*bplan, i, false) : gdnSegsSingle(i, false, n);
                hip::launch(k.gdn_step_norm, D(nh, n_seg), D(512), 0, stream, conv_out, alpha, beta, z, g.norm, gdn_out, xq, xd, I(cfg.n_k_heads),
                            scale_g, cfg.eps, I(cfg.n_v_heads), I(ch), segs_s);
                xq_src = gdn_out;
                xq_n = n;
                x16_src = 0;
                mark(OpClass::gdn_rec);
            } else {
                const u32 nh = n * cfg.n_v_heads;
                bool conv_l2 = false;
                if (gdn_in2) {
                    // the [beta; alpha] f16 GEMM straight from x16b (gdnIn2 implies the gates_ba path)
                    const Mat ba = concatRows(g.beta, g.alpha);
                    gemmF16(ba, x16 + x16bOff(), ba_buf, n, 0);
                    if (ba_buf == xq_src) xq_src = 0;
                    if (ba_buf == x16_src) x16_src = 0;
                    if (prof) prof->matmul_bytes += ba.row_bytes * ba.nrows;
                    mark(OpClass::matmul);
                    hip::launch(k.gdn_gates_ba, D((nh + 63) / 64), D(64), 0, stream, ba_buf, beta, alpha, g.dt, g.a, I(nh), I(cfg.n_v_heads));
                } else if (gdn_ba_on && k.gdn_gates_ba != nullptr && ba_buf != 0 && rowsContiguous(g.beta, g.alpha)) {
                    // one [beta; alpha] GEMM (beta's tune) into ba[t][2 * n_v_heads]
                    matmul(concatRows(g.beta, g.alpha), h, ba_buf, n, false);
                    hip::launch(k.gdn_gates_ba, D((nh + 63) / 64), D(64), 0, stream, ba_buf, beta, alpha, g.dt, g.a, I(nh), I(cfg.n_v_heads));
                } else {
                    matmul(g.beta, h, beta, n, false);
                    matmul(g.alpha, h, alpha, n, false);
                    hip::launch(k.gdn_gates, D((nh + 63) / 64), D(64), 0, stream, beta, alpha, g.dt, g.a, I(nh), I(cfg.n_v_heads));
                }
                if (!psegs.empty() && gdn_h16) {
                    // the solo f16 conv kernels per segment (segmented == solo bitwise)
                    conv_l2 = true;
                    for (const PRows& ps : psegs) {
                        selectSeq(ps.seq);
                        const u64 qh = static_cast<u64>(ps.r0) * ch * 2;
                        const u64 qo = static_cast<u64>(ps.r0) * ch * 4;
                        hip::launch(k.gdn_conv_l2n_h, D(ch / 128, (ps.n + 7) / 8), D(128), 0, stream, qkv + qh, conv_state[i], g.conv, conv_out + qo,
                                    I(ch), I(ps.n), I(2 * cfg.n_k_heads), cfg.eps);
                        hip::launch(k.gdn_conv_state_h, D((ch + 255) / 256), D(256), 0, stream, qkv + qh, conv_state[i], I(ch), I(ps.n),
                                    snap_rows > 0 ? snap_conv[0][i] : u64(0), i32(0));
                    }
                } else if (!psegs.empty()) {
                    for (const PRows& ps : psegs) {
                        selectSeq(ps.seq);
                        const u64 qo = static_cast<u64>(ps.r0) * ch * 4;
                        hip::launch(k.gdn_conv_par, D((ch + 255) / 256, ps.n), D(256), 0, stream, qkv + qo, conv_state[i], g.conv, conv_out + qo, I(ch),
                                    I(ps.n));
                        hip::launch(k.gdn_conv_state, D((ch + 255) / 256), D(256), 0, stream, qkv + qo, conv_state[i], I(ch), I(ps.n),
                                    snap_rows > 0 ? snap_conv[0][i] : u64(0), i32(0));
                    }
                } else if (cfg.d_conv == 4 && n > 1) {
                    // conv + q/k l2norm in one pass (bitwise the same) when the shapes allow
                    conv_l2 = act_fuse && k.gdn_conv_l2n != nullptr && cfg.d_state == 128 && ch % 128 == 0;
                    if (gdn_h16) {
                        hip::launch(k.gdn_conv_l2n_h, D(ch / 128, (n + 7) / 8), D(128), 0, stream, qkv, conv_state[i], g.conv, conv_out, I(ch), I(n),
                                    I(2 * cfg.n_k_heads), cfg.eps);
                    } else if (conv_l2) {
                        hip::launch(k.gdn_conv_l2n, D(ch / 128, (n + 7) / 8), D(128), 0, stream, qkv, conv_state[i], g.conv, conv_out, I(ch), I(n),
                                    I(2 * cfg.n_k_heads), cfg.eps);
                    } else {
                        hip::launch(k.gdn_conv_par, D((ch + 255) / 256, n), D(256), 0, stream, qkv, conv_state[i], g.conv, conv_out, I(ch), I(n));
                    }
                    hip::launch(gdn_h16 ? k.gdn_conv_state_h : k.gdn_conv_state, D((ch + 255) / 256), D(256), 0, stream, qkv, conv_state[i], I(ch),
                                I(n), snap_rows > 0 ? snap_conv[0][i] : u64(0), i32(0));
                } else {
                    hip::launch(k.gdn_conv_seq, D((ch + 255) / 256), D(256), 0, stream, qkv, conv_state[i], g.conv, conv_out, I(ch), I(n));
                }
                // q and k heads are contiguous at the front of each token's conv row
                if (!conv_l2)
                    hip::launch(k.l2norm, D(2 * cfg.n_k_heads, n), D(128), 0, stream, conv_out, I(cfg.d_state), I(cfg.d_state), I(ch), cfg.eps);
                mark(OpClass::gdn_prep);
                const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.d_state));
                if (!psegs.empty()) {
                    for (const PRows& ps : psegs) {
                        selectSeq(ps.seq);
                        gdnScan(i, conv_out + static_cast<u64>(ps.r0) * ch * 4, alpha + static_cast<u64>(ps.r0) * cfg.n_v_heads * 4,
                                beta + static_cast<u64>(ps.r0) * cfg.n_v_heads * 4,
                                gdn_out + static_cast<u64>(ps.r0) * cfg.n_v_heads * cfg.headV() * 4, ps.n, scale);
                    }
                } else if (gdn_chunked && n >= 64 && cfg.d_state == 128 && cfg.headV() == 128) {
                    gdnChunked(i, conv_out, alpha, beta, gdn_out, n, scale);
                } else {
                    hip::launch(k.gdn_seq_128, D(cfg.n_v_heads, cfg.headV() / 32), D(128), 0, stream, conv_out, alpha, beta, ssm_state[i], gdn_out,
                                I(n), I(cfg.n_k_heads), I(cfg.n_v_heads), I(cfg.headV()), I(ch), scale,
                                snap_rows > 0 ? snap_ssm[0][i] : u64(0), i32(0));
                }
                mark(OpClass::gdn_rec);
                // gated norm straight into the out-proj GEMM input when possible
                std::optional<ActIn> out_in;
                if (cfg.headV() == 128 && cfg.n_v_heads <= 64) out_in = actCommon(std::span<const Mat>(&g.out, 1), n, 1);
                hip::Function gn_x = nullptr;
                if (gdn_h16)
                    gn_x = *out_in == ActIn::fp8 ? x8(k.gated_norm_x8h, k.gated_norm_x8ht) : k.gated_norm_x16h;
                else if (out_in)
                    gn_x = *out_in == ActIn::fp8 ? x8(k.gated_norm_x8, k.gated_norm_x8t) : k.gated_norm_x16;
                if (gn_x != nullptr) {
                    hip::launch(gn_x, D(n), D(256), 0, stream, gdn_out, z, g.norm, x16, sx8, I(cfg.n_v_heads), cfg.eps);
                    xq_src = 0;
                    x16_src = gdn_out;
                    x16_count = n * cfg.n_v_heads * 128;
                    x16_fp8 = *out_in == ActIn::fp8;
                } else {
                    hip::launch(k.gdn_gated_norm, D(nh), D(128), 0, stream, gdn_out, z, g.norm, I(cfg.headV()), cfg.eps);
                    xq_src = 0;
                    x16_src = 0;
                }
                mark(OpClass::gdn_norm);
            }
            matmul(g.out, gdn_out, x, n, true);
        }
        if (L.moe)
            moeBlock(L.post_norm, L.ffn_gate, L.ffn_up, L.ffn_down, *L.moe, n);
        else
            ffnBlock(L.post_norm, L.ffn_gate, L.ffn_up, L.ffn_down, n);
    }
    if (cap_buf) captureLayerInput(layers.size(), n);  // id n_layer: the last layer's output
    if (keep_hidden) rmsnorm(x, output_norm, hn, E, n, E, E);
    if (!psegs.empty()) return;  // prefillSegs: per-segment logits from x
    if (keep_hidden && all_logits && n <= small_max) {
        head_phase = true;
        matmul(output, hn, logits, n, false);
        head_phase = false;
        return;
    }
    const DevPtr last = x + static_cast<u64>(n - 1) * E * 4;
    // FIX-NUM P2-1: every head input uses the 256-thread rmsnorm (the reduction order of the
    // verify / all-logits rows above, which the server's decode and the CLI's MTP / n-gram verify
    // take), never rmsnormQ8's 1024-thread block sum: plain decode, prompt-end logits and verify
    // rows then give the same logits bits for the same hidden row.
    rmsnorm(last, output_norm, h, E, 1, E, E);
    head_phase = true;
    matmul(output, h, logits, 1, false);
    head_phase = false;
}

// ---------------------------------------------------------------------------
// prefill drivers

void Model::prefillWithMtp(std::span<const u32> tokens) {
    std::size_t off = 0;
    while (off < tokens.size()) {
        const std::size_t n = std::min<std::size_t>(tokens.size() - off, max_batch);
        prefillMtpChunk(tokens, 0, off, n, std::nullopt);
        off += n;
    }
}

// One chunk of an MTP prefill: tokens[off .. off + n] at positions pos0 + off
// .., then the MTP rows pairing each of their trunk hiddens with the following
// token. The row pairing a chunk's LAST hidden with the next token is not run
// in that chunk: the chunk's last hidden goes to mtp_h, and the next chunk runs
// that boundary row on its own (one row) before its forward, exactly as a
// prefix-cache resume at that position does (prev_hidden = the checkpoint's
// hidden). So the MTP KV (and every later draft) is bitwise the same whether
// the prompt was prefilled cold or resumed from a checkpoint at a chunk end:
// small-row MTP forwards (q8 activations, fused kernels) are not bitwise equal
// to the same row inside a larger batch. Exact acceptance never showed the
// difference; relaxacc (fast) keeps drafts, so the text depended on it (FIX-FS).
// The final chunk's last hidden stays in mtp_h for the first decode draft.
void Model::prefillMtpChunk(std::span<const u32> tokens, u32 pos0, std::size_t off, std::size_t n, std::optional<DevPtr> prev_hidden) {
    const u64 E = cfg.n_embd;
    if (off == 0 && prev_hidden) hip::copyAsync(mtp_h, *prev_hidden, E * 4, stream);
    // mtp_h: the previous chunk's last hidden (off > 0) or the checkpoint's (resume)
    if (off > 0 || prev_hidden) (void)mtpForward(mtp_h, tokens.subspan(off, 1), pos0 + static_cast<u32>(off), false);
    const u32 p = pos0 + static_cast<u32>(off);
    keep_hidden = true;
    try {
        forward(tokens.subspan(off, n), p);
    } catch (...) {
        keep_hidden = false;
        throw;
    }
    keep_hidden = false;
    if (n > 1) (void)mtpForward(hn, tokens.subspan(off + 1, n - 1), p + 1, false);
    hip::copyAsync(mtp_h, hn + (n - 1) * E * 4, E * 4, stream);
}

bool Model::canSegment() const {
    return seqs.size() > 1 && k.attn_prep != nullptr && !no_fuse && !naive_attn && cfg.head_dim == 256 && cfg.d_conv == 4;
}

// Several sequences' prefill chunks as one forward (each chunk longer than
// max_small_batch). Logits of every segment that ends its prompt land in
// logits row k; with mtp, also the MTP rows and mtp_h of each sequence.
void Model::prefillSegs(std::span<const PSeg> segs, bool with_mtp) {
    if (segs.empty() || segs.size() > max_small_batch) throw ModelError("BadBatch");
    const u64 E = cfg.n_embd;
    u32 total = 0;
    PRows rows[max_small_batch];
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const PSeg& sg = segs[kk];
        if (sg.n <= max_small_batch) throw ModelError("BadBatch");
        if (sg.pos0 + sg.off + sg.n > max_ctx) throw ModelError("ContextTooLong");
        rows[kk] = {sg.seq, total, static_cast<u32>(sg.n), sg.pos0 + static_cast<u32>(sg.off)};
        total += static_cast<u32>(sg.n);
    }
    if (total > max_batch) throw ModelError("BatchTooLarge");
    // boundary MTP rows, one row each as in prefillMtpChunk (solo order): a cached
    // prefix's last hidden (off 0) or the previous chunk's (in mtp_h) with tokens[off]
    if (with_mtp)
        for (const PSeg& sg : segs)
            if (sg.off > 0 || sg.prev_hidden) {
                selectSeq(sg.seq);
                if (sg.off == 0) hip::copyAsync(mtp_h, *sg.prev_hidden, E * 4, stream);
                (void)mtpForward(mtp_h, sg.tokens.subspan(sg.off, 1), sg.pos0 + static_cast<u32>(sg.off), false);
            }
    const std::size_t cap = host_ids.size() / 2;
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const PSeg& sg = segs[kk];
        for (std::size_t j = 0; j < sg.n; ++j) {
            host_ids[rows[kk].r0 + j] = lookupId(sg.tokens[sg.off + j]);
            host_ids[cap + rows[kk].r0 + j] = static_cast<i32>(sg.pos0 + sg.off + j);
        }
    }
    // vision: RoPE rows / image rows of the segments of sequences with images
    visBegin();
    if (visAny()) {
        bool any = false;
        for (const PSeg& sg : segs)
            if (visOf(sg.seq) != nullptr) any = true;
        if (any) {
            visEnsure();
            for (std::size_t kk = 0; kk < segs.size(); ++kk) {
                const PSeg& sg = segs[kk];
                for (std::size_t j = 0; j < sg.n; ++j)
                    visRow(visOf(sg.seq), rows[kk].r0 + static_cast<u32>(j), sg.pos0 + static_cast<u32>(sg.off + j));
            }
            visCommit(total);
        }
    }
    hip::upload(ids, host_ids.data(), static_cast<std::size_t>(total) * 4);
    hip::upload(pos_buf, host_ids.data() + cap, static_cast<std::size_t>(total) * 4);
    row_n = 0;
    xq_src = 0;
    x16_src = 0;
    psegs = std::span<const PRows>(rows, segs.size());
    keep_hidden = with_mtp;
    struct Reset {
        Model& m;
        ~Reset() {
            m.psegs = {};
            m.keep_hidden = false;
        }
    } reset_flags{*this};
    run(total);
    psegs = {};
    // logits of the segments that end their prompt (before the MTP block reuses x)
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const PSeg& sg = segs[kk];
        if (sg.off + sg.n != sg.tokens.size()) continue;
        const DevPtr last = x + (static_cast<u64>(rows[kk].r0) + rows[kk].n - 1) * E * 4;
        rmsnorm(last, output_norm, h, static_cast<u32>(E), 1, static_cast<u32>(E), static_cast<u32>(E));  // P2-1: as run()
        head_phase = true;
        matmul(output, h, logits + static_cast<u64>(kk) * cfg.n_vocab * 4, 1, false);
        head_phase = false;
    }
    if (!with_mtp) return;
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const PSeg& sg = segs[kk];
        selectSeq(sg.seq);
        const DevPtr hid = hn + static_cast<u64>(rows[kk].r0) * E * 4;
        const u32 p = sg.pos0 + static_cast<u32>(sg.off);
        // the row pairing the last hidden with the next token runs with the next chunk
        if (sg.n > 1) (void)mtpForward(hid, sg.tokens.subspan(sg.off + 1, sg.n - 1), p + 1, false);
        hip::copyAsync(mtp_h, hid + (sg.n - 1) * E * 4, E * 4, stream);
    }
}

void Model::segLogitsToFront(u32 kk) {
    if (kk == 0) return;
    const u64 V = cfg.n_vocab;
    hip::copyAsync(logits, logits + static_cast<u64>(kk) * V * 4, V * 4, stream);
}

void Model::prefill(std::span<const u32> tokens, u32 pos0) {
    std::size_t off = 0;
    while (off < tokens.size()) {
        const std::size_t n = std::min<std::size_t>(tokens.size() - off, max_batch);
        forward(tokens.subspan(off, n), pos0 + static_cast<u32>(off));
        off += n;
    }
}

u32 Model::argmax() {
    hip::launch(k.argmax, D(1), D(1024), 0, stream, logits, I(cfg.n_vocab), out_tok, ids, pos_buf, i32(0));
    i32 out = 0;
    hip::download(&out, out_tok, 4);
    return static_cast<u32>(out);
}

void Model::readLogits(std::span<float> dst) { hip::download(dst.data(), logits, dst.size() * 4); }

// ---------------------------------------------------------------------------
// vision: lent prefill scratch, multi-section RoPE rows and image embedding rows
// (ported from the research prototype, item VIS)

std::size_t Model::lendScratch(std::span<std::array<u64, 2>> out) const {
    const u64 B = max_batch;
    const u64 f4 = 4;
    const u64 in_max = std::max<u64>(std::max<u64>(cfg.n_ff, 2ull * cfg.n_embd), std::max<u64>(cfg.d_inner, static_cast<u64>(cfg.n_head) * cfg.head_dim));
    u64 max_elems = 0;
    for (const Layer& L : layers) {
        const MatList all = layerMats(L);
        for (const Mat& w : all.slice()) max_elems = std::max<u64>(max_elems, static_cast<u64>(w.ncols) * w.nrows);
    }
    const u64 n_chunks = (B + 63) / 64;
    const std::array<u64, 2> list[] = {
        {ffn_g, B * ff_scratch * f4},
        {ffn_u, B * ff_scratch * f4},
        {qf, B * cfg.n_head * cfg.head_dim * 2 * f4},
        {w16, max_elems * 2},
        {qkv, B * cfg.convCh() * f4},
        {conv_out, B * cfg.convCh() * f4},
        {x16, B * std::max<u64>(in_max, cfg.n_embd) * 2},
        {z, B * cfg.d_inner * f4},
        {gdn_out, B * cfg.d_inner * f4},
        {attn_out, B * cfg.n_head * cfg.head_dim * f4},
        {gc_w, n_chunks * cfg.n_v_heads * 64 * 128 * 4},
        {gc_u, n_chunks * cfg.n_v_heads * 64 * 128 * 4},
    };
    std::size_t kk = 0;
    for (const auto& e : list) {
        if (kk == out.size()) break;
        if (e[0] == 0) continue;
        out[kk++] = e;
    }
    return kk;
}

bool Model::visAny() const {
    if (vis_single != nullptr) return true;
    for (const VisMap* v : vis_seq)
        if (v != nullptr) return true;
    return false;
}

const VisMap* Model::visOf(u32 seq) const {
    const VisMap* vm = seqs.empty() ? vis_single : vis_seq[seq];
    return vm != nullptr && vm->active() ? vm : nullptr;
}

void Model::visBegin() {
    mrope_on = false;
    rpos_rows = false;
    rdelta = 0;
    n_inj = 0;
}

void Model::visEnsure() {
    if (rpos_buf != 0) return;
    if (rpos_host.empty()) rpos_host.assign(static_cast<std::size_t>(max_batch) * 3, 0);
    // (not in `allocations`: visRelease frees it again once no sequence has images)
    rpos_own = device::DevBuf::alloc(device::hipBackend(), static_cast<std::uint64_t>(max_batch) * 3 * 4);
    rpos_buf = rpos_own.get();
}

void Model::visRelease() {
    if (rpos_buf == 0 || visAny()) return;
    try {
        hip::sync();
    } catch (...) {
    }
    rpos_own.reset();
    rpos_buf = 0;
}

// row r of the next forward is cache position pos of a sequence with image map vm
void Model::visRow(const VisMap* vm, u32 r, u32 pos) {
    const std::array<i32, 3> rp = vm != nullptr ? vm->rope(pos) : std::array<i32, 3>{static_cast<i32>(pos), static_cast<i32>(pos), static_cast<i32>(pos)};
    std::copy(rp.begin(), rp.end(), rpos_host.begin() + 3 * static_cast<std::size_t>(r));
    if (vm == nullptr) return;
    const VisSpan* sp = vm->spanAt(pos);
    if (sp == nullptr) return;
    const std::size_t E = cfg.n_embd;
    if (sp->emb == nullptr) throw ModelError("ImageNotEncoded");
    const float* src = sp->emb + static_cast<std::size_t>(pos - sp->start) * E;
    if (n_inj > 0) {
        Inj& last = inj[n_inj - 1];
        if (last.row + last.n == r && last.src + last.n * E == src) {
            last.n += 1;
            return;
        }
    }
    if (n_inj == max_inj) throw ModelError("TooManyImageRuns");
    inj[n_inj++] = Inj{r, 1, src};
}

// upload the rows' RoPE positions; mrope_on for the next forward
void Model::visCommit(u32 n) {
    if (k.attn_prep_m == nullptr) throw ModelError("MropeUnsupported");
    if (n <= max_small_batch && k.set_rpos != nullptr) {
        struct Rpos16 {
            i32 v[48];
            static bool launchPtrWord(std::uint32_t) { return false; }  // positions only (launch-log digest)
        } tab{};
        std::copy(rpos_host.begin(), rpos_host.begin() + 3 * static_cast<std::size_t>(n), tab.v);
        hip::launch(k.set_rpos, D(1), D(64), 0, stream, rpos_buf, tab, I(n));
    } else {
        hip::upload(rpos_buf, rpos_host.data(), static_cast<std::size_t>(n) * 3 * 4);
    }
    mrope_on = true;
    rpos_rows = true;
}

// rows 0 .. n-1 = positions pos0 .. of the current sequence
void Model::visRowsCur(u32 n, u32 pos0) {
    visBegin();
    if (!visAny()) return;
    const VisMap* vm = visOf(cur_seq);
    if (vm == nullptr) return;
    visEnsure();
    for (u32 r = 0; r < n; ++r) visRow(vm, r, pos0 + r);
    visCommit(n);
}

// copy the image rows recorded by visRow into x (after the token lookup)
void Model::injectRows() {
    const u64 E = cfg.n_embd;
    // the sources are pageable host memory: such a copy is not reliably ordered after
    // the work already queued on the stream (which may still read x), so drain first
    hip::streamSync(stream);
    for (u32 j = 0; j < n_inj; ++j)
        hip::copyAnyAsync(x + static_cast<u64>(inj[j].row) * E * 4, reinterpret_cast<DevPtr>(inj[j].src), static_cast<u64>(inj[j].n) * E * 4, stream);
    n_inj = 0;
}

}  // namespace whirl::qwen35
