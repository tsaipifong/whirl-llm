// Model forward, Gated-DeltaNet layers (arch025-F3, moved from src/model/forward.cpp): the per-layer state segment
// tables (gdnSegsBatch / gdnSegsSingle), the prefill recurrence (gdnChunked / gdnScan) and the DeltaNet layer
// body (gdnLayer, extracted from Model::run). The kernel choices are ops::chooseGdn* (src/ops/gdn.cpp, arch025-F6);
// here the problems are built and the plans launched.
// SPDX-License-Identifier: Apache-2.0

#include "../forward_internal.h"

namespace whirl::qwen35 {

namespace {
ops::GdnShape gdnShapeOf(const Config& c) {
    ops::GdnShape s;
    s.d_conv = c.d_conv;
    s.d_state = c.d_state;
    s.head_v = c.headV();
    s.conv_ch = c.convCh();
    s.n_v_heads = c.n_v_heads;
    s.n_k_heads = c.n_k_heads;
    return s;
}
}  // namespace

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
    ops::GdnChunkProblem cp;
    cp.n = n;
    cp.n_v_heads = cfg.n_v_heads;
    cp.n_k_heads = cfg.n_k_heads;
    cp.gc_m_bytes = gc_m_bytes;
    cp.gc_w_bytes = gc_w_bytes;
    cp.gc_u_bytes = gc_u_bytes;
    ops::GdnChunkFlags cf;
    cf.gdn_strix = gdn_strix;
    cf.gdn_strix_min = gdn_strix_min;
    cf.gdn_wmma = gdn_wmma;
    const ops::GdnChunkPlan pl = ops::chooseGdnChunk(cp, cf, k);
    if (pl.kind == ops::GdnChunk::strix) {
        // GDN-8S: strix-llama.cpp chunks (CL = 32) on the existing scratch: T' | P in gc_m, Vh in gc_w,
        // Qh | Kh | gate factors in gc_u
        const u64 H = cfg.n_v_heads, Hk = cfg.n_k_heads, nch = pl.nch, tp = pl.tp, qk = pl.qk;
        hip::launch(k.gdn_sx_prep, D(Hk, nch), D(256), 0, stream, conv_o, alpha_b, beta_b, gc_m, gc_m + tp, gc_u + 2 * qk, gc_u, gc_u + qk, gc_w,
                    I(H), I(Hk), I(n), I(nch), I(ch), scale);
        // state in place (curr_state == state); no fused output norm (on_* = null)
        hip::launch(k.gdn_sx_scan, D(H), D(256), 0, stream, gc_u, gc_u + qk, gc_w, ssm_state[i], out, ssm_state[i], gc_m, gc_m + tp,
                    gc_u + 2 * qk, I(H), I(Hk), I(n), I(nch), scale, u64(0), u64(0), u64(0), u64(0), 0.0f, false);
        return;
    }
    if (pl.kind == ops::GdnChunk::wmma) {
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
    if (ops::chooseGdnScan(gdn_chunked, n, gdnShapeOf(cfg)) == ops::GdnScan::chunked) {
        gdnChunked(i, conv_o, alpha_b, beta_b, out, n, scale);
    } else {
        hip::launch(k.gdn_seq_128, D(cfg.n_v_heads, cfg.headV() / 32), D(128), 0, stream, conv_o, alpha_b, beta_b, ssm_state[i], out, I(n),
                    I(cfg.n_k_heads), I(cfg.n_v_heads), I(cfg.headV()), I(ch), scale, snap_rows > 0 ? snap_ssm[0][i] : u64(0), i32(0));
    }
}

// The DeltaNet (gated delta rule) layer body of Model::run (arch025-F3, extract-method of run's non-attention
// branch; g = layers[i].gdn, gdn_in2 = run's flag that x16b holds this layer's normed input as f16 for [beta; alpha]).
void Model::gdnLayer(std::size_t i, const GdnW& g, u32 n, bool gdn_in2) {
    const u32 ch = cfg.convCh();
    const ops::GdnShape gs = gdnShapeOf(cfg);
    ops::GdnBaProblem bap;
    bap.gdn_ba_on = gdn_ba_on;
    bap.ba_buf = ba_buf != 0;
    bap.ba_contiguous = rowsContiguous(g.beta, g.alpha);
    // MXFP4 speed mode: the qkv / z GEMMs write f16 for the conv and the gated norm
    ops::GdnH16Problem hp;
    hp.qkv = matShape(g.qkv);
    hp.gate = matShape(g.gate);
    hp.out = matShape(g.out);
    hp.qkv_tune = g.qkv.tune;
    hp.gate_tune = g.gate.tune;
    hp.n = n;
    hp.s = gs;
    hp.fused = fused(n);
    ops::GdnH16Flags hf;
    hf.ffn_h16 = ffn_h16;
    hf.act_fuse = act_fuse;
    const bool gdn_h16 = ops::chooseGdnH16(hp, hf, matmulFlags(), h16Tune(), k);
    out_h16 = gdn_h16;
    {
        const Mat ws[2] = {g.qkv, g.gate};
        const DevPtr ys[2] = {qkv, z};
        matmulGroup(ws, h, ys, n);
    }
    out_h16 = false;
    if (ops::gdnFusable(fused(n), matShape(g.beta), matShape(g.alpha), gs, k)) {
        const u32 nh = cfg.n_v_heads;
        const u32 n_seg = bplan ? static_cast<u32>(bplan->segs.size()) : 1;
        const GdnSegs segs_c = bplan ? gdnSegsBatch(*bplan, i, true) : gdnSegsSingle(i, true, n);
        const ops::GdnDecodePlan dp = ops::chooseGdnDecode(g.beta.ty, n, small_max, prec_dec, dbg_flags, bap, k);
        if (dp.gates == ops::GdnDecodeGates::precise_ba || dp.gates == ops::GdnDecodeGates::precise_split) {
            // precise decode: [beta; alpha] on the f16 GEMM from h (f32, written by
            // rmsnormQ8), the gates, then the same conv step as below
            const u32 nhr = n * nh;
            if (dp.gates == ops::GdnDecodeGates::precise_ba) {
                matmul(concatRows(g.beta, g.alpha), h, ba_buf, n, false);
                hip::launch(k.gdn_gates_ba, D((nhr + 63) / 64), D(64), 0, stream, ba_buf, beta, alpha, g.dt, g.a, I(nhr), I(nh));
            } else {
                matmul(g.beta, h, beta, n, false);
                matmul(g.alpha, h, alpha, n, false);
                hip::launch(k.gdn_gates, D((nhr + 63) / 64), D(64), 0, stream, beta, alpha, g.dt, g.a, I(nhr), I(nh));
            }
            hip::launch(k.gdn_conv_l2, D(ch / 128, n_seg), D(128), 0, stream, qkv, g.conv, conv_out, I(ch), I(2 * cfg.n_k_heads), cfg.eps,
                        segs_c);
        } else if (dp.gates == ops::GdnDecodeGates::abconv) {
            // beta/alpha projections and the conv step share one launch
            const u32 ab_blocks = (2 * nh + 7) / 8;
            hip::launch(dp.f, D(ab_blocks * n + (ch / 128) * n_seg), D(256), 0, stream, g.beta.ptr, g.alpha.ptr, g.beta.row_bytes, xq, xd,
                        beta, alpha, g.dt, g.a, I(g.beta.ncols), I(nh), I(n), qkv, g.conv, conv_out, I(ch), I(2 * cfg.n_k_heads), cfg.eps,
                        segs_c);
        } else {
            hip::launch(dp.f, D((2 * nh + 7) / 8, n), D(256), 0, stream, g.beta.ptr, g.alpha.ptr, g.beta.row_bytes, xq,
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
        const ops::GdnGates gt = ops::chooseGdnGates(gdn_in2, bap, k);
        if (gt == ops::GdnGates::in2) {
            // the [beta; alpha] f16 GEMM straight from x16b (gdnIn2 implies the gates_ba path)
            const Mat ba = concatRows(g.beta, g.alpha);
            gemmF16(ba, x16 + x16bOff(), ba_buf, n, 0);
            if (ba_buf == xq_src) xq_src = 0;
            if (ba_buf == x16_src) x16_src = 0;
            if (prof) prof->matmul_bytes += ba.row_bytes * ba.nrows;
            mark(OpClass::matmul);
            hip::launch(k.gdn_gates_ba, D((nh + 63) / 64), D(64), 0, stream, ba_buf, beta, alpha, g.dt, g.a, I(nh), I(cfg.n_v_heads));
        } else if (gt == ops::GdnGates::ba) {
            // one [beta; alpha] GEMM (beta's tune) into ba[t][2 * n_v_heads]
            matmul(concatRows(g.beta, g.alpha), h, ba_buf, n, false);
            hip::launch(k.gdn_gates_ba, D((nh + 63) / 64), D(64), 0, stream, ba_buf, beta, alpha, g.dt, g.a, I(nh), I(cfg.n_v_heads));
        } else {
            matmul(g.beta, h, beta, n, false);
            matmul(g.alpha, h, alpha, n, false);
            hip::launch(k.gdn_gates, D((nh + 63) / 64), D(64), 0, stream, beta, alpha, g.dt, g.a, I(nh), I(cfg.n_v_heads));
        }
        const ops::GdnConvPlan cv = ops::chooseGdnConv(!psegs.empty(), gdn_h16, n, gs, act_fuse, k);
        if (cv.conv == ops::GdnConv::seg_h16) {
            // the solo f16 conv kernels per segment (segmented == solo bitwise)
            for (const PRows& ps : psegs) {
                selectSeq(ps.seq);
                const u64 qh = static_cast<u64>(ps.r0) * ch * 2;
                const u64 qo = static_cast<u64>(ps.r0) * ch * 4;
                hip::launch(k.gdn_conv_l2n_h, D(ch / 128, (ps.n + 7) / 8), D(128), 0, stream, qkv + qh, conv_state[i], g.conv, conv_out + qo,
                            I(ch), I(ps.n), I(2 * cfg.n_k_heads), cfg.eps);
                hip::launch(k.gdn_conv_state_h, D((ch + 255) / 256), D(256), 0, stream, qkv + qh, conv_state[i], I(ch), I(ps.n),
                            snap_rows > 0 ? snap_conv[0][i] : u64(0), i32(0));
            }
        } else if (cv.conv == ops::GdnConv::seg_par) {
            for (const PRows& ps : psegs) {
                selectSeq(ps.seq);
                const u64 qo = static_cast<u64>(ps.r0) * ch * 4;
                hip::launch(k.gdn_conv_par, D((ch + 255) / 256, ps.n), D(256), 0, stream, qkv + qo, conv_state[i], g.conv, conv_out + qo, I(ch),
                            I(ps.n));
                hip::launch(k.gdn_conv_state, D((ch + 255) / 256), D(256), 0, stream, qkv + qo, conv_state[i], I(ch), I(ps.n),
                            snap_rows > 0 ? snap_conv[0][i] : u64(0), i32(0));
            }
        } else if (cv.conv != ops::GdnConv::seq) {
            // conv + q/k l2norm in one pass (bitwise the same) when the shapes allow
            if (cv.conv == ops::GdnConv::l2n_h) {
                hip::launch(k.gdn_conv_l2n_h, D(ch / 128, (n + 7) / 8), D(128), 0, stream, qkv, conv_state[i], g.conv, conv_out, I(ch), I(n),
                            I(2 * cfg.n_k_heads), cfg.eps);
            } else if (cv.conv == ops::GdnConv::l2n) {
                hip::launch(k.gdn_conv_l2n, D(ch / 128, (n + 7) / 8), D(128), 0, stream, qkv, conv_state[i], g.conv, conv_out, I(ch), I(n),
                            I(2 * cfg.n_k_heads), cfg.eps);
            } else {
                hip::launch(k.gdn_conv_par, D((ch + 255) / 256, n), D(256), 0, stream, qkv, conv_state[i], g.conv, conv_out, I(ch), I(n));
            }
            hip::launch(cv.state_h ? k.gdn_conv_state_h : k.gdn_conv_state, D((ch + 255) / 256), D(256), 0, stream, qkv, conv_state[i], I(ch),
                        I(n), snap_rows > 0 ? snap_conv[0][i] : u64(0), i32(0));
        } else {
            hip::launch(k.gdn_conv_seq, D((ch + 255) / 256), D(256), 0, stream, qkv, conv_state[i], g.conv, conv_out, I(ch), I(n));
        }
        // q and k heads are contiguous at the front of each token's conv row
        if (cv.l2norm)
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
        } else if (ops::chooseGdnScan(gdn_chunked, n, gs) == ops::GdnScan::chunked) {
            gdnChunked(i, conv_out, alpha, beta, gdn_out, n, scale);
        } else {
            hip::launch(k.gdn_seq_128, D(cfg.n_v_heads, cfg.headV() / 32), D(128), 0, stream, conv_out, alpha, beta, ssm_state[i], gdn_out,
                        I(n), I(cfg.n_k_heads), I(cfg.n_v_heads), I(cfg.headV()), I(ch), scale,
                        snap_rows > 0 ? snap_ssm[0][i] : u64(0), i32(0));
        }
        mark(OpClass::gdn_rec);
        // gated norm straight into the out-proj GEMM input when possible
        const ops::GdnNormPlan gn = ops::chooseGdnGatedNorm(matShape(g.out), n, gs, gdn_h16, act_fuse, g8t, matmulFlags(), k);
        if (gn.f != nullptr) {
            hip::launch(gn.f, D(n), D(256), 0, stream, gdn_out, z, g.norm, x16, sx8, I(cfg.n_v_heads), cfg.eps);
            xq_src = 0;
            x16_src = gdn_out;
            x16_count = n * cfg.n_v_heads * 128;
            x16_fp8 = gn.fp8;
        } else {
            hip::launch(k.gdn_gated_norm, D(nh), D(128), 0, stream, gdn_out, z, g.norm, I(cfg.headV()), cfg.eps);
            xq_src = 0;
            x16_src = 0;
        }
        mark(OpClass::gdn_norm);
    }
    matmul(g.out, gdn_out, x, n, true);
}

}  // namespace whirl::qwen35
