// Model forward, matmul dispatch (arch025-F1, moved from src/model/forward.cpp): profiler marks, GEMV / GEMM
// launches and grouping, the activation-input problems (h16Out / actCommon / rmsnormIn / gdnIn2; the choices are in
// src/ops/act.cpp since arch025-F5), the fused-path predicates (rules in ops/act.h), rmsnorm / elementwise helpers
// and Model::step.
// SPDX-License-Identifier: Apache-2.0

#include "../forward_internal.h"

namespace whirl::qwen35 {

void Model::mark(OpClass cls) {
    if (prof) prof->record(stream, cls);
}

// int8 GEMV of the matrices ws (same type and ncols) on input x (n tokens) into ys. The kernel and grid come from
// ops::chooseGemvTile / chooseGemvWide (K-1); here the input is quantized and the arguments are built. Per-row
// arithmetic is that of the single launch (class q8_row).
void Model::gemvLaunch(std::span<const Mat> ws, DevPtr xin, std::span<const DevPtr> ys, u32 n, i32 acc) {
    const Mat& w = ws[0];
    if (xq_src != xin || xq_n != n) {
        const u32 cnt = n * w.ncols;
        hip::launch(k.quantize_q8, D((cnt + 255) / 256), D(256), 0, stream, xin, xq, xd, I(cnt));
        xq_src = xin;
        xq_n = n;
        mark(OpClass::quant);
    }
    if (n <= max_small_batch) {
        gemvKernels(ws, ys, n, acc, xq, xd);
        return;
    }
    // 17..32 tokens (wide batched verify): the two-tile WMMA GEMV where the type has one
    // (weights read once), else 16-token pieces through the multi-token kernels. Either
    // way every token row is computed exactly as the one-token GEMV computes it.
    const ops::MatShape w0 = matShape(w);
    const ops::GemvWide wide = ops::chooseGemvWide(w0, ws.size(), k);
    if (wide == ops::GemvWide::gemvx) {
        for (std::size_t si = 0; si < ws.size(); ++si) {
            const Mat& wi = ws[si];
            hip::launch(k.gemvx[ti(wi.ty)], ops::gemvxGrid(wi.nrows), D(256), 0, stream, wi.ptr, wi.row_bytes, xq, xd, ys[si], I(wi.ncols),
                        I(wi.nrows), acc, gate, I(n));
        }
        return;
    }
    if (wide == ops::GemvWide::head_split) {
        // output head: per range of rows, tokens [0, 16) then [n - 16, n) (rows of the overlap are
        // computed twice, to the same bits); the second pass finds the range's weights in cache
        const u32 V = w.nrows, chunk = nf().head_chunk;
        for (u32 r0 = 0; r0 < V; r0 += chunk) {
            const u32 cr = std::min(chunk, V - r0);
            for (const u32 t0 : {0u, n - max_small_batch}) {
                hip::launch(k.gemvw_head_s, ops::headSplitGrid(cr), D(256), 0, stream, w.ptr + static_cast<u64>(r0) * w.row_bytes, w.row_bytes,
                            xq + static_cast<u64>(t0) * w.ncols, xd + static_cast<u64>(t0) * (w.ncols / 32) * 4,
                            ys[0] + (static_cast<u64>(t0) * V + r0) * 4, I(w.ncols), I(cr), I(V), acc, gate);
            }
        }
        return;
    }
    for (u32 t0 = 0; t0 < n; t0 += max_small_batch) {
        const u32 nt = std::min(max_small_batch, n - t0);
        DevPtr yo[3] = {0, 0, 0};
        for (std::size_t si = 0; si < ws.size(); ++si) yo[si] = ys[si] + static_cast<u64>(t0) * ws[si].nrows * 4;
        gemvKernels(ws, std::span<const DevPtr>(yo, ws.size()), nt, acc, xq + static_cast<u64>(t0) * w.ncols,
                    xd + static_cast<u64>(t0) * (w.ncols / 32) * 4);
    }
}

// The int8 GEMV launches for n <= 16 tokens of xqp / xdp (already quantized): ops::chooseGemvTile's plan.
void Model::gemvKernels(std::span<const Mat> ws, std::span<const DevPtr> ys, u32 n, i32 acc, DevPtr xqp, DevPtr xdp) {
    ops::MatShape sh[3];
    for (std::size_t si = 0; si < ws.size(); ++si) sh[si] = matShape(ws[si]);
    const ops::GemvTune tn{&gemv_r, &gemv_w, &gemv_w_head, nf().gv_nmax};
    const ops::GemvPlan p = ops::chooseGemvTile(sh, ws.size(), n, k, tn);
    const i32 ncols = I(ws[0].ncols);
    if (p.grouped) {
        GvArgs g;
        u32 nb = 0;
        for (std::size_t si = 0; si < ws.size(); ++si) {
            const Mat& wi = ws[si];
            if (si == 1) g.b1 = I(nb);
            if (si == 2) g.b2 = I(nb);
            g.w[si] = wi.ptr;
            g.y[si] = ys[si];
            g.rb[si] = wi.row_bytes;
            g.n[si] = I(wi.nrows);
            nb += (wi.nrows + p.rows_per_block - 1) / p.rows_per_block;
        }
        hip::launch(p.f, p.grid[0], p.block, 0, stream, g, xqp, xdp, ncols, acc, gate);
    } else {
        for (std::size_t si = 0; si < ws.size(); ++si) {
            const Mat& wi = ws[si];
            hip::launch(p.f, p.grid[si], p.block, 0, stream, wi.ptr, wi.row_bytes, xqp, xdp, ys[si], ncols, I(wi.nrows), acc, gate);
        }
    }
}

// Matmuls (no accumulate) of ws on the same input into ys. On the int8 GEMV
// path the matrices of one type run as one grouped launch (bitwise the same).
void Model::matmulGroup(std::span<const Mat> ws, DevPtr xin, std::span<const DevPtr> ys, u32 n) {
    const bool ok = nf().gv_group && k.gv_grp && n <= gvMax() && !nf().float_gemv && ops::preciseAllows(ops::BitClass::q8_row, nf().prec_dec, n, gvMax());
    bool done[3] = {false, false, false};
    for (std::size_t i = 0; i < ws.size(); ++i) {
        if (done[i]) continue;
        const Mat& w = ws[i];
        if (!ok || k.gemvq[ti(w.ty)] == nullptr || w.ptr == output.ptr) {
            matmul(w, xin, ys[i], n, false);
            continue;
        }
        Mat gw[3];
        DevPtr gy[3];
        std::size_t cnt = 0;
        for (std::size_t j = i; j < ws.size(); ++j) {
            const Mat& w2 = ws[j];
            if (done[j] || w2.ty != w.ty || w2.ncols != w.ncols || w2.ptr == output.ptr) continue;
            gw[cnt] = w2;
            gy[cnt] = ys[j];
            cnt += 1;
            done[j] = true;
        }
        gemvLaunch(std::span<const Mat>(gw, cnt), xin, std::span<const DevPtr>(gy, cnt), n, 0);
        for (std::size_t j = 0; j < cnt; ++j) {
            if (gy[j] == xq_src) xq_src = 0;
            if (gy[j] == x16_src) x16_src = 0;
        }
        if (prof)
            for (std::size_t j = 0; j < cnt; ++j) prof->matmul_bytes += gw[j].row_bytes * gw[j].nrows;
        mark(head_phase ? OpClass::head : OpClass::matmul);
    }
}

// y[t][:] (+)= W x[t][:] for n tokens: GEMV for small n, tiled GEMM otherwise.
void Model::matmul(const Mat& w, DevPtr xin, DevPtr y, u32 n, bool accumulate) {
    const i32 acc = accumulate ? 1 : 0;
    const i32 ncols = I(w.ncols);
    const i32 nrows = I(w.nrows);
    const std::size_t t = ti(w.ty);
    auto finish = [&]() {
        if (y == xq_src) xq_src = 0;
        if (y == x16_src) x16_src = 0;
        if (prof) prof->matmul_bytes += w.row_bytes * w.nrows;
        mark(head_phase ? OpClass::head : OpClass::matmul);
    };
    const ops::MatmulPath path = ops::chooseMatmulPath(matShape(w), n, matmulFlags(), k);
    if (path == ops::MatmulPath::gemvq) {
        const DevPtr yy[1] = {y};
        gemvLaunch(std::span<const Mat>(&w, 1), xin, yy, n, acc);
    } else if (path == ops::MatmulPath::gemv1) {
        const u32 rows_per_block = 8;
        hip::launch(k.gemv1[t], D((w.nrows + rows_per_block - 1) / rows_per_block), D(256), 0, stream, w.ptr, w.row_bytes, xin, y, ncols,
                    nrows, ncols, nrows, acc);
    } else if (path == ops::MatmulPath::float_gemv) {
        // f32 / f16 weights, n = 2..gvMax() (16; 17..32 in a wide verify): f32 activations through
        // gemv_<T>_8 / _4 / _1, each token bitwise == the n = 1 gemv_<T>_1 (C-13; the f16 GEMM is
        // not). A code object without the _4 / _8 instances runs the _1 kernel per token.
        const u32 rows_per_block = 8;
        const hip::Dim3 grid = D((w.nrows + rows_per_block - 1) / rows_per_block);
        for (u32 t0 = 0; t0 < n;) {
            const u32 left = n - t0;
            const u32 nt = floatGemvStep(left, k.gemv8[t] != nullptr, k.gemv4[t] != nullptr);
            const hip::Function f = nt == 8 ? k.gemv8[t] : nt == 4 ? k.gemv4[t] : k.gemv1[t];
            hip::launch(f, grid, D(256), 0, stream, w.ptr, w.row_bytes, xin + static_cast<DevPtr>(t0) * w.ncols * 4,
                        y + static_cast<DevPtr>(t0) * w.nrows * 4, ncols, nrows, ncols, nrows, acc);
            t0 += nt;
        }
    } else if (path == ops::MatmulPath::gemm_fp8) {
        // MXFP4 x fp8 (per-token scaled activations, folded block exponents)
        const u32 count = n * w.ncols;
        if (x16_src != xin || x16_count != count || !x16_fp8) {
            hip::launch(x8(k.qact_fp8, k.qact_fp8t), D(n), D(256), 0, stream, xin, x16, sx8, ncols);
            x16_src = xin;
            x16_count = count;
            x16_fp8 = true;
        }
        const ops::Gemm8Plan g8 = ops::chooseGemm8(w.nrows, n, out_h16, nf().g8t, k);
        hip::launch(g8.f, g8.grid, g8.block, 0, stream, w.ptr, w.row_bytes, w.ref, x16, sx8, y, ncols, nrows, I(n), acc);
    } else {
        const u32 count = n * w.ncols;
        if (x16_src != xin || x16_count != count || x16_fp8) {
            hip::launch(k.f32_to_f16, D((count / 4 + 255) / 256), D(256), 0, stream, xin, x16, I(count));
            x16_src = xin;
            x16_count = count;
            x16_fp8 = false;
        }
        gemmF16(w, x16, y, n, acc);
    }
    finish();
}

// The f16-activation GEMM of matmul: y[t][:] (+)= W x16in[t][:] (x16in: n rows of w.ncols f16; not registered in
// the x16 cache). The kernel, the optional dequant into w16 and the grids come from ops::chooseGemmF16 (K-2).
void Model::gemmF16(const Mat& w, DevPtr x16in, DevPtr y, u32 n, i32 acc) {
    const ops::GemmPlan p = ops::chooseGemmF16(matShape(w), n, w.tune[tuneBucket(n)], gemmFlags(), k);
    const i32 ncols = I(w.ncols);
    const i32 nrows = I(w.nrows);
    switch (p.kind) {
        case ops::GemmKind::gemvh:
            hip::launch(p.f, p.grid, p.block, 0, stream, w.ptr, w.row_bytes, x16in, y, ncols, nrows, I(n), acc);
            return;
        case ops::GemmKind::gemms:
            hip::launch(p.f, p.grid, p.block, 0, stream, w.ptr, w.row_bytes, x16in, y, ncols, nrows, I(n), acc);
            return;
        case ops::GemmKind::gemmhq:
            hip::launch(p.f, p.grid, p.block, 0, stream, w.ptr, w.row_bytes, x16in, y, ncols, nrows, I(n), acc);
            return;
        default:
            break;
    }
    DevPtr wptr = w.ptr;
    u64 wrb = w.row_bytes;
    if (p.dequant) {
        hip::launch(p.dq, p.dq_grid, D(256), 0, stream, w.ptr, w.row_bytes, w16, ncols, nrows);
        wptr = w16;
        wrb = static_cast<u64>(w.ncols) * 2;
    }
    if (p.kind == ops::GemmKind::gemmh)
        hip::launch(p.f, p.grid, p.block, 0, stream, wptr, x16in, y, ncols, nrows, I(n), acc);
    else
        hip::launch(p.f, p.grid, p.block, 0, stream, wptr, wrb, x16in, y, ncols, nrows, I(n), acc);
}

// the f16 GEMM flags (precise rows, f16 outputs, the profile's precise-decode geometry). feat026-o6 (3), owner O6: the
// research switches WHIRL_PREC_GEMVH / WHIRL_PREC_GS are gone (config::removedSwitchWarnings warns at load).
ops::GemmFlags Model::gemmFlags() const {
    ops::GemmFlags f;
    f.prec_dec = nf().prec_dec;
    f.out_h16 = out_h16;
    f.h16_dq = nf().h16_dq;
    f.gemmh_on = nf().gemmh_on;
    f.gemmhq_on = nf().gemmhq_on;
    f.gemvh_cap = device::profile(arch).gemvh_grid_by_rows ? kernels::gemvhCap() : 0;
    f.gemms_tie_larger_bm = device::profile(arch).gemms_tie_larger_bm;
    f.gv_max = gvMax();
    return f;
}

// Every fp8 activation producer writes the fragment-tiled layout and the fp8
// GEMMs run gemm8t (only when the whole kernel set is present).
// arch025-N2b: a pure predicate (numerics::Target::k_gemm8t); numerics::plan sets Flags::g8t (was set here)
bool Model::hasTiledFp8() const {
    for (std::size_t ci = 0; ci < gemm8t_cfgs.size(); ++ci)
        if (k.gemm8t[ci] == nullptr || k.gemm8th[ci] == nullptr) return false;
    if (k.qact_fp8t == nullptr || k.rmsnorm_x8t == nullptr || k.silu_mul_x8t == nullptr || k.silu_mul_x8ht == nullptr ||
        k.gated_norm_x8t == nullptr || k.gated_norm_x8ht == nullptr)
        return false;
    return true;
}

void Model::rmsnorm(DevPtr xin, DevPtr w, DevPtr out, u32 n, u32 count, u32 in_stride, u32 out_stride) {
    xq_src = 0;
    x16_src = 0;
    hip::launch(k.rmsnorm, D(count), D(256), 0, stream, xin, w, out, I(n), I(in_stride), I(out_stride), cfg.eps);
    mark(OpClass::norm);
}

// The activation-input choice (arch025-F5): ops::chooseH16Out / chooseActCommon / chooseRmsnormIn / chooseGdnIn2
// (src/ops/act.cpp). Here the matrices become ops::MatShape, the flags are gathered (the device profile's
// h16_from_candidates among them) and the norm is launched.
namespace {
// the consumers of one activation as ops shapes (every caller passes 1..4 matrices)
struct ConsumerShapes {
    ops::MatShape s[4];
    std::size_t n = 0;
    std::span<const ops::MatShape> span() const { return {s, n}; }
};
}  // namespace

ops::H16Tune Model::h16Tune() const {
    ops::H16Tune h;
    h.from_candidates = device::profile(arch).h16_from_candidates;
    h.tune_mask = tune_mask;
    h.n_choices = n_choices;
    h.h16_dq = nf().h16_dq;
    return h;
}

bool Model::h16Out(const Mat& w, u32 n, std::uint8_t cls) const { return ops::chooseH16Out(matShape(w), w.tune, n, cls, matmulFlags(), h16Tune(), k); }

std::optional<ops::ActIn> Model::actCommon(std::span<const Mat> consumers, u32 n, std::uint8_t cls) const {
    if (consumers.size() > 4) throw ModelError("ActCommonConsumers");
    ConsumerShapes cs;
    for (const Mat& w : consumers) cs.s[cs.n++] = matShape(w);
    return ops::chooseActCommon(cs.span(), n, cls, nf().act_fuse, matmulFlags(), k);
}

// Prefill rmsnorm of x (n rows of n_embd) for the matmuls `consumers`: when
// they all take the same GEMM input form, the norm writes that form (x16, +
// sx8 for fp8) directly and `out` is left unwritten (x16 is registered as its
// conversion); else a plain rmsnorm into out.
void Model::rmsnormIn(DevPtr xin, DevPtr w, DevPtr out, u32 n, std::span<const Mat> consumers, std::uint8_t cls) {
    const u32 E = cfg.n_embd;
    if (consumers.size() > 4) throw ModelError("RmsnormInConsumers");
    ConsumerShapes cs;
    for (const Mat& m : consumers) cs.s[cs.n++] = matShape(m);
    const ops::RmsnormInPlan p = ops::chooseRmsnormIn(E, cs.span(), n, cls, nf().act_fuse, nf().g8t, matmulFlags(), k);
    if (p.f != nullptr) {
        hip::launch(p.f, D(n), D(256), 0, stream, xin, w, x16, sx8, I(E), cfg.eps);
        xq_src = 0;
        x16_src = out;
        x16_count = n * E;
        x16_fp8 = p.fp8;
        mark(OpClass::norm);
        return;
    }
    rmsnorm(xin, w, out, E, n, E, E);
}

// n>16 GDN prefill with qkv / gate on fp8 and contiguous [beta; alpha] on f16:
// one rmsnorm_x8h16(t) writes both inputs (WHIRL_GDN_IN2=0 off).
bool Model::gdnIn2(const GdnW& g, u32 n) const {
    ops::GdnIn2Problem p;
    p.qkv = matShape(g.qkv);
    p.gate = matShape(g.gate);
    p.beta = matShape(g.beta);
    p.alpha = matShape(g.alpha);
    p.n = n;
    p.n_embd = cfg.n_embd;
    p.fused = fused(n);
    p.ba_buf = ba_buf != 0;
    p.ba_contiguous = rowsContiguous(g.beta, g.alpha);
    ops::GdnIn2Flags f;
    f.gdn_in2_on = nf().gdn_in2_on;
    f.gdn_ba_on = nf().gdn_ba_on;
    f.act_fuse = nf().act_fuse;
    f.g8t = nf().g8t;
    return ops::chooseGdnIn2(p, f, matmulFlags(), k);
}

// The matmul path of n rows of w: ops::chooseMatmulPath (K-1; the precise rule lives there). f32 / f16 weights
// (no int8 GEMV) at n = 2..gvMax() run the f32-activation GEMV instances, bitwise == n = 1 per token (C-13).
ops::MatShape Model::matShape(const Mat& w) const {
    ops::MatShape s;
    s.ty = w.ty;
    s.nrows = w.nrows;
    s.ncols = w.ncols;
    s.row_bytes = w.row_bytes;
    s.has_ref = w.ref != 0;
    s.head = w.ptr == output.ptr;
    return s;
}

ops::MatmulFlags Model::matmulFlags() const {
    ops::MatmulFlags f;
    f.prec_dec = nf().prec_dec;
    f.float_gemv = nf().float_gemv;
    f.fp8_prefill = nf().fp8_prefill;
    f.fp8_mask = nf().fp8_mask;
    f.mm_class = mm_class;
    f.gv_max = gvMax();
    return f;
}

// Precise decode needs, for every weight type of the model, the f16 GEMM on the quantized
// bytes (gemm_c / gemms) and, for MoE experts, the grouped f16 expert GEMMs. Returns the
// first missing piece ("" = complete).
std::string Model::precMissing() const {
    if (k.f32_to_f16 == nullptr) return "f32_to_f16";
    auto tyName = [](GgmlType t) { return std::string(gguf::typeName(t)); };
    auto gemmOk = [&](const Mat& w) {
        if (w.ptr == 0) return true;
        const std::size_t t = ti(w.ty);
        if (k.gemmc[0][t] != nullptr) return true;
        if (w.ncols % 256 == 0)
            for (std::size_t si = 0; si < k.gemms.size(); ++si)
                if (k.gemms[si][t] != nullptr) return true;
        return false;
    };
    auto moeOk = [&](const MoeW& mo) -> std::string {
        if (k.silu_mul == nullptr) return std::string("silu_mul");
        for (const Mat* e : {&mo.gate, &mo.up})
            if (k.moe_gu_f[ti(e->ty)] == nullptr) return "the f32-activation decode expert kernel for " + tyName(e->ty) + " experts";
        if (k.moe_down_f[ti(mo.down.ty)] == nullptr) return "the f32-activation decode expert kernel for " + tyName(mo.down.ty) + " experts";
        return std::string();
    };
    for (const Layer& L : layers) {
        for (const Mat& w : layerMats(L).slice())
            if (!gemmOk(w)) return "the f16 GEMM for " + tyName(w.ty) + " weights";
        if (L.moe) {
            if (auto m = moeOk(*L.moe); !m.empty()) return m;
        }
    }
    if (!gemmOk(output)) return "the f16 GEMM for " + tyName(output.ty) + " weights (output head)";
    if (mtp) {
        const MtpW& mw = *mtp;
        for (const Mat* w : {&mw.eh_proj, &mw.attn.q, &mw.attn.k, &mw.attn.v, &mw.attn.o, &mw.ffn_gate, &mw.ffn_up, &mw.ffn_down})
            if (!gemmOk(*w)) return "the f16 GEMM for " + tyName(w->ty) + " weights (MTP layer)";
        if (mw.moe) {
            if (auto m = moeOk(*mw.moe); !m.empty()) return m;
        }
    }
    return {};
}

// Decode fusions apply to small batches on the int8 GEMV path (ops::fusedRows).
bool Model::fused(u32 n) const { return ops::fusedRows(n, small_max, nf().float_gemv, nf().no_fuse); }

bool Model::fusedDecode() const {
    if (!fused(max_small_batch)) return false;
    for (const Layer& L : layers) {
        if (L.kind != LayerKind::gdn) continue;
        if (!ops::gdnAbFusable(matShape(L.gdn.beta), matShape(L.gdn.alpha), k) || !ops::gdnDecodeShape(cfg.d_conv, cfg.d_state, cfg.headV(), cfg.convCh()))
            return false;
    }
    return true;
}

// MoE: balance / fast with WHIRL_MOE_WIDE=1 only (17..32 rows on the int8 decode experts, see
// moeBlock); precise MoE keeps 16-row verifies (ops::wideAllowed)
bool Model::wideCapable() const {
    return ops::wideAllowed(nf().wide_verify, cfg.moe, nf().moe_wide, nf().prec_dec) && fusedDecode() && ops::wideKernels(k);
}

bool Model::wideOk() const { return wideCapable() && gdn_replay; }

void Model::rmsnormQ8(DevPtr xin, DevPtr w, DevPtr out, u32 n) {
    hip::launch(k.rmsnorm_q8, D(n), D(1024), 0, stream, xin, w, out, xq, xd, I(cfg.n_embd), cfg.eps);
    xq_src = out;
    xq_n = n;
    x16_src = 0;
    mark(OpClass::norm);
}

void Model::elementwise(hip::Function f, DevPtr a, DevPtr b, u32 n) {
    xq_src = 0;
    x16_src = 0;
    hip::launch(f, D((n + 255) / 256), D(256), 0, stream, a, b, I(n));
    mark(OpClass::misc);
}

void Model::step(u32 token, u32 pos) {
    const u32 one[1] = {token};
    forward(one, pos);
}

}  // namespace whirl::qwen35
