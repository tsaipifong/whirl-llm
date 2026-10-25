// Model forward, matmul dispatch (arch025-F1, moved from src/model/forward.cpp): profiler marks, GEMV / GEMM
// launches and grouping, the activation-input choice (actIn / h16Out / actCommon / rmsnormIn), the fused-path
// predicates, rmsnorm / elementwise helpers and Model::step.
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
    const bool head_gemvx = w0.head && k.gemvx[ti(w.ty)] != nullptr && envFlag("HEAD_GEMVX", false);
    const ops::GemvWide wide = ops::chooseGemvWide(w0, ws.size(), k, head_gemvx);
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
        const u32 V = w.nrows, chunk = head_chunk;
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
    const ops::GemvTune tn{&gemv_r, &gemv_w, &gemv_w_head, gv_nmax};
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
    const bool ok = gv_group && k.gv_grp && n <= gvMax() && !float_gemv && ops::preciseAllows(ops::BitClass::q8_row, prec_dec, n, gvMax());
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
        const ops::Gemm8Plan g8 = ops::chooseGemm8(w.nrows, n, out_h16, g8t, k);
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

// the f16 GEMM flags (precise rows, f16 outputs, the WHIRL_PREC_GEMVH / WHIRL_PREC_GS experiments, the profile's
// precise-decode geometry)
ops::GemmFlags Model::gemmFlags() const {
    static const bool gemvh_on = [] {
        const char* v = std::getenv("WHIRL_PREC_GEMVH");
        return v == nullptr || *v != '0';
    }();
    static const int forced = [] {
        const char* v = std::getenv("WHIRL_PREC_GS");
        return v != nullptr && *v != 0 ? std::atoi(v) : -1;
    }();
    ops::GemmFlags f;
    f.prec_dec = prec_dec;
    f.out_h16 = out_h16;
    f.h16_dq = h16_dq;
    f.gemmh_on = gemmh_on;
    f.gemmhq_on = gemmhq_on;
    f.prec_gemvh = gemvh_on;
    f.prec_gs = forced;
    f.gemvh_cap = device::profile(arch).gemvh_grid_by_rows ? kernels::gemvhCap() : 0;
    f.gemms_tie_larger_bm = device::profile(arch).gemms_tie_larger_bm;
    f.gv_max = gvMax();
    return f;
}

// Every fp8 activation producer writes the fragment-tiled layout and the fp8
// GEMMs run gemm8t (only when the whole kernel set is present).
bool Model::useTiledFp8() {
    for (std::size_t ci = 0; ci < gemm8t_cfgs.size(); ++ci)
        if (k.gemm8t[ci] == nullptr || k.gemm8th[ci] == nullptr) return false;
    if (k.qact_fp8t == nullptr || k.rmsnorm_x8t == nullptr || k.silu_mul_x8t == nullptr || k.silu_mul_x8ht == nullptr ||
        k.gated_norm_x8t == nullptr || k.gated_norm_x8ht == nullptr)
        return false;
    g8t = true;
    return true;
}

void Model::rmsnorm(DevPtr xin, DevPtr w, DevPtr out, u32 n, u32 count, u32 in_stride, u32 out_stride) {
    xq_src = 0;
    x16_src = 0;
    hip::launch(k.rmsnorm, D(count), D(256), 0, stream, xin, w, out, I(n), I(in_stride), I(out_stride), cfg.eps);
    mark(OpClass::norm);
}

Model::ActIn Model::actIn(const Mat& w, u32 n, std::uint8_t cls) const {
    if (ops::preciseRows(prec_dec, n, gvMax())) return ActIn::f16;
    if (n <= gvMax() && k.gemvq[ti(w.ty)] != nullptr && !float_gemv) return ActIn::gemv;
    if (n == 1 || ops::floatGemvN(matShape(w), n, gvMax(), k)) return ActIn::f32in;
    if (fp8_prefill && (fp8_mask & cls) != 0 && w.ty == GgmlType::mxfp4 && w.ref != 0 && k.gemm8[0] != nullptr) return ActIn::fp8;
    return ActIn::f16;
}

bool Model::h16Out(const Mat& w, u32 n, std::uint8_t cls) const {
    switch (actIn(w, n, cls)) {
        case ActIn::fp8:
            return k.gemm8h[0] != nullptr;
        case ActIn::f16:
            // Radeon 8060S: decided by the kernel set and the tuner's candidate list, never by
            // the tuned choices, so the output (f16 intermediates or not) does not depend on
            // autotune timings. The gfx1151 code object has no h16 fused / dequant GEMM, so
            // this is off for quantized weights while those are candidates.
            if (device::profile(arch).h16_from_candidates) {
                for (u32 cj = 0; cj < n_choices && cj < 64; ++cj) {
                    if ((tune_mask & (1ull << cj)) == 0) continue;
                    if (cj >= 2 * gemm_cfgs.size()) {
                        const std::size_t si = cj - 2 * gemm_cfgs.size();
                        if (k.gemms[si][ti(w.ty)] == nullptr || w.ncols % 256 != 0) continue;  // not a candidate (= tune.cpp)
                        if (k.gemmsh[si][ti(w.ty)] == nullptr) return false;
                        continue;
                    }
                    if (!((cj >= gemm_cfgs.size() || w.ty == GgmlType::f16) && k.gemmch[cj % gemm_cfgs.size()] != nullptr)) return false;
                }
                // (and the table must agree: a cached table may predate the candidate list)
            }
            // every bucket (not just n's), so the choice is the same for a request's
            // solo chunks and a segmented forward of any total size
            for (std::uint8_t choice : w.tune) {
                if (choice >= 2 * gemm_cfgs.size()) {
                    if (k.gemmsh[choice - 2 * gemm_cfgs.size()][ti(w.ty)] == nullptr) return false;
                    continue;
                }
                if (!((choice >= gemm_cfgs.size() || w.ty == GgmlType::f16 || h16_dq) && k.gemmch[choice % gemm_cfgs.size()] != nullptr))
                    return false;
            }
            return true;
        default:
            return false;
    }
}

std::optional<Model::ActIn> Model::actCommon(std::span<const Mat> consumers, u32 n, std::uint8_t cls) const {
    if (!act_fuse || consumers.empty()) return std::nullopt;
    const ActIn mode = actIn(consumers[0], n, cls);
    if (mode != ActIn::fp8 && mode != ActIn::f16) return std::nullopt;
    for (std::size_t i = 1; i < consumers.size(); ++i)
        if (actIn(consumers[i], n, cls) != mode) return std::nullopt;
    return mode;
}

// Prefill rmsnorm of x (n rows of n_embd) for the matmuls `consumers`: when
// they all take the same GEMM input form, the norm writes that form (x16, +
// sx8 for fp8) directly and `out` is left unwritten (x16 is registered as its
// conversion); else a plain rmsnorm into out.
void Model::rmsnormIn(DevPtr xin, DevPtr w, DevPtr out, u32 n, std::span<const Mat> consumers, std::uint8_t cls) {
    const u32 E = cfg.n_embd;
    if (E <= 256 * 32) {
        if (auto mode = actCommon(consumers, n, cls)) {
            const hip::Function f = *mode == ActIn::fp8 ? x8(k.rmsnorm_x8, k.rmsnorm_x8t) : k.rmsnorm_x16;
            if (f != nullptr) {
                hip::launch(f, D(n), D(256), 0, stream, xin, w, x16, sx8, I(E), cfg.eps);
                xq_src = 0;
                x16_src = out;
                x16_count = n * E;
                x16_fp8 = *mode == ActIn::fp8;
                mark(OpClass::norm);
                return;
            }
        }
    }
    rmsnorm(xin, w, out, E, n, E, E);
}

// n>16 GDN prefill with qkv / gate on fp8 and contiguous [beta; alpha] on f16:
// one rmsnorm_x8h16(t) writes both inputs (WHIRL_GDN_IN2=0 off).
bool Model::gdnIn2(const GdnW& g, u32 n) const {
    if (!gdn_in2_on || fused(n) || n <= 16 || cfg.n_embd > 8192) return false;
    if (x8(k.rmsnorm_x8h16, k.rmsnorm_x8h16t) == nullptr) return false;
    if (!gdn_ba_on || k.gdn_gates_ba == nullptr || ba_buf == 0 || !rowsContiguous(g.beta, g.alpha)) return false;
    const Mat pc[2] = {g.qkv, g.gate};
    const Mat bc[2] = {g.beta, g.alpha};
    const auto pin = actCommon(pc, n, 1);
    const auto bin = actCommon(bc, n, 1);
    return pin && *pin == ActIn::fp8 && bin && *bin == ActIn::f16;
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
    f.prec_dec = prec_dec;
    f.float_gemv = float_gemv;
    f.fp8_prefill = fp8_prefill;
    f.fp8_mask = fp8_mask;
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

// Decode fusions apply to small batches on the int8 GEMV path.
bool Model::fused(u32 n) const { return n <= small_max && !float_gemv && !no_fuse; }

bool Model::fusedDecode() const {
    if (!fused(max_small_batch)) return false;
    for (const Layer& L : layers) {
        if (L.kind != LayerKind::gdn) continue;
        if (!gdnAbFusable(L.gdn) || cfg.d_conv != 4 || cfg.d_state != 128 || cfg.headV() != 128 || cfg.convCh() % 128 != 0) return false;
    }
    return true;
}

// MoE: balance / fast with WHIRL_MOE_WIDE=1 only (17..32 rows on the int8 decode experts, see
// moeBlock); precise MoE keeps 16-row verifies
bool Model::wideCapable() const {
    return wide_verify && (!cfg.moe || (moe_wide && !prec_dec)) && fusedDecode() && k.gemvx[ti(GgmlType::q4_k)] != nullptr;
}

bool Model::wideOk() const { return wideCapable() && gdn_replay; }

bool Model::gdnAbFusable(const GdnW& g) const {
    return k.gdn_ab[ti(g.beta.ty) % n_types] != nullptr && g.beta.ty == g.alpha.ty && g.beta.row_bytes == g.alpha.row_bytes &&
           g.beta.ncols % 256 == 0;
}

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
