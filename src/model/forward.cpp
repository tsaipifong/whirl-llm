// Model forward: matmul dispatch, attention / DeltaNet / FFN / MoE blocks,
// prefill, decode, MTP drafting and speculative verify.
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (Model: gemvLaunch .. readHidden).

#include "whirl/model.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;

constexpr std::size_t ti(GgmlType t) { return static_cast<std::size_t>(t); }
inline i32 I(u64 v) { return static_cast<i32>(v); }

// Positions per attn_split block (FD_CH in the kernels).
constexpr u32 fd_chunk = 64;
// Flash-decoding split count cap.
constexpr u32 fd_max_splits = 64;
constexpr u32 moe_bm = 128;
constexpr u32 moe_bn = 64;
// smallest batch that takes gemmh_f16 instead of the tuned f16 GEMM
constexpr u32 gemmh_min = 512;

// attn_split / attn_wsplit split size for n_pos positions (whole FD_CH chunks).
i32 awPer(i32 n_pos, u32 n_split) {
    const i32 span = static_cast<i32>(n_split * fd_chunk);
    const i32 q = n_pos + span - 1;
    // floor division (n_pos >= 1 here, so q >= 0)
    return (q >= 0 ? q / span : -((-q + span - 1) / span)) * static_cast<i32>(fd_chunk);
}

u32 fdSplits(u32 max_ctx) { return std::min(fd_max_splits, (max_ctx + fd_chunk - 1) / fd_chunk); }

u32 gvNmax(GgmlType ty) {
    if (gv_nmax != 0) return gv_nmax;
    return ty == GgmlType::mxfp4 ? 13 : 11;
}

// gemm8 configuration for a batch of n tokens
std::size_t gemm8Cfg(u32 n) { return n <= 64 ? 3 : n <= 256 ? 1 : 0; }
// gemm8t configuration for a batch of n tokens
std::size_t gemm8tCfg(u32 n) { return n <= 32 ? 3 : n <= 48 ? 4 : n <= 64 ? 2 : n <= 192 ? 1 : 0; }

hip::Dim3 D(u64 x, u64 y = 1, u64 z = 1) { return hip::Dim3{static_cast<unsigned>(x), static_cast<unsigned>(y), static_cast<unsigned>(z)}; }

}  // namespace

void Model::mark(OpClass cls) {
    if (prof) prof->record(stream, cls);
}

// int8 GEMV of the matrices ws (same type and ncols) on input x (n tokens)
// into ys: one launch through the grouped ABI (the one-token entries, or the
// "g" twins of the multi-token entries for groups of n <= gv_nmax), else one
// plain launch per matrix. Per-row arithmetic is that of the single launch.
void Model::gemvLaunch(std::span<const Mat> ws, DevPtr xin, std::span<const DevPtr> ys, u32 n, i32 acc) {
    const Mat& w = ws[0];
    const i32 ncols = I(w.ncols);
    if (xq_src != xin || xq_n != n) {
        const u32 cnt = n * w.ncols;
        hip::launch(k.quantize_q8, D((cnt + 255) / 256), D(256), 0, stream, xin, xq, xd, I(cnt));
        xq_src = xin;
        xq_n = n;
        mark(OpClass::quant);
    }
    const std::size_t t = ti(w.ty);
    hip::Function f = n == 1 ? k.gemvq[t] : k.gemvq_nt[n - 2][t];
    hip::Function fg = n == 1 ? nullptr : k.gemvq_nt_g[n - 2][t];
    u32 rows_per_block = 8;
    const std::uint8_t wv = n < 2 ? 0 : (w.ty == GgmlType::q6_k && w.ptr == output.ptr) ? gemv_w_head[n] : gemv_w[t % n_types][n];
    const hip::Function wf = wv > 0 ? k.gemvw[wv][n - 2][t] : nullptr;
    if (wf != nullptr) {
        f = wf;
        fg = k.gemvw_g[wv][n - 2][t];
        rows_per_block = gemvw_rows[wv];
    } else if (n >= 2 && gemv_r[n] > 1) {
        const hip::Function mr = k.gemvq_mr[gemv_r[n] / 2 - 1][n - 2][t];
        if (mr != nullptr) {
            f = mr;
            fg = k.gemvq_mr_g[gemv_r[n] / 2 - 1][n - 2][t];
            rows_per_block = 8 * static_cast<u32>(gemv_r[n]);
        }
    }
    // one-token entries always take GvArgs (gv_grp); a multi-token group takes the twin
    const bool twin = n >= 2 && ws.size() > 1 && fg != nullptr && n <= gvNmax(w.ty);
    if (twin) f = fg;
    if ((k.gv_grp && n == 1) || twin) {
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
            nb += (wi.nrows + rows_per_block - 1) / rows_per_block;
        }
        hip::launch(f, D(nb), D(256), 0, stream, g, xq, xd, ncols, acc, gate);
    } else {
        for (std::size_t si = 0; si < ws.size(); ++si) {
            const Mat& wi = ws[si];
            hip::launch(f, D((wi.nrows + rows_per_block - 1) / rows_per_block), D(256), 0, stream, wi.ptr, wi.row_bytes, xq, xd, ys[si],
                        ncols, I(wi.nrows), acc, gate);
        }
    }
}

// Matmuls (no accumulate) of ws on the same input into ys. On the int8 GEMV
// path the matrices of one type run as one grouped launch (bitwise the same).
void Model::matmulGroup(std::span<const Mat> ws, DevPtr xin, std::span<const DevPtr> ys, u32 n) {
    const bool ok = gv_group && k.gv_grp && n <= gemv_max && !float_gemv;
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
    if (n <= gemv_max && k.gemvq[t] != nullptr && !float_gemv) {
        const DevPtr yy[1] = {y};
        gemvLaunch(std::span<const Mat>(&w, 1), xin, yy, n, acc);
    } else if (n == 1) {
        const u32 rows_per_block = 8;
        hip::launch(k.gemv1[t], D((w.nrows + rows_per_block - 1) / rows_per_block), D(256), 0, stream, w.ptr, w.row_bytes, xin, y, ncols,
                    nrows, ncols, nrows, acc);
    } else if (fp8_prefill && (fp8_mask & mm_class) != 0 && w.ty == GgmlType::mxfp4 && w.ref != 0 && k.gemm8[0] != nullptr) {
        // MXFP4 x fp8 (per-token scaled activations, folded block exponents)
        const u32 count = n * w.ncols;
        if (x16_src != xin || x16_count != count || !x16_fp8) {
            hip::launch(x8(k.qact_fp8, k.qact_fp8t), D(n), D(256), 0, stream, xin, x16, sx8, ncols);
            x16_src = xin;
            x16_count = count;
            x16_fp8 = true;
        }
        const std::size_t ci = g8t ? gemm8tCfg(n) : gemm8Cfg(n);
        const GemmCfg c = g8t ? gemm8t_cfgs[ci] : gemm8_cfgs[ci];
        const hip::Function g8 = g8t ? ((out_h16 && k.gemm8th[ci] != nullptr) ? k.gemm8th[ci] : k.gemm8t[ci])
                                     : ((out_h16 && k.gemm8h[ci] != nullptr) ? k.gemm8h[ci] : k.gemm8[ci]);
        hip::launch(g8, D((n + c.bn - 1) / c.bn, (w.nrows + c.bm - 1) / c.bm), D(c.nth), 0, stream, w.ptr, w.row_bytes, w.ref, x16, sx8, y,
                    ncols, nrows, I(n), acc);
    } else {
        const u32 choice = w.tune[tuneBucket(n)];
        const std::size_t gcfg = choice % gemm_cfgs.size();
        const GemmCfg c = gemm_cfgs[gcfg];
        const u32 count = n * w.ncols;
        if (x16_src != xin || x16_count != count || x16_fp8) {
            hip::launch(k.f32_to_f16, D((count / 4 + 255) / 256), D(256), 0, stream, xin, x16, I(count));
            x16_src = xin;
            x16_count = count;
            x16_fp8 = false;
        }
        if (choice >= 2 * gemm_cfgs.size()) {
            // small-batch GEMM straight from the quantized weights (bitwise == gemm_cN)
            const std::size_t si = choice - 2 * gemm_cfgs.size();
            const GemmCfg sc = gemms_cfgs[si];
            const hip::Function sf = out_h16 ? k.gemmsh[si][t] : k.gemms[si][t];
            hip::launch(sf, D((n + sc.bn - 1) / sc.bn, (w.nrows + sc.bm - 1) / sc.bm), D(sc.nth), 0, stream, w.ptr, w.row_bytes, x16, y, ncols,
                        nrows, I(n), acc);
            finish();
            return;
        }
        DevPtr wptr = w.ptr;
        u64 wrb = w.row_bytes;
        GgmlType wty = w.ty;
        const hip::Function hq = (out_h16 && k.gemmch[gcfg] != nullptr) ? k.gemmhqh[t] : k.gemmhq[t];
        if (choice >= gemm_cfgs.size() && w.ty != GgmlType::f16 && gemmh_on && gemmhq_on && hq != nullptr && n >= gemmh_min && w.ncols % 32 == 0) {
            // dequant fused into the fragment-order f16 GEMM (bitwise == dequant_f16 + gemm_cN_f16)
            hip::launch(hq, D((n + 255) / 256, (w.nrows + 127) / 128), D(256), 0, stream, w.ptr, w.row_bytes, x16, y, ncols, nrows, I(n), acc);
            finish();
            return;
        }
        if (choice >= gemm_cfgs.size() && w.ty != GgmlType::f16) {
            const u64 groups = static_cast<u64>(w.ncols / 8) * w.nrows;
            hip::launch(k.dequant_f16[t], D(std::min<u64>((groups + 255) / 256, 65535)), D(256), 0, stream, w.ptr, w.row_bytes, w16, ncols,
                        nrows);
            wptr = w16;
            wrb = static_cast<u64>(w.ncols) * 2;
            wty = GgmlType::f16;
        }
        const bool use_ch = out_h16 && wty == GgmlType::f16 && k.gemmch[gcfg] != nullptr;
        const hip::Function gf = use_ch ? k.gemmch[gcfg] : k.gemmc[gcfg][ti(wty)];
        const hip::Function hf = use_ch ? k.gemmhh_f16 : k.gemmh_f16;
        if (gemmh_on && hf != nullptr && wty == GgmlType::f16 && n >= gemmh_min && w.ncols % 32 == 0 && wrb == static_cast<u64>(w.ncols) * 2) {
            // 128 rows x 256 tokens, bitwise the same as gf
            hip::launch(hf, D((n + 255) / 256, (w.nrows + 127) / 128), D(256), 0, stream, wptr, x16, y, ncols, nrows, I(n), acc);
        } else {
            hip::launch(gf, D((n + c.bn - 1) / c.bn, (w.nrows + c.bm - 1) / c.bm), D(c.nth), 0, stream, wptr, wrb, x16, y, ncols, nrows, I(n), acc);
        }
    }
    finish();
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
    if (n <= gemv_max && k.gemvq[ti(w.ty)] != nullptr && !float_gemv) return ActIn::gemv;
    if (n == 1) return ActIn::f32in;
    if (fp8_prefill && (fp8_mask & cls) != 0 && w.ty == GgmlType::mxfp4 && w.ref != 0 && k.gemm8[0] != nullptr) return ActIn::fp8;
    return ActIn::f16;
}

bool Model::h16Out(const Mat& w, u32 n, std::uint8_t cls) const {
    switch (actIn(w, n, cls)) {
        case ActIn::fp8:
            return k.gemm8h[0] != nullptr;
        case ActIn::f16:
            // every bucket (not just n's), so the choice is the same for a request's
            // solo chunks and a segmented forward of any total size
            for (std::uint8_t choice : w.tune) {
                if (choice >= 2 * gemm_cfgs.size()) {
                    if (k.gemmsh[choice - 2 * gemm_cfgs.size()][ti(w.ty)] == nullptr) return false;
                    continue;
                }
                if (!((choice >= gemm_cfgs.size() || w.ty == GgmlType::f16) && k.gemmch[choice % gemm_cfgs.size()] != nullptr)) return false;
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

// Decode fusions apply to small batches on the int8 GEMV path.
bool Model::fused(u32 n) const { return n <= max_small_batch && !float_gemv && !no_fuse; }

bool Model::fusedDecode() const {
    if (!fused(max_small_batch)) return false;
    for (const Layer& L : layers) {
        if (L.kind != LayerKind::gdn) continue;
        if (!gdnAbFusable(L.gdn) || cfg.d_conv != 4 || cfg.d_state != 128 || cfg.headV() != 128 || cfg.convCh() % 128 != 0) return false;
    }
    return true;
}

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
void Model::attnBlock(const AttnW& a, const KvLayer& lkv, u32 n) {
    const u32 hd = cfg.head_dim;
    const bool batched = bplan != nullptr || mtp_batch;
    const DevPtr kvb = batched ? kvbase_buf : 0;
    const KvArgs kva = kvArgs(lkv, kvb, kv_off);
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
    if (n <= max_small_batch) {
        // flash-decoding; grid.z = query (speculative verify), query t sees
        // positions up to pos_buf[t]
        const u32 n_split = fdSplits(max_ctx);
        const u32 grp_q = cfg.n_head / cfg.n_head_kv;
        if (k.attn_wsplit1 != nullptr && hd == 256 && grp_q <= 16 && (dbg_flags & 2) == 0) {
            // query groups: consecutive rows of one sequence whose split size
            // `per` (a function of the position) matches, <= 16 columns each
            AwGroups groups;
            u32 ng = 0;
            // gfx1151: one query per group (its WMMA P.V is not exact when a masked
            // key carries a real V row), so a grouped verify row could differ
            const u32 max_q = ((dbg_flags & 4) != 0 || arch == hip::Arch::gfx1151) ? 1 : 16 / grp_q;
            const bool known = row_n == n && n > 1;
            u32 r = 0;
            while (r < n) {
                u32 c = 1;
                if (known) {
                    const i32 per0 = awPer(row_pos[r] + 1, n_split);
                    while (r + c < n && c < max_q && row_base[r + c] == row_base[r] && row_pos[r + c] == row_pos[r] + static_cast<i32>(c) &&
                           awPer(row_pos[r + c] + 1, n_split) == per0)
                        c += 1;
                }
                groups.first[ng] = static_cast<i32>(r);
                groups.count[ng] = static_cast<i32>(c);
                ng += 1;
                r += c;
            }
            hip::launch(k.attn_wsplit1, D(cfg.n_head_kv, n_split, ng), D(128), 0, stream, qf, kva, part_ml, part_acc, I(cfg.n_head),
                        I(cfg.n_head_kv), I(2 * hd), pos_buf, scale, gate, groups);
        } else {
            hip::launch(k.attn_split, D(cfg.n_head_kv, n_split, n), D(256), 0, stream, qf, kva, part_ml, part_acc, I(cfg.n_head), I(cfg.n_head_kv),
                        I(hd), I(2 * hd), pos_buf, scale, i32(1), gate);
        }
        const bool combine_q8 = k.attn_combine_q8 != nullptr && fused(n) && hd == 256 && k.gemvq[ti(a.o.ty)] != nullptr && n <= gemv_max;
        if (combine_q8) {
            // the o-projection's int8 input comes out of the combine
            hip::launch(k.attn_combine_q8, D(cfg.n_head, n), D(256), 0, stream, part_ml, part_acc, qf, attn_out, I(cfg.n_head), I(hd), I(2 * hd),
                        I(n_split), xq, xd);
            mark(OpClass::attn);
            xq_src = attn_out;
            xq_n = n;
            x16_src = 0;
            matmul(a.o, attn_out, x, n, true);
            return;
        }
        hip::launch(k.attn_combine, D(cfg.n_head, n), D(256), 0, stream, part_ml, part_acc, qf, attn_out, I(cfg.n_head), I(hd), I(2 * hd),
                    I(n_split));
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

// WMMA prefill attention of n queries at positions pos0 .. (keys 0 .. pos0 +
// n - 1), split over head ranges so one launch does at most ~4096 x 128k
// query-key pairs x all heads. Same results as one launch.
// f16 / q8v KV: attn_kg (NP query heads of one KV head per block; bit-identical
// to attn_kx) when NP divides the GQA group, the head ranges stay whole groups
// of NP and the softmax scale is a power of two (it relies on exact scaling).
void Model::prefillAttn(DevPtr q, const KvArgs& kva, DevPtr out, DevPtr pos, u32 n, u32 pos0, float scale) {
    const u64 work = static_cast<u64>(n) * (static_cast<u64>(pos0) + n);
    const u64 budget = 4096ull * 131072ull;
    u32 parts = static_cast<u32>(std::min<u64>(cfg.n_head, (work + budget - 1) / budget));
    while (parts > 1 && cfg.n_head % parts != 0) parts += 1;  // whole head ranges
    const u32 hp = cfg.n_head / parts;
    const u32 grp = cfg.n_head / cfg.n_head_kv;
    hip::Function kg = nullptr;
    u32 np = 0;
    int sexp = 0;
    const bool pow2 = std::frexp(scale, &sexp) == 0.5f;
    if (attn_kg_on && cfg.head_dim == 256 && pow2) {
        for (const auto& [f, c] : {std::pair{k.attn_kg6, 6u}, std::pair{k.attn_kg4, 4u}, std::pair{k.attn_kg2, 2u}})
            if (f != nullptr && grp % c == 0 && hp % c == 0) {
                kg = f;
                np = c;
                break;
            }
    }
    const bool kx = attn_kx_on && k.attn_kx != nullptr && cfg.head_dim == 256;
    for (u32 p = 0; p < parts; ++p) {
        if (kg != nullptr)
            hip::launch(kg, D((n + 15) / 16, hp / np), D(64 * np), 0, stream, q, kva, out, I(cfg.n_head), I(cfg.n_head_kv),
                        I(2 * cfg.head_dim), pos, I(n), scale, I(p * hp));
        else
            hip::launch(kx ? k.attn_kx : k.attn_prefill_wmma, D((n + 127) / 128, hp), D(kx ? 512 : 256), 0, stream, q, kva, out,
                        I(cfg.n_head), I(cfg.n_head_kv), I(2 * cfg.head_dim), pos, I(n), scale, I(p * hp));
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
    if (fused(n)) {
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
    const bool small = n <= max_small_batch;
    if (small)
        rmsnormQ8(x, post_norm, h, n);
    else
        rmsnorm(x, post_norm, h, E, n, E, E);
    // router logits
    if (small) {
        hip::launch(k.moe_logits_f32, D((R + 7) / 8), D(256), 0, stream, mo.router.ptr, h, moe_logits, I(R), I(E), I(n), gate);
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
    if (small) {
        // experts: gate + up from the int8 h that rmsnormQ8 left in xq
        const u64 gu_rb = mo.gate.row_bytes;
        if (mo.gate.ty == mo.up.ty) {
            hip::launch(moeGu(mo.gate.ty), D((2 * F + 7) / 8, n * K), D(256), 0, stream, mo.gate.ptr, mo.up.ptr, gu_rb, I(F), I(2 * F), xq, xd, moe_ids,
                        moe_g, moe_u, I(E), I(K), gate);
        } else {
            hip::launch(moeGu(mo.gate.ty), D((F + 7) / 8, n * K), D(256), 0, stream, mo.gate.ptr, mo.gate.ptr, gu_rb, I(F), I(F), xq, xd, moe_ids, moe_g,
                        moe_g, I(E), I(K), gate);
            hip::launch(moeGu(mo.up.ty), D((F + 7) / 8, n * K), D(256), 0, stream, mo.up.ptr, mo.up.ptr, mo.up.row_bytes, I(F), I(F), xq, xd,
                        moe_ids, moe_u, moe_u, I(E), I(K), gate);
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
        hip::launch(moeDown(mo.down.ty), D((E + 7) / 8, n), D(256), 0, stream, mo.down.ptr, mo.down.row_bytes, I(E), moe_xq, moe_xd, moe_ids,
                    moe_w, moe_sg, ysh, x, I(F), I(K), gate);
        if (prof) prof->matmul_bytes += mo.down.row_bytes * E * n * K;
        xq_src = 0;
        x16_src = 0;
        mark(OpClass::matmul);
        return;
    }
    // prefill: group (token, slot) pairs by expert, grouped GEMMs, combine
    const u32 pairs = n * K;
    // token tile: 32 when experts average under ~48 tokens (short prompts), else 64
    const u32 bn = (moe_bn_force == 32 || moe_bn_force == 64) ? moe_bn_force : moeTile(n);
    const auto& gm = bn == 32 ? k.gemm_moe32 : k.gemm_moe;
    hip::launch(k.moe_route, D(1), D(1024), 0, stream, moe_ids, I(pairs), I(R), I(bn), moe_perm, moe_inv, moe_tiles, moe_ntiles);
    const u32 max_tiles = (pairs + bn - 1) / bn + R;
    if (moeFp8(mo)) {
        // MXFP4 experts x fp8 activations (per-pos scale, folded E8M0 as gemm8): gate + up in
        // one launch with f16 outputs, silu(g) * u -> fp8 rows, down with f32 output
        hip::launch(k.moe_gather_fp8, D(pairs), D(256), 0, stream, h, moe_perm, moe_x16, moe_sx, I(E), I(K));
        mark(OpClass::misc);
        const u32 nby = (F + moe_bm - 1) / moe_bm;
        hip::launch(bn == 32 ? k.gemm8_moe32h : k.gemm8_moeh, D(max_tiles, 2 * nby), D(256), 0, stream, mo.gate.ptr, mo.up.ptr, mo.gate.row_bytes,
                    mo.gate.ref, mo.up.ref, I(F), moe_x16, moe_sx, moe_yg, moe_yu, I(E), moe_tiles, moe_ntiles);
        mark(OpClass::matmul);
        hip::launch(k.silu_mul_x8h, D(pairs), D(256), 0, stream, moe_yg, moe_yu, moe_x16, moe_sx, I(F));
        mark(OpClass::misc);
        hip::launch(bn == 32 ? k.gemm8_moe32 : k.gemm8_moe, D(max_tiles, (E + moe_bm - 1) / moe_bm), D(256), 0, stream, mo.down.ptr, DevPtr{0},
                    mo.down.row_bytes, mo.down.ref, DevPtr{0}, I(E), moe_x16, moe_sx, moe_yd, DevPtr{0}, I(F), moe_tiles, moe_ntiles);
        mark(OpClass::matmul);
    } else {
        hip::launch(k.moe_gather_f16, D(pairs), D(256), 0, stream, h, moe_perm, moe_x16, I(E), I(K));
        mark(OpClass::misc);
        hip::launch(gm[ti(mo.gate.ty)], D(max_tiles, (F + moe_bm - 1) / moe_bm), D(256), 0, stream, mo.gate.ptr, mo.gate.row_bytes, I(F), moe_x16,
                    moe_yg, I(E), moe_tiles, moe_ntiles);
        hip::launch(gm[ti(mo.up.ty)], D(max_tiles, (F + moe_bm - 1) / moe_bm), D(256), 0, stream, mo.up.ptr, mo.up.row_bytes, I(F), moe_x16, moe_yu,
                    I(E), moe_tiles, moe_ntiles);
        hip::launch(k.moe_act_f16, D((pairs * F + 255) / 256), D(256), 0, stream, moe_yg, moe_yu, moe_x16, I(pairs * F));
        hip::launch(gm[ti(mo.down.ty)], D(max_tiles, (E + moe_bm - 1) / moe_bm), D(256), 0, stream, mo.down.ptr, mo.down.row_bytes, I(E), moe_x16,
                    moe_yd, I(F), moe_tiles, moe_ntiles);
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

u32 Model::moeTile(u32 n) const { return n * cfg.n_expert_used < 48 * cfg.n_expert ? 32 : moe_bn; }

// MXFP4 decode experts: the whole-block kernels when present (WHIRL_MOE_MXW=0 -> generic).
hip::Function Model::moeGu(GgmlType ty) const {
    return ty == GgmlType::mxfp4 && moe_mxw && k.moe_gu_mxw != nullptr ? k.moe_gu_mxw : k.moe_gu[ti(ty)];
}
hip::Function Model::moeDown(GgmlType ty) const {
    return ty == GgmlType::mxfp4 && moe_mxw && k.moe_down_mxw != nullptr ? k.moe_down_mxw : k.moe_down[ti(ty)];
}

// Prefill experts on the MXFP4 x fp8 grouped GEMM: all three expert tensors MXFP4 with
// load-time row exponents, fp8 prefill on (speed mode), kernels present.
bool Model::moeFp8(const MoeW& mo) const {
    return moe_fp8 && fp8_prefill && mo.gate.ty == GgmlType::mxfp4 && mo.up.ty == GgmlType::mxfp4 && mo.down.ty == GgmlType::mxfp4 &&
           mo.gate.ref != 0 && mo.up.ref != 0 && mo.down.ref != 0 && mo.gate.row_bytes == mo.up.row_bytes && moe_sx != 0 &&
           k.gemm8_moe != nullptr && k.gemm8_moe32 != nullptr && k.gemm8_moeh != nullptr && k.gemm8_moe32h != nullptr &&
           k.moe_gather_fp8 != nullptr && k.silu_mul_x8h != nullptr;
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
    attnBlock(mw.attn, mtpKv(), n);
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
    // draft head over the first draft_vocab rows only
    Mat head = draft_head ? *draft_head : output;
    if (draft_vocab > 0 && draft_vocab < head.nrows) head.nrows = draft_vocab;
    draftHeadMatmul(head, 1);
    hip::launch(k.draft_pick, D(1), D(1024), 0, stream, logits, I(head.nrows), ctlSlot(ctl_drafts), ctlSlot(ctl_probs), ctlSlot(ctl_nd),
                I(draft_slot), I(draft_n_min), draft_p_min);
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
    Idx16 dst;
    u32 n = 0;
    u32 snaps = 0;
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const VSeg& sg = segs[kk];
        if (sg.pos + sg.nd + 1 > max_ctx) throw ModelError("ContextTooLong");
        if (n + sg.nd + 1 > max_small_batch) throw ModelError("BatchTooLarge");
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
    struct Reset {
        Model& m;
        ~Reset() {
            m.keep_hidden = false;
            m.all_logits = false;
            m.bplan = nullptr;
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
    Idx16 last;
    Idx16 area;
    u32 n = 0;
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const MSeg& sg = segs[kk];
        const Seq& sq = seqs[sg.seq];
        if (r == 0) {
            if (n + sg.pend.size() > max_small_batch) throw ModelError("BatchTooLarge");
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
    struct BatchReset {
        bool& b;
        ~BatchReset() { b = false; }
    } batch_reset{mtp_batch};
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
    attnBlock(mw.attn, mtpKv(), n);
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
    if (draft_vocab > 0 && draft_vocab < head.nrows) head.nrows = draft_vocab;
    draftHeadMatmul(head, ns);
    hip::launch(k.draft_pick_rows, D(ns), D(1024), 0, stream, logits, I(head.nrows), out_tok, area, I(r), I(draft_n_min), draft_p_min);
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

void Model::verifyEnqueue(u32 next, u32 n_draft, u32 pos0) { verifyEnqueueEx(next, n_draft, std::nullopt, pos0); }

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

Model::Draft Model::readDraft(u32 r) {
    i32 buf[ctl_words];
    hip::download(buf, out_tok, sizeof buf);
    float p;
    std::memcpy(&p, &buf[ctl_probs + r], 4);
    return {static_cast<u32>(buf[ctl_drafts + r]), p};
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
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const Layer& L = layers[i];
        if (fused(n)) {
            rmsnormQ8(x, L.attn_norm, h, n);
        } else if (L.kind == LayerKind::attn) {
            const Mat c[3] = {L.attn.q, L.attn.k, L.attn.v};
            rmsnormIn(x, L.attn_norm, h, n, c, 1);
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
                if (abconv != nullptr) {
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
                matmul(g.beta, h, beta, n, false);
                matmul(g.alpha, h, alpha, n, false);
                const u32 nh = n * cfg.n_v_heads;
                bool conv_l2 = false;
                hip::launch(k.gdn_gates, D((nh + 63) / 64), D(64), 0, stream, beta, alpha, g.dt, g.a, I(nh), I(cfg.n_v_heads));
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
    if (keep_hidden) rmsnorm(x, output_norm, hn, E, n, E, E);
    if (!psegs.empty()) return;  // prefillSegs: per-segment logits from x
    if (keep_hidden && all_logits && n <= max_small_batch) {
        head_phase = true;
        matmul(output, hn, logits, n, false);
        head_phase = false;
        return;
    }
    const DevPtr last = x + static_cast<u64>(n - 1) * E * 4;
    if (fused(1))
        rmsnormQ8(last, output_norm, h, 1);
    else
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

void Model::prefillWithMtpFrom(std::span<const u32> tokens, u32 pos0, DevPtr prev_hidden) {
    std::size_t off = 0;
    while (off < tokens.size()) {
        const std::size_t n = std::min<std::size_t>(tokens.size() - off, max_batch);
        prefillMtpChunk(tokens, pos0, off, n, prev_hidden);
        off += n;
    }
}

// One chunk of an MTP prefill: tokens[off .. off + n] at positions pos0 + off
// .., then the MTP rows pairing each of their trunk hiddens with the following
// token (the final chunk's last hidden goes to mtp_h).
void Model::prefillMtpChunk(std::span<const u32> tokens, u32 pos0, std::size_t off, std::size_t n, std::optional<DevPtr> prev_hidden) {
    const u64 E = cfg.n_embd;
    if (off == 0 && prev_hidden) {
        hip::copyAsync(mtp_h, *prev_hidden, E * 4, stream);
        (void)mtpForward(mtp_h, tokens.subspan(0, 1), pos0, false);
    }
    const u32 p = pos0 + static_cast<u32>(off);
    keep_hidden = true;
    try {
        forward(tokens.subspan(off, n), p);
    } catch (...) {
        keep_hidden = false;
        throw;
    }
    keep_hidden = false;
    const bool is_last = off + n == tokens.size();
    const std::size_t nm = is_last ? n - 1 : n;
    if (nm > 0) (void)mtpForward(hn, tokens.subspan(off + 1, nm), p + 1, false);
    if (is_last) hip::copyAsync(mtp_h, hn + (n - 1) * E * 4, E * 4, stream);
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
    // MTP rows pairing a cached prefix's last hidden with tokens[0] (solo order)
    if (with_mtp)
        for (const PSeg& sg : segs)
            if (sg.off == 0 && sg.prev_hidden) {
                selectSeq(sg.seq);
                hip::copyAsync(mtp_h, *sg.prev_hidden, E * 4, stream);
                (void)mtpForward(mtp_h, sg.tokens.subspan(0, 1), sg.pos0, false);
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
        if (fused(1))
            rmsnormQ8(last, output_norm, h, 1);
        else
            rmsnorm(last, output_norm, h, static_cast<u32>(E), 1, static_cast<u32>(E), static_cast<u32>(E));
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
        const bool is_last = sg.off + sg.n == sg.tokens.size();
        const std::size_t nm = is_last ? sg.n - 1 : sg.n;
        if (nm > 0) (void)mtpForward(hid, sg.tokens.subspan(sg.off + 1, nm), p + 1, false);
        if (is_last) hip::copyAsync(mtp_h, hid + (sg.n - 1) * E * 4, E * 4, stream);
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
void Model::readHidden(std::span<float> dst) { hip::download(dst.data(), x, dst.size() * 4); }

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
    rpos_buf = hip::malloc(static_cast<std::size_t>(max_batch) * 3 * 4);
}

void Model::visRelease() {
    if (rpos_buf == 0 || visAny()) return;
    try {
        hip::sync();
    } catch (...) {
    }
    hip::free(rpos_buf);
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
