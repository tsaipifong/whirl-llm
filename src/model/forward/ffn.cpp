// Model forward, FFN block (arch025-F2, moved from src/model/forward.cpp): the dense SwiGLU FFN (ffnBlock) and
// the MoE FFN (moeBlock: router, top-K experts, shared expert).
// SPDX-License-Identifier: Apache-2.0

#include "../forward_internal.h"

namespace whirl::qwen35 {

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
    std::optional<ops::ActIn> down_in;
    if (!fused(n) && cfg.n_ff % 4 == 0 && cfg.n_ff <= 1024 * 24) down_in = actCommon(std::span<const Mat>(&down, 1), n, 4);
    // f16 gate / up outputs straight into silu -> fp8 (MXFP4 speed mode)
    const ops::ActIn gu_in = actCommon(gu, n, 2).value_or(ops::ActIn::gemv);
    const bool h16 = ffn_h16 && down_in.has_value() && cfg.n_ff % 8 == 0 && h16Out(gate_w, n, 2) && h16Out(up, n, 2) &&
                     ((gu_in == ops::ActIn::fp8 && *down_in == ops::ActIn::fp8 && k.silu_mul_x8h != nullptr) ||
                      (gu_in == ops::ActIn::f16 && *down_in == ops::ActIn::f16 && k.silu_mul_x16h != nullptr));
    out_h16 = h16;
    {
        const DevPtr ys[2] = {ffn_g, ffn_u};
        matmulGroup(gu, h, ys, n);
    }
    out_h16 = false;
    mm_class = 4;
    hip::Function silu_x = nullptr;
    if (h16)
        silu_x = *down_in == ops::ActIn::fp8 ? x8(k.silu_mul_x8h, k.silu_mul_x8ht) : k.silu_mul_x16h;
    else if (down_in)
        silu_x = *down_in == ops::ActIn::fp8 ? x8(k.silu_mul_x8, k.silu_mul_x8t) : k.silu_mul_x16;
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
        x16_fp8 = *down_in == ops::ActIn::fp8;
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

}  // namespace whirl::qwen35
