// MoE expert kernel choice (ARCH-1 K-4), see moe.h. Moved from src/model/forward.cpp (Model::moeBlock's kernel
// selection, moeGu / moeDown / moeGuFused / moeGuR / moeFp8 / moeTile); same choices and grids, pinned by the K-0
// launch tables.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/moe.h"
#include "whirl/ops/matmul.h"  // preciseRows

namespace whirl::ops {

namespace {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr std::size_t ti(GgmlType t) { return static_cast<std::size_t>(t); }
constexpr Dim3 D(u64 x, u64 y = 1, u64 z = 1) { return Dim3{static_cast<unsigned>(x), static_cast<unsigned>(y), static_cast<unsigned>(z)}; }
constexpr u32 moe_bm = 128;
constexpr u32 moe_bn = 64;

// MXFP4 decode experts: the whole-block kernels when present (WHIRL_MOE_MXW=0 -> generic)
Function moeGu(GgmlType ty, bool mxw, const kernels::MoeKernels& k) {
    return ty == GgmlType::mxfp4 && mxw && k.moe_gu_mxw != nullptr ? k.moe_gu_mxw : k.moe_gu[ti(ty)];
}
Function moeDown(GgmlType ty, bool mxw, const kernels::MoeKernels& k) {
    return ty == GgmlType::mxfp4 && mxw && k.moe_down_mxw != nullptr ? k.moe_down_mxw : k.moe_down[ti(ty)];
}
// Prefill experts on the fused gate + up + SwiGLU grouped GEMM and the 32-token row-block-fast down GEMM (gfx1151
// code object; gate and up of one type and row size).
bool fusedGu(const MoeProblem& p, const kernels::MoeKernels& k) {
    return k.moe_tiles != nullptr && k.gemm_moegu[ti(p.gate_ty)] != nullptr && k.gemm_moe32r[ti(p.down_ty)] != nullptr && p.gate_ty == p.up_ty &&
           p.gate_rb == p.up_rb && p.tiles64_buf;
}
// Prefill experts on the f16-activation MXFP4 grouped GEMMs gemmr_moe* (gfx1201; WHIRL_MOE_RBF=0 -> gemm_moe +
// moe_act_f16): all three expert tensors MXFP4, gate and up of one row size, E and F multiples of the k step.
bool gemmrGu(const MoeProblem& p, const MoeFlags& f, const kernels::MoeKernels& k) {
    return f.moe_rbf && p.gate_ty == GgmlType::mxfp4 && p.up_ty == GgmlType::mxfp4 && p.down_ty == GgmlType::mxfp4 && p.gate_rb == p.up_rb &&
           k.gemmr_moegu != nullptr && k.gemmr_moegu32 != nullptr && k.gemmr_moedn != nullptr && k.gemmr_moedn32 != nullptr &&
           kernels::gemmrMoeShapeOk(p.gate_ncols, p.down_ncols);
}
// Prefill experts on the MXFP4 x fp8 grouped GEMM: all three expert tensors MXFP4 with load-time row exponents,
// fp8 prefill on (speed mode), kernels present.
bool fp8Gu(const MoeProblem& p, const MoeFlags& f, const kernels::MoeKernels& k, Function silu_mul_x8h) {
    return f.moe_fp8 && p.gate_ty == GgmlType::mxfp4 && p.up_ty == GgmlType::mxfp4 && p.down_ty == GgmlType::mxfp4 && p.refs && p.gate_rb == p.up_rb &&
           p.sx_buf && k.gemm8_moe != nullptr && k.gemm8_moe32 != nullptr && k.gemm8_moeh != nullptr && k.gemm8_moe32h != nullptr &&
           k.moe_gather_fp8 != nullptr && silu_mul_x8h != nullptr;
}
}  // namespace

u32 moeTile(u32 n, u32 K, u32 R) { return n * K < 48 * R ? 32 : moe_bn; }

MoePlan chooseMoe(const MoeProblem& p, const MoeFlags& f, const kernels::MoeKernels& k, Function silu_mul_x8h) {
    const u32 n = p.n, E = p.E, K = p.K, F = p.F, R = p.R;
    MoePlan pl;
    // precise mode: every verify / decode batch (up to max_verify_rows rows) takes the decode experts, so a
    // generated token's MoE output never depends on whether its batch had more than 16 rows. balance / fast: a
    // wide verify / MTP batch (small_max raised to max_verify_rows) takes the int8 decode experts too.
    pl.small = n <= f.small_max || preciseRows(f.prec_dec, n, f.max_verify_rows);
    pl.small_q8 = pl.small && !f.prec_dec;
    pl.logits_grid = D((R + 7) / 8);
    if (pl.small) {
        const bool same = p.gate_ty == p.up_ty;
        pl.gu_split = !same;
        if (f.prec_dec) {
            // precise decode experts: f32 h, f32 accumulation, every pair on its own waves
            pl.bits = MoeBits::moe_f32_row;
            pl.gu = k.moe_gu_f[ti(p.gate_ty)];
            pl.gu_up = same ? nullptr : k.moe_gu_f[ti(p.up_ty)];
            pl.down = k.moe_down_f[ti(p.down_ty)];
        } else {
            pl.bits = MoeBits::moe_q8_row;
            pl.gu = moeGu(p.gate_ty, f.moe_mxw, k);
            pl.gu_up = same ? nullptr : moeGu(p.up_ty, f.moe_mxw, k);
            pl.down = moeDown(p.down_ty, f.moe_mxw, k);
        }
        pl.gu_grid = same ? D((2 * F + 7) / 8, n * K) : D((F + 7) / 8, n * K);
        pl.down_grid = D((E + 7) / 8, n);
        return pl;
    }
    // prefill: group (token, slot) pairs by expert, grouped GEMMs, combine
    const u32 pairs = n * K;
    const bool fp8 = fp8Gu(p, f, k, silu_mul_x8h), fused = fusedGu(p, k);
    pl.bn = fused && !fp8 ? 32 : (f.moe_bn_force == 32 || f.moe_bn_force == 64) ? f.moe_bn_force : moeTile(n, K, R);
    pl.max_tiles = (pairs + pl.bn - 1) / pl.bn + R;
    const u32 bn = pl.bn, max_tiles = pl.max_tiles;
    if (fp8) {
        pl.kind = MoePrefillKind::fp8;
        pl.bits = MoeBits::moe_fp8_gemm;
        const u32 nby = (F + moe_bm - 1) / moe_bm, nbyd = (E + moe_bm - 1) / moe_bm;
        pl.rbf = f.moe_rbf && k.gemm8r_moe != nullptr && k.gemm8r_moe32 != nullptr && k.gemm8r_moeh != nullptr && k.gemm8r_moe32h != nullptr;
        pl.fgu = pl.rbf ? (bn == 32 ? k.gemm8r_moe32h : k.gemm8r_moeh) : (bn == 32 ? k.gemm8_moe32h : k.gemm8_moeh);
        pl.fdn = pl.rbf ? (bn == 32 ? k.gemm8r_moe32 : k.gemm8r_moe) : (bn == 32 ? k.gemm8_moe32 : k.gemm8_moe);
        pl.gu_pgrid = pl.rbf ? D(max_tiles * 2 * nby) : D(max_tiles, 2 * nby);
        pl.dn_pgrid = pl.rbf ? D(max_tiles * nbyd) : D(max_tiles, nbyd);
    } else if (fused) {
        // Radeon 8060S: gate + up + SwiGLU in one grouped GEMM over 64-token tiles, down over 32-token tiles; the
        // same perm positions for both tile lists, every row bit-identical to the generic path
        pl.kind = MoePrefillKind::fused_gu;
        pl.max_tiles64 = (pairs + 63) / 64 + R;
        pl.fgu = k.gemm_moegu[ti(p.gate_ty)];
        pl.fdn = k.gemm_moe32r[ti(p.down_ty)];
        pl.gu_pgrid = D(pl.max_tiles64 * ((F + 63) / 64));
        pl.dn_pgrid = D(max_tiles * ((E + moe_bm - 1) / moe_bm));
    } else if (gemmrGu(p, f, k)) {
        // MXFP4 experts, f16 activations (precise): gate + up + SwiGLU in one grouped GEMM, then down; row block
        // fastest grid, empty 16-token fragments skipped - every value as the generic path
        pl.kind = MoePrefillKind::gemmr;
        pl.fgu = bn == 32 ? k.gemmr_moegu32 : k.gemmr_moegu;
        pl.fdn = bn == 32 ? k.gemmr_moedn32 : k.gemmr_moedn;
        pl.gu_pgrid = D(max_tiles * ((F + 63) / 64));
        pl.dn_pgrid = D(max_tiles * ((E + 127) / 128));
    } else {
        pl.kind = MoePrefillKind::generic;
        const auto& gm = bn == 32 ? k.gemm_moe32 : k.gemm_moe;
        pl.g_gate = gm[ti(p.gate_ty)];
        pl.g_up = gm[ti(p.up_ty)];
        pl.g_down = gm[ti(p.down_ty)];
        pl.gu_pgrid = D(max_tiles, (F + moe_bm - 1) / moe_bm);
        pl.dn_pgrid = D(max_tiles, (E + moe_bm - 1) / moe_bm);
    }
    return pl;
}

}  // namespace whirl::ops
