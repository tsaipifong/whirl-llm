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
// feat026-moep: the multi-fragment expert GEMMs (gfx1151, MXFP4 experts) when both variants are selected and present
bool moepOk(const MoeProblem& p, const MoeFlags& f, const kernels::MoeKernels& k) {
    if (f.moep_gu < 0 || f.moep_dn < 0 || f.moep_gu >= kernels::kMoepGuN || f.moep_dn >= kernels::kMoepDnN) return false;
    if (k.moep_gu[f.moep_gu] == nullptr || k.moep_dn[f.moep_dn] == nullptr || k.moe_tiles == nullptr) return false;
    // the gate + up list goes into the 64-token tile buffer: its bn must be >= 64; the down list into moe_route's
    // (sized for bn 32)
    return p.gate_ty == GgmlType::mxfp4 && p.up_ty == GgmlType::mxfp4 && p.down_ty == GgmlType::mxfp4 && p.gate_rb == p.up_rb && p.tiles64_buf &&
           kernels::kMoepGu[f.moep_gu].bn >= 64 && kernels::kMoepDn[f.moep_dn].bn >= 32;
}
// v0252-f5 (moeskip, feat027-moeskip-r1 4b6474b as NEW entry points; DISCOVERY v4 rule 3.0-1): the block drops slots only
// when the flag is on (fast; never with prec_dec, whose precise decode experts have no -1 twin), the experts are MXFP4 (the
// decode twins are MXFP4 only; the item needs MXFP4 experts) and EVERY kernel of the chosen path has its skip entry point:
// routing (moe_topk_skip, or moe_skip_apply after the fused router), decode gate + up / down (the whole-block twins of
// moe_gu_mxfp4w / moe_down_mxfp4w, or the generic MXFP4 twins; gate and up in one launch), prefill grouping / combine
// (moe_route_skip; moe_tiles_skip for the 64-token lists of fused_gu / moep; moe_combine_skip / moe_combine4_skip; the fused
// down + combine and the f16-output combines have none, so those plans keep moe_topk). All or nothing: a partial set would
// hand id -1 to a kernel without the contract (the feat027-moeskip-r1 moe_f.hip out-of-bounds read).
bool skipOk(const MoeProblem& p, const MoeFlags& f, const kernels::MoeKernels& k, const MoePlan& pl, Function* gu, Function* down) {
    if (f.moe_skip_thr <= 0.f || f.prec_dec) return false;
    if (p.gate_ty != GgmlType::mxfp4 || p.up_ty != GgmlType::mxfp4 || p.down_ty != GgmlType::mxfp4) return false;
    if (pl.router != nullptr ? k.moe_skip_apply == nullptr : k.moe_topk_skip == nullptr) return false;
    if (pl.small) {
        if (!pl.small_q8 || pl.gu_split) return false;
        const std::size_t mx = ti(GgmlType::mxfp4);
        if (pl.gu == k.moe_gu_mxw && pl.down == k.moe_down_mxw && k.moe_gu_mxw != nullptr) {
            *gu = k.moe_gu_skip_mxw;
            *down = k.moe_down_skip_mxw;
        } else if (pl.gu == k.moe_gu[mx] && pl.down == k.moe_down[mx] && k.moe_gu[mx] != nullptr) {
            *gu = k.moe_gu_skip_mx;
            *down = k.moe_down_skip_mx;
        } else {
            return false;
        }
        return *gu != nullptr && *down != nullptr;
    }
    if (k.moe_route_skip == nullptr || pl.comb_fused || pl.ydh) return false;
    if ((pl.kind == MoePrefillKind::fused_gu || pl.kind == MoePrefillKind::moep) && k.moe_tiles_skip == nullptr) return false;
    if (pl.comb_atomic) return true;  // the down GEMM adds w * y over the tiles' (kept) positions; moe_combine4 K = 0 adds the shared expert
    return pl.comb4 ? k.moe_combine4_skip != nullptr : k.moe_combine_skip != nullptr;
}
// v0251-a7pol (B26-43): the moep flags only from kMoepMinTokensPerExpert tokens per expert (moepWide, moe.h); below it
// the shipped 0.2.5 plan (no moep GEMMs, moe_combine). The one place the threshold applies.
MoeFlags moepFloor(const MoeProblem& p, const MoeFlags& f) {
    MoeFlags g = f;
    if (!moepWide(p.n, p.K, p.R)) {
        g.moep_gu = g.moep_dn = -1;
        g.moep_comb = 0;
    }
    return g;
}
}  // namespace

u32 moeTile(u32 n, u32 K, u32 R) { return n * K < 48 * R ? 32 : moe_bn; }

MoePlan chooseMoe(const MoeProblem& p, const MoeFlags& f_in, const kernels::MoeKernels& k, Function silu_mul_x8h) {
    const MoeFlags f = moepFloor(p, f_in);
    const u32 n = p.n, E = p.E, K = p.K, F = p.F, R = p.R;
    MoePlan pl;
    // precise mode: every verify / decode batch (up to max_verify_rows rows) takes the decode experts, so a
    // generated token's MoE output never depends on whether its batch had more than 16 rows. balance / fast: a
    // wide verify / MTP batch (small_max raised to max_verify_rows) takes the int8 decode experts too.
    // v0252-precfix: the segmented-prefill join bound moeSegJoinMin (moe.h) follows this rule: change both together.
    pl.small = n <= f.small_max || preciseRows(f.prec_dec, n, f.max_verify_rows);
    pl.small_q8 = pl.small && !f.prec_dec;
    pl.logits_grid = D((R + 7) / 8);
    // v0251-a11port (feat026-router): one small piece (n <= max_small_batch) of a supported shape -> the fused router
    // (the screened host condition, f06b245 forward/ffn.cpp); ceil(E / 128) picks the entry, else the two-kernel path
    if (f.moe_rfuse && pl.small && n <= f.max_small_batch && R <= 256 && K <= 32 && E % 4 == 0 && p.rcnt_buf) {
        const u32 it = (E + 127) / 128;
        pl.router = it == 16 ? k.moe_router_fused_i16 : it == 24 ? k.moe_router_fused_i24 : nullptr;
    }
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
        // v0252-f5 moeskip: the same launches with the skip twins (same arguments and grids)
        if (Function g = nullptr, d = nullptr; skipOk(p, f, k, pl, &g, &d)) {
            pl.skip = true;
            pl.gu = g;
            pl.down = d;
        }
        return pl;
    }
    // prefill: group (token, slot) pairs by expert, grouped GEMMs, combine
    const u32 pairs = n * K;
    const bool fp8 = fp8Gu(p, f, k, silu_mul_x8h), fused = fusedGu(p, k);
    const bool moep = !fp8 && moepOk(p, f, k);
    pl.bn = moep ? static_cast<u32>(kernels::kMoepDn[f.moep_dn].bn) : fused && !fp8 ? 32 : (f.moe_bn_force == 32 || f.moe_bn_force == 64) ? f.moe_bn_force : moeTile(n, K, R);
    pl.max_tiles = (pairs + pl.bn - 1) / pl.bn + R;
    const u32 bn = pl.bn, max_tiles = pl.max_tiles;
    // combine: moe_combine4 (same bits) when selected and present
    pl.comb4 = f.moep_comb == 1 && k.moe_combine4 != nullptr && E % 4 == 0;
    pl.comb_grid = pl.comb4 ? D((E + 1023) / 1024, n) : D((E + 255) / 256, n);
    if (moep) {
        // feat026-moep: several 16-token fragments of one expert per block share every decoded A fragment; every
        // stored value as gemm_moegu / gemm_moe32r (and moe_combine when fused into the down epilogue)
        const kernels::MoepGuVariant& vg = kernels::kMoepGu[f.moep_gu];
        const kernels::MoepDnVariant& vd = kernels::kMoepDn[f.moep_dn];
        pl.kind = MoePrefillKind::moep;
        pl.bits = MoeBits::moe_f16_gemm;
        pl.gu_bn = static_cast<u32>(vg.bn);
        pl.max_tiles64 = (pairs + pl.gu_bn - 1) / pl.gu_bn + R;
        const int dv = f.moep_dn;
        // not bit-identical: never in precise (owner precision policy). v0251-a7pol (B26-41): by the mode's class
        // (moep_comb_ok) as well as prec_dec (WHIRL_Q8DEC=1 clears prec_dec in precise): 3 / 4 balance + fast, 5 fast only
        const bool lossy_ok = !f.prec_dec && f.moep_comb_ok != MoeFlags::MoepComb::bitwise;
        const bool atomic_ok = !f.prec_dec && f.moep_comb_ok == MoeFlags::MoepComb::atomic;
        const bool cnt_ok = p.cnt_buf && static_cast<u32>(vd.bm) >= kernels::kMoepMinBm;
        pl.fgu = k.moep_gu[f.moep_gu];
        pl.fdn = k.moep_dn[dv];
        if (f.moep_comb == 2 && k.moep_dc[dv] != nullptr && cnt_ok) {
            pl.comb_fused = true;
            pl.fdn = k.moep_dc[dv];
        } else if (lossy_ok && f.moep_comb == 3 && k.moep_dh[dv] != nullptr && k.moe_combine4h != nullptr && E % 4 == 0) {
            pl.ydh = true;
            pl.fdn = k.moep_dh[dv];
        } else if (lossy_ok && f.moep_comb == 4 && k.moep_dhc[dv] != nullptr && cnt_ok) {
            pl.ydh = pl.comb_fused = true;
            pl.fdn = k.moep_dhc[dv];
        } else if (atomic_ok && f.moep_comb == 5 && k.moep_da[dv] != nullptr && k.moe_combine4 != nullptr && E % 4 == 0) {
            pl.comb_atomic = true;
            pl.fdn = k.moep_da[dv];
        }
        if (pl.ydh || pl.comb_atomic) {
            pl.bits = MoeBits::moe_lossy_gemm;  // not the f16 GEMM class (balance item f16 intermediates / fast candidate)
            pl.comb4 = true;
            pl.comb_grid = D((E + 1023) / 1024, n);
        }
        pl.gu_pgrid = D(pl.max_tiles64 * ((F + vg.gr - 1) / vg.gr));
        pl.dn_pgrid = D(max_tiles * ((E + vd.bm - 1) / vd.bm));
        if (pl.comb_fused) pl.comb4 = false;
    } else if (fp8) {
        pl.kind = MoePrefillKind::fp8;
        pl.bits = MoeBits::moe_fp8_gemm;
        const u32 nby = (F + moe_bm - 1) / moe_bm, nbyd = (E + moe_bm - 1) / moe_bm;
        pl.rbf = f.moe_rbf && k.gemm8r_moe != nullptr && k.gemm8r_moe32 != nullptr && k.gemm8r_moeh != nullptr && k.gemm8r_moe32h != nullptr;
        pl.fgu = pl.rbf ? (bn == 32 ? k.gemm8r_moe32h : k.gemm8r_moeh) : (bn == 32 ? k.gemm8_moe32h : k.gemm8_moeh);
        pl.fdn = pl.rbf ? (bn == 32 ? k.gemm8r_moe32 : k.gemm8r_moe) : (bn == 32 ? k.gemm8_moe32 : k.gemm8_moe);
        pl.gu_pgrid = pl.rbf ? D(max_tiles * 2 * nby) : D(max_tiles, 2 * nby);
        pl.dn_pgrid = pl.rbf ? D(max_tiles * nbyd) : D(max_tiles, nbyd);
        // v0252-moedf16 (PLAN V1-76): COMB 3 = the balance item f16 intermediates on this kind too - the down projection
        // writes moe_yd as f16 (gemm8r_moeh / gemm8r_moe32h with W2 = null: one weight set, f16 epilogue, same grid) and
        // moe_combine4h reads it. Never in precise (prec_dec, or the mode's bitwise class with WHIRL_Q8DEC=1). The COMB of
        // the caller, not moepFloor's (that floor is about the moep tile sharing; the yd bytes are saved at any n).
        if (f_in.moep_comb == 3 && !f.prec_dec && f.moep_comb_ok != MoeFlags::MoepComb::bitwise && pl.rbf && k.moe_combine4h != nullptr && E % 4 == 0) {
            pl.ydh = true;
            pl.fdn = bn == 32 ? k.gemm8r_moe32h : k.gemm8r_moeh;
            pl.bits = MoeBits::moe_lossy_gemm;
            pl.comb4 = true;
            pl.comb_grid = D((E + 1023) / 1024, n);
        }
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
    // v0252-f5 moeskip: the grouped GEMMs only read the tiles (kept positions), so they are unchanged; the grouping, the
    // 64-token list and the combine take their skip twins (qwen35::Context::moeBlock)
    if (Function g = nullptr, d = nullptr; skipOk(p, f, k, pl, &g, &d)) pl.skip = true;
    return pl;
}

// v0252-c3port (feat026-moedec-p25 0ae5530 src/ops/moe.cpp, + the plan's gate+up kernel must be moe_gu_f_mxfp4: the
// fused routed blocks reproduce that kernel only). Only the gfx1201 code object has moe_gus*_f: on the gfx1151 table
// every call returns "not taken" (the 0.2.5.1 chain).
MoeFusePlan chooseMoeFuse(const MoeFuseProblem& p, bool enabled, const MoePlan& pl, const kernels::MoeKernels& k) {
    MoeFusePlan f;
    auto no = [&](const char* why) {
        f.why = why;
        return f;
    };
    if (!enabled) return no("off (WHIRL_MOE_FUSE)");
    if (!pl.small || pl.small_q8 || pl.bits != MoeBits::moe_f32_row || pl.gu_split) return no("not the precise decode experts with one gate+up launch");
    if (pl.gu == nullptr || pl.gu != k.moe_gu_f[ti(GgmlType::mxfp4)]) return no("the plan's gate+up kernel is not moe_gu_f_mxfp4");
    if (p.gate_ty != GgmlType::mxfp4 || p.up_ty != GgmlType::mxfp4 || p.gate_rb != p.up_rb) return no("routed gate / up not MXFP4 of one row size");
    if (p.n == 0 || p.n > 32) return no("rows outside 1..32");
    if (p.F == 0 || p.F % 4 != 0 || p.C == 0 || p.C % 256 != 0 || p.K == 0) return no("shape (F % 4, C % 256)");
    if (p.shared) {
        if (!p.shared_gemvh) return no("shared expert not on the f16 gemvh path");
        if (p.sg_ty != GgmlType::mxfp4 || p.su_ty != GgmlType::mxfp4 || p.sg_rb != p.su_rb) return no("shared gate / up not MXFP4 of one row size");
        if (p.sg_ncols != p.C || p.su_ncols != p.C || p.sg_nrows != p.su_nrows || p.sg_nrows == 0 || p.sg_nrows % 64 != 0)
            return no("shared shape (ncols == n_embd, rows % 64)");
    }
    const Function fn = p.n <= 16 ? k.moe_gus_f : k.moe_gus2_f;
    if (fn == nullptr) return no("no moe_gus*_f_mxfp4 in the code object");
    f.use = true;
    f.f = fn;
    f.nsb = p.shared ? p.sg_nrows / 64 : 0;
    f.nrb = p.F / 4;
    f.grid = D(f.nsb + static_cast<u64>(f.nrb) * p.n * p.K);
    return f;
}

}  // namespace whirl::ops
