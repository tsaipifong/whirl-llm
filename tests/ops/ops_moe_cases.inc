// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.moe/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/moe.h"
#include "../test_check.h"
#include "ops_fakes.h"

#include <cstdint>
#include <string>

namespace {

namespace ops = whirl::ops;

// ---- MoE (K-4)
kn::KernelTable moeTable(bool g1151) {
    kn::KernelTable k;
    const auto mx = ti(GgmlType::mxfp4), q8 = ti(GgmlType::q8_0);
    k.moe_gu[mx] = tag("moe_gu_mxfp4");
    k.moe_down[mx] = tag("moe_down_mxfp4");
    k.moe_gu[q8] = tag("moe_gu_q8_0");
    k.moe_down[q8] = tag("moe_down_q8_0");
    k.moe_gu_mxw = tag("moe_gu_mxw");
    k.moe_down_mxw = tag("moe_down_mxw");
    k.moe_gu_f[mx] = tag("moe_gu_f_mxfp4");
    k.moe_down_f[mx] = tag("moe_down_f_mxfp4");
    k.moe_gu_f[q8] = tag("moe_gu_f_q8_0");
    k.gemm_moe[mx] = tag("gemm_moe_mxfp4");
    k.gemm_moe32[mx] = tag("gemm_moe32_mxfp4");
    if (g1151) {
        k.moe_tiles = tag("moe_tiles");
        k.gemm_moegu[mx] = tag("gemm_moegu_mxfp4");
        k.gemm_moe32r[mx] = tag("gemm_moe32r_mxfp4");
    } else {
        k.gemmr_moegu = tag("gemmr_moegu");
        k.gemmr_moegu32 = tag("gemmr_moegu32");
        k.gemmr_moedn = tag("gemmr_moedn");
        k.gemmr_moedn32 = tag("gemmr_moedn32");
        k.gemm8_moe = tag("gemm8_moe");
        k.gemm8_moe32 = tag("gemm8_moe32");
        k.gemm8_moeh = tag("gemm8_moeh");
        k.gemm8_moe32h = tag("gemm8_moe32h");
        k.gemm8r_moe = tag("gemm8r_moe");
        k.gemm8r_moe32 = tag("gemm8r_moe32");
        k.gemm8r_moeh = tag("gemm8r_moeh");
        k.gemm8r_moe32h = tag("gemm8r_moe32h");
        k.moe_gather_fp8 = tag("moe_gather_fp8");
        k.silu_mul_x8h = tag("silu_mul_x8h");
    }
    return k;
}

void testOpsMoe() {
    ops::MoeProblem p;  // Ornith-like: E 2048, 8 of 256 experts, F 512, all MXFP4 with row exponents
    p.E = 2048;
    p.K = 8;
    p.F = 512;
    p.R = 256;
    p.gate_ty = p.up_ty = p.down_ty = GgmlType::mxfp4;
    p.gate_rb = p.up_rb = 2048 / 32 * 17;
    p.gate_ncols = 2048;
    p.down_ncols = 512;
    p.refs = true;
    p.sx_buf = true;
    p.tiles64_buf = true;
    ops::MoeFlags bal;
    bal.moe_fp8 = true;
    ops::MoeFlags pre;
    pre.prec_dec = true;
    const kn::KernelTable k97 = moeTable(false), k8 = moeTable(true);
    // decode, balance: int8 experts, the MXFP4 whole-block twins, gate + up in one launch
    p.n = 1;
    auto pl = ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h);
    CHECK(pl.small && pl.small_q8 && pl.bits == ops::MoeBits::moe_q8_row && nameOf(pl.gu) == "moe_gu_mxw" && nameOf(pl.down) == "moe_down_mxw");
    CHECK(!pl.gu_split && pl.gu_grid.x == (2 * 512 + 7) / 8 && pl.gu_grid.y == 8 && pl.down_grid.x == 256 && pl.down_grid.y == 1 && pl.logits_grid.x == 32);
    ops::MoeFlags nomxw = bal;
    nomxw.moe_mxw = false;
    CHECK(nameOf(ops::chooseMoe(p, nomxw, k97, k97.silu_mul_x8h).gu) == "moe_gu_mxfp4");
    // decode, precise: the f32-activation experts, the precise class
    pl = ops::chooseMoe(p, pre, k97, k97.silu_mul_x8h);
    CHECK(pl.small && !pl.small_q8 && pl.bits == ops::MoeBits::moe_f32_row && nameOf(pl.gu) == "moe_gu_f_mxfp4" && nameOf(pl.down) == "moe_down_f_mxfp4");
    // precise: a 24-row verify (> 16) still takes the decode experts (MTP + n-gram == plain); balance does not
    p.n = 24;
    CHECK(ops::chooseMoe(p, pre, k97, k97.silu_mul_x8h).small && !ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h).small);
    // gate and up of different types: two gate / up launches of F rows each
    ops::MoeProblem q = p;
    q.n = 2;
    q.up_ty = GgmlType::q8_0;
    pl = ops::chooseMoe(q, bal, k97, k97.silu_mul_x8h);
    CHECK(pl.gu_split && nameOf(pl.gu) == "moe_gu_mxw" && nameOf(pl.gu_up) == "moe_gu_q8_0" && pl.gu_grid.x == 64 && pl.gu_grid.y == 16);
    // prefill on gfx1201, balance: MXFP4 x fp8, row-block-fast 1-D grids; 64-token tiles for a long prompt
    p.n = 4096;
    pl = ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h);
    CHECK(!pl.small && pl.kind == ops::MoePrefillKind::fp8 && pl.rbf && pl.bn == 64 && nameOf(pl.fgu) == "gemm8r_moeh" && nameOf(pl.fdn) == "gemm8r_moe");
    CHECK(pl.max_tiles == 4096 * 8 / 64 + 256 && pl.gu_pgrid.x == pl.max_tiles * 2 * 4 && pl.gu_pgrid.y == 1 && pl.dn_pgrid.x == pl.max_tiles * 16);
    // a short prompt: 32-token tiles
    p.n = 512;
    pl = ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h);
    CHECK(pl.bn == 32 && nameOf(pl.fgu) == "gemm8r_moe32h");
    // precise prefill on gfx1201: no fp8, the f16 MXFP4 gemmr grouped GEMMs
    p.n = 4096;
    pl = ops::chooseMoe(p, pre, k97, k97.silu_mul_x8h);
    CHECK(pl.kind == ops::MoePrefillKind::gemmr && nameOf(pl.fgu) == "gemmr_moegu" && pl.bits == ops::MoeBits::moe_f16_gemm);
    // WHIRL_MOE_RBF=0: the generic grouped GEMM (gate, up, down separately; 2-D grids)
    ops::MoeFlags norbf = pre;
    norbf.moe_rbf = false;
    pl = ops::chooseMoe(p, norbf, k97, k97.silu_mul_x8h);
    CHECK(pl.kind == ops::MoePrefillKind::generic && nameOf(pl.g_gate) == "gemm_moe_mxfp4" && pl.gu_pgrid.y == 4 && pl.dn_pgrid.y == 16);
    // the 8060S: the fused gate + up + SwiGLU over 64-token tiles, down over 32-token tiles (bn 32)
    pl = ops::chooseMoe(p, bal, k8, k8.silu_mul_x8h);
    CHECK(pl.kind == ops::MoePrefillKind::fused_gu && pl.bn == 32 && pl.max_tiles64 == 4096 * 8 / 64 + 256 && nameOf(pl.fgu) == "gemm_moegu_mxfp4");
    CHECK(pl.gu_pgrid.x == pl.max_tiles64 * 8 && pl.dn_pgrid.x == pl.max_tiles * 16);
    // WHIRL_MOE_BN forces the tile only where the fused path does not set it
    ops::MoeFlags bn64 = norbf;
    bn64.moe_bn_force = 64;
    p.n = 512;
    CHECK(ops::chooseMoe(p, bn64, k97, k97.silu_mul_x8h).bn == 64);
    CHECK(ops::moeTile(512, 8, 256) == 32 && ops::moeTile(4096, 8, 256) == 64);
}

}  // namespace
