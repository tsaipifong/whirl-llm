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
        k.moe_router_fused_i16 = tag("moe_router_fused_i16");  // v0251-a11port: gfx1151 code object only
        k.moe_router_fused_i24 = tag("moe_router_fused_i24");
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

// v0251-a11port (feat026-router): the fused router choice (ops::MoePlan::router), qwen35::Context::moeBlock's launch
// decision, as the screened host condition (feat026-router-p25 f06b245 forward/ffn.cpp): flag on, a small batch of one
// piece (n <= max_small_batch), R <= 256, K <= 32, E % 4 == 0, ceil(E / 128) = 16 (i16) or 24 (i24), the counter
// buffer, and the kernel in the code object (only the gfx1151 one has it). Everything else: null = moe_logits_f32 +
// moe_topk, the 0.2.5 path. Fails before the port (chooseMoe never sets router).
void testOpsMoeRouter(ops::MoeProblem p, const kn::KernelTable& k97, const kn::KernelTable& k8) {
    p.rcnt_buf = true;
    ops::MoeFlags bal, pre;
    bal.moe_fp8 = true;
    bal.moe_rfuse = pre.moe_rfuse = true;
    pre.prec_dec = true;
    CHECK(!ops::MoeFlags{}.moe_rfuse && !ops::MoeProblem{}.rcnt_buf);  // the defaults: off (callers set them)
    // 8060S, balance and precise: decode and verify pieces up to 16 rows take the i16 entry (Ornith E 2048, R 256, K 8)
    for (const std::uint32_t n : {1u, 2u, 4u, 5u, 16u}) {
        p.n = n;
        CHECK(nameOf(ops::chooseMoe(p, bal, k8, k8.silu_mul_x8h).router) == "moe_router_fused_i16");
        CHECK(nameOf(ops::chooseMoe(p, pre, k8, k8.silu_mul_x8h).router) == "moe_router_fused_i16");
    }
    // the plan's other fields do not move with the router choice
    {
        p.n = 2;
        ops::MoeFlags off = bal;
        off.moe_rfuse = false;
        const ops::MoePlan a = ops::chooseMoe(p, off, k8, k8.silu_mul_x8h), b = ops::chooseMoe(p, bal, k8, k8.silu_mul_x8h);
        CHECK(a.router == nullptr && nameOf(b.router) == "moe_router_fused_i16");  // WHIRL_MOE_RFUSE=0: the 0.2.5 path
        CHECK(a.small == b.small && a.small_q8 == b.small_q8 && a.gu == b.gu && a.down == b.down && a.logits_grid.x == b.logits_grid.x &&
              b.logits_grid.x == 32);
    }
    // more than one piece: the two-kernel path (balance wide verify 17..32 rows; precise verify up to max_verify_rows)
    {
        ops::MoeFlags wide = bal;
        wide.small_max = 32;
        p.n = 17;
        const ops::MoePlan w = ops::chooseMoe(p, wide, k8, k8.silu_mul_x8h);
        CHECK(w.small && w.router == nullptr);
        p.n = 24;
        const ops::MoePlan v = ops::chooseMoe(p, pre, k8, k8.silu_mul_x8h);
        CHECK(v.small && v.router == nullptr);
    }
    // prefill: never
    p.n = 4096;
    CHECK(ops::chooseMoe(p, bal, k8, k8.silu_mul_x8h).router == nullptr && ops::chooseMoe(p, pre, k8, k8.silu_mul_x8h).router == nullptr);
    // the R9700 table has no fused router: the two-kernel path even with the flag on
    p.n = 2;
    CHECK(ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h).router == nullptr && ops::chooseMoe(p, pre, k97, k97.silu_mul_x8h).router == nullptr);
    // no counter buffer: the two-kernel path
    {
        ops::MoeProblem q = p;
        q.rcnt_buf = false;
        CHECK(ops::chooseMoe(q, bal, k8, k8.silu_mul_x8h).router == nullptr);
    }
    // shapes: E 3072 -> i24, E 2000 -> i16 (ceil 16), E 2050 (not % 4) / 4096 (32 iterations) / R 257 / K 33 -> null
    struct S {
        std::uint32_t E, R, K;
        const char* want;
    };
    for (const S s : {S{3072, 256, 8, "moe_router_fused_i24"}, S{2000, 256, 8, "moe_router_fused_i16"}, S{2048, 128, 8, "moe_router_fused_i16"},
                      S{2048, 256, 32, "moe_router_fused_i16"}, S{2050, 256, 8, "null"}, S{4096, 256, 8, "null"}, S{1024, 256, 8, "null"},
                      S{2048, 257, 8, "null"}, S{2048, 256, 33, "null"}}) {
        ops::MoeProblem q = p;
        q.E = s.E;
        q.R = s.R;
        q.K = s.K;
        q.n = 3;
        const std::string got = nameOf(ops::chooseMoe(q, bal, k8, k8.silu_mul_x8h).router);
        CHECK_DETAIL(got == s.want, "router shape E " + std::to_string(s.E) + " R " + std::to_string(s.R) + " K " + std::to_string(s.K) +
                                        ": got " + got + ", want " + s.want);
    }
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
    testOpsMoeRouter(p, k97, k8);
}

}  // namespace
