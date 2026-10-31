// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.gdn/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/gdn.h"
#include "../test_check.h"
#include "ops_fakes.h"
#include "ops_mat.h"

#include <cstdint>
#include <string>

namespace {

namespace ops = whirl::ops;

// the DeltaNet rules (arch025-n8b1: the blocks of testGdnChoice that call only src/ops/gdn.cpp and act.cpp, in a function of
// their own so ops.gdn's isolation program runs them; whirl-ops-tests calls it from testGdnChoice, in the same place)
void testGdnRules() {
    const kn::KernelTable k = gdnTable(true);
    ops::GdnShape s;
    s.conv_ch = 10240;
    s.n_v_heads = 48;
    s.n_k_heads = 16;
    // h16: never on fused rows, one row, or more than 64 v heads
    {
        ops::MatmulFlags f;
        f.fp8_prefill = true;
        f.gv_max = 16;
        ops::H16Tune h;
        ops::GdnH16Problem p;
        p.qkv = p.gate = mat(GgmlType::mxfp4, 6144, 5120, false, true);
        p.out = mat(GgmlType::mxfp4, 5120, 6144, false, true);
        p.n = 512;
        p.s = s;
        ops::GdnH16Flags hf;
        hf.ffn_h16 = true;
        CHECK(ops::chooseGdnH16(p, hf, f, h, k));
        p.fused = true;
        CHECK(!ops::chooseGdnH16(p, hf, f, h, k));
        p.fused = false;
        p.s.n_v_heads = 96;
        CHECK(!ops::chooseGdnH16(p, hf, f, h, k));
        p.s = s;
        hf.act_fuse = false;
        CHECK(!ops::chooseGdnH16(p, hf, f, h, k));
    }
    // [beta; alpha]: one GEMM only with the buffer, contiguous rows, the kernel and WHIRL_GDN_BA
    {
        ops::GdnBaProblem ba;
        ba.ba_buf = ba.ba_contiguous = true;
        CHECK(ops::gdnBaGemm(ba, k));
        ba.ba_contiguous = false;
        CHECK(!ops::gdnBaGemm(ba, k) && ops::chooseGdnGates(false, ba, k) == ops::GdnGates::split);
        CHECK(ops::chooseGdnGates(true, ba, k) == ops::GdnGates::in2);
        ba.ba_contiguous = true;
        ba.gdn_ba_on = false;
        CHECK(!ops::gdnBaGemm(ba, k));
        // decode: WHIRL_DBG&1 turns abconv off -> gdn_ab + conv; precise rows without the one GEMM -> two GEMMs
        CHECK(ops::chooseGdnDecode(GgmlType::q8_0, 1, 16, false, 1, ba, k).gates == ops::GdnDecodeGates::ab);
        CHECK(nameOf(ops::chooseGdnDecode(GgmlType::q8_0, 1, 16, false, 1, ba, k).f) == "gdn_ab_q8_0");
        CHECK(ops::chooseGdnDecode(GgmlType::q8_0, 1, 16, true, 0, ba, k).gates == ops::GdnDecodeGates::precise_split);
    }
    // conv: one row / d_conv != 4 -> the sequential conv + l2norm; without act_fuse no fused l2norm
    {
        CHECK(ops::chooseGdnConv(false, false, 1, s, true, k).conv == ops::GdnConv::seq && ops::chooseGdnConv(false, false, 1, s, true, k).l2norm);
        ops::GdnShape s3 = s;
        s3.d_conv = 3;
        CHECK(ops::chooseGdnConv(false, false, 512, s3, true, k).conv == ops::GdnConv::seq);
        const ops::GdnConvPlan c = ops::chooseGdnConv(false, false, 512, s, false, k);
        CHECK(c.conv == ops::GdnConv::par && c.l2norm && !c.state_h);
        CHECK(!ops::gdnFusable(true, mat(GgmlType::q8_0, 48, 5120), mat(GgmlType::q8_0, 48, 5120), s3, k));
    }
    // scan / chunk: < 64 rows sequential; strix only when the scratch fits and n >= gdn_strix_min
    {
        CHECK(ops::chooseGdnScan(true, 63, s) == ops::GdnScan::seq && ops::chooseGdnScan(false, 512, s) == ops::GdnScan::seq);
        const kn::KernelTable k8 = gdnTable(false);
        ops::GdnChunkProblem cp;
        cp.n = 512;
        cp.n_v_heads = 48;
        cp.n_k_heads = 16;
        cp.gc_m_bytes = cp.gc_w_bytes = cp.gc_u_bytes = 1ull << 30;
        ops::GdnChunkFlags cf;
        cf.gdn_strix = true;
        const ops::GdnChunkPlan pl = ops::chooseGdnChunk(cp, cf, k8);
        CHECK(pl.kind == ops::GdnChunk::strix && pl.nch == 16 && pl.tp == 48ull * 16 * 32 * 32 * 2 && pl.qk == 16ull * 512 * 128 * 2);
        cp.gc_w_bytes = 48ull * 512 * 128 * 2 - 1;  // Vh does not fit
        CHECK(ops::chooseGdnChunk(cp, cf, k8).kind == ops::GdnChunk::f32);
        cf.gdn_wmma = true;
        CHECK(ops::chooseGdnChunk(cp, cf, k).kind == ops::GdnChunk::wmma);
    }
}

}  // namespace
