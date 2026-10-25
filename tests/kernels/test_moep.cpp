// whirl-kernel-test, feat026-moep: the multi-fragment MoE prefill expert GEMMs of the Radeon 8060S
// (kernels/gfx1151/moe_prefill_mt.hip; optional, gfx1151 code object only, MXFP4 experts), checked on every routed
// position against the shipped path (outputs pre-filled with NaN bytes, so a missed store fails):
//   bit-identical (precise / balance / fast):
//   * gemm_moepgu_<v>_mxfp4 (tiles: moe_tiles bn = BNMAX) == gemm_moegu_mxfp4 (f16 bits)
//   * gemm_moepdn_<v>_mxfp4 == gemm_moe32r_mxfp4 (f32 bits)
//   * gemm_moepdc_<v>_mxfp4 (fused combine) == gemm_moe32r + moe_combine (Y and x bits), counters left zero,
//     two runs equal (deterministic whatever the block order)
//   * moe_combine4 == moe_combine
//   not bit-identical, never in precise (balance item f16 intermediates / fast candidate; mode acceptance is task
//   accuracy, measured by the GPU runner), checked here within the stated numeric bound:
//   * gemm_moepdh_<v> == f16(gemm_moe32r) bits; moe_combine4h and the fused gemm_moepdhc_<v> equal each other
//     bitwise and moe_combine within one f16 rounding of every expert output
//   * gemm_moepda_<v> (atomic scatter) + moe_combine4 (K = 0) == moe_combine within f32 reassociation
// Routings: skewed (a hot expert with 1/7 of the pairs: several BNMAX tiles, tiles of 1..BNMAX tokens, empty
// 16-token fragments) at 200 and 1500 tokens, and a Zipf(1.0) routing at 2048 tokens (no repeated expert per token).
// SPDX-License-Identifier: Apache-2.0

#include "cpu_ref.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace kt {

namespace {
unsigned cdivp(std::size_t a, unsigned b) { return static_cast<unsigned>((a + b - 1) / b); }

// Zipf-like routing: token t takes K distinct experts drawn with p(e) ~ 1 / (rank(e) + 1)^s under a fixed random
// rank permutation (realistic skew; the bench can also replay a captured WHIRL_DUMP_MOE routing)
std::vector<int> zipfRouting(std::mt19937_64& rng, int n, int K, int R, double s) {
    std::vector<int> rank(static_cast<std::size_t>(R));
    std::iota(rank.begin(), rank.end(), 0);
    std::shuffle(rank.begin(), rank.end(), rng);
    std::vector<double> cdf(static_cast<std::size_t>(R));
    double acc = 0;
    for (int e = 0; e < R; ++e) cdf[static_cast<std::size_t>(e)] = acc += 1.0 / std::pow(rank[static_cast<std::size_t>(e)] + 1.0, s);
    std::uniform_real_distribution<double> u(0.0, acc);
    std::vector<int> ids(static_cast<std::size_t>(n) * K);
    for (int t = 0; t < n; ++t) {
        int got = 0;
        while (got < K) {
            const int e = static_cast<int>(std::lower_bound(cdf.begin(), cdf.end(), u(rng)) - cdf.begin());
            const int ee = std::min(e, R - 1);
            bool dup = false;
            for (int j = 0; j < got; ++j) dup = dup || ids[static_cast<std::size_t>(t) * K + j] == ee;
            if (!dup) ids[static_cast<std::size_t>(t) * K + got++] = ee;
        }
    }
    return ids;
}
}  // namespace

void moepChecks(Ctx& c, const HostMat& gate, const HostMat& up, const HostMat& down, int R, int K) {
    const int tm = static_cast<int>(QType::mxfp4);
    bool any = false;
    for (int v = 0; v < wk::kMoepGuN; ++v) any = any || c.k.moep_gu[v] != nullptr;
    if (!any || !c.k.moe_tiles || !c.k.gemm_moegu[tm] || !c.k.gemm_moe32r[tm] || gate.type != QType::mxfp4 || up.type != QType::mxfp4 ||
        down.type != QType::mxfp4) {
        // key prefix "<family> gemm_moegu": absent by design on gfx1201 (kt.cpp requiredKernelAbsentOk)
        c.rep.skip(c.rep.family, "gemm_moegu moep variants", "8060S multi-fragment expert GEMMs not in this code object");
        return;
    }
    const int E = gate.ncols, F = down.ncols;
    Buf dgate(gate.data), dup(up.data), ddown(down.data);
    struct Case {
        int n;
        bool zipf;
    };
    for (const Case cs : {Case{200, false}, Case{1500, false}, Case{2048, true}}) {
        const int n = c.quick && cs.n > 256 ? 256 : cs.n, pairs = n * K;
        std::vector<int> ids;
        if (cs.zipf) {
            ids = zipfRouting(c.rng, n, K, R, 1.0);
        } else {
            ids.resize(static_cast<std::size_t>(pairs));
            for (int p = 0; p < pairs; ++p) ids[static_cast<std::size_t>(p)] = p % 7 == 0 ? 3 : static_cast<int>(c.rng() % static_cast<unsigned>(R));
            // top-k picks distinct experts per token: re-draw duplicates (the fused combine counts each pair once)
            for (int t = 0; t < n; ++t)
                for (int a = 1; a < K; ++a)
                    for (int b = 0; b < a; ++b)
                        while (ids[static_cast<std::size_t>(t) * K + a] == ids[static_cast<std::size_t>(t) * K + b]) {
                            ids[static_cast<std::size_t>(t) * K + a] = static_cast<int>(c.rng() % static_cast<unsigned>(R));
                            b = -1;
                            break;
                        }
        }
        const std::string tag = std::string(" (") + std::to_string(n) + " tokens, " + (cs.zipf ? "zipf" : "skewed") + ")";
        const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * E);
        std::vector<float> wv = c.randu(static_cast<std::size_t>(pairs), 0.f, 0.3f), sgv = c.randu(static_cast<std::size_t>(n), 0.f, 1.f);
        const std::vector<float> x0 = c.randn(static_cast<std::size_t>(n) * E), ysh = c.randn(static_cast<std::size_t>(n) * E);
        Buf dids(ids), dh(h), dw(wv), dsg(sgv), dx0(x0), dysh(ysh);
        Buf perm(static_cast<std::size_t>(pairs) * 4), inv(static_cast<std::size_t>(pairs) * 4);
        const int mt32 = (pairs + 31) / 32 + R, mt64 = (pairs + 63) / 64 + R;
        Buf t32(static_cast<std::size_t>(mt32) * 16), n32(4), t64(static_cast<std::size_t>(mt64) * 16), n64(4);
        hip::launch(c.k.moe_route, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dids.p(), pairs, R, 32, perm.p(), inv.p(), t32.p(), n32.p());
        hip::launch(c.k.moe_tiles, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dids.p(), pairs, R, 64, t64.p(), n64.p());
        Buf xg(static_cast<std::size_t>(pairs) * E * 2);
        hip::launch(c.k.moe_gather_f16, {static_cast<unsigned>(pairs), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), perm.p(), xg.p(), E, K);
        // references: the shipped 8060S path
        Buf aref(static_cast<std::size_t>(pairs) * F * 2), yref(static_cast<std::size_t>(pairs) * E * 4), xref(static_cast<std::size_t>(n) * E * 4);
        hip::launch(c.k.gemm_moegu[tm], {static_cast<unsigned>(mt64) * cdivp(F, 64), 1, 1}, {256, 1, 1}, 0, c.s, dgate.p(), dup.p(), gate.row_bytes, F,
                    xg.p(), aref.p(), E, t64.p(), n64.p());
        hip::launch(c.k.gemm_moe32r[tm], {static_cast<unsigned>(mt32) * cdivp(E, wk::kMoeBm), 1, 1}, {256, 1, 1}, 0, c.s, ddown.p(), down.row_bytes, E,
                    aref.p(), yref.p(), F, t32.p(), n32.p());
        xref.up(x0);
        hip::launch(c.k.moe_combine, {cdivp(E, 256), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, xref.p(), dysh.p(), dsg.p(), yref.p(), inv.p(),
                    dw.p(), E, K);
        c.sync();
        const auto ar = aref.down<std::uint16_t>(static_cast<std::size_t>(pairs) * F);
        const auto yr = yref.down<float>(static_cast<std::size_t>(pairs) * E);
        const auto xr = xref.down<float>(static_cast<std::size_t>(n) * E);
        const auto hinv = inv.down<int>(static_cast<std::size_t>(pairs));
        // gate + up variants
        for (int v = 0; v < wk::kMoepGuN; ++v) {
            if (!c.k.moep_gu[v]) continue;
            const wk::MoepGuVariant& vg = wk::kMoepGu[v];
            const int mt = (pairs + vg.bn - 1) / vg.bn + R;
            Buf tv(static_cast<std::size_t>(mt) * 16), nv(4), anew(static_cast<std::size_t>(pairs) * F * 2);
            anew.fill(0xff);
            hip::launch(c.k.moe_tiles, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dids.p(), pairs, R, vg.bn, tv.p(), nv.p());
            hip::launch(c.k.moep_gu[v], {static_cast<unsigned>(mt) * cdivp(F, static_cast<unsigned>(vg.gr)), 1, 1}, {256, 1, 1}, 0, c.s, dgate.p(), dup.p(),
                        gate.row_bytes, F, xg.p(), anew.p(), E, tv.p(), nv.p());
            c.sync();
            c.rep.add(cmpExact(std::string("gemm_moepgu_") + vg.name + "_mxfp4 == gemm_moegu_mxfp4" + tag, anew.down<std::uint16_t>(ar.size()), ar,
                               Kind::invariant));
        }
        // combine4 (same bits as moe_combine)
        if (c.k.moe_combine4 && E % 4 == 0) {
            Buf x4(static_cast<std::size_t>(n) * E * 4);
            x4.up(x0);
            hip::launch(c.k.moe_combine4, {cdivp(E, 1024), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, x4.p(), dysh.p(), dsg.p(), yref.p(), inv.p(),
                        dw.p(), E, K);
            c.sync();
            c.rep.add(cmpExact("moe_combine4 == moe_combine" + tag, x4.down<float>(xr.size()), xr, Kind::invariant));
        }
        // CPU bound of the lossy candidates: sum_k |w y| (+ |sg ysh|) per element
        std::vector<double> xr_d(xr.begin(), xr.end()), sc(xr.size());
        for (int t = 0; t < n; ++t)
            for (int e = 0; e < E; ++e) {
                double s = std::fabs(static_cast<double>(sgv[static_cast<std::size_t>(t)]) * ysh[static_cast<std::size_t>(t) * E + e]);
                for (int k = 0; k < K; ++k)
                    s += std::fabs(static_cast<double>(wv[static_cast<std::size_t>(t) * K + k]) *
                                   yr[static_cast<std::size_t>(hinv[static_cast<std::size_t>(t) * K + k]) * E + e]);
                sc[static_cast<std::size_t>(t) * E + e] = s + std::fabs(static_cast<double>(x0[static_cast<std::size_t>(t) * E + e]));
            }
        for (int v = 0; v < wk::kMoepDnN; ++v) {
            if (!c.k.moep_dn[v]) continue;
            const wk::MoepDnVariant& vd = wk::kMoepDn[v];
            const std::string vn = vd.name;
            const int mt = (pairs + vd.bn - 1) / vd.bn + R;
            const unsigned grid = static_cast<unsigned>(mt) * cdivp(E, static_cast<unsigned>(vd.bm));
            const int nrb = (E + vd.bm - 1) / vd.bm;
            Buf tv(static_cast<std::size_t>(mt) * 16), nv(4), ynew(static_cast<std::size_t>(pairs) * E * 4);
            hip::launch(c.k.moe_tiles, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dids.p(), pairs, R, vd.bn, tv.p(), nv.p());
            ynew.fill(0xff);
            hip::launch(c.k.moep_dn[v], {grid, 1, 1}, {256, 1, 1}, 0, c.s, ddown.p(), down.row_bytes, E, aref.p(), ynew.p(), F, tv.p(), nv.p());
            c.sync();
            c.rep.add(cmpExact("gemm_moepdn_" + vn + "_mxfp4 == gemm_moe32r_mxfp4" + tag, ynew.down<float>(yr.size()), yr, Kind::invariant));
            Buf cnt(static_cast<std::size_t>(n) * nrb * 4);
            if (c.k.moep_dc[v]) {
                // fused combine: Y and x bitwise, counters zero after, a second run equal
                std::vector<float> xs[2];
                for (int run = 0; run < 2; ++run) {
                    Buf xf(static_cast<std::size_t>(n) * E * 4);
                    xf.up(x0);
                    cnt.zero();
                    ynew.fill(0xff);
                    hip::launch(c.k.moep_dc[v], {grid, 1, 1}, {256, 1, 1}, 0, c.s, ddown.p(), down.row_bytes, E, aref.p(), ynew.p(), F, tv.p(), nv.p(), perm.p(),
                                inv.p(), dw.p(), dsg.p(), dysh.p(), xf.p(), cnt.p(), K);
                    c.sync();
                    xs[run] = xf.down<float>(xr.size());
                    if (run == 0) {
                        c.rep.add(cmpExact("gemm_moepdc_" + vn + "_mxfp4 Y == gemm_moe32r_mxfp4" + tag, ynew.down<float>(yr.size()), yr, Kind::invariant));
                        c.rep.add(cmpExact("gemm_moepdc_" + vn + "_mxfp4 x == gemm_moe32r + moe_combine" + tag, xs[0], xr, Kind::invariant));
                        const auto hc = cnt.down<int>(static_cast<std::size_t>(n) * nrb);
                        Result r;
                        r.name = "gemm_moepdc_" + vn + "_mxfp4 counters left zero" + tag;
                        r.n = hc.size();
                        r.mismatches = static_cast<std::size_t>(std::count_if(hc.begin(), hc.end(), [](int q) { return q != 0; }));
                        r.pass = r.mismatches == 0;
                        c.rep.add(r);
                    }
                }
                c.rep.add(cmpExact("gemm_moepdc_" + vn + "_mxfp4 run 2 == run 1" + tag, xs[1], xs[0], Kind::invariant));
            }
            // balance / fast candidates
            if (c.k.moep_dh[v] && c.k.moe_combine4h && E % 4 == 0) {
                Buf yh(static_cast<std::size_t>(pairs) * E * 2), xh(static_cast<std::size_t>(n) * E * 4);
                yh.fill(0xff);
                hip::launch(c.k.moep_dh[v], {grid, 1, 1}, {256, 1, 1}, 0, c.s, ddown.p(), down.row_bytes, E, aref.p(), yh.p(), F, tv.p(), nv.p());
                xh.up(x0);
                hip::launch(c.k.moe_combine4h, {cdivp(E, 1024), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, xh.p(), dysh.p(), dsg.p(), yh.p(), inv.p(),
                            dw.p(), E, K);
                c.sync();
                std::vector<std::uint16_t> yr16(yr.size());
                for (std::size_t i = 0; i < yr.size(); ++i) yr16[i] = f2h(yr[i]);
                c.rep.add(cmpExact("gemm_moepdh_" + vn + "_mxfp4 == f16(gemm_moe32r)" + tag, yh.down<std::uint16_t>(yr16.size()), yr16, Kind::invariant));
                const auto xhv = xh.down<float>(xr.size());
                // one f16 rounding of every expert output (2^-11 relative) + f32 noise
                c.rep.add(cmpTol("moe_combine4h (f16 yd, balance candidate) vs moe_combine" + tag, xhv, xr_d, sc, std::ldexp(1.0, -10), 1e-6));
                if (c.k.moep_dhc[v]) {
                    Buf xf(static_cast<std::size_t>(n) * E * 4);
                    xf.up(x0);
                    cnt.zero();
                    hip::launch(c.k.moep_dhc[v], {grid, 1, 1}, {256, 1, 1}, 0, c.s, ddown.p(), down.row_bytes, E, aref.p(), yh.p(), F, tv.p(), nv.p(), perm.p(),
                                inv.p(), dw.p(), dsg.p(), dysh.p(), xf.p(), cnt.p(), K);
                    c.sync();
                    // the fused f16 twin: per element the scalar combine expression, = moe_combine4h's per component
                    c.rep.add(cmpExact("gemm_moepdhc_" + vn + "_mxfp4 x == gemm_moepdh + moe_combine4h" + tag, xf.down<float>(xr.size()), xhv, Kind::invariant));
                }
            }
            if (c.k.moep_da[v] && c.k.moe_combine4 && E % 4 == 0) {
                Buf xa(static_cast<std::size_t>(n) * E * 4);
                xa.up(x0);
                hip::launch(c.k.moe_combine4, {cdivp(E, 1024), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, xa.p(), dysh.p(), dsg.p(), yref.p(), inv.p(),
                            dw.p(), E, 0);
                hip::launch(c.k.moep_da[v], {grid, 1, 1}, {256, 1, 1}, 0, c.s, ddown.p(), down.row_bytes, E, aref.p(), F, tv.p(), nv.p(), perm.p(), dw.p(), xa.p(), K);
                c.sync();
                // f32 reassociation of K + 2 terms: (K + 2) * 2^-24 of sum |terms|, kept at 1e-6
                c.rep.add(cmpTol("gemm_moepda_" + vn + "_mxfp4 (atomic scatter, fast candidate) vs moe_combine" + tag, xa.down<float>(xr.size()), xr_d, sc,
                                 1e-6, 1e-7));
            }
        }
    }
}

}  // namespace kt
