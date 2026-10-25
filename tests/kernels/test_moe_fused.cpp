// whirl-kernel-test "moefuse" (feat026-moedec): the fused precise MoE decode launch moe_gus_f_mxfp4 (n <= 16) /
// moe_gus2_f_mxfp4 (n <= 32) of kernels/moe_fused.hip against the chain it replaces in Model::moeBlock:
//   routed: moe_gu_f_mxfp4 -> silu_mul                  == the fused act (= moe_g), bitwise
//   shared: f32_to_f16(h) -> gemvh / gemvh2 (gate, up) -> silu_mul -> f32_to_f16
//                                                       == the fused f16 shared activations (x16), bitwise
// over n = 1..32 rows (both entries, the n_tok edge of each token tile), n_embd 2048 (gfx1201 old path: two-stage
// LDS rounds, fused: one stage - the same k order) and 768 (one-stage rounds both), expert widths 512 / 64 (the
// Ornith shapes and a small one), shared expert 512 / 64 rows and none (nsb = 0), plus a loose CPU reference
// (weights decoded by ref::dequantRows, double accumulation) so a wrong pairing / layout cannot pass by being
// wrong the same way in both chains.
// feat026-moedec-r1 (review M1): the shared f16 activations are also memcmp-compared with the same chain on gemm_c0_mxfp4
// (the prefill GEMM every gemvh kind equals, family gemvh): on gfx1201 that pins the fused S = 1 tile against both the
// old S = 2 gemvh (C % 512 == 0) and S = 1 (C % 512 != 0: C = 768, 1280, 2304); on gfx1151 the fused paired tile against
// the build's gemvh_mxfp4 kind (WHIRL_GH_KIND_MXFP4) and, through gemm_c0, against every kind. The case names say which
// old tile / kind each comparison pins. gemvhf gets the same gemm_c0 cross-check.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>

#include "cpu_ref.h"

namespace kt {

namespace {

constexpr int kR = 12, kK = 8;

Result inv(const std::string& name, std::size_t n, std::size_t bad, const std::string& note) {
    Result r;
    r.name = name;
    r.kind = Kind::invariant;
    r.n = n;
    r.mismatches = bad;
    r.pass = bad == 0;
    r.note = note;
    return r;
}

double silu(double g) { return g / (1.0 + std::exp(-g)); }

// which old gemvh tile a comparison pins (the fused kernel: gfx1201 S = 1, gfx1151 paired)
std::string oldTile(Ctx& c, int ncols) {
    if (c.mod.arch() == hip::Arch::gfx1151) return " [old gemvh: build kind WHIRL_GH_KIND_MXFP4, fused: paired]";
    return ncols % 512 == 0 ? " [old gemvh S=2, fused S=1]" : " [old gemvh S=1, fused S=1]";
}

// y = gemm_c0_<T>(W, x16) (the reference every gemvh kind / stage count equals bit for bit, family gemvh)
void gemmC0(Ctx& c, int ti, DevPtr w, std::uint64_t row_bytes, DevPtr x16, DevPtr y, int ncols, int nrows, int n) {
    const auto& c0 = wk::kGemmCfgs[0];
    hip::launch(c.k.gemmc[0][ti], hip::Dim3{cdiv(n, c0.bn), cdiv(nrows, c0.bm), 1}, {static_cast<unsigned>(c0.nth), 1, 1}, 0, c.s, w, row_bytes, x16, y,
                ncols, nrows, n, 0);
}

void fuseCase(Ctx& c, int C, int F, int sff, int n) {
    const QType t = QType::mxfp4;
    const int ti = static_cast<int>(t);
    const std::string tag = " C=" + std::to_string(C) + " F=" + std::to_string(F) + " sff=" + std::to_string(sff) + " n=" + std::to_string(n);
    const HostMat W1 = randomMat(c, t, kR * F, C), W2 = randomMat(c, t, kR * F, C);
    Buf d1(W1.data), d2(W2.data);
    HostMat SG, SU;
    Buf dsg, dsu;
    if (sff > 0) {
        SG = randomMat(c, t, sff, C);
        SU = randomMat(c, t, sff, C);
        dsg = Buf(SG.data);
        dsu = Buf(SU.data);
    }
    std::vector<int> ids(static_cast<std::size_t>(n) * kK);
    for (int p = 0; p < n * kK; ++p) ids[static_cast<std::size_t>(p)] = (p * 5 + p / kK) % kR;  // distinct within a token
    const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * C);
    Buf did(ids), dh(h);
    const std::size_t na = static_cast<std::size_t>(n) * kK * F, ns = static_cast<std::size_t>(n) * (sff > 0 ? sff : 1);
    // the old chain
    Buf g(na * 4), u(na * 4);
    g.fill(0xff);
    u.fill(0xff);
    hip::launch(c.k.moe_gu_f[ti], {cdiv(2 * F, 8), static_cast<unsigned>(n * kK), 1}, {256, 1, 1}, 0, c.s, d1.p(), d2.p(), W1.row_bytes, F, 2 * F,
                dh.p(), did.p(), g.p(), u.p(), C, kK);
    hip::launch(c.k.silu_mul, {cdiv(na, 256), 1, 1}, {256, 1, 1}, 0, c.s, g.p(), u.p(), static_cast<int>(na));
    Buf x16(static_cast<std::size_t>(n) * C * 2), sgo(ns * 4), suo(ns * 4), xs_ref(ns * 2);
    xs_ref.fill(0xee);
    const hip::Function hv = n <= 16 ? c.k.gemvh[ti] : c.k.gemvh2[ti];
    if (sff > 0) {
        hip::launch(c.k.f32_to_f16, {cdiv(static_cast<std::uint64_t>(n) * C / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), x16.p(), n * C);
        hip::launch(hv, {cdiv(sff, 16), 1, 1}, {32, 1, 1}, 0, c.s, dsg.p(), SG.row_bytes, x16.p(), sgo.p(), C, sff, n, 0);
        hip::launch(hv, {cdiv(sff, 16), 1, 1}, {32, 1, 1}, 0, c.s, dsu.p(), SU.row_bytes, x16.p(), suo.p(), C, sff, n, 0);
        hip::launch(c.k.silu_mul, {cdiv(ns, 256), 1, 1}, {256, 1, 1}, 0, c.s, sgo.p(), suo.p(), static_cast<int>(ns));
        hip::launch(c.k.f32_to_f16, {cdiv(ns / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, sgo.p(), xs_ref.p(), static_cast<int>(ns));
    }
    // the same chain on gemm_c0 (every gemvh kind / stage count == gemm_c0)
    const bool have_c0 = sff > 0 && c.k.gemmc[0][ti];
    Buf sgc(ns * 4), suc(ns * 4), xs_c0(ns * 2);
    if (have_c0) {
        xs_c0.fill(0xdd);
        gemmC0(c, ti, dsg.p(), SG.row_bytes, x16.p(), sgc.p(), C, sff, n);
        gemmC0(c, ti, dsu.p(), SU.row_bytes, x16.p(), suc.p(), C, sff, n);
        hip::launch(c.k.silu_mul, {cdiv(ns, 256), 1, 1}, {256, 1, 1}, 0, c.s, sgc.p(), suc.p(), static_cast<int>(ns));
        hip::launch(c.k.f32_to_f16, {cdiv(ns / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, sgc.p(), xs_c0.p(), static_cast<int>(ns));
    }
    // the fused launch
    Buf act(na * 4), xs(ns * 2);
    act.fill(0xff);
    xs.fill(0xee);
    const int nsb = sff / 64, nrb = F / 4;
    const hip::Function fu = n <= 16 ? c.k.moe_gus_f : c.k.moe_gus2_f;
    hip::launch(fu, {static_cast<unsigned>(nsb + nrb * n * kK), 1, 1}, {256, 1, 1}, 0, c.s, d1.p(), d2.p(), W1.row_bytes, F, nrb, dh.p(), did.p(),
                act.p(), C, kK, sff > 0 ? dsg.p() : DevPtr{0}, sff > 0 ? dsu.p() : DevPtr{0}, sff > 0 ? SG.row_bytes : std::uint64_t{0}, sff, nsb,
                xs.p(), n);
    c.sync();
    const std::vector<float> a_ref = g.down<float>(na), a_got = act.down<float>(na);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < na; ++i) bad += std::memcmp(&a_ref[i], &a_got[i], 4) != 0;
    c.rep.add(inv("moe_gus_f routed act == moe_gu_f_mxfp4 + silu_mul" + tag, na, bad, ""));
    std::vector<std::uint16_t> s_ref, s_got;
    if (sff > 0) {
        s_ref = xs_ref.down<std::uint16_t>(ns);
        s_got = xs.down<std::uint16_t>(ns);
        std::size_t sb = 0;
        for (std::size_t i = 0; i < ns; ++i) sb += std::memcmp(&s_ref[i], &s_got[i], 2) != 0;
        c.rep.add(inv("moe_gus_f shared f16 act == f32_to_f16 + gemvh x2 + silu_mul + f32_to_f16" + tag + oldTile(c, C), ns, sb, ""));
        if (have_c0) {
            const std::vector<std::uint16_t> s_c0 = xs_c0.down<std::uint16_t>(ns);
            std::size_t cb = 0;
            for (std::size_t i = 0; i < ns; ++i) cb += std::memcmp(&s_c0[i], &s_got[i], 2) != 0;
            c.rep.add(inv("moe_gus_f shared f16 act == f32_to_f16 + gemm_c0 x2 + silu_mul + f32_to_f16 (every gemvh kind / S)" + tag, ns, cb, ""));
        }
    }
    // loose CPU reference (independent decode, double sums): the fused values are the SwiGLU of the right rows
    {
        const std::vector<float> w1 = ref::dequantRows(W1, 0, W1.nrows), w2 = ref::dequantRows(W2, 0, W2.nrows);
        std::vector<float> got;
        std::vector<double> rf, sc;
        for (int p = 0; p < n * kK; p += std::max(1, n * kK / 24)) {
            const int tok = p / kK, e = ids[static_cast<std::size_t>(p)];
            for (int r = 0; r < F; r += std::max(1, F / 16)) {
                double gd = 0, ud = 0, ga = 0, ua = 0;
                for (int i = 0; i < C; ++i) {
                    const double x = h[static_cast<std::size_t>(tok) * C + i];
                    const double pg = w1[(static_cast<std::size_t>(e) * F + r) * C + i] * x, pu = w2[(static_cast<std::size_t>(e) * F + r) * C + i] * x;
                    gd += pg;
                    ud += pu;
                    ga += std::fabs(pg);
                    ua += std::fabs(pu);
                }
                got.push_back(a_got[static_cast<std::size_t>(p) * F + r]);
                rf.push_back(silu(gd) * ud);
                sc.push_back(1.1 * ga * std::fabs(ud) + std::fabs(silu(gd)) * ua);
            }
        }
        c.rep.add(cmpTol("moe_gus_f routed act vs CPU" + tag, got, rf, sc, 1e-5, 1e-6));
    }
    if (sff > 0) {
        const std::vector<float> sgw = ref::dequantRows(SG, 0, SG.nrows), suw = ref::dequantRows(SU, 0, SU.nrows);
        std::vector<float> got;
        std::vector<double> rf, sc;
        for (int tok = 0; tok < n; ++tok)
            for (int r = 0; r < sff; r += std::max(1, sff / 32)) {
                double gd = 0, ud = 0, ga = 0, ua = 0;
                for (int i = 0; i < C; ++i) {
                    // the shared expert sees h rounded to f16 (as the f16 GEMV path does)
                    const double x = roundh(h[static_cast<std::size_t>(tok) * C + i]);
                    const double pg = sgw[static_cast<std::size_t>(r) * C + i] * x, pu = suw[static_cast<std::size_t>(r) * C + i] * x;
                    gd += pg;
                    ud += pu;
                    ga += std::fabs(pg);
                    ua += std::fabs(pu);
                }
                got.push_back(h2f(s_got[static_cast<std::size_t>(tok) * sff + r]));
                const double y = silu(gd) * ud;
                rf.push_back(y);
                sc.push_back(1.1 * ga * std::fabs(ud) + std::fabs(silu(gd)) * ua + std::fabs(y) * 1e3);  // + the f16 output rounding
            }
        c.rep.add(cmpTol("moe_gus_f shared act vs CPU" + tag, got, rf, sc, 1e-5, 1e-6));
    }
}

// gemvhf_<T> / gemvhf2_<T> (feat026 L1): f32 input rounded in the kernel == f32_to_f16 + gemvh_<T> / gemvh2_<T>, bitwise,
// every type that has both, ncols with and without the two-stage LDS round, ragged rows, n over both token tiles,
// accumulate 0 / 1 (y preset to the same random values)
void gemvhfCases(Ctx& c) {
    int ran = 0;
    for (const QType t : wk::kAllTypes) {
        const int ti = static_cast<int>(t);
        if (!c.k.gemvhf[ti] && !c.k.gemvhf2[ti]) continue;
        const std::string sfx = wk::typeSuffix(t);
        for (const int ncols : {256, 768, 2048}) {
            if (ncols % wk::blockInfo(t).values != 0) continue;
            for (const int nrows : {17, 512}) {
                const HostMat m = randomMat(c, t, nrows, ncols);
                Buf w(m.data);
                std::size_t bad = 0, cnt = 0, bad0 = 0, cnt0 = 0;
                std::string first, first0;
                for (const int n : {1, 5, 16, 17, 32}) {
                    const hip::Function hv = n <= 16 ? c.k.gemvh[ti] : c.k.gemvh2[ti];
                    const hip::Function hf = n <= 16 ? c.k.gemvhf[ti] : c.k.gemvhf2[ti];
                    if (!hv || !hf) continue;
                    const std::vector<float> x = c.randn(static_cast<std::size_t>(n) * ncols);
                    const std::vector<float> y0 = c.randn(static_cast<std::size_t>(n) * nrows);
                    Buf dx(x), x16(static_cast<std::size_t>(n) * ncols * 2);
                    for (const int acc : {0, 1}) {
                        Buf ya(y0), yb(y0);
                        hip::launch(c.k.f32_to_f16, {cdiv(static_cast<std::uint64_t>(n) * ncols / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), x16.p(), n * ncols);
                        hip::launch(hv, {cdiv(nrows, 16), 1, 1}, {32, 1, 1}, 0, c.s, w.p(), m.row_bytes, x16.p(), ya.p(), ncols, nrows, n, acc);
                        hip::launch(hf, {cdiv(nrows, 16), 1, 1}, {32, 1, 1}, 0, c.s, w.p(), m.row_bytes, dx.p(), yb.p(), ncols, nrows, n, acc);
                        c.sync();
                        const auto a = ya.down<float>(y0.size()), b = yb.down<float>(y0.size());
                        ++cnt;
                        if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
                            ++bad;
                            if (first.empty()) first = "n " + std::to_string(n) + " acc " + std::to_string(acc);
                        }
                        if (acc == 0 && c.k.gemmc[0][ti]) {  // gemm_c0 (acc 0: y overwritten): every gemvh kind / S
                            Buf yc(y0);
                            gemmC0(c, ti, w.p(), m.row_bytes, x16.p(), yc.p(), ncols, nrows, n);
                            c.sync();
                            const auto r0 = yc.down<float>(y0.size());
                            ++cnt0;
                            if (std::memcmp(r0.data(), b.data(), b.size() * 4) != 0) {
                                ++bad0;
                                if (first0.empty()) first0 = "n " + std::to_string(n);
                            }
                        }
                    }
                }
                ++ran;
                const std::string sh = " ncols=" + std::to_string(ncols) + " nrows=" + std::to_string(nrows);
                c.rep.add(inv("gemvhf_" + sfx + " == f32_to_f16 + gemvh_" + sfx + sh + oldTile(c, ncols), cnt, bad, first));
                if (cnt0 > 0) c.rep.add(inv("gemvhf_" + sfx + " == f32_to_f16 + gemm_c0_" + sfx + " (every gemvh kind / S)" + sh, cnt0, bad0, first0));
            }
        }
    }
    if (ran == 0) c.rep.skip("moefuse", "gemvhf", "kernel absent: no gemvhf_<T> in this code object");
}

}  // namespace

void testMoeFused(Ctx& c) {
    c.rep.family = "moefuse";
    const int ti = static_cast<int>(QType::mxfp4);
    if (!c.k.moe_gus_f || !c.k.moe_gus2_f) {
        c.rep.skip("moefuse", "moe_gus_f_mxfp4", "kernel absent: no fused precise MoE decode kernel in this code object");
        return;
    }
    if (!c.k.moe_gu_f[ti] || !c.k.gemvh[ti] || !c.k.gemvh2[ti] || !c.k.silu_mul || !c.k.f32_to_f16) {
        c.rep.skip("moefuse", "moe_gus_f_mxfp4", "kernel absent: the reference chain (moe_gu_f / gemvh / silu_mul / f32_to_f16)");
        return;
    }
    gemvhfCases(c);
    const int ns_all[] = {1, 2, 3, 8, 15, 16, 17, 24, 31, 32};
    for (const int n : ns_all) fuseCase(c, 2048, 512, 512, n);
    for (const int n : {1, 9, 16, 32}) fuseCase(c, 768, 64, 64, n);
    // r1 (review M1): C % 512 != 0 at the Ornith expert widths (old gfx1201 gemvh S = 1, more than one 256-k round)
    for (const int n : {1, 16, 17, 32}) fuseCase(c, 1280, 512, 512, n);
    for (const int n : {1, 4, 20}) fuseCase(c, 2048, 512, 0, n);  // no shared expert
    if (!c.quick)
        for (const int n : {1, 5, 16, 18}) fuseCase(c, 256, 128, 128, n);
    if (!c.quick)
        for (const int n : {1, 32}) fuseCase(c, 2304, 512, 512, n);  // r1: C % 512 == 256 with 9 rounds
}

}  // namespace kt
