// whirl-kernel-test "moef": precise-mode MoE decode / verify experts (kernels/moe_f.hip,
// FIX-K97 / FR-K97 P2-2), every type with a moe_gu_f_<T> / moe_down_f_<T> kernel:
//   * moe_gu_f_<T> (gate rows [0, ff) from W1, up rows from W2) and moe_down_f_<T>
//     (K experts weighted + shared expert + residual) vs an independent CPU reference
//     (weights decoded exactly by ref::dequantRow, f32 activations, double accumulation);
//   * boundary shapes: C with fewer 8-value groups than lanes, ff / E rows not a multiple of
//     the 8 rows per block, one row; the shared expert absent (ysh = null);
//   * the stated precise guarantee: a pair's result depends only on (token, expert) - the
//     rows of an n-token launch (n = 1..16, prefix and offset windows) are bitwise the same
//     rows of the 32-token launch;
//   * the code's "as gemv_<T>_1" claim (one wave per row, groups g = lane, lane + 32, ...,
//     warp_sum): moe_gu_f_<T> rows == gemv_<T>_1 on the expert's rows, and moe_down_f_<T>
//     with K = 1, w = 1, no shared expert, x = 0 == gemv_<T>_1, bitwise (not MXFP4: its
//     whole-block kernel has its own order).
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "cpu_ref.h"

namespace kt {

namespace {

Result inv(const std::string& name, std::size_t n, std::size_t bad) {
    Result r;
    r.name = name;
    r.kind = Kind::invariant;
    r.n = n;
    r.mismatches = bad;
    r.pass = bad == 0;
    return r;
}

// K distinct experts per token, positive weights summing to 1
void routing(Ctx& c, int n, int R, int K, std::vector<int>& ids, std::vector<float>& w) {
    ids.assign(static_cast<std::size_t>(n) * K, 0);
    w.assign(ids.size(), 0.f);
    std::uniform_int_distribution<int> ur(0, R - 1);
    std::uniform_real_distribution<float> uw(0.05f, 1.f);
    for (int t = 0; t < n; ++t) {
        float s = 0;
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = ur(c.rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup = dup || ids[static_cast<std::size_t>(t) * K + j] == e;
            } while (dup);
            ids[static_cast<std::size_t>(t) * K + k] = e;
            w[static_cast<std::size_t>(t) * K + k] = uw(c.rng);
            s += w[static_cast<std::size_t>(t) * K + k];
        }
        for (int k = 0; k < K; ++k) w[static_cast<std::size_t>(t) * K + k] /= s;
    }
}

int minCols(QType t) {
    const int bv = wk::blockInfo(t).values;
    // fewer 8-value groups than the 32 lanes (idle lanes) where the block allows it
    return bv <= 8 ? 40 : bv <= 32 ? 96 : 256;
}

constexpr int kR = 12, kK = 8, kN = 32;

// gate + up: (C, ff) shape
void guCase(Ctx& c, QType t, int C, int ff) {
    const int ti = static_cast<int>(t);
    const std::string sfx = wk::typeSuffix(t);
    const std::string tag = " C=" + std::to_string(C) + " ff=" + std::to_string(ff);
    const HostMat W1 = randomMat(c, t, kR * ff, C), W2 = randomMat(c, t, kR * ff, C);
    Buf d1(W1.data), d2(W2.data);
    std::vector<int> ids;
    std::vector<float> w;
    routing(c, kN, kR, kK, ids, w);
    const std::vector<float> h = c.randn(static_cast<std::size_t>(kN) * C);
    auto run = [&](int t0, int n) {
        std::vector<int> id(ids.begin() + static_cast<std::ptrdiff_t>(t0) * kK, ids.begin() + static_cast<std::ptrdiff_t>(t0 + n) * kK);
        std::vector<float> hh(h.begin() + static_cast<std::ptrdiff_t>(t0) * C, h.begin() + static_cast<std::ptrdiff_t>(t0 + n) * C);
        Buf did(id), dh(hh), y1(static_cast<std::size_t>(n) * kK * ff * 4), y2(static_cast<std::size_t>(n) * kK * ff * 4);
        y1.fill(0xff);
        y2.fill(0xff);
        hip::launch(c.k.moe_gu_f[ti], {cdiv(2 * ff, 8), static_cast<unsigned>(n * kK), 1}, {256, 1, 1}, 0, c.s, d1.p(), d2.p(), W1.row_bytes, ff,
                    2 * ff, dh.p(), did.p(), y1.p(), y2.p(), C, kK);
        c.sync();
        std::vector<float> o = y1.down<float>(static_cast<std::size_t>(n) * kK * ff), u = y2.down<float>(o.size());
        o.insert(o.end(), u.begin(), u.end());  // [gate pairs][ff] then [up pairs][ff]
        return o;
    };
    const std::vector<float> full = run(0, kN);
    const std::size_t half = static_cast<std::size_t>(kN) * kK * ff;
    // CPU reference (every pair, every row)
    {
        const std::vector<float> w1 = ref::dequantRows(W1, 0, W1.nrows), w2 = ref::dequantRows(W2, 0, W2.nrows);
        std::vector<float> got;
        std::vector<double> rf, sc;
        for (int p = 0; p < kN * kK; ++p) {
            const int tok = p / kK, e = ids[static_cast<std::size_t>(p)];
            for (int m = 0; m < 2; ++m)
                for (int r = 0; r < ff; ++r) {
                    const float* wr = (m ? w2 : w1).data() + (static_cast<std::size_t>(e) * ff + r) * C;
                    double y = 0, a = 0;
                    for (int i = 0; i < C; ++i) {
                        const double pr = static_cast<double>(wr[i]) * h[static_cast<std::size_t>(tok) * C + i];
                        y += pr;
                        a += std::fabs(pr);
                    }
                    got.push_back(full[(m ? half : 0) + static_cast<std::size_t>(p) * ff + r]);
                    rf.push_back(y);
                    sc.push_back(a);
                }
        }
        c.rep.add(cmpTol("moe_gu_f_" + sfx + " vs CPU" + tag, got, rf, sc, 1e-5, 1e-7));
    }
    // batch invariance: n = 1..16 prefix, plus offset windows, == the 32-token rows
    {
        std::size_t bad = 0, cnt = 0;
        auto cmpWin = [&](int t0, int n) {
            const std::vector<float> o = run(t0, n);
            const std::size_t hn = static_cast<std::size_t>(n) * kK * ff;
            for (int m = 0; m < 2; ++m) {
                bad += std::memcmp(o.data() + (m ? hn : 0), full.data() + (m ? half : 0) + static_cast<std::size_t>(t0) * kK * ff, hn * 4) != 0;
                ++cnt;
            }
        };
        for (int n = 1; n <= 16; ++n) cmpWin(0, n);
        cmpWin(9, 7);
        cmpWin(16, 16);
        cmpWin(31, 1);
        c.rep.add(inv("moe_gu_f_" + sfx + ": n = 1..16 / offset windows == 32-token rows" + tag, cnt, bad));
    }
    // == gemv_<T>_1 on the expert's rows (the code's stated reduction order)
    if (t != QType::mxfp4 && c.k.gemv1[ti]) {
        std::size_t bad = 0, cnt = 0;
        Buf dh(h), y(static_cast<std::size_t>(ff) * 4);
        for (int p = 0; p < kN * kK; p += 13) {
            const int tok = p / kK, e = ids[static_cast<std::size_t>(p)];
            for (int m = 0; m < 2; ++m) {
                const DevPtr Wp = (m ? d2.p() : d1.p()) + static_cast<std::uint64_t>(e) * ff * W1.row_bytes;
                hip::launch(c.k.gemv1[ti], {cdiv(ff, 8), 1, 1}, {256, 1, 1}, 0, c.s, Wp, W1.row_bytes,
                            dh.p() + static_cast<std::uint64_t>(tok) * C * 4, y.p(), C, ff, C, ff, 0);
                c.sync();
                const auto g = y.down<float>(static_cast<std::size_t>(ff));
                bad += std::memcmp(g.data(), full.data() + (m ? half : 0) + static_cast<std::size_t>(p) * ff, static_cast<std::size_t>(ff) * 4) != 0;
                ++cnt;
            }
        }
        c.rep.add(inv("moe_gu_f_" + sfx + " == gemv_" + sfx + "_1 per expert row" + tag, cnt, bad));
    }
}

// down + combine: (E rows, F cols)
void downCase(Ctx& c, QType t, int E, int F, bool shared) {
    const int ti = static_cast<int>(t);
    const std::string sfx = wk::typeSuffix(t);
    const std::string tag = " E=" + std::to_string(E) + " F=" + std::to_string(F) + (shared ? "" : " no shared");
    const HostMat W = randomMat(c, t, kR * E, F);
    Buf dW(W.data);
    std::vector<int> ids;
    std::vector<float> w;
    routing(c, kN, kR, kK, ids, w);
    const std::vector<float> act = c.randn(static_cast<std::size_t>(kN) * kK * F), x0 = c.randn(static_cast<std::size_t>(kN) * E),
                             ysh = c.randn(static_cast<std::size_t>(kN) * E), sg = c.randu(kN, 0.f, 1.f);
    auto run = [&](int t0, int n) {
        auto sl = [&](const auto& v, std::size_t per) {
            using V = std::decay_t<decltype(v)>;
            return V(v.begin() + static_cast<std::ptrdiff_t>(t0 * per), v.begin() + static_cast<std::ptrdiff_t>((t0 + n) * per));
        };
        Buf did(sl(ids, kK)), dw(sl(w, kK)), da(sl(act, static_cast<std::size_t>(kK) * F)), dx(sl(x0, E)), dys(sl(ysh, E)), dsg(sl(sg, 1));
        hip::launch(c.k.moe_down_f[ti], {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dW.p(), W.row_bytes, E, da.p(), did.p(),
                    dw.p(), dsg.p(), shared ? dys.p() : DevPtr{0}, dx.p(), F, kK);
        c.sync();
        return dx.down<float>(static_cast<std::size_t>(n) * E);
    };
    const std::vector<float> full = run(0, kN);
    {
        const std::vector<float> wf = ref::dequantRows(W, 0, W.nrows);
        std::vector<float> got;
        std::vector<double> rf, sc;
        for (int tok = 0; tok < kN; ++tok)
            for (int r = 0; r < E; ++r) {
                double acc = 0, a = 0;
                for (int k = 0; k < kK; ++k) {
                    const int p = tok * kK + k;
                    const float* wr = wf.data() + (static_cast<std::size_t>(ids[static_cast<std::size_t>(p)]) * E + r) * F;
                    double s = 0, sa = 0;
                    for (int i = 0; i < F; ++i) {
                        const double pr = static_cast<double>(wr[i]) * act[static_cast<std::size_t>(p) * F + i];
                        s += pr;
                        sa += std::fabs(pr);
                    }
                    acc += static_cast<double>(w[static_cast<std::size_t>(p)]) * s;
                    a += static_cast<double>(w[static_cast<std::size_t>(p)]) * sa;
                }
                const std::size_t o = static_cast<std::size_t>(tok) * E + r;
                const double sh = shared ? static_cast<double>(sg[static_cast<std::size_t>(tok)]) * ysh[o] : 0.0;
                got.push_back(full[o]);
                rf.push_back(x0[o] + acc + sh);
                sc.push_back(std::fabs(x0[o]) + a + std::fabs(sh));
            }
        c.rep.add(cmpTol("moe_down_f_" + sfx + " (+ shared, residual) vs CPU" + tag, got, rf, sc, 1e-5, 1e-7));
    }
    {
        std::size_t bad = 0, cnt = 0;
        auto cmpWin = [&](int t0, int n) {
            const std::vector<float> o = run(t0, n);
            bad += std::memcmp(o.data(), full.data() + static_cast<std::size_t>(t0) * E, o.size() * 4) != 0;
            ++cnt;
        };
        for (int n = 1; n <= 16; ++n) cmpWin(0, n);
        cmpWin(9, 7);
        cmpWin(16, 16);
        cmpWin(31, 1);
        c.rep.add(inv("moe_down_f_" + sfx + ": n = 1..16 / offset windows == 32-token rows" + tag, cnt, bad));
    }
    // K = 1, w = 1, no shared expert, x = 0: == gemv_<T>_1 on the expert's rows
    if (t != QType::mxfp4 && c.k.gemv1[ti] && shared) {
        std::size_t bad = 0, cnt = 0;
        Buf da(act), y(static_cast<std::size_t>(E) * 4), x(static_cast<std::size_t>(E) * 4);
        const std::vector<float> one{1.f};
        Buf d1(one);
        for (int p = 0; p < kN * kK; p += 29) {
            const int e = ids[static_cast<std::size_t>(p)];
            const DevPtr ap = da.p() + static_cast<std::uint64_t>(p) * F * 4;
            Buf did(std::vector<int>{e});
            x.zero();
            hip::launch(c.k.moe_down_f[ti], {cdiv(E, 8), 1, 1}, {256, 1, 1}, 0, c.s, dW.p(), W.row_bytes, E, ap, did.p(), d1.p(), d1.p(), DevPtr{0},
                        x.p(), F, 1);
            hip::launch(c.k.gemv1[ti], {cdiv(E, 8), 1, 1}, {256, 1, 1}, 0, c.s, dW.p() + static_cast<std::uint64_t>(e) * E * W.row_bytes, W.row_bytes,
                        ap, y.p(), F, E, F, E, 0);
            c.sync();
            const auto gx = x.down<float>(static_cast<std::size_t>(E)), gy = y.down<float>(static_cast<std::size_t>(E));
            for (int r = 0; r < E; ++r) {
                // 0 + v: only an exact -0 sum would differ in bits; compare values
                bad += gx[static_cast<std::size_t>(r)] != gy[static_cast<std::size_t>(r)];
                ++cnt;
            }
        }
        c.rep.add(inv("moe_down_f_" + sfx + " (K = 1, w = 1) == gemv_" + sfx + "_1" + tag, cnt, bad));
    }
}

}  // namespace

void testMoeF(Ctx& c) {
    c.rep.family = "moef";
    int ran = 0;
    for (const QType t : wk::kAllTypes) {
        const int ti = static_cast<int>(t);
        const std::string sfx = wk::typeSuffix(t);
        if (!c.k.moe_gu_f[ti] || !c.k.moe_down_f[ti]) {
            c.rep.skip("moef", std::string("moe_gu_f_") + sfx, "no precise MoE expert kernel for this type");
            continue;
        }
        ++ran;
        const int cmin = minCols(t);
        guCase(c, t, cmin, 7);
        guCase(c, t, 768, 100);
        if (!c.quick) guCase(c, t, 2048, 64);
        downCase(c, t, 7, cmin, true);
        downCase(c, t, 100, 768, true);
        downCase(c, t, 1, 256, false);
        if (!c.quick) downCase(c, t, 300, 512, true);
    }
    if (ran == 0) {
        Result r;
        r.name = "moe_gu_f / moe_down_f: no kernel bound";
        r.pass = false;
        c.rep.add(r);
    }
}

// whirl-kernel-test "moefx" (feat027-moeskip-r2, review moeskip-attndec8s-r1 A3): cross-object bit identity of the
// precise decode experts. WHIRL_KT_MOEF_HSACO = a code object of the device's arch built from an older moe_f.hip
// (scratch\f027mr2: b8a041a's kernels, CMake flags). For every type with moe_gu_f_<T> / moe_down_f_<T> in both
// objects, the same inputs and the same launch geometry run through both; the outputs are compared as raw 32-bit
// words (cmpExact on uint32: -0 and NaN payloads count):
//   * ids >= 0 (distinct per token), K in {1, 2, 4, 6, 8}, two shapes, down with and without the shared expert:
//     new == old, bitwise ("no-op for ids >= 0");
//   * id -1 suffix (kept prefix kt_t in [1, K], K in {2, 4, 6, 8}): gate+up kept pairs == old on the full ids,
//     dropped pairs untouched (0xff); down (NaN activations and weights on dropped slots) == old run per token on the
//     compacted kept slots (K' = kt_t, same order: bitwise, no tolerance);
//   * sensitivity control: the K = 8 down with the slots in reverse order must differ from the normal order in some
//     word (a bitwise compare that can see a reduction-order change; a pass here means it can).
// Without the env var the family records one skip (it is a cross-object check, not a property of one object).
void testMoeFx(Ctx& c) {
    c.rep.family = "moefx";
    const char* xp = std::getenv("WHIRL_KT_MOEF_HSACO");
    if (xp == nullptr || *xp == 0) {
        c.rep.skip("moefx", "moe_gu_f / moe_down_f", "WHIRL_KT_MOEF_HSACO not set (cross-object check)");
        return;
    }
    std::vector<char> img;
    if (std::FILE* fp = std::fopen(xp, "rb")) {
        std::fseek(fp, 0, SEEK_END);
        img.resize(static_cast<std::size_t>(std::ftell(fp)));
        std::fseek(fp, 0, SEEK_SET);
        if (std::fread(img.data(), 1, img.size(), fp) != img.size()) img.clear();
        std::fclose(fp);
    }
    if (img.empty()) {
        Result r;
        r.name = std::string("WHIRL_KT_MOEF_HSACO: cannot read ") + xp;
        r.pass = false;
        c.rep.add(r);
        return;
    }
    const hip::Module xmod = hip::Module::loadData(img.data(), c.mod.arch());
    std::printf("   moefx: old moe_gu_f_* / moe_down_f_* from %s (%zu bytes)\n", xp, img.size());
    std::fflush(stdout);
    constexpr int R = 16, n = 9;
    using U = std::uint32_t;
    int ran = 0;
    std::size_t sens_cnt = 0, sens_diff = 0;
    for (const QType t : wk::kAllTypes) {
        const int ti = static_cast<int>(t);
        const std::string sfx = wk::typeSuffix(t);
        if (!c.k.moe_gu_f[ti] || !c.k.moe_down_f[ti]) continue;  // no precise expert kernel (moef reports these)
        const hip::Function ogu = xmod.getFunctionOpt(("moe_gu_f_" + sfx).c_str());
        const hip::Function odn = xmod.getFunctionOpt(("moe_down_f_" + sfx).c_str());
        if (!ogu || !odn) {
            Result r;
            r.name = "moe_gu_f_" + sfx + " / moe_down_f_" + sfx + " missing in the old object (new has them)";
            r.pass = false;
            c.rep.add(r);
            continue;
        }
        ++ran;
        const hip::Function ngu = c.k.moe_gu_f[ti], ndn = c.k.moe_down_f[ti];
        struct Shape {
            int C, ff, E, F;
        };
        for (const Shape sh : {Shape{768, 100, 100, 512}, Shape{2048, 64, 300, 768}}) {
            const std::string tag = " C=" + std::to_string(sh.C) + " ff=" + std::to_string(sh.ff) + " E=" + std::to_string(sh.E) +
                                    " F=" + std::to_string(sh.F);
            const HostMat W1 = randomMat(c, t, R * sh.ff, sh.C), W2 = randomMat(c, t, R * sh.ff, sh.C), Wd = randomMat(c, t, R * sh.E, sh.F);
            Buf d1(W1.data), d2(W2.data), dd(Wd.data);
            const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * sh.C);
            const std::vector<float> x0 = c.randn(static_cast<std::size_t>(n) * sh.E), ysh = c.randn(static_cast<std::size_t>(n) * sh.E),
                                     sg = c.randu(n, 0.f, 1.f);
            Buf dh(h), dysh(ysh), dsg(sg);
            for (const int K : {1, 2, 4, 6, 8}) {
                const std::string kt = " K=" + std::to_string(K) + tag;
                std::vector<int> ids;
                std::vector<float> w;
                routing(c, n, R, K, ids, w);
                const std::size_t nk = static_cast<std::size_t>(n) * K, ng = nk * sh.ff, na = nk * sh.F, nx = static_cast<std::size_t>(n) * sh.E;
                const std::vector<float> act = c.randn(na);
                Buf did(ids), dw(w), da(act);
                auto gu = [&](const hip::Function& f, const Buf& id, Buf& y1, Buf& y2) {
                    y1.fill(0xff);
                    y2.fill(0xff);
                    hip::launch(f, {cdiv(2 * sh.ff, 8), static_cast<unsigned>(nk), 1}, {256, 1, 1}, 0, c.s, d1.p(), d2.p(), W1.row_bytes, sh.ff,
                                2 * sh.ff, dh.p(), id.p(), y1.p(), y2.p(), sh.C, K);
                };
                // ---- ids >= 0: gate + up
                {
                    Buf n1(ng * 4), n2(ng * 4), o1(ng * 4), o2(ng * 4);
                    gu(ngu, did, n1, n2);
                    gu(ogu, did, o1, o2);
                    c.sync();
                    std::vector<U> gn = n1.down<U>(ng), go = o1.down<U>(ng);
                    const std::vector<U> un = n2.down<U>(ng), uo = o2.down<U>(ng);
                    gn.insert(gn.end(), un.begin(), un.end());
                    go.insert(go.end(), uo.begin(), uo.end());
                    c.rep.add(cmpExact("moe_gu_f_" + sfx + " ids>=0 new == b8a041a (bitwise)" + kt, gn, go));
                }
                // ---- ids >= 0: down + combine, with and without the shared expert
                for (const bool shared : {true, false}) {
                    Buf xn(x0), xo(x0);
                    auto dn = [&](const hip::Function& f, Buf& x) {
                        hip::launch(f, {cdiv(sh.E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), Wd.row_bytes, sh.E, da.p(), did.p(),
                                    dw.p(), dsg.p(), shared ? dysh.p() : DevPtr{0}, x.p(), sh.F, K);
                    };
                    dn(ndn, xn);
                    dn(odn, xo);
                    c.sync();
                    c.rep.add(cmpExact("moe_down_f_" + sfx + " ids>=0 new == b8a041a (bitwise)" + kt + (shared ? "" : " no shared"), xn.down<U>(nx),
                                       xo.down<U>(nx)));
                }
                // ---- sensitivity control (K = 8): reversed slot order must change some bits
                if (K == 8) {
                    std::vector<int> ir(ids);
                    std::vector<float> wr(w), ar(act);
                    for (int tok = 0; tok < n; ++tok)
                        for (int k = 0; k < K; ++k) {
                            const std::size_t p = static_cast<std::size_t>(tok) * K + k, q = static_cast<std::size_t>(tok) * K + (K - 1 - k);
                            ir[p] = ids[q];
                            wr[p] = w[q];
                            std::copy(act.begin() + static_cast<std::ptrdiff_t>(q * sh.F), act.begin() + static_cast<std::ptrdiff_t>((q + 1) * sh.F),
                                      ar.begin() + static_cast<std::ptrdiff_t>(p * sh.F));
                        }
                    Buf dir(ir), dwr(wr), dar(ar), xa(x0), xb(x0);
                    hip::launch(ndn, {cdiv(sh.E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), Wd.row_bytes, sh.E, da.p(), did.p(), dw.p(),
                                dsg.p(), DevPtr{0}, xa.p(), sh.F, K);
                    hip::launch(ndn, {cdiv(sh.E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), Wd.row_bytes, sh.E, dar.p(), dir.p(), dwr.p(),
                                dsg.p(), DevPtr{0}, xb.p(), sh.F, K);
                    c.sync();
                    const auto a = xa.down<U>(nx), b = xb.down<U>(nx);
                    for (std::size_t i = 0; i < nx; ++i) sens_diff += a[i] != b[i];
                    sens_cnt += nx;
                }
                if (K == 1) continue;
                // ---- id -1 suffix: kept prefix kt_t in [1, K] (token 0 keeps all K, token 1 keeps 1)
                std::vector<int> ids_s(ids), keep(static_cast<std::size_t>(n));
                std::vector<float> w_s(w), act_s(act);
                for (int tok = 0; tok < n; ++tok) {
                    const int kp = tok == 0 ? K : tok == 1 ? 1 : 1 + static_cast<int>(c.rng() % static_cast<std::uint64_t>(K));
                    keep[static_cast<std::size_t>(tok)] = kp;
                    for (int k = kp; k < K; ++k) {
                        const std::size_t p = static_cast<std::size_t>(tok) * K + k;
                        ids_s[p] = -1;
                        w_s[p] = std::nanf("");
                        std::fill(act_s.begin() + static_cast<std::ptrdiff_t>(p * sh.F), act_s.begin() + static_cast<std::ptrdiff_t>((p + 1) * sh.F),
                                  std::nanf(""));
                    }
                }
                Buf dids(ids_s), dws(w_s), das(act_s);
                {
                    Buf n1(ng * 4), n2(ng * 4), o1(ng * 4), o2(ng * 4);
                    gu(ngu, dids, n1, n2);
                    gu(ogu, did, o1, o2);  // old on the full ids: the kept pairs' reference
                    c.sync();
                    const auto gn = n1.down<U>(ng), go = o1.down<U>(ng), un = n2.down<U>(ng), uo = o2.down<U>(ng);
                    std::size_t bad = 0;
                    for (std::size_t p = 0; p < nk; ++p)
                        for (int r = 0; r < sh.ff; ++r) {
                            const std::size_t i = p * sh.ff + r;
                            if (ids_s[p] < 0) bad += gn[i] != 0xffffffffu || un[i] != 0xffffffffu;
                            else bad += gn[i] != go[i] || un[i] != uo[i];
                        }
                    c.rep.add(inv("moe_gu_f_" + sfx + " id -1: dropped untouched, kept == b8a041a (bitwise)" + kt, 2 * ng, bad));
                }
                {
                    // old: one launch per token on its compacted kept slots (K' = kt_t, grid y = 1, pointers offset)
                    std::vector<int> ic;
                    std::vector<float> wc, ac;
                    std::vector<std::size_t> off(static_cast<std::size_t>(n));
                    for (int tok = 0; tok < n; ++tok) {
                        off[static_cast<std::size_t>(tok)] = ic.size();
                        for (int k = 0; k < keep[static_cast<std::size_t>(tok)]; ++k) {
                            const std::size_t p = static_cast<std::size_t>(tok) * K + k;
                            ic.push_back(ids[p]);
                            wc.push_back(w[p]);
                            ac.insert(ac.end(), act.begin() + static_cast<std::ptrdiff_t>(p * sh.F), act.begin() + static_cast<std::ptrdiff_t>((p + 1) * sh.F));
                        }
                    }
                    Buf dic(ic), dwc(wc), dac(ac);
                    for (const bool shared : {true, false}) {
                        Buf xn(x0), xo(x0);
                        hip::launch(ndn, {cdiv(sh.E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), Wd.row_bytes, sh.E, das.p(), dids.p(),
                                    dws.p(), dsg.p(), shared ? dysh.p() : DevPtr{0}, xn.p(), sh.F, K);
                        for (int tok = 0; tok < n; ++tok) {
                            const std::uint64_t o = off[static_cast<std::size_t>(tok)], ro = static_cast<std::uint64_t>(tok) * sh.E * 4;
                            hip::launch(odn, {cdiv(sh.E, 8), 1, 1}, {256, 1, 1}, 0, c.s, dd.p(), Wd.row_bytes, sh.E, dac.p() + o * sh.F * 4, dic.p() + o * 4,
                                        dwc.p() + o * 4, dsg.p() + static_cast<std::uint64_t>(tok) * 4, shared ? dysh.p() + ro : DevPtr{0}, xo.p() + ro, sh.F,
                                        keep[static_cast<std::size_t>(tok)]);
                        }
                        c.sync();
                        c.rep.add(cmpExact("moe_down_f_" + sfx + " id -1 (NaN dropped) new == b8a041a compacted (bitwise)" + kt + (shared ? "" : " no shared"),
                                           xn.down<U>(nx), xo.down<U>(nx)));
                    }
                }
            }
        }
    }
    if (ran == 0) {
        Result r;
        r.name = "moefx: no moe_gu_f / moe_down_f kernel bound";
        r.pass = false;
        c.rep.add(r);
    }
    c.rep.add(inv("moefx sensitivity: K=8 down reversed slot order differs (words differing " + std::to_string(sens_diff) + ")", sens_cnt,
                  sens_diff > 0 ? 0 : 1));
}

}  // namespace kt
