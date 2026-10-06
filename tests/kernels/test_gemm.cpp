// whirl-kernel-test: prefill GEMMs.
//   * f32_to_f16 exact;
//   * every prefill GEMM choice (gemm_c<i>_<T>, dequant_f16 + gemm_c<i>_f16,
//     gemms_c<i>_<T>, gemmhq_<T>, gemmh_f16) gives the same bits as
//     gemm_c0_<T> on a 200-token batch, and the first 40 tokens alone give the
//     same rows (the prototype's checkPrefillInvariance);
//   * f16-output twins (gemm_ch*, gemmsh*, gemmhqh*, gemmhh_f16) == f16 of
//     the f32 result;
//   * gemm_c0, gemm3, gemm_wmma (f16 x f16 -> f32) and gemm_<T> (f32 X) vs
//     a double-precision CPU product (tolerance).
// SPDX-License-Identifier: Apache-2.0

#include "cpu_ref.h"

#include <functional>

namespace kt {

namespace {

struct Gemm {
    Ctx& c;
    const HostMat& m;
    const Buf& w;
    const Buf& x16;
    int n;

    std::vector<float> run(hip::Function f, int bm, int bn, int nth, int ntok, bool tok_x = true) {
        Buf y(static_cast<std::size_t>(ntok) * m.nrows * 4);
        y.fill(0xff);
        const hip::Dim3 grid = tok_x ? hip::Dim3{cdiv(ntok, bn), cdiv(m.nrows, bm), 1} : hip::Dim3{cdiv(m.nrows, bm), cdiv(ntok, bn), 1};
        hip::launch(f, grid, {static_cast<unsigned>(nth), 1, 1}, 0, c.s, w.p(), m.row_bytes, x16.p(), y.p(), m.ncols,
                    m.nrows, ntok, 0);
        c.sync();
        return y.down<float>(static_cast<std::size_t>(ntok) * m.nrows);
    }
    std::vector<std::uint16_t> runH(hip::Function f, DevPtr wp, std::uint64_t rb, int bm, int bn, int nth) {
        Buf y(static_cast<std::size_t>(n) * m.nrows * 2);
        y.fill(0xff);
        hip::launch(f, {cdiv(n, bn), cdiv(m.nrows, bm), 1}, {static_cast<unsigned>(nth), 1, 1}, 0, c.s, wp, rb, x16.p(),
                    y.p(), m.ncols, m.nrows, n, 0);
        c.sync();
        return y.down<std::uint16_t>(static_cast<std::size_t>(n) * m.nrows);
    }
};

std::vector<std::uint16_t> toHalf(const std::vector<float>& v) {
    std::vector<std::uint16_t> h(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) h[i] = f2h(v[i]);
    return h;
}

}  // namespace

void testGemm(Ctx& c) {
    c.rep.family = "gemm";
    const int n = 200, nsub = 40;
    const std::vector<HostMat> mats = sampleMats(c, c.quick ? 512 : 2048);
    for (const HostMat& m : mats) {
        const int ti = static_cast<int>(m.type);
        const std::string sfx = wk::typeSuffix(m.type);
        const int ncols = m.ncols, nrows = m.nrows;
        Buf w(m.data);
        const std::vector<float> x = c.randn(static_cast<std::size_t>(n) * ncols);
        Buf dx(x), x16(static_cast<std::size_t>(n) * ncols * 2);
        hip::launch(c.k.f32_to_f16, {cdiv(static_cast<std::uint64_t>(n) * ncols / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s,
                    dx.p(), x16.p(), n * ncols);
        c.sync();
        const std::vector<std::uint16_t> hx16 = toHalf(x);
        c.rep.add(cmpExact("f32_to_f16 (" + std::to_string(n) + "x" + std::to_string(ncols) + ")",
                           x16.down<std::uint16_t>(hx16.size()), hx16));
        Gemm g{c, m, w, x16, n};

        // f16 weights (what dequant_f16 / the fused dequant produce)
        Buf w16(static_cast<std::size_t>(nrows) * ncols * 2);
        if (m.type != QType::f16) {
            const std::uint64_t groups = static_cast<std::uint64_t>(ncols / 8) * nrows;
            hip::launch(c.k.dequant_f16[ti], {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(groups, 256), 65535)), 1, 1},
                        {256, 1, 1}, 0, c.s, w.p(), m.row_bytes, w16.p(), ncols, nrows);
        } else {
            w16.up(m.data);
        }
        const std::uint64_t rb16 = static_cast<std::uint64_t>(ncols) * 2;

        // reference choice: gemm_c0_<T>
        const auto& c0 = wk::kGemmCfgs[0];
        const std::vector<float> ref = g.run(c.k.gemmc[0][ti], c0.bm, c0.bn, c0.nth, n);

        // CPU tolerance check of gemm_c0 on a row / token subset
        {
            const int cr = std::min(nrows, c.quick ? 64 : 256), ct = c.quick ? 8 : 24;
            std::vector<float> wf(static_cast<std::size_t>(cr) * ncols);
            std::vector<std::uint16_t> wh(static_cast<std::size_t>(ncols));
            for (int r = 0; r < cr; ++r) {
                ref::dequantRowF16(m.type, m.data.data() + static_cast<std::size_t>(r) * m.row_bytes, ncols, wh.data());
                for (int k = 0; k < ncols; ++k) wf[static_cast<std::size_t>(r) * ncols + k] = h2f(wh[static_cast<std::size_t>(k)]);
            }
            std::vector<float> xf(static_cast<std::size_t>(ct) * ncols);
            for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = h2f(hx16[i]);
            std::vector<double> ry, rs;
            ref::gemvF64(wf, cr, ncols, xf.data(), ct, ncols, ry, rs);
            auto pick = [&](const std::vector<float>& y) {
                std::vector<float> got;
                for (int t = 0; t < ct; ++t)
                    got.insert(got.end(), y.begin() + static_cast<std::ptrdiff_t>(t) * nrows,
                               y.begin() + static_cast<std::ptrdiff_t>(t) * nrows + cr);
                return got;
            };
            c.rep.add(cmpTol("gemm_c0_" + sfx + " vs CPU " + m.name, pick(ref), ry, rs, 1e-4, 1e-6));
            if (ncols % 16 == 0) {
                const auto y3 = g.run(c.k.gemm[ti], wk::kGemm3Bm, wk::kGemm3Bn, wk::kGemm3Threads, n, false);
                c.rep.add(cmpTol("gemm3_" + sfx + " vs CPU", pick(y3), ry, rs, 1e-4, 1e-6));
            }
            if (auto fw = c.fnOpt("gemm_wmma_" + sfx); fw && ncols % 64 == 0) {
                const auto yw = g.run(fw, 128, 128, 256, n, false);
                c.rep.add(cmpTol("gemm_wmma_" + sfx + " vs CPU", pick(yw), ry, rs, 1e-4, 1e-6));
            }
            // f32-activation scalar GEMM on exact f32 weights
            if (ncols % 256 == 0) {
                const std::vector<float> wx = ref::dequantRows(m, 0, cr);
                std::vector<double> ry2, rs2;
                ref::gemvF64(wx, cr, ncols, x.data(), ct, ncols, ry2, rs2);
                Buf y(static_cast<std::size_t>(n) * nrows * 4);
                hip::launch(c.fn("gemm_" + sfx), {cdiv(nrows, 16), cdiv(n, 32), 1}, {256, 1, 1}, 0, c.s, w.p(),
                            m.row_bytes, dx.p(), y.p(), ncols, nrows, n, 0);
                c.sync();
                c.rep.add(cmpTol("gemm_" + sfx + " (f32 x) vs CPU", pick(y.down<float>(static_cast<std::size_t>(n) * nrows)),
                                 ry2, rs2, 1e-4, 1e-6));
            }
        }

        // every choice == choice 0 (bitwise)
        std::size_t checked = 0, bad = 0;
        std::string first_bad;
        auto check = [&](const std::vector<float>& y, const std::string& label) {
            ++checked;
            if (std::memcmp(y.data(), ref.data(), ref.size() * 4) != 0) {
                ++bad;
                if (first_bad.empty()) first_bad = label;
            }
        };
        for (int ci = 0; ci < static_cast<int>(wk::kGemmCfgs.size()); ++ci) {
            const auto& cf = wk::kGemmCfgs[static_cast<std::size_t>(ci)];
            if (ci > 0 && c.k.gemmc[ci][ti]) check(g.run(c.k.gemmc[ci][ti], cf.bm, cf.bn, cf.nth, n), "gemm_c" + std::to_string(ci));
            if (m.type != QType::f16 && c.k.gemmc[ci][static_cast<int>(QType::f16)]) {
                // dequant + f16 GEMM (the gemm_c<i>_f16 kernel on w16)
                Buf y(static_cast<std::size_t>(n) * nrows * 4);
                y.fill(0xff);
                hip::launch(c.k.gemmc[ci][static_cast<int>(QType::f16)], {cdiv(n, cf.bn), cdiv(nrows, cf.bm), 1},
                            {static_cast<unsigned>(cf.nth), 1, 1}, 0, c.s, w16.p(), rb16, x16.p(), y.p(), ncols, nrows, n, 0);
                c.sync();
                check(y.down<float>(ref.size()), "dequant+gemm_c" + std::to_string(ci) + "_f16");
            }
        }
        if (ncols % 256 == 0) {
            for (int si = 0; si < static_cast<int>(wk::kGemmsCfgs.size()); ++si) {
                const auto& cf = c.k.gemms_geom[static_cast<std::size_t>(si)];
                if (c.k.gemms[si][ti]) check(g.run(c.k.gemms[si][ti], cf.bm, cf.bn, cf.nth, n), "gemms_c" + std::to_string(si));
            }
        }
        if (ncols % 32 == 0) {
            if (c.k.gemmhq[ti]) check(g.run(c.k.gemmhq[ti], 128, 256, 256, n), "gemmhq");
            if (c.k.gemmh_f16) {
                Buf y(static_cast<std::size_t>(n) * nrows * 4);
                y.fill(0xff);
                hip::launch(c.k.gemmh_f16, {cdiv(n, 256), cdiv(nrows, 128), 1}, {256, 1, 1}, 0, c.s, w16.p(), x16.p(),
                            y.p(), ncols, nrows, n, 0);
                c.sync();
                check(y.down<float>(ref.size()), "gemmh_f16");
            }
        }
        // 40-token subset alone
        {
            const auto sub = g.run(c.k.gemmc[0][ti], c0.bm, c0.bn, c0.nth, nsub);
            ++checked;
            if (std::memcmp(sub.data(), ref.data(), sub.size() * 4) != 0) {
                ++bad;
                if (first_bad.empty()) first_bad = "40-row subset";
            }
        }
        Result r;
        r.name = "prefill GEMM choices == gemm_c0 (" + std::to_string(checked) + " variants, n=200) " + m.name;
        r.kind = Kind::invariant;
        r.n = checked;
        r.mismatches = bad;
        r.pass = bad == 0;
        if (!first_bad.empty()) r.note = "first " + first_bad;
        c.rep.add(r);

        // Precise decode (P-8): the small-batch / verify matmuls of precise mode run these same
        // f16 GEMMs (gemms_c*, gemm_c* on the quantized bytes) at n = 1..32. Every token row
        // must equal the 200-token gemm_c0 row exactly whatever n and the kernel (fixed
        // per-row reduction: wmma over k in increasing 16-steps), and match a double CPU
        // product of the f16 activations and f16-dequantized weights.
        {
            const int cr = std::min(nrows, c.quick ? 64 : 256);
            std::vector<float> wf(static_cast<std::size_t>(cr) * ncols);
            std::vector<std::uint16_t> wh(static_cast<std::size_t>(ncols));
            for (int r2 = 0; r2 < cr; ++r2) {
                ref::dequantRowF16(m.type, m.data.data() + static_cast<std::size_t>(r2) * m.row_bytes, ncols, wh.data());
                for (int k = 0; k < ncols; ++k) wf[static_cast<std::size_t>(r2) * ncols + k] = h2f(wh[static_cast<std::size_t>(k)]);
            }
            const int nmax = 33;
            std::vector<float> xf(static_cast<std::size_t>(nmax) * ncols);
            for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = h2f(hx16[i]);
            std::vector<double> ry, rs;
            ref::gemvF64(wf, cr, ncols, xf.data(), nmax, ncols, ry, rs);
            std::size_t pchecked = 0, pbad = 0;
            std::string pfirst;
            for (int nt : {1, 2, 3, 4, 8, 16, 33}) {
                std::vector<std::pair<std::string, std::vector<float>>> outs;
                if (ncols % 256 == 0)
                    for (int si = 0; si < static_cast<int>(wk::kGemmsCfgs.size()); ++si) {
                        const auto& cf = c.k.gemms_geom[static_cast<std::size_t>(si)];
                        if (c.k.gemms[si][ti]) outs.push_back({"gemms_c" + std::to_string(si), g.run(c.k.gemms[si][ti], cf.bm, cf.bn, cf.nth, nt)});
                    }
                if (ncols % 256 == 0 && nt <= 32)
                    for (const auto& [lab, f] : {std::pair{std::string("gemvh"), c.k.gemvh[ti]}, std::pair{std::string("gemvh2"), c.k.gemvh2[ti]}}) {
                        if (!f || (lab == "gemvh" && nt > 16)) continue;
                        outs.push_back({lab, g.run(f, 16, 1 << 20, 32, nt, false)});  // one wave per 16 rows
                    }
                for (int ci = 0; ci < static_cast<int>(wk::kGemmCfgs.size()); ++ci) {
                    const auto& cf = wk::kGemmCfgs[static_cast<std::size_t>(ci)];
                    if (c.k.gemmc[ci][ti] && (ci == 0 || !c.quick)) outs.push_back({"gemm_c" + std::to_string(ci), g.run(c.k.gemmc[ci][ti], cf.bm, cf.bn, cf.nth, nt)});
                }
                for (const auto& [label, y] : outs) {
                    ++pchecked;
                    if (std::memcmp(y.data(), ref.data(), static_cast<std::size_t>(nt) * nrows * 4) != 0) {
                        ++pbad;
                        if (pfirst.empty()) pfirst = label + " n=" + std::to_string(nt);
                    }
                }
                if (!outs.empty()) {
                    const std::vector<float>& y = outs.front().second;
                    std::vector<float> got;
                    std::vector<double> ryn, rsn;
                    for (int t = 0; t < nt; ++t) {
                        got.insert(got.end(), y.begin() + static_cast<std::ptrdiff_t>(t) * nrows, y.begin() + static_cast<std::ptrdiff_t>(t) * nrows + cr);
                        ryn.insert(ryn.end(), ry.begin() + static_cast<std::ptrdiff_t>(t) * cr, ry.begin() + static_cast<std::ptrdiff_t>(t + 1) * cr);
                        rsn.insert(rsn.end(), rs.begin() + static_cast<std::ptrdiff_t>(t) * cr, rs.begin() + static_cast<std::ptrdiff_t>(t + 1) * cr);
                    }
                    c.rep.add(cmpTol("precise decode " + outs.front().first + "_" + sfx + " n=" + std::to_string(nt) + " vs CPU " + m.name, got, ryn,
                                     rsn, 1e-4, 1e-6));
                }
            }
            Result rp;
            rp.name = "precise decode rows == 200-token gemm_c0 rows, n=1,2,3,4,8,16,33 (" + std::to_string(pchecked) + " runs) " + m.name;
            rp.kind = Kind::invariant;
            rp.n = pchecked;
            rp.mismatches = pbad;
            rp.pass = pbad == 0 && pchecked > 0;
            if (!pfirst.empty()) rp.note = "first " + pfirst;
            c.rep.add(rp);
        }

        // f16-output twins == f16(f32 result)
        {
            const std::vector<std::uint16_t> want = toHalf(ref);
            std::size_t ch = 0, chbad = 0;
            std::string fb;
            auto checkH = [&](const std::vector<std::uint16_t>& y, const std::string& label) {
                ++ch;
                if (std::memcmp(y.data(), want.data(), want.size() * 2) != 0) {
                    ++chbad;
                    if (fb.empty()) fb = label;
                }
            };
            for (int ci = 0; ci < static_cast<int>(wk::kGemmCfgs.size()); ++ci) {
                const auto& cf = wk::kGemmCfgs[static_cast<std::size_t>(ci)];
                if (c.k.gemmch[ci]) checkH(g.runH(c.k.gemmch[ci], w16.p(), rb16, cf.bm, cf.bn, cf.nth), "gemm_ch" + std::to_string(ci));
            }
            if (ncols % 256 == 0)
                for (int si = 0; si < static_cast<int>(wk::kGemmsCfgs.size()); ++si) {
                    const auto& cf = c.k.gemms_geom[static_cast<std::size_t>(si)];
                    if (c.k.gemmsh[si][ti]) checkH(g.runH(c.k.gemmsh[si][ti], w.p(), m.row_bytes, cf.bm, cf.bn, cf.nth), "gemmsh_c" + std::to_string(si));
                }
            if (ncols % 32 == 0 && c.k.gemmhqh[ti]) checkH(g.runH(c.k.gemmhqh[ti], w.p(), m.row_bytes, 128, 256, 256), "gemmhqh");
            if (ncols % 32 == 0 && c.k.gemmhh_f16) {
                Buf y(static_cast<std::size_t>(n) * nrows * 2);
                hip::launch(c.k.gemmhh_f16, {cdiv(n, 256), cdiv(nrows, 128), 1}, {256, 1, 1}, 0, c.s, w16.p(), x16.p(),
                            y.p(), ncols, nrows, n, 0);
                c.sync();
                checkH(y.down<std::uint16_t>(want.size()), "gemmhh_f16");
            }
            if (ch > 0) {
                Result rh;
                rh.name = "f16-output GEMMs == f16(gemm_c0) (" + std::to_string(ch) + " variants) " + m.name;
                rh.kind = Kind::invariant;
                rh.n = ch;
                rh.mismatches = chbad;
                rh.pass = chbad == 0;
                if (!fb.empty()) rh.note = "first " + fb;
                c.rep.add(rh);
            }
        }

        // accumulate: Y = Y0 + result exactly
        {
            std::vector<float> y0 = c.randn(ref.size());
            Buf y(y0);
            hip::launch(c.k.gemmc[0][ti], {cdiv(n, c0.bn), cdiv(nrows, c0.bm), 1}, {static_cast<unsigned>(c0.nth), 1, 1}, 0,
                        c.s, w.p(), m.row_bytes, x16.p(), y.p(), ncols, nrows, n, 1);
            c.sync();
            std::vector<float> want(ref.size());
            for (std::size_t i = 0; i < want.size(); ++i) want[i] = y0[i] + ref[i];
            c.rep.add(cmpExact("gemm_c0_" + sfx + " accumulate == y0 + y", y.down<float>(want.size()), want, Kind::invariant));
        }
    }

    // GDN [beta; alpha]: one 96-row Q8_0 GEMM == the two 48-row GEMMs side by side
    // (rows 0-47 of each token from beta, 48-95 from alpha), bitwise, for every choice.
    {
        const int nh = 48, ncols = 5120;
        const QType qt = QType::q8_0;
        const int ti = static_cast<int>(qt);
        const HostMat ba = randomMat(c, qt, 2 * nh, ncols);
        const std::uint64_t rb = ba.row_bytes;
        Buf wba(ba.data);
        const std::uint64_t half = static_cast<std::uint64_t>(nh) * rb;
        Buf wb(std::vector<std::uint8_t>(ba.data.begin(), ba.data.begin() + static_cast<std::ptrdiff_t>(half)));
        Buf wa(std::vector<std::uint8_t>(ba.data.begin() + static_cast<std::ptrdiff_t>(half), ba.data.end()));
        const std::uint64_t rb16 = static_cast<std::uint64_t>(ncols) * 2;
        auto deq = [&](const Buf& w, int rows, Buf& w16) {
            const std::uint64_t groups = static_cast<std::uint64_t>(ncols / 8) * rows;
            hip::launch(c.k.dequant_f16[ti], {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(groups, 256), 65535)), 1, 1}, {256, 1, 1},
                        0, c.s, w.p(), rb, w16.p(), ncols, rows);
        };
        Buf wba16(static_cast<std::size_t>(2 * nh) * ncols * 2), wb16(static_cast<std::size_t>(nh) * ncols * 2),
            wa16(static_cast<std::size_t>(nh) * ncols * 2);
        deq(wba, 2 * nh, wba16);
        deq(wb, nh, wb16);
        deq(wa, nh, wa16);
        const std::vector<int> ns = c.quick ? std::vector<int>{17, 512} : std::vector<int>{17, 512, 4096};
        for (int n : ns) {
            const std::vector<float> x = c.randn(static_cast<std::size_t>(n) * ncols);
            Buf dx(x), x16(static_cast<std::size_t>(n) * ncols * 2);
            hip::launch(c.k.f32_to_f16, {cdiv(static_cast<std::uint64_t>(n) * ncols / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), x16.p(),
                        n * ncols);
            // one kernel choice: (quantized weights, f16 weights, rows, y)
            using L = std::function<void(DevPtr, DevPtr, int, DevPtr)>;
            std::vector<std::pair<std::string, L>> vs;
            for (int ci = 0; ci < static_cast<int>(wk::kGemmCfgs.size()); ++ci) {
                const auto cf = wk::kGemmCfgs[static_cast<std::size_t>(ci)];
                if (hip::Function f = c.k.gemmc[ci][ti])
                    vs.push_back({"gemm_c" + std::to_string(ci), [&, f, cf](DevPtr w, DevPtr, int rows, DevPtr y) {
                                      hip::launch(f, {cdiv(n, cf.bn), cdiv(rows, cf.bm), 1}, {static_cast<unsigned>(cf.nth), 1, 1}, 0, c.s, w,
                                                  rb, x16.p(), y, ncols, rows, n, 0);
                                  }});
                if (hip::Function f = c.k.gemmc[ci][static_cast<int>(QType::f16)])
                    vs.push_back({"dequant+gemm_c" + std::to_string(ci) + "_f16", [&, f, cf](DevPtr, DevPtr w16, int rows, DevPtr y) {
                                      hip::launch(f, {cdiv(n, cf.bn), cdiv(rows, cf.bm), 1}, {static_cast<unsigned>(cf.nth), 1, 1}, 0, c.s,
                                                  w16, rb16, x16.p(), y, ncols, rows, n, 0);
                                  }});
            }
            for (int si = 0; si < static_cast<int>(wk::kGemmsCfgs.size()); ++si) {
                const auto cf = c.k.gemms_geom[static_cast<std::size_t>(si)];
                if (hip::Function f = c.k.gemms[si][ti])
                    vs.push_back({"gemms_c" + std::to_string(si), [&, f, cf](DevPtr w, DevPtr, int rows, DevPtr y) {
                                      hip::launch(f, {cdiv(n, cf.bn), cdiv(rows, cf.bm), 1}, {static_cast<unsigned>(cf.nth), 1, 1}, 0, c.s, w,
                                                  rb, x16.p(), y, ncols, rows, n, 0);
                                  }});
            }
            if (hip::Function f = c.k.gemmhq[ti])
                vs.push_back({"gemmhq", [&, f](DevPtr w, DevPtr, int rows, DevPtr y) {
                                  hip::launch(f, {cdiv(n, 256), cdiv(rows, 128), 1}, {256, 1, 1}, 0, c.s, w, rb, x16.p(), y, ncols, rows, n, 0);
                              }});
            if (hip::Function f = c.k.gemmh_f16)
                vs.push_back({"gemmh_f16", [&, f](DevPtr, DevPtr w16, int rows, DevPtr y) {
                                  hip::launch(f, {cdiv(n, 256), cdiv(rows, 128), 1}, {256, 1, 1}, 0, c.s, w16, x16.p(), y, ncols, rows, n, 0);
                              }});
            std::size_t bad = 0;
            std::string first_bad;
            const std::size_t nb = static_cast<std::size_t>(n) * nh;
            Buf yba(nb * 2 * 4), yb(nb * 4), ya(nb * 4);
            for (const auto& [label, run] : vs) {
                yba.fill(0xff);
                yb.fill(0xff);
                ya.fill(0xff);
                run(wba.p(), wba16.p(), 2 * nh, yba.p());
                run(wb.p(), wb16.p(), nh, yb.p());
                run(wa.p(), wa16.p(), nh, ya.p());
                c.sync();
                const std::vector<float> gba = yba.down<float>(nb * 2), gb = yb.down<float>(nb), ga = ya.down<float>(nb);
                std::vector<float> want(nb * 2);
                for (int t = 0; t < n; ++t) {
                    std::memcpy(&want[static_cast<std::size_t>(t) * 2 * nh], &gb[static_cast<std::size_t>(t) * nh], nh * 4);
                    std::memcpy(&want[static_cast<std::size_t>(t) * 2 * nh + nh], &ga[static_cast<std::size_t>(t) * nh], nh * 4);
                }
                if (std::memcmp(gba.data(), want.data(), want.size() * 4) != 0) {
                    ++bad;
                    if (first_bad.empty()) first_bad = label;
                }
            }
            Result r;
            r.name = "gemm [beta;alpha] 96 rows == beta 48 | alpha 48 (q8_0, " + std::to_string(vs.size()) + " choices, n=" +
                     std::to_string(n) + ")";
            r.kind = Kind::invariant;
            r.n = vs.size();
            r.mismatches = bad;
            r.pass = bad == 0 && !vs.empty();
            if (!first_bad.empty()) r.note = "first " + first_bad;
            c.rep.add(r);
        }
    }
}

}  // namespace kt
