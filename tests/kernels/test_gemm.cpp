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
#include "kt_w4a4p.h"                             // v0252-w4a4k2: the W4A4 probe's CPU definitions (codes, scales, exact fold)
#include "../../kernels/shared/gemm_w4a4_sim.h"  // v0252-w4a4k2: the fake-quant arm's math (host references)
#include "../../kernels/shared/gemm_w4a4_prod.h"  // v0252-w4a4p: the product W4A4 iu4 GEMM's math (host references)
#include "whirl/ops/matmul.h"                   // v0252-w4a4k2 round 3: ops::w4sGains (the host gain rule)

#include <cstring>
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

// A device allocation of exactly `bytes` (no kTailPad): hip::malloc adds the pad itself.
struct RawAlloc {
    DevPtr p = 0;
    explicit RawAlloc(std::size_t bytes) { p = hip::malloc(bytes - hip::kTailPad); }
    RawAlloc(const RawAlloc&) = delete;
    RawAlloc& operator=(const RawAlloc&) = delete;
    ~RawAlloc() { hip::free(p); }
};

// FIX-OVR: every raw-weight GEMM with the matrix placed flush against the end of an
// allocation whose size is a multiple of 2 MiB (and a freed 2 MiB guard right after it, so
// the next page is likely unmapped): a load past the matrix end faults (hipError 719) or
// reads foreign bytes. Results must equal gemm_c0 on an ordinary buffer, bitwise, for
// n = 1, 17, 37, 40, 47, 128, 2048.
void testGemmFlush(Ctx& c, const std::vector<HostMat>& mats) {
    constexpr std::size_t kAlign = 2u << 20;
    const std::vector<int> ns = c.quick ? std::vector<int>{1, 17, 37, 40, 47, 128} : std::vector<int>{1, 17, 37, 40, 47, 128, 2048};
    const int nmax = ns.back();
    for (const HostMat& m : mats) {
        if (m.type == QType::f32) continue;
        const int ti = static_cast<int>(m.type);
        const int ncols = m.ncols, nrows = m.nrows;
        const std::size_t bytes = m.data.size();
        const std::size_t asz = (bytes + kAlign - 1) / kAlign * kAlign;
        RawAlloc big(asz);
        {
            RawAlloc guard(kAlign);  // freed at once: likely a hole right after `big`
        }
        const DevPtr wf = big.p + (asz - bytes);
        hip::upload(wf, m.data.data(), bytes);
        Buf w(m.data);
        const std::vector<float> x = c.randn(static_cast<std::size_t>(nmax) * ncols);
        Buf dx(x), x16(static_cast<std::size_t>(nmax) * ncols * 2);
        hip::launch(c.k.f32_to_f16, {cdiv(static_cast<std::uint64_t>(nmax) * ncols / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(),
                    x16.p(), nmax * ncols);
        c.sync();
        auto run = [&](hip::Function f, DevPtr wp, int bm, int bn, int nth, int nt, bool tok_x) {
            Buf y(static_cast<std::size_t>(nt) * nrows * 4);
            y.fill(0xff);
            const hip::Dim3 grid = tok_x ? hip::Dim3{cdiv(nt, bn), cdiv(nrows, bm), 1} : hip::Dim3{cdiv(nrows, bm), cdiv(nt, bn), 1};
            hip::launch(f, grid, {static_cast<unsigned>(nth), 1, 1}, 0, c.s, wp, m.row_bytes, x16.p(), y.p(), ncols, nrows, nt, 0);
            c.sync();
            return y.down<float>(static_cast<std::size_t>(nt) * nrows);
        };
        auto runH = [&](hip::Function f, DevPtr wp, int bm, int bn, int nth, int nt) {
            Buf y(static_cast<std::size_t>(nt) * nrows * 2);
            y.fill(0xff);
            hip::launch(f, {cdiv(nt, bn), cdiv(nrows, bm), 1}, {static_cast<unsigned>(nth), 1, 1}, 0, c.s, wp, m.row_bytes, x16.p(), y.p(),
                        ncols, nrows, nt, 0);
            c.sync();
            return y.down<std::uint16_t>(static_cast<std::size_t>(nt) * nrows);
        };
        const auto& c0 = wk::kGemmCfgs[0];
        const std::vector<float> ref = run(c.k.gemmc[0][ti], w.p(), c0.bm, c0.bn, c0.nth, nmax, true);
        std::size_t checked = 0, bad = 0;
        std::string first_bad;
        for (int nt : ns) {
            const std::size_t cnt = static_cast<std::size_t>(nt) * nrows;
            auto chk = [&](const std::vector<float>& y, const std::string& label) {
                ++checked;
                if (std::memcmp(y.data(), ref.data(), cnt * 4) != 0) {
                    ++bad;
                    if (first_bad.empty()) first_bad = label + " n=" + std::to_string(nt);
                }
            };
            const std::vector<std::uint16_t> want = toHalf(std::vector<float>(ref.begin(), ref.begin() + static_cast<std::ptrdiff_t>(cnt)));
            for (int ci = 0; ci < static_cast<int>(wk::kGemmCfgs.size()); ++ci) {
                const auto& cf = wk::kGemmCfgs[static_cast<std::size_t>(ci)];
                if (c.k.gemmc[ci][ti] && (!c.quick || ci == 0)) chk(run(c.k.gemmc[ci][ti], wf, cf.bm, cf.bn, cf.nth, nt, true), "gemm_c" + std::to_string(ci));
            }
            if (ncols % 256 == 0) {
                for (int si = 0; si < static_cast<int>(wk::kGemmsCfgs.size()); ++si) {
                    const auto& cf = c.k.gemms_geom[static_cast<std::size_t>(si)];
                    if (c.k.gemms[si][ti]) chk(run(c.k.gemms[si][ti], wf, cf.bm, cf.bn, cf.nth, nt, true), "gemms_c" + std::to_string(si));
                    if (c.k.gemmsh[si][ti]) {
                        ++checked;
                        if (runH(c.k.gemmsh[si][ti], wf, cf.bm, cf.bn, cf.nth, nt) != want) {
                            ++bad;
                            if (first_bad.empty()) first_bad = "gemmsh_c" + std::to_string(si) + " n=" + std::to_string(nt);
                        }
                    }
                }
                if (nt <= 16 && c.k.gemvh[ti]) chk(run(c.k.gemvh[ti], wf, 16, 1 << 20, 32, nt, false), "gemvh");
                if (nt <= 32 && c.k.gemvh2[ti]) chk(run(c.k.gemvh2[ti], wf, 16, 1 << 20, 32, nt, false), "gemvh2");
            }
            if (ncols % 32 == 0 && c.k.gemmhq[ti]) chk(run(c.k.gemmhq[ti], wf, 128, 256, 256, nt, true), "gemmhq");
        }
        Result r;
        r.name = "raw-weight GEMMs on a matrix flush at the end of a 2 MiB-multiple allocation == gemm_c0 (" + std::to_string(checked) +
                 " runs, n=1..2048) " + m.name;
        r.kind = Kind::invariant;
        r.n = checked;
        r.mismatches = bad;
        r.pass = bad == 0 && checked > 0;
        if (!first_bad.empty()) r.note = "first " + first_bad;
        c.rep.add(r);
    }
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
    testGemmFlush(c, mats);

    // Tile order coverage: a many-block grid (row-block count not a multiple of the
    // tile-order group, partial last row and token blocks); every gemm_c<i> choice and
    // dequant + gemm_c<i>_f16 vs the CPU product on ALL rows and tokens, plus the
    // choices bitwise against each other.
    {
        const int nr = 1000, nc = 512, nt = c.quick ? 300 : 700;
        const QType qt = QType::q4_k;
        const int ti = static_cast<int>(qt);
        const HostMat hm = randomMat(c, qt, nr, nc);
        Buf w(hm.data);
        const std::vector<float> x = c.randn(static_cast<std::size_t>(nt) * nc);
        Buf dx(x), x16(static_cast<std::size_t>(nt) * nc * 2);
        hip::launch(c.k.f32_to_f16, {cdiv(static_cast<std::uint64_t>(nt) * nc / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), x16.p(),
                    nt * nc);
        Buf w16(static_cast<std::size_t>(nr) * nc * 2);
        const std::uint64_t groups = static_cast<std::uint64_t>(nc / 8) * nr;
        hip::launch(c.k.dequant_f16[ti], {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(groups, 256), 65535)), 1, 1}, {256, 1, 1}, 0,
                    c.s, w.p(), hm.row_bytes, w16.p(), nc, nr);
        c.sync();
        const std::vector<std::uint16_t> hx16 = toHalf(x);
        std::vector<float> wf(static_cast<std::size_t>(nr) * nc), xf(hx16.size());
        std::vector<std::uint16_t> wh(static_cast<std::size_t>(nc));
        for (int r = 0; r < nr; ++r) {
            ref::dequantRowF16(qt, hm.data.data() + static_cast<std::size_t>(r) * hm.row_bytes, nc, wh.data());
            for (int k = 0; k < nc; ++k) wf[static_cast<std::size_t>(r) * nc + k] = h2f(wh[static_cast<std::size_t>(k)]);
        }
        for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = h2f(hx16[i]);
        std::vector<double> ry, rs;
        ref::gemvF64(wf, nr, nc, xf.data(), nt, nc, ry, rs);
        Gemm g{c, hm, w, x16, nt};
        const auto& c0 = wk::kGemmCfgs[0];
        const std::vector<float> y0 = g.run(c.k.gemmc[0][ti], c0.bm, c0.bn, c0.nth, nt);
        c.rep.add(cmpTol("gemm_c0_q4_k vs CPU, all of " + std::to_string(nr) + "x" + std::to_string(nt), y0, ry, rs, 1e-4, 1e-6));
        std::size_t checked = 0, bad = 0;
        std::string first_bad;
        for (int ci = 0; ci < static_cast<int>(wk::kGemmCfgs.size()); ++ci) {
            const auto& cf = wk::kGemmCfgs[static_cast<std::size_t>(ci)];
            std::vector<std::vector<float>> ys;
            if (ci > 0 && c.k.gemmc[ci][ti]) ys.push_back(g.run(c.k.gemmc[ci][ti], cf.bm, cf.bn, cf.nth, nt));
            if (const hip::Function f16 = c.k.gemmc[ci][static_cast<int>(QType::f16)]) {
                Buf y(static_cast<std::size_t>(nt) * nr * 4);
                y.fill(0xff);
                hip::launch(f16, {cdiv(nt, cf.bn), cdiv(nr, cf.bm), 1}, {static_cast<unsigned>(cf.nth), 1, 1}, 0, c.s, w16.p(),
                            static_cast<std::uint64_t>(nc) * 2, x16.p(), y.p(), nc, nr, nt, 0);
                c.sync();
                ys.push_back(y.down<float>(y0.size()));
            }
            for (const auto& y : ys) {
                ++checked;
                if (std::memcmp(y.data(), y0.data(), y0.size() * 4) != 0) {
                    ++bad;
                    if (first_bad.empty()) first_bad = "cfg " + std::to_string(ci);
                }
            }
        }
        Result r;
        r.name = "prefill GEMM choices == gemm_c0, " + std::to_string(nr) + "x" + std::to_string(nt) + " grid (" + std::to_string(checked) +
                 " variants)";
        r.kind = Kind::invariant;
        r.n = checked;
        r.mismatches = bad;
        r.pass = bad == 0;
        if (!first_bad.empty()) r.note = "first " + first_bad;
        c.rep.add(r);
    }

    // v0251-a3port (feat026-fdq c15f51f, the cases verbatim): the gfx1151 fused-dequant MXFP4 GEMMs gemmfq_<v>_mxfp4
    // == dequant_f16_mxfp4 + gemm_c2_f16, bitwise, on a many-block grid with partial row / token blocks, for a random
    // matrix (block exponents of randomMat: the injected decode) and one whose exponents cover 96..135 (the LUT fallback
    // outside [104, 128], subnormal / zero weights), n = 17 / 700 / 1100, plain and accumulate; plus the CPU product for
    // the first case. The gfx1151 object must export both ported slots (c2, t2) and the host table must load them
    // (kernels_abi.cpp loadMatmul); any other object (gfx1201) must have none, so the R9700 path can never select one.
    {
        const bool g1151 = c.mod.arch() == hip::Arch::gfx1151;
        std::vector<hip::Function> fq(wk::kGemmFqCfgs.size(), nullptr);
        for (std::size_t fi = 0; fi < fq.size(); ++fi) {
            const std::string kn = std::string("gemmfq_") + wk::kGemmFqNames[fi] + "_mxfp4";
            fq[fi] = c.fnOpt(kn);
            Result r;
            r.kind = Kind::invariant;
            r.n = 1;
            if (g1151) {
                r.name = kn + " in the gfx1151 code object and loaded into the host table";
                r.pass = fq[fi] != nullptr && c.k.gemmfq[fi] == fq[fi];
                if (!r.pass) r.note = fq[fi] == nullptr ? "missing from the code object" : "not in MatmulKernels::gemmfq (loadMatmul)";
            } else {
                r.name = kn + " absent from this non-gfx1151 code object and from the host table";
                r.pass = fq[fi] == nullptr && c.k.gemmfq[fi] == nullptr;
            }
            r.mismatches = r.pass ? 0 : 1;
            c.rep.add(r);
        }
        bool any = false;
        for (const auto f : fq) any = any || f != nullptr;
        const int ti = static_cast<int>(QType::mxfp4);
        const hip::Function c2f16 = c.k.gemmc[2][static_cast<int>(QType::f16)];
        if (!any || !c.k.dequant_f16[ti] || !c2f16) {
            c.rep.skip("gemm", "gemmfq_*_mxfp4 == dequant + gemm_c2_f16", "no gemmfq kernels in this code object");
        } else {
            const int nr = 1000, nc = 768;
            for (int variant = 0; variant < 2; ++variant) {
                HostMat hm = randomMat(c, QType::mxfp4, nr, nc);
                if (variant == 1) {
                    std::uniform_int_distribution<int> ex(96, 135);
                    for (std::size_t b = 0; b < hm.data.size(); b += 136)
                        for (int i = 0; i < 8; ++i) hm.data[b + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(ex(c.rng));
                }
                Buf w(hm.data), w16(static_cast<std::size_t>(nr) * nc * 2);
                const std::uint64_t groups = static_cast<std::uint64_t>(nc / 8) * nr;
                hip::launch(c.k.dequant_f16[ti], {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(groups, 256), 65535)), 1, 1}, {256, 1, 1}, 0,
                            c.s, w.p(), hm.row_bytes, w16.p(), nc, nr);
                c.sync();
                const std::vector<int> ns = c.quick ? std::vector<int>{17, 700} : std::vector<int>{17, 700, 1100};
                for (int nt : ns) {
                    const std::vector<float> x = c.randn(static_cast<std::size_t>(nt) * nc);
                    Buf dx(x), x16(static_cast<std::size_t>(nt) * nc * 2);
                    hip::launch(c.k.f32_to_f16, {cdiv(static_cast<std::uint64_t>(nt) * nc / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), x16.p(),
                                nt * nc);
                    const std::vector<float> y0 = c.randn(static_cast<std::size_t>(nt) * nr);  // accumulate base
                    const auto& c2 = wk::kGemmCfgs[2];
                    auto runRef = [&](int accumulate) {
                        Buf y(static_cast<std::size_t>(nt) * nr * 4);
                        if (accumulate) y.up(y0); else y.fill(0xff);
                        hip::launch(c2f16, {cdiv(nt, c2.bn), cdiv(nr, c2.bm), 1}, {static_cast<unsigned>(c2.nth), 1, 1}, 0, c.s, w16.p(),
                                    static_cast<std::uint64_t>(nc) * 2, x16.p(), y.p(), nc, nr, nt, accumulate);
                        c.sync();
                        return y.down<float>(static_cast<std::size_t>(nt) * nr);
                    };
                    const std::vector<float> ref0 = runRef(0), ref1 = runRef(1);
                    if (variant == 0 && nt == 700) {
                        // the CPU product of the f16 weights (ref::dequantRowF16) on all rows and tokens
                        const std::vector<std::uint16_t> hx16 = toHalf(x);
                        std::vector<float> wf(static_cast<std::size_t>(nr) * nc), xf(hx16.size());
                        std::vector<std::uint16_t> wh(static_cast<std::size_t>(nc));
                        for (int r = 0; r < nr; ++r) {
                            ref::dequantRowF16(QType::mxfp4, hm.data.data() + static_cast<std::size_t>(r) * hm.row_bytes, nc, wh.data());
                            for (int k = 0; k < nc; ++k) wf[static_cast<std::size_t>(r) * nc + k] = h2f(wh[static_cast<std::size_t>(k)]);
                        }
                        for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = h2f(hx16[i]);
                        std::vector<double> ry, rs;
                        ref::gemvF64(wf, nr, nc, xf.data(), nt, nc, ry, rs);
                        c.rep.add(cmpTol("dequant_f16_mxfp4 + gemm_c2_f16 vs CPU, 1000x700", ref0, ry, rs, 1e-4, 1e-6));
                    }
                    for (std::size_t fi = 0; fi < fq.size(); ++fi) {
                        const hip::Function f = fq[fi];
                        if (!f) continue;
                        const auto& g = wk::kGemmFqCfgs[fi];
                        for (int accumulate = 0; accumulate < 2; ++accumulate) {
                            Buf y(static_cast<std::size_t>(nt) * nr * 4);
                            if (accumulate) y.up(y0); else y.fill(0xff);
                            hip::launch(f, {cdiv(nt, g.bn), cdiv(nr, g.bm), 1}, {static_cast<unsigned>(g.nth), 1, 1}, 0, c.s, w.p(), hm.row_bytes,
                                        x16.p(), y.p(), nc, nr, nt, accumulate);
                            c.sync();
                            const std::string name = std::string("gemmfq_") + wk::kGemmFqNames[fi] + "_mxfp4 == dequant + gemm_c2_f16 (" +
                                                     (variant ? "e 96..135" : "random") + ", 1000 rows, n=" + std::to_string(nt) +
                                                     (accumulate ? ", accumulate)" : ")");
                            c.rep.add(cmpExact(name, y.down<float>(static_cast<std::size_t>(nt) * nr), accumulate ? ref1 : ref0, Kind::invariant));
                        }
                    }
                    // v0252-8sbal: the grouped-raster prototypes (kt.h kFqProto) on the same grid: every output tile has
                    // exactly one owner (1000 rows = 8 t2 / 4 c2 row tiles: full and partial groups of 4 / 8 / 16)
                    for (const FqProto& pv : kFqProto) {
                        const std::string kn = std::string("gemmfq_") + pv.name + "_mxfp4";
                        const hip::Function f = c.fnOpt(kn);
                        if (variant == 0 && nt == ns.front()) {
                            Result r;
                            r.kind = Kind::invariant;
                            r.n = 1;
                            r.name = kn + " (v0252-8sbal prototype) in the gfx1151 code object";
                            r.pass = f != nullptr;
                            if (!r.pass) r.note = "missing from the code object";
                            r.mismatches = r.pass ? 0 : 1;
                            c.rep.add(r);
                        }
                        if (!f) continue;
                        const auto& g = wk::kGemmFqCfgs[static_cast<std::size_t>(pv.base)];
                        for (int accumulate = 0; accumulate < 2; ++accumulate) {
                            Buf y(static_cast<std::size_t>(nt) * nr * 4);
                            if (accumulate) y.up(y0); else y.fill(0xff);
                            hip::launch(f, {cdiv(nt, g.bn), cdiv(nr, g.bm), 1}, {static_cast<unsigned>(g.nth), 1, 1}, 0, c.s, w.p(), hm.row_bytes,
                                        x16.p(), y.p(), nc, nr, nt, accumulate);
                            c.sync();
                            const std::string name = kn + " == dequant + gemm_c2_f16 (" + (variant ? "e 96..135" : "random") +
                                                     ", 1000 rows, n=" + std::to_string(nt) + (accumulate ? ", accumulate)" : ")");
                            c.rep.add(cmpExact(name, y.down<float>(static_cast<std::size_t>(nt) * nr), accumulate ? ref1 : ref0, Kind::invariant));
                        }
                    }
                    // v0252-valuout: the rejected v0252-8sbvalu lever prototypes are not in the shipped object (kt.h kFqRetired)
                    if (variant == 0 && nt == ns.front())
                        for (const char* rn : kFqRetired) {
                            const std::string kn = std::string("gemmfq_") + rn + "_mxfp4";
                            Result r;
                            r.kind = Kind::invariant;
                            r.n = 1;
                            r.name = kn + " (v0252-8sbvalu lever, REVERT) absent from the gfx1151 code object";
                            r.pass = c.fnOpt(kn) == nullptr;
                            if (!r.pass) r.note = "still in the code object";
                            r.mismatches = r.pass ? 0 : 1;
                            c.rep.add(r);
                        }
                }
            }
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

// v0252-w4a4k2 (PLAN V1-144, kill-screen K2) family "w4s": the W4A4 fake-quant arm of the fast gfx1151 dense prefill GEMM
// (kernels/gfx1151/gemm_w4a4s.hip; numerics flag w4a4_sim, WHIRL_W4A4_SIM). Rows:
//  - presence: both w4s_* entries in the gfx1151 object and loaded into MatmulKernels (w4s_dq / w4s_qx); absent from any other
//    object and from its host table (the R9700 can never select the arm);
//  - w4s_dequant_f16_mxfp4 == its host reference (gemm_w4a4_sim.h w4s_dequant_row_host = the probe's mxRowToI4 codes x scales,
//    pinned by whirl-kt-bench-tests) BITWISE, random blocks and an exponent sweep 96..150 (f16 subnormal ... inf);
//  - w4s_quant_x16 (absmax / 7 and MSE-best) == w4s_quant_row_host BITWISE (Gaussian rows, zero and outlier blocks);
//  - the sim GEMM (those two + the shipped dequant-path gemm_c2_f16) vs the CPU f64 product of the same f16 values: the
//    f16-path tolerance of this file (1e-4 x sum |terms| + 1e-6, the "dequant_f16_mxfp4 + gemm_c2_f16 vs CPU" row);
//  - the proof that the simulation IS the W4A4 kernel's numerics: the sim GEMM vs the CPU exact per-32 fold of the probe
//    (kt_w4a4p.h refFoldF64) and vs the probe kernel w4a4p_gemm_i4fb on the same inputs (grid w only: a product kernel
//    with another grid differs from the probe only in its code -> int4 table). Bound: each dequantized f16 value
//    carries one rounding (relative <= 2^-11), so a term moves by <= 2^-10 of its size; plus the f32 sums of both sides
//    (<= 2^-12 here, K 1024): 2^-10 + 2^-12 of sum |terms| (derived, not a hand-set band).
// v0252-w4a4p (PLAN V1-144 round 1): rows of the PRODUCT W4A4 iu4 GEMM (kernels/gfx1151/gemm_w4a4q.hip, WHIRL_W4A4=1), run
// first in family "w4s". Its reference is the sim arm wga (grid w, tensor gain g, absmax activations a):
//  - presence: w4q_wconv_mxfp4 / w4q_quant_x / w4q_gemm in the gfx1151 object and loaded into MatmulKernels::w4q_*; absent
//    from any other object and its host table;
//  - w4q_wconv_mxfp4 (with a gain) == w4q_wconv_row_host BITWISE (codes and scales; random blocks and the exponent sweep);
//  - w4q_quant_x == w4q_quant_row_host BITWISE (codes and steps; Gaussian rows with zero / outlier blocks);
//  - w4q_gemm (f32, no accumulate) == w4q_fold_host BITWISE (the kernel's own order: per 32-block fma(float(c), ws x xs, acc)),
//    at n 256 and at a token tail n 200 (not a multiple of the 128-token tile: the rows past n must stay untouched);
//  - accumulate (y + v) and the f16 store (f16(v)) == the host epilogue of the bitwise fold, BITWISE;
//  - vs the sim arm wga on the same inputs (w4s_dequant_f16_mxfp4_gn + w4s_quant_x16 + gemm_c2_f16): NOT bitwise by
//    construction (the sim rounds every dequantized weight and activation to f16; the product keeps the integers exact):
//    within 2^-10 + 2^-12 of sum |terms| (two f16 roundings per term + the f32 sums, K 1024; derived, not a hand-set band).
static void testW4q(Ctx& c) {
    const bool g1151 = c.mod.arch() == hip::Arch::gfx1151;
    const hip::Function fwc = c.fnOpt("w4q_wconv_mxfp4"), fqx = c.fnOpt("w4q_quant_x"), fgm = c.fnOpt("w4q_gemm");
    {
        Result r;
        r.kind = Kind::invariant;
        r.n = 1;
        r.name = g1151 ? "w4q_wconv_mxfp4 / w4q_quant_x / w4q_gemm in the gfx1151 code object and loaded into the host table"
                       : "w4q_* absent from this non-gfx1151 code object and from the host table";
        r.pass = g1151 ? (fwc != nullptr && fqx != nullptr && fgm != nullptr && c.k.w4q_wc == fwc && c.k.w4q_qx == fqx && c.k.w4q_gemm == fgm)
                       : (fwc == nullptr && fqx == nullptr && fgm == nullptr && c.k.w4q_wc == nullptr && c.k.w4q_qx == nullptr && c.k.w4q_gemm == nullptr);
        r.mismatches = r.pass ? 0 : 1;
        c.rep.add(r);
    }
    if (fwc == nullptr || fqx == nullptr || fgm == nullptr) return;
    const int nr = 256, nc = 1024, ntmax = 256;
    const float gain = 0.9375f;  // a tensor gain of the size the Swift tensors have (0.930-0.944, K2 round 4)
    std::vector<float> gv(static_cast<std::size_t>(nr), gain);
    Buf dgain(gv);
    std::vector<std::uint8_t> wq_h;
    std::vector<float> ws_h;
    HostMat hm0;
    for (int variant = 0; variant < 2; ++variant) {
        HostMat hm = randomMat(c, QType::mxfp4, nr, nc);
        if (variant == 1)
            for (std::size_t b = 0; b < hm.data.size(); b += 136)
                for (int i = 0; i < 8; ++i) hm.data[b + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(96 + (b / 136 * 8 + static_cast<std::size_t>(i)) % 55);
        Buf w(hm.data), dwq(static_cast<std::size_t>(nr) * nc / 2), dws(static_cast<std::size_t>(nr) * nc / 32 * 4);
        const std::uint64_t total = static_cast<std::uint64_t>(nr) * (nc / 32);
        hip::launch(fwc, {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(total, 256), 65535)), 1, 1}, {256, 1, 1}, 0, c.s, w.p(), hm.row_bytes, dwq.p(),
                    dws.p(), nc, nr, dgain.p());
        c.sync();
        std::vector<std::uint8_t> q(static_cast<std::size_t>(nr) * nc / 2);
        std::vector<float> s(static_cast<std::size_t>(nr) * nc / 32);
        for (int r = 0; r < nr; ++r)
            w4q_wconv_row_host(hm.data.data() + static_cast<std::size_t>(r) * hm.row_bytes, nc, gain, q.data() + static_cast<std::size_t>(r) * nc / 2,
                               s.data() + static_cast<std::size_t>(r) * nc / 32);
        const char* vn = variant ? "e 96..150" : "random";
        c.rep.add(cmpExact(std::string("w4q_wconv_mxfp4 codes == host reference (") + vn + ", gain 0.9375, 256 x 1024)", dwq.down<std::uint8_t>(q.size()), q));
        c.rep.add(cmpExact(std::string("w4q_wconv_mxfp4 scales == host reference (") + vn + ", gain 0.9375, 256 x 1024)", dws.down<float>(s.size()), s));
        if (variant == 0) {
            wq_h = q;
            ws_h = s;
            hm0 = hm;
        }
    }
    std::vector<float> x = c.randn(static_cast<std::size_t>(ntmax) * nc);
    for (int t = 0; t < ntmax; ++t) {
        for (int i = 64; i < 96; ++i) x[static_cast<std::size_t>(t) * nc + static_cast<std::size_t>(i)] = 0.f;
        x[static_cast<std::size_t>(t) * nc + 200 + static_cast<std::size_t>(t % 32)] = 30.f;
    }
    const std::vector<std::uint16_t> xh = toHalf(x);
    const std::uint64_t nblk = xh.size() / 32;
    Buf dx(xh), dxq(xh.size() / 2), dxs(nblk * 4);
    hip::launch(fqx, {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(nblk, 256), 65535)), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dxq.p(), dxs.p(), nblk);
    c.sync();
    std::vector<std::uint8_t> xq_h(xh.size() / 2);
    std::vector<float> xs_h(nblk);
    for (int t = 0; t < ntmax; ++t)
        w4q_quant_row_host(xh.data() + static_cast<std::size_t>(t) * nc, nc, xq_h.data() + static_cast<std::size_t>(t) * nc / 2,
                           xs_h.data() + static_cast<std::size_t>(t) * nc / 32);
    c.rep.add(cmpExact("w4q_quant_x codes == host reference (absmax / 7, 256 x 1024)", dxq.down<std::uint8_t>(xq_h.size()), xq_h));
    c.rep.add(cmpExact("w4q_quant_x steps == host reference (absmax / 7, 256 x 1024)", dxs.down<float>(xs_h.size()), xs_h));
    // the GEMM on the host-made operands (so a GEMM row fails only for the GEMM)
    Buf hwq(wq_h), hws(ws_h), hxq(xq_h), hxs(xs_h);
    std::vector<float> fold(static_cast<std::size_t>(ntmax) * nr);
    for (int t = 0; t < ntmax; ++t)
        for (int r = 0; r < nr; ++r)
            fold[static_cast<std::size_t>(t) * nr + static_cast<std::size_t>(r)] =
                w4q_fold_host(wq_h.data() + static_cast<std::size_t>(r) * nc / 2, ws_h.data() + static_cast<std::size_t>(r) * nc / 32,
                              xq_h.data() + static_cast<std::size_t>(t) * nc / 2, xs_h.data() + static_cast<std::size_t>(t) * nc / 32, nc);
    const std::vector<float> y0 = c.randn(fold.size());  // the accumulate base
    for (const int nt : {256, 200}) {
        // f32, no accumulate; the rows past nt keep their 0xff fill
        Buf y(fold.size() * 4);
        y.fill(0xff);
        hip::launch(fgm, {cdiv(nt, 128), static_cast<unsigned>(nr / 128), 1}, {256, 1, 1}, 0, c.s, hwq.p(), hws.p(), hxq.p(), hxs.p(), y.p(), nc, nr, nt, 0, 0);
        c.sync();
        std::vector<std::uint32_t> got = y.down<std::uint32_t>(fold.size()), want(fold.size(), 0xffffffffu);
        std::memcpy(want.data(), fold.data(), static_cast<std::size_t>(nt) * nr * 4);
        c.rep.add(cmpExact("w4q_gemm f32 == the host fold in the kernel's order BITWISE, n " + std::to_string(nt) + " (rows past n untouched)", got, want));
        // f32 accumulate: y0 + v
        Buf ya(y0);
        hip::launch(fgm, {cdiv(nt, 128), static_cast<unsigned>(nr / 128), 1}, {256, 1, 1}, 0, c.s, hwq.p(), hws.p(), hxq.p(), hxs.p(), ya.p(), nc, nr, nt, 1, 0);
        c.sync();
        std::vector<float> wa(y0);
        for (std::size_t i = 0; i < static_cast<std::size_t>(nt) * nr; ++i) wa[i] = y0[i] + fold[i];
        c.rep.add(cmpExact("w4q_gemm f32 accumulate == y + host fold BITWISE, n " + std::to_string(nt), ya.down<float>(wa.size()), wa));
        // f16 store: f16(v)
        Buf yh(fold.size() * 2);
        yh.fill(0xff);
        hip::launch(fgm, {cdiv(nt, 128), static_cast<unsigned>(nr / 128), 1}, {256, 1, 1}, 0, c.s, hwq.p(), hws.p(), hxq.p(), hxs.p(), yh.p(), nc, nr, nt, 0, 1);
        c.sync();
        std::vector<std::uint16_t> wh(fold.size(), 0xffff);
        for (std::size_t i = 0; i < static_cast<std::size_t>(nt) * nr; ++i) wh[i] = w4s_f2h(fold[i]);
        c.rep.add(cmpExact("w4q_gemm f16 store == f16(host fold) BITWISE, n " + std::to_string(nt), yh.down<std::uint16_t>(wh.size()), wh));
    }
    // vs the sim arm wga on the same weights and activations: w4s_dequant_f16_mxfp4_gn (gain) + w4s_quant_x16 + gemm_c2_f16
    const hip::Function c2f16 = c.k.gemmc[2][static_cast<int>(QType::f16)];
    if (c2f16 == nullptr || c.k.w4s_dqg[0] == nullptr || c.k.w4s_qx == nullptr) {
        c.rep.skip("w4s", "w4q vs the sim arm wga", "no gemm_c2_f16 / w4s gain dequant / w4s_quant_x16 in this code object");
        return;
    }
    Buf w(hm0.data), w16(static_cast<std::size_t>(nr) * nc * 2), x16q(xh.size() * 2), ys(fold.size() * 4);
    const std::uint64_t groups = static_cast<std::uint64_t>(nc / 8) * nr;
    hip::launch(c.k.w4s_dqg[0], {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(groups, 256), 65535)), 1, 1}, {256, 1, 1}, 0, c.s, w.p(), hm0.row_bytes,
                w16.p(), nc, nr, dgain.p());
    hip::launch(c.k.w4s_qx, {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(nblk, 256), 65535)), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), x16q.p(), nblk, 0);
    const auto& c2 = wk::kGemmCfgs[2];
    hip::launch(c2f16, {cdiv(ntmax, c2.bn), cdiv(nr, c2.bm), 1}, {static_cast<unsigned>(c2.nth), 1, 1}, 0, c.s, w16.p(), static_cast<std::uint64_t>(nc) * 2,
                x16q.p(), ys.p(), nc, nr, ntmax, 0);
    c.sync();
    const std::vector<float> ysim = ys.down<float>(fold.size());
    std::vector<double> yd(fold.size()), sc(fold.size());
    for (int t = 0; t < ntmax; ++t)
        for (int r = 0; r < nr; ++r) {
            double a = 0, sabs = 0;
            for (int b = 0; b < nc / 32; ++b) {
                int cb = 0;
                for (int i = b * 32; i < b * 32 + 32; ++i)
                    cb += w4q_nib(wq_h.data() + static_cast<std::size_t>(r) * nc / 2, i) * w4q_nib(xq_h.data() + static_cast<std::size_t>(t) * nc / 2, i);
                const double term = static_cast<double>(cb) * ws_h[static_cast<std::size_t>(r) * nc / 32 + static_cast<std::size_t>(b)] *
                                    xs_h[static_cast<std::size_t>(t) * nc / 32 + static_cast<std::size_t>(b)];
                a += term;
                sabs += std::fabs(term);
            }
            yd[static_cast<std::size_t>(t) * nr + static_cast<std::size_t>(r)] = static_cast<double>(ysim[static_cast<std::size_t>(t) * nr + static_cast<std::size_t>(r)]);
            sc[static_cast<std::size_t>(t) * nr + static_cast<std::size_t>(r)] = sabs;
            (void)a;
        }
    const double rtol = std::ldexp(1.0, -10) + std::ldexp(1.0, -12);
    c.rep.add(cmpTol("w4q_gemm == the sim arm wga (w4s gain dequant + w4s_quant_x16 + gemm_c2_f16) within 2^-10 + 2^-12 of sum |terms| (NOT bitwise: "
                     "f16 roundings of the sim)",
                     fold, yd, sc, rtol, 1e-6));
}

// v0252-w4a4p round 2 (WHIRL_W4A4=2 / 3, the LUT form, no converted weight copy): rows of w4q_quant_xm / w4q_gemm_m
//  - presence: both in the gfx1151 object and loaded into MatmulKernels::w4q_qxm / w4q_gemm_m; absent elsewhere;
//  - w4q_quant_xm == w4q_quant_row_m_host BITWISE (codes in the MXFP4 order, steps);
//  - w4q_gemm_m on the MXFP4 bytes + gains (random blocks and the exponent sweep e 96..150) == round 1's host fold
//    (w4q_wconv_row_host + w4q_quant_row_host + w4q_fold_host) BITWISE, f32 / accumulate / f16 store, n 256 and the tail n 200;
//  - w4q_gemm_m == w4q_gemm on the device-converted copy of the same matrix BITWISE (variant 2 == variant 1, the claim the
//    task-accuracy run relies on).
static void testW4qM(Ctx& c) {
    const bool g1151 = c.mod.arch() == hip::Arch::gfx1151;
    const hip::Function fxm = c.fnOpt("w4q_quant_xm"), fgm = c.fnOpt("w4q_gemm_m"), fg2 = c.fnOpt("w4q_gemm_m2");  // m2: round 2b
    const hip::Function fwc = c.fnOpt("w4q_wconv_mxfp4"), fqx = c.fnOpt("w4q_quant_x"), fg1 = c.fnOpt("w4q_gemm");
    {
        Result r;
        r.kind = Kind::invariant;
        r.n = 1;
        r.name = g1151 ? "w4q_quant_xm / w4q_gemm_m (round 2) in the gfx1151 code object and loaded into the host table"
                       : "w4q_quant_xm / w4q_gemm_m absent from this non-gfx1151 code object and from the host table";
        r.pass = g1151 ? (fxm != nullptr && fgm != nullptr && fg2 != nullptr && c.k.w4q_qxm == fxm && c.k.w4q_gemm_m == fgm && c.k.w4q_gemm_m2 == fg2)
                       : (fxm == nullptr && fgm == nullptr && fg2 == nullptr && c.k.w4q_qxm == nullptr && c.k.w4q_gemm_m == nullptr &&
                          c.k.w4q_gemm_m2 == nullptr);
        r.mismatches = r.pass ? 0 : 1;
        c.rep.add(r);
    }
    if (fxm == nullptr || fgm == nullptr || fg2 == nullptr || fwc == nullptr || fqx == nullptr || fg1 == nullptr) return;
    const int nr = 256, nc = 1024, ntmax = 256;
    const float gain = 0.9375f;
    std::vector<float> gv(static_cast<std::size_t>(nr), gain);
    Buf dgain(gv);
    std::vector<float> x = c.randn(static_cast<std::size_t>(ntmax) * nc);
    for (int t = 0; t < ntmax; ++t) {
        for (int i = 64; i < 96; ++i) x[static_cast<std::size_t>(t) * nc + static_cast<std::size_t>(i)] = 0.f;
        x[static_cast<std::size_t>(t) * nc + 200 + static_cast<std::size_t>(t % 32)] = 30.f;
    }
    const std::vector<std::uint16_t> xh = toHalf(x);
    const std::uint64_t nblk = xh.size() / 32;
    Buf dx(xh), dxm(xh.size() / 2), dxsm(nblk * 4), dxq(xh.size() / 2), dxs(nblk * 4);
    const unsigned qgrid = static_cast<unsigned>(std::min<std::uint64_t>(cdiv(nblk, 256), 65535));
    hip::launch(fxm, {qgrid, 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dxm.p(), dxsm.p(), nblk);
    hip::launch(fqx, {qgrid, 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dxq.p(), dxs.p(), nblk);
    c.sync();
    std::vector<std::uint8_t> xq_h(xh.size() / 2), xqm_h(xh.size() / 2);
    std::vector<float> xs_h(nblk), xsm_h(nblk);
    for (int t = 0; t < ntmax; ++t) {
        const std::size_t o = static_cast<std::size_t>(t) * nc;
        w4q_quant_row_host(xh.data() + o, nc, xq_h.data() + o / 2, xs_h.data() + o / 32);
        w4q_quant_row_m_host(xh.data() + o, nc, xqm_h.data() + o / 2, xsm_h.data() + o / 32);
    }
    c.rep.add(cmpExact("w4q_quant_xm codes (MXFP4 order) == host reference (absmax / 7, 256 x 1024)", dxm.down<std::uint8_t>(xqm_h.size()), xqm_h));
    c.rep.add(cmpExact("w4q_quant_xm steps == host reference (absmax / 7, 256 x 1024)", dxsm.down<float>(xsm_h.size()), xsm_h));
    Buf hxm(xqm_h), hxsm(xsm_h);
    for (int variant = 0; variant < 2; ++variant) {
        HostMat hm = randomMat(c, QType::mxfp4, nr, nc);
        if (variant == 1)
            for (std::size_t b = 0; b < hm.data.size(); b += 136)
                for (int i = 0; i < 8; ++i) hm.data[b + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(96 + (b / 136 * 8 + static_cast<std::size_t>(i)) % 55);
        const char* vn = variant ? "e 96..150" : "random";
        std::vector<std::uint8_t> wq_h(static_cast<std::size_t>(nr) * nc / 2);
        std::vector<float> ws_h(static_cast<std::size_t>(nr) * nc / 32);
        for (int r = 0; r < nr; ++r)
            w4q_wconv_row_host(hm.data.data() + static_cast<std::size_t>(r) * hm.row_bytes, nc, gain, wq_h.data() + static_cast<std::size_t>(r) * nc / 2,
                               ws_h.data() + static_cast<std::size_t>(r) * nc / 32);
        std::vector<float> fold(static_cast<std::size_t>(ntmax) * nr);
        for (int t = 0; t < ntmax; ++t)
            for (int r = 0; r < nr; ++r)
                fold[static_cast<std::size_t>(t) * nr + static_cast<std::size_t>(r)] =
                    w4q_fold_host(wq_h.data() + static_cast<std::size_t>(r) * nc / 2, ws_h.data() + static_cast<std::size_t>(r) * nc / 32,
                                  xq_h.data() + static_cast<std::size_t>(t) * nc / 2, xs_h.data() + static_cast<std::size_t>(t) * nc / 32, nc);
        Buf w(hm.data);
        const std::vector<float> y0 = c.randn(fold.size());
        for (const int nt : {256, 200}) {
            const std::string tag = std::string(" (") + vn + ", gain 0.9375), n " + std::to_string(nt);
            Buf y(fold.size() * 4);
            y.fill(0xff);
            hip::launch(fgm, {cdiv(nt, 128), static_cast<unsigned>(nr / 128), 1}, {256, 1, 1}, 0, c.s, w.p(), hm.row_bytes, dgain.p(), hxm.p(), hxsm.p(), y.p(), nc,
                        nr, nt, 0, 0);
            c.sync();
            std::vector<std::uint32_t> got = y.down<std::uint32_t>(fold.size()), want(fold.size(), 0xffffffffu);
            std::memcpy(want.data(), fold.data(), static_cast<std::size_t>(nt) * nr * 4);
            c.rep.add(cmpExact("w4q_gemm_m f32 == round 1 host fold BITWISE" + tag + " (rows past n untouched)", got, want));
            Buf ya(y0);
            hip::launch(fgm, {cdiv(nt, 128), static_cast<unsigned>(nr / 128), 1}, {256, 1, 1}, 0, c.s, w.p(), hm.row_bytes, dgain.p(), hxm.p(), hxsm.p(), ya.p(), nc,
                        nr, nt, 1, 0);
            c.sync();
            std::vector<float> wa(y0);
            for (std::size_t i = 0; i < static_cast<std::size_t>(nt) * nr; ++i) wa[i] = y0[i] + fold[i];
            c.rep.add(cmpExact("w4q_gemm_m f32 accumulate == y + round 1 host fold BITWISE" + tag, ya.down<float>(wa.size()), wa));
            Buf yh(fold.size() * 2);
            yh.fill(0xff);
            hip::launch(fgm, {cdiv(nt, 128), static_cast<unsigned>(nr / 128), 1}, {256, 1, 1}, 0, c.s, w.p(), hm.row_bytes, dgain.p(), hxm.p(), hxsm.p(), yh.p(), nc,
                        nr, nt, 0, 1);
            c.sync();
            std::vector<std::uint16_t> wh(fold.size(), 0xffff);
            for (std::size_t i = 0; i < static_cast<std::size_t>(nt) * nr; ++i) wh[i] = w4s_f2h(fold[i]);
            c.rep.add(cmpExact("w4q_gemm_m f16 store == f16(round 1 host fold) BITWISE" + tag, yh.down<std::uint16_t>(wh.size()), wh));
            // round 2b: the 256-token tile w4q_gemm_m2 (WHIRL_W4A4=4), 512 threads: the same bits
            Buf y2(fold.size() * 4), y2a(y0), y2h(fold.size() * 2);
            y2.fill(0xff);
            y2h.fill(0xff);
            const hip::Dim3 g2{cdiv(nt, 256), static_cast<unsigned>(nr / 128), 1};
            hip::launch(fg2, g2, {512, 1, 1}, 0, c.s, w.p(), hm.row_bytes, dgain.p(), hxm.p(), hxsm.p(), y2.p(), nc, nr, nt, 0, 0);
            hip::launch(fg2, g2, {512, 1, 1}, 0, c.s, w.p(), hm.row_bytes, dgain.p(), hxm.p(), hxsm.p(), y2a.p(), nc, nr, nt, 1, 0);
            hip::launch(fg2, g2, {512, 1, 1}, 0, c.s, w.p(), hm.row_bytes, dgain.p(), hxm.p(), hxsm.p(), y2h.p(), nc, nr, nt, 0, 1);
            c.sync();
            c.rep.add(cmpExact("w4q_gemm_m2 f32 == round 1 host fold BITWISE" + tag + " (rows past n untouched)", y2.down<std::uint32_t>(fold.size()), want));
            c.rep.add(cmpExact("w4q_gemm_m2 f32 accumulate == y + round 1 host fold BITWISE" + tag, y2a.down<float>(wa.size()), wa));
            c.rep.add(cmpExact("w4q_gemm_m2 f16 store == f16(round 1 host fold) BITWISE" + tag, y2h.down<std::uint16_t>(wh.size()), wh));
        }
        // device round 1 (w4q_wconv_mxfp4 + w4q_quant_x + w4q_gemm) vs device round 2 (w4q_quant_xm + w4q_gemm_m), f16 store, n 200
        Buf dwq(wq_h.size()), dws(ws_h.size() * 4), y1(fold.size() * 2), y2(fold.size() * 2);
        const std::uint64_t total = static_cast<std::uint64_t>(nr) * (nc / 32);
        hip::launch(fwc, {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(total, 256), 65535)), 1, 1}, {256, 1, 1}, 0, c.s, w.p(), hm.row_bytes, dwq.p(),
                    dws.p(), nc, nr, dgain.p());
        y1.fill(0xff);
        y2.fill(0xff);
        hip::launch(fg1, {cdiv(200, 128), static_cast<unsigned>(nr / 128), 1}, {256, 1, 1}, 0, c.s, dwq.p(), dws.p(), dxq.p(), dxs.p(), y1.p(), nc, nr, 200, 0, 1);
        hip::launch(fgm, {cdiv(200, 128), static_cast<unsigned>(nr / 128), 1}, {256, 1, 1}, 0, c.s, w.p(), hm.row_bytes, dgain.p(), dxm.p(), dxsm.p(), y2.p(), nc,
                    nr, 200, 0, 1);
        c.sync();
        c.rep.add(cmpExact(std::string("w4q_gemm_m (LUT form) == w4q_gemm (round 1 converted copy) BITWISE on the device, f16 store, n 200 (") + vn + ")",
                           y2.down<std::uint16_t>(fold.size()), y1.down<std::uint16_t>(fold.size())));
    }
}

void testW4s(Ctx& c) {
    testW4q(c);  // v0252-w4a4p: the product kernel rows first
    testW4qM(c);  // v0252-w4a4p round 2: the LUT form
    const bool g1151 = c.mod.arch() == hip::Arch::gfx1151;
    const hip::Function fdq = c.fnOpt("w4s_dequant_f16_mxfp4"), fqx = c.fnOpt("w4s_quant_x16");
    // round 2: one dequant entry per weight grid (kW4sDqNames: w, w1, w1e, w1d, wb), all loaded into MatmulKernels::w4s_dq
    std::array<hip::Function, 5> fg{};
    bool all_g = true, none_g = true;
    for (std::size_t g = 0; g < fg.size(); ++g) {
        fg[g] = c.fnOpt(wk::kW4sDqNames[g]);
        all_g = all_g && fg[g] != nullptr && c.k.w4s_dq[g] == fg[g];
        none_g = none_g && fg[g] == nullptr && c.k.w4s_dq[g] == nullptr;
    }
    {
        Result r;
        r.kind = Kind::invariant;
        r.n = 1;
        r.name = g1151 ? "w4s_dequant_f16_mxfp4 / w4s_quant_x16 in the gfx1151 code object and loaded into the host table"
                       : "w4s_* absent from this non-gfx1151 code object and from the host table";
        r.pass = g1151 ? (all_g && fqx != nullptr && c.k.w4s_qx == fqx) : (none_g && fqx == nullptr && c.k.w4s_qx == nullptr);
        r.mismatches = r.pass ? 0 : 1;
        c.rep.add(r);
    }
    if (fdq == nullptr || fqx == nullptr) return;
    const hip::Function c2f16 = c.k.gemmc[2][static_cast<int>(QType::f16)];
    const hip::Function i4fb = c.fnOpt("w4a4p_gemm_i4fb");
    const int nr = 256, nc = 1024, nt = 256;  // the probe's shapeOk (rows % 128, tokens % 256, K % 256)
    // weights: random blocks (variant 0) and an exponent sweep 96..150 (variant 1)
    std::vector<std::uint16_t> wsim_h;
    HostMat hm0;
    for (int variant = 0; variant < 2; ++variant) {
        HostMat hm = randomMat(c, QType::mxfp4, nr, nc);
        if (variant == 1)
            for (std::size_t b = 0; b < hm.data.size(); b += 136)
                for (int i = 0; i < 8; ++i) hm.data[b + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(96 + (b / 136 * 8 + static_cast<std::size_t>(i)) % 55);
        Buf w(hm.data), w16(static_cast<std::size_t>(nr) * nc * 2);
        const std::uint64_t groups = static_cast<std::uint64_t>(nc / 8) * nr;
        for (std::size_t g = 0; g < fg.size(); ++g) {  // round 2: every weight grid, bitwise against its host reference
            hip::launch(fg[g], {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(groups, 256), 65535)), 1, 1}, {256, 1, 1}, 0, c.s, w.p(), hm.row_bytes,
                        w16.p(), nc, nr);
            c.sync();
            std::vector<std::uint16_t> want(static_cast<std::size_t>(nr) * nc);
            for (int r = 0; r < nr; ++r)
                w4s_dequant_row_host(hm.data.data() + static_cast<std::size_t>(r) * hm.row_bytes, nc, want.data() + static_cast<std::size_t>(r) * nc,
                                     static_cast<int>(g));
            c.rep.add(cmpExact(std::string(wk::kW4sDqNames[g]) + " == host reference (" + (variant ? "e 96..150" : "random") + ", 256 x 1024)",
                               w16.down<std::uint16_t>(want.size()), want));
            if (variant == 0 && g == 0) {
                wsim_h = want;
                hm0 = hm;
            }
        }
    }
    // round 3: the least-squares gains. w4s_gain_terms (one thread per row) == w4s_gain_terms_row bitwise (num / den, double),
    // grids w and w1d; then the gain entries w4s_dequant_f16_mxfp4*_gn with the gains ops::w4sGains makes from the host terms ==
    // w4s_dequant_row_host(..., gain) bitwise for w + g (tensor), w + r (row), w1d + r. Presence: the gain kernels exist where
    // the plain ones do and are loaded into MatmulKernels::w4s_gterms / w4s_dqg.
    {
        const hip::Function gt = c.fnOpt("w4s_gain_terms");
        bool loaded = gt != nullptr && c.k.w4s_gterms == gt;
        for (std::size_t g = 0; g < 5; ++g) loaded = loaded && c.fnOpt(wk::kW4sDqgNames[g]) != nullptr && c.k.w4s_dqg[g] == c.fnOpt(wk::kW4sDqgNames[g]);
        Result r;
        r.kind = Kind::invariant;
        r.n = 1;
        r.name = "w4s_gain_terms / w4s_dequant_f16_mxfp4*_gn in the gfx1151 code object and loaded into the host table";
        r.pass = loaded;
        r.mismatches = r.pass ? 0 : 1;
        c.rep.add(r);
        if (loaded) {
            Buf w(hm0.data), dn(static_cast<std::size_t>(nr) * 8), dd(static_cast<std::size_t>(nr) * 8), w16(static_cast<std::size_t>(nr) * nc * 2);
            const struct {
                int grid, mode;
                const char* name;
            } cases[] = {{0, 1, "w + g (tensor gain)"}, {0, 2, "w + r (row gains)"}, {3, 2, "w1d + r (row gains)"}};
            for (const auto& cs : cases) {
                hip::launch(gt, {cdiv(nr, 64), 1, 1}, {64, 1, 1}, 0, c.s, w.p(), hm0.row_bytes, nc, nr, cs.grid, dn.p(), dd.p());
                c.sync();
                std::vector<double> hn(static_cast<std::size_t>(nr)), hd(static_cast<std::size_t>(nr));
                for (int r = 0; r < nr; ++r)
                    w4s_gain_terms_row(hm0.data.data() + static_cast<std::size_t>(r) * hm0.row_bytes, nc, cs.grid, &hn[static_cast<std::size_t>(r)],
                                       &hd[static_cast<std::size_t>(r)]);
                if (cs.mode == 1 || cs.grid == 3) {  // once per grid
                    c.rep.add(cmpExact(std::string("w4s_gain_terms num == host (grid ") + std::to_string(cs.grid) + ", 256 rows)", dn.down<double>(hn.size()), hn));
                    c.rep.add(cmpExact(std::string("w4s_gain_terms den == host (grid ") + std::to_string(cs.grid) + ", 256 rows)", dd.down<double>(hd.size()), hd));
                }
                std::vector<float> gains(static_cast<std::size_t>(nr));
                whirl::ops::w4sGains(hn.data(), hd.data(), static_cast<std::uint32_t>(nr), cs.mode, gains.data());
                Buf dg(gains);
                const std::uint64_t groups = static_cast<std::uint64_t>(nc / 8) * nr;
                hip::launch(c.k.w4s_dqg[static_cast<std::size_t>(cs.grid)], {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(groups, 256), 65535)), 1, 1},
                            {256, 1, 1}, 0, c.s, w.p(), hm0.row_bytes, w16.p(), nc, nr, dg.p());
                c.sync();
                std::vector<std::uint16_t> want(static_cast<std::size_t>(nr) * nc);
                for (int r = 0; r < nr; ++r)
                    w4s_dequant_row_host(hm0.data.data() + static_cast<std::size_t>(r) * hm0.row_bytes, nc, want.data() + static_cast<std::size_t>(r) * nc,
                                         cs.grid, gains[static_cast<std::size_t>(r)]);
                c.rep.add(cmpExact(std::string(wk::kW4sDqgNames[cs.grid]) + " " + cs.name + " == host reference (256 x 1024)", w16.down<std::uint16_t>(want.size()),
                                   want));
            }
        }
    }
    // activations: Gaussian f16 rows (sigma 1), a zero block and an outlier per row
    std::vector<float> x = c.randn(static_cast<std::size_t>(nt) * nc);
    for (int t = 0; t < nt; ++t) {
        for (int i = 64; i < 96; ++i) x[static_cast<std::size_t>(t) * nc + static_cast<std::size_t>(i)] = 0.f;
        x[static_cast<std::size_t>(t) * nc + 200 + static_cast<std::size_t>(t % 32)] = 30.f;
    }
    const std::vector<std::uint16_t> xh = toHalf(x);
    Buf dx(xh), dq(xh.size() * 2);
    const std::uint64_t nblk = xh.size() / 32;
    std::vector<std::uint16_t> xsim_h[2];
    for (int mse = 0; mse < 2; ++mse) {
        hip::launch(fqx, {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(nblk, 256), 65535)), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dq.p(), nblk, mse);
        c.sync();
        xsim_h[mse].resize(xh.size());
        w4s_quant_row_host(xh.data(), static_cast<int>(xh.size()), mse, xsim_h[mse].data());
        c.rep.add(cmpExact(std::string("w4s_quant_x16 == host reference (") + (mse ? "MSE-best step" : "absmax / 7") + ", 256 x 1024)",
                           dq.down<std::uint16_t>(xh.size()), xsim_h[mse]));
    }
    if (!c2f16) {
        c.rep.skip("w4s", "sim GEMM rows", "no gemm_c2_f16 in this code object");
        return;
    }
    // the sim GEMM: re-quantized f16 weights x fake-quantized f16 activations through gemm_c2_f16 (the dequant-path GEMM)
    Buf wq16(wsim_h), xq16(xsim_h[0]), y(static_cast<std::size_t>(nt) * nr * 4);
    const auto& c2 = wk::kGemmCfgs[2];
    y.fill(0xff);
    hip::launch(c2f16, {cdiv(nt, c2.bn), cdiv(nr, c2.bm), 1}, {static_cast<unsigned>(c2.nth), 1, 1}, 0, c.s, wq16.p(), static_cast<std::uint64_t>(nc) * 2,
                xq16.p(), y.p(), nc, nr, nt, 0);
    c.sync();
    const std::vector<float> ysim = y.down<float>(static_cast<std::size_t>(nt) * nr);
    {
        std::vector<float> wf(wsim_h.size()), xf(xsim_h[0].size());
        for (std::size_t i = 0; i < wf.size(); ++i) wf[i] = h2f(wsim_h[i]);
        for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = h2f(xsim_h[0][i]);
        std::vector<double> ry, rs;
        ref::gemvF64(wf, nr, nc, xf.data(), nt, nc, ry, rs);
        c.rep.add(cmpTol("w4s sim GEMM (w4s_dq + w4s_qx + gemm_c2_f16) vs CPU f64 of the same f16 values, 256 x 1024 x 256", ysim, ry, rs, 1e-4, 1e-6));
    }
    // the probe's numbers on the same inputs: packed int4 + f32 scales (kt_w4a4p.h), CPU exact fold and the i4fb kernel
    std::vector<std::uint8_t> wq(static_cast<std::size_t>(nr) * nc / 2), xq(static_cast<std::size_t>(nt) * nc / 2);
    std::vector<float> wsc(static_cast<std::size_t>(nr) * nc / 32), xsc(static_cast<std::size_t>(nt) * nc / 32);
    for (int r = 0; r < nr; ++r)
        w4a4p::mxRowToI4(hm0.data.data() + static_cast<std::size_t>(r) * hm0.row_bytes, nc, wq.data() + static_cast<std::size_t>(r) * nc / 2,
                         wsc.data() + static_cast<std::size_t>(r) * nc / 32);
    std::vector<float> xw(static_cast<std::size_t>(nc));
    for (int t = 0; t < nt; ++t) {
        for (int i = 0; i < nc; ++i) xw[static_cast<std::size_t>(i)] = h2f(xh[static_cast<std::size_t>(t) * nc + static_cast<std::size_t>(i)]);
        w4a4p::quantRow(xw.data(), nc, xq.data() + static_cast<std::size_t>(t) * nc / 2, xsc.data() + static_cast<std::size_t>(t) * nc / 32);
    }
    std::vector<double> fy(static_cast<std::size_t>(nt) * nr), fs(fy.size());
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < nr; ++r)
            w4a4p::refFoldF64(wq.data() + static_cast<std::size_t>(r) * nc / 2, wsc.data() + static_cast<std::size_t>(r) * nc / 32,
                              xq.data() + static_cast<std::size_t>(t) * nc / 2, xsc.data() + static_cast<std::size_t>(t) * nc / 32, nc,
                              fy[static_cast<std::size_t>(t) * nr + static_cast<std::size_t>(r)], fs[static_cast<std::size_t>(t) * nr + static_cast<std::size_t>(r)]);
    const double rtol = std::ldexp(1.0, -10) + std::ldexp(1.0, -12);
    c.rep.add(cmpTol("w4s sim GEMM == the W4A4 per-32 fold (CPU exact, kt_w4a4p refFoldF64) within 2^-10 + 2^-12 of sum |terms|", ysim, fy, fs, rtol, 1e-6));
    if (i4fb == nullptr) {
        c.rep.skip("w4s", "w4s sim GEMM == w4a4p_gemm_i4fb", "no w4a4p_gemm_i4fb in this code object");
        return;
    }
    Buf dwq(wq), dws(wsc), dxq(xq), dxs(xsc), yp(static_cast<std::size_t>(nt) * nr * 4);
    yp.fill(0xff);
    hip::launch(i4fb, {cdiv(nt, 128), cdiv(nr, 128), 1}, {256, 1, 1}, 0, c.s, dwq.p(), dws.p(), dxq.p(), dxs.p(), yp.p(), nc, nr, nt);
    c.sync();
    const std::vector<float> yprobe = yp.down<float>(static_cast<std::size_t>(nt) * nr);
    c.rep.add(cmpTol("w4a4p_gemm_i4fb (the probe kernel) == the CPU exact fold, 1e-5 x sum |terms|", yprobe, fy, fs, 1e-5, 1e-6));
    const std::vector<double> ysd(ysim.begin(), ysim.end());
    c.rep.add(cmpTol("w4s sim GEMM == w4a4p_gemm_i4fb on the same inputs within 2^-10 + 2^-12 of sum |terms|", yprobe, ysd, fs, rtol, 1e-6));
}

}  // namespace kt
