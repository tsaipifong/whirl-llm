// whirl-kernel-test: decode GEMVs.
//   * gemv_<T>_<1|4|8> (f32 activations) and gemvq_<T> (int8 activations) vs
//     a double-precision CPU product (tolerance);
//   * the prototype's checkGemvBitwise: every multi-token (2..16), multi-row
//     (R = 2, 4) and int8-WMMA (v1..v8) GEMV, and their grouped twins, must
//     reproduce the 1-token gemvq_<T> result bitwise per token row;
//   * grouped one-launch GEMV over 3 matrices == separate launches (bitwise);
//   * accumulate = 1 adds onto y exactly once (bitwise);
//   * the 2-bit MTP draft head GEMV (gemv_d2_nt*) multi-token == 1-token.
// SPDX-License-Identifier: Apache-2.0

#include "cpu_ref.h"

#include <cstdio>
#include <cstdlib>

namespace kt {

namespace {

constexpr int kNt = wk::kMaxSmallBatch;

// One-token int8 GEMV of token t through the GvArgs entry.
void gemvq1(Ctx& c, QType t, const Buf& w, std::uint64_t rb, int nrows, DevPtr xq, DevPtr xd, DevPtr y, int ncols,
            int acc) {
    const wk::GvArgs g = wk::gvSingle(w.p(), y, rb, nrows);
    hip::launch(c.k.gemvq[static_cast<int>(t)], {cdiv(nrows, 8), 1, 1}, {256, 1, 1}, 0, c.s, g, xq, xd, ncols, acc,
                DevPtr{0});
}

}  // namespace

void testGemv(Ctx& c) {
    c.rep.family = "gemv";
    const int cpu_rows = c.quick ? 256 : 1024;
    const std::vector<HostMat> mats = sampleMats(c, c.quick ? 1024 : 4096);
    for (const HostMat& m : mats) {
        const int ti = static_cast<int>(m.type);
        const std::string sfx = wk::typeSuffix(m.type);
        const int ncols = m.ncols, nrows = m.nrows;
        Buf w(m.data);
        const std::vector<float> x = c.randn(static_cast<std::size_t>(kNt) * ncols);
        Buf dx(x);
        const int crows = std::min(cpu_rows, nrows);
        const std::vector<float> wf = ref::dequantRows(m, 0, crows);

        // f32-activation GEMV, 1 / 4 / 8 tokens
        for (int nt : {1, 4, 8}) {
            Buf y(static_cast<std::size_t>(nt) * nrows * 4);
            hip::launch(c.fn("gemv_" + sfx + "_" + std::to_string(nt)), {cdiv(nrows, 8), 1, 1}, {256, 1, 1}, 0, c.s,
                        w.p(), m.row_bytes, dx.p(), y.p(), ncols, nrows, ncols, nrows, 0);
            c.sync();
            const auto all = y.down<float>(static_cast<std::size_t>(nt) * nrows);
            std::vector<float> got;
            for (int t = 0; t < nt; ++t)
                got.insert(got.end(), all.begin() + static_cast<std::ptrdiff_t>(t) * nrows,
                           all.begin() + static_cast<std::ptrdiff_t>(t) * nrows + crows);
            std::vector<double> ry, rs;
            ref::gemvF64(wf, crows, ncols, x.data(), nt, ncols, ry, rs);
            c.rep.add(cmpTol("gemv_" + sfx + "_" + std::to_string(nt) + " " + m.name, got, ry, rs, 1e-4, 1e-6));
        }

        // f32 / f16 (no int8 GEMV): do the multi-token instances gemv_<T>_4 / _8 reproduce
        // gemv_<T>_1 (the n = 1 decode path) bitwise per token row? (C-13: production n = 2..16
        // runs the f16 GEMM instead, which does not; _4 / _8 are candidates for a fix.)
        // Informational by default; WHIRL_KT_FLOAT_GEMV_BITWISE=1 makes a mismatch a failure.
        if (m.type == QType::f32 || m.type == QType::f16) {
            const char* sv = std::getenv("WHIRL_KT_FLOAT_GEMV_BITWISE");
            const bool strict = sv != nullptr && sv[0] == '1';
            std::vector<float> ref1(static_cast<std::size_t>(kNt) * nrows);
            Buf y1(static_cast<std::size_t>(nrows) * 4);
            for (int t = 0; t < kNt; ++t) {
                hip::launch(c.fn("gemv_" + sfx + "_1"), {cdiv(nrows, 8), 1, 1}, {256, 1, 1}, 0, c.s, w.p(), m.row_bytes,
                            dx.p() + static_cast<DevPtr>(t) * ncols * 4, y1.p(), ncols, nrows, ncols, nrows, 0);
                c.sync();
                const auto r1 = y1.down<float>(static_cast<std::size_t>(nrows));
                std::copy(r1.begin(), r1.end(), ref1.begin() + static_cast<std::ptrdiff_t>(t) * nrows);
            }
            for (int nt : {4, 8}) {
                Buf y(static_cast<std::size_t>(nt) * nrows * 4);
                hip::launch(c.fn("gemv_" + sfx + "_" + std::to_string(nt)), {cdiv(nrows, 8), 1, 1}, {256, 1, 1}, 0, c.s,
                            w.p(), m.row_bytes, dx.p(), y.p(), ncols, nrows, ncols, nrows, 0);
                c.sync();
                const auto got = y.down<float>(static_cast<std::size_t>(nt) * nrows);
                const std::vector<float> want(ref1.begin(), ref1.begin() + static_cast<std::ptrdiff_t>(nt) * nrows);
                Result r = cmpExact("gemv_" + sfx + "_" + std::to_string(nt) + " == gemv_" + sfx + "_1 per token " + m.name,
                                    got, want, Kind::invariant);
                std::printf("  [info] gemv  gemv_%s_%d vs gemv_%s_1 per token (%s): %zu of %zu differ\n", sfx.c_str(), nt,
                            sfx.c_str(), m.name.c_str(), r.mismatches, r.n);
                if (!r.pass && !strict) {
                    r.pass = true;
                    r.note += " [known: not bitwise, informational (WHIRL_KT_FLOAT_GEMV_BITWISE=1 to enforce)]";
                }
                c.rep.add(r);
            }
        }

        if (c.k.gemvq[ti] == nullptr) continue;
        // int8 activations for all 16 tokens
        Buf xq(static_cast<std::size_t>(kNt) * ncols), xd(static_cast<std::size_t>(kNt) * ncols / 32 * 4);
        hip::launch(c.k.quantize_q8, {cdiv(static_cast<std::uint64_t>(kNt) * ncols, 256), 1, 1}, {256, 1, 1}, 0, c.s,
                    dx.p(), xq.p(), xd.p(), kNt * ncols);
        // reference: each token alone through gemvq_<T>
        Buf yref(static_cast<std::size_t>(kNt) * nrows * 4);
        for (int t = 0; t < kNt; ++t)
            gemvq1(c, m.type, w, m.row_bytes, nrows, xq.p() + static_cast<std::uint64_t>(t) * ncols,
                   xd.p() + static_cast<std::uint64_t>(t) * (ncols / 32) * 4,
                   yref.p() + static_cast<std::uint64_t>(t) * nrows * 4, ncols, 0);
        c.sync();
        const auto ref1 = yref.down<float>(static_cast<std::size_t>(kNt) * nrows);
        {
            const auto hq = xq.down<std::int8_t>(static_cast<std::size_t>(kNt) * ncols);
            const auto hd = xd.down<float>(static_cast<std::size_t>(kNt) * ncols / 32);
            const int ntc = c.quick ? 2 : 4;
            std::vector<double> ry, rs;
            ref::gemvQ8F64(wf, crows, ncols, hq.data(), hd.data(), ntc, ry, rs);
            std::vector<float> got;
            for (int t = 0; t < ntc; ++t)
                got.insert(got.end(), ref1.begin() + static_cast<std::ptrdiff_t>(t) * nrows,
                           ref1.begin() + static_cast<std::ptrdiff_t>(t) * nrows + crows);
            c.rep.add(cmpTol("gemvq_" + sfx + " (int8 x) " + m.name, got, ry, rs, 1e-4, 1e-6));
        }

        // accumulate: y = y0 + v exactly
        {
            std::vector<float> y0 = c.randn(static_cast<std::size_t>(nrows));
            Buf y(y0);
            gemvq1(c, m.type, w, m.row_bytes, nrows, xq.p(), xd.p(), y.p(), ncols, 1);
            c.sync();
            std::vector<float> want(static_cast<std::size_t>(nrows));
            for (int r = 0; r < nrows; ++r) want[static_cast<std::size_t>(r)] = y0[static_cast<std::size_t>(r)] + ref1[static_cast<std::size_t>(r)];
            c.rep.add(cmpExact("gemvq_" + sfx + " accumulate == y0 + y", y.down<float>(want.size()), want, Kind::invariant));
        }

        // grouped one-token launch: three row ranges as three "matrices"
        {
            const int n0 = nrows / 3, n1 = nrows / 3, n2 = nrows - n0 - n1;
            Buf y(static_cast<std::size_t>(nrows) * 4);
            y.fill(0xff);
            wk::GvArgs g;
            g.w[0] = w.p();
            g.w[1] = w.p() + static_cast<std::uint64_t>(n0) * m.row_bytes;
            g.w[2] = w.p() + static_cast<std::uint64_t>(n0 + n1) * m.row_bytes;
            g.y[0] = y.p();
            g.y[1] = y.p() + static_cast<std::uint64_t>(n0) * 4;
            g.y[2] = y.p() + static_cast<std::uint64_t>(n0 + n1) * 4;
            g.rb[0] = g.rb[1] = g.rb[2] = m.row_bytes;
            g.n[0] = n0;
            g.n[1] = n1;
            g.n[2] = n2;
            g.b1 = static_cast<int>(cdiv(n0, 8));
            g.b2 = g.b1 + static_cast<int>(cdiv(n1, 8));
            const unsigned nb = static_cast<unsigned>(g.b2) + cdiv(n2, 8);
            hip::launch(c.k.gemvq[ti], {nb, 1, 1}, {256, 1, 1}, 0, c.s, g, xq.p(), xd.p(), ncols, 0, DevPtr{0});
            c.sync();
            const std::vector<float> want(ref1.begin(), ref1.begin() + nrows);
            c.rep.add(cmpExact("gemvq_" + sfx + " grouped (3 segments) == single", y.down<float>(want.size()), want,
                               Kind::invariant));
        }

        // multi-token / multi-row / WMMA variants (+ grouped twins) == 1-token
        std::size_t variants = 0, bad = 0;
        std::string first_bad;
        Buf y(static_cast<std::size_t>(kNt) * nrows * 4);
        for (int nt = 2; nt <= kNt; ++nt) {
            struct V {
                hip::Function f;
                int rows;
                bool grouped;
                std::string label;
            };
            std::vector<V> vs;
            vs.push_back({c.k.gemvqNt(m.type, nt, false), 8, false, "nt"});
            vs.push_back({c.k.gemvqNt(m.type, nt, true), 8, true, "g nt"});
            for (int r : {2, 4}) {
                vs.push_back({c.k.gemvqMr(m.type, nt, r, false), 8 * r, false, "R" + std::to_string(r)});
                vs.push_back({c.k.gemvqMr(m.type, nt, r, true), 8 * r, true, "g R" + std::to_string(r)});
            }
            for (int v = 1; v < wk::kNGemvw; ++v) {
                vs.push_back({c.k.gemvW(m.type, nt, v, false), wk::kGemvwRows[static_cast<std::size_t>(v)], false, "W" + std::to_string(v)});
                vs.push_back({c.k.gemvW(m.type, nt, v, true), wk::kGemvwRows[static_cast<std::size_t>(v)], true, "g W" + std::to_string(v)});
            }
            for (const V& v : vs) {
                if (v.f == nullptr) continue;
                y.fill(0xff);
                const unsigned nb = cdiv(nrows, static_cast<unsigned>(v.rows));
                if (v.grouped)
                    hip::launch(v.f, {nb, 1, 1}, {256, 1, 1}, 0, c.s, wk::gvSingle(w.p(), y.p(), m.row_bytes, nrows),
                                xq.p(), xd.p(), ncols, 0, DevPtr{0});
                else
                    hip::launch(v.f, {nb, 1, 1}, {256, 1, 1}, 0, c.s, w.p(), m.row_bytes, xq.p(), xd.p(), y.p(), ncols,
                                nrows, 0, DevPtr{0});
                c.sync();
                const auto got = y.down<float>(static_cast<std::size_t>(nt) * nrows);
                ++variants;
                if (std::memcmp(got.data(), ref1.data(), got.size() * 4) != 0) {
                    ++bad;
                    if (first_bad.empty()) first_bad = "nt=" + std::to_string(nt) + " " + v.label;
                }
            }
        }
        Result r;
        r.name = "multi-token/multi-row/WMMA GEMV == 1-token (" + std::to_string(variants) + " kernels) " + m.name;
        r.kind = Kind::invariant;
        r.n = variants;
        r.mismatches = bad;
        r.pass = bad == 0 && variants > 0;
        if (!first_bad.empty()) r.note = "first " + first_bad;
        c.rep.add(r);
    }
}

}  // namespace kt
