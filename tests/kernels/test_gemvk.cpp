// whirl-kernel-test family "gemvk" (feat026-ggemv): the gfx1151 K-balanced MXFP4 multi-token GEMV
// gemvk_nt<N>v<V>_mxfp4 / ggemvk_* (kernels/gfx1151/gemv_gk.hip, opt-in WHIRL_GEMV_K).
//   * every variant v1..v9 x 2..8 tokens, plain and grouped (3 segments), == the 1-token gemvq_mxfp4 per token
//     row, bitwise (the checkGemvBitwise contract), and == the production gemvq_nt<N>r2_mxfp4;
//   * K = 512 / 2048 / 4096 / 5120 / 6144 (2 / 8 / 16 / 20 / 24 superblocks: tails, idle groups, every KW split)
//     and row counts that leave a partial last block for 16 / 32 / 64-row blocks;
//   * accumulate = 1 adds onto y exactly once; rows past nrows are not written;
//   * the 1-token reference itself against the double CPU product (tolerance), first 128 rows.
// Skipped when the code object has no gemvk (gfx1201, older objects).
// SPDX-License-Identifier: Apache-2.0

#include "cpu_ref.h"

#include <cstdio>
#include <cstring>

namespace kt {

void testGemvk(Ctx& c) {
    c.rep.family = "gemvk";
    if (c.k.gemvk[2][0] == nullptr) {
        c.rep.skip("gemvk", "gemvk_* (gfx1151 MXFP4)", "kernel not in this code object");
        return;
    }
    constexpr int kMaxNt = wk::kGemvkMaxNt;
    struct Shape {
        int nrows, ncols;
    };
    const Shape shapes[] = {{1000, 512}, {1000, 2048}, {520, 4096}, {1000, 5120}, {777, 6144}};
    const int ti = static_cast<int>(QType::mxfp4);
    for (const Shape& sh : shapes) {
        const HostMat m = randomMat(c, QType::mxfp4, sh.nrows, sh.ncols);
        const int nrows = m.nrows, ncols = m.ncols;
        const std::string tag = std::to_string(nrows) + "x" + std::to_string(ncols);
        Buf w(m.data);
        const std::vector<float> x = c.randn(static_cast<std::size_t>(kMaxNt) * ncols);
        Buf dx(x), xq(static_cast<std::size_t>(kMaxNt) * ncols), xd(static_cast<std::size_t>(kMaxNt) * ncols / 32 * 4);
        hip::launch(c.k.quantize_q8, {cdiv(static_cast<std::uint64_t>(kMaxNt) * ncols, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), xq.p(),
                    xd.p(), kMaxNt * ncols);
        // reference: each token alone through the 1-token gemvq_mxfp4 (GvArgs entry, 8 rows per block)
        Buf yref(static_cast<std::size_t>(kMaxNt) * nrows * 4);
        for (int t = 0; t < kMaxNt; ++t) {
            const wk::GvArgs g = wk::gvSingle(w.p(), yref.p() + static_cast<std::uint64_t>(t) * nrows * 4, m.row_bytes, nrows);
            hip::launch(c.k.gemvq[ti], {cdiv(nrows, 8), 1, 1}, {256, 1, 1}, 0, c.s, g, xq.p() + static_cast<std::uint64_t>(t) * ncols,
                        xd.p() + static_cast<std::uint64_t>(t) * (ncols / 32) * 4, ncols, 0, DevPtr{0});
        }
        c.sync();
        const std::vector<float> ref1 = yref.down<float>(static_cast<std::size_t>(kMaxNt) * nrows);
        {  // the reference vs the CPU (int8 activations, double accumulation)
            const int crows = std::min(nrows, 128);
            const auto hq = xq.down<std::int8_t>(static_cast<std::size_t>(kMaxNt) * ncols);
            const auto hd = xd.down<float>(static_cast<std::size_t>(kMaxNt) * ncols / 32);
            const std::vector<float> wf = ref::dequantRows(m, 0, crows);
            std::vector<double> ry, rs;
            ref::gemvQ8F64(wf, crows, ncols, hq.data(), hd.data(), kMaxNt, ry, rs);
            std::vector<float> got;
            for (int t = 0; t < kMaxNt; ++t)
                got.insert(got.end(), ref1.begin() + static_cast<std::ptrdiff_t>(t) * nrows,
                           ref1.begin() + static_cast<std::ptrdiff_t>(t) * nrows + crows);
            c.rep.add(cmpTol("gemvq_mxfp4 (1-token reference) vs CPU " + tag, got, ry, rs, 1e-4, 1e-6));
        }
        std::size_t variants = 0, bad = 0, bad_untouched = 0;
        std::string first_bad;
        Buf y(static_cast<std::size_t>(kMaxNt + 1) * nrows * 4);
        for (int nt = 2; nt <= kMaxNt; ++nt) {
            const std::size_t cnt = static_cast<std::size_t>(nt) * nrows;
            const std::vector<float> want(ref1.begin(), ref1.begin() + static_cast<std::ptrdiff_t>(cnt));
            // the production entry first (the comparison the selector stands on)
            {
                const hip::Function f0 = c.k.gemvqMr(QType::mxfp4, nt, 2, false);
                if (f0 != nullptr) {
                    y.fill(0xff);
                    hip::launch(f0, {cdiv(nrows, 16), 1, 1}, {256, 1, 1}, 0, c.s, w.p(), m.row_bytes, xq.p(), xd.p(), y.p(), ncols, nrows, 0,
                                DevPtr{0});
                    c.sync();
                    const auto got = y.down<float>(cnt);
                    ++variants;
                    if (std::memcmp(got.data(), want.data(), cnt * 4) != 0) {
                        ++bad;
                        if (first_bad.empty()) first_bad = "nt=" + std::to_string(nt) + " gemvq_nt" + std::to_string(nt) + "r2";
                    }
                }
            }
            for (int v = 1; v < wk::kNGemvk; ++v) {
                const int rows = wk::gemvkRows(v);
                for (const bool grouped : {false, true}) {
                    const hip::Function f = c.k.gemvK(nt, v, grouped);
                    if (f == nullptr) continue;
                    const std::string lab = "nt=" + std::to_string(nt) + (grouped ? " g" : " ") + "v" + std::to_string(v);
                    y.fill(0xff);
                    if (grouped) {
                        // three row ranges of the matrix as three segments; segment starts on block boundaries of
                        // their own (block index rebased per segment)
                        const int n0 = nrows / 3, n1 = nrows / 3, n2 = nrows - n0 - n1;
                        wk::GvArgs g;
                        g.w[0] = w.p();
                        g.w[1] = w.p() + static_cast<std::uint64_t>(n0) * m.row_bytes;
                        g.w[2] = w.p() + static_cast<std::uint64_t>(n0 + n1) * m.row_bytes;
                        // multi-token output is token-major per matrix: y[t * n_seg + r]; use one y area per segment
                        g.y[0] = y.p();
                        g.y[1] = y.p() + static_cast<std::uint64_t>(nt) * n0 * 4;
                        g.y[2] = y.p() + static_cast<std::uint64_t>(nt) * (n0 + n1) * 4;
                        g.rb[0] = g.rb[1] = g.rb[2] = m.row_bytes;
                        g.n[0] = n0;
                        g.n[1] = n1;
                        g.n[2] = n2;
                        g.b1 = static_cast<int>(cdiv(n0, rows));
                        g.b2 = g.b1 + static_cast<int>(cdiv(n1, rows));
                        const unsigned nb = static_cast<unsigned>(g.b2) + cdiv(n2, rows);
                        hip::launch(f, {nb, 1, 1}, {256, 1, 1}, 0, c.s, g, xq.p(), xd.p(), ncols, 0, DevPtr{0});
                        c.sync();
                        const auto all = y.down<float>(cnt + nrows);
                        // re-assemble token-major [t][row] from the three segments
                        std::vector<float> got(cnt);
                        const int ns[3] = {n0, n1, n2}, off[3] = {0, n0, n0 + n1};
                        for (int s = 0; s < 3; ++s)
                            for (int t = 0; t < nt; ++t)
                                for (int r = 0; r < ns[s]; ++r)
                                    got[static_cast<std::size_t>(t) * nrows + off[s] + r] =
                                        all[static_cast<std::size_t>(nt) * off[s] + static_cast<std::size_t>(t) * ns[s] + r];
                        ++variants;
                        if (std::memcmp(got.data(), want.data(), cnt * 4) != 0) {
                            ++bad;
                            if (first_bad.empty()) first_bad = lab;
                        }
                        for (std::size_t i = cnt; i < all.size(); ++i) {  // nothing written past the last segment
                            std::uint32_t u;
                            std::memcpy(&u, &all[i], 4);
                            if (u != 0xffffffffu) {
                                ++bad_untouched;
                                break;
                            }
                        }
                    } else {
                        hip::launch(f, {cdiv(nrows, rows), 1, 1}, {256, 1, 1}, 0, c.s, w.p(), m.row_bytes, xq.p(), xd.p(), y.p(), ncols, nrows, 0,
                                    DevPtr{0});
                        c.sync();
                        const auto all = y.down<float>(cnt + nrows);
                        ++variants;
                        if (std::memcmp(all.data(), want.data(), cnt * 4) != 0) {
                            ++bad;
                            if (first_bad.empty()) first_bad = lab;
                        }
                        for (std::size_t i = cnt; i < all.size(); ++i) {
                            std::uint32_t u;
                            std::memcpy(&u, &all[i], 4);
                            if (u != 0xffffffffu) {
                                ++bad_untouched;
                                break;
                            }
                        }
                    }
                }
            }
            // accumulate = 1 on one variant per nt (v2: KW 4): y = y0 + gemv exactly
            if (const hip::Function fa = c.k.gemvK(nt, 2, false)) {
                const std::vector<float> y0 = c.randn(cnt);
                Buf ya(y0);
                hip::launch(fa, {cdiv(nrows, wk::gemvkRows(2)), 1, 1}, {256, 1, 1}, 0, c.s, w.p(), m.row_bytes, xq.p(), xd.p(), ya.p(), ncols,
                            nrows, 1, DevPtr{0});
                c.sync();
                std::vector<float> wa(cnt);
                for (std::size_t i = 0; i < cnt; ++i) wa[i] = y0[i] + want[i];
                c.rep.add(cmpExact("gemvk_nt" + std::to_string(nt) + "v2 accumulate == y0 + y " + tag, ya.down<float>(cnt), wa,
                                   Kind::invariant));
            }
        }
        Result r;
        r.family = "gemvk";
        r.name = "gemvk v1..v9 (+ grouped) and gemvq_nt*r2 == 1-token gemvq_mxfp4, 2.." + std::to_string(kMaxNt) + " tokens " + tag;
        r.kind = Kind::invariant;
        r.n = variants;
        r.mismatches = bad + bad_untouched;
        r.pass = bad == 0 && bad_untouched == 0 && variants > 0;
        if (!first_bad.empty()) r.note = "first mismatch " + first_bad;
        if (bad_untouched) r.note += " (rows past the matrix written)";
        std::printf("  [info] gemvk %s: %zu variant runs, %zu mismatches, %zu wrote past the end\n", tag.c_str(), variants, bad, bad_untouched);
        c.rep.add(r);
    }
}

}  // namespace kt
