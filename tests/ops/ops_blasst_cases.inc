// v0252-blasst (PLAN V1-70, T0 step 2): the CPU reference of the BLASST attention block skip (tests/kernels/blasst_ref.h),
// checked on its own before the kt rows (tests/kernels/test_attn.cpp, family attnkgl SK1 / PR) compare the GPU kernels with
// it. Test-oracle self-checks (no product code: they pass on the unchanged tree too; the product's test-first cases are
// testAttnSkipLimit / testAttnSkipSelect in ops_attention_cases.inc and testAttnSkipFlags in tests/unit):
//   (i)   lim = -inf: nothing skipped and the tile walk == the exact causal attention (rel L2 <= 1e-6);
//   (ii)  hot / cold tiles (p_cold 0.25 / 0.5 / 0.75, one pattern per head pair or different patterns for the two heads):
//         the skip count is exactly the tiles cold for that head after tile 0, the dropped mass delta is at most the
//         skipped keys' mass bound, the output error obeys |o' - o| <= 2 delta max|v| (research note 3.2), and the data
//         really moves the output (a wrong decision would show in the kt rows);
//   (iii) the first tile is never skipped (m = -inf), even when it is cold; a cold tile after a hot one is;
//   (iv)  causal: hot keys masked for every query of a block (past their positions) do not keep the diagonal tile;
//   deltaBin's edges (1e-3 / 1e-2 are bin edges: the R-M0 rule).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../test_check.h"
#include "../kernels/blasst_ref.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {

double relL2(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num += (a[i] - b[i]) * (a[i] - b[i]);
        den += b[i] * b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

void testBlasstRef() {
    const double scale = 1.0 / 16.0, ninf = -std::numeric_limits<double>::infinity();
    // (i) random data, lim = -inf: no skip, == exact (several block / tile alignments)
    {
        std::mt19937 g(20261110u);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (const auto& [n, pos0] : {std::pair{40, 70}, std::pair{16, 0}, std::pair{33, 5}}) {
            const int d = 32, nk = pos0 + n;
            std::vector<float> q(static_cast<std::size_t>(n) * d), k(static_cast<std::size_t>(nk) * d), v(static_cast<std::size_t>(nk) * d);
            for (auto& x : q) x = nd(g);
            for (auto& x : k) x = nd(g);
            for (auto& x : v) x = nd(g);
            const blasst::RefOut r = blasst::refAttention(q.data(), k.data(), v.data(), n, d, pos0, scale, ninf);
            const std::vector<double> ex = blasst::exactAttention(q.data(), k.data(), v.data(), n, d, pos0, scale);
            CHECK(r.skipped == 0 && r.tiles > 0 && relL2(r.o, ex) <= 1e-6);
            double dmax = 0.0;
            for (const double x : r.delta) dmax = std::max(dmax, x);
            CHECK(dmax == 0.0);
        }
    }
    // (ii) hot / cold tiles: n 128 queries at the end of 1024 keys (32 tiles, 8 blocks), d 64, lim -20 (margin >= 20)
    {
        const int n = 128, pos0 = 896, d = 64, nk = pos0 + n;
        std::size_t cases = 0, moved = 0;
        for (const double p_cold : {0.25, 0.5, 0.75})
            for (const bool mixed : {false, true})
                for (const bool odd : {false, true}) {
                    const std::uint32_t seed = 11u;
                    const int kvh = 1;
                    std::vector<float> k, v;
                    blasst::makeKv(nk, d, kvh, p_cold, mixed, seed, k, v);
                    const std::vector<float> q = blasst::makeQ(n, d, kvh, odd, seed, 5u + (odd ? 1u : 0u));
                    const blasst::RefOut r = blasst::refAttention(q.data(), k.data(), v.data(), n, d, pos0, scale, -20.0);
                    const std::uint64_t want = blasst::expectedSkips(n, pos0, kvh, odd, p_cold, mixed, seed);
                    if (r.skipped != want) std::printf("  blasst ref p %.2f mixed %d odd %d: %llu skipped, %llu expected\n", p_cold, mixed, odd,
                                                       static_cast<unsigned long long>(r.skipped), static_cast<unsigned long long>(want));
                    CHECK(r.skipped == want && want > 0);
                    // every skipped tile is one of the expected kinds, tile 0 never
                    bool kinds = true;
                    for (int b = 0; b < r.n_blocks; ++b) {
                        kinds = kinds && r.skip[static_cast<std::size_t>(b) * r.max_tiles] == 0;
                        for (int t = 1; t < r.max_tiles; ++t) {
                            if (!r.skip[static_cast<std::size_t>(b) * r.max_tiles + t]) continue;
                            const blasst::TileKind tk = blasst::tileKind(t, kvh, p_cold, mixed, seed);
                            kinds = kinds && tk != blasst::TileKind::both && tk != (odd ? blasst::TileKind::odd : blasst::TileKind::even);
                        }
                    }
                    CHECK(kinds);
                    // the error bound (per query): |o'_i - o_i| <= 2 delta_i max_j |v_j| (+ rounding)
                    const std::vector<double> ex = blasst::exactAttention(q.data(), k.data(), v.data(), n, d, pos0, scale);
                    double vmax = 0.0;
                    for (int j = 0; j < nk; ++j) {
                        double s2 = 0.0;
                        for (int e = 0; e < d; ++e) s2 += static_cast<double>(v[static_cast<std::size_t>(j) * d + e]) * v[static_cast<std::size_t>(j) * d + e];
                        vmax = std::max(vmax, std::sqrt(s2));
                    }
                    bool bound = true;
                    double err_max = 0.0, dmax = 0.0;
                    for (int i = 0; i < n; ++i) {
                        double e2 = 0.0;
                        for (int e = 0; e < d; ++e) {
                            const double dd = r.o[static_cast<std::size_t>(i) * d + e] - ex[static_cast<std::size_t>(i) * d + e];
                            e2 += dd * dd;
                        }
                        const double err = std::sqrt(e2), dl = r.delta[static_cast<std::size_t>(i)];
                        bound = bound && dl >= 0.0 && dl < 1.0 && err <= 2.0 * dl * vmax + 1e-12;
                        // delta below the skipped keys' bound: each skipped key's p < exp((max skipped score - M) scale), Z >= 1;
                        // with the hot / cold gap of >= ~45 q.k units, exp(-45 / 16) * 32 keys per skipped tile
                        bound = bound && dl <= std::exp(-45.0 * scale) * blasst::kK * static_cast<double>(r.max_tiles);
                        err_max = std::max(err_max, err);
                        dmax = std::max(dmax, dl);
                    }
                    CHECK(bound);
                    moved += err_max > 1e-3 ? 1u : 0u;
                    cases += 1;
                }
        CHECK(cases == 12 && moved == cases);  // every case moves the output (a skip decision is visible in o)
        std::printf("  blasst ref: %zu hot / cold cases, skip count == cold tiles, |o' - o| <= 2 delta |v|max\n", cases);
    }
    // (iii) the first tile is never skipped (m = -inf), even when cold; a cold tile after a hot one is
    {
        const int d = 32, n = 16, pos0 = 48, nk = pos0 + n;
        std::vector<double> a, b;
        blasst::headDirs(0, d, 3u, a, b);
        auto build = [&](bool first_cold, std::vector<float>& q, std::vector<float>& k, std::vector<float>& v) {
            q.assign(static_cast<std::size_t>(n) * d, 0.0f);
            k.assign(static_cast<std::size_t>(nk) * d, 0.0f);
            v.assign(static_cast<std::size_t>(nk) * d, 0.0f);
            for (int i = 0; i < n; ++i)
                for (int e = 0; e < d; ++e) q[static_cast<std::size_t>(i) * d + e] = static_cast<float>(a[e]);
            for (int j = 0; j < nk; ++j) {
                const bool cold = (j < 32) == first_cold;
                for (int e = 0; e < d; ++e) {
                    k[static_cast<std::size_t>(j) * d + e] = cold ? static_cast<float>(0.01 * b[e]) : static_cast<float>(60.0 * a[e]);
                    v[static_cast<std::size_t>(j) * d + e] = cold ? -1.0f : 1.0f;
                }
            }
        };
        std::vector<float> q, k, v;
        build(true, q, k, v);
        blasst::RefOut r = blasst::refAttention(q.data(), k.data(), v.data(), n, d, pos0, scale, -20.0);
        CHECK(r.skipped == 0 && r.skip[0] == 0 && r.tiles == 2);
        build(false, q, k, v);
        r = blasst::refAttention(q.data(), k.data(), v.data(), n, d, pos0, scale, -20.0);
        CHECK(r.skipped == 1 && r.skip[0] == 0 && r.skip[1] == 1);
        CHECK(blasst::refAttention(q.data(), k.data(), v.data(), n, d, pos0, scale, ninf).skipped == 0);
    }
    // (iv) causal: pos0 64, n 32 -> block 0 (positions 64..79) walks tiles 0..2 with tile 2's visible keys 64..79 (cold); keys
    // 80..95 are hot but masked for block 0 -> tile 2 is skipped for block 0, kept for block 1 (positions 80..95 see them)
    {
        const int d = 32, n = 32, pos0 = 64, nk = pos0 + n;
        std::vector<double> a, b;
        blasst::headDirs(0, d, 4u, a, b);
        std::vector<float> q(static_cast<std::size_t>(n) * d), k(static_cast<std::size_t>(nk) * d), v(static_cast<std::size_t>(nk) * d);
        for (int i = 0; i < n; ++i)
            for (int e = 0; e < d; ++e) q[static_cast<std::size_t>(i) * d + e] = static_cast<float>(a[e]);
        for (int j = 0; j < nk; ++j) {
            const bool cold = j >= 64 && j < 80;
            for (int e = 0; e < d; ++e) {
                k[static_cast<std::size_t>(j) * d + e] = cold ? static_cast<float>(0.01 * b[e]) : static_cast<float>(60.0 * a[e]);
                v[static_cast<std::size_t>(j) * d + e] = cold ? -1.0f : 1.0f;
            }
        }
        const blasst::RefOut r = blasst::refAttention(q.data(), k.data(), v.data(), n, d, pos0, scale, -20.0);
        CHECK(r.n_blocks == 2 && r.max_tiles == 3);
        CHECK(r.skip[0 * 3 + 2] == 1 && r.skip[1 * 3 + 2] == 0 && r.skipped == 1 && r.tiles == 6);
    }
    // the probe's bins: 1e-3 and 1e-2 are edges (bins 0..8 = delta < 1e-3, 0..10 = delta < 1e-2)
    CHECK(blasst::deltaBin(0.0) == 0 && blasst::deltaBin(9.9e-8) == 0 && blasst::deltaBin(1e-7) == 1);
    CHECK(blasst::deltaBin(9.99e-4) == 8 && blasst::deltaBin(1e-3) == 9 && blasst::deltaBin(9.99e-3) == 10 && blasst::deltaBin(1e-2) == 11);
    CHECK(blasst::deltaBin(0.5) == 14 && blasst::deltaBin(1.0) == 15 && blasst::deltaBin(std::nan("")) == 0);
}

}  // namespace
