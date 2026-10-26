// feat027-i8qk CPU error bounds (no GPU): int8 Q.K^T attention (tests/kernels/i8qk_ref.h) vs exact f64 attention,
// next to the f16 path's own error (f16 Q, f16 P) on synthetic Qwen3.8-like K: a per-channel bias (the shared
// component SageAttention's smoothing removes) and a few outlier channels. Prints the errors and fails (exit 1)
// when the int8 path breaks its bound or smoothing does not help.
// Build (standalone, CPU only): cl /std:c++20 /O2 /EHsc /I tests\kernels tests\kernels\unit\i8qk_ref_tests.cpp
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "i8qk_ref.h"

namespace {

// IEEE binary16 round to nearest even (normal and subnormal range; no overflow handling needed here)
float roundh(float f) {
    if (f == 0.f || !std::isfinite(f)) return f;
    int e = 0;
    std::frexp(f, &e);                        // |f| in [2^(e-1), 2^e)
    const int ulp_e = std::max(e - 11, -24);  // 10 mantissa bits; subnormal ulp 2^-24
    const float ulp = std::ldexp(1.f, ulp_e);
    return std::nearbyint(f / ulp) * ulp;
}

struct Data {
    i8qk::Shape s;
    int rows = 0;
    std::vector<float> q, k, v;
};

// orthonormal Walsh-Hadamard transform of one head (hd a power of 2), as attn_prep_q4 / q8h rotate q and k
void wht(float* x, int hd) {
    for (int len = 1; len < hd; len <<= 1)
        for (int i = 0; i < hd; i += 2 * len)
            for (int j = i; j < i + len; ++j) {
                const float a = x[j], b = x[j + len];
                x[j] = a + b;
                x[j + len] = a - b;
            }
    const float sc = 1.f / std::sqrt(static_cast<float>(hd));
    for (int d = 0; d < hd; ++d) x[d] *= sc;
}

// bias_amp: per-channel K bias amplitude; outl: outlier channel factor; rot: Hadamard-rotate q and k heads (the fast
// mode's q4 KV path does; it spreads the outlier channels over all 256 dims)
Data make(std::mt19937& rng, int n, int pos0, float bias_amp, float outl, bool rot = false, int n_head = 24, int n_kv = 4) {
    Data D;
    D.s.n = n;
    D.s.pos0 = pos0;
    D.s.n_head = n_head;
    D.s.n_kv = n_kv;
    D.rows = pos0 + n;
    const int hd = D.s.hd, row_el = n_kv * hd;
    std::normal_distribution<float> N(0.f, 1.f);
    std::vector<float> w(static_cast<std::size_t>(row_el), 1.f), bias(static_cast<std::size_t>(row_el));
    for (int c = 0; c < row_el; ++c) {
        if (c % hd == 3 || c % hd == 77 || c % hd == 130 || c % hd == 200) w[static_cast<std::size_t>(c)] = outl;
        bias[static_cast<std::size_t>(c)] = bias_amp * N(rng);
    }
    D.k.resize(static_cast<std::size_t>(D.rows) * row_el);
    D.v.resize(D.k.size());
    for (int r = 0; r < D.rows; ++r)
        for (int c = 0; c < row_el; ++c) {
            const std::size_t i = static_cast<std::size_t>(r) * row_el + c;
            D.k[i] = bias[static_cast<std::size_t>(c)] + w[static_cast<std::size_t>(c)] * N(rng);
            D.v[i] = roundh(N(rng));
        }
    for (int r = 0; r < D.rows; ++r)
        for (int g = 0; g < n_kv; ++g) {
            float* kr = &D.k[(static_cast<std::size_t>(r) * n_kv + g) * hd];
            if (rot) wht(kr, hd);
            for (int d = 0; d < hd; ++d) kr[d] = roundh(kr[d]);
        }
    D.q.assign(static_cast<std::size_t>(n) * n_head * D.s.q_stride, 0.f);
    for (int i = 0; i < n; ++i)
        for (int h = 0; h < n_head; ++h)
            for (int d = 0; d < hd; ++d) D.q[(static_cast<std::size_t>(i) * n_head + h) * D.s.q_stride + d] = N(rng) * (d % 64 == 5 ? outl : 1.f);
    if (rot)
        for (int i = 0; i < n; ++i)
            for (int h = 0; h < n_head; ++h) wht(&D.q[(static_cast<std::size_t>(i) * n_head + h) * D.s.q_stride], hd);
    return D;
}

// the fast mode's q4 K (kernels/attn.hip: per 32 values, scale = the signed value of largest magnitude / -8,
// q = clamp(round(x / scale), -8, 7), value q * scale in f16) on f16 K rows
std::vector<float> q4K(const std::vector<float>& k) {
    std::vector<float> o(k.size());
    for (std::size_t b = 0; b < k.size(); b += 32) {
        float big = 0.f;
        for (std::size_t i = b; i < b + 32; ++i)
            if (std::fabs(k[i]) > std::fabs(big)) big = k[i];
        const float sc = roundh(big / -8.f);
        for (std::size_t i = b; i < b + 32; ++i) {
            const float q = sc != 0.f ? std::min(std::max(std::nearbyint(k[i] / sc), -8.f), 7.f) : 0.f;
            o[i] = roundh(q * sc);
        }
    }
    return o;
}

struct Err {
    double rel_l2 = 0, max_abs = 0, cos_min = 1;
};

// out of every (query, head) vs the exact one: aggregated rel L2, max |diff|, worst cosine
Err compare(const std::vector<double>& got, const std::vector<double>& ref, int rows_qh, int hd) {
    Err e;
    double num = 0, den = 0;
    for (int r = 0; r < rows_qh; ++r) {
        double dd = 0, gg = 0, rr = 0;
        for (int d = 0; d < hd; ++d) {
            const double a = got[static_cast<std::size_t>(r) * hd + d], b = ref[static_cast<std::size_t>(r) * hd + d];
            num += (a - b) * (a - b);
            den += b * b;
            e.max_abs = std::max(e.max_abs, std::fabs(a - b));
            dd += a * b;
            gg += a * a;
            rr += b * b;
        }
        if (gg > 0 && rr > 0) e.cos_min = std::min(e.cos_min, dd / std::sqrt(gg * rr));
    }
    e.rel_l2 = std::sqrt(num / den);
    return e;
}

// out for every query / head: mode 0 exact, 1 f16 path (f16 Q, f16 P), 2 int8 with mu, 3 int8 with mu = 0,
// 4 int8 over two key ranges, each with its own mu (the FC-1c series)
std::vector<double> run(const Data& D, int mode) {
    const auto& s = D.s;
    const int row_el = s.n_kv * s.hd;
    std::vector<double> out(static_cast<std::size_t>(s.n) * s.n_head * s.hd);
    std::vector<float> mu, mu2;
    std::vector<std::int8_t> kq, kq2;
    std::vector<float> ksc, ksc2;
    int split = D.rows;
    if (mode >= 2) {
        if (mode == 4) split = (D.rows / 2) / 256 * 256;
        std::vector<float> k1(D.k.begin(), D.k.begin() + static_cast<std::ptrdiff_t>(split) * row_el);
        mu = mode == 3 ? std::vector<float>(static_cast<std::size_t>(row_el), 0.f) : i8qk::meanSampled(k1, split, row_el);
        i8qk::quantK(k1, mu, split, s.n_kv, s.hd, kq, ksc);
        if (mode == 4) {
            std::vector<float> k2(D.k.begin() + static_cast<std::ptrdiff_t>(split) * row_el, D.k.end());
            mu2 = i8qk::meanSampled(k2, D.rows - split, row_el);
            i8qk::quantK(k2, mu2, D.rows - split, s.n_kv, s.hd, kq2, ksc2);
        }
    }
    std::vector<double> sc, sc2;
    for (int i = 0; i < s.n; ++i)
        for (int h = 0; h < s.n_head; ++h) {
            if (mode <= 1) {
                i8qk::scoresRef(s, D.q, D.k, i, h, mode == 1, roundh, sc);
            } else {
                i8qk::Shape s1 = s;
                if (mode == 4) {
                    // range 1: keys [0, split) (all visible to every query here: pos0 >= split)
                    s1.pos0 = split - 1 - i;
                }
                i8qk::scoresI8(s1, D.q, kq, ksc, mu, split, i, h, sc);
                if (mode == 4) {
                    i8qk::Shape s2 = s;
                    s2.pos0 = s.pos0 - split;
                    i8qk::scoresI8(s2, D.q, kq2, ksc2, mu2, D.rows - split, i, h, sc2);
                    sc.insert(sc.end(), sc2.begin(), sc2.end());
                }
            }
            i8qk::softmaxPv(s, sc, D.v, h, mode != 0, roundh, &out[(static_cast<std::size_t>(i) * s.n_head + h) * s.hd]);
        }
    return out;
}

}  // namespace

int main() {
    std::mt19937 rng(20261026);
    int fails = 0;
    struct Case {
        const char* name;
        int n, pos0;
        float bias, outl;
        bool rot;
    };
    // pos0 >= the two-range split so every query sees the whole first range (mode 4)
    // rot = true is the fast-mode case (q4 KV: q and k Hadamard-rotated); rot = false with outlier channels is the
    // unrotated worst case (no bound: printed only, the selector never runs int8 Q.K^T on unrotated f16 / q8 KV)
    const Case cases[] = {{"no bias, no outliers", 32, 2016, 0.f, 1.f, false},
                          {"bias 2", 32, 2016, 2.f, 1.f, false},
                          {"bias 2 + outliers x8 (unrotated)", 32, 2016, 2.f, 8.f, false},
                          {"bias 4 + outliers x16 (unrotated)", 32, 4064, 4.f, 16.f, false},
                          {"bias 2 + outliers x8, rotated", 32, 2016, 2.f, 8.f, true},
                          {"bias 4 + outliers x16, rotated", 32, 4064, 4.f, 16.f, true}};
    std::printf("feat027-i8qk CPU reference: attention output vs exact f64 (rel L2 / max |diff| / worst cosine per query-head)\n");
    for (const Case& c : cases) {
        const Data D = make(rng, c.n, c.pos0, c.bias, c.outl, c.rot);
        const int qh = c.n * D.s.n_head;
        const auto ref = run(D, 0);
        const Err f16 = compare(run(D, 1), ref, qh, D.s.hd);
        const Err i8 = compare(run(D, 2), ref, qh, D.s.hd);
        const Err i8n = compare(run(D, 3), ref, qh, D.s.hd);
        const Err i8r = compare(run(D, 4), ref, qh, D.s.hd);
        // the fast path as it runs: K stored as q4, then (a) the f16 kernel on the dequantized copy (today's fast) and
        // (b) int8 Q.K^T on that copy (this item), both vs the exact attention on the unquantized K
        Data D4 = D;
        D4.k = q4K(D.k);
        const Err q4f = compare(run(D4, 1), ref, qh, D.s.hd);
        const Err q4i = compare(run(D4, 2), ref, qh, D.s.hd);
        std::printf("  %-32s keys %5d  f16 path %.2e / %.2e / %.6f   int8+smooth %.2e / %.2e / %.6f   int8 no smooth %.2e / %.2e / %.6f   int8 2 ranges %.2e / %.2e / %.6f\n",
                    c.name, D.rows, f16.rel_l2, f16.max_abs, f16.cos_min, i8.rel_l2, i8.max_abs, i8.cos_min, i8n.rel_l2, i8n.max_abs, i8n.cos_min, i8r.rel_l2,
                    i8r.max_abs, i8r.cos_min);
        std::printf("  %-32s             q4 K + f16 kernel (fast today) %.2e / %.2e / %.6f   q4 K + int8 Q.K^T (this item) %.2e / %.2e / %.6f   ratio %.2f\n", "",
                    q4f.rel_l2, q4f.max_abs, q4f.cos_min, q4i.rel_l2, q4i.max_abs, q4i.cos_min, q4i.rel_l2 / q4f.rel_l2);
        // bounds (feat027-i8qk acceptance for the kernel numerics; end-to-end acceptance is task accuracy):
        // on f16 K: int8 + smoothing within 5e-2 rel L2, worst cosine >= 0.99, the two-range series within 1.5x of one
        // range, smoothing better than none when K has a bias; on the fast path's q4 K: int8 Q.K^T adds <= 10% to the
        // q4 path's own attention error (measured 0-2%). Unrotated outlier cases: printed only (not a fast-mode input).
        const bool bounded = c.rot || c.outl == 1.f;
        const bool ok = !bounded || (i8.rel_l2 <= 5e-2 && i8.cos_min >= 0.99 && i8r.rel_l2 <= 1.5 * i8.rel_l2 + 1e-3 &&
                                     (c.bias == 0.f || i8.rel_l2 < i8n.rel_l2) && q4i.rel_l2 <= 1.10 * q4f.rel_l2);
        if (!ok) {
            std::printf("    FAIL: bound broken\n");
            ++fails;
        }
    }
    std::printf(fails == 0 ? "i8qk_ref_tests: PASS\n" : "i8qk_ref_tests: %d FAIL\n", fails);
    return fails == 0 ? 0 : 1;
}
