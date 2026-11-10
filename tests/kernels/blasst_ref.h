// v0252-blasst (PLAN V1-70, T0): CPU reference of the BLASST attention block skip as the gfx1201 attn_kgk<np>_sk / _pr
// kernels walk it (kernels/attn_kgv.hip), and the structured "hot / cold tile" data the CPU tests (tests/ops, whirl-ops-tests)
// and the kt rows (tests/kernels/test_attn.cpp, family attnkgl) both use. Header only, no HIP.
//
// The walk, per head: query blocks of 16 (q0 = 16 b, the kernel's grid.x), key tiles of 32 from key 0 up to
// n_keys(b) = pos0 + min(q0 + 16, n) (the kernel's loop); per query the running max m of the UNSCALED scores q.k (the
// kernels' units: their exp2 runs on (s - m) * scale * log2 e) and per tile the tile max mx over the keys the query sees
// (causal: key > position is masked, never counted). The tile is skipped for the block iff every query of the block has
// mx < m + lim (m = -inf on the first tile and lim = -inf both make m + lim = -inf: never skipped). A skipped tile leaves
// m unchanged (mx < m + lim < m for lim < 0) and contributes nothing (its p are 0). delta of a query = the probability
// mass of its skipped tiles over the full softmax denominator (what the probe kernels bin).
// Double precision throughout. The GPU kernels take f16 q / K / V with f32 accumulation and f16 P, so outputs compare with a
// tolerance, and skip decisions only on data with a margin around lim (makeHotCold: hot - cold >= ~45 q.k units).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace blasst {

inline constexpr int kQ = 16;  // queries per block (one PV wave)
inline constexpr int kK = 32;  // keys per tile (FA_K)

struct RefOut {
    std::vector<double> o;           // [n][d] softmax over the kept keys (no output gate)
    std::vector<std::uint8_t> skip;  // [n_blocks][max_tiles] 1 = the tile was skipped for the block
    std::vector<double> delta;       // [n] skipped mass / full softmax denominator
    int n_blocks = 0, max_tiles = 0;
    std::uint64_t tiles = 0, skipped = 0;  // (block, tile) pairs walked / skipped
};

// q [n][d] (query i at position pos0 + i), k / v [pos0 + n][d]; scale = the softmax scale (1 / sqrt(d) in WHIRL: 1 / 16)
inline RefOut refAttention(const float* q, const float* k, const float* v, int n, int d, int pos0, double scale, double lim) {
    RefOut r;
    r.n_blocks = (n + kQ - 1) / kQ;
    r.max_tiles = (pos0 + n + kK - 1) / kK;
    r.o.assign(static_cast<std::size_t>(n) * d, 0.0);
    r.skip.assign(static_cast<std::size_t>(r.n_blocks) * r.max_tiles, 0);
    r.delta.assign(static_cast<std::size_t>(n), 0.0);
    const double ninf = -std::numeric_limits<double>::infinity();
    std::vector<double> s;
    for (int b = 0; b < r.n_blocks; ++b) {
        const int q0 = b * kQ, nq = std::min(kQ, n - q0), n_keys = pos0 + std::min(q0 + kQ, n), T = (n_keys + kK - 1) / kK;
        // scores of the block's queries against every key they see (-inf where masked)
        s.assign(static_cast<std::size_t>(nq) * n_keys, ninf);
        for (int i = 0; i < nq; ++i) {
            const int pos = pos0 + q0 + i;
            for (int j = 0; j < n_keys && j <= pos; ++j) {
                double acc = 0.0;
                for (int e = 0; e < d; ++e) acc += static_cast<double>(q[static_cast<std::size_t>(q0 + i) * d + e]) * k[static_cast<std::size_t>(j) * d + e];
                s[static_cast<std::size_t>(i) * n_keys + j] = acc;
            }
        }
        // the kernel's walk: decisions only
        std::vector<double> m(static_cast<std::size_t>(nq), ninf);
        for (int t = 0; t < T; ++t) {
            const int k0 = t * kK, k1 = std::min(k0 + kK, n_keys);
            bool skip = true;
            std::vector<double> mx(static_cast<std::size_t>(nq), ninf);
            for (int i = 0; i < nq; ++i) {
                for (int j = k0; j < k1; ++j) mx[i] = std::max(mx[i], s[static_cast<std::size_t>(i) * n_keys + j]);
                if (!(mx[i] < m[i] + lim)) skip = false;  // m + lim = -inf: never
            }
            r.tiles += 1;
            if (skip) {
                r.skipped += 1;
                r.skip[static_cast<std::size_t>(b) * r.max_tiles + t] = 1;
                continue;  // m unchanged
            }
            for (int i = 0; i < nq; ++i) m[i] = std::max(m[i], mx[i]);
        }
        // outputs and delta from the decisions (exact sums, any common reference max)
        for (int i = 0; i < nq; ++i) {
            const double* si = s.data() + static_cast<std::size_t>(i) * n_keys;
            double M = ninf;
            for (int j = 0; j < n_keys; ++j) M = std::max(M, si[j]);
            double z_all = 0.0, z_skip = 0.0, z_kept = 0.0;
            std::vector<double> acc(static_cast<std::size_t>(d), 0.0);
            for (int j = 0; j < n_keys; ++j) {
                if (si[j] == ninf) continue;
                const double p = std::exp((si[j] - M) * scale);
                z_all += p;
                if (r.skip[static_cast<std::size_t>(b) * r.max_tiles + j / kK]) {
                    z_skip += p;
                    continue;
                }
                z_kept += p;
                for (int e = 0; e < d; ++e) acc[e] += p * v[static_cast<std::size_t>(j) * d + e];
            }
            for (int e = 0; e < d; ++e) r.o[static_cast<std::size_t>(q0 + i) * d + e] = acc[e] / z_kept;
            r.delta[static_cast<std::size_t>(q0 + i)] = z_skip / z_all;
        }
    }
    return r;
}

// the exact causal softmax attention (no skip): the independent check of the walk
inline std::vector<double> exactAttention(const float* q, const float* k, const float* v, int n, int d, int pos0, double scale) {
    std::vector<double> o(static_cast<std::size_t>(n) * d, 0.0), s;
    for (int i = 0; i < n; ++i) {
        const int nk = pos0 + i + 1;
        s.assign(static_cast<std::size_t>(nk), 0.0);
        double M = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < nk; ++j) {
            double acc = 0.0;
            for (int e = 0; e < d; ++e) acc += static_cast<double>(q[static_cast<std::size_t>(i) * d + e]) * k[static_cast<std::size_t>(j) * d + e];
            s[j] = acc;
            M = std::max(M, acc);
        }
        double z = 0.0;
        for (int j = 0; j < nk; ++j) {
            const double p = std::exp((s[j] - M) * scale);
            z += p;
            for (int e = 0; e < d; ++e) o[static_cast<std::size_t>(i) * d + e] += p * v[static_cast<std::size_t>(j) * d + e];
        }
        for (int e = 0; e < d; ++e) o[static_cast<std::size_t>(i) * d + e] /= z;
    }
    return o;
}

// the probe kernels' dropped-mass bin (kernels/attn_kgv.hip SK 2 flush): 0 = delta < 1e-7, b 1..15 = [10^(-7 + (b - 1) / 2),
// 10^(-7 + b / 2)), 15 also >= 1; 1e-3 is the upper edge of bin 8, 1e-2 of bin 10
inline int deltaBin(double delta) {
    if (!(delta >= 1e-7)) return 0;
    return std::min(15, 1 + static_cast<int>(std::floor((std::log10(delta) + 7.0) * 2.0)));
}

// ---- structured data: hot / cold key tiles (PLAN T0 step 2, T3 SK1 rows)
// Two unit directions a, b (a . b = 0) per KV head. A query of an even head of the pair is a + 0.01 noise, of an odd head
// b + 0.01 noise. Key tile t of a KV head is one of: both-hot (k = c (a + b)), hot-for-even (k = c a), hot-for-odd (k = c b)
// or cold (k = sigma noise). Tile 0 is always both-hot (so every query's m is hot from the first tile on). With c = 60 and
// sigma = 1, hot scores are ~60 and cold ones |s| < ~6 for the shapes used: the gap >= ~45 q.k units leaves a margin of
// >= 20 around lim = -20 (the decisions do not depend on f16 / f32 rounding).
enum class TileKind : std::uint8_t { both, even, odd, cold };
// deterministic tile kind: p_cold = share of cold tiles among tiles >= 1; mixed = the non-cold tiles split between even,
// odd and both (the two heads of a pair then skip different tiles: review focus 1 of the plan), else all non-cold are both
inline TileKind tileKind(int t, int kvh, double p_cold, bool mixed, std::uint32_t seed) {
    if (t == 0) return TileKind::both;
    std::uint32_t h = static_cast<std::uint32_t>(t) * 0x9E3779B1u ^ (static_cast<std::uint32_t>(kvh) + 1u) * 0x85EBCA77u ^ seed * 0xC2B2AE3Du;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    const double u = static_cast<double>(h % 100000u) / 100000.0;
    if (u < p_cold) return TileKind::cold;
    if (!mixed) return TileKind::both;
    const std::uint32_t w = (h >> 20) % 3u;
    return w == 0 ? TileKind::both : w == 1 ? TileKind::even : TileKind::odd;
}
// unit direction pair of KV head kvh (a, b orthogonal), d dims
inline void headDirs(int kvh, int d, std::uint32_t seed, std::vector<double>& a, std::vector<double>& b) {
    std::mt19937 g(seed * 7919u + static_cast<std::uint32_t>(kvh) * 104729u + 1u);
    std::normal_distribution<double> nd(0.0, 1.0);
    a.assign(static_cast<std::size_t>(d), 0.0);
    b.assign(static_cast<std::size_t>(d), 0.0);
    for (auto& x : a) x = nd(g);
    for (auto& x : b) x = nd(g);
    double na = 0.0;
    for (const double x : a) na += x * x;
    na = std::sqrt(na);
    for (auto& x : a) x /= na;
    double ab = 0.0;
    for (int e = 0; e < d; ++e) ab += a[e] * b[e];
    for (int e = 0; e < d; ++e) b[e] -= ab * a[e];
    double nb = 0.0;
    for (const double x : b) nb += x * x;
    nb = std::sqrt(nb);
    for (auto& x : b) x /= nb;
}
// one KV head's keys (n_keys x d) and values. The value rows' mean depends on the tile kind (both +1, even +0.3, odd -0.3,
// cold -1; + 0.25 noise), so dropping any tile moves the output by about its mass times the mean gap (a wrong skip
// decision shows in the output, well above the f16 tolerance of the kt rows)
inline void makeKv(int n_keys, int d, int kvh, double p_cold, bool mixed, std::uint32_t seed, std::vector<float>& k, std::vector<float>& v) {
    std::vector<double> a, b;
    headDirs(kvh, d, seed, a, b);
    std::mt19937 g(seed * 31u + static_cast<std::uint32_t>(kvh) * 977u + 7u);
    std::normal_distribution<double> nd(0.0, 1.0);
    constexpr double c = 60.0, sigma = 1.0;
    k.assign(static_cast<std::size_t>(n_keys) * d, 0.0f);
    v.assign(static_cast<std::size_t>(n_keys) * d, 0.0f);
    for (int j = 0; j < n_keys; ++j) {
        const TileKind tk = tileKind(j / kK, kvh, p_cold, mixed, seed);
        const double vmean = tk == TileKind::both ? 1.0 : tk == TileKind::even ? 0.3 : tk == TileKind::odd ? -0.3 : -1.0;
        for (int e = 0; e < d; ++e) {
            double x = 0.0;
            switch (tk) {
                case TileKind::both: x = c * (a[e] + b[e]); break;
                case TileKind::even: x = c * a[e]; break;
                case TileKind::odd: x = c * b[e]; break;
                case TileKind::cold: x = sigma * nd(g) / std::sqrt(static_cast<double>(d)) * 4.0; break;
            }
            k[static_cast<std::size_t>(j) * d + e] = static_cast<float>(x);
            v[static_cast<std::size_t>(j) * d + e] = static_cast<float>(vmean + 0.25 * nd(g));
        }
    }
}
// queries of head parity `odd` of KV head kvh: a (even) or b (odd) + 0.01 noise; n x d
inline std::vector<float> makeQ(int n, int d, int kvh, bool odd, std::uint32_t seed, std::uint32_t qseed) {
    std::vector<double> a, b;
    headDirs(kvh, d, seed, a, b);
    std::mt19937 g(qseed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<float> q(static_cast<std::size_t>(n) * d);
    for (int i = 0; i < n; ++i)
        for (int e = 0; e < d; ++e) q[static_cast<std::size_t>(i) * d + e] = static_cast<float>((odd ? b[e] : a[e]) + 0.01 * nd(g) / std::sqrt(static_cast<double>(d)) * 4.0);
    return q;
}
// the number of tiles a block of queries of head parity `odd` must skip at lim (with the margin above): the cold-for-this-
// head tiles after tile 0 among the block's tiles (both-hot tile 0 sets m; a tile that is cold for this head has mx far
// below it; the diagonal tile counts too: its visible keys are cold)
inline std::uint64_t expectedSkips(int n, int pos0, int kvh, bool odd, double p_cold, bool mixed, std::uint32_t seed) {
    std::uint64_t e = 0;
    for (int b = 0; b * kQ < n; ++b) {
        const int n_keys = pos0 + std::min(b * kQ + kQ, n), T = (n_keys + kK - 1) / kK;
        for (int t = 1; t < T; ++t) {
            const TileKind tk = tileKind(t, kvh, p_cold, mixed, seed);
            const bool hot = tk == TileKind::both || (odd ? tk == TileKind::odd : tk == TileKind::even);
            // a tile is only skipped when every query of the block sees none of its hot keys; with the margin all keys of a
            // hot tile are hot, and a tile some query of the block sees at all is decided by its visible keys
            e += hot ? 0u : 1u;
        }
    }
    return e;
}

}  // namespace blasst
