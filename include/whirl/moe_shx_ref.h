// moeshx (fast item, feat028-moefast97): host numerics reference of the shared-expert-as-slot-K MoE decode
// (kernels/moe_shx.hip moe_gus_mxfp4w / moe_downs_mxfp4w) and a double-precision oracle, for CPU unit tests.
// SPDX-License-Identifier: Apache-2.0
//
// Layout: MXFP4 rows in the WHIRL load-time layout (per 256 values: 8 E8M0 exponents, then 8 x 16 bytes of e2m1
// nibbles; value i of a 32-block is the low nibble of byte i for i < 16, the high nibble of byte i - 16 otherwise;
// value = 2^(e - 128) x {0, 1, 2, 3, 4, 6, 8, 12, -0, -1, ...}[nibble], kernels/common.hip k_mxfp4 / e8m0_half).
// Activations: int8 per 32-value block with an f32 scale d = amax / 127 (q8_1 class, kernels/fused_decode.hip
// wave_q8).
//
// q8Forward() follows the kernels' f32 expression and summation order (lane l owns 32-blocks l, l + 32, ...;
// xor-butterfly warp_sum; acc += wt * (scale * d * dot)); its SwiGLU uses std::exp instead of the GPU's __expf,
// so it is a numerics reference, not a bitwise model of the GPU. sharedSeparate = true runs the balance form
// (shared expert summed on its own and added as sg * ysh after the routed sum), false the moeshx form (slot K).
// oracle() is the exact math in double from the f32 h (no int8 rounding anywhere).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace whirl::moeshx_ref {

struct Shapes {
    int E = 0;  // n_embd (gate / up columns, down rows)
    int F = 0;  // expert FFN width (gate / up rows, down columns)
    int K = 0;  // routed experts per token
    int R = 0;  // routed experts
};

inline std::uint64_t mxRowBytes(int ncols) { return static_cast<std::uint64_t>(ncols) / 256 * 136; }

inline float e8m0Half(std::uint32_t e) {
    const std::uint32_t u = e < 2 ? (0x00200000u << e) : ((e - 1) << 23);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline int e2m1x2(int nib) {
    static constexpr int t[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    return t[nib & 15];
}
// value i (0..ncols) of an MXFP4 row as e2m1 x 2 (integer) and the block's E8M0 exponent
inline int mxCode(const std::uint8_t* row, int i) {
    const int b = i / 32, s = i % 32;
    const std::uint8_t* sb = row + static_cast<std::size_t>(b >> 3) * 136;
    const std::uint8_t q = sb[8 + 16 * (b & 7) + (s & 15)];
    return e2m1x2(s < 16 ? (q & 15) : (q >> 4));
}
inline std::uint32_t mxExp(const std::uint8_t* row, int b) { return row[static_cast<std::size_t>(b >> 3) * 136 + (b & 7)]; }
inline double mxValue(const std::uint8_t* row, int i) { return static_cast<double>(e8m0Half(mxExp(row, i / 32))) * mxCode(row, i); }

// q8 of n values (n % 32 == 0) as wave_q8: d = amax / 127, q = round-half-even(v * (1 / d))
inline void q8Quant(const float* v, int n, std::int8_t* q, float* d) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[b * 32 + i]));
        const float db = amax / 127.f;
        const float id = db > 0.f ? 1.f / db : 0.f;
        for (int i = 0; i < 32; ++i) q[b * 32 + i] = static_cast<std::int8_t>(std::nearbyint(v[b * 32 + i] * id));
        d[b] = db;
    }
}
inline int mxDot(const std::uint8_t* row, int b, const std::int8_t* x) {
    int a = 0;
    for (int i = 0; i < 32; ++i) a += mxCode(row, b * 32 + i) * x[b * 32 + i];
    return a;
}
// xor-butterfly warp_sum of 32 lane values; lane 0's result
inline float warpSum(float* v) {
    for (int o = 16; o > 0; o >>= 1) {
        float n[32];
        for (int i = 0; i < 32; ++i) n[i] = v[i] + v[i ^ o];
        std::memcpy(v, n, sizeof(n));
    }
    return v[0];
}
// one gate / up row from int8 x (moe_gu_mxfp4w / moe_gus_mxfp4w per-row code)
inline float rowQ8(const std::uint8_t* row, int ncols, const std::int8_t* xq, const float* xd) {
    float lane[32] = {};
    for (int l = 0; l < 32; ++l)
        for (int b = l; b < ncols / 32; b += 32) lane[l] += e8m0Half(mxExp(row, b)) * xd[b] * static_cast<float>(mxDot(row, b, xq));
    return warpSum(lane);
}

struct Problem {
    Shapes s;
    int n = 0;                               // tokens
    std::vector<std::uint8_t> gate, up, down;  // routed: [R * F] x E, [R * F] x E, [R * E] x F (rows of mxRowBytes)
    std::vector<std::uint8_t> sg, su, sd;      // shared: F x E, F x E, E x F
    std::vector<float> h;                    // [n][E] normed input
    std::vector<int> ids;                    // [n][K] (id < 0: dropped slot, moeskip)
    std::vector<float> w;                    // [n][K] routing weights
    std::vector<float> shw;                  // [n] shared weight sigmoid(shw . h)
    std::vector<float> x0;                   // [n][E] residual
};

// the int8 decode / verify experts of balance (sharedSeparate) or moeshx (slot K); returns x0 + MoE(h)
inline std::vector<float> q8Forward(const Problem& p, bool sharedSeparate) {
    const int E = p.s.E, F = p.s.F, K = p.s.K, n = p.n, S = K + 1, nb = F / 32;
    const std::uint64_t rbE = mxRowBytes(E), rbF = mxRowBytes(F);
    std::vector<std::int8_t> hq(static_cast<std::size_t>(n) * E);
    std::vector<float> hd(static_cast<std::size_t>(n) * E / 32);
    q8Quant(p.h.data(), n * E, hq.data(), hd.data());
    // gate / up per (token, slot); slot K = the shared expert
    std::vector<float> act(static_cast<std::size_t>(n) * S * F, 0.f);
    for (int t = 0; t < n; ++t)
        for (int s = 0; s < S; ++s) {
            const int e = s < K ? p.ids[static_cast<std::size_t>(t) * K + s] : 0;
            if (s < K && e < 0) continue;
            for (int r = 0; r < F; ++r) {
                const std::uint8_t* gr = s < K ? p.gate.data() + (static_cast<std::size_t>(e) * F + r) * rbE : p.sg.data() + static_cast<std::size_t>(r) * rbE;
                const std::uint8_t* ur = s < K ? p.up.data() + (static_cast<std::size_t>(e) * F + r) * rbE : p.su.data() + static_cast<std::size_t>(r) * rbE;
                const float g = rowQ8(gr, E, hq.data() + static_cast<std::size_t>(t) * E, hd.data() + static_cast<std::size_t>(t) * E / 32);
                const float u = rowQ8(ur, E, hq.data() + static_cast<std::size_t>(t) * E, hd.data() + static_cast<std::size_t>(t) * E / 32);
                act[(static_cast<std::size_t>(t) * S + s) * F + r] = g / (1.f + std::exp(-g)) * u;
            }
        }
    std::vector<std::int8_t> aq(act.size());
    std::vector<float> ad(act.size() / 32);
    q8Quant(act.data(), static_cast<int>(act.size()), aq.data(), ad.data());
    std::vector<float> x = p.x0;
    for (int t = 0; t < n; ++t)
        for (int row = 0; row < E; ++row) {
            float lane[32] = {};
            const int slots = sharedSeparate ? K : S;
            for (int l = 0; l < 32; ++l)
                for (int u = l; u < slots * nb; u += 32) {
                    const int k = u / nb, b = u - k * nb;
                    const std::size_t pp = static_cast<std::size_t>(t) * S + k;
                    const std::uint8_t* wr;
                    float wt;
                    if (k < K) {
                        const int e = p.ids[static_cast<std::size_t>(t) * K + k];
                        if (e < 0) continue;
                        wr = p.down.data() + (static_cast<std::size_t>(e) * E + row) * rbF;
                        wt = p.w[static_cast<std::size_t>(t) * K + k];
                    } else {
                        wr = p.sd.data() + static_cast<std::size_t>(row) * rbF;
                        wt = p.shw[static_cast<std::size_t>(t)];
                    }
                    lane[l] += wt * (e8m0Half(mxExp(wr, b)) * ad[pp * nb + b] * static_cast<float>(mxDot(wr, b, aq.data() + pp * F)));
                }
            float v = warpSum(lane);
            if (sharedSeparate) {
                // the balance chain: shared down as its own GEMV (sequential f32 here), added as sg * ysh
                float ysh = 0.f;
                const std::size_t pp = static_cast<std::size_t>(t) * S + K;
                for (int b = 0; b < nb; ++b)
                    ysh += e8m0Half(mxExp(p.sd.data() + static_cast<std::size_t>(row) * rbF, b)) * ad[pp * nb + b] *
                           static_cast<float>(mxDot(p.sd.data() + static_cast<std::size_t>(row) * rbF, b, aq.data() + pp * F));
                v += p.shw[static_cast<std::size_t>(t)] * ysh;
            }
            x[static_cast<std::size_t>(t) * E + row] += v;
        }
    return x;
}

// exact math in double (f32 h, exact MXFP4 weights, no int8): x0 + sum_k w_k down_k(silu(g_k) u_k) + shw down_s(...).
// scale (optional) receives, per output, x0 + the sum of |terms| (the magnitude the error is measured against).
inline std::vector<double> oracle(const Problem& p, std::vector<double>* scale = nullptr) {
    const int E = p.s.E, F = p.s.F, K = p.s.K, n = p.n, S = K + 1;
    const std::uint64_t rbE = mxRowBytes(E), rbF = mxRowBytes(F);
    std::vector<double> y(static_cast<std::size_t>(n) * E), sc(y.size());
    std::vector<double> a(static_cast<std::size_t>(F));
    for (int t = 0; t < n; ++t) {
        for (std::size_t i = 0; i < static_cast<std::size_t>(E); ++i) {
            y[static_cast<std::size_t>(t) * E + i] = p.x0[static_cast<std::size_t>(t) * E + i];
            sc[static_cast<std::size_t>(t) * E + i] = std::fabs(p.x0[static_cast<std::size_t>(t) * E + i]);
        }
        for (int s = 0; s < S; ++s) {
            const int e = s < K ? p.ids[static_cast<std::size_t>(t) * K + s] : 0;
            if (s < K && e < 0) continue;
            const double wt = s < K ? p.w[static_cast<std::size_t>(t) * K + s] : p.shw[static_cast<std::size_t>(t)];
            for (int r = 0; r < F; ++r) {
                const std::uint8_t* gr = s < K ? p.gate.data() + (static_cast<std::size_t>(e) * F + r) * rbE : p.sg.data() + static_cast<std::size_t>(r) * rbE;
                const std::uint8_t* ur = s < K ? p.up.data() + (static_cast<std::size_t>(e) * F + r) * rbE : p.su.data() + static_cast<std::size_t>(r) * rbE;
                double g = 0, u = 0;
                for (int c = 0; c < E; ++c) {
                    const double hv = p.h[static_cast<std::size_t>(t) * E + c];
                    g += mxValue(gr, c) * hv;
                    u += mxValue(ur, c) * hv;
                }
                a[static_cast<std::size_t>(r)] = g / (1.0 + std::exp(-g)) * u;
            }
            for (int row = 0; row < E; ++row) {
                const std::uint8_t* dr = s < K ? p.down.data() + (static_cast<std::size_t>(e) * E + row) * rbF : p.sd.data() + static_cast<std::size_t>(row) * rbF;
                double acc = 0, aa = 0;
                for (int c = 0; c < F; ++c) {
                    const double v = mxValue(dr, c) * a[static_cast<std::size_t>(c)];
                    acc += v;
                    aa += std::fabs(v);
                }
                y[static_cast<std::size_t>(t) * E + row] += wt * acc;
                sc[static_cast<std::size_t>(t) * E + row] += std::fabs(wt) * aa;
            }
        }
    }
    if (scale != nullptr) *scale = sc;
    return y;
}

}  // namespace whirl::moeshx_ref
