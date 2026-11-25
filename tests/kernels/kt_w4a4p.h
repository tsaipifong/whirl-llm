// whirl-kernel-test --bench w4a4p (v0252-w4a4, PLAN V1-144 kill-screen K1 on gfx1151): the HIP-free half of the W4A4 speed
// probe - int4 packing, the per-32 int4 activation quantizer, the MXFP4 -> int4 weight re-quantization (LUT form), the CPU
// references of the probe kernels, the Swift dense shape table and the end-to-end projection. Unit-tested on the CPU
// (tests\kernels\unit\kt_bench_tests.cpp, target whirl-kt-bench-tests). Header-only.
// SPDX-License-Identifier: Apache-2.0
//
// PROBE ONLY (fast-mode candidate, no product path): the kernels are bench-only entries of kernels\gfx1151\gemm_w4a4p.hip.
// Numbers (int4 = two's complement nibble, range -8..7, the probe uses -7..7):
//   activations, per 32 along K: amax = max |x|, d = amax / 7, inv = amax > 0 ? 7 / amax : 0, q = clamp(rint(x * inv), -7, 7)
//   weights from MXFP4 (LUT form): code n -> kv[n] in {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12}, q = rint(kv * 7 / 12) (ties to
//   even) -> {0, 1, 1, 2, 2, 4, 5, 7}, scale d_w = 2^(e - 128) * 12 / 7 (e = the E8M0 byte; MXFP4 value = kv * 2^(e - 128))
//   packing: byte j of a row = q[2j] & 15 | (q[2j + 1] & 15) << 4 (K order inside each byte: low nibble first)
//   i4 kernel (no fold, the int4 ceiling): Y = float(sum_k qw qx) - exact (|sum| < 2^24 for K <= 17408 and |q| <= 7)
//   i4f kernel (per-32 fold): for blocks b ascending: c_b = sum_32 qw qx (int32, exact), s = d_w[b] * d_x[b] (f32 mul),
//     acc = fma(float(c_b), s, acc) (f32) - the CPU emulation below does the same operations in the same order
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace kt::w4a4p {

inline constexpr int kBlk = 32;  // K values per scale block
// kernel tile of gemm_w4a4p.hip (the gemmfq_t2 geometry: 128 weight rows x 256 tokens, 256 threads, 4 x 4 WMMA tiles per
// wave) and its K stage (128 K values = 4 blocks); shapes must divide (no edge code in the probe)
inline constexpr int kBM = 128, kBN = 256, kBK = 128;

inline std::uint8_t packPair(int lo, int hi) { return static_cast<std::uint8_t>((lo & 15) | ((hi & 15) << 4)); }
inline int nib(std::uint8_t b, int hi) {
    const int v = hi ? (b >> 4) : (b & 15);
    return v >= 8 ? v - 16 : v;
}
// q: K int4 values (-8..7) -> K / 2 bytes
inline void packRow(const int* q, int k, std::uint8_t* out) {
    for (int j = 0; j < k / 2; ++j) out[j] = packPair(q[2 * j], q[2 * j + 1]);
}
inline int unpackAt(const std::uint8_t* row, int k) { return nib(row[k / 2], k & 1); }

// per-32 int4 activation quantizer of one row (x: K floats = f16 values widened; q: K / 2 bytes; s: K / 32 scales)
inline void quantRow(const float* x, int k, std::uint8_t* q, float* s) {
    for (int b = 0; b < k / kBlk; ++b) {
        float amax = 0.f;
        for (int i = 0; i < kBlk; ++i) amax = std::fmax(amax, std::fabs(x[b * kBlk + i]));
        const float d = amax / 7.f, inv = amax > 0.f ? 7.f / amax : 0.f;
        int c[kBlk];
        for (int i = 0; i < kBlk; ++i) {
            const float r = std::rint(x[b * kBlk + i] * inv);
            c[i] = static_cast<int>(std::fmin(std::fmax(r, -7.f), 7.f));
        }
        packRow(c, kBlk, q + b * (kBlk / 2));
        s[b] = d;
    }
}

// MXFP4 (WHIRL / ggml layout: per 256 values a 136-byte unit = 8 E8M0 bytes + 8 x 16 qs bytes; sub-block ib: qs byte j low
// nibble = value j, high nibble = value j + 16) -> int4 row (K / 2 bytes) + f32 scales (K / 32), LUT form
inline constexpr std::array<int, 16> kMxKv = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
inline int mxToI4(int code) {
    const double v = kMxKv[static_cast<std::size_t>(code & 15)] * 7.0 / 12.0;
    return static_cast<int>(std::nearbyint(v));  // default rounding mode: ties to even (3.5 -> 4)
}
inline float mxI4Scale(int e8m0) { return static_cast<float>(std::ldexp(12.0 / 7.0, e8m0 - 128)); }
inline void mxRowToI4(const std::uint8_t* mx, int k, std::uint8_t* q, float* s) {
    for (int u = 0; u < k / 256; ++u) {
        const std::uint8_t* blk = mx + static_cast<std::size_t>(u) * 136;
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* qs = blk + 8 + 16 * ib;
            int c[kBlk];
            for (int j = 0; j < 16; ++j) {
                c[j] = mxToI4(qs[j] & 15);
                c[j + 16] = mxToI4(qs[j] >> 4);
            }
            const int b = u * 8 + ib;
            packRow(c, kBlk, q + b * (kBlk / 2));
            s[b] = mxI4Scale(blk[ib]);
        }
    }
}

// CPU references for one output (row of W, token of X; rows given as packed bytes + scales)
inline std::int64_t refI4(const std::uint8_t* wq, const std::uint8_t* xq, int k) {
    std::int64_t s = 0;
    for (int i = 0; i < k; ++i) s += static_cast<std::int64_t>(unpackAt(wq, i)) * unpackAt(xq, i);
    return s;
}
// g = blocks per fold: 1 (per-32, what MXFP4 weights need) or 4 (per-128 with the scales of the group's first block: the
// i4g entry, the ceiling of an int4-native weight format)
inline std::int32_t blockDot(const std::uint8_t* wq, const std::uint8_t* xq, int b, int g = 1) {
    std::int32_t c = 0;
    for (int i = b * kBlk; i < (b + g) * kBlk; ++i) c += unpackAt(wq, i) * unpackAt(xq, i);
    return c;
}
// the i4f kernel's f32 operations in its order (bitwise target)
inline float refFoldF32(const std::uint8_t* wq, const float* ws, const std::uint8_t* xq, const float* xs, int k, int g = 1) {
    float acc = 0.f;
    for (int b = 0; b < k / kBlk; b += g) {
        const float s = ws[b] * xs[b];
        acc = std::fma(static_cast<float>(blockDot(wq, xq, b, g)), s, acc);
    }
    return acc;
}
// exact fold (double) and the scale of its terms (sum |c_b s_b|) for a tolerance check
inline void refFoldF64(const std::uint8_t* wq, const float* ws, const std::uint8_t* xq, const float* xs, int k, double& y, double& scale,
                       int g = 1) {
    y = 0;
    scale = 0;
    for (int b = 0; b < k / kBlk; b += g) {
        const double t = static_cast<double>(blockDot(wq, xq, b, g)) * (static_cast<double>(ws[b]) * xs[b]);
        y += t;
        scale += std::fabs(t);
    }
}

// ---------------------------------------------------------------- Swift-1.5 27B dense GEMM shapes of one forward
// n = launches per forward (64 layers: 48 Gated DeltaNet + 16 full attention; research\w4a8\2026-11-07-f16-p1a-w4a8-gfx1151.md
// sec 3.3: qkv 48, z 48, q 16, k + v 32, out 64, gate + up 128, down 64 = 400). rows x K = the weight matrix.
struct Shape {
    const char* name;    // short tensor name
    const char* tensor;  // GGUF tensor (blk.0 = a GDN layer, blk.3 = a full-attention layer)
    int rows, k, n;
};
inline constexpr Shape kSwift[] = {
    {"ffn_gate", "blk.0.ffn_gate.weight", 17408, 5120, 128},  // gate + up
    {"ffn_down", "blk.0.ffn_down.weight", 5120, 17408, 64},
    {"attn_qkv", "blk.0.attn_qkv.weight", 10240, 5120, 48},
    {"attn_gate", "blk.0.attn_gate.weight", 6144, 5120, 48},  // GDN z
    {"ssm_out", "blk.0.ssm_out.weight", 5120, 6144, 64},       // GDN out 48 + attention out 16 (same shape)
    {"attn_q", "blk.3.attn_q.weight", 12288, 5120, 16},
    {"attn_k", "blk.3.attn_k.weight", 1024, 5120, 32},         // k + v
};
inline int launchesPerForward() {
    int s = 0;
    for (const Shape& x : kSwift) s += x.n;
    return s;
}
// activation-producer launches per forward: one per distinct GEMM input per layer (K 5120: attention / GDN input and FFN
// input = 2 x 64; K 6144: the out-projection input x 64; K 17408: the down input x 64)
struct ProdShape {
    int k, n;
};
inline constexpr ProdShape kProd[] = {{5120, 128}, {6144, 64}, {17408, 64}};
inline bool shapeOk(int rows, int k, int ntok) { return rows % kBM == 0 && ntok % kBN == 0 && k % kBK == 0 && k % 256 == 0; }

// the GEMM entries of kernels\gfx1151\gemm_w4a4p.hip (grid x = tokens / bn, y = rows / bm, 256 threads); every bm / bn
// divides kBM / kBN, so shapeOk covers all of them
struct Gemm {
    const char* name;
    int bm, bn;
    bool fold;
    int fg;  // blocks per fold (1 = per 32; 4 = per 128, scales of the group's first block)
};
inline constexpr Gemm kGemms[] = {{"w4a4p_gemm_i4", 128, 256, false, 1},
                                  {"w4a4p_gemm_i4m", 64, 256, false, 1},
                                  {"w4a4p_gemm_i4f", 64, 256, true, 1},
                                  {"w4a4p_gemm_i4fb", 128, 128, true, 1},
                                  {"w4a4p_gemm_i4g", 64, 256, true, 4}};
// register-rate probes: A = kRegA (f16), B = each of kRegB; ops per launch = blocks * 8 waves * iters * 16 WMMAs * 8192
inline constexpr const char* kRegA = "w4a4p_reg_f16";
inline constexpr const char* kRegB[] = {"w4a4p_reg_iu8", "w4a4p_reg_iu8f", "w4a4p_reg_iu4", "w4a4p_reg_iu4f"};
inline constexpr int kRegBlocks = 320, kRegIters = 256;
inline double regOps() { return static_cast<double>(kRegBlocks) * 8 * kRegIters * 16 * 8192.0; }

// ---------------------------------------------------------------- end-to-end projection (ADR-0019: estimate +- its interval)
// Per shape g: A time t_A(g) (gemmfq, us per launch), paired ratio r_g = t_B / t_A (geometric mean of the ABBA pairs) with
// its 95 % half-width h_g. R = sum_g w_g r_g with w_g = n_g t_A(g) / sum n t_A; half-width sqrt(sum (w_g h_g)^2) (cells
// are independent runs). Producer: P = sum_K q_K t_Q(K) / sum_g n_g t_A(g) (relative to the GEMM time; its own noise is
// not propagated). With f = the measured GEMM share of prefill time (WHIRL_PROFILE matmul %):
//   saved = f (1 - R) - f P            (fraction of today's prefill time removed; > 0 = faster)
//   speed-up = 1 / (1 - saved) - 1     (pp tok/s gain)
struct CellR {
    double ta_us = 0, r = 0, half = 0;
    int n = 0;
};
struct Projection {
    double R = 0, R_half = 0, P = 0, saved = 0, saved_lo = 0, saved_hi = 0, gain = 0, gain_lo = 0, gain_hi = 0;
    bool ok = false;
};
inline Projection project(const std::vector<CellR>& cells, double prod_us_per_forward, double f) {
    Projection p;
    double ta = 0;
    for (const CellR& c : cells) ta += c.n * c.ta_us;
    if (ta <= 0 || f <= 0 || f > 1) return p;
    double h2 = 0;
    for (const CellR& c : cells) {
        const double w = c.n * c.ta_us / ta;
        p.R += w * c.r;
        h2 += (w * c.half) * (w * c.half);
    }
    p.R_half = std::sqrt(h2);
    p.P = prod_us_per_forward / ta;
    auto saved = [&](double R) { return f * (1.0 - R) - f * p.P; };
    auto gain = [](double s) { return 1.0 / (1.0 - s) - 1.0; };
    p.saved = saved(p.R);
    p.saved_lo = saved(p.R + p.R_half);
    p.saved_hi = saved(p.R - p.R_half);
    p.gain = gain(p.saved);
    p.gain_lo = gain(p.saved_lo);
    p.gain_hi = gain(p.saved_hi);
    p.ok = true;
    return p;
}
// ADR-0019 reading of a projection: "GAIN" when the whole interval of the gain is above 0, "LOSS" when it is below 0, else
// "OPEN" (no hand-set band: the interval alone decides)
inline std::string verdict(const Projection& p) {
    if (!p.ok) return "NONE";
    if (p.gain_lo > 0) return "GAIN";
    if (p.gain_hi < 0) return "LOSS";
    return "OPEN";
}

}  // namespace kt::w4a4p
