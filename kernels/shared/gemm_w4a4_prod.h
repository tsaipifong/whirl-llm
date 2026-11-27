// WHIRL v0252-w4a4p (PLAN V1-144, round 1): the math of the PRODUCT W4A4 (iu4 WMMA) dense prefill GEMM of the FAST gfx1151
// path (kernels/gfx1151/gemm_w4a4q.hip: w4q_wconv_mxfp4, w4q_quant_x, w4q_gemm), shared by the device kernels and the CPU
// references / tests (tests/kernels/unit/kt_bench_tests.cpp, tests/kernels/test_gemm.cpp family w4s). SPDX-License-Identifier: Apache-2.0
//
// Fast-only opt-in (WHIRL_W4A4=1 -> numerics flag w4a4_sim with bit w4a4_iu4; default off). Its reference is the accepted
// simulation arm WHIRL_W4A4_SIM=wga (gemm_w4a4_sim.h): the SAME integer codes and the same f32 scales:
//  - weights (grid w + tensor gain g): MXFP4 code -> int4 level w4s_wlevel (0, 1, 1, 2, 2, 4, 5, 7 + sign), per-32 scale
//    w4s_wscale(e) x gain (the gain of ops::w4sGains mode 1, folded into the stored scale once: zero cost in the loop);
//  - activations (a): per 32 along K, w4s_qblock(mse 0) codes and step d = amax / 7 (bitwise the sim's);
//  - packing (both sides, the K1 probe's verified iu4 fragment order, kt_w4a4p.h packRow): byte j of a 32-block =
//    level[2j] & 15 | (level[2j + 1] & 15) << 4 (natural K order, low nibble first, two's complement).
// The GEMM then computes per output, for the 32-blocks b in ascending K order: c_b = sum_32 qw qx (int32, exact), s_b =
// ws[b] x xs[b] (f32), acc = fma(float(c_b), s_b, acc) (f32) - w4q_fold_host below, the kernel's bitwise target.
// Where it is NOT bit-identical to the sim (by construction, both sides exact in their own form): the sim rounds every
// dequantized value to f16 (weight f16(f32(level x s) x g), activation f16(q x d)) and sums f16 products in the f16 WMMA's f32
// chain; the product keeps the integers exact and rounds only the f32 scale product s_b and the f32 fold. Per term the two
// differ by <= 2^-10 of its size (two f16 roundings) plus the f32 sums (kt row: 2^-10 + 2^-12 of sum |terms|, K 1024).
#pragma once

#include "gemm_w4a4_sim.h"

// the MXFP4 code of value k (0..31) of one sub-block (qs: its 16 bytes; value j = low nibble of byte j, j + 16 = high nibble)
W4S_HD inline int w4q_code(const uint8_t* qs, int k) { return k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4); }

// one MXFP4 sub-block -> 16 packed int4 bytes (natural K order) + its f32 scale w4s_wscale(e) x gain
W4S_HD inline void w4q_wblock(const uint8_t* qs, uint32_t e8m0, float gain, uint8_t* out, float* scale) {
    for (int j = 0; j < 16; ++j) {
        const int lo = w4s_wlevel(w4q_code(qs, 2 * j)), hi = w4s_wlevel(w4q_code(qs, 2 * j + 1));
        out[j] = static_cast<uint8_t>((lo & 15) | ((hi & 15) << 4));
    }
    *scale = w4s_wscale(e8m0) * gain;
}

// one 32-value activation block (f32 values = f16 widened) -> 16 packed int4 bytes + step d (absmax / 7: w4s_qblock mse 0)
W4S_HD inline void w4q_xblock(const float* x, uint8_t* out, float* d) {
    int q[32];
    w4s_qblock(x, 0, q, d);
    for (int j = 0; j < 16; ++j) out[j] = static_cast<uint8_t>((q[2 * j] & 15) | ((q[2 * j + 1] & 15) << 4));
}

// signed nibble k of a packed row
W4S_HD inline int w4q_nib(const uint8_t* row, int k) {
    const int v = (k & 1) ? (row[k / 2] >> 4) : (row[k / 2] & 15);
    return v >= 8 ? v - 16 : v;
}

#if !defined(__HIP_DEVICE_COMPILE__)
// one MXFP4 row of k values (k % 256 == 0) -> k / 2 packed bytes + k / 32 scales (the reference of w4q_wconv_mxfp4)
inline void w4q_wconv_row_host(const uint8_t* mx, int k, float gain, uint8_t* q, float* s) {
    for (int b = 0; b < k / 32; ++b) {
        const uint8_t* blk = mx + static_cast<size_t>(b / 8) * 136;
        const int ib = b % 8;
        w4q_wblock(blk + 8 + 16 * ib, blk[ib], gain, q + static_cast<size_t>(b) * 16, s + b);
    }
}
// one f16 activation row (bits) of k values (k % 32 == 0) -> k / 2 packed bytes + k / 32 steps (the reference of w4q_quant_x)
inline void w4q_quant_row_host(const uint16_t* x, int k, uint8_t* q, float* s) {
    for (int b = 0; b < k / 32; ++b) {
        float xf[32];
        for (int i = 0; i < 32; ++i) xf[i] = w4s_h2f(x[b * 32 + i]);
        w4q_xblock(xf, q + static_cast<size_t>(b) * 16, s + b);
    }
}
// the per-32 fold of one output in the kernel's order (w4q_gemm's bitwise target for an f32 store without accumulate)
inline float w4q_fold_host(const uint8_t* wq, const float* ws, const uint8_t* xq, const float* xs, int k) {
    float acc = 0.f;
    for (int b = 0; b < k / 32; ++b) {
        int c = 0;
        for (int i = b * 32; i < b * 32 + 32; ++i) c += w4q_nib(wq, i) * w4q_nib(xq, i);
        const float s = ws[b] * xs[b];
        acc = fmaf(static_cast<float>(c), s, acc);
    }
    return acc;
}
#endif
