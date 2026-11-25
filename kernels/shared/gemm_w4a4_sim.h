// WHIRL v0252-w4a4k2 (PLAN V1-144, kill-screen K2): the math of the W4A4 FAKE-QUANT arm of the FAST gfx1151 dense prefill
// GEMM (kernels/gfx1151/gemm_w4a4s.hip: w4s_dequant_f16_mxfp4, w4s_quant_x16), shared by the device kernels and the CPU
// references / tests (tests/kernels/unit/kt_bench_tests.cpp, tests/kernels/test_w4s.cpp). SPDX-License-Identifier: Apache-2.0
//
// Research arm, not a product path (numerics flag w4a4_sim, WHIRL_W4A4_SIM, fast mode only, default off): it gives the
// numbers a W4A4 iu4 kernel over the shipped MXFP4 weights would compute (the probe of v0252-w4a4, tests/kernels/kt_w4a4p.h,
// is the single definition; the unit tests pin every code and scale here to it), without its speed:
//  - weights: MXFP4 code n -> int4 level {0, 1, 1, 2, 2, 4, 5, 7} (+ sign) = rint(kv * 7 / 12), per-32 scale
//    2^(e - 128) x 12 / 7 (e = the block's E8M0 byte); dequantized value = f16(float(level) x scale);
//  - activations, per 32 along K: step d = amax / 7 (or the MSE-best of hipfire's 8 candidates (amax / 7) x 0.5 x (1 + j / 7),
//    block_i4_128_quant.hip:83-139, at our granularity and +-7 clamp), code = clamp(rint(x / d), -7, 7) (absmax: x x 7 / amax,
//    the probe's form); dequantized value = f16(float(code) x d).
// The f16 GEMM then multiplies the dequantized values with f32 accumulation: vs the iu4 kernel's exact int32 block sums times
// f32 scales, each value carries one extra f16 rounding (relative <= 2^-11), far below the 8-12 % re-quantization error.
#pragma once

#include <stdint.h>
#if !defined(__HIP_DEVICE_COMPILE__)
#include <math.h>
#include <string.h>
#define W4S_RINT rintf
#define W4S_FMA fmaf
#define W4S_LDEXP ldexpf
#else
#define W4S_RINT __builtin_rintf
#define W4S_FMA __builtin_fmaf
#define W4S_LDEXP __builtin_ldexpf
#endif

#if defined(__HIP__) || defined(__HIPCC__)
#define W4S_HD __host__ __device__
#else
#define W4S_HD
#endif

// round 2 (2026-11-26): the device must round every product it stores as f16 in two steps (f32 product, then the f16
// conversion), as the host references do. Round 1's GPU row: the MSE-best activation row differed in 102 of 262144 values;
// the build-3 disassembly had the store fused into v_fma_mixlo/hi_f16 (f32 operands, ONE rounding of the exact product to
// f16), and the double vs single rounding rate of the MSE candidates on Gaussian blocks (~4e-4; 0 for absmax / 7) matches.
// `#pragma clang fp contract(off)` did not stop that fold (build 4 still had 32 v_fma_mix), so W4S_F32(p) pins the f32
// product in a register (an empty asm with a "+v" operand) before the conversion. Host: a no-op (already two roundings).
#if defined(__HIP_DEVICE_COMPILE__)
#define W4S_F32(p_) __asm__ volatile("" : "+v"(p_))
#else
#define W4S_F32(p_) ((void)0)
#endif

// weight grids (Flags::w4a4_sim bits 8..10; kernels w4s_dequant_f16_mxfp4 + kW4sDqSuffix[g]): 0 = w, the 6/7 grid (absmax / 7 of
// the MXFP4 range, round 1); 1 = w1, step one e2m1 unit, halves up; 2 = w1e, halves to even; 3 = w1d, halves up at an even K
// index of the block and down at an odd one (zero-mean dither); 4 = wb, per 32-block the better of w and w1 by squared error
#define W4S_GRIDS 5

// int4 level of an MXFP4 code (low 3 bits = magnitude index, bit 3 = sign; kt_w4a4p.h mxToI4)
W4S_HD inline int w4s_wlevel(int code) {
    const int m = code & 7;
    const int v = m < 3 ? (m + 1) / 2 : (m < 5 ? 2 : (m == 5 ? 4 : (m == 6 ? 5 : 7)));
    return (code & 8) ? -v : v;
}
// per-32 weight scale of an E8M0 byte (kt_w4a4p.h mxI4Scale; bitwise equal wherever the result is a normal f32, e >= 2)
W4S_HD inline float w4s_wscale(uint32_t e8m0) { return W4S_LDEXP(12.0f / 7.0f, static_cast<int>(e8m0) - 128); }
// dequantized weight before the f16 rounding
W4S_HD inline float w4s_wval(int code, uint32_t e8m0) {
    return static_cast<float>(w4s_wlevel(code)) * w4s_wscale(e8m0);
}

// level of an MXFP4 code on grid g (0..3; k = the value's index in its 32-block: its parity drives w1d)
W4S_HD inline int w4s_wlevel_g(int code, int g, int k) {
    if (g == 0) return w4s_wlevel(code);
    const int m = code & 7;
    int v;
    if (m == 1) v = g == 1 ? 1 : (g == 2 ? 0 : ((k & 1) ? 0 : 1));       // 0.5
    else if (m == 3) v = g == 1 ? 2 : (g == 2 ? 2 : ((k & 1) ? 1 : 2));  // 1.5
    else v = m == 0 ? 0 : (m == 2 ? 1 : (m == 4 ? 2 : (m == 5 ? 3 : (m == 6 ? 4 : 6))));
    return (code & 8) ? -v : v;
}
// scale of grid g: w 2^(e-128) x 12/7; w1 / w1e / w1d one e2m1 unit = 2^(e-127)
W4S_HD inline float w4s_wscale_g(uint32_t e8m0, int g) { return g == 0 ? w4s_wscale(e8m0) : W4S_LDEXP(1.0f, static_cast<int>(e8m0) - 127); }
// MXFP4 code -> kv (2 x e2m1: 0, 1, 2, 3, 4, 6, 8, 12, + sign)
W4S_HD inline int w4s_kv(int code) {
    const int m = code & 7;
    const int v = m < 5 ? m : (m == 5 ? 6 : (m == 6 ? 8 : 12));
    return (code & 8) ? -v : v;
}
// one MXFP4 sub-block (qs: its 16 bytes, value j = low nibble of byte j, value j + 16 = high nibble; e = its E8M0 byte) on grid
// g (0..4) -> 32 values before the f16 rounding. wb (4): the grid (w or w1) with the smaller squared error against the MXFP4
// values, in exact integers (x 7: w's level L0 is 12 L0 / 7 kv units, w1's L1 is 2 L1); ties -> w. A real kernel needs one bit
// per 32-block for it (a side array of nrows x K / 32 bits) or a pre-converted int4 weight copy.
W4S_HD inline void w4s_wblock(const uint8_t* qs, uint32_t e8m0, int g, float* v) {
    int gg = g;
    if (g == 4) {
        int e0 = 0, e1 = 0;
        for (int k = 0; k < 32; ++k) {
            const int code = k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4);
            const int kv7 = 7 * w4s_kv(code);
            const int d0 = kv7 - 12 * w4s_wlevel_g(code, 0, k), d1 = kv7 - 14 * w4s_wlevel_g(code, 1, k);
            e0 += d0 * d0;
            e1 += d1 * d1;
        }
        gg = e1 < e0 ? 1 : 0;
    }
    const float s = w4s_wscale_g(e8m0, gg);
    for (int j = 0; j < 16; ++j) {
        v[j] = static_cast<float>(w4s_wlevel_g(qs[j] & 15, gg, j)) * s;
        v[j + 16] = static_cast<float>(w4s_wlevel_g(qs[j] >> 4, gg, j + 16)) * s;
    }
}

// one 32-value activation block -> codes q[32] in [-7, 7] and its step d (0 for an all-zero block). mse = 0: absmax / 7 (the
// probe's quantRow, bitwise); mse = 1: the MSE-best step of the 8-point grid (strict <: the first minimum wins)
W4S_HD inline void w4s_qblock(const float* x, int mse, int* q, float* d) {
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[i]));
    if (mse == 0 || !(amax > 0.f)) {
        const float inv = amax > 0.f ? 7.f / amax : 0.f;
        for (int i = 0; i < 32; ++i) {
            const float r = W4S_RINT(x[i] * inv);
            q[i] = static_cast<int>(fminf(fmaxf(r, -7.f), 7.f));
        }
        *d = amax / 7.f;
        return;
    }
    float best = 0.f, best_d = amax / 7.f;
    for (int j = 0; j < 8; ++j) {
        const float dj = (amax / 7.0f) * 0.5f * (1.0f + static_cast<float>(j) / 7.0f);
        float e2 = 0.f;
        for (int i = 0; i < 32; ++i) {
            const float c = fminf(fmaxf(W4S_RINT(x[i] / dj), -7.f), 7.f);
            const float err = W4S_FMA(-c, dj, x[i]);
            e2 = W4S_FMA(err, err, e2);
        }
        if (j == 0 || e2 < best) {
            best = e2;
            best_d = dj;
        }
    }
    for (int i = 0; i < 32; ++i) q[i] = static_cast<int>(fminf(fmaxf(W4S_RINT(x[i] / best_d), -7.f), 7.f));
    *d = best_d;
}

#if !defined(__HIP_DEVICE_COMPILE__)
// host f32 -> f16 bits, round to nearest even (= the device's (_Float16) conversion; subnormals kept, overflow -> inf)
inline uint16_t w4s_f2h(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t a = u & 0x7FFFFFFFu;
    if (a >= 0x7F800000u) return static_cast<uint16_t>(sign | 0x7C00u | (a > 0x7F800000u ? 0x200u : 0u));
    if (a >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);  // >= 65520: rounds to inf
    if (a < 0x38800000u) {                                               // below 2^-14: f16 subnormal (or zero)
        if (a < 0x33000000u) return static_cast<uint16_t>(sign);         // < 2^-25: rounds to 0 (2^-25 itself: tie to even 0)
        const uint32_t e = a >> 23, m = (a & 0x7FFFFFu) | 0x800000u;
        const uint32_t shift = 126u - e;  // value = m x 2^(e - 150); f16 subnormal unit 2^-24 -> m >> (126 - e)
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (r & 1u))) ++r;
        return static_cast<uint16_t>(sign | r);
    }
    uint32_t r = ((a >> 13) - (112u << 10));  // rebias 127 -> 15
    const uint32_t rem = a & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    return static_cast<uint16_t>(sign | r);
}
inline float w4s_h2f(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    float f;
    if (e == 0) {
        f = ldexpf(static_cast<float>(m), -24);
        uint32_t u;
        memcpy(&u, &f, 4);
        u |= sign;
        memcpy(&f, &u, 4);
        return f;
    }
    const uint32_t u = sign | (e == 31 ? (0xFFu << 23) | (m << 13) : ((e + 112u) << 23) | (m << 13));
    memcpy(&f, &u, 4);
    return f;
}
// one MXFP4 row (WHIRL / ggml layout: per 256 values 8 E8M0 bytes + 8 x 16 qs; sub-block ib: qs byte j low nibble = value
// j, high nibble = value j + 16) -> k f16 bits: the reference of w4s_dequant_f16_mxfp4
inline void w4s_dequant_row_host(const uint8_t* mx, int k, uint16_t* out, int g = 0) {
    for (int b = 0; b < k / 32; ++b) {
        const uint8_t* blk = mx + static_cast<size_t>(b / 8) * 136;
        const int ib = b % 8;
        float v[32];
        w4s_wblock(blk + 8 + 16 * ib, blk[ib], g, v);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = w4s_f2h(v[i]);
    }
}
// one f16 activation row of k values (k % 32 == 0) -> its fake-quantized f16 bits: the reference of w4s_quant_x16
inline void w4s_quant_row_host(const uint16_t* x, int k, int mse, uint16_t* out) {
    for (int b = 0; b < k / 32; ++b) {
        float xf[32];
        int q[32];
        float d;
        for (int i = 0; i < 32; ++i) xf[i] = w4s_h2f(x[b * 32 + i]);
        w4s_qblock(xf, mse, q, &d);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = w4s_f2h(static_cast<float>(q[i]) * d);
    }
}
#endif
