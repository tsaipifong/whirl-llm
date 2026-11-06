// v0252-f16 (F16 P1a): host + device helpers of the gfx1151 W4A8 prefill GEMM (kernels/gfx1151/gemm_w4a8.hip: MXFP4
// weights as WHIRL stores them x int8 activations, iu8 WMMA) and of its CPU reference (tests/kernels/w4a8_ref.h). The
// CPU test (tests/kernels/unit/w4a8_ref_tests.cpp) runs these same functions (v_perm emulated on the host), so the
// device decode and the CPU checks cannot drift apart. Not part of any product build yet (P1a).
// SPDX-License-Identifier: Apache-2.0
//
// Numbers (progress/v0252-f16.PRIOR_ART.md sec 3):
//   weights  MXFP4 code n -> kvalues_mxfp4[n] in {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12} (= 2 x e2m1,
//            code 8 = -0 -> 0): exact in int8; block scale 2^(e - 128) (e8m0 x 0.5; kernels/shared/common_part.h
//            e8m0_half = ggml GGML_E8M0_TO_FP32_HALF, f32 subnormal for e < 2, e = 255 -> 2^127, no NaN code)
//   activations per 32 along K (WHIRL quantize_q8, kernels/gfx1151/gemv_q.hip:14-27): d = amax / 127, q = rint(x * (1/d))
//            (0 when d == 0); a block holding a NaN / Inf -> d = NaN and q = 0, so the token's outputs become NaN like the
//            f16 path's (gemmfq propagates NaN / Inf through the f16 WMMA; WHIRL's decode quantizer zeroes them instead)
//   layout   tiled Q8_1 of Gufo (MIT, refsrc gufo prefill_quant_gemm.hpp:25-48): per 16-token tile tt and 32-block kb,
//            576 bytes at (tt * nkb + kb) * 576 = [b0: 16 tokens x 16 int8 (k 0..15)][b1: 16 x 16 (k 16..31)][16 f32 d];
//            the WMMA B fragment of token t is the 16 contiguous bytes at b0 / b1 + 16 * (t % 16); padding tokens are 0
//   GEMM     per 32-block an exact int32 dot c (|c| <= 32 * 12 * 127 = 48768); epilogue in ascending kb:
//            s = dw * dx (f32), acc = fma(float(c), s, acc); Y[t * nrows + r] = acc (or Y + acc when accumulating)
#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__HIPCC__)
#define W4A8_HD __host__ __device__ __forceinline__
#else
#define W4A8_HD inline
#endif

constexpr int kW4A8TileTok = 16, kW4A8TileBytes = 576, kW4A8ScaleOff = 512;

// kvalues_mxfp4[code] (the low 4 bits of code)
W4A8_HD int w4a8_kv(uint32_t code) {
    const uint64_t mag = 0x0C08060403020100ull;  // 0 1 2 3 4 6 8 12
    const int m = static_cast<int>((mag >> (8u * (code & 7u))) & 0xFFu);
    return (code & 8u) ? -m : m;
}

// portable v_perm_b32 D = perm(S0, S1, sel) (AMD RDNA3 ISA): per result byte, selector 0..3 = byte of S1, 4..7 = byte
// of S0, 8 / 9 / 10 / 11 = the sign of S1[15] / S1[31] / S0[15] / S0[31] as 0x00 / 0xFF, 12 = 0x00, 13..255 = 0xFF
W4A8_HD uint32_t w4a8_perm_ref(uint32_t s0, uint32_t s1, uint32_t sel) {
    const uint64_t v = (static_cast<uint64_t>(s0) << 32) | s1;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t sb = (sel >> (8 * i)) & 0xFFu;
        uint32_t b;
        if (sb < 8) b = static_cast<uint32_t>(v >> (8 * sb)) & 0xFFu;
        else if (sb == 8) b = (s1 & 0x8000u) ? 0xFFu : 0u;
        else if (sb == 9) b = (s1 & 0x80000000u) ? 0xFFu : 0u;
        else if (sb == 10) b = (s0 & 0x8000u) ? 0xFFu : 0u;
        else if (sb == 11) b = (s0 & 0x80000000u) ? 0xFFu : 0u;
        else if (sb == 12) b = 0u;
        else b = 0xFFu;
        r |= b << (8 * i);
    }
    return r;
}
W4A8_HD uint32_t w4a8_perm(uint32_t s0, uint32_t s1, uint32_t sel) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_perm(s0, s1, sel);
#else
    return w4a8_perm_ref(s0, s1, sel);
#endif
}

// 4 MXFP4 codes (bits 3:0 of each byte; bits 7:4 are ignored, so a caller may pass q or q >> 4 unmasked) -> 4 int8
// kvalues: the two-table v_perm select of WHIRL mx_lut4 (kernels/shared/gemv_q_part.h:17-24) = llama.cpp
// get_int_from_table_16 (MIT)
W4A8_HD uint32_t w4a8_lut4(uint32_t nib4) {
    const uint32_t t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    const uint32_t sel = nib4 & 0x07070707u;
    const uint32_t lo = w4a8_perm(t1, t0, sel);
    const uint32_t hi = w4a8_perm(t3, t2, sel);
    const uint32_t m = ((nib4 & 0x08080808u) >> 3) * 0xFFu;
    return (hi & m) | (lo & ~m);
}

// f32 bits of 2^(e - 128)
W4A8_HD uint32_t w4a8_e8m0_bits(uint32_t e) { return e < 2 ? (0x00200000u << e) : ((e - 1u) << 23); }

// tiled activation layout
W4A8_HD size_t w4a8_xq_bytes(int n_tok, int k) {
    return static_cast<size_t>((n_tok + kW4A8TileTok - 1) / kW4A8TileTok) * static_cast<size_t>(k / 32) * kW4A8TileBytes;
}
W4A8_HD size_t w4a8_xq_tile(int t, int kb, int k) {
    return (static_cast<size_t>(t / kW4A8TileTok) * static_cast<size_t>(k / 32) + static_cast<size_t>(kb)) * kW4A8TileBytes;
}
W4A8_HD size_t w4a8_xq_qoff(int t, int kk, int k) {
    const int j = kk % 32;
    return w4a8_xq_tile(t, kk / 32, k) + static_cast<size_t>(j / 16) * 256 + static_cast<size_t>(t % kW4A8TileTok) * 16 +
           static_cast<size_t>(j % 16);
}
W4A8_HD size_t w4a8_xq_doff(int t, int kb, int k) {
    return w4a8_xq_tile(t, kb, k) + kW4A8ScaleOff + static_cast<size_t>(t % kW4A8TileTok) * 4;
}

// quantizer arithmetic (the device kernel and the host block below use exactly these steps)
W4A8_HD float w4a8_scale(float amax) { return amax / 127.f; }
W4A8_HD float w4a8_inv(float d) { return d > 0.f ? 1.f / d : 0.f; }
W4A8_HD int w4a8_q8(float x, float id) { return static_cast<int>(rintf(x * id)); }

// one 32-block on the host (the device kernel does the same per lane with a wave reduction)
inline void w4a8_quant_block(const float *x, float *d, int8_t *q) {
    float amax = 0.f;
    bool bad = false;
    for (int i = 0; i < 32; ++i) {
        const float a = fabsf(x[i]);
        if (!(a <= 3.402823466e+38f)) bad = true;  // NaN or Inf
        else if (a > amax) amax = a;
    }
    if (bad) {
        *d = nanf("");
        for (int i = 0; i < 32; ++i) q[i] = 0;
        return;
    }
    const float dd = w4a8_scale(amax), id = w4a8_inv(dd);
    *d = dd;
    for (int i = 0; i < 32; ++i) q[i] = static_cast<int8_t>(w4a8_q8(x[i], id));
}
