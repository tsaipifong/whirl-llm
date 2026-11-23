// WHIRL v0252-8sattn (PLAN V1-148): int8 Q.K^T of the BALANCE-mode gfx1151 flash prefill attention (attn_prefill_wmma_i8 /
// attn_prefill_wmma_s_i8 in kernels/gfx1151/attn.hip), shared by the device kernels and the CPU tests
// (tests/kernels/unit/i8qk_ref_tests.cpp). SPDX-License-Identifier: Apache-2.0
//
// Why int8 here although v_wmma_i32_16x16x16_iu8 runs at the f16 rate on gfx1151 (32 cycles; FAST-0: iu8 = f16 TOPS):
// the f16 kernel holds its 16 query fragments as half8 pieces (64 VGPRs) and sits at 254-256 VGPRs, so the next K / V
// tile cannot be loaded into registers while the current one is scored (8S-P5 / feat026-attn8s: every prefetch spilled),
// and the K / V load + LDS staging is its largest loss (8S-P5 probe: no loads = +73 %). The int8 query pieces take
// 32 VGPRs (8 codes = 2 words per 16-dim step per half-wave); the freed 32 VGPRs hold the next tile's K (16) and V (16)
// while the WMMAs run (template flag PF of attn_prefill_impl). The K tile in LDS is int8 (half the bytes and LDS reads).
//
// Per row of 256 dims (absmax / 127, round to nearest even, codes in [-127, 127]):
//  - each query (per head) once per block: s_q = amax_q / 127;
//  - each key (per KV head) once per key tile in the block, when the tile is stored to LDS: s_k = amax_k / 127.
// score = s_q * s_k * sum_d code_q[d] * code_k[d] (|sum| <= 256 * 127^2 = 4,129,024 < 2^24: exact in int32 and in f32),
// then the f16 kernel's mask / online softmax / P.V (f16 P, f32 O) unchanged. Q and K are the Hadamard-rotated q8h
// values on balance's q8h path (attn_prep spreads channel outliers before the per-row absmax).
// Byte order: byte m of the packed 128-bit fragment (4 dwords, little endian) is K element m, two's complement, on both
// operands (llama.cpp ggml-cuda/mma.cuh RDNA3 int8 tiles: 16 consecutive int8 per lane in an int32x4).
// BALANCE mode only (items: int8 KV + 8060S prefill int8 activations): the output changes (numerics flag attn_i8qk,
// WHIRL_ATTN_I8QK=0 = the f16 kernel). Never precise.
#pragma once

#include <stdint.h>
#if !defined(__HIP_DEVICE_COMPILE__)
#include <math.h>
#define WHIRL_I8_RINT rintf
#else
#define WHIRL_I8_RINT __builtin_rintf
#endif

#if defined(__HIP__) || defined(__HIPCC__)
#define WHIRL_I8_HD __host__ __device__
#else
#define WHIRL_I8_HD
#endif

// 1 / scale of a row with absolute maximum amax (0 for an all-zero row: every code 0, scale 0)
WHIRL_I8_HD inline float i8qk_inv(float amax) { return amax > 0.f ? 127.f / amax : 0.f; }
// scale of a row (the factor of its codes)
WHIRL_I8_HD inline float i8qk_scale(float amax) { return amax * (1.f / 127.f); }
// code of x in [-127, 127]: round to nearest even of x * inv (|x| <= amax -> |x * inv| <= 127 up to one rounding: clamped)
WHIRL_I8_HD inline int i8qk_code(float x, float inv) {
    const float y = WHIRL_I8_RINT(x * inv);
    return y > 127.f ? 127 : (y < -127.f ? -127 : static_cast<int>(y));
}
// 4 values -> one 32-bit word, value i in bits [8 i, 8 i + 8)
WHIRL_I8_HD inline uint32_t i8qk_pack4(const float *x, float inv) {
    uint32_t w = 0;
    for (int i = 0; i < 4; ++i) w |= (static_cast<uint32_t>(i8qk_code(x[i], inv)) & 0xFFu) << (8 * i);
    return w;
}
// value i of a packed word (sign-extended)
WHIRL_I8_HD inline int i8qk_byte(uint32_t w, int i) {
    const int v = static_cast<int>((w >> (8 * i)) & 0xFFu);
    return v >= 128 ? v - 256 : v;
}
