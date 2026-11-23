// WHIRL v0252-8sfp (PLAN V1-144): int4 Q.K^T of the FAST-mode gfx1151 flash prefill attention (attn_prefill_wmma_i4 /
// attn_prefill_wmma_s_i4 in kernels/gfx1151/attn.hip), shared by the device kernels and the CPU tests
// (tests/kernels/unit/i4qk_ref_tests.cpp). SPDX-License-Identifier: Apache-2.0
//
// The gfx1151 (RDNA 3.5) matrix unit runs v_wmma_i32_16x16x16_iu4 in 16 cycles against 32 for the f16 / iu8 forms
// (AMD matrix instruction calculator, rdna3 table; FAST-0 measured iu4 87.9 vs f16 43.7 TOPS on the 8060S): the only
// lower-precision matrix rate on this card. The kernel quantizes, per row of 256 dims:
//  - each query (per head) once per block: s_q = amax_q / 7, code = rne(x * 7 / amax_q) in [-7, 7];
//  - each key (per KV head) once per key tile in the block: s_k = amax_k / 7, same codes.
// score = s_q * s_k * sum_d code_q[d] * code_k[d] (|sum| <= 256 * 49, exact in int32 and in f32), then the f16 kernel's
// mask / online softmax / P.V (f16 P, f32 O) unchanged. Q and K are the rotated q8h / q4 values on the KV-rotated paths
// (the Hadamard rotation of attn_prep spreads channel outliers before the per-row absmax).
// Nibble order (hipfire gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:21-26, probe-verified on gfx1100, MIT): bits
// [4 m, 4 m + 4) of the packed 64-bit fragment are K element m, two's complement, on both operands.
// FAST mode only: the output changes (numerics flag attn_i4qk, WHIRL_ATTN_I4QK=0 = the f16 kernel).
#pragma once

#include <stdint.h>
#if !defined(__HIP_DEVICE_COMPILE__)
#include <math.h>
#define WHIRL_I4_RINT rintf
#else
#define WHIRL_I4_RINT __builtin_rintf
#endif

#if defined(__HIP__) || defined(__HIPCC__)
#define WHIRL_I4_HD __host__ __device__
#else
#define WHIRL_I4_HD
#endif

// 1 / scale of a row with absolute maximum amax (0 for an all-zero row: every code 0, scale 0)
WHIRL_I4_HD inline float i4qk_inv(float amax) { return amax > 0.f ? 7.f / amax : 0.f; }
// scale of a row (the factor of its codes)
WHIRL_I4_HD inline float i4qk_scale(float amax) { return amax * (1.f / 7.f); }
// code of x in [-7, 7]: round to nearest even of x * inv (|x| <= amax -> |x * inv| <= 7 up to one rounding: clamped)
WHIRL_I4_HD inline int i4qk_code(float x, float inv) {
    const float y = WHIRL_I4_RINT(x * inv);
    return y > 7.f ? 7 : (y < -7.f ? -7 : static_cast<int>(y));
}
// 8 values -> one 32-bit word, value i in bits [4 i, 4 i + 4)
WHIRL_I4_HD inline uint32_t i4qk_pack8(const float *x, float inv) {
    uint32_t w = 0;
    for (int i = 0; i < 8; ++i) w |= (static_cast<uint32_t>(i4qk_code(x[i], inv)) & 0xFu) << (4 * i);
    return w;
}
// value i of a packed word (sign-extended)
WHIRL_I4_HD inline int i4qk_nib(uint32_t w, int i) {
    const int v = static_cast<int>((w >> (4 * i)) & 0xFu);
    return v >= 8 ? v - 16 : v;
}
