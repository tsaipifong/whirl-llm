// WHIRL device code (gfx1151 / RDNA 3.5): GGUF quant block decoders (dec8, MXFP4 superblocks).
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Quant formats. g = index of an 8-value group within the row.
// ---------------------------------------------------------------------------

#include "shared/common_part.h"  // also in the gfx1201 set (kernels/common.hip)

// ---- shared kernel vocabulary (arch025 N14: kernels.common; was defined in the first family that used it) ----
#include "shared/common_vocab.h"
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
template <int T> constexpr int gs_sbb() { return 256 / Q<T>::blk * Q<T>::bytes; }
typedef unsigned int gs_u32x4 __attribute__((ext_vector_type(4)));

// Quantize the 32 values held by one wave (lane i holds v) into xq/xd at block b.
__device__ __forceinline__ void wave_q8(float v, int8_t *xq, float *xd, int b) {
    const int lane = threadIdx.x % WARP;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    const int q = __float2int_rn(v * id);
    const int sum = warp_sum_i(q);
    xq[b * 32 + lane] = (int8_t)q;
    if (lane == 0) xd[b] = pk_make(d, sum);
}

// ---- gfx11 WMMA fragment vocabulary (arch025 N14: kernels.common; was in gfx1151/gemm_prefill.hip, used by gemv_h, attn, probes) ----
// gfx11 wave32 WMMA fragment layout (llama.cpp ggml-cuda/mma.cuh, RDNA3):
//   A, B: lane holds row/column (lane % 16), all 16 k (lanes 16-31 = 0-15)
//   D:    lane holds column n = lane % 16, rows m = 2 * v + (lane / 16)
// Fragment = 16 consecutive halfs of an LDS row (two 16-byte reads).
__device__ __forceinline__ half16 ld_frag16(const _Float16 *p) {
    const half8 lo = *(const half8 *)p, hi = *(const half8 *)(p + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
#define WMMA16(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
// accumulator element v of this lane -> row within the 16x16 tile
#define CROW(lane_, v_) (2 * (v_) + ((lane_) / 16))
// the other half-wave's copy of a 32-bit value (lane ^ 16), one VALU op
__device__ __forceinline__ int xhalf(int v) {
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false, false);
}
// Join per-half 8-value pieces into the full 16-k fragment both halves hold:
// k [0, 8) = half 0's piece, k [8, 16) = half 1's piece.
__device__ __forceinline__ half16 join_halves(half8 own, int lane) {
    const int4 o = __builtin_bit_cast(int4, own);
    const int4 p = make_int4(xhalf(o.x), xhalf(o.y), xhalf(o.z), xhalf(o.w));
    const bool h0 = lane < 16;
    const int4 lo = h0 ? o : p, hi = h0 ? p : o;
    return __builtin_shufflevector(__builtin_bit_cast(half8, lo), __builtin_bit_cast(half8, hi),
                                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
