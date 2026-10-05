// WHIRL device code (gfx1151 / RDNA 3.5): shared device helpers (warp reductions, packed int8 scale words).
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// HIP kernels for RDNA 3.5 (gfx1151, Radeon 8060S / Strix Halo),
// Qwen3.5-family (qwen35 / qwen35moe). A fork of the gfx1201 (RDNA 4) kernel
// source; kernel names and signatures match it so the host code is shared
// (include/whirl/kernels_abi.h). What differs is the WMMA fragment code: gfx11
// uses v_wmma_*_w32 (no _gfx12 suffix) with 16 halfs per lane for A/B (lanes
// 16-31 duplicate lanes 0-15) and C/D rows interleaved per lane; there is no
// fp8 / int8-dot WMMA path, and the int8 activation scale words (xd) carry the
// block sum (see pk_make below).
//
// Compiled at build time into the gfx1151 code object, embedded in the host
// binary and loaded with hipModuleLoadData; every entry point is extern "C".
// Weights stay in their GGUF block formats; each quant type gets a `dec8`
// that decodes 8 consecutive values. Numerics follow the GGUF block formats
// and llama.cpp's reference math; the C++ CPU references in tests/kernels/
// check them.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "iq_tables.h"

#define WARP 32

// KV cache element type (f16: half the bytes of f32, and WMMA-ready).
typedef _Float16 kv_t;  // (f16 KV element; q8 pools use int8 + scales, see KvArgs)

// Debug slots (host reads via hipModuleGetGlobal "whirl_dbg").
__device__ unsigned long long whirl_dbg[8];

__device__ __forceinline__ float h2f(const uint8_t *p) {
    uint16_t u = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    return __half2float(__ushort_as_half(u));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, WARP);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, WARP));
    return v;
}

// gfx1151 int8 activation scale word (the xd[] arrays): bits 31..13 hold f32
// bits 30..12 of d (full exponent, 11-bit mantissa, rounded), bits 12..0 the
// sum of the block's 32 int8 values + 4096. Consumers get the block sum for
// free instead of recomputing it with dp4 per row (dp4 is half rate here).
__device__ __forceinline__ float pk_d(uint32_t p) { return __uint_as_float((p >> 13) << 12); }
__device__ __forceinline__ int pk_s(uint32_t p) { return (int)(p & 0x1FFFu) - 4096; }
__device__ __forceinline__ float xdv(const float *p) { return pk_d(__float_as_uint(*p)); }
__device__ __forceinline__ float pk_make(float d, int sum) {
    const uint32_t b = __float_as_uint(d) + 0x800u;
    return __uint_as_float(((b >> 12) << 13) | (uint32_t)(sum + 4096));
}
// Capability markers (empty kernels the host looks up by name, see
// kernels::Caps in include/whirl/kernels_abi.h):
//   whirl_cap_xd_sum      the int8 activation scale words are pk_make words
//                         (scale + block sum), not plain f32 scales
//   whirl_cap_attn_group1 attn_wsplit1 query groups must hold one query (the
//                         gfx11 WMMA P.V is not bitwise invariant otherwise)
extern "C" __global__ void whirl_cap_xd_sum() {}
extern "C" __global__ void whirl_cap_attn_group1() {}

__device__ __forceinline__ int warp_sum_i(int v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, WARP);
    return v;
}

// Block-wide sum for blockDim.x <= 1024 (multiple of 32).
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + WARP - 1) / WARP;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ float block_max(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + WARP - 1) / WARP;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : -INFINITY;
    if (wid == 0) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
