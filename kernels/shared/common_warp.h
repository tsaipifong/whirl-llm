// WHIRL device code, shared by both kernel sets (gfx1201 kernels/common.hip, gfx1151 kernels/gfx1151/common.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

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
