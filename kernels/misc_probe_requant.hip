// WHIRL device code (gfx1201 / RDNA 4): WMMA peak-throughput probes, Q6_K -> Q4_K requantization (MTP head).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Peak-throughput probes: back-to-back independent WMMAs on registers only.
// 8 accumulators per wave so the matrix pipe never waits on a dependency.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) peak_wmma_f32acc(float *out, int iters) {
    half8 a, b;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = (_Float16)(threadIdx.x * 0.001f + i);
        b[i] = (_Float16)(i * 0.5f);
    }
    float8 c[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c[j]);
    }
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += c[j][0];
    if (sum == 12345.f) out[0] = sum;
}

extern "C" __global__ void __launch_bounds__(256) peak_wmma_f16acc(float *out, int iters) {
    half8 a, b;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = (_Float16)(threadIdx.x * 0.001f + i);
        b[i] = (_Float16)(i * 0.5f);
    }
    half8 c[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) c[j][i] = (_Float16)0.f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, c[j]);
    }
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += (float)c[j][0];
    if (sum == 12345.f) out[0] = sum;
}

typedef int int2v __attribute__((ext_vector_type(2)));
typedef int int8v __attribute__((ext_vector_type(8)));
extern "C" __global__ void __launch_bounds__(256) peak_wmma_i8(float *out, int iters) {
    int2v a = {(int)threadIdx.x, 3}, b = {5, (int)threadIdx.x};
    int8v c[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j] = (int8v){0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c[j], false);
    }
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += c[j][0];
    if (sum == 12345) out[0] = (float)sum;
}

// Requantize a Q6_K (or Q8_0) matrix to Q4_K (MTP block / draft head: drafts only, the
// verify stays exact). Block = one 256-value superblock, thread = value, wave = 32-value
// sub-block. Plain min/max fit per sub-block (min clamped <= 0), 6-bit scales/mins.
#include "shared/misc_probe_requant_requant_q6k_q4k.h"  // also in the gfx1151 set (kernels/gfx1151/misc_probe_requant.hip)

