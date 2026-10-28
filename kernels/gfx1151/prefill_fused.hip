// WHIRL device code (gfx1151 / RDNA 3.5): prefill activation fusions (f16).
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ===== prefill activation fusions from the gfx1201 source (f16 only) =====
// ---------------------------------------------------------------------------
// Item P: prefill activation fusions. Each kernel writes the GEMM input (x16:
// fp8 e4m3 + per-token scale, or f16) directly instead of an f32 tensor that the
// next matmul converts. The arithmetic is the unfused one step for step
// (rmsnorm / silu_mul, then qact_fp8 or f32_to_f16), so the bytes are identical.
// One 256-thread block per token row.
// ---------------------------------------------------------------------------
// gfx1151 has no fp8 conversion; only the f16 (FP8 = false) instances exist here
__device__ __forceinline__ uint8_t fp8_of(float, float) { return 0; }

// rmsnorm(x[t]) * w -> fp8 (FP8) or f16 into q[t]; n <= 256 * RN_PER.
#define RN_PER 32
template <bool FP8, bool TL = false>
__device__ void rmsnorm_x_impl(const float *x, const float *w, uint8_t *q, float *sx, int n, float eps) {
    const float *xv = x + (size_t)blockIdx.x * n;
    const int tid = threadIdx.x;
    float v[RN_PER];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < RN_PER; ++k) {
        const int i = tid + k * 256;
        v[k] = 0.f;
        if (i < n) {
            v[k] = xv[i];
            ss += v[k] * v[k];
        }
    }
    ss = block_sum(ss);
    const float r = rsqrtf(ss / n + eps);
    float m = 0.f;
#pragma unroll
    for (int k = 0; k < RN_PER; ++k) {
        const int i = tid + k * 256;
        if (i < n) {
            v[k] = v[k] * r * w[i];
            asm volatile("" : "+v"(v[k]));
            m = fmaxf(m, fabsf(v[k]));
        }
    }
    if (FP8) {
        m = block_max(m);
        const float inv = m > 0.f ? 448.f / m : 0.f;
        if (tid == 0) sx[blockIdx.x] = m > 0.f ? m / 448.f : 0.f;
        uint8_t *qr = q + (size_t)blockIdx.x * n;
#pragma unroll
        for (int k = 0; k < RN_PER; ++k) {
            const int i = tid + k * 256;
            if (i < n) {
                if (TL) q[x8t_off(blockIdx.x, i, n)] = fp8_of(v[k], inv);
                else qr[i] = fp8_of(v[k], inv);
            }
        }
    } else {
        _Float16 *qr = (_Float16 *)q + (size_t)blockIdx.x * n;
#pragma unroll
        for (int k = 0; k < RN_PER; ++k) {
            const int i = tid + k * 256;
            if (i < n) qr[i] = (_Float16)v[k];
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) rmsnorm_x16(const float *x, const float *w, uint8_t *q, float *sx, int n, float eps) {
    rmsnorm_x_impl<false>(x, w, q, sx, n, eps);
}

// silu(g[t]) * u[t] -> fp8 / f16 into q[t]; ncols % 4 == 0, ncols <= 1024 * SM_PER4.
#include "shared/prefill_fused_sm_per4.h"  // also in the gfx1201 set (kernels/prefill_fused.hip)
#include "shared/prefill_fused_silu_mul_x16.h"  // also in the gfx1201 set (kernels/prefill_fused.hip)
#include "shared/prefill_fused_gated_norm_x16.h"  // also in the gfx1201 set (kernels/prefill_fused.hip)
