// WHIRL device code (gfx1201 / RDNA 4): prefill activation fusions producing GEMM inputs (x8 / x16 variants).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Item P: prefill activation fusions. Each kernel writes the GEMM input (x16:
// fp8 e4m3 + per-token scale, or f16) directly instead of an f32 tensor that the
// next matmul converts. The arithmetic is the unfused one step for step
// (rmsnorm / silu_mul, then qact_fp8 or f32_to_f16), so the bytes are identical.
// One 256-thread block per token row.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t fp8_of(float v, float inv) {
    const float a = fminf(fmaxf(v * inv, -448.f), 448.f);
    return (uint8_t)(__builtin_amdgcn_cvt_pk_fp8_f32(a, 0.f, 0, false) & 0xFF);
}

// rmsnorm(x[t]) * w -> fp8 (FP8) or f16 into q[t]; n <= 256 * RN_PER.
// H16 (with FP8): also writes the same normed row as f16 into q16[t] (the GDN
// prefill needs fp8 for qkv / gate and f16 for [beta; alpha] from one norm).
#define RN_PER 32
template <bool FP8, bool TL = false, bool H16 = false>
__device__ void rmsnorm_x_impl(const float *x, const float *w, uint8_t *q, float *sx, int n, float eps,
                               _Float16 *q16 = nullptr) {
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
                if (H16) q16[(size_t)blockIdx.x * n + i] = (_Float16)v[k];
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
extern "C" __global__ void __launch_bounds__(256) rmsnorm_x8(const float *x, const float *w, uint8_t *q, float *sx, int n, float eps) {
    rmsnorm_x_impl<true>(x, w, q, sx, n, eps);
}
extern "C" __global__ void __launch_bounds__(256) rmsnorm_x16(const float *x, const float *w, uint8_t *q, float *sx, int n, float eps) {
    rmsnorm_x_impl<false>(x, w, q, sx, n, eps);
}
// one norm, two outputs: fp8 + sx (row-major / tiled) and f16 (row-major)
extern "C" __global__ void __launch_bounds__(256) rmsnorm_x8h16(const float *x, const float *w, uint8_t *q, float *sx, _Float16 *q16, int n, float eps) {
    rmsnorm_x_impl<true, false, true>(x, w, q, sx, n, eps, q16);
}
extern "C" __global__ void __launch_bounds__(256) rmsnorm_x8h16t(const float *x, const float *w, uint8_t *q, float *sx, _Float16 *q16, int n, float eps) {
    rmsnorm_x_impl<true, true, true>(x, w, q, sx, n, eps, q16);
}

// silu(g[t]) * u[t] -> fp8 / f16 into q[t]; ncols % 4 == 0, ncols <= 1024 * SM_PER4.
#include "shared/prefill_fused_sm_per4.h"  // also in the gfx1151 set (kernels/gfx1151/prefill_fused.hip)
extern "C" __global__ void __launch_bounds__(256) silu_mul_x8(const float *g, const float *u, uint8_t *q, float *sx, int ncols) {
    silu_mul_x_impl<true>(g, u, q, sx, ncols);
}
#include "shared/prefill_fused_silu_mul_x16.h"  // also in the gfx1151 set (kernels/gfx1151/prefill_fused.hip)
extern "C" __global__ void __launch_bounds__(256) gated_norm_x8(const float *o, const float *z, const float *w, uint8_t *q, float *sx, int n_heads, float eps) {
    gated_norm_x_impl<true>(o, z, w, q, sx, n_heads, eps);
}
// z as f16 (item P: MXFP4 speed mode writes the z GEMM output as f16)
extern "C" __global__ void __launch_bounds__(256) gated_norm_x8h(const float *o, const _Float16 *z, const float *w, uint8_t *q, float *sx, int n_heads, float eps) {
    gated_norm_x_impl<true, _Float16>(o, z, w, q, sx, n_heads, eps);
}
#include "shared/prefill_fused_gated_norm_x16.h"  // also in the gfx1151 set (kernels/gfx1151/prefill_fused.hip)

