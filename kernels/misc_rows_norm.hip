// WHIRL device code (gfx1201 / RDNA 4): embedding row lookup (get_rows_*), rmsnorm, l2norm, add, silu_mul, NEOX RoPE.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// Dequantize rows `ids[t]` of W into out[t][ncols] (embedding lookup).
#include "shared/misc_rows_norm_get_rows.h"  // also in the gfx1151 set (kernels/gfx1151/fused_decode.hip)

GET_ROWS(f32, QT_F32)
GET_ROWS(f16, QT_F16)
GET_ROWS(q8_0, QT_Q8_0)
GET_ROWS(q3_k, QT_Q3_K)
GET_ROWS(q4_k, QT_Q4_K)
GET_ROWS(q5_k, QT_Q5_K)
GET_ROWS(q6_k, QT_Q6_K)
GET_ROWS(iq4_nl, QT_IQ4_NL)
GET_ROWS(iq3_s, QT_IQ3_S)
GET_ROWS(iq4_xs, QT_IQ4_XS)
GET_ROWS(mxfp4, QT_MXFP4)

// ---------------------------------------------------------------------------
// Norms and elementwise ops. "Vector v" of a launch = blockIdx.x.
// ---------------------------------------------------------------------------

// out[v][i] = x[v][i] / sqrt(mean(x^2) + eps) * w[i]; vectors at x + v*stride.
extern "C" __global__ void rmsnorm(const float *x, const float *w, float *out, int n, int in_stride,
                                   int out_stride, float eps) {
    const float *xv = x + (size_t)blockIdx.x * in_stride;
    float *ov = out + (size_t)blockIdx.x * out_stride;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xv[i] * xv[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) ov[i] = xv[i] * r * w[i];
}

// In-place L2 norm: x / sqrt(sum x^2 + eps) (llama.cpp build_gdn_l2_norm).
// Vector (blockIdx.x, blockIdx.y) at x + y*tok_stride + x*stride.
#include "shared/misc_rows_norm_l2norm.h"  // also in the gfx1151 set (kernels/gfx1151/misc_rows_norm.hip)

