// WHIRL device code (gfx1151 / RDNA 3.5): norms, RoPE, elementwise.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

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
#include "shared/misc_rows_norm_l2norm.h"  // also in the gfx1201 set (kernels/misc_rows_norm.hip)
