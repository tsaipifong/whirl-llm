// WHIRL device code, shared by both kernel sets (gfx1201 kernels/fused_decode.hip, gfx1151 kernels/gfx1151/fused_decode.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

extern "C" __global__ void __launch_bounds__(1024)
    rmsnorm_q8(const float *x, const float *w, float *out, int8_t *xq, float *xd, int n, float eps) {
    // row blockIdx.x (small multi-token batches: speculative verify)
    x += (size_t)blockIdx.x * n;
    out += (size_t)blockIdx.x * n;
    xq += (size_t)blockIdx.x * n;
    xd += (size_t)blockIdx.x * (n / 32);
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / n + eps);
    const int wid = threadIdx.x / WARP, lane = threadIdx.x % WARP, nw = blockDim.x / WARP;
    for (int b = wid; b < n / 32; b += nw) {
        const int i = b * 32 + lane;
        const float v = x[i] * r * w[i];
        out[i] = v;
        wave_q8(v, xq, xd, b);
    }
}

// g = silu(g) * u and its int8 copy. One wave per 32-block.
extern "C" __global__ void silu_mul_q8(float *g, const float *u, int8_t *xq, float *xd, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float a = g[i];
    const float v = a / (1.f + __expf(-a)) * u[i];
    g[i] = v;
    wave_q8(v, xq, xd, i >> 5);
}

// beta = sigmoid(Wb x), g = softplus(Wa x + dt) * A for n_heads heads, from
// int8 x. One wave per row; rows [0, nh) are beta, [nh, 2nh) alpha.
template <int T>
__device__ void gdn_ab_impl(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq,
                            const float *xd, float *beta, float *g, const float *dt, const float *A, int ncols,
                            int nh, int blk) {
    const int lane = threadIdx.x % WARP;
    const int row = blk * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= 2 * nh) return;
    const bool is_b = row < nh;
    const int h = is_b ? row : row - nh;
    const uint8_t *wr = (is_b ? Wb : Wa) + (size_t)h * row_bytes;
    const int nu = ncols / QD<T>::sb * QD<T>::per_sb;
    float acc = 0.f;
    for (int u = lane; u < nu; u += WARP) acc += QD<T>::dot(wr, u, xq, xd);
    acc = warp_sum(acc);
    if (lane == 0) {
        if (is_b) {
            beta[h] = 1.f / (1.f + __expf(-acc));
        } else {
            const float z = acc + dt[h];
            g[h] = (z > 20.f ? z : log1pf(__expf(z))) * A[h];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256)
    gdn_ab_q8_0(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq, const float *xd,
                float *beta, float *g, const float *dt, const float *A, int ncols, int nh) {
    const int t = blockIdx.y;  // token
    gdn_ab_impl<QT_Q8_0>(Wb, Wa, row_bytes, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32), beta + (size_t)t * nh,
                         g + (size_t)t * nh, dt, A, ncols, nh, blockIdx.x);
}

// Recurrent-state segments for the fused DeltaNet decode kernels: rows
// [row0, row0 + nrows) of the batch belong to one sequence whose state lives
// at `state`; snap[t] (if non-null) receives the state right after local row t
// (speculative verify rollback). Passed by value (kernel args, no host copy).
#include "kernel_args.h"  // GdnSeg (defined once for both kernel sets, arch025 N13)
