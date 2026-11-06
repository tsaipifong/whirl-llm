// WHIRL device code (gfx1201 / RDNA 4): moeskip (fast item; v0252-f5, ported from feat027-moeskip-r1 4b6474b as NEW entry
// points; included LAST in the gfx1201 block of whirl_kernels.hip, so no existing kernel moves or changes).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: uses moe_mxfp4.hip (mxw_dot / mxw_scale), common.hip (WARP, warp_sum, block_sum), the generic QDN
// decoders, shared/moe_gu_pair.h.
//
// The shared entry points (routing, grouping, combine, generic MXFP4 decode experts): shared/moe_skip.h. Here: the twins of
// the whole-block MXFP4 decode experts moe_gu_mxfp4w / moe_down_mxfp4w (the R9700 fast decode / verify path) - the same
// per-row code; gate + up returns for a dropped pair (id -1) before forming a weight address, down sums the kept prefix only.

#include "shared/moe_skip.h"

extern "C" __global__ void __launch_bounds__(256)
    moe_gu_skip_mxfp4w(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, uint64_t row_bytes, int ff, int nrows,
                       const int8_t *__restrict__ xq, const float *__restrict__ xd, const int *__restrict__ ids, float *__restrict__ y1,
                       float *__restrict__ y2, int C, int K, const int *skip) {
    if (skip != nullptr && *skip) return;
    if (ids[blockIdx.y] < 0) return;  // moeskip: a dropped pair
#include "shared/moe_gu_pair.h"  // lane, row, p, t, e, second, r, wr (ADR-0005)
    const int8_t *xt = xq + (size_t)t * C;
    const float *dt = xd + (size_t)t * (C / 32);
    float acc = 0.f;
    for (int b = lane; b < C / 32; b += WARP) acc += mxw_scale(wr, b) * dt[b] * (float)mxw_dot(wr, b, xt);
    acc = warp_sum(acc);
    if (lane == 0) (second ? y2 : y1)[(size_t)p * ff + r] = acc;
}

extern "C" __global__ void __launch_bounds__(256)
    moe_down_skip_mxfp4w(const uint8_t *__restrict__ W, uint64_t row_bytes, int E, const int8_t *__restrict__ xq, const float *__restrict__ xd,
                         const int *__restrict__ ids, const float *__restrict__ w, const float *__restrict__ sg, const float *__restrict__ ysh,
                         float *__restrict__ x, int F, int K, const int *skip) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= E) return;
    const int t = blockIdx.y;
    const int nb = F / 32;
    const int kt = moe_kept(ids, t, K);  // moeskip: the kept prefix
    float acc = 0.f;
    for (int u = lane; u < kt * nb; u += WARP) {
        const int k = u / nb, b = u - k * nb;
        const int p = t * K + k;
        const uint8_t *wr = W + ((size_t)ids[p] * E + row) * row_bytes;
        acc += w[p] * (mxw_scale(wr, b) * xd[(size_t)p * nb + b] * (float)mxw_dot(wr, b, xq + (size_t)p * F));
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        float v = acc;
        if (ysh != nullptr) v += sg[t] * ysh[(size_t)t * E + row];
        x[(size_t)t * E + row] += v;
    }
}
