// WHIRL device code (gfx1151 / RDNA 3.5, Radeon 8060S): moeskip (fast item; v0252-f5, ported from feat027-moeskip-r1 4b6474b
// as NEW entry points; included LAST by kernels/gfx1151/whirl_kernels_gfx1151.hip, so no existing kernel moves or changes).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: uses common.hip (WARP, warp_sum, block_sum), the generic QDN decoders, shared/moe_gu_pair.h.
//
// The shared entry points (routing incl. moe_skip_apply after the fused router, grouping, combine, generic MXFP4 decode
// experts): shared/moe_skip.h. Here: the twins of the 8060S prefill's moe_tiles (the 64-token / moep gate + up lists) and
// moe_combine4 (the moep default combine), with the dropped pairs (id -1 / inv -1) left out.

#include "shared/moe_skip.h"

// moe_tiles over the kept pairs only (the expert counts of moe_route_skip, so the positions match its perm). One 1024-thread
// block; R <= 1024.
extern "C" __global__ void __launch_bounds__(1024)
    moe_tiles_skip(const int *__restrict__ ids, int n_pairs, int R, int bn, int4 *__restrict__ tiles, int *__restrict__ n_tiles) {
    __shared__ int cnt[1024];
    for (int r = threadIdx.x; r < R; r += blockDim.x) cnt[r] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < n_pairs; p += blockDim.x)
        if (ids[p] >= 0) atomicAdd(&cnt[ids[p]], 1);  // moeskip: id -1 pairs are in no tile
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, nt = 0;
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < cnt[r]; s += bn) tiles[nt++] = make_int4(r, o + s, min(bn, cnt[r] - s), 0);
            o += cnt[r];
        }
        *n_tiles = nt;
    }
}

// moe_combine4 with the dropped slots (inv -1) left out; per element the moe_combine_skip expression and order.
// grid (C / 1024, n), C % 4 == 0.
extern "C" __global__ void __launch_bounds__(256)
    moe_combine4_skip(float *__restrict__ x, const float *__restrict__ ysh, const float *__restrict__ sg, const float *__restrict__ yd,
                      const int *__restrict__ inv, const float *__restrict__ w, int C, int K) {
    const int t = blockIdx.y;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (c >= C) return;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < K; ++k) {
        const int q = inv[(size_t)t * K + k];
        if (q < 0) continue;  // moeskip: a dropped slot
        const float wk = w[(size_t)t * K + k];
        const float4 v = *(const float4 *)(yd + (size_t)q * C + c);
        acc.x += wk * v.x;
        acc.y += wk * v.y;
        acc.z += wk * v.z;
        acc.w += wk * v.w;
    }
    if (ysh != nullptr) {
        const float s = sg[t];
        const float4 v = *(const float4 *)(ysh + (size_t)t * C + c);
        acc.x += s * v.x;
        acc.y += s * v.y;
        acc.z += s * v.z;
        acc.w += s * v.w;
    }
    float4 *xo = (float4 *)(x + (size_t)t * C + c);
    float4 xv = *xo;
    xv.x += acc.x;
    xv.y += acc.y;
    xv.z += acc.z;
    xv.w += acc.w;
    *xo = xv;
}
