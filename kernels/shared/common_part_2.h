// WHIRL device code, shared by both kernel sets (gfx1201 kernels/common.hip, gfx1151 kernels/gfx1151/common.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

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
