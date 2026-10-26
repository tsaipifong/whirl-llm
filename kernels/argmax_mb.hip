// WHIRL device code (both kernel sets): multi-block greedy argmax with last-block finish (feat026-argmax).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by whirl_kernels.hip after both kernel sets (needs WARP from common.hip).
//
// argmax_rows_to / argmax_rows run one 1024-thread block per row: 4 verify rows x 248k vocab use 4 CUs and
// each wave keeps one load in flight, so the kernel is latency bound (audit\PROF-8S.md 2.3: 1.5 ms per Swift
// 8060S verify cycle, 1.8%). Here grid = (tiles, rows): each 256-thread block scans AM_TILE values of one
// row with 4 independent float4 loads per thread, folds them into an argmax key (argmax_key.h), combines
// it into ws_key[row] with atomicMax, and takes a ticket on ws_cnt[row] (CUDA threadFenceReduction
// pattern). The block that draws the last ticket of the row writes out[dst.v[row]] and resets the key
// and the counter to 0 for the next launch (the workspace is zeroed once at allocation). Same index as
// argmax_rows / argmax_rows_to for every input (ties -> lowest index, NaN never wins, -0 == +0).
//
// Prior art: gufo qwen/hip/kernels/sample.hip:968-1056 (tiled 4096-value argmax, 2 launches, MIT);
// llama.cpp ggml-cuda/argmax.cu:8-67 (one block per row). progress\feat026-argmax.PRIOR_ART.md.

#include "argmax_key.h"

#define AM_TILE 4096
#define AM_THREADS 256

__device__ __forceinline__ am_key_t am_block_max(am_key_t k) {
    __shared__ am_key_t am_sh[AM_THREADS / WARP];
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) k = am_max(k, am_shfl_xor(k, o));
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    if (lane == 0) am_sh[wid] = k;
    __syncthreads();
    if (wid == 0) {
        k = lane < AM_THREADS / WARP ? am_sh[lane] : 0ull;
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) k = am_max(k, am_shfl_xor(k, o));
    }
    return k;  // valid in thread 0
}

// x: [rows][n] f32; out[dst.v[row]] = argmax of row; ws_key[rows], ws_cnt[rows]: zeroed workspace.
extern "C" __global__ void __launch_bounds__(AM_THREADS)
    argmax_rows_to_mb(const float *__restrict__ x, int n, int *__restrict__ out, AmRowIdx dst, am_key_t *ws_key, unsigned *ws_cnt) {
    const int row = blockIdx.y;
    const float *xr = x + (size_t)row * n;
    const int beg = blockIdx.x * AM_TILE;
    const int end = min(n, beg + AM_TILE);
    am_key_t best = 0ull;
    if ((n & 3) == 0 && end - beg == AM_TILE) {
        // full tile, 16-byte aligned rows: 4 float4 loads per thread, all issued before the compares
        float4 v[AM_TILE / (4 * AM_THREADS)];
#pragma unroll
        for (int j = 0; j < AM_TILE / (4 * AM_THREADS); ++j) v[j] = *(const float4 *)(xr + beg + 4 * (threadIdx.x + j * AM_THREADS));
#pragma unroll
        for (int j = 0; j < AM_TILE / (4 * AM_THREADS); ++j) {
            const int i = beg + 4 * (threadIdx.x + j * AM_THREADS);
            best = am_max(best, am_key(v[j].x, i));
            best = am_max(best, am_key(v[j].y, i + 1));
            best = am_max(best, am_key(v[j].z, i + 2));
            best = am_max(best, am_key(v[j].w, i + 3));
        }
    } else {
        for (int i = beg + threadIdx.x; i < end; i += AM_THREADS) best = am_max(best, am_key(xr[i], i));
    }
    best = am_block_max(best);
    __shared__ int am_last;
    if (threadIdx.x == 0) {
        am_publish(ws_key + row, best);
        __threadfence();
        const unsigned t = atomicAdd(ws_cnt + row, 1u);
        am_last = t == gridDim.x - 1;
    }
    __syncthreads();
    if (am_last && threadIdx.x == 0) {
        __threadfence();
        const am_key_t k = am_load(ws_key + row);
        out[dst.v[row]] = am_idx(k);
        ws_key[row] = 0ull;
        ws_cnt[row] = 0u;
    }
}
