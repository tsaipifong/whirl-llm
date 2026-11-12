// WHIRL device code (gfx1201 / RDNA 4 only): multi-block greedy argmax of verify rows, with the F1 draft rank
// (v0252-f1arg, PLAN V1-63 / V1-64; DISCOVERY C6 port to gfx1201 plus the rank twin).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by whirl_kernels.hip at the end of the gfx1201 set (needs WARP from common.hip and
// RowIdx from kernel_args.h). The gfx1151 set does not get these kernels (its argmax path is verified separately).
//
// argmax_rows_to / argmax_rank_rows_to (kernels/shared/sample_pick_set_tokens.h) run ONE 1024-thread block per row:
// 3-4 verify rows x 248,320 logits keep 3-4 of the R9700's 64 CUs busy and each wave keeps one 128-byte load in
// flight, so they are latency bound: 72-148 us per call for 1-4 rows (RGP int0252 spans read with the parser before its
// PLAN V1-68 pairing fix; research/f1arg/2026-11-10-r9700-argmax-rank.md), against ~5 us to read 3 rows at 640 GB/s.
// Here grid = (AM_TILES(n), rows), AM_THREADS (256) threads, both from argmax_key.h (the one source every launcher uses;
// v0252-f1arg2): each block scans AM_TILE values of one row with AM_V4 independent float4 loads per thread, folds them
// into an argmax key (argmax_key.h) and (rank twin) counts the entries ahead of the row's draft, combines into the row's
// workspace with atomics, and takes a ticket (CUDA threadFenceReduction pattern). The block that draws the row's last
// ticket writes the outputs and resets the row's workspace for the next launch. A grid with fewer blocks is still exact
// (the remaining tiles in a slow loop), so a launcher's grid mistake costs time, never a wrong token.
// Results are bitwise those of argmax_rows_to / argmax_rank_rows_to for every input (ties -> lowest index, NaN never
// wins, -0 == +0, all -inf / NaN -> 0; rank = the same integer count; row max = x[index] or -inf; l_d = x[d] or -inf):
// tests/kernels/test_misc.cpp, tests/unit/kernels_sample_tests.cpp.
//
// Prior art (progress\v0252-f1arg.PRIOR_ART.md): strata src/kernels/cuda/verify_kernels.cu:880-951 (up to 128 blocks a
// row, the row's last block merges, on by default for gfx12); PyTorch Reduce.cuh global_reduce (ctas_per_output,
// last CTA done); gufo qwen/hip/kernels/sample.hip:968-1056 (4096-value tiles, 2 launches, MIT); llama.cpp
// ggml-cuda/top-k.cu (64 blocks a row for the radix top-k).

#include "kernel_args.h"  // RowIdx
#include "argmax_key.h"   // keys, AmWs and the launch geometry AM_TILE / AM_THREADS / AM_TILES shared with every launcher

#define AM_V4 (AM_TILE / (4 * AM_THREADS))  // float4 loads per thread in a full tile (4)
static_assert(AM_TILE % (4 * AM_THREADS) == 0 && AM_V4 >= 1, "a full tile is AM_V4 float4 loads per thread");
static_assert(AM_MAX_ROWS == (int)whirl::kernels::abi::kMaxVerifyRows, "AmWs rows = RowIdx rows = kMaxVerifyRows");

__device__ __forceinline__ am_key_t am_shfl_xor(am_key_t k, int o) {
    const uint32_t lo = __shfl_xor((uint32_t)(k & 0xFFFFFFFFull), o, WARP);
    const uint32_t hi = __shfl_xor((uint32_t)(k >> 32), o, WARP);
    return ((am_key_t)hi << 32) | lo;
}

// element (v, i) into the thread's key and (RANK) its count of entries ahead of the draft (d, ld)
template <bool RANK>
__device__ __forceinline__ void am_take(float v, int i, int d, float ld, am_key_t &best, int &cnt) {
    best = am_max(best, am_key(v, i));
    if (RANK) cnt += am_ahead(v, i, ld, d);
}

// a full 16-byte-aligned tile already loaded (v[j] = the float4 at element beg + 4 * (tid + j * AM_THREADS))
template <bool RANK>
__device__ __forceinline__ void am_vec(const float4 (&v)[AM_V4], int beg, int d, float ld, am_key_t &best, int &cnt) {
#pragma unroll
    for (int j = 0; j < AM_V4; ++j) {
        const int i = beg + 4 * (int)(threadIdx.x + j * AM_THREADS);
        am_take<RANK>(v[j].x, i, d, ld, best, cnt);
        am_take<RANK>(v[j].y, i + 1, d, ld, best, cnt);
        am_take<RANK>(v[j].z, i + 2, d, ld, best, cnt);
        am_take<RANK>(v[j].w, i + 3, d, ld, best, cnt);
    }
}

// a partial or unaligned tile [beg, end) of row xr, scalar strided loads (the row's last tile at n = 248320)
template <bool RANK>
__device__ __forceinline__ void am_scalar(const float *__restrict__ xr, int beg, int end, int d, float ld, am_key_t &best, int &cnt) {
    for (int i = beg + (int)threadIdx.x; i < end; i += AM_THREADS) am_take<RANK>(xr[i], i, d, ld, best, cnt);
}

// block max of the keys and (RANK) block sum of the counts; valid in thread 0
template <bool RANK>
__device__ __forceinline__ void am_block(am_key_t &k, int &c) {
    __shared__ am_key_t am_sk[AM_THREADS / WARP];
    __shared__ int am_sc[AM_THREADS / WARP];
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) {
        k = am_max(k, am_shfl_xor(k, o));
        if (RANK) c += __shfl_xor(c, o, WARP);
    }
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    if (lane == 0) {
        am_sk[wid] = k;
        if (RANK) am_sc[wid] = c;
    }
    __syncthreads();
    if (wid == 0) {
        k = lane < AM_THREADS / WARP ? am_sk[lane] : 0ull;
        if (RANK) c = lane < AM_THREADS / WARP ? am_sc[lane] : 0;
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) {
            k = am_max(k, am_shfl_xor(k, o));
            if (RANK) c += __shfl_xor(c, o, WARP);
        }
    }
}

__device__ __forceinline__ am_key_t am_load(const am_key_t *p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Common body. Row = blockIdx.y, tile = blockIdx.x; gridDim.x = AM_TILES(n) (argmax_key.h, every launcher), gridDim.y =
// rows (<= AM_MAX_ROWS), n >= 1. Any other gridDim.x >= 1 gives the same bits (REVIEW-v0252-f1arg C1, CUB's strip-mine
// GridEvenShare): with fewer blocks each block also scans tiles blockIdx.x + k * gridDim.x in a scalar loop, with more
// blocks the extra ones find nothing (key 0, count 0).
// RANK: si = src.v[row] >= 0 names the row's draft word out[si] (argmax_rank_rows_to's contract: src and dst words
// never overlap, so the last block's out[dst] store cannot race another block's out[src] read; static_assert in
// include/whirl/kernels_abi.h for the control area the host uses).
template <bool RANK>
__device__ __forceinline__ void am_rows(const float *__restrict__ x, int n, int *out, const RowIdx &dst, const RowIdx &src, int *stats,
                                        int stride, AmWs *ws) {
    const int row = blockIdx.y;
    const float *__restrict__ xr = x + (size_t)row * n;
    const int beg = blockIdx.x * AM_TILE;
    const int end = min(n, beg + AM_TILE);
    // full 16-byte-aligned tile: issue all AM_V4 float4 loads first, so the draft reads below overlap them
    const bool full = end - beg == AM_TILE && (reinterpret_cast<uintptr_t>(xr + beg) & 15u) == 0;
    float4 v[AM_V4] = {};  // read only when full (the initializer is dead there and folds away)
    if (full) {
        const float4 *xv = reinterpret_cast<const float4 *>(xr + beg);
#pragma unroll
        for (int j = 0; j < AM_V4; ++j) v[j] = xv[threadIdx.x + j * AM_THREADS];
    }
    int d = -1;
    float ld = -INFINITY;
    bool has = false;
    if (RANK) {  // the row's draft: its id from the control area and its logit (uniform per block)
        const int si = src.v[row];
        d = si >= 0 ? out[si] : -1;
        has = d >= 0 && d < n;
        ld = has ? xr[d] : -INFINITY;
    }
    am_key_t k = 0ull;
    int c = 0;
    if (full) {
        if (RANK && has)
            am_vec<true>(v, beg, d, ld, k, c);
        else
            am_vec<false>(v, beg, d, ld, k, c);
    } else {
        if (RANK && has)
            am_scalar<true>(xr, beg, end, d, ld, k, c);
        else
            am_scalar<false>(xr, beg, end, d, ld, k, c);
    }
    // a grid smaller than AM_TILES(n): the tiles no block owns (never taken at the launchers' grid; uniform per block)
    for (int t = (int)(blockIdx.x + gridDim.x); t < AM_TILES(n); t += (int)gridDim.x) {
        if (RANK && has)
            am_scalar<true>(xr, t * AM_TILE, min(n, (t + 1) * AM_TILE), d, ld, k, c);
        else
            am_scalar<false>(xr, t * AM_TILE, min(n, (t + 1) * AM_TILE), d, ld, k, c);
    }
    am_block<RANK>(k, c);
    if (threadIdx.x != 0) return;
    atomicMax(ws->key + row, k);  // no return value used: posted, the fence below waits for it
    if (RANK && has && c != 0) atomicAdd(ws->rank + row, c);
    __threadfence();
    if (atomicAdd(ws->cnt + row, 1u) != gridDim.x - 1) return;
    // the row's last block: every other block's key / count is in (their fence came before their ticket)
    __threadfence();
    const am_key_t g = am_load(ws->key + row);
    const int idx = am_idx(g);
    out[dst.v[row]] = idx;
    if (RANK) {
        const int r = __hip_atomic_load(ws->rank + row, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        stats[row] = has ? r : 0x7fffffff;
        ((float *)stats)[stride + row] = (uint32_t)(g >> 32) == AM_ORD_NEG_INF ? -INFINITY : xr[idx];
        ((float *)stats)[2 * stride + row] = ld;
        ws->rank[row] = 0;
    }
    ws->key[row] = 0ull;
    ws->cnt[row] = 0u;
}

// = argmax_rows_to: out[dst.v[row]] = the first index of the maximum of row `row` of x ([rows][n]).
extern "C" __global__ void __launch_bounds__(AM_THREADS)
    argmax_rows_to_mb(const float *__restrict__ x, int n, int *out, RowIdx dst, AmWs *ws) {
    am_rows<false>(x, n, out, dst, dst, nullptr, 0, ws);
}

// = argmax_rank_rows_to: the argmax as above plus, for rows with a draft (src.v[row] >= 0, id out[src.v[row]] in [0, n)),
// stats[row] = its rank in the whole row (INT_MAX: no draft), ((float *)stats)[stride + row] = the row max,
// ((float *)stats)[2 * stride + row] = the draft's logit (-inf without a draft).
extern "C" __global__ void __launch_bounds__(AM_THREADS)
    argmax_rank_rows_to_mb(const float *__restrict__ x, int n, int *out, RowIdx dst, RowIdx src, int *stats, int stride, AmWs *ws) {
    am_rows<true>(x, n, out, dst, src, stats, stride, ws);
}
