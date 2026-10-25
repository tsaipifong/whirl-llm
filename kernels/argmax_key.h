// WHIRL device code (both kernel sets): order-independent greedy argmax keys (feat026-argmax).
// SPDX-License-Identifier: Apache-2.0
//
// A 64-bit key whose unsigned order is the legacy argmax_rows order: a larger value wins, and among
// equal values the lower index wins (the first index of the maximum). Because max is associative and
// commutative, blocks may combine keys with atomicMax in any order and the result is the same bits.
// Matching the legacy scan exactly (sample_pick.hip argmax_rows: best = -inf, idx = 0, x[i] > best):
//   * NaN never wins a comparison there, so it is mapped to -inf here;
//   * -0 == +0 there, so -0 is mapped to +0 (equal keys, lower index wins);
//   * a row that is all -inf / NaN gives index 0 there; here every element maps to (-inf, i) and
//     the largest key is (-inf, 0), so index 0 too.
// Integer-only (no float compares), so -ffast-math / -ffinite-math cannot fold the NaN test away.
// am_key / am_idx / am_max also compile on the host (tests/unit/unit_tests.cpp checks them against the
// legacy scan with random block partitions and combine orders); the rest is device code.
#ifndef WHIRL_ARGMAX_KEY_H
#define WHIRL_ARGMAX_KEY_H

#include <stdint.h>
#if defined(__HIP__) || defined(__HIPCC__)
#define WHIRL_AM_FN __device__ __forceinline__
#define WHIRL_AM_BITS(v) __float_as_uint(v)
#else
#include <string.h>
#define WHIRL_AM_FN inline
inline uint32_t whirl_am_bits(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    return u;
}
#define WHIRL_AM_BITS(v) whirl_am_bits(v)
#endif

typedef unsigned long long am_key_t;

// Destination slot per row: the same layout as RowIdx (sample_pick.hip / draft_pick.hip, kernels_abi.h).
struct AmRowIdx {
    int v[32];
};

WHIRL_AM_FN am_key_t am_key(float v, int i) {
    uint32_t u = WHIRL_AM_BITS(v);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) u = 0xFF800000u;  // NaN -> -inf
    if (u == 0x80000000u) u = 0u;                          // -0 -> +0
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);         // float order -> unsigned order
    return ((am_key_t)u << 32) | (am_key_t)(0xFFFFFFFFu - (uint32_t)i);
}

WHIRL_AM_FN int am_idx(am_key_t k) { return (int)(0xFFFFFFFFu - (uint32_t)(k & 0xFFFFFFFFull)); }

WHIRL_AM_FN am_key_t am_max(am_key_t a, am_key_t b) { return a > b ? a : b; }

#if defined(__HIP__) || defined(__HIPCC__)

__device__ __forceinline__ am_key_t am_shfl_xor(am_key_t k, int o) {
    const uint32_t lo = __shfl_xor((uint32_t)(k & 0xFFFFFFFFull), o, WARP);
    const uint32_t hi = __shfl_xor((uint32_t)(k >> 32), o, WARP);
    return ((am_key_t)hi << 32) | lo;
}

__device__ __forceinline__ am_key_t am_load(const am_key_t *p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Combine a block's key into the device-wide key (skips the atomic when it cannot win).
__device__ __forceinline__ void am_publish(am_key_t *p, am_key_t k) {
    if (k > am_load(p)) atomicMax(p, k);
}

// Fused-argmax epilogue arguments of a GEMV launch (one launch computes all NT token rows of logits):
// key[t] (t < NT) and *cnt are zeroed workspace; the block with the last ticket writes out[dst.v[t]].
struct AmArgs {
    am_key_t *key;
    unsigned *cnt;
    int *out;
    AmRowIdx dst;
};

// Called by every block of the GEMV after its logits stores; ks[t][r] = key of the block's row r of
// token t (0 for rows past nrows). blockDim.x >= NT. Workspace is reset by the last block.
template <int NT, int ROWS>
__device__ __forceinline__ void am_gemv_finish(const am_key_t (&ks)[NT][ROWS], const AmArgs &am) {
    __shared__ int am_last_g;
    __syncthreads();
    if (threadIdx.x < NT) {
        am_key_t k = 0ull;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) k = am_max(k, ks[threadIdx.x][r]);
        am_publish(am.key + threadIdx.x, k);
        __threadfence();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned t = atomicAdd(am.cnt, 1u);
        am_last_g = t == gridDim.x - 1;
    }
    __syncthreads();
    if (am_last_g) {
        if (threadIdx.x < NT) {
            __threadfence();
            const am_key_t k = am_load(am.key + threadIdx.x);
            am.out[am.dst.v[threadIdx.x]] = am_idx(k);
            am.key[threadIdx.x] = 0ull;
        }
        if (threadIdx.x == 0) am.cnt[0] = 0u;
    }
}

#endif  // __HIP__

#endif  // WHIRL_ARGMAX_KEY_H
