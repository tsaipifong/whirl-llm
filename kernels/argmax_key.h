// WHIRL device code (gfx1201 set) + host: order-independent greedy argmax keys and the multi-block argmax workspace
// (v0252-f1arg; the key is feat026-argmax's kernels/argmax_key.h from feat026-argmax-p25 f4fa25b, DISCOVERY C6).
// SPDX-License-Identifier: Apache-2.0
//
// A 64-bit key whose unsigned order is the legacy argmax order (kernels/shared/block_argmax.h: best = -inf, idx = 0,
// x[i] > best, the tree takes the lower index on equal values): a larger value wins, and among equal values the lower
// index wins (the first index of the maximum). max is associative and commutative, so blocks may combine keys with
// atomicMax in any order and the result is the same bits. Matching the legacy scan exactly:
//   * NaN never wins a comparison there, so it is mapped to -inf here;
//   * -0 == +0 there, so -0 is mapped to +0 (equal keys, lower index wins);
//   * a row that is all -inf / NaN gives index 0 there; here every element maps to (-inf, i) and the largest key is
//     (-inf, 0), so index 0 too.
// Integer-only (no float compares), so -ffast-math / -ffinite-math cannot fold the NaN test away.
// The key never carries the sign of a zero maximum: the row max a kernel reports is read back from x[index] (the
// legacy kernels report x[first index of the maximum], or -inf when no element is above -inf: AM_ORD_NEG_INF).
// am_key / am_idx / am_max / AmWs / the AM_* geometry also compile on the host (tests/unit/kernels_sample_tests.cpp checks
// them against the legacy scans with random block partitions and combine orders); the rest is device code.
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

// high word of the key of -inf (and of NaN): the row's maximum is -inf (no element above -inf)
#define AM_ORD_NEG_INF 0x007FFFFFu

// Launch geometry of the multi-block twins (kernels/sample_argmax_mb.hip), the one source for the kernel and every
// launcher (kt misc / bench, the host switch; REVIEW-v0252-f1arg C1): grid (AM_TILES(n), rows), AM_THREADS threads, one
// AM_TILE-value tile per block. The kernel is exact for any gridDim.x >= 1 (fewer blocks also scan the remaining tiles in
// a slow loop, extra blocks find nothing), so AM_TILES(n) is the fast grid, not a correctness condition.
#define AM_TILE 4096
#define AM_THREADS 256
#define AM_TILES(n) (((n) + AM_TILE - 1) / AM_TILE)
// rows one launch may have = kMaxVerifyRows (include/whirl/kernels_abi_layout.h) = RowIdx's 32 (static_assert in the kernel)
#define AM_MAX_ROWS 32

// Workspace of one multi-block argmax launch (rows <= AM_MAX_ROWS): zeroed once at allocation; every launch leaves it
// zeroed (the last block of each row resets its row). One workspace per stream: two launches that may run at the same
// time must not share one.
struct AmWs {
    am_key_t key[AM_MAX_ROWS];  // per row: max of the blocks' keys (atomicMax)
    unsigned cnt[AM_MAX_ROWS];  // per row: blocks done (ticket)
    int rank[AM_MAX_ROWS];      // per row: argmax_rank_rows_to_mb, entries ahead of the draft (atomicAdd)
};
#define AM_WS_BYTES 512
#if defined(__cplusplus) && __cplusplus >= 201103L
static_assert(sizeof(AmWs) == AM_WS_BYTES, "AmWs layout");
#endif

WHIRL_AM_FN am_key_t am_key(float v, int i) {
    uint32_t u = WHIRL_AM_BITS(v);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) u = 0xFF800000u;  // NaN -> -inf
    if (u == 0x80000000u) u = 0u;                          // -0 -> +0
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);         // float order -> unsigned order
    return ((am_key_t)u << 32) | (am_key_t)(0xFFFFFFFFu - (uint32_t)i);
}

WHIRL_AM_FN int am_idx(am_key_t k) { return (int)(0xFFFFFFFFu - (uint32_t)(k & 0xFFFFFFFFull)); }

WHIRL_AM_FN am_key_t am_max(am_key_t a, am_key_t b) { return a > b ? a : b; }

// the F1 rank predicate (argmax_rank_rows_to, kernels/shared/sample_pick_set_tokens.h): element (v, i) is ahead of the
// draft (ld = its logit, d = its id) when it has a greater logit, or an equal logit and a lower id; float compares as
// there (a NaN element is never ahead; a NaN draft logit has nothing ahead)
WHIRL_AM_FN int am_ahead(float v, int i, float ld, int d) { return (v > ld || (v == ld && i < d)) ? 1 : 0; }

#endif  // WHIRL_ARGMAX_KEY_H
