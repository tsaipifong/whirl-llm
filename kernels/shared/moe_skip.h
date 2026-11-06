// WHIRL device code, shared by both kernel sets: moeskip (fast mode) - the routing, prefill grouping / combine and generic
// MXFP4 decode-expert ENTRY POINTS of v0252-f5 (ported from feat027-moeskip-r1 4b6474b; there the existing moe_gu / moe_down
// / moe_route / moe_tiles / moe_combine were edited in place: here every existing kernel is left byte for byte as it is,
// DISCOVERY v4 rule 3.0-1, and the fast path launches these twins instead; host reference include/whirl/moe_skip.h).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by kernels/moe_skip.hip (gfx1201, LAST in its set) and kernels/gfx1151/moe_skip.hip (gfx1151,
// LAST in its set); uses WARP, warp_sum, block_sum, QD / QDN, the moe_gu_pair.h prologue.
//
// Contract (every consumer below): a dropped (token, slot) pair has id -1 and weight 0; the kept slots of a token are a
// prefix of its K slots (the router's selection order, weights non-increasing); the kept weights are renormalized.

// The rule on one token's K slots in place (moe_skip.h applyRow, the same float operations in the same order): with
// mk = clamp(min_keep, 1, K), keep = the first k >= mk with w[k] < thr (K when none); nothing dropped -> unchanged; else the
// kept weights divided by their sum (k order), the rest id -1 / weight 0. thr >= 1: exactly the top mk (static top-k).
__device__ __forceinline__ void moe_skip_rule(int *ids, float *w, int K, float thr, int min_keep) {
    const int mk = min_keep < 1 ? 1 : (min_keep > K ? K : min_keep);
    int keep = K;
    for (int k = mk; k < K; ++k)
        if (w[k] < thr) {
            keep = k;
            break;
        }
    if (keep == K) return;
    float s = 0.f;
    for (int k = 0; k < keep; ++k) s += w[k];
    for (int k = 0; k < K; ++k) {
        if (k < keep) {
            w[k] = w[k] / s;
        } else {
            w[k] = 0.f;
            ids[k] = -1;
        }
    }
}

// Kept slots of token t (a kept prefix, then -1); K when nothing was dropped.
__device__ __forceinline__ int moe_kept(const int *__restrict__ ids, int t, int K) {
    int kt = K;
    while (kt > 1 && ids[(size_t)t * K + kt - 1] < 0) --kt;
    return kt;
}

// moe_topk (shared/moe_logits_f32.h) verbatim, then moe_skip_rule on the token's ids / weights: bitwise == moe_topk followed
// by moe_skip_apply; thr 0 (or nothing below thr) gives moe_topk's output. One 256-thread block per token.
extern "C" __global__ void __launch_bounds__(256)
    moe_topk_skip(const float *__restrict__ logits, int ld, const float *__restrict__ h, const float *__restrict__ shw, int *__restrict__ ids,
                  float *__restrict__ w, float *__restrict__ sg, int R, int K, int E, float thr, int min_keep) {
    const int t = blockIdx.x;
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, nw = blockDim.x / WARP;
    __shared__ float lv[1024];
    __shared__ float rv[32];
    __shared__ int ri[32];
    __shared__ float sel_v[64];
    __shared__ int sel_i[64];
    const float *l = logits + (size_t)t * ld;
    for (int r = threadIdx.x; r < R; r += blockDim.x) lv[r] = l[r];
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int r = threadIdx.x; r < R; r += blockDim.x) {
            const float v = lv[r];
            if (v > bv || (v == bv && r < bi)) {
                bv = v;
                bi = r;
            }
        }
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o, WARP);
            const int oi = __shfl_xor(bi, o, WARP);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            rv[wid] = bv;
            ri[wid] = bi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = rv[0];
            int i = ri[0];
            for (int q = 1; q < nw; ++q) {
                if (rv[q] > b || (rv[q] == b && ri[q] < i)) {
                    b = rv[q];
                    i = ri[q];
                }
            }
            if (i < 0 || i >= R) i = 0;  // all -inf/NaN: keep a valid id
            sel_v[k] = b;
            sel_i[k] = i;
            lv[i] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float m = sel_v[0];
        float s = 0.f;
        for (int k = 0; k < K; ++k) s += __expf(sel_v[k] - m);
        for (int k = 0; k < K; ++k) {
            ids[(size_t)t * K + k] = sel_i[k];
            w[(size_t)t * K + k] = __expf(sel_v[k] - m) / s;
        }
        moe_skip_rule(ids + (size_t)t * K, w + (size_t)t * K, K, thr, min_keep);  // moeskip
    }
    if (shw != nullptr) {
        float d = 0.f;
        for (int c = threadIdx.x; c < E; c += blockDim.x) d += h[(size_t)t * E + c] * shw[c];
        d = block_sum(d);
        if (threadIdx.x == 0) sg[t] = 1.f / (1.f + __expf(-d));
    }
}

// moe_skip_rule on n tokens of a router's output in place (after moe_router_fused_i16 / _i24); one thread per token.
extern "C" __global__ void __launch_bounds__(256) moe_skip_apply(int *__restrict__ ids, float *__restrict__ w, int n, int K, float thr, int min_keep) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    moe_skip_rule(ids + (size_t)t * K, w + (size_t)t * K, K, thr, min_keep);
}

// moe_route (shared/moe_moe_route.h) over the kept pairs: pairs with id -1 are in no tile; they are parked after the kept
// positions (perm = a valid pair, so a gather over every position reads a real token) with inv -1 (moe_combine_skip skips
// them). Without -1: moe_route's tiles and grouping. One 1024-thread block; R <= 1024.
extern "C" __global__ void __launch_bounds__(1024)
    moe_route_skip(const int *__restrict__ ids, int n_pairs, int R, int bn, int *__restrict__ perm, int *__restrict__ inv,
                   int4 *__restrict__ tiles, int *__restrict__ n_tiles) {
    __shared__ int cnt[1024], off[1024], fill[1024];
    __shared__ int n_kept, tail;
    for (int r = threadIdx.x; r < R; r += blockDim.x) {
        cnt[r] = 0;
        fill[r] = 0;
    }
    if (threadIdx.x == 0) tail = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < n_pairs; p += blockDim.x)
        if (ids[p] >= 0) atomicAdd(&cnt[ids[p]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, nt = 0;
        for (int r = 0; r < R; ++r) {
            off[r] = o;
            for (int s = 0; s < cnt[r]; s += bn) tiles[nt++] = make_int4(r, o + s, min(bn, cnt[r] - s), 0);
            o += cnt[r];
        }
        *n_tiles = nt;
        n_kept = o;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < n_pairs; p += blockDim.x) {
        const int e = ids[p];
        if (e < 0) {
            perm[n_kept + atomicAdd(&tail, 1)] = p;
            inv[p] = -1;
            continue;
        }
        const int pos = off[e] + atomicAdd(&fill[e], 1);
        perm[pos] = p;
        inv[p] = pos;
    }
}

// moe_combine with the dropped slots (inv -1) left out: x[t][c] += sum over kept k of w[t,k] * yd[inv[t*K+k]][c]
// (+ sg[t] * ysh[t][c]); per element moe_combine's expression and order. grid (C/256, n).
extern "C" __global__ void moe_combine_skip(float *__restrict__ x, const float *__restrict__ ysh, const float *__restrict__ sg,
                                            const float *__restrict__ yd, const int *__restrict__ inv, const float *__restrict__ w, int C, int K) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int q = inv[(size_t)t * K + k];
        if (q < 0) continue;  // moeskip: a dropped slot
        acc += w[(size_t)t * K + k] * yd[(size_t)q * C + c];
    }
    if (ysh != nullptr) acc += sg[t] * ysh[(size_t)t * C + c];
    x[(size_t)t * C + c] += acc;
}

// Generic MXFP4 decode experts (moe_gu_mxfp4 / moe_down_mxfp4 of either set, the generic QDN path of moe_gu_impl /
// moe_down_impl): gate + up returns for a dropped pair before forming any weight address; down sums the kept prefix only
// (a dropped slot's weight row and activations are never read). A kept pair's value is the shipped kernel's (same code,
// same order); without -1 every launch is the shipped one.
extern "C" __global__ void __launch_bounds__(256)
    moe_gu_skip_mxfp4(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, uint64_t row_bytes, int ff, int nrows,
                      const int8_t *__restrict__ xq, const float *__restrict__ xd, const int *__restrict__ ids, float *__restrict__ y1,
                      float *__restrict__ y2, int C, int K, const int *skip) {
    if (skip != nullptr && *skip) return;
    if (ids[blockIdx.y] < 0) return;  // moeskip: a dropped pair
#include "shared/moe_gu_pair.h"  // lane, row, p, t, e, second, r, wr (ADR-0005)
    const int8_t *xt = xq + (size_t)t * C;
    const float *dt = xd + (size_t)t * (C / 32);
    const int nu = C / QD<QT_MXFP4>::sb * QD<QT_MXFP4>::per_sb;
    float acc = 0.f;
    for (int u = lane; u < nu; u += WARP) QDN<QT_MXFP4>::dot<1>(wr, u, xt, dt, C, &acc);
    acc = warp_sum(acc);
    if (lane == 0) (second ? y2 : y1)[(size_t)p * ff + r] = acc;
}

extern "C" __global__ void __launch_bounds__(256)
    moe_down_skip_mxfp4(const uint8_t *__restrict__ W, uint64_t row_bytes, int E, const int8_t *__restrict__ xq, const float *__restrict__ xd,
                        const int *__restrict__ ids, const float *__restrict__ w, const float *__restrict__ sg, const float *__restrict__ ysh,
                        float *__restrict__ x, int F, int K, const int *skip) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= E) return;
    const int t = blockIdx.y;
    const int kt = moe_kept(ids, t, K);  // moeskip: the kept prefix
    const int nu = F / QD<QT_MXFP4>::sb * QD<QT_MXFP4>::per_sb;
    float acc = 0.f;
    for (int u = lane; u < kt * nu; u += WARP) {
        const int k = u / nu, uu = u - k * nu;
        const int p = t * K + k;
        const uint8_t *wr = W + ((size_t)ids[p] * E + row) * row_bytes;
        float a = 0.f;
        QDN<QT_MXFP4>::dot<1>(wr, uu, xq + (size_t)p * F, xd + (size_t)p * (F / 32), F, &a);
        acc += w[p] * a;
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        float v = acc;
        if (ysh != nullptr) v += sg[t] * ysh[(size_t)t * E + row];
        x[(size_t)t * E + row] += v;
    }
}
