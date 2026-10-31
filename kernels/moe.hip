// WHIRL device code (gfx1201 / RDNA 4): mixture of experts: router, top-k, decode experts, prefill routing and grouped GEMM.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Mixture of experts (qwen35moe): softmax router, top-k, renormalized weights,
// SwiGLU experts plus a sigmoid-gated shared expert (llama.cpp build_moe_ffn
// with LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, norm_w = true).
// Expert tensors are 3D GGUF tensors: expert e owns rows [e * rows_e, (e+1) * rows_e).
// ---------------------------------------------------------------------------

// Router logits in f32 (moe_logits_f32).
#include "shared/moe_logits_f32.h"  // also in the gfx1151 set (kernels/gfx1151/moe.hip)

// Decode experts, gate and up: pair p = blockIdx.y = t * K + k runs expert
// ids[p] on token t's int8 activation. Rows [0, ff) come from W1 into y1,
// [ff, nrows) from W2 into y2 (nrows = 2 ff when both share a quant type).
template <int T>
__device__ void moe_gu_impl(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, uint64_t row_bytes, int ff,
                            int nrows, const int8_t *__restrict__ xq, const float *__restrict__ xd,
                            const int *__restrict__ ids, float *__restrict__ y1, float *__restrict__ y2, int C, int K) {
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const int p = blockIdx.y;
    const int t = p / K, e = ids[p];
    const bool second = row >= ff;
    const int r = second ? row - ff : row;
    const uint8_t *wr = (second ? W2 : W1) + ((size_t)e * ff + r) * row_bytes;
    const int8_t *xt = xq + (size_t)t * C;
    const float *dt = xd + (size_t)t * (C / 32);
    const int nu = C / QD<T>::sb * QD<T>::per_sb;
    float acc = 0.f;
    for (int u = lane; u < nu; u += WARP) QDN<T>::template dot<1>(wr, u, xt, dt, C, &acc);
    acc = warp_sum(acc);
    if (lane == 0) (second ? y2 : y1)[(size_t)p * ff + r] = acc;
}

// Decode experts, down + combine: x[t][row] += sum_k w[t,k] * (W_{ids[t,k]} a[t,k])[row]
// (+ sg[t] * ysh[t][row]). a: int8 [n * K][F]. One wave per output row, all K
// experts' units spread over its lanes.
template <int T>
__device__ void moe_down_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, int E, const int8_t *__restrict__ xq,
                              const float *__restrict__ xd, const int *__restrict__ ids, const float *__restrict__ w,
                              const float *__restrict__ sg, const float *__restrict__ ysh, float *__restrict__ x, int F,
                              int K) {
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= E) return;
    const int t = blockIdx.y;
    const int nu = F / QD<T>::sb * QD<T>::per_sb;
    float acc = 0.f;
    for (int u = lane; u < K * nu; u += WARP) {
        const int k = u / nu, uu = u - k * nu;
        const int p = t * K + k;
        const uint8_t *wr = W + ((size_t)ids[p] * E + row) * row_bytes;
        float a = 0.f;
        QDN<T>::template dot<1>(wr, uu, xq + (size_t)p * F, xd + (size_t)p * (F / 32), F, &a);
        acc += w[p] * a;
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        float v = acc;
        if (ysh != nullptr) v += sg[t] * ysh[(size_t)t * E + row];
        x[(size_t)t * E + row] += v;
    }
}

// Prefill: group the n_pairs (token, slot) pairs by expert. perm[pos] = pair
// (sorted by expert), inv[pair] = pos, tiles[i] = (expert, first pos, count <= bn),
// *n_tiles = number of tiles. One 1024-thread block; R <= 1024.
#include "shared/moe_moe_route.h"  // also in the gfx1151 set (kernels/gfx1151/moe.hip)

// One (BM rows x BN tokens) WMMA tile at (m0, t0): gemm3_impl with the tile
// origin passed in instead of taken from blockIdx (grouped expert GEMM).
template <int T, int BM, int BN, int BK, int WAVES_M, int NTH>
__device__ void gemm_tile_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                               float *__restrict__ Y, int ncols, int nrows, int n_tok, int m0, int t0) {
    constexpr int WAVES_N = NTH / 32 / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int LD = BK + 8;
    constexpr int GPR = BK / 8;
    constexpr int A_PER = BM * GPR / NTH;
    constexpr int B_PER = BN * GPR / NTH;
    static_assert(BM * GPR % NTH == 0 && BN * GPR % NTH == 0, "tile/threads");
    __shared__ _Float16 As[BM][LD];
    __shared__ _Float16 Bs[BN][LD];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    half8 ra[A_PER], rb[B_PER];
#define GTILE_FETCH(k0_)                                                                            \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < A_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int r = i / GPR, gg = i % GPR;                                                    \
            if (m0 + r < nrows) {                                                                   \
                ra[e] = dec_group_h<T>(W + (size_t)(m0 + r) * row_bytes, (k0_) / 8 + gg);            \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) ra[e][q] = (_Float16)0.f;              \
            }                                                                                       \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < B_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int t = i / GPR, gg = i % GPR;                                                    \
            if (t0 + t < n_tok) {                                                                   \
                rb[e] = *(const half8 *)(X + (size_t)(t0 + t) * ncols + (k0_) + gg * 8);            \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) rb[e][q] = (_Float16)0.f;              \
            }                                                                                       \
        }                                                                                           \
    } while (0)
    GTILE_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(half8 *)&As[i / GPR][(i % GPR) * 8] = ra[e];
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(half8 *)&Bs[i / GPR][(i % GPR) * 8] = rb[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) GTILE_FETCH(k0 + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            half8 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = *(const half8 *)&As[(wm * TM + i) * 16 + lane % 16][kk + 8 * (lane / 16)];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = *(const half8 *)&Bs[(wn * TN + j) * 16 + lane % 16][kk + 8 * (lane / 16)];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
        }
    }
#undef GTILE_FETCH
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int t = t0 + (wn * TN + j) * 16 + lane % 16;
            if (t >= n_tok) continue;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + 8 * (lane / 16) + v;
                if (row < nrows) Y[(size_t)t * nrows + row] = acc[i][j][v];
            }
        }
    }
}

// Grouped expert GEMM over the tiles built by moe_route: tile blockIdx.x
// (expert e, first pos s, count c) computes Y[s..s+c)[rows_e] = W_e X[s..s+c)^T
// for row block blockIdx.y. X: f16 [pos][ncols], Y: f32 [pos][rows_e].
// Token tile BN = 32 (gemm_moe32_*, short prompts) or 64 (gemm_moe_*); BK = 64.
#include "shared/moe_moe_bm.h"  // also in the gfx1151 set (kernels/gfx1151/moe.hip)

#define MOE_ENTRIES(SUF, T)                                                                                    \
    extern "C" __global__ void __launch_bounds__(256)                                                          \
        moe_gu_##SUF(const uint8_t *W1, const uint8_t *W2, uint64_t row_bytes, int ff, int nrows,              \
                     const int8_t *xq, const float *xd, const int *ids, float *y1, float *y2, int C, int K,    \
                     const int *skip) {                                                                        \
        if (skip != nullptr && *skip) return;                                                                  \
        moe_gu_impl<T>(W1, W2, row_bytes, ff, nrows, xq, xd, ids, y1, y2, C, K);                               \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                          \
        moe_down_##SUF(const uint8_t *W, uint64_t row_bytes, int E, const int8_t *xq, const float *xd,         \
                       const int *ids, const float *w, const float *sg, const float *ysh, float *x, int F,     \
                       int K, const int *skip) {                                                               \
        if (skip != nullptr && *skip) return;                                                                  \
        moe_down_impl<T>(W, row_bytes, E, xq, xd, ids, w, sg, ysh, x, F, K);                                   \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                          \
        gemm_moe_##SUF(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X, float *Y,          \
                       int ncols, const int4 *tiles, const int *n_tiles) {                                     \
        gemm_moe_impl<T, 64>(W, row_bytes, rows_e, X, Y, ncols, tiles, n_tiles);                               \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                          \
        gemm_moe32_##SUF(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X, float *Y,        \
                         int ncols, const int4 *tiles, const int *n_tiles) {                                   \
        gemm_moe_impl<T, 32>(W, row_bytes, rows_e, X, Y, ncols, tiles, n_tiles);                               \
    }
MOE_ENTRIES(q4_k, QT_Q4_K)
MOE_ENTRIES(q5_k, QT_Q5_K)
MOE_ENTRIES(q6_k, QT_Q6_K)
MOE_ENTRIES(q8_0, QT_Q8_0)
MOE_ENTRIES(iq4_xs, QT_IQ4_XS)
MOE_ENTRIES(iq4_nl, QT_IQ4_NL)
MOE_ENTRIES(q3_k, QT_Q3_K)
MOE_ENTRIES(iq3_s, QT_IQ3_S)

