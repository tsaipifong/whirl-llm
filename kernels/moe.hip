// WHIRL device code (gfx1201 / RDNA 4): mixture of experts: GDN beta/alpha projections, router, top-k, decode experts, prefill routing and grouped GEMM.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Mixture of experts (qwen35moe): softmax router, top-k, renormalized weights,
// SwiGLU experts plus a sigmoid-gated shared expert (llama.cpp build_moe_ffn
// with LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, norm_w = true).
// Expert tensors are 3D GGUF tensors: expert e owns rows [e * rows_e, (e+1) * rows_e).
// ---------------------------------------------------------------------------

// GDN beta/alpha projections for other quant types (see gdn_ab_q8_0).
#define GDN_AB_ENTRY(SUF, T)                                                                              \
    extern "C" __global__ void __launch_bounds__(256)                                                     \
        gdn_ab_##SUF(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq,         \
                     const float *xd, float *beta, float *g, const float *dt, const float *A, int ncols,  \
                     int nh) {                                                                            \
        const int t = blockIdx.y;                                                                         \
        gdn_ab_impl<T>(Wb, Wa, row_bytes, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32),          \
                       beta + (size_t)t * nh, g + (size_t)t * nh, dt, A, ncols, nh, blockIdx.x);                      \
    }
GDN_AB_ENTRY(q4_k, QT_Q4_K)
GDN_AB_ENTRY(q5_k, QT_Q5_K)
GDN_AB_ENTRY(q6_k, QT_Q6_K)
GDN_AB_ENTRY(iq4_xs, QT_IQ4_XS)
GDN_AB_ENTRY(mxfp4, QT_MXFP4)
GDN_AB_ENTRY(iq4_nl, QT_IQ4_NL)
GDN_AB_ENTRY(q3_k, QT_Q3_K)
GDN_AB_ENTRY(iq3_s, QT_IQ3_S)

// gdn_ab + gdn_conv_l2 in one launch (independent inputs): blocks
// [0, n_ab * ab_blocks) = gdn_ab (token b / ab_blocks, 8 rows per block),
// the rest = gdn_conv_l2 (segment, 128-channel block) with 128 of the 256
// threads active. Per block identical to the two separate kernels.
template <int T>
__device__ void gdn_abconv_impl(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq,
                                const float *xd, float *beta, float *g, const float *dt, const float *A, int ncols,
                                int nh, int n_ab, const float *xin, const float *w, float *out, int ch, int n_qk_heads,
                                float eps, const GdnSegs &segs) {
    const int ab_blocks = (2 * nh + 7) / 8;
    int b = blockIdx.x;
    if (b < n_ab * ab_blocks) {
        const int t = b / ab_blocks;
        gdn_ab_impl<T>(Wb, Wa, row_bytes, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32), beta + (size_t)t * nh,
                       g + (size_t)t * nh, dt, A, ncols, nh, b % ab_blocks);
        return;
    }
    b -= n_ab * ab_blocks;
    const int nblk = ch / 128;
    gdn_conv_l2_impl(xin, w, out, ch, n_qk_heads, eps, segs, b % nblk, b / nblk, threadIdx.x < 128);
}
#define GDN_ABCONV_ENTRY(SUF, T)                                                                               \
    extern "C" __global__ void __launch_bounds__(256)                                                          \
        gdn_abconv_##SUF(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq,          \
                         const float *xd, float *beta, float *g, const float *dt, const float *A, int ncols,   \
                         int nh, int n_ab, const float *xin, const float *w, float *out, int ch, int n_qk_heads, \
                         float eps, GdnSegs segs) {                                                            \
        gdn_abconv_impl<T>(Wb, Wa, row_bytes, xq, xd, beta, g, dt, A, ncols, nh, n_ab, xin, w, out, ch,       \
                           n_qk_heads, eps, segs);                                                             \
    }
GDN_ABCONV_ENTRY(q8_0, QT_Q8_0)
GDN_ABCONV_ENTRY(q4_k, QT_Q4_K)
GDN_ABCONV_ENTRY(q5_k, QT_Q5_K)
GDN_ABCONV_ENTRY(q6_k, QT_Q6_K)
GDN_ABCONV_ENTRY(iq4_xs, QT_IQ4_XS)
GDN_ABCONV_ENTRY(mxfp4, QT_MXFP4)
GDN_ABCONV_ENTRY(iq4_nl, QT_IQ4_NL)
GDN_ABCONV_ENTRY(q3_k, QT_Q3_K)
GDN_ABCONV_ENTRY(iq3_s, QT_IQ3_S)

// Router logits in f32 for n <= 16 tokens: out[t][r] = W[r] . x[t] (C % 4 == 0).
// One wave per router row, 8 rows per block.
extern "C" __global__ void __launch_bounds__(256)
    moe_logits_f32(const float *__restrict__ W, const float *__restrict__ x, float *__restrict__ out, int R, int C,
                   int n, const int *skip) {
    if (skip != nullptr && *skip) return;
    constexpr int MT = 16;
    const int lane = threadIdx.x % WARP;
    const int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (r >= R) return;
    const float4 *wr = (const float4 *)(W + (size_t)r * C);
    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.f;
    for (int c = lane; c < C / 4; c += WARP) {
        const float4 w = wr[c];
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t < n) {
                const float4 v = ((const float4 *)(x + (size_t)t * C))[c];
                acc[t] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (t < n) {
            const float s = warp_sum(acc[t]);
            if (lane == 0) out[(size_t)t * R + r] = s;
        }
    }
}

// Per token (one 256-thread block): top-K of the R router logits (ties -> lower
// index), weights = softmax restricted to the selected experts (= softmax over
// all R, then renormalized), sg[t] = sigmoid(shw . h[t]) when shw != null.
extern "C" __global__ void __launch_bounds__(256)
    moe_topk(const float *__restrict__ logits, int ld, const float *__restrict__ h, const float *__restrict__ shw,
             int *__restrict__ ids, float *__restrict__ w, float *__restrict__ sg, int R, int K, int E) {
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
    }
    if (shw != nullptr) {
        float d = 0.f;
        for (int c = threadIdx.x; c < E; c += blockDim.x) d += h[(size_t)t * E + c] * shw[c];
        d = block_sum(d);
        if (threadIdx.x == 0) sg[t] = 1.f / (1.f + __expf(-d));
    }
}

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
extern "C" __global__ void __launch_bounds__(1024)
    moe_route(const int *__restrict__ ids, int n_pairs, int R, int bn, int *__restrict__ perm, int *__restrict__ inv,
              int4 *__restrict__ tiles, int *__restrict__ n_tiles) {
    __shared__ int cnt[1024], off[1024], fill[1024];
    for (int r = threadIdx.x; r < R; r += blockDim.x) {
        cnt[r] = 0;
        fill[r] = 0;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < n_pairs; p += blockDim.x) atomicAdd(&cnt[ids[p]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, nt = 0;
        for (int r = 0; r < R; ++r) {
            off[r] = o;
            for (int s = 0; s < cnt[r]; s += bn) tiles[nt++] = make_int4(r, o + s, min(bn, cnt[r] - s), 0);
            o += cnt[r];
        }
        *n_tiles = nt;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < n_pairs; p += blockDim.x) {
        const int e = ids[p];
        const int pos = off[e] + atomicAdd(&fill[e], 1);
        perm[pos] = p;
        inv[p] = pos;
    }
}

// xg[pos][c] = f16(x[perm[pos] / K][c]); one block per pos, C % 4 == 0.
extern "C" __global__ void moe_gather_f16(const float *__restrict__ x, const int *__restrict__ perm,
                                          _Float16 *__restrict__ xg, int C, int K) {
    const int pos = blockIdx.x;
    const int t = perm[pos] / K;
    for (int c = threadIdx.x * 4; c < C; c += blockDim.x * 4) {
        const float4 v = *(const float4 *)(x + (size_t)t * C + c);
        _Float16 *o = xg + (size_t)pos * C + c;
        o[0] = (_Float16)v.x;
        o[1] = (_Float16)v.y;
        o[2] = (_Float16)v.z;
        o[3] = (_Float16)v.w;
    }
}

// a[i] = f16(silu(g[i]) * u[i])
extern "C" __global__ void moe_act_f16(const float *__restrict__ g, const float *__restrict__ u,
                                       _Float16 *__restrict__ a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = g[i];
    a[i] = (_Float16)(v / (1.f + __expf(-v)) * u[i]);
}

// x[t][c] += sum_k w[t,k] * yd[inv[t*K+k]][c] (+ sg[t] * ysh[t][c]); grid (C/256, n).
extern "C" __global__ void moe_combine(float *__restrict__ x, const float *__restrict__ ysh, const float *__restrict__ sg,
                                       const float *__restrict__ yd, const int *__restrict__ inv,
                                       const float *__restrict__ w, int C, int K) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += w[(size_t)t * K + k] * yd[(size_t)inv[(size_t)t * K + k] * C + c];
    if (ysh != nullptr) acc += sg[t] * ysh[(size_t)t * C + c];
    x[(size_t)t * C + c] += acc;
}

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
#define MOE_BM 128
template <int T, int BN>
__device__ void gemm_moe_impl(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X, float *Y, int ncols,
                              const int4 *tiles, const int *n_tiles) {
    if ((int)blockIdx.x >= *n_tiles) return;
    const int4 tl = tiles[blockIdx.x];
    gemm_tile_impl<T, MOE_BM, BN, 64, (BN == 32 ? 8 : 4), 256>(W + (size_t)tl.x * rows_e * row_bytes, row_bytes,
                                                   X + (size_t)tl.y * ncols, Y + (size_t)tl.y * rows_e, ncols, rows_e,
                                                   tl.z, blockIdx.y * MOE_BM, 0);
}

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

