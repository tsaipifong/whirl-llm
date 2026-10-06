// WHIRL device code (gfx1151 / RDNA 3.5): mixture of experts.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

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

// Wide-unit dots for the decode MoE kernels (gemv1s / gemv1s6 unit layout).
#ifndef MOE_WIDE
#define MOE_WIDE true
#endif
template <int T>
__device__ __forceinline__ float moe_wide_unit(const uint8_t *wr, int u, const int8_t *xt, const float *dt, float scale) {
    WUnit wu;
    wunit<T>(wr, u, wu);
    const int base = (u >> 2) * 256 + 64 * (u & 3);
    const int4 a0 = *(const int4 *)(xt + base), a1 = *(const int4 *)(xt + base + 16);
    const int4 c0 = *(const int4 *)(xt + base + 32), c1 = *(const int4 *)(xt + base + 48);
    const int xa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const int xc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    int dl = 0, dh = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        dl = dp4(wu.w[k], xa[k], dl);
        dh = dp4(wu.w[8 + k], xc[k], dh);
    }
    const uint2 p = *(const uint2 *)((const uint32_t *)dt + (base >> 5));
    return scale * wfold(wfold(0.f, p.x, wu.s0, wu.m0, dl), p.y, wu.s1, wu.m1, dh);
}
template <int T>
__device__ __forceinline__ float moe_wide_dot(const uint8_t *wr, int u0, int nu, int lane, const int8_t *xt,
                                              const float *dt, float scale) {
    float acc = 0.f;
    for (int u = u0 + lane; u < nu; u += WARP) acc += moe_wide_unit<T>(wr, u, xt, dt, scale);
    return acc;
}
__device__ __forceinline__ float moe_wide_unit6(const uint8_t *wr, int u, const int8_t *xt, const float *dt, float scale) {
    WUnit6 wu;
    wunit6(wr, u, wu);
    const int base = (u >> 3) * 256 + 128 * ((u >> 2) & 1) + 32 * (u & 3);
    const int4 a0 = *(const int4 *)(xt + base), a1 = *(const int4 *)(xt + base + 16);
    const int xa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    int c0 = 0, c1 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        c0 = dp4(wu.w[k], xa[k], c0);
        c1 = dp4(wu.w[4 + k], xa[4 + k], c1);
    }
    return scale * wfold6(0.f, ((const uint32_t *)dt)[base >> 5], wu.s0, wu.s1, c0, c1);
}
__device__ __forceinline__ float moe_wide_dot6(const uint8_t *wr, int u0, int nu, int lane, const int8_t *xt,
                                               const float *dt, float scale) {
    float acc = 0.f;
    for (int u = u0 + lane; u < nu; u += WARP) acc += moe_wide_unit6(wr, u, xt, dt, scale);
    return acc;
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
    float acc = 0.f;
    if constexpr (MOE_WIDE && (T == QT_Q4_K || T == QT_Q5_K)) {
        acc = moe_wide_dot<T>(wr, 0, C / 256 * 4, lane, xt, dt, 1.f);
    } else if constexpr (MOE_WIDE && T == QT_Q6_K) {
        acc = moe_wide_dot6(wr, 0, C / 256 * 8, lane, xt, dt, 1.f);
    } else {
        const int nu = C / QD<T>::sb * QD<T>::per_sb;
        for (int u = lane; u < nu; u += WARP) QDN<T>::template dot<1>(wr, u, xt, dt, C, &acc);
    }
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
    float acc = 0.f;
    if constexpr (MOE_WIDE && (T == QT_Q4_K || T == QT_Q5_K || T == QT_Q6_K)) {
        // wide units of all K experts spread over the lanes
        constexpr int per_sb = T == QT_Q6_K ? 8 : 4;
        const int nu = F / 256 * per_sb;
        for (int u = lane; u < K * nu; u += WARP) {
            const int k = u / nu, uu = u - k * nu;
            const int p = t * K + k;
            const uint8_t *wr = W + ((size_t)ids[p] * E + row) * row_bytes;
            const int8_t *xt = xq + (size_t)p * F;
            const float *dt = xd + (size_t)p * (F / 32);
            if constexpr (T == QT_Q6_K) acc += moe_wide_unit6(wr, uu, xt, dt, w[p]);
            else acc += moe_wide_unit<T>(wr, uu, xt, dt, w[p]);
        }
    } else {
        const int nu = F / QD<T>::sb * QD<T>::per_sb;
        for (int u = lane; u < K * nu; u += WARP) {
            const int k = u / nu, uu = u - k * nu;
            const int p = t * K + k;
            const uint8_t *wr = W + ((size_t)ids[p] * E + row) * row_bytes;
            float a = 0.f;
            QDN<T>::template dot<1>(wr, uu, xq + (size_t)p * F, xd + (size_t)p * (F / 32), F, &a);
            acc += w[p] * a;
        }
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
template <int T, int BM, int BN, int BK, int WAVES_M, int NTH, bool SKIP = false>
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
        // SKIP: a 16-token fragment column past n_tok only multiplies padding (its outputs
        // are never stored), so it is left out; the live columns run the same WMMA chain
        if (SKIP && wn * TN * 16 >= n_tok) continue;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            half16 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = ld_frag16(&As[(wm * TM + i) * 16 + lane % 16][kk]);
#pragma unroll
            for (int j = 0; j < TN; ++j)
                if (!SKIP || (wn * TN + j) * 16 < n_tok) b[j] = ld_frag16(&Bs[(wn * TN + j) * 16 + lane % 16][kk]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    if (!SKIP || (wn * TN + j) * 16 < n_tok) acc[i][j] = WMMA16(a[i], b[j], acc[i][j]);
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
                const int row = m0 + (wm * TM + i) * 16 + CROW(lane, v);
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

// Tiles of another token-tile size over the same grouping as moe_route (expert counts
// only, so the positions match moe_route's perm whatever its order within an expert).
// One 1024-thread block; R <= 1024.
extern "C" __global__ void __launch_bounds__(1024)
    moe_tiles(const int *__restrict__ ids, int n_pairs, int R, int bn, int4 *__restrict__ tiles, int *__restrict__ n_tiles) {
    __shared__ int cnt[1024];
    for (int r = threadIdx.x; r < R; r += blockDim.x) cnt[r] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < n_pairs; p += blockDim.x) atomicAdd(&cnt[ids[p]], 1);
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

// Grouped expert GEMM, 32-token tiles, for the Radeon 8060S prefill: 128 rows x 32 tokens
// per block with the 8 waves as 4 (rows) x 2 (16-token halves), so a tile of <= 16 tokens
// leaves half the waves without WMMA work (SKIP); 1-D grid with the row block fastest
// (blockIdx.x = tile * nrb + row block) so the blocks reading one tile's X rows run
// together. Per output the same WMMA chain over K as gemm_moe_<T> (bit-identical rows).
template <int T>
__device__ void gemm_moe32r_impl(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X, float *Y, int ncols,
                                 const int4 *tiles, const int *n_tiles) {
    const int nrb = (rows_e + MOE_BM - 1) / MOE_BM;
    const int ti = (int)blockIdx.x / nrb, rb = (int)blockIdx.x % nrb;
    if (ti >= *n_tiles) return;
    const int4 tl = tiles[ti];
    gemm_tile_impl<T, MOE_BM, 32, 64, 4, 256, true>(W + (size_t)tl.x * rows_e * row_bytes, row_bytes, X + (size_t)tl.y * ncols,
                                                    Y + (size_t)tl.y * rows_e, ncols, rows_e, tl.z, rb * MOE_BM, 0);
}

// Gate + up + SwiGLU in one grouped GEMM (64-token tiles): a block takes 64 gate rows
// (W1) and the same 64 up rows (W2) of one expert, i.e. a 128-row A tile, against one
// token tile; wave (wm, wn) holds gate rows wm*16.. and up rows 64 + wm*16.. for tokens
// wn*32.. (16-token fragments past the tile's count skipped). The f32 sums are those of
// gemm_moe_<T> for the same rows (same K order and WMMA chain); A[pos][r] =
// f16(silu(g) * u) with the moe_act_f16 expression, so A equals gemm_moe_<T> (gate),
// gemm_moe_<T> (up), moe_act_f16 bit for bit. 1-D grid, row block fastest.
// X: f16 [pos][ncols], A: f16 [pos][ff]; A must not alias X.
template <int T>
__device__ void gemm_moegu_impl(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, uint64_t row_bytes, int ff,
                                const _Float16 *__restrict__ Xall, _Float16 *__restrict__ Aout, int ncols,
                                const int4 *__restrict__ tiles, const int *__restrict__ n_tiles) {
    constexpr int NTH = 256, BK = 64, BN = 64, GR = 64, WM = 4;
    constexpr int BM = 2 * GR;
    constexpr int WAVES_N = NTH / 32 / WM;
    constexpr int TN = BN / 16 / WAVES_N;
    constexpr int LD = BK + 8;
    constexpr int GPR = BK / 8;
    constexpr int A_PER = BM * GPR / NTH;
    constexpr int B_PER = BN * GPR / NTH;
    static_assert(GR == WM * 16 && BM * GPR % NTH == 0 && BN * GPR % NTH == 0, "tile/threads");
    const int nrb = (ff + GR - 1) / GR;
    const int ti = (int)blockIdx.x / nrb, rb = (int)blockIdx.x % nrb;
    if (ti >= *n_tiles) return;
    const int4 tl = tiles[ti];
    const int n_tok = tl.z, m0 = rb * GR;
    const uint8_t *Wg = W1 + (size_t)tl.x * ff * row_bytes, *Wu = W2 + (size_t)tl.x * ff * row_bytes;
    const _Float16 *X = Xall + (size_t)tl.y * ncols;
    __shared__ _Float16 As[BM][LD];
    __shared__ _Float16 Bs[BN][LD];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WM, wn = wid / WM;
    float8 acc[2][TN];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    half8 ra[A_PER], rbx[B_PER];
#define GU_FETCH(k0_)                                                                               \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < A_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int r = i / GPR, gg = i % GPR;                                                    \
            const int rr = m0 + (r % GR);                                                           \
            if (rr < ff) {                                                                          \
                ra[e] = dec_group_h<T>((r < GR ? Wg : Wu) + (size_t)rr * row_bytes, (k0_) / 8 + gg); \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) ra[e][q] = (_Float16)0.f;              \
            }                                                                                       \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < B_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int t = i / GPR, gg = i % GPR;                                                    \
            if (t < n_tok) {                                                                        \
                rbx[e] = *(const half8 *)(X + (size_t)t * ncols + (k0_) + gg * 8);                  \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) rbx[e][q] = (_Float16)0.f;             \
            }                                                                                       \
        }                                                                                           \
    } while (0)
    GU_FETCH(0);
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
            *(half8 *)&Bs[i / GPR][(i % GPR) * 8] = rbx[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) GU_FETCH(k0 + BK);
        if (wn * TN * 16 >= n_tok) continue;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            half16 a[2], b[TN];
            a[0] = ld_frag16(&As[wm * 16 + lane % 16][kk]);
            a[1] = ld_frag16(&As[GR + wm * 16 + lane % 16][kk]);
#pragma unroll
            for (int j = 0; j < TN; ++j)
                if ((wn * TN + j) * 16 < n_tok) b[j] = ld_frag16(&Bs[(wn * TN + j) * 16 + lane % 16][kk]);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    if ((wn * TN + j) * 16 < n_tok) acc[i][j] = WMMA16(a[i], b[j], acc[i][j]);
        }
    }
#undef GU_FETCH
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = (wn * TN + j) * 16 + lane % 16;
        if (t >= n_tok) continue;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int row = m0 + wm * 16 + CROW(lane, v);
            if (row < ff) {
                const float g = acc[0][j][v], u = acc[1][j][v];
                Aout[(size_t)(tl.y + t) * ff + row] = (_Float16)(g / (1.f + __expf(-g)) * u);
            }
        }
    }
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
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                          \
        gemm_moe32r_##SUF(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X, float *Y,       \
                          int ncols, const int4 *tiles, const int *n_tiles) {                                  \
        gemm_moe32r_impl<T>(W, row_bytes, rows_e, X, Y, ncols, tiles, n_tiles);                                \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                          \
        gemm_moegu_##SUF(const uint8_t *W1, const uint8_t *W2, uint64_t row_bytes, int ff, const _Float16 *X,  \
                         _Float16 *A, int ncols, const int4 *tiles, const int *n_tiles) {                      \
        gemm_moegu_impl<T>(W1, W2, row_bytes, ff, X, A, ncols, tiles, n_tiles);                                \
    }
MOE_ENTRIES(q4_k, QT_Q4_K)
MOE_ENTRIES(q5_k, QT_Q5_K)
MOE_ENTRIES(q6_k, QT_Q6_K)
MOE_ENTRIES(q8_0, QT_Q8_0)
MOE_ENTRIES(iq4_xs, QT_IQ4_XS)
MOE_ENTRIES(iq4_nl, QT_IQ4_NL)
MOE_ENTRIES(q3_k, QT_Q3_K)
MOE_ENTRIES(iq3_s, QT_IQ3_S)
// MXFP4 experts on the generic paths (int8 decode, f16 WMMA prefill) over the load-time
// MXFP4 layout (136-byte superblocks); gfx1151 has no fp8 grouped GEMM.
MOE_ENTRIES(mxfp4, QT_MXFP4)
