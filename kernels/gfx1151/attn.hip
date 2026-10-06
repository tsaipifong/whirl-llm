// WHIRL device code (gfx1151 / RDNA 3.5): paged KV, decode / split / prefill attention.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Paged KV cache (one shared block pool). Each attention layer's K and V
// caches are pools of KV_PAGE-token pages; logical position p of a sequence
// lives in pool row ptab[tb + p / KV_PAGE] * KV_PAGE + p % KV_PAGE, tb = the
// sequence's page-table offset (per batched row: kvbase[row]; else tab0).
// Element format: f16, or q8 (int8 + one f16 scale per 32 consecutive
// values of a row; scales in ks / vs as [pool row][row elements / 32]).
// Every kernel reading K/V dequantizes with the same f16 multiply
// (kv_ld8 / kv_ld1), and K/V are quantized once, when stored (kv_st32).
// ---------------------------------------------------------------------------
#define KV_PAGE 256
struct KvArgs {
    void *k;
    void *v;
    _Float16 *ks;
    _Float16 *vs;
    const int *ptab;
    const int *kvbase;
    int tab0;
    int win;  // MTP draft window (attn_wsplit1 only): W / 64 | (threshold / 1024) << 16, 0 = off
};
typedef char kvq8_t __attribute__((ext_vector_type(8)));
__device__ __forceinline__ int kv_tb(const KvArgs &a, int row) { return a.kvbase != nullptr ? a.kvbase[row] : a.tab0; }
__device__ __forceinline__ int kv_row(const KvArgs &a, int tb, int p) { return a.ptab[tb + p / KV_PAGE] * KV_PAGE + p % KV_PAGE; }
// pool row of p minus p: the same for every position of p's page (aligned key tiles)
__device__ __forceinline__ int kv_delta(const KvArgs &a, int tb, int p) {
    const int pg = p / KV_PAGE;
    return (a.ptab[tb + pg] - pg) * KV_PAGE;
}
template <bool Q8> __device__ __forceinline__ half8 kv_ld8(const void *base, const _Float16 *sc, size_t e) {
    if constexpr (Q8) {
        const kvq8_t q = *(const kvq8_t *)((const int8_t *)base + e);
        const _Float16 s = sc[e >> 5];
        half8 h;
#pragma unroll
        for (int i = 0; i < 8; ++i) h[i] = (_Float16)q[i] * s;
        return h;
    } else {
        return *(const half8 *)((const _Float16 *)base + e);
    }
}
template <bool Q8> __device__ __forceinline__ float kv_ld1(const void *base, const _Float16 *sc, size_t e) {
    if constexpr (Q8) {
        return (float)((_Float16)((const int8_t *)base)[e] * sc[e >> 5]);
    } else {
        return (float)((const _Float16 *)base)[e];
    }
}
// gfx11 WMMA fragment: 16 consecutive values (one scale group half)
template <bool Q8> __device__ __forceinline__ half16 kv_ld16(const void *base, const _Float16 *sc, size_t e) {
    const half8 lo = kv_ld8<Q8>(base, sc, e), hi = kv_ld8<Q8>(base, sc, e + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
// Store x at element e; the 32 lanes of the wave hold the 32 elements of one
// aligned scale group (q8: scale = amax / 127 rounded to f16, q = round(x / scale)).
template <bool Q8> __device__ __forceinline__ void kv_st32(void *base, _Float16 *sc, size_t e, float x) {
    if constexpr (Q8) {
        const float amax = warp_max(fabsf(x));
        const _Float16 s = (_Float16)(amax * (1.f / 127.f));
        const float sf = (float)s;
        int qv = sf > 0.f ? (int)rintf(x / sf) : 0;
        qv = max(-127, min(127, qv));
        ((int8_t *)base)[e] = (int8_t)qv;
        if ((threadIdx.x % WARP) == 0) sc[e >> 5] = s;
    } else {
        ((_Float16 *)base)[e] = (_Float16)x;
    }
}

// ---------------------------------------------------------------------------
// Full attention (decode, one query token per launch row; the scalar
// reference path, single sequence).
// q: [n_tok][n_head][q_stride] (query at offset 0, gate at +head_dim).
// out: [n_tok][n_head*head_dim], gated by sigmoid.
// scores scratch: [n_tok][n_head][max_ctx].
// Token t attends to positions [0, pos0 + t].
// ---------------------------------------------------------------------------
template <bool Q8>
__device__ __forceinline__ void attn_decode_impl(const float *q, const KvArgs &kv, float *out, float *scores,
                int n_head, int n_kv, int head_dim, int q_stride, int max_ctx, const int *pos, float scale) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int pos0 = pos[0];
    const int kvh = h / (n_head / n_kv);
    const int n_pos = pos0 + t + 1;
    const int tb = kv.tab0;
    const float *qv = q + ((size_t)t * n_head + h) * q_stride;
    float *sc = scores + ((size_t)t * n_head + h) * max_ctx;

    __shared__ float qs[512];
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) qs[i] = qv[i];
    __syncthreads();

    // scores: one wave per position
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, nw = blockDim.x / WARP;
    for (int p = wid; p < n_pos; p += nw) {
        const size_t kb = ((size_t)kv_row(kv, tb, p) * n_kv + kvh) * head_dim;
        float s = 0.f;
        for (int i = lane; i < head_dim; i += WARP) s += qs[i] * kv_ld1<Q8>(kv.k, kv.ks, kb + i);
        s = warp_sum(s);
        if (lane == 0) sc[p] = s * scale;
    }
    __syncthreads();

    float m = -INFINITY;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) m = fmaxf(m, sc[p]);
    m = block_max(m);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        const float e = __expf(sc[p] - m);
        sc[p] = e;
        sum += e;
    }
    sum = block_sum(sum);
    const float inv = 1.f / sum;
    __syncthreads();

    const float *gate = qv + head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < n_pos; ++p)
            acc += sc[p] * kv_ld1<Q8>(kv.v, kv.vs, ((size_t)kv_row(kv, tb, p) * n_kv + kvh) * head_dim + d);
        const float g = gate[d];
        out[(size_t)t * n_head * head_dim + h * head_dim + d] = acc * inv / (1.f + __expf(-g));
    }
}
#define ATTN_DECODE_ENTRY(NAME, Q8)                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                        \
        NAME(const float *q, KvArgs kv, float *out, float *scores, int n_head, int n_kv, int head_dim,     \
             int q_stride, int max_ctx, const int *pos, float scale) {                                      \
        attn_decode_impl<Q8>(q, kv, out, scores, n_head, n_kv, head_dim, q_stride, max_ctx, pos, scale);  \
    }
ATTN_DECODE_ENTRY(attn_decode, false)
ATTN_DECODE_ENTRY(attn_decode_q8, true)

// ---------------------------------------------------------------------------
// Split-K decode attention (flash-decoding), one query token.
// Block (g, split): kv head g, positions [split*FD_CH, +FD_CH). The GQA group's
// query heads share every K/V row the block reads. Partials (running max m,
// sum l, unnormalized acc) go to scratch; attn_combine merges the splits and
// applies the sigmoid output gate. Grid is sized for max_ctx so the launch is
// position-independent; splits past the current length exit early.
// part layout: [n_split][n_head] of {m, l} and [n_split][n_head][head_dim] acc.
// ---------------------------------------------------------------------------
#define FD_CH 64
#define FD_MAXG 8

#define FD_MAXQ 4
#define FD_ROWS (FD_MAXQ * FD_MAXG)

// nq queries (speculative verify; 1 for decode) share one pass over K/V: row
// hh = t * grp + j is query t, head j of the group; query t sees positions
// < pos[t] + 1 (later positions score -inf and add exact zeros).
template <int NQ, bool Q8>
__device__ __forceinline__ void attn_split_impl(const float *q, const KvArgs &kv, int tb, float *part_ml,
                                                float *part_acc, int n_head, int n_kv, int head_dim, int q_stride,
                                                const int *pos, float scale) {
    constexpr int nq = NQ, ROWS = NQ * FD_MAXG;
    const int g = blockIdx.x, split = blockIdx.y, n_split = gridDim.y;
    const int grp = n_head / n_kv, nh = nq * grp;
    {
        // grid.z = independent query groups (one block set per verify query:
        // they run concurrently and share K/V through the cache)
        const int tz = blockIdx.z;
        q += (size_t)tz * NQ * n_head * q_stride;
        pos += tz * NQ;
        part_ml += (size_t)tz * NQ * n_split * n_head * 2;
        part_acc += (size_t)tz * NQ * n_split * n_head * head_dim;
    }
    const int n_pos = pos[nq - 1] + 1;
    // gridDim.y splits share the positions in whole FD_CH chunks, so a fixed
    // grid (graph-captured decode) adapts to the device-side position; each
    // block walks its chunks with an online softmax.
    const int per = (n_pos + n_split * FD_CH - 1) / (n_split * FD_CH) * FD_CH;
    const int b0 = split * per, b1 = min(n_pos, b0 + per);
    const size_t ml_q = (size_t)n_split * n_head * 2, acc_q = (size_t)n_split * n_head * head_dim;
    __shared__ __align__(16) float qs[ROWS][256];
    __shared__ float sc[ROWS][FD_CH];
    __shared__ float sp[4][ROWS][FD_CH];
    __shared__ float mrun[ROWS], lrun[ROWS], fac[ROWS];
    __shared__ int npq[NQ];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, nw = blockDim.x / WARP;
    if (b0 >= n_pos) {
        if (threadIdx.x < nh) {
            const int t = threadIdx.x / grp, h = g * grp + threadIdx.x % grp;
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = -INFINITY;
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = 0.f;
        }
        return;
    }
    for (int i = threadIdx.x; i < nh * head_dim; i += blockDim.x) {
        const int hh = i / head_dim, d = i % head_dim;
        const int t = hh / grp, j = hh % grp;
        qs[hh][d] = q[(size_t)t * n_head * q_stride + (size_t)(g * grp + j) * q_stride + d];
    }
    if (threadIdx.x < ROWS) {
        mrun[threadIdx.x] = -INFINITY;
        lrun[threadIdx.x] = 0.f;
    }
    if (threadIdx.x < nq) npq[threadIdx.x] = pos[threadIdx.x] + 1;
    // thread d owns output column d (head_dim <= blockDim)
    const int d = threadIdx.x;
    float acc[ROWS];
#pragma unroll
    for (int hh = 0; hh < ROWS; ++hh) acc[hh] = 0.f;
    __syncthreads();
    for (int c0 = b0; c0 < b1; c0 += FD_CH) {
        const int c1 = min(b1, c0 + FD_CH);
        const int dl = kv_delta(kv, tb, c0);  // FD_CH chunks never straddle a page
        if (head_dim == 256 && nw == 8) {
            // scores, lane = position: wave w takes positions (w & 1) * 32 + lane
            // over dims [64 (w >> 1), +64); q reads are LDS broadcasts and the
            // four dim-slice partials are summed through LDS (no shuffles)
            const int pi = (wid & 1) * 32 + lane, sl = wid >> 1, p = c0 + pi;
            float part[ROWS];
#pragma unroll
            for (int hh = 0; hh < ROWS; ++hh) part[hh] = 0.f;
            if (p < c1) {
                const size_t kr = ((size_t)(p + dl) * n_kv + g) * head_dim + 64 * sl;
#pragma unroll
                for (int e8 = 0; e8 < 8; ++e8) {
                    const half8 kh8 = kv_ld8<Q8>(kv.k, kv.ks, kr + 8 * e8);
                    float kv[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) kv[e] = (float)kh8[e];
#pragma unroll
                    for (int hh = 0; hh < ROWS; ++hh) {
                        if (hh < nh) {
                            const float4 qa = *(const float4 *)&qs[hh][64 * sl + 8 * e8];
                            const float4 qb = *(const float4 *)&qs[hh][64 * sl + 8 * e8 + 4];
                            part[hh] += qa.x * kv[0] + qa.y * kv[1] + qa.z * kv[2] + qa.w * kv[3] + qb.x * kv[4] +
                                        qb.y * kv[5] + qb.z * kv[6] + qb.w * kv[7];
                        }
                    }
                }
            }
#pragma unroll
            for (int hh = 0; hh < ROWS; ++hh)
                if (hh < nh) sp[sl][hh][pi] = part[hh];
            __syncthreads();
            for (int i = threadIdx.x; i < nh * FD_CH; i += blockDim.x) {
                const int hh = i / FD_CH, pp = i % FD_CH;
                if (c0 + pp < c1) {
                    const float s = (sp[0][hh][pp] + sp[1][hh][pp]) + (sp[2][hh][pp] + sp[3][hh][pp]);
                    sc[hh][pp] = c0 + pp < npq[hh / grp] ? s * scale : -INFINITY;
                }
            }
        } else {
            // scores: one wave per position, lanes over head_dim (8 each for 256)
            for (int p = c0 + wid; p < c1; p += nw) {
                const half8 kh8 = kv_ld8<Q8>(kv.k, kv.ks, ((size_t)(p + dl) * n_kv + g) * head_dim + lane * 8);
                float kv[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) kv[e] = (float)kh8[e];
                for (int hh = 0; hh < nh; ++hh) {
                    float s = 0.f;
#pragma unroll
                    for (int e = 0; e < 8; ++e) s += qs[hh][lane * 8 + e] * kv[e];
                    s = warp_sum(s);
                    if (lane == 0) sc[hh][p - c0] = p < npq[hh / grp] ? s * scale : -INFINITY;
                }
            }
        }
        __syncthreads();
        // per-row running max / exp / sum: wave w handles rows w, w + nw, ...
        for (int hh = wid; hh < nh; hh += nw) {
            float m = -INFINITY;
            for (int p = lane; p < c1 - c0; p += WARP) m = fmaxf(m, sc[hh][p]);
            m = warp_max(m);
            const float mo = mrun[hh];
            const float M = fmaxf(mo, m);
            float l = 0.f;
            if (M == -INFINITY) {  // nothing visible yet for this query
                for (int p = lane; p < c1 - c0; p += WARP) sc[hh][p] = 0.f;
            } else {
                for (int p = lane; p < c1 - c0; p += WARP) {
                    const float e = __expf(sc[hh][p] - M);
                    sc[hh][p] = e;
                    l += e;
                }
            }
            l = warp_sum(l);
            if (lane == 0) {
                const float f = M == -INFINITY ? 1.f : __expf(mo - M);
                fac[hh] = f;
                lrun[hh] = lrun[hh] * f + l;
                mrun[hh] = M;
            }
        }
        __syncthreads();
        if (d < head_dim) {
#pragma unroll
            for (int hh = 0; hh < ROWS; ++hh)
                if (hh < nh) acc[hh] *= fac[hh];
            for (int p = c0; p < c1; ++p) {
                const float v = kv_ld1<Q8>(kv.v, kv.vs, ((size_t)(p + dl) * n_kv + g) * head_dim + d);
#pragma unroll
                for (int hh = 0; hh < ROWS; ++hh)
                    if (hh < nh) acc[hh] += sc[hh][p - c0] * v;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < nh) {
        const int t = threadIdx.x / grp, h = g * grp + threadIdx.x % grp;
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = mrun[threadIdx.x];
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = lrun[threadIdx.x];
    }
    if (d < head_dim)
#pragma unroll
        for (int hh = 0; hh < ROWS; ++hh)
            if (hh < nh) {
                const int t = hh / grp, h = g * grp + hh % grp;
                part_acc[t * acc_q + ((size_t)split * n_head + h) * head_dim + d] = acc[hh];
            }
}

#define ATTN_SPLIT_ENTRY(NAME, NQ, Q8)                                                                        \
    extern "C" __global__ void __launch_bounds__(256)                                                         \
        NAME(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head,                          \
             int n_kv, int head_dim, int q_stride, const int *pos, float scale, int nq, const int *skip) {    \
        if (skip != nullptr && *skip) return;                                                                 \
        const int tb = kv_tb(kv, blockIdx.z * NQ);                                                            \
        attn_split_impl<NQ, Q8>(q, kv, tb, part_ml, part_acc, n_head, n_kv, head_dim, q_stride, pos, scale);  \
    }
// NQ > 1 (one pass over K/V for several queries) measured slower than
// grid.z = query with NQ = 1: the score reductions, not K/V bytes, bound it.
ATTN_SPLIT_ENTRY(attn_split, 1, false)
ATTN_SPLIT_ENTRY(attn_split_q8, 1, true)

// out[h][d] = sum_i e^(m_i - M) acc_i[d] / sum_i e^(m_i - M) l_i, gated by sigmoid.
extern "C" __global__ void attn_combine(const float *part_ml, const float *part_acc, const float *q, float *out,
                                        int n_head, int head_dim, int q_stride, int n_split) {
    const int h = blockIdx.x;
    const int tq = blockIdx.y;  // query token
    part_ml += (size_t)tq * n_split * n_head * 2;
    part_acc += (size_t)tq * n_split * n_head * head_dim;
    q += (size_t)tq * n_head * q_stride;
    out += (size_t)tq * n_head * head_dim;
    float M = -INFINITY;
    for (int i = 0; i < n_split; ++i) M = fmaxf(M, part_ml[((size_t)i * n_head + h) * 2]);
    float L = 0.f;
    for (int i = 0; i < n_split; ++i) {
        const float m = part_ml[((size_t)i * n_head + h) * 2];
        if (m > -INFINITY) L += __expf(m - M) * part_ml[((size_t)i * n_head + h) * 2 + 1];
    }
    const float inv = 1.f / L;
    const float *gate = q + (size_t)h * q_stride + head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < n_split; ++i) {
            const float m = part_ml[((size_t)i * n_head + h) * 2];
            if (m > -INFINITY) acc += __expf(m - M) * part_acc[((size_t)i * n_head + h) * head_dim + d];
        }
        const float gg = gate[d];
        out[(size_t)h * head_dim + d] = acc * inv / (1.f + __expf(-gg));
    }
}

// ---------------------------------------------------------------------------
// Fused attention prep: per (head, token) block the q_norm / k_norm RMSNorm,
// NEOX RoPE on the first n_rot dims and, for k heads, the K/V cache store.
// Same float expressions as rmsnorm + rope_neox + kv_store (same block_sum
// order), so the results are unchanged; 5 launches per attention layer -> 1.
// grid = (n_head + n_kv, n_tok), 256 threads, hd <= 512.
// ---------------------------------------------------------------------------
template <bool Q8>
__device__ __forceinline__ void attn_prep_impl(float *qf, float *kk, const float *vv, const float *qw, const float *kw, const KvArgs &kv,
              const int *pos, int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps) {
    __shared__ float nv[512];
    const int hb = blockIdx.x, t = blockIdx.y;
    const bool isq = hb < n_head;
    const int h = isq ? hb : hb - n_head;
    float *xv = isq ? qf + ((size_t)t * n_head + h) * 2 * hd : kk + ((size_t)t * n_kv + h) * hd;
    const float *w = isq ? qw : kw;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += xv[i] * xv[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) nv[i] = xv[i] * r * w[i];
    __syncthreads();
    const int half = n_rot / 2;
    const float p = (float)pos[t];
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float theta = p * powf(theta_scale, (float)i);
        float s, c;
        sincosf(theta, &s, &c);
        const float x0 = nv[i], x1 = nv[i + half];
        nv[i] = x0 * c - x1 * s;
        nv[i + half] = x0 * s + x1 * c;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) xv[i] = nv[i];
    if (isq) return;
    const int row = n_kv * hd;
    // single sequence: consecutive positions from pos[0]; batched rows: pos[t]
    const int lp = kv.kvbase == nullptr ? pos[0] + t : pos[t];
    const size_t prow = (size_t)kv_row(kv, kv_tb(kv, t), lp);
    const float *vr = vv + (size_t)t * row + (size_t)h * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {  // hd % 32 == 0: whole waves per scale group
        kv_st32<Q8>(kv.k, kv.ks, prow * row + (size_t)h * hd + d, nv[d]);
        kv_st32<Q8>(kv.v, kv.vs, prow * row + (size_t)h * hd + d, vr[d]);
    }
}
extern "C" __global__ void __launch_bounds__(256)
    attn_prep(float *qf, float *kk, const float *vv, const float *qw, const float *kw, KvArgs kv,
              const int *pos, int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps) {
    attn_prep_impl<false>(qf, kk, vv, qw, kw, kv, pos, n_head, n_kv, hd, n_rot, theta_scale, eps);
}
extern "C" __global__ void __launch_bounds__(256)
    attn_prep_q8(float *qf, float *kk, const float *vv, const float *qw, const float *kw, KvArgs kv,
              const int *pos, int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps) {
    attn_prep_impl<true>(qf, kk, vv, qw, kw, kv, pos, n_head, n_kv, hd, n_rot, theta_scale, eps);
}

// attn_combine + the int8 copy of its output for the o-projection GEMV
// (same values as quantize_q8 over out); needs blockDim == head_dim, a
// multiple of 32.
extern "C" __global__ void attn_combine_q8(const float *part_ml, const float *part_acc, const float *q, float *out,
                                           int n_head, int head_dim, int q_stride, int n_split, int8_t *xq, float *xd) {
    const int h = blockIdx.x;
    const int tq = blockIdx.y;  // query token
    part_ml += (size_t)tq * n_split * n_head * 2;
    part_acc += (size_t)tq * n_split * n_head * head_dim;
    q += (size_t)tq * n_head * q_stride;
    out += (size_t)tq * n_head * head_dim;
    float M = -INFINITY;
    for (int i = 0; i < n_split; ++i) M = fmaxf(M, part_ml[((size_t)i * n_head + h) * 2]);
    float L = 0.f;
    for (int i = 0; i < n_split; ++i) {
        const float m = part_ml[((size_t)i * n_head + h) * 2];
        if (m > -INFINITY) L += __expf(m - M) * part_ml[((size_t)i * n_head + h) * 2 + 1];
    }
    const float inv = 1.f / L;
    const float *gate = q + (size_t)h * q_stride + head_dim;
    const int d = threadIdx.x;
    float acc = 0.f;
    for (int i = 0; i < n_split; ++i) {
        const float m = part_ml[((size_t)i * n_head + h) * 2];
        if (m > -INFINITY) acc += __expf(m - M) * part_acc[((size_t)i * n_head + h) * head_dim + d];
    }
    const float gg = gate[d];
    const float o = acc * inv / (1.f + __expf(-gg));
    out[(size_t)h * head_dim + d] = o;
    wave_q8(o, xq, xd, (int)((((size_t)tq * n_head + h) * head_dim + d) >> 5));
}

// ---------------------------------------------------------------------------
// Split-K decode / verify attention on WMMA (gfx11 port of the kernels.hip
// attn_wsplit; fragment code from attn_prefill_impl here). Block (kv head g,
// split, group z): the group's queries x the GQA heads of g are up to 64
// columns, wave cg takes columns [16 cg, 16 cg + 16); every column walks the
// split's keys in 32-key chunks in order with masking, so its partial is the
// same whether it runs alone (decode) or grouped (verify). Partials in the
// attn_split layout for attn_combine / attn_combine_q8.
// ---------------------------------------------------------------------------
#define AW_K 32
#define AW_D 256
#define AW_SINK 256  // draft window: positions [0, AW_SINK) are always attended
struct AwGroups {  // kMaxVerifyRows (32) groups, as include/whirl/kernels_abi.h
    int first[32];
    int count[32];
};

template <int CG, bool Q8>
__device__ __forceinline__ void attn_wsplit_impl(const float *q, const KvArgs &kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                int q_stride, const int *pos, float scale, const int *skip, AwGroups groups) {
    // no implicit fma contraction: every instantiation (decode 128-thread,
    // verify 256-thread) must round each column identically
#pragma clang fp contract(off)
    if (skip != nullptr && *skip) return;
    __shared__ _Float16 Vt[AW_D][AW_K + 8];
    __shared__ _Float16 Qs[16 * CG][AW_D + 8];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int g = blockIdx.x, split = blockIdx.y, n_split = gridDim.y, z = blockIdx.z;
    const int grp = n_head / n_kv;
    const int first = groups.first[z], count = groups.count[z];
    const int ncol = count * grp;
    const int tb = kv_tb(kv, first);
    const int n_pos = pos[first + count - 1] + 1;
    int per = (n_pos + n_split * 64 - 1) / (n_split * 64) * 64;
    int b0 = split * per, b1 = min(n_pos, b0 + per);
    // MTP draft window (kv.win = W / 64 | (threshold / 1024) << 16; the trunk passes 0): once
    // n_pos >= threshold, split 0 takes the sink [0, AW_SINK) and the other splits share the last
    // W positions from a 64-aligned start; keys between are skipped. A window that reaches the
    // sink leaves the ranges above unchanged (same bits as without it). Same as kernels/attn.hip.
    if (kv.win != 0 && n_split > 1) {
        const int w = (kv.win & 0xffff) * 64, min_pos = (kv.win >> 16) * 1024;
        const int w0 = max(AW_SINK, (n_pos - w) & ~63);
        if (n_pos >= min_pos && w0 > AW_SINK) {
            if (split == 0) {
                b0 = 0;
                b1 = AW_SINK;
            } else {
                per = (n_pos - w0 + (n_split - 1) * 64 - 1) / ((n_split - 1) * 64) * 64;
                b0 = w0 + (split - 1) * per;
                b1 = min(n_pos, b0 + per);
            }
        }
    }
    const size_t ml_q = (size_t)n_split * n_head * 2, acc_q = (size_t)n_split * n_head * AW_D;
    const bool cw = wid * 16 < ncol;
    const int col = wid * 16 + lane % 16;
    const bool cvalid = cw && col < ncol;
    const int colc = cvalid ? col : 0;
    const int t = first + colc / grp, h = g * grp + colc % grp;
    const int my_pos = cvalid ? pos[t] : -1;
    if (b0 >= n_pos) {
        if (cvalid && lane < 16) {
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = -INFINITY;
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = 0.f;
        }
        return;
    }
    // Q^T (f16) for the group's columns in LDS; column c = (query c / grp, head c % grp)
    for (int i = threadIdx.x; i < 16 * CG * (AW_D / 8); i += blockDim.x) {
        const int c = i / (AW_D / 8), d8 = (i % (AW_D / 8)) * 8;
        half8 v8;
        if (c < ncol) {
            const float *qr = q + ((size_t)(first + c / grp) * n_head + g * grp + c % grp) * q_stride + d8;
            const float4 a = *(const float4 *)qr, b = *(const float4 *)(qr + 4);
            v8 = (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                         (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) v8[e] = (_Float16)0.f;
        }
        *(half8 *)&Qs[c][d8] = v8;
    }
    float8 o[AW_D / 16];
#pragma unroll
    for (int dt = 0; dt < AW_D / 16; ++dt) o[dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    for (int k0 = b0; k0 < b1; k0 += AW_K) {
        const int dl = kv_delta(kv, tb, k0);  // AW_K-key tiles never straddle a page
        __syncthreads();
        if (threadIdx.x < 128) {
            const int tv = threadIdx.x;
            const int kb = tv / (AW_D / 8);
            const int d8 = (tv % (AW_D / 8)) * 8;
            half8 rows[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int key = k0 + kb * 8 + r;
                if (key < b1) {
                    rows[r] = kv_ld8<Q8>(kv.v, kv.vs, ((size_t)(key + dl) * n_kv + g) * AW_D + d8);
                } else {
#pragma unroll
                    for (int e2 = 0; e2 < 8; ++e2) rows[r][e2] = (_Float16)0.f;
                }
            }
            typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
            const int tb = (kb / 2) * 16 + (kb % 2) * 4;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                half4_t ev, od;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    ev[r] = rows[2 * r][c];
                    od[r] = rows[2 * r + 1][c];
                }
                *(half4_t *)&Vt[d8 + c][tb] = ev;
                *(half4_t *)&Vt[d8 + c][tb + 8] = od;
            }
        }
        __syncthreads();
        if (!cw) continue;
        float8 sc[AW_K / 16];
#pragma unroll
        for (int kt = 0; kt < AW_K / 16; ++kt) sc[kt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kt = 0; kt < AW_K / 16; ++kt) {
            const int krow = min(k0 + kt * 16 + lane % 16, b1 - 1);  // clamp (masked below)
            const size_t kr = ((size_t)(krow + dl) * n_kv + g) * AW_D;
#pragma unroll
            for (int st = 0; st < AW_D / 16; ++st) {
                const half16 qb = ld_frag16(&Qs[wid * 16 + lane % 16][16 * st]);
                const half16 a = kv_ld16<Q8>(kv.k, kv.ks, kr + 16 * st);
                sc[kt] = WMMA16(a, qb, sc[kt]);
            }
        }
        float mx = -INFINITY;
#pragma unroll
        for (int kt = 0; kt < AW_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int key = k0 + kt * 16 + CROW(lane, v);
                const float x = (key <= my_pos && key < b1) ? sc[kt][v] * scale : -INFINITY;
                sc[kt][v] = x;
                mx = fmaxf(mx, x);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16, WARP));
        const float m_new = fmaxf(m, mx);
        const float alpha = (m == -INFINITY) ? 0.f : __expf(m - m_new);
        float rs = 0.f;
        half8 pt[AW_K / 16];
#pragma unroll
        for (int kt = 0; kt < AW_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float p = (sc[kt][v] == -INFINITY) ? 0.f : __expf(sc[kt][v] - m_new);
                rs += p;
                pt[kt][v] = (_Float16)p;
            }
        rs += __shfl_xor(rs, 16, WARP);
        l = __builtin_fmaf(l, alpha, rs);
        m = m_new;
#pragma unroll
        for (int dt = 0; dt < AW_D / 16; ++dt)
#pragma unroll
            for (int v = 0; v < 8; ++v) o[dt][v] *= alpha;
        half16 pb[AW_K / 16];
#pragma unroll
        for (int kt = 0; kt < AW_K / 16; ++kt) pb[kt] = join_halves(pt[kt], lane);
#pragma unroll
        for (int dt = 0; dt < AW_D / 16; ++dt)
#pragma unroll
            for (int kt = 0; kt < AW_K / 16; ++kt) {
                const half16 a = ld_frag16(&Vt[dt * 16 + lane % 16][kt * 16]);
                o[dt] = WMMA16(a, pb[kt], o[dt]);
            }
    }
    if (!cvalid) return;
    if (lane < 16) {
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = m;
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = l;
    }
    float *pa = part_acc + t * acc_q + ((size_t)split * n_head + h) * AW_D;
#pragma unroll
    for (int dt = 0; dt < AW_D / 16; ++dt)
#pragma unroll
        for (int v = 0; v < 8; ++v) pa[dt * 16 + CROW(lane, v)] = o[dt][v];
}
extern "C" __global__ void __launch_bounds__(128)
    attn_wsplit1(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                 int q_stride, const int *pos, float scale, const int *skip, AwGroups groups) {
    attn_wsplit_impl<1, false>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);
}
extern "C" __global__ void __launch_bounds__(128)
    attn_wsplit1_q8(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                 int q_stride, const int *pos, float scale, const int *skip, AwGroups groups) {
    attn_wsplit_impl<1, true>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);
}



// ---------------------------------------------------------------------------
// Prefill attention on WMMA (flash-attention style), causal, head_dim 256.
// Block = 128 queries (8 waves x 16) of one head. Per 32-key block:
//   S^T = K Q^T   (A = K rows from LDS, B = Q^T fragments kept in registers)
//   online softmax per query column (lane % 16), P^T stays in registers and
//   is directly the B operand of
//   O^T += V^T P^T (A = V^T rows from LDS, stored transposed).
// Output gated by sigmoid(gate) like attn_decode. K/V cache is f16 (or q8).
// Tile loads use all 256 threads: K as 4 x 16-byte row pieces per thread; V as
// 4 dims x 8 keys per thread (8-byte loads), transposed in registers and stored
// with 8-byte writes into V^T rows whose 16-byte chunks are XOR-swizzled by row
// bits 3..4 (fa_vt_off): the transposed stores are ~4-way instead of ~16-way
// bank conflicted and the fragment reads stay conflict-free. Keys past the
// block's last key are zero, as before (gfx11 WMMA results depend on the V
// values where P = 0, so the zero-fill boundary is part of the output bits).
// ---------------------------------------------------------------------------
#define FA_Q 128
#define FA_K 32
#define FA_D 256
#define FA_VS 40  // V^T row stride (halves)

typedef _Float16 fa_h4 __attribute__((ext_vector_type(4)));
typedef char fa_q4 __attribute__((ext_vector_type(4)));
template <bool Q8> __device__ __forceinline__ fa_h4 kv_ld4(const void *base, const _Float16 *sc, size_t e) {
    if constexpr (Q8) {
        const fa_q4 q = *(const fa_q4 *)((const int8_t *)base + e);
        const _Float16 s = sc[e >> 5];
        fa_h4 h;
#pragma unroll
        for (int i = 0; i < 4; ++i) h[i] = (_Float16)q[i] * s;
        return h;
    } else {
        return *(const fa_h4 *)((const _Float16 *)base + e);
    }
}
// V^T element (dim row, permuted key position h)
__device__ __forceinline__ int fa_vt_off(int row, int h) { return row * FA_VS + ((((h >> 3) ^ (row >> 3)) & 3) << 3) + (h & 7); }

// q8 KV: the previous tile loader (the 256-thread loader below spills with the
// q8 dequantization); same arithmetic, same bits.
template <int MODE, bool Q8>
__device__ __forceinline__ void attn_prefill_impl_q8kv(const float *q, const KvArgs &kv, float *out, int n_head, int n_kv,
                      int q_stride, const int *pos, int n_tok, float scale, int h0) {
    const int tb = kv.tab0;  // one sequence per launch
    __shared__ _Float16 Ks[FA_K][FA_D + 8];
    __shared__ _Float16 Vt[FA_D][FA_K + 8];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    // h0: first head of this launch (the host splits long prefills over head ranges)
    const int h = blockIdx.y + h0, kvh = h / (n_head / n_kv);
    const int q0 = blockIdx.x * FA_Q;
    const int pos0 = pos[0];
    const int my_q = q0 + wid * 16 + lane % 16;       // this lane's query column
    const int my_qc = min(my_q, n_tok - 1);            // clamp for loads
    const int my_pos = pos0 + my_qc;

    // Q^T fragments (B operand): column = query, k = dims 16s + 8*(lane/16) ..
    half8 qf[FA_D / 16];
    {
        const float *qr = q + ((size_t)my_qc * n_head + h) * q_stride;
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
            const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
            qf[st] = (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                             (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
        }
    }
    float8 o[FA_D / 16];
#pragma unroll
    for (int dt = 0; dt < FA_D / 16; ++dt) o[dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;

    const int last_q = min(q0 + FA_Q, n_tok) - 1;
    const int n_keys = pos0 + last_q + 1;
    for (int k0 = 0; k0 < n_keys; k0 += FA_K) {
        const int dl = kv_delta(kv, tb, k0);  // FA_K-key tiles never straddle a page
        __syncthreads();
        // load K (row-major) and V (transposed) tiles; the cache is already f16.
        // Threads [0,128): K, 8 x 16-byte rows each. Threads [128,256): V as
        // 8 keys x 8 dims blocks, transposed in registers, 8 x 16-byte stores.
        if (MODE != 1 || k0 == 0) {
            if (threadIdx.x < 128) {
#pragma unroll
                for (int e = 0; e < FA_K * FA_D / 8 / 128; ++e) {
                    const int i = threadIdx.x + e * 128;
                    const int kr = i / (FA_D / 8), d8 = (i % (FA_D / 8)) * 8;
                    const int key = k0 + kr;
                    half8 kv8;
                    if (key < n_keys) {
                        kv8 = kv_ld8<Q8>(kv.k, kv.ks, ((size_t)(key + dl) * n_kv + kvh) * FA_D + d8);
                    } else {
#pragma unroll
                        for (int q = 0; q < 8; ++q) kv8[q] = (_Float16)0.f;
                    }
                    *(half8 *)&Ks[kr][d8] = kv8;
                }
            } else {
                const int tv = threadIdx.x - 128;              // 128 threads
                const int kb = tv / (FA_D / 8);                 // key group (8 keys): 0..FA_K/8-1
                const int d8 = (tv % (FA_D / 8)) * 8;           // dim group
                half8 rows[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int key = k0 + kb * 8 + r;
                    if (key < n_keys) {
                        rows[r] = kv_ld8<Q8>(kv.v, kv.vs, ((size_t)(key + dl) * n_kv + kvh) * FA_D + d8);
                    } else {
#pragma unroll
                        for (int q = 0; q < 8; ++q) rows[r][q] = (_Float16)0.f;
                    }
                }
                // keys kb*8 + r: tile kb / 2, j = (kb % 2) * 8 + r ->
                // position (r % 2) * 8 + (kb % 2) * 4 + r / 2 within the tile
                typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
                const int tb = (kb / 2) * 16 + (kb % 2) * 4;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    half4_t ev, od;
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        ev[r] = rows[2 * r][c];
                        od[r] = rows[2 * r + 1][c];
                    }
                    *(half4_t *)&Vt[d8 + c][tb] = ev;
                    *(half4_t *)&Vt[d8 + c][tb + 8] = od;
                }
            }
        }
        __syncthreads();
        // S^T tiles: rows = keys kt*16 + 8*(lane/16) + v, column = my query
        // gfx11: Q^T keeps only this half's 8 dims per st in registers; the
        // full 16-dim B fragment is rebuilt with one cross-half exchange.
        float8 sc[FA_K / 16];
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt) sc[kt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const half16 qb = join_halves(qf[st], lane);
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt) {
                const half16 a = ld_frag16(&Ks[kt * 16 + lane % 16][16 * st]);
                sc[kt] = WMMA16(a, qb, sc[kt]);
            }
        }
        // causal mask + online softmax (per lane = per query; partner lane ^16)
        float mx = -INFINITY;
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int key = k0 + kt * 16 + CROW(lane, v);
                const float x = (key <= my_pos) ? sc[kt][v] * scale : -INFINITY;
                sc[kt][v] = x;
                mx = fmaxf(mx, x);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16, WARP));
        const float m_new = fmaxf(m, mx);
        const float alpha = (m == -INFINITY) ? 0.f : __expf(m - m_new);
        float rs = 0.f;
        half8 pt[FA_K / 16];
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float p = (sc[kt][v] == -INFINITY) ? 0.f : __expf(sc[kt][v] - m_new);
                rs += p;
                pt[kt][v] = (_Float16)p;
            }
        rs += __shfl_xor(rs, 16, WARP);
        l = l * alpha + rs;
        m = m_new;
        if (MODE != 2) {
#pragma unroll
        for (int dt = 0; dt < FA_D / 16; ++dt)
#pragma unroll
            for (int v = 0; v < 8; ++v) o[dt][v] *= alpha;
        }
        // O^T += V^T P^T. This lane holds keys 2v + (lane / 16) of P^T; the
        // B fragment takes k = [even keys; odd keys], matching the permuted
        // key order Vt is stored in.
        // one key tile's B fragment live at a time (register pressure: the
        // kernel spilled); per dt the accumulation order is unchanged
        // (kt = 0 then 1), so results are identical
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt) {
            const half16 pb = join_halves(pt[kt], lane);
#pragma unroll
            for (int dt = 0; dt < FA_D / 16; ++dt) {
                const half16 a = ld_frag16(&Vt[dt * 16 + lane % 16][kt * 16]);
                if (MODE != 3) o[dt] = WMMA16(a, pb, o[dt]);
                // pin: one V fragment in flight at a time (else all 16 are
                // preloaded and the kernel spills)
                asm volatile("" : "+v"(o[dt]));
            }
        }
    }
    if (my_q >= n_tok) return;
    const float inv = 1.f / l;
    const float *gate = q + ((size_t)my_q * n_head + h) * q_stride + FA_D;
    float *orow = out + (size_t)my_q * n_head * FA_D + (size_t)h * FA_D;
#pragma unroll
    for (int dt = 0; dt < FA_D / 16; ++dt)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int d = dt * 16 + CROW(lane, v);
            orow[d] = o[dt][v] * inv / (1.f + __expf(-gate[d]));
        }
}
template <int MODE, bool Q8>
__device__ __forceinline__ void attn_prefill_impl(const float *q, const KvArgs &kv, float *out, int n_head, int n_kv,
                      int q_stride, const int *pos, int n_tok, float scale, int h0) {
    if constexpr (Q8) {
        attn_prefill_impl_q8kv<MODE, Q8>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);
        return;
    }
    const int tb = kv.tab0;  // one sequence per launch
    __shared__ _Float16 Ks[FA_K][FA_D + 8];
    __shared__ _Float16 Vt[FA_D * FA_VS];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, tid = threadIdx.x;
    // h0: first head of this launch (the host splits long prefills over head ranges)
    const int h = blockIdx.y + h0, kvh = h / (n_head / n_kv);
    const int q0 = blockIdx.x * FA_Q;
    const int pos0 = pos[0];
    const int my_q = q0 + wid * 16 + lane % 16;       // this lane's query column
    const int my_qc = min(my_q, n_tok - 1);            // clamp for loads
    const int my_pos = pos0 + my_qc;

    // Q^T fragments (B operand): column = query, k = dims 16s + 8*(lane/16) ..
    half8 qf[FA_D / 16];
    {
        const float *qr = q + ((size_t)my_qc * n_head + h) * q_stride;
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
            const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
            qf[st] = (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                             (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
        }
    }
    float8 o[FA_D / 16];
#pragma unroll
    for (int dt = 0; dt < FA_D / 16; ++dt) o[dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;

    const int last_q = min(q0 + FA_Q, n_tok) - 1;
    const int n_keys = pos0 + last_q + 1;
    // per-thread load / store geometry (loop invariant and kept small: the kernel
    // sits at 256 VGPRs, and per-element 64-bit addresses made it spill)
    const int vd4 = (tid % 64) * 4, vkb = tid / 64;                      // V: dims vd4 .. +3 of keys vkb*8 .. +7
    const int vtb = (vkb / 2) * 16 + (vkb % 2) * 4;                      // permuted position of its even keys
    const int vsA = fa_vt_off(vd4, vtb), vsB = fa_vt_off(vd4, vtb + 8);  // rows vd4 + c share row >> 3
    const int rstep = n_kv * FA_D;
    const int kro = kvh * FA_D + (tid % 32) * 8, vro = kvh * FA_D + vd4;
    half8 kr8[4];
    fa_h4 vr[8];
    // keys >= n_keys: read the last key's row (same page), store zeros
    auto loadK = [&](int k0) {
        const int dl = kv_delta(kv, tb, k0);  // FA_K-key tiles never straddle a page
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int key = k0 + tid / 32 + 8 * e;
            const int kc = min(key, n_keys - 1);
            const half8 x = kv_ld8<Q8>(kv.k, kv.ks, (size_t)(unsigned)(kc + dl) * rstep + kro);
            const half8 z = {0, 0, 0, 0, 0, 0, 0, 0};
            kr8[e] = key < n_keys ? x : z;
        }
    };
    auto loadV = [&](int k0) {
        const int dl = kv_delta(kv, tb, k0);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int key = k0 + vkb * 8 + r;
            const int kc = min(key, n_keys - 1);
            const fa_h4 x = kv_ld4<Q8>(kv.v, kv.vs, (size_t)(unsigned)(kc + dl) * rstep + vro);
            const fa_h4 z = {0, 0, 0, 0};
            vr[r] = key < n_keys ? x : z;
        }
    };
    for (int k0 = 0; k0 < n_keys; k0 += FA_K) {
        __syncthreads();
        if (MODE != 1 || k0 == 0) {
            loadK(k0);
            loadV(k0);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = tid + e * 256;
                *(half8 *)&Ks[i / 32][(i % 32) * 8] = kr8[e];
            }
            // keys vkb*8 + r: tile vkb / 2, j = (vkb % 2) * 8 + r ->
            // position (r % 2) * 8 + (vkb % 2) * 4 + r / 2 within the tile
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                fa_h4 ev, od;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    ev[r] = vr[2 * r][c];
                    od[r] = vr[2 * r + 1][c];
                }
                *(fa_h4 *)&Vt[vsA + c * FA_VS] = ev;
                *(fa_h4 *)&Vt[vsB + c * FA_VS] = od;
            }
        }
        __syncthreads();
        // S^T tiles: rows = keys kt*16 + 8*(lane/16) + v, column = my query
        // gfx11: Q^T keeps only this half's 8 dims per st in registers; the
        // full 16-dim B fragment is rebuilt with one cross-half exchange.
        float8 sc[FA_K / 16];
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt) sc[kt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const half16 qb = join_halves(qf[st], lane);
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt) {
                const half16 a = ld_frag16(&Ks[kt * 16 + lane % 16][16 * st]);
                sc[kt] = WMMA16(a, qb, sc[kt]);
            }
        }
        // causal mask + online softmax (per lane = per query; partner lane ^16)
        float mx = -INFINITY;
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int key = k0 + kt * 16 + CROW(lane, v);
                const float x = (key <= my_pos) ? sc[kt][v] * scale : -INFINITY;
                sc[kt][v] = x;
                mx = fmaxf(mx, x);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16, WARP));
        const float m_new = fmaxf(m, mx);
        const float alpha = (m == -INFINITY) ? 0.f : __expf(m - m_new);
        float rs = 0.f;
        half8 pt[FA_K / 16];
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float p = (sc[kt][v] == -INFINITY) ? 0.f : __expf(sc[kt][v] - m_new);
                rs += p;
                pt[kt][v] = (_Float16)p;
            }
        rs += __shfl_xor(rs, 16, WARP);
        l = l * alpha + rs;
        m = m_new;
        if (MODE != 2) {
#pragma unroll
        for (int dt = 0; dt < FA_D / 16; ++dt)
#pragma unroll
            for (int v = 0; v < 8; ++v) o[dt][v] *= alpha;
        }
        // O^T += V^T P^T. This lane holds keys 2v + (lane / 16) of P^T; the
        // B fragment takes k = [even keys; odd keys], matching the permuted
        // key order Vt is stored in.
        // one key tile's B fragment live at a time (register pressure: the
        // kernel spilled); per dt the accumulation order is unchanged
        // (kt = 0 then 1), so results are identical
        // V^T fragment reads, row dt*16 + l (l = lane % 16): chunk = (c ^ 2 (dt & 1)) ^ (l / 8)
        const int s0 = (lane % 16) >> 3;
        const _Float16 *vlo = Vt + (lane % 16) * FA_VS + s0 * 8;
        const _Float16 *vhi = Vt + (lane % 16) * FA_VS + 8 - s0 * 8;
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt) {
            const half16 pb = join_halves(pt[kt], lane);
#pragma unroll
            for (int dt = 0; dt < FA_D / 16; ++dt) {
                const int C = (2 * kt) ^ (2 * (dt & 1));
                const half8 lo = *(const half8 *)(vlo + dt * 16 * FA_VS + C * 8);
                const half8 hi = *(const half8 *)(vhi + dt * 16 * FA_VS + C * 8);
                const half16 a = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                if (MODE != 3) o[dt] = WMMA16(a, pb, o[dt]);
                // pin: one V fragment in flight at a time (else all 16 are
                // preloaded and the kernel spills)
                asm volatile("" : "+v"(o[dt]));
            }
        }
    }
    if (my_q >= n_tok) return;
    const float inv = 1.f / l;
    const float *gate = q + ((size_t)my_q * n_head + h) * q_stride + FA_D;
    float *orow = out + (size_t)my_q * n_head * FA_D + (size_t)h * FA_D;
#pragma unroll
    for (int dt = 0; dt < FA_D / 16; ++dt)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int d = dt * 16 + CROW(lane, v);
            orow[d] = o[dt][v] * inv / (1.f + __expf(-gate[d]));
        }
}
#define FA_ENTRY(NAME, MODE, Q8)                                                                    \
    extern "C" __global__ void __launch_bounds__(256)                                               \
        NAME(const float *q, KvArgs kv, float *out, int n_head, int n_kv,                           \
             int q_stride, const int *pos, int n_tok, float scale, int h0) {                        \
        attn_prefill_impl<MODE, Q8>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);     \
    }
FA_ENTRY(attn_prefill_wmma, 0, false)
FA_ENTRY(attn_prefill_wmma_q8, 0, true)
FA_ENTRY(fa_probe_noload, 1, false)
FA_ENTRY(fa_probe_norescale, 2, false)
FA_ENTRY(fa_probe_nopv, 3, false)
