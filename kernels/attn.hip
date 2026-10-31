// WHIRL device code (gfx1201 / RDNA 4): paged KV cache, decode attention, split-K attention, fused attention prep, WMMA split / prefill attention.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

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
#include "shared/attn_kv_page.h"  // also in the gfx1151 set (kernels/gfx1151/attn.hip)
// Store x at element e; the 32 lanes of the wave hold the 32 elements of one
// aligned scale group, lane = e % 32 (q8: scale = amax / 127 rounded to f16,
// q = round(x / scale); q4: see above, q = clamp(round(x / scale), -8, 7)).
#include "shared/attn_attn_decode_entry.h"  // also in the gfx1151 set (kernels/gfx1151/attn.hip)
ATTN_DECODE_ENTRY(attn_decode, false)
ATTN_DECODE_ENTRY(attn_decode_q8, true)
ATTN_DECODE_ENTRY(attn_decode_q8v, 2)
ATTN_DECODE_ENTRY(attn_decode_q4, 3)

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
template <int NQ, int Q8>
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
                    const half8 kh8 = kv_ld8<kv_kf(Q8)>(kv.k, kv.ks, kr + 8 * e8);
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
            // lanes past head_dim (head_dim < 256) load nothing and add nothing: their 8 K
            // values would be past the row (the last pool row: past the K pool) and their
            // qs entries are unwritten LDS (FIX-OVR; head_dim == 256 is unchanged)
            const bool lv = lane * 8 < head_dim;
            for (int p = c0 + wid; p < c1; p += nw) {
                half8 kh8 = (half8){0, 0, 0, 0, 0, 0, 0, 0};
                if (lv) kh8 = kv_ld8<kv_kf(Q8)>(kv.k, kv.ks, ((size_t)(p + dl) * n_kv + g) * head_dim + lane * 8);
                float kv[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) kv[e] = (float)kh8[e];
                for (int hh = 0; hh < nh; ++hh) {
                    float s = 0.f;
                    if (lv)
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
                const float v = kv_ld1<kv_vf(Q8)>(kv.v, kv.vs, ((size_t)(p + dl) * n_kv + g) * head_dim + d);
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

#include "shared/attn_attn_combine.h"  // also in the gfx1151 set (kernels/gfx1151/attn.hip)

// ---------------------------------------------------------------------------
// Fused attention prep: per (head, token) block the q_norm / k_norm RMSNorm,
// NEOX RoPE on the first n_rot dims and, for k heads, the K/V cache store.
// Same float expressions as rmsnorm + rope_neox + kv_store (same block_sum
// order), so the results are unchanged; 5 launches per attention layer -> 1.
// grid = (n_head + n_kv, n_tok), 256 threads, hd <= 512.
// ---------------------------------------------------------------------------
// ROT (q8 KV only): rotate every q and k head by the orthonormal Walsh-Hadamard
// transform after RoPE (q.k is unchanged in exact arithmetic). Outlier
// channels of K get spread over all head dims, so the per-32 int8 scales of
// the stored K fit much better. Every attention path reads q from qf and K
// from the cache written here, so all of them see the rotated pair.
// MR (vision): multi-section RoPE for image prompts (ggml GGML_ROPE_TYPE_IMROPE, the
// Qwen3-VL / Qwen3.5 "mrope_interleaved" layout): rotary pair i takes the h position when
// i % 3 == 1 and i < 3 * s1, the w position when i % 3 == 2 and i < 3 * s2, else t.
// rpos = [row][3] {t, h, w} positions; rpos == nullptr: all three = pos[t] - rdelta.
// The KV row is still pos[t] (the cache index). Same theta expression as the plain path,
// so a row whose three positions are equal gives the plain path's bits at that position.
template <int Q8, bool ROT = false, bool MR = false>
__device__ __forceinline__ void attn_prep_impl(float *qf, float *kk, const float *vv, const float *qw, const float *kw, const KvArgs &kv,
              const int *pos, int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps,
              const int *rpos = nullptr, int rdelta = 0, int s0 = 0, int s1 = 0, int s2 = 0) {
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
    if constexpr (MR) {
        int pt, ph, pw;
        if (rpos != nullptr) {
            pt = rpos[3 * t];
            ph = rpos[3 * t + 1];
            pw = rpos[3 * t + 2];
        } else {
            pt = ph = pw = pos[t] - rdelta;
        }
        for (int i = threadIdx.x; i < half; i += blockDim.x) {
            const int sec = i % 3;
            const int pi = (sec == 1 && i < 3 * s1) ? ph : (sec == 2 && i < 3 * s2) ? pw : pt;
            (void)s0;
            const float p = (float)pi;
            const float theta = p * powf(theta_scale, (float)i);
            float s, c;
            sincosf(theta, &s, &c);
            const float x0 = nv[i], x1 = nv[i + half];
            nv[i] = x0 * c - x1 * s;
            nv[i + half] = x0 * s + x1 * c;
        }
    } else {
    const float p = (float)pos[t];
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float theta = p * powf(theta_scale, (float)i);
        float s, c;
        sincosf(theta, &s, &c);
        const float x0 = nv[i], x1 = nv[i + half];
        nv[i] = x0 * c - x1 * s;
        nv[i + half] = x0 * s + x1 * c;
    }
    }
    __syncthreads();
    if constexpr (ROT) {  // in-place fast Walsh-Hadamard transform, hd a power of 2
        for (int len = 1; len < hd; len <<= 1) {
            for (int i = threadIdx.x; i < hd / 2; i += blockDim.x) {
                const int a = (i / len) * 2 * len + i % len, b = a + len;
                const float x = nv[a], y = nv[b];
                nv[a] = x + y;
                nv[b] = x - y;
            }
            __syncthreads();
        }
        const float rs = rsqrtf((float)hd);
        for (int i = threadIdx.x; i < hd; i += blockDim.x) nv[i] *= rs;
        __syncthreads();
    }
    for (int i = threadIdx.x; i < hd; i += blockDim.x) xv[i] = nv[i];
    if (isq) return;
    const int row = n_kv * hd;
    // single sequence: consecutive positions from pos[0]; batched rows: pos[t]
    const int lp = kv.kvbase == nullptr ? pos[0] + t : pos[t];
    const size_t prow = (size_t)kv_row(kv, kv_tb(kv, t), lp);
    const float *vr = vv + (size_t)t * row + (size_t)h * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {  // hd % 32 == 0: whole waves per scale group
        kv_st32<kv_kf(Q8)>(kv.k, kv.ks, prow * row + (size_t)h * hd + d, nv[d]);
        kv_st32<kv_vf(Q8)>(kv.v, kv.vs, prow * row + (size_t)h * hd + d, vr[d]);
    }
}
#include "shared/attn_attn_prep.h"  // also in the gfx1151 set (kernels/gfx1151/attn.hip)
extern "C" __global__ void __launch_bounds__(256)
    attn_prep_q8h(float *qf, float *kk, const float *vv, const float *qw, const float *kw, KvArgs kv,
              const int *pos, int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps) {
    attn_prep_impl<true, true>(qf, kk, vv, qw, kw, kv, pos, n_head, n_kv, hd, n_rot, theta_scale, eps);
}
// q8v: K f16 (not rotated), V q8
extern "C" __global__ void __launch_bounds__(256)
    attn_prep_q8v(float *qf, float *kk, const float *vv, const float *qw, const float *kw, KvArgs kv,
              const int *pos, int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps) {
    attn_prep_impl<2>(qf, kk, vv, qw, kw, kv, pos, n_head, n_kv, hd, n_rot, theta_scale, eps);
}
// q4 (fast mode): 4-bit K and V (format above), q and k Hadamard-rotated as q8h
extern "C" __global__ void __launch_bounds__(256)
    attn_prep_q4(float *qf, float *kk, const float *vv, const float *qw, const float *kw, KvArgs kv,
              const int *pos, int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps) {
    attn_prep_impl<3, true>(qf, kk, vv, qw, kw, kv, pos, n_head, n_kv, hd, n_rot, theta_scale, eps);
}
// vision: the same five, with multi-section RoPE positions (see attn_prep_impl MR)
#define ATTN_PREP_M(NAME, Q8, ROT)                                                                                        \
    extern "C" __global__ void __launch_bounds__(256)                                                                     \
    NAME(float *qf, float *kk, const float *vv, const float *qw, const float *kw, KvArgs kv, const int *pos,               \
         int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps, const int *rpos, int rdelta,               \
         int s0, int s1, int s2) {                                                                                        \
        attn_prep_impl<Q8, ROT, true>(qf, kk, vv, qw, kw, kv, pos, n_head, n_kv, hd, n_rot, theta_scale, eps, rpos,       \
                                      rdelta, s0, s1, s2);                                                                \
    }
ATTN_PREP_M(attn_prep_m, false, false)
ATTN_PREP_M(attn_prep_m_q8, true, false)
ATTN_PREP_M(attn_prep_m_q8h, true, true)
ATTN_PREP_M(attn_prep_m_q8v, 2, false)
ATTN_PREP_M(attn_prep_m_q4, 3, true)

// vision: rows' multi-section RoPE positions {t, h, w} for a small batch (by value)
struct Rpos16 {
    int v[48];
};
#include "shared/attn_set_rpos.h"  // also in the gfx1151 set (kernels/gfx1151/attn.hip)
// struct AwGroups: kernels/kernel_args.h

template <int CG, int Q8>
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
    const int n_pos = pos[first + count - 1] + 1;  // the group's last query
    int per = (n_pos + n_split * 64 - 1) / (n_split * 64) * 64;
    int b0 = split * per, b1 = min(n_pos, b0 + per);
    // MTP draft window (kv.win = W / 64 | (threshold / 1024) << 16; the trunk passes 0): once
    // n_pos >= threshold, split 0 takes the sink [0, AW_SINK) and the other splits share the last
    // W positions from a 64-aligned start; keys between are skipped. A window that reaches the
    // sink leaves the ranges above unchanged (same bits as without it).
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
    // this lane's column
    const bool cw = wid * 16 < ncol;              // computing wave
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
    // Q^T fragments (B operand): column = (query, head), k = dims 16s + 8*(lane/16) ..
    // Q^T (f16) for the group's columns in LDS; column c = (query c / grp, head c % grp)
    for (int i = threadIdx.x; i < 16 * CG * (AW_D / 8); i += blockDim.x) {
#include "shared/attn_qs_f16.h"  // Qs[c][d8..+8) = f16 Q of column c (ADR-0005)
    }
    float8 o[AW_D / 16];
#pragma unroll
    for (int dt = 0; dt < AW_D / 16; ++dt) o[dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    // V^T staging: thread (kb, d8) < 128 loads keys kb*8..+8 at dims d8..d8+8
    // (4 key groups x 32 dim groups), transposes in registers
    const int kb = threadIdx.x / (AW_D / 8), d8 = (threadIdx.x % (AW_D / 8)) * 8;
    for (int k0 = b0; k0 < b1; k0 += AW_K) {
        const int dl = kv_delta(kv, tb, k0);  // AW_K-key tiles never straddle a page
        __syncthreads();
        if (threadIdx.x < 128) {  // 4 key groups x 32 dim groups
            half8 rows[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int key = k0 + kb * 8 + r;
                if (key < b1) {
                    rows[r] = kv_ld8<kv_vf(Q8)>(kv.v, kv.vs, ((size_t)(key + dl) * n_kv + g) * AW_D + d8);
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) rows[r][e] = (_Float16)0.f;
                }
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                half8 cv;
#pragma unroll
                for (int r = 0; r < 8; ++r) cv[r] = rows[r][c];
                *(half8 *)&Vt[d8 + c][kb * 8] = cv;
            }
        }
        __syncthreads();
        if (!cw) continue;
        // S^T tiles: rows = keys kt*16 + 8*(lane/16) + v, column = this lane's column
        float8 sc[AW_K / 16];
#pragma unroll
        for (int kt = 0; kt < AW_K / 16; ++kt) {
            sc[kt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
            const int krow = min(k0 + kt * 16 + lane % 16, b1 - 1);  // clamp (masked below)
            const size_t kr = ((size_t)(krow + dl) * n_kv + g) * AW_D + 8 * (lane / 16);
            // q8 K: the row's 8 scales in one load (step st uses scale st / 2)
            half8 ks8;
            if constexpr (Q8 == 1) ks8 = *(const half8 *)(kv.ks + (kr >> 5));
#pragma unroll
            for (int st = 0; st < AW_D / 16; ++st) {
                half8 a;
                if constexpr (Q8 == 1)
                    a = kv_dq8(*(const kvq8_t *)((const int8_t *)kv.k + kr + 16 * st), ks8[st >> 1]);
                else
                    a = kv_ld8<kv_kf(Q8)>(kv.k, kv.ks, kr + 16 * st);
                const half8 qb = *(const half8 *)&Qs[wid * 16 + lane % 16][16 * st + 8 * (lane / 16)];
                sc[kt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, qb, sc[kt]);
            }
        }
        float mx = -INFINITY;
#pragma unroll
        for (int kt = 0; kt < AW_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int key = k0 + kt * 16 + 8 * (lane / 16) + v;
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
#pragma unroll
        for (int dt = 0; dt < AW_D / 16; ++dt)
#pragma unroll
            for (int kt = 0; kt < AW_K / 16; ++kt) {
                const half8 a = *(const half8 *)&Vt[dt * 16 + lane % 16][kt * 16 + 8 * (lane / 16)];
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, pt[kt], o[dt]);
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
        for (int v = 0; v < 8; ++v) pa[dt * 16 + 8 * (lane / 16) + v] = o[dt][v];
}
#include "shared/attn_attn_wsplit1.h"  // also in the gfx1151 set (kernels/gfx1151/attn.hip)
// Up to 32 columns per group (27B: 5 verify rows x 6 GQA heads): waves 0 and 1 each
// compute 16 columns on the same staged V^T tile, so a sequence's K/V is read once
// for up to 5 verify rows instead of once per 2. Per column the arithmetic is the
// CG = 1 kernel's (same code path), so the partials are the same bits.
extern "C" __global__ void __launch_bounds__(128)
    attn_wsplit2(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                 int q_stride, const int *pos, float scale, const int *skip, AwGroups groups) {
    attn_wsplit_impl<2, false>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);
}
extern "C" __global__ void __launch_bounds__(128)
    attn_wsplit2_q8(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                 int q_stride, const int *pos, float scale, const int *skip, AwGroups groups) {
    attn_wsplit_impl<2, true>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);
}
extern "C" __global__ void __launch_bounds__(128)
    attn_wsplit2_q8v(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                 int q_stride, const int *pos, float scale, const int *skip, AwGroups groups) {
    attn_wsplit_impl<2, 2>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);
}
extern "C" __global__ void __launch_bounds__(128)
    attn_wsplit2_q4(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                 int q_stride, const int *pos, float scale, const int *skip, AwGroups groups) {
    attn_wsplit_impl<2, 3>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);
}



// ---------------------------------------------------------------------------
// attn_dq4: q4 KV decode / verify attention for fast mode (ported from
// kernels/gfx1151/attn.hip; v_dot4_i32_iu8 and v_dot2_f32_f16 exist on gfx12 as well). Same
// arguments, grid (kv head g, split, group z) and partial layout as attn_wsplit1
// (attn_combine / attn_combine_q8 unchanged), one query per group (the host uses it
// with caps.attn_group1), GQA group <= DQ_C heads, head_dim 256, 256 threads.
// Not bitwise equal to attn_wsplit1_q4 (lossy fast-mode path, KL-gated):
//  * scores: q (already Hadamard-rotated by attn_prep_q4) is quantized per 32 values to
//    int8 (scale amax / 127) in LDS, K nibbles stay unsigned 0..15 and go straight into
//    v_dot4_i32_iu8 (int8 q x uint8 k); per 32-group the exact integer sum minus 8 * sum(q)
//    is scaled once by q scale x k scale (no per-element dequantization);
//  * every wave walks its own 32-key tiles (lane = key for the scores) with its own
//    online softmax (base 2), no block barrier in the key loop;
//  * P.V: lane owns 8 dims of the 32-key tile; key pairs (p0 sv0, p1 sv1) in f16 against
//    the exact (n - 8) f16 pairs of the two keys through v_dot2_f32_f16 (f32 accumulate);
//  * the waves' partials are merged in a fixed order through LDS (deterministic).
// Split size: >= DQ_NW * 32 keys (each wave gets a tile) and at most DQ_MAXS splits carry keys
// (measured on the 8060S at 64k / 128k: 64-128 active splits beat 256 by 5-15%); the other
// blocks of the fixed (graph-captured) grid exit at once.
// ---------------------------------------------------------------------------
#include "shared/attn_dq_nw.h"  // also in the gfx1151 set (kernels/gfx1151/attn.hip)
extern "C" __global__ void __launch_bounds__(256)
    attn_dq4(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv, int q_stride, const int *pos,
             float scale, const int *skip, AwGroups groups) {
    if (skip != nullptr && *skip) return;
    __shared__ __align__(16) uint2 Q8[DQ_C][32];               // col, 8-value chunk: int8 x4 of the even / odd values
    __shared__ float Qsc[DQ_C][8];                             // q scale per col, 32-group
    __shared__ int Qs8[DQ_C][8];                               // 8 * sum of the group's q ints
    __shared__ __align__(16) _Float16 Pw[DQ_NW][16][DQ_C][2];  // wave, key pair, col: p of the pair's keys
    __shared__ __align__(16) _Float16 Sv[DQ_NW][16][8][2];     // wave, key pair, group: V scales of the pair's keys
    __shared__ float Rm[DQ_NW][DQ_C], Rl[DQ_NW][DQ_C];
    __shared__ __align__(16) float Acc[DQ_C][AW_D];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    // gfx1201: block order with the query group fastest, so the verify rows of a sequence read the
    // same K / V tiles at about the same time (L2 hits instead of one DRAM pass per row); a
    // bijection of the grid, every block's partial is unchanged
    const int lin = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const int z = lin % gridDim.z, gs = lin / gridDim.z;
    const int g = gs % gridDim.x, split = gs / gridDim.x, n_split = gridDim.y;
    const int grp = n_head / n_kv;
    const int t = groups.first[z];
    const int tb = kv_tb(kv, t);
    const int n_pos = pos[t] + 1;
    int per = dq_per(n_pos, n_split);
    int b0 = split * per, b1 = min(n_pos, b0 + per);
    if (kv.win != 0 && n_split > 1) {  // MTP draft window: same key set as attn_wsplit1
        const int w = (kv.win & 0xffff) * 64, min_pos = (kv.win >> 16) * 1024;
        const int w0 = max(AW_SINK, (n_pos - w) & ~63);
        if (n_pos >= min_pos && w0 > AW_SINK) {
            if (split == 0) {
                b0 = 0;
                b1 = AW_SINK;
            } else {
                per = dq_per(n_pos - w0, n_split - 1);
                b0 = w0 + (split - 1) * per;
                b1 = min(n_pos, b0 + per);
            }
        }
    }
    const size_t ml_q = (size_t)n_split * n_head * 2, acc_q = (size_t)n_split * n_head * AW_D;
    if (b0 >= n_pos) {
        if (threadIdx.x < grp) {
            const int h = g * grp + threadIdx.x;
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = -INFINITY;
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = 0.f;
        }
        return;
    }
    {
        // q -> int8 per 32 values: wave = column, lane = values [8 lane, 8 lane + 8)
        const int c = wid;
        float x[8];
        if (c < grp) {
            const float *qr = q + ((size_t)t * n_head + g * grp + c) * q_stride + 8 * lane;
            const float4 a = *(const float4 *)qr, b = *(const float4 *)(qr + 4);
            x[0] = a.x;
            x[1] = a.y;
            x[2] = a.z;
            x[3] = a.w;
            x[4] = b.x;
            x[5] = b.y;
            x[6] = b.z;
            x[7] = b.w;
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) x[e] = 0.f;
        }
        float am = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) am = fmaxf(am, fabsf(x[e]));
        am = fmaxf(am, __shfl_xor(am, 1, WARP));
        am = fmaxf(am, __shfl_xor(am, 2, WARP));
        const float qs = am * (1.f / 127.f), inv = am > 0.f ? 127.f / am : 0.f;
        int qi[8], sum = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            qi[e] = max(-127, min(127, (int)rintf(x[e] * inv)));
            sum += qi[e];
        }
        sum += __shfl_xor(sum, 1, WARP);
        sum += __shfl_xor(sum, 2, WARP);
        uint2 pk;
        pk.x = (uint32_t)(qi[0] & 255) | (uint32_t)(qi[2] & 255) << 8 | (uint32_t)(qi[4] & 255) << 16 | (uint32_t)(qi[6] & 255) << 24;
        pk.y = (uint32_t)(qi[1] & 255) | (uint32_t)(qi[3] & 255) << 8 | (uint32_t)(qi[5] & 255) << 16 | (uint32_t)(qi[7] & 255) << 24;
        Q8[c][lane] = pk;
        if ((lane & 3) == 0) {
            Qsc[c][lane >> 2] = qs;
            Qs8[c][lane >> 2] = 8 * sum;
        }
    }
    __syncthreads();
    const float sl2 = scale * 1.4426950408889634f;  // scores in base-2 units
    float m[DQ_C], ls[DQ_C], o[DQ_C][8];
#pragma unroll
    for (int c = 0; c < DQ_C; ++c) {
        m[c] = -INFINITY;
        ls[c] = 0.f;
#pragma unroll
        for (int d = 0; d < 8; ++d) o[c][d] = 0.f;
    }
    const uint32_t M4 = 0x0f0f0f0fu;
    const int gl = lane >> 2;  // the 32-group of this lane's P.V dims
    for (int k0 = b0 + wid * 32; k0 < b1; k0 += DQ_NW * 32) {
        const int dl = kv_delta(kv, tb, k0);  // 32-key tiles never straddle a page
        const int key = k0 + lane;
        const bool valid = key < b1;
        const size_t r = (size_t)(min(key, b1 - 1) + dl) * n_kv + g;
        uint4 kw[8];
        const uint4 *kp = (const uint4 *)((const uint8_t *)kv.k + r * (AW_D / 2));
#pragma unroll
        for (int i = 0; i < 8; ++i) kw[i] = kp[i];
        const uint4 ksw = *(const uint4 *)(kv.ks + r * (AW_D / 32));
        const uint4 vsw = *(const uint4 *)(kv.vs + r * (AW_D / 32));
        uint32_t vw[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const size_t rv = (size_t)(min(k0 + j, b1 - 1) + dl) * n_kv + g;
            vw[j] = *(const uint32_t *)((const uint8_t *)kv.v + rv * (AW_D / 2) + 4 * lane);
        }
        // scores (lane = key). zo: an opaque 0 so the q reads stay in the loop (hoisted, the
        // 512 q words and their scales would spill)
        int zo;
        asm volatile("s_mov_b32 %0, 0" : "=s"(zo));
        float s[DQ_C];
#pragma unroll
        for (int c = 0; c < DQ_C; ++c) s[c] = 0.f;
        _Float16 ksh[8];
        __builtin_memcpy(ksh, &ksw, 16);
#pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
            const uint32_t w4[4] = {kw[gi].x, kw[gi].y, kw[gi].z, kw[gi].w};
            uint32_t lo[4], hi[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                lo[j] = w4[j] & M4;
                hi[j] = (w4[j] >> 4) & M4;
            }
            const float sk = (float)ksh[gi];
#pragma unroll
            for (int c = 0; c < DQ_C; ++c) {
                const uint4 qa = *(const uint4 *)&Q8[c][4 * gi + zo], qb = *(const uint4 *)&Q8[c][4 * gi + 2 + zo];
                int S = 0;
                S = __builtin_amdgcn_sudot4(true, (int)qa.x, false, (int)lo[0], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qa.y, false, (int)hi[0], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qa.z, false, (int)lo[1], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qa.w, false, (int)hi[1], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qb.x, false, (int)lo[2], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qb.y, false, (int)hi[2], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qb.z, false, (int)lo[3], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qb.w, false, (int)hi[3], S, false);
                s[c] = fmaf((float)(S - Qs8[c][gi + zo]), Qsc[c][gi + zo] * sk, s[c]);
            }
            __builtin_amdgcn_sched_barrier(0);  // one group's q reads in flight at a time (no spills)
        }
        // online softmax (base 2) per column; the running max is wave-uniform
        float p[DQ_C];
#pragma unroll
        for (int c = 0; c < DQ_C; ++c) {
            const float x = valid ? s[c] * sl2 : -INFINITY;
            const float mt = warp_max(x);
            const float mn = fmaxf(m[c], mt);
            const float alpha = __builtin_amdgcn_exp2f(m[c] - mn);  // m = -inf: 0
            p[c] = __builtin_amdgcn_exp2f(x - mn);
            ls[c] = ls[c] * alpha + p[c];
            m[c] = mn;
#pragma unroll
            for (int d = 0; d < 8; ++d) o[c][d] *= alpha;
        }
        {
            _Float16 vsh[8];
            __builtin_memcpy(vsh, &vsw, 16);
#pragma unroll
            for (int c = 0; c < DQ_C; ++c) Pw[wid][lane >> 1][c][lane & 1] = (_Float16)p[c];
#pragma unroll
            for (int gi = 0; gi < 8; ++gi) Sv[wid][lane >> 1][gi][lane & 1] = valid ? vsh[gi] : (_Float16)0.f;
        }
        dq_wave_sync();
        // P.V (lane = dims [8 lane, 8 lane + 8)), key pairs
        const dq_h2 off = (dq_h2){(_Float16)1032.0f, (_Float16)1032.0f};
#pragma unroll 4
        for (int pr = 0; pr < 16; ++pr) {
            const uint4 pa = *(const uint4 *)&Pw[wid][pr][0][0], pb = *(const uint4 *)&Pw[wid][pr][4][0];
            uint32_t svw;
            __builtin_memcpy(&svw, &Sv[wid][pr][gl][0], 4);
            const dq_h2 sv2 = dq_bits(svw);
            const uint32_t pu[8] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
            dq_h2 w2[DQ_C];
#pragma unroll
            for (int c = 0; c < DQ_C; ++c) w2[c] = dq_bits(pu[c]) * sv2;
            const uint32_t a0 = vw[2 * pr], a1 = vw[2 * pr + 1];
            const uint32_t lo0 = a0 & M4, hi0 = (a0 >> 4) & M4, lo1 = a1 & M4, hi1 = (a1 >> 4) & M4;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t sel = (uint32_t)j | 0x0c00u | (uint32_t)(4 + j) << 16 | 0x0c000000u;
                const dq_h2 ne = dq_bits(__builtin_amdgcn_perm(lo1, lo0, sel) | 0x64006400u) - off;  // dim 2j: (n0 - 8, n1 - 8)
                const dq_h2 no = dq_bits(__builtin_amdgcn_perm(hi1, hi0, sel) | 0x64006400u) - off;  // dim 2j + 1
#pragma unroll
                for (int c = 0; c < DQ_C; ++c) {
                    o[c][2 * j] = __builtin_amdgcn_fdot2(w2[c], ne, o[c][2 * j], false);
                    o[c][2 * j + 1] = __builtin_amdgcn_fdot2(w2[c], no, o[c][2 * j + 1], false);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        dq_wave_sync();
    }
    // merge the waves' partials in wave order
#pragma unroll
    for (int c = 0; c < DQ_C; ++c) {
        const float l = warp_sum(ls[c]);
        if (lane == 0) {
            Rm[wid][c] = m[c];
            Rl[wid][c] = l;
        }
    }
    __syncthreads();
    float fac[DQ_C];
#pragma unroll
    for (int c = 0; c < DQ_C; ++c) {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < DQ_NW; ++w) M = fmaxf(M, Rm[w][c]);
        fac[c] = m[c] == -INFINITY ? 0.f : __builtin_amdgcn_exp2f(m[c] - M);
    }
    for (int w = 0; w < DQ_NW; ++w) {
        if (wid == w) {
#pragma unroll
            for (int c = 0; c < DQ_C; ++c) {
                float4 *ap = (float4 *)&Acc[c][8 * lane];
                const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
                float4 x0 = w == 0 ? z4 : ap[0], x1 = w == 0 ? z4 : ap[1];
                x0.x += fac[c] * o[c][0];
                x0.y += fac[c] * o[c][1];
                x0.z += fac[c] * o[c][2];
                x0.w += fac[c] * o[c][3];
                x1.x += fac[c] * o[c][4];
                x1.y += fac[c] * o[c][5];
                x1.z += fac[c] * o[c][6];
                x1.w += fac[c] * o[c][7];
                ap[0] = x0;
                ap[1] = x1;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < grp) {
        const int c = threadIdx.x, h = g * grp + c;
        float M = -INFINITY, L = 0.f;
        for (int w = 0; w < DQ_NW; ++w) M = fmaxf(M, Rm[w][c]);
        for (int w = 0; w < DQ_NW; ++w)
            if (Rm[w][c] != -INFINITY) L += __builtin_amdgcn_exp2f(Rm[w][c] - M) * Rl[w][c];
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = M * 0.6931471805599453f;  // natural-log units
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = L;
    }
    for (int i = threadIdx.x; i < grp * AW_D; i += blockDim.x) {
        const int c = i / AW_D, d = i % AW_D, h = g * grp + c;
        part_acc[t * acc_q + ((size_t)split * n_head + h) * AW_D + d] = Acc[c][d];
    }
}

// ---------------------------------------------------------------------------
// Prefill attention on WMMA (flash-attention style), causal, head_dim 256.
// Block = 64 queries (4 waves x 16) of one head. Per 32-key block:
//   S^T = K Q^T   (A = K rows from LDS, B = Q^T fragments kept in registers)
//   online softmax per query column (lane % 16), P^T stays in registers and
//   is directly the B operand of
//   O^T += V^T P^T (A = V^T rows from LDS, stored transposed).
// Output gated by sigmoid(gate) like attn_decode. K/V cache is f32.
// ---------------------------------------------------------------------------
#define FA_Q 128
#define FA_K 32
#define FA_D 256

template <int MODE, int Q8>
// h0: first head of this launch (grid.y = heads h0 ..): a long prefill chunk at a
// deep position is split into several launches over head ranges, each short
// enough for the Windows GPU watchdog (TDR, 2 s); results are unchanged.
__device__ __forceinline__ void attn_prefill_impl(const float *q, const KvArgs &kv, float *out, int n_head, int n_kv,
                      int q_stride, const int *pos, int n_tok, float scale, int h0) {
    const int tb = kv.tab0;  // one sequence per launch
    __shared__ _Float16 Ks[FA_K][FA_D + 8];
    __shared__ _Float16 Vt[FA_D][FA_K + 8];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int h = blockIdx.y + h0, kvh = h / (n_head / n_kv);
    const int q0 = blockIdx.x * FA_Q;
    const int pos0 = pos[0];
    const int my_q = q0 + wid * 16 + lane % 16;       // this lane's query column
    const int my_qc = min(my_q, n_tok - 1);            // clamp for loads
    const int my_pos = pos0 + my_qc;

    // Q^T fragments (B operand): column = query, k = dims 16s + 8*(lane/16) ..
#include "shared/attn_fa_qfrag.h"  // half8 qf[FA_D / 16]: Q^T fragments of query my_qc (ADR-0005)
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
                        kv8 = kv_ld8<kv_kf(Q8)>(kv.k, kv.ks, ((size_t)(key + dl) * n_kv + kvh) * FA_D + d8);
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
                        rows[r] = kv_ld8<kv_vf(Q8)>(kv.v, kv.vs, ((size_t)(key + dl) * n_kv + kvh) * FA_D + d8);
                    } else {
#pragma unroll
                        for (int q = 0; q < 8; ++q) rows[r][q] = (_Float16)0.f;
                    }
                }
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    half8 col;
#pragma unroll
                    for (int r = 0; r < 8; ++r) col[r] = rows[r][c];
                    *(half8 *)&Vt[d8 + c][kb * 8] = col;
                }
            }
        }
        __syncthreads();
        // S^T tiles: rows = keys kt*16 + 8*(lane/16) + v, column = my query
        float8 sc[FA_K / 16];
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt) {
            sc[kt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
                const half8 a = *(const half8 *)&Ks[kt * 16 + lane % 16][16 * st + 8 * (lane / 16)];
                sc[kt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, qf[st], sc[kt]);
            }
        }
        // causal mask + online softmax (per lane = per query; partner lane ^16)
#include "shared/attn_fa_softmax.h"  // mask + online softmax of the key tile (ADR-0005)
        // alpha == 1 on every lane (running max unchanged: the common case
        // deep into a long context) makes the rescale an exact no-op; skip it
        if (MODE != 2 && __builtin_amdgcn_ballot_w32(alpha != 1.f)) {
#pragma unroll
        for (int dt = 0; dt < FA_D / 16; ++dt)
#pragma unroll
            for (int v = 0; v < 8; ++v) o[dt][v] *= alpha;
        }
        // O^T += V^T P^T (a pinned kt-outer order removed the scratch spills
        // but measured 5% slower at 120k: item D probe, 243 vs 255 ms/layer)
#pragma unroll
        for (int dt = 0; dt < FA_D / 16; ++dt)
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt) {
                const half8 a = *(const half8 *)&Vt[dt * 16 + lane % 16][kt * 16 + 8 * (lane / 16)];
                if (MODE != 3) o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, pt[kt], o[dt]);
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
            const int d = dt * 16 + 8 * (lane / 16) + v;
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
FA_ENTRY(attn_prefill_wmma_q8v, 0, 2)
FA_ENTRY(attn_prefill_wmma_q4, 0, 3)
FA_ENTRY(fa_probe_noload, 1, false)
FA_ENTRY(fa_probe_norescale, 2, false)
FA_ENTRY(fa_probe_nopv, 3, false)

