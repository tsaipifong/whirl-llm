// WHIRL device code (gfx1151 / RDNA 3.5): paged KV, decode / split / prefill attention.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Paged KV cache (one shared block pool). Each attention layer's K and V
// caches are pools of KV_PAGE-token pages; logical position p of a sequence
// lives in pool row ptab[tb + p / KV_PAGE] * KV_PAGE + p % KV_PAGE, tb = the
// sequence's page-table offset (per batched row: kvbase[row]; else tab0).
// Element format: f16, int8 or int4 (+ one f16 scale per 32 consecutive
// values of a row; scales in ks / vs as [pool row][row elements / 32]).
// Every kernel reading K/V dequantizes with the same f16 multiply
// (kv_ld8 / kv_ld1), and K/V are quantized once, when stored (kv_st32).
// ---------------------------------------------------------------------------
#include "shared/attn_kv_page.h"  // also in the gfx1201 set (kernels/attn.hip)
#include "shared/attn_i4qk.h"     // v0252-8sfp: int4 Q.K^T codes of the fast prefill attention (attn_prefill_wmma_i4)
#include "shared/attn_i8qk.h"     // v0252-8sattn: int8 Q.K^T codes of the balance prefill attention (attn_prefill_wmma_i8)
// gfx11 WMMA fragment: 16 consecutive values (one scale group half)
template <int F> __device__ __forceinline__ half16 kv_ld16(const void *base, const _Float16 *sc, size_t e) {
    if constexpr (F == 2) {
        const uint2 w = *(const uint2 *)((const uint8_t *)base + (e >> 1));
        const _Float16 s = sc[e >> 5];
        const half8 lo = kv_dq4(w.x, s), hi = kv_dq4(w.y, s);
        return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    } else {
        const half8 lo = kv_ld8<F>(base, sc, e), hi = kv_ld8<F>(base, sc, e + 8);
        return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    }
}
// Store x at element e; the 32 lanes of the wave hold the 32 elements of one
// aligned scale group, lane = e % 32 (q8: scale = amax / 127 rounded to f16,
// q = round(x / scale); q4: see above, q = clamp(round(x / scale), -8, 7)).
#include "shared/attn_attn_decode_entry.h"  // also in the gfx1201 set (kernels/attn.hip)
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
            for (int p = c0 + wid; p < c1; p += nw) {
                const half8 kh8 = kv_ld8<kv_kf(Q8)>(kv.k, kv.ks, ((size_t)(p + dl) * n_kv + g) * head_dim + lane * 8);
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

#include "shared/attn_attn_combine.h"  // also in the gfx1201 set (kernels/attn.hip)

// ---------------------------------------------------------------------------
// Fused attention prep: per (head, token) block the q_norm / k_norm RMSNorm,
// NEOX RoPE on the first n_rot dims and, for k heads, the K/V cache store.
// Same float expressions as rmsnorm + rope_neox + kv_store (same block_sum
// order), so the results are unchanged; 5 launches per attention layer -> 1.
// grid = (n_head + n_kv, n_tok), 256 threads, hd <= 512.
// ---------------------------------------------------------------------------
// MR (vision): multi-section RoPE for image prompts, as kernels/attn.hip (ggml
// GGML_ROPE_TYPE_IMROPE): rotary pair i takes the h position when i % 3 == 1 and
// i < 3 * s1, the w position when i % 3 == 2 and i < 3 * s2, else t. rpos = [row][3]
// {t, h, w}; rpos == nullptr: all three = pos[t] - rdelta. The KV row stays pos[t].
template <int Q8, bool ROT = false, bool MR = false>
__device__ __forceinline__ void attn_prep_impl(float *qf, float *kk, const float *vv, const float *qw, const float *kw, const KvArgs &kv,
              const int *pos, int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps,
              const int *rpos = nullptr, int rdelta = 0, int s1 = 0, int s2 = 0) {
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
            const float theta = (float)pi * powf(theta_scale, (float)i);
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
    if constexpr (ROT) {  // in-place fast Walsh-Hadamard transform, hd a power of 2 (as kernels/attn.hip)
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
#include "shared/attn_attn_prep.h"  // also in the gfx1201 set (kernels/attn.hip)

// ROT: rotate every q and k head by the orthonormal Walsh-Hadamard transform after
// RoPE (q.k unchanged in exact arithmetic): K outlier channels spread over the head,
// so the per-32 int scales fit better. Every attention path reads q from qf and K
// from the cache written here. q8h: int8 K / V; q4: 4-bit K / V (always rotated).
#define ATTN_PREP_ENTRY(NAME, Q8, ROT)                                                                                  extern "C" __global__ void __launch_bounds__(256)                                                                     NAME(float *qf, float *kk, const float *vv, const float *qw, const float *kw, KvArgs kv, const int *pos,               int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps) {                                     attn_prep_impl<Q8, ROT>(qf, kk, vv, qw, kw, kv, pos, n_head, n_kv, hd, n_rot, theta_scale, eps);              }
ATTN_PREP_ENTRY(attn_prep_q8h, 1, true)
ATTN_PREP_ENTRY(attn_prep_q8v, 2, false)  // K f16 (not rotated), V int8
ATTN_PREP_ENTRY(attn_prep_q4, 3, true)

// vision: the same five with multi-section RoPE positions (see attn_prep_impl MR);
// argument ABI as kernels/attn.hip ATTN_PREP_M (s0 unused)
#define ATTN_PREP_M(NAME, Q8, ROT)                                                                                            extern "C" __global__ void __launch_bounds__(256)                                                                         NAME(float *qf, float *kk, const float *vv, const float *qw, const float *kw, KvArgs kv, const int *pos,                        int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps, const int *rpos, int rdelta,                        int s0, int s1, int s2) {                                                                                                (void)s0;                                                                                                                 attn_prep_impl<Q8, ROT, true>(qf, kk, vv, qw, kw, kv, pos, n_head, n_kv, hd, n_rot, theta_scale, eps, rpos,                                             rdelta, s1, s2);                                                                        }
ATTN_PREP_M(attn_prep_m, 0, false)
ATTN_PREP_M(attn_prep_m_q8, 1, false)
ATTN_PREP_M(attn_prep_m_q8h, 1, true)
ATTN_PREP_M(attn_prep_m_q8v, 2, false)
ATTN_PREP_M(attn_prep_m_q4, 3, true)

// vision: rows' multi-section RoPE positions {t, h, w} for a small batch (by value)
struct Rpos16 {
    int v[48];
};
#include "shared/attn_set_rpos.h"  // also in the gfx1201 set (kernels/attn.hip)
// struct AwGroups: kernels/kernel_args.h

// Per-column math of attn_wsplit_impl, shared with attn_ws4 (kernels/gfx1151/attn_ws4.hip) so
// both run the same operation graph per column (bitwise identical by construction).
// QK chain of one 16-key tile: sc = WMMA(A(st), Q^T fragment st, sc) for st 0..15 in order;
// qrow = this lane's Qs row (column lane % 16), afrag(st) = K rows' values [16 st, 16 st + 16).
template <class FA> __device__ __forceinline__ float8 aw_qk_chain(float8 sc, const _Float16 *qrow, FA afrag) {
#pragma clang fp contract(off)
#pragma unroll
    for (int st = 0; st < AW_D / 16; ++st) {
        const half16 qb = ld_frag16(qrow + 16 * st);
        const half16 a = afrag(st);
        sc = WMMA16(a, qb, sc);
    }
    return sc;
}
// Online-softmax step of one 32-key chunk for this lane's column: mask (key <= my_pos, key < b1),
// scale, chunk max over both half-waves, rescale factor alpha, p (f16) and the row sum in the
// fixed order (kt outer, v inner, then lane ^ 16); updates m, l; returns alpha.
__device__ __forceinline__ float aw_softmax_step(float8 (&sc)[AW_K / 16], half8 (&pt)[AW_K / 16], int k0, int lane, int my_pos, int b1,
                                                 float scale, float &m, float &l) {
#pragma clang fp contract(off)
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
    return alpha;
}
// P.V of one chunk for NDT 16-dim tiles of this lane's column: o *= alpha, then per tile
// o = WMMA(V^T tile kt0, p kt0, o), WMMA(.. kt1, p kt1, o); vfrag(j, kt) = V^T fragment of tile j.
template <int NDT, class FV>
__device__ __forceinline__ void aw_pv(float8 (&o)[NDT], float alpha, const half8 (&pt)[AW_K / 16], int lane, FV vfrag) {
#pragma clang fp contract(off)
#pragma unroll
    for (int dt = 0; dt < NDT; ++dt)
#pragma unroll
        for (int v = 0; v < 8; ++v) o[dt][v] *= alpha;
    half16 pb[AW_K / 16];
#pragma unroll
    for (int kt = 0; kt < AW_K / 16; ++kt) pb[kt] = join_halves(pt[kt], lane);
#pragma unroll
    for (int dt = 0; dt < NDT; ++dt)
#pragma unroll
        for (int kt = 0; kt < AW_K / 16; ++kt) {
            const half16 a = vfrag(dt, kt);
            o[dt] = WMMA16(a, pb[kt], o[dt]);
        }
}

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
#include "shared/attn_qs_f16.h"  // Qs[c][d8..+8) = f16 Q of column c (ADR-0005)
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
                    rows[r] = kv_ld8<kv_vf(Q8)>(kv.v, kv.vs, ((size_t)(key + dl) * n_kv + g) * AW_D + d8);
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
            sc[kt] = aw_qk_chain(sc[kt], &Qs[wid * 16 + lane % 16][0],
                                 [&](int st) { return kv_ld16<kv_kf(Q8)>(kv.k, kv.ks, kr + 16 * st); });
        }
        half8 pt[AW_K / 16];
        const float alpha = aw_softmax_step(sc, pt, k0, lane, my_pos, b1, scale, m, l);
        aw_pv<AW_D / 16>(o, alpha, pt, lane, [&](int dt, int kt) { return ld_frag16(&Vt[dt * 16 + lane % 16][kt * 16]); });
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
#include "shared/attn_attn_wsplit1.h"  // also in the gfx1201 set (kernels/attn.hip)

// ---------------------------------------------------------------------------
// attn_dq4: q4 KV decode / verify attention for fast mode (gfx1151). Same
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
#include "shared/attn_dq_nw.h"  // also in the gfx1201 set (kernels/attn.hip)
// attn_dq4 block mapping (shared/attn_dq4.h): the block index as it is
#define ADQ_GRID const int g = blockIdx.x, split = blockIdx.y, n_split = gridDim.y, z = blockIdx.z;
#include "shared/attn_dq4.h"  // attn_dq4, both kernel sets (ADR-0005)
#undef ADQ_GRID



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
template <int F> __device__ __forceinline__ fa_h4 kv_ld4(const void *base, const _Float16 *sc, size_t e) {
    if constexpr (F == 1) {
        const fa_q4 q = *(const fa_q4 *)((const int8_t *)base + e);
        const _Float16 s = sc[e >> 5];
        fa_h4 h;
#pragma unroll
        for (int i = 0; i < 4; ++i) h[i] = (_Float16)q[i] * s;
        return h;
    } else if constexpr (F == 2) {
        const half8 h8 = kv_dq4(*(const uint16_t *)((const uint8_t *)base + (e >> 1)), sc[e >> 5]);
        return (fa_h4){h8[0], h8[1], h8[2], h8[3]};
    } else {
        return *(const fa_h4 *)((const _Float16 *)base + e);
    }
}
// V^T element (dim row, permuted key position h)
__device__ __forceinline__ int fa_vt_off(int row, int h) { return row * FA_VS + ((((h >> 3) ^ (row >> 3)) & 3) << 3) + (h & 7); }

// q8 KV: the previous tile loader (the 256-thread loader below spills with the
// q8 dequantization); same arithmetic, same bits.
template <int MODE, int Q8>
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
#include "shared/attn_fa_softmax.h"  // mask + online softmax of the key tile (ADR-0005)
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
// SEG (f16 only, KG-3; as attn_kgs on gfx1201): keys [kb0, kb1) of the sequence (kb0, kb1
// multiples of FA_K, or kb1 = the last key + 1) in one launch of a series covering all keys: the
// running softmax state (m, l in st_ml[query][head][2], the unnormalized O^T in out) is stored at
// the end of a non-final range and loaded at the start of the next, so the series walks the same
// tiles in the same order with the same arithmetic as one launch over all keys - bit-identical
// output (a block whose keys all lie before kb0 returns at once; the range that holds a block's
// last key writes its output).
// I4 (v0252-8sfp, FAST mode only, kernels/shared/attn_i4qk.h): Q.K^T on v_wmma_i32_16x16x16_iu4 (16 cycles, the f16 form
// 32): each query row (per head) quantized to int4 once per block, each key row of a tile once when it is stored in LDS
// (per-row absmax / 7 scales), score = s_q s_k int dot; mask / softmax / P.V as the f16 kernel. Not bitwise.
// I8 (v0252-8sattn, BALANCE mode only, kernels/shared/attn_i8qk.h): Q.K^T on v_wmma_i32_16x16x16_iu8 (32 cycles, the
// f16 rate): per-row int8 query / key codes (absmax / 127). The query pieces take 32 VGPRs instead of the f16 kernel's
// 64, and the freed registers hold the NEXT key tile's K / V loads while this tile is scored (register prefetch: the
// loads of tile k0 + FA_K are issued right after the second barrier of tile k0 and stored at the top of the next step).
// mask / softmax / P.V as the f16 kernel (f16 P, f32 O). Not bitwise (scores from int8 codes).
typedef int fa_i2 __attribute__((ext_vector_type(2)));
typedef int fa_i4v __attribute__((ext_vector_type(4)));
typedef int fa_i8 __attribute__((ext_vector_type(8)));
#define FA_K4S 34  // int4 K row stride in LDS (32-bit words: 32 words of 8 dims + 2 pad, conflict-free 8-byte fragment reads)
#define FA_K8S 68  // int8 K row stride in LDS (32-bit words: 64 words of 4 dims + 4 pad; 16-byte fragment reads)
// barrier that orders LDS only (strix-llama.cpp fattn-rdna35.cu lds_barrier, as attn_sx.hip): __syncthreads() also waits
// for every outstanding global load (vmcnt(0)), which would end the register prefetch of the PF forms at the barrier
__device__ __forceinline__ void fa_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// PF (v0252-8sattn, the int8 form only; same tiles, same values, same arithmetic as PF 0):
//   0: K and V of tile k0 loaded at the top of step k0 and stored to LDS before the scores (as the shipped kernels);
//   1: K and V of tile k0 + FA_K loaded right after the second (LDS-only) barrier of step k0, in flight during the step.
// The f16 kernel cannot take PF 1: compiled here (ROCm 7.2) every arrangement spills (K+V whole step 45 VGPRs, K whole
// step 21, K / V after the softmax 46 / 26), because its Q (64 VGPRs) and O (128) leave no room (HANDOFF, research note).
//   2 (v0252-8sattn2, bench variants only): as 1, but nothing consumes the prefetched registers before the next step's store
//     - the loads are raw (key index clamped for the address only), the zero masking of keys >= n_keys is done at the
//     store (block-uniform branch, last tile only), and the page-table delta of the next tile is read at the top of the
//     step. The PF 1 code object waits for its loads right after issuing them (the select of the masking consumes them in
//     the same step: progress\v0252-8sattn2.PRIOR_ART.md sec. 2), so PF 1 hides nothing; PF 2 is the strix structure.
// QX (v0252-8sattn2, bench variants only; same values, same arithmetic): the other placement of the query codes - I8: in
// registers (this lane's 8 bytes per 16-dim step, the partner half's 8 joined per step as the f16 kernel's join_halves),
// I4: in LDS (128 x FA_K4S words = 17.4 KB, the B fragment read per step as the I8 form). Bitwise == the shipping form.
// PIN (v0252-8sattn2, bench variant only; same values): the shipping int4 score loop with the one-A-fragment pin of the int8
// form (attn_prefill_wmma_i4 hoists the step's fragment reads and spills 6 VGPRs). v0252-8sattn4: also the f16 score loop
// (attn_pfx_f16pin, F4; the f16 kernel does not spill, so the pin can only change its schedule).
template <int MODE, int Q8, bool SEG = false, bool I4 = false, bool I8 = false, int PF = 0, bool QX = false, bool PIN = false>
__device__ __forceinline__ void attn_prefill_impl(const float *q, const KvArgs &kv, float *out, int n_head, int n_kv,
                      int q_stride, const int *pos, int n_tok, float scale, int h0,
                      int kb0 = 0, int kb1 = 0, float *st_ml = nullptr) {
    static_assert(!SEG || Q8 == 0, "attn_prefill_wmma SEG: f16 K / V only");
    static_assert(!I4 || (Q8 == 0 && MODE == 0), "attn_prefill_wmma I4: f16 K / V, no probe mode");
    static_assert(!I8 || (Q8 == 0 && MODE == 0 && !I4), "attn_prefill_wmma I8: f16 K / V, no probe mode, not with I4");
    static_assert(PF == 0 || I8 || (I4 && QX), "attn_prefill_wmma PF: the int8 form or the int4 form with Q in LDS only");
    static_assert(!QX || ((I8 || I4) && !SEG), "attn_prefill_wmma QX: int8 / int4 forms, one key range");
    static_assert(!PIN || (!I8 && !QX && !SEG && MODE == 0), "attn_prefill_wmma PIN: the int4 / f16 forms with Q in registers, one key range");
    if constexpr (Q8) {
        attn_prefill_impl_q8kv<MODE, Q8>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);
        return;
    }
    const int tb = kv.tab0;  // one sequence per launch
    __shared__ _Float16 Ks[I8 ? 1 : FA_K][FA_D + 8];  // I8: its own int8 tile (ks8_tile below; the f16 tile would not fit)
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
#include "shared/attn_fa_qfrag.h"  // half8 qf[FA_D / 16]: Q^T fragments of query my_qc (ADR-0005)
    // I4: int4 Q^T fragments (B operand, 16 dims each: word 0 = dims 16 st .. +7 from the lane of the low half, word 1 =
    // dims 16 st + 8 .. +15 from the high half) and the query scale s_q, from the f32 q (one absmax over the 256 dims)
    fa_i2 qi4[I4 && !QX ? FA_D / 16 : 1];
    float sq = 0.f;
    if constexpr (I4 && !QX) {
        const float *qr = q + ((size_t)my_qc * n_head + h) * q_stride;
        float amax = 0.f;
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
            const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w))));
        }
        amax = fmaxf(amax, __shfl_xor(amax, 16, WARP));
        sq = i4qk_scale(amax);
        const float iq = i4qk_inv(amax);
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
            const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
            const float x[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            const int w = (int)i4qk_pack8(x, iq);
            const int pw = __shfl_xor(w, 16, WARP);
            qi4[st] = lane < 16 ? (fa_i2){w, pw} : (fa_i2){pw, w};
        }
    }
    float8 o[FA_D / 16];
#pragma unroll
    for (int dt = 0; dt < FA_D / 16; ++dt) o[dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    // I4: int4 K tile [FA_K][FA_K4S words] and its row scales, inside the f16 tile's LDS (4.6 KB of its 16.9)
    uint32_t *const Ks4 = reinterpret_cast<uint32_t *>(&Ks[0][0]);
    float *const sk4 = reinterpret_cast<float *>(Ks4 + FA_K * FA_K4S);

    const int last_q = min(q0 + FA_Q, n_tok) - 1;
    const int n_keys = pos0 + last_q + 1;
    if constexpr (SEG) {
        if (kb0 >= n_keys) return;  // done in an earlier range (block-uniform)
        if (kb0 > 0) {
            const float *orow = out + (size_t)my_qc * n_head * FA_D + (size_t)h * FA_D;
            m = st_ml[((size_t)my_qc * n_head + h) * 2];
            l = st_ml[((size_t)my_qc * n_head + h) * 2 + 1];
#pragma unroll
            for (int dt = 0; dt < FA_D / 16; ++dt)
#pragma unroll
                for (int v = 0; v < 8; ++v) o[dt][v] = orow[dt * 16 + CROW(lane, v)];
        }
    }
    const int k_beg = SEG ? kb0 : 0, k_end = SEG ? min(n_keys, kb1) : n_keys;
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
            const half8 x = kv_ld8<kv_kf(Q8)>(kv.k, kv.ks, (size_t)(unsigned)(kc + dl) * rstep + kro);
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
            const fa_h4 x = kv_ld4<kv_vf(Q8)>(kv.v, kv.vs, (size_t)(unsigned)(kc + dl) * rstep + vro);
            const fa_h4 z = {0, 0, 0, 0};
            vr[r] = key < n_keys ? x : z;
        }
    };
    if constexpr (I8 || (I4 && QX)) {
        // I8: the int8 Q codes of the block's 128 queries in LDS (row = query column 16 wid + lane % 16 of FA_K8S words; words
        // 4 st + 2 (lane / 16), +1 = dims 16 st + 8 (lane / 16) .. +7, written by this lane) and this lane's query scale s_q (sq).
        // The score loop reads the whole 16-byte B fragment of step st from the row (lanes l and l + 16 read the same bytes), so
        // no query register stays live: that is what makes room for the K / V prefetch registers (the f16 kernel holds its Q as
        // 64 VGPRs of half8 pieces). LDS: K tile 8.8 KB + V^T 20.5 KB + Q 34.8 KB = 64.1 KB per block, two blocks per WGP as before.
        // QX (I8): the same codes in registers (qi8: this lane's two words per step), no Q LDS.
        // QX (I4): the int4 codes in LDS (Qs4: row as Qs8 of FA_K4S words; word 2 st + lane / 16 = dims 16 st + 8 (lane / 16)
        // .. +7), the K tile as the shipping int4 form (Ks4 / sk4 inside the f16 tile's LDS): 16.9 + 20.5 + 17.4 = 54.8 KB.
        __shared__ uint32_t ks8_tile[I8 ? FA_K * FA_K8S + FA_K : 1];
        __shared__ uint32_t qs8_tile[I8 && !QX ? FA_Q * FA_K8S : 1];
        __shared__ uint32_t qs4_tile[I4 ? FA_Q * FA_K4S : 1];
        uint32_t *const Ks8 = ks8_tile;
        uint32_t *const Qs8 = qs8_tile;
        uint32_t *const Qs4 = qs4_tile;
        float *const sk8 = reinterpret_cast<float *>(ks8_tile + FA_K * FA_K8S);
        fa_i2 qi8[I8 && QX ? FA_D / 16 : 1];
        if constexpr (I8) {
            const float *qr = q + ((size_t)my_qc * n_head + h) * q_stride;
            float amax = 0.f;
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
                const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
                const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))));
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w))));
            }
            amax = fmaxf(amax, __shfl_xor(amax, 16, WARP));
            sq = i8qk_scale(amax);
            const float iq = i8qk_inv(amax);
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
                const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
                const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
                const float xa[4] = {a.x, a.y, a.z, a.w}, xb[4] = {b.x, b.y, b.z, b.w};
                if constexpr (QX)
                    qi8[st] = (fa_i2){(int)i8qk_pack4(xa, iq), (int)i8qk_pack4(xb, iq)};
                else
                *(fa_i2 *)&Qs8[(wid * 16 + lane % 16) * FA_K8S + 4 * st + 2 * (lane / 16)] =
                    (fa_i2){(int)i8qk_pack4(xa, iq), (int)i8qk_pack4(xb, iq)};
            }
        } else {
            // I4 + QX: the shipping int4 query codes (same absmax, same packing), stored to LDS instead of kept as qi4
            const float *qr = q + ((size_t)my_qc * n_head + h) * q_stride;
            float amax = 0.f;
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
                const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
                const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))));
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w))));
            }
            amax = fmaxf(amax, __shfl_xor(amax, 16, WARP));
            sq = i4qk_scale(amax);
            const float iq = i4qk_inv(amax);
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
                const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
                const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
                const float x[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                Qs4[(wid * 16 + lane % 16) * FA_K4S + 2 * st + lane / 16] = i4qk_pack8(x, iq);
            }
        }
        // PF 2: raw loads of a tile (no masking: its keys >= n_keys are zeroed at the store) with the page delta read earlier
        auto loadKr = [&](int k0, int dl) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int kc = min(k0 + tid / 32 + 8 * e, n_keys - 1);
                kr8[e] = kv_ld8<kv_kf(Q8)>(kv.k, kv.ks, (size_t)(unsigned)(kc + dl) * rstep + kro);
            }
        };
        auto loadVr = [&](int k0, int dl) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int kc = min(k0 + vkb * 8 + r, n_keys - 1);
                vr[r] = kv_ld4<kv_vf(Q8)>(kv.v, kv.vs, (size_t)(unsigned)(kc + dl) * rstep + vro);
            }
        };
        // PF 2: page deltas two tiles ahead (dl2 = the delta of tile k0 + FA_K at the top of step k0), so the dependent
        // page-table read of a prefetch was issued a whole step before its addresses are formed (r2 compile: read one step
        // ahead, the compiler formed the addresses at once and waited for the read at the top of every step)
        int dl2 = 0;
        if constexpr (PF == 2)
            if (k_beg + FA_K < k_end) dl2 = kv_delta(kv, tb, k_beg + FA_K);
        for (int k0 = k_beg; k0 < k_end; k0 += FA_K) {
            if constexpr (PF != 0)
                fa_lds_barrier();
            else
                __syncthreads();
            // PF 1: the previous step's prefetch (below) holds tile k0's K and V; the stores wait for the registers
            // (vmcnt), not for the barrier
            int dln = 0;
            if constexpr (PF == 2) {
                dln = dl2;
                if (k0 + 2 * FA_K < k_end) dl2 = kv_delta(kv, tb, k0 + 2 * FA_K);
                if (k0 == k_beg) {
                    loadK(k0);
                    loadV(k0);
                } else if (k0 + FA_K > n_keys) {  // block-uniform: the last tile holds keys >= n_keys, zero them
                    const half8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};
                    const fa_h4 z4 = {0, 0, 0, 0};
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        if (k0 + tid / 32 + 8 * e >= n_keys) kr8[e] = z8;
#pragma unroll
                    for (int r = 0; r < 8; ++r)
                        if (k0 + vkb * 8 + r >= n_keys) vr[r] = z4;
                }
            } else {
            if (PF == 0 || k0 == k_beg) {
                loadK(k0);
                loadV(k0);
            }
            }
            // as I4: the 32 lanes of a wave hold the 256 dims of one key (key wid + 8 e, dims 8 lane .. +7): wave absmax,
            // then each lane packs its 8 dims into words 2 lane, 2 lane + 1 of the key's int8 row
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float x[8], amax = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    x[i] = (float)kr8[e][i];
                    amax = fmaxf(amax, fabsf(x[i]));
                }
#pragma unroll
                for (int sh = 16; sh >= 1; sh >>= 1) amax = fmaxf(amax, __shfl_xor(amax, sh, WARP));
                const int key = tid / 32 + 8 * e;
                if constexpr (I8) {
                const float ik = i8qk_inv(amax);
                *(fa_i2 *)&Ks8[key * FA_K8S + 2 * (tid % 32)] = (fa_i2){(int)i8qk_pack4(x, ik), (int)i8qk_pack4(x + 4, ik)};
                if ((tid % 32) == 0) sk8[key] = i8qk_scale(amax);
                } else {
                    Ks4[key * FA_K4S + (tid % 32)] = i4qk_pack8(x, i4qk_inv(amax));
                    if ((tid % 32) == 0) sk4[key] = i4qk_scale(amax);
                }
            }
            // keys vkb*8 + r: tile vkb / 2, j = (vkb % 2) * 8 + r ->
            // position (r % 2) * 8 + (vkb % 2) * 4 + r / 2 within the tile (as the f16 kernel's V^T store)
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
            if constexpr (PF != 0)
                fa_lds_barrier();
            else
                __syncthreads();
            // PF 1: issue the next tile's K and V loads now; they land in kr8 / vr while this tile is scored (no use before
            // the next step's store). Same rows, same zero masking past n_keys as the load at the top of the step.
            if constexpr (PF == 1) {
                if (k0 + FA_K < k_end) {
                    loadK(k0 + FA_K);
                    loadV(k0 + FA_K);
                }
            }
            if constexpr (PF == 2) {
                if (k0 + FA_K < k_end) {
                    loadKr(k0 + FA_K, dln);
                    loadVr(k0 + FA_K, dln);
                }
            }
            float8 sc[FA_K / 16];
            if constexpr (I8) {
            // int8 S^T: A = key rows kt*16 + lane%16 (16-byte fragment of dims 16 st ..), B = this lane's query row in Qs8
            {
                fa_i8 acc[FA_K / 16];
#pragma unroll
                for (int kt = 0; kt < FA_K / 16; ++kt) acc[kt] = (fa_i8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int st = 0; st < FA_D / 16; ++st) {
                    fa_i4v qb;
                    if constexpr (QX) {
                        // this lane's 8 bytes and the partner half's 8 (lane ^ 16), low half first (as join_halves)
                        const int p0 = xhalf(qi8[st][0]), p1 = xhalf(qi8[st][1]);
                        qb = lane < 16 ? (fa_i4v){qi8[st][0], qi8[st][1], p0, p1} : (fa_i4v){p0, p1, qi8[st][0], qi8[st][1]};
                    } else {
                        qb = *(const fa_i4v *)&Qs8[(wid * 16 + lane % 16) * FA_K8S + 4 * st];
                    }
#pragma unroll
                    for (int kt = 0; kt < FA_K / 16; ++kt) {
                        const fa_i4v a = *(const fa_i4v *)&Ks8[(kt * 16 + lane % 16) * FA_K8S + 4 * st];
                        acc[kt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, qb, acc[kt], false);
                        asm volatile("" : "+v"(acc[kt]));  // pin: one A fragment in flight at a time (as the P.V loop)
                    }
                }
#pragma unroll
                for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
                    for (int v = 0; v < 8; ++v) sc[kt][v] = (float)acc[kt][v] * (sk8[kt * 16 + CROW(lane, v)] * sq);
            }
            } else {
                // int4 S^T as the shipping form, B = this lane's query row in Qs4 (8-byte fragment of dims 16 st ..)
                fa_i8 acc[FA_K / 16];
#pragma unroll
                for (int kt = 0; kt < FA_K / 16; ++kt) acc[kt] = (fa_i8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int st = 0; st < FA_D / 16; ++st) {
                    const fa_i2 qb = *(const fa_i2 *)&Qs4[(wid * 16 + lane % 16) * FA_K4S + 2 * st];
#pragma unroll
                    for (int kt = 0; kt < FA_K / 16; ++kt) {
                        const fa_i2 a = *(const fa_i2 *)&Ks4[(kt * 16 + lane % 16) * FA_K4S + 2 * st];
                        acc[kt] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, qb, acc[kt], false);
                        // pin (as the int8 form): one A fragment in flight at a time. Without it the compiler hoists all 48
                        // fragment reads of the step (96 VGPRs) and the variant spills (r1 compile: 7 / 74 spilled VGPRs)
                        asm volatile("" : "+v"(acc[kt]));
                    }
                }
#pragma unroll
                for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
                    for (int v = 0; v < 8; ++v) sc[kt][v] = (float)acc[kt][v] * (sk4[kt * 16 + CROW(lane, v)] * sq);
            }
            // causal mask + online softmax (per lane = per query; partner lane ^16), O rescale, as the f16 kernel
#include "shared/attn_fa_softmax.h"  // mask + online softmax of the key tile (ADR-0005)
#pragma unroll
            for (int dt = 0; dt < FA_D / 16; ++dt)
#pragma unroll
                for (int v = 0; v < 8; ++v) o[dt][v] *= alpha;
            // O^T += V^T P^T exactly as the f16 kernel (same V^T layout, same order)
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
                    o[dt] = WMMA16(a, pb, o[dt]);
                    asm volatile("" : "+v"(o[dt]));
                }
            }
        }
    } else {
    for (int k0 = k_beg; k0 < k_end; k0 += FA_K) {
        __syncthreads();
        if (MODE != 1 || k0 == 0) {
            loadK(k0);
            loadV(k0);
            if constexpr (I4) {
                // the 32 lanes of a wave hold the 256 dims of one key (key wid + 8 e, dims 8 lane .. +7): wave absmax,
                // then each lane packs its 8 dims into word `lane` of the key's int4 row
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float x[8], amax = 0.f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        x[i] = (float)kr8[e][i];
                        amax = fmaxf(amax, fabsf(x[i]));
                    }
#pragma unroll
                    for (int sh = 16; sh >= 1; sh >>= 1) amax = fmaxf(amax, __shfl_xor(amax, sh, WARP));
                    const int key = tid / 32 + 8 * e;
                    Ks4[key * FA_K4S + (tid % 32)] = i4qk_pack8(x, i4qk_inv(amax));
                    if ((tid % 32) == 0) sk4[key] = i4qk_scale(amax);
                }
            } else {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = tid + e * 256;
                *(half8 *)&Ks[i / 32][(i % 32) * 8] = kr8[e];
            }
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
        if constexpr (I4) {
            // int4 S^T: A = key rows kt*16 + lane%16 (8-byte fragment of dims 16 st ..), B = qi4; C layout as f32
            fa_i8 acc[FA_K / 16];
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt) acc[kt] = (fa_i8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
#pragma unroll
                for (int kt = 0; kt < FA_K / 16; ++kt) {
                    const fa_i2 a = *(const fa_i2 *)&Ks4[(kt * 16 + lane % 16) * FA_K4S + 2 * st];
                    acc[kt] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, qi4[st], acc[kt], false);
                    if constexpr (PIN) asm volatile("" : "+v"(acc[kt]));
                }
            }
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
                for (int v = 0; v < 8; ++v) sc[kt][v] = (float)acc[kt][v] * (sk4[kt * 16 + CROW(lane, v)] * sq);
        } else {
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt) sc[kt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const half16 qb = join_halves(qf[st], lane);
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt) {
                const half16 a = ld_frag16(&Ks[kt * 16 + lane % 16][16 * st]);
                sc[kt] = WMMA16(a, qb, sc[kt]);
                if constexpr (PIN) asm volatile("" : "+v"(sc[kt]));  // F4 (attn_pfx_f16pin) only
            }
        }
        }
        // causal mask + online softmax (per lane = per query; partner lane ^16)
#include "shared/attn_fa_softmax.h"  // mask + online softmax of the key tile (ADR-0005)
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
    }
    if (my_q >= n_tok) return;
    if constexpr (SEG) {
        if (k_end < n_keys) {  // not this block's last range: keep the running state
            float *orow = out + (size_t)my_q * n_head * FA_D + (size_t)h * FA_D;
            if (lane < 16) {
                st_ml[((size_t)my_q * n_head + h) * 2] = m;
                st_ml[((size_t)my_q * n_head + h) * 2 + 1] = l;
            }
#pragma unroll
            for (int dt = 0; dt < FA_D / 16; ++dt)
#pragma unroll
                for (int v = 0; v < 8; ++v) orow[dt * 16 + CROW(lane, v)] = o[dt][v];
            return;
        }
    }
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
FA_ENTRY(attn_prefill_wmma_q8v, 0, 2)
FA_ENTRY(attn_prefill_wmma_q4, 0, 3)
FA_ENTRY(fa_probe_noload, 1, false)
FA_ENTRY(fa_probe_norescale, 2, false)
FA_ENTRY(fa_probe_nopv, 3, false)
// KG-3: key range [kb0, kb1) of the f16 kernel with the softmax state carried in st_ml / out
// (see attn_prefill_impl); same grid and block as attn_prefill_wmma
extern "C" __global__ void __launch_bounds__(256)
    attn_prefill_wmma_s(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos,
                        int n_tok, float scale, int h0, int kb0, int kb1, float *st_ml) {
    attn_prefill_impl<0, 0, true>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0, kb0, kb1, st_ml);
}
// v0252-8sfp (FAST mode only): the int4 Q.K^T forms of attn_prefill_wmma / attn_prefill_wmma_s (same arguments, grid,
// block; not bitwise), on the f16 key copy of the q8h / q4 paths (ops::chooseAttnPrefill, numerics flag attn_i4qk)
extern "C" __global__ void __launch_bounds__(256)
    attn_prefill_wmma_i4(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos,
                         int n_tok, float scale, int h0) {
    attn_prefill_impl<0, 0, false, true>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);
}
extern "C" __global__ void __launch_bounds__(256)
    attn_prefill_wmma_s_i4(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos,
                           int n_tok, float scale, int h0, int kb0, int kb1, float *st_ml) {
    attn_prefill_impl<0, 0, true, true>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0, kb0, kb1, st_ml);
}
// v0252-8sattn (BALANCE mode only): the int8 Q.K^T + K / V register-prefetch forms of attn_prefill_wmma /
// attn_prefill_wmma_s (same arguments, grid, block; not bitwise), on the f16 key copy of the q8h path
// (ops::chooseAttnPrefill, numerics flag attn_i8qk)
extern "C" __global__ void __launch_bounds__(256)
    attn_prefill_wmma_i8(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos,
                         int n_tok, float scale, int h0) {
    attn_prefill_impl<0, 0, false, false, true, 1>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);
}
extern "C" __global__ void __launch_bounds__(256)
    attn_prefill_wmma_s_i8(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos,
                           int n_tok, float scale, int h0, int kb0, int kb1, float *st_ml) {
    attn_prefill_impl<0, 0, true, false, true, 1>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0, kb0, kb1, st_ml);
}
// attn_prefill_wmma_i8n: the same int8 arithmetic WITHOUT the register prefetch (PF 0) - a kt probe (attn_prefill_wmma_i8n
// == attn_prefill_wmma_i8 bitwise: the prefetch moves loads, never values) and a kernel-time reference; never selected
extern "C" __global__ void __launch_bounds__(256)
    attn_prefill_wmma_i8n(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos,
                          int n_tok, float scale, int h0) {
    attn_prefill_impl<0, 0, false, false, true, 0>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);
}
// v0252-8sattn2 (PLAN V1-148): bench-only variants (kt-bench family attnpv, tests/kernels/kt_bench.h; kt invariants in
// tests/kernels/test_attn.cpp); never selected by the engine, same arguments / grid / block as attn_prefill_wmma.
// Each moves the query codes or the K / V loads, never a value: bitwise == attn_prefill_wmma_i8 (i8*) / _i4 (i4*).
#define FA_PFX_ENTRY(NAME, I4_, I8_, PF_, QX_)                                                                        \
    extern "C" __global__ void __launch_bounds__(256)                                                               \
        NAME(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos, int n_tok, \
             float scale, int h0) {                                                                                 \
        attn_prefill_impl<0, 0, false, I4_, I8_, PF_, QX_>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0); \
    }
FA_PFX_ENTRY(attn_pfx_i8r, false, true, 0, true)    // V4: int8, Q codes in registers, no prefetch
FA_PFX_ENTRY(attn_pfx_i8p2, false, true, 2, false)  // V7: int8, Q codes in LDS, deferred-consumer prefetch
FA_PFX_ENTRY(attn_pfx_i4l, true, false, 0, true)    // V5: int4, Q codes in LDS, no prefetch
FA_PFX_ENTRY(attn_pfx_i4lp, true, false, 2, true)   // V6: int4, Q codes in LDS, deferred-consumer prefetch
FA_PFX_ENTRY(attn_pfx_i4lp1, true, false, 1, true)  // V6p1: int4, Q codes in LDS, PF 1 (as attn_prefill_wmma_i8)
// V1p: the shipping int4 kernel with its score loop pinned (one A fragment in flight)
extern "C" __global__ void __launch_bounds__(256)
    attn_pfx_i4p(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos, int n_tok,
                 float scale, int h0) {
    attn_prefill_impl<0, 0, false, true, false, 0, false, true>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);
}

// v0252-8sattn4 (PLAN V1-148): bench-only f16 structure variants of attn_prefill_wmma (kt-bench family attnpv F1..F6;
// kt invariants in tests/kernels/test_attn.cpp); never selected by the engine. All class P: the same f16 Q / K / V values,
// the same 16-WMMA f32 score chain per key tile in the same order, the same 32-key tiles from key 0, the shared softmax
// (written out below with an equivalent integer causal test) and the same P.V order - only where Q lives, who stages K / V and when the K / V loads are issued differ, so the output
// is bitwise == attn_prefill_wmma (all-masked tiles past a smaller block's last key are exact no-ops: alpha 1, P 0).
// A separate function, so the shipping attn_prefill_impl is instruction-identical.
//   NW  consumer waves (16 queries each): 8 -> 128-query block (as shipped), 4 -> 64-query block
//   NP  producer waves (0: every thread stages K / V as shipped; 4: waves NW.. only load / stage K / V, the NW consumer
//       waves only compute - they hold no staging registers)
//   QM  Q placement: 0 registers (64 VGPRs, as shipped), 1 re-read from the f32 q (global, L0 / L2) at every 16-dim step of
//       every key tile, 2 f16 in LDS (64-query block only: Q 32 KB + K 16 KB + V^T 16 KB = 64 KB, all three unpadded and
//       XOR-swizzled by 16-byte chunk)
//   PF  0 load at the top of the step (as shipped), 2 (deferred-consumer prefetch issued after the second barrier, as
//       attn_prefill_impl PF 2) or 3 (NP 0 only: the same raw prefetch issued after the score loop, so it flies over the
//       softmax and P.V - no VMEM wait of this wave can then fall behind it: vmcnt counts in order, and a QM 1 Q re-read
//       inside the score loop would otherwise wait for the whole prefetch)
//   PIN one score chain operand at a time (asm pin after every score WMMA, as the int forms); every built form passes 0 (F4,
//       the pinned form, is the shipping attn_prefill_impl with its PIN: compiled here PIN only moved spills around)
// Compile notes (ROCm 7.2, HANDOFF sec. 2): all 8 waves staging with PF 2 spills (Q in registers 55, 64-query + Q in LDS
// 41), so the prefetch forms stage through producer waves; producers + Q in registers spills 10, so F6's consumers re-read Q.
template <bool B> struct fa_bc {  // compile-time role of an attn_f16x_impl staging loop instance
    static constexpr bool v = B;
};
template <int NW, int NP, int QM, int PF, int PIN>
__device__ __forceinline__ void attn_f16x_impl(const float *q, const KvArgs &kv, float *out, int n_head, int n_kv,
                      int q_stride, const int *pos, int n_tok, float scale, int h0) {
    constexpr int QB = NW * 16;                 // queries per block
    constexpr int NL = (NP ? NP : NW) * WARP;   // threads that stage K / V
    constexpr int KE = FA_K * FA_D / 8 / NL;    // K half8 per staging thread
    constexpr int VP = FA_K * FA_D / 32 / NL;   // V passes of (4 dims x 8 keys) per staging thread
    constexpr bool QL = QM == 2;
    static_assert(!QL || NW == 4, "attn_f16x: Q in LDS fits the 64-query block only");
    static_assert(PF == 0 || PF == 2 || (PF == 3 && NP == 0), "attn_f16x: PF 0, 2 or 3 (3 without producer waves)");
    static_assert(KE * NL * 8 == FA_K * FA_D && VP * NL * 32 == FA_K * FA_D, "attn_f16x: staging layout");
    constexpr int KS = QL ? FA_D : FA_D + 8;  // K row stride (halves)
    constexpr int VS = QL ? FA_K : FA_VS;     // V^T row stride (halves)
    __shared__ _Float16 Ks[FA_K * KS];
    __shared__ _Float16 Vt[FA_D * VS];
    __shared__ _Float16 Qs[QL ? QB * FA_D : 8];
    // QL: 16-byte chunk c of row r sits at chunk c ^ (r & 15) (K and Q: 16 rows of one fragment read hit 16 chunk slots);
    // V^T: chunk (h >> 3) of row r at ((h >> 3) ^ (r >> 2)) & 3 (4 rows of 64 B per 256 B, 16 rows conflict-free)
    auto kq_ofs = [](int r, int c) { return QL ? r * FA_D + ((c ^ (r & 15)) << 3) : r * KS + c * 8; };
    auto vt_ofs = [](int r, int hh) {
        return QL ? r * FA_K + ((((hh >> 3) ^ (r >> 2)) & 3) << 3) + (hh & 7) : fa_vt_off(r, hh);
    };
    const int tb = kv.tab0;
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, tid = threadIdx.x;
    const int h = blockIdx.y + h0, kvh = h / (n_head / n_kv);
    const int q0 = blockIdx.x * QB;
    const int pos0 = pos[0];
    const int last_q = min(q0 + QB, n_tok) - 1;
    const int n_keys = pos0 + last_q + 1;
    const int k_beg = 0, k_end = n_keys;
    const int rstep = n_kv * FA_D;
    const bool producer = NP != 0 && wid >= NW;  // wave-uniform
    if (QL && !producer) {
        // the block's f16 Q (same conversion as attn_fa_qfrag.h) into LDS by the consumer waves (rows 16 wid ..; producer
        // waves have no rows); the first loop barrier publishes it
        const int my_qc = min(q0 + wid * 16 + lane % 16, n_tok - 1);
        const float *qr = q + ((size_t)my_qc * n_head + h) * q_stride;
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
            const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
            *(half8 *)&Qs[kq_ofs(wid * 16 + lane % 16, 2 * st + lane / 16)] =
                (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                        (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
        }
    }
    // ---- staging: K / V of tile k0 into LDS (the shipping per-thread geometry generalised to NL threads)
    // STG: this instance stages K / V; CONS: it computes (consume). NP 0: both; NP > 0: producers STG only, consumers CONS
    // only - separate instances, so a consumer holds no staging registers and a producer no Q / O
    auto stage_loop = [&](auto stg, auto cons, auto consume) {
        constexpr bool STG = decltype(stg)::v, CONS = decltype(cons)::v;
        const int lt = NP ? tid - NW * WARP : tid;
        half8 kr8[KE];
        fa_h4 vr[VP * 8];
        auto vgeo = [&](int p, int &vd4, int &vkb) {
            const int s = lt + NL * p;
            vd4 = (s % 64) * 4;
            vkb = s / 64;
        };
        // V^T store offsets of each pass (rows vd4 .. vd4 + 3 share the swizzle: + c * VS), as the shipping vsA / vsB
        int vsA[VP], vsB[VP];
#pragma unroll
        for (int p = 0; p < VP; ++p) {
            int vd4, vkb;
            vgeo(p, vd4, vkb);
            const int vtb = (vkb / 2) * 16 + (vkb % 2) * 4;
            vsA[p] = vt_ofs(vd4, vtb);
            vsB[p] = vt_ofs(vd4, vtb + 8);
        }
        auto loadKV = [&](int k0, int dl, bool mask) {
#pragma unroll
            for (int e = 0; e < KE; ++e) {
                const int i = lt + e * NL, key = k0 + i / 32;
                const int kc = min(key, n_keys - 1);
                const half8 x = kv_ld8<0>(kv.k, kv.ks, (size_t)(unsigned)(kc + dl) * rstep + kvh * FA_D + (i % 32) * 8);
                const half8 z = {0, 0, 0, 0, 0, 0, 0, 0};
                kr8[e] = (mask && key >= n_keys) ? z : x;
            }
#pragma unroll
            for (int p = 0; p < VP; ++p) {
                int vd4, vkb;
                vgeo(p, vd4, vkb);
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int key = k0 + vkb * 8 + r;
                    const int kc = min(key, n_keys - 1);
                    const fa_h4 x = kv_ld4<0>(kv.v, kv.vs, (size_t)(unsigned)(kc + dl) * rstep + kvh * FA_D + vd4);
                    const fa_h4 z = {0, 0, 0, 0};
                    vr[p * 8 + r] = (mask && key >= n_keys) ? z : x;
                }
            }
        };
        int dl2 = 0;
        if constexpr (PF >= 2)
            if (STG && k_beg + FA_K < k_end) dl2 = kv_delta(kv, tb, k_beg + FA_K);
        for (int k0 = k_beg; k0 < k_end; k0 += FA_K) {
            if constexpr (PF >= 2) fa_lds_barrier(); else __syncthreads();
            int dln = 0;
            if constexpr (STG) {
                if constexpr (PF >= 2) {
                    dln = dl2;
                    if (k0 + 2 * FA_K < k_end) dl2 = kv_delta(kv, tb, k0 + 2 * FA_K);
                    if (k0 == k_beg) {
                        loadKV(k0, kv_delta(kv, tb, k0), true);
                    } else if (k0 + FA_K > n_keys) {  // block-uniform: the last tile holds keys >= n_keys, zero them
                        const half8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};
                        const fa_h4 z4 = {0, 0, 0, 0};
#pragma unroll
                        for (int e = 0; e < KE; ++e)
                            if (k0 + (lt + e * NL) / 32 >= n_keys) kr8[e] = z8;
#pragma unroll
                        for (int p = 0; p < VP; ++p) {
                            int vd4, vkb;
                            vgeo(p, vd4, vkb);
#pragma unroll
                            for (int r = 0; r < 8; ++r)
                                if (k0 + vkb * 8 + r >= n_keys) vr[p * 8 + r] = z4;
                        }
                    }
                } else {
                    loadKV(k0, kv_delta(kv, tb, k0), true);
                }
#pragma unroll
                for (int e = 0; e < KE; ++e) {
                    const int i = lt + e * NL;
                    *(half8 *)&Ks[kq_ofs(i / 32, i % 32)] = kr8[e];
                }
#pragma unroll
                for (int p = 0; p < VP; ++p) {
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        fa_h4 ev, od;
#pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            ev[r] = vr[p * 8 + 2 * r][c];
                            od[r] = vr[p * 8 + 2 * r + 1][c];
                        }
                        *(fa_h4 *)&Vt[vsA[p] + c * VS] = ev;
                        *(fa_h4 *)&Vt[vsB[p] + c * VS] = od;
                    }
                }
            }
            if constexpr (PF >= 2) fa_lds_barrier(); else __syncthreads();
            if constexpr (STG && PF == 2)
                if (k0 + FA_K < k_end) loadKV(k0 + FA_K, dln, false);  // raw: masked at the next store
            // PF 3: the consumer calls mid() right after its score loop
            auto mid = [&]() {
                if constexpr (STG && PF == 3) {
                    // the score loop and the loads are one unrolled block: without the barrier the scheduler hoists the
                    // loads into the score loop and the 48 registers spill O
                    __builtin_amdgcn_sched_barrier(0);
                    if (k0 + FA_K < k_end) loadKV(k0 + FA_K, dln, false);
                }
            };
            if constexpr (CONS) consume(k0, mid);
        }
    };
    // ---- consumers: the shipping score / softmax / P.V of one tile; O and (QM 0) Q in registers
    auto consumer = [&]() {
        const int my_q = q0 + wid * 16 + lane % 16;
        const int my_qc = min(my_q, n_tok - 1);
        const int my_pos = pos0 + my_qc;
        float8 o[FA_D / 16];
#pragma unroll
        for (int dt = 0; dt < FA_D / 16; ++dt) o[dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        float m = -INFINITY, l = 0.f;
        half8 qf[QM == 0 ? FA_D / 16 : 1];
        const float *qrow = q + ((size_t)my_qc * n_head + h) * q_stride;
        if constexpr (QM == 0) {
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
                const float4 a = *(const float4 *)(qrow + 16 * st + 8 * (lane / 16));
                const float4 b = *(const float4 *)(qrow + 16 * st + 8 * (lane / 16) + 4);
                qf[st] = (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                                 (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
            }
        }
        // per-lane LDS bases; kt, dt, st and the chunk are immediates (the shipping ld_frag16 / vlo / vhi form):
        //   V^T logical chunk j of row dt * 16 + l16: QL vbo[j] + dt * 16 * VS (four bases, the swizzle does not depend on
        //   dt); else with jj = j ^ 2 (dt & 1) as the shipping vlo / vhi: vbo[0] + jj * 8 (jj even), vbo[1] + (jj - 1) * 8
        //   QL K / Q: element kt / wid * 4096 + (kx ^ (c << 3)) (chunk c ^ l16 of row l16; kx pinned per tile so the 32
        //   xors are formed at their reads instead of hoisted into 32 VGPRs)
        const int l16 = lane % 16;
        int vbo[QL ? 4 : 2];
        if constexpr (QL) {
#pragma unroll
            for (int j = 0; j < 4; ++j) vbo[j] = l16 * FA_K + (((j ^ (l16 >> 2)) & 3) << 3);
        } else {
            vbo[0] = l16 * FA_VS + (l16 >> 3) * 8;
            vbo[1] = l16 * FA_VS + 8 - (l16 >> 3) * 8;
        }
        const int kx0 = l16 * FA_D + (l16 << 3);
        auto tile = [&](int k0, auto mid) {
            int kx = kx0;
            float8 sc[FA_K / 16];
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt) sc[kt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
            // QM 1: the f32 q of steps st, st + 1 in a two-slot ring; the load of step st + 2 is issued after step st's last
            // score WMMA (asm pin on the row pointer that reads sc), so two loads are in flight and the 32 loads of a tile
            // are never hoisted together (the PF 3 compile without the ring: all 32 at the top, 128 VGPRs, 372 spilled)
            const float *qp = qrow + 8 * (lane / 16);
            float4 qra[QM == 1 ? 2 : 1], qrb[QM == 1 ? 2 : 1];
            if constexpr (QM == 1) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    qra[j] = *(const float4 *)(qp + 16 * j);
                    qrb[j] = *(const float4 *)(qp + 16 * j + 4);
                }
            }
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
                if constexpr (QL) asm volatile("" : "+v"(kx));  // per step: this step's 4 xors only
                half16 qb;
                if constexpr (QM == 0) {
                    qb = join_halves(qf[st], lane);
                } else if constexpr (QM == 1) {
                    const float4 a = qra[st & 1], b = qrb[st & 1];
                    qb = join_halves((half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                                             (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w}, lane);
                } else {
                    const _Float16 *qs = Qs + wid * 16 * FA_D;
                    const half8 lo = *(const half8 *)&qs[kx ^ ((2 * st) << 3)], hi = *(const half8 *)&qs[kx ^ ((2 * st + 1) << 3)];
                    qb = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                }
#pragma unroll
                for (int kt = 0; kt < FA_K / 16; ++kt) {
                    half8 lo, hi;
                    if constexpr (QL) {
                        lo = *(const half8 *)&Ks[kt * 16 * FA_D + (kx ^ ((2 * st) << 3))];
                        hi = *(const half8 *)&Ks[kt * 16 * FA_D + (kx ^ ((2 * st + 1) << 3))];
                    } else {
                        lo = *(const half8 *)&Ks[(kt * 16 + l16) * KS + 16 * st];
                        hi = *(const half8 *)&Ks[(kt * 16 + l16) * KS + 16 * st + 8];
                    }
                    const half16 a = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                    sc[kt] = WMMA16(a, qb, sc[kt]);
                    if constexpr (PIN != 0) asm volatile("" : "+v"(sc[kt]));
                }
                if constexpr (QM == 1) {
                    if (st + 2 < FA_D / 16) {
                        asm volatile("" : "+v"(qp) : "v"(sc[FA_K / 16 - 1]));
                        qra[st & 1] = *(const float4 *)(qp + 16 * (st + 2));
                        qrb[st & 1] = *(const float4 *)(qp + 16 * (st + 2) + 4);
                    }
                }
            }
            mid();  // PF 3: the next tile's K / V loads
            // mask + online softmax: shared/attn_fa_softmax.h written out with the causal test as
            // `2 v + 16 kt <= my_pos - k0 - lane / 16` (constants on the left, one per-lane value) instead of forming the 16
            // key indices k0 + 16 kt + CROW(lane, v) - the same integer test, the same float operations in the same order
            float mx = -INFINITY;
            const int dlim = my_pos - k0 - lane / 16;
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const float x = (2 * v + 16 * kt <= dlim) ? sc[kt][v] * scale : -INFINITY;
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
#pragma unroll
            for (int dt = 0; dt < FA_D / 16; ++dt)
#pragma unroll
                for (int v = 0; v < 8; ++v) o[dt][v] *= alpha;
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt) {
                const half16 pb = join_halves(pt[kt], lane);
#pragma unroll
                for (int dt = 0; dt < FA_D / 16; ++dt) {
                    // logical chunks 2 kt (lo) and 2 kt + 1 (hi) of V^T row dt * 16 + l16 (the shipping vlo / vhi reads)
                    half8 lo, hi;
                    if constexpr (QL) {
                        lo = *(const half8 *)&Vt[vbo[2 * kt] + dt * 16 * VS];
                        hi = *(const half8 *)&Vt[vbo[2 * kt + 1] + dt * 16 * VS];
                    } else {
                        const int C = (2 * kt) ^ (2 * (dt & 1));  // even
                        lo = *(const half8 *)&Vt[vbo[0] + dt * 16 * VS + C * 8];
                        hi = *(const half8 *)&Vt[vbo[1] + dt * 16 * VS + C * 8];
                    }
                    const half16 a = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                    o[dt] = WMMA16(a, pb, o[dt]);
                    asm volatile("" : "+v"(o[dt]));  // one V fragment in flight at a time (as shipped)
                }
            }
        };
        stage_loop(fa_bc<NP == 0>{}, fa_bc<true>{}, tile);
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
    };
    if (producer) {
        if constexpr (NP != 0) stage_loop(fa_bc<true>{}, fa_bc<false>{}, [](int, auto) {});
    } else {
        consumer();
    }
}
#define FA_F16X_ENTRY(NAME, NTHR, NW_, NP_, QM_, PF_)                                                         \
    extern "C" __global__ void __launch_bounds__(NTHR)                                                       \
        NAME(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos, int n_tok, \
             float scale, int h0) {                                                                          \
        attn_f16x_impl<NW_, NP_, QM_, PF_, 0>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);   \
    }
// compile resources (ROCm 7.2, progress\v0252-8sattn4.HANDOFF.md sec. 2); the forms that spilled are not built (HANDOFF)
FA_F16X_ENTRY(attn_pfx_f16qg, 256, 8, 0, 1, 2)    // F1: Q re-read from memory every step + PF 2 (all 8 waves stage)
FA_F16X_ENTRY(attn_pfx_f16l64p, 256, 4, 4, 2, 2)  // F2: 64-query block, Q in LDS, 4 producer waves + PF 2
FA_F16X_ENTRY(attn_pfx_f16l64, 256, 4, 4, 2, 0)   // F3: F2 without the prefetch (producers load at the top of the step)
FA_F16X_ENTRY(attn_pfx_f16ws, 384, 8, 4, 1, 2)    // F6: F1's consumers (Q re-read) + 4 producer waves + PF 2
// (F7 = F1 with PF 3, attn_f16x_impl<8, 0, 1, 3, 0>: compiled 256 VGPR + 284 spilled - not built, HANDOFF sec. 2)
// F4: the shipping f16 kernel with its score loop pinned (one chain operand at a time, as the int forms)
extern "C" __global__ void __launch_bounds__(256)
    attn_pfx_f16pin(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos, int n_tok,
                    float scale, int h0) {
    attn_prefill_impl<0, 0, false, false, false, 0, false, true>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);
}

// q8 / q8h / q4 KV prefill (KG-3, the gfx1151 port of Q8P / FC-1c): the K and V of keys
// [r0, r0 + n_rows) of one sequence dequantized into contiguous f16 rows (k16 / v16:
// [row][row_el], pool layout without pages), so the prefill attention runs the f16
// attn_prefill_wmma (or attn_prefill_wmma_s over key ranges) on them. Same dequantization as
// the q8 / q4 tile loaders of attn_prefill_wmma_q8 / _q4 (kv_ld8: kv_dq8 / kv_dq4), and the f16
// kernel does the same arithmetic on the same f16 values, so the output is bit-identical; the
// old loaders converted K / V in every block (each query block and head again).
// grid ceil(n_rows * row_el / 2048), block 256; row_el % 32 == 0
template <int F>
__device__ __forceinline__ void kv_dq_rows_impl(const KvArgs &kv, _Float16 *k16, _Float16 *v16, int r0, int n_rows, int row_el) {
    const size_t o = ((size_t)blockIdx.x * 256 + threadIdx.x) * 8;
    if (o >= (size_t)n_rows * row_el) return;
    const int p = r0 + (int)(o / row_el), e = (int)(o % row_el);
    const size_t src = (size_t)kv_row(kv, kv.tab0, p) * row_el + e;
    *(half8 *)(k16 + o) = kv_ld8<F>(kv.k, kv.ks, src);
    *(half8 *)(v16 + o) = kv_ld8<F>(kv.v, kv.vs, src);
}
#include "shared/attn_prefill_kp_kv_dq_rows.h"  // also in the gfx1201 set (kernels/attn_prefill_kp.hip)
