// WHIRL device code (gfx1201 / RDNA 4): attn_ws4r, 4-wave pipelined split-K decode / verify attention.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included LAST in the gfx1201 block of kernels/whirl_kernels.hip (no existing kernel moves); uses
// KvArgs, AwGroups, kv_delta / kv_tb / kv_dq8, AW_K / AW_D / AW_SINK and the Q^T staging step of kernels/attn.hip.
// v0252-ws4r97 (PLAN V1-87, phase 1): the gfx1201 version of the gfx1151 attn_ws4 design (audit/DES-ATTN-8S.md 3.4-3.7,
// kernels/gfx1151/attn_ws4.hip): every wave loads, the next 32-key chunk is prefetched into registers while the current
// one is computed, block barriers order LDS only, and a block reads each K / V row of its split once for all of its
// query columns. The LDS-scoped barrier is the idea of libr4d r4d_common.h:65-74 and strix-llama.cpp fattn-rdna35.cu
// (on gfx12 __syncthreads also emits global_inv, a vector-cache invalidate that drops the prefetched lines); the code
// here is written from scratch (progress/v0252-ws4r97.PRIOR_ART.md).
//
// attn_ws4r<CG, Q8, B>: CG 16-column tiles (1 = attn_wsplit1, 2 = attn_wsplit2), Q8 the KV format of attn_wsplit_impl
// (0 f16, 1 int8 K and V = q8 / q8h), B the barrier (0 LDS-scoped fences + s_barrier, 1 inline asm that waits only for
// this wave's LDS traffic, see wr_bar). Same arguments, grid (kv head, split, group), 128 threads, column mapping and
// partial layout as attn_wsplit_impl<CG, Q8>, and BITWISE the same partials for ANY groups / grid / window (the host
// keeps its attn_wsplit1 / attn_wsplit2 grouping, so a row's partials, and every mode's output, do not change):
//  - split range: the same expression of n_pos, gridDim.y and the draft window (copied from attn_wsplit_impl);
//  - K: the A operand of every QK WMMA has the same bits (f16 values as stored; int8 through the same kv_dq8 with the
//    same scale ks[e >> 5]; keys >= b1 read row b1 - 1, the attn_wsplit_impl clamp), only staged through LDS;
//  - V^T: the same LDS values (keys >= b1 +0) at the same fragment addresses;
//  - per column: the QK chain over st 0..15 per 16-key tile kt, the softmax step and the P.V order (o * alpha, then
//    kt 0, then kt 1, per 16-dim tile dt) are attn_wsplit_impl's lines; only WHICH wave computes a (tile, kt) score
//    tile or a dt output tile changes (WMMA output tiles are independent), and the softmax runs in every wave with the
//    same inputs (same bits everywhere).
// Per 32-key chunk:
//   LB0  Ks / Vt of the chunk (and Qs) written
//   QK   wave j < 2 CG: tile j / 2, key tile j % 2 -> Sc (f32, the C fragment of that lane)
//   LB1  every wave: softmax of every tile, P.V of its own dims [64 w, 64 w + 64) for every tile
//   LB2  stage the prefetched raw chunk into Ks / Vt, then issue the raw loads of the chunk after it (branch-free,
//        clamped: past the last chunk the last one is reloaded, never used)
// WS4R_MUT (kernel development / test mutants, default 0 = the product): 1 P.V key tiles in the order kt 1, kt 0;
// 2 stub: every partial (-inf, 0) / 0; 3 the draft window ignored; 4 the gfx1151 split rule (64 splits, not gridDim.y);
// 5 q8 K: piece 0's scale for every piece; 6 V keys >= b1 not zeroed (probe: is gfx12 WMMA exact with p = 0 rows?);
// 7 no LB2 barrier (race probe).
// ---------------------------------------------------------------------------
#ifndef WS4R_MUT
#define WS4R_MUT 0
#endif

namespace ws4r {
constexpr int kKs = AW_D + 8;  // Ks row stride (halves), as attn_wsplit_impl's Qs
constexpr int kVt = AW_K + 8;  // Vt row stride (halves), as attn_wsplit_impl's Vt
}  // namespace ws4r

__device__ __forceinline__ void wr_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// B = 1: the barrier as inline asm (opaque to the waitcnt pass, as kernels/gfx1151/attn_ws4.hip ws4_bar<1>): this wave's
// LDS reads / writes done (dscnt 0), then the gfx12 split barrier; global loads stay in flight. Only LDS is exchanged
// between the waves here, and every VGPR use of an in-flight load still gets its s_wait_loadcnt from SIInsertWaitcnts.
// Same instructions on the data, so the same bits as B = 0. Risk as on gfx1151 (review C2): a wave waiting in the
// barrier with loads outstanding cannot be backed out on an exception; all loads are clamped to [b0, b1 - 1].
template <int B>
__device__ __forceinline__ void wr_bar() {
    if constexpr (B == 0)
        wr_lds_barrier();
    else
        asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}

typedef _Float16 wr_h4 __attribute__((ext_vector_type(4)));
// raw bytes of one chunk held by one thread (f16: K 8 x 8 values, V 8 keys x 8 values; int8: K 4 x 16 values + scale,
// V 8 keys x 8 values + scale)
template <int Q8> struct WrRaw;
template <> struct WrRaw<0> {
    uint4 k[8];
    uint4 v[8];
};
template <> struct WrRaw<1> {
    uint4 k[4];
    uint2 v[8];
    wr_h4 ks;
    half8 vs;
};
// thread maps (tv 0..127): f16 K piece p: key tv / 32 + 4 p, dims (tv % 32) * 8 .. +8 (a wave reads one 512-byte row);
// int8 K piece p: key tv / 16 + 8 p, dims (tv % 16) * 16 .. +16; V (both, attn_wsplit_impl's map): key tile kb = tv / 32,
// keys 8 kb + i, dims d8 = (tv % 32) * 8 .. +8
__device__ __forceinline__ constexpr int wr_kpieces(int q8) { return q8 ? 4 : 8; }
__device__ __forceinline__ int wr_kkey(int q8, int tv, int p) { return q8 ? tv / 16 + 8 * p : tv / 32 + 4 * p; }
__device__ __forceinline__ int wr_kd0(int q8, int tv) { return q8 ? (tv % 16) * 16 : (tv % 32) * 8; }

// issue the raw loads of chunk k0 (dl = kv_delta of k0: 32-key chunks never straddle a page), every load unconditional
// (a load under a branch is waited for at the join): keys >= b1 read row b1 - 1 (K: the attn_wsplit_impl clamp; V:
// zeroed when staged)
template <int Q8>
__device__ __forceinline__ void wr_load(WrRaw<Q8> &r, const KvArgs &kv, int dl, int k0, int b1, int g, int n_kv, int tv) {
#pragma unroll
    for (int p = 0; p < wr_kpieces(Q8); ++p) {
        const int key = min(k0 + wr_kkey(Q8, tv, p), b1 - 1);
        const size_t e = ((size_t)(key + dl) * n_kv + g) * AW_D + wr_kd0(Q8, tv);
        if constexpr (Q8 == 0) {
            r.k[p] = *(const uint4 *)((const _Float16 *)kv.k + e);
        } else {
            r.k[p] = *(const uint4 *)((const int8_t *)kv.k + e);
#if WS4R_MUT == 5
            r.ks[p] = kv.ks[(((size_t)(min(k0 + wr_kkey(Q8, tv, 0), b1 - 1) + dl) * n_kv + g) * AW_D + wr_kd0(Q8, tv)) >> 5];
#else
            r.ks[p] = kv.ks[e >> 5];
#endif
        }
    }
    const int kb = tv / 32, d8 = (tv % 32) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int key = min(k0 + kb * 8 + i, b1 - 1);
        const size_t e = ((size_t)(key + dl) * n_kv + g) * AW_D + d8;
        if constexpr (Q8 == 0) {
            r.v[i] = *(const uint4 *)((const _Float16 *)kv.v + e);
        } else {
            r.v[i] = *(const uint2 *)((const int8_t *)kv.v + e);
            r.vs[i] = kv.vs[e >> 5];
        }
    }
}

// raw chunk k0 -> Ks [32][kKs] (f16) and Vt [AW_D][kVt] (attn_wsplit_impl's V^T layout)
template <int Q8>
__device__ __forceinline__ void wr_stage(const WrRaw<Q8> &r, _Float16 (*Ks)[ws4r::kKs], _Float16 (*Vt)[ws4r::kVt], int k0, int b1, int tv) {
#pragma clang fp contract(off)
#pragma unroll
    for (int p = 0; p < wr_kpieces(Q8); ++p) {
        _Float16 *dst = &Ks[wr_kkey(Q8, tv, p)][wr_kd0(Q8, tv)];
        if constexpr (Q8 == 0) {
            *(half8 *)dst = __builtin_bit_cast(half8, r.k[p]);
        } else {
            *(half8 *)dst = kv_dq8(__builtin_bit_cast(kvq8_t, make_uint2(r.k[p].x, r.k[p].y)), r.ks[p]);
            *(half8 *)(dst + 8) = kv_dq8(__builtin_bit_cast(kvq8_t, make_uint2(r.k[p].z, r.k[p].w)), r.ks[p]);
        }
    }
    const int kb = tv / 32, d8 = (tv % 32) * 8;
    half8 rows[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        half8 x;
        if constexpr (Q8 == 0)
            x = __builtin_bit_cast(half8, r.v[i]);
        else
            x = kv_dq8(__builtin_bit_cast(kvq8_t, r.v[i]), r.vs[i]);
#if WS4R_MUT == 6
        const bool valid = true;
#else
        const bool valid = k0 + kb * 8 + i < b1;  // keys >= b1: +0 as attn_wsplit_impl
#endif
#pragma unroll
        for (int e2 = 0; e2 < 8; ++e2) rows[i][e2] = valid ? x[e2] : (_Float16)0.f;
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        half8 cv;
#pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = rows[i][c];
        *(half8 *)&Vt[d8 + c][kb * 8] = cv;
    }
}

template <int CG, int Q8, int B>
__device__ __forceinline__ void attn_ws4r_impl(const float *q, const KvArgs &kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                                               int q_stride, const int *pos, float scale, const int *skip, const AwGroups &groups) {
    // no implicit fma contraction (attn_wsplit_impl's rule: every column rounds as there)
#pragma clang fp contract(off)
    if (skip != nullptr && *skip) return;
    __shared__ __align__(16) _Float16 Qs[16 * CG][AW_D + 8];
    __shared__ __align__(16) _Float16 Ks[AW_K][ws4r::kKs];
    __shared__ __align__(16) _Float16 Vt[AW_D][ws4r::kVt];
    __shared__ __align__(16) float Sc[CG * 2 * 32 * 8];
    const int tv = threadIdx.x, lane = tv % WARP, wid = tv / WARP;
    const int g = blockIdx.x, split = blockIdx.y, n_split = gridDim.y, z = blockIdx.z;
    const int grp = n_head / n_kv;
    const int first = groups.first[z], count = groups.count[z];
    const int ncol = count * grp;
    const int tb = kv_tb(kv, first);
    const int n_pos = pos[first + count - 1] + 1;  // the group's last query
    // split range: attn_wsplit_impl's expression (kernels/attn.hip)
#if WS4R_MUT == 4
    const int nsr = 64;
#else
    const int nsr = n_split;
#endif
    int per = (n_pos + nsr * 64 - 1) / (nsr * 64) * 64;
    int b0 = split * per, b1 = min(n_pos, b0 + per);
#if WS4R_MUT != 3
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
#endif
    const size_t ml_q = (size_t)n_split * n_head * 2, acc_q = (size_t)n_split * n_head * AW_D;
    // tile c: this lane's column c * 16 + lane % 16 (the column of wave c in attn_wsplit_impl)
    bool cvalid[CG];
    int mpos[CG];
#pragma unroll
    for (int c = 0; c < CG; ++c) {
        const int col = c * 16 + lane % 16;
        cvalid[c] = col < ncol;
        mpos[c] = cvalid[c] ? pos[first + col / grp] : -1;
    }
    auto ct = [&](int c) { const int col = cvalid[c] ? c * 16 + lane % 16 : 0; return first + col / grp; };
    auto ch = [&](int c) { const int col = cvalid[c] ? c * 16 + lane % 16 : 0; return g * grp + col % grp; };
    if (b0 >= n_pos || WS4R_MUT == 2) {
        if (wid == 0 && lane < 16) {
#pragma unroll
            for (int c = 0; c < CG; ++c)
                if (cvalid[c]) {
                    part_ml[ct(c) * ml_q + ((size_t)split * n_head + ch(c)) * 2 + 0] = -INFINITY;
                    part_ml[ct(c) * ml_q + ((size_t)split * n_head + ch(c)) * 2 + 1] = 0.f;
                }
        }
#if WS4R_MUT == 2
        if (b0 < n_pos) {  // stub: zero acc where the product writes one
#pragma unroll
            for (int c = 0; c < CG; ++c)
                if (cvalid[c]) {
                    float *pa = part_acc + ct(c) * acc_q + ((size_t)split * n_head + ch(c)) * AW_D;
#pragma unroll
                    for (int j = 0; j < 4; ++j)
#pragma unroll
                        for (int v = 0; v < 8; ++v) pa[(4 * wid + j) * 16 + 8 * (lane / 16) + v] = 0.f;
                }
        }
#endif
        return;
    }
    const int nch = (b1 - b0 + AW_K - 1) / AW_K;
    WrRaw<Q8> raw;
    wr_load<Q8>(raw, kv, kv_delta(kv, tb, b0), b0, b1, g, n_kv, tv);
    // Q^T (f16) of the group's columns, as attn_wsplit_impl (128 threads)
    for (int i = tv; i < 16 * CG * (AW_D / 8); i += 128) {
#include "shared/attn_qs_f16.h"  // Qs[c][d8..+8) = f16 Q of column c (ADR-0005)
    }
    wr_stage<Q8>(raw, Ks, Vt, b0, b1, tv);
    {
        const int kn = b0 + min(1, nch - 1) * AW_K;
        wr_load<Q8>(raw, kv, kv_delta(kv, tb, kn), kn, b1, g, n_kv, tv);
    }
    float8 o[CG][4];
#pragma unroll
    for (int c = 0; c < CG; ++c)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[c][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m[CG], l[CG];
#pragma unroll
    for (int c = 0; c < CG; ++c) {
        m[c] = -INFINITY;
        l[c] = 0.f;
    }
    for (int cidx = 0; cidx < nch; ++cidx) {
        const int k0 = b0 + cidx * AW_K;
        wr_bar<B>();  // LB0: Ks / Vt (and Qs) of the chunk written
        // QK of score tile j = (column tile j / 2, key tile j % 2) in wave j: attn_wsplit_impl's chain
        if (wid < 2 * CG && 16 * (wid / 2) < ncol) {  // block-uniform per wave
            const int c = wid / 2, kt = wid % 2;
            float8 s = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int st = 0; st < AW_D / 16; ++st) {
                const half8 a = *(const half8 *)&Ks[kt * 16 + lane % 16][16 * st + 8 * (lane / 16)];
                const half8 qb = *(const half8 *)&Qs[c * 16 + lane % 16][16 * st + 8 * (lane / 16)];
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, qb, s);
            }
            *(float4 *)&Sc[(wid * 32 + lane) * 8] = (float4){s[0], s[1], s[2], s[3]};
            *(float4 *)&Sc[(wid * 32 + lane) * 8 + 4] = (float4){s[4], s[5], s[6], s[7]};
        }
        wr_bar<B>();  // LB1: Sc written
#pragma unroll
        for (int c = 0; c < CG; ++c) {
            if (16 * c >= ncol) continue;  // block-uniform: no column of this tile
            float8 sc[AW_K / 16];
#pragma unroll
            for (int kt = 0; kt < AW_K / 16; ++kt) {
                const float4 a = *(const float4 *)&Sc[((c * 2 + kt) * 32 + lane) * 8];
                const float4 b = *(const float4 *)&Sc[((c * 2 + kt) * 32 + lane) * 8 + 4];
                sc[kt] = (float8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            }
            // softmax step: attn_wsplit_impl's lines (same column, same inputs -> same bits in every wave)
            const int my_pos = mpos[c];
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
            const float m_new = fmaxf(m[c], mx);
            const float alpha = (m[c] == -INFINITY) ? 0.f : __expf(m[c] - m_new);
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
            l[c] = __builtin_fmaf(l[c], alpha, rs);
            m[c] = m_new;
            // P.V of this wave's 16-dim tiles dt = 4 wid + j: o * alpha, then kt 0, kt 1 (attn_wsplit_impl's order)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int v = 0; v < 8; ++v) o[c][j][v] *= alpha;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int kk = 0; kk < AW_K / 16; ++kk) {
#if WS4R_MUT == 1
                    const int kt = AW_K / 16 - 1 - kk;
#else
                    const int kt = kk;
#endif
                    const half8 a = *(const half8 *)&Vt[(4 * wid + j) * 16 + lane % 16][kt * 16 + 8 * (lane / 16)];
                    o[c][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, pt[kt], o[c][j]);
                }
        }
        if (cidx + 1 < nch) {
#if WS4R_MUT != 7
            wr_bar<B>();  // LB2: Ks, Vt, Sc read
#endif
            wr_stage<Q8>(raw, Ks, Vt, k0 + AW_K, b1, tv);
            const int kn = min(k0 + 2 * AW_K, b0 + (nch - 1) * AW_K);
            wr_load<Q8>(raw, kv, kv_delta(kv, tb, kn), kn, b1, g, n_kv, tv);
        }
    }
#pragma unroll
    for (int c = 0; c < CG; ++c) {
        if (!cvalid[c]) continue;
        if (wid == 0 && lane < 16) {
            part_ml[ct(c) * ml_q + ((size_t)split * n_head + ch(c)) * 2 + 0] = m[c];
            part_ml[ct(c) * ml_q + ((size_t)split * n_head + ch(c)) * 2 + 1] = l[c];
        }
        float *pa = part_acc + ct(c) * acc_q + ((size_t)split * n_head + ch(c)) * AW_D;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) pa[(4 * wid + j) * 16 + 8 * (lane / 16) + v] = o[c][j][v];
    }
}

#define WS4R_ENTRY(name, CG, Q8, B)                                                                                                       \
    extern "C" __global__ void __launch_bounds__(128)                                                                                 \
        name(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv, int q_stride, const int *pos, float scale, \
             const int *skip, AwGroups groups) {                                                                                      \
        attn_ws4r_impl<CG, Q8, B>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);                       \
    }
// == attn_wsplit1 / attn_wsplit1_q8 (groups of <= 16 columns) and attn_wsplit2 / attn_wsplit2_q8 (<= 32 columns);
// suffix b = the inline-asm barrier (B = 1). No host selects them yet (phase 2, after the R9700 kt result).
WS4R_ENTRY(attn_ws4r1, 1, 0, 0)
WS4R_ENTRY(attn_ws4r1_q8, 1, 1, 0)
WS4R_ENTRY(attn_ws4r2, 2, 0, 0)
WS4R_ENTRY(attn_ws4r2_q8, 2, 1, 0)
WS4R_ENTRY(attn_ws4r1b, 1, 0, 1)
WS4R_ENTRY(attn_ws4r1b_q8, 1, 1, 1)
WS4R_ENTRY(attn_ws4r2b, 2, 0, 1)
WS4R_ENTRY(attn_ws4r2b_q8, 2, 1, 1)
#undef WS4R_ENTRY
