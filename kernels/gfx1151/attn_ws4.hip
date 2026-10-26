// WHIRL device code (gfx1151 / RDNA 3.5): attn_ws4, 4-wave pipelined split-K decode attention.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by kernels/gfx1151/whirl_kernels_gfx1151.hip after attn.hip
// (KvArgs, kv_dq8 / kv_dq4, AwGroups, aw_qk_chain / aw_softmax_step / aw_pv).
// Design: audit/DES-ATTN-8S.md (sections 3.4-3.7, T1 F4 / F7, T0). The LDS-only barrier idea
// (s_barrier fenced on the local address space only, so in-flight global loads are not drained
// and the vector L0 is not invalidated) comes from asf0/strix-llama.cpp fattn-rdna35.cu
// (MIT, Copyright (c) 2023-2026 The ggml authors); the code here is written from scratch.
//
// attn_ws4<F, PD, BF> (F: 0 f16, 1 int8 (q8 / q8h), 2 int4 (q4); PD: chunks prefetched in
// registers, 1 for all (2 spills); BF: branch-free loads, see ws4_load). Same arguments, grid
// (kv head, split, group), 128 threads and partial layout as attn_wsplit1[_q8|_q4], and bitwise
// the same partials for groups of one query (the gfx1151 host's only kind, caps.attn_group1). Per 32-key chunk:
//   LB0  Ks / Vt of the chunk ready
//   QK   wave kt (0, 1): aw_qk_chain over its 16 keys from Ks -> Sc (f32, C-fragment order)
//   LB1  every wave: aw_softmax_step on both tiles (identical bits in all waves), aw_pv on its
//        own 64 dims (dt 4 w .. 4 w + 3) from Vt
//   LB2  stage the next chunk's raw K / V bytes (registers, issued PD chunks ahead) into
//        Ks / Vt (in-register dequant: kv_dq8 / kv_dq4, the same rounding as kv_ld16 / kv_ld8),
//        then issue the raw loads of chunk + 1 + PD into the freed registers
// K / V bytes are loaded by all 128 threads, once per block, except: keys >= b1 of the last
// chunk read row b1 - 1 again (the K clamp of attn_wsplit_impl; V is zeroed when staged), and
// with BF the prefetch past the last chunk reloads the last chunk (L2; < 2% at 32k+).
// The split rule uses the constant ws4::kNs (64), not gridDim.y: the splits depend on n_pos
// only (the same ranges as attn_wsplit1 whenever n_pos <= 64 * gridDim.y, i.e. always).
// ---------------------------------------------------------------------------
#include "attn_ws4_index.h"

__device__ __forceinline__ void ws4_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// feat026-attndec8s: on gfx11 the compiler puts s_waitcnt vmcnt(0) before EVERY s_barrier (no
// FeatureBackOffBarrier for GFX11, LLVM 4b89e8adda), fences or not, so ws4_lds_barrier drains the
// register prefetch. B = 1: the barrier as inline asm (opaque to the waitcnt pass, as hipfire
// AT_BARRIER): this wave's LDS reads / writes done (lgkmcnt(0)), then s_barrier; global loads stay
// in flight. Only LDS is exchanged between the waves here. Same instructions on the data, so the
// same bits; B = 0 is the shipped barrier.
// Why the compiler cannot break it: asm volatile has unmodeled side effects (a scheduling barrier at
// the MI level; the "memory" clobber orders it at the IR level), and clang marks inline asm in HIP
// device code convergent, so it is never duplicated into divergent paths (every caller's condition is
// block-uniform: c, nch, cidx, kAlias). VGPR dependencies on the in-flight loads still get their
// vmcnt from SIInsertWaitcnts (the asm does not touch those registers).
// Risk accepted (review C2): without FeatureBackOffBarrier (GFX11) a wave waiting in s_barrier with
// VMEM still outstanding cannot be backed out of the barrier on an exception (trap / debugger / bad
// address), so a corrupt KV pointer may hang the GPU (TDR) instead of a clean page fault. All loads
// here are clamped to [b0, b1 - 1]; only a bad KV pool pointer can trigger it. hipfire ships the same
// barrier (AT_BARRIER).
template <int B>
__device__ __forceinline__ void ws4_bar() {
    if constexpr (B == 0)
        ws4_lds_barrier();
    else
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// scales as f16 vectors (constant lane indices after unrolling; a _Float16 array was promoted
// to LDS by the compiler)
typedef _Float16 ws4_h4 __attribute__((ext_vector_type(4)));
template <int F> struct Ws4Raw;
template <> struct Ws4Raw<0> {  // f16: K 8 x 8 values, V 8 keys x 8 values
    uint4 k[8];
    uint4 v[8];
};
template <> struct Ws4Raw<1> {  // int8: K 4 x 16 values + scale, V 8 keys x 8 values + scale
    uint4 k[4];
    uint2 v[8];
    ws4_h4 ks;
    half8 vs;
};
template <> struct Ws4Raw<2> {  // int4: K 4 x 16 nibbles + scale, V 8 keys x 8 nibbles + scale
    uint2 k[4];
    uint32_t v[8];
    ws4_h4 ks;
    half8 vs;
};

// issue the raw loads of chunk k0 (keys < b1) of kv head g for thread tv
// BF (branch-free): every load unconditional. Otherwise V rows >= b1 are not read (zero) and
// the caller skips loads past the last chunk; the loads under a branch make the compiler copy
// their results at the join, i.e. wait right away (the prefetch then only covers the staging).
// f16 / int8 use BF (+13% for int8); int4 does not (BF needs 4 more VGPRs than 256 there).
// dl: kv_delta(kv, tb, k0), looked up by the caller (32-key chunks never straddle a page)
template <int F, bool BF>
__device__ __forceinline__ void ws4_load_dl(Ws4Raw<F> &r, const KvArgs &kv, int dl, int k0, int b1, int g, int n_kv, int tv) {
#pragma unroll
    for (int p = 0; p < ws4::k_pieces(F); ++p) {
        const int key = min(k0 + ws4::k_key(F, tv, p), b1 - 1);  // clamp (attn_wsplit_impl krow)
        const size_t e = ((size_t)(key + dl) * n_kv + g) * AW_D + ws4::k_d0(F, tv);
        if constexpr (F == 0) {
            r.k[p] = *(const uint4 *)((const _Float16 *)kv.k + e);
        } else if constexpr (F == 1) {
            r.k[p] = *(const uint4 *)((const int8_t *)kv.k + e);
            r.ks[p] = kv.ks[e >> 5];
        } else {
            r.k[p] = *(const uint2 *)((const uint8_t *)kv.k + (e >> 1));
            r.ks[p] = kv.ks[e >> 5];
        }
    }
    // V rows (BF: keys >= b1 read row b1 - 1 again; zeroed when staged either way)
    const int kb = ws4::v_kb(tv), d8 = ws4::v_d8(tv);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int key = min(k0 + kb * 8 + i, b1 - 1);
        const size_t e = ((size_t)(key + dl) * n_kv + g) * AW_D + d8;
        if (!BF && k0 + kb * 8 + i >= b1) {
            if constexpr (F == 0) {
                r.v[i] = make_uint4(0, 0, 0, 0);
            } else if constexpr (F == 1) {
                r.v[i] = make_uint2(0, 0);
                r.vs[i] = (_Float16)0.f;
            } else {
                r.v[i] = 0;
                r.vs[i] = (_Float16)0.f;
            }
        } else if constexpr (F == 0) {
            r.v[i] = *(const uint4 *)((const _Float16 *)kv.v + e);
        } else if constexpr (F == 1) {
            r.v[i] = *(const uint2 *)((const int8_t *)kv.v + e);
            r.vs[i] = kv.vs[e >> 5];
        } else {
            r.v[i] = *(const uint32_t *)((const uint8_t *)kv.v + (e >> 1));
            r.vs[i] = kv.vs[e >> 5];
        }
    }
}
template <int F, bool BF>
__device__ __forceinline__ void ws4_load(Ws4Raw<F> &r, const KvArgs &kv, int tb, int k0, int b1, int g, int n_kv, int tv) {
    ws4_load_dl<F, BF>(r, kv, kv_delta(kv, tb, k0), k0, b1, g, n_kv, tv);
}

// dequantize / move the raw chunk k0 into Ks [32][kQs] and Vt (attn_wsplit_impl key permutation)
template <int F>
__device__ __forceinline__ void ws4_stage(const Ws4Raw<F> &r, _Float16 (*Ks)[ws4::kQs], _Float16 *Vt, int k0, int b1, int tv) {
#pragma clang fp contract(off)
#pragma unroll
    for (int p = 0; p < ws4::k_pieces(F); ++p) {
        _Float16 *dst = &Ks[ws4::k_key(F, tv, p)][ws4::k_d0(F, tv)];
        if constexpr (F == 0) {
            *(half8 *)dst = __builtin_bit_cast(half8, r.k[p]);
        } else if constexpr (F == 1) {
            *(half8 *)dst = kv_dq8(__builtin_bit_cast(kvq8_t, make_uint2(r.k[p].x, r.k[p].y)), r.ks[p]);
            *(half8 *)(dst + 8) = kv_dq8(__builtin_bit_cast(kvq8_t, make_uint2(r.k[p].z, r.k[p].w)), r.ks[p]);
        } else {
            *(half8 *)dst = kv_dq4(r.k[p].x, r.ks[p]);
            *(half8 *)(dst + 8) = kv_dq4(r.k[p].y, r.ks[p]);
        }
    }
    // V^T: even rows (keys 8 kb + 0, 2, 4, 6) to column tb, then odd rows to tb + 8 (half the
    // temporaries of dequantizing all 8 rows at once; same values and layout)
    typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
    const int kb = ws4::v_kb(tv), d8 = ws4::v_d8(tv), tb = ws4::vt_tb(kb);
#pragma unroll
    for (int par = 0; par < 2; ++par) {
        half8 rows[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = 2 * j + par;
            half8 x;
            if constexpr (F == 0)
                x = __builtin_bit_cast(half8, r.v[i]);
            else if constexpr (F == 1)
                x = kv_dq8(__builtin_bit_cast(kvq8_t, r.v[i]), r.vs[i]);
            else
                x = kv_dq4(r.v[i], r.vs[i]);
            const bool valid = k0 + kb * 8 + i < b1;  // keys >= b1: +0 as attn_wsplit_impl
#pragma unroll
            for (int e2 = 0; e2 < 8; ++e2) rows[j][e2] = valid ? x[e2] : (_Float16)0.f;
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            half4_t w4;
#pragma unroll
            for (int j = 0; j < 4; ++j) w4[j] = rows[j][c];
            *(half4_t *)&Vt[ws4::vt_addr(d8 + c, tb + 8 * par)] = w4;
        }
    }
}

#ifndef WS4_DBG  // kernel development: 1 = no QK / softmax / P.V, 2 = no global loads after the prologue
#define WS4_DBG 0
#endif
// TU (feat026-attndec8s tune bits, 0 = shipped): 1 inline-asm barriers (ws4_bar<1>), 2 page-table
// lookahead (the kv_delta of the chunk loaded next is read one load earlier, after the current
// chunk's K / V loads, so its wait is not a serial round trip right before them). Same addresses,
// same data, same arithmetic: bitwise the shipped partials.
// The lookahead is applied to the int8 / int4 entries only: with f16 K / V (64 raw VGPRs) it spilled 3 VGPRs
// (attn_ws4 245 -> 256 VGPR, 16 B scratch; scratch\f26attndec8s\regs.tsv, first m31 build).
template <int F, int PD, bool BF, int TU0 = 0>
__device__ __forceinline__ void attn_ws4_impl(const float *q, const KvArgs &kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                                              int q_stride, const int *pos, float scale, const int *skip, const AwGroups &groups) {
#pragma clang fp contract(off)
    constexpr int TU = F == 0 ? (TU0 & 1) : TU0;
    if (skip != nullptr && *skip) return;
    __shared__ __align__(16) _Float16 Qs[16][ws4::kQs];
    __shared__ __align__(16) _Float16 Ks[AW_K][ws4::kQs];
    __shared__ __align__(16) _Float16 Vt[ws4::kVtSize];
    __shared__ __align__(16) float Sc[ws4::kScSize];
    const int tv = threadIdx.x, lane = tv % WARP, wid = tv / WARP;
    const int g = blockIdx.x, split = blockIdx.y, n_split = gridDim.y, z = blockIdx.z;
    const int grp = n_head / n_kv;
    const int first = groups.first[z], count = groups.count[z];
    const int ncol = count * grp;  // <= 16 (one query per group on gfx1151)
    const int tb = kv_tb(kv, first);
    const int n_pos = pos[first + count - 1] + 1;
    const ws4::Range rg = ws4::block_range(n_pos, split, ws4::kNs, kv.win);
    const int b0 = rg.b0, b1 = rg.b1;
    const size_t ml_q = (size_t)n_split * n_head * 2, acc_q = (size_t)n_split * n_head * AW_D;
    const int col = lane % 16;
    const bool cvalid = col < ncol;
    const int colc = cvalid ? col : 0;
    const int t = first + colc / grp, h = g * grp + colc % grp;
    const int my_pos = cvalid ? pos[t] : -1;
    if (b0 >= n_pos) {
        if (wid == 0 && cvalid && lane < 16) {
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = -INFINITY;
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = 0.f;
        }
        return;
    }
    const int nch = (b1 - b0 + AW_K - 1) / AW_K;
    Ws4Raw<F> raw[PD];
    ws4_load<F, BF>(raw[0], kv, tb, b0, b1, g, n_kv, tv);
    // Q^T (f16) of the group's columns, as attn_wsplit_impl
    for (int i = tv; i < 16 * (AW_D / 8); i += 128) {
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
    ws4_stage<F>(raw[0], Ks, Vt, b0, b1, tv);
#pragma unroll
    for (int j = 1; j <= PD; ++j)
        if (BF || j < nch) ws4_load<F, BF>(raw[j % PD], kv, tb, b0 + min(j, nch - 1) * AW_K, b1, g, n_kv, tv);
    // TU & 2: page-table entry of the first in-loop load (chunk min(1 + PD, nch - 1))
    int dl_next = 0;
    if constexpr ((TU & 2) != 0) dl_next = kv_delta(kv, tb, b0 + min(1 + PD, nch - 1) * AW_K);
    float8 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    for (int i0 = 0; i0 < nch; i0 += PD) {
#pragma unroll
        for (int s = 0; s < PD; ++s) {
            const int c = i0 + s;
            if (c < nch) {
                const int k0 = b0 + c * AW_K;
                ws4_bar<TU & 1>();  // LB0: Ks / Vt (and Qs) of chunk c written
#if WS4_DBG != 1
                if (wid < 2) {
                    const int kt = wid;
                    const float8 sc = aw_qk_chain((float8){0, 0, 0, 0, 0, 0, 0, 0}, &Qs[lane % 16][0],
                                                  [&](int st) { return ld_frag16(&Ks[kt * 16 + lane % 16][16 * st]); });
                    *(float4 *)&Sc[ws4::sc_addr(kt, lane, 0)] = (float4){sc[0], sc[1], sc[2], sc[3]};
                    *(float4 *)&Sc[ws4::sc_addr(kt, lane, 4)] = (float4){sc[4], sc[5], sc[6], sc[7]};
                }
                ws4_bar<TU & 1>();  // LB1: Sc written, Ks free
                float8 sc[AW_K / 16];
#pragma unroll
                for (int kt = 0; kt < AW_K / 16; ++kt) {
                    const float4 a = *(const float4 *)&Sc[ws4::sc_addr(kt, lane, 0)];
                    const float4 b = *(const float4 *)&Sc[ws4::sc_addr(kt, lane, 4)];
                    sc[kt] = (float8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                }
                half8 pt[AW_K / 16];
                const float alpha = aw_softmax_step(sc, pt, k0, lane, my_pos, b1, scale, m, l);
                aw_pv<4>(o, alpha, pt, lane,
                         [&](int j, int kt) { return ld_frag16(&Vt[ws4::vt_addr((4 * wid + j) * 16 + lane % 16, kt * 16)]); });
#endif
                if (c + 1 < nch) {
                    ws4_bar<TU & 1>();  // LB2: Vt, Ps read
                    ws4_stage<F>(raw[(s + 1) % PD], Ks, Vt, k0 + AW_K, b1, tv);
                    // BF: unconditional (see ws4_load); past the last chunk it reloads the last
                    // one (L2), never used
                    if constexpr ((TU & 2) == 0) {
                        if (WS4_DBG != 2 && (BF || c + 1 + PD < nch))
                            ws4_load<F, BF>(raw[(s + 1) % PD], kv, tb, min(k0 + (1 + PD) * AW_K, b0 + (nch - 1) * AW_K), b1, g, n_kv, tv);
                    } else {
                        if (WS4_DBG != 2 && (BF || c + 1 + PD < nch))
                            ws4_load_dl<F, BF>(raw[(s + 1) % PD], kv, dl_next, min(k0 + (1 + PD) * AW_K, b0 + (nch - 1) * AW_K), b1, g, n_kv, tv);
                        // the entry of the following load (clamped as that load), issued after this one
                        dl_next = kv_delta(kv, tb, min(k0 + (2 + PD) * AW_K, b0 + (nch - 1) * AW_K));
                    }
                }
            }
        }
    }
    if (!cvalid) return;
    if (wid == 0 && lane < 16) {
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = m;
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = l;
    }
    float *pa = part_acc + t * acc_q + ((size_t)split * n_head + h) * AW_D;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) pa[(4 * wid + j) * 16 + CROW(lane, v)] = o[j][v];
}

#define WS4_ENTRY(name, F, PD, BF)                                                                                                        \
    extern "C" __global__ void __launch_bounds__(128)                                                                                 \
        name(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv, int q_stride, const int *pos, float scale, \
             const int *skip, AwGroups groups) {                                                                                      \
        attn_ws4_impl<F, PD, BF, AD8S_WS4_TU>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);                            \
    }
WS4_ENTRY(attn_ws4, 0, 1, true)
WS4_ENTRY(attn_ws4_q8, 1, 1, true)
WS4_ENTRY(attn_ws4_q4, 2, 1, false)
// Prefetch depth 2 (template PD, design F7) does not fit any format with ROCm 7.2: 256 VGPRs
// and 50-140 spilled once the prefetch really overlaps (and slower than depth 1), so all use 1.
#undef WS4_ENTRY

// ---------------------------------------------------------------------------
// attn_ws4g<CG>: multi-row verify (design 3.3 mode A, T0 result; task T3). One block reads the
// KV of its split once for up to rows_per_block(CG, grp) query rows x the grp GQA heads of kv
// head g (CG 16-column WMMA tiles, columns packed densely: column c = (row c / grp, head c % grp);
// 8 rows x 6 heads = 48 columns at CG 3). A group with more rows uses several column-group
// blocks on grid.x (blockIdx.x = g + n_kv * column group), dispatched next to each other so
// the later reads of the same split hit L2. Grid (n_kv * column groups, splits, groups),
// 256 threads, otherwise the attn_wsplit1 ABI / partial layout.
// 8 waves (not 4 as attn_ws4): the 48 columns' accumulators are 96 VGPRs per lane with 128
// threads, plus a 64-VGPR f16 prefetch, which spilled; with 256 threads waves 0-3 prefetch /
// stage K and waves 4-7 V (32 VGPRs each, one register set: the role only selects addresses),
// QK + softmax of tile c in wave c (p / alpha to LDS), P.V dims [32 w, 32 w + 32) of every tile in wave w.
// Bitwise row-count invariance: every row's partials equal its single-row (attn_ws4 /
// attn_wsplit1) partials whenever the rows of a group share the split rule (host: same awPer and
// draft-window parameters). Per column:
//  - QK: the column's Q operand bits and the K rows of its valid keys are those of the solo run;
//    the other columns and the K rows of keys past its position (real here, clamped in solo) do
//    not change its valid scores (T0: WMMA columns and output rows are independent, (a) (c)).
//  - softmax: aw_softmax_step masks with the row's own position (same bits as solo).
//  - P.V: WMMA is NOT independent of A columns whose B entries are 0 (T0 (b)), so in a chunk
//    holding keys past row r's position (only in r's last split) the row's columns take a
//    variant P.V whose V^T fragment has those keys zeroed in registers, exactly the solo V^T.
//  - chunks entirely past row r's position (and splits past it) are skipped for r: m, l, o kept
//    by select, (-inf, 0) written for the split (T0 (d): even WMMA(C, B = 0) can change C).
// The block's range uses its own rows' largest position (n_pos); the tail logic runs only in
// chunks with a key past the smallest position (block-uniform branch).
// ---------------------------------------------------------------------------
#ifndef WS4G_QK_UNROLL  // QK st loop unroll (full unrolling hoists fragment loads: CG 3 spilled)
#define WS4G_QK_UNROLL 4
#endif
#ifndef WS4G_NOTAIL  // kernel development: 1 = no tail path (wrong bits for groups)
#define WS4G_NOTAIL 0
#endif

// one thread's raw part of a chunk: K (waves 0-3, the f16 K map of ws4::k_key / k_d0 for every
// format: 8 pieces of 8 values) or V (waves 4-7: key block v_kb, dims v_d8, 8 keys)
template <int F> struct Ws4gRaw;
template <> struct Ws4gRaw<0> {
    uint4 x[8];
};
template <> struct Ws4gRaw<1> {
    uint2 x[8];
    half8 s;
};

// branch-free: the role only selects base pointers and offsets; keys >= b1 read row b1 - 1
// (K: the attn_wsplit_impl clamp; V: zeroed when staged)
// LA (feat026-attndec8s page-table lookahead): dl_la = kv_delta(kv, tb, k0), looked up by the caller. LA = false is the
// shipped body token for token (a wrapper function changed the q8 entries' register allocation)
template <int F, bool LA = false>
__device__ __forceinline__ void ws4g_load(Ws4gRaw<F> &r, const KvArgs &kv, int tb, int k0, int b1, int g, int n_kv, bool isv, int tq, int dl_la = 0) {
    const int dl = LA ? dl_la : kv_delta(kv, tb, k0);
    const int kb = ws4::v_kb(tq);
    const int d = isv ? ws4::v_d8(tq) : ws4::k_d0(0, tq);
    // pointer arithmetic, not a select of the two pointers (a select became a private array
    // indexed per chunk: scratch loads queued behind the K / V loads)
    const char *base = (const char *)kv.k + (isv ? (const char *)kv.v - (const char *)kv.k : 0);
    const _Float16 *sb = kv.ks + (isv ? kv.vs - kv.ks : 0);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int kk = isv ? kb * 8 + i : ws4::k_key(0, tq, i);
        const int key = min(k0 + kk, b1 - 1);
        const size_t e = ((size_t)(key + dl) * n_kv + g) * AW_D + d;
        if constexpr (F == 0) {
            r.x[i] = *(const uint4 *)((const _Float16 *)base + e);
        } else {
            r.x[i] = *(const uint2 *)((const int8_t *)base + e);
            r.s[i] = sb[e >> 5];
        }
    }
}

template <int F>
__device__ __forceinline__ half8 ws4g_val(const Ws4gRaw<F> &r, int i) {
    if constexpr (F == 0)
        return __builtin_bit_cast(half8, r.x[i]);
    else
        return kv_dq8(__builtin_bit_cast(kvq8_t, r.x[i]), r.s[i]);  // kv_ld8 / kv_ld16 rounding
}

template <int F>
__device__ __forceinline__ void ws4g_stage(const Ws4gRaw<F> &r, _Float16 (*Ks)[ws4::kQs], _Float16 *Vt, int k0, int b1, bool isv, int tq) {
#pragma clang fp contract(off)
    if (!isv) {
#pragma unroll
        for (int p = 0; p < 8; ++p) *(half8 *)&Ks[ws4::k_key(0, tq, p)][ws4::k_d0(0, tq)] = ws4g_val<F>(r, p);
        return;
    }
    typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
    const int kb = ws4::v_kb(tq), d8 = ws4::v_d8(tq), tb = ws4::vt_tb(kb);
#pragma unroll
    for (int par = 0; par < 2; ++par) {
        half8 rows[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = 2 * j + par;
            const half8 x = ws4g_val<F>(r, i);
            const bool valid = k0 + kb * 8 + i < b1;  // keys >= b1: +0 as attn_wsplit_impl
#pragma unroll
            for (int e2 = 0; e2 < 8; ++e2) rows[j][e2] = valid ? x[e2] : (_Float16)0.f;
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            half4_t w4;
#pragma unroll
            for (int j = 0; j < 4; ++j) w4[j] = rows[j][c];
            *(half4_t *)&Vt[ws4::vt_addr(d8 + c, tb + 8 * par)] = w4;
        }
    }
}

// TU: the attn_ws4_impl tune bits (1 inline-asm barriers, 2 page-table lookahead; 0 = shipped)
// (lookahead: int8 entries only; f16 CG 3 spilled 5 -> 14 VGPRs with it)
template <int F, int CG, int TU0 = 0>
__device__ __forceinline__ void attn_ws4g_impl(const float *q, const KvArgs &kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                                               int q_stride, const int *pos, float scale, const int *skip, const AwGroups &groups) {
#pragma clang fp contract(off)
    constexpr int TU = F == 0 ? (TU0 & 1) : TU0;
    if (skip != nullptr && *skip) return;
    // CG 3: Qs 25.3 KB + Ks 16.9 + Vt 21 = 63.2 KB, so p / alpha live in the Ks space (one
    // more LDS barrier per chunk: every wave's QK reads of Ks done before they are written)
    constexpr bool kAlias = CG == 3;
    constexpr int NDT = AW_D / 16 / 8;  // P.V 16-dim tiles per wave
    __shared__ __align__(16) _Float16 Qs[16 * CG][ws4::kQs];
    __shared__ __align__(16) _Float16 Ks[AW_K][ws4::kQs];
    __shared__ __align__(16) _Float16 Vt[ws4::kVtSize];
    // Ps: p (f16) [tile][kt][lane][8]; As: alpha [tile][16 columns] (CG 3: in the Ks space)
    constexpr int kPs = CG * 2 * 32 * 8, kAs = CG * 16;
    static_assert(kPs * 2 + kAs * 4 <= AW_K * ws4::kQs * 2, "Ps / As fit in Ks");
    __shared__ __align__(16) _Float16 PsOwn[kAlias ? 8 : kPs];
    __shared__ __align__(16) float AsOwn[kAlias ? 4 : kAs];
    _Float16 *Ps = kAlias ? reinterpret_cast<_Float16 *>(&Ks[0][0]) : PsOwn;
    float *As = kAlias ? reinterpret_cast<float *>(Ps + kPs) : AsOwn;
    const int tv = threadIdx.x, lane = tv % WARP, wid = tv / WARP;
    const bool isv = wid >= 4;
    const int tq = tv & 127;
    const int grp = n_head / n_kv, rpb = ws4::rows_per_block(CG, grp);
    const int g = blockIdx.x % n_kv, cgi = blockIdx.x / n_kv;
    const int split = blockIdx.y, n_split = gridDim.y, z = blockIdx.z;
    const int first = groups.first[z] + cgi * rpb, nrows = min(groups.count[z] - cgi * rpb, rpb);
    if (nrows <= 0) return;
    const int ncol = nrows * grp;
    int pmin = pos[first], pmax = pos[first];
    for (int r = 1; r < nrows; ++r) {
        pmin = min(pmin, pos[first + r]);
        pmax = max(pmax, pos[first + r]);
    }
    const int tb = kv_tb(kv, first);
    const int n_pos = pmax + 1;
    const ws4::Range rg = ws4::block_range(n_pos, split, ws4::kNs, kv.win);
    const int b0 = rg.b0, b1 = rg.b1;
    const size_t ml_q = (size_t)n_split * n_head * 2, acc_q = (size_t)n_split * n_head * AW_D;
    // per tile c: this lane's column, its row (crow_), position (mp); output row / head
    // recomputed where written (fewer registers live through the chunk loop: f16 CG 3 spilled)
    bool cvalid[CG];
    int crow_[CG], mp[CG];
    auto colc = [&](int c) { return 16 * c + lane % 16 < ncol ? 16 * c + lane % 16 : 0; };
#pragma unroll
    for (int c = 0; c < CG; ++c) {
        cvalid[c] = 16 * c + lane % 16 < ncol;
        crow_[c] = colc(c) / grp;
        mp[c] = cvalid[c] ? pos[first + crow_[c]] : -1;
    }
    auto ct = [&](int c) { return first + colc(c) / grp; };
    auto ch = [&](int c) { return g * grp + colc(c) % grp; };
    if (b0 >= n_pos) {
        if (wid == 0 && lane < 16) {
#pragma unroll
            for (int c = 0; c < CG; ++c)
                if (cvalid[c]) {
                    part_ml[ct(c) * ml_q + ((size_t)split * n_head + ch(c)) * 2 + 0] = -INFINITY;
                    part_ml[ct(c) * ml_q + ((size_t)split * n_head + ch(c)) * 2 + 1] = 0.f;
                }
        }
        return;
    }
    const int nch = (b1 - b0 + AW_K - 1) / AW_K;
    Ws4gRaw<F> raw;
    ws4g_load<F>(raw, kv, tb, b0, b1, g, n_kv, isv, tq);
    for (int i = tv; i < 16 * CG * (AW_D / 8); i += 256) {
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
    ws4g_stage<F>(raw, Ks, Vt, b0, b1, isv, tq);
    ws4g_load<F>(raw, kv, tb, b0 + min(1, nch - 1) * AW_K, b1, g, n_kv, isv, tq);
    float8 o[CG][NDT];
#pragma unroll
    for (int c = 0; c < CG; ++c)
#pragma unroll
        for (int j = 0; j < NDT; ++j) o[c][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    // softmax state of tile wid (waves < CG): m, l and the column's position
    float mw = -INFINITY, lw = 0.f;
    int mpw = mp[0];
#pragma unroll
    for (int c = 1; c < CG; ++c) mpw = wid == c ? mp[c] : mpw;
    auto vfrag = [&](int j, int kt) { return ld_frag16(&Vt[ws4::vt_addr((NDT * wid + j) * 16 + lane % 16, kt * 16)]); };
    for (int cidx = 0; cidx < nch; ++cidx) {
        const int k0 = b0 + cidx * AW_K;
        const int kend = min(k0 + AW_K, b1) - 1;  // last key of the chunk
        const bool tail = pmin < kend;            // some row has keys of this chunk past its position
        ws4_bar<TU & 1>();                        // LB0: Ks / Vt (and Qs) of the chunk written
        // TU & 2 (r1 schedule): the page-table entry of the load issued after LB2 of THIS chunk, read here
        // (unconditional, clamped as that load), a whole chunk of compute earlier. It is younger than the
        // in-flight prefetch, which ws4g_stage consumes first at LB2 anyway, so no young load is drained
        // for it. (The first version read the entry of the load after next AFTER the 16 prefetch loads;
        // the loop end then got a vmcnt(12) draining 5 of them.)
        int dl_la = 0;
        if constexpr ((TU & 2) != 0) dl_la = kv_delta(kv, tb, min(k0 + 2 * AW_K, b0 + (nch - 1) * AW_K));
        // QK + softmax of tile c in wave c (its m, l live there): the two kt chains of aw_qk_chain
        // interleaved (one Q fragment load per st for both; each chain's order unchanged)
        half8 ptw[AW_K / 16];
        float alphaw = 0.f;
        if (wid < CG) {  // one code copy: the tile is wid (wave-uniform)
            float8 sc[AW_K / 16];
#pragma unroll
            for (int kt = 0; kt < AW_K / 16; ++kt) sc[kt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
            const _Float16 *qrow = &Qs[wid * 16 + lane % 16][0];
#pragma unroll WS4G_QK_UNROLL
            for (int st = 0; st < AW_D / 16; ++st) {
                const half16 qb = ld_frag16(qrow + 16 * st);
#pragma unroll
                for (int kt = 0; kt < AW_K / 16; ++kt) sc[kt] = WMMA16(ld_frag16(&Ks[kt * 16 + lane % 16][16 * st]), qb, sc[kt]);
            }
            // tail chunk: rows before the chunk keep m, l (select)
            float m2 = mw, l2 = lw;
            alphaw = aw_softmax_step(sc, ptw, k0, lane, mpw, b1, scale, m2, l2);
            if (WS4G_NOTAIL || !tail || k0 <= mpw) {
                mw = m2;
                lw = l2;
            }
        }
        if constexpr (kAlias) ws4_bar<TU & 1>();  // every wave's Ks reads done (Ps / As overwrite Ks)
        if (wid < CG) {
#pragma unroll
            for (int kt = 0; kt < AW_K / 16; ++kt) *(half8 *)&Ps[(wid * 2 + kt) * 32 * 8 + lane * 8] = ptw[kt];
            if (lane < 16) As[wid * 16 + lane] = alphaw;  // same in both half-waves (m, mx reduced over lane ^ 16)
        }
        ws4_bar<TU & 1>();  // LBP: p, alpha written
        if (WS4G_NOTAIL || !tail) {
            // P.V: each V^T fragment loaded once for every tile; per column (and 16-dim tile) the
            // operations of aw_pv in its order: o * alpha, then WMMA kt 0, then kt 1
#pragma unroll
            for (int c = 0; c < CG; ++c) {
                const float alpha = As[c * 16 + lane % 16];
#pragma unroll
                for (int j = 0; j < NDT; ++j)
#pragma unroll
                    for (int v = 0; v < 8; ++v) o[c][j][v] *= alpha;
            }
#pragma unroll
            for (int kt = 0; kt < AW_K / 16; ++kt) {
                half16 pb[CG];
#pragma unroll
                for (int c = 0; c < CG; ++c) pb[c] = join_halves(*(const half8 *)&Ps[(c * 2 + kt) * 32 * 8 + lane * 8], lane);
#pragma unroll
                for (int j = 0; j < NDT; ++j) {
                    const half16 a = vfrag(j, kt);
#pragma unroll
                    for (int c = 0; c < CG; ++c) o[c][j] = WMMA16(a, pb[c], o[c][j]);
                }
            }
        } else {
#pragma unroll
        for (int c = 0; c < CG; ++c) {
            half8 pt[AW_K / 16];
#pragma unroll
            for (int kt = 0; kt < AW_K / 16; ++kt) pt[kt] = *(const half8 *)&Ps[(c * 2 + kt) * 32 * 8 + lane * 8];
            const float alpha = As[c * 16 + lane % 16];
            {
                // tail chunk: rows before the chunk are skipped (kept by select), rows ending
                // inside it take the variant P.V (keys past the row's position zeroed in V^T)
                const bool act = k0 <= mp[c];
                half16 pb[AW_K / 16];
#pragma unroll
                for (int kt = 0; kt < AW_K / 16; ++kt) pb[kt] = join_halves(pt[kt], lane);
                const int rlo = (16 * c) / grp, rhi = min(nrows - 1, (16 * c + 15) / grp);
#pragma unroll
                for (int j = 0; j < NDT; ++j) {
                    // same operations as aw_pv: o * alpha, then WMMA kt 0, kt 1
                    float8 x = o[c][j];
#pragma unroll
                    for (int v = 0; v < 8; ++v) x[v] *= alpha;
                    float8 d = WMMA16(vfrag(j, 0), pb[0], x);
                    d = WMMA16(vfrag(j, 1), pb[1], d);
                    for (int r = rlo; r <= rhi; ++r) {
                        const int pr = pos[first + r];
                        if (pr >= k0 && pr < kend) {  // block-uniform
                            const int pk = pr - k0;
                            half16 a0 = vfrag(j, 0), a1 = vfrag(j, 1);
#pragma unroll
                            for (int e = 0; e < 16; ++e) {
                                if (ws4::vt_key(e) > pk) a0[e] = (_Float16)0.f;
                                if (ws4::vt_key(16 + e) > pk) a1[e] = (_Float16)0.f;
                            }
                            float8 dv = WMMA16(a0, pb[0], x);
                            dv = WMMA16(a1, pb[1], dv);
                            const bool mine = cvalid[c] && crow_[c] == r;
#pragma unroll
                            for (int v = 0; v < 8; ++v) d[v] = mine ? dv[v] : d[v];
                        }
                    }
                    if (act) o[c][j] = d;
                }
            }
        }
        }
        if (cidx + 1 < nch) {
            ws4_bar<TU & 1>();  // LB2: Vt, Ps read
            ws4g_stage<F>(raw, Ks, Vt, k0 + AW_K, b1, isv, tq);
            // unconditional (a load under a branch is waited for at the join); past the last
            // chunk it reloads the last one (L2), never used
            if constexpr ((TU & 2) == 0) {
                ws4g_load<F>(raw, kv, tb, min(k0 + 2 * AW_K, b0 + (nch - 1) * AW_K), b1, g, n_kv, isv, tq);
            } else {
                ws4g_load<F, true>(raw, kv, tb, min(k0 + 2 * AW_K, b0 + (nch - 1) * AW_K), b1, g, n_kv, isv, tq, dl_la);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < CG; ++c) {
        if (!cvalid[c]) continue;
        if (wid == c && lane < 16) {  // the tile's softmax wave
            part_ml[ct(c) * ml_q + ((size_t)split * n_head + ch(c)) * 2 + 0] = mw;
            part_ml[ct(c) * ml_q + ((size_t)split * n_head + ch(c)) * 2 + 1] = lw;
        }
        if (mp[c] < b0) continue;  // split past the row: (-inf, 0), no acc (as the solo run)
        float *pa = part_acc + ct(c) * acc_q + ((size_t)split * n_head + ch(c)) * AW_D;
#pragma unroll
        for (int j = 0; j < NDT; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) pa[(NDT * wid + j) * 16 + CROW(lane, v)] = o[c][j][v];
    }
}

#define WS4G_ENTRY(name, F, CG)                                                                                                           \
    extern "C" __global__ void __launch_bounds__(256)                                                                                 \
        name(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv, int q_stride, const int *pos, float scale, \
             const int *skip, AwGroups groups) {                                                                                      \
        attn_ws4g_impl<F, CG, AD8S_WS4G_TU>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);                               \
    }
WS4G_ENTRY(attn_ws4g1, 0, 1)
WS4G_ENTRY(attn_ws4g2, 0, 2)
WS4G_ENTRY(attn_ws4g3, 0, 3)
WS4G_ENTRY(attn_ws4g1_q8, 1, 1)
WS4G_ENTRY(attn_ws4g2_q8, 1, 2)
WS4G_ENTRY(attn_ws4g3_q8, 1, 3)
#undef WS4G_ENTRY
