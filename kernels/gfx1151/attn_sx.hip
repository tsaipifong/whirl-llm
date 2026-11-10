// WHIRL device code (gfx1151 / RDNA 3.5): balance-mode prefill flash attention attn_sx2 / attn_sx4 / attn_sx8
// (v0252-strixc2, PLAN V1-09; never precise: the plan flag attn_sx is balance-only, see src/common/numerics.cpp).
// SPDX-License-Identifier: MIT
//
// Adapted from strix-llama.cpp (rdna35-perf branch, commit 62e9d99),
// ggml/src/ggml-cuda/fattn-rdna35.cu lines 4-861: flash_attn_rdna35 (https://github.com/asf0/strix-llama.cpp),
// through the KLIB-1 / v0252-b8s-c2 microbench extract (scratch/v0252_b8s/c2/fa_rdna35_kernel.h). Original code:
//   MIT License, Copyright (c) 2023-2026 The ggml authors.
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
//   associated documentation files (the "Software"), to deal in the Software without restriction,
//   including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
//   and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
//   subject to the following conditions: The above copyright notice and this permission notice shall be
//   included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
//   WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
//
// Changes for WHIRL (v0252-strixc2): the WHIRL prefill ABI (q, KvArgs, out, n_head, n_kv, q_stride, pos, n_tok,
// scale, h0; same as attn_prefill_wmma, so the step executor launches it unchanged), one sequence per launch; Q rows
// q[(token * n_head + head) * q_stride], output gate o / (1 + exp(-gate)) as attn_prefill_wmma; the causal mask is
// derived from the positions (pos[0] + token; no mask tensor, no mask_classify pass, no KV_max buffer): a step is
// unmasked when its last key <= the block's first position, else key > pos(token) -> -inf; the paged KV cache is
// walked with one kv_delta per 32-key step (KV_PAGE % NKEYS == 0); keys >= the block's last position + 1 are loaded
// from the clamped last row and staged as zero (unwritten pool rows may hold anything); head range h0 (TDR split);
// f16 K / V only, D = 256 only, no sinks / softcap / split-KV; the half-wave swap is ds_swizzle xor 16 (KLIB-1:
// permlanex16(fi=false) miscompiles on HIP SDK 7.2). Not bitwise equal to attn_prefill_wmma (GQA-packed columns,
// split-D scores, lazy rescale): f32 scores / O / row sums, f16 P - the balance item "f16 intermediates".
// Not self-contained: included LAST by kernels/gfx1151/whirl_kernels_gfx1151.hip (no existing kernel moves).
#include <hip/hip_fp16.h>

// Flash attention for RDNA3.5 WMMA (v_wmma_f32_16x16x16_f16), D = 64, 128, 256 or 512, f16 K/V. For prefill a K/V cache of
// another type (q8_0, q4_0, ...) is converted to f16 first, into the scratch space behind dst: at 32k keys and 4096
// tokens that is ~1% of the attention time, while the generic kernels such a cache fell back to are ~2x slower.
// Small batches at depth read q8_0 K/V directly (dequantized while staged to LDS): they are bound by the K/V reads.
//
// The generic MMA kernel keeps fp32 accumulators for 16 query columns x all V dims per warp (128 VGPRs at
// D = 256), so Q cannot stay in registers, every WMMA re-reads both operands from LDS and the kernel
// spills; D = 512 is not supported by it at all on RDNA and falls back to the FMA tile kernel.
// Here a block of 4*DQ warps owns 64 query columns (4 column tiles of 16) and walks the KV cache in steps of
// 64 / 32 / 32 / 16 keys at D = 64 / 128 / 256 / 512 (32 WMMAs per warp and step):
//   - S^T = K Q^T: the DQ = D/128 warps of a column tile split D (one warp at D <= 128); each keeps its share
//     of Q in registers (64 VGPRs), scores all key tiles of the step over its share (one LDS operand per WMMA) and the shares
//     are summed through LDS. Every warp of a column tile then holds all scores of its columns, so the
//     softmax needs no max exchange.
//   - P stays in the lane that owns the column; one permlanex16 turns the WMMA C layout into the B operand
//     layout.
//   - O^T += V^T P: warp w owns V dims [32w, 32w+32) for all 64 columns (64 VGPRs of accumulators). V is
//     transposed once per block while it is staged into LDS.
//   - K and V^T have separate LDS buffers: the next K and this step's V^T are both staged between the two
//     barriers of a step, from registers that were loaded while the scores were computed.
//   - a pre-pass classifies the mask per (token tile, 32 keys): all-zero steps ignore the mask, all -inf
//     steps are skipped, only mixed steps (near the diagonal) read it.
//   - the head group varies fastest across blocks, so the blocks in flight read whole KV cache rows.
//   - small batches (decode, speculative verification) have too few column tiles to fill the GPU: the key steps are
//     split over gridDim.y blocks, which write unnormalized partials and (max, row sum) for flash_attn_combine_results.
// Column c of a block is (token jt*NCOLS1 + c/NCOLS2, Q head head0 + c%NCOLS2); all NCOLS2 heads share
// one K/V head.

#define FATTN_RDNA35_AVAILABLE  // WHIRL: only in the gfx1151 code object

namespace fattn_rdna35 {

constexpr int CLS_KEYS   = 32;     // granularity of the mask classes
constexpr int MAX_CLS    = 8192;   // mask classes kept in LDS (2 bits each): up to 262144 keys
constexpr float m_init   = -1e30f; // finite so that fully masked columns never produce inf - inf
constexpr float log2e    = 1.44269504088896340736f;
constexpr float T_LAZY   = 8.0f;   // OWN: the reference max moves when a step max is more than this above it (log2 units)
constexpr float T_HI     = 14.0f;  // OWN: above this P would leave the f16 range -> redo the step against the new max

template <int D>
struct cfg {
    static constexpr int NCOLS  = 64;                   // query columns per block (32 at D = 512: 16.3 vs 25.6 TFLOPS, K/V reuse halves)
    static constexpr int NCT    = NCOLS / 16;           // column tiles
    static constexpr int DQ     = D > 128 ? D / 128 : 1; // warps per column tile, splitting D in the score phase
    static constexpr int QCH    = D / DQ / 16;          // 16-dim chunks of Q per warp
    static constexpr int NWARPS = NCT * DQ;
    static constexpr int NTHR   = NWARPS * 32;
    static constexpr int OT     = D / NWARPS / 16;      // warp w owns V dims [16*OT*w, 16*OT*(w+1))
    static constexpr int NKEYS  = D == 512 ? 16 : D == 64 ? 64 : 32; // keys per step: 32 WMMAs per warp and step
    static constexpr int KT     = NKEYS / 16;           // key tiles per step
    static constexpr int SEGS   = D / 8;                // 16 B segments per K/V row
    static constexpr int KPT    = NKEYS * SEGS / NTHR;  // K and V rows (keys) per thread and step
    // D = 512 runs at 192 VGPRs (2 blocks per WGP). With V prefetched as well and all P operands of a key tile loaded at
    // once, the loop spilled and its scratch reloads waited for the K/V loads (vmcnt is in order): 26.7 TFLOPS. Loading V
    // after the first barrier, one score chain and P one operand at a time: 29.5. At D = 256 both cost ~1%.
    static constexpr bool TIGHT = D == 512;
    // D = 256: warp dq of a column tile finishes key tile dq of the step (half the score exchange and softmax per warp).
    static constexpr bool OWN   = KT == DQ;
    static constexpr int SQK    = D + 8;                // row stride (halves) of sK:     conflict-free 16-row reads
    static constexpr int SVT    = NKEYS + 8;            // row stride (halves) of sVt/sP: conflict-free 16-row reads
    static constexpr int NUNITS = NKEYS / 8;            // 8-key units of a V^T row (swizzled)

    static constexpr int SZ_K   = NKEYS * SQK * 2;
    static constexpr int SZ_VT  = D * SVT * 2;
    static constexpr int SZ_P   = NCOLS * SVT * 2;
    // partial scores [warp][key tile][lane][8]; one warp per column tile (D <= 128) needs no exchange, only the column maxima
    static constexpr int SZ_SX  = DQ == 1 ? NCOLS * 4 : NWARPS * (OWN ? 1 : KT) * 32 * 8 * 4;
    static constexpr int SZ_RED = (OWN ? 4 : 2) * NCOLS * 4;            // per column: step maxima and row sums (OWN), or rescale factors and row sums
    static constexpr int SZ_CLS = MAX_CLS / 4;

    static constexpr int OFF_K   = 0;
    static constexpr int OFF_VT  = OFF_K   + SZ_K;
    static constexpr int OFF_P   = OFF_VT  + SZ_VT;
    static constexpr int OFF_SX  = OFF_P   + SZ_P;
    static constexpr int OFF_RED = OFF_SX  + SZ_SX;
    static constexpr int OFF_CLS = OFF_RED + SZ_RED;
    static constexpr int SZ_SMEM = OFF_CLS + SZ_CLS;

    static_assert(KPT == 2 || KPT == 4, "V transpose handles 2 or 4 keys per thread");
    static_assert(SEGS % 8 == 0 && NTHR % SEGS == 0, "thread layout");
    static_assert(QCH == 8 || (DQ == 1 && QCH == 4), "Q chunks per warp");
    static_assert(NKEYS % CLS_KEYS == 0 || CLS_KEYS % NKEYS == 0, "mask classes per step");
    static_assert(SZ_SMEM + 16 <= 65536, "LDS budget");
};

typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float    float8_t __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4_t  __attribute__((ext_vector_type(4))); // plain vector: stays in registers (HIP's int4 class may not)

#ifdef FATTN_RDNA35_AVAILABLE

static __device__ __forceinline__ float8_t wmma(const half16_t a, const half16_t b, const float8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// Barrier that only orders LDS: __syncthreads() also waits for all outstanding global loads (vmcnt(0)) and
// invalidates the vector L0 cache, which would cut the K/V prefetch short at every barrier.
static __device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// value of the lane in the other half of the wave (lane ^ 16). WHIRL: ds_swizzle xor 16 (the microbenched KLIB_SWAP 4;
// strix's permlanex16(fi=false) gives wrong results under HIP SDK 7.2, KLIB-1)
static __device__ __forceinline__ int swap_halves(const int x) {
    return __builtin_amdgcn_ds_swizzle(x, 0x401F);
}
static __device__ __forceinline__ float swap_halves(const float x) {
    return __int_as_float(swap_halves(__float_as_int(x)));
}

// q8_0 K/V (small batches, see the launcher): a 16 B f16 segment of 8 dims is a quarter of a q8_0 block. Its 8 quants
// are loaded raw with the block scale and dequantized when staged to LDS (same f16 values as dequantize_q8_0).
// The quants of segment s start at byte 34*(s/4) + 2 + 8*(s%4) of the row: 2 mod 4 in even blocks, where a third dword
// is loaded and the two data dwords are funnel-shifted out; rows and heads start at multiples of 4 B (host check).
static __device__ __forceinline__ u32x4_t load_seg_q8(const char * row, const int seg) {
    const uint32_t  blk = seg / 4;
    const uint32_t  off = 34u*blk + 2u + 8u*(seg % 4);
    const uint32_t * p  = (const uint32_t *) (row + (off & ~3u));
    u32x4_t r;
    r[0] = p[0];
    r[1] = p[1];
    r[2] = blk % 2 == 0 ? p[2] : 0u;
    r[3] = *(const uint16_t *) (row + 34u*blk);
    return r;
}
static __device__ __forceinline__ u32x4_t dequant_seg_q8(const u32x4_t r, const int seg) {
    typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
    const uint32_t sh = (seg / 4) % 2 == 0 ? 2u : 0u;
    // q + 128 as the low byte of an f16 with exponent 10: 1152 + q, exact
    const uint32_t lo = __builtin_amdgcn_alignbyte(r[1], r[0], sh) ^ 0x80808080u;
    const uint32_t hi = __builtin_amdgcn_alignbyte(r[2], r[1], sh) ^ 0x80808080u;
    const uint32_t b[4] = {
        __builtin_amdgcn_perm(0x64646464u, lo, 0x04010400u), __builtin_amdgcn_perm(0x64646464u, lo, 0x04030402u),
        __builtin_amdgcn_perm(0x64646464u, hi, 0x04010400u), __builtin_amdgcn_perm(0x64646464u, hi, 0x04030402u),
    };
    const h2_t d = __builtin_bit_cast(h2_t, r[3] * 0x00010001u);
    const h2_t o = {(_Float16) -1152.0f, (_Float16) -1152.0f};
    u32x4_t out;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        out[e] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2_t, b[e]) + o) * d); // q exact, one rounding of d*q
    }
    return out;
}

// K tile: KPT x 16 B per thread; the lanes of a wave read 512 contiguous bytes of a key row.
// WHIRL: K / V point at the step's page (row k of the step = K + k*nb11, see kv_page_base), keys >= kv_end read
// the clamped row kv_end - 1 (same page) and are staged as zero
template <int D, bool Q8>
static __device__ __forceinline__ void load_K(u32x4_t * regs, const char * K, const uint32_t nb11, const int k0, const int kv_end) {
    using C = cfg<D>;
#pragma unroll
    for (int p = 0; p < C::KPT; ++p) {
        const int idx = p*C::NTHR + threadIdx.x;
        const int key = k0 + idx/C::SEGS;
        if constexpr (Q8) {
            regs[p] = load_seg_q8(K + uint32_t(key)*nb11, idx % C::SEGS);
        } else {
            const u32x4_t x = *(const u32x4_t *) (K + (uint32_t(min(key, kv_end - 1))*nb11 + uint32_t(idx % C::SEGS)*16u));
            regs[p] = key < kv_end ? x : u32x4_t{0u, 0u, 0u, 0u};
        }
    }
}
template <int D, bool Q8>
static __device__ __forceinline__ void store_K(const u32x4_t * regs, _Float16 * sK) {
    using C = cfg<D>;
#pragma unroll
    for (int p = 0; p < C::KPT; ++p) {
        const int idx = p*C::NTHR + threadIdx.x;
        *(u32x4_t *) (sK + (idx/C::SEGS)*C::SQK + (idx % C::SEGS)*8) = Q8 ? dequant_seg_q8(regs[p], idx % C::SEGS) : regs[p];
    }
}

// V tile: thread = (group kg of KPT keys, 8-dim segment seg). Lanes 8j..8j+7 of a wave read 8 consecutive
// segments of one key, i.e. full 128 B cache lines.
template <int D>
static __device__ __forceinline__ void v_thread_coords(int & kg, int & seg) {
    constexpr int WPR = cfg<D>::SEGS / 8; // warps per V row
    const int lane = threadIdx.x % 32;
    const int w    = threadIdx.x / 32;
    seg = lane % 8 + 8*(w % WPR);
    kg  = lane / 8 + 4*(w / WPR);
}
template <int D, bool Q8>
static __device__ __forceinline__ void load_V(u32x4_t * regs, const char * V, const uint32_t nb21, const int k0, const int kv_end) {
    using C = cfg<D>;
    int kg, seg;
    v_thread_coords<D>(kg, seg);
#pragma unroll
    for (int kk = 0; kk < C::KPT; ++kk) {
        const int key = k0 + C::KPT*kg + kk;
        if constexpr (Q8) {
            regs[kk] = load_seg_q8(V + uint32_t(key)*nb21, seg);
        } else {
            const u32x4_t x = *(const u32x4_t *) (V + (uint32_t(min(key, kv_end - 1))*nb21 + uint32_t(seg)*16u));
            regs[kk] = key < kv_end ? x : u32x4_t{0u, 0u, 0u, 0u};
        }
    }
}
// V^T[dim][key] in LDS: within a row the 8-key units are swizzled with the dim's segment, so that the lanes
// of a store (same key group, consecutive segments) spread over more banks.
template <int D>
static __device__ __forceinline__ int vt_unit(const int unit, const int dim) {
    return unit ^ ((dim / 8) % cfg<D>::NUNITS);
}
// registers -> LDS as V^T: the KPT keys x 8 dims block of the thread is transposed with v_perm_b32.
template <int D, bool Q8>
static __device__ __forceinline__ void store_Vt(const u32x4_t * regs_in, _Float16 * sVt) {
    using C = cfg<D>;
    int kg, seg;
    v_thread_coords<D>(kg, seg);
    u32x4_t regs[C::KPT];
#pragma unroll
    for (int kk = 0; kk < C::KPT; ++kk) {
        regs[kk] = Q8 ? dequant_seg_q8(regs_in[kk], seg) : regs_in[kk];
    }
    // regs[kk][q]: dims (2q, 2q+1) of key kk
    const int col = 8*vt_unit<D>(C::KPT*kg/8, 8*seg) + (C::KPT*kg) % 8;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        _Float16 * row_even = sVt + (8*seg + 2*q + 0)*C::SVT + col;
        _Float16 * row_odd  = sVt + (8*seg + 2*q + 1)*C::SVT + col;
        const uint32_t e01 = __builtin_amdgcn_perm(regs[1][q], regs[0][q], 0x05040100u); // dim 2q:   keys 0, 1
        const uint32_t o01 = __builtin_amdgcn_perm(regs[1][q], regs[0][q], 0x07060302u); // dim 2q+1: keys 0, 1
        if constexpr (C::KPT == 4) {
            const uint32_t e23 = __builtin_amdgcn_perm(regs[3][q], regs[2][q], 0x05040100u);
            const uint32_t o23 = __builtin_amdgcn_perm(regs[3][q], regs[2][q], 0x07060302u);
            *(uint2 *) row_even = make_uint2(e01, e23);
            *(uint2 *) row_odd  = make_uint2(o01, o23);
        } else {
            *(uint32_t *) row_even = e01;
            *(uint32_t *) row_odd  = o01;
        }
    }
}
// A operand of V^T for dims [dim0, dim0+16) (dim0 % 16 == 0) and keys [16c, 16c+16): lane row dim0 + l16.
template <int D>
static __device__ __forceinline__ half16_t load_Vt_operand(const _Float16 * sVt, const int dim, const int c) {
    typedef int int8v_t __attribute__((ext_vector_type(8)));
    const _Float16 * row = sVt + dim*cfg<D>::SVT;
    const int4 lo = *(const int4 *) (row + 8*vt_unit<D>(2*c + 0, dim));
    const int4 hi = *(const int4 *) (row + 8*vt_unit<D>(2*c + 1, dim));
    const int8v_t v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    return __builtin_bit_cast(half16_t, v);
}

#endif // FATTN_RDNA35_AVAILABLE

// D = 512: a 16-wave block at up to 256 VGPRs fits once per WGP, so the SIMDs idle while its warps wait at the barriers.
// Capped at 192 VGPRs (8 waves per SIMD) a second block fits (LDS allows two) and fills these gaps (see cfg::TIGHT).
// WHIRL: the kernel body of flash_attn_rdna35 as a device function on the WHIRL prefill ABI (see the file header).
// Grid: x = ceil(n_tok / NCOLS1) query tiles x (heads of this launch / NCOLS2) head groups (head group fastest),
// block cfg<D>::NTHR. h0 = the first head of this launch.
// pool row of key p minus p for p's page (KV_PAGE = 256 rows, a multiple of every step's NKEYS): attn_kv_page.h kv_delta
template <int D, int NCOLS2, bool Q8>
static __device__ __forceinline__ void attn_sx_impl(const float * __restrict__ q, const KvArgs & kv, float * __restrict__ out,
        const int n_head, const int n_kv, const int q_stride, const int * __restrict__ pos, const int n_tok, const float scale,
        const int h0) {
#ifdef FATTN_RDNA35_AVAILABLE
    using C = cfg<D>;
    constexpr int NCOLS  = C::NCOLS;
    constexpr int NCOLS1 = NCOLS / NCOLS2;
    constexpr int NUSED  = NCOLS1 * NCOLS2; // NCOLS2 = 6: 60 of the 64 columns
    constexpr int NCT    = C::NCT;
    constexpr int OT     = C::OT;
    constexpr int KT     = C::KT;
    constexpr int DQ     = C::DQ;
    constexpr int NKEYS  = C::NKEYS;
    constexpr int SVT    = C::SVT;
    static_assert(!Q8, "WHIRL attn_sx: f16 K / V only");
    static_assert(KV_PAGE % NKEYS == 0, "a key step never straddles a KV page");

    __shared__ __attribute__((aligned(16))) char smem[C::SZ_SMEM];
    __shared__ int sRescaleStep; // last step in which some column max moved
    __shared__ int sRedo;        // OWN: last step whose P had to be redone against a new max
    _Float16 * sK     = (_Float16 *) (smem + C::OFF_K);
    _Float16 * sVt    = (_Float16 *) (smem + C::OFF_VT);
    _Float16 * sP     = (_Float16 *) (smem + C::OFF_P);
    float    * sSx    = (float    *) (smem + C::OFF_SX);
    float    * sAlpha = (float    *) (smem + C::OFF_RED);
    float    * sMax   = sAlpha;                              // OWN: [key tile][l16][column tile]
    float    * sL     = sAlpha + (C::OWN ? 2*NCOLS : NCOLS); // OWN: [key tile][column]

    const int lane = threadIdx.x % 32;
    const int w    = threadIdx.x / 32;
    const int l16  = lane % 16;
    const int hh   = lane / 16;
    const int ct   = w / DQ;      // column tile this warp scores
    const int dq   = w % DQ;      // share of D this warp scores: dims [128dq, 128dq+128)
    const int c_s  = 16*ct + l16; // column of this lane in the score phase

    // WHIRL: query tiles and head groups from the grid (one sequence per launch); Q / gate rows q[(token*n_head + head)*q_stride]
    const int ne01      = n_tok;
    const int ntiles    = (n_tok + NCOLS1 - 1) / NCOLS1;
    const int n_hg      = gridDim.x / ntiles;
    const int jt        = blockIdx.x / n_hg;
    const int head0     = h0 + (blockIdx.x % n_hg)*NCOLS2;
    const int gqa_ratio = n_head / n_kv;
    const int z_KV      = head0 / gqa_ratio;
    const int64_t nb01  = int64_t(n_head) * q_stride * 4;  // bytes per token
    const int64_t nb02  = int64_t(q_stride) * 4;           // bytes per head
    const float scale_q = scale * log2e;
    const int tb        = kv.tab0;

    const uint32_t nb11u = uint32_t(n_kv) * D * 2; // bytes per key row (all KV heads); key offsets of one step's page < 4 GiB
    const uint32_t nb21u = nb11u;
    const char * K_h = (const char *) kv.k + int64_t(z_KV) * D * 2;
    const char * V_h = (const char *) kv.v + int64_t(z_KV) * D * 2;
    // the page of key k0: row k of the step = base + k*nb11 (kv_delta: pool row of k0 minus k0)
    auto page_k = [&](const int k0) { return K_h + int64_t(kv_delta(kv, tb, k0)) * nb11u; };
    auto page_v = [&](const int k0) { return V_h + int64_t(kv_delta(kv, tb, k0)) * nb21u; };

    // causal mask from the positions: token t of the sequence sits at pos0 + t and sees keys 0 .. pos0 + t
    const int pos0    = pos[0];
    const int p_first = pos0 + jt*NCOLS1;                     // first position of the tile
    const int kv_end  = pos0 + min((jt + 1)*NCOLS1, n_tok);   // = KV_max of the tile (its last position + 1)
    const int n_steps = (kv_end + NKEYS - 1) / NKEYS;
    const int st_begin = 0;
    const int st_end   = n_steps;

    if (threadIdx.x == 0) {
        sRescaleStep = -1;
        sRedo        = -1;
    }
    // step class (strix mask_classify): 0 = no key of the step is masked for any column of the tile, 2 = mixed
    // (diagonal steps; 1 = all masked cannot occur below kv_end)
    auto step_class = [&](const int st) -> int {
        return st*NKEYS + NKEYS - 1 <= p_first ? 0 : 2;
    };
    auto next_active = [&](int st) -> int {
        while (st < st_end && step_class(st) == 1) {
            ++st;
        }
        return st;
    };
    // the position of column c_s (padding columns of the last tile: the last token, as strix clamps the mask row)
    const int pos_c = pos0 + min(jt*NCOLS1 + c_s/NCOLS2, ne01 - 1);

    // Q of column c_s, dims [128dq, 128dq+128) -> registers as B operands, pre-scaled by scale*log2(e)
    // so that the softmax can use exp2.
    half16_t qb[C::QCH];
    {
        const int  token = jt*NCOLS1 + c_s/NCOLS2;
        const bool valid = token < ne01 && c_s < NUSED;
        const float * qrow = (const float *) ((const char *) q + int64_t(head0 + c_s % NCOLS2)*nb02 + int64_t(valid ? token : 0)*nb01) + (D/DQ)*dq;
#pragma unroll
        for (int k = 0; k < C::QCH; ++k) {
#pragma unroll
            for (int e = 0; e < 16; e += 4) {
                const float4 v = valid ? *(const float4 *) (qrow + 16*k + e) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                qb[k][e + 0] = (_Float16) (v.x*scale_q);
                qb[k][e + 1] = (_Float16) (v.y*scale_q);
                qb[k][e + 2] = (_Float16) (v.z*scale_q);
                qb[k][e + 3] = (_Float16) (v.w*scale_q);
            }
        }
    }
    __syncthreads(); // sRescaleStep visible

    u32x4_t kregs[C::KPT];
    u32x4_t vregs[C::KPT];
    int cur = next_active(st_begin);
    if (cur < st_end) {
        load_K<D, Q8>(kregs, page_k(cur*NKEYS), nb11u, cur*NKEYS, kv_end);
        store_K<D, Q8>(kregs, sK);
    }
    __syncthreads();

    float8_t O[OT][NCT]; // [dim tile][column tile]
#pragma unroll
    for (int i = 0; i < OT; ++i) {
#pragma unroll
        for (int t = 0; t < NCT; ++t) {
            O[i][t] = float8_t{};
        }
    }
    float m = m_init; // running max and row sum of column c_s (identical in all warps of a column tile)
    float l = 0.0f;   // OWN: reference max of column c_s and row sum over key tile dq
    float l_prev = 0.0f;
    float mref[NCT];  // OWN: reference max of the PV columns 16t + l16
#pragma unroll
    for (int t = 0; t < NCT; ++t) {
        mref[t] = m_init;
    }

    // OWN: P = exp2(S - mr) of key tile dq -> B operand layout in sP; returns the column sum of P.
    // own[q] = (key 4q + hh, key 4q + 2 + hh), other[q] from lane ^ 16 = (key 4q + 1 - hh, key 4q + 3 - hh)
    auto store_P = [&](const float8_t & sc, const float mr) -> float {
        float p[8];
        float rowsum = 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            p[r] = __builtin_amdgcn_exp2f(sc[r] - mr);
            rowsum += p[r];
        }
        uint32_t Pb[8];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const half2 h2 = __floats2half2_rn(p[2*q], p[2*q + 1]);
            const uint32_t own   = __builtin_bit_cast(uint32_t, h2);
            const uint32_t other = (uint32_t) swap_halves((int) own);
            const uint32_t lo_src = hh ? other : own;
            const uint32_t hi_src = hh ? own   : other;
            Pb[2*q + 0] = __builtin_amdgcn_perm(hi_src, lo_src, 0x05040100u);
            Pb[2*q + 1] = __builtin_amdgcn_perm(hi_src, lo_src, 0x07060302u);
        }
        if (hh == 0) {
            int4 * dstp = (int4 *) (sP + c_s*SVT + 16*dq);
            dstp[0] = make_int4(Pb[0], Pb[1], Pb[2], Pb[3]);
            dstp[1] = make_int4(Pb[4], Pb[5], Pb[6], Pb[7]);
        }
        return rowsum + swap_halves(rowsum);
    };

    // logit softcap: none in WHIRL's models (qwen35); the call sites stay as in strix
    auto softcap = [&](float8_t & s) {
        (void) s;
    };

    while (cur < st_end) {
        const int  k0       = cur*NKEYS;
        const int  nxt      = next_active(cur + 1);
        const bool has_next = nxt < st_end;
        const bool mixed    = step_class(cur) == 2;

        // V of this step (not at D = 512) and K of the next one: in flight while the scores are computed.
        if constexpr (!C::TIGHT) {
            load_V<D, Q8>(vregs, page_v(k0), nb21u, k0, kv_end);
        }
        if (has_next) {
            load_K<D, Q8>(kregs, page_k(nxt*NKEYS), nb11u, nxt*NKEYS, kv_end);
        }

        // ---- partial S^T = K Q^T over dims [128dq, 128dq+128), all key tiles of the step ----
        float8_t S[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            S[kt] = float8_t{};
        }
        if constexpr (!C::TIGHT) {
            // K operands double-buffered: the loads of chunk k + 1 are in flight during the WMMAs of chunk k
            const _Float16 * ka[KT];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                ka[kt] = sK + (16*(C::OWN ? kt ^ dq : kt) + l16)*C::SQK + (D/DQ)*dq;
            }
            half16_t a[KT];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                a[kt] = *(const half16_t *) ka[kt];
            }
#pragma unroll
            for (int k = 0; k < C::QCH; ++k) {
                half16_t an[KT];
                if (k < C::QCH - 1) {
#pragma unroll
                    for (int kt = 0; kt < KT; ++kt) {
                        an[kt] = *(const half16_t *) (ka[kt] + 16*(k + 1));
                    }
                }
                __builtin_amdgcn_sched_barrier(0); // keep the loads of chunk k + 1 ahead of the WMMAs of chunk k
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    S[kt] = wmma(a[kt], qb[k], S[kt]);
                    if (k < C::QCH - 1) {
                        a[kt] = an[kt];
                    }
                }
            }
        } else {
#pragma unroll
            for (int k = 0; k < C::QCH; ++k) {
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    S[kt] = wmma(*(const half16_t *) (sK + (16*(C::OWN ? kt ^ dq : kt) + l16)*C::SQK + (D/DQ)*dq + 16*k), qb[k], S[kt]);
                }
            }
        }
        // S[kt][r]: key k0 + 16kt + 2r + hh, column c_s (OWN: S[0] is key tile dq, S[1] key tile dq ^ 1)
#pragma unroll
        for (int kt = C::OWN ? 1 : 0; kt < KT && DQ > 1; ++kt) {
            float4 * px = (float4 *) (sSx + ((C::OWN ? w : KT*w + kt)*32 + lane)*8);
            px[0] = make_float4(S[kt][0], S[kt][1], S[kt][2], S[kt][3]);
            px[1] = make_float4(S[kt][4], S[kt][5], S[kt][6], S[kt][7]);
        }
        lds_barrier(); // B1: partial scores visible, all warps are done reading K

        if constexpr (C::OWN) {
            // Own key tile: add the share of the partner warp. The reference max m moves lazily (as in FlashAttention-4),
            // after V^T P and only when a step max is more than T_LAZY above it; P up to 2^T_HI is exact in f16.
            {
                const float4 * px = (const float4 *) (sSx + ((ct*DQ + (dq ^ 1))*32 + lane)*8);
                const float4 x0 = px[0];
                const float4 x1 = px[1];
                S[0][0] += x0.x; S[0][1] += x0.y; S[0][2] += x0.z; S[0][3] += x0.w;
                S[0][4] += x1.x; S[0][5] += x1.y; S[0][6] += x1.z; S[0][7] += x1.w;
            }
            softcap(S[0]);
            if (mixed) {
                // WHIRL: causal mask from the positions; lane key k0 + 16dq + 2r + hh (strix: mask row of the column's token)
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    S[0][r] = k0 + 16*dq + 2*r + hh > pos_c ? -INFINITY : S[0][r];
                }
            }
            float mx = S[0][0];
#pragma unroll
            for (int r = 1; r < 8; ++r) {
                mx = fmaxf(mx, S[0][r]);
            }
            mx = fmaxf(mx, swap_halves(mx));
            if (hh == 0) {
                sMax[(dq*16 + l16)*NCT + ct] = mx;
            }
            if (mx > m + T_HI) {
                sRedo = cur;
            }
            l_prev = l;
            l += store_P(S[0], m);
        } else if (dq == 0) {
            // With one key tile (D = 512) only warp dq = 0 of a column tile uses the softmax (P, rescale factors, row sums);
            // at D <= 128 one warp owns a column tile.
            // ---- full scores: add the shares of the other warps of the column tile ----
#pragma unroll
            for (int j = 1; j < DQ; ++j) {
                const int wp = ct*DQ + (dq + j) % DQ;
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    const float4 * px = (const float4 *) (sSx + ((KT*wp + kt)*32 + lane)*8);
                    const float4 x0 = px[0];
                    const float4 x1 = px[1];
                    S[kt][0] += x0.x; S[kt][1] += x0.y; S[kt][2] += x0.z; S[kt][3] += x0.w;
                    S[kt][4] += x1.x; S[kt][5] += x1.y; S[kt][6] += x1.z; S[kt][7] += x1.w;
                }
            }
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                softcap(S[kt]);
            }
            if (mixed) {
                // WHIRL: causal mask from the positions; lane key k0 + 16kt + 2r + hh, r = 0..7
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        S[kt][r] = k0 + 16*kt + 2*r + hh > pos_c ? -INFINITY : S[kt][r];
                    }
                }
            }

            // ---- softmax over the keys of the step ----
            {
                float mx = S[0][0];
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        mx = fmaxf(mx, S[kt][r]);
                    }
                }
                const float m_new = fmaxf(m, fmaxf(mx, swap_halves(mx)));
                const float alpha = __builtin_amdgcn_exp2f(m - m_new);
                m = m_new;

                float rowsum = 0.0f;
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        S[kt][r] = __builtin_amdgcn_exp2f(S[kt][r] - m_new);
                        rowsum += S[kt][r];
                    }
                }
                l = l*alpha + rowsum + swap_halves(rowsum);
                if (dq == 0 && hh == 0) {
                    sAlpha[c_s] = alpha;
                    if (alpha != 1.0f) {
                        sRescaleStep = cur;
                    }
                }

                // P of the key tiles kt % DQ == dq -> B operand layout.
                // own[q] = (key 4q + hh, key 4q + 2 + hh), other[q] from lane ^ 16 = (key 4q + 1 - hh, key 4q + 3 - hh)
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    if (kt % DQ != dq) {
                        continue;
                    }
                    uint32_t Pb[8];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const half2 h2 = __floats2half2_rn(S[kt][2*q], S[kt][2*q + 1]);
                        const uint32_t own   = __builtin_bit_cast(uint32_t, h2);
                        const uint32_t other = (uint32_t) swap_halves((int) own);
                        const uint32_t lo_src = hh ? other : own; // provides the even key of each pair
                        const uint32_t hi_src = hh ? own   : other;
                        Pb[2*q + 0] = __builtin_amdgcn_perm(hi_src, lo_src, 0x05040100u); // keys 4q,   4q+1
                        Pb[2*q + 1] = __builtin_amdgcn_perm(hi_src, lo_src, 0x07060302u); // keys 4q+2, 4q+3
                    }
                    if (hh == 0) {
                        int4 * dstp = (int4 *) (sP + c_s*SVT + 16*kt);
                        dstp[0] = make_int4(Pb[0], Pb[1], Pb[2], Pb[3]);
                        dstp[1] = make_int4(Pb[4], Pb[5], Pb[6], Pb[7]);
                    }
                }
            }
        }

        // ---- stage V^T of this step and K of the next one (both buffers are free after B1) ----
        if constexpr (C::TIGHT) {
            load_V<D, Q8>(vregs, page_v(k0), nb21u, k0, kv_end);
        }
        store_Vt<D, Q8>(vregs, sVt);
        if (has_next) {
            store_K<D, Q8>(kregs, sK);
        }
        lds_barrier(); // B2: P, V^T, rescale factors and the next K visible

        // ---- O^T = alpha*O^T + V^T P for dims [32w, 32w+32) ----
        float alpha[NCT];
        bool  redo  = false;
        bool  moved = false;
        if constexpr (C::OWN) {
            redo = sRedo == cur;
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                const float smax = fmaxf(sMax[l16*NCT + t], sMax[(16 + l16)*NCT + t]);
                const float mn   = redo || smax > mref[t] + T_LAZY ? fmaxf(mref[t], smax) : mref[t];
                alpha[t] = __builtin_amdgcn_exp2f(mref[t] - mn);
                moved    = moved || mn != mref[t];
                mref[t]  = mn;
            }
            float alpha_s = alpha[0];
            float m_s     = mref[0];
#pragma unroll
            for (int t = 1; t < NCT; ++t) {
                if (ct == t) {
                    alpha_s = alpha[t];
                    m_s     = mref[t];
                }
            }
            m = m_s;
            if (redo) {
                // P of this step left the f16 range against the old max: rescale first and redo P (first steps only, in practice)
#pragma unroll
                for (int t = 0; t < NCT; ++t) {
#pragma unroll
                    for (int i = 0; i < OT; ++i) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            O[i][t][r] *= alpha[t];
                        }
                    }
                }
                l = l_prev*alpha_s + store_P(S[0], m);
                lds_barrier();
            } else {
                l *= alpha_s;
            }
        } else if (sRescaleStep == cur) {
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                const float alpha = sAlpha[16*t + l16];
#pragma unroll
                for (int i = 0; i < OT; ++i) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        O[i][t][r] *= alpha;
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < KT && !C::TIGHT; ++c) {
            half16_t b[NCT];
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                b[t] = *(const half16_t *) (sP + (16*t + l16)*SVT + 16*c);
            }
#pragma unroll
            for (int i = 0; i < OT; ++i) {
                const half16_t a = load_Vt_operand<D>(sVt, 16*OT*w + 16*i + l16, c);
#pragma unroll
                for (int t = 0; t < NCT; ++t) {
                    O[i][t] = wmma(a, b[t], O[i][t]);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < KT && C::TIGHT; ++c) {
            half16_t a[OT];
#pragma unroll
            for (int i = 0; i < OT; ++i) {
                a[i] = load_Vt_operand<D>(sVt, 16*OT*w + 16*i + l16, c);
            }
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                const half16_t b = *(const half16_t *) (sP + (16*t + l16)*SVT + 16*c);
#pragma unroll
                for (int i = 0; i < OT; ++i) {
                    O[i][t] = wmma(a[i], b, O[i][t]);
                }
            }
        }
        if (C::OWN && !redo && __any(moved)) {
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
#pragma unroll
                for (int i = 0; i < OT; ++i) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        O[i][t][r] *= alpha[t];
                    }
                }
            }
        }
        cur = nxt;
    }

    // ---- normalize by the row sums, gate and write (WHIRL: no split-KV, no sinks) ----
    float * sM = sSx; // column maxima (not OWN), nobody reads sSx after the last B2
    if (C::OWN ? hh == 0 : dq == 0 && hh == 0) {
        sL[(C::OWN ? dq*NCOLS : 0) + c_s] = l;
        if (!C::OWN) {
            sM[c_s] = m;
        }
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < NCT; ++t) {
        const int c = 16*t + l16;
        const int token = jt*NCOLS1 + c/NCOLS2;
        if (token >= ne01 || c >= NUSED) {
            continue;
        }
        const int head = head0 + c % NCOLS2;
        const float l_sum = C::OWN ? sL[c] + sL[NCOLS + c] : sL[c];
        const float inv = l_sum > 0.0f ? 1.0f/l_sum : 0.0f;
        const int64_t j_dst = int64_t(token)*n_head + head;
        float * dst_c = out + j_dst*D;
        const float * gate = q + j_dst*q_stride + D; // WHIRL: gated attention output (as attn_prefill_wmma)
#pragma unroll
        for (int i = 0; i < OT; ++i) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = 16*OT*w + 16*i + 2*r + hh;
                dst_c[d] = O[i][t][r]*inv / (1.f + __expf(-gate[d]));
            }
        }
    }
#endif // FATTN_RDNA35_AVAILABLE
}

} // namespace fattn_rdna35

// WHIRL entries: NCOLS2 query heads of one KV head per block column group (strix fattn_rdna35_ncols2: the first of 8 / 4 / 2
// that divides the GQA ratio; ops::attnSxFor), D = 256, f16 K / V. Same arguments as attn_prefill_wmma.
#define WHIRL_ATTN_SX_ENTRY(NAME, NC2)                                                                                 \
    extern "C" __global__ void __launch_bounds__(fattn_rdna35::cfg<256>::NTHR, 1)                                      \
        NAME(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos, int n_tok,   \
             float scale, int h0) {                                                                                  \
        fattn_rdna35::attn_sx_impl<256, NC2, false>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);       \
    }
WHIRL_ATTN_SX_ENTRY(attn_sx2, 2)
WHIRL_ATTN_SX_ENTRY(attn_sx4, 4)
WHIRL_ATTN_SX_ENTRY(attn_sx8, 8)
#undef WHIRL_ATTN_SX_ENTRY

