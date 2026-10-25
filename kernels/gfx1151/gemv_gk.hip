// WHIRL device code (gfx1151 / RDNA 3.5): K-balanced MXFP4 multi-token GEMV (gemvk / ggemvk), feat026-ggemv.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by kernels/gfx1151/whirl_kernels_gfx1151.hip after gemv_wunit.hip / gemv_q6.hip.
//
// The production 2..16-token MXFP4 GEMV on gfx1151 (gemvq_nt*r2 / ggemvq_nt*r2 = gemvw_impl<MXFP4, NT, 1, false>)
// gives wave w (of 8) the superblocks sb = w (mod 8) of one 16-row tile. With nsb = ncols / 256 superblocks:
//   Swift 5120 -> 20 sb -> 3,3,3,3,2,2,2,2 per wave (the block waits for 3, the mean is 2.5: 83%),
//   Ornith 2048 -> 8 sb -> 1 per wave (nothing to overlap), Ornith 512 -> 2 sb -> 6 of 8 waves idle.
// gemvk keeps EVERY float operation of gemvw_impl and only changes which wave does which superblock:
// KW waves split K (wave kw walks sb = kw, kw + KW, ...), 8 / KW waves split rows (one 16-row tile each),
// so a block covers 16 * 8 / KW rows. Superblock sb still belongs to group g = sb mod 8 (slices 4g .. 4g+3);
// a wave holds the accumulators of its 8 / KW groups (g = kw + KW * i, and the k-th sb of the wave is in
// group index i = k mod (8 / KW)), each group's slices still fold their units in increasing sb with the same
// wfold expression, and the LDS tree over G0..G7 is the same ((G0+G1)+(G2+G3))+((G4+G5)+(G6+G7)).
// Hence every (row, token) is bit-identical to gemv1s_impl (1 token) and to gemvw_impl (checkGemvBitwise).
// KW = 8 is gemvw_impl's own mapping (control). PF = 1 issues the next superblock's weight loads (4 x 16 B +
// the 8-byte e8m0 header) before computing the current one (the gfx1201 gw2 ring-of-2 idea, kernels.md §3.1).
// The integer decode is the same (mx_lut4 on the same nibbles, permlanex16 exchange of the other half), and the
// e8m0 pair of unit pp is bits [16 pp, 16 pp + 16) of the header's first 8 bytes = the u16 that wunit_half reads.
//
// Variants (host: kernels::kGemvkKW / kGemvkPF / kGemvkRows, selector WHIRL_GEMV_K; default off):
//   v1 = KW 8, PF 1 (16 rows)   v2 = KW 4, PF 0 (32 rows)   v3 = KW 4, PF 1 (32 rows)
//   v4 = KW 2, PF 0 (64 rows)   v5 = KW 2, PF 1 (64 rows)   v6 = KW 8, PF 0 (16 rows: production mapping,
//   branch-free x; the control)   v7 = KW 8, PF 2   v8 = KW 4, PF 2   v9 = KW 2, PF 2 (PF 2: x first, see mx_step_pf2)

// Scheduling fence between superblocks: without it the compiler hoists the loads of all 8 / KW unrolled
// superblocks of a round (VGPRs 256 + spills for KW 4 / 2, probe 2026-10-26). Only moves instructions, never math.
#ifndef GK_NO_FENCE
#define GK_SCHED_FENCE() __builtin_amdgcn_sched_barrier(0)
#else
#define GK_SCHED_FENCE() ((void)0)
#endif
// per-unit fence inside a superblock (GK_UNIT_FENCE=0 lets the scheduler interleave the 4 units)
#if !defined(GK_UNIT_FENCE_OFF)
#define GK_UNIT_FENCE() __builtin_amdgcn_sched_barrier(0)
#else
#define GK_UNIT_FENCE() ((void)0)
#endif

struct MxSb {          // raw words of one superblock half-row for one lane (half h): 4 units
    uint4 q[4];        // quants of sub-block h of unit pp (16 B at 8 + 32 pp + 16 h)
    uint2 e;           // the 8 header bytes: e8m0 of the 8 sub-blocks
};

__device__ __forceinline__ void mx_fetch(const uint8_t *__restrict__ row, int sb, int h, MxSb &r) {
    const uint8_t *blk = row + (size_t)sb * 136;
    r.e = *(const uint2 *)blk;
#pragma unroll
    for (int pp = 0; pp < 4; ++pp) r.q[pp] = *(const uint4 *)(blk + 8 + 32 * pp + 16 * h);
}

// Unlike gemvw_impl the lanes of token slots n >= NT load token 0's x instead of zeros (no divergent branch,
// one basic block per superblock so the scheduling fences hold): WMMA row m of C only depends on A row m, so the
// C rows of slots >= NT are never read for a stored (row, token) - the odd-NT spare fold is never stored either.
// The activation side of one superblock: x of the lane's token slot (4 units x 64 B) and the packed scale words.
template <int NT> struct MxX {
    int4v x[4][4];
    uint2 p[4][(NT + 1) / 2];
};
template <int NT>
__device__ __forceinline__ void mx_xload(int sb, int h, const int8_t *__restrict__ xt, const uint32_t *__restrict__ xp,
                                         int nb32, MxX<NT> &o) {
    constexpr int NTV = (NT + 1) / 2;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int base = sb * 256 + 64 * q, xi = base >> 5;
#pragma unroll
        for (int c = 0; c < 4; ++c) o.x[q][c] = *(const int4v *)(xt + base + 16 * c);
#pragma unroll
        for (int v = 0; v < NTV; ++v) {
            const int t = min(2 * v + h, NT - 1);
            o.p[q][v] = *(const uint2 *)(xp + (size_t)t * nb32 + xi);
        }
    }
}
// One superblock (4 units) of one 16-row tile for NT tokens into acc[4][NTV]: the body of gemvw_impl's
// q loop (SplitDec<MXFP4> path), operation for operation.
template <int NT>
__device__ __forceinline__ void mx_sb_x(const MxSb &r, const MxX<NT> &xx, int h, float (&acc)[4][(NT + 1) / 2]) {
    constexpr int NTV = (NT + 1) / 2;
    const int8v zero = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t ee = (q < 2 ? r.e.x >> (16 * q) : r.e.y >> (16 * (q - 2))) & 0xFFFF;
        const float s0 = e8m0_half(ee & 0xFF), s1 = e8m0_half(ee >> 8);
        const uint32_t a[4] = {r.q[q].x, r.q[q].y, r.q[q].z, r.q[q].w};
        int own[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            own[k] = mx_lut4(a[k] & 0x0F0F0F0F);
            own[4 + k] = mx_lut4((a[k] >> 4) & 0x0F0F0F0F);
        }
        int w[16];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int oth = xhalf(own[k]);
            w[k] = h ? oth : own[k];
            w[8 + k] = h ? own[k] : oth;
        }
        const int4v b0 = {w[0], w[1], w[2], w[3]}, b1 = {w[4], w[5], w[6], w[7]};
        const int4v b2 = {w[8], w[9], w[10], w[11]}, b3 = {w[12], w[13], w[14], w[15]};
        int8v c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, xx.x[q][0], true, b0, zero, false);
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, xx.x[q][1], true, b1, c0, false);
        int8v c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, xx.x[q][2], true, b2, zero, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, xx.x[q][3], true, b3, c1, false);
#pragma unroll
        for (int v = 0; v < NTV; ++v) {
            const uint2 p = xx.p[q][v];
            acc[q][v] = wfold(wfold(acc[q][v], p.x, s0, 0.f, c0[v]), p.y, s1, 0.f, c1[v]);
        }
        GK_UNIT_FENCE();
    }
}
// PF 0 / 1: the activations are loaded per unit, right before their WMMAs (as gemvw_impl; loading all four
// units' x first costs ~70 VGPRs, probe 2026-10-26: nt4 KW4 99 -> 170).
template <int NT>
__device__ __forceinline__ void mx_sb(const MxSb &r, int sb, int h, const int8_t *__restrict__ xt,
                                      const uint32_t *__restrict__ xp, int nb32, float (&acc)[4][(NT + 1) / 2]) {
    constexpr int NTV = (NT + 1) / 2;
    const int8v zero = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int base = sb * 256 + 64 * q, xi = base >> 5;
        const int4v x0 = *(const int4v *)(xt + base);
        const int4v x1 = *(const int4v *)(xt + base + 16);
        const int4v x2 = *(const int4v *)(xt + base + 32);
        const int4v x3 = *(const int4v *)(xt + base + 48);
        const uint32_t ee = (q < 2 ? r.e.x >> (16 * q) : r.e.y >> (16 * (q - 2))) & 0xFFFF;
        const float s0 = e8m0_half(ee & 0xFF), s1 = e8m0_half(ee >> 8);
        const uint32_t a[4] = {r.q[q].x, r.q[q].y, r.q[q].z, r.q[q].w};
        int own[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            own[k] = mx_lut4(a[k] & 0x0F0F0F0F);
            own[4 + k] = mx_lut4((a[k] >> 4) & 0x0F0F0F0F);
        }
        int w[16];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int oth = xhalf(own[k]);
            w[k] = h ? oth : own[k];
            w[8 + k] = h ? own[k] : oth;
        }
        const int4v b0 = {w[0], w[1], w[2], w[3]}, b1 = {w[4], w[5], w[6], w[7]};
        const int4v b2 = {w[8], w[9], w[10], w[11]}, b3 = {w[12], w[13], w[14], w[15]};
        int8v c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, x0, true, b0, zero, false);
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, x1, true, b1, c0, false);
        int8v c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, x2, true, b2, zero, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, x3, true, b3, c1, false);
#pragma unroll
        for (int v = 0; v < NTV; ++v) {
            const int t = min(2 * v + h, NT - 1);
            const uint2 p = *(const uint2 *)(xp + (size_t)t * nb32 + xi);
            acc[q][v] = wfold(wfold(acc[q][v], p.x, s0, 0.f, c0[v]), p.y, s1, 0.f, c1[v]);
        }
        GK_UNIT_FENCE();
    }
}
// PF = 2: activation loads of the current superblock first, then the next superblock's weight prefetch, then
// compute: s_waitcnt vmcnt is in order, so with PF = 1 (prefetch first) waiting for x also waits for the prefetch.
template <int NT>
__device__ __forceinline__ void mx_step_pf2(MxSb &cur, const uint8_t *__restrict__ wr, int sb, int sb_next, int h,
                                            const int8_t *__restrict__ xt, const uint32_t *__restrict__ xp, int nb32,
                                            float (&acc)[4][(NT + 1) / 2]) {
    MxX<NT> xx;
    mx_xload<NT>(sb, h, xt, xp, nb32, xx);
    const MxSb r = cur;
    mx_fetch(wr, sb_next, h, cur);
    GK_SCHED_FENCE();
    mx_sb_x<NT>(r, xx, h, acc);
    GK_SCHED_FENCE();
}

template <int NT, int KW, int PF>
__device__ __forceinline__ void gemvk_mx_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                                              const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows,
                                              int accumulate, const int *__restrict__ skip, int bid = -1) {
    static_assert(KW == 2 || KW == 4 || KW == 8, "gemvk: KW must divide 8");
    constexpr int NG = 8 / KW;          // groups per wave = 16-row tiles per block
    constexpr int ROWS = 16 * NG;
    constexpr int NTV = (NT + 1) / 2;
    __shared__ float red[8][NT][ROWS];
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP, h = lane >> 4, n = lane & 15;
    const int wv = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WARP));
    const int kw = wv % KW, rt = wv / KW;
    const int r0 = (bid < 0 ? (int)blockIdx.x : bid) * ROWS;
    const uint8_t *wr = W + (size_t)min(r0 + rt * 16 + n, nrows - 1) * row_bytes;
    const int8_t *xt = xq + (size_t)(n < NT ? n : 0) * ncols;
    const uint32_t *xp = (const uint32_t *)xd;
    const int nb32 = ncols / 32;
    const int nsb = ncols / 256;
    const int cnt = nsb > kw ? (nsb - kw + KW - 1) / KW : 0;  // superblocks of this wave: kw + KW * k, k < cnt
    float acc[NG][4][NTV];
#pragma unroll
    for (int i = 0; i < NG; ++i)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int v = 0; v < NTV; ++v) acc[i][q][v] = 0.f;
    // full rounds of NG superblocks (no guards inside), then the tail (< NG superblocks; k is a multiple of NG
    // there, so the group index stays i)
    int k = 0;
    if constexpr (PF == 2) {
        MxSb cur;
        if (cnt > 0) mx_fetch(wr, kw, h, cur);
        for (; k + NG <= cnt; k += NG) {
#pragma unroll
            for (int i = 0; i < NG; ++i)
                mx_step_pf2<NT>(cur, wr, kw + KW * (k + i), min(kw + KW * (k + i + 1), nsb - 1), h, xt, xp, nb32, acc[i]);
        }
#pragma unroll
        for (int i = 0; i < NG - 1; ++i)
            if (k + i < cnt)
                mx_step_pf2<NT>(cur, wr, kw + KW * (k + i), min(kw + KW * (k + i + 1), nsb - 1), h, xt, xp, nb32, acc[i]);
    } else if constexpr (PF == 1) {
        MxSb cur;
        if (cnt > 0) mx_fetch(wr, kw, h, cur);
        for (; k + NG <= cnt; k += NG) {
#pragma unroll
            for (int i = 0; i < NG; ++i) {
                const MxSb r = cur;
                // next superblock of this wave, clamped (the last prefetch re-reads a line already in cache)
                mx_fetch(wr, min(kw + KW * (k + i + 1), nsb - 1), h, cur);
                GK_SCHED_FENCE();
                mx_sb<NT>(r, kw + KW * (k + i), h, xt, xp, nb32, acc[i]);
                GK_SCHED_FENCE();
            }
        }
#pragma unroll
        for (int i = 0; i < NG - 1; ++i) {
            if (k + i < cnt) {
                const MxSb r = cur;
                mx_fetch(wr, min(kw + KW * (k + i + 1), nsb - 1), h, cur);
                GK_SCHED_FENCE();
                mx_sb<NT>(r, kw + KW * (k + i), h, xt, xp, nb32, acc[i]);
                GK_SCHED_FENCE();
            }
        }
    } else {
        for (; k + NG <= cnt; k += NG) {
#pragma unroll
            for (int i = 0; i < NG; ++i) {
                MxSb r;
                mx_fetch(wr, kw + KW * (k + i), h, r);
                mx_sb<NT>(r, kw + KW * (k + i), h, xt, xp, nb32, acc[i]);
                GK_SCHED_FENCE();
            }
        }
#pragma unroll
        for (int i = 0; i < NG - 1; ++i) {
            if (k + i < cnt) {
                MxSb r;
                mx_fetch(wr, kw + KW * (k + i), h, r);
                mx_sb<NT>(r, kw + KW * (k + i), h, xt, xp, nb32, acc[i]);
                GK_SCHED_FENCE();
            }
        }
    }
    // group of acc[i] is g = (kw + KW * i): the k-th superblock kw + KW k has (kw + KW k) mod 8 = kw + KW (k mod NG)
#pragma unroll
    for (int i = 0; i < NG; ++i)
#pragma unroll
        for (int v = 0; v < NTV; ++v) {
            const int t = 2 * v + h;
            if (t < NT) red[kw + KW * i][t][rt * 16 + n] = (acc[i][0][v] + acc[i][1][v]) + (acc[i][2][v] + acc[i][3][v]);
        }
    __syncthreads();
    for (int i = threadIdx.x; i < NT * ROWS; i += blockDim.x) {
        const int t = i / ROWS, nn = i % ROWS, row = r0 + nn;
        const float s = ((red[0][t][nn] + red[1][t][nn]) + (red[2][t][nn] + red[3][t][nn])) +
                        ((red[4][t][nn] + red[5][t][nn]) + (red[6][t][nn] + red[7][t][nn]));
        if (row < nrows) {
            float *dst = y + (size_t)t * nrows + row;
            *dst = accumulate ? *dst + s : s;
        }
    }
}

// plain entry (one matrix) and grouped twin (GvArgs, up to 3 matrices sharing the input; kernels.md §4: separate
// entries, the impl force-inlined, the weight pointer only through gv_sel)
#define GEMVK_ENTRY(NT, V, KW, PF)                                                                         \
    extern "C" __global__ void __launch_bounds__(256)                                                      \
        gemvk_nt##NT##v##V##_mxfp4(const uint8_t *W, uint64_t row_bytes, const int8_t *xq, const float *xd, \
                                   float *y, int ncols, int nrows, int accumulate, const int *skip) {     \
        gemvk_mx_impl<NT, KW, PF>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip);                \
    }                                                                                                      \
    extern "C" __global__ void __launch_bounds__(256) ggemvk_nt##NT##v##V##_mxfp4(GV_PARAMS) {             \
        const GvSel gs = gv_sel(g);                                                                        \
        gemvk_mx_impl<NT, KW, PF>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b);     \
    }
#define GEMVK_VARIANTS(NT)   \
    GEMVK_ENTRY(NT, 1, 8, 1) \
    GEMVK_ENTRY(NT, 2, 4, 0) \
    GEMVK_ENTRY(NT, 3, 4, 1) \
    GEMVK_ENTRY(NT, 4, 2, 0) \
    GEMVK_ENTRY(NT, 5, 2, 1) \
    GEMVK_ENTRY(NT, 6, 8, 0) \
    GEMVK_ENTRY(NT, 7, 8, 2) \
    GEMVK_ENTRY(NT, 8, 4, 2) \
    GEMVK_ENTRY(NT, 9, 2, 2)
// verify rows that occur in balance decode on the 8060S (MTP drafts 1..7 -> 2..8 rows); host table is 2..16
GEMVK_VARIANTS(2)
GEMVK_VARIANTS(3)
GEMVK_VARIANTS(4)
GEMVK_VARIANTS(5)
GEMVK_VARIANTS(6)
GEMVK_VARIANTS(7)
GEMVK_VARIANTS(8)
// marker: the code object carries gemvk (host: kernels::Caps / L.opt)
extern "C" __global__ void gemvk_abi_mxfp4() {}
