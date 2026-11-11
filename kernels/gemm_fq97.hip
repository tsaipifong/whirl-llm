// WHIRL device code (gfx1201 / RDNA 4): fused-dequant MXFP4 prefill GEMM gemmfq97_<v>_mxfp4 (v0252-fdq97, PLAN V1-89
// phase 1: the gfx1201 counterpart of the 8060S gemm_fq.hip, KB v0251-a3port). Phase 1 is kernel level only: no host
// path selects these kernels; whirl-kernel-test runs them (kt gemm bitwise rows, kt --bench fdq97).
// SPDX-License-Identifier: Apache-2.0
//
// A fragment of the gfx1201 translation unit: kernels/whirl_kernels.hip includes it LAST in the gfx1201 block, so no
// gfx1201 kernel before it moves in the TU order. It uses the set's helpers: half8 / float8 / half2_t, WARP,
// g8t_lds_barrier (gemm_fp8t.hip), e8m0_half (shared/common_part.h), mx_lut4 + bytes8_to_h8
// (shared/gemm_prefill_wmma_entry.h), and the integer decode helpers of kernels/gfx1151/mx_inject.h (host + device and
// arch neutral; the CPU unit test testMxInject runs the same functions). Family kernels.gemm (families.json glob
// kernels/gemm_*.hip).
//
// BITWISE == dequant_f16_mxfp4 + gemmh_f16 on gfx1201 (the precise prefill pair of an MXFP4 dequant choice at >= 512
// rows, ops::chooseGemmF16); the full argument is section 1 of progress/v0252-fdq97.HANDOFF.md:
//  (a) tile 128 x 256 x 32, 8 waves 4 x 2 (TM 2 x TN 8), the LDS fragment layout, the k loop (32-k slabs, two 16-k WMMA
//      steps each, ascending), v_wmma_f32_16x16x16_f16 with a = weights / b = tokens and an f32 accumulator from 0, the
//      TG-4 raster and the f32 epilogue (accumulate = old + acc, the partial-row element path) are gemmh_impl's
//      (kernels/shared/gemmh_impl.h, the gfx12 branch), so every output element is the same WMMA chain over the same
//      operands in the same order;
//  (b) the A operand: lane slot (row r, group gg) of slab k0 holds dec_group_h<QT_MXFP4>(row r, k0 / 8 + gg), the 8 f16
//      values dequant_f16_mxfp4 writes and gemmh_f16 loads for that (row, k):
//        DEC 0  the same math on registers (mx_lut4 + bytes8_to_h8 with the same arguments: the calls dec_group_h makes),
//        DEC 1  exponent injection + one v_pk_fma_f16 by 2^(e - 113) with a +0 addend for block exponents e in
//               [104, 128] (bit-identical to the LUT math there: mx_inject.h, unit test testMxInject), DEC 0 otherwise;
//  (c) only WHEN the f16 weights are made differs: the raw MXFP4 bytes (8 qs bytes + the E8M0 byte per lane per slab) are
//      loaded before the previous slab's WMMAs and decoded after them, before the LDS store (strix-llama.cpp mmb
//      load_regs / store_lds, the 8060S gemm_fq pattern), instead of a separate pass that writes and re-reads 2 bytes per
//      weight. gemmhq_mxfp4 (gemm_fp8t.hip) is the same fusion with the decode AT the fetch (kept as a kt reference).
// Variants (kt only in phase 1; the kt job's KEEP rule judges s1 alone, the others are reported):
//   gemmfq97_s1_mxfp4  one LDS buffer, two barriers per slab (gemmh_impl's schedule), DEC 1   (PRE-REGISTERED candidate)
//   gemmfq97_s0_mxfp4  one LDS buffer, DEC 0                                                 (decode-cost control)
//   gemmfq97_d1_mxfp4  two LDS buffers, one barrier per slab (gemm8t_impl's schedule), DEC 1 (barrier / overlap probe)
// Same ABI and grid as gemmhq_mxfp4: (W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate), grid {ceil(n / 256),
// ceil(nrows / 128)}, 256 threads, f32 output. Requires WHIRL's MXFP4 repack (136-byte superblocks of 256 values,
// row_bytes == ncols / 256 * 136) and ncols % 256 == 0: the host must check it (phase 2).
// -DWHIRL_FQ97_MUTANTS=1 (a kt-only object built by the item's CPU chain, never the product) adds the RED controls
// gemmfq97_m<k>_mxfp4 = s1 with mutation k (whirl-kernel-test WHIRL_KT_FQ97_MUT=<k> runs them in place of s1).

#include "gfx1151/mx_inject.h"

// 8 values of one LUT group: dec_group_h<QT_MXFP4>'s math on registers (same calls -> same bits)
__device__ __forceinline__ half8 fq97_lut8(uint32_t n0, uint32_t n1, _Float16 dl) {
    const uint32_t u0 = (uint32_t)mx_lut4(n0) ^ 0x80808080u, u1 = (uint32_t)mx_lut4(n1) ^ 0x80808080u;
    return bytes8_to_h8(u0, u1, (_Float16)128.f, dl, (_Float16)0.f);
}
// 8 values from two dwords of injected high bytes (t0: values 0-3, t1: values 4-7) times S (+0)
__device__ __forceinline__ half8 fq97_inj8(uint32_t t0, uint32_t t1, half2_t s2) {
    const uint32_t w[4] = {__builtin_amdgcn_perm(0u, t0, kMxiSpread0), __builtin_amdgcn_perm(0u, t0, kMxiSpread1),
                           __builtin_amdgcn_perm(0u, t1, kMxiSpread0), __builtin_amdgcn_perm(0u, t1, kMxiSpread1)};
    const half2_t z2 = {(_Float16)0.f, (_Float16)0.f};
    half8 out;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const half2_t v = __builtin_elementwise_fma(__builtin_bit_cast(half2_t, w[i]), s2, z2);
        out[2 * i] = v[0];
        out[2 * i + 1] = v[1];
    }
    return out;
}
// one lane's 16 weights of a slab: q = 8 qs bytes [8h, 8h + 8) of the slab's 32-value sub-block, e = its E8M0 byte;
// lo = group h (the low nibbles: values 8h .. 8h + 7 of the sub-block), hi = group h + 2 (the high nibbles: 16 + 8h ..)
template <int DEC, int MUT>
__device__ __forceinline__ void fq97_decode(uint2 q, uint32_t e, half8 &lo, half8 &hi) {
    // MUT 2 / 3: the injected range one exponent too wide at the bottom / top (RED controls of the E8M0 edge rows)
    const bool inj = DEC == 1 && (mxi_fast_ok(e) || (MUT == 2 && e == 103u) || (MUT == 3 && e == 129u));
    if (inj) {
        const uint32_t sb = mxi_scale_bits(e);
        const half2_t s2 = __builtin_bit_cast(half2_t, sb | (sb << 16));
        lo = fq97_inj8(mxi_hib_lo(q.x), mxi_hib_lo(q.y), s2);
        hi = fq97_inj8(mxi_hib_hi(q.x), mxi_hib_hi(q.y), s2);
    } else {
        const _Float16 dl = (_Float16)e8m0_half(e);
        lo = fq97_lut8(q.x & 0x0F0F0F0Fu, q.y & 0x0F0F0F0Fu, dl);
        hi = fq97_lut8((q.x >> 4) & 0x0F0F0F0Fu, (q.y >> 4) & 0x0F0F0F0Fu, dl);
    }
    if (MUT == 1) {  // RED control: the low / high nibble groups swapped
        const half8 t = lo;
        lo = hi;
        hi = t;
    }
}

// MUT (RED controls, -DWHIRL_FQ97_MUTANTS objects only): 1 nibble groups swapped, 2 / 3 injected range +1 at 103 / 129,
// 4 the two WMMA k steps of a slab in reverse order, 5 accumulate ignored, 6 the E8M0 byte of the neighbouring sub-block
template <int DEC, bool DBUF, int MUT = 0>
__device__ void gemmfq97_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                              float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int BM = 128, BN = 256, BK = 32, WAVES_M = 4, NTH = 256, TG = 4;  // = gemmh_impl (shared/gemmh_impl.h:1)
    constexpr int WAVES_N = NTH / 32 / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int KSPS = BK / 16, GPR = BK / 8;
    constexpr int B_PER = BN * GPR / NTH;
    constexpr int ABUF = BM * BK, SBUF = ABUF + BN * BK;
    static_assert(2 * BM == NTH && BK == 32, "one lane per (row, 8-byte half of the slab's 16 qs bytes)");
    __shared__ __attribute__((aligned(16))) _Float16 S[(DBUF ? 2 : 1) * SBUF];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    int bx, by;
    {  // = gemmh_impl's raster (TG token blocks x all row blocks per group)
        const int gx = gridDim.x, gy = gridDim.y, bid = blockIdx.y * gx + blockIdx.x;
        const int per = TG * gy, g = bid / per, rem = bid % per;
        const int tg = min(TG, gx - g * TG);
        bx = g * TG + rem % tg;
        by = rem / tg;
    }
    const int m0 = by * BM, t0 = bx * BN;
    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    // A: lane x owns row ar = x % BM (clamped: never stored) and qs bytes [8 ah, 8 ah + 8) of the slab's sub-block,
    // ah = x / BM: gemmh_impl's slots i = x (row ar, group ah) and i = x + NTH (row ar, group ah + 2)
    const int ar = threadIdx.x % BM, ah = threadIdx.x / BM;
    int agr = m0 + ar;
    agr = agr < nrows ? agr : nrows - 1;
    const uint8_t *arow = W + (size_t)agr * row_bytes;
    const int slo = (((ar >> 4) * KSPS + 0) * 32 + (ar & 15) + 16 * ah) * 8;  // group ah     (gg >> 1 = 0, gg & 1 = ah)
    const int shi = (((ar >> 4) * KSPS + 1) * 32 + (ar & 15) + 16 * ah) * 8;  // group ah + 2 (gg >> 1 = 1, gg & 1 = ah)
    uint2 qr;
    uint32_t qe;
    half8 rb[B_PER];
    auto fetch = [&](int k0) {
        const uint8_t *blk = arow + (size_t)(k0 >> 8) * 136;
        const int ib = (k0 & 255) >> 5;
        qr = *(const uint2 *)(blk + 8 + 16 * ib + 8 * ah);
        qe = blk[MUT == 6 ? (ib ^ 1) : ib];
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {  // = gemmh_impl
            const int i = threadIdx.x + e * NTH;
            const int t = i % BN, gg = i / BN;
            int gt = t0 + t;
            gt = gt < n_tok ? gt : n_tok - 1;  // clamped: never stored
            rb[e] = *(const half8 *)(X + (size_t)gt * ncols + k0 + gg * 8);
        }
    };
    auto store = [&](int buf, const half8 &alo, const half8 &ahi) {
        _Float16 *sa = S + buf * SBUF;
        _Float16 *sb = sa + ABUF;
        *(half8 *)(sa + slo) = alo;
        *(half8 *)(sa + shi) = ahi;
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {  // = gemmh_impl
            const int i = threadIdx.x + e * NTH;
            const int t = i % BN, gg = i / BN;
            *(half8 *)(sb + (((t >> 4) * KSPS + (gg >> 1)) * 32 + (t & 15) + 16 * (gg & 1)) * 8) = rb[e];
        }
    };
    auto compute = [&](int buf) {  // = gemmh_impl (gfx12 branch)
        const _Float16 *sa = S + buf * SBUF;
        const _Float16 *sb = sa + ABUF;
#pragma unroll
        for (int st0 = 0; st0 < KSPS; ++st0) {
            const int st = MUT == 4 ? KSPS - 1 - st0 : st0;
            half8 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = *(const half8 *)(sa + (((wm * TM + i) * KSPS + st) * 32 + lane) * 8);
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = *(const half8 *)(sb + (((wn * TN + j) * KSPS + st) * 32 + lane) * 8);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
        }
    };
    const int nslabs = ncols / BK;
    fetch(0);
    for (int s = 0; s < nslabs; ++s) {
        const int buf = DBUF ? (s & 1) : 0;
        half8 alo, ahi;
        fq97_decode<DEC, MUT>(qr, qe, alo, ahi);  // slab s (its bytes arrived during slab s - 1's WMMAs)
        // one buffer: every wave must be done reading slab s - 1 before the store; two buffers: the store of slab s
        // overwrites the buffer of slab s - 2, which every wave finished before the previous barrier (gemm8t_impl)
        if (!DBUF && s > 0) g8t_lds_barrier();
        store(buf, alo, ahi);
        g8t_lds_barrier();
        if (s + 1 < nslabs) fetch((s + 1) * BK);
        compute(buf);
    }
    const bool acm = MUT == 5 ? false : accumulate != 0;
#pragma unroll
    for (int j = 0; j < TN; ++j) {  // = gemmh_impl's f32 epilogue (gfx12)
        const int t = t0 + (wn * TN + j) * 16 + lane % 16;
        if (t >= n_tok) continue;
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int row0 = m0 + (wm * TM + i) * 16 + 8 * (lane / 16);
            const bool full = (nrows & 7) == 0 && row0 + 8 <= nrows;
            float *dst = Y + (size_t)t * nrows + row0;
            if (full) {
                float4 y0 = {acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]};
                float4 y1 = {acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]};
                if (acm) {
                    const float4 o0 = *(const float4 *)dst, o1 = *(const float4 *)(dst + 4);
                    y0.x = o0.x + y0.x; y0.y = o0.y + y0.y; y0.z = o0.z + y0.z; y0.w = o0.w + y0.w;
                    y1.x = o1.x + y1.x; y1.y = o1.y + y1.y; y1.z = o1.z + y1.z; y1.w = o1.w + y1.w;
                }
                *(float4 *)dst = y0;
                *(float4 *)(dst + 4) = y1;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v)
                    if (row0 + v < nrows) dst[v] = acm ? dst[v] + acc[i][j][v] : acc[i][j][v];
            }
        }
    }
}

#define GEMMFQ97_ENTRY(NAME, DEC, DBUF, MUT)                                                                            \
    extern "C" __global__ void __launch_bounds__(256) NAME(const uint8_t *W, uint64_t row_bytes, const _Float16 *X,     \
                                                           float *Y, int ncols, int nrows, int n_tok, int accumulate) { \
        gemmfq97_impl<DEC, DBUF, MUT>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                             \
    }
GEMMFQ97_ENTRY(gemmfq97_s1_mxfp4, 1, false, 0)
GEMMFQ97_ENTRY(gemmfq97_s0_mxfp4, 0, false, 0)
GEMMFQ97_ENTRY(gemmfq97_d1_mxfp4, 1, true, 0)
#if defined(WHIRL_FQ97_MUTANTS)
GEMMFQ97_ENTRY(gemmfq97_m1_mxfp4, 1, false, 1)
GEMMFQ97_ENTRY(gemmfq97_m2_mxfp4, 1, false, 2)
GEMMFQ97_ENTRY(gemmfq97_m3_mxfp4, 1, false, 3)
GEMMFQ97_ENTRY(gemmfq97_m4_mxfp4, 1, false, 4)
GEMMFQ97_ENTRY(gemmfq97_m5_mxfp4, 1, false, 5)
GEMMFQ97_ENTRY(gemmfq97_m6_mxfp4, 1, false, 6)
#endif
#undef GEMMFQ97_ENTRY
