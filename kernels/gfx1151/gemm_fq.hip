// gfx1151 (Radeon 8060S) fused-dequant MXFP4 prefill GEMM gemmfq_<v>_mxfp4 (feat026-fdq 90d5cf7 / feat026-fdq-p25
// c15f51f, ported to 0.2.5 by v0251-a3port; screening: progress\v0251-judge1.md sec 1, KB feat026:fdq).
// SPDX-License-Identifier: Apache-2.0
//
// A fragment of the gfx1151 translation unit (kernels/gfx1151/whirl_kernels_gfx1151.hip includes it LAST, so no
// existing kernel moves): it uses the set's helpers (half8 / half16 / half2_t, mx_lut4, bytes8_to_h8, e8m0_half,
// ld_frag16, WMMA16, CROW, WARP). Family kernels.gemm (kernels/families.json glob kernels/gfx1151/gemm_*.hip).
//
// Bitwise == dequant_f16_mxfp4 + gemm_c2_f16 (and every other gemm_c<i>_f16): the LDS tile holds the same f16
// weights (dec_group_h<QT_MXFP4> values; see mx_inject.h for the injected decode), the WMMAs run in the same k order
// with the same operands (a = weights, b = tokens). Differences from gemm3_impl, all for speed:
//   * the registers carry the RAW MXFP4 bytes of the next K chunk across the WMMA loop (2 x 8 bytes + the E8M0 byte
//     per 32-value sub-block instead of 4 half8) and decode them at the top of the next iteration, before the
//     barrier (strix-llama.cpp mmb load_regs / store_lds: the load latency hides behind the WMMAs);
//   * one lane per 32-value sub-block: no two lanes load the same bytes (dec_group_h's groups g and g + 2 do);
//   * rows / tokens past the end are clamped, not branched (their outputs are never stored, and each output reads
//     only its own row and token): no divergent guard that drains the loads;
//   * exponent-injected decode + one v_pk_fma_f16 per 2 values (~2 VALU / value), the LUT math (mx_lut4 +
//     bytes8_to_h8, = dec_group_h) for block exponents outside [104, 128].
// Geometry: c2 = 256 x 128 WAVES_M 4 (the kept gemm_c2_f16 layout, 2 sub-blocks per lane); t2 = 128 x 256 WAVES_M 2
// (the hipBLASLt gfx1151 W4A16 orientation: the quantized side is the short one, 1 sub-block per lane = half the
// decode VALU per WMMA). Both are 4 x 4 WMMA tiles per wave, 256 threads, BK 64, TOK_X grid (x = token blocks).
// Requires ncols % 256 == 0 (whole MXFP4 superblocks) and row_bytes % 8 == 0 (136-byte units).
// Modes: bitwise -> the same in precise, balance and fast; the gate is the bitwise kernel test (tests/kernels
// test_gemm.cpp) and golden / seqtest identical, no KL needed.
// The source branch's LUT / decode-at-fetch probes (c2l, t2l, c2f) were screening-only and are not ported.

#include "mx_inject.h"
#include "fq_raster.h"  // v0252-8sbal: grouped-raster block order (GRP > 0 prototypes)

// v0252-8sbal: the group width of the PRODUCT entry points (gemmfq_c2 / _t2). 0 = the shipped plain order; a screen
// object is compiled with -DWHIRL_FQ_GRP_T2=<g> / -DWHIRL_FQ_GRP_C2=<g> (WHIRL_CODE_OBJECT A/B, bitwise-equal output).
#ifndef WHIRL_FQ_GRP_T2
#define WHIRL_FQ_GRP_T2 0
#endif
#ifndef WHIRL_FQ_GRP_C2
#define WHIRL_FQ_GRP_C2 0
#endif

// 8 values of one LUT group: dec_group_h<QT_MXFP4>'s math on registers (same calls -> same bits)
__device__ __forceinline__ half8 fq_lut8(uint32_t n0, uint32_t n1, _Float16 dl) {
    const uint32_t u0 = (uint32_t)mx_lut4(n0) ^ 0x80808080u, u1 = (uint32_t)mx_lut4(n1) ^ 0x80808080u;
    return bytes8_to_h8(u0, u1, (_Float16)128.f, dl, (_Float16)0.f);
}
// 8 values from two dwords of injected high bytes (t0: values 0-3, t1: values 4-7) times S (+0)
__device__ __forceinline__ half8 fq_inj8(uint32_t t0, uint32_t t1, half2_t s2) {
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
// one 32-value sub-block (qa = bytes 0-7, qb = bytes 8-15 of its qs, e = its E8M0 byte) -> 4 half8 in k order:
// [0, 8) low nibbles of qa, [8, 16) low of qb, [16, 24) high of qa, [24, 32) high of qb (= dec_group_h's groups)
__device__ __forceinline__ void fq_decode_sb(uint2 qa, uint2 qb, uint32_t e, half8 *o) {
    if (mxi_fast_ok(e)) {
        const uint32_t sb = mxi_scale_bits(e);
        const half2_t s2 = __builtin_bit_cast(half2_t, sb | (sb << 16));
        o[0] = fq_inj8(mxi_hib_lo(qa.x), mxi_hib_lo(qa.y), s2);
        o[1] = fq_inj8(mxi_hib_lo(qb.x), mxi_hib_lo(qb.y), s2);
        o[2] = fq_inj8(mxi_hib_hi(qa.x), mxi_hib_hi(qa.y), s2);
        o[3] = fq_inj8(mxi_hib_hi(qb.x), mxi_hib_hi(qb.y), s2);
    } else {
        const _Float16 dl = (_Float16)e8m0_half(e);
        o[0] = fq_lut8(qa.x & 0x0F0F0F0Fu, qa.y & 0x0F0F0F0Fu, dl);
        o[1] = fq_lut8(qb.x & 0x0F0F0F0Fu, qb.y & 0x0F0F0F0Fu, dl);
        o[2] = fq_lut8((qa.x >> 4) & 0x0F0F0F0Fu, (qa.y >> 4) & 0x0F0F0F0Fu, dl);
        o[3] = fq_lut8((qb.x >> 4) & 0x0F0F0F0Fu, (qb.y >> 4) & 0x0F0F0F0Fu, dl);
    }
}

template <int BM, int BN, int WAVES_M, int GRP = 0>
__device__ void gemmfq_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                            float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int NTH = 256, BK = 64;
    constexpr int WAVES_N = NTH / 32 / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int LD = BK + 8;
    constexpr int GPR = BK / 8;
    constexpr int A_SB = BM * (BK / 32) / NTH;  // 32-value sub-blocks per lane per K chunk
    constexpr int B_PER = BN * GPR / NTH;
    static_assert(A_SB >= 1 && BM * (BK / 32) % NTH == 0 && BN * GPR % NTH == 0, "tile/threads");
    __shared__ _Float16 As[BM][LD];
    __shared__ _Float16 Bs[BN][LD];

    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    int m0, t0;
    if constexpr (GRP == 0) {
        m0 = blockIdx.y * BM;
        t0 = blockIdx.x * BN;
    } else {  // v0252-8sbal: block -> (row tile, token tile) in groups of GRP row tiles (fq_raster.h)
        unsigned rt, tt;
        fq_group_tile(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, GRP, &rt, &tt);
        m0 = static_cast<int>(rt) * BM;
        t0 = static_cast<int>(tt) * BN;
    }

    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    // lane slot s = threadIdx.x + e * NTH: row s / 2, sub-block s % 2 of the chunk (adjacent lanes: one row's 32 bytes)
    const uint8_t *arow[A_SB];
#pragma unroll
    for (int e = 0; e < A_SB; ++e) {
        const int s = threadIdx.x + e * NTH;
        arow[e] = W + (size_t)min(m0 + s / 2, nrows - 1) * row_bytes;
    }
    const _Float16 *brow[B_PER];
#pragma unroll
    for (int e = 0; e < B_PER; ++e) {
        const int i = threadIdx.x + e * NTH;
        brow[e] = X + (size_t)min(t0 + i / GPR, n_tok - 1) * ncols + (i % GPR) * 8;
    }
    uint2 qa[A_SB], qb[A_SB];
    uint32_t qe[A_SB];
    half8 rb[B_PER];
#define GEMMFQ_FETCH(k0_)                                                                           \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < A_SB; ++e) {                                          \
            const int s = threadIdx.x + e * NTH;                                                    \
            const uint8_t *blk = arow[e] + (size_t)((k0_) >> 8) * 136;                              \
            const int ib = (((k0_) & 255) >> 5) + (s & 1);                                          \
            const uint8_t *q = blk + 8 + 16 * ib;                                                   \
            qa[e] = *(const uint2 *)q;                                                              \
            qb[e] = *(const uint2 *)(q + 8);                                                        \
            qe[e] = blk[ib];                                                                        \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < B_PER; ++e) rb[e] = *(const half8 *)(brow[e] + (k0_)); \
    } while (0)

    GEMMFQ_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        half8 ra[A_SB][4];
#pragma unroll
        for (int e = 0; e < A_SB; ++e) fq_decode_sb(qa[e], qb[e], qe[e], ra[e]);
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_SB; ++e) {
            const int s = threadIdx.x + e * NTH;
#pragma unroll
            for (int j = 0; j < 4; ++j) *(half8 *)&As[s / 2][(s & 1) * 32 + j * 8] = ra[e][j];
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(half8 *)&Bs[i / GPR][(i % GPR) * 8] = rb[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) GEMMFQ_FETCH(k0 + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            half16 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = ld_frag16(&As[(wm * TM + i) * 16 + lane % 16][kk]);
#pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = ld_frag16(&Bs[(wn * TN + j) * 16 + lane % 16][kk]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = WMMA16(a[i], b[j], acc[i][j]);
        }
    }
#undef GEMMFQ_FETCH
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int t = t0 + (wn * TN + j) * 16 + lane % 16;
            if (t >= n_tok) continue;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + CROW(lane, v);
                if (row < nrows) {
                    float *dst = Y + (size_t)t * nrows + row;
                    *dst = accumulate ? *dst + acc[i][j][v] : acc[i][j][v];
                }
            }
        }
    }
}

// Entry points (host: MatmulKernels::gemmfq, kernels::kGemmFqCfgs / kGemmFqNames; selector ops::chooseGemmF16 with
// GemmFlags::fdq: the device default device::Profile::fdq_default, WHIRL_FDQ8S overrides). Same ABI as gemm_c*.
//   gemmfq_c2_mxfp4   256 x 128 WAVES_M 4, injected decode   (the kept c2 layout; slot 1)
//   gemmfq_t2_mxfp4   128 x 256 WAVES_M 2, injected decode   (quantized side short; slot 2, the gfx1151 default)
extern "C" __global__ void __launch_bounds__(256) gemmfq_c2_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmfq_impl<256, 128, 4, WHIRL_FQ_GRP_C2>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmfq_t2_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmfq_impl<128, 256, 2, WHIRL_FQ_GRP_T2>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }

// v0252-8sbal prototypes (kt only: tests/kernels/kt.h kFqProto; not in kernels::kGemmFqNames, so the product never
// selects them): the c2 / t2 tiles with the grouped raster of GRP row tiles. Same ABI and grid as their base slot.
#define GEMMFQ_PROTO(name_, bm_, bn_, wm_, g_)                                                                          \
    extern "C" __global__ void __launch_bounds__(256) gemmfq_##name_##_mxfp4(const uint8_t *W, uint64_t row_bytes,    \
        const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) {                               \
        gemmfq_impl<bm_, bn_, wm_, g_>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                          \
    }
GEMMFQ_PROTO(t2g4, 128, 256, 2, 4)
GEMMFQ_PROTO(t2g8, 128, 256, 2, 8)
GEMMFQ_PROTO(t2g16, 128, 256, 2, 16)
GEMMFQ_PROTO(c2g4, 256, 128, 4, 4)
GEMMFQ_PROTO(c2g8, 256, 128, 4, 8)
#undef GEMMFQ_PROTO

