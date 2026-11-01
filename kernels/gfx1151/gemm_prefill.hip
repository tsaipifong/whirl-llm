// WHIRL device code (gfx1151 / RDNA 3.5): prefill GEMM (WMMA f16 v_wmma_f32_16x16x16_f16, configurations).
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Prefill matmul: Y[t][r] (+)= sum_c W[r][c] * X[t][c] for a tile of TT tokens.
// X chunks of 256 columns are staged in LDS and shared by the 8 waves of the
// block; each wave owns RW rows and keeps RW*TT accumulators per lane.
// Requires ncols % 256 == 0.
// ---------------------------------------------------------------------------
#include "shared/gemm_prefill_gemm_entry.h"  // also in the gfx1201 set (kernels/gemm_prefill.hip)


#include "shared/gemm_prefill_f32_to_f16.h"  // also in the gfx1201 set (kernels/gemm_prefill.hip)

template <int T, int BM, int BN, int BK, int MODE = 0>
__device__ void gemm_wmma_impl(const uint8_t *__restrict__ W, uint64_t row_bytes,
                               const _Float16 *__restrict__ X, float *__restrict__ Y,
                               int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int LD = BK + 8;
    constexpr int MT = BM / 16, NT = BN / 16;
    constexpr int PER_WAVE = MT * NT / 8;  // 8 waves
    constexpr int GPR = BK / 8;            // 8-value groups per row per chunk
    __shared__ _Float16 As[BM][LD];
    __shared__ _Float16 Bs[BN][LD];

    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;

    float8 acc[PER_WAVE];
#pragma unroll
    for (int i = 0; i < PER_WAVE; ++i) acc[i] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
        for (int i = threadIdx.x; i < BM * GPR; i += 256) {
            const int r = i / GPR, gg = i % GPR;
            half8 h;
            if (m0 + r < nrows) {
                if (MODE == 1) {
                    h = *(const half8 *)(W + (size_t)(m0 + r) * row_bytes + ((k0 / 8 + gg) * 16) % row_bytes);
                } else {
                    float w[8];
                    dec_group<T>(W + (size_t)(m0 + r) * row_bytes, k0 / 8 + gg, w);
#pragma unroll
                    for (int e = 0; e < 8; ++e) h[e] = (_Float16)w[e];
                }
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) h[e] = (_Float16)0.f;
            }
            *(half8 *)&As[r][gg * 8] = h;
        }
        for (int i = threadIdx.x; i < BN * GPR; i += 256) {
            const int t = i / GPR, gg = i % GPR;
            half8 h;
            if (t0 + t < n_tok) {
                h = *(const half8 *)(X + (size_t)(t0 + t) * ncols + k0 + gg * 8);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) h[e] = (_Float16)0.f;
            }
            *(half8 *)&Bs[t][gg * 8] = h;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // each wave owns a fixed mi and PER_WAVE consecutive ni
            const int mi = wid % MT;
            const half16 a = ld_frag16(&As[mi * 16 + lane % 16][kk]);
#pragma unroll
            for (int p = 0; p < PER_WAVE; ++p) {
                const int ni = (wid / MT) * PER_WAVE + p;
                const half16 b = ld_frag16(&Bs[ni * 16 + lane % 16][kk]);
                if (MODE == 2) {
                    acc[p][0] += (float)a[0] * (float)b[0];
                } else {
                    acc[p] = WMMA16(a, b, acc[p]);
                }
            }
        }
    }
    const int mi = wid % MT;
#pragma unroll
    for (int p = 0; p < PER_WAVE; ++p) {
        const int ni = (wid / MT) * PER_WAVE + p;
        const int t = t0 + ni * 16 + lane % 16;
        if (t >= n_tok) continue;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int row = m0 + mi * 16 + CROW(lane, v);
            if (row < nrows) {
                float *dst = Y + (size_t)t * nrows + row;
                *dst = accumulate ? *dst + acc[p][v] : acc[p][v];
            }
        }
    }
}

// BM=128 rows x BN=128 tokens, BK=64: 8 waves, each 1 row-tile x 8 token-tiles.
#include "shared/gemm_prefill_wmma_entry.h"  // also in the gfx1201 set (kernels/gemm_prefill.hip)

// Word-load decoders for the types whose blocks are only 2-byte aligned (Q6_K, Q8_0,
// IQ4_NL, Q3_K). Same values as the generic dec_group + f16 conversion, bit for bit:
// every weight is one f32 product (scale x exact small integer) rounded to f16 once;
// only the loads (dwords from 2-aligned addresses instead of bytes) and the integer
// unpacking differ.
__device__ __forceinline__ uint32_t ld32a2(const uint8_t *p) {
    uint32_t v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 4);
    return v;
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_Q6_K>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 5) * 210;
    const int p = (g & 31) * 8;
    const int n = p >> 7, r = p & 127, quarter = r >> 5, l = r & 31;
    const float d = h2f(b + 208);
    const float ds = d * ((const int8_t *)(b + 192))[8 * n + (l >> 4) + 2 * quarter];
    const uint8_t *qlp = b + 64 * n + l + ((quarter & 1) ? 32 : 0), *qhp = b + 128 + 32 * n + l;
    const int lsh = (quarter >> 1) ? 4 : 0, hsh = 2 * quarter;
    const uint32_t w[2] = {((ld32a2(qlp) >> lsh) & 0x0F0F0F0Fu) | (((ld32a2(qhp) >> hsh) & 0x03030303u) << 4),
                           ((ld32a2(qlp + 4) >> lsh) & 0x0F0F0F0Fu) | (((ld32a2(qhp + 4) >> hsh) & 0x03030303u) << 4)};
    half8 h;
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = (_Float16)(ds * (float)((int)((w[i >> 2] >> (8 * (i & 3))) & 0xFF) - 32));
    return h;
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_Q8_0>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 2) * 34;
    const float d = h2f(b);
    const uint8_t *q = b + 2 + 8 * (g & 3);
    const uint32_t w[2] = {ld32a2(q), ld32a2(q + 4)};
    half8 h;
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = (_Float16)(d * (float)(int8_t)(w[i >> 2] >> (8 * (i & 3))));
    return h;
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_IQ4_NL>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 2) * 18;
    const float d = h2f(b);
    const int p = (g & 3) * 8;
    const uint8_t *q = b + 2 + (p & 15);
    const int sh = p < 16 ? 0 : 4;
    const uint32_t w[2] = {(uint32_t)iq4_lut4((ld32a2(q) >> sh) & 0x0F0F0F0Fu), (uint32_t)iq4_lut4((ld32a2(q + 4) >> sh) & 0x0F0F0F0Fu)};
    half8 h;
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = (_Float16)(d * (float)(int8_t)(w[i >> 2] >> (8 * (i & 3))));
    return h;
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_Q3_K>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 5) * 110;
    const int p = (g & 31) * 8;
    const uint8_t *hm = b, *qs = b + 32, *sb = b + 96;
    const float d = h2f(b + 108);
    const int n = p >> 7, r = p & 127, j = r >> 5, s = r & 31;
    const int k = 8 * n + 2 * j + (s >> 4);
    const int lo = (k < 8) ? (sb[k] & 0xF) : (sb[k - 8] >> 4);
    const int hi = (sb[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
    const float dl = d * ((lo | (hi << 4)) - 32);
    const int bit = 4 * n + j, shift = 2 * j;
    const uint8_t *q = qs + 32 * n + s;
    // per byte: ((q >> shift) & 3) - (hmask bit set ? 0 : 4)  ->  ((q >> shift) & 3) + 4 * hbit - 4
    const uint32_t w[2] = {((ld32a2(q) >> shift) & 0x03030303u) | (((ld32a2(hm + s) >> bit) & 0x01010101u) << 2),
                           ((ld32a2(q + 4) >> shift) & 0x03030303u) | (((ld32a2(hm + s + 4) >> bit) & 0x01010101u) << 2)};
    half8 h;
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = (_Float16)(dl * (float)((int)((w[i >> 2] >> (8 * (i & 3))) & 0xFF) - 4));
    return h;
}

template <int T, int BM, int BN, int BK, int WAVES_M, int MODE = 0, int NTH = 256, bool TOK_X = false>
__device__ void gemm3_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                           float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    // main loop: kernels/shared/gemm3_mainloop.h (shared with gemm_tile_impl and the gfx1201 set, ADR-0005)
    // TOK_X: token blocks along grid x so the blocks sharing a weight row
    // block run back to back and reuse it from L2 instead of re-streaming W.
#define G3_TILE_ORIGIN const int m0 = (TOK_X ? blockIdx.y : blockIdx.x) * BM, t0 = (TOK_X ? blockIdx.x : blockIdx.y) * BN;
    // MODE 1: raw f16 rows
#define G3_A_MODE1(k_)                                                                                  \
    if (MODE == 1) {                                                                                    \
        ra[e] = *(const half8 *)(W + (size_t)(m0 + r) * row_bytes + (((k_) / 8 + gg) * 16) % row_bytes); \
    } else
#define G3_SKIP_CHUNK
#define G3_COL_IF(j)
#include "shared/gemm3_mainloop.h"
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
#include "shared/gemm_store_ij_f32.h"  // f32 Y[t][row] of accumulator (i, j) (ADR-0005)
        }
    }
}

#define GEMM3_BM 256
#define GEMM3_BN 256
#define GEMM3_NTH 512
#define GEMM3_ENTRY(SUF, T)                                                                         \
    extern "C" __global__ void __launch_bounds__(512)                                               \
        gemm3_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols,  \
                    int nrows, int n_tok, int accumulate) {                                        \
        gemm3_impl<T, GEMM3_BM, GEMM3_BN, 16, 4, 0, GEMM3_NTH>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); \
    }

GEMM3_ENTRY(f32, QT_F32)
GEMM3_ENTRY(f16, QT_F16)
GEMM3_ENTRY(q8_0, QT_Q8_0)
GEMM3_ENTRY(q3_k, QT_Q3_K)
GEMM3_ENTRY(q4_k, QT_Q4_K)
GEMM3_ENTRY(q5_k, QT_Q5_K)
GEMM3_ENTRY(q6_k, QT_Q6_K)
GEMM3_ENTRY(iq4_nl, QT_IQ4_NL)
GEMM3_ENTRY(iq3_s, QT_IQ3_S)
GEMM3_ENTRY(iq4_xs, QT_IQ4_XS)
GEMM3_ENTRY(mxfp4, QT_MXFP4)

// ---------------------------------------------------------------------------
// Prefill GEMM v4: gemm3 + double-buffered LDS (one barrier per K chunk).
// Chunk k+1 is fetched into registers, chunk k multiplied from LDS buffer
// `cur`, then the registers are stored into the other buffer.
// ---------------------------------------------------------------------------
template <int T, int BM, int BN, int BK, int WAVES_M, int NTH = 256>
__device__ void gemm4_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                           float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int WAVES_N = NTH / 32 / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int LD = BK + 8;
    constexpr int GPR = BK / 8;
    constexpr int A_PER = BM * GPR / NTH;
    constexpr int B_PER = BN * GPR / NTH;
    static_assert(BM * GPR % NTH == 0 && BN * GPR % NTH == 0, "tile/threads");
    __shared__ _Float16 As[2][BM][LD];
    __shared__ _Float16 Bs[2][BN][LD];

    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;

    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    half8 ra[A_PER], rb[B_PER];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            const int r = i / GPR, gg = i % GPR;
            if (m0 + r < nrows) {
                ra[e] = dec_group_h<T>(W + (size_t)(m0 + r) * row_bytes, k0 / 8 + gg);
            } else {
#pragma unroll
                for (int q = 0; q < 8; ++q) ra[e][q] = (_Float16)0.f;
            }
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            const int t = i / GPR, gg = i % GPR;
            if (t0 + t < n_tok) {
                rb[e] = *(const half8 *)(X + (size_t)(t0 + t) * ncols + k0 + gg * 8);
            } else {
#pragma unroll
                for (int q = 0; q < 8; ++q) rb[e][q] = (_Float16)0.f;
            }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(half8 *)&As[buf][i / GPR][(i % GPR) * 8] = ra[e];
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(half8 *)&Bs[buf][i / GPR][(i % GPR) * 8] = rb[e];
        }
    };

    fetch(0);
    store(0);
    __syncthreads();
    int cur = 0;
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        const bool more = k0 + BK < ncols;
        if (more) fetch(k0 + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            half16 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = ld_frag16(&As[cur][(wm * TM + i) * 16 + lane % 16][kk]);
#pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = ld_frag16(&Bs[cur][(wn * TN + j) * 16 + lane % 16][kk]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = WMMA16(a[i], b[j], acc[i][j]);
        }
        if (more) store(cur ^ 1);
        __syncthreads();
        cur ^= 1;
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
#include "shared/gemm_store_ij_f32.h"  // f32 Y[t][row] of accumulator (i, j) (ADR-0005)
        }
    }
}

#include "shared/gemm_prefill_dequant_f16_f32.h"  // also in the gfx1201 set (kernels/gemm_prefill.hip)
extern "C" __global__ void __launch_bounds__(256) gemm_c2_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
#include "shared/gemm_prefill_gemm_c2_q8_0.h"  // also in the gfx1201 set (kernels/gemm_prefill.hip)
extern "C" __global__ void __launch_bounds__(512) gemm_c10_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 128, 256, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c10_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c10_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 128, 256, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c10_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 128, 256, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c10_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 128, 256, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c10_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 128, 256, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c10_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 128, 256, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c10_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 128, 256, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c10_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 128, 256, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c10_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 128, 256, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c10_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 128, 256, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c11_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 256, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c11_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c11_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 256, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c11_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 256, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c11_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 256, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c11_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 256, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c11_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 256, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c11_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 256, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c11_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 256, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c11_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 256, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c11_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 256, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
#include "shared/gemm_prefill_gemm_c12_f32.h"  // also in the gfx1201 set (kernels/gemm_prefill.hip)
