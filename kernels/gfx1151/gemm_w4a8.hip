// v0252-f16 (F16 P1a): gfx1151 (Radeon 8060S) W4A8 prefill GEMM - MXFP4 weights exactly as WHIRL stores them x int8
// activations (one f32 scale per 32 along K), iu8 WMMA (__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 / _w64), int32 per
// 32-block, one f32 epilogue per block. Plus the activation quantizer (f16 -> the tiled Q8_1 layout).
// SPDX-License-Identifier: Apache-2.0
//
// P1a status: a NEW file, not included by kernels/whirl_kernels.hip or any product build. The P1a harness
// (scratch/v0252_f16) compiles it as its own code object (twice: wave32, and wave64 with -mwavefrontsize64
// -DW4A8_WAVE64=1) and times it against the v0.2.5.1 gemmfq (kernels/gfx1151/gemm_fq.hip) from the product's code
// object. Lossy (int8 activations): an allowed item of the 8060S balance (ADR-0013) and fast; never precise.
//
// Structure from Gufo's WKQuantA8BlockedWmmaGEMMKernel (refsrc gufo src/models/qwen/hip/kernels/prefill_quant_gemm.hpp
// :137-554 and prefill_quant_wave64.hip, MIT License, Copyright (c) 2026 gufo contributors): BM x BN output tile,
// BK 32-blocks per stage, weights decoded to int8 in WMMA-A order in LDS with slot-permuted per-row scales, activations
// pre-tiled so the B fragments are contiguous, two iu8 WMMAs per 32-block into a fresh int32 tile, then
// acc += float(c) * (dw * dx). What differs (progress/v0252-f16.PRIOR_ART.md sec 3):
//   * weights: WHIRL's MXFP4 layout (per 256 values e[8] then qs[8][16]), decoded with the two-table v_perm LUT
//     (w4a8_lut4 = WHIRL mx_lut4 = llama.cpp get_int_from_table_16, MIT); no Q4_K offset term;
//   * the raw weight / activation bytes of the next stage ride in registers across the WMMA loop and are decoded at the
//     top of the next iteration (gemmfq's pattern, from strix-llama.cpp mmb, MIT), not right after their loads;
//   * the epilogue is pinned: s = dw * dx, acc = fma(float(c), s, acc), ascending 32-blocks - the GPU equals the CPU
//     reference (tests/kernels/w4a8_ref.h) bit for bit, f32 output included;
//   * ABI = gemmfq's with the tiled int8 activations instead of f16: (W, row_bytes, Xq, Y, ncols, nrows, n_tok,
//     accumulate), Y[t * nrows + row] f32, row-clamped loads, token tiles clamped (outputs past n_tok never stored).
// Requires ncols % 256 == 0 (whole MXFP4 superblocks) and row_bytes % 8 == 0 (136-byte units), like gemmfq.

#include <hip/hip_runtime.h>

#include "w4a8_math.h"

typedef int w4a8_v4i __attribute__((ext_vector_type(4)));
typedef int w4a8_v8i __attribute__((ext_vector_type(8)));

#if defined(W4A8_WAVE64) && W4A8_WAVE64
#define W4A8_WS 64
typedef w4a8_v4i w4a8_acc_t;  // 16x16 int32 C tile over 64 lanes: 4 per lane (row 4 v + lane / 16, token lane % 16)
#define W4A8_NE 4
#define W4A8_WMMA(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, (a), true, (b), (c), false)
#else
#define W4A8_WS 32
typedef w4a8_v8i w4a8_acc_t;  // over 32 lanes: 8 per lane (row 2 v + lane / 16, token lane % 16) = CROW of gfx1151
#define W4A8_NE 8
#define W4A8_WMMA(a, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, (a), true, (b), (c), false)
#endif

// LDS slot of row rl (0..15) of a 16-row tile in the per-row scale array: lane group h (lane / 16) reads W4A8_NE
// consecutive slots h * NE .. h * NE + NE - 1 = rows NE-interleaved (Gufo prefill_quant_gemm.hpp:349-351)
__device__ __forceinline__ int w4a8_slot(int rl) {
    return W4A8_WS == 32 ? (rl & 1) * 8 + (rl >> 1) : (rl & 3) * 4 + (rl >> 2);
}

template <int BM, int BN, int BK, int WM, int WN>
__device__ __forceinline__ void w4a8_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const uint8_t *__restrict__ Xq,
                                          float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int NTH = WM * WN * W4A8_WS;
    static_assert(NTH == 256, "256 threads per block");
    constexpr int RT = BM / 16, TT = BN / 16;  // 16-row / 16-token tiles per block
    constexpr int TM = RT / WM, TN = TT / WN;  // per wave
    static_assert(RT % WM == 0 && TT % WN == 0 && TM >= 1 && TN >= 1, "tile split");
    constexpr int WU = BM * BK / NTH;           // weight units (one row's 32-block) per thread per stage
    static_assert(WU >= 1 && BM * BK % NTH == 0, "weight units");
    constexpr int XQ = TT * BK * 32;            // 16-byte activation chunks (int8 part) per stage
    constexpr int XU = (XQ + NTH - 1) / NTH;
    constexpr int XD = TT * BK * 4;             // 16-byte scale chunks per stage
    static_assert(XD <= NTH, "one scale chunk per thread at most");
    static_assert(8 % BK == 0, "BK divides a superblock's 8 blocks: no K tail for ncols % 256 == 0");
    constexpr int NE = W4A8_NE;

    __shared__ w4a8_v4i sA[BK][RT][32];   // [kb][row tile][rl: k 0..15 | 16 + rl: k 16..31]
    __shared__ __attribute__((aligned(16))) float sDw[BK][RT][16];  // 2^(e - 128) per row, slot-permuted
    __shared__ w4a8_v4i sB[BK][TT][32];   // [kb][token tile][tl: b0 | 16 + tl: b1] (the tile's first 512 bytes as is)
    __shared__ __attribute__((aligned(16))) float sDx[BK][TT][16];  // per token

    const int tid = threadIdx.x;
    const int lane = tid % W4A8_WS, wid = tid / W4A8_WS;
    const int l16 = lane & 15, grp = lane >> 4;
    const int wm = wid % WM, wn = wid / WM;
    const int m0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
    const int nkb = ncols / 32;
    const int ntt = (n_tok + 15) / 16;

    w4a8_acc_t zero;
#pragma unroll
    for (int v = 0; v < NE; ++v) zero[v] = 0;
    float acc[TM][TN][NE];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int v = 0; v < NE; ++v) acc[i][j][v] = 0.f;

    // weight unit u = tid + e * NTH: row u / BK of the block, 32-block u % BK of the stage (adjacent threads: one row's
    // consecutive blocks = contiguous qs bytes)
    const uint8_t *wrow[WU];
#pragma unroll
    for (int e = 0; e < WU; ++e) {
        const int u = tid + e * NTH;
        wrow[e] = W + (size_t)min(m0 + u / BK, nrows - 1) * row_bytes;
    }
    // activation tiles of this block, clamped to the last tile (rows past n_tok are never stored), as per-thread tile
    // rows: an array of TT tile pointers indexed per thread would sit in 2 x TT VGPRs
    size_t xrow[XU];
#pragma unroll
    for (int e = 0; e < XU; ++e) xrow[e] = (size_t)min(t0 / 16 + (tid + e * NTH) / 32 / BK % TT, ntt - 1) * nkb;
    const size_t drow = (size_t)min(t0 / 16 + tid / 4 / BK % TT, ntt - 1) * nkb;

    uint2 qa[WU], qb[WU];
    uint32_t qe[WU];
    w4a8_v4i xr[XU];
    w4a8_v4i dr = {0, 0, 0, 0};
#define W4A8_FETCH(kb0_)                                                                                    \
    do {                                                                                                    \
        _Pragma("unroll") for (int e = 0; e < WU; ++e) {                                                    \
            const int u = tid + e * NTH;                                                                    \
            const int kbg = min((kb0_) + u % BK, nkb - 1);                                                  \
            const uint8_t *sb = wrow[e] + (size_t)(kbg >> 3) * 136;                                         \
            const uint8_t *q = sb + 8 + 16 * (kbg & 7);                                                     \
            qa[e] = *(const uint2 *)q;                                                                      \
            qb[e] = *(const uint2 *)(q + 8);                                                                \
            qe[e] = sb[kbg & 7];                                                                            \
        }                                                                                                   \
        _Pragma("unroll") for (int e = 0; e < XU; ++e) {                                                    \
            const int u = tid + e * NTH;                                                                    \
            if (XQ % NTH == 0 || u < XQ) {                                                                  \
                const int c = u % 32, kb = (u / 32) % BK;                                                   \
                const int kbg = min((kb0_) + kb, nkb - 1);                                                  \
                xr[e] = *(const w4a8_v4i *)(Xq + (xrow[e] + kbg) * kW4A8TileBytes + 16 * c);                \
            }                                                                                               \
        }                                                                                                   \
        if (tid < XD) {                                                                                     \
            const int c = tid % 4, kb = (tid / 4) % BK;                                                     \
            const int kbg = min((kb0_) + kb, nkb - 1);                                                      \
            dr = *(const w4a8_v4i *)(Xq + (drow + kbg) * kW4A8TileBytes + kW4A8ScaleOff + 16 * c);          \
        }                                                                                                   \
    } while (0)

    W4A8_FETCH(0);
    for (int kb0 = 0; kb0 < nkb; kb0 += BK) {
        // decode the weights of this stage (registers), then publish the stage
        w4a8_v4i lo[WU], hi[WU];
        float dw[WU];
#pragma unroll
        for (int e = 0; e < WU; ++e) {
            lo[e] = (w4a8_v4i){(int)w4a8_lut4(qa[e].x), (int)w4a8_lut4(qa[e].y), (int)w4a8_lut4(qb[e].x), (int)w4a8_lut4(qb[e].y)};
            hi[e] = (w4a8_v4i){(int)w4a8_lut4(qa[e].x >> 4), (int)w4a8_lut4(qa[e].y >> 4), (int)w4a8_lut4(qb[e].x >> 4),
                               (int)w4a8_lut4(qb[e].y >> 4)};
            dw[e] = __uint_as_float(w4a8_e8m0_bits(qe[e]));
        }
        __syncthreads();
#pragma unroll
        for (int e = 0; e < WU; ++e) {
            const int u = tid + e * NTH;
            const int r = u / BK, kb = u % BK;
            sA[kb][r >> 4][r & 15] = lo[e];
            sA[kb][r >> 4][16 + (r & 15)] = hi[e];
            sDw[kb][r >> 4][w4a8_slot(r & 15)] = dw[e];
        }
#pragma unroll
        for (int e = 0; e < XU; ++e) {
            const int u = tid + e * NTH;
            if (XQ % NTH == 0 || u < XQ) {
                const int c = u % 32, kb = (u / 32) % BK, tt = u / 32 / BK;
                sB[kb][tt][c] = xr[e];
            }
        }
        if (tid < XD) {
            const int c = tid % 4, kb = (tid / 4) % BK, tt = tid / 4 / BK;
            *(w4a8_v4i *)&sDx[kb][tt][4 * c] = dr;
        }
        __syncthreads();
        if (kb0 + BK < nkb) W4A8_FETCH(kb0 + BK);

        // ncols % 256 == 0 and BK | 8: every stage is whole (no K tail)
#pragma unroll
        for (int kb = 0; kb < BK; ++kb) {
            if constexpr (TM <= 2) {
                // Gufo's order: the wave's A fragments and row scales stay in registers, B is read once per token tile
                w4a8_v4i alo[TM], ahi[TM];
                float sw[TM][NE];
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const int rs = wm * TM + i;
                    alo[i] = sA[kb][rs][l16];
                    ahi[i] = sA[kb][rs][16 + l16];
#pragma unroll
                    for (int v = 0; v < NE; ++v) sw[i][v] = sDw[kb][rs][grp * NE + v];
                }
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int ts = wn * TN + j;
                    const w4a8_v4i blo = sB[kb][ts][l16], bhi = sB[kb][ts][16 + l16];
                    const float dx = sDx[kb][ts][l16];
                    w4a8_acc_t c[TM];
#pragma unroll
                    for (int i = 0; i < TM; ++i) c[i] = W4A8_WMMA(alo[i], blo, zero);
#pragma unroll
                    for (int i = 0; i < TM; ++i) c[i] = W4A8_WMMA(ahi[i], bhi, c[i]);
#pragma unroll
                    for (int i = 0; i < TM; ++i)
#pragma unroll
                        for (int v = 0; v < NE; ++v) {
                            const float s = sw[i][v] * dx;
                            acc[i][j][v] = __builtin_fmaf((float)c[i][v], s, acc[i][j][v]);
                        }
                }
            } else {
                // 4 or more row tiles per wave (4 x 4 / 4 x 8 output tiles = 128+ accumulator VGPRs): one row tile at a
                // time, B re-read from LDS per (row tile, token tile) so the live fragments fit beside the accumulators
                float dx[TN];
#pragma unroll
                for (int j = 0; j < TN; ++j) dx[j] = sDx[kb][wn * TN + j][l16];
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const int rs = wm * TM + i;
                    const w4a8_v4i alo = sA[kb][rs][l16], ahi = sA[kb][rs][16 + l16];
                    float sw[NE];
#pragma unroll
                    for (int v = 0; v < NE; ++v) sw[v] = sDw[kb][rs][grp * NE + v];
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const int ts = wn * TN + j;
                        w4a8_acc_t c = W4A8_WMMA(alo, sB[kb][ts][l16], zero);
                        c = W4A8_WMMA(ahi, sB[kb][ts][16 + l16], c);
#pragma unroll
                        for (int v = 0; v < NE; ++v) {
                            const float s = sw[v] * dx[j];
                            acc[i][j][v] = __builtin_fmaf((float)c[v], s, acc[i][j][v]);
                        }
                    }
                }
            }
        }
    }
#undef W4A8_FETCH

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int t = t0 + (wn * TN + j) * 16 + l16;
            if (t >= n_tok) continue;
#pragma unroll
            for (int v = 0; v < NE; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + (W4A8_WS == 32 ? 2 * v + grp : 4 * v + grp);
                if (row < nrows) {
                    float *dst = Y + (size_t)t * nrows + row;
                    *dst = accumulate ? *dst + acc[i][j][v] : acc[i][j][v];
                }
            }
        }
    }
}

// Entry points (P1a candidates; the harness screens them per shape). <tag>: tile BM x BN, waves WM x WN.
#define W4A8_ENTRY(name, BM, BN, BK, WM, WN)                                                                    \
    extern "C" __global__ void __launch_bounds__(256)                                                           \
        name(const uint8_t *W, uint64_t row_bytes, const uint8_t *Xq, float *Y, int ncols, int nrows, int n_tok, \
             int accumulate) {                                                                                  \
        w4a8_impl<BM, BN, BK, WM, WN>(W, row_bytes, Xq, Y, ncols, nrows, n_tok, accumulate);                     \
    }
#if W4A8_WS == 32
W4A8_ENTRY(w4a8_g32_mxfp4, 128, 128, 2, 4, 2)  // Gufo's wave32 tile (2 x 4 tiles per wave)
W4A8_ENTRY(w4a8_g32b4_mxfp4, 128, 128, 4, 4, 2)  // the same with 4 blocks (128 K) per stage: half the barriers
W4A8_ENTRY(w4a8_t32_mxfp4, 128, 256, 2, 2, 4)  // gemmfq t2's tile (4 x 4 tiles per wave)
W4A8_ENTRY(w4a8_c32_mxfp4, 256, 128, 2, 4, 2)  // gemmfq c2's tile (4 x 4 tiles per wave)
#else
W4A8_ENTRY(w4a8_g64_mxfp4, 128, 128, 2, 2, 2)  // Gufo's wave64 "square" tile (4 x 4 tiles per wave, 4 C values per lane)
W4A8_ENTRY(w4a8_t64_mxfp4, 128, 256, 2, 2, 2)  // 4 x 8 tiles per wave
#endif

// f16 activations [n_tok][k] -> the tiled int8 layout (kernels/gfx1151/w4a8_math.h). 64 threads per (16-token tile,
// 32-block): thread = token tl (0..15) x part (0..3, 8 values each); 4 tiles per 256-thread block; grid = ceil(tiles / 4)
// with tiles = ceil(n_tok / 16) * k / 32. Padding tokens get q = 0, d = 0. Same steps as w4a8_quant_block (host).
extern "C" __global__ void __launch_bounds__(256) w4a8_quant_f16(const _Float16 *__restrict__ X, uint8_t *__restrict__ Xq, int n_tok, int k) {
    const int nkb = k / 32, ntt = (n_tok + 15) / 16;
    const long tile = (long)blockIdx.x * 4 + (threadIdx.x >> 6);
    if (tile >= (long)ntt * nkb) return;
    const int tt = (int)(tile / nkb), kb = (int)(tile % nkb);
    const int l = threadIdx.x & 63, tl = l >> 2, part = l & 3;
    const int t = tt * 16 + tl;
    float v[8];
    if (t < n_tok) {
        typedef _Float16 h8 __attribute__((ext_vector_type(8)));
        const h8 h = *(const h8 *)(X + (size_t)t * k + kb * 32 + part * 8);
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = (float)h[i];
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = 0.f;
    }
    float am = 0.f;
    int bad = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float a = fabsf(v[i]);
        bad |= !(a <= 3.402823466e+38f);
        am = fmaxf(am, a);
    }
    am = fmaxf(am, __shfl_xor(am, 1, 4));
    am = fmaxf(am, __shfl_xor(am, 2, 4));
    bad |= __shfl_xor(bad, 1, 4);
    bad |= __shfl_xor(bad, 2, 4);
    const float d = bad ? __builtin_nanf("") : w4a8_scale(am);
    const float id = w4a8_inv(d);
    uint32_t w0 = 0, w1 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q0 = bad ? 0 : w4a8_q8(v[i], id), q1 = bad ? 0 : w4a8_q8(v[4 + i], id);
        w0 |= (uint32_t)(q0 & 0xFF) << (8 * i);
        w1 |= (uint32_t)(q1 & 0xFF) << (8 * i);
    }
    uint8_t *dst = Xq + (size_t)tile * kW4A8TileBytes;
    *(uint2 *)(dst + (part >> 1) * 256 + tl * 16 + (part & 1) * 8) = make_uint2(w0, w1);
    if (part == 0) *(float *)(dst + kW4A8ScaleOff + tl * 4) = d;
}
