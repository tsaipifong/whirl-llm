// gfx1151 (Radeon 8060S) PRODUCT W4A4 (iu4 WMMA) dense prefill GEMM of the FAST mode (v0252-w4a4p, PLAN V1-144 round 1).
// SPDX-License-Identifier: Apache-2.0
//
// A fragment of the gfx1151 translation unit (kernels/gfx1151/whirl_kernels_gfx1151.hip includes it LAST, so no existing
// kernel moves). Family kernels.gemm (glob kernels/gfx1151/gemm_*.hip). Fast-only opt-in: WHIRL_W4A4=1 -> numerics flag
// w4a4_sim with bit w4a4_iu4 (= the accepted sim arm wga + this kernel); host MatmulKernels::w4q_wc / w4q_qx / w4q_gemm,
// ops::w4a4Takes, Context::matmul. Never used by precise / balance, the R9700 or any default plan.
// The math (codes, scales, packing, fold order) is kernels/shared/gemm_w4a4_prod.h (pinned to the sim arm and to the K1 probe
// by whirl-kt-bench-tests; the kernels to its host references by whirl-kernel-test --only w4s).
//   w4q_wconv_mxfp4(W, row_bytes, Wq, Ws, ncols, nrows, gain): once per matrix (Context's cache): every MXFP4 32-sub-block ->
//     16 packed int4 bytes (Wq [rows][ncols / 2]) + f32 scale w4s_wscale(e) x gain[row] (Ws [rows][ncols / 32]). One thread per
//     sub-block, grid-stride. ncols % 256 == 0.
//   w4q_quant_x(X, Xq, Xs, nblk): nblk 32-value f16 blocks (rows of K % 32 == 0, back to back) -> packed int4 + step (absmax / 7).
//   w4q_gemm(Wq, Ws, Xq, Xs, Y, ncols, nrows, n_tok, acc, h16): Y[t][row] (+)= sum over 32-blocks b ascending of
//     fma(float(c_b), ws[b] x xs[b], .) - the K1 probe's i4fb form (128 rows x 128 tokens, 256 threads = 4 waves along M x 2
//     along N, 2 x 4 WMMA tiles per wave, K stage 128, register prefetch of the next stage) with a token guard (any n_tok:
//     rows of a partial tile read zeros and are not stored) and the f16 GEMM epilogue (h16: f16 output; acc: y + v).
//     Requires nrows % 128 == 0, ncols % 128 == 0 (ops::w4a4Takes checks it).

#include "shared/gemm_w4a4_prod.h"

typedef int w4q_v2i __attribute__((ext_vector_type(2)));
typedef int w4q_v4i __attribute__((ext_vector_type(4)));
typedef int w4q_v8i __attribute__((ext_vector_type(8)));
#define W4Q_WMMA_I4(a_, b_, c_) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, (a_), true, (b_), (c_), false)

extern "C" __global__ void __launch_bounds__(256) w4q_wconv_mxfp4(const uint8_t *W, uint64_t row_bytes, uint8_t *Wq, float *Ws, int ncols, int nrows,
                                                                  const float *gain) {
    const size_t spr = static_cast<size_t>(ncols) / 32;
    const size_t total = spr * static_cast<size_t>(nrows);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / spr, b = i % spr;
        const uint8_t *blk = W + r * row_bytes + (b / 8) * 136;
        const int ib = static_cast<int>(b % 8);
        uint8_t q[16];
        float s;
        w4q_wblock(blk + 8 + 16 * ib, blk[ib], gain[r], q, &s);
        uint32_t u[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) u[j] = (uint32_t)q[4 * j] | ((uint32_t)q[4 * j + 1] << 8) | ((uint32_t)q[4 * j + 2] << 16) | ((uint32_t)q[4 * j + 3] << 24);
        *(uint4 *)(Wq + r * (static_cast<size_t>(ncols) / 2) + b * 16) = make_uint4(u[0], u[1], u[2], u[3]);
        Ws[r * spr + b] = s;
    }
}

extern "C" __global__ void __launch_bounds__(256) w4q_quant_x(const _Float16 *X, uint8_t *Xq, float *Xs, uint64_t nblk) {
    for (uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; g < nblk; g += (uint64_t)gridDim.x * blockDim.x) {
        float x[32];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half8 h = *(const half8 *)(X + g * 32 + i * 8);
#pragma unroll
            for (int j = 0; j < 8; ++j) x[i * 8 + j] = (float)h[j];
        }
        uint8_t q[16];
        float d;
        w4q_xblock(x, q, &d);
        uint32_t u[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) u[j] = (uint32_t)q[4 * j] | ((uint32_t)q[4 * j + 1] << 8) | ((uint32_t)q[4 * j + 2] << 16) | ((uint32_t)q[4 * j + 3] << 24);
        *(uint4 *)(Xq + g * 16) = make_uint4(u[0], u[1], u[2], u[3]);
        Xs[g] = d;
    }
}

extern "C" __global__ void __launch_bounds__(256) w4q_gemm(const uint8_t *__restrict__ Wq, const float *__restrict__ Ws, const uint8_t *__restrict__ Xq,
                                                           const float *__restrict__ Xs, void *__restrict__ Yv, int ncols, int nrows, int n_tok, int acc,
                                                           int h16) {
    constexpr int BM = 128, BN = 128, WAVES_M = 4, NTH = 256, WAVES_N = NTH / WARP / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int BK = 128, BKB = BK / 2, NB = BK / 32;
    constexpr int LDB = BKB + 16;
    constexpr int A_V = BM * BKB / 16 / NTH, B_V = BN * BKB / 16 / NTH;
    constexpr int A_S = BM * NB / NTH, B_S = BN * NB / NTH;
    static_assert(A_V >= 1 && B_V >= 1 && A_S >= 1 && B_S >= 1 && TM >= 1 && TN >= 1, "tile/threads");
    __shared__ __attribute__((aligned(16))) uint8_t As[BM * LDB];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[BN * LDB];
    __shared__ __attribute__((aligned(16))) float Ass[NB][BM];
    __shared__ __attribute__((aligned(16))) float Bss[NB][BN];

    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    const int m0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
    const size_t rbytes = (size_t)ncols / 2, rsc = (size_t)ncols / 32;

    w4q_v4i ra[A_V], rb[B_V];
    float sa[A_S], sb[B_S];
    // tokens past n_tok read zero codes and zero steps (their outputs are not stored)
#define W4Q_FETCH(k0_)                                                                                                          \
    do {                                                                                                                        \
        _Pragma("unroll") for (int e = 0; e < A_V; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH;                                                                                \
            ra[e] = *(const w4q_v4i *)(Wq + (size_t)(m0 + s / 4) * rbytes + (k0_) / 2 + (s % 4) * 16);                           \
        }                                                                                                                       \
        _Pragma("unroll") for (int e = 0; e < B_V; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH;                                                                                \
            rb[e] = t0 + s / 4 < n_tok ? *(const w4q_v4i *)(Xq + (size_t)(t0 + s / 4) * rbytes + (k0_) / 2 + (s % 4) * 16)      \
                                       : (w4q_v4i){0, 0, 0, 0};                                                                 \
        }                                                                                                                       \
        _Pragma("unroll") for (int e = 0; e < A_S; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH;                                                                                \
            sa[e] = Ws[(size_t)(m0 + s / 4) * rsc + (k0_) / 32 + s % 4];                                                        \
        }                                                                                                                       \
        _Pragma("unroll") for (int e = 0; e < B_S; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH;                                                                                \
            sb[e] = t0 + s / 4 < n_tok ? Xs[(size_t)(t0 + s / 4) * rsc + (k0_) / 32 + s % 4] : 0.f;                             \
        }                                                                                                                       \
    } while (0)

    float8 facc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) facc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    W4Q_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_V; ++e) {
            const int s = threadIdx.x + e * NTH;
            *(w4q_v4i *)&As[(s / 4) * LDB + (s % 4) * 16] = ra[e];
        }
#pragma unroll
        for (int e = 0; e < B_V; ++e) {
            const int s = threadIdx.x + e * NTH;
            *(w4q_v4i *)&Bs[(s / 4) * LDB + (s % 4) * 16] = rb[e];
        }
#pragma unroll
        for (int e = 0; e < A_S; ++e) {
            const int s = threadIdx.x + e * NTH, r = s / 4, rr = r & 15;
            Ass[s % 4][(r & ~15) + (rr & 1) * 8 + rr / 2] = sa[e];  // CROW order inside each 16-row tile
        }
#pragma unroll
        for (int e = 0; e < B_S; ++e) {
            const int s = threadIdx.x + e * NTH;
            Bss[s % 4][s / 4] = sb[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) W4Q_FETCH(k0 + BK);
#pragma unroll
        for (int bl = 0; bl < NB; ++bl) {
            w4q_v2i a0[TM], a1[TM], b0[TN], b1[TN];
            float sx[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const uint8_t *p = &As[((wm * TM + i) * 16 + lane % 16) * LDB + bl * 16];
                a0[i] = *(const w4q_v2i *)p;
                a1[i] = *(const w4q_v2i *)(p + 8);
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const uint8_t *p = &Bs[((wn * TN + j) * 16 + lane % 16) * LDB + bl * 16];
                b0[j] = *(const w4q_v2i *)p;
                b1[j] = *(const w4q_v2i *)(p + 8);
                sx[j] = Bss[bl][(wn * TN + j) * 16 + lane % 16];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float4 w0 = *(const float4 *)&Ass[bl][(wm * TM + i) * 16 + (lane / 16) * 8];
                const float4 w1 = *(const float4 *)&Ass[bl][(wm * TM + i) * 16 + (lane / 16) * 8 + 4];
                const float sw[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    w4q_v8i c = (w4q_v8i){0, 0, 0, 0, 0, 0, 0, 0};
                    c = W4Q_WMMA_I4(a0[i], b0[j], c);
                    c = W4Q_WMMA_I4(a1[i], b1[j], c);
#pragma unroll
                    for (int v = 0; v < 8; ++v) facc[i][j][v] = __builtin_fmaf((float)c[v], sw[v] * sx[j], facc[i][j][v]);
                }
            }
        }
    }
#undef W4Q_FETCH
    float *Y = (float *)Yv;
    _Float16 *Yh = (_Float16 *)Yv;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int t = t0 + (wn * TN + j) * 16 + lane % 16;
            if (t >= n_tok) continue;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + CROW(lane, v);
                const size_t o = (size_t)t * nrows + row;
                if (h16)
                    Yh[o] = (_Float16)(acc ? (float)Yh[o] + facc[i][j][v] : facc[i][j][v]);
                else
                    Y[o] = acc ? Y[o] + facc[i][j][v] : facc[i][j][v];
            }
        }
}

// ---- round 2 (v0252-w4a4p r2: WHIRL_W4A4=2 / 3, the LUT form; no converted weight copy)
//   w4q_quant_xm(X, Xq, Xs, nblk): as w4q_quant_x, the bytes in the MXFP4 order (byte j = code j | code j + 16 << 4: w4q_xblock_m).
//   w4q_gemm_m(W, row_bytes, gain, Xq, Xs, Y, ncols, nrows, n_tok, acc, h16): w4q_gemm's tiles, stages, fold order and
//     epilogue, but the A side reads the MXFP4 matrix itself: per stage 64 code bytes per row (8-byte loads), remapped to int4
//     levels at the LDS store (w4q_lvl_word: v_perm table, nibble positions kept), and the E8M0 byte per 32 turned into
//     w4q_mscale(e, gain[row]) = the scale round 1 stores. With MXFP4-order activations every 32-block sum is the same exact
//     int32 as round 1's, so the output is BITWISE w4q_gemm's on the round-1 converted copy (kt rows). ncols % 256 == 0.
extern "C" __global__ void __launch_bounds__(256) w4q_quant_xm(const _Float16 *X, uint8_t *Xq, float *Xs, uint64_t nblk) {
    for (uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; g < nblk; g += (uint64_t)gridDim.x * blockDim.x) {
        float x[32];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half8 h = *(const half8 *)(X + g * 32 + i * 8);
#pragma unroll
            for (int j = 0; j < 8; ++j) x[i * 8 + j] = (float)h[j];
        }
        uint8_t q[16];
        float d;
        w4q_xblock_m(x, q, &d);
        uint32_t u[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) u[j] = (uint32_t)q[4 * j] | ((uint32_t)q[4 * j + 1] << 8) | ((uint32_t)q[4 * j + 2] << 16) | ((uint32_t)q[4 * j + 3] << 24);
        *(uint4 *)(Xq + g * 16) = make_uint4(u[0], u[1], u[2], u[3]);
        Xs[g] = d;
    }
}

extern "C" __global__ void __launch_bounds__(256) w4q_gemm_m(const uint8_t *__restrict__ W, uint64_t row_bytes, const float *__restrict__ gain,
                                                             const uint8_t *__restrict__ Xq, const float *__restrict__ Xs, void *__restrict__ Yv, int ncols,
                                                             int nrows, int n_tok, int acc, int h16) {
    constexpr int BM = 128, BN = 128, WAVES_M = 4, NTH = 256, WAVES_N = NTH / WARP / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int BK = 128, BKB = BK / 2, NB = BK / 32;
    constexpr int LDB = BKB + 16;
    constexpr int A_V = BM * BKB / 8 / NTH, B_V = BN * BKB / 16 / NTH;  // A: 8-byte loads (an MXFP4 sub-block is 8-aligned)
    constexpr int A_S = BM * NB / NTH, B_S = BN * NB / NTH;
    static_assert(A_V >= 1 && B_V >= 1 && A_S >= 1 && B_S >= 1 && TM >= 1 && TN >= 1, "tile/threads");
    __shared__ __attribute__((aligned(16))) uint8_t As[BM * LDB];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[BN * LDB];
    __shared__ __attribute__((aligned(16))) float Ass[NB][BM];
    __shared__ __attribute__((aligned(16))) float Bss[NB][BN];

    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    const int m0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
    const size_t rbytes = (size_t)ncols / 2, rsc = (size_t)ncols / 32;

    w4q_v2i ra[A_V];
    w4q_v4i rb[B_V];
    uint32_t ea[A_S];
    float ga[A_S], sb[B_S];
#pragma unroll
    for (int e = 0; e < A_S; ++e) ga[e] = gain[m0 + (threadIdx.x + e * NTH) / 4];  // the row of a scale slot is fixed per thread
    // k0_ % 256 is 0 or 128: the stage is the first or second half of one 136-byte MXFP4 block (8 E8M0 + 8 x 16 code bytes)
#define W4QM_FETCH(k0_)                                                                                                         \
    do {                                                                                                                        \
        const size_t blk0 = (size_t)((k0_) >> 8) * 136;                                                                          \
        const int ib0 = ((k0_) >> 5) & 7;                                                                                       \
        _Pragma("unroll") for (int e = 0; e < A_V; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH, c = s % 8;                                                                     \
            ra[e] = *(const w4q_v2i *)(W + (size_t)(m0 + s / 8) * row_bytes + blk0 + 8 + (ib0 + c / 2) * 16 + (c % 2) * 8);      \
        }                                                                                                                       \
        _Pragma("unroll") for (int e = 0; e < B_V; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH;                                                                                \
            rb[e] = t0 + s / 4 < n_tok ? *(const w4q_v4i *)(Xq + (size_t)(t0 + s / 4) * rbytes + (k0_) / 2 + (s % 4) * 16)      \
                                       : (w4q_v4i){0, 0, 0, 0};                                                                 \
        }                                                                                                                       \
        _Pragma("unroll") for (int e = 0; e < A_S; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH;                                                                                \
            ea[e] = W[(size_t)(m0 + s / 4) * row_bytes + blk0 + ib0 + s % 4];                                                   \
        }                                                                                                                       \
        _Pragma("unroll") for (int e = 0; e < B_S; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH;                                                                                \
            sb[e] = t0 + s / 4 < n_tok ? Xs[(size_t)(t0 + s / 4) * rsc + (k0_) / 32 + s % 4] : 0.f;                             \
        }                                                                                                                       \
    } while (0)

    float8 facc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) facc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    W4QM_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_V; ++e) {
            const int s = threadIdx.x + e * NTH;
            const w4q_v2i lv = (w4q_v2i){(int)w4q_lvl_word((uint32_t)ra[e].x), (int)w4q_lvl_word((uint32_t)ra[e].y)};
            *(w4q_v2i *)&As[(s / 8) * LDB + (s % 8) * 8] = lv;
        }
#pragma unroll
        for (int e = 0; e < B_V; ++e) {
            const int s = threadIdx.x + e * NTH;
            *(w4q_v4i *)&Bs[(s / 4) * LDB + (s % 4) * 16] = rb[e];
        }
#pragma unroll
        for (int e = 0; e < A_S; ++e) {
            const int s = threadIdx.x + e * NTH, r = s / 4, rr = r & 15;
            Ass[s % 4][(r & ~15) + (rr & 1) * 8 + rr / 2] = w4q_mscale(ea[e], ga[e]);  // CROW order inside each 16-row tile
        }
#pragma unroll
        for (int e = 0; e < B_S; ++e) {
            const int s = threadIdx.x + e * NTH;
            Bss[s % 4][s / 4] = sb[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) W4QM_FETCH(k0 + BK);
#pragma unroll
        for (int bl = 0; bl < NB; ++bl) {
            w4q_v2i a0[TM], a1[TM], b0[TN], b1[TN];
            float sx[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const uint8_t *p = &As[((wm * TM + i) * 16 + lane % 16) * LDB + bl * 16];
                a0[i] = *(const w4q_v2i *)p;
                a1[i] = *(const w4q_v2i *)(p + 8);
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const uint8_t *p = &Bs[((wn * TN + j) * 16 + lane % 16) * LDB + bl * 16];
                b0[j] = *(const w4q_v2i *)p;
                b1[j] = *(const w4q_v2i *)(p + 8);
                sx[j] = Bss[bl][(wn * TN + j) * 16 + lane % 16];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float4 w0 = *(const float4 *)&Ass[bl][(wm * TM + i) * 16 + (lane / 16) * 8];
                const float4 w1 = *(const float4 *)&Ass[bl][(wm * TM + i) * 16 + (lane / 16) * 8 + 4];
                const float sw[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    w4q_v8i c = (w4q_v8i){0, 0, 0, 0, 0, 0, 0, 0};
                    c = W4Q_WMMA_I4(a0[i], b0[j], c);
                    c = W4Q_WMMA_I4(a1[i], b1[j], c);
#pragma unroll
                    for (int v = 0; v < 8; ++v) facc[i][j][v] = __builtin_fmaf((float)c[v], sw[v] * sx[j], facc[i][j][v]);
                }
            }
        }
    }
#undef W4QM_FETCH
    float *Y = (float *)Yv;
    _Float16 *Yh = (_Float16 *)Yv;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int t = t0 + (wn * TN + j) * 16 + lane % 16;
            if (t >= n_tok) continue;  // token tail of the LUT form
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + CROW(lane, v);
                const size_t o = (size_t)t * nrows + row;
                if (h16)
                    Yh[o] = (_Float16)(acc ? (float)Yh[o] + facc[i][j][v] : facc[i][j][v]);
                else
                    Y[o] = acc ? Y[o] + facc[i][j][v] : facc[i][j][v];
            }
        }
}

// round 2b: w4q_gemm_m2 = w4q_gemm_m's body on 512 threads x 256 tokens (4 waves along N: the same per-wave tiles and fold,
// so the same bits; each remapped weight stage serves twice the tokens). Kernel probe: +25 % time vs the copy (slower than
// w4q_gemm_m's +5-6 %): kept selectable as WHIRL_W4A4=4, not the screen arm.
template <int NTH_, int BN_>
__device__ __forceinline__ void w4q_gemm_m_body(const uint8_t *__restrict__ W, uint64_t row_bytes, const float *__restrict__ gain,
                                                const uint8_t *__restrict__ Xq, const float *__restrict__ Xs, void *__restrict__ Yv, int ncols, int nrows,
                                                int n_tok, int acc, int h16) {
    constexpr int BM = 128, BN = BN_, WAVES_M = 4, NTH = NTH_, WAVES_N = NTH / WARP / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int BK = 128, BKB = BK / 2, NB = BK / 32;
    constexpr int LDB = BKB + 16;
    constexpr int A_V = BM * BKB / 8 / NTH, B_V = BN * BKB / 16 / NTH;  // A: 8-byte loads (an MXFP4 sub-block is 8-aligned)
    constexpr int A_S = BM * NB / NTH, B_S = BN * NB / NTH;
    static_assert(A_V >= 1 && B_V >= 1 && A_S >= 1 && B_S >= 1 && TM >= 1 && TN >= 1, "tile/threads");
    __shared__ __attribute__((aligned(16))) uint8_t As[BM * LDB];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[BN * LDB];
    __shared__ __attribute__((aligned(16))) float Ass[NB][BM];
    __shared__ __attribute__((aligned(16))) float Bss[NB][BN];

    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    const int m0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
    const size_t rbytes = (size_t)ncols / 2, rsc = (size_t)ncols / 32;

    w4q_v2i ra[A_V];
    w4q_v4i rb[B_V];
    uint32_t ea[A_S];
    float gr[A_S], sb[B_S];
#pragma unroll
    for (int e = 0; e < A_S; ++e) gr[e] = gain[m0 + (threadIdx.x + e * NTH) / 4];  // the row of a scale slot is fixed per thread
    // k0_ % 256 is 0 or 128: the stage is the first or second half of one 136-byte MXFP4 block (8 E8M0 + 8 x 16 code bytes)
#define W4QM2_FETCH(k0_)                                                                                                         \
    do {                                                                                                                        \
        const size_t blk0 = (size_t)((k0_) >> 8) * 136;                                                                          \
        const int ib0 = ((k0_) >> 5) & 7;                                                                                       \
        _Pragma("unroll") for (int e = 0; e < A_V; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH, c = s % 8;                                                                     \
            ra[e] = *(const w4q_v2i *)(W + (size_t)(m0 + s / 8) * row_bytes + blk0 + 8 + (ib0 + c / 2) * 16 + (c % 2) * 8);      \
        }                                                                                                                       \
        _Pragma("unroll") for (int e = 0; e < B_V; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH;                                                                                \
            rb[e] = t0 + s / 4 < n_tok ? *(const w4q_v4i *)(Xq + (size_t)(t0 + s / 4) * rbytes + (k0_) / 2 + (s % 4) * 16)      \
                                       : (w4q_v4i){0, 0, 0, 0};                                                                 \
        }                                                                                                                       \
        _Pragma("unroll") for (int e = 0; e < A_S; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH;                                                                                \
            ea[e] = W[(size_t)(m0 + s / 4) * row_bytes + blk0 + ib0 + s % 4];                                                   \
        }                                                                                                                       \
        _Pragma("unroll") for (int e = 0; e < B_S; ++e) {                                                                       \
            const int s = threadIdx.x + e * NTH;                                                                                \
            sb[e] = t0 + s / 4 < n_tok ? Xs[(size_t)(t0 + s / 4) * rsc + (k0_) / 32 + s % 4] : 0.f;                             \
        }                                                                                                                       \
    } while (0)

    float8 facc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) facc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    W4QM2_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_V; ++e) {
            const int s = threadIdx.x + e * NTH;
            const w4q_v2i lv = (w4q_v2i){(int)w4q_lvl_word((uint32_t)ra[e].x), (int)w4q_lvl_word((uint32_t)ra[e].y)};
            *(w4q_v2i *)&As[(s / 8) * LDB + (s % 8) * 8] = lv;
        }
#pragma unroll
        for (int e = 0; e < B_V; ++e) {
            const int s = threadIdx.x + e * NTH;
            *(w4q_v4i *)&Bs[(s / 4) * LDB + (s % 4) * 16] = rb[e];
        }
#pragma unroll
        for (int e = 0; e < A_S; ++e) {
            const int s = threadIdx.x + e * NTH, r = s / 4, rr = r & 15;
            Ass[s % 4][(r & ~15) + (rr & 1) * 8 + rr / 2] = w4q_mscale(ea[e], gr[e]);  // CROW order inside each 16-row tile
        }
#pragma unroll
        for (int e = 0; e < B_S; ++e) {
            const int s = threadIdx.x + e * NTH;
            Bss[s % 4][s / 4] = sb[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) W4QM2_FETCH(k0 + BK);
#pragma unroll
        for (int bl = 0; bl < NB; ++bl) {
            w4q_v2i a0[TM], a1[TM], b0[TN], b1[TN];
            float sx[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const uint8_t *p = &As[((wm * TM + i) * 16 + lane % 16) * LDB + bl * 16];
                a0[i] = *(const w4q_v2i *)p;
                a1[i] = *(const w4q_v2i *)(p + 8);
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const uint8_t *p = &Bs[((wn * TN + j) * 16 + lane % 16) * LDB + bl * 16];
                b0[j] = *(const w4q_v2i *)p;
                b1[j] = *(const w4q_v2i *)(p + 8);
                sx[j] = Bss[bl][(wn * TN + j) * 16 + lane % 16];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float4 w0 = *(const float4 *)&Ass[bl][(wm * TM + i) * 16 + (lane / 16) * 8];
                const float4 w1 = *(const float4 *)&Ass[bl][(wm * TM + i) * 16 + (lane / 16) * 8 + 4];
                const float sw[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    w4q_v8i c = (w4q_v8i){0, 0, 0, 0, 0, 0, 0, 0};
                    c = W4Q_WMMA_I4(a0[i], b0[j], c);
                    c = W4Q_WMMA_I4(a1[i], b1[j], c);
#pragma unroll
                    for (int v = 0; v < 8; ++v) facc[i][j][v] = __builtin_fmaf((float)c[v], sw[v] * sx[j], facc[i][j][v]);
                }
            }
        }
    }
#undef W4QM2_FETCH
    float *Y = (float *)Yv;
    _Float16 *Yh = (_Float16 *)Yv;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int t = t0 + (wn * TN + j) * 16 + lane % 16;
            if (t >= n_tok) continue;  // token tail (256-token tile)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + CROW(lane, v);
                const size_t o = (size_t)t * nrows + row;
                if (h16)
                    Yh[o] = (_Float16)(acc ? (float)Yh[o] + facc[i][j][v] : facc[i][j][v]);
                else
                    Y[o] = acc ? Y[o] + facc[i][j][v] : facc[i][j][v];
            }
        }
}

// grid (ceil(n_tok / 256), nrows / 128) x 512 threads
extern "C" __global__ void __launch_bounds__(512) w4q_gemm_m2(const uint8_t *__restrict__ W, uint64_t row_bytes, const float *__restrict__ gain,
                                                              const uint8_t *__restrict__ Xq, const float *__restrict__ Xs, void *__restrict__ Yv, int ncols,
                                                              int nrows, int n_tok, int acc, int h16) {
    w4q_gemm_m_body<512, 256>(W, row_bytes, gain, Xq, Xs, Yv, ncols, nrows, n_tok, acc, h16);
}
