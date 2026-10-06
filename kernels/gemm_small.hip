// WHIRL device code (gfx1201 / RDNA 4): small-batch prefill GEMM on raw quantized bytes (gemms*, gemmsd*).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---- item T2: small-batch prefill GEMM (begin) ----
// gemms_c{i}_<type> (f32 out, optional accumulate) / gemmsh_c{i}_<type> (f16 out): the
// gemm3_impl numerics for short prompts (n <= ~256): every A fragment is dec_group_h<T>
// (lane: row lane%16, k 8*(lane/16)..+8), B = the same f16 activations, and each
// accumulator runs wmma(a, b, acc) over k in increasing 16-steps -> bitwise == gemm_cN_<t>.
// What differs is the staging: 256-k stages of RAW quantized bytes in LDS (a 16-byte
// aligned copy keeps every super-block's global alignment mod 16, so dec_group_h reads the
// same bytes the same way) plus the X slab (BN tokens x 256 k) in LDS, next stage
// prefetched into registers; token tiles of 16..64 instead of 128/256 (no padding work).
template <int T> constexpr int t2_sbb() { return 256 / Q<T>::blk * Q<T>::bytes; }
typedef unsigned int t2_u32x4 __attribute__((ext_vector_type(4)));
#include "gemm_small_addr.h"  // CH and which W chunks a stage loads (shared with the unit tests)

template <int T, int BM, int BN, int NTH, int WAVES_M, bool OH>
__device__ __forceinline__ void gemms_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                                           float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int SB = t2_sbb<T>();
    constexpr int CH = gemms_stage_chunks(SB);  // 16-byte chunks per row stage (covers the misaligned start)
    constexpr int SLOT = CH * 16;
    constexpr int WCH = BM * CH;
    constexpr int WPER = (WCH + NTH - 1) / NTH;
    constexpr int XLD = 256 + 8;
    constexpr int XCH = BN * 32;
    static_assert(XCH % NTH == 0, "x chunks");
    constexpr int XPER = XCH / NTH;
    constexpr int WAVES_N = NTH / 32 / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    static_assert(TM >= 1 && TN >= 1 && BM % (16 * WAVES_M) == 0 && BN % (16 * WAVES_N) == 0, "tile");
    __shared__ __attribute__((aligned(16))) uint8_t Ws[BM * SLOT];
    __shared__ __attribute__((aligned(16))) _Float16 Xs[BN * XLD];
    const int tid = threadIdx.x, lane = tid % WARP, wid = tid / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    const int m0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
    const int nst = ncols / 256;
    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    t2_u32x4 rw[WPER];
    half8 rx[XPER];
#define GEMMS_FETCH(js)                                                                             \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < WPER; ++e) {                                          \
            const int idx = tid + e * NTH;                                                          \
            if (WCH % NTH == 0 || idx < WCH) {                                                      \
                const int r = idx / CH, c = idx % CH;                                               \
                const int row = min(m0 + r, nrows - 1);                                             \
                const size_t st = (size_t)row * row_bytes + (size_t)(js) * SB;                      \
                /* only chunks overlapping the stage: the matrix may end right after it */          \
                rw[e] = gemms_chunk_needed(st, c, SB) ? *(const t2_u32x4 *)(W + (st & ~(size_t)15) + c * 16) : (t2_u32x4){0, 0, 0, 0}; \
            }                                                                                       \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < XPER; ++e) {                                          \
            const int idx = tid + e * NTH;                                                          \
            const int t = idx / 32, c = idx % 32;                                                   \
            if (t0 + t < n_tok) rx[e] = *(const half8 *)(X + (size_t)(t0 + t) * ncols + (js) * 256 + c * 8); \
            else _Pragma("unroll") for (int q = 0; q < 8; ++q) rx[e][q] = (_Float16)0.f;           \
        }                                                                                           \
    } while (0)
    GEMMS_FETCH(0);
    for (int js = 0; js < nst; ++js) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < WPER; ++e) {
            const int idx = tid + e * NTH;
            if (WCH % NTH == 0 || idx < WCH) *(t2_u32x4 *)&Ws[(idx / CH) * SLOT + (idx % CH) * 16] = rw[e];
        }
#pragma unroll
        for (int e = 0; e < XPER; ++e) {
            const int idx = tid + e * NTH;
            *(half8 *)&Xs[(idx / 32) * XLD + (idx % 32) * 8] = rx[e];
        }
        __syncthreads();
        if (js + 1 < nst) GEMMS_FETCH(js + 1);
        const uint8_t *wp[TM];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int rl = (wm * TM + i) * 16 + lane % 16;
            const int row = min(m0 + rl, nrows - 1);
            wp[i] = Ws + rl * SLOT + (int)(((size_t)row * row_bytes + (size_t)js * SB) & 15);
        }
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            half8 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = dec_group_h<T>(wp[i], 2 * kk + lane / 16);
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = *(const half8 *)&Xs[((wn * TN + j) * 16 + lane % 16) * XLD + kk * 16 + 8 * (lane / 16)];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
        }
    }
#undef GEMMS_FETCH
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int t = t0 + (wn * TN + j) * 16 + lane % 16;
            if (t >= n_tok) continue;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + 8 * (lane / 16) + v;
                if (row < nrows) {
                    if (OH) {
                        ((_Float16 *)Y)[(size_t)t * nrows + row] = (_Float16)acc[i][j][v];
                    } else {
                        float *dst = Y + (size_t)t * nrows + row;
                        *dst = accumulate ? *dst + acc[i][j][v] : acc[i][j][v];
                    }
                }
            }
        }
}

// gemmsd (item T2 v5, larger small batches): decode once per block: raw W 256-k stage in LDS -> cooperative dec_group_h into an f16 A
// tile (128-k half stages) -> WMMA from LDS (A and the X half slab), like gemm3 but with the
// raw global loads prefetched a whole stage ahead.
template <int T, int BM, int BN, int NTH, int WAVES_M, bool OH>
__device__ __forceinline__ void gemmsd_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                            float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int SB = t2_sbb<T>();
    constexpr int CH = gemms_stage_chunks(SB);
    constexpr int SLOT = CH * 16;
    constexpr int WCH = BM * CH;
    constexpr int WPER = (WCH + NTH - 1) / NTH;
    constexpr int LD = 128 + 8;
    constexpr int XCH = BN * 16;
    static_assert(XCH % NTH == 0, "x chunks");
    constexpr int XPER = XCH / NTH;
    constexpr int DG = BM * 16;  // 8-value groups per half stage
    static_assert(DG % NTH == 0, "dec groups");
    constexpr int DPER = DG / NTH;
    constexpr int WAVES_N = NTH / 32 / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    static_assert(TM >= 1 && TN >= 1 && BM % (16 * WAVES_M) == 0 && BN % (16 * WAVES_N) == 0, "tile");
    __shared__ __attribute__((aligned(16))) uint8_t Wr[BM * SLOT];
    __shared__ __attribute__((aligned(16))) _Float16 As[BM * LD];
    __shared__ __attribute__((aligned(16))) _Float16 Xs[BN * LD];
    const int tid = threadIdx.x, lane = tid % WARP, wid = tid / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    const int m0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
    const int nst = ncols / 256;
    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    t2_u32x4 rw[WPER];
    half8 rx0[XPER], rx1[XPER];
    uint32_t wst[WPER];
#pragma unroll
    for (int e = 0; e < WPER; ++e) {
        const int idx = min(tid + e * NTH, WCH - 1);
        wst[e] = (uint32_t)((size_t)min(m0 + idx / CH, nrows - 1) * row_bytes);
    }
    const _Float16 *xrow[XPER];
#pragma unroll
    for (int e = 0; e < XPER; ++e) {
        const int idx = tid + e * NTH;
        xrow[e] = X + (size_t)min(t0 + idx / 16, n_tok - 1) * ncols + (idx % 16) * 8;
    }
    auto fetch_w = [&](int js) {
#pragma unroll
        for (int e = 0; e < WPER; ++e) {
            const int idx = tid + e * NTH;
            if (WCH % NTH == 0 || idx < WCH) {
                const uint32_t st = wst[e] + (uint32_t)js * SB;
                // only chunks overlapping the stage: the matrix may end right after it
                rw[e] = gemms_chunk_needed(st, idx % CH, SB) ? *(const t2_u32x4 *)(W + (st & ~15u) + (idx % CH) * 16) : (t2_u32x4){0, 0, 0, 0};
            }
        }
    };
    auto fetch_x = [&](half8 *rx, int k0) {
#pragma unroll
        for (int e = 0; e < XPER; ++e) rx[e] = *(const half8 *)(xrow[e] + k0);
    };
    auto store_x = [&](const half8 *rx) {
#pragma unroll
        for (int e = 0; e < XPER; ++e) {
            const int idx = tid + e * NTH;
            *(half8 *)&Xs[(idx / 16) * LD + (idx % 16) * 8] = rx[e];
        }
    };
    auto decode = [&](int js, int h) {
#pragma unroll
        for (int e = 0; e < DPER; ++e) {
            const int idx = tid + e * NTH;
            const int r = idx / 16, g = idx % 16;
            const int row = min(m0 + r, nrows - 1);
            const uint8_t *p = Wr + r * SLOT + (int)(((size_t)row * row_bytes + (size_t)js * SB) & 15);
            *(half8 *)&As[r * LD + g * 8] = dec_group_h<T>(p, h * 16 + g);
        }
    };
    auto compute = [&]() {
#pragma unroll 2
        for (int kq = 0; kq < 8; ++kq) {
            half8 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = *(const half8 *)&As[((wm * TM + i) * 16 + lane % 16) * LD + kq * 16 + 8 * (lane / 16)];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = *(const half8 *)&Xs[((wn * TN + j) * 16 + lane % 16) * LD + kq * 16 + 8 * (lane / 16)];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
        }
    };
    fetch_w(0);
    fetch_x(rx0, 0);
    for (int js = 0; js < nst; ++js) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < WPER; ++e) {
            const int idx = tid + e * NTH;
            if (WCH % NTH == 0 || idx < WCH) *(t2_u32x4 *)&Wr[(idx / CH) * SLOT + (idx % CH) * 16] = rw[e];
        }
        store_x(rx0);
        __syncthreads();
        fetch_x(rx1, js * 256 + 128);
        if (js + 1 < nst) fetch_w(js + 1);
        decode(js, 0);
        __syncthreads();
        compute();
        __syncthreads();
        store_x(rx1);
        decode(js, 1);
        __syncthreads();
        if (js + 1 < nst) fetch_x(rx0, (js + 1) * 256);
        compute();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int t = t0 + (wn * TN + j) * 16 + lane % 16;
            if (t >= n_tok) continue;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + 8 * (lane / 16) + v;
                if (row < nrows) {
                    if (OH) {
                        ((_Float16 *)Y)[(size_t)t * nrows + row] = (_Float16)acc[i][j][v];
                    } else {
                        float *dst = Y + (size_t)t * nrows + row;
                        *dst = accumulate ? *dst + acc[i][j][v] : acc[i][j][v];
                    }
                }
            }
        }
}
extern "C" __global__ void __launch_bounds__(128) gemms_c0_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q8_0, 64, 32, 128, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemmsh_c0_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q8_0, 64, 32, 128, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemms_c0_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q3_K, 64, 32, 128, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemmsh_c0_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q3_K, 64, 32, 128, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemms_c0_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q4_K, 64, 32, 128, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemmsh_c0_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q4_K, 64, 32, 128, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemms_c0_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q5_K, 64, 32, 128, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemmsh_c0_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q5_K, 64, 32, 128, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemms_c0_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q6_K, 64, 32, 128, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemmsh_c0_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q6_K, 64, 32, 128, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemms_c0_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_NL, 64, 32, 128, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemmsh_c0_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_NL, 64, 32, 128, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemms_c0_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ3_S, 64, 32, 128, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemmsh_c0_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ3_S, 64, 32, 128, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemms_c0_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_XS, 64, 32, 128, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemmsh_c0_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_XS, 64, 32, 128, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemms_c0_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_MXFP4, 64, 32, 128, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(128) gemmsh_c0_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_MXFP4, 64, 32, 128, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c1_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q8_0, 128, 32, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c1_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q8_0, 128, 32, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c1_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q3_K, 128, 32, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c1_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q3_K, 128, 32, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c1_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q4_K, 128, 32, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c1_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q4_K, 128, 32, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c1_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q5_K, 128, 32, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c1_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q5_K, 128, 32, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c1_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q6_K, 128, 32, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c1_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q6_K, 128, 32, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c1_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_NL, 128, 32, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c1_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_NL, 128, 32, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c1_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ3_S, 128, 32, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c1_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ3_S, 128, 32, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c1_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_XS, 128, 32, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c1_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_XS, 128, 32, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c1_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_MXFP4, 128, 32, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c1_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_MXFP4, 128, 32, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c2_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q8_0, 128, 48, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c2_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q8_0, 128, 48, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c2_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q3_K, 128, 48, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c2_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q3_K, 128, 48, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c2_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q4_K, 128, 48, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c2_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q4_K, 128, 48, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c2_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q5_K, 128, 48, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c2_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q5_K, 128, 48, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c2_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q6_K, 128, 48, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c2_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q6_K, 128, 48, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c2_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_NL, 128, 48, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c2_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_NL, 128, 48, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c2_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ3_S, 128, 48, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c2_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ3_S, 128, 48, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c2_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_XS, 128, 48, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c2_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_XS, 128, 48, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c2_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_MXFP4, 128, 48, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c2_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_MXFP4, 128, 48, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c3_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q3_K, 128, 64, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c3_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q3_K, 128, 64, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c3_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q4_K, 128, 64, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c3_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q4_K, 128, 64, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c3_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q5_K, 128, 64, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c3_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q5_K, 128, 64, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c3_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q6_K, 128, 64, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c3_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q6_K, 128, 64, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c3_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_NL, 128, 64, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c3_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_NL, 128, 64, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c3_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ3_S, 128, 64, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c3_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ3_S, 128, 64, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c3_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_XS, 128, 64, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c3_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_XS, 128, 64, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c3_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_MXFP4, 128, 64, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c3_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_MXFP4, 128, 64, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c4_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q8_0, 128, 16, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c4_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q8_0, 128, 16, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c4_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q3_K, 128, 16, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c4_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q3_K, 128, 16, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c4_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q4_K, 128, 16, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c4_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q4_K, 128, 16, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c4_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q5_K, 128, 16, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c4_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q5_K, 128, 16, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c4_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q6_K, 128, 16, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c4_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_Q6_K, 128, 16, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c4_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_NL, 128, 16, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c4_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_NL, 128, 16, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c4_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ3_S, 128, 16, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c4_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ3_S, 128, 16, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c4_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_XS, 128, 16, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c4_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_IQ4_XS, 128, 16, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c4_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_MXFP4, 128, 16, 256, 8, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c4_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemms_impl<QT_MXFP4, 128, 16, 256, 8, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c5_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q8_0, 64, 64, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c5_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q8_0, 64, 64, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c5_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q3_K, 64, 64, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c5_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q3_K, 64, 64, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c5_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q4_K, 64, 64, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c5_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q4_K, 64, 64, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c5_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q5_K, 64, 64, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c5_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q5_K, 64, 64, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c5_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q6_K, 64, 64, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c5_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q6_K, 64, 64, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c5_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_NL, 64, 64, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c5_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_NL, 64, 64, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c5_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ3_S, 64, 64, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c5_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ3_S, 64, 64, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c5_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_XS, 64, 64, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c5_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_XS, 64, 64, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c5_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_MXFP4, 64, 64, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c5_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_MXFP4, 64, 64, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c6_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q8_0, 64, 96, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c6_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q8_0, 64, 96, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c6_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q3_K, 64, 96, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c6_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q3_K, 64, 96, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c6_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q4_K, 64, 96, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c6_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q4_K, 64, 96, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c6_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q5_K, 64, 96, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c6_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q5_K, 64, 96, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c6_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q6_K, 64, 96, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c6_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q6_K, 64, 96, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c6_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_NL, 64, 96, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c6_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_NL, 64, 96, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c6_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ3_S, 64, 96, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c6_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ3_S, 64, 96, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c6_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_XS, 64, 96, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c6_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_XS, 64, 96, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c6_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_MXFP4, 64, 96, 256, 4, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c6_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_MXFP4, 64, 96, 256, 4, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c7_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q3_K, 64, 128, 256, 2, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c7_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q3_K, 64, 128, 256, 2, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c7_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q4_K, 64, 128, 256, 2, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c7_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q4_K, 64, 128, 256, 2, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c7_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q5_K, 64, 128, 256, 2, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c7_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_Q5_K, 64, 128, 256, 2, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c7_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_NL, 64, 128, 256, 2, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c7_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_NL, 64, 128, 256, 2, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c7_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ3_S, 64, 128, 256, 2, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c7_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ3_S, 64, 128, 256, 2, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c7_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_XS, 64, 128, 256, 2, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c7_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_IQ4_XS, 64, 128, 256, 2, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemms_c7_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_MXFP4, 64, 128, 256, 2, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemmsh_c7_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemmsd_impl<QT_MXFP4, 64, 128, 256, 2, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
// ---- item T2: small-batch prefill GEMM (end) ----
