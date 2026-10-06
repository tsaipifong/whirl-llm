// WHIRL device code (gfx1151 / RDNA 3.5): small-batch prefill GEMM on raw quantized bytes (gemms*).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// gemms_c<i>_<type> (f32 out, optional accumulate) / gemmsh_c<i>_<type> (f16 out) for
// short prompts (n up to a few hundred tokens). Numerics are those of gemm3_impl: every
// weight fragment is dec_group_h<T> (lane: row lane % 16, all 16 k of the step), the
// activations are the same f16 values, and each accumulator runs one WMMA per 16-k step
// in increasing k from zero -> bitwise == gemm_c<i>_<type> for every row and token.
//
// What differs is the data flow. Per 256-k stage the block copies the RAW quantized
// bytes of its BM rows into LDS (a 16-byte aligned copy keeps every super-block's
// global alignment mod 16, so dec_group_h reads the same bytes the same way) together
// with an X slab (BN tokens x XK k, f16); the next stage is prefetched into registers
// while the current one is multiplied. Each wave decodes its own row tile from LDS:
// lanes 0-15 decode k 0..7 and lanes 16-31 k 8..15 of the step, joined with one
// permlanex16 per dword. Waves are laid out 8 x 1 (BM = 128: every wave owns one row
// tile and all BN / 16 token tiles, so no fragment is decoded twice) or 4 x 2 (BM = 64).
// The weights are the WMMA B operand (D lane = weight row, element v = token pair), so
// the stores of a row's tokens are contiguous; A / B order does not change the bits.
//
// gfx11 has no gfx12-style 8-k-per-lane fragments, and the gemm_c tiles (BN 128 / 256)
// waste most of their work and bandwidth on short batches: measured on the Radeon
// 8060S, Q4_K 17408 x 5120 at 88 tokens 1.18 ms (gemm_c3) -> 0.79 ms (128 x 96 here),
// at 32 tokens 1.10 -> 0.26 ms.
//
// The block geometry of each slot is exported in whirl_gemms_geom (BM, BN, threads per
// slot); the host reads it instead of the gfx1201 table (kernels::KernelTable::gemms_geom).
// ---------------------------------------------------------------------------
template <int T> constexpr int gs_sbb() { return 256 / Q<T>::blk * Q<T>::bytes; }
typedef unsigned int gs_u32x4 __attribute__((ext_vector_type(4)));

template <int T, int BM, int BN, int WAVES_M, int XK, bool OH>
__device__ __forceinline__ void gemms11_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                                             float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int NTH = 256;
    constexpr int SB = gs_sbb<T>();
    constexpr int CH = (SB + 15) / 16 + 1;  // 16-byte chunks per row stage (covers the misaligned start)
    constexpr int SLOT = CH * 16;
    constexpr int WCH = BM * CH;
    constexpr int WPER = (WCH + NTH - 1) / NTH;
    constexpr int XLD = XK + 8;
    constexpr int XC = XK / 8;  // 16-byte chunks per token per X stage
    constexpr int XCH = BN * XC;
    static_assert(XCH % NTH == 0, "x chunks");
    constexpr int XPER = XCH / NTH;
    constexpr int NH = 256 / XK;  // X stages per W stage
    constexpr int KS = XK / 16;   // k-steps per X stage
    constexpr int WAVES_N = NTH / 32 / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int KU = TN >= 8 || SB > 256 ? 1 : 2;  // k-step unroll (register pressure)
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
    gs_u32x4 rw[WPER];
    half8 rx[XPER];
    // (macros, not lambdas: a by-reference capture of rw / rx pushes them to scratch)
#define GEMMS11_FETCHW(js_)                                                                         \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < WPER; ++e) {                                          \
            const int idx = tid + e * NTH;                                                          \
            if (WCH % NTH == 0 || idx < WCH) {                                                      \
                const int r = idx / CH, c = idx % CH;                                               \
                const int row = min(m0 + r, nrows - 1);                                             \
                const size_t st = (size_t)row * row_bytes + (size_t)(js_) * SB;                     \
                const size_t a = (st & ~(size_t)15) + c * 16;                                       \
                /* no 16-byte chunk wholly past the stage (the matrix may end there) */             \
                if (a < st + SB) rw[e] = *(const gs_u32x4 *)(W + a);                                \
            }                                                                                       \
        }                                                                                           \
    } while (0)
#define GEMMS11_FETCHX(hs_)                                                                         \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < XPER; ++e) {                                          \
            const int idx = tid + e * NTH;                                                          \
            const int t = idx / XC, c = idx % XC;                                                   \
            if (t0 + t < n_tok) rx[e] = *(const half8 *)(X + (size_t)(t0 + t) * ncols + (hs_) * XK + c * 8); \
            else _Pragma("unroll") for (int q = 0; q < 8; ++q) rx[e][q] = (_Float16)0.f;           \
        }                                                                                           \
    } while (0)
    GEMMS11_FETCHW(0);
    GEMMS11_FETCHX(0);
    for (int js = 0; js < nst; ++js) {
#pragma unroll 1
        for (int hf = 0; hf < NH; ++hf) {
            __syncthreads();
            if (hf == 0) {
#pragma unroll
                for (int e = 0; e < WPER; ++e) {
                    const int idx = tid + e * NTH;
                    if (WCH % NTH == 0 || idx < WCH) *(gs_u32x4 *)&Ws[(idx / CH) * SLOT + (idx % CH) * 16] = rw[e];
                }
            }
#pragma unroll
            for (int e = 0; e < XPER; ++e) {
                const int idx = tid + e * NTH;
                *(half8 *)&Xs[(idx / XC) * XLD + (idx % XC) * 8] = rx[e];
            }
            __syncthreads();
            if (hf == 0 && js + 1 < nst) GEMMS11_FETCHW(js + 1);
            if (js * NH + hf + 1 < nst * NH) GEMMS11_FETCHX(js * NH + hf + 1);
            const uint8_t *wp[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int rl = (wm * TM + i) * 16 + lane % 16;
                const int row = min(m0 + rl, nrows - 1);
                wp[i] = Ws + rl * SLOT + (int)(((size_t)row * row_bytes + (size_t)js * SB) & 15);
            }
            // (partial unroll: a full unroll lets the scheduler hoist every step's LDS reads and spill)
#pragma unroll KU
            for (int k8 = 0; k8 < KS; ++k8) {
                const int kk = hf * KS + k8;
                half16 a[TM], b[TN];
#pragma unroll
                for (int i = 0; i < TM; ++i) a[i] = join_halves(dec_group_h<T>(wp[i], 2 * kk + lane / 16), lane);
#pragma unroll
                for (int j = 0; j < TN; ++j) b[j] = ld_frag16(&Xs[((wn * TN + j) * 16 + lane % 16) * XLD + k8 * 16]);
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] = WMMA16(b[j], a[i], acc[i][j]);
            }
        }
    }
#undef GEMMS11_FETCHW
#undef GEMMS11_FETCHX
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = m0 + (wm * TM + i) * 16 + lane % 16;
        if (row >= nrows) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int t = t0 + (wn * TN + j) * 16 + CROW(lane, v);
                if (t < n_tok) {
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

// Slots (BM, BN, threads; WAVES_M, X stage k). Must match whirl_gemms_geom below.
//   c0 128 x  32 (8, 256)   c1 128 x  48 (8, 256)   c2 128 x  64 (8, 256)   c3 128 x 96 (8, 128)
//   c4 128 x 128 (8, 128)   c5  64 x  32 (4, 256)   c6 128 x  64 (8, 128)   c7  64 x 64 (4, 256)
// q8_0 has no c2 / c4 (LDS: 288-byte row stages); f32 / f16 weights have no gemms entries.
extern "C" __global__ void whirl_cap_gemms_geom() {}
extern "C" __device__ int whirl_gemms_geom[24] = {
    128, 32, 256, 128, 48, 256, 128, 64, 256, 128, 96, 256, 128, 128, 256, 64, 32, 256, 128, 64, 256, 64, 64, 256,
};

#define GEMMS11_ENTRY(CI, SUF, T, BM, BN, WM, XK)                                                                    \
    extern "C" __global__ void __launch_bounds__(256)                                                                \
        gemms_c##CI##_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, \
                            int n_tok, int accumulate) {                                                             \
        gemms11_impl<T, BM, BN, WM, XK, false>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                 \
    }                                                                                                                \
    extern "C" __global__ void __launch_bounds__(256)                                                                \
        gemmsh_c##CI##_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, \
                             int n_tok, int accumulate) {                                                            \
        gemms11_impl<T, BM, BN, WM, XK, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                  \
    }
#define GEMMS11_SLOTS_A(SUF, T)                     \
    GEMMS11_ENTRY(0, SUF, T, 128, 32, 8, 256)       \
    GEMMS11_ENTRY(1, SUF, T, 128, 48, 8, 256)       \
    GEMMS11_ENTRY(3, SUF, T, 128, 96, 8, 128)       \
    GEMMS11_ENTRY(5, SUF, T, 64, 32, 4, 256)        \
    GEMMS11_ENTRY(6, SUF, T, 128, 64, 8, 128)       \
    GEMMS11_ENTRY(7, SUF, T, 64, 64, 4, 256)
#define GEMMS11_SLOTS(SUF, T)  \
    GEMMS11_SLOTS_A(SUF, T)    \
    GEMMS11_ENTRY(2, SUF, T, 128, 64, 8, 256) \
    GEMMS11_ENTRY(4, SUF, T, 128, 128, 8, 128)

GEMMS11_SLOTS_A(q8_0, QT_Q8_0)
GEMMS11_SLOTS(q3_k, QT_Q3_K)
GEMMS11_SLOTS(q4_k, QT_Q4_K)
GEMMS11_SLOTS(q5_k, QT_Q5_K)
GEMMS11_SLOTS(q6_k, QT_Q6_K)
GEMMS11_SLOTS(iq4_nl, QT_IQ4_NL)
GEMMS11_SLOTS(iq3_s, QT_IQ3_S)
GEMMS11_SLOTS(iq4_xs, QT_IQ4_XS)
GEMMS11_SLOTS(mxfp4, QT_MXFP4)
