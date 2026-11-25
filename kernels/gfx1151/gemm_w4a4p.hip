// gfx1151 (Radeon 8060S) W4A4 SPEED PROBE (v0252-w4a4, PLAN V1-144 kill-screen K1): bench-only entries w4a4p_*, timed by
// whirl-kernel-test --bench w4a4p (tests/kernels/bench_kernels.cpp, CPU side tests/kernels/kt_w4a4p.h). NO product path:
// no host code outside the bench calls these entries, no selector, no default, no numerics change in any mode.
// SPDX-License-Identifier: Apache-2.0
//
// A fragment of the gfx1151 translation unit (kernels/gfx1151/whirl_kernels_gfx1151.hip includes it LAST, so no existing
// kernel moves; the code-object comparison of the shipping entries is in progress\v0252-w4a4.HANDOFF.md). Uses the set's
// helpers (half8, float8, CROW, WARP).
//
// The question: how fast is an int4 x int4 GEMM (iu4 WMMA, RDNA 3 / 3.5: v_wmma_i32_16x16x16_iu4 = 16 cycles per wave on a
// SIMD vs 32 for the f16 and iu8 16x16x16 WMMAs - AMD matrix instruction calculator, rdna3 table) at the Swift dense
// shapes, with and without the per-32 fold a W4A4 kernel over MXFP4-derived weights needs, against the shipped f16 GEMM
// (gemmfq_t2_mxfp4) in the same run?
// Entries:
//   w4a4p_gemm_i4   int4 W x int4 X, int32 accumulate over all of K, Y = float(acc)          (the iu4 ceiling, no scales)
//   w4a4p_gemm_i4f  the same with the per-32 fold: per K32 block c = 2 iu4 WMMAs into a zero int tile, then per element
//                   acc = fma(float(c), d_w * d_x, acc)                                       (what a real W4A4 kernel pays)
//   w4a4p_quant_x   the activation producer: f16 X -> packed int4 + f32 scale per 32 (amax / 7, rint, clamp -7..7)
//   w4a4p_reg_*     register-only WMMA rate probes (P1a style, research\w4a8\2026-11-07-f16-p1a-w4a8-gfx1151.md): f16, iu8,
//                   iu8 + fold, iu4, iu4 + fold; 8 independent accumulators per wave, operands from a random buffer
// GEMM geometry = gemmfq_t2's: 128 weight rows x 256 tokens per block, 256 threads (8 waves, 2 along M x 4 along N), 4 x 4
// WMMA tiles per wave; K stage 128 values (64 bytes per row = the bytes of gemmfq's BK 64 f16 stage), operands staged in LDS
// from a register prefetch of the next stage (gemmfq's load / compute overlap). Data: Wq [rows][K / 2] bytes, Ws [rows][K / 32]
// f32, Xq [tokens][K / 2], Xs [tokens][K / 32], Y [tokens][rows] f32 (gemmfq's layout). Requires rows % 128 == 0,
// tokens % 256 == 0, K % 128 == 0 (no edge code in a probe). Fragment: lane l holds row (A) / token (B) l % 16, 16 K values
// = 8 bytes (an int dot product does not depend on the K order inside a fragment as long as A and B share it).

typedef int w4p_v2i __attribute__((ext_vector_type(2)));
typedef int w4p_v4i __attribute__((ext_vector_type(4)));
typedef int w4p_v8i __attribute__((ext_vector_type(8)));
typedef _Float16 w4p_v16h __attribute__((ext_vector_type(16)));
typedef float w4p_v8f __attribute__((ext_vector_type(8)));
#define W4P_WMMA_I4(a_, b_, c_) __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, (a_), true, (b_), (c_), false)
#define W4P_WMMA_I8(a_, b_, c_) __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, (a_), true, (b_), (c_), false)

template <bool FOLD, int BM, int BN, int WAVES_M, int FG = 1>
__device__ void w4a4p_gemm_impl(const uint8_t *__restrict__ Wq, const float *__restrict__ Ws, const uint8_t *__restrict__ Xq,
                                const float *__restrict__ Xs, float *__restrict__ Y, int ncols, int nrows, int n_tok) {
    constexpr int NTH = 256, WAVES_N = NTH / WARP / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;  // WMMA tiles per wave
    constexpr int BK = 128, BKB = BK / 2, NB = BK / 32;            // 64 bytes, 4 blocks per stage
    constexpr int LDB = BKB + 16;                                  // 80-byte rows: 16-byte aligned stores, 8-byte fragment reads
    constexpr int A_V = BM * BKB / 16 / NTH, B_V = BN * BKB / 16 / NTH;  // uint4 per thread (BM / 64, BN / 64)
    constexpr int A_S = BM * NB / NTH, B_S = BN * NB / NTH;              // scales per thread (BM / 64, BN / 64)
    static_assert(A_V >= 1 && B_V >= 1 && A_S >= 1 && B_S >= 1 && TM >= 1 && TN >= 1 && BM % (16 * WAVES_M) == 0, "tile/threads");
    __shared__ __attribute__((aligned(16))) uint8_t As[BM * LDB];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[BN * LDB];
    // scales per stage: A rows in CROW order inside each 16-row tile (position (r & 1) * 8 + r / 2: lane half h reads 8
    // consecutive floats for its rows 2v + h), B tokens in order
    __shared__ __attribute__((aligned(16))) float Ass[NB][BM];
    __shared__ __attribute__((aligned(16))) float Bss[NB][BN];

    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    const int m0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
    const size_t rbytes = (size_t)ncols / 2, rsc = (size_t)ncols / 32;

    // slot s = threadIdx.x + e * NTH: row s / 4, 16-byte chunk s % 4 of the stage (scales: row s / 4, block s % 4)
    w4p_v4i ra[A_V], rb[B_V];  // ext vectors, not HIP uint4 structs (those arrays stayed in scratch: 96 B / lane)
    float sa[A_S], sb[B_S];
#define W4P_FETCH(k0_)                                                                                       \
    do {                                                                                                     \
        _Pragma("unroll") for (int e = 0; e < A_V; ++e) {                                                    \
            const int s = threadIdx.x + e * NTH;                                                             \
            ra[e] = *(const w4p_v4i *)(Wq + (size_t)(m0 + s / 4) * rbytes + (k0_) / 2 + (s % 4) * 16);         \
        }                                                                                                    \
        _Pragma("unroll") for (int e = 0; e < B_V; ++e) {                                                    \
            const int s = threadIdx.x + e * NTH;                                                             \
            rb[e] = *(const w4p_v4i *)(Xq + (size_t)(t0 + s / 4) * rbytes + (k0_) / 2 + (s % 4) * 16);         \
        }                                                                                                    \
        if constexpr (FOLD) {                                                                                \
            _Pragma("unroll") for (int e = 0; e < A_S; ++e) {                                                \
                const int s = threadIdx.x + e * NTH;                                                         \
                sa[e] = Ws[(size_t)(m0 + s / 4) * rsc + (k0_) / 32 + s % 4];                                 \
            }                                                                                                \
            _Pragma("unroll") for (int e = 0; e < B_S; ++e) {                                                \
                const int s = threadIdx.x + e * NTH;                                                         \
                sb[e] = Xs[(size_t)(t0 + s / 4) * rsc + (k0_) / 32 + s % 4];                                 \
            }                                                                                                \
        }                                                                                                    \
    } while (0)

    w4p_v8i iacc[TM][TN];
    float8 facc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if constexpr (FOLD)
                facc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
            else
                iacc[i][j] = (w4p_v8i){0, 0, 0, 0, 0, 0, 0, 0};
        }

    W4P_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_V; ++e) {
            const int s = threadIdx.x + e * NTH;
            *(w4p_v4i *)&As[(s / 4) * LDB + (s % 4) * 16] = ra[e];
        }
#pragma unroll
        for (int e = 0; e < B_V; ++e) {
            const int s = threadIdx.x + e * NTH;
            *(w4p_v4i *)&Bs[(s / 4) * LDB + (s % 4) * 16] = rb[e];
        }
        if constexpr (FOLD) {
#pragma unroll
            for (int e = 0; e < A_S; ++e) {
                const int s = threadIdx.x + e * NTH, r = s / 4, rr = r & 15;
                Ass[s % 4][(r & ~15) + (rr & 1) * 8 + rr / 2] = sa[e];
            }
#pragma unroll
            for (int e = 0; e < B_S; ++e) {
                const int s = threadIdx.x + e * NTH;
                Bss[s % 4][s / 4] = sb[e];
            }
        }
        __syncthreads();
        if (k0 + BK < ncols) W4P_FETCH(k0 + BK);
        if constexpr (!FOLD || FG == NB) {
            // no fold, or one fold per stage (FG == 4 blocks = 128 K: 8 iu4 WMMAs per int tile, then one fold with the scales
            // of the stage's first block - the hipfire X128 granularity an int4-native weight format would allow)
            w4p_v8i ct[TM][TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) ct[i][j] = FOLD ? (w4p_v8i){0, 0, 0, 0, 0, 0, 0, 0} : iacc[i][j];
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                w4p_v2i a[TM], b[TN];
#pragma unroll
                for (int i = 0; i < TM; ++i) a[i] = *(const w4p_v2i *)&As[((wm * TM + i) * 16 + lane % 16) * LDB + kk / 2];
#pragma unroll
                for (int j = 0; j < TN; ++j) b[j] = *(const w4p_v2i *)&Bs[((wn * TN + j) * 16 + lane % 16) * LDB + kk / 2];
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) ct[i][j] = W4P_WMMA_I4(a[i], b[j], ct[i][j]);
            }
            if constexpr (!FOLD) {
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) iacc[i][j] = ct[i][j];
            } else {
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const float4 w0 = *(const float4 *)&Ass[0][(wm * TM + i) * 16 + (lane / 16) * 8];
                    const float4 w1 = *(const float4 *)&Ass[0][(wm * TM + i) * 16 + (lane / 16) * 8 + 4];
                    const float sw[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const float sx = Bss[0][(wn * TN + j) * 16 + lane % 16];
#pragma unroll
                        for (int v = 0; v < 8; ++v) facc[i][j][v] = __builtin_fmaf((float)ct[i][j][v], sw[v] * sx, facc[i][j][v]);
                    }
                }
            }
        } else {
            static_assert(FG == 1, "fold every block (FG 1) or every stage (FG 4)");
#pragma unroll
            for (int bl = 0; bl < NB; ++bl) {
                w4p_v2i a0[TM], a1[TM], b0[TN], b1[TN];
                float sx[TN];
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const uint8_t *p = &As[((wm * TM + i) * 16 + lane % 16) * LDB + bl * 16];
                    a0[i] = *(const w4p_v2i *)p;
                    a1[i] = *(const w4p_v2i *)(p + 8);
                }
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const uint8_t *p = &Bs[((wn * TN + j) * 16 + lane % 16) * LDB + bl * 16];
                    b0[j] = *(const w4p_v2i *)p;
                    b1[j] = *(const w4p_v2i *)(p + 8);
                    sx[j] = Bss[bl][(wn * TN + j) * 16 + lane % 16];
                }
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const float4 w0 = *(const float4 *)&Ass[bl][(wm * TM + i) * 16 + (lane / 16) * 8];
                    const float4 w1 = *(const float4 *)&Ass[bl][(wm * TM + i) * 16 + (lane / 16) * 8 + 4];
                    const float sw[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        w4p_v8i c = (w4p_v8i){0, 0, 0, 0, 0, 0, 0, 0};
                        c = W4P_WMMA_I4(a0[i], b0[j], c);
                        c = W4P_WMMA_I4(a1[i], b1[j], c);
#pragma unroll
                        for (int v = 0; v < 8; ++v) facc[i][j][v] = __builtin_fmaf((float)c[v], sw[v] * sx[j], facc[i][j][v]);
                    }
                }
            }
        }
    }
#undef W4P_FETCH
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int t = t0 + (wn * TN + j) * 16 + lane % 16;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + CROW(lane, v);
                Y[(size_t)t * nrows + row] = FOLD ? facc[i][j][v] : (float)iacc[i][j][v];
            }
        }
}

// Entries (grid x = tokens / BN, y = rows / BM, 256 threads; the bench reads the tile from kt_w4a4p.h kGemms):
//   w4a4p_gemm_i4    128 x 256, WAVES_M 2, 4 x 4 tiles / wave (gemmfq_t2's geometry), no fold
//   w4a4p_gemm_i4m   64 x 256, WAVES_M 2, 2 x 4 tiles / wave, no fold (the fold variants' geometry, to split tile from fold)
//   w4a4p_gemm_i4f   64 x 256, WAVES_M 2, 2 x 4 tiles / wave, per-32 fold (4 x 4 with the fold state needs > 256 VGPRs: spills)
//   w4a4p_gemm_i4fb  128 x 128, WAVES_M 4, 2 x 4 tiles / wave, per-32 fold
//   w4a4p_gemm_i4g   64 x 256, WAVES_M 2, 2 x 4 tiles / wave, per-128 fold (scales of block 0 of each 128: NOT a form MXFP4
//                    weights allow - the ceiling of an int4-native weight format with groups of 128, hipfire's X128)
#define W4P_GEMM_ENTRY(name_, fold_, bm_, bn_, wm_, fg_)                                                                              \
    extern "C" __global__ void __launch_bounds__(256) name_(const uint8_t *Wq, const float *Ws, const uint8_t *Xq, const float *Xs, \
                                                            float *Y, int ncols, int nrows, int n_tok) {                            \
        w4a4p_gemm_impl<fold_, bm_, bn_, wm_, fg_>(Wq, Ws, Xq, Xs, Y, ncols, nrows, n_tok);                                         \
    }
W4P_GEMM_ENTRY(w4a4p_gemm_i4, false, 128, 256, 2, 1)
W4P_GEMM_ENTRY(w4a4p_gemm_i4m, false, 64, 256, 2, 1)
W4P_GEMM_ENTRY(w4a4p_gemm_i4f, true, 64, 256, 2, 1)
W4P_GEMM_ENTRY(w4a4p_gemm_i4fb, true, 128, 128, 4, 1)
W4P_GEMM_ENTRY(w4a4p_gemm_i4g, true, 64, 256, 2, 4)
#undef W4P_GEMM_ENTRY

// activation producer: one thread per 32-value block of X (row-major [tokens][K], K % 32 == 0): nblk = tokens * K / 32
extern "C" __global__ void __launch_bounds__(256) w4a4p_quant_x(const _Float16 *X, uint8_t *Xq, float *Xs, int nblk) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nblk) return;
    half8 h[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = *(const half8 *)(X + (size_t)g * 32 + i * 8);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf((float)h[i][j]));
    const float d = amax / 7.f, inv = amax > 0.f ? 7.f / amax : 0.f;
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t u = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float r = fminf(fmaxf(rintf((float)h[i][j] * inv), -7.f), 7.f);
            u |= ((uint32_t)(int)r & 15u) << (4 * j);
        }
        w[i] = u;
    }
    *(uint4 *)(Xq + (size_t)g * 16) = make_uint4(w[0], w[1], w[2], w[3]);
    Xs[g] = d;
}

// ---------------------------------------------------------------- register-only rate probes (8 accumulators per wave,
// 2 WMMAs per "block" = K32 for the 16x16x16 instructions; the fold variants run P1a's per-block epilogue: per element
// cvt, mul by the per-block scale product, fma; loop-carried int tiles so the compiler cannot hoist the WMMAs)
#define W4P_NACC 8
extern "C" __global__ void __launch_bounds__(256) w4a4p_reg_f16(const w4p_v16h *src, float *out, int iters) {
    const int l = threadIdx.x & 31;
    const w4p_v16h a0 = src[l], a1 = src[32 + l], b0 = src[64 + l], b1 = src[96 + l];
    w4p_v8f c[W4P_NACC];
#pragma unroll
    for (int j = 0; j < W4P_NACC; ++j) c[j] = (w4p_v8f){0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < W4P_NACC; ++j) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, c[j]);
#pragma unroll
        for (int j = 0; j < W4P_NACC; ++j) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, c[j]);
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < W4P_NACC; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) s += c[j][v];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <bool I4, bool FOLD>
__device__ void w4a4p_reg_int(const w4p_v4i *src, const float *sc, float *out, int iters) {
    const int l = threadIdx.x & 31;
    const w4p_v4i a0 = src[l], a1 = src[32 + l], b0 = src[64 + l], b1 = src[96 + l];
    const w4p_v2i h0 = {a0.x, a0.y}, h1 = {a1.x, a1.y}, g0 = {b0.x, b0.y}, g1 = {b1.x, b1.y};
    float sw[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) sw[v] = sc[(l + v) & 63];
    float dx = sc[64 + l];
    float acc[W4P_NACC][8];
    w4p_v8i c[W4P_NACC];
#pragma unroll
    for (int j = 0; j < W4P_NACC; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            acc[j][v] = 0.f;
            c[j][v] = j;
        }
    for (int i = 0; i < iters; ++i) {
        if constexpr (I4) {
#pragma unroll
            for (int j = 0; j < W4P_NACC; ++j) c[j] = W4P_WMMA_I4(h0, g0, c[j]);
#pragma unroll
            for (int j = 0; j < W4P_NACC; ++j) c[j] = W4P_WMMA_I4(h1, g1, c[j]);
        } else {
#pragma unroll
            for (int j = 0; j < W4P_NACC; ++j) c[j] = W4P_WMMA_I8(a0, b0, c[j]);
#pragma unroll
            for (int j = 0; j < W4P_NACC; ++j) c[j] = W4P_WMMA_I8(a1, b1, c[j]);
        }
        if constexpr (FOLD) {
#pragma unroll
            for (int j = 0; j < W4P_NACC; ++j)
#pragma unroll
                for (int v = 0; v < 8; ++v) acc[j][v] = __builtin_fmaf((float)c[j][v], sw[v] * dx, acc[j][v]);
            dx = dx * 0.999f + 1e-7f;
        }
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < W4P_NACC; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) s += FOLD ? acc[j][v] : (float)c[j][v];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
extern "C" __global__ void __launch_bounds__(256) w4a4p_reg_iu8(const w4p_v4i *src, const float *sc, float *out, int iters) { w4a4p_reg_int<false, false>(src, sc, out, iters); }
extern "C" __global__ void __launch_bounds__(256) w4a4p_reg_iu8f(const w4p_v4i *src, const float *sc, float *out, int iters) { w4a4p_reg_int<false, true>(src, sc, out, iters); }
extern "C" __global__ void __launch_bounds__(256) w4a4p_reg_iu4(const w4p_v4i *src, const float *sc, float *out, int iters) { w4a4p_reg_int<true, false>(src, sc, out, iters); }
extern "C" __global__ void __launch_bounds__(256) w4a4p_reg_iu4f(const w4p_v4i *src, const float *sc, float *out, int iters) { w4a4p_reg_int<true, true>(src, sc, out, iters); }
#undef W4P_NACC
