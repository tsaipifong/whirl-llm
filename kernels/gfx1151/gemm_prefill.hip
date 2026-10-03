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
template <int T, int TT, int RW>
__device__ void gemm_impl(const uint8_t *__restrict__ W, uint64_t row_bytes,
                          const float *__restrict__ X, float *__restrict__ Y,
                          int ncols, int nrows, int n_tok, int accumulate) {
    __shared__ float xs[TT][256 + 4];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int nw = blockDim.x / WARP;
    const int row0 = (blockIdx.x * nw + wid) * RW;
    const int t0 = blockIdx.y * TT;
    float acc[RW][TT];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int t = 0; t < TT; ++t) acc[r][t] = 0.f;

    for (int k0 = 0; k0 < ncols; k0 += 256) {
        __syncthreads();
        for (int i = threadIdx.x; i < TT * 64; i += blockDim.x) {
            const int t = i / 64, c4 = i % 64;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t0 + t < n_tok) v = *(const float4 *)(X + (size_t)(t0 + t) * ncols + k0 + 4 * c4);
            xs[t][4 * c4 + 0] = v.x;
            xs[t][4 * c4 + 1] = v.y;
            xs[t][4 * c4 + 2] = v.z;
            xs[t][4 * c4 + 3] = v.w;
        }
        __syncthreads();
        const int g = k0 / 8 + lane;
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int row = row0 + r;
            if (row >= nrows) break;
            float w[8];
            dec_group<T>(W + (size_t)row * row_bytes, g, w);
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                const float *xp = &xs[t][8 * lane];
                acc[r][t] += w[0] * xp[0] + w[1] * xp[1] + w[2] * xp[2] + w[3] * xp[3] +
                             w[4] * xp[4] + w[5] * xp[5] + w[6] * xp[6] + w[7] * xp[7];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const float s = warp_sum(acc[r][t]);
            if (lane == 0 && row < nrows && t0 + t < n_tok) {
                float *dst = Y + (size_t)(t0 + t) * nrows + row;
                *dst = accumulate ? *dst + s : s;
            }
        }
    }
}

#define GEMM_ENTRY(SUF, T)                                                                         \
    extern "C" __global__ void __launch_bounds__(256)                                              \
        gemm_##SUF(const uint8_t *W, uint64_t row_bytes, const float *X, float *Y, int ncols,     \
                   int nrows, int n_tok, int accumulate) {                                        \
        gemm_impl<T, 32, 2>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                  \
    }

GEMM_ENTRY(f32, QT_F32)
GEMM_ENTRY(f16, QT_F16)
GEMM_ENTRY(q8_0, QT_Q8_0)
GEMM_ENTRY(q3_k, QT_Q3_K)
GEMM_ENTRY(q4_k, QT_Q4_K)
GEMM_ENTRY(q5_k, QT_Q5_K)
GEMM_ENTRY(q6_k, QT_Q6_K)
GEMM_ENTRY(iq4_nl, QT_IQ4_NL)
GEMM_ENTRY(iq3_s, QT_IQ3_S)
GEMM_ENTRY(iq4_xs, QT_IQ4_XS)
GEMM_ENTRY(mxfp4, QT_MXFP4)

// ---------------------------------------------------------------------------
// Prefill matmul on the RDNA 3.5 matrix cores (v_wmma_f32_16x16x16_f16).
// Block tile: BM weight rows x BN tokens; K staged BK at a time in LDS as f16.
// Weights are dequantized on the fly; activations arrive pre-converted to f16
// (f32_to_f16) so each row-block streams half the bytes and does no converts.
// gfx11 wave32 fragment layout: see ld_frag16 / CROW below.
// Requires ncols % BK == 0.
// ---------------------------------------------------------------------------
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));

// gfx11 wave32 WMMA fragment layout (llama.cpp ggml-cuda/mma.cuh, RDNA3):
//   A, B: lane holds row/column (lane % 16), all 16 k (lanes 16-31 = 0-15)
//   D:    lane holds column n = lane % 16, rows m = 2 * v + (lane / 16)
// Fragment = 16 consecutive halfs of an LDS row (two 16-byte reads).
__device__ __forceinline__ half16 ld_frag16(const _Float16 *p) {
    const half8 lo = *(const half8 *)p, hi = *(const half8 *)(p + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
#define WMMA16(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
// accumulator element v of this lane -> row within the 16x16 tile
#define CROW(lane_, v_) (2 * (v_) + ((lane_) / 16))
// the other half-wave's copy of a 32-bit value (lane ^ 16), one VALU op
__device__ __forceinline__ int xhalf(int v) {
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false, false);
}
// Join per-half 8-value pieces into the full 16-k fragment both halves hold:
// k [0, 8) = half 0's piece, k [8, 16) = half 1's piece.
__device__ __forceinline__ half16 join_halves(half8 own, int lane) {
    const int4 o = __builtin_bit_cast(int4, own);
    const int4 p = make_int4(xhalf(o.x), xhalf(o.y), xhalf(o.z), xhalf(o.w));
    const bool h0 = lane < 16;
    const int4 lo = h0 ? o : p, hi = h0 ? p : o;
    return __builtin_shufflevector(__builtin_bit_cast(half8, lo), __builtin_bit_cast(half8, hi),
                                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

extern "C" __global__ void f32_to_f16(const float *x, _Float16 *y, int n) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        const float4 v = *(const float4 *)(x + i);
        y[i] = (_Float16)v.x;
        y[i + 1] = (_Float16)v.y;
        y[i + 2] = (_Float16)v.z;
        y[i + 3] = (_Float16)v.w;
    } else {
        for (int j = i; j < n; ++j) y[j] = (_Float16)x[j];
    }
}

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
#pragma unroll
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
#pragma unroll
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
#define WMMA_ENTRY(SUF, T)                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                               \
        gemm_wmma_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, \
                        int nrows, int n_tok, int accumulate) {                                    \
        gemm_wmma_impl<T, 128, 128, 64>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);       \
    }

WMMA_ENTRY(f32, QT_F32)
WMMA_ENTRY(f16, QT_F16)
WMMA_ENTRY(q8_0, QT_Q8_0)
WMMA_ENTRY(q3_k, QT_Q3_K)
WMMA_ENTRY(q4_k, QT_Q4_K)
WMMA_ENTRY(q5_k, QT_Q5_K)
WMMA_ENTRY(q6_k, QT_Q6_K)
WMMA_ENTRY(iq4_nl, QT_IQ4_NL)
WMMA_ENTRY(iq3_s, QT_IQ3_S)
WMMA_ENTRY(iq4_xs, QT_IQ4_XS)
WMMA_ENTRY(mxfp4, QT_MXFP4)

// ---------------------------------------------------------------------------
// Prefill GEMM v3: register-blocked, software-pipelined WMMA.
// Block tile BM x BN (weights x tokens), 8 waves arranged WAVES_M x WAVES_N,
// each wave owning TM x TN 16x16 tiles (fragment reuse: TM + TN LDS loads per
// TM * TN WMMAs). The next K chunk is fetched/dequantized into registers
// while the current chunk is multiplied from LDS.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Fast dequant to f16 for the prefill GEMM: vector loads + packed f16 math.
// Small unsigned ints u (< 1024) become exact f16 via bits 0x6400 | u
// (= 1024 + u); subtracting 1024 is exact, then one packed FMA applies the
// scale. Types without a fast path fall back to dec_group.
// ---------------------------------------------------------------------------
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ int iq4_lut4(uint32_t nib4);  // defined with the gemvq kernels
__device__ __forceinline__ int mx_lut4(uint32_t nib4);   // (MXFP4 codes, same place)

// bytes b0..b3 (each < 256) of x -> two half2: (b0,b1), (b2,b3) as 1024 + b
__device__ __forceinline__ void bytes_to_h2(uint32_t x, half2_t &lo, half2_t &hi) {
    const uint32_t a = (x & 0xFFu) | ((x & 0xFF00u) << 8) | 0x64006400u;
    const uint32_t b = ((x >> 16) & 0xFFu) | ((x >> 8) & 0xFF0000u) | 0x64006400u;
    lo = __builtin_bit_cast(half2_t, a);
    hi = __builtin_bit_cast(half2_t, b);
}

// 8 packed bytes (u0,u1) -> half8 of (byte - bias) * scale + add
__device__ __forceinline__ half8 bytes8_to_h8(uint32_t u0, uint32_t u1, _Float16 bias, _Float16 scale, _Float16 add) {
    half2_t p[4];
    bytes_to_h2(u0, p[0], p[1]);
    bytes_to_h2(u1, p[2], p[3]);
    const half2_t b2 = {(_Float16)1024.f + bias, (_Float16)1024.f + bias};
    const half2_t s2 = {scale, scale}, a2 = {add, add};
    half8 out;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const half2_t v = (p[i] - b2) * s2 + a2;
        out[2 * i] = v[0];
        out[2 * i + 1] = v[1];
    }
    return out;
}

template <int T>
__device__ __forceinline__ half8 dec_group_h(const uint8_t *row, int g) {
    float w[8];
    dec_group<T>(row, g, w);
    half8 h;
#pragma unroll
    for (int e = 0; e < 8; ++e) h[e] = (_Float16)w[e];
    return h;
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_F16>(const uint8_t *row, int g) {
    return *(const half8 *)(row + (size_t)g * 16);
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_Q4_K>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 5) * 144;
    const int p = (g & 31) * 8;
    const int j64 = p >> 6, in = p & 63, hi = in >> 5, l = in & 31;
    const uint32_t dm = *(const uint32_t *)b;
    const float d = __half2float(__ushort_as_half((uint16_t)dm)), dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
    int sc, m;
    scale_min_k4(2 * j64 + hi, b + 4, sc, m);
    const uint2 q = *(const uint2 *)(b + 16 + 32 * j64 + l);
    const uint32_t q0 = hi ? (q.x >> 4) & 0x0F0F0F0F : q.x & 0x0F0F0F0F;
    const uint32_t q1 = hi ? (q.y >> 4) & 0x0F0F0F0F : q.y & 0x0F0F0F0F;
    return bytes8_to_h8(q0, q1, (_Float16)0.f, (_Float16)(d * sc), (_Float16)(-dmin * m));
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_Q5_K>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 5) * 176;
    const int p = (g & 31) * 8;
    const int j64 = p >> 6, in = p & 63, hi = in >> 5, l = in & 31;
    const uint32_t dm = *(const uint32_t *)b;
    const float d = __half2float(__ushort_as_half((uint16_t)dm)), dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
    int sc, m;
    scale_min_k4(2 * j64 + hi, b + 4, sc, m);
    const uint2 q = *(const uint2 *)(b + 48 + 32 * j64 + l);
    const uint2 h = *(const uint2 *)(b + 16 + l);
    const int sh = 2 * j64 + hi;
    const uint32_t q0 = (hi ? (q.x >> 4) & 0x0F0F0F0F : q.x & 0x0F0F0F0F) | (((h.x >> sh) & 0x01010101) << 4);
    const uint32_t q1 = (hi ? (q.y >> 4) & 0x0F0F0F0F : q.y & 0x0F0F0F0F) | (((h.y >> sh) & 0x01010101) << 4);
    return bytes8_to_h8(q0, q1, (_Float16)0.f, (_Float16)(d * sc), (_Float16)(-dmin * m));
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_IQ4_XS>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 5) * 136;
    const int p = (g & 31) * 8;
    const int ib = p >> 5, s = p & 31;
    const uint32_t hdr = *(const uint32_t *)b;
    const float d = __half2float(__ushort_as_half((uint16_t)hdr));
    const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) | ((((hdr >> 16) >> (2 * ib)) & 3) << 4);
    const uint2 q = *(const uint2 *)(b + 8 + 16 * ib + (s & 15));
    const uint32_t n0 = s < 16 ? q.x & 0x0F0F0F0F : (q.x >> 4) & 0x0F0F0F0F;
    const uint32_t n1 = s < 16 ? q.y & 0x0F0F0F0F : (q.y >> 4) & 0x0F0F0F0F;
    // signed LUT bytes -> unsigned (v + 128), then subtract 128
    const uint32_t u0 = (uint32_t)iq4_lut4(n0) ^ 0x80808080u, u1 = (uint32_t)iq4_lut4(n1) ^ 0x80808080u;
    return bytes8_to_h8(u0, u1, (_Float16)128.f, (_Float16)(d * (ls - 32)), (_Float16)0.f);
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_MXFP4>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 5) * 136;
    const int p = (g & 31) * 8;
    const int ib = p >> 5, s = p & 31;
    const float dl = e8m0_half(b[ib]);
    const uint2 q = *(const uint2 *)(b + 8 + 16 * ib + (s & 15));
    const uint32_t n0 = s < 16 ? q.x & 0x0F0F0F0F : (q.x >> 4) & 0x0F0F0F0F;
    const uint32_t n1 = s < 16 ? q.y & 0x0F0F0F0F : (q.y >> 4) & 0x0F0F0F0F;
    const uint32_t u0 = (uint32_t)mx_lut4(n0) ^ 0x80808080u, u1 = (uint32_t)mx_lut4(n1) ^ 0x80808080u;
    return bytes8_to_h8(u0, u1, (_Float16)128.f, (_Float16)dl, (_Float16)0.f);
}

template <int T, int BM, int BN, int BK, int WAVES_M, int MODE = 0, int NTH = 256, bool TOK_X = false>
__device__ void gemm3_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                           float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int WAVES_N = NTH / 32 / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int LD = BK + 8;
    constexpr int GPR = BK / 8;                   // 8-value groups per row per chunk
    constexpr int A_PER = BM * GPR / NTH;         // groups per thread (weights)
    constexpr int B_PER = BN * GPR / NTH;         // groups per thread (activations)
    static_assert(BM * GPR % NTH == 0 && BN * GPR % NTH == 0, "tile/threads");
    __shared__ _Float16 As[BM][LD];
    __shared__ _Float16 Bs[BN][LD];

    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    // TOK_X: token blocks along grid x so the blocks sharing a weight row
    // block run back to back and reuse it from L2 instead of re-streaming W.
    const int m0 = (TOK_X ? blockIdx.y : blockIdx.x) * BM, t0 = (TOK_X ? blockIdx.x : blockIdx.y) * BN;

    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    half8 ra[A_PER], rb[B_PER];
    // (a macro, not a lambda: a by-reference capture of ra/rb takes their
    // address and pushes them to scratch)
#define GEMM3_FETCH(k0_)                                                                            \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < A_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int r = i / GPR, gg = i % GPR;                                                    \
            if (MODE == 1) {                                                                        \
                ra[e] = *(const half8 *)(W + (size_t)(m0 + r) * row_bytes + (((k0_) / 8 + gg) * 16) % row_bytes); \
            } else if (m0 + r < nrows) {                                                            \
                ra[e] = dec_group_h<T>(W + (size_t)(m0 + r) * row_bytes, (k0_) / 8 + gg);            \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) ra[e][q] = (_Float16)0.f;              \
            }                                                                                       \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < B_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int t = i / GPR, gg = i % GPR;                                                    \
            if (t0 + t < n_tok) {                                                                   \
                rb[e] = *(const half8 *)(X + (size_t)(t0 + t) * ncols + (k0_) + gg * 8);            \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) rb[e][q] = (_Float16)0.f;              \
            }                                                                                       \
        }                                                                                           \
    } while (0)

    GEMM3_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(half8 *)&As[i / GPR][(i % GPR) * 8] = ra[e];
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(half8 *)&Bs[i / GPR][(i % GPR) * 8] = rb[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) GEMM3_FETCH(k0 + BK);
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
#undef GEMM3_FETCH

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

#define GEMM4_PROBE(NAME, BM, BN, BK, WM)                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                              \
        NAME(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, \
             int n_tok, int accumulate) {                                                         \
        gemm4_impl<QT_Q4_K, BM, BN, BK, WM>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);  \
    }
GEMM4_PROBE(g4p_128x256_32, 128, 256, 32, 2)
GEMM4_PROBE(g4p_256x128_32, 256, 128, 32, 4)
GEMM4_PROBE(g4p_128x128_32, 128, 128, 32, 2)

// Dequantize a whole weight matrix to f16 (row-major), for the
// dequant-then-f16-GEMM prefill path. One thread per 8-value group.
template <int T>
__device__ void dequant_f16_impl(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) {
    const size_t gpr = ncols / 8;
    const size_t total = gpr * nrows;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / gpr, g = i % gpr;
        *(half8 *)(out + r * ncols + g * 8) = dec_group_h<T>(W + r * row_bytes, (int)g);
    }
}
extern "C" __global__ void __launch_bounds__(256) dequant_f16_f32(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { dequant_f16_impl<QT_F32>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) dequant_f16_q8_0(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { dequant_f16_impl<QT_Q8_0>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) dequant_f16_q3_k(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { dequant_f16_impl<QT_Q3_K>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) dequant_f16_q4_k(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { dequant_f16_impl<QT_Q4_K>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) dequant_f16_q5_k(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { dequant_f16_impl<QT_Q5_K>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) dequant_f16_q6_k(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { dequant_f16_impl<QT_Q6_K>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) dequant_f16_iq4_nl(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { dequant_f16_impl<QT_IQ4_NL>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) dequant_f16_iq3_s(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { dequant_f16_impl<QT_IQ3_S>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) dequant_f16_iq4_xs(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { dequant_f16_impl<QT_IQ4_XS>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) dequant_f16_mxfp4(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { dequant_f16_impl<QT_MXFP4>(W, row_bytes, out, ncols, nrows); }

// Prefill GEMM configurations, chosen per matrix by the load-time autotuner.
// (BM, BN, threads) must match kGemmCfgs in include/whirl/kernels_abi.h; BK / WAVES_M are free
// per arch. gfx1151: c10 uses WAVES_M 2 (4x2 WMMA tiles per wave instead of
// 1x8) and c11 WAVES_M 4 (4x2 instead of 1x8): the gfx11 A/B fragments are 32
// bytes per lane (lanes 16-31 duplicate), so fewer fragment loads per WMMA pay.
extern "C" __global__ void __launch_bounds__(512) gemm_c0_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 128, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c0_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c0_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 128, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c0_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 128, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c0_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 128, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c0_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 128, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c0_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 128, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c0_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 128, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c0_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 128, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c0_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 128, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c0_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 128, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c1_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 128, 256, 64, 8, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c1_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 64, 8, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c1_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 128, 256, 64, 8, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c1_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 128, 256, 64, 8, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c1_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 128, 256, 64, 8, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c1_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 128, 256, 64, 8, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c1_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 128, 256, 64, 8, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c1_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 128, 256, 64, 8, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c1_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 128, 256, 64, 8, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c1_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 128, 256, 64, 8, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c1_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 128, 256, 64, 8, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c2_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 256, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c2_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c2_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 256, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c2_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 256, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c2_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 256, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c2_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 256, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c2_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 256, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c2_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 256, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c2_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 256, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c2_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 256, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c2_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 256, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c3_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 64, 128, 32, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c3_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 32, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c3_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 64, 128, 32, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c3_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 64, 128, 32, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c3_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 64, 128, 32, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c3_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 64, 128, 32, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c3_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 64, 128, 32, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c3_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 64, 128, 32, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c3_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 64, 128, 32, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c3_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 64, 128, 32, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c3_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 64, 128, 32, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c4_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 64, 128, 32, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c4_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 32, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c4_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 64, 128, 32, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c4_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 64, 128, 32, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c4_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 64, 128, 32, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c4_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 64, 128, 32, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c4_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 64, 128, 32, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c4_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 64, 128, 32, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c4_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 64, 128, 32, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c4_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 64, 128, 32, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c4_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 64, 128, 32, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c5_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 64, 128, 32, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c5_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 32, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c5_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 64, 128, 32, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c5_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 64, 128, 32, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c5_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 64, 128, 32, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c5_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 64, 128, 32, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c5_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 64, 128, 32, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c5_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 64, 128, 32, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c5_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 64, 128, 32, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c5_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 64, 128, 32, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c5_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 64, 128, 32, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c6_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 64, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c6_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c6_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 64, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c6_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 64, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c6_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 64, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c6_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 64, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c6_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 64, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c6_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 64, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c6_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 64, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c6_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 64, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c6_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 64, 128, 64, 1, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c7_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 64, 128, 64, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c7_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 64, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c7_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 64, 128, 64, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c7_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 64, 128, 64, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c7_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 64, 128, 64, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c7_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 64, 128, 64, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c7_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 64, 128, 64, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c7_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 64, 128, 64, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c7_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 64, 128, 64, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c7_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 64, 128, 64, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c7_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 64, 128, 64, 2, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c8_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 64, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c8_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c8_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 64, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c8_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 64, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c8_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 64, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c8_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 64, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c8_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 64, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c8_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 64, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c8_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 64, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c8_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 64, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gemm_c8_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 64, 128, 64, 4, 0, 256, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c9_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 128, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c9_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c9_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 128, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c9_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 128, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c9_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 128, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c9_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 128, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c9_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 128, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c9_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 128, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c9_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 128, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c9_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 128, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c9_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 128, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
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
extern "C" __global__ void __launch_bounds__(512) gemm_c12_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 256, 128, 64, 16, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c12_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 64, 16, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c12_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 256, 128, 64, 16, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c12_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 256, 128, 64, 16, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c12_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 256, 128, 64, 16, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c12_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 256, 128, 64, 16, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c12_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 256, 128, 64, 16, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c12_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 256, 128, 64, 16, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c12_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 256, 128, 64, 16, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c12_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 256, 128, 64, 16, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c12_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 256, 128, 64, 16, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c13_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 128, 128, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c13_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c13_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 128, 128, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c13_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 128, 128, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c13_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 128, 128, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c13_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 128, 128, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c13_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 128, 128, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c13_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 128, 128, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c13_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 128, 128, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c13_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 128, 128, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c13_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 128, 128, 32, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c14_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 128, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c14_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c14_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 128, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c14_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 128, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c14_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 128, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c14_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 128, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c14_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 128, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c14_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 128, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c14_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 128, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c14_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 128, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c14_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 128, 128, 32, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c15_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 128, 128, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c15_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c15_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 128, 128, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c15_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 128, 128, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c15_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 128, 128, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c15_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 128, 128, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c15_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 128, 128, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c15_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 128, 128, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c15_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 128, 128, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c15_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 128, 128, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c15_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 128, 128, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c16_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 128, 128, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c16_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c16_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 128, 128, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c16_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 128, 128, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c16_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 128, 128, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c16_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 128, 128, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c16_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 128, 128, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c16_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 128, 128, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c16_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 128, 128, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c16_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 128, 128, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c16_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 128, 128, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c17_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 128, 128, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c17_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c17_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 128, 128, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c17_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 128, 128, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c17_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 128, 128, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c17_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 128, 128, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c17_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 128, 128, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c17_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 128, 128, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c17_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 128, 128, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c17_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 128, 128, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c17_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 128, 128, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c18_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 128, 128, 64, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c18_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 64, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c18_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 128, 128, 64, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c18_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 128, 128, 64, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c18_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 128, 128, 64, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c18_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 128, 128, 64, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c18_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 128, 128, 64, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c18_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 128, 128, 64, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c18_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 128, 128, 64, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c18_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 128, 128, 64, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c18_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 128, 128, 64, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c19_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 64, 256, 64, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c19_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 64, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c19_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 64, 256, 64, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c19_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 64, 256, 64, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c19_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 64, 256, 64, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c19_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 64, 256, 64, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c19_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 64, 256, 64, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c19_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 64, 256, 64, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c19_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 64, 256, 64, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c19_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 64, 256, 64, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c19_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 64, 256, 64, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c20_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 64, 256, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c20_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c20_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 64, 256, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c20_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 64, 256, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c20_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 64, 256, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c20_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 64, 256, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c20_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 64, 256, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c20_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 64, 256, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c20_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 64, 256, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c20_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 64, 256, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c20_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 64, 256, 64, 2, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c21_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 64, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c21_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c21_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 64, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c21_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 64, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c21_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 64, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c21_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 64, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c21_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 64, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c21_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 64, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c21_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 64, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c21_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 64, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c21_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 64, 256, 64, 4, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c22_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 256, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c22_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c22_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 256, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c22_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 256, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c22_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 256, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c22_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 256, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c22_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 256, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c22_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 256, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c22_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 256, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c22_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 256, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c22_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 256, 256, 32, 1, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c23_f32(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F32, 128, 512, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c23_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 512, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c23_q8_0(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q8_0, 128, 512, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c23_q3_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q3_K, 128, 512, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c23_q4_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q4_K, 128, 512, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c23_q5_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q5_K, 128, 512, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c23_q6_k(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_Q6_K, 128, 512, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c23_iq4_nl(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_NL, 128, 512, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c23_iq3_s(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ3_S, 128, 512, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c23_iq4_xs(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_IQ4_XS, 128, 512, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gemm_c23_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_MXFP4, 128, 512, 32, 8, 0, 512, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }

// gemm3 shape probes (bench only)
#define GEMM3_PROBE(NAME, BM, BN, BK, WM, MODE)                                                    \
    extern "C" __global__ void __launch_bounds__(256)                                              \
        NAME(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, \
             int n_tok, int accumulate) {                                                         \
        gemm3_impl<QT_Q4_K, BM, BN, BK, WM, MODE>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); \
    }
GEMM3_PROBE(g3p_nodeq, 128, 256, 64, 2, 1)
GEMM3_PROBE(g3p_bk32, 128, 256, 32, 2, 0)
GEMM3_PROBE(g3p_256x128, 256, 128, 64, 4, 0)
GEMM3_PROBE(g3p_128x128, 128, 128, 64, 2, 0)
GEMM3_PROBE(g3p_nodeq32, 128, 256, 32, 2, 1)
GEMM3_PROBE(g3p_bk16, 128, 256, 16, 2, 0)

// GRID-BEGIN
GEMM3_PROBE(gx3_128x256_k16_w1, 128, 256, 16, 1, 0)
GEMM3_PROBE(gx3_128x256_k16_w2, 128, 256, 16, 2, 0)
GEMM3_PROBE(gx3_128x256_k16_w4, 128, 256, 16, 4, 0)
GEMM3_PROBE(gx3_128x256_k32_w1, 128, 256, 32, 1, 0)
GEMM3_PROBE(gx3_128x256_k32_w2, 128, 256, 32, 2, 0)
GEMM3_PROBE(gx3_128x256_k32_w4, 128, 256, 32, 4, 0)
GEMM3_PROBE(gx3_128x256_k64_w1, 128, 256, 64, 1, 0)
GEMM3_PROBE(gx3_128x256_k64_w2, 128, 256, 64, 2, 0)
GEMM3_PROBE(gx3_128x256_k64_w4, 128, 256, 64, 4, 0)
GEMM3_PROBE(gx3_256x128_k16_w2, 256, 128, 16, 2, 0)
GEMM3_PROBE(gx3_256x128_k16_w4, 256, 128, 16, 4, 0)
GEMM3_PROBE(gx3_256x128_k16_w8, 256, 128, 16, 8, 0)
GEMM3_PROBE(gx3_256x128_k32_w2, 256, 128, 32, 2, 0)
GEMM3_PROBE(gx3_256x128_k32_w4, 256, 128, 32, 4, 0)
GEMM3_PROBE(gx3_256x128_k32_w8, 256, 128, 32, 8, 0)
GEMM3_PROBE(gx3_256x128_k64_w2, 256, 128, 64, 2, 0)
GEMM3_PROBE(gx3_256x128_k64_w4, 256, 128, 64, 4, 0)
GEMM3_PROBE(gx3_256x128_k64_w8, 256, 128, 64, 8, 0)
GEMM3_PROBE(gx3_128x128_k16_w1, 128, 128, 16, 1, 0)
GEMM3_PROBE(gx3_128x128_k16_w2, 128, 128, 16, 2, 0)
GEMM3_PROBE(gx3_128x128_k16_w4, 128, 128, 16, 4, 0)
GEMM3_PROBE(gx3_128x128_k16_w8, 128, 128, 16, 8, 0)
GEMM3_PROBE(gx3_128x128_k32_w1, 128, 128, 32, 1, 0)
GEMM3_PROBE(gx3_128x128_k32_w2, 128, 128, 32, 2, 0)
GEMM3_PROBE(gx3_128x128_k32_w4, 128, 128, 32, 4, 0)
GEMM3_PROBE(gx3_128x128_k32_w8, 128, 128, 32, 8, 0)
GEMM3_PROBE(gx3_128x128_k64_w1, 128, 128, 64, 1, 0)
GEMM3_PROBE(gx3_128x128_k64_w2, 128, 128, 64, 2, 0)
GEMM3_PROBE(gx3_128x128_k64_w4, 128, 128, 64, 4, 0)
GEMM3_PROBE(gx3_128x128_k64_w8, 128, 128, 64, 8, 0)
GEMM3_PROBE(gx3_64x256_k32_w1, 64, 256, 32, 1, 0)
GEMM3_PROBE(gx3_64x256_k32_w2, 64, 256, 32, 2, 0)
GEMM3_PROBE(gx3_64x256_k32_w4, 64, 256, 32, 4, 0)
GEMM3_PROBE(gx3_64x256_k64_w1, 64, 256, 64, 1, 0)
GEMM3_PROBE(gx3_64x256_k64_w2, 64, 256, 64, 2, 0)
GEMM3_PROBE(gx3_64x256_k64_w4, 64, 256, 64, 4, 0)
GEMM4_PROBE(gx4_128x256_k16_w1, 128, 256, 16, 1)
GEMM4_PROBE(gx4_128x256_k16_w2, 128, 256, 16, 2)
GEMM4_PROBE(gx4_128x256_k16_w4, 128, 256, 16, 4)
GEMM4_PROBE(gx4_128x256_k32_w1, 128, 256, 32, 1)
GEMM4_PROBE(gx4_128x256_k32_w2, 128, 256, 32, 2)
GEMM4_PROBE(gx4_128x256_k32_w4, 128, 256, 32, 4)
GEMM4_PROBE(gx4_256x128_k16_w2, 256, 128, 16, 2)
GEMM4_PROBE(gx4_256x128_k16_w4, 256, 128, 16, 4)
GEMM4_PROBE(gx4_256x128_k16_w8, 256, 128, 16, 8)
GEMM4_PROBE(gx4_256x128_k32_w2, 256, 128, 32, 2)
GEMM4_PROBE(gx4_256x128_k32_w4, 256, 128, 32, 4)
GEMM4_PROBE(gx4_256x128_k32_w8, 256, 128, 32, 8)
GEMM4_PROBE(gx4_128x128_k16_w1, 128, 128, 16, 1)
GEMM4_PROBE(gx4_128x128_k16_w2, 128, 128, 16, 2)
GEMM4_PROBE(gx4_128x128_k16_w4, 128, 128, 16, 4)
GEMM4_PROBE(gx4_128x128_k16_w8, 128, 128, 16, 8)
GEMM4_PROBE(gx4_128x128_k32_w1, 128, 128, 32, 1)
GEMM4_PROBE(gx4_128x128_k32_w2, 128, 128, 32, 2)
GEMM4_PROBE(gx4_128x128_k32_w4, 128, 128, 32, 4)
GEMM4_PROBE(gx4_128x128_k32_w8, 128, 128, 32, 8)
GEMM4_PROBE(gx4_64x256_k32_w1, 64, 256, 32, 1)
GEMM4_PROBE(gx4_64x256_k32_w2, 64, 256, 32, 2)
GEMM4_PROBE(gx4_64x256_k32_w4, 64, 256, 32, 4)
// GRID-END
#define GEMM3W_PROBE(NAME, BM, BN, BK, WM)                                                         \
    extern "C" __global__ void __launch_bounds__(512)                                              \
        NAME(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, \
             int n_tok, int accumulate) {                                                         \
        gemm3_impl<QT_Q4_K, BM, BN, BK, WM, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); \
    }
GEMM3W_PROBE(gw_256x256_k32_w2, 256, 256, 32, 2)
GEMM3W_PROBE(gw_256x256_k32_w4, 256, 256, 32, 4)
GEMM3W_PROBE(gw_256x256_k32_w8, 256, 256, 32, 8)
GEMM3W_PROBE(gw_256x256_k16_w4, 256, 256, 16, 4)
GEMM3W_PROBE(gw_256x256_k16_w2, 256, 256, 16, 2)
GEMM3W_PROBE(gw_128x512_k32_w1, 128, 512, 32, 1)
GEMM3W_PROBE(gw_128x512_k32_w2, 128, 512, 32, 2)

// F16GRID-BEGIN
extern "C" __global__ void __launch_bounds__(256) gf3_128x256_k16_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 16, 1, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x256_k16_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 16, 2, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x256_k16_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 16, 4, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x256_k32_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 32, 1, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x256_k32_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 32, 2, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x256_k32_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 32, 4, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x256_k64_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 64, 1, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x256_k64_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 64, 2, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x256_k64_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 64, 4, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_256x128_k16_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 16, 2, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_256x128_k16_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 16, 4, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_256x128_k16_w8_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 16, 8, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_256x128_k32_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 32, 2, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_256x128_k32_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 32, 4, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_256x128_k32_w8_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 32, 8, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_256x128_k64_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 64, 2, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_256x128_k64_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 64, 4, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_256x128_k64_w8_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 64, 8, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k16_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 16, 1, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k16_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 16, 2, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k16_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 16, 4, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k16_w8_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 16, 8, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k32_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 32, 1, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k32_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 32, 2, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k32_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 32, 4, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k32_w8_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 32, 8, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k64_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 64, 1, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k64_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 64, 2, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k64_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 64, 4, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_128x128_k64_w8_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 64, 8, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_64x256_k32_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 32, 1, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_64x256_k32_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 32, 2, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_64x256_k32_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 32, 4, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_64x256_k64_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 64, 1, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_64x256_k64_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 64, 2, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf3_64x256_k64_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 64, 4, 0, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_128x256_k32_w1_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 32, 1, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_128x256_k32_w2_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 32, 2, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_128x256_k32_w4_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 32, 4, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_128x256_k32_w8_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 32, 8, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_128x256_k64_w1_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 64, 1, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_128x256_k64_w2_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 64, 2, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_128x256_k64_w4_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 64, 4, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_128x256_k64_w8_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 64, 8, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x128_k32_w2_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 32, 2, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x128_k32_w4_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 32, 4, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x128_k32_w8_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 32, 8, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x128_k64_w2_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 64, 2, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x128_k64_w4_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 64, 4, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x128_k64_w8_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 64, 8, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x256_k16_w2_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 256, 16, 2, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x256_k16_w4_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 256, 16, 4, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x256_k16_w8_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 256, 16, 8, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x256_k32_w2_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 256, 32, 2, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x256_k32_w4_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 256, 32, 4, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_256x256_k32_w8_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 256, 32, 8, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_128x512_k32_w1_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 512, 32, 1, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_128x512_k32_w2_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 512, 32, 2, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf3_128x512_k32_w4_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 512, 32, 4, 0, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x256_k16_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 256, 16, 1, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x256_k16_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 256, 16, 2, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x256_k16_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 256, 16, 4, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x256_k32_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 256, 32, 1, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x256_k32_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 256, 32, 2, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x256_k32_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 256, 32, 4, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_256x128_k16_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 128, 16, 2, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_256x128_k16_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 128, 16, 4, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_256x128_k16_w8_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 128, 16, 8, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_256x128_k32_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 128, 32, 2, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_256x128_k32_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 128, 32, 4, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_256x128_k32_w8_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 128, 32, 8, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x128_k16_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 128, 16, 1, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x128_k16_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 128, 16, 2, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x128_k16_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 128, 16, 4, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x128_k16_w8_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 128, 16, 8, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x128_k32_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 128, 32, 1, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x128_k32_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 128, 32, 2, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x128_k32_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 128, 32, 4, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_128x128_k32_w8_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 128, 32, 8, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_64x256_k32_w1_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 64, 256, 32, 1, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_64x256_k32_w2_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 64, 256, 32, 2, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(256) gf4_64x256_k32_w4_t256(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 64, 256, 32, 4, 256>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf4_128x256_k32_w1_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 256, 32, 1, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf4_128x256_k32_w2_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 256, 32, 2, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf4_128x256_k32_w4_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 256, 32, 4, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf4_128x256_k32_w8_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 128, 256, 32, 8, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf4_256x128_k32_w2_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 128, 32, 2, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf4_256x128_k32_w4_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 128, 32, 4, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf4_256x128_k32_w8_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 128, 32, 8, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf4_256x256_k16_w2_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 256, 16, 2, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf4_256x256_k16_w4_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 256, 16, 4, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
extern "C" __global__ void __launch_bounds__(512) gf4_256x256_k16_w8_t512(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm4_impl<QT_F16, 256, 256, 16, 8, 512>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate); }
// F16GRID-END
// Bottleneck probes (bench only): 1 = skip dequant, 2 = skip WMMA.
extern "C" __global__ void __launch_bounds__(256)
    gemm_probe_nodeq(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows,
                     int n_tok, int accumulate) {
    gemm_wmma_impl<QT_Q4_K, 128, 128, 64, 1>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);
}
extern "C" __global__ void __launch_bounds__(256)
    gemm_probe_nomma(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows,
                     int n_tok, int accumulate) {
    gemm_wmma_impl<QT_Q4_K, 128, 128, 64, 2>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);
}
