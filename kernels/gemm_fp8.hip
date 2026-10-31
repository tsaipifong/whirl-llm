// WHIRL device code (gfx1201 / RDNA 4): MXFP4-weight x FP8(e4m3)-activation prefill GEMM (gemm8*).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Item M: MXFP4-weight x FP8(e4m3)-activation prefill GEMM
// (v_wmma_f32_16x16x16_fp8_fp8). Activations are quantized per token
// (qact_fp8: q = e4m3(x * 448 / amax_t), sx[t] = amax_t / 448), lossy.
// Weights stay MXFP4: each 32-value block is unpacked to e4m3 with its E8M0
// exponent folded in, w8 = e2m1 * 2^-(ref[r] - e), ref[r] = the row's largest
// block exponent (Mat.ref, computed at load), so no per-block scaling is left
// after the WMMA: Y = acc * sx[t] * 2^(ref[r] - 127). The fold is exact while
// ref - e <= 8 (e4m3 subnormals); the loader counts blocks beyond that.
// Method (folded block exponent) as in bkvargyas/r9700-stack
// kernels/r9k_moe_mxfp4a8.hip (Apache-2.0, c03aa1e); the table below is
// generated independently (offline script, not in this repository) and the kernel
// follows the gemm3 structure (register prefetch, LDS tiles, TM x TN per wave).
// ---------------------------------------------------------------------------
typedef int fp8x8 __attribute__((ext_vector_type(2)));

// e4m3 bytes of the e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} * 2^-s, s = 0..15
__device__ __constant__ uint32_t k_fp8tab[16][2] = {
    {0x3c383000u, 0x4c484440u}, {0x34302800u, 0x44403c38u}, {0x2c282000u, 0x3c383430u}, {0x24201800u, 0x34302c28u},
    {0x1c181000u, 0x2c282420u}, {0x14100800u, 0x24201c18u}, {0x0c080400u, 0x1c181410u}, {0x06040200u, 0x14100c08u},
    {0x03020100u, 0x0c080604u}, {0x02010000u, 0x06040302u}, {0x01000000u, 0x03020201u}, {0x00000000u, 0x02010100u},
    {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
};

__device__ __forceinline__ uint32_t mx_fp8x4(uint32_t n, uint32_t t0, uint32_t t1) {
    return __builtin_amdgcn_perm(t1, t0, n & 0x07070707u) | ((n & 0x08080808u) << 4);
}

// per-token fp8 activations: one block per token row of x [n][ncols] (ncols % 4 == 0)
template <bool TL>
__device__ void qact_fp8_impl(const float *x, uint8_t *q, float *sx, int ncols) {
    const float *xr = x + (size_t)blockIdx.x * ncols;
    float m = 0.f;
    for (int i = threadIdx.x * 4; i < ncols; i += 1024) {
        const float4 v = *(const float4 *)(xr + i);
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    m = block_max(m);
    const float inv = m > 0.f ? 448.f / m : 0.f;
    if (threadIdx.x == 0) sx[blockIdx.x] = m > 0.f ? m / 448.f : 0.f;
    uint8_t *qr = q + (size_t)blockIdx.x * ncols;
    for (int i = threadIdx.x * 4; i < ncols; i += 1024) {
        const float4 v = *(const float4 *)(xr + i);
        const float a = fminf(fmaxf(v.x * inv, -448.f), 448.f), b = fminf(fmaxf(v.y * inv, -448.f), 448.f);
        const float c = fminf(fmaxf(v.z * inv, -448.f), 448.f), d = fminf(fmaxf(v.w * inv, -448.f), 448.f);
        int p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
        p = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, p, true);
        if (TL) *(int *)(q + x8t_off(blockIdx.x, i, ncols)) = p;
        else *(int *)(qr + i) = p;
    }
}
extern "C" __global__ void __launch_bounds__(256) qact_fp8(const float *x, uint8_t *q, float *sx, int ncols) {
    qact_fp8_impl<false>(x, q, sx, ncols);
}

// OH (item P): write Y as f16 (no accumulate), for GEMM outputs that only feed an
// elementwise op (FFN gate / up -> silu_mul_x8h); halves the output traffic.
template <int BM, int BN, int BK, int WAVES_M, int NTH, bool OH = false>
__device__ void gemm8_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const uint8_t *__restrict__ wref,
                           const uint8_t *__restrict__ X8, const float *__restrict__ sx, float *__restrict__ Y,
                           int ncols, int nrows, int n_tok, int accumulate) {
    // main loop: kernels/shared/gemm8_mainloop.h (shared with gemm8_moe_impl, ADR-0005)
#define G8_TILE_ORIGIN m0 = blockIdx.y * BM, t0 = blockIdx.x * BN  // token blocks along x (weight reuse in L2)
#define G8_TROW(t) t0 + t
#define G8_TOFF(t) (size_t)(t0 + t)
#define G8_WMMA_IF(j)
#include "shared/gemm8_mainloop.h"
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = t0 + (wn * TN + j) * 16 + lane % 16;
        if (t >= n_tok) continue;
        const float st = sx[t];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            if (OH) {
                const int row0 = m0 + (wm * TM + i) * 16 + 8 * (lane / 16);
                _Float16 *dh = (_Float16 *)Y + (size_t)t * nrows + row0;
                if (row0 + 8 <= nrows && (nrows & 7) == 0) {
                    half8 hv;
#pragma unroll
                    for (int v = 0; v < 8; ++v) hv[v] = (_Float16)(acc[i][j][v] * ldexpf(st, (int)wref[row0 + v] - 127));
                    *(half8 *)dh = hv;
                } else {
#pragma unroll
                    for (int v = 0; v < 8; ++v)
                        if (row0 + v < nrows) dh[v] = (_Float16)(acc[i][j][v] * ldexpf(st, (int)wref[row0 + v] - 127));
                }
                continue;
            }
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + 8 * (lane / 16) + v;
                if (row < nrows) {
                    const float y = acc[i][j][v] * ldexpf(st, (int)wref[row] - 127);
                    float *dst = Y + (size_t)t * nrows + row;
                    *dst = accumulate ? *dst + y : y;
                }
            }
        }
    }
}

// Configurations (must match the gemm8 table in kernels_abi.h): (BM, BN, BK, WAVES_M, threads)
#define GEMM8_ENTRY(I, BM, BN, BK, WM, NTH)                                                                   \
    extern "C" __global__ void __launch_bounds__(NTH)                                                         \
        gemm8_c##I(const uint8_t *W, uint64_t row_bytes, const uint8_t *wref, const uint8_t *X8, const float *sx, \
                   float *Y, int ncols, int nrows, int n_tok, int accumulate) {                               \
        gemm8_impl<BM, BN, BK, WM, NTH>(W, row_bytes, wref, X8, sx, Y, ncols, nrows, n_tok, accumulate);        \
    }
GEMM8_ENTRY(0, 128, 256, 128, 2, 256)
GEMM8_ENTRY(1, 128, 128, 128, 2, 256)
GEMM8_ENTRY(2, 256, 128, 64, 4, 256)
GEMM8_ENTRY(3, 128, 64, 128, 2, 256)
GEMM8_ENTRY(4, 64, 128, 128, 1, 256)
#define GEMM8H_ENTRY(I, BM, BN, BK, WM, NTH)                                                                  \
    extern "C" __global__ void __launch_bounds__(NTH)                                                         \
        gemm8h_c##I(const uint8_t *W, uint64_t row_bytes, const uint8_t *wref, const uint8_t *X8, const float *sx, \
                    float *Y, int ncols, int nrows, int n_tok, int accumulate) {                              \
        gemm8_impl<BM, BN, BK, WM, NTH, true>(W, row_bytes, wref, X8, sx, Y, ncols, nrows, n_tok, 0);          \
    }
GEMM8H_ENTRY(0, 128, 256, 128, 2, 256)
GEMM8H_ENTRY(1, 128, 128, 128, 2, 256)
GEMM8H_ENTRY(2, 256, 128, 64, 4, 256)
GEMM8H_ENTRY(3, 128, 64, 128, 2, 256)
GEMM8H_ENTRY(4, 64, 128, 128, 1, 256)

// silu(g) * u from f16 gate / up rows (gemm8h outputs) -> fp8 + per-token scale.
template <bool TL>
__device__ void silu_mul_x8h_impl(const _Float16 *g, const _Float16 *u, uint8_t *q, float *sx, int ncols) {
    const _Float16 *gr = g + (size_t)blockIdx.x * ncols, *ur = u + (size_t)blockIdx.x * ncols;
    const int tid = threadIdx.x;
    float y[SM_PER4 / 2][8];
    float m = 0.f;
#pragma unroll
    for (int k = 0; k < SM_PER4 / 2; ++k) {
        const int i = (tid + k * 256) * 8;
        if (i < ncols) {
            const half8 a = *(const half8 *)(gr + i), b = *(const half8 *)(ur + i);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float gv = (float)a[e];
                y[k][e] = gv / (1.f + __expf(-gv)) * (float)b[e];
                m = fmaxf(m, fabsf(y[k][e]));
            }
        }
    }
    m = block_max(m);
    const float inv = m > 0.f ? 448.f / m : 0.f;
    if (tid == 0) sx[blockIdx.x] = m > 0.f ? m / 448.f : 0.f;
    uint8_t *qr = q + (size_t)blockIdx.x * ncols;
#pragma unroll
    for (int k = 0; k < SM_PER4 / 2; ++k) {
        const int i = (tid + k * 256) * 8;
        if (i < ncols) {
            int p[2];
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const float a = fminf(fmaxf(y[k][4 * h2] * inv, -448.f), 448.f), b = fminf(fmaxf(y[k][4 * h2 + 1] * inv, -448.f), 448.f);
                const float c = fminf(fmaxf(y[k][4 * h2 + 2] * inv, -448.f), 448.f), d = fminf(fmaxf(y[k][4 * h2 + 3] * inv, -448.f), 448.f);
                int w = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
                p[h2] = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
            }
            if (TL) *(int2 *)(q + x8t_off(blockIdx.x, i, ncols)) = make_int2(p[0], p[1]);
            else *(int2 *)(qr + i) = make_int2(p[0], p[1]);
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) silu_mul_x8h(const _Float16 *g, const _Float16 *u, uint8_t *q, float *sx, int ncols) {
    silu_mul_x8h_impl<false>(g, u, q, sx, ncols);
}

