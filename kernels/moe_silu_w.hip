// WHIRL device code (gfx1201 / RDNA 4): silu_mul_x8h for short rows - one wave32 per row (v0252-silu, PLAN V1-77
// follow-up; the fp8 MoE prefill activation of balance / fast, F = 512 at Ornith-1.5-35B-A3B).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at the END of the gfx1201 block of whirl_kernels.hip (no existing kernel moves);
// WARP, warp_max, half8 from common.hip. Family kernels.moe.
//
// silu_mul_x8h (gemm_fp8.hip) runs one 256-thread block per row with 8 values per thread: at F = 512 only 64 threads
// (2 of 8 waves) load anything, and the row max goes through block_max (3 __syncthreads + LDS) for all 8 waves
// (V1-77: 6.7 ms RGP per 2048-row Ornith forward, about 2.2x its byte floor). Here one wave owns one row, each lane
// 16 contiguous values per chunk (2 x 16 B of g, 2 x 16 B of u, one 16 B store of q), the row max is warp_max (DPP /
// shuffles, no LDS, no barrier), WPB rows per block (TRT-LLM doActivationKernel sizes its block to the row the same way
// when it does not need a block-wide amax; nv/flashinfer cutlass_fused_moe_kernels.cuh:2690-2711).
//
// Bit-identity with silu_mul_x8h (the kt family "silu" checks every byte and every sx bit):
//   y      the same source expression on the same f16 inputs, same TU and flags: gv / (1.f + __expf(-gv)) * (float)b
//   m      max over |y| from 0 with fmaxf: exact and order independent (fmaxf drops NaN in any order; no -0 occurs
//          since |y| >= +0 and m starts at +0), so warp_max over lane partials == block_max over thread partials
//   sx/inv the same expressions of m; clamp and the cvt_pk_fp8 pairs (a,b low word, c,d high word, per 4 values) as is
//   q      row-major [row][ncols] like silu_mul_x8h (not the TL layout)
// Shape: ncols % 16 == 0 and ncols <= kSiluWMaxF (kernels::siluWShapeOk); nrows = rows; grid ceil(nrows / WPB),
// block WPB * 32. Rows >= nrows are not touched.
//
// SILU_MUT (default 0 = the product) are the kt RED controls built only by scratch\v0252_silu\mk_mutants.ps1:
//   1  warp_max without its last xor step (lanes 0-15 and 16-31 reduce apart) -> wrong sx / q when the max is in the
//      other half
//   2  stub: q = 0 and sx = 0 for every row
//   3  SwiGLU as (g * u) / (1 + e^-g) (multiply before the divide: other rounding) -> wrong q bytes in some rows

#ifndef SILU_MUT
#define SILU_MUT 0
#endif
#define SILUW_CH 2  // chunks of 32 lanes x 16 values: ncols <= 1024 (= kSiluWMaxF in kernels_abi.h)

template <int WPB>
__device__ void silu_mul_x8w_impl(const _Float16 *g, const _Float16 *u, uint8_t *q, float *sx, int ncols, int nrows) {
    const int lane = threadIdx.x % WARP, row = blockIdx.x * WPB + threadIdx.x / WARP;
    if (row >= nrows) return;  // whole wave; no block barrier below
    const _Float16 *gr = g + (size_t)row * ncols, *ur = u + (size_t)row * ncols;
    float y[SILUW_CH][16];
    float m = 0.f;
#pragma unroll
    for (int k = 0; k < SILUW_CH; ++k) {
        const int i = (lane + k * WARP) * 16;
        if (i < ncols) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const half8 a = *(const half8 *)(gr + i + 8 * h), b = *(const half8 *)(ur + i + 8 * h);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float gv = (float)a[e];
#if SILU_MUT == 3
                    y[k][8 * h + e] = gv * (float)b[e] / (1.f + __expf(-gv));
#else
                    y[k][8 * h + e] = gv / (1.f + __expf(-gv)) * (float)b[e];
#endif
                    m = fmaxf(m, fabsf(y[k][8 * h + e]));
                }
            }
        }
    }
#if SILU_MUT == 1
#pragma unroll
    for (int o = WARP / 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, WARP));
#else
    m = warp_max(m);
#endif
    const float inv = m > 0.f ? 448.f / m : 0.f;
#if SILU_MUT == 2
    if (lane == 0) sx[row] = 0.f;
#else
    if (lane == 0) sx[row] = m > 0.f ? m / 448.f : 0.f;
#endif
    uint8_t *qr = q + (size_t)row * ncols;
#pragma unroll
    for (int k = 0; k < SILUW_CH; ++k) {
        const int i = (lane + k * WARP) * 16;
        if (i < ncols) {
            int p[4];
#pragma unroll
            for (int h2 = 0; h2 < 4; ++h2) {
                const float a = fminf(fmaxf(y[k][4 * h2] * inv, -448.f), 448.f), b = fminf(fmaxf(y[k][4 * h2 + 1] * inv, -448.f), 448.f);
                const float c = fminf(fmaxf(y[k][4 * h2 + 2] * inv, -448.f), 448.f), d = fminf(fmaxf(y[k][4 * h2 + 3] * inv, -448.f), 448.f);
                int w = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
                p[h2] = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
            }
#if SILU_MUT == 2
            p[0] = p[1] = p[2] = p[3] = 0;
#endif
            *(int4 *)(qr + i) = make_int4(p[0], p[1], p[2], p[3]);
        }
    }
}
// silu_mul_x8w: 8 rows per 256-thread block; silu_mul_x8w4: 4 rows per 128-thread block (the kt bench picks)
extern "C" __global__ void __launch_bounds__(256) silu_mul_x8w(const _Float16 *g, const _Float16 *u, uint8_t *q, float *sx, int ncols, int nrows) {
    silu_mul_x8w_impl<8>(g, u, q, sx, ncols, nrows);
}
extern "C" __global__ void __launch_bounds__(128) silu_mul_x8w4(const _Float16 *g, const _Float16 *u, uint8_t *q, float *sx, int ncols, int nrows) {
    silu_mul_x8w_impl<4>(g, u, q, sx, ncols, nrows);
}
#undef SILUW_CH
