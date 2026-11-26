// gfx1151 (Radeon 8060S) W4A4 FAKE-QUANT arm of the fast dense prefill GEMM (v0252-w4a4k2, PLAN V1-144 kill-screen K2).
// SPDX-License-Identifier: Apache-2.0
//
// A fragment of the gfx1151 translation unit (kernels/gfx1151/whirl_kernels_gfx1151.hip includes it LAST, so no existing
// kernel moves). Family kernels.gemm (kernels/families.json glob kernels/gfx1151/gemm_*.hip). Research arm only: numerics flag
// w4a4_sim (WHIRL_W4A4_SIM, fast mode, gfx1151, dense MXFP4 models; default off), host MatmulKernels::w4s_dq / w4s_qx,
// ops::chooseGemmF16 (GemmFlags::w4s_w) and Context::matmul. Never used by precise / balance or by any default plan.
// The math (codes, scales, f16 rounding) is kernels/shared/gemm_w4a4_sim.h, pinned to the W4A4 probe's CPU definitions
// (tests/kernels/kt_w4a4p.h) by whirl-kt-bench-tests and to its own host reference by whirl-kernel-test --only w4s.
//   w4s_dequant_f16_mxfp4[_w1|_w1e|_w1d|_wb](W, row_bytes, out, ncols, nrows) (one per weight grid): the dequant_f16_<T> ABI (the plan's dequant pass into the f16
//     weight scratch); every 32-value MXFP4 sub-block -> f16(int4 level x 2^(e - 128) x 12 / 7). One thread per sub-block,
//     grid-stride (any grid). Requires ncols % 256 == 0 (whole superblocks).
//   w4s_quant_x16(X, Xq, nblk, mse): nblk 32-value blocks of f16 activations (rows of K % 32 == 0 values, back to back) ->
//     their per-32 int4 round trip in f16 (mse 0: step amax / 7; 1: the MSE-best step). One thread per block, grid-stride.
// Speed is irrelevant (K2 measures quality only): the GEMM after them is the shipped f16 GEMM on the dequantized values.

#include "shared/gemm_w4a4_sim.h"

// round 2: one entry per weight grid (gemm_w4a4_sim.h W4S_GRIDS; host MatmulKernels::w4s_dq[g], kernels::kW4sDqNames)
template <int G>
__device__ void w4s_dequant_impl(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) {
    const size_t spr = static_cast<size_t>(ncols) / 32;
    const size_t total = spr * static_cast<size_t>(nrows);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / spr, b = i % spr;
        const uint8_t *blk = W + r * row_bytes + (b / 8) * 136;
        const int ib = static_cast<int>(b % 8);
        float v[32];
        w4s_wblock(blk + 8 + 16 * ib, blk[ib], G, v);
        _Float16 *o = out + r * static_cast<size_t>(ncols) + b * 32;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            float p = v[k];
            W4S_F32(p);  // round 2: f32 product, then f16 (no single-rounding v_fma_mix fold)
            o[k] = (_Float16)p;
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) w4s_dequant_f16_mxfp4(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { w4s_dequant_impl<0>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) w4s_dequant_f16_mxfp4_w1(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { w4s_dequant_impl<1>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) w4s_dequant_f16_mxfp4_w1e(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { w4s_dequant_impl<2>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) w4s_dequant_f16_mxfp4_w1d(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { w4s_dequant_impl<3>(W, row_bytes, out, ncols, nrows); }
extern "C" __global__ void __launch_bounds__(256) w4s_dequant_f16_mxfp4_wb(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows) { w4s_dequant_impl<4>(W, row_bytes, out, ncols, nrows); }

// round 3: the least-squares gain of the weight grids. w4s_gain_terms: one thread per row, num / den of the row on grid g
// (gemm_w4a4_sim.h w4s_gain_terms_row, double; the host turns them into gains, ops::w4sGains). The _gn entries = the plain
// ones with the row's gain multiplied into the f32 value before the f16 store (gain[r]; a tensor gain = the same value in
// every row). Launch: (nrows + 63) / 64 x 64 threads for the terms, the dequant grid for the _gn entries.
extern "C" __global__ void __launch_bounds__(64) w4s_gain_terms(const uint8_t *W, uint64_t row_bytes, int ncols, int nrows, int grid, double *num,
                                                              double *den) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    double n, d;
    w4s_gain_terms_row(W + static_cast<size_t>(r) * row_bytes, ncols, grid, &n, &d);
    num[r] = n;
    den[r] = d;
}
template <int G>
__device__ void w4s_dequant_gain_impl(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows, const float *gain) {
    const size_t spr = static_cast<size_t>(ncols) / 32;
    const size_t total = spr * static_cast<size_t>(nrows);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / spr, b = i % spr;
        const uint8_t *blk = W + r * row_bytes + (b / 8) * 136;
        const int ib = static_cast<int>(b % 8);
        float v[32];
        w4s_wblock(blk + 8 + 16 * ib, blk[ib], G, v);
        const float gr = gain[r];
        _Float16 *o = out + r * static_cast<size_t>(ncols) + b * 32;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            float p = v[k] * gr;
            W4S_F32(p);  // f32 product, then f16 (the round-2 rule)
            o[k] = (_Float16)p;
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) w4s_dequant_f16_mxfp4_gn(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows, const float *gain) { w4s_dequant_gain_impl<0>(W, row_bytes, out, ncols, nrows, gain); }
extern "C" __global__ void __launch_bounds__(256) w4s_dequant_f16_mxfp4_w1_gn(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows, const float *gain) { w4s_dequant_gain_impl<1>(W, row_bytes, out, ncols, nrows, gain); }
extern "C" __global__ void __launch_bounds__(256) w4s_dequant_f16_mxfp4_w1e_gn(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows, const float *gain) { w4s_dequant_gain_impl<2>(W, row_bytes, out, ncols, nrows, gain); }
extern "C" __global__ void __launch_bounds__(256) w4s_dequant_f16_mxfp4_w1d_gn(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows, const float *gain) { w4s_dequant_gain_impl<3>(W, row_bytes, out, ncols, nrows, gain); }
extern "C" __global__ void __launch_bounds__(256) w4s_dequant_f16_mxfp4_wb_gn(const uint8_t *W, uint64_t row_bytes, _Float16 *out, int ncols, int nrows, const float *gain) { w4s_dequant_gain_impl<4>(W, row_bytes, out, ncols, nrows, gain); }

extern "C" __global__ void __launch_bounds__(256) w4s_quant_x16(const _Float16 *X, _Float16 *Xq, uint64_t nblk, int mse) {
    for (uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; g < nblk; g += (uint64_t)gridDim.x * blockDim.x) {
        float x[32];
        int q[32];
        float d;
#pragma unroll
        for (int i = 0; i < 32; ++i) x[i] = (float)X[g * 32 + i];
        w4s_qblock(x, mse, q, &d);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            float p = static_cast<float>(q[i]) * d;
            W4S_F32(p);  // round 2: f32 product, then f16 (round 1: a v_fma_mix single rounding = the MSE-row mismatches)
            Xq[g * 32 + i] = (_Float16)p;
        }
    }
}
