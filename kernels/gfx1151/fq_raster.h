// gfx1151 fused-dequant GEMM (gemm_fq.hip) block order: grouped raster (v0252-8sbal, PLAN V1-53).
// SPDX-License-Identifier: Apache-2.0
//
// Host + device (the CPU unit tests in tests/unit/kernels_gemm_tests.cpp run the same function). The grid stays the
// gemmfq grid (x = token blocks, y = row blocks; dispatch order = linear id y * nx + x). grp = 0 keeps that order (row
// tile = y, token tile = x: consecutive blocks share one weight row tile and walk ALL token tiles, so the whole
// activation matrix is live between two reuses of a token tile). grp > 0 walks groups of grp row tiles: inside a group
// the row tile changes fastest, so consecutive blocks share one activation (token) tile and grp neighbouring weight
// row tiles (Gufo prefill_quant_gemm.hpp:207-223, "reuse each activation panel across up to eight neighboring row
// tiles", MIT). The last group may hold fewer rows; every (row, token) tile keeps exactly one owner. Pure index
// arithmetic: the tile each block computes changes, never how it computes it (bitwise output).
#pragma once

#if defined(__HIPCC__)
#define FQR_HD __host__ __device__ __forceinline__
#else
#define FQR_HD inline
#endif

FQR_HD void fq_group_tile(unsigned bx, unsigned by, unsigned nx, unsigned ny, unsigned grp, unsigned* row, unsigned* tok) {
    if (grp == 0) {
        *row = by;
        *tok = bx;
        return;
    }
    const unsigned lin = by * nx + bx;
    const unsigned first = lin / (grp * nx) * grp;
    const unsigned rows = ny - first < grp ? ny - first : grp;
    const unsigned within = lin - first * nx;
    *row = first + within % rows;
    *tok = within / rows;
}
