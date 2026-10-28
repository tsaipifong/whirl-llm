// WHIRL device code, shared by both kernel sets (gfx1201 kernels/moe.hip, gfx1151 kernels/gfx1151/moe.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

#define MOE_BM 128
template <int T, int BN>
__device__ void gemm_moe_impl(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X, float *Y, int ncols,
                              const int4 *tiles, const int *n_tiles) {
    if ((int)blockIdx.x >= *n_tiles) return;
    const int4 tl = tiles[blockIdx.x];
    gemm_tile_impl<T, MOE_BM, BN, 64, (BN == 32 ? 8 : 4), 256>(W + (size_t)tl.x * rows_e * row_bytes, row_bytes,
                                                   X + (size_t)tl.y * ncols, Y + (size_t)tl.y * rows_e, ncols, rows_e,
                                                   tl.z, blockIdx.y * MOE_BM, 0);
}
