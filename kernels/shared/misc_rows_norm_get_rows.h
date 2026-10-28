// WHIRL device code, shared by both kernel sets (gfx1201 kernels/misc_rows_norm.hip, gfx1151 kernels/gfx1151/fused_decode.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

template <int T>
__device__ void get_rows_impl(const uint8_t *W, uint64_t row_bytes, const int *ids, float *out, int ncols) {
    const int t = blockIdx.y;
    const uint8_t *wr = W + (size_t)ids[t] * row_bytes;
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < ncols / 8; g += gridDim.x * blockDim.x) {
        float w[8];
        dec_group<T>(wr, g, w);
#pragma unroll
        for (int i = 0; i < 8; ++i) out[(size_t)t * ncols + 8 * g + i] = w[i];
    }
}

#define GET_ROWS(SUF, T)                                                                               \
    extern "C" __global__ void get_rows_##SUF(const uint8_t *W, uint64_t row_bytes, const int *ids, \
                                              float *out, int ncols) {                               \
        get_rows_impl<T>(W, row_bytes, ids, out, ncols);                                             \
    }
