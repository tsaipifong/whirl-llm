// WHIRL device code (gfx1151 / RDNA 3.5): f32-activation matvec.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Matvec: y[r] (+)= sum_c W[r][c] * x[t][c], for NT tokens at once.
// One wave per row; 8 rows per 256-thread block.
// ---------------------------------------------------------------------------

template <int T, int NT>
__device__ void gemv_impl(const uint8_t *__restrict__ W, uint64_t row_bytes,
                          const float *__restrict__ x, float *__restrict__ y,
                          int ncols, int nrows, int x_stride, int y_stride, int accumulate) {
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    const int ng = ncols / 8;
    for (int g = lane; g < ng; g += WARP) {
        float w[8];
        dec_group<T>(wr, g, w);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 *xp = (const float4 *)(x + (size_t)t * x_stride + 8 * g);
            const float4 a = xp[0], b = xp[1];
            acc[t] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w +
                      w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) {
            float *dst = y + (size_t)t * y_stride + row;
            *dst = accumulate ? *dst + s : s;
        }
    }
}

#define GEMV_ENTRY(NAME, T, NT)                                                                   \
    extern "C" __global__ void __launch_bounds__(256)                                             \
        NAME(const uint8_t *W, uint64_t row_bytes, const float *x, float *y, int ncols,           \
             int nrows, int x_stride, int y_stride, int accumulate) {                             \
        gemv_impl<T, NT>(W, row_bytes, x, y, ncols, nrows, x_stride, y_stride, accumulate);        \
    }

#define GEMV_TYPE(SUF, T)                    \
    GEMV_ENTRY(gemv_##SUF##_1, T, 1)         \
    GEMV_ENTRY(gemv_##SUF##_4, T, 4)         \
    GEMV_ENTRY(gemv_##SUF##_8, T, 8)

GEMV_TYPE(f32, QT_F32)
GEMV_TYPE(f16, QT_F16)
GEMV_TYPE(q8_0, QT_Q8_0)
GEMV_TYPE(q3_k, QT_Q3_K)
GEMV_TYPE(q4_k, QT_Q4_K)
GEMV_TYPE(q5_k, QT_Q5_K)
GEMV_TYPE(q6_k, QT_Q6_K)
GEMV_TYPE(iq4_nl, QT_IQ4_NL)
GEMV_TYPE(iq3_s, QT_IQ3_S)
GEMV_TYPE(iq4_xs, QT_IQ4_XS)
GEMV_TYPE(mxfp4, QT_MXFP4)
