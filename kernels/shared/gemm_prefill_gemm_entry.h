// WHIRL device code, shared by both kernel sets (gfx1201 kernels/gemm_prefill.hip, gfx1151 kernels/gfx1151/gemm_prefill.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

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
// Prefill matmul on the RDNA4 matrix cores (v_wmma_f32_16x16x16_f16).
// Block tile: BM weight rows x BN tokens; K staged BK at a time in LDS as f16.
// Weights are dequantized on the fly; activations arrive pre-converted to f16
// (f32_to_f16) so each row-block streams half the bytes and does no converts.
// gfx12 wave32 fragment layout (see llama.cpp ggml-cuda/mma.cuh):
//   A, B: lane holds row (lane % 16), k = 8 * (lane / 16) + [0, 8)
//   D:    lane holds column n = lane % 16, rows m = 8 * (lane / 16) + [0, 8)
// Requires ncols % BK == 0.
// ---------------------------------------------------------------------------
