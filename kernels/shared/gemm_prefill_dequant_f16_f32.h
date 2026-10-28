// WHIRL device code, shared by both kernel sets (gfx1201 kernels/gemm_prefill.hip, gfx1151 kernels/gfx1151/gemm_prefill.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

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
// (BM, BN, BK, WAVES_M, threads) must match the gemm table in kernels_abi.h. All are
// spill-free for every quant type (checked with -Rpass-analysis=kernel-resource-usage).
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
