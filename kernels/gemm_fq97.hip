// WHIRL device code (gfx1201 / RDNA 4): fused-dequant MXFP4 prefill GEMM gemmfq97_<v>_mxfp4 (v0252-fdq97, PLAN V1-89
// phase 1). TEST-FIRST STUB (commit 1 of the item): the entry points exist with the final ABI and grid, but every one
// writes 0.0f to the outputs it owns, so the kt gemm rows "gemmfq97_<v>_mxfp4 == dequant_f16_mxfp4 + gemmh_f16" must
// FAIL on this object (the RED of tests/kernels/test_gemm.cpp testGemmFq97). The real kernel replaces this file.
// SPDX-License-Identifier: Apache-2.0
//
// A fragment of the gfx1201 translation unit (kernels/whirl_kernels.hip includes it LAST in the gfx1201 block).
// ABI: (W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate), grid {ceil(n / 256), ceil(nrows / 128)}, 256 threads.

__device__ void gemmfq97_stub(float *__restrict__ Y, int nrows, int n_tok) {
    const int t = blockIdx.x * 256 + threadIdx.x, m0 = blockIdx.y * 128;
    if (t >= n_tok) return;
    for (int r = 0; r < 128 && m0 + r < nrows; ++r) Y[(size_t)t * nrows + m0 + r] = 0.0f;
}

#define GEMMFQ97_ENTRY(NAME)                                                                                            \
    extern "C" __global__ void __launch_bounds__(256) NAME(const uint8_t *W, uint64_t row_bytes, const _Float16 *X,     \
                                                           float *Y, int ncols, int nrows, int n_tok, int accumulate) { \
        gemmfq97_stub(Y, nrows, n_tok);                                                                                 \
    }
GEMMFQ97_ENTRY(gemmfq97_s1_mxfp4)
GEMMFQ97_ENTRY(gemmfq97_s0_mxfp4)
GEMMFQ97_ENTRY(gemmfq97_d1_mxfp4)
#undef GEMMFQ97_ENTRY
