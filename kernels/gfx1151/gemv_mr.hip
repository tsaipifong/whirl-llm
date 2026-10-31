// WHIRL device code (gfx1151 / RDNA 3.5): multi-row multi-token int8 GEMV.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Multi-row multi-token int8 GEMV (verify / batched decode, 2..16 tokens):
// each wave owns R rows, so every int8 x unit it loads (the bottleneck once
// NT grows) serves R rows, and row-independent terms (x sums) are computed
// once. Per (row, token) the arithmetic is exactly that of gemvq_impl /
// gemvq_nt_impl (same lane -> unit mapping, same float expressions in the
// same order, same warp_sum), so the results match them bitwise.
// ---------------------------------------------------------------------------
// Activation-scale reads of the shared unit decoders (gfx1151: packed scale words, xdv()); ADR-0005.
#define GMR_X0 xdv(xdt)
#define GMR_X1 xdv(xdt + 1)
#define GMR_XQ(e) xdv(xdt + (e))
#include "shared/gemv_mr_kw.h"  // MRD, MRD_KW, MRD<Q4_K / Q5_K / IQ4_XS>, both kernel sets (ADR-0005)

// Q6_K: per row 4 quarters of 2 int8x4 weights, 4 scales and d.
#include "shared/gemv_mr_q6.h"  // MRD<Q6_K>, both kernel sets (ADR-0005)
#undef GMR_X0
#undef GMR_X1
#undef GMR_XQ

// KLIB-1 (opt-in, WHIRL_Q3K_U16=1): Q3_K llama.cpp-split units (QD<QT_Q3_K_U16> in gemv_q.hip), R rows per wave
// sharing the int8 x loads. Per (row, token): the same q3u_dec / q3u_tok as gemvq_q3_k16 / gemvq_nt*_q3_k16, same
// lane -> unit mapping and order, so the results equal them bit for bit.
template <> struct MRD<QT_Q3_K_U16> {
    static constexpr int per_sb = QD<QT_Q3_K_U16>::per_sb;
    template <int NT, int R>
    static __device__ __forceinline__ void step(const uint8_t *const *wr, int u, const int8_t *xq, const float *xd,
                                                int ncols, float (*acc)[NT]) {
        Q3uW w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) w[r] = q3u_dec(wr[r], u);
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r][t] += q3u_tok(w[r], xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32));
    }
};
