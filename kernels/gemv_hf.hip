// WHIRL device code (gfx1201 / RDNA 4): precise decode / verify gemvh on f32 activations (gemvhf*) and the gemvh tile
// the fused MoE decode kernel (moe_fused.hip) runs.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at the END of the gfx1201 block of whirl_kernels.hip (after gemv_h.hip / gemv_hv.hip, so
// no existing kernel moves). Uses dec_group_h<T>, t2_sbb<T>, half8 / float8, t2_u32x4 (gemm_prefill.hip / gemm_small.hip
// / common.hip) and gemms_stage_chunks / gemms_chunk_needed (gemm_small_addr.h, included by gemv_h.hip); family
// kernels.gemm (kernels/families.json).
// Port: v0252-c3port from feat026-moedec-p25 0ae5530 (kernels/gemv_h.hip hunks), gfx1201 only (DISCOVERY 2e C3). The
// branch made gemvh_impl a template on the activation type and added gemvh_tile to gemv_h.hip; here the shipped
// gemv_h.hip is NOT touched: gemvh_tile is a copy of gemvh_impl's tile loop (the same fragments, k order and WMMA chain)
// and the gemvhf entries are built from it, so every existing gemvh kernel's machine code stays as it is.
//
// gemvhf_<T> / gemvhf2_<T>: gemvh_<T> / gemvh2_<T> on f32 x - each B element rounded to f16 in the load with (_Float16)v,
// round to nearest even, exactly as f32_to_f16 rounds it - so y is bitwise f32_to_f16 + gemvh (kt moefuse, gemvhf rows,
// also against gemm_c0). Saves the f32_to_f16 launch of a precise decode / verify matmul whose input is not in the x16
// cache (WHIRL_GEMVH_F32, ops::gemvhF32). No f16 entry: with f16 weights the f32-activation variant spills (256 VGPR,
// about 150 spilled, feat026-moedec), so f16 keeps f32_to_f16 + gemvh.

// B operand loads of gemvh_tile: the f16 activations as stored, or f32 activations rounded to f16 as f32_to_f16 does
__device__ __forceinline__ half8 gemvh_ldb(const _Float16 *p) { return *(const half8 *)p; }
__device__ __forceinline__ half8 gemvh_ldb(const float *p) {
    const float4 a = ((const float4 *)p)[0], b = ((const float4 *)p)[1];
    return (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w, (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
}
// LDS bytes of one wave's staged rows (16 rows x SLOT)
template <int T, int S> constexpr int gemvh_slot() { return gemms_stage_chunks(S * t2_sbb<T>()) * 16; }

// A copy of gemvh_impl's tile loop (gemv_h.hip): one wave's 16-row tile starting at row m0 (< nrows), the accumulators
// acc[j] (tokens j * 16 + lane % 16; lane holds rows m0 + 8 * (lane / 16) + v). Ws: this wave's 16 * gemvh_slot<T, S>()
// LDS bytes. The __syncthreads are block-wide: every wave of the block must run the same number of stages (same ncols).
// S = 1 and S = 2 run the same k order (S only groups stages per LDS round), so the same bits.
template <int T, int TN, int S, typename XT>
__device__ __forceinline__ void gemvh_tile(uint8_t *__restrict__ Ws, const uint8_t *__restrict__ W, uint64_t row_bytes,
                                           const XT *__restrict__ X, int ncols, int nrows, int n_tok, int m0, float8 (&acc)[TN]) {
    constexpr int SB = S * t2_sbb<T>();  // S super-block stages (S * 256 k) per LDS round
    constexpr int CH = gemms_stage_chunks(SB);  // 16-byte chunks per row stage (covers the misaligned start)
    constexpr int SLOT = CH * 16;
    constexpr int WCH = 16 * CH;
    constexpr int WPER = (WCH + WARP - 1) / WARP;
    const int lane = threadIdx.x % WARP;
    const int nst = ncols / (256 * S);
    const int g0 = lane / 16;
    const int rl = lane % 16;
    const int row = min(m0 + rl, nrows - 1);
    const XT *xr[TN];
    bool xv[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const int t = j * 16 + lane % 16;
        xv[j] = t < n_tok;
        xr[j] = X + (size_t)min(t, n_tok - 1) * ncols + 8 * g0;
    }
    t2_u32x4 rw[WPER];
    size_t rbase[WPER];
#pragma unroll
    for (int e = 0; e < WPER; ++e) {
        const int idx = min(lane + e * WARP, WCH - 1);
        rbase[e] = (size_t)min(m0 + idx / CH, nrows - 1) * row_bytes;
    }
    auto fetch = [&](int js) {
#pragma unroll
        for (int e = 0; e < WPER; ++e) {
            const int idx = lane + e * WARP;
            if (WCH % WARP == 0 || idx < WCH) {
                const size_t st = rbase[e] + (size_t)js * SB;
                // only chunks overlapping the stage: the matrix may end right after it (gemm_small_addr.h)
                rw[e] = gemms_chunk_needed(st, idx % CH, SB) ? *(const t2_u32x4 *)(W + (st & ~(size_t)15) + (idx % CH) * 16)
                                                             : (t2_u32x4){0, 0, 0, 0};
            }
        }
    };
    fetch(0);
    for (int js = 0; js < nst; ++js) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < WPER; ++e) {
            const int idx = lane + e * WARP;
            if (WCH % WARP == 0 || idx < WCH) *(t2_u32x4 *)&Ws[(idx / CH) * SLOT + (idx % CH) * 16] = rw[e];
        }
        __syncthreads();
        if (js + 1 < nst) fetch(js + 1);
        const uint8_t *wp = Ws + rl * SLOT + (int)(((size_t)row * row_bytes + (size_t)js * SB) & 15);
#pragma unroll
        for (int kk = 0; kk < 16 * S; ++kk) {
            const half8 a = dec_group_h<T>(wp, 2 * kk + g0);
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                half8 b = gemvh_ldb(xr[j] + js * 256 * S + kk * 16);
                if (!xv[j]) b = (half8){0, 0, 0, 0, 0, 0, 0, 0};
                acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[j]);
            }
        }
    }
}

// gemvhf: gemvh_impl's body (one wave = one block per 16 rows, the tile above on f32 x, the same store)
template <int T, int TN, int S>
__device__ __forceinline__ void gemvhf_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const float *__restrict__ X,
                                            float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    __shared__ __attribute__((aligned(16))) uint8_t Ws[16 * gemvh_slot<T, S>()];
    const int lane = threadIdx.x % WARP;
    const int m0 = blockIdx.x * 16;
    if (m0 >= nrows) return;
    const int g0 = lane / 16;
    float8 acc[TN];
    gemvh_tile<T, TN, S, float>(Ws, W, row_bytes, X, ncols, nrows, n_tok, m0, acc);
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = j * 16 + lane % 16;
        if (t >= n_tok) continue;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int r = m0 + 8 * g0 + v;
            if (r < nrows) {
                float *dst = Y + (size_t)t * nrows + r;
                *dst = accumulate ? *dst + acc[j][v] : acc[j][v];
            }
        }
    }
}

#define GEMVHF_TYPE(SUF, T)                                                                                                     \
    extern "C" __global__ void __launch_bounds__(32) gemvhf_##SUF(const uint8_t *W, uint64_t row_bytes, const float *X, float *Y,  \
                                                                    int ncols, int nrows, int n_tok, int accumulate) {          \
        if (ncols % 512 == 0) gemvhf_impl<T, 1, 2>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                         \
        else gemvhf_impl<T, 1, 1>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                          \
    }                                                                                                                           \
    extern "C" __global__ void __launch_bounds__(32) gemvhf2_##SUF(const uint8_t *W, uint64_t row_bytes, const float *X, float *Y, \
                                                                     int ncols, int nrows, int n_tok, int accumulate) {         \
        if (ncols % 512 == 0) gemvhf_impl<T, 2, 2>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                         \
        else gemvhf_impl<T, 2, 1>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                          \
    }

GEMVHF_TYPE(q8_0, QT_Q8_0)
GEMVHF_TYPE(q3_k, QT_Q3_K)
GEMVHF_TYPE(q4_k, QT_Q4_K)
GEMVHF_TYPE(q5_k, QT_Q5_K)
GEMVHF_TYPE(q6_k, QT_Q6_K)
GEMVHF_TYPE(iq4_nl, QT_IQ4_NL)
GEMVHF_TYPE(iq3_s, QT_IQ3_S)
GEMVHF_TYPE(iq4_xs, QT_IQ4_XS)
GEMVHF_TYPE(mxfp4, QT_MXFP4)
