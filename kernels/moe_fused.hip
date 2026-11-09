// WHIRL device code (gfx1201 / RDNA 4): precise-mode MoE decode, routed + shared expert gate/up and SwiGLU in one
// launch (feat026-moedec, kernel technique audit gap #4, research-moedec levers L1 + L3).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at the END of the gfx1201 block of whirl_kernels.hip, after moe_f.hip (mxf_blk) and
// gemv_hf.hip (gemvh_tile / gemvh_slot); warp_sum, WARP, half8 / float8 from common.hip. Family kernels.moe.
// Port: v0252-c3port from feat026-moedec-p25 0ae5530 (kernels/moe_fused.hip), gfx1201 only (DISCOVERY 2e C3): the
// gfx1151 branches (paired tile) are dropped, the file sits inside the `#if !defined(__gfx1151__)` block, and the
// shared kernels/moe_f.hip is not touched (the branch's mxf_gu_dot refactor of moe_gu_f_mxfp4 is replaced by the copy
// mfu_gu_dot below, so moe_gu_f_mxfp4's machine code stays as shipped).
//
// moe_gus_f_mxfp4 (n_tok <= 16) / moe_gus2_f_mxfp4 (n_tok <= 32) replace, in precise decode / verify of a MoE layer
// with MXFP4 routed and shared experts, the chain
//     moe_gu_f_mxfp4 -> silu_mul (routed)
//     f32_to_f16(h) -> gemvh(sg) -> gemvh(su) -> silu_mul -> f32_to_f16 (shared)
// by one launch, bit for bit:
//   * blocks [0, nsb): the shared expert. Block b runs the 16-row gemvh tiles 4b..4b+3 of the shared gate matrix
//     (waves 0-3) and of the shared up matrix (waves 4-7) with the gemvh tile code (gemvh_tile: same fragments, same k
//     order, same WMMA chain), reading the f32 h and rounding every B element with (_Float16)v as f32_to_f16 does.
//     The up wave hands its accumulators lane by lane to its gate partner through LDS; the gate wave stores
//     (_Float16)(g / (1 + exp(-g)) * u) with the f32 product materialized first - silu_mul then f32_to_f16, the
//     input of the shared down gemvh (xs: [n_tok][sff] f16).
//   * blocks [nsb, nsb + nrb * npairs): the routed experts. Block nsb + p * nrb + i runs rows 4i..4i+3 of pair p's
//     gate matrix (waves 0-3) and of its up matrix (waves 4-7) with mfu_gu_dot (the code of moe_gu_f_mxfp4: lanes
//     over 32-value blocks, warp_sum); the up sums go through LDS and the gate wave stores silu_mul's
//     g / (1 + exp(-g)) * u into act[p][row] (= moe_g after moe_gu_f + silu_mul).
// The shared blocks come first: they are the latency-bound part (dependent LDS stages per wave), so they start before
// the routed blocks fill the GPU. Each block's bits are those of the kernels it replaces; only the scheduling changes.
// Requirements (checked by the host, ops::chooseMoeFuse): ff % 4 == 0, sff % 64 == 0 (whole groups of 4 tiles),
// C % 256 == 0 (gemvh), n_tok <= 16 * TN.

// the LDS-staged tile with one 256-k stage per round (S = 1: the same k order as gemvh's S = 2, so the same bits,
// half the LDS); 8 waves x 16 rows x gemvh_slot; the up wave's slice is reused as its exchange buffer
#define MFU_WS (16 * gemvh_slot<QT_MXFP4, 1>())
#define MFU_LDS_BYTES (8 * MFU_WS)

// silu_mul's value g / (1 + __expf(-g)) * u (kernels/shared/misc_rows_norm_l2norm.h), materialized in f32 (no
// contraction into a later fma / f16 round)
__device__ __forceinline__ float mfu_swiglu(float g, float u) {
    float y = g / (1.f + __expf(-g)) * u;
    asm volatile("" : "+v"(y));
    return y;
}

// one gate / up row of one pair: moe_gu_f_mxfp4's sum (moe_f.hip: lane b, b + 32, ... over the row's 32-value blocks,
// mxf_blk, warp_sum), every lane holding the result
__device__ __forceinline__ float mfu_gu_dot(const uint8_t *__restrict__ wr, const float *__restrict__ x, int C) {
    const int lane = threadIdx.x % WARP;
    float acc = 0.f;
    for (int b = lane; b < C / 32; b += WARP) acc += mxf_blk(wr, b, x);
    return warp_sum(acc);
}

template <int TN>
__device__ __forceinline__ void mfu_shared(uint8_t *__restrict__ smem, const uint8_t *__restrict__ SG, const uint8_t *__restrict__ SU,
                                           uint64_t s_row_bytes, int sff, const float *__restrict__ h, int C, int n_tok,
                                           _Float16 *__restrict__ xs) {
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, q = w & 3;
    const bool up = w >= 4;
    const int m0 = (blockIdx.x * 4 + q) * 16;
    float8 acc[TN];
    gemvh_tile<QT_MXFP4, TN, 1, float>(smem + w * MFU_WS, up ? SU : SG, s_row_bytes, h, C, sff, n_tok, m0, acc);
    __syncthreads();  // every wave is past its last staged-row read
    float *xch = (float *)(smem + (q + 4) * MFU_WS);  // the up wave's own slice
    if (up) {
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) xch[(j * 8 + v) * WARP + lane] = acc[j][v];
    }
    __syncthreads();
    if (up) return;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = j * 16 + lane % 16;
        if (t >= n_tok) continue;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int r = m0 + 8 * (lane / 16) + v;
            if (r < sff) xs[(size_t)t * sff + r] = (_Float16)mfu_swiglu(acc[j][v], xch[(j * 8 + v) * WARP + lane]);
        }
    }
}

__device__ __forceinline__ void mfu_routed(uint8_t *__restrict__ smem, int bi, const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2,
                                           uint64_t row_bytes, int ff, int nrb, const float *__restrict__ h, const int *__restrict__ ids,
                                           float *__restrict__ act, int C, int K) {
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, q = w & 3;
    const bool second = w >= 4;
    const int p = bi / nrb;
    const int r = (bi - p * nrb) * 4 + q;
    const int t = p / K, e = ids[p];
    const uint8_t *wr = (second ? W2 : W1) + ((size_t)e * ff + r) * row_bytes;
    const float acc = mfu_gu_dot(wr, h + (size_t)t * C, C);
    float *us = (float *)smem;
    if (second && lane == 0) us[q] = acc;
    __syncthreads();
    if (!second && lane == 0) act[(size_t)p * ff + r] = mfu_swiglu(acc, us[q]);
}

#define MOE_GUS_ENTRY(NAME, TN)                                                                                                     \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                                                          \
        const uint8_t *W1, const uint8_t *W2, uint64_t row_bytes, int ff, int nrb, const float *h, const int *ids, float *act, int C, \
        int K, const uint8_t *SG, const uint8_t *SU, uint64_t s_row_bytes, int sff, int nsb, _Float16 *xs, int n_tok) {                \
        __shared__ __attribute__((aligned(16))) uint8_t smem[MFU_LDS_BYTES];                                                         \
        if ((int)blockIdx.x < nsb) mfu_shared<TN>(smem, SG, SU, s_row_bytes, sff, h, C, n_tok, xs);                                 \
        else mfu_routed(smem, (int)blockIdx.x - nsb, W1, W2, row_bytes, ff, nrb, h, ids, act, C, K);                               \
    }

MOE_GUS_ENTRY(moe_gus_f_mxfp4, 1)
MOE_GUS_ENTRY(moe_gus2_f_mxfp4, 2)
