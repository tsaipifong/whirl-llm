// WHIRL device code (gfx1201 and gfx1151): fused MoE router for small batches (feat026-router).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included last by whirl_kernels.hip (uses WARP, warp_sum and block_sum of the
// active kernel set's common.hip).
//
// moe_router_fused_i<ITER> = moe_logits_f32 + moe_topk in one launch, bit-identical to the pair:
//   * logits: one wave per router row, 8 rows per block, lane L sums c = L, L+32, ... in order with
//     the same expression as moe_logits_f32, then warp_sum. Only the memory schedule differs: the
//     row's ITER float4 W loads are issued before any FMA, and for n <= 4 tokens x is staged in LDS
//     (same values, different source).
//   * top-k: the last block to finish (agent-scope atomic count; it resets the count, so the kernel
//     is self-contained under graph capture) does, per token, K rounds of a wave-level argmax with
//     the moe_topk comparator: larger value wins, equal (==) -> lower index, NaN never wins, start
//     (-inf, INT_MAX); an id outside [0, R) becomes 0. That comparator is a lexicographic maximum,
//     so any reduction tree picks the same element as moe_topk's block tree. Weights: the same
//     __expf sum in k order on one lane; sg: the same 256-thread block_sum as moe_topk.
//   * skip (ctl_stop) skips only the logits, like moe_logits_f32; the count and top-k still run on
//     the buffer, like the separate moe_topk launch.
// Limits (host checks, else the two-kernel path): blockDim 256, n <= 16, R <= 256, K <= 32,
// C % 4 == 0, ceil(C / 128) == ITER, E == C (the router input is the shared-gate input h).
// Grid: ceil(R / 8) blocks. cnt: one zeroed u32 per concurrently running router (Model::moe_rcnt).

#define MOE_RF_XS 12288  // f32 of x staged in LDS (n * C <= this: n <= 4 at C <= 3072)

template <int ITER, int NT, bool EXACT, bool LDSX>
__device__ __forceinline__ void moe_rf_row(const float *__restrict__ W, const float *__restrict__ x, const float *xs,
                                           float *__restrict__ out, int R, int C, int n, int r, int lane) {
    const int C4 = C / 4;
    const float4 *wr = (const float4 *)(W + (size_t)r * C);
    float4 wv[ITER];
#pragma unroll
    for (int i = 0; i < ITER; ++i) {
        const int c = lane + i * WARP;
        wv[i] = c < C4 ? wr[c] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
#pragma unroll
    for (int i = 0; i < ITER; ++i) {
        const int c = lane + i * WARP;
        if (c < C4) {
            const float4 w = wv[i];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (EXACT || t < n) {
                    const float4 v = LDSX ? ((const float4 *)(xs + (size_t)t * C))[c]
                                          : ((const float4 *)(x + (size_t)t * C))[c];
                    acc[t] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (EXACT || t < n) {
            const float s = warp_sum(acc[t]);
            if (lane == 0) out[(size_t)t * R + r] = s;
        }
    }
}

template <int ITER, int NT>
__device__ __forceinline__ void moe_rf_rows_lds(const float *__restrict__ W, const float *__restrict__ x, float *xs,
                                                float *__restrict__ out, int R, int C, int n, int r, int lane) {
    const int tot4 = NT * C / 4;
    for (int i = threadIdx.x; i < tot4; i += blockDim.x) ((float4 *)xs)[i] = ((const float4 *)x)[i];
    __syncthreads();
    if (r < R) moe_rf_row<ITER, NT, true, true>(W, x, xs, out, R, C, n, r, lane);
}

template <int ITER>
__device__ void moe_router_fused_impl(const float *__restrict__ W, const float *__restrict__ x, float *logits,
                                      int R, int C, int n, const int *skip, const float *__restrict__ shw,
                                      int *__restrict__ ids, float *__restrict__ w, float *__restrict__ sg, int K, int E,
                                      unsigned *cnt) {
    __shared__ __attribute__((aligned(16))) float xs[MOE_RF_XS];
    __shared__ float sel_v[256 / WARP][32];
    __shared__ int sel_i[256 / WARP][32];
    __shared__ unsigned s_last;
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, nw = blockDim.x / WARP;
    const int r = blockIdx.x * nw + wid;
    // ---- phase 1: this block's router rows ----
    if (skip == nullptr || !*skip) {
        switch (n) {
            case 1: moe_rf_rows_lds<ITER, 1>(W, x, xs, logits, R, C, n, r, lane); break;
            case 2: moe_rf_rows_lds<ITER, 2>(W, x, xs, logits, R, C, n, r, lane); break;
            case 3: moe_rf_rows_lds<ITER, 3>(W, x, xs, logits, R, C, n, r, lane); break;
            case 4: moe_rf_rows_lds<ITER, 4>(W, x, xs, logits, R, C, n, r, lane); break;
            default:
                if (r < R) moe_rf_row<ITER, 16, false, false>(W, x, nullptr, logits, R, C, n, r, lane);
                break;
        }
    }
    // ---- hand-off: the last block to arrive sees every block's logits ----
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned prev = __hip_atomic_fetch_add(cnt, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = prev == gridDim.x - 1 ? 1u : 0u;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (threadIdx.x == 0) __hip_atomic_store(cnt, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    // ---- phase 2: top-K per token, one wave per token ----
    for (int t = wid; t < n; t += nw) {
        const float *l = logits + (size_t)t * R;
        float v[256 / WARP];
#pragma unroll
        for (int j = 0; j < 256 / WARP; ++j) {
            const int idx = lane + j * WARP;
            v[j] = idx < R ? l[idx] : __builtin_nanf("");
        }
        for (int k = 0; k < K; ++k) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < 256 / WARP; ++j) {
                const int idx = lane + j * WARP;
                if (v[j] > bv || (v[j] == bv && idx < bi)) {
                    bv = v[j];
                    bi = idx;
                }
            }
#pragma unroll
            for (int o = WARP / 2; o > 0; o >>= 1) {
                const float ov = __shfl_xor(bv, o, WARP);
                const int oi = __shfl_xor(bi, o, WARP);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
            int i = bi;
            if (i < 0 || i >= R) i = 0;  // all -inf/NaN: keep a valid id (as moe_topk)
#pragma unroll
            for (int j = 0; j < 256 / WARP; ++j)
                if (lane + j * WARP == i) v[j] = -INFINITY;
            if (lane == 0) {
                sel_v[wid][k] = bv;
                sel_i[wid][k] = i;
            }
        }
        if (lane == 0) {
            const float m = sel_v[wid][0];
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += __expf(sel_v[wid][k] - m);
            for (int k = 0; k < K; ++k) {
                ids[(size_t)t * K + k] = sel_i[wid][k];
                w[(size_t)t * K + k] = __expf(sel_v[wid][k] - m) / s;
            }
        }
    }
    // ---- shared-expert gate, same block reduction as moe_topk ----
    if (shw != nullptr) {
        for (int t = 0; t < n; ++t) {
            float d = 0.f;
            for (int c = threadIdx.x; c < E; c += blockDim.x) d += x[(size_t)t * E + c] * shw[c];
            d = block_sum(d);
            if (threadIdx.x == 0) sg[t] = 1.f / (1.f + __expf(-d));
        }
    }
}

#define MOE_ROUTER_FUSED_ENTRY(IT)                                                                                       \
    extern "C" __global__ void __launch_bounds__(256)                                                                    \
        moe_router_fused_i##IT(const float *__restrict__ W, const float *__restrict__ x, float *logits,     \
                               int R, int C, int n, const int *skip, const float *__restrict__ shw,                      \
                               int *__restrict__ ids, float *__restrict__ w, float *__restrict__ sg, int K, int E,       \
                               unsigned *cnt) {                                                                          \
        moe_router_fused_impl<IT>(W, x, logits, R, C, n, skip, shw, ids, w, sg, K, E, cnt);                              \
    }
MOE_ROUTER_FUSED_ENTRY(16)  // C in (1920, 2048]: Ornith 35B-A3B (2048)
MOE_ROUTER_FUSED_ENTRY(24)  // C in (2944, 3072]: 122B-A10B class (3072)
