// moeskip (fast mode, feat027): top-K routing that drops low-weight slots.
// SPDX-License-Identifier: Apache-2.0
//
// Shared by both kernel sets (included after moe.hip: WARP, block_sum). moe_topk itself is left
// unchanged; precise / balance never launch this kernel. Host reference: include/whirl/moe_skip.h.
//
// Per token (one 256-thread block): the same top-K selection as moe_topk (ties -> lower index),
// then w[k] = exp(v_k - v_0) / sum_j exp(v_j - v_0) over the K slots (moe_topk's weights); slots
// k >= min_keep with w[k] < thr are dropped (id -1, weight 0, a suffix since v is sorted), and the
// kept weights are renormalized (exp(v_k - v_0) / sum over the kept slots). sg as moe_topk.

extern "C" __global__ void __launch_bounds__(256)
    moe_topk_skip(const float *__restrict__ logits, int ld, const float *__restrict__ h, const float *__restrict__ shw,
                  int *__restrict__ ids, float *__restrict__ w, float *__restrict__ sg, int R, int K, int E, float thr,
                  int min_keep) {
    const int t = blockIdx.x;
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, nw = blockDim.x / WARP;
    __shared__ float lv[1024];
    __shared__ float rv[32];
    __shared__ int ri[32];
    __shared__ float sel_v[64];
    __shared__ int sel_i[64];
    const float *l = logits + (size_t)t * ld;
    for (int r = threadIdx.x; r < R; r += blockDim.x) lv[r] = l[r];
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int r = threadIdx.x; r < R; r += blockDim.x) {
            const float v = lv[r];
            if (v > bv || (v == bv && r < bi)) {
                bv = v;
                bi = r;
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
        if (lane == 0) {
            rv[wid] = bv;
            ri[wid] = bi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = rv[0];
            int i = ri[0];
            for (int q = 1; q < nw; ++q) {
                if (rv[q] > b || (rv[q] == b && ri[q] < i)) {
                    b = rv[q];
                    i = ri[q];
                }
            }
            if (i < 0 || i >= R) i = 0;  // all -inf/NaN: keep a valid id
            sel_v[k] = b;
            sel_i[k] = i;
            lv[i] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float m = sel_v[0];
        float s = 0.f;
        for (int k = 0; k < K; ++k) s += __expf(sel_v[k] - m);
        const int mk = min_keep < 1 ? 1 : (min_keep > K ? K : min_keep);
        int keep = K;
        if (thr > 0.f) {
            for (int k = mk; k < K; ++k) {
                if (__expf(sel_v[k] - m) / s < thr) {
                    keep = k;
                    break;
                }
            }
        }
        float s2 = 0.f;
        for (int k = 0; k < keep; ++k) s2 += __expf(sel_v[k] - m);
        for (int k = 0; k < K; ++k) {
            const bool kept = k < keep;
            ids[(size_t)t * K + k] = kept ? sel_i[k] : -1;
            w[(size_t)t * K + k] = kept ? __expf(sel_v[k] - m) / s2 : 0.f;
        }
    }
    if (shw != nullptr) {
        float d = 0.f;
        for (int c = threadIdx.x; c < E; c += blockDim.x) d += h[(size_t)t * E + c] * shw[c];
        d = block_sum(d);
        if (threadIdx.x == 0) sg[t] = 1.f / (1.f + __expf(-d));
    }
}

// Kept slots of token t: ids of a token are a kept prefix then -1 (moe_topk_skip); K without moeskip.
__device__ __forceinline__ int moe_kept(const int *__restrict__ ids, int t, int K) {
    int kt = K;
    while (kt > 1 && ids[(size_t)t * K + kt - 1] < 0) --kt;
    return kt;
}
