// WHIRL device code, shared by both kernel sets (gfx1201 kernels/moe.hip, gfx1151 kernels/gfx1151/moe.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).
// Router logits in f32 for n <= 16 tokens: out[t][r] = W[r] . x[t] (C % 4 == 0).
// One wave per router row, 8 rows per block.
extern "C" __global__ void __launch_bounds__(256)
    moe_logits_f32(const float *__restrict__ W, const float *__restrict__ x, float *__restrict__ out, int R, int C,
                   int n, const int *skip) {
    if (skip != nullptr && *skip) return;
    constexpr int MT = 16;
    const int lane = threadIdx.x % WARP;
    const int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (r >= R) return;
    const float4 *wr = (const float4 *)(W + (size_t)r * C);
    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.f;
    for (int c = lane; c < C / 4; c += WARP) {
        const float4 w = wr[c];
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t < n) {
                const float4 v = ((const float4 *)(x + (size_t)t * C))[c];
                acc[t] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (t < n) {
            const float s = warp_sum(acc[t]);
            if (lane == 0) out[(size_t)t * R + r] = s;
        }
    }
}

// Per token (one 256-thread block): top-K of the R router logits (ties -> lower
// index), weights = softmax restricted to the selected experts (= softmax over
// all R, then renormalized), sg[t] = sigmoid(shw . h[t]) when shw != null.
extern "C" __global__ void __launch_bounds__(256)
    moe_topk(const float *__restrict__ logits, int ld, const float *__restrict__ h, const float *__restrict__ shw,
             int *__restrict__ ids, float *__restrict__ w, float *__restrict__ sg, int R, int K, int E) {
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
        for (int k = 0; k < K; ++k) {
            ids[(size_t)t * K + k] = sel_i[k];
            w[(size_t)t * K + k] = __expf(sel_v[k] - m) / s;
        }
    }
    if (shw != nullptr) {
        float d = 0.f;
        for (int c = threadIdx.x; c < E; c += blockDim.x) d += h[(size_t)t * E + c] * shw[c];
        d = block_sum(d);
        if (threadIdx.x == 0) sg[t] = 1.f / (1.f + __expf(-d));
    }
}
