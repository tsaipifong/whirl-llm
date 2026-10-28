// WHIRL device code, shared by both kernel sets (gfx1201 kernels/moe.hip, gfx1151 kernels/gfx1151/moe.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

#define GDN_AB_ENTRY(SUF, T)                                                                              \
    extern "C" __global__ void __launch_bounds__(256)                                                     \
        gdn_ab_##SUF(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq,         \
                     const float *xd, float *beta, float *g, const float *dt, const float *A, int ncols,  \
                     int nh) {                                                                            \
        const int t = blockIdx.y;                                                                         \
        gdn_ab_impl<T>(Wb, Wa, row_bytes, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32),          \
                       beta + (size_t)t * nh, g + (size_t)t * nh, dt, A, ncols, nh, blockIdx.x);                      \
    }
GDN_AB_ENTRY(q4_k, QT_Q4_K)
GDN_AB_ENTRY(q5_k, QT_Q5_K)
GDN_AB_ENTRY(q6_k, QT_Q6_K)
GDN_AB_ENTRY(iq4_xs, QT_IQ4_XS)
GDN_AB_ENTRY(mxfp4, QT_MXFP4)
GDN_AB_ENTRY(iq4_nl, QT_IQ4_NL)
GDN_AB_ENTRY(q3_k, QT_Q3_K)
GDN_AB_ENTRY(iq3_s, QT_IQ3_S)

// gdn_ab + gdn_conv_l2 in one launch (independent inputs): blocks
// [0, n_ab * ab_blocks) = gdn_ab (token b / ab_blocks, 8 rows per block),
// the rest = gdn_conv_l2 (segment, 128-channel block) with 128 of the 256
// threads active. Per block identical to the two separate kernels.
template <int T>
__device__ void gdn_abconv_impl(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq,
                                const float *xd, float *beta, float *g, const float *dt, const float *A, int ncols,
                                int nh, int n_ab, const float *xin, const float *w, float *out, int ch, int n_qk_heads,
                                float eps, const GdnSegs &segs) {
    const int ab_blocks = (2 * nh + 7) / 8;
    int b = blockIdx.x;
    if (b < n_ab * ab_blocks) {
        const int t = b / ab_blocks;
        gdn_ab_impl<T>(Wb, Wa, row_bytes, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32), beta + (size_t)t * nh,
                       g + (size_t)t * nh, dt, A, ncols, nh, b % ab_blocks);
        return;
    }
    b -= n_ab * ab_blocks;
    const int nblk = ch / 128;
    gdn_conv_l2_impl(xin, w, out, ch, n_qk_heads, eps, segs, b % nblk, b / nblk, threadIdx.x < 128);
}
#define GDN_ABCONV_ENTRY(SUF, T)                                                                               \
    extern "C" __global__ void __launch_bounds__(256)                                                          \
        gdn_abconv_##SUF(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq,          \
                         const float *xd, float *beta, float *g, const float *dt, const float *A, int ncols,   \
                         int nh, int n_ab, const float *xin, const float *w, float *out, int ch, int n_qk_heads, \
                         float eps, GdnSegs segs) {                                                            \
        gdn_abconv_impl<T>(Wb, Wa, row_bytes, xq, xd, beta, g, dt, A, ncols, nh, n_ab, xin, w, out, ch,       \
                           n_qk_heads, eps, segs);                                                             \
    }
GDN_ABCONV_ENTRY(q8_0, QT_Q8_0)
GDN_ABCONV_ENTRY(q4_k, QT_Q4_K)
GDN_ABCONV_ENTRY(q5_k, QT_Q5_K)
GDN_ABCONV_ENTRY(q6_k, QT_Q6_K)
GDN_ABCONV_ENTRY(iq4_xs, QT_IQ4_XS)
GDN_ABCONV_ENTRY(mxfp4, QT_MXFP4)
GDN_ABCONV_ENTRY(iq4_nl, QT_IQ4_NL)
GDN_ABCONV_ENTRY(q3_k, QT_Q3_K)
GDN_ABCONV_ENTRY(iq3_s, QT_IQ3_S)

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
