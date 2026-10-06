// WHIRL device code (gfx1201 and gfx1151): precise-mode MoE decode experts (moe_gu_f_*, moe_down_f_*).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included after common.hip / gemv_f.hip (dec_group<T>, warp_sum) by both kernel sets.
//
// The decode / verify experts of precise mode: f32 activations (no int8, no f16 rounding of
// the activations), the weights decoded by dec_group<T> to f32, f32 accumulation. A pair's
// (token, expert slot) result is computed by its own waves with a fixed reduction order (lane
// partials over the 8-value groups g = lane, lane + 32, ... as gemv_<T>_1, then warp_sum),
// so it does not depend on how many tokens are in the batch or on the other tokens.

// gate + up: pair p = blockIdx.y = t * K + k runs expert ids[p] on h[t] (f32, C values).
// Rows [0, ff) from W1 into y1, [ff, nrows) from W2 into y2 (nrows = 2 ff when both share a type).
template <int T>
__device__ void moe_gu_f_impl(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, uint64_t row_bytes, int ff, int nrows,
                              const float *__restrict__ h, const int *__restrict__ ids, float *__restrict__ y1, float *__restrict__ y2,
                              int C, int K) {
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const int p = blockIdx.y;
    const int t = p / K, e = ids[p];
    const bool second = row >= ff;
    const int r = second ? row - ff : row;
    const uint8_t *wr = (second ? W2 : W1) + ((size_t)e * ff + r) * row_bytes;
    const float *x = h + (size_t)t * C;
    float acc = 0.f;
    for (int g = lane; g < C / 8; g += WARP) {
        float w[8];
        dec_group<T>(wr, g, w);
        const float4 *xp = (const float4 *)(x + 8 * g);
        const float4 a = xp[0], b = xp[1];
        acc += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) (second ? y2 : y1)[(size_t)p * ff + r] = acc;
}

// down + combine: x[t][row] += sum_k w[t,k] * (W_{ids[t,k]} a[t,k])[row] (+ sg[t] * ysh[t][row]),
// a: f32 [n * K][F] (silu(g) * u). One wave per output row; the K experts in slot order.
template <int T>
__device__ void moe_down_f_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, int E, const float *__restrict__ act,
                                const int *__restrict__ ids, const float *__restrict__ w, const float *__restrict__ sg,
                                const float *__restrict__ ysh, float *__restrict__ x, int F, int K) {
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= E) return;
    const int t = blockIdx.y;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int p = t * K + k;
        const uint8_t *wr = W + ((size_t)ids[p] * E + row) * row_bytes;
        const float *a = act + (size_t)p * F;
        float s = 0.f;
        for (int g = lane; g < F / 8; g += WARP) {
            float wv[8];
            dec_group<T>(wr, g, wv);
            const float4 *xp = (const float4 *)(a + 8 * g);
            const float4 u = xp[0], v = xp[1];
            s += wv[0] * u.x + wv[1] * u.y + wv[2] * u.z + wv[3] * u.w + wv[4] * v.x + wv[5] * v.y + wv[6] * v.z + wv[7] * v.w;
        }
        acc += w[p] * s;
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        float v = acc;
        if (ysh != nullptr) v += sg[t] * ysh[(size_t)t * E + row];
        x[(size_t)t * E + row] += v;
    }
}

#define MOE_F_ENTRIES(SUF, T)                                                                                                       \
    extern "C" __global__ void __launch_bounds__(256) moe_gu_f_##SUF(const uint8_t *W1, const uint8_t *W2, uint64_t row_bytes, int ff, \
                                                                    int nrows, const float *h, const int *ids, float *y1, float *y2,  \
                                                                    int C, int K) {                                                   \
        moe_gu_f_impl<T>(W1, W2, row_bytes, ff, nrows, h, ids, y1, y2, C, K);                                                         \
    }                                                                                                                                 \
    extern "C" __global__ void __launch_bounds__(256) moe_down_f_##SUF(const uint8_t *W, uint64_t row_bytes, int E, const float *act,  \
                                                                      const int *ids, const float *w, const float *sg, const float *ysh, \
                                                                      float *x, int F, int K) {                                       \
        moe_down_f_impl<T>(W, row_bytes, E, act, ids, w, sg, ysh, x, F, K);                                                           \
    }

MOE_F_ENTRIES(q4_k, QT_Q4_K)
MOE_F_ENTRIES(q5_k, QT_Q5_K)
MOE_F_ENTRIES(q6_k, QT_Q6_K)
MOE_F_ENTRIES(q8_0, QT_Q8_0)
MOE_F_ENTRIES(iq4_xs, QT_IQ4_XS)
MOE_F_ENTRIES(iq4_nl, QT_IQ4_NL)
MOE_F_ENTRIES(q3_k, QT_Q3_K)
MOE_F_ENTRIES(iq3_s, QT_IQ3_S)
MOE_F_ENTRIES(mxfp4, QT_MXFP4)
