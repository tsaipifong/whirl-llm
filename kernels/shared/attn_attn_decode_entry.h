// WHIRL device code, shared by both kernel sets (gfx1201 kernels/attn.hip, gfx1151 kernels/gfx1151/attn.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

template <int F> __device__ __forceinline__ void kv_st32(void *base, _Float16 *sc, size_t e, float x) {
    if constexpr (F == 1) {
        const float amax = warp_max(fabsf(x));
        const _Float16 s = (_Float16)(amax * (1.f / 127.f));
        const float sf = (float)s;
        int qv = sf > 0.f ? (int)rintf(x / sf) : 0;
        qv = max(-127, min(127, qv));
        ((int8_t *)base)[e] = (int8_t)qv;
        if ((threadIdx.x % WARP) == 0) sc[e >> 5] = s;
    } else if constexpr (F == 2) {
        const float mx = warp_max(x), mn = -warp_max(-x);
        const float m = mx >= -mn ? mx : mn;  // signed value of largest magnitude
        const _Float16 s = (_Float16)(m * (-1.f / 8.f));
        const float sf = (float)s;
        int qv = sf != 0.f ? (int)rintf(x / sf) : 0;
        qv = max(-8, min(7, qv)) + 8;
        const int qo = __shfl_xor(qv, 1, WARP);
        if ((e & 1) == 0) ((uint8_t *)base)[e >> 1] = (uint8_t)(qv | (qo << 4));
        if ((threadIdx.x % WARP) == 0) sc[e >> 5] = s;
    } else {
        ((_Float16 *)base)[e] = (_Float16)x;
    }
}

// ---------------------------------------------------------------------------
// Full attention (decode, one query token per launch row; the scalar
// reference path, single sequence).
// q: [n_tok][n_head][q_stride] (query at offset 0, gate at +head_dim).
// out: [n_tok][n_head*head_dim], gated by sigmoid.
// scores scratch: [n_tok][n_head][max_ctx].
// Token t attends to positions [0, pos0 + t].
// ---------------------------------------------------------------------------
template <int Q8>
__device__ __forceinline__ void attn_decode_impl(const float *q, const KvArgs &kv, float *out, float *scores,
                int n_head, int n_kv, int head_dim, int q_stride, int max_ctx, const int *pos, float scale) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int pos0 = pos[0];
    const int kvh = h / (n_head / n_kv);
    const int n_pos = pos0 + t + 1;
    const int tb = kv.tab0;
    const float *qv = q + ((size_t)t * n_head + h) * q_stride;
    float *sc = scores + ((size_t)t * n_head + h) * max_ctx;

    __shared__ float qs[512];
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) qs[i] = qv[i];
    __syncthreads();

    // scores: one wave per position
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, nw = blockDim.x / WARP;
    for (int p = wid; p < n_pos; p += nw) {
        const size_t kb = ((size_t)kv_row(kv, tb, p) * n_kv + kvh) * head_dim;
        float s = 0.f;
        for (int i = lane; i < head_dim; i += WARP) s += qs[i] * kv_ld1<kv_kf(Q8)>(kv.k, kv.ks, kb + i);
        s = warp_sum(s);
        if (lane == 0) sc[p] = s * scale;
    }
    __syncthreads();

    float m = -INFINITY;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) m = fmaxf(m, sc[p]);
    m = block_max(m);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        const float e = __expf(sc[p] - m);
        sc[p] = e;
        sum += e;
    }
    sum = block_sum(sum);
    const float inv = 1.f / sum;
    __syncthreads();

    const float *gate = qv + head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < n_pos; ++p)
            acc += sc[p] * kv_ld1<kv_vf(Q8)>(kv.v, kv.vs, ((size_t)kv_row(kv, tb, p) * n_kv + kvh) * head_dim + d);
        const float g = gate[d];
        out[(size_t)t * n_head * head_dim + h * head_dim + d] = acc * inv / (1.f + __expf(-g));
    }
}
#define ATTN_DECODE_ENTRY(NAME, Q8)                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                        \
        NAME(const float *q, KvArgs kv, float *out, float *scores, int n_head, int n_kv, int head_dim,     \
             int q_stride, int max_ctx, const int *pos, float scale) {                                      \
        attn_decode_impl<Q8>(q, kv, out, scores, n_head, n_kv, head_dim, q_stride, max_ctx, pos, scale);  \
    }
