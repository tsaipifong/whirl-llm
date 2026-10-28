// WHIRL device code, shared by both kernel sets (gfx1201 kernels/attn.hip, gfx1151 kernels/gfx1151/attn.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

#define ATTN_SPLIT_ENTRY(NAME, NQ, Q8)                                                                        \
    extern "C" __global__ void __launch_bounds__(256)                                                         \
        NAME(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head,                          \
             int n_kv, int head_dim, int q_stride, const int *pos, float scale, int nq, const int *skip) {    \
        if (skip != nullptr && *skip) return;                                                                 \
        const int tb = kv_tb(kv, blockIdx.z * NQ);                                                            \
        attn_split_impl<NQ, Q8>(q, kv, tb, part_ml, part_acc, n_head, n_kv, head_dim, q_stride, pos, scale);  \
    }
// NQ > 1 (one pass over K/V for several queries) measured slower than
// grid.z = query with NQ = 1: the score reductions, not K/V bytes, bound it.
ATTN_SPLIT_ENTRY(attn_split, 1, false)
ATTN_SPLIT_ENTRY(attn_split_q8, 1, true)
ATTN_SPLIT_ENTRY(attn_split_q8v, 1, 2)
ATTN_SPLIT_ENTRY(attn_split_q4, 1, 3)

// out[h][d] = sum_i e^(m_i - M) acc_i[d] / sum_i e^(m_i - M) l_i, gated by sigmoid.
extern "C" __global__ void attn_combine(const float *part_ml, const float *part_acc, const float *q, float *out,
                                        int n_head, int head_dim, int q_stride, int n_split) {
    const int h = blockIdx.x;
    const int tq = blockIdx.y;  // query token
    part_ml += (size_t)tq * n_split * n_head * 2;
    part_acc += (size_t)tq * n_split * n_head * head_dim;
    q += (size_t)tq * n_head * q_stride;
    out += (size_t)tq * n_head * head_dim;
    float M = -INFINITY;
    for (int i = 0; i < n_split; ++i) M = fmaxf(M, part_ml[((size_t)i * n_head + h) * 2]);
    float L = 0.f;
    for (int i = 0; i < n_split; ++i) {
        const float m = part_ml[((size_t)i * n_head + h) * 2];
        if (m > -INFINITY) L += __expf(m - M) * part_ml[((size_t)i * n_head + h) * 2 + 1];
    }
    const float inv = 1.f / L;
    const float *gate = q + (size_t)h * q_stride + head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < n_split; ++i) {
            const float m = part_ml[((size_t)i * n_head + h) * 2];
            if (m > -INFINITY) acc += __expf(m - M) * part_acc[((size_t)i * n_head + h) * head_dim + d];
        }
        const float gg = gate[d];
        out[(size_t)h * head_dim + d] = acc * inv / (1.f + __expf(-gg));
    }
}
