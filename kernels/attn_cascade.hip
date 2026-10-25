// WHIRL device code (gfx1201 and gfx1151): feat026-cascade shared-prefix decode attention helpers
// (whirl/ops/cascade.h). Included by both kernel sets after attn.hip; plain f32 code, no WMMA.
// SPDX-License-Identifier: Apache-2.0
//
// The prefix pass and the suffix pass are the existing split-K kernels (attn_wsplit* / attn_ws4* / attn_dq4)
// with other position / page-table rows; these two kernels only prepare those rows and merge the passes.

#define CAS_MAX_ROWS 32
// whirl::ops::CascadeRows (same layout, by value)
struct CascadeRows {
    int spos[CAS_MAX_ROWS];   // suffix position: pos - P
    int sbase[CAS_MAX_ROWS];  // suffix page-table offset: kvbase + P / KV_PAGE
    int ppos[CAS_MAX_ROWS];   // prefix position P - 1 (-1: the row is not in a cascade group)
};

// dst[0, 32) suffix positions, dst[32, 64) suffix page-table offsets, dst[64, 96) prefix positions
extern "C" __global__ void attn_cascade_rows(int *dst, CascadeRows t, int n) {
    const int i = threadIdx.x;
    if (i >= n || i >= CAS_MAX_ROWS) return;
    dst[i] = t.spos[i];
    dst[CAS_MAX_ROWS + i] = t.sbase[i];
    dst[2 * CAS_MAX_ROWS + i] = t.ppos[i];
}

// For every row with a prefix (rows[64 + r] >= 0) and head h: the prefix partials (cas_ml / cas_acc, n_split
// splits, the attn_wsplit1 layout) merged by log-sum-exp into split 0 of the suffix partials (part_ml /
// part_acc), so attn_combine / attn_combine_q8 run unchanged over the suffix's n_split splits.
//   Mp = max_i m_i, Lp = sum_i e^(m_i - Mp) l_i, Op[d] = sum_i e^(m_i - Mp) acc_i[d]   (prefix splits i)
//   M = max(m0, Mp), split 0 := (M, l0 e^(m0 - M) + Lp e^(Mp - M), acc0 e^(m0 - M) + Op e^(Mp - M))
// Empty partials (m = -inf) carry no acc (the split kernels do not write it) and are skipped.
// grid (n_head, rows), 256 threads (thread d owns columns d, d + 256, ...). CPU twin: ops::cascadeMergeRef.
extern "C" __global__ void __launch_bounds__(256)
    attn_cascade_merge(float *part_ml, float *part_acc, const float *cas_ml, const float *cas_acc, const int *rows, int n_head,
                       int head_dim, int n_split) {
    const int h = blockIdx.x, r = blockIdx.y;
    if (rows[2 * CAS_MAX_ROWS + r] < 0) return;
    const size_t ml_r = (size_t)r * n_split * n_head * 2, acc_r = (size_t)r * n_split * n_head * head_dim;
    const float *pml = cas_ml + ml_r;
    float mp = -INFINITY;
    for (int i = 0; i < n_split; ++i) mp = fmaxf(mp, pml[((size_t)i * n_head + h) * 2]);
    float lp = 0.f;
    for (int i = 0; i < n_split; ++i) {
        const float m = pml[((size_t)i * n_head + h) * 2];
        if (m > -INFINITY) lp += __expf(m - mp) * pml[((size_t)i * n_head + h) * 2 + 1];
    }
    float *s0 = part_ml + ml_r + (size_t)h * 2;
    const float m0 = s0[0], l0 = s0[1];
    const float M = fmaxf(m0, mp);
    __syncthreads();  // every thread has read split 0's (m, l) before thread 0 rewrites it
    if (M == -INFINITY) return;
    const float f0 = m0 > -INFINITY ? __expf(m0 - M) : 0.f, fp = mp > -INFINITY ? __expf(mp - M) : 0.f;
    float *a0 = part_acc + acc_r + (size_t)h * head_dim;
    const float *pacc = cas_acc + acc_r;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float op = 0.f;
        for (int i = 0; i < n_split; ++i) {
            const float m = pml[((size_t)i * n_head + h) * 2];
            if (m > -INFINITY) op += __expf(m - mp) * pacc[((size_t)i * n_head + h) * head_dim + d];
        }
        const float x0 = m0 > -INFINITY ? a0[d] : 0.f;
        a0[d] = x0 * f0 + op * fp;
    }
    if (threadIdx.x == 0) {
        s0[0] = M;
        s0[1] = (m0 > -INFINITY ? l0 * f0 : 0.f) + lp * fp;
    }
}
#undef CAS_MAX_ROWS
