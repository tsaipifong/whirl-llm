// WHIRL device code, shared by both kernel sets (gfx1201 kernels/attn.hip, gfx1151 kernels/gfx1151/attn.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

extern "C" __global__ void set_rpos(int *rpos, Rpos16 tab, int n) {
    const int i = threadIdx.x;
    if (i < 3 * n) rpos[i] = tab.v[i];
}

// attn_combine + the int8 copy of its output for the o-projection GEMV
// (same values as quantize_q8 over out); needs blockDim == head_dim, a
// multiple of 32.
extern "C" __global__ void attn_combine_q8(const float *part_ml, const float *part_acc, const float *q, float *out,
                                           int n_head, int head_dim, int q_stride, int n_split, int8_t *xq, float *xd) {
#include "shared/attn_combine_ml.h"  // h, tq, M, L, inv, gate (ADR-0005)
    const int d = threadIdx.x;
    float acc = 0.f;
    for (int i = 0; i < n_split; ++i) {
        const float m = part_ml[((size_t)i * n_head + h) * 2];
        if (m > -INFINITY) acc += __expf(m - M) * part_acc[((size_t)i * n_head + h) * head_dim + d];
    }
    const float gg = gate[d];
    const float o = acc * inv / (1.f + __expf(-gg));
    out[(size_t)h * head_dim + d] = o;
    wave_q8(o, xq, xd, (int)((((size_t)tq * n_head + h) * head_dim + d) >> 5));
}

// ---------------------------------------------------------------------------
// Split-K decode / verify attention on WMMA (gfx12), K/V read once per
// query group. Block (kv head g, split, group z): the group's queries
// (verify rows of one sequence) times the GQA heads of g form up to 64
// "columns" (column c = query c / grp, head j = c % grp); wave cg handles
// columns [16 cg, 16 cg + 16) like attn_prefill_impl (queries as columns):
//   S^T = K Q^T  (A = K rows straight from the f16 cache, B = Q^T in registers)
//   online softmax per column (lane % 16, partner lane ^ 16)
//   O^T += V^T P^T (A = V^T from LDS, stored transposed per 32-key chunk).
// Every column walks the split's keys [b0, b1) in 32-key chunks in order;
// keys past its own position are masked (exact zeros), so a column's
// partial (m, l, acc) is the same bits whether it runs alone (decode) or
// with other queries (verify) - the split range comes from the group's last
// query and the host only groups queries with the same split size `per`.
// Partials go to part_ml / part_acc in the attn_split layout, merged by
// attn_combine / attn_combine_q8.
// ---------------------------------------------------------------------------
#define AW_K 32
#define AW_D 256
#define AW_SINK 256  // draft window: positions [0, AW_SINK) are always attended
