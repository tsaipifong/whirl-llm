// WHIRL device code, shared by both kernel sets (gfx1201 kernels/sample_pick.hip, gfx1151 kernels/gfx1151/draft_pick.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

#include "kernel_args.h"  // Tok16 (defined once for both kernel sets, arch025 N13)
extern "C" __global__ void set_tokens(int *ids, int *pos, const int *dev_src, Tok16 toks, int n, int n_host, int pos0) {
    const int i = threadIdx.x;
    if (i >= n) return;
    ids[i] = i < n_host ? toks.t[i] : dev_src[i - n_host];
    pos[i] = pos0 + i;
}

// Greedy token of each of gridDim.x rows of x ([rows][n]) into out[row].
extern "C" __global__ void __launch_bounds__(1024) argmax_rows(const float *x, int n, int *out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    x += (size_t)blockIdx.x * n;
#include "shared/block_argmax.h"  // argmax of x[0..n) -> bv[0], bi[0] (ADR-0005)
    if (threadIdx.x == 0) out[blockIdx.x] = bi[0];
}

// Batched decode rows over several sequences: token (>= 0 from the table,
// < 0: dev_src[-tok - 1], e.g. a sequence's draft slot), position and KV
// region base per row, all from kernel args.
// struct RowTab: kernels/kernel_args.h
extern "C" __global__ void set_rows(int *ids, int *pos, int *kvbase, const int *dev_src, RowTab tab, int n) {
    const int i = threadIdx.x;
    if (i >= n) return;
    const int t = tab.tok[i];
    ids[i] = t >= 0 ? t : dev_src[-t - 1];
    pos[i] = tab.pos[i];
    kvbase[i] = tab.base[i];
}

// struct RowIdx: kernels/kernel_args.h

// rmsnorm_q8 over gathered rows: output row b = norm(x row src.v[b]).
extern "C" __global__ void __launch_bounds__(1024)
    rmsnorm_q8_rows(const float *x, const float *w, float *out, int8_t *xq, float *xd, int n, float eps, RowIdx src) {
    x += (size_t)src.v[blockIdx.x] * n;
#include "shared/rmsnorm_q8_row.h"  // norm + int8 copy of output row blockIdx.x (ADR-0005)
}

// Greedy token of row b of x into out[dst.v[b]] (verify rows -> each
// sequence's control area).
extern "C" __global__ void __launch_bounds__(1024) argmax_rows_to(const float *x, int n, int *out, RowIdx dst) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    x += (size_t)blockIdx.x * n;
#include "shared/block_argmax.h"  // argmax of x[0..n) -> bv[0], bi[0] (ADR-0005)
    if (threadIdx.x == 0) out[dst.v[blockIdx.x]] = bi[0];
}

// relaxacc one-pass check (v0252-f1, fast mode; DISCOVERY F1): argmax_rows_to plus, for row b with a draft (src.v[b] >= 0:
// the draft token is out[src.v[b]], the sequence's control area), the draft's rank in the WHOLE row = the entries with a
// greater logit, or an equal logit and a lower id (the argmax's tie rule; an integer count, so every reduction order gives
// the same value), the row max and the draft's logit, in ONE read of the row:
//   stats[b] = rank (INT_MAX: no draft, or a draft id outside the row), ((float *)stats)[stride + b] = max,
//   ((float *)stats)[2 * stride + b] = l_d (-inf without a draft).
// The argmax scan / tree are block_argmax.h's (first index of the maximum). src and dst words must not overlap.
extern "C" __global__ void __launch_bounds__(1024)
    argmax_rank_rows_to(const float *x, int n, int *out, RowIdx dst, RowIdx src, int *stats, int stride) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    __shared__ int bc[1024];
    x += (size_t)blockIdx.x * n;
    const int si = src.v[blockIdx.x];
    const int d = si >= 0 ? out[si] : -1;
    const bool has = d >= 0 && d < n;
    const float ld = has ? x[d] : -INFINITY;
    float best = -INFINITY;
    int idx = 0, cnt = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        if (v > best) {
            best = v;
            idx = i;
        }
        cnt += (v > ld || (v == ld && i < d)) ? 1 : 0;
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = idx;
    bc[threadIdx.x] = cnt;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float o = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (o > bv[threadIdx.x] || (o == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = o;
                bi[threadIdx.x] = oi;
            }
            bc[threadIdx.x] += bc[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[dst.v[blockIdx.x]] = bi[0];
        stats[blockIdx.x] = has ? bc[0] : 0x7fffffff;
        ((float *)stats)[stride + blockIdx.x] = bv[0];
        ((float *)stats)[2 * stride + blockIdx.x] = ld;
    }
}

// MTP draft r: greedy token + softmax probability into tok[r] / prob[r], and
// the device-side p-min cutoff (llama.cpp --spec-draft-p-min / n-min):
// ctl[0] = drafts kept this cycle, ctl[1] = chain stopped (later drafts'
// weight-streaming kernels see it through their skip pointer and return).
// map (may be null): token id of each logits row (draft-head vocabulary subset).
__device__ __forceinline__ void draft_pick_body(const float *x, int n, int *tok, float *prob, int *ctl, int r, int n_min,
                                                float p_min, const int *map) {
    if (r > 0 && ctl[1] != 0) return;
    __shared__ float bv[1024];
    __shared__ int bi[1024];
#include "shared/block_argmax.h"  // argmax of x[0..n) -> bv[0], bi[0] (ADR-0005)
#include "shared/block_argmax_psum.h"  // m, top, bv[0] = sum e^(x - m) (ADR-0005)
    const float p = 1.f / bv[0];
    if (threadIdx.x == 0) {
        tok[r] = map ? map[top] : top;
        prob[r] = p;
        if (p_min > 0.f && r >= n_min && p < p_min) {
            ctl[0] = r;
            ctl[1] = 1;
        } else {
            ctl[0] = r + 1;
            if (r == 0) ctl[1] = 0;
        }
    }
}

extern "C" __global__ void __launch_bounds__(1024)
    draft_pick(const float *x, int n, int *tok, float *prob, int *ctl, int r, int n_min, float p_min, const int *map) {
    draft_pick_body(x, n, tok, prob, ctl, r, n_min, p_min, map);
}

// Batched drafts: logits row b belongs to the sequence whose control area
// starts at ctl_all + area.v[b] (ints; see the control-area layout in the model code).
extern "C" __global__ void __launch_bounds__(1024)
    draft_pick_rows(const float *x, int n, int *ctl_all, RowIdx area, int r, int n_min, float p_min, const int *map) {
    int *c = ctl_all + area.v[blockIdx.x];
    draft_pick_body(x + (size_t)blockIdx.x * n, n, c + 16, (float *)(c + 32), c + 48, r, n_min, p_min, map);
}

// Greedy token of one logits row plus its softmax probability (MTP draft
// confidence for the p-min cutoff): out_tok[0] = argmax, out_prob[0] = p.
extern "C" __global__ void __launch_bounds__(1024) argmax_prob(const float *x, int n, int *out_tok, float *out_prob) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
#include "shared/block_argmax.h"  // argmax of x[0..n) -> bv[0], bi[0] (ADR-0005)
#include "shared/block_argmax_psum.h"  // m, top, bv[0] = sum e^(x - m) (ADR-0005)
    if (threadIdx.x == 0) {
        out_tok[0] = top;
        out_prob[0] = 1.f / bv[0];
    }
}
