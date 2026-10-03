// WHIRL device code (gfx1201 / RDNA 4): token tables, greedy picks for batched / verify rows, MTP draft pick with probability.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// Token ids + positions for up to 16 rows without a host copy (a blocking
// hipMemcpy would drain the queue): rows [0, n_host) from the by-value table,
// the rest from dev_src (earlier on-device argmax results).
struct Tok16 {
    int t[16];
};
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
    float best = -INFINITY;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > best) {
            best = x[i];
            idx = i;
        }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float o = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (o > bv[threadIdx.x] || (o == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = o;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = bi[0];
}

// Batched decode rows over several sequences: token (>= 0 from the table,
// < 0: dev_src[-tok - 1], e.g. a sequence's draft slot), position and KV
// region base per row, all from kernel args.
struct RowTab {
    int tok[32];
    int pos[32];
    int base[32];
};
extern "C" __global__ void set_rows(int *ids, int *pos, int *kvbase, const int *dev_src, RowTab tab, int n) {
    const int i = threadIdx.x;
    if (i >= n) return;
    const int t = tab.tok[i];
    ids[i] = t >= 0 ? t : dev_src[-t - 1];
    pos[i] = tab.pos[i];
    kvbase[i] = tab.base[i];
}

struct RowIdx {
    int v[32];
};

// rmsnorm_q8 over gathered rows: output row b = norm(x row src.v[b]).
extern "C" __global__ void __launch_bounds__(1024)
    rmsnorm_q8_rows(const float *x, const float *w, float *out, int8_t *xq, float *xd, int n, float eps, RowIdx src) {
    x += (size_t)src.v[blockIdx.x] * n;
    out += (size_t)blockIdx.x * n;
    xq += (size_t)blockIdx.x * n;
    xd += (size_t)blockIdx.x * (n / 32);
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / n + eps);
    const int wid = threadIdx.x / WARP, lane = threadIdx.x % WARP, nw = blockDim.x / WARP;
    for (int b = wid; b < n / 32; b += nw) {
        const int i = b * 32 + lane;
        const float v = x[i] * r * w[i];
        out[i] = v;
        wave_q8(v, xq, xd, b);
    }
}

// Greedy token of row b of x into out[dst.v[b]] (verify rows -> each
// sequence's control area).
extern "C" __global__ void __launch_bounds__(1024) argmax_rows_to(const float *x, int n, int *out, RowIdx dst) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    x += (size_t)blockIdx.x * n;
    float best = -INFINITY;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > best) {
            best = x[i];
            idx = i;
        }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float o = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (o > bv[threadIdx.x] || (o == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = o;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[dst.v[blockIdx.x]] = bi[0];
}

// MTP draft r: greedy token + softmax probability into tok[r] / prob[r], and
// the device-side p-min cutoff (llama.cpp --spec-draft-p-min / n-min):
// ctl[0] = drafts kept this cycle, ctl[1] = chain stopped (later drafts'
// weight-streaming kernels see it through their skip pointer and return).
__device__ __forceinline__ void draft_pick_body(const float *x, int n, int *tok, float *prob, int *ctl, int r, int n_min,
                                                float p_min) {
    if (r > 0 && ctl[1] != 0) return;
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float best = -INFINITY;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > best) {
            best = x[i];
            idx = i;
        }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float o = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (o > bv[threadIdx.x] || (o == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = o;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    const float m = bv[0];
    const int top = bi[0];
    __syncthreads();
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += __expf(x[i] - m);
    bv[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) bv[threadIdx.x] += bv[threadIdx.x + s];
        __syncthreads();
    }
    const float p = 1.f / bv[0];
    if (threadIdx.x == 0) {
        tok[r] = top;
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
    draft_pick(const float *x, int n, int *tok, float *prob, int *ctl, int r, int n_min, float p_min) {
    draft_pick_body(x, n, tok, prob, ctl, r, n_min, p_min);
}

// Batched drafts: logits row b belongs to the sequence whose control area
// starts at ctl_all + area.v[b] (ints; see the control-area layout in the model code).
extern "C" __global__ void __launch_bounds__(1024)
    draft_pick_rows(const float *x, int n, int *ctl_all, RowIdx area, int r, int n_min, float p_min) {
    int *c = ctl_all + area.v[blockIdx.x];
    draft_pick_body(x + (size_t)blockIdx.x * n, n, c + 16, (float *)(c + 32), c + 48, r, n_min, p_min);
}

// Greedy token of one logits row plus its softmax probability (MTP draft
// confidence for the p-min cutoff): out_tok[0] = argmax, out_prob[0] = p.
extern "C" __global__ void __launch_bounds__(1024) argmax_prob(const float *x, int n, int *out_tok, float *out_prob) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float best = -INFINITY;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > best) {
            best = x[i];
            idx = i;
        }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float o = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (o > bv[threadIdx.x] || (o == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = o;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    const float m = bv[0];
    const int top = bi[0];
    __syncthreads();
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += __expf(x[i] - m);
    bv[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) bv[threadIdx.x] += bv[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_tok[0] = top;
        out_prob[0] = 1.f / bv[0];
    }
}

