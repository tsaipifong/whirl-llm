// WHIRL device code (gfx1201 and gfx1151): speculative sampling of MTP drafts (numerics item specsample).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included last by whirl_kernels.hip (uses RowIdx of the active kernel set).
//
// draft_sample_rows runs right after draft_pick_rows (same rows, same control areas) for the rows
// whose request samples (temperature > 0): it replaces the greedy draft r with a token drawn from
// the draft distribution q and writes q for the host (accept min(1, p/q), residual max(0, p - q)).
// q = softmax(x * inv_t) over the K = min(top_k, 32) (32 when top_k = 0) largest logits, then the
// request's top-k / top-p / min-p, renormalized; host reference: whirl::spec::draftDist.
// The p-min cutoff of draft_pick_rows (argmax probability at temperature 1, a function of the row
// only, never of the drawn token) is kept: a row whose chain stopped at or before r is skipped.
// Rows with on = 0 keep the greedy draft.

#define WHIRL_SPEC_QMAX 32
#define WHIRL_SPEC_QWORDS 72

struct DSampArgs {
    int on[32];
    int top_k[32];
    int qoff[32];  // i32 word offset of this row's q entry in qbuf
    float inv_t[32];
    float top_p[32];
    float min_p[32];
    float u[32];
};

__device__ __forceinline__ unsigned spec_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ void __launch_bounds__(1024)
    draft_sample_rows(const float *x, int n, int *ctl_all, RowIdx area, int r, const int *map, DSampArgs a, int *qbuf) {
    __shared__ unsigned hist[32][256];
    __shared__ float red[1024];
    __shared__ unsigned s_prefix, s_remain, s_gt, s_eq;
    __shared__ int s_id[WHIRL_SPEC_QMAX];
    __shared__ float s_lg[WHIRL_SPEC_QMAX];
    const int b = blockIdx.x;
    if (a.on[b] == 0) return;
    int *c = ctl_all + area.v[b];
    if (c[49] != 0 && c[48] <= r) return;  // chain stopped by the p-min cutoff
    const int t = threadIdx.x;
    const int w = (t >> 5) & 31;
    x += (size_t)b * n;
    const int tk = a.top_k[b];
    int K = (tk > 0 && tk < WHIRL_SPEC_QMAX) ? tk : WHIRL_SPEC_QMAX;
    if (K > n) K = n;
    const float inv_t = a.inv_t[b];
    float m = -INFINITY;
    for (int i = t; i < n; i += blockDim.x) m = fmaxf(m, x[i]);
    red[t] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float sum = 0.f;
    for (int i = t; i < n; i += blockDim.x) sum += __expf((x[i] - m) * inv_t);
    red[t] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    sum = red[0];
    if (t == 0) {
        s_prefix = 0;
        s_remain = (unsigned)K;
    }
    // K-th largest by an 8-bit radix select (as topk_rows)
    unsigned mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 32 * 256; i += blockDim.x) (&hist[0][0])[i] = 0;
        __syncthreads();
        const unsigned prefix = s_prefix;
        for (int i = t; i < n; i += blockDim.x) {
            const unsigned k = spec_key(x[i]);
            if ((k & mask) == prefix) atomicAdd(&hist[w][(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (t < 256) {
            unsigned cnt = 0;
            for (int j = 0; j < 32; ++j) cnt += hist[j][t];
            hist[0][t] = cnt;
        }
        __syncthreads();
        if (t == 0) {
            unsigned rem = s_remain;
            int bb = 255;
            for (; bb > 0; --bb) {
                const unsigned cnt = hist[0][bb];
                if (cnt >= rem) break;
                rem -= cnt;
            }
            s_prefix = prefix | ((unsigned)bb << shift);
            s_remain = rem;
        }
        mask |= 255u << shift;
        __syncthreads();
    }
    if (t == 0) {
        s_gt = 0;
        s_eq = 0;
    }
    __syncthreads();
    const unsigned thr = s_prefix, need = s_remain, base = (unsigned)K - need;
    for (int i = t; i < n; i += blockDim.x) {
        const float v = x[i];
        const unsigned k = spec_key(v);
        if (k > thr) {
            const unsigned j = atomicAdd(&s_gt, 1u);
            if (j < base) {
                s_id[j] = i;
                s_lg[j] = v;
            }
        } else if (k == thr) {
            const unsigned j = atomicAdd(&s_eq, 1u);
            if (j < need) {
                s_id[base + j] = i;
                s_lg[base + j] = v;
            }
        }
    }
    __syncthreads();
    if (t != 0) return;
    // tie order of equal keys depends on the atomics: ids are mapped first and sorted by (logit desc, id asc)
    int id[WHIRL_SPEC_QMAX];
    float lg[WHIRL_SPEC_QMAX], p[WHIRL_SPEC_QMAX];
    for (int i = 0; i < K; ++i) {
        id[i] = map ? map[s_id[i]] : s_id[i];
        lg[i] = s_lg[i];
    }
    for (int i = 1; i < K; ++i) {
        int j = i;
        while (j > 0 && (lg[j - 1] < lg[j] || (lg[j - 1] == lg[j] && id[j - 1] > id[j]))) {
            const float tl = lg[j - 1];
            lg[j - 1] = lg[j];
            lg[j] = tl;
            const int ti = id[j - 1];
            id[j - 1] = id[j];
            id[j] = ti;
            --j;
        }
    }
    for (int i = 0; i < K; ++i) p[i] = expf((lg[i] - m) * inv_t) / sum;
    int nk = K;
    const bool closed = tk > 0 && tk <= K;
    if (closed) nk = tk;
    const float top_p = a.top_p[b];
    if (top_p < 1.f) {
        float z = 1.f;
        if (closed) {
            z = 0.f;
            for (int i = 0; i < nk; ++i) z += p[i];
        }
        float cum = 0.f;
        for (int j = 0; j < nk; ++j) {
            cum += p[j] / z;
            if (cum >= top_p) {
                nk = j + 1;
                break;
            }
        }
    }
    const float min_p = a.min_p[b];
    if (min_p > 0.f && nk > 0) {
        const float th = min_p * p[0];
        int j = 0;
        while (j < nk && p[j] >= th) ++j;
        nk = j > 1 ? j : 1;
    }
    float z = 0.f;
    for (int i = 0; i < nk; ++i) z += p[i];
    for (int i = 0; i < nk; ++i) p[i] /= z;
    // draw (same arithmetic as whirl::spec::sampleQ)
    const float u = a.u[b];
    int pick = nk - 1;
    float cum = 0.f;
    for (int i = 0; i < nk; ++i) {
        cum += p[i];
        if (cum > u) {
            pick = i;
            break;
        }
    }
    c[16 + r] = id[pick];
    int *q = qbuf + a.qoff[b];
    q[0] = nk;
    for (int i = 0; i < nk; ++i) {
        q[1 + i] = id[i];
        q[1 + WHIRL_SPEC_QMAX + i] = __float_as_int(p[i]);
    }
}
