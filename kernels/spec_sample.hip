// WHIRL device code (gfx1201 and gfx1151): speculative sampling of MTP drafts (numerics item specsample).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included last by whirl_kernels.hip (uses RowIdx of the active kernel set).
//
// draft_sample_rows runs right after draft_pick_rows (same rows, same control areas) for the rows
// whose request samples (temperature > 0): it replaces the greedy draft r with a token drawn from
// the draft distribution q by the race coupled to the target sampler (argmin E(v) / q(v), E from the
// row's race key, whirl/spec_sample.h) and writes q (tests / diagnostics).
// q = softmax(x * inv_t) over the K = min(top_k, 32) (32 when top_k = 0) largest logits, then the
// request's top-k / top-p / min-p, renormalized; host reference: whirl::spec::draftDist.
// The p-min cutoff of draft_pick_rows (argmax probability at temperature 1, a function of the row
// only, never of the drawn token) is kept: a row whose chain stopped at or before r is skipped.
// Rows with on = 0 keep the greedy draft.
// Ties at the K-th logit (feat027-relaxdet-r1, as topk_rows): when more logits equal the K-th value
// than places are left, the LOWEST row indices among them are kept, chosen by an index-ordered block
// scan, so q (and the drawn draft) never depend on shared-atomic arrival order. The row index order is
// the token id order: the draft-vocabulary map is strictly ascending (qwen35::draftVocabIds), so this
// is the (logit desc, id asc) rule of whirl::spec::draftDist and the kernel test oracle. Without
// overflowing ties (the usual case) every tied logit is kept and the atomic path runs as before.

#define WHIRL_SPEC_QMAX 32
#define WHIRL_SPEC_QWORDS 72

struct DSampArgs {
    int on[32];
    int top_k[32];
    int qoff[32];  // i32 word offset of this row's q entry in qbuf
    float inv_t[32];
    float top_p[32];
    float min_p[32];
    unsigned key_lo[32];  // race key base (whirl::spec::raceBase)
    unsigned key_hi[32];
};

__device__ __forceinline__ unsigned long long spec_splitmix(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// order-preserving key; NaN -> 0, below every number (-inf is 0x007fffff), so a NaN logit is never a
// candidate ahead of a finite one (FIX-K97 / FR-K97 P3-2; the max below ignores NaN through fmaxf too)
__device__ __forceinline__ unsigned spec_key(float f) {
    const unsigned u = __float_as_uint(f);
    if (f != f) return 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ void __launch_bounds__(1024)
    draft_sample_rows(const float *x, int n, int *ctl_all, RowIdx area, int r, const int *map, DSampArgs a, int *qbuf) {
    __shared__ unsigned hist[32][256];
    __shared__ float red[1024];
    __shared__ unsigned s_prefix, s_remain, s_gt, s_eq, s_eqn, s_taken;
    __shared__ unsigned s_wcnt[32];
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
    for (int i = t; i < n; i += blockDim.x) {
        const float xi = x[i];
        if (xi == xi) sum += __expf((xi - m) * inv_t);  // NaN counts as -inf (adds 0)
    }
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
            if (shift == 0) s_eqn = hist[0][bb];  // logits whose key equals the K-th key
        }
        mask |= 255u << shift;
        __syncthreads();
    }
    if (t == 0) {
        s_gt = 0;
        s_eq = 0;
        s_taken = 0;
    }
    __syncthreads();
    const unsigned thr = s_prefix, need = s_remain, base = (unsigned)K - need;
    const bool ordered = s_eqn > need;  // ties overflow the places left: lowest indices win
    for (int i = t; i < n; i += blockDim.x) {
        const float xv = x[i];
        const float v = xv == xv ? xv : -INFINITY;  // a NaN only fills a slot past the finite logits, with p = 0
        const unsigned k = spec_key(xv);
        if (k > thr) {
            const unsigned j = atomicAdd(&s_gt, 1u);
            if (j < base) {
                s_id[j] = i;
                s_lg[j] = v;
            }
        } else if (k == thr && !ordered) {
            const unsigned j = atomicAdd(&s_eq, 1u);
            if (j < need) {
                s_id[base + j] = i;
                s_lg[base + j] = v;
            }
        }
    }
    if (ordered) {
        // index-ordered scan over blockDim-wide sweeps: a tied logit's place is the number of tied
        // logits with a lower index (earlier sweeps + earlier waves + lower lanes); same code as topk_rows
        const int lane = t % WARP, wv = t / WARP, nw = (int)blockDim.x / WARP;
        for (int c0 = 0; c0 < n; c0 += blockDim.x) {
            const int i = c0 + t;
            const bool f = i < n && spec_key(x[i]) == thr;
            const unsigned long long bal = __ballot(f);
            if (lane == 0) s_wcnt[wv] = (unsigned)__popcll(bal);
            __syncthreads();
            unsigned j = s_taken;
            for (int q = 0; q < wv; ++q) j += s_wcnt[q];
            j += (unsigned)__popcll(bal & ((1ull << lane) - 1ull));
            if (f && j < need) {
                const float xv = x[i];
                s_id[base + j] = i;
                s_lg[base + j] = xv == xv ? xv : -INFINITY;
            }
            __syncthreads();
            if (t == 0) {
                unsigned s = 0;
                for (int q = 0; q < nw; ++q) s += s_wcnt[q];
                s_taken += s;
            }
            __syncthreads();
            if (s_taken >= need) break;  // block-uniform (shared, after the barrier)
        }
    }
    __syncthreads();
    if (t != 0) return;
    // the candidate SET is deterministic (ties above); the slot order is not: ids are mapped first and
    // sorted by (logit desc, id asc)
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
    // race draw (same arithmetic as whirl::spec::sampleRace / raceE)
    const unsigned long long rkey = (static_cast<unsigned long long>(a.key_hi[b]) << 32) | a.key_lo[b];
    int pick = 0;
    double bk = __longlong_as_double(0x7FF0000000000000ll);  // +inf
    for (int i = 0; i < nk; ++i) {
        if (!(p[i] > 0.f)) continue;
        const unsigned long long z = spec_splitmix(rkey + (static_cast<unsigned long long>(static_cast<unsigned>(id[i])) + 1ull) * 0xE7037ED1A0B428DBull);
        const double u = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
        const double k = -log(1.0 - u) / static_cast<double>(p[i]);
        if (k < bk) {
            bk = k;
            pick = i;
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
