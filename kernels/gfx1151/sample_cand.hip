// WHIRL device code (gfx1151 / RDNA 3.5): sampling candidates.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Sampling candidates (server sampling). For each of gridDim.x logits rows
// ([rows][n]): stats[row] = {max, sum_i exp((x_i - max) * inv_t)} (the full
// softmax normalizer at temperature 1/inv_t) and the K largest logits in
// ids/vals[row][0..K) (unordered). The K-th largest value is found with an
// 8-bit radix select over order-preserving float keys (4 passes, per-wave
// histograms to keep shared-atomic contention low).
// Ties at the K-th value (feat027-relaxdet): when more logits equal the K-th
// value than places are left, the LOWEST indices among them are kept (the
// argmax tie rule), chosen by an index-ordered block scan, so the candidate
// set never depends on shared-atomic arrival order. Without overflowing ties
// (the usual case) every tied logit is kept and the atomic path below runs as
// before; the slot order inside ids/vals stays unspecified (callers sort or
// test membership).
__device__ __forceinline__ unsigned topk_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ void __launch_bounds__(1024) topk_rows(const float *x, int n, int K, float inv_t, int *ids, float *vals,
                                                              float *stats) {
    __shared__ unsigned hist[32][256];
    __shared__ float red[1024];
    __shared__ unsigned s_prefix, s_remain, s_gt, s_eq, s_eqn, s_taken;
    __shared__ unsigned s_wcnt[32];
    const int t = threadIdx.x;
    const int w = (t >> 5) & 31;
    x += (size_t)blockIdx.x * n;
    ids += (size_t)blockIdx.x * K;
    vals += (size_t)blockIdx.x * K;
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
    if (t == 0) {
        stats[2 * blockIdx.x] = m;
        stats[2 * blockIdx.x + 1] = red[0];
        s_prefix = 0;
        s_remain = (unsigned)K;
    }
    unsigned mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 32 * 256; i += blockDim.x) (&hist[0][0])[i] = 0;
        __syncthreads();
        const unsigned prefix = s_prefix;
        for (int i = t; i < n; i += blockDim.x) {
            const unsigned k = topk_key(x[i]);
            if ((k & mask) == prefix) atomicAdd(&hist[w][(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (t < 256) {
            unsigned c = 0;
            for (int j = 0; j < 32; ++j) c += hist[j][t];
            hist[0][t] = c;
        }
        __syncthreads();
        if (t == 0) {
            unsigned rem = s_remain;
            int b = 255;
            for (; b > 0; --b) {
                const unsigned c = hist[0][b];
                if (c >= rem) break;
                rem -= c;
            }
            s_prefix = prefix | ((unsigned)b << shift);
            s_remain = rem;
            if (shift == 0) s_eqn = hist[0][b];  // logits equal to the K-th value
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
        const float v = x[i];
        const unsigned k = topk_key(v);
        if (k > thr) {
            const unsigned j = atomicAdd(&s_gt, 1u);
            if (j < base) {
                ids[j] = i;
                vals[j] = v;
            }
        } else if (k == thr && !ordered) {
            const unsigned j = atomicAdd(&s_eq, 1u);
            if (j < need) {
                ids[base + j] = i;
                vals[base + j] = v;
            }
        }
    }
    if (ordered) {
        // index-ordered scan over blockDim-wide sweeps: a tied logit's place is the number of
        // tied logits with a lower index (earlier sweeps + earlier waves + lower lanes)
        const int lane = t % WARP, wv = t / WARP, nw = (int)blockDim.x / WARP;
        for (int c0 = 0; c0 < n; c0 += blockDim.x) {
            const int i = c0 + t;
            const bool f = i < n && topk_key(x[i]) == thr;
            const unsigned long long bal = __ballot(f);
            if (lane == 0) s_wcnt[wv] = (unsigned)__popcll(bal);
            __syncthreads();
            unsigned j = s_taken;
            for (int q = 0; q < wv; ++q) j += s_wcnt[q];
            j += (unsigned)__popcll(bal & ((1ull << lane) - 1ull));
            if (f && j < need) {
                ids[base + j] = i;
                vals[base + j] = x[i];
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
}
