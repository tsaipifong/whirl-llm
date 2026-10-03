// WHIRL device code (gfx1201 / RDNA 4): sampling candidates (top-k logits per row).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Sampling candidates (server). For each of gridDim.x logits rows
// ([rows][n]): stats[row] = {max, sum_i exp((x_i - max) * inv_t)} (the full
// softmax normalizer at temperature 1/inv_t) and the K largest logits in
// ids/vals[row][0..K) (unordered). The K-th largest value is found with an
// 8-bit radix select over order-preserving float keys (4 passes, per-wave
// histograms to keep shared-atomic contention low).
__device__ __forceinline__ unsigned topk_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ void __launch_bounds__(1024) topk_rows(const float *x, int n, int K, float inv_t, int *ids, float *vals,
                                                              float *stats) {
    __shared__ unsigned hist[32][256];
    __shared__ float red[1024];
    __shared__ unsigned s_prefix, s_remain, s_gt, s_eq;
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
        const unsigned k = topk_key(v);
        if (k > thr) {
            const unsigned j = atomicAdd(&s_gt, 1u);
            if (j < base) {
                ids[j] = i;
                vals[j] = v;
            }
        } else if (k == thr) {
            const unsigned j = atomicAdd(&s_eq, 1u);
            if (j < need) {
                ids[base + j] = i;
                vals[base + j] = v;
            }
        }
    }
}

