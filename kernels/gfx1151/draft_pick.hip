// WHIRL device code (gfx1151 / RDNA 3.5): 2-bit MTP draft head, token tables, greedy / draft picks.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

#define D2_XD(p) xdv(p)  // gfx1151 packed scale words
// ---------------------------------------------------------------------------
// 2-bit MTP draft head ("D2"). Only the draft argmax matters (a wrong draft
// costs one rejected verify row), and measured acceptance does not depend on
// the head's precision (Q6_K / Q4_K / Q3_K heads accept the same), so the
// draft head streams 2.5 bits per weight instead of Q4_K's 4.5.
// Row layout (K columns): codes [K/4 bytes] then f16 scales [K/32]:
//   unit u = 64 values = 16 code bytes at 16u; value at unit position
//   16k + 4kk + j is the 2-bit field (bits 8j + 2kk) of word k, so
//   (w_k >> 2kk) & 0x03030303 gives 4 codes in x order;
//   block b (32 values) has scale s_b: w = s_b * (2c - 3), c in 0..3.
// Arithmetic is free-form (drafts never have to match anything bitwise).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
    requant_q6k_d2(const uint8_t *src, uint64_t src_rb, uint8_t *dst, int ncols) {
    const int row = blockIdx.x, sbi = blockIdx.y, t = threadIdx.x, lane = t & 31, bj = t >> 5;
    const uint8_t *b = src + (size_t)row * src_rb + (size_t)sbi * 210;
    uint8_t *o = dst + (size_t)row * (ncols / 4 + ncols / 16);
    __shared__ float v[256];
    __shared__ uint32_t words[16];
    if (t < 16) words[t] = 0;
    if ((t & 7) == 0) Q<QT_Q6_K>::dec8(b, t, v + t);
    __syncthreads();
    const float x = v[t];
    const float amax = warp_max(fabsf(x));
    // pick the scale (of a few candidates) with the least squared error
    const float fs[6] = {1.f, 0.85f, 0.72f, 0.62f, 0.54f, 0.46f};
    float best_s = amax / 3.f, best_e = INFINITY;
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const float s = amax / 3.f * fs[i];
        const int c = s > 0.f ? max(0, min(3, (int)rintf((x / s + 3.f) * 0.5f))) : 1;
        const float d = x - s * (float)(2 * c - 3);
        const float e = warp_sum(d * d);
        if (e < best_e) {
            best_e = e;
            best_s = s;
        }
    }
    const _Float16 sh = (_Float16)best_s;
    const float sf = (float)sh;
    const int c = sf > 0.f ? max(0, min(3, (int)rintf((x / sf + 3.f) * 0.5f))) : 1;
    const int blk = sbi * 8 + bj;                   // block index in the row
    const int p = 32 * (blk & 1) + lane;            // position in its 64-value unit
    const int k = p >> 4, q = p & 15, kk = q >> 2, j = q & 3;
    atomicOr(&words[(bj >> 1) * 4 + k], (uint32_t)c << (8 * j + 2 * kk));
    __syncthreads();
    if (t < 16) ((uint32_t *)(o + (size_t)sbi * 64))[t] = words[t];  // 4 units x 4 words
    if (lane == 0) *(_Float16 *)(o + ncols / 4 + 2 * blk) = sh;
}

// y[t][r] = draft-head logits for NT tokens (int8 x, xd per 32), one wave per
// row, lane per 64-value unit.
template <int NT>
__device__ void gemv_d2_impl(const uint8_t *__restrict__ W, const int8_t *__restrict__ xq, const float *__restrict__ xd,
                             float *__restrict__ y, int ncols, int nrows, const int *__restrict__ skip) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const size_t rb = (size_t)(ncols / 4 + ncols / 16);
    const uint8_t *codes = W + (size_t)row * rb;
    const uint16_t *sc = (const uint16_t *)(codes + ncols / 4);
    const int nu = ncols / 64, nb = ncols / 32;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int u = lane; u < nu; u += WARP) {
        const uint4 cw = *(const uint4 *)(codes + 16 * u);
        const uint32_t s2 = *(const uint32_t *)(sc + 2 * u);
        const float s0 = __half2float(__ushort_as_half((uint16_t)(s2 & 0xFFFF)));
        const float s1 = __half2float(__ushort_as_half((uint16_t)(s2 >> 16)));
        const uint32_t w[4] = {cw.x, cw.y, cw.z, cw.w};
        int cb[16];
#pragma unroll
        for (int k = 0; k < 4; ++k)
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) cb[4 * k + kk] = (int)((w[k] >> (2 * kk)) & 0x03030303u);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int8_t *xt = xq + (size_t)t * ncols + 64 * u;
            const int4 x0 = *(const int4 *)xt, x1 = *(const int4 *)(xt + 16);
            const int4 x2 = *(const int4 *)(xt + 32), x3 = *(const int4 *)(xt + 48);
            const int xv[16] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w, x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
            int d0 = 0, d1 = 0, e0 = 0, e1 = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                d0 = dp4(cb[i], xv[i], d0);
                e0 = dp4(0x01010101, xv[i], e0);
                d1 = dp4(cb[8 + i], xv[8 + i], d1);
                e1 = dp4(0x01010101, xv[8 + i], e1);
            }
            const float *xdt = xd + (size_t)t * nb + 2 * u;
            acc[t] += s0 * D2_XD(xdt) * (float)(2 * d0 - 3 * e0) + s1 * D2_XD(xdt + 1) * (float)(2 * d1 - 3 * e1);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) y[(size_t)t * nrows + row] = s;
    }
}
#define GEMV_D2_ENTRY(NT)                                                                                   \
    extern "C" __global__ void __launch_bounds__(256)                                                       \
        gemv_d2_nt##NT(const uint8_t *W, const int8_t *xq, const float *xd, float *y, int ncols, int nrows, \
                       const int *skip) {                                                                   \
        gemv_d2_impl<NT>(W, xq, xd, y, ncols, nrows, skip);                                                 \
    }
GEMV_D2_ENTRY(1)
GEMV_D2_ENTRY(2)
GEMV_D2_ENTRY(3)
GEMV_D2_ENTRY(4)
GEMV_D2_ENTRY(5)
GEMV_D2_ENTRY(6)
GEMV_D2_ENTRY(7)
GEMV_D2_ENTRY(8)
GEMV_D2_ENTRY(9)
GEMV_D2_ENTRY(10)
GEMV_D2_ENTRY(11)
GEMV_D2_ENTRY(12)
GEMV_D2_ENTRY(13)
GEMV_D2_ENTRY(14)
GEMV_D2_ENTRY(15)
GEMV_D2_ENTRY(16)

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
struct RowTab {  // kMaxVerifyRows (32) rows, as include/whirl/kernels_abi.h
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

struct RowIdx {  // kMaxVerifyRows (32) rows, as include/whirl/kernels_abi.h
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
                                                float p_min, const int *map) {
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
// starts at ctl_all + area.v[b] (ints; see kCtl* in include/whirl/kernels_abi.h).
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
