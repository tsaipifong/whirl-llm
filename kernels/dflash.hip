// DFlash2 block-diffusion drafter: the kernels it needs beyond the trunk's
// (the math follows llama.cpp's src/models/dflash.cpp and z-lab/dflash, both
// MIT; written from the formulas, no code copied):
//   dflash_conv    grouped dynamic causal convolution inside one draft block
//   dflash_topk    deterministic top-k per logits row (ties: smaller index)
//   dflash_lattice selector scores for each predecessor x candidate pair
//   dflash_walk    greedy walk over the lattice (p-min / n-min) -> draft tokens
// Correctness first: none of these is on a bandwidth-critical path (8-row
// draft blocks; the vocabulary pass of dflash_topk is the only large read).

// Dynamic convolution of one side (0: on the norm output before the projection,
// 1: on the attention / FFN output before the residual add), x / y [T][C]:
//   y[t, c] = sum_{tap < min(K, T), tap <= t} (base[side][tap][c] + dyn[t][g(c) + G (tap + K side)]) x[t - tap, c]
// with g(c) = c / group and G = C / group (the projection's output layout); the
// window never reaches before the block start. grid (ceil(C / 256), T), block 256.
extern "C" __global__ void __launch_bounds__(256)
    dflash_conv(const float *x, const float *dyn, const float *base, float *y, int T, int C, int K, int group, int side,
                int dyn_stride) {
#pragma clang fp contract(off)
    const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (c >= C || t >= T) return;
    const int G = C / group;
    float acc = 0.f;
    for (int tap = 0; tap < K && tap <= t; ++tap) {
        const float w = base[(size_t)(side * K + tap) * C + c] + dyn[(size_t)t * dyn_stride + c / group + G * (tap + K * side)];
        acc += w * x[(size_t)(t - tap) * C + c];
    }
    y[(size_t)t * C + c] = acc;
}

// Ordering of (value, index) keys: larger value first, then smaller index; NaN
// counts as -inf so it never wins over a number.
__device__ __forceinline__ bool dflash_better(float va, int ia, float vb, int ib) {
    if (va != va) va = -INFINITY;
    if (vb != vb) vb = -INFINITY;
    return va > vb || (va == vb && ia < ib);
}

// Top-k (k <= 64) of each logits row, in key order (deterministic, unlike a
// radix select): round r takes the best key below round r-1's. One block of
// 1024 per row; ids / vals [row][k]. grid (rows), block 1024.
extern "C" __global__ void __launch_bounds__(1024)
    dflash_topk(const float *x, int n, int ld, int k, int *ids, float *vals) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float *row = x + (size_t)blockIdx.x * ld;
    float pv = INFINITY;  // previous key: (+inf, -1) is above every key
    int pi = -1;
    for (int r = 0; r < k; ++r) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float v = row[i];
            if (dflash_better(pv, pi, v, i) && dflash_better(v, i, best, idx)) {
                best = v;
                idx = i;
            }
        }
        bv[threadIdx.x] = best;
        bi[threadIdx.x] = idx;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s && dflash_better(bv[threadIdx.x + s], bi[threadIdx.x + s], bv[threadIdx.x], bi[threadIdx.x])) {
                bv[threadIdx.x] = bv[threadIdx.x + s];
                bi[threadIdx.x] = bi[threadIdx.x + s];
            }
            __syncthreads();
        }
        pv = bv[0];
        pi = bi[0];
        if (threadIdx.x == 0) {
            ids[blockIdx.x * k + r] = pi;
            vals[blockIdx.x * k + r] = pv;
        }
        __syncthreads();
    }
}

// Codebook row element: f32 (q8 = 0) or Q8_0 (q8 = 1: 34-byte blocks of 32).
__device__ __forceinline__ float dflash_cb(const uint8_t *cb, uint64_t rb, int row, int j, int q8) {
    const uint8_t *r = cb + (size_t)row * rb;
    if (!q8) return ((const float *)r)[j];
    const uint8_t *b = r + (size_t)(j / 32) * 34;
    return h2f(b) * (float)((const int8_t *)(b + 2))[j % 32];
}

// Selector lattice for draft positions p = 0 .. P-1 (block rows 1 .. P):
//   lat[p][i][j] = sum_r Prev[pred_i][r] * gate[p][r] * Next[cand[p][j]][r] + unary[p][j]
// pred_i = cand[p-1][i], or the anchor token for p = 0 (all K rows the same).
// cand / unary [P][K], gate [P][R], lat [P][K][K]. grid (P, K), block 32 * K
// (wave j: candidate j).
extern "C" __global__ void __launch_bounds__(512)
    dflash_lattice(const int *cand, const float *unary, const float *gate, const uint8_t *prev_cb, const uint8_t *next_cb,
                   uint64_t cb_rb, int q8, int R, int K, int anchor, float *lat) {
#pragma clang fp contract(off)
    const int p = blockIdx.x, i = blockIdx.y, lane = threadIdx.x % WARP, j = threadIdx.x / WARP;
    if (j >= K) return;
    const int pred = p == 0 ? anchor : cand[(p - 1) * K + i];
    const int nx = cand[p * K + j];
    float s = 0.f;
    for (int r = lane; r < R; r += WARP)
        s += dflash_cb(prev_cb, cb_rb, pred, r, q8) * gate[(size_t)p * R + r] * dflash_cb(next_cb, cb_rb, nx, r, q8);
    s = warp_sum(s);
    if (lane == 0) lat[((size_t)p * K + i) * K + j] = s + unary[p * K + j];
}

// Greedy walk (llama.cpp's DFlash2 walk): start at predecessor 0, take the first
// maximum of row pred at each position; stop when its softmax probability over
// the K scores is below p_min (p_min > 0); drop everything if fewer than n_min.
// out[0] = number of drafts, out[1 ..] = tokens. One thread (P x K is tiny).
extern "C" __global__ void dflash_walk(const float *lat, const int *cand, int P, int K, float p_min, int n_min, int *out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int n = 0, pred = 0;
    for (int p = 0; p < P; ++p) {
        const float *s = lat + ((size_t)p * K + pred) * K;
        int jb = 0;
        for (int j = 1; j < K; ++j)
            if (s[j] > s[jb]) jb = j;
        if (p_min > 0.f) {
            float sum = 0.f;
            for (int j = 0; j < K; ++j) sum += __expf(s[j] - s[jb]);
            if (1.f / sum < p_min) break;
        }
        out[1 + n] = cand[p * K + jb];
        n += 1;
        pred = jb;
    }
    out[0] = n < n_min ? 0 : n;
}

// Drafter attention, non-causal over the context ring plus the block's own rows.
// q [B][nh][hd], kb / vb [B][nkv][hd] (the block), kc / vc [W][nkv][hd] (context
// ring, f32), cpos [W] = position of each ring row (-1 empty). Query row t sits at
// qpos + t; a context key is visible iff 0 <= cpos < qpos and (window == 0 or
// qpos + t - cpos < window); every block key is visible. out [B][nh][hd].
// grid (nh, B), block hd (one thread per dimension; hd <= 256, W + B <= 2304).
// Longer rings (W + B > DF_MAXK, e.g. a 4096-row window with 16-row blocks) use
// dflash_attn_long below; this kernel stays as it is for every drafter that fits.
#define DF_MAXK 2304  // = kernels::kDFlashAttnKeys
extern "C" __global__ void __launch_bounds__(256)
    dflash_attn(const float *q, const float *kb, const float *vb, const float *kc, const float *vc, const int *cpos, int W, int B,
                int nh, int nkv, int hd, int qpos, int window, float scale, float *out) {
    __shared__ float qs[256];
    __shared__ float sc[DF_MAXK];
    const int h = blockIdx.x, t = blockIdx.y, g = h / (nh / nkv), tid = threadIdx.x;
    const int my_pos = qpos + t, nk = W + B;
    if (tid < hd) qs[tid] = q[((size_t)t * nh + h) * hd + tid];
    __syncthreads();
    float mloc = -INFINITY;
    for (int k = tid; k < nk; k += blockDim.x) {
#include "shared/dflash_kscore.h"  // s = scaled q.k of key k, -inf if not visible (ADR-0005)
        sc[k] = s;
        mloc = fmaxf(mloc, s);
    }
    const float m = block_max(mloc);
    float lloc = 0.f;
    for (int k = tid; k < nk; k += blockDim.x) {
        const float e = sc[k] == -INFINITY ? 0.f : __expf(sc[k] - m);
        sc[k] = e;
        lloc += e;
    }
    const float l = block_sum(lloc);  // block_sum's barriers also publish sc[]
    if (tid < hd) {
        float o = 0.f;
        for (int k = 0; k < nk; ++k) {
            const float p = sc[k];
            if (p == 0.f) continue;
            const float *vr = k < W ? vc + ((size_t)k * nkv + g) * hd : vb + ((size_t)(k - W) * nkv + g) * hd;
            o += p * vr[tid];
        }
        out[((size_t)t * nh + h) * hd + tid] = o / l;
    }
}

// dflash_attn for W + B > DF_MAXK (C-52): the keys in chunks of DF_MAXK with the
// softmax combined online. Per chunk the same steps as dflash_attn (scores,
// block max, exp, block sum, the thread's output dimension); the chunks are
// merged with the running max: m' = max(m, m_c), l = l e^(m - m') + l_c e^(m_c - m'),
// o likewise; out = o / l. A chunk with no visible key is skipped (the block's own
// keys are always visible, so l > 0). The LDS use is that of dflash_attn
// (DF_MAXK + 256 floats); same grid / block, no limit on W.
extern "C" __global__ void __launch_bounds__(256)
    dflash_attn_long(const float *q, const float *kb, const float *vb, const float *kc, const float *vc, const int *cpos, int W, int B,
                     int nh, int nkv, int hd, int qpos, int window, float scale, float *out) {
    __shared__ float qs[256];
    __shared__ float sc[DF_MAXK];
    const int h = blockIdx.x, t = blockIdx.y, g = h / (nh / nkv), tid = threadIdx.x;
    const int my_pos = qpos + t, nk = W + B;
    if (tid < hd) qs[tid] = q[((size_t)t * nh + h) * hd + tid];
    __syncthreads();
    float m_run = -INFINITY, l_run = 0.f, o_run = 0.f;
    for (int c0 = 0; c0 < nk; c0 += DF_MAXK) {
        const int nc = min(nk - c0, DF_MAXK);
        float mloc = -INFINITY;
        for (int j = tid; j < nc; j += blockDim.x) {
            const int k = c0 + j;
#include "shared/dflash_kscore.h"  // s = scaled q.k of key k, -inf if not visible (ADR-0005)
            sc[j] = s;
            mloc = fmaxf(mloc, s);
        }
        const float m = block_max(mloc);  // uniform across the block
        if (m == -INFINITY) continue;     // no visible key in this chunk (sc[] was not read: no barrier needed)
        float lloc = 0.f;
        for (int j = tid; j < nc; j += blockDim.x) {
            const float e = sc[j] == -INFINITY ? 0.f : __expf(sc[j] - m);
            sc[j] = e;
            lloc += e;
        }
        const float l = block_sum(lloc);  // block_sum's barriers also publish sc[]
        float o = 0.f;
        if (tid < hd) {
            for (int j = 0; j < nc; ++j) {
                const float p = sc[j];
                if (p == 0.f) continue;
                const int k = c0 + j;
                const float *vr = k < W ? vc + ((size_t)k * nkv + g) * hd : vb + ((size_t)(k - W) * nkv + g) * hd;
                o += p * vr[tid];
            }
        }
        const float mn = fmaxf(m_run, m);
        const float a = __expf(m_run - mn), b = __expf(m - mn);  // m_run = -inf: a = 0
        l_run = l_run * a + l * b;
        o_run = o_run * a + o * b;
        m_run = mn;
        __syncthreads();  // every thread has read sc[] before the next chunk writes it
    }
    if (tid < hd) out[((size_t)t * nh + h) * hd + tid] = o_run / l_run;
}
