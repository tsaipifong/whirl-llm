// WHIRL vision encoder kernels: Qwen3-VL / Qwen3.5 ViT ("qwen3vl_merger"
// mmproj, F16 weights) for RDNA 4 (gfx1201) and RDNA 3.5 (gfx1151, VIS-8S).
// SPDX-License-Identifier: Apache-2.0 (graph adapted from llama.cpp, MIT; see
// THIRD_PARTY_NOTICES.md). Ported from the research prototype (item VIS).
//
// A separate code object from kernels/whirl_kernels.hip: it is loaded only when a request
// carries an image (hipModuleLoadData on first use) and unloaded after the
// vision idle timeout, so a text-only process never has it in VRAM.
//
// Graph (ported from llama.cpp tools/mtmd/models/qwen3vl.cpp, MIT):
//   patches (2x2 merge order) -> conv(W0)+conv(W1) (temporal patch 2, same frame)
//   + patch bias + bilinear(align corners) learned position embedding
//   27 x [LN -> qkv (+b) -> 2D vision RoPE -> full attention -> out (+b) -> +res
//         LN -> up (+b) -> GELU(tanh) -> down (+b) -> +res]
//   post LN -> view 4 patches as one row -> mm.0 (+b) -> GELU -> mm.2 (+b)
// Activations: f32 residual stream, f16 GEMM inputs, f32 accumulation.
//
// One source, two code objects (cmake/vision.cmake): gfx1201 compiles exactly the
// code in the `#else` branches (unchanged from the gfx1201-only file); gfx1151
// (__GFX11__) takes the `#if defined(__GFX11__)` branches. The arithmetic is the same
// (f16 inputs, f32 accumulation, 16x16x16 WMMA); only the wave32 fragment layouts
// differ: gfx12 lanes hold 8 k values (k = 8 * (lane / 16) ..) and accumulator rows
// v + 8 * (lane / 16); gfx11 lanes hold all 16 k values (lanes 16-31 repeat 0-15) and
// accumulator rows 2 * v + lane / 16 (llama.cpp ggml-cuda/mma.cuh, RDNA3).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define WARP 32
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
#if defined(__GFX11__)
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
// 16 consecutive halfs (two 16-byte reads) = one gfx11 A / B fragment
__device__ __forceinline__ half16 cat_frag(half8 lo, half8 hi) {
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
__device__ __forceinline__ half16 ld_frag16(const _Float16 *p) { return cat_frag(*(const half8 *)p, *(const half8 *)(p + 8)); }
#define VWMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
// the other half-wave's copy of a 32-bit value (lane ^ 16)
__device__ __forceinline__ int xhalf(int v) {
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false, false);
}
// P^T values of this lane (keys 2v + h of one query, gfx11 accumulator order) -> the
// full 16-key B fragment (both half-waves hold it). Every lane must call it.
__device__ __forceinline__ half16 p_to_frag(half8 p, int lane) {
    const int4 o = __builtin_bit_cast(int4, p);
    const int4 q = make_int4(xhalf(o.x), xhalf(o.y), xhalf(o.z), xhalf(o.w));
    const bool h = lane >= 16;
    const half8 ev = __builtin_bit_cast(half8, h ? q : o), od = __builtin_bit_cast(half8, h ? o : q);
    return __builtin_shufflevector(ev, od, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
}
#endif

#define VD 72    // head dim
#define VDP 80   // padded head dim (5 WMMA k-steps)
#define VH 16    // heads
#define VE 1152  // embedding

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, WARP);
    return v;
}

__device__ float block_sum(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + WARP - 1) / WARP;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ __forceinline__ void lds_barrier() {
#if defined(__GFX11__)
    __syncthreads();
#else
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
#endif
}

// token index t (2x2 merge order) -> patch (px, py); pw = patches per row
__device__ __forceinline__ void tok_xy(int t, int pw, int &px, int &py) {
    const int blk = t >> 2, q = t & 3;
    const int bw = pw >> 1;
    const int y2 = blk / bw, x2 = blk % bw;
    px = 2 * x2 + (q & 1);
    py = 2 * y2 + (q >> 1);
}

// ---------------------------------------------------------------------------
// Patch input: RGB u8 image (W x H, preprocessed size, multiple of 32) ->
// P[t][1536] f16 = [patch (c, ky, kx) ; same patch again] (the two temporal
// kernels of a still image see the same frame; one GEMM over K = 1536 is
// conv(W0) + conv(W1)). Normalization as llama.cpp: (u8/255 - mean)/std.
// grid = N tokens, block 256.
extern "C" __global__ void vis_patchify(const uint8_t *img, _Float16 *P, int W, int pw, float m0, float m1, float m2,
                                        float s0, float s1, float s2) {
    const int t = blockIdx.x;
    int px, py;
    tok_xy(t, pw, px, py);
    _Float16 *dst = P + (size_t)t * 1536;
    for (int i = threadIdx.x; i < 768; i += blockDim.x) {
        const int c = i / 256, ky = (i / 16) % 16, kx = i % 16;
        const int x = px * 16 + kx, y = py * 16 + ky;
        const float v = (float)img[((size_t)y * W + x) * 3 + c] / 255.0f;
        const float mean = c == 0 ? m0 : (c == 1 ? m1 : m2);
        const float sd = c == 0 ? s0 : (c == 1 ? s1 : s2);
        const _Float16 h = (_Float16)((v - mean) / sd);
        dst[i] = h;
        dst[i + 768] = h;
    }
}

// X[t][c] += patch_bias[c] + bilinear(align corners) learned position embedding
// (src [n_side * n_side][VE], row-major y, x), then LN -> H (f16) for layer 0.
// grid = N tokens, block 256.
extern "C" __global__ void vis_pos_ln(float *X, const float *pos_src, const float *pbias, int pw, int ph, int n_side,
                                      const float *g, const float *b, _Float16 *H, float eps) {
    __shared__ float row[VE];
    const int t = blockIdx.x;
    int px, py;
    tok_xy(t, pw, px, py);
    // ggml upscale bilinear, GGML_SCALE_FLAG_ALIGN_CORNERS (pixel offset 0)
    const float sf0 = (pw > 1 && n_side > 1) ? (float)(pw - 1) / (n_side - 1) : (float)pw / n_side;
    const float sf1 = (ph > 1 && n_side > 1) ? (float)(ph - 1) / (n_side - 1) : (float)ph / n_side;
    const bool same = pw == n_side && ph == n_side;
    const float ysf = (float)py / sf1;
    int y0 = (int)floorf(ysf), y1 = y0 + 1;
    y0 = max(0, min(y0, n_side - 1));
    y1 = max(0, min(y1, n_side - 1));
    float dy = ysf - (float)y0;
    dy = fmaxf(0.0f, fminf(dy, 1.0f));
    const float xsf = (float)px / sf0;
    int x0 = (int)floorf(xsf), x1 = x0 + 1;
    x0 = max(0, min(x0, n_side - 1));
    x1 = max(0, min(x1, n_side - 1));
    float dx = xsf - (float)x0;
    dx = fmaxf(0.0f, fminf(dx, 1.0f));
    float *xr = X + (size_t)t * VE;
    float s = 0.f;
    for (int c = threadIdx.x; c < VE; c += blockDim.x) {
        float pe;
        if (same) {
            pe = pos_src[((size_t)py * n_side + px) * VE + c];
        } else {
            const float va = pos_src[((size_t)y0 * n_side + x0) * VE + c];
            const float vb = pos_src[((size_t)y0 * n_side + x1) * VE + c];
            const float vc = pos_src[((size_t)y1 * n_side + x0) * VE + c];
            const float vd = pos_src[((size_t)y1 * n_side + x1) * VE + c];
            pe = va * (1.0f - dx) * (1.0f - dy) + vb * dx * (1.0f - dy) + vc * (1.0f - dx) * dy + vd * dx * dy;
        }
        const float v = (xr[c] + pbias[c]) + pe;
        xr[c] = v;
        row[c] = v;
        s += v;
    }
    const float mean = block_sum(s) / VE;
    float q = 0.f;
    for (int c = threadIdx.x; c < VE; c += blockDim.x) {
        const float d = row[c] - mean;
        q += d * d;
    }
    const float r = rsqrtf(block_sum(q) / VE + eps);
    _Float16 *hr = H + (size_t)t * VE;
    for (int c = threadIdx.x; c < VE; c += blockDim.x) hr[c] = (_Float16)((row[c] - mean) * r * g[c] + b[c]);
}

// X[t] += T[t] + bias (residual), then H = LN(X[t]) (f16; g == null: no LN).
// grid = N tokens, block 256.
extern "C" __global__ void vis_resid_ln(float *X, const float *T, const float *bias, const float *g, const float *b,
                                        _Float16 *H, float eps) {
    __shared__ float row[VE];
    const int t = blockIdx.x;
    float *xr = X + (size_t)t * VE;
    const float *tr = T + (size_t)t * VE;
    float s = 0.f;
    for (int c = threadIdx.x; c < VE; c += blockDim.x) {
        const float v = xr[c] + (tr[c] + bias[c]);
        xr[c] = v;
        row[c] = v;
        s += v;
    }
    if (g == nullptr) return;
    const float mean = block_sum(s) / VE;
    float q = 0.f;
    for (int c = threadIdx.x; c < VE; c += blockDim.x) {
        const float d = row[c] - mean;
        q += d * d;
    }
    const float r = rsqrtf(block_sum(q) / VE + eps);
    _Float16 *hr = H + (size_t)t * VE;
    for (int c = threadIdx.x; c < VE; c += blockDim.x) hr[c] = (_Float16)((row[c] - mean) * r * g[c] + b[c]);
}

// qkv GEMM output T[t][3 * VE] (+ bias) -> 2D vision RoPE on q, k (NEOX pairs
// (i, i + 36): i < 18 rotate by row * theta^i, i >= 18 by col * theta^(i - 18);
// ggml GGML_ROPE_TYPE_VISION, sections {18, 18}) -> Q, K, V [VH][N][VDP] f16
// (dims 72..79 zero). grid = (N, VH), block 64.
extern "C" __global__ void vis_qkv_prep(const float *T, const float *bias, _Float16 *Q, _Float16 *K, _Float16 *V, int N,
                                        int pw, float theta_scale) {
    const int t = blockIdx.x, h = blockIdx.y;
    int px, py;
    tok_xy(t, pw, px, py);
    const float *tr = T + (size_t)t * 3 * VE + h * VD;
    const float *br = bias + h * VD;
    const size_t o = ((size_t)h * N + t) * VDP;
    const int i = threadIdx.x;
    if (i < VD / 2) {
        const int sec = i;  // sector = i (sect_dims = 36)
        const float p = sec < 18 ? (float)py : (float)px;
        const int e = sec < 18 ? sec : sec - 18;
        const float theta = p * powf(theta_scale, (float)e);
        const float c = cosf(theta), s = sinf(theta);
        {
            const float x0 = tr[i] + br[i], x1 = tr[i + 36] + br[i + 36];
            Q[o + i] = (_Float16)(x0 * c - x1 * s);
            Q[o + i + 36] = (_Float16)(x0 * s + x1 * c);
        }
        {
            const float x0 = tr[VE + i] + br[VE + i], x1 = tr[VE + i + 36] + br[VE + i + 36];
            K[o + i] = (_Float16)(x0 * c - x1 * s);
            K[o + i + 36] = (_Float16)(x0 * s + x1 * c);
        }
    }
    if (i < VD) V[o + i] = (_Float16)(tr[2 * VE + i] + br[2 * VE + i]);
    if (i >= VD && i < VDP) {
        Q[o + i] = (_Float16)0.f;
        K[o + i] = (_Float16)0.f;
        V[o + i] = (_Float16)0.f;
    }
}

// ---------------------------------------------------------------------------
// Full (non-causal) flash attention over N tokens, head dim 72 (padded 80),
// f16 WMMA (gfx12 16x16x16, f32 accumulate). Block = 4 waves x 16 queries of
// one head; 64-key tiles of K and V^T in LDS. Each wave computes S^T = K Q^T
// (lane: one query, 8 keys) so P^T is directly the B fragment of O^T += V^T P^T.
// Online softmax in base 2. O -> [N][VE] f16 (head h at columns h*72 ..).
// grid = (ceil(N / 64), VH), block 128.
#define FA_BQ 64
#define FA_BK 64
#if !defined(__GFX11__)  // v1 (the host uses vis_attn2): gfx1201 only
extern "C" __global__ void __launch_bounds__(128) vis_attn(const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                                                           _Float16 *O, int N, float scale_log2) {
    __shared__ __attribute__((aligned(16))) _Float16 Ks[FA_BK * VDP];
    __shared__ __attribute__((aligned(16))) _Float16 Vt[VDP * FA_BK];
    const int h = blockIdx.y;
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int q0 = blockIdx.x * FA_BQ + wid * 16;
    const _Float16 *Qh = Q + (size_t)h * N * VDP;
    const _Float16 *Kh = K + (size_t)h * N * VDP;
    const _Float16 *Vh = V + (size_t)h * N * VDP;
    // B fragments of Q^T: lane = query q0 + lane%16, dims ks*16 + (lane/16)*8 .. +8
    half8 qf[5];
    {
        int qi = q0 + lane % 16;
        qi = qi < N ? qi : N - 1;
#pragma unroll
        for (int ks = 0; ks < 5; ++ks) qf[ks] = *(const half8 *)(Qh + (size_t)qi * VDP + ks * 16 + (lane / 16) * 8);
    }
    float8 acc[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) acc[i] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    for (int k0 = 0; k0 < N; k0 += FA_BK) {
        // K tile [64][80] and V^T tile [80][64]
        for (int e = threadIdx.x; e < FA_BK * VDP / 8; e += blockDim.x) {
            const int r = e / (VDP / 8), c8 = (e % (VDP / 8)) * 8;
            int kr = k0 + r;
            const bool ok = kr < N;
            kr = ok ? kr : N - 1;
            const half8 kv = *(const half8 *)(Kh + (size_t)kr * VDP + c8);
            *(half8 *)(Ks + r * VDP + c8) = kv;
            const half8 vv = *(const half8 *)(Vh + (size_t)kr * VDP + c8);
#pragma unroll
            for (int j = 0; j < 8; ++j) Vt[(c8 + j) * FA_BK + r] = ok ? vv[j] : (_Float16)0.f;
        }
        lds_barrier();
#pragma unroll
        for (int sub = 0; sub < FA_BK / 16; ++sub) {
            float8 s = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int ks = 0; ks < 5; ++ks) {
                const half8 a = *(const half8 *)(Ks + (sub * 16 + lane % 16) * VDP + ks * 16 + (lane / 16) * 8);
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, qf[ks], s);
            }
            // lane holds keys k0 + sub*16 + 8*(lane/16) + v for query lane%16
            float mt = -INFINITY;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int key = k0 + sub * 16 + 8 * (lane / 16) + v;
                s[v] = key < N ? s[v] * scale_log2 : -INFINITY;
                mt = fmaxf(mt, s[v]);
            }
            mt = fmaxf(mt, __shfl_xor(mt, 16, WARP));
            const float mn = fmaxf(m, mt);
            const float corr = exp2f(m - mn);  // m = -inf on the first tile: 0
            m = mn;
            half8 p;
            float ps = 0.f;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float e = exp2f(s[v] - mn);
                ps += e;
                p[v] = (_Float16)e;
            }
            l = l * corr + ps;
#pragma unroll
            for (int i = 0; i < 5; ++i) {
#pragma unroll
                for (int v = 0; v < 8; ++v) acc[i][v] *= corr;
                const half8 a = *(const half8 *)(Vt + (i * 16 + lane % 16) * FA_BK + sub * 16 + (lane / 16) * 8);
                acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, p, acc[i]);
            }
        }
        lds_barrier();
    }
    const float L = l + __shfl_xor(l, 16, WARP);
    const float inv = 1.0f / L;
    const int q = q0 + lane % 16;
    if (q >= N) return;
    _Float16 *orow = O + (size_t)q * VE + h * VD;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        const int d0 = i * 16 + 8 * (lane / 16);
        if (d0 >= VD) continue;
        half8 hv;
#pragma unroll
        for (int v = 0; v < 8; ++v) hv[v] = (_Float16)(acc[i][v] * inv);
        *(half8 *)(orow + d0) = hv;  // h*72 + d0: 16-byte aligned (72*2 = 144 = 9*16)
    }
}
#endif  // !__GFX11__

// ---------------------------------------------------------------------------
// v2 (faster): qkv prep per 64-token tile with V written transposed, and flash
// attention with 32 queries per wave, 64-key softmax steps, double-buffered
// LDS tiles. Q, K: [VH][Np][VDP]; Vt: [VH][VDP][Np] (Np = N rounded up to 128,
// rows / columns >= N are zero).
// grid = (Np / 64, VH), block 256.
extern "C" __global__ void __launch_bounds__(256) vis_qkv_prep2(const float *T, const float *bias, _Float16 *Q, _Float16 *K,
                                                                _Float16 *Vt, int N, int Np, int pw, float theta_scale) {
    __shared__ _Float16 vs[VDP][65];
    const int t0 = blockIdx.x * 64, h = blockIdx.y;
    // q / k: thread = (token tt, pair i) for 36 pairs + pads
    for (int e = threadIdx.x; e < 64 * 40; e += blockDim.x) {
        const int tt = e / 40, j = e % 40;
        const int t = t0 + tt;
        const size_t o = ((size_t)h * Np + t) * VDP;
        if (t >= N) {
            Q[o + j] = Q[o + j + 40] = (_Float16)0.f;
            K[o + j] = K[o + j + 40] = (_Float16)0.f;
            continue;
        }
        if (j >= 36) {  // pads 72..79 (j = 36..39 -> 72..75 and 76..79; the buffers are lent
                        // scratch with old contents, so every pad must be written)
            Q[o + 36 + j] = Q[o + 40 + j] = (_Float16)0.f;
            K[o + 36 + j] = K[o + 40 + j] = (_Float16)0.f;
            continue;
        }
        int px, py;
        tok_xy(t, pw, px, py);
        const float *tr = T + (size_t)t * 3 * VE + h * VD;
        const float *br = bias + h * VD;
        const float p = j < 18 ? (float)py : (float)px;
        const int ex = j < 18 ? j : j - 18;
        const float theta = p * powf(theta_scale, (float)ex);
        const float c = cosf(theta), s = sinf(theta);
        {
            const float x0 = tr[j] + br[j], x1 = tr[j + 36] + br[j + 36];
            Q[o + j] = (_Float16)(x0 * c - x1 * s);
            Q[o + j + 36] = (_Float16)(x0 * s + x1 * c);
        }
        {
            const float x0 = tr[VE + j] + br[VE + j], x1 = tr[VE + j + 36] + br[VE + j + 36];
            K[o + j] = (_Float16)(x0 * c - x1 * s);
            K[o + j + 36] = (_Float16)(x0 * s + x1 * c);
        }
    }
    // v -> LDS [d][tt] -> Vt rows
    for (int e = threadIdx.x; e < 64 * VDP; e += blockDim.x) {
        const int tt = e / VDP, d = e % VDP;
        const int t = t0 + tt;
        float v = 0.f;
        if (t < N && d < VD) v = T[(size_t)t * 3 * VE + 2 * VE + h * VD + d] + bias[2 * VE + h * VD + d];
        vs[d][tt] = (_Float16)v;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < VDP * 64; e += blockDim.x) {
        const int d = e / 64, tt = e % 64;
        Vt[((size_t)h * VDP + d) * Np + t0 + tt] = vs[d][tt];
    }
}

#if !defined(__GFX11__)
// grid = (Np / 128, VH), block 128 (4 waves x 32 queries).
#define FB_BQ 128
#define FB_BK 64
#define FB_KS 80  // LDS row stride of K (halfs; padding it cost occupancy, measured)
#define FB_VS 68  // LDS row stride of V^T (padded: 64 gave 8-way bank conflicts; 1080p attn 170 -> 150 ms)
extern "C" __global__ void __launch_bounds__(128) vis_attn2(const _Float16 *__restrict__ Q, const _Float16 *__restrict__ K,
                                                            const _Float16 *__restrict__ Vt, _Float16 *__restrict__ O, int N,
                                                            int Np, float scale_log2) {
    __shared__ __attribute__((aligned(16))) _Float16 Ks[2][FB_BK * FB_KS];
    __shared__ __attribute__((aligned(16))) _Float16 Vs[2][VDP * FB_VS];
    const int h = blockIdx.y;
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int q0 = blockIdx.x * FB_BQ + wid * 32;
    const _Float16 *Qh = Q + (size_t)h * Np * VDP;
    const _Float16 *Kh = K + (size_t)h * Np * VDP;
    const _Float16 *Vh = Vt + (size_t)h * VDP * Np;
    half8 qf[2][5];
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
        for (int ks = 0; ks < 5; ++ks)
            qf[f][ks] = *(const half8 *)(Qh + (size_t)(q0 + f * 16 + lane % 16) * VDP + ks * 16 + (lane / 16) * 8);
    float8 acc[2][5];
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
        for (int i = 0; i < 5; ++i) acc[f][i] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const int ntile = (N + FB_BK - 1) / FB_BK;
    auto load = [&](int tile, int b) {
        const int k0 = tile * FB_BK;
        // K: 64 rows x 80 = contiguous 640 half8; Vt: 80 rows x 64 keys (row stride Np)
#pragma nounroll  // trip count depends on threadIdx.x: never unrolled (an unroll hint here only warned)
        for (int e = threadIdx.x; e < FB_BK * VDP / 8; e += 128) {
            const int r = e / (VDP / 8), c = (e % (VDP / 8)) * 8;
            *(half8 *)(&Ks[b][r * FB_KS + c]) = *(const half8 *)(Kh + (size_t)k0 * VDP + e * 8);
        }
#pragma nounroll
        for (int e = threadIdx.x; e < VDP * FB_BK / 8; e += 128) {
            const int d = e / (FB_BK / 8), c = (e % (FB_BK / 8)) * 8;
            *(half8 *)(&Vs[b][d * FB_VS + c]) = *(const half8 *)(Vh + (size_t)d * Np + k0 + c);
        }
    };
    load(0, 0);
    lds_barrier();
    for (int tile = 0; tile < ntile; ++tile) {
        const int b = tile & 1;
        if (tile + 1 < ntile) load(tile + 1, b ^ 1);
        const int k0 = tile * FB_BK;
        float8 s[2][4];
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            s[0][sub] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
            s[1][sub] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int ks = 0; ks < 5; ++ks) {
                const half8 a = *(const half8 *)(&Ks[b][(sub * 16 + lane % 16) * FB_KS + ks * 16 + (lane / 16) * 8]);
                s[0][sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, qf[0][ks], s[0][sub]);
                s[1][sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, qf[1][ks], s[1][sub]);
            }
        }
        const bool tail = k0 + FB_BK > N;
        half8 p[2][4];
#pragma unroll
        for (int f = 0; f < 2; ++f) {
            float mt = -INFINITY;
#pragma unroll
            for (int sub = 0; sub < 4; ++sub)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    float x = s[f][sub][v] * scale_log2;
                    if (tail && k0 + sub * 16 + 8 * (lane / 16) + v >= N) x = -INFINITY;
                    s[f][sub][v] = x;
                    mt = fmaxf(mt, x);
                }
            mt = fmaxf(mt, __shfl_xor(mt, 16, WARP));
            const float mn = fmaxf(m[f], mt);
            // rescale only when some query's running max moved (rare after the first tiles)
            if (__any(mn > m[f])) {
                const float corr = __builtin_amdgcn_exp2f(m[f] - mn);
                l[f] *= corr;
#pragma unroll
                for (int i = 0; i < 5; ++i)
#pragma unroll
                    for (int v = 0; v < 8; ++v) acc[f][i][v] *= corr;
            }
            m[f] = mn;
            float ps = 0.f;
#pragma unroll
            for (int sub = 0; sub < 4; ++sub)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const float e = __builtin_amdgcn_exp2f(s[f][sub][v] - mn);
                    ps += e;
                    p[f][sub][v] = (_Float16)e;
                }
            l[f] += ps;
        }
#pragma unroll
        for (int sub = 0; sub < 4; ++sub)
#pragma unroll
            for (int i = 0; i < 5; ++i) {
                const half8 a = *(const half8 *)(&Vs[b][(i * 16 + lane % 16) * FB_VS + sub * 16 + (lane / 16) * 8]);
                acc[0][i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, p[0][sub], acc[0][i]);
                acc[1][i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, p[1][sub], acc[1][i]);
            }
        lds_barrier();
    }
#pragma unroll
    for (int f = 0; f < 2; ++f) {
        const float L = l[f] + __shfl_xor(l[f], 16, WARP);
        const float inv = 1.0f / L;
        const int q = q0 + f * 16 + lane % 16;
        if (q >= N) continue;
        _Float16 *orow = O + (size_t)q * VE + h * VD;
#pragma unroll
        for (int i = 0; i < 5; ++i) {
            const int d0 = i * 16 + 8 * (lane / 16);
            if (d0 >= VD) continue;
            half8 hv;
#pragma unroll
            for (int v = 0; v < 8; ++v) hv[v] = (_Float16)(acc[f][i][v] * inv);
            *(half8 *)(orow + d0) = hv;
        }
    }
}
#else
// gfx11 (RDNA 3.5): the same algorithm with 16 queries per wave (the 32-query
// form needs ~330 VGPRs with the 16-k gfx11 fragments and spilled), 8 waves per
// block so a block still covers 128 queries and shares each K / V^T tile.
// grid = (Np / 128, VH), block 256 (host: Vision::attnThreads()).
#define FB_BQ 128
#define FB_BK 64
#define FB_KS 80
#define FB_VS 72  // 16-byte aligned rows for the 32-byte fragment reads
extern "C" __global__ void __launch_bounds__(256) vis_attn2(const _Float16 *__restrict__ Q, const _Float16 *__restrict__ K,
                                                            const _Float16 *__restrict__ Vt, _Float16 *__restrict__ O, int N,
                                                            int Np, float scale_log2) {
    __shared__ __attribute__((aligned(16))) _Float16 Ks[2][FB_BK * FB_KS];
    __shared__ __attribute__((aligned(16))) _Float16 Vs[2][VDP * FB_VS];
    const int h = blockIdx.y;
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int q0 = blockIdx.x * FB_BQ + wid * 16;
    const _Float16 *Qh = Q + (size_t)h * Np * VDP;
    const _Float16 *Kh = K + (size_t)h * Np * VDP;
    const _Float16 *Vh = Vt + (size_t)h * VDP * Np;
    // B fragments of Q^T: lane = query q0 + lane % 16, all 16 dims of k-step ks
    half16 qf[5];
#pragma unroll
    for (int ks = 0; ks < 5; ++ks) qf[ks] = ld_frag16(Qh + (size_t)(q0 + lane % 16) * VDP + ks * 16);
    float8 acc[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) acc[i] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    const int ntile = (N + FB_BK - 1) / FB_BK;
    auto load = [&](int tile, int b) {
        const int k0 = tile * FB_BK;
#pragma nounroll
        for (int e = threadIdx.x; e < FB_BK * VDP / 8; e += 256) {
            const int r = e / (VDP / 8), c = (e % (VDP / 8)) * 8;
            *(half8 *)(&Ks[b][r * FB_KS + c]) = *(const half8 *)(Kh + (size_t)k0 * VDP + e * 8);
        }
#pragma nounroll
        for (int e = threadIdx.x; e < VDP * FB_BK / 8; e += 256) {
            const int d = e / (FB_BK / 8), c = (e % (FB_BK / 8)) * 8;
            *(half8 *)(&Vs[b][d * FB_VS + c]) = *(const half8 *)(Vh + (size_t)d * Np + k0 + c);
        }
    };
    load(0, 0);
    lds_barrier();
    for (int tile = 0; tile < ntile; ++tile) {
        const int b = tile & 1;
        if (tile + 1 < ntile) load(tile + 1, b ^ 1);
        const int k0 = tile * FB_BK;
        // S^T = K Q^T: element v of this lane = key k0 + sub * 16 + 2v + lane / 16, query lane % 16
        float8 s[4];
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            s[sub] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int ks = 0; ks < 5; ++ks) s[sub] = VWMMA(ld_frag16(&Ks[b][(sub * 16 + lane % 16) * FB_KS + ks * 16]), qf[ks], s[sub]);
        }
        const bool tail = k0 + FB_BK > N;
        float mt = -INFINITY;
#pragma unroll
        for (int sub = 0; sub < 4; ++sub)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                float x = s[sub][v] * scale_log2;
                if (tail && k0 + sub * 16 + 2 * v + lane / 16 >= N) x = -INFINITY;
                s[sub][v] = x;
                mt = fmaxf(mt, x);
            }
        mt = fmaxf(mt, __shfl_xor(mt, 16, WARP));
        const float mn = fmaxf(m, mt);
        if (__any(mn > m)) {
            const float corr = __builtin_amdgcn_exp2f(m - mn);
            l *= corr;
#pragma unroll
            for (int i = 0; i < 5; ++i)
#pragma unroll
                for (int v = 0; v < 8; ++v) acc[i][v] *= corr;
        }
        m = mn;
        float ps = 0.f;
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            half8 p;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float e = __builtin_amdgcn_exp2f(s[sub][v] - mn);
                ps += e;
                p[v] = (_Float16)e;
            }
            // O^T += V^T P^T: P^T fragment = all 16 keys of this sub-tile for query lane % 16
            const half16 pb = p_to_frag(p, lane);
#pragma unroll
            for (int i = 0; i < 5; ++i) acc[i] = VWMMA(ld_frag16(&Vs[b][(i * 16 + lane % 16) * FB_VS + sub * 16]), pb, acc[i]);
        }
        l += ps;
        lds_barrier();
    }
    const float L = l + __shfl_xor(l, 16, WARP);
    const float inv = 1.0f / L;
    const int q = q0 + lane % 16;
    if (q >= N) return;
    // element v of this lane = dim i * 16 + 2v + lane / 16; per-element stores (a cross-lane
    // exchange here could be placed after the divergent checks, where permlanex16 reads
    // inactive lanes as 0)
    _Float16 *orow = O + (size_t)q * VE + h * VD;
#pragma unroll
    for (int i = 0; i < 5; ++i)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int d = i * 16 + 2 * v + lane / 16;
            if (d < VD) orow[d] = (_Float16)(acc[i][v] * inv);
        }
}
#endif  // __GFX11__

// GELU (ggml_gelu, tanh approximation) of T[t][n_in] + bias -> M[t][n_out] f16,
// columns n_in .. n_out-1 = 0 (K padding of the next GEMM). grid = N, block 256.
extern "C" __global__ void vis_gelu(const float *T, const float *bias, _Float16 *M, int n_in, int n_out) {
    const int t = blockIdx.x;
    const float *tr = T + (size_t)t * n_in;
    _Float16 *mr = M + (size_t)t * n_out;
    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    for (int c = threadIdx.x; c < n_out; c += blockDim.x) {
        float y = 0.f;
        if (c < n_in) {
            const float x = tr[c] + bias[c];
            y = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
        }
        mr[c] = (_Float16)y;
    }
}

// out[t][c] = T[t][c] + bias[c]  (n = rows * cols elements)
extern "C" __global__ void vis_bias(const float *T, const float *bias, float *out, int cols, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = T[i] + bias[i % cols];
}

// ---------------------------------------------------------------------------
// f16 x f16 GEMM, f32 out: Y[t][r] (+)= sum_k W[r][k] X[t][k]. The body is gemmh_impl of the gfx1201 kernel set
// (kernels/shared/gemmh_impl.h, ADR-0005; it was a copy here), compiled into this code object so it stands alone.
// Row-major W (row stride ncols) and X, ncols % 32 == 0. grid = (ceil(n/256), ceil(nrows/128)).
#define GEMMH_LDS_BARRIER() lds_barrier()
extern "C" __global__ void __launch_bounds__(256) vis_gemm(const _Float16 *__restrict__ Wp, const _Float16 *__restrict__ X,
                                                           float *__restrict__ Y, int ncols, int nrows, int n_tok,
                                                           int accumulate) {
    constexpr bool OH = false;
    constexpr int T = -1;
    const uint8_t *Wb = (const uint8_t *)Wp;
    const uint64_t row_bytes = (uint64_t)ncols * 2;
#include "../shared/gemmh_impl.h"
}
