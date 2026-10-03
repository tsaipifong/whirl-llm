// WHIRL device code (gfx1151 / RDNA 3.5): decode fusions (norm + quant, fused DeltaNet).
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Decode-path fusions (one token). Each produces the int8 activation (xq, xd)
// that the following gemvq consumes, so no separate quantize launch is needed.
// ---------------------------------------------------------------------------

// Quantize the 32 values held by one wave (lane i holds v) into xq/xd at block b.
__device__ __forceinline__ void wave_q8(float v, int8_t *xq, float *xd, int b) {
    const int lane = threadIdx.x % WARP;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    const int q = __float2int_rn(v * id);
    const int sum = warp_sum_i(q);
    xq[b * 32 + lane] = (int8_t)q;
    if (lane == 0) xd[b] = pk_make(d, sum);
}

// out = rmsnorm(x) * w (f32, kept for any f32 consumer) and its int8 copy.
// Single block of 1024 threads; n % 32 == 0.
extern "C" __global__ void __launch_bounds__(1024)
    rmsnorm_q8(const float *x, const float *w, float *out, int8_t *xq, float *xd, int n, float eps) {
    // row blockIdx.x (small multi-token batches: speculative verify)
    x += (size_t)blockIdx.x * n;
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

// g = silu(g) * u and its int8 copy. One wave per 32-block.
extern "C" __global__ void silu_mul_q8(float *g, const float *u, int8_t *xq, float *xd, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float a = g[i];
    const float v = a / (1.f + __expf(-a)) * u[i];
    g[i] = v;
    wave_q8(v, xq, xd, i >> 5);
}

// beta = sigmoid(Wb x), g = softplus(Wa x + dt) * A for n_heads heads, from
// int8 x. One wave per row; rows [0, nh) are beta, [nh, 2nh) alpha.
template <int T>
__device__ void gdn_ab_impl(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq,
                            const float *xd, float *beta, float *g, const float *dt, const float *A, int ncols,
                            int nh, int blk) {
    const int lane = threadIdx.x % WARP;
    const int row = blk * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= 2 * nh) return;
    const bool is_b = row < nh;
    const int h = is_b ? row : row - nh;
    const uint8_t *wr = (is_b ? Wb : Wa) + (size_t)h * row_bytes;
    const int nu = ncols / QD<T>::sb * QD<T>::per_sb;
    float acc = 0.f;
    for (int u = lane; u < nu; u += WARP) acc += QD<T>::dot(wr, u, xq, xd);
    acc = warp_sum(acc);
    if (lane == 0) {
        if (is_b) {
            beta[h] = 1.f / (1.f + __expf(-acc));
        } else {
            const float z = acc + dt[h];
            g[h] = (z > 20.f ? z : log1pf(__expf(z))) * A[h];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256)
    gdn_ab_q8_0(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq, const float *xd,
                float *beta, float *g, const float *dt, const float *A, int ncols, int nh) {
    const int t = blockIdx.y;  // token
    gdn_ab_impl<QT_Q8_0>(Wb, Wa, row_bytes, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32), beta + (size_t)t * nh,
                         g + (size_t)t * nh, dt, A, ncols, nh, blockIdx.x);
}

// Recurrent-state segments for the fused DeltaNet decode kernels: rows
// [row0, row0 + nrows) of the batch belong to one sequence whose state lives
// at `state`; snap[t] (if non-null) receives the state right after local row t
// (speculative verify rollback). Passed by value (kernel args, no host copy).
#define GDN_MAX_SEG 16
#define GDN_MAX_SNAP 15
struct GdnSeg {
    float *state;
    float *snap[GDN_MAX_SNAP];
    int row0, nrows;
    // replay mode (pend != null): the inputs of this segment's rows are kept in
    // pend (conv: raw input rows [row][ch]; delta rule: [row][head][k 128 | v 128
    // | decay | beta]); the first npend rows kept by the previous verify are
    // applied to the state first and the state is written once (the new base),
    // then the new rows run without writing state or snapshots
    float *pend;
    int npend, pad_;
};
struct GdnSegs {
    GdnSeg s[GDN_MAX_SEG];
};

// Causal conv step (d_conv = 4) + SiLU for 128 channels per block, then L2
// norm when the block is a q/k head (block < n_qk_heads). blockDim = 128.
// grid.y = segment.
// cb = 128-channel block, seg = segment; act = false: an idle thread of a
// wider block that only joins the block_sum reductions (adds 0).
__device__ __forceinline__ void gdn_conv_l2_impl(const float *xin, const float *w, float *out, int ch, int n_qk_heads,
                                                 float eps, const GdnSegs &segs, int cb, int seg, bool act) {
    const GdnSeg &sg = segs.s[seg];
    float *state = sg.state;
    const int r0 = sg.row0, n_tok = sg.nrows;
    const int c = cb * 128 + (act ? (int)threadIdx.x : 0);
    float h0 = 0.f, h1 = 0.f, h2 = 0.f;
    float4 wv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (act) {
        h0 = state[c];
        h1 = state[(size_t)ch + c];
        h2 = state[(size_t)2 * ch + c];
        wv = *(const float4 *)(w + (size_t)c * 4);
    }
    for (int t = 0; t < n_tok; ++t) {
        float v = 0.f;
        if (act) {
            const float xn = xin[(size_t)(r0 + t) * ch + c];
            // explicit fmas = the contraction the original kernel compiled to, so the
            // result does not depend on how the compiler contracts it here
            const float acc = __builtin_fmaf(xn, wv.w, __builtin_fmaf(h2, wv.z, __builtin_fmaf(h1, wv.y, h0 * wv.x)));
            h0 = h1;
            h1 = h2;
            h2 = xn;
            float *sp = t < GDN_MAX_SNAP ? sg.snap[t] : nullptr;
            if (sp != nullptr) {
                sp[c] = h0;
                sp[(size_t)ch + c] = h1;
                sp[(size_t)2 * ch + c] = h2;
            }
            v = acc / (1.f + __expf(-acc));
        }
        if (cb < n_qk_heads) {
            const float ss = block_sum(v * v);
            v *= rsqrtf(ss + eps);
        }
        if (act) out[(size_t)(r0 + t) * ch + c] = v;
    }
    if (act) {
        state[c] = h0;
        state[(size_t)ch + c] = h1;
        state[(size_t)2 * ch + c] = h2;
    }
}

extern "C" __global__ void __launch_bounds__(128)
    gdn_conv_l2(const float *xin, const float *w, float *out, int ch, int n_qk_heads, float eps, GdnSegs segs) {
    gdn_conv_l2_impl(xin, w, out, ch, n_qk_heads, eps, segs, blockIdx.x, blockIdx.y, true);
}

// Delta-rule step per token + gated RMSNorm + int8 quantization.
// Block = one v head of one segment (grid.y), 512 threads = 128 columns j x
// 4 key-dim quarters. Thread (j, qi) keeps S[i][j] for i in [32 qi, 32 qi + 32)
// in registers.
extern "C" __global__ void __launch_bounds__(512)
    gdn_step_norm(const float *qkv, const float *g, const float *beta, const float *z, const float *wn, float *out,
                  int8_t *xq, float *xd, int n_k_heads, float scale, float eps, int n_v_heads, int qkv_stride,
                  GdnSegs segs) {
    constexpr int DK = 128, DV = 128, QI = 4, PER = DK / QI;
    const GdnSeg &sgm = segs.s[blockIdx.y];
    const int r0 = sgm.row0, n_tok = sgm.nrows;
    const int h = blockIdx.x;
    const int j = threadIdx.x % DV, qi = threadIdx.x / DV;
    const int kh = h % n_k_heads;
    const int d_inner = n_v_heads * DV;
    __shared__ float ks[DK], qs[DK];
    __shared__ float red[QI][DV];
    __shared__ float ov[DV];
    float *Sg = sgm.state + (size_t)h * DK * DV;
    float S[PER];
#pragma unroll
    for (int e = 0; e < PER; ++e) S[e] = Sg[(size_t)(qi * PER + e) * DV + j];
    for (int tl = 0; tl < n_tok; ++tl) {
        const int t = r0 + tl;
        const float *row = qkv + (size_t)t * qkv_stride;
        __syncthreads();
        if (threadIdx.x < DK) {
            qs[threadIdx.x] = row[(size_t)kh * DK + threadIdx.x];
            ks[threadIdx.x] = row[(size_t)n_k_heads * DK + (size_t)kh * DK + threadIdx.x];
        }
        __syncthreads();
        const float decay = __expf(g[(size_t)t * n_v_heads + h]);
        float sk = 0.f;
#pragma unroll
        for (int e = 0; e < PER; ++e) {
            S[e] *= decay;
            sk += S[e] * ks[qi * PER + e];
        }
        red[qi][j] = sk;
        __syncthreads();
        sk = red[0][j] + red[1][j] + red[2][j] + red[3][j];
        const float v = row[(size_t)2 * n_k_heads * DK + (size_t)h * DV + j];
        const float delta = (v - sk) * beta[(size_t)t * n_v_heads + h];
        float o = 0.f;
#pragma unroll
        for (int e = 0; e < PER; ++e) {
            S[e] += ks[qi * PER + e] * delta;
            o += S[e] * qs[qi * PER + e];
        }
        float *sp = tl < GDN_MAX_SNAP ? sgm.snap[tl] : nullptr;
        if (sp != nullptr) {
            float *Ss = sp + (size_t)h * DK * DV;
#pragma unroll
            for (int e = 0; e < PER; ++e) Ss[(size_t)(qi * PER + e) * DV + j] = S[e];
        }
        __syncthreads();
        red[qi][j] = o;
        __syncthreads();
        if (qi == 0) ov[j] = (red[0][j] + red[1][j] + red[2][j] + red[3][j]) * scale;
        __syncthreads();
        float y = 0.f;
        if (threadIdx.x < DV) y = ov[threadIdx.x];
        const float ss = block_sum(threadIdx.x < DV ? y * y : 0.f);
        if (threadIdx.x < DV) {
            const float r = rsqrtf(ss / DV + eps);
            const float zz = z[(size_t)t * d_inner + (size_t)h * DV + threadIdx.x];
            y = y * r * wn[threadIdx.x] * (zz / (1.f + __expf(-zz)));
            out[(size_t)t * d_inner + (size_t)h * DV + threadIdx.x] = y;
            wave_q8(y, xq + (size_t)t * d_inner, xd + (size_t)t * (d_inner / 32), (h * DV + threadIdx.x) >> 5);
        }
    }
#pragma unroll
    for (int e = 0; e < PER; ++e) Sg[(size_t)(qi * PER + e) * DV + j] = S[e];
}

// Dequantize rows `ids[t]` of W into out[t][ncols] (embedding lookup).
template <int T>
__device__ void get_rows_impl(const uint8_t *W, uint64_t row_bytes, const int *ids, float *out, int ncols) {
    const int t = blockIdx.y;
    const uint8_t *wr = W + (size_t)ids[t] * row_bytes;
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < ncols / 8; g += gridDim.x * blockDim.x) {
        float w[8];
        dec_group<T>(wr, g, w);
#pragma unroll
        for (int i = 0; i < 8; ++i) out[(size_t)t * ncols + 8 * g + i] = w[i];
    }
}

#define GET_ROWS(SUF, T)                                                                               \
    extern "C" __global__ void get_rows_##SUF(const uint8_t *W, uint64_t row_bytes, const int *ids, \
                                              float *out, int ncols) {                               \
        get_rows_impl<T>(W, row_bytes, ids, out, ncols);                                             \
    }

GET_ROWS(f32, QT_F32)
GET_ROWS(f16, QT_F16)
GET_ROWS(q8_0, QT_Q8_0)
GET_ROWS(q3_k, QT_Q3_K)
GET_ROWS(q4_k, QT_Q4_K)
GET_ROWS(q5_k, QT_Q5_K)
GET_ROWS(q6_k, QT_Q6_K)
GET_ROWS(iq4_nl, QT_IQ4_NL)
GET_ROWS(iq3_s, QT_IQ3_S)
GET_ROWS(iq4_xs, QT_IQ4_XS)
GET_ROWS(mxfp4, QT_MXFP4)
