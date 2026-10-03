// WHIRL device code (gfx1201 / RDNA 4): one-token decode fusions: norm + int8 quantization, fused Gated DeltaNet conv / delta-rule steps.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

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
    xq[b * 32 + lane] = (int8_t)__float2int_rn(v * id);
    if (lane == 0) xd[b] = d;
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
    float *pend = sg.pend;
    if (act) {
        h0 = state[c];
        h1 = state[(size_t)ch + c];
        h2 = state[(size_t)2 * ch + c];
        wv = *(const float4 *)(w + (size_t)c * 4);
        if (pend != nullptr && sg.npend > 0) {
            for (int t = 0; t < sg.npend; ++t) {
                h0 = h1;
                h1 = h2;
                h2 = pend[(size_t)t * ch + c];
            }
            state[c] = h0;
            state[(size_t)ch + c] = h1;
            state[(size_t)2 * ch + c] = h2;
        }
    }
    for (int t = 0; t < n_tok; ++t) {
        float v = 0.f;
        if (act) {
            const float xn = xin[(size_t)(r0 + t) * ch + c];
            if (pend != nullptr) pend[(size_t)t * ch + c] = xn;
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
    if (act && pend == nullptr) {
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
    gdn_step_norm_v0(const float *qkv, const float *g, const float *beta, const float *z, const float *wn, float *out,
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
    float *pend = sgm.pend;
    constexpr int PW = DK + DV + 2; // pend row width per head
    if (pend != nullptr && sgm.npend > 0) {
        // the rows the previous verify kept: the same state update as below
        for (int tl = 0; tl < sgm.npend; ++tl) {
            const float *pr = pend + ((size_t)tl * n_v_heads + h) * PW;
            __syncthreads();
            if (threadIdx.x < DK) ks[threadIdx.x] = pr[threadIdx.x];
            __syncthreads();
            const float decay = pr[DK + DV];
            float sk = 0.f;
#pragma unroll
            for (int e = 0; e < PER; ++e) {
                S[e] *= decay;
                sk += S[e] * ks[qi * PER + e];
            }
            red[qi][j] = sk;
            __syncthreads();
            sk = red[0][j] + red[1][j] + red[2][j] + red[3][j];
            const float v = pr[DK + j];
            const float delta = (v - sk) * pr[DK + DV + 1];
#pragma unroll
            for (int e = 0; e < PER; ++e) S[e] += ks[qi * PER + e] * delta;
        }
#pragma unroll
        for (int e = 0; e < PER; ++e) Sg[(size_t)(qi * PER + e) * DV + j] = S[e];
    }
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
        const float bt = beta[(size_t)t * n_v_heads + h];
        const float delta = (v - sk) * bt;
        if (pend != nullptr) {
            // this row's inputs, for the replay in the next verify
            float *pr = pend + ((size_t)tl * n_v_heads + h) * PW;
            if (qi == 0) {
                pr[j] = ks[j];
                pr[DK + j] = v;
            }
            if (threadIdx.x == 0) {
                pr[DK + DV] = decay;
                pr[DK + DV + 1] = bt;
            }
        }
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
    if (pend == nullptr) {
#pragma unroll
        for (int e = 0; e < PER; ++e) Sg[(size_t)(qi * PER + e) * DV + j] = S[e];
    }
}

// Delta-rule step per token + gated RMSNorm + int8 quantization.
// Block = one v head of one segment (grid.y), 512 threads = 128 columns j x
// 4 key-dim quarters. Thread (j, qi) keeps S[i][j] for i in [32 qi, 32 qi + 32)
// in registers.
// Item E: the rows' q / k / v / decay / beta are prefetched into LDS, the
// sequential per-row chain has one barrier per row (double-buffered
// reductions; the o reduction of row t runs after the barrier of row t + 1),
// and the gated RMSNorm + int8 quantization of all rows run afterwards, one
// wave per row, with the block_sum / wave_q8 arithmetic of the per-row
// version (same values, bit for bit). gdn_step_norm_v0 = the per-row version.
#define GDN_MR 16
extern "C" __global__ void __launch_bounds__(512)
    gdn_step_norm(const float *qkv, const float *g, const float *beta, const float *z, const float *wn, float *out,
                  int8_t *xq, float *xd, int n_k_heads, float scale, float eps, int n_v_heads, int qkv_stride,
                  GdnSegs segs) {
    constexpr int DK = 128, DV = 128, QI = 4, PER = DK / QI, MR = GDN_MR;
    const GdnSeg &sgm = segs.s[blockIdx.y];
    const int r0 = sgm.row0, n_tok = sgm.nrows;
    const int h = blockIdx.x;
    const int j = threadIdx.x % DV, qi = threadIdx.x / DV;
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int kh = h % n_k_heads;
    const int d_inner = n_v_heads * DV;
    __shared__ float ks[MR][DK], qs[MR][DK], vs[MR][DV], ov[MR][DV];
    __shared__ float dec[MR], bet[MR];
    __shared__ float red[2][QI][DV], redo[2][QI][DV];
    float *Sg = sgm.state + (size_t)h * DK * DV;
    float S[PER];
#pragma unroll
    for (int e = 0; e < PER; ++e) S[e] = Sg[(size_t)(qi * PER + e) * DV + j];
    float *pend = sgm.pend;
    constexpr int PW = DK + DV + 2;  // pend row width per head
    const int np = pend != nullptr ? sgm.npend : 0;
    if (np > 0) {
        // the rows the previous verify kept: the same state update as below
        for (int i = threadIdx.x; i < np * DK; i += blockDim.x) {
            const int t = i / DK, d = i % DK;
            const float *pr = pend + ((size_t)t * n_v_heads + h) * PW;
            ks[t][d] = pr[d];
            vs[t][d] = pr[DK + d];
        }
        if (threadIdx.x < np) {
            const float *pr = pend + ((size_t)threadIdx.x * n_v_heads + h) * PW;
            dec[threadIdx.x] = pr[DK + DV];
            bet[threadIdx.x] = pr[DK + DV + 1];
        }
        __syncthreads();
        for (int tl = 0; tl < np; ++tl) {
            const float decay = dec[tl];
            float sk = 0.f;
#pragma unroll
            for (int e = 0; e < PER; ++e) {
                S[e] *= decay;
                sk += S[e] * ks[tl][qi * PER + e];
            }
            red[tl & 1][qi][j] = sk;
            __syncthreads();
            sk = red[tl & 1][0][j] + red[tl & 1][1][j] + red[tl & 1][2][j] + red[tl & 1][3][j];
            const float delta = (vs[tl][j] - sk) * bet[tl];
#pragma unroll
            for (int e = 0; e < PER; ++e) S[e] += ks[tl][qi * PER + e] * delta;
        }
#pragma unroll
        for (int e = 0; e < PER; ++e) Sg[(size_t)(qi * PER + e) * DV + j] = S[e];
        __syncthreads();  // ks / vs / red are reused below
    }
    for (int c0 = 0; c0 < n_tok; c0 += MR) {
        const int nr = min(MR, n_tok - c0);
        for (int i = threadIdx.x; i < nr * DK; i += blockDim.x) {
            const int t = i / DK, d = i % DK;
            const float *row = qkv + (size_t)(r0 + c0 + t) * qkv_stride;
            qs[t][d] = row[(size_t)kh * DK + d];
            ks[t][d] = row[(size_t)n_k_heads * DK + (size_t)kh * DK + d];
            vs[t][d] = row[(size_t)2 * n_k_heads * DK + (size_t)h * DV + d];
        }
        if (threadIdx.x < nr) {
            const int t = r0 + c0 + threadIdx.x;
            dec[threadIdx.x] = __expf(g[(size_t)t * n_v_heads + h]);
            bet[threadIdx.x] = beta[(size_t)t * n_v_heads + h];
        }
        __syncthreads();
        if (pend != nullptr) {
            // the rows' inputs, for the replay in the next verify (pend was read above)
            for (int i = threadIdx.x; i < nr * DK; i += blockDim.x) {
                const int t = i / DK, d = i % DK;
                float *pr = pend + ((size_t)(c0 + t) * n_v_heads + h) * PW;
                pr[d] = ks[t][d];
                pr[DK + d] = vs[t][d];
            }
            if (threadIdx.x < nr) {
                float *pr = pend + ((size_t)(c0 + threadIdx.x) * n_v_heads + h) * PW;
                pr[DK + DV] = dec[threadIdx.x];
                pr[DK + DV + 1] = bet[threadIdx.x];
            }
        }
        for (int tl = 0; tl < nr; ++tl) {
            const float decay = dec[tl];
            float sk = 0.f;
#pragma unroll
            for (int e = 0; e < PER; ++e) {
                S[e] *= decay;
                sk += S[e] * ks[tl][qi * PER + e];
            }
            red[tl & 1][qi][j] = sk;
            __syncthreads();
            // o of the previous row (its partials were written before this barrier)
            if (tl > 0 && qi == 0)
                ov[tl - 1][j] = (redo[(tl - 1) & 1][0][j] + redo[(tl - 1) & 1][1][j] + redo[(tl - 1) & 1][2][j] +
                                 redo[(tl - 1) & 1][3][j]) *
                                scale;
            sk = red[tl & 1][0][j] + red[tl & 1][1][j] + red[tl & 1][2][j] + red[tl & 1][3][j];
            const float delta = (vs[tl][j] - sk) * bet[tl];
            float o = 0.f;
#pragma unroll
            for (int e = 0; e < PER; ++e) {
                S[e] += ks[tl][qi * PER + e] * delta;
                o += S[e] * qs[tl][qi * PER + e];
            }
            const int ta = c0 + tl;
            float *sp = ta < GDN_MAX_SNAP ? sgm.snap[ta] : nullptr;
            if (sp != nullptr) {
                float *Ss = sp + (size_t)h * DK * DV;
#pragma unroll
                for (int e = 0; e < PER; ++e) Ss[(size_t)(qi * PER + e) * DV + j] = S[e];
            }
            redo[tl & 1][qi][j] = o;
        }
        __syncthreads();
        if (qi == 0)
            ov[nr - 1][j] = (redo[(nr - 1) & 1][0][j] + redo[(nr - 1) & 1][1][j] + redo[(nr - 1) & 1][2][j] +
                             redo[(nr - 1) & 1][3][j]) *
                            scale;
        __syncthreads();
        // gated RMSNorm + int8 copy, one wave per row: lane l holds columns
        // l, 32 + l, 64 + l, 96 + l (= waves 0..3 of block_sum; the other
        // waves' zeros leave its sum unchanged)
        for (int tl = wid; tl < nr; tl += blockDim.x / WARP) {
            const int t = r0 + c0 + tl;
            float y[4], p[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                y[c] = ov[tl][32 * c + lane];
                // a separate multiply (as block_sum's select input): no fma into the first shuffle add
                float sq = y[c] * y[c];
                asm volatile("" : "+v"(sq));
                p[c] = warp_sum(sq);
            }
            const float ss = (p[0] + p[2]) + (p[1] + p[3]);
            const float r = rsqrtf(ss / DV + eps);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int jj = 32 * c + lane;
                const float zz = z[(size_t)t * d_inner + (size_t)h * DV + jj];
                const float yy = y[c] * r * wn[jj] * (zz / (1.f + __expf(-zz)));
                out[(size_t)t * d_inner + (size_t)h * DV + jj] = yy;
                wave_q8(yy, xq + (size_t)t * d_inner, xd + (size_t)t * (d_inner / 32), (h * DV + jj) >> 5);
            }
        }
        __syncthreads();  // LDS reuse by the next chunk
    }
    if (pend == nullptr) {
#pragma unroll
        for (int e = 0; e < PER; ++e) Sg[(size_t)(qi * PER + e) * DV + j] = S[e];
    }
}

