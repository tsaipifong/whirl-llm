// WHIRL device code (gfx1151 / RDNA 3.5): Gated DeltaNet (per token, chunked prefill, sequential scan), kv_store, argmax.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Gated DeltaNet (one token per launch).
// ---------------------------------------------------------------------------

// Causal depthwise conv over [state(d_conv-1) | new tokens], SiLU; the last
// d_conv-1 inputs become the new state. w: [ch][d_conv]; state: [d_conv-1][ch]
// (oldest first); xin/out: [n_tok][ch]. Requires d_conv == 4.
extern "C" __global__ void gdn_conv_seq(const float *xin, float *state, const float *w, float *out,
                                        int ch, int n_tok) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const float w0 = w[(size_t)c * 4 + 0], w1 = w[(size_t)c * 4 + 1];
    const float w2 = w[(size_t)c * 4 + 2], w3 = w[(size_t)c * 4 + 3];
    float h0 = state[c], h1 = state[(size_t)ch + c], h2 = state[(size_t)2 * ch + c];
    for (int t = 0; t < n_tok; ++t) {
        const float xn = xin[(size_t)t * ch + c];
        const float acc = h0 * w0 + h1 * w1 + h2 * w2 + xn * w3;
        out[(size_t)t * ch + c] = acc / (1.f + __expf(-acc));
        h0 = h1;
        h1 = h2;
        h2 = xn;
    }
    state[c] = h0;
    state[(size_t)ch + c] = h1;
    state[(size_t)2 * ch + c] = h2;
}

// beta = sigmoid(b); g = softplus(a + dt_bias) * A   (A = -exp(A_log), stored)
extern "C" __global__ void gdn_gates(float *b, float *a, const float *dt_bias, const float *A, int n,
                                     int n_heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % n_heads;
    b[i] = 1.f / (1.f + __expf(-b[i]));
    const float x = a[i] + dt_bias[h];
    const float sp = x > 20.f ? x : log1pf(__expf(x));
    a[i] = sp * A[h];
}

// gdn_gates on the concatenated [beta | alpha] projection output: ba[t][0, nh) = b,
// ba[t][nh, 2nh) = a (one 2nh-row GEMM). Same math as gdn_gates (bitwise), written
// to b / a laid out [t][nh]. n = n_tok * n_heads. Same kernel as the gfx1201 set.
extern "C" __global__ void gdn_gates_ba(const float *ba, float *b, float *a, const float *dt_bias, const float *A,
                                        int n, int n_heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % n_heads;
    const int t = i / n_heads;
    const float bi = ba[(size_t)t * 2 * n_heads + h];
    const float ai = ba[(size_t)t * 2 * n_heads + n_heads + h];
    b[i] = 1.f / (1.f + __expf(-bi));
    const float x = ai + dt_bias[h];
    const float sp = x > 20.f ? x : log1pf(__expf(x));
    a[i] = sp * A[h];
}

// One v-head per block, blockDim = dv. Thread j keeps column j of the state
// (S[i][j], i = key dim) in registers for the whole token run.
// State layout: state[h*DK*dv + i*dv + j]. qkv: [n_tok][qkv_stride] with q at
// [0, nk*DK), k at [nk*DK, 2*nk*DK), v after. g/beta: [n_tok][n_v_heads].
// ---------------------------------------------------------------------------
// Chunkwise Gated DeltaNet for prefill (C = 64 tokens per chunk), f32.
// Recurrence per v head (S: dk x dv, a_t = exp(g_t)):
//   u_t = beta_t (v_t - a_t S_{t-1}^T k_t);  S_t = a_t S_{t-1} + k_t u_t^T;  o_t = S_t^T q_t
// With G_t = cumsum(g) inside the chunk, A_t = exp(G_t), S0 = state at chunk start:
//   (I + B) U = diag(beta) (V - diag(A) K S0),  B[t][s] = beta_t e^{G_t-G_s} k_t.k_s (s < t)
//   => U = Ut - W S0 with T = (I+B)^-1, Ut = T diag(beta) V, W = T diag(beta A) K
//   O = diag(A) Q S0 + M U,  M[t][s] = e^{G_t-G_s} q_t.k_s (s <= t)
//   S_C = A_C S0 + Kt^T U,   Kt[s] = e^{G_C-G_s} k_s
// gdn_chunk_prep builds W, Ut, M, G for every (chunk, head) in parallel;
// gdn_chunk_scan walks the chunks sequentially per (head, 32 value columns).
// Padded rows (t >= n_tok) carry k = v = 0, beta = 0, g = 0 and drop out.
// ---------------------------------------------------------------------------
#define GCH 64
#define GDK 128

extern "C" __global__ void __launch_bounds__(256)
    gdn_chunk_prep(const float *qkv, const float *gl, const float *beta, float *Wout, float *Uout, float *Mout,
                   float *Gout, int n_tok, int n_k_heads, int n_v_heads, int qkv_stride) {
    const int c = blockIdx.x, h = blockIdx.y, kh = h % n_k_heads;
    const int t0 = c * GCH;
    const int tid = threadIdx.x;
    __shared__ float Ks[GCH][GDK + 4];   // K, later V
    __shared__ float Ts[GCH][GCH + 1];   // B, then T
    __shared__ float G[GCH], Bt[GCH], Av[GCH];
    for (int i = tid; i < GCH * GDK / 4; i += blockDim.x) {
        const int t = i / (GDK / 4), d4 = (i % (GDK / 4)) * 4, tok = t0 + t;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tok < n_tok) v = *(const float4 *)(qkv + (size_t)tok * qkv_stride + (size_t)n_k_heads * GDK + (size_t)kh * GDK + d4);
        *(float4 *)&Ks[t][d4] = v;
    }
    if (tid < GCH) {
        const int tok = t0 + tid;
        G[tid] = tok < n_tok ? gl[(size_t)tok * n_v_heads + h] : 0.f;
        Bt[tid] = tok < n_tok ? beta[(size_t)tok * n_v_heads + h] : 0.f;
    }
    __syncthreads();
    if (tid < WARP) {  // inclusive scan of g over 64 values (2 per lane)
        float a0 = G[2 * tid], a1 = G[2 * tid + 1];
        a1 += a0;
        float x = a1;
#pragma unroll
        for (int o = 1; o < WARP; o <<= 1) {
            const float y = __shfl_up(x, o, WARP);
            if (tid >= o) x += y;
        }
        const float base = x - a1;
        G[2 * tid] = base + a0;
        G[2 * tid + 1] = x;
    }
    __syncthreads();
    if (tid < GCH) Av[tid] = __expf(G[tid]);
    // 4x4 register tiles: thread -> rows t4*4.., cols s4*4..
    const int t4 = (tid >> 4) * 4, s4 = (tid & 15) * 4;
    {
        float acc[4][4] = {};
        for (int d = 0; d < GDK; ++d) {
            float kt[4], ks[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                kt[e] = Ks[t4 + e][d];
                ks[e] = Ks[s4 + e][d];
            }
#pragma unroll
            for (int e = 0; e < 4; ++e)
#pragma unroll
                for (int f = 0; f < 4; ++f) acc[e][f] += kt[e] * ks[f];
        }
        __syncthreads();
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int t = t4 + e, sI = s4 + f;
                Ts[t][sI] = (sI < t) ? Bt[t] * __expf(G[t] - G[sI]) * acc[e][f] : 0.f;
            }
    }
    // M = e^{G_t-G_s} q_t.k_s (s <= t); q rows straight from global (4 rows/thread)
    {
        float acc[4][4] = {};
        const float *qr[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) qr[e] = qkv + (size_t)min(t0 + t4 + e, n_tok - 1) * qkv_stride + (size_t)kh * GDK;
        for (int d = 0; d < GDK; d += 4) {
            float4 q4[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) q4[e] = *(const float4 *)(qr[e] + d);
#pragma unroll
            for (int dd = 0; dd < 4; ++dd) {
                float ks[4];
#pragma unroll
                for (int f = 0; f < 4; ++f) ks[f] = Ks[s4 + f][d + dd];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float qv = dd == 0 ? q4[e].x : dd == 1 ? q4[e].y : dd == 2 ? q4[e].z : q4[e].w;
#pragma unroll
                    for (int f = 0; f < 4; ++f) acc[e][f] += qv * ks[f];
                }
            }
        }
        const size_t ch = (size_t)c * n_v_heads + h;
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int t = t4 + e, sI = s4 + f;
                Mout[ch * GCH * GCH + (size_t)t * GCH + sI] =
                    (sI <= t && t0 + t < n_tok) ? __expf(G[t] - G[sI]) * acc[e][f] : 0.f;
            }
    }
    __syncthreads();
    // T = (I + B)^-1 by column sweep: T starts as -B below the diagonal (B's
    // row r is still needed while T's row r is final): for r = 0..62, rows
    // t > r get T[t][s] -= B[t][r] * T[r][s] for s < r, and T[t][r] = -B[t][r]
    // is already in place. Implemented on the combined array X = -B:
    //   X[t][s] (s < r) += X[t][r] * X[r][s]   ... with X[t][r] = -B[t][r] = T-coefficient.
    for (int i = tid; i < GCH * GCH; i += blockDim.x) Ts[i / GCH][i % GCH] = -Ts[i / GCH][i % GCH];
    __syncthreads();
    for (int r = 1; r < GCH - 1; ++r) {
        // rows t > r: X[t][s] += X[t][r] * X[r][s] for s < r
        const int rows = GCH - 1 - r;
        for (int i = tid; i < rows * r; i += blockDim.x) {
            const int t = r + 1 + i / r, sI = i % r;
            Ts[t][sI] += Ts[t][r] * Ts[r][sI];
        }
        __syncthreads();
    }
    // Ts now holds T strictly below the diagonal (unit diagonal implicit).
    const size_t ch = (size_t)c * n_v_heads + h;
    // W = T diag(beta A) K : thread -> rows t4.., 8 dims d8..
    {
        const int d8 = (tid & 15) * 8;
        float acc[4][8];
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int f = 0; f < 8; ++f) acc[e][f] = (t4 + e < GCH) ? Bt[t4 + e] * Av[t4 + e] * Ks[t4 + e][d8 + f] : 0.f;
        for (int sI = 0; sI < t4 + 3; ++sI) {
            const float cs = Bt[sI] * Av[sI];
            float kv[8];
#pragma unroll
            for (int f = 0; f < 8; ++f) kv[f] = cs * Ks[sI][d8 + f];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float tv = (sI < t4 + e) ? Ts[t4 + e][sI] : 0.f;
#pragma unroll
                for (int f = 0; f < 8; ++f) acc[e][f] += tv * kv[f];
            }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int f = 0; f < 8; f += 4)
                *(float4 *)(Wout + ch * GCH * GDK + (size_t)(t4 + e) * GDK + d8 + f) =
                    make_float4(acc[e][f], acc[e][f + 1], acc[e][f + 2], acc[e][f + 3]);
    }
    __syncthreads();
    for (int i = tid; i < GCH * GDK / 4; i += blockDim.x) {
        const int t = i / (GDK / 4), d4 = (i % (GDK / 4)) * 4, tok = t0 + t;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tok < n_tok) v = *(const float4 *)(qkv + (size_t)tok * qkv_stride + (size_t)2 * n_k_heads * GDK + (size_t)h * GDK + d4);
        *(float4 *)&Ks[t][d4] = v;
    }
    __syncthreads();
    // Ut = T diag(beta) V
    {
        const int d8 = (tid & 15) * 8;
        float acc[4][8];
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int f = 0; f < 8; ++f) acc[e][f] = Bt[t4 + e] * Ks[t4 + e][d8 + f];
        for (int sI = 0; sI < t4 + 3; ++sI) {
            const float cs = Bt[sI];
            float vv[8];
#pragma unroll
            for (int f = 0; f < 8; ++f) vv[f] = cs * Ks[sI][d8 + f];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float tv = (sI < t4 + e) ? Ts[t4 + e][sI] : 0.f;
#pragma unroll
                for (int f = 0; f < 8; ++f) acc[e][f] += tv * vv[f];
            }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int f = 0; f < 8; f += 4)
                *(float4 *)(Uout + ch * GCH * GDK + (size_t)(t4 + e) * GDK + d8 + f) =
                    make_float4(acc[e][f], acc[e][f + 1], acc[e][f + 2], acc[e][f + 3]);
    }
    if (tid < GCH) Gout[ch * GCH + tid] = G[tid];
}

// Stage sources per chunk, in order: 0 = W (64x128), 1 = Q (64x128),
// 2 = M (64x64), 3 = K (64x128, scaled by e^{G_C - G_s} when stored).
// The next stage's global loads are issued before the current stage's math.
#define SCAN_FETCH(stage_id)                                                                        \
    do {                                                                                            \
        const int c_ = (stage_id) >> 2, p_ = (stage_id) & 3;                                        \
        const size_t ch_ = (size_t)c_ * n_v_heads + h;                                              \
        const int t0_ = c_ * GCH;                                                                   \
        _Pragma("unroll") for (int e = 0; e < 8; ++e) {                                             \
            const int i = tid + e * 256;                                                            \
            const int t = i / (GDK / 4), d4 = (i % (GDK / 4)) * 4;                                  \
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);                                             \
            if (p_ == 0) {                                                                          \
                v = *(const float4 *)(Win + ch_ * GCH * GDK + (size_t)t * GDK + d4);                \
            } else if (p_ == 1) {                                                                   \
                const int tok = min(t0_ + t, n_tok - 1);                                            \
                v = *(const float4 *)(qkv + (size_t)tok * qkv_stride + (size_t)kh * GDK + d4);      \
            } else if (p_ == 2) {                                                                   \
                if (e < 4) {                                                                        \
                    const int tm = i / (GCH / 4), s4 = (i % (GCH / 4)) * 4;                         \
                    v = *(const float4 *)(Min + ch_ * GCH * GCH + (size_t)tm * GCH + s4);           \
                }                                                                                   \
            } else {                                                                                \
                const int tok = t0_ + t;                                                            \
                if (tok < n_tok)                                                                    \
                    v = *(const float4 *)(qkv + (size_t)tok * qkv_stride + (size_t)n_k_heads * GDK + (size_t)kh * GDK + d4); \
            }                                                                                       \
            pre[e] = v;                                                                             \
        }                                                                                           \
    } while (0)

#define SCAN_STORE(p)                                                                               \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < 8; ++e) {                                             \
            const int i = tid + e * 256;                                                            \
            if ((p) == 2) {                                                                         \
                if (e < 4) {                                                                        \
                    const int tm = i / (GCH / 4), s4 = (i % (GCH / 4)) * 4;                         \
                    *(float4 *)&X[tm][s4] = pre[e];                                                 \
                }                                                                                   \
            } else {                                                                                \
                const int t = i / (GDK / 4), d4 = (i % (GDK / 4)) * 4;                              \
                float4 v = pre[e];                                                                  \
                if ((p) == 3) {                                                                     \
                    const float f = __expf(Gs[GCH - 1] - Gs[t]);                                    \
                    v.x *= f; v.y *= f; v.z *= f; v.w *= f;                                         \
                }                                                                                   \
                *(float4 *)&X[t][d4] = v;                                                           \
            }                                                                                       \
        }                                                                                           \
    } while (0)

extern "C" __global__ void __launch_bounds__(256)
    gdn_chunk_scan(const float *qkv, const float *Win, const float *Uin, const float *Min, const float *Gin,
                   float *state, float *out, int n_tok, int n_k_heads, int n_v_heads, int qkv_stride, float scale) {
    const int h = blockIdx.x, j0 = blockIdx.y * 32, kh = h % n_k_heads;
    const int tid = threadIdx.x;
    __shared__ float S[GDK][33];
    __shared__ float U[GCH][33];
    __shared__ float X[GCH][GDK + 4];
    __shared__ float Gs[GCH];
    float *Sg = state + (size_t)h * GDK * GDK;
    for (int i = tid; i < GDK * 32; i += blockDim.x) S[i / 32][i % 32] = Sg[(size_t)(i / 32) * GDK + j0 + i % 32];
    const int n_chunks = (n_tok + GCH - 1) / GCH;
    const int n_stages = 4 * n_chunks;
    const int pt = tid >> 2, pj = (tid & 3) * 8;
    const int si = tid >> 1, sj = (tid & 1) * 16;
    float4 pre[8];
    SCAN_FETCH(0);
    float o[8];
    for (int st = 0; st < n_stages; ++st) {
        const int c = st >> 2, p = st & 3;
        const int t0 = c * GCH;
        const size_t ch = (size_t)c * n_v_heads + h;
        __syncthreads();
        if (p == 0 && tid < GCH) Gs[tid] = Gin[ch * GCH + tid];
        if (p == 3) __syncthreads();  // Gs visible for the K scaling
        SCAN_STORE(p);
        __syncthreads();
        if (st + 1 < n_stages) SCAN_FETCH(st + 1);
        if (p == 0) {  // U = Ut - W S
            const float *Ut = Uin + ch * GCH * GDK + (size_t)pt * GDK + j0 + pj;
            float acc[8];
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) acc[jj] = Ut[jj];
            for (int i = 0; i < GDK; ++i) {
                const float w = X[pt][i];
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) acc[jj] -= w * S[i][pj + jj];
            }
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) U[pt][pj + jj] = acc[jj];
        } else if (p == 1) {  // o = A (Q S)
            const float a = __expf(Gs[pt]);
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) o[jj] = 0.f;
            for (int i = 0; i < GDK; ++i) {
                const float q = X[pt][i];
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) o[jj] += q * S[i][pj + jj];
            }
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) o[jj] *= a;
        } else if (p == 2) {  // o += M U, write
            for (int sI = 0; sI <= pt; ++sI) {
                const float m = X[pt][sI];
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) o[jj] += m * U[sI][pj + jj];
            }
            const int tok = t0 + pt;
            if (tok < n_tok) {
                float *orow = out + ((size_t)tok * n_v_heads + h) * GDK + j0 + pj;
#pragma unroll
                for (int jj = 0; jj < 8; jj += 4)
                    *(float4 *)(orow + jj) = make_float4(o[jj] * scale, o[jj + 1] * scale, o[jj + 2] * scale, o[jj + 3] * scale);
            }
        } else {  // S = A_C S + Kt^T U
            const float ac = __expf(Gs[GCH - 1]);
            float acc[16];
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) acc[jj] = ac * S[si][sj + jj];
            for (int sI = 0; sI < GCH; ++sI) {
                const float kt = X[sI][si];
#pragma unroll
                for (int jj = 0; jj < 16; ++jj) acc[jj] += kt * U[sI][sj + jj];
            }
            __syncthreads();
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) S[si][sj + jj] = acc[jj];
        }
    }
    __syncthreads();
    for (int i = tid; i < GDK * 32; i += blockDim.x) Sg[(size_t)(i / 32) * GDK + j0 + i % 32] = S[i / 32][i % 32];
}
#undef SCAN_FETCH
#undef SCAN_STORE

// Causal conv over a token block, parallel over (channel, token): the first
// d_conv-1 taps before the block come from the rolling state.
extern "C" __global__ void gdn_conv_par(const float *xin, const float *state, const float *w, float *out, int ch,
                                        int n_tok) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= ch) return;
    float acc = 0.f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        const int tt = t - 3 + kk;  // input time index
        const float x = tt >= 0 ? xin[(size_t)tt * ch + c] : state[(size_t)(3 + tt) * ch + c];
        acc += x * w[(size_t)c * 4 + kk];
    }
    out[(size_t)t * ch + c] = acc / (1.f + __expf(-acc));
}

// New rolling state = last 3 inputs (mixing in old state when n_tok < 3).
// Runs after gdn_conv_par (which reads the old state).
extern "C" __global__ void gdn_conv_state(const float *xin, float *state, int ch, int n_tok, float *snap,
                                          int snap_after) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    if (snap != nullptr) {  // rolling state as it was right after token snap_after
#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            const int tt = snap_after - 2 + kk;
            snap[(size_t)kk * ch + c] = tt >= 0 ? xin[(size_t)tt * ch + c] : state[(size_t)(3 + tt) * ch + c];
        }
    }
    float v[3];
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        const int tt = n_tok - 3 + kk;
        v[kk] = tt >= 0 ? xin[(size_t)tt * ch + c] : state[(size_t)(3 + tt) * ch + c];
    }
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) state[(size_t)kk * ch + c] = v[kk];
}

// Sum over the 4 lanes of a quad (lanes 4k..4k+3) with DPP quad permutes
// (VALU, no LDS round trip like ds_bpermute).
__device__ __forceinline__ float quad_sum(float x) {
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0xB1, 0xF, 0xF, false));  // xor 1
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x4E, 0xF, 0xF, false));  // xor 2
    return x;
}

// Sum over groups of 8 lanes: quad permutes + row_xmask:4 (xor 4).
__device__ __forceinline__ float oct_sum(float x) {
    x = quad_sum(x);
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x164, 0xF, 0xF, false));
    return x;
}

template <int DK, int MODE = 0, int LPC = 4>
__device__ void gdn_seq_impl(const float *qkv, const float *g, const float *beta, float *state, float *out,
                             int n_tok, int n_k_heads, int n_v_heads, int dv, int qkv_stride, float scale,
                             float *snap = nullptr, int snap_after = -1) {
    // Each value column j is owned by the 4 lanes of a quad (qi = lane & 3),
    // each holding 32 of the DK state entries S[i][j] in registers; the two
    // dot products per token use 4 independent accumulators and reduce with
    // DPP quad permutes. Inputs are staged GC tokens at a time into LDS
    // (double-buffered; the next chunk's loads overlap this chunk's steps), so
    // the steps inside a chunk touch only LDS and need no barrier.
    // blockDim.x = 128 (32 columns x 4); blockIdx.y selects the column group.
    constexpr int QP = DK / LPC;
    constexpr int GC = 8;
    constexpr int NCOL = 128 / LPC;
    constexpr int QK4 = 2 * DK / 4;                     // float4 per token (q then k)
    constexpr int QK_PER = GC * QK4 / 128;              // float4 per thread per chunk
    constexpr int V_PER = GC * NCOL / 128;
    const int h = blockIdx.x;
    const int qi = threadIdx.x & (LPC - 1), jl = threadIdx.x / LPC;
    const bool dbg = blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0;
    unsigned long long c_start = 0, w_start = 0;
    if (dbg) {
        c_start = clock64();
        w_start = wall_clock64();
    }
    const int j0 = blockIdx.y * NCOL;
    const int j = j0 + jl;
    const int kh = h % n_k_heads;
    __shared__ float4 qk4[2][GC][QK4];
    __shared__ float vs[2][GC][NCOL];
    __shared__ float gs[2][GC], bs[2][GC];
    float S[QP];
    float *Sg = state + (size_t)h * DK * dv;
#pragma unroll
    for (int e = 0; e < QP; ++e) S[e] = Sg[(size_t)(qi * QP + e) * dv + j];

    float4 rqk[QK_PER];
    float rvv[V_PER];
    float rgb = 0.f;
    // loads use a clamped token index (always valid, no branches) so the
    // staging registers stay in VGPRs
#define GDN_FETCH(c0)                                                                               \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < QK_PER; ++e) {                                        \
            const int idx = threadIdx.x + e * 128;                                                  \
            const int tt = idx / QK4, c4 = idx % QK4;                                               \
            const int t = min((c0) + tt, n_tok - 1);                                                \
            const float *row = qkv + (size_t)t * qkv_stride;                                        \
            const float *src = (c4 < DK / 4) ? row + (size_t)kh * DK + 4 * c4                        \
                                             : row + (size_t)n_k_heads * DK + (size_t)kh * DK + 4 * (c4 - DK / 4); \
            rqk[e] = *(const float4 *)src;                                                          \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < V_PER; ++e) {                                         \
            const int idx = threadIdx.x + e * 128;                                                  \
            const int t = min((c0) + idx / NCOL, n_tok - 1);                                        \
            rvv[e] = qkv[(size_t)t * qkv_stride + (size_t)2 * n_k_heads * DK + (size_t)h * dv + j0 + idx % NCOL]; \
        }                                                                                           \
        {                                                                                           \
            const int t = min((c0) + (int)(threadIdx.x % GC), n_tok - 1);                           \
            rgb = (threadIdx.x < GC) ? g[(size_t)t * n_v_heads + h] : beta[(size_t)t * n_v_heads + h]; \
        }                                                                                           \
    } while (0)
#define GDN_STAGE(buf)                                                                              \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < QK_PER; ++e) {                                        \
            const int idx = threadIdx.x + e * 128;                                                  \
            qk4[buf][idx / QK4][idx % QK4] = rqk[e];                                                \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < V_PER; ++e) {                                         \
            const int idx = threadIdx.x + e * 128;                                                  \
            vs[buf][idx / NCOL][idx % NCOL] = rvv[e];                                               \
        }                                                                                           \
        if (threadIdx.x < GC) gs[buf][threadIdx.x] = rgb;                                           \
        else if (threadIdx.x < 2 * GC) bs[buf][threadIdx.x - GC] = rgb;                             \
    } while (0)

    GDN_FETCH(0);
    GDN_STAGE(0);
    __syncthreads();
    int cur = 0;
    for (int c0 = 0; c0 < n_tok; c0 += GC) {
        const bool more = c0 + GC < n_tok;
        if (more && MODE != 1) GDN_FETCH(c0 + GC);
        const int steps = min(GC, n_tok - c0);
        for (int tt = 0; tt < steps; ++tt) {
            const float4 *q4 = &qk4[cur][tt][qi * (QP / 4)];
            const float4 *k4 = &qk4[cur][tt][DK / 4 + qi * (QP / 4)];
            const float decay = __expf(gs[cur][tt]);
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
            for (int e4 = 0; e4 < QP / 4; ++e4) {
                const float4 kv = k4[e4];
                S[4 * e4 + 0] *= decay;
                S[4 * e4 + 1] *= decay;
                S[4 * e4 + 2] *= decay;
                S[4 * e4 + 3] *= decay;
                a0 += S[4 * e4 + 0] * kv.x;
                a1 += S[4 * e4 + 1] * kv.y;
                a2 += S[4 * e4 + 2] * kv.z;
                a3 += S[4 * e4 + 3] * kv.w;
            }
            const float sk = (LPC == 8 ? oct_sum((a0 + a1) + (a2 + a3)) : quad_sum((a0 + a1) + (a2 + a3)));
            const float delta = (vs[cur][tt][jl] - sk) * bs[cur][tt];
            if (MODE == 3) {
                if (qi == 0) out[((size_t)(c0 + tt) * n_v_heads + h) * dv + j] = delta;
                continue;
            }
            a0 = a1 = a2 = a3 = 0.f;
#pragma unroll
            for (int e4 = 0; e4 < QP / 4; ++e4) {
                const float4 kv = k4[e4], qv = q4[e4];
                S[4 * e4 + 0] += kv.x * delta;
                S[4 * e4 + 1] += kv.y * delta;
                S[4 * e4 + 2] += kv.z * delta;
                S[4 * e4 + 3] += kv.w * delta;
                a0 += S[4 * e4 + 0] * qv.x;
                a1 += S[4 * e4 + 1] * qv.y;
                a2 += S[4 * e4 + 2] * qv.z;
                a3 += S[4 * e4 + 3] * qv.w;
            }
            const float o = (LPC == 8 ? oct_sum((a0 + a1) + (a2 + a3)) : quad_sum((a0 + a1) + (a2 + a3)));
            if (MODE != 2 && qi == 0) out[((size_t)(c0 + tt) * n_v_heads + h) * dv + j] = o * scale;
            if (snap != nullptr && c0 + tt == snap_after) {
                float *Ss = snap + (size_t)h * DK * dv;
#pragma unroll
                for (int e = 0; e < QP; ++e) Ss[(size_t)(qi * QP + e) * dv + j] = S[e];
            }
            if (MODE == 2 && tt == steps - 1 && qi == 0) out[((size_t)(c0 + tt) * n_v_heads + h) * dv + j] = o * scale;
        }
        if (more) {
            __syncthreads();
            GDN_STAGE(cur ^ 1);
            __syncthreads();
            cur ^= 1;
        }
    }
#undef GDN_FETCH
#undef GDN_STAGE
#pragma unroll
    for (int e = 0; e < QP; ++e) Sg[(size_t)(qi * QP + e) * dv + j] = S[e];
    if (dbg) {
        whirl_dbg[0] = clock64() - c_start;
        whirl_dbg[1] = wall_clock64() - w_start;
    }
}

#define GDN_PROBE(NAME, MODE)                                                                       \
    extern "C" __global__ void __launch_bounds__(128)                                              \
        NAME(const float *qkv, const float *g, const float *beta, float *state, float *out, int n_tok, \
             int n_k_heads, int n_v_heads, int dv, int qkv_stride, float scale) {                  \
        gdn_seq_impl<128, MODE>(qkv, g, beta, state, out, n_tok, n_k_heads, n_v_heads, dv, qkv_stride, scale); \
    }
GDN_PROBE(gdn_probe_nofetch, 1)
GDN_PROBE(gdn_probe_nostore, 2)
GDN_PROBE(gdn_probe_half, 3)
extern "C" __global__ void __launch_bounds__(128)
    gdn_seq_128_l8(const float *qkv, const float *g, const float *beta, float *state, float *out, int n_tok,
                   int n_k_heads, int n_v_heads, int dv, int qkv_stride, float scale, float *snap, int snap_after) {
    gdn_seq_impl<128, 0, 8>(qkv, g, beta, state, out, n_tok, n_k_heads, n_v_heads, dv, qkv_stride, scale);
}

extern "C" __global__ void __launch_bounds__(128)
    gdn_seq_128(const float *qkv, const float *g, const float *beta, float *state, float *out, int n_tok,
                int n_k_heads, int n_v_heads, int dv, int qkv_stride, float scale, float *snap, int snap_after) {
    gdn_seq_impl<128>(qkv, g, beta, state, out, n_tok, n_k_heads, n_v_heads, dv, qkv_stride, scale, snap, snap_after);
}

// out[h][j] = rmsnorm(o[h])[j] * w[j] * silu(z[h][j]), in place on o.
extern "C" __global__ void gdn_gated_norm(float *o, const float *z, const float *w, int dv, float eps) {
    float *ov = o + (size_t)blockIdx.x * dv;
    const float *zv = z + (size_t)blockIdx.x * dv;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) ss += ov[i] * ov[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / dv + eps);
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        const float zz = zv[i];
        ov[i] = ov[i] * r * w[i] * (zz / (1.f + __expf(-zz)));
    }
}

// Write n_tok new K/V rows into the (paged) cache: row t at logical position
// pos[0] + t (single sequence) or pos[t] (batched rows, kvbase). One block
// per row; row % 32 == 0 (whole waves per q8 scale group).
template <int Q8>
__device__ __forceinline__ void kv_store_impl(const float *k, const float *v, const KvArgs &kv, const int *pos, int row) {
    const int t = blockIdx.x;
    const int lp = kv.kvbase == nullptr ? pos[0] + t : pos[t];
    const size_t prow = (size_t)kv_row(kv, kv_tb(kv, t), lp);
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        kv_st32<kv_kf(Q8)>(kv.k, kv.ks, prow * row + i, k[(size_t)t * row + i]);
        kv_st32<kv_vf(Q8)>(kv.v, kv.vs, prow * row + i, v[(size_t)t * row + i]);
    }
}
extern "C" __global__ void kv_store(const float *k, const float *v, KvArgs kv, const int *pos, int row) {
    kv_store_impl<false>(k, v, kv, pos, row);
}
extern "C" __global__ void kv_store_q8(const float *k, const float *v, KvArgs kv, const int *pos, int row) {
    kv_store_impl<true>(k, v, kv, pos, row);
}
extern "C" __global__ void kv_store_q8v(const float *k, const float *v, KvArgs kv, const int *pos, int row) {
    kv_store_impl<2>(k, v, kv, pos, row);
}
// q4: no Hadamard rotation here (attn_prep_q4 rotates); the host always uses attn_prep for q4
extern "C" __global__ void kv_store_q4(const float *k, const float *v, KvArgs kv, const int *pos, int row) {
    kv_store_impl<3>(k, v, kv, pos, row);
}

// Greedy argmax over n logits -> out[0]; also feeds the next decode step:
// ids[0] = token, pos[0] += advance.
extern "C" __global__ void __launch_bounds__(1024) argmax(const float *x, int n, int *out, int *ids, int *pos,
                                                          int advance) {
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
    if (threadIdx.x == 0) {
        out[0] = bi[0];
        ids[0] = bi[0];
        pos[0] += advance;
    }
}
