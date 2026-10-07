// WHIRL device code (gfx1201 / RDNA 4): Gated DeltaNet: per-token conv and recurrence, chunkwise f32 prefill.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

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
// ba[t][nh, 2nh) = a (one 2nh-row GEMM). Same math as gdn_gates, written to b / a
// laid out [t][nh]. n = n_tok * n_heads.
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

// gdn_chunk_prep (P-9 rewrite, the same f32 operations per output as the original row sweep):
//  - K K^T and Q K^T on 4x4 register tiles with rows interleaved by 16 (conflict-free
//    16-byte LDS loads along d)
//  - T = (I + B)^-1 by columns: column s of X = -B is updated only from its own entries
//    X[r][s] and the never-updated X[t][r] (t > r) of step r, so one lane per column runs
//    the whole sweep in registers (same per-entry fma order as the 62-barrier row sweep)
extern "C" __global__ void __launch_bounds__(256)
    gdn_chunk_prep(const float *qkv, const float *gl, const float *beta, float *Wout, float *Uout, float *Mout,
                    float *Gout, int n_tok, int n_k_heads, int n_v_heads, int qkv_stride) {
    const int c = blockIdx.x, h = blockIdx.y, kh = h % n_k_heads;
    const int t0 = c * GCH;
    const int tid = threadIdx.x;
    __shared__ __attribute__((aligned(16))) float Ks[GCH][GDK + 4];  // K, later V
    __shared__ __attribute__((aligned(16))) float Ts[GCH][GCH + 4];  // -B transposed, then T
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
    const int tq = tid >> 4, sq = tid & 15;  // rows tq + 16 e, columns sq + 16 f
    {
        float acc[4][4] = {};
        for (int d = 0; d < GDK; d += 4) {
            float4 kt[4], ks[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                kt[e] = *(const float4 *)&Ks[tq + 16 * e][d];
                ks[e] = *(const float4 *)&Ks[sq + 16 * e][d];
            }
#pragma unroll
            for (int dd = 0; dd < 4; ++dd)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float kv = dd == 0 ? kt[e].x : dd == 1 ? kt[e].y : dd == 2 ? kt[e].z : kt[e].w;
#pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        const float sv = dd == 0 ? ks[f].x : dd == 1 ? ks[f].y : dd == 2 ? ks[f].z : ks[f].w;
                        acc[e][f] += kv * sv;
                    }
                }
        }
        // X = -B, stored transposed: Ts[s][t] = X[t][s]
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int t = tq + 16 * e, sI = sq + 16 * f;
                const float b = (sI < t) ? Bt[t] * __expf(G[t] - G[sI]) * acc[e][f] : 0.f;
                Ts[sI][t] = -b;
            }
    }
    // M = e^{G_t-G_s} q_t.k_s (s <= t); q rows straight from global
    {
        float acc[4][4] = {};
        const float *qr[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) qr[e] = qkv + (size_t)min(t0 + tq + 16 * e, n_tok - 1) * qkv_stride + (size_t)kh * GDK;
        for (int d = 0; d < GDK; d += 4) {
            float4 q4[4], ks[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                q4[e] = *(const float4 *)(qr[e] + d);
                ks[e] = *(const float4 *)&Ks[sq + 16 * e][d];
            }
#pragma unroll
            for (int dd = 0; dd < 4; ++dd)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float qv = dd == 0 ? q4[e].x : dd == 1 ? q4[e].y : dd == 2 ? q4[e].z : q4[e].w;
#pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        const float sv = dd == 0 ? ks[f].x : dd == 1 ? ks[f].y : dd == 2 ? ks[f].z : ks[f].w;
                        acc[e][f] += qv * sv;
                    }
                }
        }
        const size_t ch = (size_t)c * n_v_heads + h;
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int t = tq + 16 * e, sI = sq + 16 * f;
                Mout[ch * GCH * GCH + (size_t)t * GCH + sI] =
                    (sI <= t && t0 + t < n_tok) ? __expf(G[t] - G[sI]) * acc[e][f] : 0.f;
            }
    }
    __syncthreads();
    // T: lane s (s < 64) owns column s. Step r (r = 1..62, r > s): X[t][s] += X[t][r] * X[r][s]
    // for t > r, X[t][r] = Ts[r][t] (column r is only updated at steps > r).
    float col[GCH];
    if (tid < GCH) {
        const int s = tid;
#pragma unroll
        for (int t = 0; t < GCH; t += 4) {
            const float4 v = *(const float4 *)&Ts[s][t];
            col[t] = v.x; col[t + 1] = v.y; col[t + 2] = v.z; col[t + 3] = v.w;
        }
#pragma unroll
        for (int r = 1; r < GCH - 1; ++r) {
            if (r > s) {
                const float xr = col[r];
#pragma unroll
                for (int t = r + 1; t < GCH; ++t) col[t] += Ts[r][t] * xr;
            }
        }
    }
    __syncthreads();
    if (tid < GCH) {
        const int s = tid;
#pragma unroll
        for (int t = 1; t < GCH; ++t)
            if (t > s) Ts[t][s] = col[t];
    }
    __syncthreads();
    // Ts now holds T strictly below the diagonal (unit diagonal implicit).
    const int t4 = (tid >> 4) * 4;
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

// gdn_chunk_scan (P-9 rewrite: register-tiled, bitwise == the original per-element arithmetic):
// per chunk four stages, each staging <= 32 KB of f32 in LDS (next stage's global
// loads issued before the current stage's math):
//   0, 1 = W and Q, k dims [0, 64) and [64, 128): U = Ut - W S and o = Q S accumulate
//          in registers over the two halves, in the same i order (then o *= A, U -> LDS)
//   2    = M (64x64): o += M U (s <= t), write
//   3    = K (64x128, scaled by e^{G_C - G_s} when stored): S = A_C S + Kt^T U
// Thread tiles: stages 0..2 rows (tr, tr + 32) x 4 columns, stage 3 4 key dims x 4
// columns; operands come as 16-byte LDS loads (S unpadded, rows broadcast).
#define GSX 68   // W / Q / M row stride (floats) in the staging buffer
#define GKX 132  // K row stride
#define SCAN_FETCH(stage_id)                                                                       \
    do {                                                                                            \
        const int c_ = (stage_id) >> 2, p_ = (stage_id) & 3;                                        \
        const size_t ch_ = (size_t)c_ * n_v_heads + h;                                              \
        const int t0_ = c_ * GCH;                                                                   \
        if (p_ <= 1) {                                                                              \
            _Pragma("unroll") for (int e = 0; e < 4; ++e) {                                         \
                const int i = tid + e * 256;                                                        \
                const int t = i >> 4, d4 = (i & 15) * 4 + p_ * 64;                                  \
                pre[e] = *(const float4 *)(Win + ch_ * GCH * GDK + (size_t)t * GDK + d4);           \
                const int tok = min(t0_ + t, n_tok - 1);                                            \
                pre[e + 4] = *(const float4 *)(qkv + (size_t)tok * qkv_stride + (size_t)kh * GDK + d4); \
            }                                                                                       \
        } else if (p_ == 2) {                                                                       \
            _Pragma("unroll") for (int e = 0; e < 4; ++e) {                                         \
                const int i = tid + e * 256;                                                        \
                const int t = i >> 4, s4 = (i & 15) * 4;                                            \
                pre[e] = *(const float4 *)(Min + ch_ * GCH * GCH + (size_t)t * GCH + s4);           \
            }                                                                                       \
        } else {                                                                                    \
            _Pragma("unroll") for (int e = 0; e < 8; ++e) {                                         \
                const int i = tid + e * 256;                                                        \
                const int t = i >> 5, d4 = (i & 31) * 4;                                            \
                const int tok = t0_ + t;                                                            \
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);                                         \
                if (tok < n_tok)                                                                    \
                    v = *(const float4 *)(qkv + (size_t)tok * qkv_stride + (size_t)n_k_heads * GDK + (size_t)kh * GDK + d4); \
                pre[e] = v;                                                                         \
            }                                                                                       \
        }                                                                                           \
    } while (0)

#define F4C(v, k) ((k) == 0 ? (v).x : (k) == 1 ? (v).y : (k) == 2 ? (v).z : (v).w)

extern "C" __global__ void __launch_bounds__(256)
    gdn_chunk_scan(const float *qkv, const float *Win, const float *Uin, const float *Min, const float *Gin,
                    float *state, float *out, int n_tok, int n_k_heads, int n_v_heads, int qkv_stride, float scale) {
    const int h = blockIdx.x, j0 = blockIdx.y * 32, kh = h % n_k_heads;
    const int tid = threadIdx.x;
    __shared__ __attribute__((aligned(16))) float S[GDK * 32];
    __shared__ __attribute__((aligned(16))) float U[GCH * 32];
    __shared__ __attribute__((aligned(16))) float X[2 * GCH * GSX];
    __shared__ float Gs[GCH];
    float *Sg = state + (size_t)h * GDK * GDK;
    for (int i = tid; i < GDK * 32; i += blockDim.x) S[i] = Sg[(size_t)(i / 32) * GDK + j0 + i % 32];
    const int n_chunks = (n_tok + GCH - 1) / GCH;
    const int n_stages = 4 * n_chunks;
    const int tr = tid >> 3, tc = (tid & 7) * 4;  // stages 0..2: rows tr, tr + 32; columns tc..tc+3
    const int si0 = (tid >> 3) * 4;               // stage 3: key dims si0..si0+3; columns tc..tc+3
    float4 pre[8];
    SCAN_FETCH(0);
    float u0[4], u1[4], o0[4], o1[4];
    for (int st = 0; st < n_stages; ++st) {
        const int c = st >> 2, p = st & 3;
        const int t0 = c * GCH;
        const size_t ch = (size_t)c * n_v_heads + h;
        __syncthreads();
        if (p == 0 && tid < GCH) Gs[tid] = Gin[ch * GCH + tid];
        if (p == 3) __syncthreads();  // Gs visible for the K scaling
        if (p <= 1) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = tid + e * 256;
                const int t = i >> 4, d4 = (i & 15) * 4;
                *(float4 *)&X[t * GSX + d4] = pre[e];
                *(float4 *)&X[GCH * GSX + t * GSX + d4] = pre[e + 4];
            }
        } else if (p == 2) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = tid + e * 256;
                const int t = i >> 4, s4 = (i & 15) * 4;
                *(float4 *)&X[t * GSX + s4] = pre[e];
            }
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = tid + e * 256;
                const int t = i >> 5, d4 = (i & 31) * 4;
                float4 v = pre[e];
                const float f = __expf(Gs[GCH - 1] - Gs[t]);
                v.x *= f; v.y *= f; v.z *= f; v.w *= f;
                *(float4 *)&X[t * GKX + d4] = v;
            }
        }
        __syncthreads();
        if (st + 1 < n_stages) SCAN_FETCH(st + 1);
        if (p <= 1) {
            if (p == 0) {  // U starts at Ut, o at 0
                const float4 a = *(const float4 *)(Uin + ch * GCH * GDK + (size_t)tr * GDK + j0 + tc);
                const float4 b = *(const float4 *)(Uin + ch * GCH * GDK + (size_t)(tr + 32) * GDK + j0 + tc);
                u0[0] = a.x; u0[1] = a.y; u0[2] = a.z; u0[3] = a.w;
                u1[0] = b.x; u1[1] = b.y; u1[2] = b.z; u1[3] = b.w;
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) o0[jj] = o1[jj] = 0.f;
            }
            const float *Wl = X, *Ql = X + GCH * GSX, *Sp = S + (p * 64) * 32 + tc;
            for (int ii = 0; ii < 64; ii += 4) {
                const float4 w0 = *(const float4 *)&Wl[tr * GSX + ii], w1 = *(const float4 *)&Wl[(tr + 32) * GSX + ii];
                const float4 q0 = *(const float4 *)&Ql[tr * GSX + ii], q1 = *(const float4 *)&Ql[(tr + 32) * GSX + ii];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float4 s = *(const float4 *)&Sp[(ii + k) * 32];
                    const float sv[4] = {s.x, s.y, s.z, s.w};
                    const float wa = F4C(w0, k), wb = F4C(w1, k), qa = F4C(q0, k), qb = F4C(q1, k);
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        u0[jj] -= wa * sv[jj];
                        u1[jj] -= wb * sv[jj];
                        o0[jj] += qa * sv[jj];
                        o1[jj] += qb * sv[jj];
                    }
                }
            }
            if (p == 1) {
                const float a0 = __expf(Gs[tr]), a1 = __expf(Gs[tr + 32]);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    o0[jj] *= a0;
                    o1[jj] *= a1;
                }
                *(float4 *)&U[tr * 32 + tc] = make_float4(u0[0], u0[1], u0[2], u0[3]);
                *(float4 *)&U[(tr + 32) * 32 + tc] = make_float4(u1[0], u1[1], u1[2], u1[3]);
            }
        } else if (p == 2) {  // o += M U (s <= t), write
            const float *Ml = X;
            for (int sI = 0; sI <= tr + 32; ++sI) {
                const float4 uu = *(const float4 *)&U[sI * 32 + tc];
                const float uv[4] = {uu.x, uu.y, uu.z, uu.w};
                const float mb = Ml[(tr + 32) * GSX + sI];
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) o1[jj] += mb * uv[jj];
                if (sI <= tr) {
                    const float ma = Ml[tr * GSX + sI];
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) o0[jj] += ma * uv[jj];
                }
            }
            const int tok0 = t0 + tr, tok1 = t0 + tr + 32;
            if (tok0 < n_tok)
                *(float4 *)(out + ((size_t)tok0 * n_v_heads + h) * GDK + j0 + tc) =
                    make_float4(o0[0] * scale, o0[1] * scale, o0[2] * scale, o0[3] * scale);
            if (tok1 < n_tok)
                *(float4 *)(out + ((size_t)tok1 * n_v_heads + h) * GDK + j0 + tc) =
                    make_float4(o1[0] * scale, o1[1] * scale, o1[2] * scale, o1[3] * scale);
        } else {  // S = A_C S + Kt^T U (each thread updates only its own S entries)
            const float ac = __expf(Gs[GCH - 1]);
            float acc[4][4];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float4 s = *(const float4 *)&S[(si0 + a) * 32 + tc];
                acc[a][0] = ac * s.x; acc[a][1] = ac * s.y; acc[a][2] = ac * s.z; acc[a][3] = ac * s.w;
            }
            const float *Kl = X;
            for (int sI = 0; sI < GCH; ++sI) {
                const float4 kk = *(const float4 *)&Kl[sI * GKX + si0];
                const float4 uu = *(const float4 *)&U[sI * 32 + tc];
                const float kv[4] = {kk.x, kk.y, kk.z, kk.w}, uv[4] = {uu.x, uu.y, uu.z, uu.w};
#pragma unroll
                for (int a = 0; a < 4; ++a)
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) acc[a][jj] += kv[a] * uv[jj];
            }
#pragma unroll
            for (int a = 0; a < 4; ++a) *(float4 *)&S[(si0 + a) * 32 + tc] = make_float4(acc[a][0], acc[a][1], acc[a][2], acc[a][3]);
        }
    }
    __syncthreads();
    for (int i = tid; i < GDK * 32; i += blockDim.x) Sg[(size_t)(i / 32) * GDK + j0 + i % 32] = S[i];
}
#undef SCAN_FETCH
#undef F4C
#undef GSX
#undef GKX

