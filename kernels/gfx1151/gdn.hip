// WHIRL device code (gfx1151 / RDNA 3.5): Gated DeltaNet (per token, chunked prefill, sequential scan).
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Gated DeltaNet (one token per launch).
// ---------------------------------------------------------------------------

// Causal depthwise conv over [state(d_conv-1) | new tokens], SiLU; the last
// d_conv-1 inputs become the new state. w: [ch][d_conv]; state: [d_conv-1][ch]
// (oldest first); xin/out: [n_tok][ch]. Requires d_conv == 4.
#include "shared/gdn_prefill_gdn_conv_seq.h"  // also in the gfx1201 set (kernels/gdn_prefill.hip)

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
#include "shared/gdn_seq_gdn_conv_par.h"  // also in the gfx1201 set (kernels/gdn_seq.hip)
