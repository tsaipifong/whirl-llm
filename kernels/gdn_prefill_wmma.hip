// WHIRL device code (gfx1201 / RDNA 4): chunked Gated DeltaNet prefill on f16 WMMA.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---- item P: WMMA DeltaNet prefill (begin) ----
#ifndef GW_FLOOR
#define GW_FLOOR 0
#endif
// ---------------------------------------------------------------------------
// Item P: chunked Gated DeltaNet prefill on the matrix cores (f16 WMMA, f32
// accumulate), same chunk algebra as gdn_chunk_prep / gdn_chunk_scan (C = 64):
//   X = beta (V - diag(A) K S0);  U = (I + Tstrict) X;  O = diag(A) Q S0 + M U
//   S_C = A_C S0 + K^T diag(e^{G_C - G}) U
// gdn_wprep (grid chunks x v-heads): T = (I+B)^-1 (strict lower part) and M,
//   both as f16 WMMA A-fragments of the 10 lower 16x16 tiles, plus G.
// gdn_wscan (grid v-heads x column groups): one wave owns 16 state columns
//   (8 C-fragment tiles of S in registers, which double as B-fragments); the
//   block stages q/k (f16, swizzled), k^T, T and M of each chunk in LDS.
// gfx12 wave32 fragment layout:
//   A, B: lane holds row (lane % 16), k = 8 * (lane / 16) + [0, 8)
//   D:    lane holds column n = lane % 16, rows m = 8 * (lane / 16) + [0, 8)
// ---------------------------------------------------------------------------
typedef _Float16 wh8 __attribute__((ext_vector_type(8)));
typedef float wf8 __attribute__((ext_vector_type(8)));
#define GW_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c)

// 64 x 128 f16 matrix in LDS, 16 chunks of 8 per row, chunk swizzled by row.
__device__ __forceinline__ int sw128(int t, int c) { return t * 16 + (c ^ (t & 15)); }
// 128 x 64 f16 (k^T), 8 chunks per row.
__device__ __forceinline__ int sw64(int i, int c) { return i * 8 + (c ^ (i & 7)); }
__device__ __forceinline__ int tri_idx(int r, int c) { return r * (r + 1) / 2 + c; }
__device__ __forceinline__ wh8 cvt8(wf8 x) {
    wh8 h;
#pragma unroll
    for (int e = 0; e < 8; ++e) h[e] = (_Float16)x[e];
    return h;
}
__device__ __forceinline__ wh8 cvt44(float4 a, float4 b) {
    wh8 h;
    h[0] = (_Float16)a.x; h[1] = (_Float16)a.y; h[2] = (_Float16)a.z; h[3] = (_Float16)a.w;
    h[4] = (_Float16)b.x; h[5] = (_Float16)b.y; h[6] = (_Float16)b.z; h[7] = (_Float16)b.w;
    return h;
}

__device__ __forceinline__ float gw_quad_sum(float x) {
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0xB1, 0xF, 0xF, false));  // xor 1
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x4E, 0xF, 0xF, false));  // xor 2
    return x;
}

#define GW_TILES 10           // lower 16x16 tiles of a 64x64 matrix
#define GW_TM_HALVES (2 * GW_TILES * 256)   // T then M, per (chunk, head)

extern "C" __global__ void __launch_bounds__(256)
    gdn_wprep(const float *qkv, const float *gl, const float *beta, _Float16 *TMout, float *Gout, int n_tok,
              int n_k_heads, int n_v_heads, int qkv_stride) {
    // grid (v-heads, chunks): the 3 v-heads of a k-head run close together (L2 reuse of q/k)
    const int c = blockIdx.y, h = blockIdx.x, kh = h % n_k_heads;
    const int t0 = c * GCH;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    __shared__ union {
        wh8 kq[2][64 * 16];         // K, Q (f16, swizzled)
        struct {
            float Bq[GCH][4][16];   // beta_t e^{G_t-G_s} k_t.k_s at [t][s % 4][s / 4]
            wh8 Tf[GW_TILES * 32], Mf[GW_TILES * 32];
        } b;
    } sm;
    wh8 *Tf = sm.b.Tf, *Mf = sm.b.Mf;
    __shared__ float G[GCH], Bt[GCH];
    // K, Q: 2 x 64 rows x 16 chunks of 8 = 2048 half8, 8 per thread
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int i = tid + e * 256;
        const int m = i >> 10, t = (i >> 4) & 63, cc = i & 15, tok = t0 + t;
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
        if (tok < n_tok) {
            const float *p = qkv + (size_t)tok * qkv_stride + (size_t)(m == 0 ? n_k_heads + kh : kh) * GDK + cc * 8;
            a = *(const float4 *)p;
            b = *(const float4 *)(p + 4);
        }
        sm.kq[m][sw128(t, cc)] = cvt44(a, b);
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
    // KK and QK lower tiles: 20 tiles over 8 waves (tile list: 0..9 KK, 10..19 QK)
    wf8 acc[3];
    int tr[3], tc[3];
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        const int tt = wid + 8 * p;
        const int ti = tt % 10;
        // ti -> (r, c) with c <= r
        const int r = ti < 1 ? 0 : ti < 3 ? 1 : ti < 6 ? 2 : 3;
        tr[p] = r;
        tc[p] = ti - r * (r + 1) / 2;
        acc[p] = (wf8){0, 0, 0, 0, 0, 0, 0, 0};
        if (tt < 20) {
            const int am = tt < 10 ? 0 : 1;  // A = K or Q
#pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                const wh8 a = sm.kq[am][sw128(r * 16 + (lane & 15), kt * 2 + (lane >> 4))];
                const wh8 b = sm.kq[0][sw128(tc[p] * 16 + (lane & 15), kt * 2 + (lane >> 4))];
                acc[p] = GW_WMMA(a, b, acc[p]);
            }
        }
    }
    __syncthreads();  // K/Q dead; G scanned
    // Write B (f32, to LDS) and M (f16 A-fragments) from the accumulators.
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        const int tt = wid + 8 * p;
        if (tt >= 20) continue;
        const int r = tr[p], cI = tc[p];
        const int s = cI * 16 + (lane & 15);
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = r * 16 + 8 * (lane >> 4) + v;
            const float d = __expf(G[t] - G[s]);
            if (tt < 10) {
                sm.b.Bq[t][s & 3][s >> 2] = (s < t) ? Bt[t] * d * acc[p][v] : 0.f;
            } else {
                const float mv = (s <= t && t0 + t < n_tok) ? d * acc[p][v] : 0.f;
                // A-fragment slot of M[t][s]: tile (r, cI), lane (t%16) + 16*((s%16)/8), elem s%8
                ((_Float16 *)Mf)[tri_idx(r, cI) * 256 + ((t & 15) + 16 * ((s & 15) >> 3)) * 8 + (s & 7)] = (_Float16)mv;
            }
        }
    }
    __syncthreads();
    // T = (I + B)^-1: column j = tid / 4, the 4 lanes of a quad split s (s % 4 == q).
    {
        const int j = tid >> 2, q = tid & 3;
        float x[16];
#pragma unroll
        for (int m = 0; m < 16; ++m) x[m] = 0.f;
#ifndef GW_NOSUB
#pragma unroll
        for (int t = 0; t < GCH; ++t) {
            // B[t][s] for s = 4m + q; entries s >= t are zero (only lower tiles are read)
            float part = 0.f;
            if (t > 0) {
                const float4 *brow = (const float4 *)sm.b.Bq[t][q];
#pragma unroll
                for (int m4 = 0; m4 <= (t - 1) / 16; ++m4) {
                    const float4 bv = brow[m4];
                    part += bv.x * x[4 * m4] + bv.y * x[4 * m4 + 1] + bv.z * x[4 * m4 + 2] + bv.w * x[4 * m4 + 3];
                }
            }
            const float tot = gw_quad_sum(part);
            const float xt = (t == j ? 1.f : 0.f) - tot;
            if ((t & 3) == q) x[t >> 2] = xt;
        }
#endif
        // strict lower part T[t][j] (t > j) into the A-fragments of tiles (t/16, j/16), j/16 <= t/16
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int t = 4 * m + q;
            if ((t >> 4) >= (j >> 4)) {
                const float tv = t > j ? x[m] : 0.f;
                ((_Float16 *)Tf)[tri_idx(t >> 4, j >> 4) * 256 + ((t & 15) + 16 * ((j & 15) >> 3)) * 8 + (j & 7)] = (_Float16)tv;
            }
        }
    }
    __syncthreads();
    const size_t chh = (size_t)c * n_v_heads + h;
    wh8 *dst = (wh8 *)(TMout + chh * GW_TM_HALVES);
    for (int i = tid; i < 2 * GW_TILES * 32; i += 256) dst[i] = i < GW_TILES * 32 ? Tf[i] : Mf[i - GW_TILES * 32];
    if (tid < GCH) Gout[chh * GCH + tid] = G[tid];
}

// WPB waves per block, 16 state columns each; grid (n_v_heads, 128 / (16 * WPB)).
// Keeps a per-lane offset opaque inside the chunk loop, so the compiler does not
// hoist dozens of loop-invariant 64-bit addresses (each load/store then uses a
// uniform row base + this one 32-bit lane offset).
#define GW_OPAQUE(x) asm volatile("" : "+v"(x))

template <int WPB>
__device__ void gdn_wscan_impl(const float *qkv, const _Float16 *TMin, const float *Gin, const float *beta, float *state,
                               float *out, int n_tok, int n_k_heads, int n_v_heads, int qkv_stride, float scale) {
    constexpr int NT = WPB * 32;
    constexpr int BLK_PER = 256 / NT;                          // 8x8 staging blocks per thread
    constexpr int NPTM = (2 * GW_TILES * 32 + NT - 1) / NT;    // T/M fragments per thread
    const int h = blockIdx.x, kh = h % n_k_heads;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, hi = lane >> 4, L = lane & 15;
    const int j = blockIdx.y * WPB * 16 + wid * 16 + L;
    __shared__ wh8 Ks[64 * 17], Qs[64 * 17], KT[128 * 9];    // padded rows: conflict-free fragment loads
    __shared__ wh8 TM[2 * GW_TILES * 32];
    __shared__ float Gs[GCH], Bs[GCH];
    const int abase = L * 17 + hi, kbase = L * 9 + hi;
    const int v_lane0 = 8 * hi * qkv_stride + 2 * n_k_heads * GDK + h * GDK + j;
    const int o_lane0 = (8 * hi * n_v_heads + h) * GDK + j;
    const size_t o_row = (size_t)n_v_heads * GDK;
    int s_lane0[BLK_PER];
#pragma unroll
    for (int bb = 0; bb < BLK_PER; ++bb) {
        const int b = tid + bb * NT, m = b >> 7, blk = b & 127;
        s_lane0[bb] = (blk >> 4) * 8 * qkv_stride + (m == 0 ? n_k_heads + kh : kh) * GDK + (blk & 15) * 8;
    }
    float *Sg = state + (size_t)h * GDK * GDK;
    wf8 S[8];
#pragma unroll
    for (int it = 0; it < 8; ++it)
#pragma unroll
        for (int v = 0; v < 8; ++v) S[it][v] = Sg[(size_t)(it * 16 + 8 * hi + v) * GDK + j];
    const int n_chunks = (n_tok + GCH - 1) / GCH;
    float4 pre[BLK_PER][16];
    uint4 ptm[NPTM];
    float pg = 0.f, pb = 0.f;
#define GW_FETCH(cf)                                                                                      \
    do {                                                                                                  \
        const int t0f = (cf) * GCH;                                                                       \
        const bool fullf = t0f + GCH <= n_tok;                                                            \
        _Pragma("unroll") for (int bb = 0; bb < BLK_PER; ++bb) {                                          \
            int so = s_lane0[bb];                                                                         \
            GW_OPAQUE(so);                                                                                \
            if (fullf) {                                                                                  \
                _Pragma("unroll") for (int e = 0; e < 8; ++e) {                                           \
                    const float *p = qkv + (size_t)(t0f + e) * qkv_stride + so;                           \
                    pre[bb][2 * e] = *(const float4 *)p;                                                  \
                    pre[bb][2 * e + 1] = *(const float4 *)(p + 4);                                        \
                }                                                                                         \
            } else {                                                                                      \
                const int t8 = ((tid + bb * NT) & 127) >> 4;                                              \
                _Pragma("unroll") for (int e = 0; e < 8; ++e) {                                           \
                    const int tok = t0f + t8 * 8 + e;                                                     \
                    const float *p = qkv + (size_t)min(tok, n_tok - 1) * qkv_stride + (so - t8 * 8 * qkv_stride); \
                    float4 a = *(const float4 *)p, b2 = *(const float4 *)(p + 4);                         \
                    if (tok >= n_tok) a = b2 = make_float4(0.f, 0.f, 0.f, 0.f);                           \
                    pre[bb][2 * e] = a;                                                                   \
                    pre[bb][2 * e + 1] = b2;                                                              \
                }                                                                                         \
            }                                                                                             \
        }                                                                                                 \
        const uint4 *src = (const uint4 *)(TMin + ((size_t)(cf) * n_v_heads + h) * GW_TM_HALVES);         \
        int to = tid;                                                                                     \
        GW_OPAQUE(to);                                                                                    \
        _Pragma("unroll") for (int e = 0; e < NPTM; ++e) ptm[e] = src[min(to + e * NT, 2 * GW_TILES * 32 - 1)]; \
        if (tid < GCH) {                                                                                  \
            pg = Gin[((size_t)(cf) * n_v_heads + h) * GCH + tid];                                         \
            pb = t0f + tid < n_tok ? beta[(size_t)(t0f + tid) * n_v_heads + h] : 0.f;                     \
        }                                                                                                 \
    } while (0)
#ifdef GW_PREFETCH
    GW_FETCH(0);
#endif
    for (int c = 0; c < n_chunks; ++c) {
        const int t0 = c * GCH;
        const bool full = t0 + GCH <= n_tok;
        __syncthreads();
#ifndef GW_PREFETCH
        if (!GW_FLOOR || c < 2) GW_FETCH(c);
#endif
        // stage K, Q (f16) and k^T, T/M, G, beta of chunk c
#pragma unroll
        for (int bb = 0; bb < BLK_PER; ++bb) {
            const int b = tid + bb * NT, m = b >> 7, blk = b & 127;
            const int t8 = blk >> 4, i8 = blk & 15;
            wh8 *dst = m == 0 ? Ks : Qs;
#pragma unroll
            for (int e = 0; e < 8; ++e) dst[(t8 * 8 + e) * 17 + i8] = cvt44(pre[bb][2 * e], pre[bb][2 * e + 1]);
            if (m == 0) {
#pragma unroll
                for (int d = 0; d < 8; ++d) {
                    wh8 hv;
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const float4 f = pre[bb][2 * e + (d >> 2)];
                        const int dd = d & 3;
                        hv[e] = (_Float16)(dd == 0 ? f.x : dd == 1 ? f.y : dd == 2 ? f.z : f.w);
                    }
                    KT[(i8 * 8 + d) * 9 + t8] = hv;
                }
            }
        }
#pragma unroll
        for (int e = 0; e < NPTM; ++e) {
            const int i = tid + e * NT;
            if (i < 2 * GW_TILES * 32) ((uint4 *)TM)[i] = ptm[e];
        }
        if (tid < GCH) {
            Gs[tid] = pg;
            Bs[tid] = pb;
        }
        __syncthreads();
        int v_lane = v_lane0;
        GW_OPAQUE(v_lane);
        // V (C-fragment layout) for this chunk
        float vv[4][8];
        if (full) {
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int v = 0; v < 8; ++v) vv[r][v] = (qkv + (size_t)(t0 + r * 16 + v) * qkv_stride)[v_lane];
        } else {
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int tok = t0 + r * 16 + 8 * hi + v;
                    const float x = qkv[(size_t)min(tok, n_tok - 1) * qkv_stride + (v_lane - 8 * hi * qkv_stride)];
                    vv[r][v] = tok < n_tok ? x : 0.f;
                }
        }
        wf8 X[4], O[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            X[r] = (wf8){0, 0, 0, 0, 0, 0, 0, 0};
            O[r] = X[r];
        }
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            const wh8 sb = cvt8(S[kt]);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                X[r] = GW_WMMA(Ks[abase + 2 * kt + r * 16 * 17], sb, X[r]);
                O[r] = GW_WMMA(Qs[abase + 2 * kt + r * 16 * 17], sb, O[r]);
            }
        }
        // X = beta (V - A X), O *= A
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int t = r * 16 + 8 * hi + v;
                const float a = __expf(Gs[t]);
                X[r][v] = Bs[t] * (vv[r][v] - a * X[r][v]);
                O[r][v] *= a;
            }
#ifdef GW_PREFETCH
        __builtin_amdgcn_sched_barrier(0);
        if (c + 1 < n_chunks) GW_FETCH(c + 1);
        __builtin_amdgcn_sched_barrier(0);
#endif
        wh8 xb[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) xb[r] = cvt8(X[r]);
        // U = X + Tstrict X
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int cc = 0; cc <= r; ++cc) X[r] = GW_WMMA(TM[tri_idx(r, cc) * 32 + lane], xb[cc], X[r]);
#pragma unroll
        for (int r = 0; r < 4; ++r) xb[r] = cvt8(X[r]);
        // O += M U
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int cc = 0; cc <= r; ++cc) O[r] = GW_WMMA(TM[(GW_TILES + tri_idx(r, cc)) * 32 + lane], xb[cc], O[r]);
        {
            int o_lane = o_lane0;
            GW_OPAQUE(o_lane);
            if (full) {
#pragma unroll
                for (int r = 0; r < 4; ++r)
#pragma unroll
                    for (int v = 0; v < 8; ++v) (out + (size_t)(t0 + r * 16 + v) * o_row)[o_lane] = O[r][v] * scale;
            } else {
#pragma unroll
                for (int r = 0; r < 4; ++r)
#pragma unroll
                    for (int v = 0; v < 8; ++v)
                        if (t0 + r * 16 + 8 * hi + v < n_tok) (out + (size_t)(t0 + r * 16 + v) * o_row)[o_lane] = O[r][v] * scale;
            }
        }
        // S = A_C S + K^T diag(e^{G_C - G}) U
        const float gc = Gs[GCH - 1];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int v = 0; v < 8; ++v) X[r][v] *= __expf(gc - Gs[r * 16 + 8 * hi + v]);
            xb[r] = cvt8(X[r]);
        }
        const float ac = __expf(gc);
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            wf8 acc = S[it] * ac;
#pragma unroll
            for (int cc = 0; cc < 4; ++cc) acc = GW_WMMA(KT[kbase + 2 * cc + it * 16 * 9], xb[cc], acc);
            S[it] = acc;
        }
    }
#undef GW_FETCH
#pragma unroll
    for (int it = 0; it < 8; ++it)
#pragma unroll
        for (int v = 0; v < 8; ++v) Sg[(size_t)(it * 16 + 8 * hi + v) * GDK + j] = S[it][v];
}

extern "C" __global__ void __launch_bounds__(256)
    gdn_wscan8(const float *qkv, const _Float16 *TMin, const float *Gin, const float *beta, float *state, float *out,
               int n_tok, int n_k_heads, int n_v_heads, int qkv_stride, float scale) {
    gdn_wscan_impl<8>(qkv, TMin, Gin, beta, state, out, n_tok, n_k_heads, n_v_heads, qkv_stride, scale);
}
extern "C" __global__ void __launch_bounds__(128)
    gdn_wscan4(const float *qkv, const _Float16 *TMin, const float *Gin, const float *beta, float *state, float *out,
               int n_tok, int n_k_heads, int n_v_heads, int qkv_stride, float scale) {
    gdn_wscan_impl<4>(qkv, TMin, Gin, beta, state, out, n_tok, n_k_heads, n_v_heads, qkv_stride, scale);
}

