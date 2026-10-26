// WHIRL device code (gfx1201 + gfx1151): gdn_step_norm candidates (feat026-gdnnorm). Not self-contained: included at
// the end of kernels/fused_decode.hip and kernels/gfx1151/fused_decode.hip (needs GdnSegs, WARP, warp_sum, wave_q8).
// SPDX-License-Identifier: Apache-2.0
//
// Every candidate computes the SAME values as the shipped kernel of its arch, bit for bit: the 4 key quarters, the
// 32-term fma chains per quarter, the ((r0 + r1) + r2) + r3 combines, the (p0 + p2) + (p1 + p3) norm tree of block_sum
// and wave_q8 are unchanged; only the schedule moves (the gfx1201 item E body, ported to gfx1151, plus knobs below).
//   MR    rows per LDS chunk (16 = item E; 8 = 24 KiB less LDS: gfx1151 WGPs hold 4 blocks instead of 3)
//   EARLY store the state right after the last row's update (before the o reduction + norm + quant tail) instead of
//         at the end (pend == nullptr only; replay mode never writes the state after new rows)
//   ZPRE  load z (one row per wave) and wn into registers with the row inputs, not inside the norm tail
//   PEND  honour the replay buffers (gfx1201); gfx1151 ships without replay, so its candidates ignore pend as it does
//   WPE   __launch_bounds__(512, W) = amdgpu_waves_per_eu(W) (register cap only; none by default). The shipped gfx1151 kernel takes 208 VGPRs:
//         4 waves per SIMD = ONE 16-wave block per WGP, so 48 heads run in 3 rounds over the 20 WGPs of the 8060S.
//         8 -> <= 192 VGPRs (2 blocks / WGP), 12 -> <= 128 (3 blocks / WGP: 48 heads in one round)
//
// Tune table (WHIRL_GDNN = id replaces the exported gdn_step_norm; default 0 = the shipped kernel):
//   1: MR 16           2: MR 8           3: MR 16 + EARLY        4: MR 16 + EARLY + ZPRE        5: MR 8 + EARLY + ZPRE
//   6: id 4 + WPE 8    7: id 4 + WPE 12   (LLVM drops WPE for 41 KiB LDS: its LDS occupancy model allows 1 block)
//   8: MR 4 + EARLY + ZPRE + WPE 8   9: same + WPE 12   12: MR 4 + EARLY + ZPRE (MR 4 = the Swift 4-row verify in one chunk)
//   10 / 11 (gfx1151 only): the SHIPPED gfx1151 per-row body verbatim + WPE 8 / 12 (3.7 KiB LDS: register cap only)
// WHIRL_GDNN_DEV=1 also exports gdnn<id>_step_norm for ids 1..9 (kt family gdnn compares them with the shipped kernel).
#ifndef WHIRL_GDNN
#define WHIRL_GDNN 0
#endif
#ifndef WHIRL_GDNN_DEV
#define WHIRL_GDNN_DEV 0
#endif

#if WHIRL_GDNN || WHIRL_GDNN_DEV
template <int MR, bool EARLY, bool ZPRE, bool PEND>
__device__ __forceinline__ void gdnn_impl(const float *qkv, const float *g, const float *beta, const float *z,
                                          const float *wn, float *out, int8_t *xq, float *xd, int n_k_heads,
                                          float scale, float eps, int n_v_heads, int qkv_stride, const GdnSegs &segs) {
    constexpr int DK = 128, DV = 128, QI = 4, PER = DK / QI;
    static_assert(MR <= 16, "the norm tail runs one row per wave (512 threads = 16 waves)");
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
    float wnr[4];
    if (ZPRE) {
#pragma unroll
        for (int c = 0; c < 4; ++c) wnr[c] = wn[32 * c + lane];
    }
    float S[PER];
#pragma unroll
    for (int e = 0; e < PER; ++e) S[e] = Sg[(size_t)(qi * PER + e) * DV + j];
    float *pend = PEND ? sgm.pend : nullptr;
    constexpr int PW = DK + DV + 2;  // pend row width per head
    if (PEND) {
        const int np = pend != nullptr ? sgm.npend : 0;
        if (np > 0) {
            // the rows the previous verify kept: the same state update as below (npend <= 16 = MR of item E: MR 16
            // runs the shipped replay verbatim; MR 8 replays in chunks of 8, the same sequential row order)
            if (MR == 16) {
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
            } else {
            for (int p0 = 0; p0 < np; p0 += MR) {
                const int npc = min(MR, np - p0);
                for (int i = threadIdx.x; i < npc * DK; i += blockDim.x) {
                    const int t = i / DK, d = i % DK;
                    const float *pr = pend + ((size_t)(p0 + t) * n_v_heads + h) * PW;
                    ks[t][d] = pr[d];
                    vs[t][d] = pr[DK + d];
                }
                if (threadIdx.x < npc) {
                    const float *pr = pend + ((size_t)(p0 + threadIdx.x) * n_v_heads + h) * PW;
                    dec[threadIdx.x] = pr[DK + DV];
                    bet[threadIdx.x] = pr[DK + DV + 1];
                }
                __syncthreads();
                for (int tl = 0; tl < npc; ++tl) {
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
                __syncthreads();  // ks / vs / dec / bet / red are reused by the next replay chunk or below
            }
#pragma unroll
            for (int e = 0; e < PER; ++e) Sg[(size_t)(qi * PER + e) * DV + j] = S[e];
            }
        }
    }
    bool stored = false;
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
        float zr[4] = {0.f, 0.f, 0.f, 0.f};
        if (ZPRE) {
            if (wid < nr) {
                const int t = r0 + c0 + wid;
#pragma unroll
                for (int c = 0; c < 4; ++c) zr[c] = z[(size_t)t * d_inner + (size_t)h * DV + 32 * c + lane];
            }
        }
        __syncthreads();
        if (PEND && pend != nullptr) {
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
            if (EARLY && pend == nullptr && ta == n_tok - 1) {
                // the final state: written now, overlapping the o reduction + norm tail
#pragma unroll
                for (int e = 0; e < PER; ++e) Sg[(size_t)(qi * PER + e) * DV + j] = S[e];
                stored = true;
            }
            redo[tl & 1][qi][j] = o;
        }
        __syncthreads();
        if (qi == 0)
            ov[nr - 1][j] = (redo[(nr - 1) & 1][0][j] + redo[(nr - 1) & 1][1][j] + redo[(nr - 1) & 1][2][j] +
                             redo[(nr - 1) & 1][3][j]) *
                            scale;
        __syncthreads();
        // gated RMSNorm + int8 copy, one wave per row: lane l holds columns l, 32 + l, 64 + l, 96 + l (= waves 0..3
        // of block_sum; the other waves' zeros leave its sum unchanged)
        // (ZPRE: MR <= 16 waves, so each wave runs at most one row: the row its z was prefetched for)
        for (int tl = wid; tl < (ZPRE ? min(nr, wid + 1) : nr); tl += blockDim.x / WARP) {
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
                const float zz = ZPRE ? zr[c] : z[(size_t)t * d_inner + (size_t)h * DV + jj];
                const float w = ZPRE ? wnr[c] : wn[jj];
                const float yy = y[c] * r * w * (zz / (1.f + __expf(-zz)));
                out[(size_t)t * d_inner + (size_t)h * DV + jj] = yy;
                wave_q8(yy, xq + (size_t)t * d_inner, xd + (size_t)t * (d_inner / 32), (h * DV + jj) >> 5);
            }
        }
        __syncthreads();  // LDS reuse by the next chunk
    }
    if (pend == nullptr && !stored) {
#pragma unroll
        for (int e = 0; e < PER; ++e) Sg[(size_t)(qi * PER + e) * DV + j] = S[e];
    }
}

#if defined(__gfx1151__)
// The shipped gfx1151 gdn_step_norm body, verbatim (kernels/gfx1151/fused_decode.hip), for ids 10 / 11.
__device__ __forceinline__ void gdnn1151_row_impl(const float *qkv, const float *g, const float *beta, const float *z,
                                                  const float *wn, float *out, int8_t *xq, float *xd, int n_k_heads,
                                                  float scale, float eps, int n_v_heads, int qkv_stride,
                                                  const GdnSegs &segs) {
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
#define GDNN1151_ROW_ENTRY(NAME, LB)                                                                                 \
    extern "C" __global__ void LB NAME(const float *qkv, const float *g, const float *beta, const float *z,         \
                                       const float *wn, float *out, int8_t *xq, float *xd, int n_k_heads,           \
                                       float scale, float eps, int n_v_heads, int qkv_stride, GdnSegs segs) {      \
        gdnn1151_row_impl(qkv, g, beta, z, wn, out, xq, xd, n_k_heads, scale, eps, n_v_heads, qkv_stride, segs);   \
    }
#endif
#if defined(__gfx1151__)
#define GDNN_PEND false
#else
#define GDNN_PEND true
#endif
#define GDNN_CFG_1 16, false, false, GDNN_PEND
#define GDNN_CFG_2 8, false, false, GDNN_PEND
#define GDNN_CFG_3 16, true, false, GDNN_PEND
#define GDNN_CFG_4 16, true, true, GDNN_PEND
#define GDNN_CFG_5 8, true, true, GDNN_PEND
#define GDNN_CFG_6 GDNN_CFG_4
#define GDNN_CFG_7 GDNN_CFG_4
#define GDNN_CFG_8 4, true, true, GDNN_PEND
#define GDNN_CFG_9 4, true, true, GDNN_PEND
#define GDNN_CFG_12 4, true, true, GDNN_PEND
#define GDNN_LB_1 __launch_bounds__(512)
#define GDNN_LB_2 __launch_bounds__(512)
#define GDNN_LB_3 __launch_bounds__(512)
#define GDNN_LB_4 __launch_bounds__(512)
#define GDNN_LB_5 __launch_bounds__(512)
#define GDNN_LB_6 __launch_bounds__(512, 8)
#define GDNN_LB_7 __launch_bounds__(512, 12)
#define GDNN_LB_8 __launch_bounds__(512, 8)
#define GDNN_LB_9 __launch_bounds__(512, 12)
#define GDNN_LB_12 __launch_bounds__(512)
#define GDNN_ENTRY(NAME, ID)                                                                                         \
    extern "C" __global__ void GDNN_LB_##ID                                                                          \
        NAME(const float *qkv, const float *g, const float *beta, const float *z, const float *wn, float *out,      \
             int8_t *xq, float *xd, int n_k_heads, float scale, float eps, int n_v_heads, int qkv_stride,           \
             GdnSegs segs) {                                                                                         \
        gdnn_impl<GDNN_CFG_##ID>(qkv, g, beta, z, wn, out, xq, xd, n_k_heads, scale, eps, n_v_heads, qkv_stride,   \
                                 segs);                                                                              \
    }
#define GDNN_ENTRY_X(NAME, ID) GDNN_ENTRY(NAME, ID)
#if WHIRL_GDNN == 10 || WHIRL_GDNN == 11
GDNN1151_ROW_ENTRY(gdn_step_norm, __launch_bounds__(512, WHIRL_GDNN == 10 ? 8 : 12))
#elif WHIRL_GDNN
GDNN_ENTRY_X(gdn_step_norm, WHIRL_GDNN)
#endif
#if WHIRL_GDNN_DEV
GDNN_ENTRY(gdnn1_step_norm, 1)
GDNN_ENTRY(gdnn2_step_norm, 2)
GDNN_ENTRY(gdnn3_step_norm, 3)
GDNN_ENTRY(gdnn4_step_norm, 4)
GDNN_ENTRY(gdnn5_step_norm, 5)
GDNN_ENTRY(gdnn6_step_norm, 6)
GDNN_ENTRY(gdnn7_step_norm, 7)
GDNN_ENTRY(gdnn8_step_norm, 8)
GDNN_ENTRY(gdnn9_step_norm, 9)
GDNN_ENTRY(gdnn12_step_norm, 12)
#if defined(__gfx1151__)
GDNN1151_ROW_ENTRY(gdnn10_step_norm, __launch_bounds__(512, 8))
GDNN1151_ROW_ENTRY(gdnn11_step_norm, __launch_bounds__(512, 12))
#endif
#endif
#endif  // WHIRL_GDNN || WHIRL_GDNN_DEV
