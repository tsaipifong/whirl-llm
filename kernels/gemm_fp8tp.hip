// WHIRL device code (gfx1201 / RDNA 4): schedule variants of the fragment-tiled fp8 GEMM gemm8t_c0 (gemm8tp_v*).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by whirl_kernels.hip after gemm_fp8t.hip (uses fp8x8, k_fp8tab, mx_fp8x4,
// g8t_lds_barrier, float8, half8, WARP).
//
// feat026-gemm8tpp (kernel audit gap #11). Opt-in only (host: WHIRL_GEMM8T_V=<v>, default 0 = gemm8t_c0); the
// geometry is gemm8t_c0's (128 rows x 256 tokens, 8 waves 4 x 2, BK 64, raster 4), so the host launches them with
// kGemm8tCfgs[0]. Every variant issues, for each accumulator acc[i][j], the same WMMA sequence in k as gemm8t_c0
// (slab 0..n-1, k-step 0..3, same A / B bytes) and the same epilogue, so the output is bitwise gemm8t_c0's (and so
// gemm8_c0's). Only the issue order of loads / LDS traffic and the waits change:
//   v1  rolling B prefetch: bf[j][st] of slab s+1 is loaded right after the last WMMA that reads bf[j][st] of slab s
//       (no extra VGPR; item K's rejected B prefetch doubled bf to 256 VGPR) + the last slab peeled so the compiler's
//       wait counters are exact (the shipped loop shares one compute block between the last slab and the others, and
//       the conservative merge makes the last WMMAs wait for the next slab's weight fetch).
//   v2  v1 + sched_group_barrier interleave (TM WMMA : 1 B load), so each k-step's reloads issue right after its
//       WMMAs (without it the compiler sinks the reloads of k-steps 0..2 behind the WMMAs of k-step 2, ISA probe).
//   v3  v2 + s_setprio(1) around the WMMA burst.
//   v4  v2 + weights fetched two slabs ahead (two register sets, loop unrolled by 2 so they stay static).
//   v5  ping-pong (CDNA4 8-wave pattern, HipKittens / AMD AI Developer Hub fp8_gemm_hip_cdna4 s14): the two wave
//       groups wn = 0 / 1 (waves w and w + 4, which share a SIMD) run one barrier apart; memory phase = B global
//       loads, A fragments LDS -> registers, unpack + store of the next weight slab, fetch of the slab after;
//       MMA phase = 64 WMMA under s_setprio(1). The MMA phase touches no LDS, so the 2-slab LDS ring stays safe.
//   v6  v5 without s_setprio (isolates the barrier stagger).
// ---------------------------------------------------------------------------

// epilogue of gemm8t_impl (same expressions, same order)
template <int TM, int TN, int WAVES_M, bool OH>
__device__ __forceinline__ void g8tp_epilogue(float8 (&acc)[TM][TN], const uint8_t *__restrict__ wref, const float *__restrict__ sx,
                                              float *__restrict__ Y, int nrows, int n_tok, int accumulate, int m0, int t0, int lane,
                                              int wm, int wn) {
    float sc[TM][8];
    int rows0[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        rows0[i] = m0 + (wm * TM + i) * 16 + 8 * (lane / 16);
#pragma unroll
        for (int v = 0; v < 8; ++v) sc[i][v] = rows0[i] + v < nrows ? ldexpf(1.f, (int)wref[rows0[i] + v] - 127) : 0.f;
    }
    const bool rows8 = (nrows & 7) == 0;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = t0 + (wn * TN + j) * 16 + lane % 16;
        if (t >= n_tok) continue;
        const float st = sx[t];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int row0 = rows0[i];
            if (OH) {
                _Float16 *dh = (_Float16 *)Y + (size_t)t * nrows + row0;
                if (rows8 && row0 + 8 <= nrows) {
                    half8 hv;
#pragma unroll
                    for (int v = 0; v < 8; ++v) hv[v] = (_Float16)(acc[i][j][v] * (st * sc[i][v]));
                    *(half8 *)dh = hv;
                } else {
#pragma unroll
                    for (int v = 0; v < 8; ++v)
                        if (row0 + v < nrows) dh[v] = (_Float16)(acc[i][j][v] * ldexpf(st, (int)wref[row0 + v] - 127));
                }
                continue;
            }
            float *dst = Y + (size_t)t * nrows + row0;
            if (rows8 && row0 + 8 <= nrows) {
                float4 y0, y1;
                y0.x = acc[i][j][0] * (st * sc[i][0]); y0.y = acc[i][j][1] * (st * sc[i][1]);
                y0.z = acc[i][j][2] * (st * sc[i][2]); y0.w = acc[i][j][3] * (st * sc[i][3]);
                y1.x = acc[i][j][4] * (st * sc[i][4]); y1.y = acc[i][j][5] * (st * sc[i][5]);
                y1.z = acc[i][j][6] * (st * sc[i][6]); y1.w = acc[i][j][7] * (st * sc[i][7]);
                if (accumulate) {
                    const float4 o0 = *(const float4 *)dst, o1 = *(const float4 *)(dst + 4);
                    y0.x = o0.x + y0.x; y0.y = o0.y + y0.y; y0.z = o0.z + y0.z; y0.w = o0.w + y0.w;
                    y1.x = o1.x + y1.x; y1.y = o1.y + y1.y; y1.z = o1.z + y1.z; y1.w = o1.w + y1.w;
                }
                *(float4 *)dst = y0;
                *(float4 *)(dst + 4) = y1;
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v)
                    if (row0 + v < nrows) {
                        const float y = acc[i][j][v] * ldexpf(st, (int)wref[row0 + v] - 127);
                        dst[v] = accumulate ? dst[v] + y : y;
                    }
            }
        }
    }
}

__device__ __forceinline__ void g8tp_barrier() {
    asm volatile("s_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}

// Shared setup of every variant: block raster, B bases, weight-unit addressing (as gemm8t_impl).
#define G8TP_SETUP                                                                                                   \
    constexpr int BM = 128, BN = 256, WAVES_M = 4, TG = 4, NTH = 256, BK = 64;                                        \
    constexpr int WAVES_N = NTH / 32 / WAVES_M;                                                                       \
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;                                                     \
    constexpr int KSPS = BK / 16, UPR = BK / 32;                                                                      \
    constexpr int A_UNITS = BM * UPR;                                                                                 \
    constexpr int A_PER = (A_UNITS + NTH - 1) / NTH;                                                                  \
    static_assert(A_UNITS % NTH == 0, "units");                                                                       \
    constexpr int BUF = BM * BK;                                                                                      \
    __shared__ __attribute__((aligned(16))) uint8_t As[2 * BUF];                                                      \
    __shared__ uint32_t tab[16][2];                                                                                   \
    if (threadIdx.x < 32) tab[threadIdx.x >> 1][threadIdx.x & 1] = k_fp8tab[threadIdx.x >> 1][threadIdx.x & 1];       \
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;                                                    \
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;                                                                 \
    int bx, by;                                                                                                       \
    {                                                                                                                 \
        const int gx = gridDim.x, gy = gridDim.y, bid = blockIdx.y * gx + blockIdx.x;                                \
        const int per = TG * gy, g = bid / per, rem = bid % per;                                                      \
        const int tg = min(TG, gx - g * TG);                                                                          \
        bx = g * TG + rem % tg;                                                                                       \
        by = rem / tg;                                                                                                \
    }                                                                                                                 \
    const int m0 = by * BM, t0 = bx * BN;                                                                             \
    const int ksteps = ncols >> 4;                                                                                    \
    const int tb_last = (n_tok - 1) >> 4;                                                                             \
    float8 acc[TM][TN];                                                                                               \
    _Pragma("unroll") for (int i = 0; i < TM; ++i)                                                                    \
        _Pragma("unroll") for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};                  \
    const uint8_t *bbase[TN];                                                                                         \
    _Pragma("unroll") for (int j = 0; j < TN; ++j) {                                                                  \
        int tb = (t0 >> 4) + wn * TN + j;                                                                             \
        tb = tb < tb_last ? tb : tb_last;                                                                             \
        bbase[j] = XT + (size_t)__builtin_amdgcn_readfirstlane(tb * ksteps * 256);                                    \
    }                                                                                                                 \
    const unsigned boff = (unsigned)lane * 8u;                                                                        \
    int rref[A_PER];                                                                                                  \
    const uint8_t *wrow[A_PER];                                                                                       \
    unsigned wdst[A_PER];                                                                                             \
    _Pragma("unroll") for (int e = 0; e < A_PER; ++e) {                                                               \
        const int i = (threadIdx.x + e * NTH) % A_UNITS;                                                              \
        const int r = i % BM, u = i / BM;                                                                             \
        int gr = m0 + r;                                                                                              \
        gr = gr < nrows ? gr : nrows - 1;                                                                             \
        rref[e] = wref[gr];                                                                                           \
        wrow[e] = W + (size_t)gr * row_bytes;                                                                         \
        wdst[e] = (unsigned)((((r >> 4) * KSPS + 2 * u) * 32 + (r & 15)) * 8);                                        \
    }                                                                                                                 \
    auto fetchW = [&](int k0, uint4(&rq)[A_PER], uint32_t(&re)[A_PER]) {                                              \
        _Pragma("unroll") for (int e = 0; e < A_PER; ++e) {                                                           \
            const int i = (threadIdx.x + e * NTH) % A_UNITS;                                                          \
            const int k = k0 + (i / BM) * 32;                                                                         \
            const uint8_t *sb = wrow[e] + (size_t)(k >> 8) * 136;                                                     \
            rq[e] = *(const uint4 *)(sb + 8 + ((k & 255) >> 5) * 16);                                                 \
            re[e] = sb[(k & 255) >> 5];                                                                               \
        }                                                                                                             \
    };                                                                                                                \
    auto storeW = [&](int buf, const uint4(&rq)[A_PER], const uint32_t(&re)[A_PER]) {                                 \
        uint8_t *base = As + buf * BUF;                                                                               \
        _Pragma("unroll") for (int e = 0; e < A_PER; ++e) {                                                           \
            int d = rref[e] - (int)re[e];                                                                             \
            d = d < 0 ? 0 : (d > 15 ? 15 : d);                                                                        \
            const uint32_t ta = tab[d][0], tb = tab[d][1];                                                            \
            fp8x8 l0, l1, h0, h1;                                                                                     \
            l0.x = mx_fp8x4(rq[e].x & 0x0F0F0F0Fu, ta, tb);                                                           \
            l0.y = mx_fp8x4(rq[e].y & 0x0F0F0F0Fu, ta, tb);                                                           \
            l1.x = mx_fp8x4(rq[e].z & 0x0F0F0F0Fu, ta, tb);                                                           \
            l1.y = mx_fp8x4(rq[e].w & 0x0F0F0F0Fu, ta, tb);                                                           \
            h0.x = mx_fp8x4((rq[e].x >> 4) & 0x0F0F0F0Fu, ta, tb);                                                    \
            h0.y = mx_fp8x4((rq[e].y >> 4) & 0x0F0F0F0Fu, ta, tb);                                                    \
            h1.x = mx_fp8x4((rq[e].z >> 4) & 0x0F0F0F0Fu, ta, tb);                                                    \
            h1.y = mx_fp8x4((rq[e].w >> 4) & 0x0F0F0F0Fu, ta, tb);                                                    \
            *(fp8x8 *)(base + wdst[e]) = l0;                                                                          \
            *(fp8x8 *)(base + wdst[e] + 16 * 8) = l1;                                                                 \
            *(fp8x8 *)(base + wdst[e] + 256) = h0;                                                                    \
            *(fp8x8 *)(base + wdst[e] + 256 + 16 * 8) = h1;                                                           \
        }                                                                                                             \
    };                                                                                                                \
    fp8x8 bf[TN][KSPS];                                                                                               \
    const int nslabs = ncols / BK

// v1..v4: gemm8t_c0's single-phase loop with a rolling B prefetch.
template <int V, bool OH>
__device__ void gemm8tq_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const uint8_t *__restrict__ wref,
                             const uint8_t *__restrict__ XT, const float *__restrict__ sx, float *__restrict__ Y, int ncols,
                             int nrows, int n_tok, int accumulate) {
    constexpr bool SGB = V >= 2, PRIO = V == 3, W2 = V == 4;
    G8TP_SETUP;
    uint4 rqa[A_PER], rqb[A_PER];
    uint32_t rea[A_PER], reb[A_PER];
    auto loadB_st = [&](int s, int st) {
        const unsigned ko = boff + (unsigned)((s * KSPS + st) * 256);
#pragma unroll
        for (int j = 0; j < TN; ++j) bf[j][st] = *(const fp8x8 *)(bbase[j] + ko);
    };
    // WMMAs of slab s (buffer s & 1); with NEXT the B fragments of slab s + 1 replace those just consumed. With SGB
    // the A fragments of the whole slab are read first, the WMMAs run j-outer (bf[j][st]'s last read is then the
    // TM-th WMMA of column j; each accumulator still sees the same k order) and every k-step is a fenced region
    // "TM WMMA, 1 B load" x TN.
    auto compute = [&](int s, bool next) {
        const uint8_t *base = As + (s & 1) * BUF + (wm * TM * KSPS * 32 + lane) * 8;
        if (PRIO) __builtin_amdgcn_s_setprio(1);
        if constexpr (SGB) {
            fp8x8 af[TM][KSPS];
#pragma unroll
            for (int st = 0; st < KSPS; ++st)
#pragma unroll
                for (int i = 0; i < TM; ++i) af[i][st] = *(const fp8x8 *)(base + (i * KSPS + st) * 256);
#pragma unroll
            for (int st = 0; st < KSPS; ++st) {
#pragma unroll
                for (int j = 0; j < TN; ++j)
#pragma unroll
                    for (int i = 0; i < TM; ++i)
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][st], bf[j][st], acc[i][j]);
                if (next) {
                    loadB_st(s + 1, st);
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        __builtin_amdgcn_sched_group_barrier(0x008, TM, 0);  // TM WMMAs of column j ...
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);   // ... then its B reload
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
            }
        } else {
#pragma unroll
            for (int st = 0; st < KSPS; ++st) {
                fp8x8 a[TM];
#pragma unroll
                for (int i = 0; i < TM; ++i) a[i] = *(const fp8x8 *)(base + (i * KSPS + st) * 256);
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a[i], bf[j][st], acc[i][j]);
                if (next) loadB_st(s + 1, st);
            }
        }
        if (PRIO) __builtin_amdgcn_s_setprio(0);
    };
    // one slab: store its weights (fetched NW slabs ago), barrier, fetch slab s + NW, WMMAs
    auto step = [&](int s, uint4(&rq)[A_PER], uint32_t(&re)[A_PER], bool fetch, bool next) {
        storeW(s & 1, rq, re);
        g8t_lds_barrier();
        if (fetch) fetchW((s + (W2 ? 2 : 1)) * BK, rq, re);
        compute(s, next);
        if (W2) __builtin_amdgcn_sched_barrier(0);  // keep the two register sets' live ranges apart
    };
    fetchW(0, rqa, rea);
    if (W2) fetchW(BK, rqb, reb);
#pragma unroll
    for (int st = 0; st < KSPS; ++st) loadB_st(0, st);
    __syncthreads();  // tab
    // One barrier per slab, as gemm8t_impl: the store of slab s overwrites buffer s & 1, last read in slab s - 2.
    int s = 0;
    if constexpr (!W2) {
        for (; s + 1 < nslabs; ++s) step(s, rqa, rea, true, true);
        step(s, rqa, rea, false, false);
    } else {
        // two register sets: slab s uses set s & 1 (loop unrolled by 2). Even slab counts >= 2 only (every product
        // shape; the host takes gemm8t_c0 otherwise, ops::gemm8tpOk).
        for (; s + 2 < nslabs; s += 2) {
            step(s, rqa, rea, true, true);
            step(s + 1, rqb, reb, true, true);
        }
        step(s, rqa, rea, false, true);
        step(s + 1, rqb, reb, false, false);
    }
    g8tp_epilogue<TM, TN, WAVES_M, OH>(acc, wref, sx, Y, nrows, n_tok, accumulate, m0, t0, lane, wm, wn);
}

// v5 / v6: ping-pong. Group wn = 1 runs one barrier behind group wn = 0, so on each SIMD one wave is in its memory
// phase while the other is in its MMA phase. LDS ring (2 slabs): a group writes buffer (s+1)&1 and reads buffer s&1
// only in its memory phase; the other group is then in its MMA phase (registers only), and the barrier between two
// memory phases of different groups orders every write before the reads of the next slab.
template <int V, bool OH>
__device__ void gemm8tpp_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const uint8_t *__restrict__ wref,
                              const uint8_t *__restrict__ XT, const float *__restrict__ sx, float *__restrict__ Y, int ncols,
                              int nrows, int n_tok, int accumulate) {
    constexpr bool PRIO = V == 5;
    G8TP_SETUP;
    uint4 rq[A_PER];
    uint32_t re[A_PER];
    fp8x8 af[TM][KSPS];
    const unsigned abase = (unsigned)((wm * TM * KSPS * 32 + lane) * 8);
    fetchW(0, rq, re);
    __syncthreads();  // tab
    storeW(0, rq, re);
    if (nslabs > 1) fetchW(BK, rq, re);
    g8t_lds_barrier();
    if (wn == 1) g8tp_barrier();  // phase offset
    __builtin_amdgcn_sched_barrier(0);
    // one slab: memory phase, barrier, MMA phase, barrier. STORE / FETCH are literals at every call (last two slabs
    // peeled) so the compiler's wait counters are exact: the MMA phase waits for the B loads only, never for the
    // weight fetch issued after them.
    auto phase = [&](int s, bool store, bool fetch) {
        {
            const unsigned ko = boff + (unsigned)(s * KSPS * 256);
#pragma unroll
            for (int j = 0; j < TN; ++j)
#pragma unroll
                for (int st = 0; st < KSPS; ++st) bf[j][st] = *(const fp8x8 *)(bbase[j] + ko + (unsigned)(st * 256));
            const uint8_t *ab = As + (s & 1) * BUF + abase;
#pragma unroll
            for (int st = 0; st < KSPS; ++st)
#pragma unroll
                for (int i = 0; i < TM; ++i) af[i][st] = *(const fp8x8 *)(ab + (i * KSPS + st) * 256);
            if (store) storeW((s + 1) & 1, rq, re);
            if (fetch) fetchW((s + 2) * BK, rq, re);
        }
        g8t_lds_barrier();
        __builtin_amdgcn_sched_barrier(0);
        if (PRIO) __builtin_amdgcn_s_setprio(1);
#pragma unroll
        for (int st = 0; st < KSPS; ++st)
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][st], bf[j][st], acc[i][j]);
        if (PRIO) __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        g8tp_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };
    if (nslabs == 1) {
        phase(0, false, false);
    } else {
        int s = 0;
        for (; s + 2 < nslabs; ++s) phase(s, true, true);
        phase(s, true, false);
        phase(s + 1, false, false);
    }
    if (wn == 0) g8tp_barrier();  // balance the phase offset
    g8tp_epilogue<TM, TN, WAVES_M, OH>(acc, wref, sx, Y, nrows, n_tok, accumulate, m0, t0, lane, wm, wn);
}
#undef G8TP_SETUP

#define GEMM8TP_ENTRY(V, IMPL)                                                                                        \
    extern "C" __global__ void __launch_bounds__(256)                                                                 \
        gemm8tp_v##V(const uint8_t *W, uint64_t row_bytes, const uint8_t *wref, const uint8_t *X8, const float *sx,   \
                     float *Y, int ncols, int nrows, int n_tok, int accumulate) {                                     \
        IMPL<V, false>(W, row_bytes, wref, X8, sx, Y, ncols, nrows, n_tok, accumulate);                               \
    }                                                                                                                 \
    extern "C" __global__ void __launch_bounds__(256)                                                                 \
        gemm8tph_v##V(const uint8_t *W, uint64_t row_bytes, const uint8_t *wref, const uint8_t *X8, const float *sx,  \
                      float *Y, int ncols, int nrows, int n_tok, int accumulate) {                                    \
        IMPL<V, true>(W, row_bytes, wref, X8, sx, Y, ncols, nrows, n_tok, 0);                                         \
    }
GEMM8TP_ENTRY(1, gemm8tq_impl)
GEMM8TP_ENTRY(2, gemm8tq_impl)
GEMM8TP_ENTRY(3, gemm8tq_impl)
GEMM8TP_ENTRY(4, gemm8tq_impl)
GEMM8TP_ENTRY(5, gemm8tpp_impl)
GEMM8TP_ENTRY(6, gemm8tpp_impl)
#undef GEMM8TP_ENTRY
