// WHIRL device code (gfx1201 / RDNA 4): fragment-tiled fp8 GEMM (gemm8t*), f16 x f16 fragment-tiled GEMM (gemmh*), gemmhq*.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---- item K: fragment-tiled fp8 prefill GEMM (begin) ----
// ---------------------------------------------------------------------------
// Item K: MXFP4 x fp8 prefill GEMM with fragment-tiled activations (gemm8t).
// The fp8 activation producers write x16 in WMMA-fragment-tiled layout: the
// 16-token x 16-k fragment (tb, ks) is 256 contiguous bytes at
// (tb * K/16 + ks) * 256, lane l holding token tb*16 + l%16, k ks*16 + (l/16)*8
// .. +8 (x8t_off) -- exactly the B operand bytes of v_wmma_f32_16x16x16_fp8_fp8,
// so each wave loads its B fragments with one coalesced global_load_b64 per
// fragment straight into the WMMA register (no activation tile in LDS: 16 KB
// instead of 55 KB of LDS, several blocks per CU). Weights are unpacked (folded
// E8M0 exponent, as gemm8) into LDS in fragment order (conflict-free
// ds_load_b64), double-buffered with one barrier per 64-k slab. Epilogue: the
// per-row 2^(ref-127) is hoisted and each tile is written with 16-byte stores.
// Every accumulator sees the same WMMA sequence in k as gemm8_impl and the scale
// is the same value (st * 2^e == ldexpf(st, e) exactly), so the output is
// bitwise gemm8's.
// Fragment-tiled activation idea from bkvargyas/r9700-stack (Apache-2.0,
// c03aa1e) kernels/r9k_moe_mxfp4a8.hip (r9k_moe_4bit_prefill_at_kernel,
// r9k_quant_rows_fp8_tiled); the kernel below is written for WHIRL's weight
// layout and tile structure.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void g8t_lds_barrier() {
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}

// BM weight rows x BN tokens per block, 64 k per slab, WAVES_M x (8 / WAVES_M) waves;
// TG token blocks x all row blocks per rasterization group.
template <int BM, int BN, int WAVES_M, int TG, bool OH>
__device__ void gemm8t_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const uint8_t *__restrict__ wref,
                            const uint8_t *__restrict__ XT, const float *__restrict__ sx, float *__restrict__ Y,
                            int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int NTH = 256, BK = 64;
    constexpr int WAVES_N = NTH / 32 / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int KSPS = BK / 16, UPR = BK / 32;
    constexpr int A_UNITS = BM * UPR;
    constexpr int A_PER = (A_UNITS + NTH - 1) / NTH;
    static_assert(A_UNITS % NTH == 0 || NTH % A_UNITS == 0, "units");
    constexpr int BUF = BM * BK;
    __shared__ __attribute__((aligned(16))) uint8_t As[2 * BUF];
    __shared__ uint32_t tab[16][2];
    if (threadIdx.x < 32) tab[threadIdx.x >> 1][threadIdx.x & 1] = k_fp8tab[threadIdx.x >> 1][threadIdx.x & 1];

    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    int bx, by;
    {
        const int gx = gridDim.x, gy = gridDim.y, bid = blockIdx.y * gx + blockIdx.x;
        const int per = TG * gy, g = bid / per, rem = bid % per;
        const int tg = min(TG, gx - g * TG);
        bx = g * TG + rem % tg;
        by = rem / tg;
    }
    const int m0 = by * BM, t0 = bx * BN;
    const int ksteps = ncols >> 4;
    const int tb_last = (n_tok - 1) >> 4;

    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    const uint8_t *bbase[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        int tb = (t0 >> 4) + wn * TN + j;
        tb = tb < tb_last ? tb : tb_last;  // clamped: tokens past n_tok are never stored
        bbase[j] = XT + (size_t)__builtin_amdgcn_readfirstlane(tb * ksteps * 256);
    }
    const unsigned boff = (unsigned)lane * 8u;

    uint4 rq[A_PER];
    uint32_t re[A_PER];
    int rref[A_PER];
    const uint8_t *wrow[A_PER];
    unsigned wdst[A_PER];
#pragma unroll
    for (int e = 0; e < A_PER; ++e) {
        const int i = (threadIdx.x + e * NTH) % A_UNITS;
        const int r = i % BM, u = i / BM;
        int gr = m0 + r;
        gr = gr < nrows ? gr : nrows - 1;  // clamped: rows past nrows are never stored
        rref[e] = wref[gr];
        wrow[e] = W + (size_t)gr * row_bytes;
        wdst[e] = (unsigned)((((r >> 4) * KSPS + 2 * u) * 32 + (r & 15)) * 8);
    }
    auto fetchW = [&](int k0) {
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = (threadIdx.x + e * NTH) % A_UNITS;
            const int k = k0 + (i / BM) * 32;
            const uint8_t *sb = wrow[e] + (size_t)(k >> 8) * 136;
            rq[e] = *(const uint4 *)(sb + 8 + ((k & 255) >> 5) * 16);
            re[e] = sb[(k & 255) >> 5];
        }
    };
    auto storeW = [&](int buf) {
        uint8_t *base = As + buf * BUF;
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            if (A_UNITS < NTH && threadIdx.x >= A_UNITS) break;
            int d = rref[e] - (int)re[e];
            d = d < 0 ? 0 : (d > 15 ? 15 : d);
            const uint32_t ta = tab[d][0], tb = tab[d][1];
            fp8x8 l0, l1, h0, h1;
            l0.x = mx_fp8x4(rq[e].x & 0x0F0F0F0Fu, ta, tb);
            l0.y = mx_fp8x4(rq[e].y & 0x0F0F0F0Fu, ta, tb);
            l1.x = mx_fp8x4(rq[e].z & 0x0F0F0F0Fu, ta, tb);
            l1.y = mx_fp8x4(rq[e].w & 0x0F0F0F0Fu, ta, tb);
            h0.x = mx_fp8x4((rq[e].x >> 4) & 0x0F0F0F0Fu, ta, tb);
            h0.y = mx_fp8x4((rq[e].y >> 4) & 0x0F0F0F0Fu, ta, tb);
            h1.x = mx_fp8x4((rq[e].z >> 4) & 0x0F0F0F0Fu, ta, tb);
            h1.y = mx_fp8x4((rq[e].w >> 4) & 0x0F0F0F0Fu, ta, tb);
            *(fp8x8 *)(base + wdst[e]) = l0;
            *(fp8x8 *)(base + wdst[e] + 16 * 8) = l1;
            *(fp8x8 *)(base + wdst[e] + 256) = h0;
            *(fp8x8 *)(base + wdst[e] + 256 + 16 * 8) = h1;
        }
    };
    fp8x8 bf[TN][KSPS];
    auto loadB = [&](int s) {
        const unsigned ko = boff + (unsigned)(s * KSPS * 256);
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int st = 0; st < KSPS; ++st) bf[j][st] = *(const fp8x8 *)(bbase[j] + ko + (unsigned)(st * 256));
    };
    auto compute = [&](int buf) {
        const uint8_t *base = As + buf * BUF + (wm * TM * KSPS * 32 + lane) * 8;
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
        }
    };
    const int nslabs = ncols / BK;
    fetchW(0);
    __syncthreads();  // tab
    // One barrier per slab: the store of slab s overwrites buffer s&1, last read in slab s-2;
    // a wave stores slab s only after barrier s-1, which every wave reaches after its slab s-2 WMMAs.
    for (int s = 0; s < nslabs; ++s) {
        const int buf = s & 1;
        loadB(s);
        storeW(buf);
        g8t_lds_barrier();
        if (s + 1 < nslabs) fetchW((s + 1) * BK);
        compute(buf);
    }

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

// Configurations (must match the gemm8t table in kernels_abi.h): (BM, BN, WAVES_M, TG)
#define GEMM8T_ENTRY(I, BM, BN, WM, TG)                                                                         \
    extern "C" __global__ void __launch_bounds__(256)                                                           \
        gemm8t_c##I(const uint8_t *W, uint64_t row_bytes, const uint8_t *wref, const uint8_t *X8, const float *sx, \
                    float *Y, int ncols, int nrows, int n_tok, int accumulate) {                                \
        gemm8t_impl<BM, BN, WM, TG, false>(W, row_bytes, wref, X8, sx, Y, ncols, nrows, n_tok, accumulate);      \
    }                                                                                                           \
    extern "C" __global__ void __launch_bounds__(256)                                                           \
        gemm8th_c##I(const uint8_t *W, uint64_t row_bytes, const uint8_t *wref, const uint8_t *X8, const float *sx, \
                     float *Y, int ncols, int nrows, int n_tok, int accumulate) {                               \
        gemm8t_impl<BM, BN, WM, TG, true>(W, row_bytes, wref, X8, sx, Y, ncols, nrows, n_tok, 0);                \
    }
GEMM8T_ENTRY(0, 128, 256, 4, 4)
GEMM8T_ENTRY(1, 128, 128, 2, 4)
GEMM8T_ENTRY(2, 128, 64, 4, 4)
GEMM8T_ENTRY(3, 128, 32, 8, 4)
GEMM8T_ENTRY(4, 128, 48, 8, 4)

// fragment-tiled fp8 producers (same bytes as the row-major kernels, only the address differs)
extern "C" __global__ void __launch_bounds__(256) qact_fp8t(const float *x, uint8_t *q, float *sx, int ncols) {
    qact_fp8_impl<true>(x, q, sx, ncols);
}
extern "C" __global__ void __launch_bounds__(256) silu_mul_x8ht(const _Float16 *g, const _Float16 *u, uint8_t *q, float *sx, int ncols) {
    silu_mul_x8h_impl<true>(g, u, q, sx, ncols);
}
extern "C" __global__ void __launch_bounds__(256) rmsnorm_x8t(const float *x, const float *w, uint8_t *q, float *sx, int n, float eps) {
    rmsnorm_x_impl<true, true>(x, w, q, sx, n, eps);
}
extern "C" __global__ void __launch_bounds__(256) silu_mul_x8t(const float *g, const float *u, uint8_t *q, float *sx, int ncols) {
    silu_mul_x_impl<true, true>(g, u, q, sx, ncols);
}
extern "C" __global__ void __launch_bounds__(256) gated_norm_x8t(const float *o, const float *z, const float *w, uint8_t *q, float *sx, int n_heads, float eps) {
    gated_norm_x_impl<true, float, true>(o, z, w, q, sx, n_heads, eps);
}
extern "C" __global__ void __launch_bounds__(256) gated_norm_x8ht(const float *o, const _Float16 *z, const float *w, uint8_t *q, float *sx, int n_heads, float eps) {
    gated_norm_x_impl<true, _Float16, true>(o, z, w, q, sx, n_heads, eps);
}

// ---------------------------------------------------------------------------
// Item K: f16-weight x f16-activation prefill GEMM (Q4_K_M etc. after dequant_f16),
// bitwise the same as gemm3_impl (gemm_cN_f16): same per-accumulator WMMA order.
// Both operands are staged through LDS in WMMA fragment order (conflict-free
// ds_load_b128, 16 x 16 fragment = 512 contiguous bytes, lane l = row l%16,
// k (l/16)*8 .. +8), 32-k slabs, 128 rows x 256 tokens, 8 waves (TM 2 x TN 8),
// rasterized in groups of 4 token blocks, 16-byte epilogue stores.
// Row-major W (row stride ncols) and X, ncols % 32 == 0.
// ---------------------------------------------------------------------------
template <bool OH, int T = -1>
__device__ void gemmh_impl(const uint8_t *__restrict__ Wb, uint64_t row_bytes, const _Float16 *__restrict__ X,
                           float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int BM = 128, BN = 256, BK = 32, WAVES_M = 4, NTH = 256, TG = 4;
    constexpr int WAVES_N = NTH / 32 / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int KSPS = BK / 16, GPR = BK / 8;
    constexpr int A_PER = BM * GPR / NTH, B_PER = BN * GPR / NTH;
    constexpr int ABUF = BM * BK;
    __shared__ __attribute__((aligned(16))) _Float16 S[ABUF + BN * BK];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    int bx, by;
    {
        const int gx = gridDim.x, gy = gridDim.y, bid = blockIdx.y * gx + blockIdx.x;
        const int per = TG * gy, g = bid / per, rem = bid % per;
        const int tg = min(TG, gx - g * TG);
        bx = g * TG + rem % tg;
        by = rem / tg;
    }
    const int m0 = by * BM, t0 = bx * BN;
    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    half8 ra[A_PER], rb[B_PER];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            const int r = i % BM, gg = i / BM;
            int gr = m0 + r;
            gr = gr < nrows ? gr : nrows - 1;  // clamped: never stored
            if constexpr (T < 0) ra[e] = *(const half8 *)(Wb + (size_t)gr * row_bytes + (size_t)(k0 + gg * 8) * 2);
            else ra[e] = dec_group_h<T>(Wb + (size_t)gr * row_bytes, k0 / 8 + gg);
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            const int t = i % BN, gg = i / BN;
            int gt = t0 + t;
            gt = gt < n_tok ? gt : n_tok - 1;  // clamped: never stored
            rb[e] = *(const half8 *)(X + (size_t)gt * ncols + k0 + gg * 8);
        }
    };
    auto store = [&]() {
        _Float16 *sb = S + ABUF;
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            const int r = i % BM, gg = i / BM;
            *(half8 *)(S + (((r >> 4) * KSPS + (gg >> 1)) * 32 + (r & 15) + 16 * (gg & 1)) * 8) = ra[e];
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            const int t = i % BN, gg = i / BN;
            *(half8 *)(sb + (((t >> 4) * KSPS + (gg >> 1)) * 32 + (t & 15) + 16 * (gg & 1)) * 8) = rb[e];
        }
    };
    auto compute = [&]() {
        const _Float16 *sb = S + ABUF;
#pragma unroll
        for (int st = 0; st < KSPS; ++st) {
            half8 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = *(const half8 *)(S + (((wm * TM + i) * KSPS + st) * 32 + lane) * 8);
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = *(const half8 *)(sb + (((wn * TN + j) * KSPS + st) * 32 + lane) * 8);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
        }
    };
    const int nslabs = ncols / BK;
    fetch(0);
    for (int s = 0; s < nslabs; ++s) {
        if (s > 0) g8t_lds_barrier();
        store();
        g8t_lds_barrier();
        if (s + 1 < nslabs) fetch((s + 1) * BK);
        compute();
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = t0 + (wn * TN + j) * 16 + lane % 16;
        if (t >= n_tok) continue;
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int row0 = m0 + (wm * TM + i) * 16 + 8 * (lane / 16);
            const bool full = (nrows & 7) == 0 && row0 + 8 <= nrows;
            if (OH) {
                _Float16 *dh = (_Float16 *)Y + (size_t)t * nrows + row0;
                if (full) {
                    half8 hv;
#pragma unroll
                    for (int v = 0; v < 8; ++v) hv[v] = (_Float16)acc[i][j][v];
                    *(half8 *)dh = hv;
                } else {
#pragma unroll
                    for (int v = 0; v < 8; ++v)
                        if (row0 + v < nrows) dh[v] = (_Float16)acc[i][j][v];
                }
                continue;
            }
            float *dst = Y + (size_t)t * nrows + row0;
            if (full) {
                float4 y0 = {acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]};
                float4 y1 = {acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]};
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
                    if (row0 + v < nrows) dst[v] = accumulate ? dst[v] + acc[i][j][v] : acc[i][j][v];
            }
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) gemmh_f16(const _Float16 *W, const _Float16 *X, float *Y, int ncols, int nrows,
                                                            int n_tok, int accumulate) {
    gemmh_impl<false>((const uint8_t *)W, (uint64_t)ncols * 2, X, Y, ncols, nrows, n_tok, accumulate);
}
extern "C" __global__ void __launch_bounds__(256) gemmhh_f16(const _Float16 *W, const _Float16 *X, float *Y, int ncols, int nrows,
                                                             int n_tok, int accumulate) {
    gemmh_impl<true>((const uint8_t *)W, (uint64_t)ncols * 2, X, Y, ncols, nrows, n_tok, 0);
}
// fused-dequant twins (f32 output; f16 output = gemmhqh_*)
#define GEMMHQ_ENTRY(SUF, T)                                                                                        \
    extern "C" __global__ void __launch_bounds__(256) gemmhq_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, \
                                                                  float *Y, int ncols, int nrows, int n_tok, int accumulate) { \
        gemmh_impl<false, T>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                   \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(256) gemmhqh_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, \
                                                                   float *Y, int ncols, int nrows, int n_tok, int accumulate) { \
        gemmh_impl<true, T>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0);                                             \
    }
GEMMHQ_ENTRY(q8_0, QT_Q8_0)
GEMMHQ_ENTRY(q3_k, QT_Q3_K)
GEMMHQ_ENTRY(q4_k, QT_Q4_K)
GEMMHQ_ENTRY(q5_k, QT_Q5_K)
GEMMHQ_ENTRY(q6_k, QT_Q6_K)
GEMMHQ_ENTRY(iq4_xs, QT_IQ4_XS)
GEMMHQ_ENTRY(mxfp4, QT_MXFP4)
// ---- item K: fragment-tiled fp8 prefill GEMM (end) ----
