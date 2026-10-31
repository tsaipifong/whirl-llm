    constexpr int BM = 128, BN = 256, BK = 32, WAVES_M = 4, NTH = 256, TG = 4;  // shared f16 x f16 GEMM body (gemmh_impl / vis_gemm, ADR-0005); see the end of this file
#if !defined(GEMMH_DEQUANT)
    static_assert(T < 0, "gemmh_impl: dequantizing instantiations need GEMMH_DEQUANT (dec_group_h)");
#endif
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
#if defined(GEMMH_DEQUANT)
            if constexpr (T < 0) ra[e] = *(const half8 *)(Wb + (size_t)gr * row_bytes + (size_t)(k0 + gg * 8) * 2);
            else ra[e] = dec_group_h<T>(Wb + (size_t)gr * row_bytes, k0 / 8 + gg);
#else
            ra[e] = *(const half8 *)(Wb + (size_t)gr * row_bytes + (size_t)(k0 + gg * 8) * 2);
#endif
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
#if defined(__GFX11__)
        // the slab holds k 0..7 of row r at slot r and k 8..15 at slot r + 16: a gfx11
        // fragment (all 16 k of row lane % 16) joins the two slots
        const int r16 = lane % 16;
#pragma unroll
        for (int st = 0; st < KSPS; ++st) {
            half16 a[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const _Float16 *pa = S + (((wm * TM + i) * KSPS + st) * 32) * 8;
                a[i] = cat_frag(*(const half8 *)(pa + r16 * 8), *(const half8 *)(pa + (r16 + 16) * 8));
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const _Float16 *pb = sb + (((wn * TN + j) * KSPS + st) * 32) * 8;
                const half16 bj = cat_frag(*(const half8 *)(pb + r16 * 8), *(const half8 *)(pb + (r16 + 16) * 8));
#pragma unroll
                for (int i = 0; i < TM; ++i) acc[i][j] = VWMMA(a[i], bj, acc[i][j]);
            }
        }
#else
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
#endif
    };
    const int nslabs = ncols / BK;
    fetch(0);
    for (int s = 0; s < nslabs; ++s) {
        if (s > 0) GEMMH_LDS_BARRIER();
        store();
        GEMMH_LDS_BARRIER();
        if (s + 1 < nslabs) fetch((s + 1) * BK);
        compute();
    }
#if defined(__GFX11__)
    static_assert(!OH, "gemmh_impl: f16 output is gfx1201-only");
    // gfx11: element v of this lane is row 2v + lane / 16 of the 16-row tile; per-element stores
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = t0 + (wn * TN + j) * 16 + lane % 16;
        if (t >= n_tok) continue;
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + 2 * v + lane / 16;
                if (row < nrows) {
                    float *dst = Y + (size_t)t * nrows + row;
                    *dst = accumulate ? *dst + acc[i][j][v] : acc[i][j][v];
                }
            }
    }
    return;
#endif
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

// WHIRL device code: f16 x f16 fragment-tiled prefill GEMM body (gemmh_impl), shared by the gfx1201 kernel set
// (kernels/gemm_fp8t.hip: gemmh_f16, gemmhh_f16, gemmhq*_*) and the vision code object (kernels/vision/vision.hip:
// vis_gemm, gfx1201 and gfx1151). ADR-0005 / arch025-sc11c; vision.hip had a copy of it.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: the STATEMENTS of the body, #included between the braces of a function that has the parameters
//   (const uint8_t *Wb, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate)
// and constants OH (bool, f16 output) and T (int, < 0 = f16 weights): gemm_fp8t.hip gemmh_impl<OH, T> (a template),
// vision.hip vis_gemm (the kernel itself, OH = false, T = -1). Not a callable template for vision: calling an inlined
// template from vis_gemm changed its ISA (128 commuted v_add_f32 operands, progress/arch025-sc11c.HANDOFF.md), the
// body in the kernel keeps it identical. The includer defines before it:
//   GEMMH_LDS_BARRIER()  the LDS-only workgroup barrier of that code object (gemm_fp8t.hip: g8t_lds_barrier(),
//                        vision.hip: lds_barrier(), which is __syncthreads() on gfx11)
//   GEMMH_DEQUANT        (optional) T >= 0 instantiations dequantize the weights with dec_group_h<T> (the kernel set
//                        has it; vision does not)
// and provides WARP, half8, float8; on gfx11 also half16, cat_frag and VWMMA (vision.hip).
// The #if lines vanish in the gfx1201 kernel set (no __GFX11__), so its preprocessed TU is the one before the move.
// Line 1 stays a code line and this comment stays at the end: clang -E -P prints the first token after an #include
// (or after skipped #if lines) with other whitespace, which would change the preprocessed kernel-set text.
//
// Item K: f16-weight x f16-activation prefill GEMM (Q4_K_M etc. after dequant_f16),
// bitwise the same as gemm3_impl (gemm_cN_f16): same per-accumulator WMMA order.
// Both operands are staged through LDS in WMMA fragment order (conflict-free
// ds_load_b128, 16 x 16 fragment = 512 contiguous bytes, lane l = row l%16,
// k (l/16)*8 .. +8), 32-k slabs, 128 rows x 256 tokens, 8 waves (TM 2 x TN 8),
// rasterized in groups of 4 token blocks, 16-byte epilogue stores.
// gfx11 (vision only): the gfx11 fragment joins the two 8-k slots of a row, accumulator element v is row 2v + lane / 16
// (per-element stores, f32 output only).
// Row-major W (row stride ncols) and X, ncols % 32 == 0.
