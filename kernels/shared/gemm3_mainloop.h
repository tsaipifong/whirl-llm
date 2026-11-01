    constexpr int WAVES_N = NTH / 32 / WAVES_M;  // shared f16 prefill GEMM main loop (gemm3_impl / gemm_tile_impl, ADR-0005); see the end of this file
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int LD = BK + 8;
    constexpr int GPR = BK / 8;
    constexpr int A_PER = BM * GPR / NTH;
    constexpr int B_PER = BN * GPR / NTH;
    static_assert(BM * GPR % NTH == 0 && BN * GPR % NTH == 0, "tile/threads");
    __shared__ _Float16 As[BM][LD];
    __shared__ _Float16 Bs[BN][LD];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    G3_TILE_ORIGIN
    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    half8 ra[A_PER], rb[B_PER];
#define G3_FETCH(k0_)                                                                               \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < A_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int r = i / GPR, gg = i % GPR;                                                    \
            G3_A_MODE1(k0_) if (m0 + r < nrows) {                                                   \
                ra[e] = dec_group_h<T>(W + (size_t)(m0 + r) * row_bytes, (k0_) / 8 + gg);            \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) ra[e][q] = (_Float16)0.f;              \
            }                                                                                       \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < B_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int t = i / GPR, gg = i % GPR;                                                    \
            if (t0 + t < n_tok) {                                                                   \
                rb[e] = *(const half8 *)(X + (size_t)(t0 + t) * ncols + (k0_) + gg * 8);            \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) rb[e][q] = (_Float16)0.f;              \
            }                                                                                       \
        }                                                                                           \
    } while (0)
    G3_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(half8 *)&As[i / GPR][(i % GPR) * 8] = ra[e];
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(half8 *)&Bs[i / GPR][(i % GPR) * 8] = rb[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) G3_FETCH(k0 + BK);
        G3_SKIP_CHUNK
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
#if defined(__GFX11__)
            half16 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = ld_frag16(&As[(wm * TM + i) * 16 + lane % 16][kk]);
#pragma unroll
            for (int j = 0; j < TN; ++j)
                G3_COL_IF(j) b[j] = ld_frag16(&Bs[(wn * TN + j) * 16 + lane % 16][kk]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    G3_COL_IF(j) acc[i][j] = WMMA16(a[i], b[j], acc[i][j]);
#else
            half8 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = *(const half8 *)&As[(wm * TM + i) * 16 + lane % 16][kk + 8 * (lane / 16)];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                G3_COL_IF(j) b[j] = *(const half8 *)&Bs[(wn * TN + j) * 16 + lane % 16][kk + 8 * (lane / 16)];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    G3_COL_IF(j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
#endif
        }
    }
#undef G3_FETCH

#undef G3_TILE_ORIGIN
#undef G3_A_MODE1
#undef G3_SKIP_CHUNK
#undef G3_COL_IF

// kernels/shared/gemm3_mainloop.h -- the f16 WMMA prefill GEMM main loop (fetch -> LDS -> WMMA over the K chunks)
// shared by gemm3_impl (kernels/gemm_prefill.hip, dense prefill GEMM) and gemm_tile_impl (kernels/moe.hip, one tile
// of the grouped expert GEMM), in both kernel sets (kernels/ and kernels/gfx1151/); ADR-0005 / arch025-sc11d.
// #included inside each function's braces (statement block, not a helper call: arch025-sc11c showed that an
// inlined helper can change the ISA). Before the #include the includer defines:
//   G3_TILE_ORIGIN  statement declaring the tile origin m0, t0 (gemm3: from blockIdx / TOK_X; tile: empty,
//                   m0 and t0 are parameters)
//   G3_A_MODE1(k)   prefix of the weight fetch for chunk k (gemm3: the MODE == 1 raw-f16 branch ending in else;
//                   tile: empty)
//   G3_SKIP_CHUNK   statement before the WMMA loop of a K chunk (gfx1151 tile: the SKIP early continue; else empty)
//   G3_COL_IF(j)    prefix of the column-j fragment load and WMMA (gfx1151 tile: the SKIP column guard; else empty)
// The fragment loads / WMMA are per architecture: gfx12 half8 + v_wmma_..._gfx12, gfx11 (__GFX11__, the gfx1151
// set) half16 ld_frag16 + WMMA16 of kernels/gfx1151. The #if lines vanish in the preprocessed kernel set.
// and provides W, row_bytes, X, ncols, nrows, n_tok and the template parameters T, BM, BN, BK, WAVES_M, NTH
// (dec_group_h<T>). Leaves acc[TM][TN], lane, wm, wn, TM, TN for the caller's epilogue. The hooks are #undef'd
// here. Expanded, the token sequence is the one before the merge in both kernels.
// The weight / activation fetch is a macro, not a lambda: a by-reference capture of ra/rb takes their address and
// pushes them to scratch.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
