    constexpr int WAVES_N = NTH / 32 / WAVES_M;  // shared fp8 GEMM main loop (gemm8_impl / gemm8_moe_impl, ADR-0005); see the end of this file
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int LD = BK + 16;  // bytes; 16-byte aligned rows
    constexpr int UPR = BK / 32;  // 32-value weight units per row per chunk
    constexpr int A_UNITS = BM * UPR;
    constexpr int A_PER = (A_UNITS + NTH - 1) / NTH;
    constexpr int CPR = BK / 16;  // 16-byte activation pieces per row per chunk
    constexpr int B_PER = BN * CPR / NTH;
    static_assert((A_UNITS % NTH == 0 || NTH % A_UNITS == 0) && BN * CPR % NTH == 0, "tile/threads");
    __shared__ __attribute__((aligned(16))) uint8_t As[BM * LD];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[BN * LD];
    __shared__ uint32_t tab[16][2];
    if (threadIdx.x < 32) tab[threadIdx.x >> 1][threadIdx.x & 1] = k_fp8tab[threadIdx.x >> 1][threadIdx.x & 1];

    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    const int G8_TILE_ORIGIN;

    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    uint4 rq[A_PER];
    uint32_t re[A_PER];
    int rref[A_PER];
    uint4 rb[B_PER];
#pragma unroll
    for (int e = 0; e < A_PER; ++e) {
        const int r = (threadIdx.x + e * NTH) / UPR;
        rref[e] = (r < BM && m0 + r < nrows) ? wref[m0 + r] : 0;
    }
#define G8_FETCH(k0_)                                                                                      \
    do {                                                                                                   \
        _Pragma("unroll") for (int e = 0; e < A_PER; ++e) {                                                \
            const int i = threadIdx.x + e * NTH;                                                           \
            const int r = i / UPR, u = i % UPR;                                                            \
            const int k = (k0_) + u * 32;                                                                  \
            if (i < A_UNITS && m0 + r < nrows) {                                                           \
                const uint8_t *sb = W + (size_t)(m0 + r) * row_bytes + (size_t)(k >> 8) * 136;             \
                rq[e] = *(const uint4 *)(sb + 8 + ((k & 255) >> 5) * 16);                                  \
                re[e] = sb[(k & 255) >> 5];                                                                \
            } else {                                                                                       \
                rq[e] = make_uint4(0, 0, 0, 0);                                                            \
                re[e] = 0;                                                                                 \
            }                                                                                              \
        }                                                                                                  \
        _Pragma("unroll") for (int e = 0; e < B_PER; ++e) {                                                \
            const int i = threadIdx.x + e * NTH;                                                           \
            const int t = i / CPR, c = i % CPR;                                                            \
            rb[e] = G8_TROW(t) < n_tok ? *(const uint4 *)(X8 + G8_TOFF(t) * ncols + (k0_) + c * 16)        \
                                        : make_uint4(0, 0, 0, 0);                                          \
        }                                                                                                  \
    } while (0)

    G8_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            if (i >= A_UNITS) break;
            const int r = i / UPR, u = i % UPR;
            int d = rref[e] - (int)re[e];
            d = d < 0 ? 0 : (d > 15 ? 15 : d);
            const uint32_t ta = tab[d][0], tb = tab[d][1];
            uint4 lo, hi;
            lo.x = mx_fp8x4(rq[e].x & 0x0F0F0F0Fu, ta, tb);
            lo.y = mx_fp8x4(rq[e].y & 0x0F0F0F0Fu, ta, tb);
            lo.z = mx_fp8x4(rq[e].z & 0x0F0F0F0Fu, ta, tb);
            lo.w = mx_fp8x4(rq[e].w & 0x0F0F0F0Fu, ta, tb);
            hi.x = mx_fp8x4((rq[e].x >> 4) & 0x0F0F0F0Fu, ta, tb);
            hi.y = mx_fp8x4((rq[e].y >> 4) & 0x0F0F0F0Fu, ta, tb);
            hi.z = mx_fp8x4((rq[e].z >> 4) & 0x0F0F0F0Fu, ta, tb);
            hi.w = mx_fp8x4((rq[e].w >> 4) & 0x0F0F0F0Fu, ta, tb);
            *(uint4 *)&As[r * LD + u * 32] = lo;
            *(uint4 *)&As[r * LD + u * 32 + 16] = hi;
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(uint4 *)&Bs[(i / CPR) * LD + (i % CPR) * 16] = rb[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) G8_FETCH(k0 + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            fp8x8 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = *(const fp8x8 *)&As[((wm * TM + i) * 16 + lane % 16) * LD + kk + 8 * (lane / 16)];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = *(const fp8x8 *)&Bs[((wn * TN + j) * 16 + lane % 16) * LD + kk + 8 * (lane / 16)];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    G8_WMMA_IF(j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a[i], b[j], acc[i][j]);
        }
    }
#undef G8_FETCH
#undef G8_TILE_ORIGIN
#undef G8_TROW
#undef G8_TOFF
#undef G8_WMMA_IF

// kernels/shared/gemm8_mainloop.h -- the fp8 x fp8 WMMA GEMM main loop shared by gemm8_impl (kernels/gemm_fp8.hip,
// dense) and gemm8_moe_impl (kernels/moe_mxfp4.hip, grouped MoE experts); ADR-0005 / arch025-sc11d.
// #included inside each function's braces (statement block, not a helper call: arch025-sc11c showed that an
// inlined helper can change the ISA). Before the #include the includer defines:
//   G8_TILE_ORIGIN  declarators of the block origin (m0, and t0 for the dense GEMM)
//   G8_TROW(t)      activation row of tile token t, compared with n_tok      (dense: t0 + t, MoE: t)
//   G8_TOFF(t)      the same row as size_t for the X8 offset                  (dense: (size_t)(t0 + t), MoE: (size_t)t)
//   G8_WMMA_IF(j)   statement prefix of the WMMA of accumulator column j    (dense: empty, MoE: the SKIP guard)
// and provides W, row_bytes, wref, X8, ncols, nrows, n_tok and the template constants BM, BN, BK, WAVES_M, NTH.
// The hooks are #undef'd here. Expanded, the token sequence is the one before the merge in both kernels.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
