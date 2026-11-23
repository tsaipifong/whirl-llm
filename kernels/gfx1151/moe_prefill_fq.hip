// v0252-8smoe2 (PLAN V1-147, levers C and E): twins of the Radeon 8060S (gfx1151) MoE prefill expert GEMMs
// (kernels/gfx1151/moe_prefill_mt.hip). A fragment of the gfx1151 translation unit, included LAST
// (whirl_kernels_gfx1151.hip: no existing kernel moves); uses moe_prefill_mt.hip's moepdn_impl / moep_store_h16_staged and
// gemm_fq.hip's fq_decode_sb / mx_inject.h. Every entry is OPT-IN (host: ops::chooseMoe with MoeFlags::moep_dhs /
// moep_fq; device::Profile rows 0 on both cards).
// SPDX-License-Identifier: Apache-2.0
//
// C  gemm_moepdhs_<v>_mxfp4 (gemm_moepdh arguments and grid): the f16 expert outputs stored through LDS with 16-byte
//    writes (strix-llama.cpp mmb.cu:115-148 mmb_store_tile). Bitwise == gemm_moepdh_<v> (the same (_Float16)acc values).
//    Balance item f16 intermediates, as gemm_moepdh.
// E  gemm_moepfq{gu,dn,dh,dhs,da}_<v>_mxfp4 (the arguments and grid of gemm_moepgu / dn / dh / dhs / da of variant v):
//    the expert GEMMs on the gemmfq main loop (kernels/gfx1151/gemm_fq.hip, KB feat026:fdq):
//      * the registers carry the RAW MXFP4 bytes of the next K chunk across the WMMA loop and decode them at the top of
//        the next iteration (the base kernels decode right after issuing the loads, BEFORE the WMMAs of the step, so
//        every K step waits for its global loads);
//      * one lane per 32-value sub-block (dec_group_h lanes g and g + 2 load the same 8 bytes);
//      * the exponent-injected decode (~2 VALU / value; LUT math for block exponents outside [104, 128]);
//      * rows / tokens past the end clamped (never stored; an output reads only its own row and token) instead of
//        guarded loads.
//    Bit-identity: the LDS A tile holds the dec_group_h<QT_MXFP4> values (fq_decode_sb: kt gemm 60/60 bitwise for
//    gemmfq, CPU testMxInject), the B tile the same f16 X rows, the WMMA k order and the epilogues are the base
//    kernels' - so each twin stores the bits of its base kernel (gemm_moepda: within f32 atomic reassociation, as the
//    base). Host requirement: ncols % 256 == 0 (whole MXFP4 super-blocks; Ornith E 2048, F 512).

// ---------------------------------------------------------------- C: coalesced f16 down epilogue
#define MOEPDHS_ENTRY(NAME, BM, WM, BN, BK)                                                                       \
    extern "C" __global__ void __launch_bounds__(256)                                                             \
        gemm_moepdhs_##NAME##_mxfp4(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X,          \
                                    _Float16 *Y, int ncols, const int4 *tiles, const int *n_tiles) {              \
        moepdn_impl<QT_MXFP4, BK, BM, WM, BN, 5>(W, row_bytes, rows_e, X, Y, ncols, tiles, n_tiles, nullptr,      \
                                                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0);       \
    }
MOEPDHS_ENTRY(m128w4n32k64, 128, 4, 32, 64)
MOEPDHS_ENTRY(m128w4n64k64, 128, 4, 64, 64)
MOEPDHS_ENTRY(m128w4n128k64, 128, 4, 128, 64)
MOEPDHS_ENTRY(m128w8n128k64, 128, 8, 128, 64)
MOEPDHS_ENTRY(m64w4n128k64, 64, 4, 128, 64)
MOEPDHS_ENTRY(m128w4n256k32, 128, 4, 256, 32)
#undef MOEPDHS_ENTRY

// ---------------------------------------------------------------- E: fused-dequant fetch (shared by both twins)
// A slab of BMA rows (row r -> its MXFP4 row pointer, clamped) and up to BNMAX tokens of X; per K chunk of BK values.
template <int BK, int BMA, int BNMAX>
struct MoepFqFetch {
    static constexpr int NTH = 256, SBR = BK / 32, GPR = BK / 8;
    static constexpr int A_SB = BMA * SBR / NTH, B_PER = BNMAX * GPR / NTH;
    static_assert(BK == 32 || BK == 64, "BK");
    static_assert(A_SB >= 1 && BMA * SBR % NTH == 0 && BNMAX * GPR % NTH == 0, "one lane per sub-block / tile threads");
    const uint8_t *arow[A_SB];
    const _Float16 *brow[B_PER];
    uint2 qa[A_SB], qb[A_SB];
    uint32_t qe[A_SB];
    half8 rb[B_PER];
    __device__ __forceinline__ void fetch(int k0) {
#pragma unroll
        for (int e = 0; e < A_SB; ++e) {
            const int s = threadIdx.x + e * NTH;
            const uint8_t *blk = arow[e] + (size_t)(k0 >> 8) * 136;
            const int ib = moep::fqIb(k0, moep::fqSb(s, BK));
            const uint8_t *q = blk + 8 + 16 * ib;
            qa[e] = *(const uint2 *)q;
            qb[e] = *(const uint2 *)(q + 8);
            qe[e] = blk[ib];
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) rb[e] = *(const half8 *)(brow[e] + k0);
    }
    // decode the carried bytes (BEFORE the K step's first block barrier, as gemmfq: the VALU overlaps the other waves'
    // WMMAs), then store both tiles between the two barriers; B rows past the last live fragment are never read, so
    // not stored (as the base kernels)
    __device__ __forceinline__ void decode(half8 (&ra)[A_SB][4]) const {
#pragma unroll
        for (int e = 0; e < A_SB; ++e) fq_decode_sb(qa[e], qb[e], qe[e], ra[e]);
    }
    template <int LD>
    __device__ __forceinline__ void store(const half8 (&ra)[A_SB][4], _Float16 (*As)[LD], _Float16 (*Bs)[LD], int live16) const {
#pragma unroll
        for (int e = 0; e < A_SB; ++e) {
            const int s = threadIdx.x + e * NTH, r = moep::fqRow(s, BK), sb = moep::fqSb(s, BK);
#pragma unroll
            for (int j = 0; j < 4; ++j) *(half8 *)&As[r][sb * 32 + j * 8] = ra[e][j];
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            if (i / GPR < live16) *(half8 *)&Bs[i / GPR][(i % GPR) * 8] = rb[e];
        }
    }
};

// ---------------------------------------------------------------- E: gate + up + SwiGLU (twin of moepgu_impl)
template <int BK, int GR, int BNMAX>
__device__ void moepfq_gu_impl(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, uint64_t row_bytes, int ff,
                               const _Float16 *__restrict__ Xall, _Float16 *__restrict__ Aout, int ncols,
                               const int4 *__restrict__ tiles, const int *__restrict__ n_tiles) {
    constexpr int NTH = 256, NW = NTH / 32;
    constexpr int WM = GR / 16, WN = NW / WM;
    constexpr int NF = BNMAX / 16, FPW = NF / WN;
    constexpr int BM = 2 * GR, LD = BK + 8;
    static_assert(GR % 16 == 0 && NW % WM == 0 && NF % WN == 0 && FPW >= 1, "wave layout");
    using Fetch = MoepFqFetch<BK, BM, BNMAX>;
    const int nrb = moep::nrb(ff, GR);
    const int ti = moep::blockTile((int)blockIdx.x, nrb), rb = moep::blockRb((int)blockIdx.x, nrb);
    if (ti >= *n_tiles) return;
    const int4 tl = tiles[ti];
    const int n_tok = tl.z, m0 = rb * GR;
    const int live = moep::liveFrags(n_tok), live16 = live * 16;
    const uint8_t *Wg = W1 + (size_t)tl.x * ff * row_bytes, *Wu = W2 + (size_t)tl.x * ff * row_bytes;
    const _Float16 *X = Xall + (size_t)tl.y * ncols;
    __shared__ _Float16 As[BM][LD];
    __shared__ _Float16 Bs[BNMAX][LD];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WM, wn = wid / WM;
    Fetch F;
#pragma unroll
    for (int e = 0; e < Fetch::A_SB; ++e) {
        const int r = moep::fqRow(threadIdx.x + e * NTH, BK);
        F.arow[e] = (r < GR ? Wg : Wu) + (size_t)min(m0 + (r % GR), ff - 1) * row_bytes;
    }
#pragma unroll
    for (int e = 0; e < Fetch::B_PER; ++e) {
        const int i = threadIdx.x + e * NTH;
        F.brow[e] = X + (size_t)min(i / Fetch::GPR, n_tok - 1) * ncols + (i % Fetch::GPR) * 8;
    }
    float8 acc[2][FPW];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < FPW; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    F.fetch(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        half8 ra[Fetch::A_SB][4];
        F.decode(ra);
        __syncthreads();
        F.template store<LD>(ra, As, Bs, live16);
        __syncthreads();
        if (k0 + BK < ncols) F.fetch(k0 + BK);
        if (wn >= live) continue;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            const half16 a0 = ld_frag16(&As[wm * 16 + lane % 16][kk]);
            const half16 a1 = ld_frag16(&As[GR + wm * 16 + lane % 16][kk]);
#pragma unroll
            for (int i = 0; i < FPW; ++i) {
                const int j = moep::frag(wn, WN, i);
                if (j < live) {
                    const half16 b = ld_frag16(&Bs[j * 16 + lane % 16][kk]);
                    acc[0][i] = WMMA16(a0, b, acc[0][i]);
                    acc[1][i] = WMMA16(a1, b, acc[1][i]);
                }
            }
        }
    }
    // the moepgu_impl epilogue, verbatim
#pragma unroll
    for (int i = 0; i < FPW; ++i) {
        const int j = moep::frag(wn, WN, i);
        const int t = j * 16 + lane % 16;
        if (j >= live || t >= n_tok) continue;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int row = m0 + wm * 16 + CROW(lane, v);
            if (row < ff) {
                const float g = acc[0][i][v], u = acc[1][i][v];
                Aout[(size_t)(tl.y + t) * ff + row] = (_Float16)(g / (1.f + __expf(-g)) * u);
            }
        }
    }
}

// ---------------------------------------------------------------- E: down (twin of moepdn_impl, EPI 0 / 2 / 4 / 5)
template <int BK, int BM, int WM, int BNMAX, int EPI>
__device__ void moepfq_dn_impl(const uint8_t *__restrict__ Wall, uint64_t row_bytes, int rows_e, const _Float16 *__restrict__ Xall,
                               void *__restrict__ Yv, int ncols, const int4 *__restrict__ tiles, const int *__restrict__ n_tiles,
                               const int *__restrict__ perm, const float *__restrict__ w, float *__restrict__ x, int K) {
    constexpr int NTH = 256, NW = NTH / 32;
    constexpr int WN = NW / WM, TM = BM / 16 / WM;
    constexpr int NF = BNMAX / 16, FPW = NF / WN;
    constexpr int LD = BK + 8;
    static_assert(NW % WM == 0 && BM % (16 * WM) == 0 && TM >= 1 && NF % WN == 0 && FPW >= 1, "wave layout");
    static_assert(EPI == 0 || EPI == 2 || EPI == 4 || EPI == 5, "epilogue (no fused-combine twins)");
    using Fetch = MoepFqFetch<BK, BM, BNMAX>;
    float *__restrict__ Y = (float *)Yv;
    _Float16 *__restrict__ Yh = (_Float16 *)Yv;
    const int nrb = moep::nrb(rows_e, BM);
    const int ti = moep::blockTile((int)blockIdx.x, nrb), rb = moep::blockRb((int)blockIdx.x, nrb);
    if (ti >= *n_tiles) return;
    const int4 tl = tiles[ti];
    const int n_tok = tl.z, m0 = rb * BM;
    const int live = moep::liveFrags(n_tok), live16 = live * 16;
    const uint8_t *W = Wall + (size_t)tl.x * rows_e * row_bytes;
    const _Float16 *X = Xall + (size_t)tl.y * ncols;
    __shared__ _Float16 As[BM][LD];
    __shared__ _Float16 Bs[BNMAX][LD];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WM, wn = wid / WM;
    Fetch F;
#pragma unroll
    for (int e = 0; e < Fetch::A_SB; ++e) F.arow[e] = W + (size_t)min(m0 + moep::fqRow(threadIdx.x + e * NTH, BK), rows_e - 1) * row_bytes;
#pragma unroll
    for (int e = 0; e < Fetch::B_PER; ++e) {
        const int i = threadIdx.x + e * NTH;
        F.brow[e] = X + (size_t)min(i / Fetch::GPR, n_tok - 1) * ncols + (i % Fetch::GPR) * 8;
    }
    float8 acc[TM][FPW];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < FPW; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    F.fetch(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        half8 ra[Fetch::A_SB][4];
        F.decode(ra);
        __syncthreads();
        F.template store<LD>(ra, As, Bs, live16);
        __syncthreads();
        if (k0 + BK < ncols) F.fetch(k0 + BK);
        if (wn >= live) continue;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            half16 a[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = ld_frag16(&As[(wm * TM + i) * 16 + lane % 16][kk]);
#pragma unroll
            for (int f = 0; f < FPW; ++f) {
                const int j = moep::frag(wn, WN, f);
                if (j < live) {
                    const half16 b = ld_frag16(&Bs[j * 16 + lane % 16][kk]);
#pragma unroll
                    for (int i = 0; i < TM; ++i) acc[i][f] = WMMA16(a[i], b, acc[i][f]);
                }
            }
        }
    }
    if constexpr (EPI == 5) {
        static_assert(NW * 16 * (TM * 16 + 8) <= BM * LD, "the staging slices fit in As");
        __syncthreads();
        moep_store_h16_staged<TM, FPW, WN>(acc, &As[0][0] + wid * 16 * (TM * 16 + 8), Yh, rows_e, m0, wm, wn, live, n_tok, tl.y, lane);
    } else {
        // the moepdn_impl epilogue (EPI 0 / 2 / 4), verbatim
#pragma unroll
        for (int f = 0; f < FPW; ++f) {
            const int j = moep::frag(wn, WN, f);
            const int t = j * 16 + lane % 16;
            if (j >= live || t >= n_tok) continue;
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                if constexpr (EPI == 4) {
                    const int p = perm[tl.y + t], tok = p / K;
                    const float wp = w[p];
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int row = m0 + (wm * TM + i) * 16 + CROW(lane, v);
                        if (row < rows_e) unsafeAtomicAdd(&x[(size_t)tok * rows_e + row], wp * acc[i][f][v]);
                    }
                } else {
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int row = m0 + (wm * TM + i) * 16 + CROW(lane, v);
                        if (row < rows_e) {
                            if constexpr (EPI == 2) Yh[(size_t)(tl.y + t) * rows_e + row] = (_Float16)acc[i][f][v];
                            else Y[(size_t)(tl.y + t) * rows_e + row] = acc[i][f][v];
                        }
                    }
                }
            }
        }
    }
}

// Entry points (names = kernels::kMoepGu / kMoepDn variant names; only the variants with one lane per sub-block:
// 2 * GR * BK / 32 and BM * BK / 32 multiples of 256). Host: kernels::MoeKernels::moep_fq_* (null for the others).
#define MOEPFQGU_ENTRY(NAME, GR, BN, BK)                                                                            \
    extern "C" __global__ void __launch_bounds__(256)                                                               \
        gemm_moepfqgu_##NAME##_mxfp4(const uint8_t *W1, const uint8_t *W2, uint64_t row_bytes, int ff,              \
                                     const _Float16 *X, _Float16 *A, int ncols, const int4 *tiles, const int *n_tiles) { \
        moepfq_gu_impl<BK, GR, BN>(W1, W2, row_bytes, ff, X, A, ncols, tiles, n_tiles);                             \
    }
#define MOEPFQDN_ENTRY(NAME, BM, WM, BN, BK)                                                                        \
    extern "C" __global__ void __launch_bounds__(256)                                                               \
        gemm_moepfqdn_##NAME##_mxfp4(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X, float *Y,  \
                                     int ncols, const int4 *tiles, const int *n_tiles) {                            \
        moepfq_dn_impl<BK, BM, WM, BN, 0>(W, row_bytes, rows_e, X, Y, ncols, tiles, n_tiles, nullptr, nullptr, nullptr, 0); \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(256)                                                               \
        gemm_moepfqdh_##NAME##_mxfp4(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X,           \
                                     _Float16 *Y, int ncols, const int4 *tiles, const int *n_tiles) {               \
        moepfq_dn_impl<BK, BM, WM, BN, 2>(W, row_bytes, rows_e, X, Y, ncols, tiles, n_tiles, nullptr, nullptr, nullptr, 0); \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(256)                                                               \
        gemm_moepfqdhs_##NAME##_mxfp4(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X,          \
                                      _Float16 *Y, int ncols, const int4 *tiles, const int *n_tiles) {              \
        moepfq_dn_impl<BK, BM, WM, BN, 5>(W, row_bytes, rows_e, X, Y, ncols, tiles, n_tiles, nullptr, nullptr, nullptr, 0); \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(256)                                                               \
        gemm_moepfqda_##NAME##_mxfp4(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X,           \
                                     int ncols, const int4 *tiles, const int *n_tiles, const int *perm,             \
                                     const float *w, float *x, int K) {                                             \
        moepfq_dn_impl<BK, BM, WM, BN, 4>(W, row_bytes, rows_e, X, nullptr, ncols, tiles, n_tiles, perm, w, x, K);   \
    }
MOEPFQGU_ENTRY(g64n64k64, 64, 64, 64)
MOEPFQGU_ENTRY(g64n128k64, 64, 128, 64)
MOEPFQGU_ENTRY(g128n128k32, 128, 128, 32)
MOEPFQDN_ENTRY(m128w4n32k64, 128, 4, 32, 64)
MOEPFQDN_ENTRY(m128w4n64k64, 128, 4, 64, 64)
MOEPFQDN_ENTRY(m128w4n128k64, 128, 4, 128, 64)
MOEPFQDN_ENTRY(m128w8n128k64, 128, 8, 128, 64)
#undef MOEPFQGU_ENTRY
#undef MOEPFQDN_ENTRY
