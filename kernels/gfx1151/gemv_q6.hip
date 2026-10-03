// WHIRL device code (gfx1151 / RDNA 3.5): Q6_K GEMV family.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// gfx1151 Q6_K GEMV family (output head, MTP layer, some layer weights), same
// canonical order as the Q4_K family above: unit u = one 32-value block,
// u = 8 * superblock + 4 * n + quarter (values 128n + 32 quarter + [0, 32)),
// slice = u % 32, wfold6 per unit, G-tree over the slices. WMMA wave w runs
// half n = w & 1 of the superblocks b = w / 2 (mod 4), i.e. slices 4w .. 4w+3.
// Values are fed as signed int8 (q - 32), so no x sums are needed.
// ---------------------------------------------------------------------------
struct WUnit6 {
    int w[8];
    float s0, s1;
};

__device__ __forceinline__ uint4 ld128a2(const uint8_t *p) {  // 2-byte aligned 16-byte load
    uint4 v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 16);
    return v;
}

__device__ __forceinline__ void wunit6(const uint8_t *row, int u, WUnit6 &o) {
    const int b = u >> 3, hn = (u >> 2) & 1, qq = u & 3;
    const uint8_t *blk = row + (size_t)b * 210;
    const uint4 l0 = ld128a2(blk + 64 * hn + 32 * (qq & 1)), l1 = ld128a2(blk + 64 * hn + 32 * (qq & 1) + 16);
    const uint4 h0 = ld128a2(blk + 128 + 32 * hn), h1 = ld128a2(blk + 128 + 32 * hn + 16);
    const float d = h2f(blk + 208);
    const int8_t *sc = (const int8_t *)(blk + 192) + 8 * hn + 2 * qq;
    o.s0 = d * (float)sc[0];
    o.s1 = d * (float)sc[1];
    const int sl = 4 * (qq >> 1), sh = 2 * qq;
    const uint32_t L[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
    const uint32_t H[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t v = ((L[k] >> sl) & 0x0F0F0F0Fu) | (((H[k] >> sh) & 0x03030303u) << 4);
        o.w[k] = (int)(((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u);  // bytewise v - 32
    }
}

__device__ __forceinline__ float wfold6(float acc, uint32_t xp, float s0, float s1, int c0, int c1) {
    return __builtin_fmaf(pk_d(xp), __builtin_fmaf(s1, (float)c1, s0 * (float)c0), acc);
}

__device__ void gemv1s6_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                             const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                             const int *__restrict__ skip, int bid = -1) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = (bid < 0 ? (int)blockIdx.x : bid) * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const uint32_t *xp = (const uint32_t *)xd;
    const int nu = ncols / 256 * 8;
    float acc = 0.f;
    for (int u = lane; u < nu; u += WARP) {
        WUnit6 wu;
        wunit6(wr, u, wu);
        const int base = (u >> 3) * 256 + 128 * ((u >> 2) & 1) + 32 * (u & 3);
        const int4 a0 = *(const int4 *)(xq + base), a1 = *(const int4 *)(xq + base + 16);
        const int xa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        int c0 = 0, c1 = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            c0 = dp4(wu.w[k], xa[k], c0);
            c1 = dp4(wu.w[4 + k], xa[4 + k], c1);
        }
        acc = wfold6(acc, xp[base >> 5], wu.s0, wu.s1, c0, c1);
    }
    acc = slice_sum(acc);
    if (lane == 0) y[row] = accumulate ? y[row] + acc : acc;
}

template <int NT, int TILES, bool HALF>
__device__ void gemvw6_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                            const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                            const int *__restrict__ skip) {
    constexpr int ROWS = HALF ? 8 : 16 * TILES;
    constexpr int NTV = (NT + 1) / 2;
    __shared__ float red[8][NT][ROWS];
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, h = lane >> 4, n = lane & 15;
    const int r0 = blockIdx.x * ROWS;
    const uint8_t *wr[TILES];
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt) wr[tt] = W + (size_t)min(r0 + tt * 16 + (HALF ? (n & 7) : n), nrows - 1) * row_bytes;
    const bool tok_ok = n < NT;
    const int8_t *xt = xq + (size_t)(tok_ok ? n : 0) * ncols;
    const uint32_t *xp = (const uint32_t *)xd;
    const int nb32 = ncols / 32;
    const int nsb = ncols / 256;
    const int hn = w & 1;
    float acc[TILES][4][NTV];
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int v = 0; v < NTV; ++v) acc[tt][q][v] = 0.f;
    const int8v zero = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int sb = w >> 1; sb < nsb; sb += 4) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int u = 8 * sb + 4 * hn + q;
            const int base = sb * 256 + 128 * hn + 32 * q, xi = base >> 5;
            int4v x0 = {0, 0, 0, 0}, x1 = x0;
            if (tok_ok) {
                x0 = *(const int4v *)(xt + base);
                x1 = *(const int4v *)(xt + base + 16);
            }
#pragma unroll
            for (int tt = 0; tt < TILES; ++tt) {
                WUnit6 wu;
                wunit6(wr[tt], u, wu);
                const int4v b0 = {wu.w[0], wu.w[1], wu.w[2], wu.w[3]}, b1 = {wu.w[4], wu.w[5], wu.w[6], wu.w[7]};
                const int8v c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, x0, true, b0, zero, false);
                const int8v c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, x1, true, b1, zero, false);
#pragma unroll
                for (int v = 0; v < NTV; ++v) {
                    const int t = min(2 * v + h, NT - 1);
                    acc[tt][q][v] = wfold6(acc[tt][q][v], xp[(size_t)t * nb32 + xi], wu.s0, wu.s1, c0[v], c1[v]);
                }
            }
        }
    }
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
        for (int v = 0; v < NTV; ++v) {
            const int t = 2 * v + h;
            if (t < NT && (!HALF || n < 8)) red[w][t][tt * 16 + n] = (acc[tt][0][v] + acc[tt][1][v]) + (acc[tt][2][v] + acc[tt][3][v]);
        }
    __syncthreads();
    for (int i = threadIdx.x; i < NT * ROWS; i += blockDim.x) {
        const int t = i / ROWS, nn = i % ROWS, row = r0 + nn;
        const float s = ((red[0][t][nn] + red[1][t][nn]) + (red[2][t][nn] + red[3][t][nn])) +
                        ((red[4][t][nn] + red[5][t][nn]) + (red[6][t][nn] + red[7][t][nn]));
        if (row < nrows) {
            float *dst = y + (size_t)t * nrows + row;
            *dst = accumulate ? *dst + s : s;
        }
    }
}

template <int T, int NT, int R>
__device__ void gemvq_mr_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                              const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                              const int *__restrict__ skip) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row0 = (blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP) * R;
    if (row0 >= nrows) return;
    const uint8_t *wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wr[r] = W + (size_t)min(row0 + r, nrows - 1) * row_bytes;
    const int nu = ncols / 256 * MRD<T>::per_sb;
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.f;
    for (int u = lane; u < nu; u += WARP) MRD<T>::template step<NT, R>(wr, u, xq, xd, ncols, acc);
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0 && row0 + r < nrows) {
                float *dst = y + (size_t)t * nrows + row0 + r;
                *dst = accumulate ? *dst + v : v;
            }
        }
    }
}

#define GEMVQ_ENTRY(SUF, T)                                                                      \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_##SUF(GV_PARAMS) {                                                                 \
        const GvSel gs = gv_sel(g);                                                              \
        gemvq_impl<T>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b);       \
    }
#define GEMVQ_NT_ENTRY(NT, SUF, T)                                                               \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_nt##NT##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,            \
                             const float *xd, float *y, int ncols, int nrows, int accumulate,   \
                             const int *skip) {                                                  \
        gemvq_nt_impl<T, NT>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip);           \
    }
#define GEMVQ_W_ENTRY(SUF, T)                                                                    \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq, const float *xd,    \
                    float *y, int ncols, int nrows, int accumulate, const int *skip) {          \
        gemvq_mr_impl<T, 1, 1>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip);        \
    }
#define GEMVQ_W_NT_ENTRY(NT, SUF, T)                                                             \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_nt##NT##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,            \
                             const float *xd, float *y, int ncols, int nrows, int accumulate,   \
                             const int *skip) {                                                  \
        gemvq_mr_impl<T, NT, 1>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip);        \
    }
// gfx1151: Q4_K / Q5_K / IQ4_XS use the slice-ordered dp4 (1 token) and
// WMMA (2..16 tokens) kernels above; R=1 entries run a half-used 16-row tile.
#define GEMV1S_ENTRY(SUF, T)                                                                     \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_##SUF(GV_PARAMS) {                                                                 \
        const GvSel gs = gv_sel(g);                                                              \
        gemv1s_impl<T>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b);      \
    }
#define GEMVW_ENTRY(NAME, T, NT, TILES, HALF)                                                    \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        NAME(const uint8_t *W, uint64_t row_bytes, const int8_t *xq, const float *xd, float *y, \
             int ncols, int nrows, int accumulate, const int *skip) {                            \
        gemvw_impl<T, NT, TILES, HALF>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip); \
    }
#define GEMVW6_ENTRY(NAME, NT, TILES, HALF)                                                      \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        NAME(const uint8_t *W, uint64_t row_bytes, const int8_t *xq, const float *xd, float *y, \
             int ncols, int nrows, int accumulate, const int *skip) {                            \
        gemvw6_impl<NT, TILES, HALF>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip);   \
    }
#define GEMVQ_NT_TYPES(NT)                \
    GEMVW_ENTRY(gemvq_nt##NT##_q4_k, QT_Q4_K, NT, 1, true)     \
    GEMVW_ENTRY(gemvq_nt##NT##_q5_k, QT_Q5_K, NT, 1, true)     \
    GEMVW_ENTRY(gemvq_nt##NT##_iq4_xs, QT_IQ4_XS, NT, 1, true) \
    GEMVW_ENTRY(gemvq_nt##NT##_mxfp4, QT_MXFP4, NT, 1, true) \
    GEMVW6_ENTRY(gemvq_nt##NT##_q6_k, NT, 1, true) \
    GEMVQ_NT_ENTRY(NT, iq4_nl, QT_IQ4_NL) \
    GEMVQ_NT_ENTRY(NT, q8_0, QT_Q8_0)     \
    GEMVQ_NT_ENTRY(NT, q3_k, QT_Q3_K)     \
    GEMVQ_NT_ENTRY(NT, iq3_s, QT_IQ3_S)

GEMV1S_ENTRY(q4_k, QT_Q4_K)
GEMV1S_ENTRY(q5_k, QT_Q5_K)
GEMV1S_ENTRY(iq4_xs, QT_IQ4_XS)
GEMV1S_ENTRY(mxfp4, QT_MXFP4)
extern "C" __global__ void __launch_bounds__(256)
    gemvq_q6_k(GV_PARAMS) {
    const GvSel gs = gv_sel(g);
    gemv1s6_impl(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b);
}
GEMVQ_ENTRY(iq4_nl, QT_IQ4_NL)
GEMVQ_ENTRY(q8_0, QT_Q8_0)
GEMVQ_ENTRY(q3_k, QT_Q3_K)
GEMVQ_ENTRY(iq3_s, QT_IQ3_S)
GEMVQ_NT_TYPES(2)
GEMVQ_NT_TYPES(3)
GEMVQ_NT_TYPES(4)
GEMVQ_NT_TYPES(5)
GEMVQ_NT_TYPES(6)
GEMVQ_NT_TYPES(7)
GEMVQ_NT_TYPES(8)
GEMVQ_NT_TYPES(9)
GEMVQ_NT_TYPES(10)
GEMVQ_NT_TYPES(11)
GEMVQ_NT_TYPES(12)
GEMVQ_NT_TYPES(13)
GEMVQ_NT_TYPES(14)
GEMVQ_NT_TYPES(15)
GEMVQ_NT_TYPES(16)

#define GEMVQ_MR_ENTRY(NT, R, SUF, T)                                                            \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_nt##NT##r##R##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,      \
                                   const float *xd, float *y, int ncols, int nrows,              \
                                   int accumulate, const int *skip) {                            \
        gemvq_mr_impl<T, NT, R>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip);        \
    }
// R = 2 -> one 16-row WMMA tile per block, R = 4 -> two tiles sharing x
#define GEMVQ_MR_TYPES(NT, R)                    \
    GEMVW_ENTRY(gemvq_nt##NT##r##R##_q4_k, QT_Q4_K, NT, R / 2, false)     \
    GEMVW_ENTRY(gemvq_nt##NT##r##R##_q5_k, QT_Q5_K, NT, R / 2, false)     \
    GEMVW_ENTRY(gemvq_nt##NT##r##R##_iq4_xs, QT_IQ4_XS, NT, R / 2, false) \
    GEMVW_ENTRY(gemvq_nt##NT##r##R##_mxfp4, QT_MXFP4, NT, R / 2, false) \
    GEMVW6_ENTRY(gemvq_nt##NT##r##R##_q6_k, NT, R / 2, false)
GEMVQ_MR_TYPES(2, 2)
GEMVQ_MR_TYPES(2, 4)
GEMVQ_MR_TYPES(3, 2)
GEMVQ_MR_TYPES(3, 4)
GEMVQ_MR_TYPES(4, 2)
GEMVQ_MR_TYPES(4, 4)
GEMVQ_MR_TYPES(5, 2)
GEMVQ_MR_TYPES(5, 4)
GEMVQ_MR_TYPES(6, 2)
GEMVQ_MR_TYPES(6, 4)
GEMVQ_MR_TYPES(7, 2)
GEMVQ_MR_TYPES(7, 4)
GEMVQ_MR_TYPES(8, 2)
GEMVQ_MR_TYPES(8, 4)
GEMVQ_MR_TYPES(9, 2)
GEMVQ_MR_TYPES(9, 4)
GEMVQ_MR_TYPES(10, 2)
GEMVQ_MR_TYPES(10, 4)
GEMVQ_MR_TYPES(11, 2)
GEMVQ_MR_TYPES(11, 4)
GEMVQ_MR_TYPES(12, 2)
GEMVQ_MR_TYPES(12, 4)
GEMVQ_MR_TYPES(13, 2)
GEMVQ_MR_TYPES(13, 4)
GEMVQ_MR_TYPES(14, 2)
GEMVQ_MR_TYPES(14, 4)
GEMVQ_MR_TYPES(15, 2)
GEMVQ_MR_TYPES(15, 4)
GEMVQ_MR_TYPES(16, 2)
GEMVQ_MR_TYPES(16, 4)
