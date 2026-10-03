// WHIRL device code (gfx1151 / RDNA 3.5): Q4_K / Q5_K / IQ4_XS / MXFP4 GEMV units.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// gfx1151 GEMV for Q4_K / Q5_K / IQ4_XS (91% of the 27B's weight bytes).
// On RDNA 3.5 v_dot4 runs at half the int8 WMMA rate and every extra token
// of the dp4 multi-token GEMV re-reads x through L0 per row, so verify rows
// cost ~9% each. Here the multi-token kernels (2..16 tokens) run the integer
// dots on v_wmma_i32_16x16x16_iu8 with A = x (16 tokens) and B = the decoded
// weights (16 rows): x is loaded once per 16 rows.
//
// All variants (1 token on dp4, 2..16 tokens on WMMA, 8/16/32 rows per block)
// share one arithmetic per (row, token), so MTP verify equals plain decode
// bitwise:
//   * unit u = 64 weights (two 32-value sub-blocks, u = 4 * superblock + j);
//     slice s = u % 32 sums its units in increasing u with wfold() (exact
//     int dot per sub-block, then the same float ops);
//   * the 32 slice sums are combined as G_g = (s_4g + s_4g+1) + (s_4g+2 + s_4g+3),
//     y = ((G0 + G1) + (G2 + G3)) + ((G4 + G5) + (G6 + G7))
//     (a butterfly with xor offsets 1, 2, 4, 8, 16).
// The WMMA kernels give wave w the slices 4w .. 4w+3, i.e. whole superblocks
// b = w (mod 8): one 144..176-byte stretch per row per iteration.
// ---------------------------------------------------------------------------
typedef int int4v __attribute__((ext_vector_type(4)));
typedef int int8v __attribute__((ext_vector_type(8)));

// One decoded wide unit of one row: w[0..7] = first sub-block (values 0..31,
// 4 per int, in x order), w[8..15] = second; scales s, mins m (0 for IQ4_XS).
struct WUnit {
    int w[16];
    float s0, s1, m0, m1;
};

template <int T> __device__ __forceinline__ void wunit(const uint8_t *row, int u, WUnit &o);

template <> __device__ __forceinline__ void wunit<QT_Q4_K>(const uint8_t *row, int u, WUnit &o) {
    const int b = u >> 2, j = u & 3;
    const uint8_t *blk = row + (size_t)b * 144;
    const uint32_t dm = *(const uint32_t *)blk;
    const float d = __half2float(__ushort_as_half((uint16_t)(dm & 0xFFFF)));
    const float dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
    int s0, m0, s1, m1;
    scale_min_k4(2 * j, blk + 4, s0, m0);
    scale_min_k4(2 * j + 1, blk + 4, s1, m1);
    o.s0 = d * s0;
    o.s1 = d * s1;
    o.m0 = dmin * m0;
    o.m1 = dmin * m1;
    const uint4 qa = *(const uint4 *)(blk + 16 + 32 * j);
    const uint4 qb = *(const uint4 *)(blk + 32 + 32 * j);
    const uint32_t q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        o.w[k] = q[k] & 0x0F0F0F0F;
        o.w[8 + k] = (q[k] >> 4) & 0x0F0F0F0F;
    }
}

template <> __device__ __forceinline__ void wunit<QT_Q5_K>(const uint8_t *row, int u, WUnit &o) {
    const int b = u >> 2, j = u & 3;
    const uint8_t *blk = row + (size_t)b * 176;
    const uint32_t dm = *(const uint32_t *)blk;
    const float d = __half2float(__ushort_as_half((uint16_t)(dm & 0xFFFF)));
    const float dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
    int s0, m0, s1, m1;
    scale_min_k4(2 * j, blk + 4, s0, m0);
    scale_min_k4(2 * j + 1, blk + 4, s1, m1);
    o.s0 = d * s0;
    o.s1 = d * s1;
    o.m0 = dmin * m0;
    o.m1 = dmin * m1;
    const uint4 qa = *(const uint4 *)(blk + 48 + 32 * j);
    const uint4 qb = *(const uint4 *)(blk + 64 + 32 * j);
    const uint4 ha = *(const uint4 *)(blk + 16);
    const uint4 hb = *(const uint4 *)(blk + 32);
    const uint32_t q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
    const uint32_t h[8] = {ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        o.w[k] = (q[k] & 0x0F0F0F0F) | (((h[k] >> (2 * j)) & 0x01010101) << 4);
        o.w[8 + k] = ((q[k] >> 4) & 0x0F0F0F0F) | (((h[k] >> (2 * j + 1)) & 0x01010101) << 4);
    }
}

template <> __device__ __forceinline__ void wunit<QT_IQ4_XS>(const uint8_t *row, int u, WUnit &o) {
    const int b = u >> 2, pp = u & 3;
    const uint8_t *blk = row + (size_t)b * 136;
    const uint32_t hdr = *(const uint32_t *)blk;
    const float d = __half2float(__ushort_as_half((uint16_t)(hdr & 0xFFFF)));
    const uint32_t sh = hdr >> 16;
    const int ib0 = 2 * pp, ib1 = 2 * pp + 1;
    const int ls0 = (blk[4 + pp] & 0xF) | (((sh >> (2 * ib0)) & 3) << 4);
    const int ls1 = (blk[4 + pp] >> 4) | (((sh >> (2 * ib1)) & 3) << 4);
    o.s0 = d * (ls0 - 32);
    o.s1 = d * (ls1 - 32);
    o.m0 = 0.f;
    o.m1 = 0.f;
    const uint4 qa = *(const uint4 *)(blk + 8 + 32 * pp);
    const uint4 qb = *(const uint4 *)(blk + 24 + 32 * pp);
    const uint32_t a[4] = {qa.x, qa.y, qa.z, qa.w}, c[4] = {qb.x, qb.y, qb.z, qb.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        o.w[k] = iq4_lut4(a[k] & 0x0F0F0F0F);             // sub-block 2pp values 0..15
        o.w[4 + k] = iq4_lut4((a[k] >> 4) & 0x0F0F0F0F);  // 16..31
        o.w[8 + k] = iq4_lut4(c[k] & 0x0F0F0F0F);         // sub-block 2pp+1
        o.w[12 + k] = iq4_lut4((c[k] >> 4) & 0x0F0F0F0F);
    }
}

template <> __device__ __forceinline__ void wunit<QT_MXFP4>(const uint8_t *row, int u, WUnit &o) {
    const int b = u >> 2, pp = u & 3;
    const uint8_t *blk = row + (size_t)b * 136;
    const uint32_t ee = *(const uint16_t *)(blk + 2 * pp);
    o.s0 = e8m0_half(ee & 0xFF);
    o.s1 = e8m0_half(ee >> 8);
    o.m0 = 0.f;
    o.m1 = 0.f;
    const uint4 qa = *(const uint4 *)(blk + 8 + 32 * pp);
    const uint4 qb = *(const uint4 *)(blk + 24 + 32 * pp);
    const uint32_t a[4] = {qa.x, qa.y, qa.z, qa.w}, c[4] = {qb.x, qb.y, qb.z, qb.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        o.w[k] = mx_lut4(a[k] & 0x0F0F0F0F);
        o.w[4 + k] = mx_lut4((a[k] >> 4) & 0x0F0F0F0F);
        o.w[8 + k] = mx_lut4(c[k] & 0x0F0F0F0F);
        o.w[12 + k] = mx_lut4((c[k] >> 4) & 0x0F0F0F0F);
    }
}

// Half of a wide unit for the WMMA kernels: sub-block hs (0 / 1) of unit u
// as 8 ints (values 0..31 of that sub-block, 4 per int, x order) plus both
// sub-blocks' scales (same expressions as wunit). SPLIT_T: types that decode
// per half-wave (the decode is heavy enough to pay for the lane swap).
#ifndef GFX1151_SPLIT_Q5K
#define GFX1151_SPLIT_Q5K true
#endif
template <int T> struct SplitDec { static constexpr bool on = false; };
template <> struct SplitDec<QT_IQ4_XS> { static constexpr bool on = true; };
template <> struct SplitDec<QT_MXFP4> { static constexpr bool on = true; };
template <> struct SplitDec<QT_Q5_K> { static constexpr bool on = GFX1151_SPLIT_Q5K; };

template <int T> __device__ __forceinline__ void wunit_half(const uint8_t *row, int u, int hs, int *w, WUnit &o);

template <> __device__ __forceinline__ void wunit_half<QT_IQ4_XS>(const uint8_t *row, int u, int hs, int *w, WUnit &o) {
    const int b = u >> 2, pp = u & 3;
    const uint8_t *blk = row + (size_t)b * 136;
    const uint32_t hdr = *(const uint32_t *)blk;
    const float d = __half2float(__ushort_as_half((uint16_t)(hdr & 0xFFFF)));
    const uint32_t sh = hdr >> 16;
    const int ib0 = 2 * pp, ib1 = 2 * pp + 1;
    const int ls0 = (blk[4 + pp] & 0xF) | (((sh >> (2 * ib0)) & 3) << 4);
    const int ls1 = (blk[4 + pp] >> 4) | (((sh >> (2 * ib1)) & 3) << 4);
    o.s0 = d * (ls0 - 32);
    o.s1 = d * (ls1 - 32);
    o.m0 = 0.f;
    o.m1 = 0.f;
    const uint4 qa = *(const uint4 *)(blk + 8 + 32 * pp + 16 * hs);
    const uint32_t a[4] = {qa.x, qa.y, qa.z, qa.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        w[k] = iq4_lut4(a[k] & 0x0F0F0F0F);
        w[4 + k] = iq4_lut4((a[k] >> 4) & 0x0F0F0F0F);
    }
}

template <> __device__ __forceinline__ void wunit_half<QT_MXFP4>(const uint8_t *row, int u, int hs, int *w, WUnit &o) {
    const int b = u >> 2, pp = u & 3;
    const uint8_t *blk = row + (size_t)b * 136;
    const uint32_t ee = *(const uint16_t *)(blk + 2 * pp);
    o.s0 = e8m0_half(ee & 0xFF);
    o.s1 = e8m0_half(ee >> 8);
    o.m0 = 0.f;
    o.m1 = 0.f;
    const uint4 qa = *(const uint4 *)(blk + 8 + 32 * pp + 16 * hs);
    const uint32_t a[4] = {qa.x, qa.y, qa.z, qa.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        w[k] = mx_lut4(a[k] & 0x0F0F0F0F);
        w[4 + k] = mx_lut4((a[k] >> 4) & 0x0F0F0F0F);
    }
}

template <> __device__ __forceinline__ void wunit_half<QT_Q5_K>(const uint8_t *row, int u, int hs, int *w, WUnit &o) {
    const int b = u >> 2, j = u & 3;
    const uint8_t *blk = row + (size_t)b * 176;
    const uint32_t dm = *(const uint32_t *)blk;
    const float d = __half2float(__ushort_as_half((uint16_t)(dm & 0xFFFF)));
    const float dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
    int s0, m0, s1, m1;
    scale_min_k4(2 * j, blk + 4, s0, m0);
    scale_min_k4(2 * j + 1, blk + 4, s1, m1);
    o.s0 = d * s0;
    o.s1 = d * s1;
    o.m0 = dmin * m0;
    o.m1 = dmin * m1;
    const uint4 qa = *(const uint4 *)(blk + 48 + 32 * j);
    const uint4 qb = *(const uint4 *)(blk + 64 + 32 * j);
    const uint4 ha = *(const uint4 *)(blk + 16);
    const uint4 hb = *(const uint4 *)(blk + 32);
    const uint32_t q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
    const uint32_t h[8] = {ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w};
    const int qs = 4 * hs, hsh = 2 * j + hs;
#pragma unroll
    for (int k = 0; k < 8; ++k) w[k] = ((q[k] >> qs) & 0x0F0F0F0F) | (((h[k] >> hsh) & 0x01010101) << 4);
}

// acc + d_x * (s * dot - m * sum_x), with explicit fmas so every kernel
// rounds identically.
__device__ __forceinline__ float slice_sum(float v) {  // xor offsets 1, 2, 4, 8, 16
#pragma unroll
    for (int o = 1; o < WARP; o <<= 1) v += __shfl_xor(v, o, WARP);
    return v;
}
__device__ __forceinline__ float wfold(float acc, uint32_t xp, float s, float m, int ci) {
    return __builtin_fmaf(pk_d(xp), __builtin_fmaf(s, (float)ci, -(m * (float)pk_s(xp))), acc);
}

// One token, one wave per row (8 rows per 256-thread block), lane = slice.
template <int T>
__device__ void gemv1s_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                            const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                            const int *__restrict__ skip, int bid = -1) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = (bid < 0 ? (int)blockIdx.x : bid) * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const uint32_t *xp = (const uint32_t *)xd;
    const int nu = ncols / 256 * 4;
    float acc = 0.f;
    for (int u = lane; u < nu; u += WARP) {
        WUnit wu;
        wunit<T>(wr, u, wu);
        const int base = (u >> 2) * 256 + 64 * (u & 3);
        const int4 a0 = *(const int4 *)(xq + base), a1 = *(const int4 *)(xq + base + 16);
        const int4 c0 = *(const int4 *)(xq + base + 32), c1 = *(const int4 *)(xq + base + 48);
        const int xa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const int xc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        int dl = 0, dh = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            dl = dp4(wu.w[k], xa[k], dl);
            dh = dp4(wu.w[8 + k], xc[k], dh);
        }
        const uint2 p = *(const uint2 *)(xp + (base >> 5));
        acc = wfold(acc, p.x, wu.s0, wu.m0, dl);
        acc = wfold(acc, p.y, wu.s1, wu.m1, dh);
    }
    acc = slice_sum(acc);
    if (lane == 0) y[row] = accumulate ? y[row] + acc : acc;
}

// NT tokens (2..16) on int8 WMMA. Block = ROWS rows: HALF -> 8 (tile half
// used), else TILES x 16 (TILES tiles share each x load). Wave w runs the
// superblocks b = w (mod 8), i.e. slices 4w .. 4w+3.
template <int T, int NT, int TILES, bool HALF>
__device__ void gemvw_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
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
    float acc[TILES][4][NTV];
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int v = 0; v < NTV; ++v) acc[tt][q][v] = 0.f;
    const int8v zero = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int sb = w; sb < nsb; sb += 8) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int u = 4 * sb + q;
            {
                const int base = sb * 256 + 64 * q, xi = base >> 5;
                int4v x0 = {0, 0, 0, 0}, x1 = x0, x2 = x0, x3 = x0;
                if (tok_ok) {
                    x0 = *(const int4v *)(xt + base);
                    x1 = *(const int4v *)(xt + base + 16);
                    x2 = *(const int4v *)(xt + base + 32);
                    x3 = *(const int4v *)(xt + base + 48);
                }
#pragma unroll
                for (int tt = 0; tt < TILES; ++tt) {
                    WUnit wu;
                    if constexpr (SplitDec<T>::on) {
                        // half-wave h decodes sub-block h; the other comes over lane ^ 16
                        int own[8];
                        wunit_half<T>(wr[tt], u, h, own, wu);
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            const int oth = xhalf(own[k]);
                            wu.w[k] = h ? oth : own[k];
                            wu.w[8 + k] = h ? own[k] : oth;
                        }
                    } else {
                        wunit<T>(wr[tt], u, wu);
                    }
                    const int4v b0 = {wu.w[0], wu.w[1], wu.w[2], wu.w[3]}, b1 = {wu.w[4], wu.w[5], wu.w[6], wu.w[7]};
                    const int4v b2 = {wu.w[8], wu.w[9], wu.w[10], wu.w[11]}, b3 = {wu.w[12], wu.w[13], wu.w[14], wu.w[15]};
                    int8v c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, x0, true, b0, zero, false);
                    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, x1, true, b1, c0, false);
                    int8v c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, x2, true, b2, zero, false);
                    c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, x3, true, b3, c1, false);
#pragma unroll
                    for (int v = 0; v < NTV; ++v) {
                        // no divergence for odd NT: the spare row folds a
                        // valid token's scales and is never stored
                        const int t = min(2 * v + h, NT - 1);
                        const uint2 p = *(const uint2 *)(xp + (size_t)t * nb32 + xi);
                        acc[tt][q][v] = wfold(wfold(acc[tt][q][v], p.x, wu.s0, wu.m0, c0[v]), p.y, wu.s1, wu.m1, c1[v]);
                    }
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
