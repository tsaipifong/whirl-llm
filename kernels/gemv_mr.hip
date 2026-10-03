// WHIRL device code (gfx1201 / RDNA 4): multi-row multi-token int8 GEMV (gemvq_nt*r*).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Multi-row multi-token int8 GEMV (verify / batched decode, 2..16 tokens):
// each wave owns R rows, so every int8 x unit it loads (the bottleneck once
// NT grows) serves R rows, and row-independent terms (x sums) are computed
// once. Per (row, token) the arithmetic is exactly that of gemvq_impl /
// gemvq_nt_impl (same lane -> unit mapping, same float expressions in the
// same order, same warp_sum), so the results match them bitwise.
// ---------------------------------------------------------------------------
template <int T> struct MRD;

// Q4_K / Q5_K wide unit = (superblock b, pair j): the 32 bytes qs[32j..32j+32]
// hold sub-block 2j (low nibbles) and 2j+1 (high nibbles), 64 weights. Integer
// dots run over whole 32-value sub-blocks, so the float work per weight is a
// quarter of the 8-value units above.
template <int T>
struct MRD_KW {
    static constexpr int per_sb = 4;
    template <int NT, int R>
    static __device__ __forceinline__ void step(const uint8_t *const *wr, int u, const int8_t *xq, const float *xd,
                                                int ncols, float (*acc)[NT]) {
        const int b = u >> 2, j = u & 3;
        int lo[R][8], hi[R][8];
        float ds0[R], ds1[R], dm0[R], dm1[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t *blk = wr[r] + (size_t)b * (T == QT_Q4_K ? 144 : 176);
            const uint32_t dm = *(const uint32_t *)blk;
            const float d = __half2float(__ushort_as_half((uint16_t)(dm & 0xFFFF)));
            const float dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
            int s0, m0, s1, m1;
            scale_min_k4(2 * j, blk + 4, s0, m0);
            scale_min_k4(2 * j + 1, blk + 4, s1, m1);
            ds0[r] = d * s0;
            ds1[r] = d * s1;
            dm0[r] = dmin * m0;
            dm1[r] = dmin * m1;
            const uint4 qa = *(const uint4 *)(blk + (T == QT_Q4_K ? 16 : 48) + 32 * j);
            const uint4 qb = *(const uint4 *)(blk + (T == QT_Q4_K ? 32 : 64) + 32 * j);
            const uint32_t q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
            if constexpr (T == QT_Q4_K) {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    lo[r][k] = q[k] & 0x0F0F0F0F;
                    hi[r][k] = (q[k] >> 4) & 0x0F0F0F0F;
                }
            } else {
                const uint4 ha = *(const uint4 *)(blk + 16);
                const uint4 hb = *(const uint4 *)(blk + 32);
                const uint32_t h[8] = {ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w};
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    lo[r][k] = (q[k] & 0x0F0F0F0F) | (((h[k] >> (2 * j)) & 0x01010101) << 4);
                    hi[r][k] = ((q[k] >> 4) & 0x0F0F0F0F) | (((h[k] >> (2 * j + 1)) & 0x01010101) << 4);
                }
            }
        }
        const int base = b * 256 + 64 * j, xi = base >> 5;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int8_t *xt = xq + (size_t)t * ncols + base;
            const int4 a0 = *(const int4 *)xt, a1 = *(const int4 *)(xt + 16);
            const int4 c0 = *(const int4 *)(xt + 32), c1 = *(const int4 *)(xt + 48);
            const int xa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const int xc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            int sl = 0, sh = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                sl = dp4(0x01010101, xa[k], sl);
                sh = dp4(0x01010101, xc[k], sh);
            }
            const float *xdt = xd + (size_t)t * (ncols / 32) + xi;
            const float x0 = xdt[0], x1 = xdt[1];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int dl = 0, dh = 0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    dl = dp4(lo[r][k], xa[k], dl);
                    dh = dp4(hi[r][k], xc[k], dh);
                }
                acc[r][t] += x0 * (ds0[r] * dl - dm0[r] * sl) + x1 * (ds1[r] * dh - dm1[r] * sh);
            }
        }
    }
};
template <> struct MRD<QT_Q4_K> : MRD_KW<QT_Q4_K> {};
template <> struct MRD<QT_Q5_K> : MRD_KW<QT_Q5_K> {};

// IQ4_XS wide unit = (superblock b, pair p): sub-blocks 2p and 2p+1, each 16
// bytes of qs (low nibbles -> values 0..15, high -> 16..31) and a 6-bit scale.
template <> struct MRD<QT_IQ4_XS> {
    static constexpr int per_sb = 4;
    template <int NT, int R>
    static __device__ __forceinline__ void step(const uint8_t *const *wr, int u, const int8_t *xq, const float *xd,
                                                int ncols, float (*acc)[NT]) {
        const int b = u >> 2, pp = u & 3;
        int lo[R][8], hi[R][8];
        float dl0[R], dl1[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t *blk = wr[r] + (size_t)b * 136;
            const uint32_t hdr = *(const uint32_t *)blk;
            const float d = __half2float(__ushort_as_half((uint16_t)(hdr & 0xFFFF)));
            const uint32_t sh = hdr >> 16;
            const int ib0 = 2 * pp, ib1 = 2 * pp + 1;
            const int ls0 = (blk[4 + pp] & 0xF) | (((sh >> (2 * ib0)) & 3) << 4);
            const int ls1 = (blk[4 + pp] >> 4) | (((sh >> (2 * ib1)) & 3) << 4);
            dl0[r] = d * (ls0 - 32);
            dl1[r] = d * (ls1 - 32);
            const uint4 qa = *(const uint4 *)(blk + 8 + 32 * pp);
            const uint4 qb = *(const uint4 *)(blk + 24 + 32 * pp);
            const uint32_t q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                lo[r][k] = iq4_lut4(q[k] & 0x0F0F0F0F);
                hi[r][k] = iq4_lut4((q[k] >> 4) & 0x0F0F0F0F);
            }
        }
        const int base = b * 256 + 64 * pp, xi = base >> 5;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int8_t *xt = xq + (size_t)t * ncols + base;
            // sub-block 2p: cols [0, 32) of the unit, sub-block 2p+1: [32, 64)
            const int4 a0 = *(const int4 *)xt, a1 = *(const int4 *)(xt + 16);
            const int4 c0 = *(const int4 *)(xt + 32), c1 = *(const int4 *)(xt + 48);
            const float *xdt = xd + (size_t)t * (ncols / 32) + xi;
            const float x0 = xdt[0], x1 = xdt[1];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                // lo[0..3]: sub-block 2p bytes 0..15 (low -> a0, high -> a1)
                int s0 = dp4(lo[r][0], a0.x, 0);
                s0 = dp4(lo[r][1], a0.y, s0);
                s0 = dp4(lo[r][2], a0.z, s0);
                s0 = dp4(lo[r][3], a0.w, s0);
                s0 = dp4(hi[r][0], a1.x, s0);
                s0 = dp4(hi[r][1], a1.y, s0);
                s0 = dp4(hi[r][2], a1.z, s0);
                s0 = dp4(hi[r][3], a1.w, s0);
                int s1 = dp4(lo[r][4], c0.x, 0);
                s1 = dp4(lo[r][5], c0.y, s1);
                s1 = dp4(lo[r][6], c0.z, s1);
                s1 = dp4(lo[r][7], c0.w, s1);
                s1 = dp4(hi[r][4], c1.x, s1);
                s1 = dp4(hi[r][5], c1.y, s1);
                s1 = dp4(hi[r][6], c1.z, s1);
                s1 = dp4(hi[r][7], c1.w, s1);
                acc[r][t] += dl0[r] * x0 * s0 + dl1[r] * x1 * s1;
            }
        }
    }
};

// MXFP4 wide unit: as IQ4_XS (sub-blocks 2p, 2p+1), scales 2^(e - 128).
template <> struct MRD<QT_MXFP4> {
    static constexpr int per_sb = 4;
    template <int NT, int R>
    static __device__ __forceinline__ void step(const uint8_t *const *wr, int u, const int8_t *xq, const float *xd,
                                                int ncols, float (*acc)[NT]) {
        const int b = u >> 2, pp = u & 3;
        int lo[R][8], hi[R][8];
        float dl0[R], dl1[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t *blk = wr[r] + (size_t)b * 136;
            const uint32_t ee = *(const uint16_t *)(blk + 2 * pp);
            dl0[r] = e8m0_half(ee & 0xFF);
            dl1[r] = e8m0_half(ee >> 8);
            const uint4 qa = *(const uint4 *)(blk + 8 + 32 * pp);
            const uint4 qb = *(const uint4 *)(blk + 24 + 32 * pp);
            const uint32_t q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                lo[r][k] = mx_lut4(q[k] & 0x0F0F0F0F);
                hi[r][k] = mx_lut4((q[k] >> 4) & 0x0F0F0F0F);
            }
        }
        const int base = b * 256 + 64 * pp, xi = base >> 5;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int8_t *xt = xq + (size_t)t * ncols + base;
            const int4 a0 = *(const int4 *)xt, a1 = *(const int4 *)(xt + 16);
            const int4 c0 = *(const int4 *)(xt + 32), c1 = *(const int4 *)(xt + 48);
            const float *xdt = xd + (size_t)t * (ncols / 32) + xi;
            const float x0 = xdt[0], x1 = xdt[1];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int s0 = dp4(lo[r][0], a0.x, 0);
                s0 = dp4(lo[r][1], a0.y, s0);
                s0 = dp4(lo[r][2], a0.z, s0);
                s0 = dp4(lo[r][3], a0.w, s0);
                s0 = dp4(hi[r][0], a1.x, s0);
                s0 = dp4(hi[r][1], a1.y, s0);
                s0 = dp4(hi[r][2], a1.z, s0);
                s0 = dp4(hi[r][3], a1.w, s0);
                int s1 = dp4(lo[r][4], c0.x, 0);
                s1 = dp4(lo[r][5], c0.y, s1);
                s1 = dp4(lo[r][6], c0.z, s1);
                s1 = dp4(lo[r][7], c0.w, s1);
                s1 = dp4(hi[r][4], c1.x, s1);
                s1 = dp4(hi[r][5], c1.y, s1);
                s1 = dp4(hi[r][6], c1.z, s1);
                s1 = dp4(hi[r][7], c1.w, s1);
                acc[r][t] += dl0[r] * x0 * s0 + dl1[r] * x1 * s1;
            }
        }
    }
};

// Q6_K: per row 4 quarters of 2 int8x4 weights, 4 scales and d.
template <> struct MRD<QT_Q6_K> {
    static constexpr int per_sb = QD<QT_Q6_K>::per_sb;
    template <int NT, int R>
    static __device__ __forceinline__ void step(const uint8_t *const *wr, int u, const int8_t *xq, const float *xd,
                                                int ncols, float (*acc)[NT]) {
        const int b = u >> 3, n = (u >> 2) & 1, g = u & 3;
        const int is = g >> 1;
        int v0[R][4], v1[R][4];
        float scf[R][4], dd[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t *blk = wr[r] + (size_t)b * 210;
            const uint8_t *ql = blk + 64 * n + 8 * g, *qh = blk + 128 + 32 * n + 8 * g;
            const int8_t *sc = (const int8_t *)(blk + 192) + 8 * n;
            dd[r] = h2f(blk + 208);
            const uint32_t a0 = ld32u(ql), a1 = ld32u(ql + 4), c0 = ld32u(ql + 32), c1 = ld32u(ql + 36);
            const uint32_t h0 = ld32u(qh), h1 = ld32u(qh + 4);
#pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                const uint32_t l0 = (quarter & 1) ? c0 : a0, l1 = (quarter & 1) ? c1 : a1;
                const int ls = (quarter >> 1) ? 4 : 0;
                v0[r][quarter] = ((l0 >> ls) & 0x0F0F0F0F) | (((h0 >> (2 * quarter)) & 0x03030303) << 4);
                v1[r][quarter] = ((l1 >> ls) & 0x0F0F0F0F) | (((h1 >> (2 * quarter)) & 0x03030303) << 4);
                scf[r][quarter] = (float)sc[is + 2 * quarter];
            }
        }
        const int base = b * 256 + 128 * n + 8 * g;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int8_t *xt = xq + (size_t)t * ncols;
            const float *xdt = xd + (size_t)t * (ncols / 32);
            int2 xv[4];
            int sv[4];
            float xs[4];
#pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                xv[quarter] = *(const int2 *)(xt + base + 32 * quarter);
                sv[quarter] = dp4(0x01010101, xv[quarter].y, dp4(0x01010101, xv[quarter].x, 0));
                xs[quarter] = xdt[(base + 32 * quarter) >> 5];
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float total = 0.f;
#pragma unroll
                for (int quarter = 0; quarter < 4; ++quarter) {
                    const int dv = dp4(v1[r][quarter], xv[quarter].y, dp4(v0[r][quarter], xv[quarter].x, 0));
                    total += scf[r][quarter] * xs[quarter] * (float)(dv - 32 * sv[quarter]);
                }
                acc[r][t] += dd[r] * total;
            }
        }
    }
};

template <int T, int NT, int R>
__device__ __forceinline__ void gemvq_mr_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                              const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                              const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row0 = (bid * (blockDim.x / WARP) + threadIdx.x / WARP) * R;
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
        gemvq_##SUF(GV_PARAMS) {          \
        { const GvSel gs = gv_sel(g); gemvq_impl<T>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }                  \
    }
#define GEMVQ_NT_ENTRY(NT, SUF, T)                                                               \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_nt##NT##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,            \
                             const float *xd, float *y, int ncols, int nrows, int accumulate,   \
                             const int *skip) {                                                  \
        gemvq_nt_impl<T, NT>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x);           \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvq_nt##NT##_##SUF(GV_PARAMS) {                                                  \
        { const GvSel gs = gv_sel(g); gemvq_nt_impl<T, NT>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }           \
    }
#define GEMVQ_W_ENTRY(SUF, T)                                                                    \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_##SUF(GV_PARAMS) {          \
        { const GvSel gs = gv_sel(g); gemvq_mr_impl<T, 1, 1>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }        \
    }
#define GEMVQ_W_NT_ENTRY(NT, SUF, T)                                                             \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_nt##NT##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,            \
                             const float *xd, float *y, int ncols, int nrows, int accumulate,   \
                             const int *skip) {                                                  \
        gemvq_mr_impl<T, NT, 1>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x);        \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvq_nt##NT##_##SUF(GV_PARAMS) {                                                  \
        { const GvSel gs = gv_sel(g); gemvq_mr_impl<T, NT, 1>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }        \
    }
// item T1: q3_k / iq3_s one-token and q3_k dp4 few-token entries (bitwise == gemvq_impl / gemvq_nt_impl)
#define GEMVQ_T1_ENTRY(SUF, IMPL)                                                                \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_##SUF(GV_PARAMS) {          \
        { const GvSel gs = gv_sel(g); IMPL(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }                           \
    }
#define GEMVQ_T1_Q3K_NT_ENTRY(NT)                                                                \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_nt##NT##_q3_k(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,             \
                            const float *xd, float *y, int ncols, int nrows, int accumulate,    \
                            const int *skip) {                                                   \
        t1_q3k_nt_impl<NT>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x);             \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvq_nt##NT##_q3_k(GV_PARAMS) {                                                   \
        { const GvSel gs = gv_sel(g); t1_q3k_nt_impl<NT>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }             \
    }
#define GEMVQ_NT_TYPES(NT)                \
    GEMVQ_W_NT_ENTRY(NT, q4_k, QT_Q4_K)     \
    GEMVQ_W_NT_ENTRY(NT, q5_k, QT_Q5_K)     \
    GEMVQ_W_NT_ENTRY(NT, iq4_xs, QT_IQ4_XS) \
    GEMVQ_W_NT_ENTRY(NT, mxfp4, QT_MXFP4)   \
    GEMVQ_NT_ENTRY(NT, q6_k, QT_Q6_K)     \
    GEMVQ_NT_ENTRY(NT, iq4_nl, QT_IQ4_NL) \
    GEMVQ_NT_ENTRY(NT, q8_0, QT_Q8_0)     \
    GEMVQ_T1_Q3K_NT_ENTRY(NT)             \
    GEMVQ_NT_ENTRY(NT, iq3_s, QT_IQ3_S)

GEMVQ_W_ENTRY(q4_k, QT_Q4_K)
GEMVQ_W_ENTRY(q5_k, QT_Q5_K)
GEMVQ_W_ENTRY(iq4_xs, QT_IQ4_XS)
GEMVQ_W_ENTRY(mxfp4, QT_MXFP4)
GEMVQ_ENTRY(q6_k, QT_Q6_K)
GEMVQ_ENTRY(iq4_nl, QT_IQ4_NL)
GEMVQ_ENTRY(q8_0, QT_Q8_0)
GEMVQ_T1_ENTRY(q3_k, t1_q3k_1_impl<4>)
GEMVQ_T1_ENTRY(iq3_s, t1_iq3s_1_impl<4>)
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
        gemvq_mr_impl<T, NT, R>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x);        \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvq_nt##NT##r##R##_##SUF(GV_PARAMS) {                            \
        { const GvSel gs = gv_sel(g); gemvq_mr_impl<T, NT, R>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }        \
    }
#define GEMVQ_MR_TYPES(NT, R)                    \
    GEMVQ_MR_ENTRY(NT, R, q4_k, QT_Q4_K)         \
    GEMVQ_MR_ENTRY(NT, R, q5_k, QT_Q5_K)         \
    GEMVQ_MR_ENTRY(NT, R, iq4_xs, QT_IQ4_XS)     \
    GEMVQ_MR_ENTRY(NT, R, mxfp4, QT_MXFP4)       \
    GEMVQ_MR_ENTRY(NT, R, q6_k, QT_Q6_K)
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

