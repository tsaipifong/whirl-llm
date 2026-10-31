template <int T> struct MRD;  // shared multi-row GEMV unit decoders (both kernel sets, ADR-0005); see the end of this file

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
            const float x0 = GMR_X0, x1 = GMR_X1;
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
#include "shared/gemv_mr_unit4_dot.h"  // acc[r][t] += unit dots (ADR-0005)
    }
};

// kernels/shared/gemv_mr_kw.h -- the per-format unit decoders of the multi-row multi-token int8 GEMV gemvq_mr_impl:
// MRD (declaration), MRD_KW (Q4_K / Q5_K wide unit), MRD<QT_Q4_K>, MRD<QT_Q5_K>, MRD<QT_IQ4_XS> (its token side is
// shared/gemv_mr_unit4_dot.h). One copy for both kernel sets: kernels/gemv_mr.hip (gfx1201) and
// kernels/gfx1151/gemv_mr.hip; ADR-0005 / arch025-sc11e (the two copies differed only in the activation-scale read).
// File scope, included at the same TU position as the code it replaces. Before the #include the includer defines
//   GMR_X0, GMR_X1  the scales of sub-blocks 2p / 2p+1 (relative to xdt; gfx1201 xdt[0] / xdt[1], gfx1151 packed
//                   scale words xdv(xdt) / xdv(xdt + 1))
// and #undefs them after its last use (gemv_mr_q6.h, MRD<QT_MXFP4> of gfx1201).
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
