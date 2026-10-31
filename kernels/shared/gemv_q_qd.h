template <int T> struct QD;  // shared int8 GEMV unit dots QD<Q4_K .. Q3_K> (both kernel sets, ADR-0005); see the end of this file

// Q4_K unit = (superblock, j in 0..3, i in 0..3): 8 low nibbles (sub-block 2j)
// and 8 high nibbles (sub-block 2j+1) from qs[32j + 8i .. +8].
template <> struct QD<QT_Q4_K> {
    static constexpr int per_sb = 16, sb = 256, bytes = 144;
    static __device__ __forceinline__ float dot(const uint8_t *row, int u, const int8_t *xq, const float *xd) {
        const int b = u >> 4, j = (u >> 2) & 3, i = u & 3;
        const uint8_t *blk = row + (size_t)b * bytes;
        const uint32_t dm = *(const uint32_t *)blk;
        const float d = __half2float(__ushort_as_half((uint16_t)(dm & 0xFFFF)));
        const float dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
        const uint2 q = *(const uint2 *)(blk + 16 + 32 * j + 8 * i);
        const int base = b * 256 + 64 * j + 8 * i;
        const int2 xl = *(const int2 *)(xq + base);
        const int2 xh = *(const int2 *)(xq + base + 32);
        const int lo0 = q.x & 0x0F0F0F0F, lo1 = q.y & 0x0F0F0F0F;
        const int hi0 = (q.x >> 4) & 0x0F0F0F0F, hi1 = (q.y >> 4) & 0x0F0F0F0F;
        const int dl = dp4(lo1, xl.y, dp4(lo0, xl.x, 0));
        const int dh = dp4(hi1, xh.y, dp4(hi0, xh.x, 0));
        const int sl = dp4(0x01010101, xl.y, dp4(0x01010101, xl.x, 0));
        const int sh = dp4(0x01010101, xh.y, dp4(0x01010101, xh.x, 0));
        int s0, m0, s1, m1;
        scale_min_k4(2 * j, blk + 4, s0, m0);
        scale_min_k4(2 * j + 1, blk + 4, s1, m1);
        const float x0 = GQ_XD((b * 256 + 64 * j) >> 5), x1 = GQ_XD((b * 256 + 64 * j + 32) >> 5);
        return d * (s0 * x0 * dl + s1 * x1 * dh) - dmin * (m0 * x0 * sl + m1 * x1 * sh);
    }
};

// Q5_K: as Q4_K plus the 5th bit from qh[8i .. +8], bits 2j / 2j+1.
template <> struct QD<QT_Q5_K> {
    static constexpr int per_sb = 16, sb = 256, bytes = 176;
    static __device__ __forceinline__ float dot(const uint8_t *row, int u, const int8_t *xq, const float *xd) {
        const int b = u >> 4, j = (u >> 2) & 3, i = u & 3;
        const uint8_t *blk = row + (size_t)b * bytes;
        const uint32_t dm = *(const uint32_t *)blk;
        const float d = __half2float(__ushort_as_half((uint16_t)(dm & 0xFFFF)));
        const float dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
        const uint2 q = *(const uint2 *)(blk + 48 + 32 * j + 8 * i);
        const uint2 h = *(const uint2 *)(blk + 16 + 8 * i);
        const int base = b * 256 + 64 * j + 8 * i;
        const int2 xl = *(const int2 *)(xq + base);
        const int2 xh = *(const int2 *)(xq + base + 32);
        const int lo0 = (q.x & 0x0F0F0F0F) | (((h.x >> (2 * j)) & 0x01010101) << 4);
        const int lo1 = (q.y & 0x0F0F0F0F) | (((h.y >> (2 * j)) & 0x01010101) << 4);
        const int hi0 = ((q.x >> 4) & 0x0F0F0F0F) | (((h.x >> (2 * j + 1)) & 0x01010101) << 4);
        const int hi1 = ((q.y >> 4) & 0x0F0F0F0F) | (((h.y >> (2 * j + 1)) & 0x01010101) << 4);
        const int dl = dp4(lo1, xl.y, dp4(lo0, xl.x, 0));
        const int dh = dp4(hi1, xh.y, dp4(hi0, xh.x, 0));
        const int sl = dp4(0x01010101, xl.y, dp4(0x01010101, xl.x, 0));
        const int sh = dp4(0x01010101, xh.y, dp4(0x01010101, xh.x, 0));
        int s0, m0, s1, m1;
        scale_min_k4(2 * j, blk + 4, s0, m0);
        scale_min_k4(2 * j + 1, blk + 4, s1, m1);
        const float x0 = GQ_XD((b * 256 + 64 * j) >> 5), x1 = GQ_XD((b * 256 + 64 * j + 32) >> 5);
        return d * (s0 * x0 * dl + s1 * x1 * dh) - dmin * (m0 * x0 * sl + m1 * x1 * sh);
    }
};

// Map 4 packed nibbles (one per byte, low 4 bits) through kvalues_iq4nl with
// two v_perm_b32 byte selects (entries 0..7 and 8..15) and a bit-3 blend.
#include "shared/gemv_q_part.h"  // also in the gfx1151 set (kernels/gfx1151/gemv_q.hip)

// MXFP4: the IQ4_XS unit layout with a power-of-2 block scale 2^(e - 128).
template <> struct QD<QT_MXFP4> {
    static constexpr int per_sb = 16, sb = 256, bytes = 136;
    static __device__ __forceinline__ float dot(const uint8_t *row, int u, const int8_t *xq, const float *xd) {
        const int b = u >> 4, ib = (u >> 1) & 7, hh = u & 1;
        const uint8_t *blk = row + (size_t)b * bytes;
        const float dl = e8m0_half(blk[ib]);
        const uint2 q = *(const uint2 *)(blk + 8 + 16 * ib + 8 * hh);
        const int base = b * 256 + 32 * ib + 8 * hh;
        const int2 xl = *(const int2 *)(xq + base);
        const int2 xh = *(const int2 *)(xq + base + 16);
        int acc = dp4(mx_lut4(q.x & 0x0F0F0F0F), xl.x, 0);
        acc = dp4(mx_lut4(q.y & 0x0F0F0F0F), xl.y, acc);
        acc = dp4(mx_lut4((q.x >> 4) & 0x0F0F0F0F), xh.x, acc);
        acc = dp4(mx_lut4((q.y >> 4) & 0x0F0F0F0F), xh.y, acc);
        return dl * GQ_XD((b * 256 + 32 * ib) >> 5) * acc;
    }
};

// IQ4_XS unit = (superblock, ib 0..7, h 0..1): qs[16ib + 8h .. +8] ->
// low nibbles at 32ib + 8h, high nibbles at 32ib + 16 + 8h.
template <> struct QD<QT_IQ4_XS> {
    static constexpr int per_sb = 16, sb = 256, bytes = 136;
    static __device__ __forceinline__ float dot(const uint8_t *row, int u, const int8_t *xq, const float *xd) {
        const int b = u >> 4, ib = (u >> 1) & 7, hh = u & 1;
        const uint8_t *blk = row + (size_t)b * bytes;
        const uint32_t hdr = *(const uint32_t *)blk;
        const float d = __half2float(__ushort_as_half((uint16_t)(hdr & 0xFFFF)));
        const uint32_t sh = hdr >> 16;
        const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        const uint2 q = *(const uint2 *)(blk + 8 + 16 * ib + 8 * hh);
        const int base = b * 256 + 32 * ib + 8 * hh;
        const int2 xl = *(const int2 *)(xq + base);
        const int2 xh = *(const int2 *)(xq + base + 16);
        int acc = dp4(iq4_lut4(q.x & 0x0F0F0F0F), xl.x, 0);
        acc = dp4(iq4_lut4(q.y & 0x0F0F0F0F), xl.y, acc);
        acc = dp4(iq4_lut4((q.x >> 4) & 0x0F0F0F0F), xh.x, acc);
        acc = dp4(iq4_lut4((q.y >> 4) & 0x0F0F0F0F), xh.y, acc);
        return d * (ls - 32) * GQ_XD((b * 256 + 32 * ib) >> 5) * acc;
    }
};

// Q6_K unit = (superblock, n 0..1, g 0..3): positions l = 8g..8g+8 in each of
// the 4 quarters of half n. Blocks are 210 bytes (2-aligned): byte loads.
template <> struct QD<QT_Q6_K> {
    static constexpr int per_sb = 8, sb = 256, bytes = 210;
    static __device__ __forceinline__ float dot(const uint8_t *row, int u, const int8_t *xq, const float *xd) {
        const int b = u >> 3, n = (u >> 2) & 1, g = u & 3;
        const uint8_t *blk = row + (size_t)b * bytes;
        const uint8_t *ql = blk + 64 * n + 8 * g, *qh = blk + 128 + 32 * n + 8 * g;
        const int8_t *sc = (const int8_t *)(blk + 192) + 8 * n;
        const float d = h2f(blk + 208);
        const uint32_t a0 = ld32u(ql), a1 = ld32u(ql + 4), c0 = ld32u(ql + 32), c1 = ld32u(ql + 36);
        const uint32_t h0 = ld32u(qh), h1 = ld32u(qh + 4);
        const int base = b * 256 + 128 * n + 8 * g;
        const int is = g >> 1;  // l / 16
        float total = 0.f;
#pragma unroll
        for (int quarter = 0; quarter < 4; ++quarter) {
            const uint32_t l0 = (quarter & 1) ? c0 : a0, l1 = (quarter & 1) ? c1 : a1;
            const int ls = (quarter >> 1) ? 4 : 0;
            const int v0 = ((l0 >> ls) & 0x0F0F0F0F) | (((h0 >> (2 * quarter)) & 0x03030303) << 4);
            const int v1 = ((l1 >> ls) & 0x0F0F0F0F) | (((h1 >> (2 * quarter)) & 0x03030303) << 4);
            const int2 xv = *(const int2 *)(xq + base + 32 * quarter);
            const int dv = dp4(v1, xv.y, dp4(v0, xv.x, 0));
            const int sv = dp4(0x01010101, xv.y, dp4(0x01010101, xv.x, 0));
            total += (float)sc[is + 2 * quarter] * GQ_XD((base + 32 * quarter) >> 5) * (float)(dv - 32 * sv);
        }
        return d * total;
    }
};

// IQ4_NL unit = (block of 32, h 0..1): qs[8h .. +8] -> low nibbles at 8h,
// high nibbles at 16 + 8h. 18-byte blocks: byte loads.
template <> struct QD<QT_IQ4_NL> {
    static constexpr int per_sb = 2, sb = 32, bytes = 18;
    static __device__ __forceinline__ float dot(const uint8_t *row, int u, const int8_t *xq, const float *xd) {
        const int b = u >> 1, hh = u & 1;
        const uint8_t *blk = row + (size_t)b * bytes;
        const float d = h2f(blk);
        const uint32_t q0 = ld32u(blk + 2 + 8 * hh), q1 = ld32u(blk + 6 + 8 * hh);
        const int base = b * 32 + 8 * hh;
        const int2 xl = *(const int2 *)(xq + base);
        const int2 xh = *(const int2 *)(xq + base + 16);
        int acc = dp4(iq4_lut4(q0 & 0x0F0F0F0F), xl.x, 0);
        acc = dp4(iq4_lut4(q1 & 0x0F0F0F0F), xl.y, acc);
        acc = dp4(iq4_lut4((q0 >> 4) & 0x0F0F0F0F), xh.x, acc);
        acc = dp4(iq4_lut4((q1 >> 4) & 0x0F0F0F0F), xh.y, acc);
        return d * GQ_XD_B * acc;
    }
};

// Q8_0 unit = (block of 32, quarter 0..3): 8 int8 weights. 34-byte blocks.
template <> struct QD<QT_Q8_0> {
    static constexpr int per_sb = 4, sb = 32, bytes = 34;
    static __device__ __forceinline__ float dot(const uint8_t *row, int u, const int8_t *xq, const float *xd) {
        const int b = u >> 2, qq = u & 3;
        const uint8_t *blk = row + (size_t)b * bytes;
        const float d = h2f(blk);
        const int w0 = (int)ld32u(blk + 2 + 8 * qq), w1 = (int)ld32u(blk + 6 + 8 * qq);
        const int2 xv = *(const int2 *)(xq + b * 32 + 8 * qq);
        return d * GQ_XD_B * dp4(w1, xv.y, dp4(w0, xv.x, 0));
    }
};

// Q3_K unit = (superblock, n 0..1, j 0..3, sg 0..3): positions
// 128n + 32j + 8sg .. +8; q = ((qs[32n+s] >> 2j) & 3) - (hmask bit clear ? 4 : 0).
template <> struct QD<QT_Q3_K> {
    static constexpr int per_sb = 32, sb = 256, bytes = 110;
    static __device__ __forceinline__ float dot(const uint8_t *row, int u, const int8_t *xq, const float *xd) {
        const int b = u >> 5, n = (u >> 4) & 1, j = (u >> 2) & 3, sg = u & 3;
        const uint8_t *blk = row + (size_t)b * bytes;
        const uint8_t *hm = blk, *qs = blk + 32, *sb8 = blk + 96;
        const float d = h2f(blk + 108);
        const int kk = 8 * n + 2 * j + (sg >> 1);
        const int lo = (kk < 8) ? (sb8[kk] & 0xF) : (sb8[kk - 8] >> 4);
        const int hi = (sb8[8 + (kk & 3)] >> (2 * (kk >> 2))) & 3;
        const float dl = d * ((lo | (hi << 4)) - 32);
        const uint32_t q0 = ld32u(qs + 32 * n + 8 * sg), q1 = ld32u(qs + 32 * n + 8 * sg + 4);
        const uint32_t h0 = ld32u(hm + 8 * sg), h1 = ld32u(hm + 8 * sg + 4);
        const int sh = 4 * n + j;
        const int w0 = (q0 >> (2 * j)) & 0x03030303, w1 = (q1 >> (2 * j)) & 0x03030303;
        const int m0 = ((~h0) >> sh) & 0x01010101, m1 = ((~h1) >> sh) & 0x01010101;
        const int base = b * 256 + 128 * n + 32 * j + 8 * sg;
        const int2 xv = *(const int2 *)(xq + base);
        const int dv = dp4(w1, xv.y, dp4(w0, xv.x, 0)) - 4 * dp4(m1, xv.y, dp4(m0, xv.x, 0));
        return dl * GQ_XD(base >> 5) * dv;
    }
};

// kernels/shared/gemv_q_qd.h -- the single-token unit dots of the int8 decode GEMV (gemvq_impl and, through QDN's
// generic fallback, the multi-token paths): the QD declaration and QD<QT_Q4_K> .. QD<QT_Q3_K> in the order of the
// original files (it #includes gemv_q_part.h, already shared, at its old place). One copy for both kernel sets:
// kernels/gemv_q.hip (gfx1201) and kernels/gfx1151/gemv_q.hip; ADR-0005 / arch025-sc11e (the two copies differed
// only in the activation-scale reads). File scope, included at the same TU position as the code it replaces. Before
// the #include the includer defines (relative to the dot's xd argument)
//   GQ_XD(e)  the scale of 32-group e (gfx1201 xd[e], gfx1151 packed scale words xdv(xd + (e)))
//   GQ_XD_B   the scale of 32-group b (gfx1201 xd[b], gfx1151 xdv(xd + b))
// and #undefs them after gemv_q_qd_iq3s.h.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
