// WHIRL device code (gfx1201 / RDNA 4): decode GEMV with int8 activations (gemvq_*), multi-token gemvq_nt*, grouped ABI, Q3_K / IQ3_S paths.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Decode GEMV with int8 activations (llama.cpp mmvq idea): x is quantized to
// int8 in blocks of 32 (scale xd), weights are consumed as packed nibbles/bytes
// and multiplied with v_dot4 (4 int8 MACs per instruction).
// Each lane handles one "unit" of 16 or 32 weights per iteration.
// ---------------------------------------------------------------------------

// x (f32, n) -> xq (int8, n), xd (f32, n/32). One wave per 32-block.
extern "C" __global__ void quantize_q8(const float *x, int8_t *xq, float *xd, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float v = i < n ? x[i] : 0.f;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    if (i < n) {
        xq[i] = (int8_t)__float2int_rn(v * id);
        if ((i & 31) == 0) xd[i >> 5] = d;
    }
}

__device__ __forceinline__ int dp4(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }

__device__ __forceinline__ uint32_t ld32u(const uint8_t *p) {  // unaligned-safe
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

template <int T> struct QD;

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
        const float x0 = xd[(b * 256 + 64 * j) >> 5], x1 = xd[(b * 256 + 64 * j + 32) >> 5];
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
        const float x0 = xd[(b * 256 + 64 * j) >> 5], x1 = xd[(b * 256 + 64 * j + 32) >> 5];
        return d * (s0 * x0 * dl + s1 * x1 * dh) - dmin * (m0 * x0 * sl + m1 * x1 * sh);
    }
};

// Map 4 packed nibbles (one per byte, low 4 bits) through kvalues_iq4nl with
// two v_perm_b32 byte selects (entries 0..7 and 8..15) and a bit-3 blend.
__device__ __forceinline__ int iq4_lut4(uint32_t nib4) {
    const uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    const uint32_t sel = nib4 & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(t1, t0, sel);
    const uint32_t hi = __builtin_amdgcn_perm(t3, t2, sel);
    const uint32_t m = ((nib4 & 0x08080808u) >> 3) * 0xFFu;
    return (int)((hi & m) | (lo & ~m));
}

// kvalues_mxfp4 (0 1 2 3 4 6 8 12 and negatives) through the same two-table
// v_perm_b32 select as iq4_lut4.
__device__ __forceinline__ int mx_lut4(uint32_t nib4) {
    const uint32_t t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    const uint32_t sel = nib4 & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(t1, t0, sel);
    const uint32_t hi = __builtin_amdgcn_perm(t3, t2, sel);
    const uint32_t m = ((nib4 & 0x08080808u) >> 3) * 0xFFu;
    return (int)((hi & m) | (lo & ~m));
}

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
        return dl * xd[(b * 256 + 32 * ib) >> 5] * acc;
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
        return d * (ls - 32) * xd[(b * 256 + 32 * ib) >> 5] * acc;
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
            total += (float)sc[is + 2 * quarter] * xd[(base + 32 * quarter) >> 5] * (float)(dv - 32 * sv);
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
        return d * xd[b] * acc;
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
        return d * xd[b] * dp4(w1, xv.y, dp4(w0, xv.x, 0));
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
        return dl * xd[base >> 5] * dv;
    }
};

// IQ3_S unit = (superblock, ib32 0..7, l 0..3): 8 values from two grid entries.
template <> struct QD<QT_IQ3_S> {
    static constexpr int per_sb = 32, sb = 256, bytes = 110;
    static __device__ __forceinline__ float dot(const uint8_t *row, int u, const int8_t *xq, const float *xd) {
        const int b = u >> 5, ib32 = (u >> 2) & 7, l = u & 3;
        const uint8_t *blk = row + (size_t)b * bytes;
        const float d = h2f(blk);
        const uint8_t *qs = blk + 2, *qh = blk + 66, *signs = blk + 74, *scales = blk + 106;
        const float db = d * (1 + 2 * ((scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xF));
        const uint8_t h = qh[ib32];
        const uint32_t g1 = k_iq3s_grid[qs[8 * ib32 + 2 * l] | ((h << (8 - 2 * l)) & 256)];
        const uint32_t g2 = k_iq3s_grid[qs[8 * ib32 + 2 * l + 1] | ((h << (7 - 2 * l)) & 256)];
        const uint32_t sg = signs[4 * ib32 + l];
        const uint32_t m1 = (((sg & 0xF) * 0x00204081u) & 0x01010101u) * 0xFFu;
        const uint32_t m2 = (((sg >> 4) * 0x00204081u) & 0x01010101u) * 0xFFu;
        const int v1 = (int)((g1 ^ m1) + (m1 & 0x01010101u));
        const int v2 = (int)((g2 ^ m2) + (m2 & 0x01010101u));
        const int base = b * 256 + 32 * ib32 + 8 * l;
        const int2 xv = *(const int2 *)(xq + base);
        return db * xd[base >> 5] * dp4(v2, xv.y, dp4(v1, xv.x, 0));
    }
};

// Multi-token unit dot: decode the weight unit once, dot it with NT tokens.
// Generic fallback re-runs the single-token dot (weight part left to CSE).
template <int T> struct QDN {
    template <int NT>
    static __device__ __forceinline__ void dot(const uint8_t *row, int u, const int8_t *xq, const float *xd, int ncols,
                                               float *acc) {
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] += QD<T>::dot(row, u, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32));
    }
};

// Q4_K / Q5_K share the tail: 4 packed int8x4 weights (two 32-sub-blocks),
// per-sub-block scale/min, and x sums for the min term.
template <int NT>
__device__ __forceinline__ void kq_tail(int lo0, int lo1, int hi0, int hi1, float ds0, float ds1, float dm0, float dm1,
                                        int base, int xi, const int8_t *xq, const float *xd, int ncols, float *acc) {
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int8_t *xt = xq + (size_t)t * ncols + base;
        const int2 xl = *(const int2 *)xt;
        const int2 xh = *(const int2 *)(xt + 32);
        const int dl = dp4(lo1, xl.y, dp4(lo0, xl.x, 0));
        const int dh = dp4(hi1, xh.y, dp4(hi0, xh.x, 0));
        const int sl = dp4(0x01010101, xl.y, dp4(0x01010101, xl.x, 0));
        const int sh = dp4(0x01010101, xh.y, dp4(0x01010101, xh.x, 0));
        const float *xdt = xd + (size_t)t * (ncols / 32) + xi;
        acc[t] += xdt[0] * (ds0 * dl - dm0 * sl) + xdt[1] * (ds1 * dh - dm1 * sh);
    }
}

template <> struct QDN<QT_Q4_K> {
    template <int NT>
    static __device__ __forceinline__ void dot(const uint8_t *row, int u, const int8_t *xq, const float *xd, int ncols,
                                               float *acc) {
        const int b = u >> 4, j = (u >> 2) & 3, i = u & 3;
        const uint8_t *blk = row + (size_t)b * 144;
        const uint32_t dm = *(const uint32_t *)blk;
        const float d = __half2float(__ushort_as_half((uint16_t)(dm & 0xFFFF)));
        const float dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
        const uint2 q = *(const uint2 *)(blk + 16 + 32 * j + 8 * i);
        int s0, m0, s1, m1;
        scale_min_k4(2 * j, blk + 4, s0, m0);
        scale_min_k4(2 * j + 1, blk + 4, s1, m1);
        kq_tail<NT>(q.x & 0x0F0F0F0F, q.y & 0x0F0F0F0F, (q.x >> 4) & 0x0F0F0F0F, (q.y >> 4) & 0x0F0F0F0F, d * s0, d * s1,
                    dmin * m0, dmin * m1, b * 256 + 64 * j + 8 * i, (b * 256 + 64 * j) >> 5, xq, xd, ncols, acc);
    }
};

template <> struct QDN<QT_Q5_K> {
    template <int NT>
    static __device__ __forceinline__ void dot(const uint8_t *row, int u, const int8_t *xq, const float *xd, int ncols,
                                               float *acc) {
        const int b = u >> 4, j = (u >> 2) & 3, i = u & 3;
        const uint8_t *blk = row + (size_t)b * 176;
        const uint32_t dm = *(const uint32_t *)blk;
        const float d = __half2float(__ushort_as_half((uint16_t)(dm & 0xFFFF)));
        const float dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
        const uint2 q = *(const uint2 *)(blk + 48 + 32 * j + 8 * i);
        const uint2 h = *(const uint2 *)(blk + 16 + 8 * i);
        int s0, m0, s1, m1;
        scale_min_k4(2 * j, blk + 4, s0, m0);
        scale_min_k4(2 * j + 1, blk + 4, s1, m1);
        kq_tail<NT>((q.x & 0x0F0F0F0F) | (((h.x >> (2 * j)) & 0x01010101) << 4),
                    (q.y & 0x0F0F0F0F) | (((h.y >> (2 * j)) & 0x01010101) << 4),
                    ((q.x >> 4) & 0x0F0F0F0F) | (((h.x >> (2 * j + 1)) & 0x01010101) << 4),
                    ((q.y >> 4) & 0x0F0F0F0F) | (((h.y >> (2 * j + 1)) & 0x01010101) << 4), d * s0, d * s1, dmin * m0,
                    dmin * m1, b * 256 + 64 * j + 8 * i, (b * 256 + 64 * j) >> 5, xq, xd, ncols, acc);
    }
};

// ---- item T1b: grouped GEMV ABI (begin) ----
// One launch of a one-token gemvq_<type> covers up to 3 matrices that share the input
// (attention q/k/v, FFN gate/up, DeltaNet qkv/z): blocks [0, b1) run segment 0,
// [b1, b2) segment 1, the rest segment 2, each with its own (W, row_bytes, y, nrows)
// and segment-local block index. Per-row arithmetic is that of the single launch
// (bitwise the same). The multi-token entries keep their own signature (selecting W
// per block changes their codegen: +30..90% VGPRs, spills on Q6_K / Q5_K v7); each
// has a grouped twin "g<name>" that the host uses only for same-input groups.
struct GvArgs {
    const uint8_t *W0, *W1, *W2;
    float *y0, *y1, *y2;
    uint64_t rb0, rb1, rb2;
    int n0, n1, n2;
    int b1, b2;
};
struct GvSel {
    const uint8_t *W;
    float *y;
    uint64_t rb;
    int nrows, b;
};
__device__ __forceinline__ GvSel gv_sel(const GvArgs &g) {
    const int b = blockIdx.x;
    if (b < g.b1) return {g.W0, g.y0, g.rb0, g.n0, b};
    if (b < g.b2) return {g.W1, g.y1, g.rb1, g.n1, b - g.b1};
    return {g.W2, g.y2, g.rb2, g.n2, b - g.b2};
}
#define GV_PARAMS const GvArgs g, const int8_t *xq, const float *xd, int ncols, int accumulate, const int *skip
// marker: the host passes GvArgs to the one-token gemvq_<type> entries
extern "C" __global__ void gemv_grouped_abi() {}
// ---- item T1b: grouped GEMV ABI (end) ----
template <int T>
__device__ __forceinline__ void gemvq_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                           const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                           const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = bid * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const int nu = ncols / QD<T>::sb * QD<T>::per_sb;
    float acc = 0.f;
    // same arithmetic as the multi-token path, so verify matches decode bitwise
    for (int u = lane; u < nu; u += WARP) QDN<T>::template dot<1>(wr, u, xq, xd, ncols, &acc);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = accumulate ? y[row] + acc : acc;
}

// Multi-token int8 GEMV (speculative verify / batched decode, 2..16 tokens): the same
// weight unit is dotted with every token's x; repeated weight loads hit the
// same addresses back to back, so DRAM traffic stays one pass over W.
// xq: [NT][ncols] int8, xd: [NT][ncols/32]; y: [NT][nrows].
template <int T, int NT>
__device__ __forceinline__ void gemvq_nt_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                              const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                              const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = bid * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const int nu = ncols / QD<T>::sb * QD<T>::per_sb;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int u = lane; u < nu; u += WARP) QDN<T>::template dot<NT>(wr, u, xq, xd, ncols, acc);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) {
            float *dst = y + (size_t)t * nrows + row;
            *dst = accumulate ? *dst + v : v;
        }
    }
}

// ---- item T1: Q3_K / IQ3_S decode GEMV (begin) ----
// One-token (and Q3_K few-token dp4) GEMV for the 8-value-unit types. Lane l owns
// in-superblock unit l of every superblock (per_sb == WARP), exactly as in
// gemvq_impl / gemvq_nt_impl, and every (unit, token) term is the QD<T>::dot float
// expression, accumulated in superblock order, then the same warp_sum => bitwise
// identical to gemvq_impl<T> / gemvq_nt_impl<T, NT>. What changes: (1) the Q3_K
// scale nibble is read branch-free (kk < 8 selected a different byte through a
// divergent branch, which serialized a scale load before the weight loads), (2) U
// superblocks' loads are issued before their terms are computed (one 110-byte
// superblock per wave per step was too little memory-level parallelism):
// 17408x5120 q3_k 491 -> 590 GB/s, iq3_s 502 -> 583 GB/s (pure read ceiling ~595).
typedef uint32_t t1_u32a2 __attribute__((aligned(2)));
typedef uint16_t t1_u16a2 __attribute__((aligned(2)));
struct T1Q3W {
    uint32_t dh, slo, shi, q0, q1, h0, h1;
};
struct T1Q3Lane {
    int j, sh, lo_off, lo_sh, hi_off, hi_sh, q_off, h_off, bl;
    __device__ __forceinline__ T1Q3Lane(int lane) {
        const int n = lane >> 4, sg = lane & 3;
        j = (lane >> 2) & 3;
        const int kk = 8 * n + 2 * j + (sg >> 1);
        lo_off = 96 + (kk & 7);
        lo_sh = kk < 8 ? 0 : 4;
        hi_off = 96 + 8 + (kk & 3);
        hi_sh = 2 * (kk >> 2);
        q_off = 32 + 32 * n + 8 * sg;
        h_off = 8 * sg;
        sh = 4 * n + j;
        bl = 128 * n + 32 * j + 8 * sg;
    }
    __device__ __forceinline__ T1Q3W load(const uint8_t *blk) const {
        T1Q3W c;
        c.dh = *(const t1_u16a2 *)(blk + 108);
        c.slo = blk[lo_off];
        c.shi = blk[hi_off];
        c.q0 = *(const t1_u32a2 *)(blk + q_off);
        c.q1 = *(const t1_u32a2 *)(blk + q_off + 4);
        c.h0 = *(const t1_u32a2 *)(blk + h_off);
        c.h1 = *(const t1_u32a2 *)(blk + h_off + 4);
        return c;
    }
};

template <int U>
__device__ __forceinline__ void t1_q3k_1_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                              const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                              const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = bid * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const int nsb = ncols / 256;
    const T1Q3Lane L(lane);
    float acc = 0.f;
    auto term = [&](const T1Q3W &c, int2 xv, float xs) {
        const float d = __half2float(__ushort_as_half((uint16_t)c.dh));
        const int lo = (c.slo >> L.lo_sh) & 0xF;
        const int hi = (c.shi >> L.hi_sh) & 3;
        const float dl = d * ((lo | (hi << 4)) - 32);
        const int w0 = (c.q0 >> (2 * L.j)) & 0x03030303, w1 = (c.q1 >> (2 * L.j)) & 0x03030303;
        const int m0 = ((~c.h0) >> L.sh) & 0x01010101, m1 = ((~c.h1) >> L.sh) & 0x01010101;
        const int dv = dp4(w1, xv.y, dp4(w0, xv.x, 0)) - 4 * dp4(m1, xv.y, dp4(m0, xv.x, 0));
        return dl * xs * dv;
    };
    int b = 0;
    for (; b + U <= nsb; b += U) {
        T1Q3W c[U];
        int2 xv[U];
        float xs[U];
#pragma unroll
        for (int k = 0; k < U; ++k) {
            c[k] = L.load(wr + (size_t)(b + k) * 110);
            xv[k] = *(const int2 *)(xq + (b + k) * 256 + L.bl);
            xs[k] = xd[((b + k) * 256 + L.bl) >> 5];
        }
#pragma unroll
        for (int k = 0; k < U; ++k) acc += term(c[k], xv[k], xs[k]);
    }
    for (; b < nsb; ++b)
        acc += term(L.load(wr + (size_t)b * 110), *(const int2 *)(xq + b * 256 + L.bl), xd[(b * 256 + L.bl) >> 5]);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = accumulate ? y[row] + acc : acc;
}

// Q3_K few-token (the dp4 path; tokens share the decoded unit). Replaces
// gemvq_nt_impl<QT_Q3_K, NT>: 17408x5120 T=2 0.104 -> 0.073 ms, T=5 0.198 -> 0.116 ms.
template <int NT>
__device__ __forceinline__ void t1_q3k_nt_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                               const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                               const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = bid * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const int nsb = ncols / 256;
    const T1Q3Lane L(lane);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int b = 0; b < nsb; ++b) {
        const T1Q3W c = L.load(wr + (size_t)b * 110);
        int2 xv[NT];
        float xs[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            xv[t] = *(const int2 *)(xq + (size_t)t * ncols + b * 256 + L.bl);
            xs[t] = xd[(size_t)t * (ncols / 32) + ((b * 256 + L.bl) >> 5)];
        }
        const float d = __half2float(__ushort_as_half((uint16_t)c.dh));
        const int lo = (c.slo >> L.lo_sh) & 0xF;
        const int hi = (c.shi >> L.hi_sh) & 3;
        const float dl = d * ((lo | (hi << 4)) - 32);
        const int w0 = (c.q0 >> (2 * L.j)) & 0x03030303, w1 = (c.q1 >> (2 * L.j)) & 0x03030303;
        const int m0 = ((~c.h0) >> L.sh) & 0x01010101, m1 = ((~c.h1) >> L.sh) & 0x01010101;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int dv = dp4(w1, xv[t].y, dp4(w0, xv[t].x, 0)) - 4 * dp4(m1, xv[t].y, dp4(m0, xv[t].x, 0));
            acc[t] += dl * xs[t] * dv;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) {
            float *dst = y + (size_t)t * nrows + row;
            *dst = accumulate ? *dst + v : v;
        }
    }
}

struct T1I3W {
    uint32_t dh, sc, h, qa, sg;
};
template <int U>
__device__ __forceinline__ void t1_iq3s_1_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                               const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                               const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = bid * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const int nsb = ncols / 256;
    const int ib32 = lane >> 2, l = lane & 3;
    const int bl = 32 * ib32 + 8 * l;
    auto ld = [&](int b) {
        const uint8_t *blk = wr + (size_t)b * 110;
        T1I3W r;
        r.dh = *(const t1_u16a2 *)blk;
        r.sc = blk[106 + (ib32 >> 1)];
        r.h = blk[66 + ib32];
        r.qa = *(const t1_u16a2 *)(blk + 2 + 8 * ib32 + 2 * l);
        r.sg = blk[74 + 4 * ib32 + l];
        return r;
    };
    auto term = [&](const T1I3W &c, int2 xv, float xs) {
        const float d = __half2float(__ushort_as_half((uint16_t)c.dh));
        const float db = d * (1 + 2 * ((c.sc >> (4 * (ib32 & 1))) & 0xF));
        const uint32_t g1 = k_iq3s_grid[(c.qa & 0xFF) | ((c.h << (8 - 2 * l)) & 256)];
        const uint32_t g2 = k_iq3s_grid[(c.qa >> 8) | ((c.h << (7 - 2 * l)) & 256)];
        const uint32_t m1 = (((c.sg & 0xF) * 0x00204081u) & 0x01010101u) * 0xFFu;
        const uint32_t m2 = (((c.sg >> 4) * 0x00204081u) & 0x01010101u) * 0xFFu;
        const int v1 = (int)((g1 ^ m1) + (m1 & 0x01010101u));
        const int v2 = (int)((g2 ^ m2) + (m2 & 0x01010101u));
        return db * xs * dp4(v2, xv.y, dp4(v1, xv.x, 0));
    };
    float acc = 0.f;
    int b = 0;
    for (; b + U <= nsb; b += U) {
        T1I3W c[U];
        int2 xv[U];
        float xs[U];
#pragma unroll
        for (int k = 0; k < U; ++k) {
            c[k] = ld(b + k);
            xv[k] = *(const int2 *)(xq + (b + k) * 256 + bl);
            xs[k] = xd[((b + k) * 256 + bl) >> 5];
        }
#pragma unroll
        for (int k = 0; k < U; ++k) acc += term(c[k], xv[k], xs[k]);
    }
    for (; b < nsb; ++b) acc += term(ld(b), *(const int2 *)(xq + b * 256 + bl), xd[(b * 256 + bl) >> 5]);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = accumulate ? y[row] + acc : acc;
}
// ---- item T1 (end) ----


