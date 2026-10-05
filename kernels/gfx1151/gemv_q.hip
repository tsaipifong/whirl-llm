// WHIRL device code (gfx1151 / RDNA 3.5): int8 decode GEMV (one token, grouped ABI) and multi-token WMMA GEMV.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

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
    const int q = i < n ? __float2int_rn(v * id) : 0;
    const int sum = warp_sum_i(q);
    if (i < n) {
        xq[i] = (int8_t)q;
        if ((i & 31) == 0) xd[i >> 5] = pk_make(d, sum);
    }
}

__device__ __forceinline__ int dp4(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }


__device__ __forceinline__ uint32_t ld32u(const uint8_t *p) {  // 2-byte aligned (all call sites)
    uint32_t v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 4);
    return v;
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
        const float x0 = xdv(xd + ((b * 256 + 64 * j) >> 5)), x1 = xdv(xd + ((b * 256 + 64 * j + 32) >> 5));
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
        const float x0 = xdv(xd + ((b * 256 + 64 * j) >> 5)), x1 = xdv(xd + ((b * 256 + 64 * j + 32) >> 5));
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
        return dl * xdv(xd + ((b * 256 + 32 * ib) >> 5)) * acc;
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
        return d * (ls - 32) * xdv(xd + ((b * 256 + 32 * ib) >> 5)) * acc;
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
            total += (float)sc[is + 2 * quarter] * xdv(xd + ((base + 32 * quarter) >> 5)) * (float)(dv - 32 * sv);
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
        return d * xdv(xd + b) * acc;
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
        return d * xdv(xd + b) * dp4(w1, xv.y, dp4(w0, xv.x, 0));
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
        return dl * xdv(xd + (base >> 5)) * dv;
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
        return db * xdv(xd + (base >> 5)) * dp4(v2, xv.y, dp4(v1, xv.x, 0));
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
        acc[t] += xdv(xdt) * (ds0 * dl - dm0 * sl) + xdv(xdt + 1) * (ds1 * dh - dm1 * sh);
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

// ---- grouped one-token GEMV ABI ----
// One launch of a one-token gemvq_<type> covers up to 3 matrices that share the input:
// blocks [0, b1) run segment 0, [b1, b2) segment 1, the rest segment 2, each with its
// own (W, row_bytes, y, nrows) and segment-local block index (bitwise the same rows).
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
// ---- grouped ABI (end) ----
template <int T>
__device__ void gemvq_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                           const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                           const int *__restrict__ skip, int bid = -1) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = (bid < 0 ? (int)blockIdx.x : bid) * (blockDim.x / WARP) + threadIdx.x / WARP;
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
__device__ void gemvq_nt_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                              const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                              const int *__restrict__ skip, int bid = -1) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = (bid < 0 ? (int)blockIdx.x : bid) * (blockDim.x / WARP) + threadIdx.x / WARP;
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
