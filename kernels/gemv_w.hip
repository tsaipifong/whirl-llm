// WHIRL device code (gfx1201 / RDNA 4): mid-batch GEMV on int8 WMMA (gemvw_nt*v1..4).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Mid-batch GEMV (verify / batched decode, 2..16 tokens) on the int8 matrix
// cores: v_wmma_i32_16x16x16_iu8 with A = x (16 token rows) and B = decoded
// weights (16 matrix rows), so each D[token][row] is an exact int32 dot of a
// k-slice. Integer dots are exact in any order, so only the float work has to
// follow the canonical one-token order of gemvq_mr_impl<T, 1, 1> (MRD):
//   * unit u = 64 weights (sub-blocks 2j, 2j+1 of superblock u / 4);
//   * slice s = u % 32 accumulates its units in increasing u with the MRD
//     float expression (same source text, same operand types);
//   * the 32 slices are reduced like warp_sum: pairs at xor 16, 8, 4, 2, 1.
// Wave w of 8 runs the units u = w (mod 8), i.e. slices w, w+8, w+16, w+24
// (slot k % 4 of its k-th unit): the xor-16 / xor-8 levels are in registers,
// the xor-4/2/1 levels are done through LDS. So the results equal the one-
// token GEMV bitwise (MTP verify == plain greedy decode).
// The k order inside a WMMA is free (the dot is exact), so each lane loads
// whole 16-byte weight / x pieces and both A and B use the same permutation.
// gfx12 fragments: A/B lane = (row or column lane % 16, k half lane / 16),
// D lane = column lane % 16, rows 8 * (lane / 16) + [0, 8).
// ---------------------------------------------------------------------------
typedef int wi2 __attribute__((ext_vector_type(2)));
typedef int wi8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ wi8 wmma_iu8(int a0, int a1, int b0, int b1, wi8 c) {
    const wi2 a = {a0, a1}, b = {b0, b1};
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}

// One unit of one tile: integer dots of both sub-blocks (dl, dh), their x
// sums (sl, sh; K-quants only) and the per-row float scales.
template <int T> struct WMU;

template <int T>
struct WMU_K {  // Q4_K / Q5_K
    static constexpr int per_sb = 4;
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * per_sb; }
    static constexpr int bytes = T == QT_Q4_K ? 144 : 176;
    // x fragment of one unit: 16 bytes of sub-block 2j (cols 16h..16h+16)
    // and 16 of sub-block 2j+1
    struct X {
        int4 a, c;
        wi8 sl, sh;
    };
    static __device__ __forceinline__ void load_x(const int8_t *xt, int u, int h, X &x) {
        const int base = (u >> 2) * 256 + 64 * (u & 3);
        x.a = *(const int4 *)(xt + base + 16 * h);
        x.c = *(const int4 *)(xt + base + 32 + 16 * h);
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        const int one = 0x01010101;
        x.sl = wmma_iu8(x.a.z, x.a.w, one, one, wmma_iu8(x.a.x, x.a.y, one, one, z));
        x.sh = wmma_iu8(x.c.z, x.c.w, one, one, wmma_iu8(x.c.x, x.c.y, one, one, z));
    }
    template <int NV>
    static __device__ __forceinline__ void run(const uint8_t *row, int u, int h, const X &x, const float *xd, int nb32,
                                               int t0, int ntm1, float *acc) {
        const int b = u >> 2, j = u & 3;
        const uint8_t *blk = row + (size_t)b * bytes;
        // header in one 16-byte load: d, dmin, scales[12]; scale_min_k4 for
        // sub-blocks 2j, 2j+1 without byte loads or branches (j is wave-uniform)
        const uint4 hd = *(const uint4 *)blk;
        const float d = __half2float(__ushort_as_half((uint16_t)(hd.x & 0xFFFF)));
        const float dmin = __half2float(__ushort_as_half((uint16_t)(hd.x >> 16)));
        const int sel = 16 * (j & 1);
        const uint32_t A = hd.y >> sel, B = hd.z >> sel, C = hd.w >> sel;  // sc[k], sc[k+4], sc[k+8], k = 2(j&1)
        int s0, m0, s1, m1;
        if (j < 2) {
            s0 = A & 63;
            s1 = (A >> 8) & 63;
            m0 = B & 63;
            m1 = (B >> 8) & 63;
        } else {
            s0 = (C & 0xF) | (((A >> 6) & 3) << 4);
            s1 = ((C >> 8) & 0xF) | (((A >> 14) & 3) << 4);
            m0 = ((C >> 4) & 0xF) | (((B >> 6) & 3) << 4);
            m1 = ((C >> 12) & 0xF) | (((B >> 14) & 3) << 4);
        }
        const float ds0 = d * s0, ds1 = d * s1, dm0 = dmin * m0, dm1 = dmin * m1;
        const uint4 qv = *(const uint4 *)(blk + (T == QT_Q4_K ? 16 : 48) + 32 * j + 16 * h);
        const uint32_t q[4] = {qv.x, qv.y, qv.z, qv.w};
        int lo[4], hi[4];
        if constexpr (T == QT_Q4_K) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                lo[k] = q[k] & 0x0F0F0F0F;
                hi[k] = (q[k] >> 4) & 0x0F0F0F0F;
            }
        } else {
            const uint4 hv = *(const uint4 *)(blk + 16 + 16 * h);
            const uint32_t hb[4] = {hv.x, hv.y, hv.z, hv.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                lo[k] = (q[k] & 0x0F0F0F0F) | (((hb[k] >> (2 * j)) & 0x01010101) << 4);
                hi[k] = ((q[k] >> 4) & 0x0F0F0F0F) | (((hb[k] >> (2 * j + 1)) & 0x01010101) << 4);
            }
        }
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        const wi8 cl = wmma_iu8(x.a.z, x.a.w, lo[2], lo[3], wmma_iu8(x.a.x, x.a.y, lo[0], lo[1], z));
        const wi8 ch = wmma_iu8(x.c.z, x.c.w, hi[2], hi[3], wmma_iu8(x.c.x, x.c.y, hi[0], hi[1], z));
        const int xi = (b * 256 + 64 * j) >> 5;
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const int t = min(t0 + v, ntm1);
            const float *xdt = xd + (size_t)t * nb32 + xi;
            const float x0 = xdt[0], x1 = xdt[1];
            const int dl = cl[v], dh = ch[v], sl = x.sl[v], sh = x.sh[v];
            acc[v] += x0 * (ds0 * dl - dm0 * sl) + x1 * (ds1 * dh - dm1 * sh);
        }
    }
};
template <> struct WMU<QT_Q4_K> : WMU_K<QT_Q4_K> {};
template <> struct WMU<QT_Q5_K> : WMU_K<QT_Q5_K> {};

template <> struct WMU<QT_IQ4_XS> {
    static constexpr int per_sb = 4;
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * per_sb; }
    // sub-block 2p: values 8h..8h+8 (low nibbles) and 16+8h.. (high); same
    // for sub-block 2p+1 -> four 8-byte x pieces
    struct X {
        int2 a0, a1, c0, c1;
    };
    static __device__ __forceinline__ void load_x(const int8_t *xt, int u, int h, X &x) {
        const int base = (u >> 2) * 256 + 64 * (u & 3) + 8 * h;
        x.a0 = *(const int2 *)(xt + base);
        x.a1 = *(const int2 *)(xt + base + 16);
        x.c0 = *(const int2 *)(xt + base + 32);
        x.c1 = *(const int2 *)(xt + base + 48);
    }
    template <int NV>
    static __device__ __forceinline__ void run(const uint8_t *row, int u, int h, const X &x, const float *xd, int nb32,
                                               int t0, int ntm1, float *acc) {
        const int b = u >> 2, pp = u & 3;
        const uint8_t *blk = row + (size_t)b * 136;
        const uint32_t hdr = *(const uint32_t *)blk;
        const float d = __half2float(__ushort_as_half((uint16_t)(hdr & 0xFFFF)));
        const uint32_t sh = hdr >> 16;
        const int ib0 = 2 * pp, ib1 = 2 * pp + 1;
        const int ls0 = (blk[4 + pp] & 0xF) | (((sh >> (2 * ib0)) & 3) << 4);
        const int ls1 = (blk[4 + pp] >> 4) | (((sh >> (2 * ib1)) & 3) << 4);
        const float dl0 = d * (ls0 - 32), dl1 = d * (ls1 - 32);
        const uint2 g0 = *(const uint2 *)(blk + 8 + 32 * pp + 8 * h);
        const uint2 g1 = *(const uint2 *)(blk + 24 + 32 * pp + 8 * h);
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        wi8 c0 = wmma_iu8(x.a0.x, x.a0.y, iq4_lut4(g0.x & 0x0F0F0F0F), iq4_lut4(g0.y & 0x0F0F0F0F), z);
        c0 = wmma_iu8(x.a1.x, x.a1.y, iq4_lut4((g0.x >> 4) & 0x0F0F0F0F), iq4_lut4((g0.y >> 4) & 0x0F0F0F0F), c0);
        wi8 c1 = wmma_iu8(x.c0.x, x.c0.y, iq4_lut4(g1.x & 0x0F0F0F0F), iq4_lut4(g1.y & 0x0F0F0F0F), z);
        c1 = wmma_iu8(x.c1.x, x.c1.y, iq4_lut4((g1.x >> 4) & 0x0F0F0F0F), iq4_lut4((g1.y >> 4) & 0x0F0F0F0F), c1);
        const int xi = (b * 256 + 64 * pp) >> 5;
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const int t = min(t0 + v, ntm1);
            const float *xdt = xd + (size_t)t * nb32 + xi;
            const float x0 = xdt[0], x1 = xdt[1];
            const int s0 = c0[v], s1 = c1[v];
            acc[v] += dl0 * x0 * s0 + dl1 * x1 * s1;
        }
    }
};

// MXFP4: the IQ4_XS WMMA unit with power-of-2 scales (same float expression
// as MRD<QT_MXFP4>, so gemvw == gemvq bitwise).
template <> struct WMU<QT_MXFP4> {
    static constexpr int per_sb = 4;
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * per_sb; }
    struct X {
        int2 a0, a1, c0, c1;
    };
    static __device__ __forceinline__ void load_x(const int8_t *xt, int u, int h, X &x) {
        const int base = (u >> 2) * 256 + 64 * (u & 3) + 8 * h;
        x.a0 = *(const int2 *)(xt + base);
        x.a1 = *(const int2 *)(xt + base + 16);
        x.c0 = *(const int2 *)(xt + base + 32);
        x.c1 = *(const int2 *)(xt + base + 48);
    }
    template <int NV>
    static __device__ __forceinline__ void run(const uint8_t *row, int u, int h, const X &x, const float *xd, int nb32,
                                               int t0, int ntm1, float *acc) {
        const int b = u >> 2, pp = u & 3;
        const uint8_t *blk = row + (size_t)b * 136;
        const uint32_t ee = *(const uint16_t *)(blk + 2 * pp);
        const float dl0 = e8m0_half(ee & 0xFF), dl1 = e8m0_half(ee >> 8);
        const uint2 g0 = *(const uint2 *)(blk + 8 + 32 * pp + 8 * h);
        const uint2 g1 = *(const uint2 *)(blk + 24 + 32 * pp + 8 * h);
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        wi8 c0 = wmma_iu8(x.a0.x, x.a0.y, mx_lut4(g0.x & 0x0F0F0F0F), mx_lut4(g0.y & 0x0F0F0F0F), z);
        c0 = wmma_iu8(x.a1.x, x.a1.y, mx_lut4((g0.x >> 4) & 0x0F0F0F0F), mx_lut4((g0.y >> 4) & 0x0F0F0F0F), c0);
        wi8 c1 = wmma_iu8(x.c0.x, x.c0.y, mx_lut4(g1.x & 0x0F0F0F0F), mx_lut4(g1.y & 0x0F0F0F0F), z);
        c1 = wmma_iu8(x.c1.x, x.c1.y, mx_lut4((g1.x >> 4) & 0x0F0F0F0F), mx_lut4((g1.y >> 4) & 0x0F0F0F0F), c1);
        const int xi = (b * 256 + 64 * pp) >> 5;
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const int t = min(t0 + v, ntm1);
            const float *xdt = xd + (size_t)t * nb32 + xi;
            const float x0 = xdt[0], x1 = xdt[1];
            const int s0 = c0[v], s1 = c1[v];
            acc[v] += dl0 * x0 * s0 + dl1 * x1 * s1;
        }
    }
};

// Q6_K keeps its canonical unit (QD<Q6_K>: u = 8b + 4n + g, positions
// 128n + 32q + 8g + [0, 8) of quarter q, per-quarter 8-value int dots).
// An 8-value dot is half a WMMA k-slice: A holds quarter q (lanes h = 0) and
// quarter q+1 (h = 1) of x, and B gives quarter q only to lanes h = 0 (the
// other half zero), then quarter q+1 only to h = 1. Weights enter as signed
// q - 32, so the dot equals dv - 32 * sv exactly and no x sums are needed.
// Lane half h decodes quarters h and h + 2 (one ql byte run, two nibbles).
__device__ __forceinline__ uint2 ld64a2(const uint8_t *p) {  // 2-byte aligned 8-byte load
    uint2 v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 8);
    return v;
}
template <> struct WMU<QT_Q6_K> {
    static constexpr int per_sb = 8;
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * per_sb; }
    struct X {
        int2 a01, a23;  // lane half h: quarter h, quarter h + 2
    };
    static __device__ __forceinline__ void load_x(const int8_t *xt, int u, int h, X &x) {
        const int base = (u >> 3) * 256 + 128 * ((u >> 2) & 1) + 8 * (u & 3) + 32 * h;
        x.a01 = *(const int2 *)(xt + base);
        x.a23 = *(const int2 *)(xt + base + 64);
    }
    template <int NV>
    static __device__ __forceinline__ void run(const uint8_t *row, int u, int h, const X &x, const float *xd, int nb32,
                                               int t0, int ntm1, float *acc) {
        const int b = u >> 3, n = (u >> 2) & 1, g = u & 3;
        const uint8_t *blk = row + (size_t)b * 210;
        const uint2 l = ld64a2(blk + 64 * n + 8 * g + 32 * h);
        const uint2 hq = ld64a2(blk + 128 + 32 * n + 8 * g);
        const float dd = h2f(blk + 208);
        const uint2 scw = ld64a2(blk + 192 + 8 * n);  // sc[8n .. 8n + 8]
        const int is = g >> 1;
        float scf[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint32_t word = (is + 2 * q) < 4 ? scw.x : scw.y;
            scf[q] = (float)(int8_t)(word >> (8 * ((is + 2 * q) & 3)));
        }
        // quarter h (low nibbles) and h + 2 (high nibbles), bytewise q - 32
        int wq[2][2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int qq = h + 2 * e;
            const uint32_t L[2] = {l.x, l.y}, H[2] = {hq.x, hq.y};
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const uint32_t v = ((L[k] >> (4 * e)) & 0x0F0F0F0Fu) | (((H[k] >> (2 * qq)) & 0x03030303u) << 4);
                wq[e][k] = (int)(((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
            }
        }
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        const int m0 = h == 0 ? -1 : 0, m1 = ~m0;  // quarter q -> h = 0 lanes, q + 1 -> h = 1
        const wi8 c0 = wmma_iu8(x.a01.x, x.a01.y, wq[0][0] & m0, wq[0][1] & m0, z);
        const wi8 c1 = wmma_iu8(x.a01.x, x.a01.y, wq[0][0] & m1, wq[0][1] & m1, z);
        const wi8 c2 = wmma_iu8(x.a23.x, x.a23.y, wq[1][0] & m0, wq[1][1] & m0, z);
        const wi8 c3 = wmma_iu8(x.a23.x, x.a23.y, wq[1][0] & m1, wq[1][1] & m1, z);
        const int xi = (b * 256 + 128 * n) >> 5;  // + quarter
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const int t = min(t0 + v, ntm1);
            const float4 xs = *(const float4 *)(xd + (size_t)t * nb32 + xi);
            const int dv[4] = {c0[v], c1[v], c2[v], c3[v]};
            const float xsq[4] = {xs.x, xs.y, xs.z, xs.w};
            float total = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) total += scf[q] * xsq[q] * (float)dv[q];
            acc[v] += dd * total;
        }
    }
};

// IQ4_NL keeps QD<IQ4_NL>: unit u = 2 * block + hh (16 values: low nibbles of
// qs[8hh..8hh+8] -> 8hh.., high -> 16+8hh..): exactly one WMMA k-slice, lane
// half h = 0 the low nibbles, h = 1 the high ones.
template <> struct WMU<QT_IQ4_NL> {
    static __device__ __forceinline__ int units(int ncols) { return ncols / 32 * 2; }
    struct X {
        int2 a;
    };
    static __device__ __forceinline__ void load_x(const int8_t *xt, int u, int h, X &x) {
        x.a = *(const int2 *)(xt + (u >> 1) * 32 + 8 * (u & 1) + 16 * h);
    }
    template <int NV>
    static __device__ __forceinline__ void run(const uint8_t *row, int u, int h, const X &x, const float *xd, int nb32,
                                               int t0, int ntm1, float *acc) {
        const int b = u >> 1, hh = u & 1;
        const uint8_t *blk = row + (size_t)b * 18;
        const float d = h2f(blk);
        const uint2 q = ld64a2(blk + 2 + 8 * hh);
        const int sh = 4 * h;
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        const wi8 c = wmma_iu8(x.a.x, x.a.y, iq4_lut4((q.x >> sh) & 0x0F0F0F0F), iq4_lut4((q.y >> sh) & 0x0F0F0F0F), z);
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const int t = min(t0 + v, ntm1);
            const int dv = c[v];
            acc[v] += d * xd[(size_t)t * nb32 + b] * dv;
        }
    }
};

// 8-value units (Q3_K, IQ3_S, Q8_0: QD<T> unit = 8 weights, one float scale
// times one int dot): half a WMMA k-slice. Two units share a WMMA pair: lane
// half h = 0 carries unit u0 (A: its x, B: its weights), h = 1 unit u1; the
// first WMMA masks B to h = 0 (-> dot of u0), the second to h = 1 (-> u1).
// dec(row, u, wv, sc) gives the lane's 8 signed weights and the unit scale
// (the dl factor of QD<T>::dot), xpos(u) the unit's first column.
template <int T> struct WMU8;
template <> struct WMU8<QT_Q3_K> {
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * 32; }
    static __device__ __forceinline__ int xpos(int u) {
        const int b = u >> 5, n = (u >> 4) & 1, j = (u >> 2) & 3, sg = u & 3;
        return b * 256 + 128 * n + 32 * j + 8 * sg;
    }
    static __device__ __forceinline__ void dec(const uint8_t *row, int u, int2 &wv, float &dl) {
        const int b = u >> 5, n = (u >> 4) & 1, j = (u >> 2) & 3, sg = u & 3;
        const uint8_t *blk = row + (size_t)b * 110;
        const uint8_t *hm = blk, *qs = blk + 32, *sb8 = blk + 96;
        const float d = h2f(blk + 108);
        const int kk = 8 * n + 2 * j + (sg >> 1);
        const int lo = (kk < 8) ? (sb8[kk] & 0xF) : (sb8[kk - 8] >> 4);
        const int hi = (sb8[8 + (kk & 3)] >> (2 * (kk >> 2))) & 3;
        dl = d * ((lo | (hi << 4)) - 32);
        const uint2 q = ld64a2(qs + 32 * n + 8 * sg), hmv = ld64a2(hm + 8 * sg);
        const int sh = 4 * n + j;
        const uint32_t w0 = (q.x >> (2 * j)) & 0x03030303u, w1 = (q.y >> (2 * j)) & 0x03030303u;
        const uint32_t m0 = ((~hmv.x) >> sh) & 0x01010101u, m1 = ((~hmv.y) >> sh) & 0x01010101u;
        wv.x = (int)(((w0 | 0x80808080u) - (m0 << 2)) ^ 0x80808080u);  // bytewise w - 4m
        wv.y = (int)(((w1 | 0x80808080u) - (m1 << 2)) ^ 0x80808080u);
    }
};
template <> struct WMU8<QT_IQ3_S> {
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * 32; }
    static __device__ __forceinline__ int xpos(int u) { return (u >> 5) * 256 + 32 * ((u >> 2) & 7) + 8 * (u & 3); }
    static __device__ __forceinline__ void dec(const uint8_t *row, int u, int2 &wv, float &db) {
        const int b = u >> 5, ib32 = (u >> 2) & 7, l = u & 3;
        const uint8_t *blk = row + (size_t)b * 110;
        const float d = h2f(blk);
        const uint8_t *qs = blk + 2, *qh = blk + 66, *signs = blk + 74, *scales = blk + 106;
        db = d * (1 + 2 * ((scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xF));
        const uint8_t hb = qh[ib32];
        const uint32_t g1 = k_iq3s_grid[qs[8 * ib32 + 2 * l] | ((hb << (8 - 2 * l)) & 256)];
        const uint32_t g2 = k_iq3s_grid[qs[8 * ib32 + 2 * l + 1] | ((hb << (7 - 2 * l)) & 256)];
        const uint32_t sg = signs[4 * ib32 + l];
        const uint32_t m1 = (((sg & 0xF) * 0x00204081u) & 0x01010101u) * 0xFFu;
        const uint32_t m2 = (((sg >> 4) * 0x00204081u) & 0x01010101u) * 0xFFu;
        wv.x = (int)((g1 ^ m1) + (m1 & 0x01010101u));
        wv.y = (int)((g2 ^ m2) + (m2 & 0x01010101u));
    }
};
template <> struct WMU8<QT_Q8_0> {
    static __device__ __forceinline__ int units(int ncols) { return ncols / 32 * 4; }
    static __device__ __forceinline__ int xpos(int u) { return (u >> 2) * 32 + 8 * (u & 3); }
    static __device__ __forceinline__ void dec(const uint8_t *row, int u, int2 &wv, float &d) {
        const uint8_t *blk = row + (size_t)(u >> 2) * 34;
        d = h2f(blk);
        const uint2 q = ld64a2(blk + 2 + 8 * (u & 3));
        wv.x = (int)q.x;
        wv.y = (int)q.y;
    }
};

// Paired-unit variant of gemvw_impl for WMU8 types: step pairs (s, s+1) =
// units (u, u + 8) share A (lane half h carries unit u + 8h) and a WMMA pair.
template <int T, int NT, int TILES>
__device__ __forceinline__ void gemvw8_pair(const uint8_t *const *wr, const int8_t *xt, const float *xd, int nb32,
                                            int h, int n, int u0, bool has1, float (*acc0)[8], float (*acc1)[8]) {
    constexpr int NV = NT < 8 ? NT : 8;
    const int t0 = 8 * h;
    const int mk0 = h == 0 ? -1 : 0, mk1 = ~mk0;
    const int u1 = has1 ? u0 + 8 : u0;
    const int uh = h ? u1 : u0;
    const int2 xa = *(const int2 *)(xt + WMU8<T>::xpos(uh));
    const int xi0 = WMU8<T>::xpos(u0) >> 5, xi1 = WMU8<T>::xpos(u1) >> 5;
    const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt) {
        int2 wv;
        float sc;
        WMU8<T>::dec(wr[tt], uh, wv, sc);
        // row n's scale for u0 (lane n) and u1 (lane n + 16)
        const float sc0 = __shfl(sc, n, WARP), sc1 = __shfl(sc, n + 16, WARP);
        const wi8 c0 = wmma_iu8(xa.x, xa.y, wv.x & mk0, wv.y & mk0, z);
        const wi8 c1 = wmma_iu8(xa.x, xa.y, wv.x & mk1, wv.y & mk1, z);
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const int t = min(t0 + v, NT - 1);
            const int dv0 = c0[v];
            acc0[tt][v] += sc0 * xd[(size_t)t * nb32 + xi0] * dv0;
        }
        if (has1) {
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const int t = min(t0 + v, NT - 1);
                const int dv1 = c1[v];
                acc1[tt][v] += sc1 * xd[(size_t)t * nb32 + xi1] * dv1;
            }
        }
    }
}

template <int T, int NT, int TILES, int PIN>
__device__ __forceinline__ void gemvw8_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                            const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                            const int *__restrict__ skip, int bid) {
    constexpr int ROWS = 16 * TILES;
    constexpr int NV = NT < 8 ? NT : 8;
    __shared__ float red[8][NT][ROWS];
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP, h = lane >> 4, n = lane & 15;
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP);
    const int r0 = bid * ROWS;
    const uint8_t *wr[TILES];
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt) wr[tt] = W + (size_t)min(r0 + tt * 16 + n, nrows - 1) * row_bytes;
    const int8_t *xt = xq + (size_t)min(n, NT - 1) * ncols;
    const int nb32 = ncols / 32;
    const int nu = WMU8<T>::units(ncols);
    const int t0 = 8 * h;
    // slot-major so a slot is a float[TILES][8] block
    float acc[4][TILES][8];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
            for (int v = 0; v < 8; ++v) acc[q][tt][v] = 0.f;
    int k = 0;
    for (; w + 8 * (k + 3) < nu; k += 4) {
        gemvw8_pair<T, NT, TILES>(wr, xt, xd, nb32, h, n, w + 8 * k, true, acc[0], acc[1]);
        if (PIN == 1) {
#pragma unroll
            for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
                for (int v = 0; v < NV; ++v) asm volatile("" : "+v"(acc[0][tt][v]), "+v"(acc[1][tt][v]));
        }
        gemvw8_pair<T, NT, TILES>(wr, xt, xd, nb32, h, n, w + 8 * (k + 2), true, acc[2], acc[3]);
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
            for (int v = 0; v < NV; ++v)
                asm volatile("" : "+v"(acc[0][tt][v]), "+v"(acc[1][tt][v]), "+v"(acc[2][tt][v]), "+v"(acc[3][tt][v]));
    }
    if (w + 8 * k < nu) gemvw8_pair<T, NT, TILES>(wr, xt, xd, nb32, h, n, w + 8 * k, w + 8 * (k + 1) < nu, acc[0], acc[1]);
    if (w + 8 * (k + 2) < nu) gemvw8_pair<T, NT, TILES>(wr, xt, xd, nb32, h, n, w + 8 * (k + 2), false, acc[2], acc[3]);
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const int t = t0 + v;
            if (t < NT) red[w][t][tt * 16 + n] = (acc[0][tt][v] + acc[2][tt][v]) + (acc[1][tt][v] + acc[3][tt][v]);
        }
    __syncthreads();
    for (int i = threadIdx.x; i < NT * ROWS; i += blockDim.x) {
        const int t = i / ROWS, nn = i % ROWS, row = r0 + nn;
        const float s = ((red[0][t][nn] + red[4][t][nn]) + (red[2][t][nn] + red[6][t][nn])) +
                        ((red[1][t][nn] + red[5][t][nn]) + (red[3][t][nn] + red[7][t][nn]));
        if (row < nrows) {
            float *dst = y + (size_t)t * nrows + row;
            *dst = accumulate ? *dst + s : s;
        }
    }
}

template <int T, int NT, int TILES, int PIN>
__device__ __forceinline__ void gemvw_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                           const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                           const int *__restrict__ skip, int bid, int ystride = 0) {
    constexpr int ROWS = 16 * TILES;
    constexpr int NV = NT < 8 ? NT : 8;  // D rows (tokens) per lane half
    __shared__ float red[8][NT][ROWS];
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP, h = lane >> 4, n = lane & 15;
    // wave-uniform unit index -> scalar block math and uniform scale branches
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP);
    const int r0 = bid * ROWS;
    const uint8_t *wr[TILES];
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt) wr[tt] = W + (size_t)min(r0 + tt * 16 + n, nrows - 1) * row_bytes;
    const int8_t *xt = xq + (size_t)min(n, NT - 1) * ncols;  // A row = token n
    const int nb32 = ncols / 32;
    const int nu = WMU<T>::units(ncols);
    const int t0 = 8 * h;
    float acc[TILES][4][NV];
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int v = 0; v < NV; ++v) acc[tt][q][v] = 0.f;
    int k = 0;
    for (; w + 8 * (k + 3) < nu; k += 4) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int u = w + 8 * (k + s);
            typename WMU<T>::X x;
            WMU<T>::load_x(xt, u, h, x);
#pragma unroll
            for (int tt = 0; tt < TILES; ++tt) WMU<T>::template run<NV>(wr[tt], u, h, x, xd, nb32, t0, NT - 1, acc[tt][s]);
            // Pin this unit's results here: the four slots are independent, and
            // without this the compiler interleaves all four units' WMMAs and
            // keeps every fragment live (spills).
            if (PIN == 1 || (s & 1)) {  // PIN units in flight per wave
#pragma unroll
                for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
                    for (int v = 0; v < NV; ++v) {
                        if (PIN == 1)
                            asm volatile("" : "+v"(acc[tt][s][v]));
                        else
                            asm volatile("" : "+v"(acc[tt][s - 1][v]), "+v"(acc[tt][s][v]));
                    }
            }
        }
    }
#pragma unroll
    for (int s = 0; s < 3; ++s) {
        const int u = w + 8 * (k + s);
        if (u < nu) {
            typename WMU<T>::X x;
            WMU<T>::load_x(xt, u, h, x);
#pragma unroll
            for (int tt = 0; tt < TILES; ++tt) WMU<T>::template run<NV>(wr[tt], u, h, x, xd, nb32, t0, NT - 1, acc[tt][s]);
        }
    }
    // slots 0..3 = slices w, w+8, w+16, w+24: xor-16 then xor-8 level
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const int t = t0 + v;
            if (t < NT) red[w][t][tt * 16 + n] = (acc[tt][0][v] + acc[tt][2][v]) + (acc[tt][1][v] + acc[tt][3][v]);
        }
    __syncthreads();
    // xor 4, 2, 1 over the waves
    for (int i = threadIdx.x; i < NT * ROWS; i += blockDim.x) {
        const int t = i / ROWS, nn = i % ROWS, row = r0 + nn;
        const float s = ((red[0][t][nn] + red[4][t][nn]) + (red[2][t][nn] + red[6][t][nn])) +
                        ((red[1][t][nn] + red[5][t][nn]) + (red[3][t][nn] + red[7][t][nn]));
        if (row < nrows) {
            float *dst = y + (size_t)t * (ystride ? ystride : nrows) + row;
            *dst = accumulate ? *dst + s : s;
        }
    }
}

// Variants V = (tiles, units in flight): 1 = (1, 1), 2 = (2, 1), 3 = (1, 2),
// 4 = (2, 2); the host picks one per (type, token count), 16 * tiles rows
// per block.
#define GEMVW_ENTRY(NT, V, TILES, PIN, SUF, T)                                                  \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvw_nt##NT##v##V##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,      \
                                   const float *xd, float *y, int ncols, int nrows,              \
                                   int accumulate, const int *skip) {                            \
        gemvw_impl<T, NT, TILES, PIN>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x);  \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvw_nt##NT##v##V##_##SUF(GV_PARAMS) {                            \
        { const GvSel gs = gv_sel(g); gemvw_impl<T, NT, TILES, PIN>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }  \
    }
#define GEMVW8_ENTRY(NT, V, TILES, PIN, SUF, T)                                                 \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvw_nt##NT##v##V##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,      \
                                   const float *xd, float *y, int ncols, int nrows,              \
                                   int accumulate, const int *skip) {                            \
        gemvw8_impl<T, NT, TILES, PIN>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x); \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvw_nt##NT##v##V##_##SUF(GV_PARAMS) {                            \
        { const GvSel gs = gv_sel(g); gemvw8_impl<T, NT, TILES, PIN>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); } \
    }
#define GEMVW_TYPES(NT, V, TILES, PIN)                  \
    GEMVW_ENTRY(NT, V, TILES, PIN, q4_k, QT_Q4_K)       \
    GEMVW_ENTRY(NT, V, TILES, PIN, q5_k, QT_Q5_K)       \
    GEMVW_ENTRY(NT, V, TILES, PIN, iq4_xs, QT_IQ4_XS)   \
    GEMVW_ENTRY(NT, V, TILES, PIN, mxfp4, QT_MXFP4)     \
    GEMVW_ENTRY(NT, V, TILES, PIN, q6_k, QT_Q6_K)       \
    GEMVW_ENTRY(NT, V, TILES, PIN, iq4_nl, QT_IQ4_NL)   \
    GEMVW8_ENTRY(NT, V, TILES, PIN, q3_k, QT_Q3_K)
// (v3 = (1, 2) measured never better than v1 / v2 on the 27B; not built)
// Output head of a wide (17..32-row) verify: 16 tokens over a range of head rows at a time,
// with the logits row stride passed separately (ystride = n_vocab), so the host can run the
// two token passes of each row range back to back and the second pass reads the weights
// from cache. Same kernel body as gemvw_nt16v2_q6_k.
extern "C" __global__ void __launch_bounds__(256)
    gemvw_nt16v2s_q6_k(const uint8_t *W, uint64_t row_bytes, const int8_t *xq, const float *xd, float *y, int ncols, int nrows,
                       int ystride, int accumulate, const int *skip) {
    gemvw_impl<QT_Q6_K, 16, 2, 1>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x, ystride);
}
#define GEMVW_NT(NT) GEMVW_TYPES(NT, 1, 1, 1) GEMVW_TYPES(NT, 2, 2, 1) GEMVW_TYPES(NT, 4, 2, 2)
GEMVW_NT(2)
GEMVW_NT(3)
GEMVW_NT(4)
GEMVW_NT(5)
GEMVW_NT(6)
GEMVW_NT(7)
GEMVW_NT(8)
GEMVW_NT(9)
GEMVW_NT(10)
GEMVW_NT(11)
GEMVW_NT(12)
GEMVW_NT(13)
GEMVW_NT(14)
GEMVW_NT(15)
GEMVW_NT(16)
