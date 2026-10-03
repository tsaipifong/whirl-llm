// WHIRL device code (gfx1201 / RDNA 4): pipelined mid-batch GEMV on int8 WMMA (gemvw_nt*v5..8).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---- item V: pipelined mid-batch GEMV (begin) ----
// gw2_impl: the gemvw_impl lane / unit / slot mapping, float expressions and xor-16/8 +
// LDS xor-4/2/1 reduction (so bitwise == gemvq 1-token), restructured for overlap:
//  * u = w + 8k, so j = u & 3 (Q4_K/Q5_K scale class, IQ4_XS/MXFP4 pair, Q6_K n/g) is a wave
//    constant: the loop is specialised per class, no branches inside;
//  * the loads of unit k + DEPTH are issued before unit k is computed (ring of 2 / 4);
//  * xd: one dword per lane (token lane % 16) + shuffles instead of 8 x b64 per unit;
//  * LUT nibble blend without v_mul_lo_u32.
// Variants (host table gemv_w): v5 = (1 tile, depth 1), v6 = (1, 3), v7 = (2, 1), v8 = (2, 2);
// Q3_K / IQ3_S: v5 = gw8 (8-value units, 1 tile, IQ3_S grid in LDS).
template <int T, int TILES, int NV, int XDS> struct GU;

template <int TILES, int NV, int XDS, int QT> struct GU_K {
    static constexpr int bytes = QT == QT_Q4_K ? 144 : 176;
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * 4; }
    struct P {
        int4 a, c;
        uint4 hd[TILES], qv[TILES];
        uint4 hb[QT == QT_Q5_K ? TILES : 1];
        float2 xs[XDS ? 1 : NV];
    };
    static __device__ __forceinline__ void fetch(P &p, const uint8_t *const *wr, const int8_t *xt, const float *xd, int nb32,
                                                 int u, int h, int t0, int ntm1, int lane) {
        const int b = u >> 2, j = u & 3;
        const int base = b * 256 + 64 * j;
        p.a = *(const int4 *)(xt + base + 16 * h);
        p.c = *(const int4 *)(xt + base + 32 + 16 * h);
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint8_t *blk = wr[tt] + (size_t)b * bytes;
            p.hd[tt] = *(const uint4 *)blk;
            p.qv[tt] = *(const uint4 *)(blk + (QT == QT_Q4_K ? 16 : 48) + 32 * j + 16 * h);
            if constexpr (QT == QT_Q5_K) p.hb[tt] = *(const uint4 *)(blk + 16 + 16 * h);
        }
        const int xi = base >> 5;
        if constexpr (XDS) {
            p.xs[0].x = xd[(size_t)min(lane & 15, ntm1) * nb32 + xi + (lane >> 4)];
        } else {
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const int t = min(t0 + v, ntm1);
                p.xs[v] = *(const float2 *)(xd + (size_t)t * nb32 + xi);
            }
        }
    }
    template <bool J2>
    static __device__ __forceinline__ void compute(const P &p, int u, int h, int t0, float (*acc)[NV]) {
        const int j = u & 3;
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        const int one = 0x01010101;
        const wi8 xsl = wmma_iu8(p.a.z, p.a.w, one, one, wmma_iu8(p.a.x, p.a.y, one, one, z));
        const wi8 xsh = wmma_iu8(p.c.z, p.c.w, one, one, wmma_iu8(p.c.x, p.c.y, one, one, z));
        float x0v[NV], x1v[NV];
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if constexpr (XDS) {
                x0v[v] = __shfl(p.xs[0].x, t0 + v, WARP);
                x1v[v] = __shfl(p.xs[0].x, 16 + t0 + v, WARP);
            } else {
                x0v[v] = p.xs[v].x;
                x1v[v] = p.xs[v].y;
            }
        }
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint4 hd = p.hd[tt];
            const float d = __half2float(__ushort_as_half((uint16_t)(hd.x & 0xFFFF)));
            const float dmin = __half2float(__ushort_as_half((uint16_t)(hd.x >> 16)));
            const int sel = 16 * (j & 1);
            const uint32_t A = hd.y >> sel, B = hd.z >> sel, C = hd.w >> sel;
            int s0, m0, s1, m1;
            if constexpr (!J2) {
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
            const uint32_t q[4] = {p.qv[tt].x, p.qv[tt].y, p.qv[tt].z, p.qv[tt].w};
            int lo[4], hi[4];
            if constexpr (QT == QT_Q4_K) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    lo[k] = q[k] & 0x0F0F0F0F;
                    hi[k] = (q[k] >> 4) & 0x0F0F0F0F;
                }
            } else {
                const uint32_t hb[4] = {p.hb[tt].x, p.hb[tt].y, p.hb[tt].z, p.hb[tt].w};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    lo[k] = (q[k] & 0x0F0F0F0F) | (((hb[k] >> (2 * j)) & 0x01010101) << 4);
                    hi[k] = ((q[k] >> 4) & 0x0F0F0F0F) | (((hb[k] >> (2 * j + 1)) & 0x01010101) << 4);
                }
            }
            const wi8 cl = wmma_iu8(p.a.z, p.a.w, lo[2], lo[3], wmma_iu8(p.a.x, p.a.y, lo[0], lo[1], z));
            const wi8 ch = wmma_iu8(p.c.z, p.c.w, hi[2], hi[3], wmma_iu8(p.c.x, p.c.y, hi[0], hi[1], z));
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const float x0 = x0v[v], x1 = x1v[v];
                const int dl = cl[v], dh = ch[v], sl = xsl[v], sh = xsh[v];
                acc[tt][v] += x0 * (ds0 * dl - dm0 * sl) + x1 * (ds1 * dh - dm1 * sh);
            }
        }
    }
};
template <int TILES, int NV, int XDS> struct GU<QT_Q4_K, TILES, NV, XDS> : GU_K<TILES, NV, XDS, QT_Q4_K> {};
template <int TILES, int NV, int XDS> struct GU<QT_Q5_K, TILES, NV, XDS> : GU_K<TILES, NV, XDS, QT_Q5_K> {};

// v_mul_lo_u32-free LUT blend (same bytes as iq4_lut4 / mx_lut4)
template <uint32_t T0, uint32_t T1, uint32_t T2, uint32_t T3>
__device__ __forceinline__ int lut4f(uint32_t nib4) {
    const uint32_t sel = nib4 & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(T1, T0, sel);
    const uint32_t hi = __builtin_amdgcn_perm(T3, T2, sel);
    const uint32_t b = (nib4 >> 3) & 0x01010101u;
    uint32_t m = b << 8;
    asm("" : "+v"(m));
    m -= b;
    return (int)((hi & m) | (lo & ~m));
}
#define IQ4F(x) lut4f<0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u>(x)
#define MXF(x) lut4f<0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu>(x)
// IQ4_XS / MXFP4 (4-bit LUT, 136-byte superblocks)
template <int TILES, int NV, int XDS, int QT> struct GU_4L {
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * 4; }
    struct P {
        int2 a0, a1, c0, c1;
        uint32_t hdr[TILES], sc[QT == QT_IQ4_XS ? TILES : 1];
        uint2 g0[TILES], g1[TILES];
        float2 xs[XDS ? 1 : NV];
    };
    static __device__ __forceinline__ void fetch(P &p, const uint8_t *const *wr, const int8_t *xt, const float *xd, int nb32,
                                                 int u, int h, int t0, int ntm1, int lane) {
        const int b = u >> 2, pp = u & 3;
        const int base = b * 256 + 64 * pp + 8 * h;
        p.a0 = *(const int2 *)(xt + base);
        p.a1 = *(const int2 *)(xt + base + 16);
        p.c0 = *(const int2 *)(xt + base + 32);
        p.c1 = *(const int2 *)(xt + base + 48);
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint8_t *blk = wr[tt] + (size_t)b * 136;
            if constexpr (QT == QT_IQ4_XS) {
                p.hdr[tt] = *(const uint32_t *)blk;
                p.sc[tt] = *(const uint32_t *)(blk + 4);
            } else {
                p.hdr[tt] = *(const uint16_t *)(blk + 2 * pp);
            }
            p.g0[tt] = *(const uint2 *)(blk + 8 + 32 * pp + 8 * h);
            p.g1[tt] = *(const uint2 *)(blk + 24 + 32 * pp + 8 * h);
        }
        const int xi = (b * 256 + 64 * pp) >> 5;
        if constexpr (XDS) {
            p.xs[0].x = xd[(size_t)min(lane & 15, ntm1) * nb32 + xi + (lane >> 4)];
        } else {
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const int t = min(t0 + v, ntm1);
                p.xs[v] = *(const float2 *)(xd + (size_t)t * nb32 + xi);
            }
        }
    }
    template <bool J2>
    static __device__ __forceinline__ void compute(const P &p, int u, int h, int t0, float (*acc)[NV]) {
        const int pp = u & 3;
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        float x0v[NV], x1v[NV];
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if constexpr (XDS) {
                x0v[v] = __shfl(p.xs[0].x, t0 + v, WARP);
                x1v[v] = __shfl(p.xs[0].x, 16 + t0 + v, WARP);
            } else {
                x0v[v] = p.xs[v].x;
                x1v[v] = p.xs[v].y;
            }
        }
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            float dl0, dl1;
            if constexpr (QT == QT_IQ4_XS) {
                const uint32_t hdr = p.hdr[tt];
                const float d = __half2float(__ushort_as_half((uint16_t)(hdr & 0xFFFF)));
                const uint32_t sh = hdr >> 16;
                const int ib0 = 2 * pp, ib1 = 2 * pp + 1;
                const uint32_t sl = (p.sc[tt] >> (8 * pp)) & 0xFF;
                const int ls0 = (sl & 0xF) | (((sh >> (2 * ib0)) & 3) << 4);
                const int ls1 = (sl >> 4) | (((sh >> (2 * ib1)) & 3) << 4);
                dl0 = d * (ls0 - 32);
                dl1 = d * (ls1 - 32);
            } else {
                const uint32_t ee = p.hdr[tt];
                dl0 = e8m0_half(ee & 0xFF);
                dl1 = e8m0_half(ee >> 8);
            }
            const uint2 g0 = p.g0[tt], g1 = p.g1[tt];
            wi8 c0, c1;
            if constexpr (QT == QT_IQ4_XS) {
                c0 = wmma_iu8(p.a0.x, p.a0.y, IQ4F(g0.x & 0x0F0F0F0F), IQ4F(g0.y & 0x0F0F0F0F), z);
                c0 = wmma_iu8(p.a1.x, p.a1.y, IQ4F((g0.x >> 4) & 0x0F0F0F0F), IQ4F((g0.y >> 4) & 0x0F0F0F0F), c0);
                c1 = wmma_iu8(p.c0.x, p.c0.y, IQ4F(g1.x & 0x0F0F0F0F), IQ4F(g1.y & 0x0F0F0F0F), z);
                c1 = wmma_iu8(p.c1.x, p.c1.y, IQ4F((g1.x >> 4) & 0x0F0F0F0F), IQ4F((g1.y >> 4) & 0x0F0F0F0F), c1);
            } else {
                c0 = wmma_iu8(p.a0.x, p.a0.y, MXF(g0.x & 0x0F0F0F0F), MXF(g0.y & 0x0F0F0F0F), z);
                c0 = wmma_iu8(p.a1.x, p.a1.y, MXF((g0.x >> 4) & 0x0F0F0F0F), MXF((g0.y >> 4) & 0x0F0F0F0F), c0);
                c1 = wmma_iu8(p.c0.x, p.c0.y, MXF(g1.x & 0x0F0F0F0F), MXF(g1.y & 0x0F0F0F0F), z);
                c1 = wmma_iu8(p.c1.x, p.c1.y, MXF((g1.x >> 4) & 0x0F0F0F0F), MXF((g1.y >> 4) & 0x0F0F0F0F), c1);
            }
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const float x0 = x0v[v], x1 = x1v[v];
                const int s0 = c0[v], s1 = c1[v];
                acc[tt][v] += dl0 * x0 * s0 + dl1 * x1 * s1;
            }
        }
    }
};
template <int TILES, int NV, int XDS> struct GU<QT_IQ4_XS, TILES, NV, XDS> : GU_4L<TILES, NV, XDS, QT_IQ4_XS> {};
template <int TILES, int NV, int XDS> struct GU<QT_MXFP4, TILES, NV, XDS> : GU_4L<TILES, NV, XDS, QT_MXFP4> {};

#define GW2_STEP(S, GUARD)                                                                          \
    if (GUARD) {                                                                                    \
        U::fetch(p[(S + DEPTH) % RING], wr, xt, xd, nb32, w + 8 * min(kk + S + DEPTH, cnt - 1), h, t0, NT - 1, lane); \
        U::template compute<J2>(p[S % RING], w + 8 * (kk + S), h, t0, acc[S]);                            \
        _Pragma("unroll") for (int tt = 0; tt < TILES; ++tt)                                        \
            _Pragma("unroll") for (int v = 0; v < NV; ++v) asm volatile("" : "+v"(acc[S][tt][v])); \
    }

template <int T, int NT, int TILES, int XDS, bool J2, int DEPTH>
__device__ __forceinline__ void gw2_loop(const uint8_t *const *wr, const int8_t *xt, const float *xd, int nb32, int nu,
                                         int w, int h, int t0, int lane, float (*acc)[TILES][NT < 8 ? NT : 8]) {
    constexpr int NV = NT < 8 ? NT : 8;
    using U = GU<T, TILES, NV, XDS>;
    const int cnt = (nu - w + 7) >> 3;
    if (cnt <= 0) return;
    constexpr int RING = DEPTH == 1 ? 2 : 4;
    typename U::P p[RING];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) U::fetch(p[d], wr, xt, xd, nb32, w + 8 * min(d, cnt - 1), h, t0, NT - 1, lane);
    int kk = 0;
    for (; kk + 4 <= cnt; kk += 4) {
        GW2_STEP(0, true)
        GW2_STEP(1, true)
        GW2_STEP(2, true)
        GW2_STEP(3, true)
    }
    GW2_STEP(0, kk < cnt)
    GW2_STEP(1, kk + 1 < cnt)
    GW2_STEP(2, kk + 2 < cnt)
}

template <int T, int NT, int TILES, int XDS, int DEPTH = 1>
__device__ __forceinline__ void gw2_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
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
    const int nu = GU<T, TILES, NV, XDS>::units(ncols);
    const int t0 = 8 * h;
    float acc[4][TILES][NV];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
            for (int v = 0; v < NV; ++v) acc[q][tt][v] = 0.f;
    if ((T == QT_Q4_K || T == QT_Q5_K) && (w & 3) >= 2)
        gw2_loop<T, NT, TILES, XDS, true, DEPTH>(wr, xt, xd, nb32, nu, w, h, t0, lane, acc);
    else
        gw2_loop<T, NT, TILES, XDS, false, DEPTH>(wr, xt, xd, nb32, nu, w, h, t0, lane, acc);
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
// item V: Q6_K pipelined unit for gw2 (same unit / fragment / float expression as WMU<QT_Q6_K>).
// u = 8b + 4n + g with u = w + 8k: n = (w >> 2) & 1 and g = w & 3 are wave constants, b = k.
// XDS 0: 8 float4 xd loads per unit; XDS 1: 2 dwords per lane + 32 shuffles.
template <int TILES, int NV, int XDS> struct GU<QT_Q6_K, TILES, NV, XDS> {
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * 8; }
    struct P {
        int2 a01, a23;
        uint2 l[TILES], hq[TILES], scw[TILES];
        uint32_t dd[TILES];
        float4 xs[XDS ? 1 : NV];
    };
    static __device__ __forceinline__ void fetch(P &p, const uint8_t *const *wr, const int8_t *xt, const float *xd, int nb32,
                                                 int u, int h, int t0, int ntm1, int lane) {
        const int b = u >> 3, n = (u >> 2) & 1, g = u & 3;
        const int base = b * 256 + 128 * n + 8 * g + 32 * h;
        p.a01 = *(const int2 *)(xt + base);
        p.a23 = *(const int2 *)(xt + base + 64);
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint8_t *blk = wr[tt] + (size_t)b * 210;
            p.l[tt] = ld64a2(blk + 64 * n + 8 * g + 32 * h);
            p.hq[tt] = ld64a2(blk + 128 + 32 * n + 8 * g);
            p.scw[tt] = ld64a2(blk + 192 + 8 * n);
            p.dd[tt] = *(const uint16_t *)(blk + 208);
        }
        const int xi = (b * 256 + 128 * n) >> 5;
        if constexpr (XDS) {
            // lane: token lane & 15, quarters 2 (lane >> 4) .. + 1
            p.xs[0].x = xd[(size_t)min(lane & 15, ntm1) * nb32 + xi + 2 * (lane >> 4)];
            p.xs[0].y = xd[(size_t)min(lane & 15, ntm1) * nb32 + xi + 2 * (lane >> 4) + 1];
        } else {
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const int t = min(t0 + v, ntm1);
                p.xs[v] = *(const float4 *)(xd + (size_t)t * nb32 + xi);
            }
        }
    }
    template <bool J2>
    static __device__ __forceinline__ void compute(const P &p, int u, int h, int t0, float (*acc)[NV]) {
        const int g = u & 3;
        const int is = g >> 1;
        float xq4[NV][4];
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if constexpr (XDS) {
                xq4[v][0] = __shfl(p.xs[0].x, t0 + v, WARP);
                xq4[v][1] = __shfl(p.xs[0].y, t0 + v, WARP);
                xq4[v][2] = __shfl(p.xs[0].x, 16 + t0 + v, WARP);
                xq4[v][3] = __shfl(p.xs[0].y, 16 + t0 + v, WARP);
            } else {
                xq4[v][0] = p.xs[v].x;
                xq4[v][1] = p.xs[v].y;
                xq4[v][2] = p.xs[v].z;
                xq4[v][3] = p.xs[v].w;
            }
        }
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        const int m0 = h == 0 ? -1 : 0, m1 = ~m0;
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint2 l = p.l[tt], hq = p.hq[tt], scw = p.scw[tt];
            const float dd = __half2float(__ushort_as_half((uint16_t)p.dd[tt]));
            float scf[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t word = (is + 2 * q) < 4 ? scw.x : scw.y;
                scf[q] = (float)(int8_t)(word >> (8 * ((is + 2 * q) & 3)));
            }
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
            const wi8 c0 = wmma_iu8(p.a01.x, p.a01.y, wq[0][0] & m0, wq[0][1] & m0, z);
            const wi8 c1 = wmma_iu8(p.a01.x, p.a01.y, wq[0][0] & m1, wq[0][1] & m1, z);
            const wi8 c2 = wmma_iu8(p.a23.x, p.a23.y, wq[1][0] & m0, wq[1][1] & m0, z);
            const wi8 c3 = wmma_iu8(p.a23.x, p.a23.y, wq[1][0] & m1, wq[1][1] & m1, z);
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const int dv[4] = {c0[v], c1[v], c2[v], c3[v]};
                const float xsq[4] = {xq4[v][0], xq4[v][1], xq4[v][2], xq4[v][3]};
                float total = 0.f;
#pragma unroll
                for (int q = 0; q < 4; ++q) total += scf[q] * xsq[q] * (float)dv[q];
                acc[tt][v] += dd * total;
            }
        }
    }
};
// item V: pipelined mid-batch GEMV for the 8-value-unit types Q3_K / IQ3_S (gw8).
// Same canonical order as gemvw8_impl / gemvq_impl (unit u = 8 weights, slice u % 32, wave w
// owns slices w, w+8, w+16, w+24 = slots 0..3, same float expression acc += sc * xd * dv,
// same xor-16/8 register + LDS xor-4/2/1 reduction) -> bitwise == the 1-token GEMV. Per wave
// and superblock b the 4 units u = 32b + w + 8s (s = slot) are one step: pairs (s = 0, 1) and
// (2, 3) share a WMMA pair (lane half h carries unit 2p + h, B masked per WMMA); every lane
// decodes all 4 unit scales (no shuffles); xd via one float2 per lane + shuffles; the loads of
// superblock b + 1 are issued before b is computed.
template <int T, int TILES> struct G8;

template <int TILES> struct G8<QT_Q3_K, TILES> {
    struct P {
        int2 xa[2];
        uint2 hm[TILES], q0[TILES], q1[TILES];
        uint2 sa[TILES];
        uint32_t sb[TILES];
        float2 xs;
    };
    static __device__ __forceinline__ void fetch(P &p, const uint8_t *const *wr, const int8_t *xt, const float *xd, int nb32,
                                                 int b, int w, int h, int lane, int ntm1) {
        const int sg = w & 3, j0 = (w >> 2) & 1;
#pragma unroll
        for (int pp = 0; pp < 2; ++pp) p.xa[pp] = *(const int2 *)(xt + b * 256 + 128 * pp + 32 * (j0 + 2 * h) + 8 * sg);
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint8_t *blk = wr[tt] + (size_t)b * 110;
            p.hm[tt] = ld64a2(blk + 8 * sg);
            p.q0[tt] = ld64a2(blk + 32 + 8 * sg);
            p.q1[tt] = ld64a2(blk + 64 + 8 * sg);
            p.sa[tt] = ld64a2(blk + 96);                                  // scale bytes 0..7
            p.sb[tt] = *(const uint16_t *)(blk + 104) | ((uint32_t) * (const uint16_t *)(blk + 106) << 16);  // 8..11
        }
        const float *xr = xd + (size_t)min(lane & 15, ntm1) * nb32 + b * 8 + j0 + 4 * (lane >> 4);
        p.xs.x = xr[0];
        p.xs.y = xr[2];
    }
    // d is read separately (2 bytes at 108)
    static __device__ __forceinline__ float dval(const uint8_t *const *wr, int tt, int b) { return h2f(wr[tt] + (size_t)b * 110 + 108); }
};

template <int TILES> struct G8<QT_IQ3_S, TILES> {
    struct P {
        int2 xa[2];
        uint32_t sc[TILES];
        uint2 qh[TILES];
        uint32_t qs[TILES][2];
        uint32_t sg[TILES][2];
        float2 xs;
    };
    static __device__ __forceinline__ void fetch(P &p, const uint8_t *const *wr, const int8_t *xt, const float *xd, int nb32,
                                                 int b, int w, int h, int lane, int ntm1) {
        const int l = w & 3, i0 = (w >> 2) & 1;
#pragma unroll
        for (int pp = 0; pp < 2; ++pp) p.xa[pp] = *(const int2 *)(xt + b * 256 + 32 * (i0 + 4 * pp + 2 * h) + 8 * l);
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint8_t *blk = wr[tt] + (size_t)b * 110;
            p.sc[tt] = *(const uint16_t *)(blk + 106) | ((uint32_t) * (const uint16_t *)(blk + 108) << 16);
            p.qh[tt] = ld64a2(blk + 66);
#pragma unroll
            for (int pp = 0; pp < 2; ++pp) {
                const int ib32 = i0 + 4 * pp + 2 * h;
                p.qs[tt][pp] = *(const uint16_t *)(blk + 2 + 8 * ib32 + 2 * l);
                p.sg[tt][pp] = blk[74 + 4 * ib32 + l];
            }
        }
        const float *xr = xd + (size_t)min(lane & 15, ntm1) * nb32 + b * 8 + i0 + 4 * (lane >> 4);
        p.xs.x = xr[0];
        p.xs.y = xr[2];
    }
    static __device__ __forceinline__ float dval(const uint8_t *const *wr, int tt, int b) { return h2f(wr[tt] + (size_t)b * 110); }
};

// one superblock step of wave w: 4 units (slots), TILES tiles
template <int T, int NT, int TILES>
__device__ __forceinline__ void gw8_compute(const typename G8<T, TILES>::P &p, const uint8_t *const *wr, int b, int w, int h,
                                            int n, float (*acc)[TILES][NT < 8 ? NT : 8], const uint32_t *grid) {
    constexpr int NV = NT < 8 ? NT : 8;
    const int t0 = 8 * h;
    const int mk0 = h == 0 ? -1 : 0, mk1 = ~mk0;
    const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
    // xd of unit s, token t0 + v: lane t0 + v + 16 (s >> 1), component s & 1
    float xv[4][NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        xv[0][v] = __shfl(p.xs.x, t0 + v, WARP);
        xv[1][v] = __shfl(p.xs.y, t0 + v, WARP);
        xv[2][v] = __shfl(p.xs.x, 16 + t0 + v, WARP);
        xv[3][v] = __shfl(p.xs.y, 16 + t0 + v, WARP);
    }
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt) {
        const float d = G8<T, TILES>::dval(wr, tt, b);
        float sc[4];
        int2 wv[2];
        if constexpr (T == QT_Q3_K) {
            const int sg = w & 3, j0 = (w >> 2) & 1;
            const uint32_t sb4[3] = {p.sa[tt].x, p.sa[tt].y, p.sb[tt]};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int nn = s >> 1, j = j0 + 2 * (s & 1);
                const int kk = 8 * nn + 2 * j + (sg >> 1);
                const int lo = (kk < 8) ? ((sb4[kk >> 2] >> (8 * (kk & 3))) & 0xF) : ((sb4[(kk - 8) >> 2] >> (8 * ((kk - 8) & 3) + 4)) & 0xF);
                const int hi = ((sb4[2] >> (8 * (kk & 3))) >> (2 * (kk >> 2))) & 3;
                sc[s] = d * ((lo | (hi << 4)) - 32);
            }
            // lane half h decodes unit 2 pp + h: j = j0 + 2 h, nn = pp
#pragma unroll
            for (int pp = 0; pp < 2; ++pp) {
                const int j = j0 + 2 * h;
                const uint2 q = pp ? p.q1[tt] : p.q0[tt];
                const int sh = 4 * pp + j;
                const uint32_t w0 = (q.x >> (2 * j)) & 0x03030303u, w1 = (q.y >> (2 * j)) & 0x03030303u;
                const uint32_t m0 = ((~p.hm[tt].x) >> sh) & 0x01010101u, m1 = ((~p.hm[tt].y) >> sh) & 0x01010101u;
                wv[pp].x = (int)(((w0 | 0x80808080u) - (m0 << 2)) ^ 0x80808080u);
                wv[pp].y = (int)(((w1 | 0x80808080u) - (m1 << 2)) ^ 0x80808080u);
            }
        } else {
            const int l = w & 3, i0 = (w >> 2) & 1;
#pragma unroll
            for (int s = 0; s < 4; ++s) sc[s] = d * (1 + 2 * ((p.sc[tt] >> (8 * s + 4 * i0)) & 0xF));
#pragma unroll
            for (int pp = 0; pp < 2; ++pp) {
                const int ib32 = i0 + 4 * pp + 2 * h;
                const uint32_t hb = (ib32 < 4 ? p.qh[tt].x : p.qh[tt].y) >> (8 * (ib32 & 3));
                const uint32_t qa = p.qs[tt][pp] & 0xFF, qb = p.qs[tt][pp] >> 8;
                const uint32_t g1 = grid[qa | ((hb << (8 - 2 * l)) & 256)];
                const uint32_t g2 = grid[qb | ((hb << (7 - 2 * l)) & 256)];
                const uint32_t sgn = p.sg[tt][pp];
                const uint32_t m1 = (((sgn & 0xF) * 0x00204081u) & 0x01010101u) * 0xFFu;
                const uint32_t m2 = (((sgn >> 4) * 0x00204081u) & 0x01010101u) * 0xFFu;
                wv[pp].x = (int)((g1 ^ m1) + (m1 & 0x01010101u));
                wv[pp].y = (int)((g2 ^ m2) + (m2 & 0x01010101u));
            }
        }
#pragma unroll
        for (int pp = 0; pp < 2; ++pp) {
            const wi8 c0 = wmma_iu8(p.xa[pp].x, p.xa[pp].y, wv[pp].x & mk0, wv[pp].y & mk0, z);
            const wi8 c1 = wmma_iu8(p.xa[pp].x, p.xa[pp].y, wv[pp].x & mk1, wv[pp].y & mk1, z);
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const int dv0 = c0[v];
                acc[2 * pp][tt][v] += sc[2 * pp] * xv[2 * pp][v] * dv0;
            }
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const int dv1 = c1[v];
                acc[2 * pp + 1][tt][v] += sc[2 * pp + 1] * xv[2 * pp + 1][v] * dv1;
            }
        }
    }
}

template <int T, int NT, int TILES>
__device__ __forceinline__ void gw8_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                         const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                         const int *__restrict__ skip, int bid) {
    constexpr int ROWS = 16 * TILES;
    constexpr int NV = NT < 8 ? NT : 8;
    __shared__ float red[8][NT][ROWS];
    __shared__ uint32_t grid_s[T == QT_IQ3_S ? 512 : 1];
    if (skip != nullptr && *skip) return;
    if constexpr (T == QT_IQ3_S) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) grid_s[i] = k_iq3s_grid[i];
        __syncthreads();
    }
    const int lane = threadIdx.x % WARP, h = lane >> 4, n = lane & 15;
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP);
    const int r0 = bid * ROWS;
    const uint8_t *wr[TILES];
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt) wr[tt] = W + (size_t)min(r0 + tt * 16 + n, nrows - 1) * row_bytes;
    const int8_t *xt = xq + (size_t)min(n, NT - 1) * ncols;
    const int nb32 = ncols / 32;
    const int nsb = ncols / 256;
    const int t0 = 8 * h;
    float acc[4][TILES][NV];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
            for (int v = 0; v < NV; ++v) acc[q][tt][v] = 0.f;
    typename G8<T, TILES>::P p[2];
    G8<T, TILES>::fetch(p[0], wr, xt, xd, nb32, 0, w, h, lane, NT - 1);
    int b = 0;
    for (; b + 2 <= nsb; b += 2) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            G8<T, TILES>::fetch(p[e ^ 1], wr, xt, xd, nb32, min(b + e + 1, nsb - 1), w, h, lane, NT - 1);
            gw8_compute<T, NT, TILES>(p[e], wr, b + e, w, h, n, acc, grid_s);
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
                for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
                    for (int v = 0; v < NV; ++v) asm volatile("" : "+v"(acc[q][tt][v]));
        }
    }
    if (b < nsb) gw8_compute<T, NT, TILES>(p[0], wr, b, w, h, n, acc, grid_s);
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

#define GEMVW2_ENTRY(NT, V, TILES, DEPTH, SUF, T)                                                 \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvw_nt##NT##v##V##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,      \
                                   const float *xd, float *y, int ncols, int nrows,              \
                                   int accumulate, const int *skip) {                            \
        gw2_impl<T, NT, TILES, 1, DEPTH>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x); \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvw_nt##NT##v##V##_##SUF(GV_PARAMS) {                            \
        { const GvSel gs = gv_sel(g); gw2_impl<T, NT, TILES, 1, DEPTH>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); } \
    }
#define GEMVW2_TYPES(NT, V, TILES, DEPTH)                  \
    GEMVW2_ENTRY(NT, V, TILES, DEPTH, q4_k, QT_Q4_K)       \
    GEMVW2_ENTRY(NT, V, TILES, DEPTH, q5_k, QT_Q5_K)       \
    GEMVW2_ENTRY(NT, V, TILES, DEPTH, iq4_xs, QT_IQ4_XS)   \
    GEMVW2_ENTRY(NT, V, TILES, DEPTH, mxfp4, QT_MXFP4)     \
    GEMVW2_ENTRY(NT, V, TILES, DEPTH, q6_k, QT_Q6_K)
// (v8 = (2, 2) spills on Q4_K / Q5_K: not built for them)
#define GEMVW2_NT(NT) GEMVW2_TYPES(NT, 5, 1, 1) GEMVW2_TYPES(NT, 6, 1, 3) GEMVW2_TYPES(NT, 7, 2, 1) \
    GEMVW2_ENTRY(NT, 8, 2, 2, iq4_xs, QT_IQ4_XS) GEMVW2_ENTRY(NT, 8, 2, 2, mxfp4, QT_MXFP4) GEMVW2_ENTRY(NT, 8, 2, 2, q6_k, QT_Q6_K) \
    GEMVW8B_ENTRY(NT, q3_k, QT_Q3_K) GEMVW8B_ENTRY(NT, iq3_s, QT_IQ3_S)
// Q3_K / IQ3_S: v5 = gw8 (1 tile)
#define GEMVW8B_ENTRY(NT, SUF, T)                                                                 \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvw_nt##NT##v5_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,          \
                               const float *xd, float *y, int ncols, int nrows,                  \
                               int accumulate, const int *skip) {                                \
        gw8_impl<T, NT, 1>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x);             \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvw_nt##NT##v5_##SUF(GV_PARAMS) {                                \
        { const GvSel gs = gv_sel(g); gw8_impl<T, NT, 1>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }             \
    }
GEMVW2_NT(2)
GEMVW2_NT(3)
GEMVW2_NT(4)
GEMVW2_NT(5)
GEMVW2_NT(6)
GEMVW2_NT(7)
GEMVW2_NT(8)
GEMVW2_NT(9)
GEMVW2_NT(10)
GEMVW2_NT(11)
GEMVW2_NT(12)
GEMVW2_NT(13)
GEMVW2_NT(14)
GEMVW2_NT(15)
GEMVW2_NT(16)
// ---- item V: pipelined mid-batch GEMV (end) ----

