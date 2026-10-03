// WHIRL device code (gfx1201 / RDNA 4): wide mid-batch GEMV on int8 WMMA, 17..32 tokens (gemvx_v*).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included in order by whirl_kernels.hip (after gemv_wp.hip).

// ---------------------------------------------------------------------------
// gemvx: the gw2 kernels (gemv_wp.hip) with two 16-token activation tiles. Each unit's
// weights are loaded and decoded once and feed one WMMA chain per token tile, so a
// verify of up to 32 rows reads the weights once. Per (token, weight row) everything
// is the gw2 / one-token arithmetic: the lane / unit / slot mapping (u = w + 8k, slice
// u % 32, wave w owns slots w, w+8, w+16, w+24), the per-unit float expression (same
// source text, same operand types), the xor-16/8 register + LDS xor-4/2/1 reduction.
// Integer dots are exact, so token t's result is bitwise the one-token gemvq result.
// The token count n (1..32) is a runtime argument: rows >= n are neither read past
// row n - 1 (activations clamp to row n - 1) nor written.
// One variant (v6 = 1 weight tile, prefetch depth 3): in a probe over the 27B shapes it beat
// depth 1 and 2 weight tiles (those spilled for the K-quants), and two 16-token launches by
// 1.5-1.9x. No grouped twin (it spilled); one launch per matrix.
// ---------------------------------------------------------------------------
template <int T, int TILES> struct GX;

// Q4_K / Q5_K (as GU_K)
template <int TILES, int QT> struct GX_K {
    static constexpr int KT = 2, NV = 8;
    static constexpr int bytes = QT == QT_Q4_K ? 144 : 176;
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * 4; }
    struct P {
        int4 a[KT], c[KT];
        uint4 hd[TILES], qv[TILES];
        uint4 hb[QT == QT_Q5_K ? TILES : 1];
        float xs[KT];
    };
    static __device__ __forceinline__ void fetch(P &p, const uint8_t *const *wr, const int8_t *const *xt, const float *xd,
                                                 int nb32, int u, int h, int ntm1, int lane) {
        const int b = u >> 2, j = u & 3;
        const int base = b * 256 + 64 * j;
#pragma unroll
        for (int k = 0; k < KT; ++k) {
            p.a[k] = *(const int4 *)(xt[k] + base + 16 * h);
            p.c[k] = *(const int4 *)(xt[k] + base + 32 + 16 * h);
        }
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint8_t *blk = wr[tt] + (size_t)b * bytes;
            p.hd[tt] = *(const uint4 *)blk;
            p.qv[tt] = *(const uint4 *)(blk + (QT == QT_Q4_K ? 16 : 48) + 32 * j + 16 * h);
            if constexpr (QT == QT_Q5_K) p.hb[tt] = *(const uint4 *)(blk + 16 + 16 * h);
        }
        const int xi = base >> 5;
#pragma unroll
        for (int k = 0; k < KT; ++k) p.xs[k] = xd[(size_t)min(16 * k + (lane & 15), ntm1) * nb32 + xi + (lane >> 4)];
    }
    template <bool J2>
    static __device__ __forceinline__ void compute(const P &p, int u, int h, int t0, float (*acc)[NV]) {
        const int j = u & 3;
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        const int one = 0x01010101;
        // token tile outermost: one tile's activation scales live at a time
#pragma unroll
        for (int k = 0; k < KT; ++k) {
            const wi8 xsl = wmma_iu8(p.a[k].z, p.a[k].w, one, one, wmma_iu8(p.a[k].x, p.a[k].y, one, one, z));
            const wi8 xsh = wmma_iu8(p.c[k].z, p.c[k].w, one, one, wmma_iu8(p.c[k].x, p.c[k].y, one, one, z));
            float x0v[NV], x1v[NV];
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                x0v[v] = __shfl(p.xs[k], t0 + v, WARP);
                x1v[v] = __shfl(p.xs[k], 16 + t0 + v, WARP);
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
                    for (int e = 0; e < 4; ++e) {
                        lo[e] = q[e] & 0x0F0F0F0F;
                        hi[e] = (q[e] >> 4) & 0x0F0F0F0F;
                    }
                } else {
                    const uint32_t hb[4] = {p.hb[tt].x, p.hb[tt].y, p.hb[tt].z, p.hb[tt].w};
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        lo[e] = (q[e] & 0x0F0F0F0F) | (((hb[e] >> (2 * j)) & 0x01010101) << 4);
                        hi[e] = ((q[e] >> 4) & 0x0F0F0F0F) | (((hb[e] >> (2 * j + 1)) & 0x01010101) << 4);
                    }
                }
                const wi8 cl = wmma_iu8(p.a[k].z, p.a[k].w, lo[2], lo[3], wmma_iu8(p.a[k].x, p.a[k].y, lo[0], lo[1], z));
                const wi8 ch = wmma_iu8(p.c[k].z, p.c[k].w, hi[2], hi[3], wmma_iu8(p.c[k].x, p.c[k].y, hi[0], hi[1], z));
#pragma unroll
                for (int v = 0; v < NV; ++v) {
                    const float x0 = x0v[v], x1 = x1v[v];
                    const int dl = cl[v], dh = ch[v], sl = xsl[v], sh = xsh[v];
                    acc[k * TILES + tt][v] += x0 * (ds0 * dl - dm0 * sl) + x1 * (ds1 * dh - dm1 * sh);
                }
            }
        }
    }
};
template <int TILES> struct GX<QT_Q4_K, TILES> : GX_K<TILES, QT_Q4_K> {};
template <int TILES> struct GX<QT_Q5_K, TILES> : GX_K<TILES, QT_Q5_K> {};

// IQ4_XS / MXFP4 (as GU_4L)
template <int TILES, int QT> struct GX_4L {
    static constexpr int KT = 2, NV = 8;
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * 4; }
    struct P {
        int2 a0[KT], a1[KT], c0[KT], c1[KT];
        uint32_t hdr[TILES], sc[QT == QT_IQ4_XS ? TILES : 1];
        uint2 g0[TILES], g1[TILES];
        float xs[KT];
    };
    static __device__ __forceinline__ void fetch(P &p, const uint8_t *const *wr, const int8_t *const *xt, const float *xd,
                                                 int nb32, int u, int h, int ntm1, int lane) {
        const int b = u >> 2, pp = u & 3;
        const int base = b * 256 + 64 * pp + 8 * h;
#pragma unroll
        for (int k = 0; k < KT; ++k) {
            p.a0[k] = *(const int2 *)(xt[k] + base);
            p.a1[k] = *(const int2 *)(xt[k] + base + 16);
            p.c0[k] = *(const int2 *)(xt[k] + base + 32);
            p.c1[k] = *(const int2 *)(xt[k] + base + 48);
        }
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
#pragma unroll
        for (int k = 0; k < KT; ++k) p.xs[k] = xd[(size_t)min(16 * k + (lane & 15), ntm1) * nb32 + xi + (lane >> 4)];
    }
    template <bool J2>
    static __device__ __forceinline__ void compute(const P &p, int u, int h, int t0, float (*acc)[NV]) {
        const int pp = u & 3;
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        float x0v[KT][NV], x1v[KT][NV];
#pragma unroll
        for (int k = 0; k < KT; ++k)
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                x0v[k][v] = __shfl(p.xs[k], t0 + v, WARP);
                x1v[k][v] = __shfl(p.xs[k], 16 + t0 + v, WARP);
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
            int b0[4], b1[4];
            if constexpr (QT == QT_IQ4_XS) {
                b0[0] = IQ4F(g0.x & 0x0F0F0F0F);
                b0[1] = IQ4F(g0.y & 0x0F0F0F0F);
                b0[2] = IQ4F((g0.x >> 4) & 0x0F0F0F0F);
                b0[3] = IQ4F((g0.y >> 4) & 0x0F0F0F0F);
                b1[0] = IQ4F(g1.x & 0x0F0F0F0F);
                b1[1] = IQ4F(g1.y & 0x0F0F0F0F);
                b1[2] = IQ4F((g1.x >> 4) & 0x0F0F0F0F);
                b1[3] = IQ4F((g1.y >> 4) & 0x0F0F0F0F);
            } else {
                b0[0] = MXF(g0.x & 0x0F0F0F0F);
                b0[1] = MXF(g0.y & 0x0F0F0F0F);
                b0[2] = MXF((g0.x >> 4) & 0x0F0F0F0F);
                b0[3] = MXF((g0.y >> 4) & 0x0F0F0F0F);
                b1[0] = MXF(g1.x & 0x0F0F0F0F);
                b1[1] = MXF(g1.y & 0x0F0F0F0F);
                b1[2] = MXF((g1.x >> 4) & 0x0F0F0F0F);
                b1[3] = MXF((g1.y >> 4) & 0x0F0F0F0F);
            }
#pragma unroll
            for (int k = 0; k < KT; ++k) {
                wi8 c0, c1;
                c0 = wmma_iu8(p.a0[k].x, p.a0[k].y, b0[0], b0[1], z);
                c0 = wmma_iu8(p.a1[k].x, p.a1[k].y, b0[2], b0[3], c0);
                c1 = wmma_iu8(p.c0[k].x, p.c0[k].y, b1[0], b1[1], z);
                c1 = wmma_iu8(p.c1[k].x, p.c1[k].y, b1[2], b1[3], c1);
#pragma unroll
                for (int v = 0; v < NV; ++v) {
                    const float x0 = x0v[k][v], x1 = x1v[k][v];
                    const int s0 = c0[v], s1 = c1[v];
                    acc[k * TILES + tt][v] += dl0 * x0 * s0 + dl1 * x1 * s1;
                }
            }
        }
    }
};
template <int TILES> struct GX<QT_IQ4_XS, TILES> : GX_4L<TILES, QT_IQ4_XS> {};
template <int TILES> struct GX<QT_MXFP4, TILES> : GX_4L<TILES, QT_MXFP4> {};

// Q6_K (as GU<QT_Q6_K>, XDS = 1)
template <int TILES> struct GX<QT_Q6_K, TILES> {
    static constexpr int KT = 2, NV = 8;
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * 8; }
    struct P {
        int2 a01[KT], a23[KT];
        uint2 l[TILES], hq[TILES], scw[TILES];
        uint32_t dd[TILES];
        float2 xs[KT];
    };
    static __device__ __forceinline__ void fetch(P &p, const uint8_t *const *wr, const int8_t *const *xt, const float *xd,
                                                 int nb32, int u, int h, int ntm1, int lane) {
        const int b = u >> 3, n = (u >> 2) & 1, g = u & 3;
        const int base = b * 256 + 128 * n + 8 * g + 32 * h;
#pragma unroll
        for (int k = 0; k < KT; ++k) {
            p.a01[k] = *(const int2 *)(xt[k] + base);
            p.a23[k] = *(const int2 *)(xt[k] + base + 64);
        }
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint8_t *blk = wr[tt] + (size_t)b * 210;
            p.l[tt] = ld64a2(blk + 64 * n + 8 * g + 32 * h);
            p.hq[tt] = ld64a2(blk + 128 + 32 * n + 8 * g);
            p.scw[tt] = ld64a2(blk + 192 + 8 * n);
            p.dd[tt] = *(const uint16_t *)(blk + 208);
        }
        const int xi = (b * 256 + 128 * n) >> 5;
#pragma unroll
        for (int k = 0; k < KT; ++k) {
            const size_t row = (size_t)min(16 * k + (lane & 15), ntm1) * nb32;
            p.xs[k].x = xd[row + xi + 2 * (lane >> 4)];
            p.xs[k].y = xd[row + xi + 2 * (lane >> 4) + 1];
        }
    }
    template <bool J2>
    static __device__ __forceinline__ void compute(const P &p, int u, int h, int t0, float (*acc)[NV]) {
        const int g = u & 3;
        const int is = g >> 1;
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        const int m0 = h == 0 ? -1 : 0, m1 = ~m0;
        // token tile outermost: one tile's activation scales live at a time
#pragma unroll
        for (int k = 0; k < KT; ++k) {
            float xq4[NV][4];
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                xq4[v][0] = __shfl(p.xs[k].x, t0 + v, WARP);
                xq4[v][1] = __shfl(p.xs[k].y, t0 + v, WARP);
                xq4[v][2] = __shfl(p.xs[k].x, 16 + t0 + v, WARP);
                xq4[v][3] = __shfl(p.xs[k].y, 16 + t0 + v, WARP);
            }
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
                    for (int f = 0; f < 2; ++f) {
                        const uint32_t v = ((L[f] >> (4 * e)) & 0x0F0F0F0Fu) | (((H[f] >> (2 * qq)) & 0x03030303u) << 4);
                        wq[e][f] = (int)(((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
                    }
                }
                const wi8 c0 = wmma_iu8(p.a01[k].x, p.a01[k].y, wq[0][0] & m0, wq[0][1] & m0, z);
                const wi8 c1 = wmma_iu8(p.a01[k].x, p.a01[k].y, wq[0][0] & m1, wq[0][1] & m1, z);
                const wi8 c2 = wmma_iu8(p.a23[k].x, p.a23[k].y, wq[1][0] & m0, wq[1][1] & m0, z);
                const wi8 c3 = wmma_iu8(p.a23[k].x, p.a23[k].y, wq[1][0] & m1, wq[1][1] & m1, z);
#pragma unroll
                for (int v = 0; v < NV; ++v) {
                    const int dv[4] = {c0[v], c1[v], c2[v], c3[v]};
                    const float xsq[4] = {xq4[v][0], xq4[v][1], xq4[v][2], xq4[v][3]};
                    float total = 0.f;
#pragma unroll
                    for (int q = 0; q < 4; ++q) total += scf[q] * xsq[q] * (float)dv[q];
                    acc[k * TILES + tt][v] += dd * total;
                }
            }
        }
    }
};

#define GWX_STEP(S, GUARD)                                                                          \
    if (GUARD) {                                                                                    \
        U::fetch(p[(S + DEPTH) % RING], wr, xt, xd, nb32, w + 8 * min(kk + S + DEPTH, cnt - 1), h, ntm1, lane); \
        U::template compute<J2>(p[S % RING], w + 8 * (kk + S), h, t0, acc[S]);                      \
        _Pragma("unroll") for (int tt = 0; tt < 2 * TILES; ++tt)                                    \
            _Pragma("unroll") for (int v = 0; v < 8; ++v) asm volatile("" : "+v"(acc[S][tt][v]));  \
    }

template <int T, int TILES, bool J2, int DEPTH>
__device__ __forceinline__ void gwx_loop(const uint8_t *const *wr, const int8_t *const *xt, const float *xd, int nb32, int nu,
                                         int w, int h, int t0, int ntm1, int lane, float (*acc)[2 * TILES][8]) {
    using U = GX<T, TILES>;
    const int cnt = (nu - w + 7) >> 3;
    if (cnt <= 0) return;
    constexpr int RING = DEPTH == 1 ? 2 : 4;
    typename U::P p[RING];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) U::fetch(p[d], wr, xt, xd, nb32, w + 8 * min(d, cnt - 1), h, ntm1, lane);
    int kk = 0;
    for (; kk + 4 <= cnt; kk += 4) {
        GWX_STEP(0, true)
        GWX_STEP(1, true)
        GWX_STEP(2, true)
        GWX_STEP(3, true)
    }
    GWX_STEP(0, kk < cnt)
    GWX_STEP(1, kk + 1 < cnt)
    GWX_STEP(2, kk + 2 < cnt)
}

template <int T, int TILES, int DEPTH>
__device__ __forceinline__ void gwx_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                                         const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows,
                                         int accumulate, const int *__restrict__ skip, int bid, int ntok) {
    constexpr int ROWS = 16 * TILES, NT = 32;
    __shared__ float red[8][NT][ROWS];
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP, h = lane >> 4, n = lane & 15;
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP);
    const int r0 = bid * ROWS;
    const int ntm1 = ntok - 1;
    const uint8_t *wr[TILES];
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt) wr[tt] = W + (size_t)min(r0 + tt * 16 + n, nrows - 1) * row_bytes;
    const int8_t *xt[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) xt[k] = xq + (size_t)min(16 * k + n, ntm1) * ncols;
    const int nb32 = ncols / 32;
    const int nu = GX<T, TILES>::units(ncols);
    const int t0 = 8 * h;
    float acc[4][2 * TILES][8];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int tt = 0; tt < 2 * TILES; ++tt)
#pragma unroll
            for (int v = 0; v < 8; ++v) acc[q][tt][v] = 0.f;
    if ((T == QT_Q4_K || T == QT_Q5_K) && (w & 3) >= 2)
        gwx_loop<T, TILES, true, DEPTH>(wr, xt, xd, nb32, nu, w, h, t0, ntm1, lane, acc);
    else
        gwx_loop<T, TILES, false, DEPTH>(wr, xt, xd, nb32, nu, w, h, t0, ntm1, lane, acc);
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int t = 16 * k + t0 + v;
                red[w][t][tt * 16 + n] = (acc[0][k * TILES + tt][v] + acc[2][k * TILES + tt][v]) +
                                         (acc[1][k * TILES + tt][v] + acc[3][k * TILES + tt][v]);
            }
    __syncthreads();
    for (int i = threadIdx.x; i < ntok * ROWS; i += blockDim.x) {
        const int t = i / ROWS, nn = i % ROWS, row = r0 + nn;
        const float s = ((red[0][t][nn] + red[4][t][nn]) + (red[2][t][nn] + red[6][t][nn])) +
                        ((red[1][t][nn] + red[5][t][nn]) + (red[3][t][nn] + red[7][t][nn]));
        if (row < nrows) {
            float *dst = y + (size_t)t * nrows + row;
            *dst = accumulate ? *dst + s : s;
        }
    }
}

#define GEMVX_ENTRY(V, TILES, DEPTH, SUF, T)                                                                      \
    extern "C" __global__ void __launch_bounds__(256)                                                             \
        gemvx_v##V##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq, const float *xd, float *y,     \
                           int ncols, int nrows, int accumulate, const int *skip, int ntok) {                     \
        gwx_impl<T, TILES, DEPTH>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x, ntok);     \
    }
GEMVX_ENTRY(5, 1, 1, q4_k, QT_Q4_K)
GEMVX_ENTRY(5, 1, 1, q5_k, QT_Q5_K)
GEMVX_ENTRY(5, 1, 1, iq4_xs, QT_IQ4_XS)
GEMVX_ENTRY(5, 1, 1, mxfp4, QT_MXFP4)
GEMVX_ENTRY(5, 1, 1, q6_k, QT_Q6_K)
GEMVX_ENTRY(6, 1, 3, q4_k, QT_Q4_K)
GEMVX_ENTRY(6, 1, 3, q5_k, QT_Q5_K)
GEMVX_ENTRY(6, 1, 3, iq4_xs, QT_IQ4_XS)
GEMVX_ENTRY(6, 1, 3, mxfp4, QT_MXFP4)
GEMVX_ENTRY(6, 1, 3, q6_k, QT_Q6_K)

