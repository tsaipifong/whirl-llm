        const uint2 g0 = *(const uint2 *)(blk + 8 + 32 * pp + 8 * h);  // shared 4-bit WMMA unit dots + fold (ADR-0005); see the end of this file
        const uint2 g1 = *(const uint2 *)(blk + 24 + 32 * pp + 8 * h);
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        wi8 c0 = wmma_iu8(x.a0.x, x.a0.y, GW4_LUT(g0.x & 0x0F0F0F0F), GW4_LUT(g0.y & 0x0F0F0F0F), z);
        c0 = wmma_iu8(x.a1.x, x.a1.y, GW4_LUT((g0.x >> 4) & 0x0F0F0F0F), GW4_LUT((g0.y >> 4) & 0x0F0F0F0F), c0);
        wi8 c1 = wmma_iu8(x.c0.x, x.c0.y, GW4_LUT(g1.x & 0x0F0F0F0F), GW4_LUT(g1.y & 0x0F0F0F0F), z);
        c1 = wmma_iu8(x.c1.x, x.c1.y, GW4_LUT((g1.x >> 4) & 0x0F0F0F0F), GW4_LUT((g1.y >> 4) & 0x0F0F0F0F), c1);
        const int xi = (b * 256 + 64 * pp) >> 5;
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const int t = min(t0 + v, ntm1);
            const float *xdt = xd + (size_t)t * nb32 + xi;
            const float x0 = xdt[0], x1 = xdt[1];
            const int s0 = c0[v], s1 = c1[v];
            acc[v] += dl0 * x0 * s0 + dl1 * x1 * s1;
        }
#undef GW4_LUT

// kernels/shared/gemvw_unit4_run.h -- body of WMU<T>::run for the 4-bit types (IQ4_XS, MXFP4 in kernels/gemv_w.hip)
// after the type's scale decode (dl0, dl1): the two 16-byte nibble groups g0 / g1 of sub-blocks 2p, 2p+1 through
// the type's 4-bit LUT GW4_LUT (iq4_lut4 / mx_lut4, defined by the includer, #undef'd here) into two int8 WMMA
// dot pairs c0, c1, then acc[v] += dl0 x0 s0 + dl1 x1 s1 for the NV tokens (the MRD float expression, so
// gemvw == gemvq bitwise). ADR-0005 / arch025-sc11d. The includer provides row block blk, b, pp, h, x, xd, nb32,
// t0, ntm1, acc.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
