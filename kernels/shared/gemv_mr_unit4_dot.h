        const int base = b * 256 + 64 * pp, xi = base >> 5;  // shared 4-bit wide-unit dot of gemvq_mr (ADR-0005); see the end of this file
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int8_t *xt = xq + (size_t)t * ncols + base;  // sub-block 2p: cols [0, 32) of the unit, sub-block 2p+1: [32, 64)
            const int4 a0 = *(const int4 *)xt, a1 = *(const int4 *)(xt + 16);
            const int4 c0 = *(const int4 *)(xt + 32), c1 = *(const int4 *)(xt + 48);
            const float *xdt = xd + (size_t)t * (ncols / 32) + xi;
            const float x0 = xdt[0], x1 = xdt[1];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int s0 = dp4(lo[r][0], a0.x, 0);  // lo[0..3]: sub-block 2p bytes 0..15 (low -> a0, high -> a1)
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

// kernels/shared/gemv_mr_unit4_dot.h -- the token side of a 4-bit wide unit of the multi-row GEMV (IQ4_XS and
// MXFP4 in kernels/gemv_mr.hip): for each of the NT tokens the int8 activations of sub-blocks 2p and 2p+1 and
// their scales, the two 32-value integer dots per row and acc[r][t] += dl0 x0 s0 + dl1 x1 s1 (the canonical
// MRD float expression). #included in MRD<QT_IQ4_XS>::step and MRD<QT_MXFP4>::step after the type's own
// decode into lo / hi / dl0 / dl1; ADR-0005 / arch025-sc11d.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
