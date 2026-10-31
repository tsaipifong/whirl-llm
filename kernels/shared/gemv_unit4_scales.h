            float dl0, dl1;  // shared IQ4_XS / MXFP4 unit scales (ADR-0005); see the end of this file
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

// kernels/shared/gemv_unit4_scales.h -- tile tt of a 4-bit unit (sub-blocks 2pp, 2pp+1) in the pipelined /
// wide WMMA GEMVs: IQ4_XS d * (6-bit scale - 32) from the fetched header and scale byte, MXFP4 the two e8m0
// exponents; and the tile's two nibble groups g0, g1. #included in GU_4L::compute (kernels/gemv_wp.hip) and
// GX_4L::compute (kernels/gemv_wx.hip); ADR-0005 / arch025-sc11d. The includer provides p (hdr, sc, g0, g1),
// tt, pp and QT.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
