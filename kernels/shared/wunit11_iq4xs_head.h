    const int b = u >> 2, pp = u & 3;  // shared gfx1151 IQ4_XS unit head (ADR-0005); see the end of this file
    const uint8_t *blk = row + (size_t)b * 136;
    const uint32_t hdr = *(const uint32_t *)blk;
    const float d = __half2float(__ushort_as_half((uint16_t)(hdr & 0xFFFF)));
    const uint32_t sh = hdr >> 16;
    const int ib0 = 2 * pp, ib1 = 2 * pp + 1;
    const int ls0 = (blk[4 + pp] & 0xF) | (((sh >> (2 * ib0)) & 3) << 4);
    const int ls1 = (blk[4 + pp] >> 4) | (((sh >> (2 * ib1)) & 3) << 4);
    o.s0 = d * (ls0 - 32);
    o.s1 = d * (ls1 - 32);
    o.m0 = 0.f;
    o.m1 = 0.f;

// kernels/shared/wunit11_iq4xs_head.h -- IQ4_XS unit u = (superblock b, pair pp) of the gfx1151 WMMA GEMV:
// d * (6-bit scale - 32) of sub-blocks 2pp, 2pp+1 into o.s0 / o.s1, no mins. #included in wunit<QT_IQ4_XS> and
// wunit_half<QT_IQ4_XS> (kernels/gfx1151/gemv_wunit.hip); ADR-0005 / arch025-sc11d. The includer provides
// row, u, o.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
