    const int b = u >> 2, j = u & 3;  // shared gfx1151 Q5_K unit head (ADR-0005); see the end of this file
    const uint8_t *blk = row + (size_t)b * 176;
    const uint32_t dm = *(const uint32_t *)blk;
    const float d = __half2float(__ushort_as_half((uint16_t)(dm & 0xFFFF)));
    const float dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
    int s0, m0, s1, m1;
    scale_min_k4(2 * j, blk + 4, s0, m0);
    scale_min_k4(2 * j + 1, blk + 4, s1, m1);
    o.s0 = d * s0;
    o.s1 = d * s1;
    o.m0 = dmin * m0;
    o.m1 = dmin * m1;
    const uint4 qa = *(const uint4 *)(blk + 48 + 32 * j);
    const uint4 qb = *(const uint4 *)(blk + 64 + 32 * j);
    const uint4 ha = *(const uint4 *)(blk + 16);
    const uint4 hb = *(const uint4 *)(blk + 32);
    const uint32_t q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
    const uint32_t h[8] = {ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w};

// kernels/shared/wunit11_q5k_head.h -- Q5_K unit u = (superblock b, pair j) of the gfx1151 WMMA GEMV:
// d / dmin, the 6-bit scales and mins of sub-blocks 2j, 2j+1 into o (s0, s1, m0, m1), the 32 quant bytes q[8]
// and the high-bit words h[8]. #included in wunit<QT_Q5_K> and wunit_half<QT_Q5_K> (kernels/gfx1151/
// gemv_wunit.hip); ADR-0005 / arch025-sc11d. The includer provides row, u, o.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
