    const float d = __half2float(__ushort_as_half((uint16_t)(hd.x & 0xFFFF)));  // shared K-quant scale unpack (ADR-0005); see the end of this file
    const float dmin = __half2float(__ushort_as_half((uint16_t)(hd.x >> 16)));
    const int sel = 16 * (j & 1);
    const uint32_t A = hd.y >> sel, B = hd.z >> sel, C = hd.w >> sel;  // sc[k], sc[k+4], sc[k+8], k = 2(j&1)
    int s0, m0, s1, m1;
    if GKS_LOW_J {
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
#undef GKS_LOW_J

// kernels/shared/gemv_kscale.h -- Q4_K / Q5_K: the 6-bit scales and mins of sub-blocks 2j and 2j+1 from the
// superblock header hd (d, dmin f16 + 12 scale bytes) -> ds0 = d s0, ds1 = d s1, dm0 = dmin m0, dm1 = dmin m1.
// Sub-blocks 0..3 (j < 2) keep the low 6 bits of bytes 0..7; 4..7 join a nibble of bytes 8..11 with the top two
// bits of bytes 0..7. #included in WMU_K::unit (kernels/gemv_w.hip), GU_K::compute (kernels/gemv_wp.hip) and
// GX_K::compute (kernels/gemv_wx.hip); ADR-0005 / arch025-sc11d. The includer defines GKS_LOW_J, the
// parenthesised condition after if: (j < 2) where j is a runtime value, constexpr (!J2) where it is a template
// constant (#undef'd here), and provides hd (uint4) and j.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
