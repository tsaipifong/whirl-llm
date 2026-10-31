        float scf[4];  // shared Q6_K unit scales + weights (ADR-0005); see the end of this file
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

// kernels/shared/gemv_q6_unit.h -- Q6_K WMMA unit (u = 8b + 4n + g): the int8 scales of the four quarters
// (scf[q] = sc[is + 2q] of the 8-scale word pair scw, is = g >> 1) and the signed weights of quarters h and
// h + 2 (ql nibble | 2 qh bits, bytewise q - 32) in wq[e][k]. #included in WMU<QT_Q6_K>::run (kernels/gemv_w.hip)
// and the gw2 Q6_K unit (kernels/gemv_wp.hip); ADR-0005 / arch025-sc11d. The includer provides l, hq, scw, is, h.
// (GX's Q6_K unit in kernels/gemv_wx.hip names its inner index f: same expression, kept as is.)
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
