template <> struct QD<QT_IQ3_S> {  // shared int8 GEMV unit dots QD<IQ3_S> (both kernel sets, ADR-0005); see the end of this file
    static constexpr int per_sb = 32, sb = 256, bytes = 110;
    static __device__ __forceinline__ float dot(const uint8_t *row, int u, const int8_t *xq, const float *xd) {
        const int b = u >> 5, ib32 = (u >> 2) & 7, l = u & 3;
        const uint8_t *blk = row + (size_t)b * bytes;
        const float d = h2f(blk);
        const uint8_t *qs = blk + 2, *qh = blk + 66, *signs = blk + 74, *scales = blk + 106;
        const float db = d * (1 + 2 * ((scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xF));
        const uint8_t h = qh[ib32];
        const uint32_t g1 = k_iq3s_grid[qs[8 * ib32 + 2 * l] | ((h << (8 - 2 * l)) & 256)];
        const uint32_t g2 = k_iq3s_grid[qs[8 * ib32 + 2 * l + 1] | ((h << (7 - 2 * l)) & 256)];
        const uint32_t sg = signs[4 * ib32 + l];
        const uint32_t m1 = (((sg & 0xF) * 0x00204081u) & 0x01010101u) * 0xFFu;
        const uint32_t m2 = (((sg >> 4) * 0x00204081u) & 0x01010101u) * 0xFFu;
        const int v1 = (int)((g1 ^ m1) + (m1 & 0x01010101u));
        const int v2 = (int)((g2 ^ m2) + (m2 & 0x01010101u));
        const int base = b * 256 + 32 * ib32 + 8 * l;
        const int2 xv = *(const int2 *)(xq + base);
        return db * GQ_XD(base >> 5) * dp4(v2, xv.y, dp4(v1, xv.x, 0));
    }
};

// kernels/shared/gemv_q_qd_iq3s.h -- QD<QT_IQ3_S>, the IQ3_S unit dot of the int8 decode GEMV (8 values from two grid
// entries). One copy for both kernel sets (kernels/gemv_q.hip, kernels/gfx1151/gemv_q.hip); ADR-0005 / arch025-sc11e.
// Uses the includer's GQ_XD(e) (see gemv_q_qd.h).
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
