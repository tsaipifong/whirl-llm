template <> struct MRD<QT_Q6_K> {  // shared multi-row GEMV Q6_K unit (both kernel sets, ADR-0005); see the end of this file
    static constexpr int per_sb = QD<QT_Q6_K>::per_sb;
    template <int NT, int R>
    static __device__ __forceinline__ void step(const uint8_t *const *wr, int u, const int8_t *xq, const float *xd,
                                                int ncols, float (*acc)[NT]) {
        const int b = u >> 3, n = (u >> 2) & 1, g = u & 3;
        const int is = g >> 1;
        int v0[R][4], v1[R][4];
        float scf[R][4], dd[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t *blk = wr[r] + (size_t)b * 210;
            const uint8_t *ql = blk + 64 * n + 8 * g, *qh = blk + 128 + 32 * n + 8 * g;
            const int8_t *sc = (const int8_t *)(blk + 192) + 8 * n;
            dd[r] = h2f(blk + 208);
            const uint32_t a0 = ld32u(ql), a1 = ld32u(ql + 4), c0 = ld32u(ql + 32), c1 = ld32u(ql + 36);
            const uint32_t h0 = ld32u(qh), h1 = ld32u(qh + 4);
#pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                const uint32_t l0 = (quarter & 1) ? c0 : a0, l1 = (quarter & 1) ? c1 : a1;
                const int ls = (quarter >> 1) ? 4 : 0;
                v0[r][quarter] = ((l0 >> ls) & 0x0F0F0F0F) | (((h0 >> (2 * quarter)) & 0x03030303) << 4);
                v1[r][quarter] = ((l1 >> ls) & 0x0F0F0F0F) | (((h1 >> (2 * quarter)) & 0x03030303) << 4);
                scf[r][quarter] = (float)sc[is + 2 * quarter];
            }
        }
        const int base = b * 256 + 128 * n + 8 * g;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int8_t *xt = xq + (size_t)t * ncols;
            const float *xdt = xd + (size_t)t * (ncols / 32);
            int2 xv[4];
            int sv[4];
            float xs[4];
#pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                xv[quarter] = *(const int2 *)(xt + base + 32 * quarter);
                sv[quarter] = dp4(0x01010101, xv[quarter].y, dp4(0x01010101, xv[quarter].x, 0));
                xs[quarter] = GMR_XQ((base + 32 * quarter) >> 5);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float total = 0.f;
#pragma unroll
                for (int quarter = 0; quarter < 4; ++quarter) {
                    const int dv = dp4(v1[r][quarter], xv[quarter].y, dp4(v0[r][quarter], xv[quarter].x, 0));
                    total += scf[r][quarter] * xs[quarter] * (float)(dv - 32 * sv[quarter]);
                }
                acc[r][t] += dd[r] * total;
            }
        }
    }
};

// kernels/shared/gemv_mr_q6.h -- MRD<QT_Q6_K>, the Q6_K unit of the multi-row multi-token int8 GEMV: per row 4
// quarters of 2 int8x4 weights, 4 scales and d. One copy for both kernel sets (kernels/gemv_mr.hip,
// kernels/gfx1151/gemv_mr.hip); ADR-0005 / arch025-sc11e. The includer defines GMR_XQ(e), the activation scale of
// 32-group e of the token (gfx1201 xdt[e], gfx1151 xdv(xdt + (e))), and #undefs it afterwards.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
