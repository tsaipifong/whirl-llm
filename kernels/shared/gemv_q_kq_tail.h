template <int NT>  // shared Q4_K / Q5_K multi-token tail (both kernel sets, ADR-0005); see the end of this file
__device__ __forceinline__ void kq_tail(int lo0, int lo1, int hi0, int hi1, float ds0, float ds1, float dm0, float dm1,
                                        int base, int xi, const int8_t *xq, const float *xd, int ncols, float *acc) {
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int8_t *xt = xq + (size_t)t * ncols + base;
        const int2 xl = *(const int2 *)xt;
        const int2 xh = *(const int2 *)(xt + 32);
        const int dl = dp4(lo1, xl.y, dp4(lo0, xl.x, 0));
        const int dh = dp4(hi1, xh.y, dp4(hi0, xh.x, 0));
        const int sl = dp4(0x01010101, xl.y, dp4(0x01010101, xl.x, 0));
        const int sh = dp4(0x01010101, xh.y, dp4(0x01010101, xh.x, 0));
        const float *xdt = xd + (size_t)t * (ncols / 32) + xi;
        acc[t] += GQ_X0 * (ds0 * dl - dm0 * sl) + GQ_X1 * (ds1 * dh - dm1 * sh);
    }
}

// kernels/shared/gemv_q_kq_tail.h -- kq_tail<NT>, the Q4_K / Q5_K multi-token tail of the int8 GEMV (QDN<QT_Q4_K> /
// QDN<QT_Q5_K> in gemv_q_gemv_grouped_abi.h): per token the four int8 dots, the x sums of the min term and the two
// sub-block terms. One copy for both kernel sets (kernels/gemv_q.hip, kernels/gfx1151/gemv_q.hip); ADR-0005 /
// arch025-sc11e. File scope. The includer defines GQ_X0 / GQ_X1, the scales of the two sub-blocks relative to xdt
// (gfx1201 xdt[0] / xdt[1], gfx1151 packed scale words xdv(xdt) / xdv(xdt + 1)), and #undefs them afterwards.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
