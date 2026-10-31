// WHIRL device code, shared by both kernel sets (gfx1201 kernels/gdn_ab.hip, gfx1151 kernels/gfx1151/gdn_ab.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).
// GDN beta/alpha projections for other quant types (see gdn_ab_q8_0); moved out of the moe family (ADR-0005).
#define GDN_AB_ENTRY(SUF, T)                                                                              \
    extern "C" __global__ void __launch_bounds__(256)                                                     \
        gdn_ab_##SUF(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq,         \
                     const float *xd, float *beta, float *g, const float *dt, const float *A, int ncols,  \
                     int nh) {                                                                            \
        const int t = blockIdx.y;                                                                         \
        gdn_ab_impl<T>(Wb, Wa, row_bytes, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32),          \
                       beta + (size_t)t * nh, g + (size_t)t * nh, dt, A, ncols, nh, blockIdx.x);                      \
    }
GDN_AB_ENTRY(q4_k, QT_Q4_K)
GDN_AB_ENTRY(q5_k, QT_Q5_K)
GDN_AB_ENTRY(q6_k, QT_Q6_K)
GDN_AB_ENTRY(iq4_xs, QT_IQ4_XS)
GDN_AB_ENTRY(mxfp4, QT_MXFP4)
GDN_AB_ENTRY(iq4_nl, QT_IQ4_NL)
GDN_AB_ENTRY(q3_k, QT_Q3_K)
GDN_AB_ENTRY(iq3_s, QT_IQ3_S)

// gdn_ab + gdn_conv_l2 in one launch (independent inputs): blocks
// [0, n_ab * ab_blocks) = gdn_ab (token b / ab_blocks, 8 rows per block),
// the rest = gdn_conv_l2 (segment, 128-channel block) with 128 of the 256
// threads active. Per block identical to the two separate kernels.
template <int T>
__device__ void gdn_abconv_impl(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq,
                                const float *xd, float *beta, float *g, const float *dt, const float *A, int ncols,
                                int nh, int n_ab, const float *xin, const float *w, float *out, int ch, int n_qk_heads,
                                float eps, const GdnSegs &segs) {
    const int ab_blocks = (2 * nh + 7) / 8;
    int b = blockIdx.x;
    if (b < n_ab * ab_blocks) {
        const int t = b / ab_blocks;
        gdn_ab_impl<T>(Wb, Wa, row_bytes, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32), beta + (size_t)t * nh,
                       g + (size_t)t * nh, dt, A, ncols, nh, b % ab_blocks);
        return;
    }
    b -= n_ab * ab_blocks;
    const int nblk = ch / 128;
    gdn_conv_l2_impl(xin, w, out, ch, n_qk_heads, eps, segs, b % nblk, b / nblk, threadIdx.x < 128);
}
#define GDN_ABCONV_ENTRY(SUF, T)                                                                               \
    extern "C" __global__ void __launch_bounds__(256)                                                          \
        gdn_abconv_##SUF(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq,          \
                         const float *xd, float *beta, float *g, const float *dt, const float *A, int ncols,   \
                         int nh, int n_ab, const float *xin, const float *w, float *out, int ch, int n_qk_heads, \
                         float eps, GdnSegs segs) {                                                            \
        gdn_abconv_impl<T>(Wb, Wa, row_bytes, xq, xd, beta, g, dt, A, ncols, nh, n_ab, xin, w, out, ch,       \
                           n_qk_heads, eps, segs);                                                             \
    }
GDN_ABCONV_ENTRY(q8_0, QT_Q8_0)
GDN_ABCONV_ENTRY(q4_k, QT_Q4_K)
GDN_ABCONV_ENTRY(q5_k, QT_Q5_K)
GDN_ABCONV_ENTRY(q6_k, QT_Q6_K)
GDN_ABCONV_ENTRY(iq4_xs, QT_IQ4_XS)
GDN_ABCONV_ENTRY(mxfp4, QT_MXFP4)
GDN_ABCONV_ENTRY(iq4_nl, QT_IQ4_NL)
GDN_ABCONV_ENTRY(q3_k, QT_Q3_K)
GDN_ABCONV_ENTRY(iq3_s, QT_IQ3_S)
