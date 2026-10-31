    const int lane = threadIdx.x % WARP;  // shared decode-expert gate/up pair prologue (ADR-0005); see the end of this file
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const int p = blockIdx.y;
    const int t = p / K, e = ids[p];
    const bool second = row >= ff;
    const int r = second ? row - ff : row;
    const uint8_t *wr = (second ? W2 : W1) + ((size_t)e * ff + r) * row_bytes;

// kernels/shared/moe_gu_pair.h -- prologue of the decode-expert gate + up kernels: one wave per output row, pair
// p = blockIdx.y = t * K + k runs expert e = ids[p] on token t; rows [0, ff) come from W1, [ff, nrows) from W2.
// #included inside the bodies of moe_gu_impl (kernels/moe.hip, kernels/gfx1151/moe.hip), moe_gu_f_impl and
// moe_gu_f_mxfp4 (kernels/moe_f.hip), moe_gu_mxfp4w (kernels/moe_mxfp4.hip); ADR-0005 / arch025-sc11d.
// The includer provides W1, W2, row_bytes, ff, nrows, ids, K; this block declares lane, row, p, t, e, second, r
// and wr (the weight row) and returns early for rows past nrows.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
