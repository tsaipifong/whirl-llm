// WHIRL device code, shared by both kernel sets (gfx1201 kernels/gemv_q.hip, gfx1151 kernels/gfx1151/gemv_q.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

template <> struct QDN<QT_Q4_K> {
    template <int NT>
    static __device__ __forceinline__ void dot(const uint8_t *row, int u, const int8_t *xq, const float *xd, int ncols,
                                               float *acc) {
        const int b = u >> 4, j = (u >> 2) & 3, i = u & 3;
        const uint8_t *blk = row + (size_t)b * 144;
        const uint32_t dm = *(const uint32_t *)blk;
        const float d = __half2float(__ushort_as_half((uint16_t)(dm & 0xFFFF)));
        const float dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
        const uint2 q = *(const uint2 *)(blk + 16 + 32 * j + 8 * i);
        int s0, m0, s1, m1;
        scale_min_k4(2 * j, blk + 4, s0, m0);
        scale_min_k4(2 * j + 1, blk + 4, s1, m1);
        kq_tail<NT>(q.x & 0x0F0F0F0F, q.y & 0x0F0F0F0F, (q.x >> 4) & 0x0F0F0F0F, (q.y >> 4) & 0x0F0F0F0F, d * s0, d * s1,
                    dmin * m0, dmin * m1, b * 256 + 64 * j + 8 * i, (b * 256 + 64 * j) >> 5, xq, xd, ncols, acc);
    }
};

template <> struct QDN<QT_Q5_K> {
    template <int NT>
    static __device__ __forceinline__ void dot(const uint8_t *row, int u, const int8_t *xq, const float *xd, int ncols,
                                               float *acc) {
        const int b = u >> 4, j = (u >> 2) & 3, i = u & 3;
        const uint8_t *blk = row + (size_t)b * 176;
        const uint32_t dm = *(const uint32_t *)blk;
        const float d = __half2float(__ushort_as_half((uint16_t)(dm & 0xFFFF)));
        const float dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
        const uint2 q = *(const uint2 *)(blk + 48 + 32 * j + 8 * i);
        const uint2 h = *(const uint2 *)(blk + 16 + 8 * i);
        int s0, m0, s1, m1;
        scale_min_k4(2 * j, blk + 4, s0, m0);
        scale_min_k4(2 * j + 1, blk + 4, s1, m1);
        kq_tail<NT>((q.x & 0x0F0F0F0F) | (((h.x >> (2 * j)) & 0x01010101) << 4),
                    (q.y & 0x0F0F0F0F) | (((h.y >> (2 * j)) & 0x01010101) << 4),
                    ((q.x >> 4) & 0x0F0F0F0F) | (((h.x >> (2 * j + 1)) & 0x01010101) << 4),
                    ((q.y >> 4) & 0x0F0F0F0F) | (((h.y >> (2 * j + 1)) & 0x01010101) << 4), d * s0, d * s1, dmin * m0,
                    dmin * m1, b * 256 + 64 * j + 8 * i, (b * 256 + 64 * j) >> 5, xq, xd, ncols, acc);
    }
};

// ---- item T1b: grouped GEMV ABI (begin) ----
// One launch of a one-token gemvq_<type> covers up to 3 matrices that share the input
// (attention q/k/v, FFN gate/up, DeltaNet qkv/z): blocks [0, b1) run segment 0,
// [b1, b2) segment 1, the rest segment 2, each with its own (W, row_bytes, y, nrows)
// and segment-local block index. Per-row arithmetic is that of the single launch
// (bitwise the same). The multi-token entries keep their own signature (selecting W
// per block changes their codegen: +30..90% VGPRs, spills on Q6_K / Q5_K v7); each
// has a grouped twin "g<name>" that the host uses only for same-input groups.
#include "kernel_args.h"  // GvArgs (defined once for both kernel sets, arch025 N13)
struct GvSel {
    const uint8_t *W;
    float *y;
    uint64_t rb;
    int nrows, b;
};
__device__ __forceinline__ GvSel gv_sel(const GvArgs &g) {
    const int b = blockIdx.x;
    if (b < g.b1) return {g.W0, g.y0, g.rb0, g.n0, b};
    if (b < g.b2) return {g.W1, g.y1, g.rb1, g.n1, b - g.b1};
    return {g.W2, g.y2, g.rb2, g.n2, b - g.b2};
}
#define GV_PARAMS const GvArgs g, const int8_t *xq, const float *xd, int ncols, int accumulate, const int *skip
// marker: the host passes GvArgs to the one-token gemvq_<type> entries
extern "C" __global__ void gemv_grouped_abi() {}
