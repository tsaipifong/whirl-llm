// WHIRL device code, shared by both kernel sets (gfx1201 kernels/prefill_fused.hip, gfx1151 kernels/gfx1151/prefill_fused.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

extern "C" __global__ void __launch_bounds__(256) silu_mul_x16(const float *g, const float *u, uint8_t *q, float *sx, int ncols) {
    silu_mul_x_impl<false>(g, u, q, sx, ncols);
}

// gdn_gated_norm (per head of 128: rmsnorm * w * silu(z)) fused with the out-proj
// input conversion: one 256-thread block per token, wave w takes heads w, w + 8, ..
// The per-head sum of squares follows block_sum over 128 threads exactly:
// s_k = warp_sum over elements 32k + lane, total = (s0 + s2) + (s1 + s3).
#define GN_HPW 8
template <bool FP8, typename ZT = float, bool TL = false>
__device__ void gated_norm_x_impl(const float *o, const ZT *z, const float *w, uint8_t *q, float *sx, int n_heads, float eps) {
    int dv = 128;
    asm volatile("" : "+s"(dv));  // runtime divisor like gdn_gated_norm (ss / dv)
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int row = n_heads * 128;
    const float *orow = o + (size_t)blockIdx.x * row;
    const ZT *zrow = z + (size_t)blockIdx.x * row;
    float y[GN_HPW][4];
    float m = 0.f;
#pragma unroll
    for (int hh = 0; hh < GN_HPW; ++hh) {
        const int h = wid + 8 * hh;
        if (h < n_heads) {
            float s[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                y[hh][k] = orow[h * 128 + 32 * k + lane];
                float sq = y[hh][k] * y[hh][k];
                asm volatile("" : "+v"(sq));  // no fma into the first butterfly add (block_sum sums rounded squares)
                s[k] = warp_sum(sq);
            }
            // second level exactly as block_sum does it: wave 0 lanes 0..3 hold s0..s3, warp_sum, lane 0
            float t4 = lane == 0 ? s[0] : lane == 1 ? s[1] : lane == 2 ? s[2] : lane == 3 ? s[3] : 0.f;
            t4 = warp_sum(t4);
            const float ss = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, t4)));  // uniform: the norm factor goes through the scalar unit as in gdn_gated_norm
            const float r = rsqrtf(ss / dv + eps);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int i = 32 * k + lane;
                const float zz = (float)zrow[h * 128 + i];
                y[hh][k] = y[hh][k] * r * w[i] * (zz / (1.f + __expf(-zz)));
                asm volatile("" : "+v"(y[hh][k]));
                m = fmaxf(m, fabsf(y[hh][k]));
            }
        }
    }
    if (FP8) {
        m = block_max(m);
        const float inv = m > 0.f ? 448.f / m : 0.f;
        if (tid == 0) sx[blockIdx.x] = m > 0.f ? m / 448.f : 0.f;
        uint8_t *qr = q + (size_t)blockIdx.x * row;
#pragma unroll
        for (int hh = 0; hh < GN_HPW; ++hh) {
            const int h = wid + 8 * hh;
            if (h < n_heads)
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (TL) q[x8t_off(blockIdx.x, h * 128 + 32 * k + lane, row)] = fp8_of(y[hh][k], inv);
                    else qr[h * 128 + 32 * k + lane] = fp8_of(y[hh][k], inv);
                }
        }
    } else {
        _Float16 *qr = (_Float16 *)q + (size_t)blockIdx.x * row;
#pragma unroll
        for (int hh = 0; hh < GN_HPW; ++hh) {
            const int h = wid + 8 * hh;
            if (h < n_heads)
#pragma unroll
                for (int k = 0; k < 4; ++k) qr[h * 128 + 32 * k + lane] = (_Float16)y[hh][k];
        }
    }
}
