// WHIRL device code (gfx1201 / RDNA 4): 2-bit MTP draft head GEMV (gemv_d2_nt*).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

#define D2_XD(p) (*(p))  // plain f32 activation scales
// ---------------------------------------------------------------------------
// 2-bit MTP draft head ("D2"). Only the draft argmax matters (a wrong draft
// costs one rejected verify row), and measured acceptance does not depend on
// the head's precision (Q6_K / Q4_K / Q3_K heads accept the same), so the
// draft head streams 2.5 bits per weight instead of Q4_K's 4.5.
// Row layout (K columns): codes [K/4 bytes] then f16 scales [K/32]:
//   unit u = 64 values = 16 code bytes at 16u; value at unit position
//   16k + 4kk + j is the 2-bit field (bits 8j + 2kk) of word k, so
//   (w_k >> 2kk) & 0x03030303 gives 4 codes in x order;
//   block b (32 values) has scale s_b: w = s_b * (2c - 3), c in 0..3.
// Arithmetic is free-form (drafts never have to match anything bitwise).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
    requant_q6k_d2(const uint8_t *src, uint64_t src_rb, uint8_t *dst, int ncols) {
    const int row = blockIdx.x, sbi = blockIdx.y, t = threadIdx.x, lane = t & 31, bj = t >> 5;
    const uint8_t *b = src + (size_t)row * src_rb + (size_t)sbi * 210;
    uint8_t *o = dst + (size_t)row * (ncols / 4 + ncols / 16);
    __shared__ float v[256];
    __shared__ uint32_t words[16];
    if (t < 16) words[t] = 0;
    if ((t & 7) == 0) Q<QT_Q6_K>::dec8(b, t, v + t);
    __syncthreads();
    const float x = v[t];
    const float amax = warp_max(fabsf(x));
    // pick the scale (of a few candidates) with the least squared error
    const float fs[6] = {1.f, 0.85f, 0.72f, 0.62f, 0.54f, 0.46f};
    float best_s = amax / 3.f, best_e = INFINITY;
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const float s = amax / 3.f * fs[i];
        const int c = s > 0.f ? max(0, min(3, (int)rintf((x / s + 3.f) * 0.5f))) : 1;
        const float d = x - s * (float)(2 * c - 3);
        const float e = warp_sum(d * d);
        if (e < best_e) {
            best_e = e;
            best_s = s;
        }
    }
    const _Float16 sh = (_Float16)best_s;
    const float sf = (float)sh;
    const int c = sf > 0.f ? max(0, min(3, (int)rintf((x / sf + 3.f) * 0.5f))) : 1;
    const int blk = sbi * 8 + bj;                   // block index in the row
    const int p = 32 * (blk & 1) + lane;            // position in its 64-value unit
    const int k = p >> 4, q = p & 15, kk = q >> 2, j = q & 3;
    atomicOr(&words[(bj >> 1) * 4 + k], (uint32_t)c << (8 * j + 2 * kk));
    __syncthreads();
    if (t < 16) ((uint32_t *)(o + (size_t)sbi * 64))[t] = words[t];  // 4 units x 4 words
    if (lane == 0) *(_Float16 *)(o + ncols / 4 + 2 * blk) = sh;
}

// y[t][r] = draft-head logits for NT tokens (int8 x, xd per 32), one wave per
// row, lane per 64-value unit.
template <int NT>
__device__ void gemv_d2_impl(const uint8_t *__restrict__ W, const int8_t *__restrict__ xq, const float *__restrict__ xd,
                             float *__restrict__ y, int ncols, int nrows, const int *__restrict__ skip) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const size_t rb = (size_t)(ncols / 4 + ncols / 16);
    const uint8_t *codes = W + (size_t)row * rb;
    const uint16_t *sc = (const uint16_t *)(codes + ncols / 4);
    const int nu = ncols / 64, nb = ncols / 32;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int u = lane; u < nu; u += WARP) {
        const uint4 cw = *(const uint4 *)(codes + 16 * u);
        const uint32_t s2 = *(const uint32_t *)(sc + 2 * u);
        const float s0 = __half2float(__ushort_as_half((uint16_t)(s2 & 0xFFFF)));
        const float s1 = __half2float(__ushort_as_half((uint16_t)(s2 >> 16)));
        const uint32_t w[4] = {cw.x, cw.y, cw.z, cw.w};
        int cb[16];
#pragma unroll
        for (int k = 0; k < 4; ++k)
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) cb[4 * k + kk] = (int)((w[k] >> (2 * kk)) & 0x03030303u);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int8_t *xt = xq + (size_t)t * ncols + 64 * u;
            const int4 x0 = *(const int4 *)xt, x1 = *(const int4 *)(xt + 16);
            const int4 x2 = *(const int4 *)(xt + 32), x3 = *(const int4 *)(xt + 48);
            const int xv[16] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w, x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
            int d0 = 0, d1 = 0, e0 = 0, e1 = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                d0 = dp4(cb[i], xv[i], d0);
                e0 = dp4(0x01010101, xv[i], e0);
                d1 = dp4(cb[8 + i], xv[8 + i], d1);
                e1 = dp4(0x01010101, xv[8 + i], e1);
            }
            const float *xdt = xd + (size_t)t * nb + 2 * u;
            acc[t] += s0 * D2_XD(xdt) * (float)(2 * d0 - 3 * e0) + s1 * D2_XD(xdt + 1) * (float)(2 * d1 - 3 * e1);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) y[(size_t)t * nrows + row] = s;
    }
}
#define GEMV_D2_ENTRY(NT)                                                                                   \
    extern "C" __global__ void __launch_bounds__(256)                                                       \
        gemv_d2_nt##NT(const uint8_t *W, const int8_t *xq, const float *xd, float *y, int ncols, int nrows, \
                       const int *skip) {                                                                   \
        gemv_d2_impl<NT>(W, xq, xd, y, ncols, nrows, skip);                                                 \
    }
GEMV_D2_ENTRY(1)
GEMV_D2_ENTRY(2)
GEMV_D2_ENTRY(3)
GEMV_D2_ENTRY(4)
GEMV_D2_ENTRY(5)
GEMV_D2_ENTRY(6)
GEMV_D2_ENTRY(7)
GEMV_D2_ENTRY(8)
GEMV_D2_ENTRY(9)
GEMV_D2_ENTRY(10)
GEMV_D2_ENTRY(11)
GEMV_D2_ENTRY(12)
GEMV_D2_ENTRY(13)
GEMV_D2_ENTRY(14)
GEMV_D2_ENTRY(15)
GEMV_D2_ENTRY(16)

