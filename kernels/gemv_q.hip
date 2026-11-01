// WHIRL device code (gfx1201 / RDNA 4): decode GEMV with int8 activations (gemvq_*), multi-token gemvq_nt*, grouped ABI, Q3_K / IQ3_S paths.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Decode GEMV with int8 activations (llama.cpp mmvq idea): x is quantized to
// int8 in blocks of 32 (scale xd), weights are consumed as packed nibbles/bytes
// and multiplied with v_dot4 (4 int8 MACs per instruction).
// Each lane handles one "unit" of 16 or 32 weights per iteration.
// ---------------------------------------------------------------------------

// x (f32, n) -> xq (int8, n), xd (f32, n/32). One wave per 32-block.
extern "C" __global__ void quantize_q8(const float *x, int8_t *xq, float *xd, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float v = i < n ? x[i] : 0.f;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    if (i < n) {
        xq[i] = (int8_t)__float2int_rn(v * id);
        if ((i & 31) == 0) xd[i >> 5] = d;
    }
}


__device__ __forceinline__ uint32_t ld32u(const uint8_t *p) {  // unaligned-safe
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Activation-scale reads of the shared unit dots (gfx1201: plain float scales); ADR-0005.
#define GQ_XD(e) xd[e]
#define GQ_XD_B xd[b]
#include "shared/gemv_q_qd.h"  // QD, QD<Q4_K .. Q3_K>, both kernel sets (ADR-0005)

// IQ3_S unit = (superblock, ib32 0..7, l 0..3): 8 values from two grid entries.
#include "shared/gemv_q_qd_iq3s.h"  // QD<IQ3_S>, both kernel sets (ADR-0005)
#undef GQ_XD
#undef GQ_XD_B

// Multi-token unit dot: decode the weight unit once, dot it with NT tokens.
// Generic fallback re-runs the single-token dot (weight part left to CSE).
#include "shared/gemv_q_part_2.h"  // also in the gfx1151 set (kernels/gfx1151/gemv_q.hip)

// Q4_K / Q5_K share the tail: 4 packed int8x4 weights (two 32-sub-blocks),
// per-sub-block scale/min, and x sums for the min term.
#define GQ_X0 xdt[0]  // gfx1201: plain float scales
#define GQ_X1 xdt[1]
#include "shared/gemv_q_kq_tail.h"  // kq_tail<NT>, both kernel sets (ADR-0005)
#undef GQ_X0
#undef GQ_X1

#include "shared/gemv_q_gemv_grouped_abi.h"  // also in the gfx1151 set (kernels/gfx1151/gemv_q.hip)
// ---- item T1b: grouped GEMV ABI (end) ----
template <int T>
__device__ __forceinline__ void gemvq_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                           const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                           const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = bid * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const int nu = ncols / QD<T>::sb * QD<T>::per_sb;
    float acc = 0.f;
    // same arithmetic as the multi-token path, so verify matches decode bitwise
    for (int u = lane; u < nu; u += WARP) QDN<T>::template dot<1>(wr, u, xq, xd, ncols, &acc);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = accumulate ? y[row] + acc : acc;
}

// Multi-token int8 GEMV (speculative verify / batched decode, 2..16 tokens): the same
// weight unit is dotted with every token's x; repeated weight loads hit the
// same addresses back to back, so DRAM traffic stays one pass over W.
// xq: [NT][ncols] int8, xd: [NT][ncols/32]; y: [NT][nrows].
template <int T, int NT>
__device__ __forceinline__ void gemvq_nt_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                              const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                              const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = bid * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const int nu = ncols / QD<T>::sb * QD<T>::per_sb;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int u = lane; u < nu; u += WARP) QDN<T>::template dot<NT>(wr, u, xq, xd, ncols, acc);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) {
            float *dst = y + (size_t)t * nrows + row;
            *dst = accumulate ? *dst + v : v;
        }
    }
}

// ---- item T1: Q3_K / IQ3_S decode GEMV (begin) ----
// One-token (and Q3_K few-token dp4) GEMV for the 8-value-unit types. Lane l owns
// in-superblock unit l of every superblock (per_sb == WARP), exactly as in
// gemvq_impl / gemvq_nt_impl, and every (unit, token) term is the QD<T>::dot float
// expression, accumulated in superblock order, then the same warp_sum => bitwise
// identical to gemvq_impl<T> / gemvq_nt_impl<T, NT>. What changes: (1) the Q3_K
// scale nibble is read branch-free (kk < 8 selected a different byte through a
// divergent branch, which serialized a scale load before the weight loads), (2) U
// superblocks' loads are issued before their terms are computed (one 110-byte
// superblock per wave per step was too little memory-level parallelism):
// 17408x5120 q3_k 491 -> 590 GB/s, iq3_s 502 -> 583 GB/s (pure read ceiling ~595).
typedef uint32_t t1_u32a2 __attribute__((aligned(2)));
typedef uint16_t t1_u16a2 __attribute__((aligned(2)));
struct T1Q3W {
    uint32_t dh, slo, shi, q0, q1, h0, h1;
};
struct T1Q3Lane {
    int j, sh, lo_off, lo_sh, hi_off, hi_sh, q_off, h_off, bl;
    __device__ __forceinline__ T1Q3Lane(int lane) {
        const int n = lane >> 4, sg = lane & 3;
        j = (lane >> 2) & 3;
        const int kk = 8 * n + 2 * j + (sg >> 1);
        lo_off = 96 + (kk & 7);
        lo_sh = kk < 8 ? 0 : 4;
        hi_off = 96 + 8 + (kk & 3);
        hi_sh = 2 * (kk >> 2);
        q_off = 32 + 32 * n + 8 * sg;
        h_off = 8 * sg;
        sh = 4 * n + j;
        bl = 128 * n + 32 * j + 8 * sg;
    }
    __device__ __forceinline__ T1Q3W load(const uint8_t *blk) const {
        T1Q3W c;
        c.dh = *(const t1_u16a2 *)(blk + 108);
        c.slo = blk[lo_off];
        c.shi = blk[hi_off];
        c.q0 = *(const t1_u32a2 *)(blk + q_off);
        c.q1 = *(const t1_u32a2 *)(blk + q_off + 4);
        c.h0 = *(const t1_u32a2 *)(blk + h_off);
        c.h1 = *(const t1_u32a2 *)(blk + h_off + 4);
        return c;
    }
};

template <int U>
__device__ __forceinline__ void t1_q3k_1_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                              const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                              const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = bid * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const int nsb = ncols / 256;
    const T1Q3Lane L(lane);
    float acc = 0.f;
    auto term = [&](const T1Q3W &c, int2 xv, float xs) {
        const float d = __half2float(__ushort_as_half((uint16_t)c.dh));
        const int lo = (c.slo >> L.lo_sh) & 0xF;
        const int hi = (c.shi >> L.hi_sh) & 3;
        const float dl = d * ((lo | (hi << 4)) - 32);
        const int w0 = (c.q0 >> (2 * L.j)) & 0x03030303, w1 = (c.q1 >> (2 * L.j)) & 0x03030303;
        const int m0 = ((~c.h0) >> L.sh) & 0x01010101, m1 = ((~c.h1) >> L.sh) & 0x01010101;
        const int dv = dp4(w1, xv.y, dp4(w0, xv.x, 0)) - 4 * dp4(m1, xv.y, dp4(m0, xv.x, 0));
        return dl * xs * dv;
    };
    int b = 0;
    for (; b + U <= nsb; b += U) {
        T1Q3W c[U];
        int2 xv[U];
        float xs[U];
#pragma unroll
        for (int k = 0; k < U; ++k) {
            c[k] = L.load(wr + (size_t)(b + k) * 110);
            xv[k] = *(const int2 *)(xq + (b + k) * 256 + L.bl);
            xs[k] = xd[((b + k) * 256 + L.bl) >> 5];
        }
#pragma unroll
        for (int k = 0; k < U; ++k) acc += term(c[k], xv[k], xs[k]);
    }
    for (; b < nsb; ++b)
        acc += term(L.load(wr + (size_t)b * 110), *(const int2 *)(xq + b * 256 + L.bl), xd[(b * 256 + L.bl) >> 5]);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = accumulate ? y[row] + acc : acc;
}

// Q3_K few-token (the dp4 path; tokens share the decoded unit). Replaces
// gemvq_nt_impl<QT_Q3_K, NT>: 17408x5120 T=2 0.104 -> 0.073 ms, T=5 0.198 -> 0.116 ms.
template <int NT>
__device__ __forceinline__ void t1_q3k_nt_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                               const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                               const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = bid * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const int nsb = ncols / 256;
    const T1Q3Lane L(lane);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int b = 0; b < nsb; ++b) {
        const T1Q3W c = L.load(wr + (size_t)b * 110);
        int2 xv[NT];
        float xs[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            xv[t] = *(const int2 *)(xq + (size_t)t * ncols + b * 256 + L.bl);
            xs[t] = xd[(size_t)t * (ncols / 32) + ((b * 256 + L.bl) >> 5)];
        }
        const float d = __half2float(__ushort_as_half((uint16_t)c.dh));
        const int lo = (c.slo >> L.lo_sh) & 0xF;
        const int hi = (c.shi >> L.hi_sh) & 3;
        const float dl = d * ((lo | (hi << 4)) - 32);
        const int w0 = (c.q0 >> (2 * L.j)) & 0x03030303, w1 = (c.q1 >> (2 * L.j)) & 0x03030303;
        const int m0 = ((~c.h0) >> L.sh) & 0x01010101, m1 = ((~c.h1) >> L.sh) & 0x01010101;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int dv = dp4(w1, xv[t].y, dp4(w0, xv[t].x, 0)) - 4 * dp4(m1, xv[t].y, dp4(m0, xv[t].x, 0));
            acc[t] += dl * xs[t] * dv;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) {
            float *dst = y + (size_t)t * nrows + row;
            *dst = accumulate ? *dst + v : v;
        }
    }
}

struct T1I3W {
    uint32_t dh, sc, h, qa, sg;
};
template <int U>
__device__ __forceinline__ void t1_iq3s_1_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                               const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                               const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = bid * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const uint8_t *wr = W + (size_t)row * row_bytes;
    const int nsb = ncols / 256;
    const int ib32 = lane >> 2, l = lane & 3;
    const int bl = 32 * ib32 + 8 * l;
    auto ld = [&](int b) {
        const uint8_t *blk = wr + (size_t)b * 110;
        T1I3W r;
        r.dh = *(const t1_u16a2 *)blk;
        r.sc = blk[106 + (ib32 >> 1)];
        r.h = blk[66 + ib32];
        r.qa = *(const t1_u16a2 *)(blk + 2 + 8 * ib32 + 2 * l);
        r.sg = blk[74 + 4 * ib32 + l];
        return r;
    };
    auto term = [&](const T1I3W &c, int2 xv, float xs) {
        const float d = __half2float(__ushort_as_half((uint16_t)c.dh));
        const float db = d * (1 + 2 * ((c.sc >> (4 * (ib32 & 1))) & 0xF));
        const uint32_t g1 = k_iq3s_grid[(c.qa & 0xFF) | ((c.h << (8 - 2 * l)) & 256)];
        const uint32_t g2 = k_iq3s_grid[(c.qa >> 8) | ((c.h << (7 - 2 * l)) & 256)];
        const uint32_t m1 = (((c.sg & 0xF) * 0x00204081u) & 0x01010101u) * 0xFFu;
        const uint32_t m2 = (((c.sg >> 4) * 0x00204081u) & 0x01010101u) * 0xFFu;
        const int v1 = (int)((g1 ^ m1) + (m1 & 0x01010101u));
        const int v2 = (int)((g2 ^ m2) + (m2 & 0x01010101u));
        return db * xs * dp4(v2, xv.y, dp4(v1, xv.x, 0));
    };
    float acc = 0.f;
    int b = 0;
    for (; b + U <= nsb; b += U) {
        T1I3W c[U];
        int2 xv[U];
        float xs[U];
#pragma unroll
        for (int k = 0; k < U; ++k) {
            c[k] = ld(b + k);
            xv[k] = *(const int2 *)(xq + (b + k) * 256 + bl);
            xs[k] = xd[((b + k) * 256 + bl) >> 5];
        }
#pragma unroll
        for (int k = 0; k < U; ++k) acc += term(c[k], xv[k], xs[k]);
    }
    for (; b < nsb; ++b) acc += term(ld(b), *(const int2 *)(xq + b * 256 + bl), xd[(b * 256 + bl) >> 5]);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = accumulate ? y[row] + acc : acc;
}
// ---- item T1 (end) ----


