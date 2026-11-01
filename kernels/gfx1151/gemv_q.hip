// WHIRL device code (gfx1151 / RDNA 3.5): int8 decode GEMV (one token, grouped ABI) and multi-token WMMA GEMV.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

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
    const int q = i < n ? __float2int_rn(v * id) : 0;
    const int sum = warp_sum_i(q);
    if (i < n) {
        xq[i] = (int8_t)q;
        if ((i & 31) == 0) xd[i >> 5] = pk_make(d, sum);
    }
}



__device__ __forceinline__ uint32_t ld32u(const uint8_t *p) {  // 2-byte aligned (all call sites)
    uint32_t v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 4);
    return v;
}

// Activation-scale reads of the shared unit dots (gfx1151: packed scale words, xdv()); ADR-0005.
#define GQ_XD(e) xdv(xd + (e))
#define GQ_XD_B xdv(xd + b)
#include "shared/gemv_q_qd.h"  // QD, QD<Q4_K .. Q3_K>, both kernel sets (ADR-0005)

// KLIB-1 (opt-in, WHIRL_Q3K_U16=1): Q3_K with the unit split of llama.cpp's vec_dot_q3_K_q8_1 (ggml-cuda/vecdotq.cuh,
// MIT, Copyright (c) 2023-2026 The ggml authors). Unit = (superblock b, iqs 0..15): one 4-byte qs word (4 positions x
// 4 two-bit planes) and one 4-byte hmask word give 16 weights; plane i is 32-group 4 (iqs / 8) + i, positions
// 4 (iqs % 8) .. +3. The QD<QT_Q3_K> unit above decodes 8 weights of one plane from 16 bytes (each qs byte read by four
// units). Different summation order than QD<QT_Q3_K> (not bit-identical to it); the 1-token and multi-token
// entries share q3u_tok, so verify == decode bit for bit within this variant.
#define QT_Q3_K_U16 1011
struct Q3uW {
    int vl[4], vh[4], sc[4];
    float d;
    int base;
};
__device__ __forceinline__ Q3uW q3u_dec(const uint8_t *row, int u) {
    const int b = u >> 4, iqs = u & 15, n = iqs >> 3, l4 = iqs & 7;
    const uint8_t *blk = row + (size_t)b * 110;
    const uint32_t ql = ld32u(blk + 32 + 4 * iqs);
    const uint32_t qh = ~ld32u(blk + 4 * l4) >> (4 * n);
    const uint8_t *scales = blk + 96;
    Q3uW w;
    w.d = h2f(blk + 108);
    w.base = b * 256 + 128 * n + 4 * l4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = 8 * n + (l4 >> 2) + 2 * i;
        const int lo = (scales[isc & 7] >> (4 * (isc >> 3))) & 0xF;
        const int hi = ((scales[8 + (isc & 3)] >> (2 * (isc >> 2))) & 3) << 4;
        w.sc[i] = (lo | hi) - 32;
        w.vl[i] = (int)((ql >> (2 * i)) & 0x03030303u);
        w.vh[i] = (int)(((qh >> i) << 2) & 0x04040404u);
    }
    return w;
}
__device__ __forceinline__ float q3u_tok(const Q3uW &w, const int8_t *xq, const float *xd) {
    float sumf = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = w.base + 32 * i;
        const int x = *(const int *)(xq + e);
        sumf += xdv(xd + (e >> 5)) * (float)((dp4(w.vl[i], x, 0) - dp4(w.vh[i], x, 0)) * w.sc[i]);
    }
    return w.d * sumf;
}
template <> struct QD<QT_Q3_K_U16> {
    static constexpr int per_sb = 16, sb = 256, bytes = 110;
    static __device__ __forceinline__ float dot(const uint8_t *row, int u, const int8_t *xq, const float *xd) {
        return q3u_tok(q3u_dec(row, u), xq, xd);
    }
};

// IQ3_S unit = (superblock, ib32 0..7, l 0..3): 8 values from two grid entries.
#include "shared/gemv_q_qd_iq3s.h"  // QD<IQ3_S>, both kernel sets (ADR-0005)
#undef GQ_XD
#undef GQ_XD_B

// Multi-token unit dot: decode the weight unit once, dot it with NT tokens.
// Generic fallback re-runs the single-token dot (weight part left to CSE).
#include "shared/gemv_q_part_2.h"  // also in the gfx1201 set (kernels/gemv_q.hip)

template <> struct QDN<QT_Q3_K_U16> {
    template <int NT>
    static __device__ __forceinline__ void dot(const uint8_t *row, int u, const int8_t *xq, const float *xd, int ncols,
                                               float *acc) {
        const Q3uW w = q3u_dec(row, u);
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] += q3u_tok(w, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32));
    }
};

// Q4_K / Q5_K share the tail: 4 packed int8x4 weights (two 32-sub-blocks),
// per-sub-block scale/min, and x sums for the min term.
#define GQ_X0 xdv(xdt)  // gfx1151: packed scale words
#define GQ_X1 xdv(xdt + 1)
#include "shared/gemv_q_kq_tail.h"  // kq_tail<NT>, both kernel sets (ADR-0005)
#undef GQ_X0
#undef GQ_X1

#include "shared/gemv_q_gemv_grouped_abi.h"  // also in the gfx1201 set (kernels/gemv_q.hip)
// ---- grouped ABI (end) ----
template <int T>
__device__ void gemvq_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                           const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                           const int *__restrict__ skip, int bid = -1) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = (bid < 0 ? (int)blockIdx.x : bid) * (blockDim.x / WARP) + threadIdx.x / WARP;
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
__device__ void gemvq_nt_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                              const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                              const int *__restrict__ skip, int bid = -1) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = (bid < 0 ? (int)blockIdx.x : bid) * (blockDim.x / WARP) + threadIdx.x / WARP;
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
