// WHIRL device code (gfx1201 / RDNA 4): shared device helpers (warp/block reductions) and GGUF quant block decoders (dec8).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// HIP kernels for RDNA 4 (gfx1201), Qwen3.5-family (qwen35 / qwen35moe).
//
// Compiled at build time to a gfx1201 code object, embedded in the host
// binary and loaded with hipModuleLoadData; every entry point is extern "C"
// so the host finds it by name (see include/whirl/kernels_abi.h). Weights
// stay in their GGUF block formats; each quant type gets a `dec8` that
// decodes 8 consecutive values, and all matvecs share one loop.
//
// Numerics follow the GGUF block formats and llama.cpp's reference math
// (dequantization, Gated DeltaNet recurrence, NEOX RoPE); the C++ CPU
// references in tests/kernels/ check them.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "iq_tables.h"

#define WARP 32

// KV cache element type (f16: half the bytes of f32, and WMMA-ready).
typedef _Float16 kv_t;  // (f16 KV element; q8 pools use int8 + scales, see KvArgs)

// Debug slots (host reads via hipModuleGetGlobal "whirl_dbg").
__device__ unsigned long long whirl_dbg[8];

__device__ __forceinline__ float h2f(const uint8_t *p) {
    uint16_t u = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    return __half2float(__ushort_as_half(u));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, WARP);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, WARP));
    return v;
}

// Block-wide sum for blockDim.x <= 1024 (multiple of 32).
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + WARP - 1) / WARP;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ float block_max(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + WARP - 1) / WARP;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : -INFINITY;
    if (wid == 0) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

// ---------------------------------------------------------------------------
// Quant formats. g = index of an 8-value group within the row.
// ---------------------------------------------------------------------------

enum QType {
    QT_F32 = 0, QT_F16 = 1, QT_Q8_0 = 8, QT_Q3_K = 11, QT_Q4_K = 12, QT_Q5_K = 13,
    QT_Q6_K = 14, QT_IQ4_NL = 20, QT_IQ3_S = 21, QT_IQ4_XS = 23, QT_MXFP4 = 39,
};

__device__ __constant__ static const int8_t k_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t *q, int &d, int &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

template <int T> struct Q;

template <> struct Q<QT_F32> {
    static constexpr int blk = 8, bytes = 32;
    static __device__ __forceinline__ void dec8(const uint8_t *b, int, float *o) {
        const float *f = (const float *)b;
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = f[i];
    }
};

template <> struct Q<QT_F16> {
    static constexpr int blk = 8, bytes = 16;
    static __device__ __forceinline__ void dec8(const uint8_t *b, int, float *o) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = h2f(b + 2 * i);
    }
};

template <> struct Q<QT_Q8_0> {
    static constexpr int blk = 32, bytes = 34;
    static __device__ __forceinline__ void dec8(const uint8_t *b, int p, float *o) {
        const float d = h2f(b);
        const int8_t *q = (const int8_t *)(b + 2) + p;
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = d * q[i];
    }
};

template <> struct Q<QT_IQ4_NL> {
    static constexpr int blk = 32, bytes = 18;
    static __device__ __forceinline__ void dec8(const uint8_t *b, int p, float *o) {
        const float d = h2f(b);
        const uint8_t *qs = b + 2;
        if (p < 16) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = d * k_iq4nl[qs[p + i] & 0xF];
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = d * k_iq4nl[qs[p - 16 + i] >> 4];
        }
    }
};

// block_q4_K: d, dmin (f16), scales[12], qs[128]
template <> struct Q<QT_Q4_K> {
    static constexpr int blk = 256, bytes = 144;
    static __device__ __forceinline__ void dec8(const uint8_t *b, int p, float *o) {
        const float d = h2f(b), dmin = h2f(b + 2);
        const uint8_t *sc = b + 4, *qs = b + 16;
        const int j64 = p >> 6, in = p & 63, hi = in >> 5, l = in & 31;
        int s, m;
        scale_min_k4(2 * j64 + hi, sc, s, m);
        const float d1 = d * s, m1 = dmin * m;
        const uint8_t *q = qs + 32 * j64 + l;
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = d1 * (hi ? (q[i] >> 4) : (q[i] & 0xF)) - m1;
    }
};

// block_q5_K: d, dmin, scales[12], qh[32], qs[128]
template <> struct Q<QT_Q5_K> {
    static constexpr int blk = 256, bytes = 176;
    static __device__ __forceinline__ void dec8(const uint8_t *b, int p, float *o) {
        const float d = h2f(b), dmin = h2f(b + 2);
        const uint8_t *sc = b + 4, *qh = b + 16, *qs = b + 48;
        const int j64 = p >> 6, in = p & 63, hi = in >> 5, l = in & 31;
        int s, m;
        scale_min_k4(2 * j64 + hi, sc, s, m);
        const float d1 = d * s, m1 = dmin * m;
        const uint8_t *q = qs + 32 * j64 + l;
        const uint8_t hb = (uint8_t)(1u << (2 * j64 + hi));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int v = (hi ? (q[i] >> 4) : (q[i] & 0xF)) + ((qh[l + i] & hb) ? 16 : 0);
            o[i] = d1 * v - m1;
        }
    }
};

// block_q6_K: ql[128], qh[64], scales[16] (int8), d
template <> struct Q<QT_Q6_K> {
    static constexpr int blk = 256, bytes = 210;
    static __device__ __forceinline__ void dec8(const uint8_t *b, int p, float *o) {
        const uint8_t *ql = b, *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 192);
        const float d = h2f(b + 208);
        const int n = p >> 7, r = p & 127, quarter = r >> 5, l = r & 31;
        ql += 64 * n;
        qh += 32 * n;
        sc += 8 * n;
        const float ds = d * sc[(l >> 4) + 2 * quarter];
        const uint8_t *qlp = ql + l + ((quarter & 1) ? 32 : 0);
        const int lshift = (quarter >> 1) ? 4 : 0;
        const int hshift = 2 * quarter;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int v = (((qlp[i] >> lshift) & 0xF) | (((qh[l + i] >> hshift) & 3) << 4)) - 32;
            o[i] = ds * v;
        }
    }
};

// block_q3_K: hmask[32], qs[64], scales[12], d
template <> struct Q<QT_Q3_K> {
    static constexpr int blk = 256, bytes = 110;
    static __device__ __forceinline__ void dec8(const uint8_t *b, int p, float *o) {
        const uint8_t *hm = b, *qs = b + 32, *sb = b + 96;
        const float d = h2f(b + 108);
        const int n = p >> 7, r = p & 127, j = r >> 5, s = r & 31;
        const int k = 8 * n + 2 * j + (s >> 4);
        const int lo = (k < 8) ? (sb[k] & 0xF) : (sb[k - 8] >> 4);
        const int hi = (sb[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
        const float dl = d * ((lo | (hi << 4)) - 32);
        const uint8_t m = (uint8_t)(1u << (4 * n + j));
        const uint8_t *q = qs + 32 * n + s;
        const int shift = 2 * j;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int v = ((q[i] >> shift) & 3) - ((hm[s + i] & m) ? 0 : 4);
            o[i] = dl * v;
        }
    }
};

// block_iq4_xs: d, scales_h (u16), scales_l[4], qs[128]
template <> struct Q<QT_IQ4_XS> {
    static constexpr int blk = 256, bytes = 136;
    static __device__ __forceinline__ void dec8(const uint8_t *b, int p, float *o) {
        const float d = h2f(b);
        const uint16_t sh = (uint16_t)b[2] | ((uint16_t)b[3] << 8);
        const uint8_t *sl = b + 4, *qs = b + 8;
        const int ib = p >> 5, s = p & 31;
        const int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * (ls - 32);
        const uint8_t *q = qs + 16 * ib;
        if (s < 16) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = dl * k_iq4nl[q[s + i] & 0xF];
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = dl * k_iq4nl[q[s - 16 + i] >> 4];
        }
    }
};

// MXFP4 (ggml type 39) in WHIRL's load-time layout (the loader's MXFP4 repack): per
// 256 values e[8] (E8M0 block exponents of the 8 blocks of 32) then qs[8][16]
// (block ib: low nibbles -> values 32ib + 0..15, high -> 32ib + 16..31), 136
// bytes like IQ4_XS. Value = kvalues_mxfp4[code] * 2^(e - 128), i.e. the e2m1
// value times 2^(e - 127) (llama.cpp GGML_E8M0_TO_FP32_HALF).
__device__ __constant__ static const int8_t k_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
__device__ __forceinline__ float e8m0_half(uint32_t e) {
    return __uint_as_float(e < 2 ? (0x00200000u << e) : ((e - 1) << 23));
}
template <> struct Q<QT_MXFP4> {
    static constexpr int blk = 256, bytes = 136;
    static __device__ __forceinline__ void dec8(const uint8_t *b, int p, float *o) {
        const int ib = p >> 5, s = p & 31;
        const float dl = e8m0_half(b[ib]);
        const uint8_t *q = b + 8 + 16 * ib;
        if (s < 16) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = dl * k_mxfp4[q[s + i] & 0xF];
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = dl * k_mxfp4[q[s - 16 + i] >> 4];
        }
    }
};

// block_iq3_s: d, qs[64], qh[8], signs[32], scales[4]
template <> struct Q<QT_IQ3_S> {
    static constexpr int blk = 256, bytes = 110;
    static __device__ __forceinline__ void dec8(const uint8_t *b, int p, float *o) {
        const float d = h2f(b);
        const uint8_t *qs = b + 2, *qh = b + 66, *signs = b + 74, *scales = b + 106;
        const int ib32 = p >> 5, l = (p & 31) >> 3;
        const float db = d * (1 + 2 * ((scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xF));
        const uint8_t h = qh[ib32];
        const int i1 = qs[8 * ib32 + 2 * l + 0] | ((h << (8 - 2 * l)) & 256);
        const int i2 = qs[8 * ib32 + 2 * l + 1] | ((h << (7 - 2 * l)) & 256);
        const uint32_t g1 = k_iq3s_grid[i1], g2 = k_iq3s_grid[i2];
        const uint8_t sg = signs[4 * ib32 + l];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o[i] = db * (float)((g1 >> (8 * i)) & 0xFF) * ((sg >> i) & 1 ? -1.f : 1.f);
            o[i + 4] = db * (float)((g2 >> (8 * i)) & 0xFF) * ((sg >> (i + 4)) & 1 ? -1.f : 1.f);
        }
    }
};

template <int T>
__device__ __forceinline__ void dec_group(const uint8_t *row, int g, float *o) {
    constexpr int gpb = Q<T>::blk / 8;
    Q<T>::dec8(row + (size_t)(g / gpb) * Q<T>::bytes, (g % gpb) * 8, o);
}

