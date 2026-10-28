// WHIRL device code, shared by both kernel sets (gfx1201 kernels/gemm_prefill.hip, gfx1151 kernels/gfx1151/gemm_prefill.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

#define WMMA_ENTRY(SUF, T)                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                               \
        gemm_wmma_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, \
                        int nrows, int n_tok, int accumulate) {                                    \
        gemm_wmma_impl<T, 128, 128, 64>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);       \
    }

WMMA_ENTRY(f32, QT_F32)
WMMA_ENTRY(f16, QT_F16)
WMMA_ENTRY(q8_0, QT_Q8_0)
WMMA_ENTRY(q3_k, QT_Q3_K)
WMMA_ENTRY(q4_k, QT_Q4_K)
WMMA_ENTRY(q5_k, QT_Q5_K)
WMMA_ENTRY(q6_k, QT_Q6_K)
WMMA_ENTRY(iq4_nl, QT_IQ4_NL)
WMMA_ENTRY(iq3_s, QT_IQ3_S)
WMMA_ENTRY(iq4_xs, QT_IQ4_XS)
WMMA_ENTRY(mxfp4, QT_MXFP4)

// ---------------------------------------------------------------------------
// Prefill GEMM v3: register-blocked, software-pipelined WMMA.
// Block tile BM x BN (weights x tokens), 8 waves arranged WAVES_M x WAVES_N,
// each wave owning TM x TN 16x16 tiles (fragment reuse: TM + TN LDS loads per
// TM * TN WMMAs). The next K chunk is fetched/dequantized into registers
// while the current chunk is multiplied from LDS.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Fast dequant to f16 for the prefill GEMM: vector loads + packed f16 math.
// Small unsigned ints u (< 1024) become exact f16 via bits 0x6400 | u
// (= 1024 + u); subtracting 1024 is exact, then one packed FMA applies the
// scale. Types without a fast path fall back to dec_group.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int iq4_lut4(uint32_t nib4);  // defined with the gemvq kernels
__device__ __forceinline__ int mx_lut4(uint32_t nib4);   // (MXFP4 codes, same place)

// bytes b0..b3 (each < 256) of x -> two half2: (b0,b1), (b2,b3) as 1024 + b
__device__ __forceinline__ void bytes_to_h2(uint32_t x, half2_t &lo, half2_t &hi) {
    const uint32_t a = (x & 0xFFu) | ((x & 0xFF00u) << 8) | 0x64006400u;
    const uint32_t b = ((x >> 16) & 0xFFu) | ((x >> 8) & 0xFF0000u) | 0x64006400u;
    lo = __builtin_bit_cast(half2_t, a);
    hi = __builtin_bit_cast(half2_t, b);
}

// 8 packed bytes (u0,u1) -> half8 of (byte - bias) * scale + add
__device__ __forceinline__ half8 bytes8_to_h8(uint32_t u0, uint32_t u1, _Float16 bias, _Float16 scale, _Float16 add) {
    half2_t p[4];
    bytes_to_h2(u0, p[0], p[1]);
    bytes_to_h2(u1, p[2], p[3]);
    const half2_t b2 = {(_Float16)1024.f + bias, (_Float16)1024.f + bias};
    const half2_t s2 = {scale, scale}, a2 = {add, add};
    half8 out;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const half2_t v = (p[i] - b2) * s2 + a2;
        out[2 * i] = v[0];
        out[2 * i + 1] = v[1];
    }
    return out;
}

template <int T>
__device__ __forceinline__ half8 dec_group_h(const uint8_t *row, int g) {
    float w[8];
    dec_group<T>(row, g, w);
    half8 h;
#pragma unroll
    for (int e = 0; e < 8; ++e) h[e] = (_Float16)w[e];
    return h;
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_F16>(const uint8_t *row, int g) {
    return *(const half8 *)(row + (size_t)g * 16);
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_Q4_K>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 5) * 144;
    const int p = (g & 31) * 8;
    const int j64 = p >> 6, in = p & 63, hi = in >> 5, l = in & 31;
    const uint32_t dm = *(const uint32_t *)b;
    const float d = __half2float(__ushort_as_half((uint16_t)dm)), dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
    int sc, m;
    scale_min_k4(2 * j64 + hi, b + 4, sc, m);
    const uint2 q = *(const uint2 *)(b + 16 + 32 * j64 + l);
    const uint32_t q0 = hi ? (q.x >> 4) & 0x0F0F0F0F : q.x & 0x0F0F0F0F;
    const uint32_t q1 = hi ? (q.y >> 4) & 0x0F0F0F0F : q.y & 0x0F0F0F0F;
    return bytes8_to_h8(q0, q1, (_Float16)0.f, (_Float16)(d * sc), (_Float16)(-dmin * m));
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_Q5_K>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 5) * 176;
    const int p = (g & 31) * 8;
    const int j64 = p >> 6, in = p & 63, hi = in >> 5, l = in & 31;
    const uint32_t dm = *(const uint32_t *)b;
    const float d = __half2float(__ushort_as_half((uint16_t)dm)), dmin = __half2float(__ushort_as_half((uint16_t)(dm >> 16)));
    int sc, m;
    scale_min_k4(2 * j64 + hi, b + 4, sc, m);
    const uint2 q = *(const uint2 *)(b + 48 + 32 * j64 + l);
    const uint2 h = *(const uint2 *)(b + 16 + l);
    const int sh = 2 * j64 + hi;
    const uint32_t q0 = (hi ? (q.x >> 4) & 0x0F0F0F0F : q.x & 0x0F0F0F0F) | (((h.x >> sh) & 0x01010101) << 4);
    const uint32_t q1 = (hi ? (q.y >> 4) & 0x0F0F0F0F : q.y & 0x0F0F0F0F) | (((h.y >> sh) & 0x01010101) << 4);
    return bytes8_to_h8(q0, q1, (_Float16)0.f, (_Float16)(d * sc), (_Float16)(-dmin * m));
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_IQ4_XS>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 5) * 136;
    const int p = (g & 31) * 8;
    const int ib = p >> 5, s = p & 31;
    const uint32_t hdr = *(const uint32_t *)b;
    const float d = __half2float(__ushort_as_half((uint16_t)hdr));
    const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) | ((((hdr >> 16) >> (2 * ib)) & 3) << 4);
    const uint2 q = *(const uint2 *)(b + 8 + 16 * ib + (s & 15));
    const uint32_t n0 = s < 16 ? q.x & 0x0F0F0F0F : (q.x >> 4) & 0x0F0F0F0F;
    const uint32_t n1 = s < 16 ? q.y & 0x0F0F0F0F : (q.y >> 4) & 0x0F0F0F0F;
    // signed LUT bytes -> unsigned (v + 128), then subtract 128
    const uint32_t u0 = (uint32_t)iq4_lut4(n0) ^ 0x80808080u, u1 = (uint32_t)iq4_lut4(n1) ^ 0x80808080u;
    return bytes8_to_h8(u0, u1, (_Float16)128.f, (_Float16)(d * (ls - 32)), (_Float16)0.f);
}

template <>
__device__ __forceinline__ half8 dec_group_h<QT_MXFP4>(const uint8_t *row, int g) {
    const uint8_t *b = row + (size_t)(g >> 5) * 136;
    const int p = (g & 31) * 8;
    const int ib = p >> 5, s = p & 31;
    const float dl = e8m0_half(b[ib]);
    const uint2 q = *(const uint2 *)(b + 8 + 16 * ib + (s & 15));
    const uint32_t n0 = s < 16 ? q.x & 0x0F0F0F0F : (q.x >> 4) & 0x0F0F0F0F;
    const uint32_t n1 = s < 16 ? q.y & 0x0F0F0F0F : (q.y >> 4) & 0x0F0F0F0F;
    const uint32_t u0 = (uint32_t)mx_lut4(n0) ^ 0x80808080u, u1 = (uint32_t)mx_lut4(n1) ^ 0x80808080u;
    return bytes8_to_h8(u0, u1, (_Float16)128.f, (_Float16)dl, (_Float16)0.f);
}
