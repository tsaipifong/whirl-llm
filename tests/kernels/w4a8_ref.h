// v0252-f16 (F16 P1a): CPU reference of the gfx1151 W4A8 prefill GEMM (kernels/gfx1151/gemm_w4a8.hip). Header-only, no
// HIP: used by the CPU test (tests/kernels/unit/w4a8_ref_tests.cpp) and by the P1a GPU harness
// (scratch/v0252_f16/w4a8_bench.cpp: the GPU kernel must equal gemm() bit for bit, f32 output included).
// SPDX-License-Identifier: Apache-2.0
//
// The weight side is decoded with the plain kvalues table (w4a8_kv), not with the kernel's v_perm decode (w4a8_lut4),
// so the GPU check compares two implementations of the mapping. The math is spelled out in kernels/gfx1151/w4a8_math.h.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../../kernels/gfx1151/w4a8_math.h"

namespace w4a8ref {

// MXFP4 code and block exponent of element k of a row in WHIRL's load-time layout (per 256 values: e[8], qs[8][16];
// element j of block ib = low nibble of qs[ib][j] for j < 16, high nibble of qs[ib][j - 16] otherwise)
inline void codeExp(const std::uint8_t *row, int k, int &code, int &e) {
    const std::uint8_t *sb = row + static_cast<std::size_t>(k / 256) * 136;
    const int kk = k % 256, ib = kk / 32, j = kk % 32;
    e = sb[ib];
    const std::uint8_t b = sb[8 + 16 * ib + (j & 15)];
    code = j < 16 ? (b & 0xF) : (b >> 4);
}

// the exact weight value kvalues[code] * 2^(e - 128)
inline double weightValue(const std::uint8_t *row, int k) {
    int code, e;
    codeExp(row, k, code, e);
    return std::ldexp(static_cast<double>(w4a8_kv(static_cast<std::uint32_t>(code))), e - 128);
}

inline float bitsF(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// x [n_tok][k] f32 (the f16 activations converted exactly) -> the tiled int8 layout (padding tokens zero)
inline void quantizeTiled(const float *x, int n_tok, int k, std::uint8_t *xq) {
    std::memset(xq, 0, w4a8_xq_bytes(n_tok, k));
    for (int t = 0; t < n_tok; ++t)
        for (int kb = 0; kb < k / 32; ++kb) {
            float d;
            std::int8_t q[32];
            w4a8_quant_block(x + static_cast<std::size_t>(t) * k + 32 * kb, &d, q);
            for (int j = 0; j < 32; ++j) xq[w4a8_xq_qoff(t, 32 * kb + j, k)] = static_cast<std::uint8_t>(q[j]);
            std::memcpy(xq + w4a8_xq_doff(t, kb, k), &d, 4);
        }
}

inline float scaleX(const std::uint8_t *xq, int t, int kb, int k) {
    float d;
    std::memcpy(&d, xq + w4a8_xq_doff(t, kb, k), 4);
    return d;
}

// exact int32 dot of block kb: row r of W x token t of xq
inline std::int32_t blockDot(const std::uint8_t *W, std::size_t row_bytes, int r, const std::uint8_t *xq, int t, int kb, int k) {
    const std::uint8_t *row = W + static_cast<std::size_t>(r) * row_bytes;
    std::int32_t s = 0;
    for (int j = 0; j < 32; ++j) {
        int code, e;
        codeExp(row, 32 * kb + j, code, e);
        s += w4a8_kv(static_cast<std::uint32_t>(code)) * static_cast<std::int8_t>(xq[w4a8_xq_qoff(t, 32 * kb + j, k)]);
    }
    return s;
}

// one output: the pinned epilogue order
inline float dotRow(const std::uint8_t *W, std::size_t row_bytes, const std::uint8_t *xq, int k, int r, int t) {
    const std::uint8_t *row = W + static_cast<std::size_t>(r) * row_bytes;
    float acc = 0.f;
    for (int kb = 0; kb < k / 32; ++kb) {
        const std::int32_t c = blockDot(W, row_bytes, r, xq, t, kb, k);
        const float dw = bitsF(w4a8_e8m0_bits(row[static_cast<std::size_t>(kb / 8) * 136 + kb % 8]));
        const float s = dw * scaleX(xq, t, kb, k);
        acc = std::fma(static_cast<float>(c), s, acc);
    }
    return acc;
}

// Y[t * nrows + r] = sum over blocks (or Y + that when accumulate != 0)
inline void gemm(const std::uint8_t *W, std::size_t row_bytes, const std::uint8_t *xq, float *Y, int k, int nrows, int n_tok,
                 int accumulate) {
    for (int t = 0; t < n_tok; ++t)
        for (int r = 0; r < nrows; ++r) {
            const float acc = dotRow(W, row_bytes, xq, k, r, t);
            float &y = Y[static_cast<std::size_t>(t) * nrows + r];
            y = accumulate ? y + acc : acc;
        }
}

// the f16-activation path in double (exact weights x the f16 activations given as f32): the diagnostic reference
inline void gemmF64(const std::uint8_t *W, std::size_t row_bytes, const float *x, double *Y, int k, int nrows, int n_tok) {
    for (int t = 0; t < n_tok; ++t)
        for (int r = 0; r < nrows; ++r) {
            double s = 0;
            for (int c = 0; c < k; ++c)
                s += weightValue(W + static_cast<std::size_t>(r) * row_bytes, c) * static_cast<double>(x[static_cast<std::size_t>(t) * k + c]);
            Y[static_cast<std::size_t>(t) * nrows + r] = s;
        }
}

}  // namespace w4a8ref
