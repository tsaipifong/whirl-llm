// whirl-kernel-test: C++ CPU reference implementations.
// SPDX-License-Identifier: Apache-2.0
//
// Written from the GGUF / ggml block-format definitions and the model math
// (RMSNorm, NEOX RoPE, softmax attention, Gated DeltaNet recurrence, MoE
// routing), not from the kernels. Where a kernel is bit-exact by design the
// reference reproduces the exact rounding steps (documented per function);
// otherwise the references accumulate in double.

#pragma once

#include <cstdint>
#include <vector>

#include "kt.h"

namespace kt::ref {

// Exact f32 values of one row (every supported type decodes exactly in f32).
void dequantRow(QType t, const std::uint8_t* row, int ncols, float* out);
std::vector<float> dequantRows(const HostMat& m, int r0, int nrows);

// The f16 weights dequant_f16_<T> produces: F16 copies; Q4_K / Q5_K / IQ4_XS
// / MXFP4 round the block scale (and Q4_K/Q5_K the -dmin*m offset) to f16
// first and apply them with one f16 fused multiply-add (q * s + a rounded
// once); the other types round the exact f32 value to f16 (Q8_0 / Q3_K / Q6_K
// turn a -0 product into +0, see the implementation).
void dequantRowF16(QType t, const std::uint8_t* row, int ncols, std::uint16_t* out);

// quantize_q8: per 32 values d = amax / 127 (f32), q = rint(x * (1 / d)).
void quantizeQ8(const float* x, int n, std::int8_t* xq, float* xd);

// y[t][r] = sum_c W[r][c] * x[t][c] in double (W exact), plus scale[t][r] =
// sum |W x| for the tolerance bound.
void gemvF64(const std::vector<float>& W, int nrows, int ncols, const float* x, int ntok, int x_stride,
             std::vector<double>& y, std::vector<double>& scale);
// Same with int8 activations: x[t][c] = xq * xd[block] (exactly what gemvq_* consume).
void gemvQ8F64(const std::vector<float>& W, int nrows, int ncols, const std::int8_t* xq, const float* xd, int ntok,
               std::vector<double>& y, std::vector<double>& scale);

// OCP e4m3 (fn) with round to nearest even; |v| <= 448 (callers clamp).
std::uint8_t e4m3(float v);
float e4m3ToF(std::uint8_t b);
// qact_fp8 of one row: sx = amax / 448, q = e4m3(clamp(x * (448 / amax))).
void qactFp8Row(const float* x, int n, std::uint8_t* q, float& sx);
// Byte offset of fp8 activation (t, k) in the fragment-tiled layout (K % 16 == 0).
inline std::size_t x8tOff(int t, int k, int K) {
    return (static_cast<std::size_t>(t >> 4) * static_cast<std::size_t>(K >> 4) + static_cast<std::size_t>(k >> 4)) * 256 +
           static_cast<std::size_t>((((((k >> 3) & 1) << 4) | (t & 15)) * 8)) + static_cast<std::size_t>(k & 7);
}

}  // namespace kt::ref
