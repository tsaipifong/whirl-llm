// v0252-f16 (F16 P1a) STUB - test-first state (commit 1): every helper returns 0 so tests/kernels/unit/w4a8_ref_tests.cpp
// fails before the real code. Replaced by the real header in commit 2.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__HIPCC__)
#define W4A8_HD __host__ __device__ __forceinline__
#else
#define W4A8_HD inline
#endif

constexpr int kW4A8TileTok = 16, kW4A8TileBytes = 576, kW4A8ScaleOff = 512;

W4A8_HD int w4a8_kv(uint32_t) { return 0; }
W4A8_HD uint32_t w4a8_perm_ref(uint32_t, uint32_t, uint32_t) { return 0; }
W4A8_HD uint32_t w4a8_lut4(uint32_t) { return 0; }
W4A8_HD uint32_t w4a8_e8m0_bits(uint32_t) { return 0; }
W4A8_HD size_t w4a8_xq_bytes(int, int) { return 0; }
W4A8_HD size_t w4a8_xq_qoff(int, int, int) { return 0; }
W4A8_HD size_t w4a8_xq_doff(int, int, int) { return 0; }
inline void w4a8_quant_block(const float *, float *d, int8_t *q) {
    *d = 0.f;
    for (int i = 0; i < 32; ++i) q[i] = 0;
}
