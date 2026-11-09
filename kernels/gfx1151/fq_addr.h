// RED STUB (v0252-8sbvalu test-first): every function returns 0
// gfx1151 fused-dequant GEMM (gemm_fq.hip) load addressing of the v0252-8sbvalu levers (PLAN V1-53 N2).
// SPDX-License-Identifier: Apache-2.0
//
// Host + device (tests/unit/kernels_gemm_tests.cpp testFqAddr runs these against the shipped kernel's pointer math).
// The shipped loop keeps 64-bit pointers per lane and advances them with VALU every K step (~40 of ~103 VALU per BK-64
// step in gemmfq_t2, research\b8s\2026-11-09-v0252-8sbal-gemmfq-raster.md s2 / progress\v0252-8sbvalu.PRIOR_ART.md s0).
// The buffer-load levers split every load offset into a per-lane part (voffset VGPR, fixed for the whole loop) and a
// uniform part that depends only on k0 (soffset SGPR, SALU) - hipfire qwen4_gemm_mqv2_wmma_gfx11_bt.hip:849-856 (MIT).
// Offsets are bytes relative to the block's base (W + m0 * row_bytes, X + t0 * ncols); every one fits 32 bits for the
// product shapes (W: BM rows x row_bytes <= 256 x 9248 B; X: BN tokens x ncols x 2 <= 512 x 17408 x 2 B).
//
// MXFP4 row layout: 136-byte blocks of 256 values (8 E8M0 bytes, then 8 sub-blocks x 16 qs bytes). In a K chunk at
// k0 (k0 % 32 == 0) the 32-value sub-block sb (0 = the first) has qs at block + 8 + 16 * ib and its E8M0 at block + ib,
// ib = ((k0 & 255) >> 5) + sb. A "part" p (0..3) of a sub-block is 8 values in k order [8p, 8p + 8): qs bytes
// (p & 1) * 8 .. + 8, the low nibbles for p < 2, the high nibbles for p >= 2 (= fq_decode_sb's o[p]).
#pragma once

#include <stdint.h>

#if defined(__HIPCC__)
#define FQA_HD __host__ __device__ __forceinline__
#else
#define FQA_HD inline
#endif

// uniform (SGPR) parts: qs bytes of sub-block 0 of the chunk at k0, and its E8M0 byte
FQA_HD uint32_t fq_q_soff(uint32_t k0) { return 0u; }
FQA_HD uint32_t fq_e_soff(uint32_t k0) { return 0u; }
// per-lane (VGPR) parts: local row lrow (0 = the block's first row), sub-block sb of the chunk, 8-byte half h of its qs
FQA_HD uint32_t fq_q_voff(uint32_t lrow, uint32_t row_bytes, uint32_t sb, uint32_t h) {
    return 0u;
}
FQA_HD uint32_t fq_e_voff(uint32_t lrow, uint32_t row_bytes, uint32_t sb) { return 0u; }
// activations (f16): local token ltok, 8-value column group c8 of the chunk; the uniform part is k0 * 2
FQA_HD uint32_t fq_b_voff(uint32_t ltok, uint32_t ncols, uint32_t c8) { return 0u; }
FQA_HD uint32_t fq_b_soff(uint32_t k0) { return 0u; }
// part p of a sub-block: which 8 qs bytes (0 = bytes 0-7, 1 = bytes 8-15), which nibble (0 low, 1 high), LDS column
FQA_HD uint32_t fq_part_half(uint32_t p) { return 0u; }
FQA_HD uint32_t fq_part_hi(uint32_t p) { return 0u; }
FQA_HD uint32_t fq_part_col(uint32_t p) { return 0u; }
