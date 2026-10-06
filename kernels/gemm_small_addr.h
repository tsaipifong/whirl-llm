// WHIRL: W-stage addressing of the small-batch prefill GEMMs (gemms_impl / gemmsd_impl in
// gemm_small.hip), shared by the device code and the host unit tests.
// SPDX-License-Identifier: Apache-2.0
//
// A stage is the SB raw bytes of one row covering 256 k (SB = 256 / blk * bytes). It is
// copied to LDS as 16-byte chunks starting at the 16-byte aligned address at or below the
// stage start st, so that every super-block keeps its global alignment mod 16. CH chunks
// cover the worst-case misalignment (st % 16 == 15). A chunk is loaded only when it
// overlaps the stage: a chunk that starts at or after st + SB holds no byte the decoder
// reads, and for the last row's last stage it lies wholly past the end of the matrix (16 B
// for Q4_K/Q5_K/IQ4_XS, up to 30 B for Q6_K), which faulted (hipError 719) whenever the
// allocation ended at an unmapped page (X16-4, FIX-OVR). Skipped chunks are zero-filled;
// the decoder never reads them, so results are bit-identical.
//
// The last needed chunk may still extend up to 15 bytes past st + SB, but it is a 16-byte
// aligned block that contains a byte of the matrix, so it never crosses a page boundary
// (and the allocation helper adds kTailPad spare bytes anyway).
#pragma once

#include <stdint.h>

#if defined(__HIP__) || defined(__HIPCC__)
#define WHIRL_GS_HD __host__ __device__
#else
#define WHIRL_GS_HD
#endif

// 16-byte chunks per row stage (covers the misaligned start)
WHIRL_GS_HD constexpr int gemms_stage_chunks(int sb) { return (sb + 15) / 16 + 1; }
// byte offset (from W) of chunk c of the stage starting at byte st
WHIRL_GS_HD constexpr uint64_t gemms_chunk_addr(uint64_t st, int c) { return (st & ~(uint64_t)15) + (uint64_t)c * 16; }
// whether chunk c of the stage starting at byte st overlaps the stage [st, st + sb), i.e.
// gemms_chunk_addr(st, c) < st + sb, written on the low 4 bits of st (32-bit in the kernels)
template <class U>
WHIRL_GS_HD constexpr bool gemms_chunk_needed(U st, int c, int sb) { return (int)(st & 15) + sb > c * 16; }
