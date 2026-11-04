// feat026-moep: index maps of the multi-fragment MoE prefill expert GEMMs (kernels/gfx1151/moe_prefill_mt.hip),
// shared by the device code and the CPU schedule test (tests/kernels/unit/moep_sched_tests.cpp), so the test checks
// the very arithmetic the kernels use: 1-D grid -> (tile, row block), wave -> (row fragment, token fragments),
// fragment liveness, and the last-arriver combine counters.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(__HIP__) || defined(__HIPCC__)
#define MOEP_HD __host__ __device__ __forceinline__
#else
#define MOEP_HD inline
#endif

namespace moep {

// blocks along one tile's rows (row block fastest in the 1-D grid)
MOEP_HD int nrb(int rows, int bm) { return (rows + bm - 1) / bm; }
// 1-D grid block index -> (tile index, row block)
MOEP_HD int blockTile(int bx, int nrb_) { return bx / nrb_; }
MOEP_HD int blockRb(int bx, int nrb_) { return bx % nrb_; }
// upper bound of moe_tiles' list length for token tile bn (the grid is sized with it; extra blocks exit)
MOEP_HD int maxTiles(int pairs, int bn, int R) { return (pairs + bn - 1) / bn + R; }
// 16-token fragments that hold live tokens of a tile of n_tok tokens
MOEP_HD int liveFrags(int n_tok) { return (n_tok + 15) / 16; }
// wave column wn of WN takes fragments wn, wn + WN, wn + 2 WN, ... (interleaved: a small tile keeps every wave
// column busy before any column takes a second fragment)
MOEP_HD int frag(int wn, int WN, int i) { return wn + WN * i; }
// counter slot of (token, row block) for the fused combine
MOEP_HD int cntSlot(int tok, int rb, int nrb_) { return tok * nrb_ + rb; }

}  // namespace moep
