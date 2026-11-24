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

// v0252-8smoe2 (C): the coalesced f16 epilogue (gemm_moepdhs_*, EPI 5). A wave stages the TM row fragments x one 16-token
// fragment of its f32 WMMA accumulators as f16 in a private LDS slice [16 tokens][TM * 16 rows + 8 pad], then every lane
// writes 8 consecutive rows (16 bytes) of one token. Slice row stride (halves; a multiple of 8: 16-byte aligned reads).
MOEP_HD int dhsLd(int TM) { return TM * 16 + 8; }
// slice offset of accumulator element v of row fragment i held by lane (WMMA f32 layout: token lane % 16, row CROW)
MOEP_HD int dhsWrite(int lane, int i, int v, int ld) { return (lane % 16) * ld + i * 16 + 2 * v + lane / 16; }
// 8-row chunks of one staged fragment (16 tokens x TM * 16 rows / 8)
MOEP_HD int dhsChunks(int TM) { return 16 * TM * 2; }
// chunk c -> (token in the fragment, first row in the wave's TM * 16 rows)
MOEP_HD void dhsRead(int c, int TM, int* tok, int* row8) {
    *tok = c / (2 * TM);
    *row8 = (c % (2 * TM)) * 8;
}

// v0252-8smoe2 (E): the fused-dequant (gemmfq) A fetch of the moep twins gemm_moepfq*: per K chunk of BK values a row has
// BK / 32 MXFP4 sub-blocks; lane slot s (threadIdx.x + e * 256) takes row s / (BK / 32) and its sub-block s % (BK / 32)
// (adjacent lanes: one row's bytes). The four half8 of a sub-block go to LDS columns sb * 32 + j * 8, j = 0..3 (the
// dec_group_h groups g .. g + 3 of that sub-block: low nibbles of bytes 0-7, of bytes 8-15, high of 0-7, high of 8-15).
MOEP_HD int fqRow(int s, int BK) { return s / (BK / 32); }
MOEP_HD int fqSb(int s, int BK) { return s % (BK / 32); }
// MXFP4 sub-block index inside the 256-value super-block of chunk k0 (k0 % BK == 0, BK in {32, 64})
MOEP_HD int fqIb(int k0, int sb) { return ((k0 & 255) >> 5) + sb; }

// v0252-8sdet (PLAN V1-140, COMB 6, fast only): the order-independent fixed-point MoE combine (gemm_moepdi_*, EPI 6).
// Each routed addend a = w * y is rounded to the grid 2^-kFxShift (q = fxQuant(a, lim), an i32) and added into an i32 sum
// per (token, row) by integer atomics: integer addition mod 2^32 is exact, commutative and associative, so the sum does
// not depend on the order the blocks run in. The last arriver of a (token, row block) converts once (fxValue) and adds
// it (+ sg * ysh) into x. Error: |q * 2^-S - a| <= 2^-(S+1) per unclamped addend (S = 14: 3.05e-5, below one f16 rounding
// of any |a| >= 1/16), so |fxValue(sum) - sum a| <= kept * 2^-(S+1) + one f32 rounding of the value. Overflow: each q is
// clamped to +-lim with K * lim <= INT32_MAX, so the true sum of <= K addends fits i32 (partial sums may wrap mod 2^32:
// harmless, only the final sum is read); lim * 2^-S = 16383.999 for K = 8. NaN -> 0, +-inf -> +-lim (deterministic).
constexpr int kFxShift = 14;
MOEP_HD float fxScale() { return static_cast<float>(1 << kFxShift); }
// the largest integral float lim with K * lim <= INT32_MAX (host: passed to the kernel as an argument)
inline float fxLimit(int K) {
    const long long cap = 2147483647LL / (K < 1 ? 1 : K);
    long long m = cap;
    while (static_cast<long long>(static_cast<float>(m)) > cap) --m;
    return static_cast<float>(m);
}
// round a * 2^S to the nearest integer (ties to even), clamped to +-lim; no library call (host and device the same)
MOEP_HD int fxQuant(float a, float lim) {
    float s = a * fxScale();
    s = s > lim ? lim : s;
    s = s < -lim ? -lim : s;
    if (!(s == s)) s = 0.f;  // NaN
    int i = static_cast<int>(s);  // toward zero; |s| <= lim < 2^31
    const float r = s - static_cast<float>(i);  // exact: s and i share the binade (or i = 0)
    if (r > 0.5f || (r == 0.5f && (i & 1))) ++i;
    else if (r < -0.5f || (r == -0.5f && (i & 1))) --i;
    return i;
}
MOEP_HD float fxValue(int sum) { return static_cast<float>(sum) * (1.f / fxScale()); }
// kept (routed, not moeskip-dropped) slots of token tok: inv[tok * K + k] >= 0 (moe_route_skip parks the dropped ones at -1)
MOEP_HD int fxKept(const int* inv, int tok, int K) {
    int c = 0;
    for (int k = 0; k < K; ++k) c += inv[tok * K + k] >= 0 ? 1 : 0;
    return c;
}

}  // namespace moep
