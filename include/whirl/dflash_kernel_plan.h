// DFlash2 drafter kernel plan (feat029-dflashcost, T4): which attention / top-k kernels the drafter
// launches, their split sizes and workspace, plus host models of the split kernels for CPU tests.
// SPDX-License-Identifier: Apache-2.0
//
// Attention: dflash_attn_part (grid nkv x block x splits, one block per kv head, token and key chunk of
// kAttnChunk keys; the kv head's G = nh / nkv q heads share every K / V read) writes per (split, token,
// q head) the chunk max m, the chunk sum l and the unnormalized output o; dflash_attn_merge combines the
// splits with the running max (out = sum_s o_s e^(m_s - M) / sum_s l_s e^(m_s - M)). f32 throughout; only
// the summation order differs from dflash_attn (drafter tolerance, kernel test 1e-4 vs CPU).
// Top-k: dflash_topk_part (grid rows x slices of kTopkSlice values) keeps each slice's top k in key order,
// dflash_topk_merge takes the top k of the slices' lists. Under the strict key order (value desc, index asc,
// NaN = -inf) the top k of the union of per-slice top-k lists is the global top k, so ids / values are
// bit-identical to dflash_topk.
// WHIRL_DFLASH_KERNELS=old keeps the v0.2.x kernels (A/B in the GPU runner); the drafter only proposes
// tokens and the target verifies them exactly, so the output does not depend on this choice.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace whirl::dflash {

inline constexpr std::uint32_t kAttnChunk = 256;   // keys per dflash_attn_part block (= DFA_CH in kernels/dflash.hip)
inline constexpr std::uint32_t kAttnMaxGroup = 8;  // q heads per kv head the split kernel holds (DFA_MAXG)
inline constexpr std::uint32_t kAttnThreads = 256; // block of dflash_attn_part (8 waves of 32)
inline constexpr std::uint32_t kTopkSlice = 8192;  // values per dflash_topk_part block (= DFT_SLICE)
inline constexpr std::uint32_t kTopkMaxK = 64;     // as dflash_topk

enum class AttnKernel : std::uint8_t { split, single, chunked };
enum class TopkKernel : std::uint8_t { split, single };

inline const char* attnName(AttnKernel a) { return a == AttnKernel::split ? "split" : a == AttnKernel::single ? "single" : "chunked"; }
inline const char* topkName(TopkKernel t) { return t == TopkKernel::split ? "split" : "single"; }

// WHIRL_DFLASH_KERNELS: "old" -> the v0.2.x kernels; unset / anything else -> the split kernels when present.
inline bool forceOldKernels(const char* env) { return env != nullptr && std::string_view(env) == "old"; }

// The split attention needs: its two kernels, nh a multiple of nkv with G <= kAttnMaxGroup, and a head
// dimension dividing the 256-thread block (64 / 128 / 256; the V pass gives each thread one dimension of
// one key lane). Otherwise the old kernels: dflash_attn while the keys fit kDFlashAttnKeys, else the chunked one.
inline AttnKernel chooseAttn(bool have_split, bool force_old, std::uint32_t nh, std::uint32_t nkv, std::uint32_t hd, std::uint32_t keys,
                             std::uint32_t max_single_keys) {
    const bool shape_ok = nkv > 0 && nh % nkv == 0 && nh / nkv >= 1 && nh / nkv <= kAttnMaxGroup && hd >= 32 && hd <= kAttnThreads &&
                          kAttnThreads % hd == 0;
    if (have_split && !force_old && shape_ok) return AttnKernel::split;
    return keys > max_single_keys ? AttnKernel::chunked : AttnKernel::single;
}

inline TopkKernel chooseTopk(bool have_split, bool force_old, std::uint32_t k) {
    return have_split && !force_old && k >= 1 && k <= kTopkMaxK && k <= kTopkSlice ? TopkKernel::split : TopkKernel::single;
}

inline std::uint32_t attnSplits(std::uint32_t keys) { return (keys + kAttnChunk - 1) / kAttnChunk; }
inline std::uint32_t topkSlices(std::uint32_t n) { return std::max<std::uint32_t>(1, (n + kTopkSlice - 1) / kTopkSlice); }

// Workspace (floats) of dflash_attn_part: [split][token][q head][2 + hd] (m, l, o[hd]).
inline std::uint64_t attnPartFloats(std::uint32_t keys, std::uint32_t block, std::uint32_t nh, std::uint32_t hd) {
    return static_cast<std::uint64_t>(attnSplits(keys)) * block * nh * (2ull + hd);
}
// Workspace (entries; one int id and one float value each) of dflash_topk_part: [row][slice][k].
inline std::uint64_t topkPartEntries(std::uint32_t rows, std::uint32_t n, std::uint32_t k) {
    return static_cast<std::uint64_t>(rows) * topkSlices(n) * k;
}

// ---- host models of the split kernels (same algorithm, same order of the merge; CPU tests)

// Key order of dflash_topk: larger value first, then smaller index; NaN counts as -inf.
inline bool better(float va, int ia, float vb, int ib) {
    if (va != va) va = -INFINITY;
    if (vb != vb) vb = -INFINITY;
    return va > vb || (va == vb && ia < ib);
}

// k rounds over (v, i) pairs: round r takes the best key below round r-1's; (-inf, INT_MAX) when none is left.
inline void topkRounds(const float* v, const int* idx, std::size_t n, int k, int* out_i, float* out_v) {
    float pv = INFINITY;
    int pi = -1;
    for (int r = 0; r < k; ++r) {
        float best = -INFINITY;
        int bi = std::numeric_limits<int>::max();
        for (std::size_t j = 0; j < n; ++j)
            if (better(pv, pi, v[j], idx[j]) && better(v[j], idx[j], best, bi)) {
                best = v[j];
                bi = idx[j];
            }
        pv = best;
        pi = bi;
        out_i[r] = pi;
        out_v[r] = pv;
    }
}

// dflash_topk on one row (reference) and the two-stage version (dflash_topk_part + merge).
inline void topkSingle(const float* row, int n, int k, int* ids, float* vals) {
    std::vector<int> ix(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) ix[static_cast<std::size_t>(i)] = i;
    topkRounds(row, ix.data(), static_cast<std::size_t>(n), k, ids, vals);
}
inline void topkSplit(const float* row, int n, int k, int* ids, float* vals, std::uint32_t slice = kTopkSlice) {
    const std::uint32_t S = std::max<std::uint32_t>(1, (static_cast<std::uint32_t>(n) + slice - 1) / slice);
    std::vector<int> pi(static_cast<std::size_t>(S) * k);
    std::vector<float> pv(pi.size());
    std::vector<int> ix;
    for (std::uint32_t s = 0; s < S; ++s) {
        const int lo = static_cast<int>(s * slice), hi = std::min(n, static_cast<int>((s + 1) * slice));
        ix.resize(static_cast<std::size_t>(std::max(0, hi - lo)));
        for (int i = lo; i < hi; ++i) ix[static_cast<std::size_t>(i - lo)] = i;
        topkRounds(row + lo, ix.data(), ix.size(), k, pi.data() + static_cast<std::size_t>(s) * k, pv.data() + static_cast<std::size_t>(s) * k);
    }
    topkRounds(pv.data(), pi.data(), pv.size(), k, ids, vals);
}

// Attention inputs of one (token, q head): the visible keys' scores already scaled (s_k), the values v_k
// [hd]. attnDirect = dflash_attn's math (one max, one sum); attnSplitModel = the split kernels' math over
// chunks of `chunk` keys (invisible keys = -inf; an all-invisible chunk has m = -inf, l = 0 and is skipped).
inline void attnDirect(const std::vector<float>& s, const std::vector<float>& v, int hd, float* out) {
    float m = -INFINITY;
    for (float x : s) m = std::max(m, x);
    float l = 0.f;
    std::vector<float> o(static_cast<std::size_t>(hd), 0.f);
    for (std::size_t k = 0; k < s.size(); ++k) {
        if (s[k] == -INFINITY) continue;
        const float e = std::exp(s[k] - m);
        l += e;
        for (int d = 0; d < hd; ++d) o[static_cast<std::size_t>(d)] += e * v[k * static_cast<std::size_t>(hd) + static_cast<std::size_t>(d)];
    }
    for (int d = 0; d < hd; ++d) out[d] = o[static_cast<std::size_t>(d)] / l;
}
inline void attnSplitModel(const std::vector<float>& s, const std::vector<float>& v, int hd, float* out, std::uint32_t chunk = kAttnChunk) {
    const std::size_t nk = s.size();
    struct Part {
        float m, l;
        std::vector<float> o;
    };
    std::vector<Part> parts;
    for (std::size_t c0 = 0; c0 < nk; c0 += chunk) {
        const std::size_t c1 = std::min(nk, c0 + chunk);
        Part p{-INFINITY, 0.f, std::vector<float>(static_cast<std::size_t>(hd), 0.f)};
        for (std::size_t k = c0; k < c1; ++k) p.m = std::max(p.m, s[k]);
        if (p.m != -INFINITY)
            for (std::size_t k = c0; k < c1; ++k) {
                if (s[k] == -INFINITY) continue;
                const float e = std::exp(s[k] - p.m);
                p.l += e;
                for (int d = 0; d < hd; ++d) p.o[static_cast<std::size_t>(d)] += e * v[k * static_cast<std::size_t>(hd) + static_cast<std::size_t>(d)];
            }
        parts.push_back(std::move(p));
    }
    float M = -INFINITY;
    for (const Part& p : parts) M = std::max(M, p.m);
    float l = 0.f;
    std::vector<float> o(static_cast<std::size_t>(hd), 0.f);
    for (const Part& p : parts) {
        if (p.m == -INFINITY) continue;
        const float w = std::exp(p.m - M);
        l += p.l * w;
        for (int d = 0; d < hd; ++d) o[static_cast<std::size_t>(d)] += p.o[static_cast<std::size_t>(d)] * w;
    }
    for (int d = 0; d < hd; ++d) out[d] = o[static_cast<std::size_t>(d)] / l;
}

}  // namespace whirl::dflash
