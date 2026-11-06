// moeskip (fast mode): drop routed experts whose renormalized router weight is below a threshold.
// SPDX-License-Identifier: Apache-2.0
//
// v0252-f5 (ported from feat027-moeskip-r1 4b6474b, include/whirl/moe_skip.h; progress\v0252-f5.PRIOR_ART.md): the host
// reference of the moeskip entry points (kernels/shared/moe_skip.h moe_skip_rule, moe_topk_skip, moe_skip_apply,
// moe_route_skip, moe_combine_skip; kernels/moe_skip.hip, kernels/gfx1151/moe_skip.hip). Pure functions for the CPU unit
// tests (tests/unit/kernels_moe_tests.cpp testMoeSkipRef) and the GPU kernel tests (tests/kernels/test_moe_skip.cpp).
//
// Rule (one token): ids[0..K) / w[0..K) = a router's top-K output in selection order (moe_topk: logits descending, ties ->
// lower index; w[k] = exp(v_k - v_0) / sum_j exp(v_j - v_0), the weights the model uses, norm_topk_prob), so w is
// non-increasing. With mk = clamp(min_keep, 1, K), the kept slots are [0, keep): keep = the first k >= mk with w[k] < thr
// (K when none). Nothing dropped -> ids / w unchanged (bitwise moe_topk). Else the dropped slots get id -1 and weight 0 and
// the kept weights are divided by their sum s (float, summed in k order): w[k] / s. Same float operations in the same order
// as the kernels, so the reference is bitwise the kernels' result where the device divides exactly (IEEE, the HIP default).
// thr <= 0: nothing is ever dropped. thr >= 1: exactly the top mk slots (static top-k, as llama.cpp's expert_used_count = mk
// with the weights renormalized over the kept experts).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace whirl::moeskip {

struct Params {
    float thr = 0.f;   // <= 0: off
    int min_keep = 1;  // the top min_keep slots are always kept (clamped to [1, K])
    bool on() const { return thr > 0.f; }
};

// The rule on one token, in place (see the file comment); returns keep (the kept prefix length).
inline int applyRow(int* ids, float* w, int K, const Params& p) {
    const int mk = p.min_keep < 1 ? 1 : (p.min_keep > K ? K : p.min_keep);
    int keep = K;
    for (int k = mk; k < K; ++k)
        if (w[k] < p.thr) {
            keep = k;
            break;
        }
    if (keep == K) return K;
    float s = 0.f;
    for (int k = 0; k < keep; ++k) s += w[k];
    for (int k = 0; k < K; ++k) {
        if (k < keep) {
            w[k] = w[k] / s;
        } else {
            w[k] = 0.f;
            ids[k] = -1;
        }
    }
    return keep;
}

// Kept slots of a token (ids of a token: a kept prefix, then -1); K when nothing was dropped.
inline int keptPrefix(const int* ids_t, int K) {
    int kt = K;
    while (kt > 1 && ids_t[kt - 1] < 0) --kt;
    return kt;
}

// Prefill grouping with dropped pairs (moe_route_skip): pairs with id >= 0 grouped by expert in [0, n_kept), each
// expert's pairs contiguous and in ascending expert order, tiles of at most bn over the kept positions only; pairs with
// id < 0 parked at [n_kept, n_pairs) with inv = -1 (perm there = a dropped pair, so a gather over every position reads
// a valid token). The order inside an expert and inside the parked tail is ascending pair index here (the kernel's
// atomics may order them differently; no output depends on it). n_kept = n_pairs and no -1: moe_route's grouping.
struct Route {
    std::vector<int> perm, inv;
    struct Tile {
        int expert, first, count;
    };
    std::vector<Tile> tiles;
    int n_kept = 0;
};
inline Route route(const std::vector<int>& ids, int R, int bn) {
    Route r;
    const int np = static_cast<int>(ids.size());
    r.perm.assign(static_cast<std::size_t>(np), -1);
    r.inv.assign(static_cast<std::size_t>(np), -1);
    std::vector<int> cnt(static_cast<std::size_t>(R), 0), off(static_cast<std::size_t>(R), 0), fill(static_cast<std::size_t>(R), 0);
    for (int e : ids)
        if (e >= 0) ++cnt[static_cast<std::size_t>(e)];
    int o = 0;
    for (int e = 0; e < R; ++e) {
        off[static_cast<std::size_t>(e)] = o;
        const int c = cnt[static_cast<std::size_t>(e)];
        for (int s = 0; s < c; s += bn) r.tiles.push_back({e, o + s, c - s < bn ? c - s : bn});
        o += c;
    }
    r.n_kept = o;
    int tail = o;
    for (int p = 0; p < np; ++p) {
        const int e = ids[static_cast<std::size_t>(p)];
        const int pos = e >= 0 ? off[static_cast<std::size_t>(e)] + fill[static_cast<std::size_t>(e)]++ : tail++;
        r.perm[static_cast<std::size_t>(pos)] = p;
        r.inv[static_cast<std::size_t>(p)] = e >= 0 ? pos : -1;
    }
    return r;
}

// moe_combine_skip of one output value: x + sum over k ascending of w[t,k] * yd[inv[t,k]] for inv >= 0 (+ sg * ysh),
// the kernel's f32 order (yd(q) = the expert output at position q of this column)
template <class YdAt>
inline float combine(float x, const int* inv_t, const float* w_t, int K, YdAt yd, bool has_sh, float sg, float ysh) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int q = inv_t[k];
        if (q < 0) continue;
        acc += w_t[k] * yd(q);
    }
    if (has_sh) acc += sg * ysh;
    return x + acc;
}

}  // namespace whirl::moeskip
