// moeskip (fast mode): drop routed experts whose renormalized router weight is below a threshold.
// SPDX-License-Identifier: Apache-2.0
//
// The host reference of what kernels/moe_skip.hip (moe_topk_skip) and the id -1 handling in the
// decode experts, moe_route / moe_tiles and moe_combine do. Pure functions for the CPU unit tests.
//
// Rule (feat027, progress\feat027-moeskip.PRIOR_ART.md): the K selected slots come sorted by router
// logit (descending), w[k] = softmax over the K selected logits (the weights the model uses,
// norm_topk_prob). Slots k >= min_keep with w[k] < thr are dropped: id -1, weight 0. Because w is
// sorted, the kept slots are always a prefix. The kept weights are renormalized to sum 1
// (computed as exp(v_k - v_0) / sum over the kept slots, one division, as the kernel).
// thr <= 0: nothing is dropped and the weights are those of moe_topk (precise / balance never call it).

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace whirl::moeskip {

// Mode-plan defaults (fast). thr is an estimate until the GPU sweep (feat027 runner) measures the
// skip fraction and task accuracy; WHIRL_MOESKIP_THR / WHIRL_MOESKIP_MIN override it.
inline constexpr float k_default_thr = 0.05f;
inline constexpr int k_default_min_keep = 1;

struct Params {
    float thr = 0.f;   // <= 0: off
    int min_keep = 1;  // always keep the top min_keep slots (>= 1)
    bool on() const { return thr > 0.f; }
};

// One token: v[0..K) = the selected logits in selection order (descending). Writes w[0..K) and
// keep (number of kept slots, a prefix); ids[k] for k >= keep must be set to -1 by the caller.
// Returns keep.
inline int selectRow(const float* v, int K, const Params& p, float* w) {
    const double m = v[0];
    std::vector<double> e(static_cast<std::size_t>(K));
    double s = 0;
    for (int k = 0; k < K; ++k) s += (e[static_cast<std::size_t>(k)] = std::exp(static_cast<double>(v[k]) - m));
    int keep = K;
    if (p.on()) {
        const int mk = p.min_keep < 1 ? 1 : (p.min_keep > K ? K : p.min_keep);
        for (int k = mk; k < K; ++k)
            if (e[static_cast<std::size_t>(k)] / s < static_cast<double>(p.thr)) {
                keep = k;
                break;
            }
    }
    double s2 = 0;
    for (int k = 0; k < keep; ++k) s2 += e[static_cast<std::size_t>(k)];
    for (int k = 0; k < K; ++k) w[k] = k < keep ? static_cast<float>(e[static_cast<std::size_t>(k)] / s2) : 0.f;
    return keep;
}

// Number of kept slots of token t (ids of a token: a kept prefix, then -1).
inline int keptPrefix(const int* ids_t, int K) {
    int kt = K;
    while (kt > 1 && ids_t[kt - 1] < 0) --kt;
    return kt;
}

// Prefill grouping with skipped pairs (moe_route semantics): pairs with id >= 0 grouped by expert in
// [0, n_kept), each expert's pairs contiguous and in ascending expert order; pairs with id < 0 parked at
// [n_kept, n_pairs) with inv = -1. Tiles cover kept positions only. Order inside an expert is ascending
// pair index here (the kernel's atomics may order them differently; outputs do not depend on it).
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
        for (int s = 0; s < cnt[static_cast<std::size_t>(e)]; s += bn)
            r.tiles.push_back({e, o + s, cnt[static_cast<std::size_t>(e)] - s < bn ? cnt[static_cast<std::size_t>(e)] - s : bn});
        o += cnt[static_cast<std::size_t>(e)];
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

}  // namespace whirl::moeskip
