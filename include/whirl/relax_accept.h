// SPDX-License-Identifier: Apache-2.0
// Relaxed speculative acceptance (numerics item relaxacc, fast mode only). MTP drafts only:
// n-gram drafts copy the history, and relaxing them fed a repetition loop (FAST-2: a 13-line
// duplicated import in 1 of 24 replies at k4/a0.1); they keep exact acceptance.
//
// Exact acceptance keeps a draft token only when it equals the token the target would pick
// (greedy: the argmax of the verify row; sampling: the sampled token). relaxacc also keeps a
// draft that is "close enough" to the target, so more drafts survive per verify cycle:
//
//   greedy   : the draft is among the top-k tokens of the verify row (k = WHIRL_RELAX_K,
//              default 4) and p(draft) >= alpha * p(argmax) (alpha = WHIRL_RELAX_ALPHA,
//              default 0.1): exp(l_draft - l_max) >= alpha on the raw logits.
//   sampling : typical acceptance (Medusa, Cai et al. 2024): the draft is kept when
//              p(draft) >= min(eps, delta * exp(-H(p))) on the request's filtered sampling
//              distribution (eps = WHIRL_RELAX_EPS, default 0.09; delta = WHIRL_RELAX_DELTA,
//              default 0.3). A rejected position draws the token from the row as usual.
//
// The output is NOT the target model's greedy / sampled output (that is the point: fast mode);
// it is deterministic for the same prompt, seed and settings: the kept tokens depend on which
// drafts were proposed, so with relaxacc the draft-count policy sees a fixed synthetic cycle cost
// instead of wall-clock times and a solo request starts from a reset policy state. Concurrent
// requests share the verify batch's draft count, so their outputs can depend on each other.
// Kept drafts also depend on the last bits of the MTP / trunk rows, so a prompt prefilled in
// other chunks (resumed from another prompt's shared prefix) can give another text; a repeat
// resumed from its own checkpoints is bit-identical to the cold run (FIX-FS: the MTP boundary
// row of a chunk runs alone in both, see Model::prefillMtpChunk).
// The bonus token after the last
// accepted draft is the target's own pick, so every emitted non-draft token is exact.
// WHIRL_RELAX=0 turns the item off in fast mode; WHIRL_RELAX_K=1 is exact greedy acceptance.
//
// Defaults (FAST-2, Radeon 8060S, Qwen3.8-27B Q4_K_M, 24 zh/en coding + prose / agent prompts,
// greedy, vs fast with exact acceptance): tok/cycle k2/a0.5 +0.3%, k3/a0.3 +3.6%, k4/a0.2 +7.4%,
// k4/a0.1 +10.4%; execution-graded tasks 12/12 for every setting, tool call unchanged, no rise in
// repetition. A kept relaxed draft is on average e^1.0 = 2.7x less likely than the argmax at
// k4/a0.1. Decode tok/s (3 prompts, short / 32k): Q4_K_M +23% / +21%, Swift +14% / +22%.
#pragma once

#include <cmath>
#include <cstdint>
#include <span>

namespace whirl::relax {

struct Params {
    bool on = false;
    std::uint32_t k = 4;     // greedy: draft rank limit (1 = exact)
    float alpha = 0.1f;      // greedy: p(draft) / p(max) floor
    float eps = 0.09f;       // sampling: hard probability threshold
    float delta = 0.3f;      // sampling: entropy-scaled threshold
    std::uint32_t topk() const { return k < 1 ? 1 : (k > 16 ? 16 : k); }
};

// Greedy rule on a row's top-K candidates (ids / logits, any order, K >= p.topk()); m = the
// row's max logit. The argmax itself always passes (alpha <= 1).
inline bool acceptGreedy(const Params& p, std::span<const std::int32_t> ids, std::span<const float> logits, float m,
                         std::uint32_t draft) {
    const std::size_t n = ids.size() < logits.size() ? ids.size() : logits.size();
    std::size_t at = n;
    for (std::size_t i = 0; i < n; ++i)
        if (static_cast<std::uint32_t>(ids[i]) == draft) {
            at = i;
            break;
        }
    if (at == n) return false;
    const float ld = logits[at];
    std::uint32_t rank = 0;  // candidates strictly ahead (ties: lower id first, as the argmax)
    for (std::size_t i = 0; i < n; ++i) {
        if (i == at) continue;
        if (logits[i] > ld || (logits[i] == ld && static_cast<std::uint32_t>(ids[i]) < draft)) ++rank;
    }
    if (rank >= p.topk()) return false;
    return std::exp(static_cast<double>(ld) - static_cast<double>(m)) >= static_cast<double>(p.alpha);
}

// Repetition guard: true when hist followed by tail ends in an n-gram (n = 6) that already occurs
// earlier in the last `window` tokens. A relaxed (non-exact) acceptance never extends a repeat:
// at k4/a0.1 relaxed drafts started duplicated-line loops that greedy then continued.
inline bool extendsRepeat(std::span<const std::uint32_t> hist, std::span<const std::uint32_t> tail, std::size_t n = 6,
                          std::size_t window = 4096) {
    const std::size_t hn = hist.size() < window ? hist.size() : window;
    const std::size_t s = hn + tail.size();
    if (s < n + 1) return false;
    auto at = [&](std::size_t i) { return i < hn ? hist[hist.size() - hn + i] : tail[i - hn]; };
    for (std::size_t i = 0; i + n < s; ++i) {
        std::size_t j = 0;
        while (j < n && at(i + j) == at(s - n + j)) ++j;
        if (j == n) return true;
    }
    return false;
}

// Typical acceptance threshold for a row with entropy h (nats).
inline bool acceptTypical(const Params& p, double p_draft, double h) {
    const double thr = std::fmin(static_cast<double>(p.eps), static_cast<double>(p.delta) * std::exp(-h));
    return p_draft >= thr;
}

}  // namespace whirl::relax
