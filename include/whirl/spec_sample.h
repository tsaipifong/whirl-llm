// Speculative sampling of MTP drafts (numerics item "specsample", balance / fast).
// SPDX-License-Identifier: Apache-2.0
//
// With temperature > 0 the server used to take the MTP draft as the draft head's argmax and accept it
// only when the token sampled from the target equals it: lossless and token for token equal to plain
// sampling, but a deterministic draft is accepted with probability p(draft) only.
//
// specsample couples the draft to the target draw (Gumbel-max / exponential-race coupling). For output
// index i every token v gets E(v) = -ln(1 - U(seed, i, v)), U a counter hash; the target token is
// argmin_v E(v) / p(v) over the target's candidates (an exact draw from p), and draft r (output index
// pos0 + r) is argmin_v E(v) / q(v) over q's support with the same E. The draft is accepted iff it
// equals the target token. So:
//  - the emitted token at every index is a function of (target logits, seed, index) only: the same
//    whatever the drafts, the draft count, the prefix cache or the draft head's state, and the same
//    as with MTP off (same seed -> same text within the mode);
//  - the acceptance is P(both races pick the same token), close to the optimum sum min(p, q) (vs
//    p(argmax q) for a greedy draft); an inverse-CDF coupling of the two draws reaches far less
//    when p and q rank tokens differently.
// An open target (no top-k / top-p / min-p cut inside the device candidates) first picks "candidates
// vs tail" with the stream-0 uniform u (candidates when u < their mass S), then races over the
// candidates or walks the tail by inverse CDF: still an exact draw from p.
// (The first specsample used the Leviathan / Chen accept-min(1, p/q)-else-residual rule: same output
// distribution, but the emitted token depended on q and on where the draft boundaries fell - draft
// counts come from timing-trained cost models - so one seed could give different texts: removed.)
// Precise mode (specsample off) keeps the inverse-CDF sampler and greedy drafts, unchanged.
//
// q here: the draft head's softmax at the request temperature over its q_max largest logits, then
// the request's top-k / top-p / min-p (the same rule as the target's candidate filter), renormalized.
// Tokens outside the draft vocabulary subset have q = 0. The device kernel draft_sample_rows
// computes q, draws the draft and writes q (token ids + probabilities, for tests / diagnostics).
// The pure parts live here for the unit tests.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace whirl::spec {

inline constexpr std::uint32_t q_max = 32;  // support of q (draft head's largest logits)
// Device buffer of q per (sequence, draft step): [0] = n, [1, 1 + q_max) = token ids, [1 + q_max,
// 1 + 2 q_max) = probabilities (f32 bits); q_words i32 words per entry.
inline constexpr std::uint32_t q_words = 72;
inline constexpr std::uint32_t q_off_ids = 1;
inline constexpr std::uint32_t q_off_p = 1 + q_max;

// Uniform streams keyed by (seed, output token index): 0 = the plain sampler's draw (inverse CDF;
// with specsample: the candidates-vs-tail choice of an open target). Other streams: reserved.
enum : std::uint32_t { u_sample = 0 };

// Race keys: raceBase(seed, i) per output index, raceE(base, v) = E(v) ~ Exp(1) per token
// (the device kernel computes the same bits: splitmix64 finalizer, double log).
inline std::uint64_t splitmix(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
inline std::uint64_t raceBase(std::uint64_t seed, std::uint64_t pos_idx) {
    return splitmix((seed ^ 0xA0761D6478BD642Full) + (pos_idx + 1) * 0x9E3779B97F4A7C15ull);
}
double raceE(std::uint64_t base, std::uint32_t token);
double uniform(std::uint64_t seed, std::uint64_t pos_idx, std::uint32_t stream);

// Per-sequence draft sampling parameters (set by the server before the MTP draft steps).
struct DraftSample {
    bool on = false;
    std::uint32_t top_k = 0;
    float inv_t = 1;
    float top_p = 1;
    float min_p = 0;
    std::uint64_t seed = 0;
    std::uint64_t pos0 = 0;  // output token index of draft 0
};

struct QDist {
    std::uint32_t n = 0;
    std::uint32_t id[q_max] = {};
    float p[q_max] = {};
    // q(t); 0 outside the support
    float of(std::uint32_t t) const {
        for (std::uint32_t i = 0; i < n; ++i)
            if (id[i] == t) return p[i];
        return 0.f;
    }
};
// Decode one q entry of the device buffer (n clamped to q_max).
QDist qFromWords(const std::int32_t* w);

// CPU reference of the device filter: cand ids / logits (the K largest draft logits, any order,
// K <= q_max), m = max logit, sum = sum over the whole draft row of exp((x - m) * inv_t).
// Sorted by logit (ties: smaller id first), then top-k / top-p / min-p as the target filter, renormalized.
QDist draftDist(std::span<const std::uint32_t> ids, std::span<const float> logits, float m, float sum, float inv_t,
                std::uint32_t top_k, float top_p, float min_p);
// Number of draft logits the device keeps for top_k (min(top_k, q_max), q_max when top_k = 0).
inline std::uint32_t draftCands(std::uint32_t top_k) { return top_k > 0 && top_k < q_max ? top_k : q_max; }
// Race draw from q: argmin over q's support (q > 0) of raceE(base, id) / q (first on ties).
std::uint32_t sampleRace(const QDist& q, std::uint64_t base);

}  // namespace whirl::spec
