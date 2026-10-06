// Speculative sampling of MTP drafts (numerics item "specsample", balance / fast).
// SPDX-License-Identifier: Apache-2.0
//
// With temperature > 0 the server used to take the MTP draft as the draft head's argmax and accept it
// only when the token sampled from the target equals it. That is lossless and matches plain sampling
// token for token, but a deterministic draft is accepted with probability p(draft) only. Standard
// speculative sampling (Leviathan et al. 2023, Chen et al. 2023) draws the draft from a draft
// distribution q, accepts it with probability min(1, p(d) / q(d)) and otherwise samples from the
// normalized residual max(0, p - q). The output distribution equals plain sampling from p exactly,
// for any q, as long as the draft really is drawn from q and q is known exactly; the sampled
// sequence differs from plain sampling with the same seed, which is why precise mode keeps the
// exact-match rule (bit-identical MTP == plain).
//
// q here: the draft head's softmax at the request temperature over its q_max largest logits, then
// the request's top-k / top-p / min-p (the same rule as the target's candidate filter), renormalized.
// Tokens outside the draft vocabulary subset have q = 0. The device kernel draft_sample_rows
// computes q, draws the draft and writes q (token ids + probabilities) for the host; the host
// computes p and runs accept / residual. The pure parts live here for the unit tests.

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

// Uniform streams keyed by (seed, output token index): 0 = the plain sampler's draw (unchanged),
// 1 = accept test, 2 = residual draw, 3 = the draft draw on the device.
enum : std::uint32_t { u_sample = 0, u_accept = 1, u_residual = 2, u_draft = 3 };
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
// Inverse-CDF draw from q (u in [0, 1)).
std::uint32_t sampleQ(const QDist& q, double u);

// Accept test: u < p / q (q > 0).
inline bool accept(double p, double q, double u) { return u * q < p; }

// Residual draw over a closed target distribution (ids / p, sorted or not, summing to 1):
// r(x) proportional to max(0, p(x) - q(x)). u in [0, 1). When the residual mass is 0 (p <= q
// everywhere, the accept test then always passes) the most likely target token is returned.
std::uint32_t sampleResidual(std::span<const std::uint32_t> ids, std::span<const double> p, const QDist& q, double u);

}  // namespace whirl::spec
