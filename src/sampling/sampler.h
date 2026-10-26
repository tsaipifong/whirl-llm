// Token sampling of the server (arch025-E1a, moved from src/server/engine.{h,cpp}): the per-request sampler
// state, the candidate filter (top-k / top-p / min-p) and the draws (inverse CDF; the specsample race,
// whirl/spec_sample.h). Host-only: the device side of a verify row comes in through RowSource.
//
// RNG: no stream state. Every draw is a counter hash keyed by (seed, output token index):
// Sampler::uniform() (inverse CDF, the open-vs-tail pick of the race) and spec::raceE (the race). The order
// of the calls only matters through pos_idx, which the caller (the Acceptor) sets per output index. The
// server's only stateful PRNG is IdGen (src/server/stream.h: response ids and default seeds), not here.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace whirl::sampling {

struct Cand {
    std::uint32_t id;
    float logit;
    double p;
};

struct Sampler {
    double temp = 0;
    float inv_t = 1;
    std::uint32_t top_k = 0;
    double top_p = 1;
    double min_p = 0;
    std::uint64_t seed = 0;
    std::uint64_t pos_idx = 0;
    std::vector<Cand>* cands = nullptr;
    std::size_t n = 0;
    bool closed = false;
    float m = 0;
    float sum = 1;
    std::uint32_t row = 0;
    std::uint32_t row_base = 0;
    std::uint32_t full_used = 0;
    std::uint32_t big_used = 0;
    bool race = false;  // specsample: race draw coupled to the MTP drafts (whirl/spec_sample.h)

    // Uniform in [0, 1) keyed by (seed, output token index).
    double uniform() const {
        std::uint64_t z = seed + (pos_idx + 1) * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    }
};

// Candidates [0, count) -> sorted, probabilities, top-k / top-p / min-p.
// false when the kept set may extend past the candidates.
bool filterCands(Sampler& s, std::size_t count, bool complete);

// The device side of the verify rows a sampler reads (the server engine; tests: a host fake).
class RowSource {
public:
    virtual ~RowSource() = default;
    // Candidates of verify row s.row_base + r into *s.cands, filtered (sets s.row, s.m, s.sum, s.n, s.closed;
    // escalates to a larger device top-k and then the full row while filterCands says the set is open).
    virtual void prepareRow(Sampler& s, std::uint32_t r) = 0;
    // The full logits row gr on the host (n_vocab floats, valid until the next call).
    virtual const float* loadFullRow(std::uint32_t gr) = 0;
    // One logit of row gr (a token outside an open row's candidates).
    virtual float logitAt(std::uint32_t gr, std::uint32_t t) = 0;
    virtual std::uint32_t nVocab() const = 0;
    // n_vocab bytes of host scratch (the tail walk's candidate marks).
    virtual std::uint8_t* inCands() = 0;
};

// p(t) on the sampler's filtered distribution of the prepared row (relaxacc typical acceptance); an open row
// reads t's logit from the device when t is not a candidate.
double probOf(RowSource& src, Sampler& s, std::uint32_t t);
// specsample target draw: race over the candidates; an open distribution first takes the tail with
// probability 1 - S (S = candidate mass) and walks it by inverse CDF.
std::uint32_t sampleRace(RowSource& src, Sampler& s);
// One draw from the prepared row (s.race: the race; else inverse CDF). ex / p_ex: the residual draw that
// excludes token ex of mass p_ex (no caller since the Leviathan / Chen rule was removed, spec_sample.h).
std::uint32_t sampleRow(RowSource& src, Sampler& s, std::optional<std::uint32_t> ex, double p_ex);

}  // namespace whirl::sampling
