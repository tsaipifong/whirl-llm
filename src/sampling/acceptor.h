// Draft acceptance of one slot's verify rows (arch025-E1a, moved from the decodeCycle acceptance block of
// src/server/engine.cpp): which drafts survive a verify and which token follows them.
//
//   greedy            a draft is kept iff it is the verify row's argmax
//   relaxacc_greedy   + a near-argmax MTP draft (top-k and p >= alpha p_max, whirl/relax_accept.h)
//   sample            temperature > 0: kept iff it equals the sampler's inverse-CDF draw at that index
//   specsample        the same with the race draw (Sampler::race, whirl/spec_sample.h)
//   relaxacc_typical  + Medusa typical acceptance of an MTP draft on the filtered distribution
//
// RNG: the target token at output index n_gen + acc is the sampler's draw keyed by (seed, that index); the
// loop sets Sampler::pos_idx right after preparing each row and before any draw (prepareRow -> pos_idx ->
// typical check -> sampleRow). That order is the RNG contract the acceptor tests pin.
// Known, kept as is (move only): F2-02, the typical entropy h of an open row covers the candidates only
// (audit\review2\codereview_fast2-relaxacc.md:63).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sampling/sampler.h"
#include "whirl/relax_accept.h"

#include <cstdint>
#include <functional>
#include <span>

namespace whirl::sampling {

enum class AcceptRule : std::uint8_t { greedy, relaxacc_greedy, sample, specsample, relaxacc_typical };

// The rule a slot's verify uses: greedy request?, relaxacc applies to its drafts (MTP drafts in fast mode)?,
// race sampler (specsample)?
AcceptRule acceptRule(bool greedy, bool relax, bool race);
const char* ruleName(AcceptRule r);

// One slot's drafts in a verify batch.
struct Drafts {
    std::span<const std::uint32_t> dr;  // the proposed drafts (dr[0..nd) are valid)
    std::uint32_t nd = 0;               // drafts the verify checked
    std::uint32_t row0 = 0;             // the slot's first verify row in the batch
};

// relaxacc greedy: the device top-k of every verify row, as the engine downloads it: ids
// [max_rows * k] int32, logits [max_rows * k] float, then (max, sum) [max_rows * 2] float.
struct RelaxRows {
    const std::uint8_t* host = nullptr;
    std::uint32_t max_rows = 0;
    std::uint32_t k = 0;
};

struct Accepted {
    std::uint32_t acc = 0;     // drafts kept
    std::uint32_t chosen = 0;  // the token after them (the target's own pick)
};

// relaxacc repetition guard: would keeping draft acc extend an n-gram of the slot's recent tokens?
using RepeatGuard = std::function<bool(std::uint32_t acc)>;

class Acceptor {
public:
    explicit Acceptor(const relax::Params& relax) : relax_(relax) {}

    // argmax: the verify rows' target argmax (row 0..nd, the slot's control area); relax: relaxacc on for
    // these drafts. relaxed / relax_gap: the slot's counters (added to in place, as before the move).
    Accepted greedy(const std::int32_t* argmax, const Drafts& d, bool relax, const RelaxRows& rr,
                    const RepeatGuard& repeats, std::uint32_t& relaxed, double& relax_gap) const;
    // temperature > 0: rows prepared through src (sm.row_base = d.row0), drawn at output index n_gen + acc;
    // typ: relaxacc typical acceptance for these drafts.
    Accepted sample(RowSource& src, Sampler& sm, const Drafts& d, bool typ, std::uint32_t n_gen,
                    const RepeatGuard& repeats, std::uint32_t& relaxed) const;

private:
    const relax::Params& relax_;
};

}  // namespace whirl::sampling
