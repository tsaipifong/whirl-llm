// Draft acceptance (see acceptor.h). The two loops are the old decodeCycle acceptance block
// (src/server/engine.cpp before arch025-E1a) with the slot's fields passed in.
// SPDX-License-Identifier: Apache-2.0

#include "sampling/acceptor.h"

#include <cmath>
#include <optional>
#include <vector>

namespace whirl::sampling {

AcceptRule acceptRule(bool greedy, bool relax, bool race) {
    if (greedy) return relax ? AcceptRule::relaxacc_greedy : AcceptRule::greedy;
    if (relax) return AcceptRule::relaxacc_typical;
    return race ? AcceptRule::specsample : AcceptRule::sample;
}

const char* ruleName(AcceptRule r) {
    switch (r) {
    case AcceptRule::greedy: return "greedy";
    case AcceptRule::relaxacc_greedy: return "relaxacc_greedy";
    case AcceptRule::sample: return "sample";
    case AcceptRule::specsample: return "specsample";
    case AcceptRule::relaxacc_typical: return "relaxacc_typical";
    }
    return "?";
}

Accepted Acceptor::greedy(const std::int32_t* argmax, const Drafts& d, bool relax, const RelaxRows& rr,
                          const RepeatGuard& repeats, std::uint32_t& relaxed, double& relax_gap) const {
    const std::span<const std::uint32_t> dr = d.dr;
    const std::uint32_t nd_dev = d.nd;
    const std::uint32_t row0 = d.row0;
    const std::uint32_t rk = rr.k;
    std::uint32_t acc = 0;
    std::uint32_t chosen;
    while (acc < nd_dev) {
        if (static_cast<std::uint32_t>(argmax[acc]) == dr[acc]) {
            ++acc;
            continue;
        }
        // n-gram drafts are copies of the history: relaxing them feeds repetition loops
        if (!relax) break;
        // relaxacc: verify row acc (conditioned on drafts 0..acc-1) keeps a near-argmax draft
        const std::size_t gr = row0 + acc;
        const auto* rids = reinterpret_cast<const std::int32_t*>(rr.host) + gr * rk;
        const auto* rvals = reinterpret_cast<const float*>(rr.host + static_cast<std::size_t>(rr.max_rows) * rk * 4) + gr * rk;
        const auto* rst = reinterpret_cast<const float*>(rr.host + static_cast<std::size_t>(rr.max_rows) * rk * 8);
        if (!relax::acceptGreedy(relax_, std::span<const std::int32_t>(rids, rk), std::span<const float>(rvals, rk),
                                 rst[2 * gr], dr[acc]) ||
            repeats(acc))
            break;
        for (std::uint32_t j = 0; j < rk; ++j)
            if (static_cast<std::uint32_t>(rids[j]) == dr[acc]) relax_gap += static_cast<double>(rst[2 * gr] - rvals[j]);
        ++acc;
        relaxed += 1;
    }
    chosen = static_cast<std::uint32_t>(argmax[acc]);
    return {acc, chosen};
}

Accepted Acceptor::sample(RowSource& src, Sampler& sm, const Drafts& d, bool typ, std::uint32_t n_gen,
                          const RepeatGuard& repeats, std::uint32_t& relaxed) const {
    const std::span<const std::uint32_t> dr = d.dr;
    const std::uint32_t nd_dev = d.nd;
    std::uint32_t acc = 0;
    std::uint32_t chosen;
    // The target token at every output index is the sampler's draw keyed by (seed, output
    // index); a draft is accepted iff it is that token. With specsample the sampler is the race
    // and MTP draft r was drawn on the device from the draft distribution q by the same race
    // (whirl/spec_sample.h); otherwise inverse CDF and the draft argmax. Either way the reply
    // is a function of (target logits, seed) only: drafts change speed, never text.
    sm.row_base = d.row0;
    for (;;) {
        src.prepareRow(sm, acc);
        sm.pos_idx = n_gen + acc;
        if (typ && acc < nd_dev) {
            // relaxacc typical acceptance on the filtered sampling distribution of row acc
            const double pd = probOf(src, sm, dr[acc]);
            double h = 0;
            const std::vector<Cand>& cv = *sm.cands;
            for (std::size_t i = 0; i < sm.n; ++i)
                if (cv[i].p > 0) h -= cv[i].p * std::log(cv[i].p);
            if (relax::acceptTypical(relax_, pd, h) && !repeats(acc)) {
                ++acc;
                relaxed += 1;
                continue;
            }
        }
        const std::uint32_t t = sampleRow(src, sm, std::nullopt, 0);
        if (acc < nd_dev && t == dr[acc]) {
            ++acc;
            continue;
        }
        chosen = t;
        break;
    }
    return {acc, chosen};
}

}  // namespace whirl::sampling
