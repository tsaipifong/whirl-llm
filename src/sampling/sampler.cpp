// Token sampling (see sampler.h). Moved from src/server/engine.cpp (arch025-E1a); the bodies are the old
// ones with the device reads going through RowSource (loadFullRow / logitAt / nVocab / inCands).
// SPDX-License-Identifier: Apache-2.0

#include "sampling/sampler.h"

#include "whirl/spec_sample.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace whirl::sampling {

bool filterCands(Sampler& s, std::size_t count, bool complete) {
    std::vector<Cand>& cv = *s.cands;
    Cand* c = cv.data();
    std::sort(c, c + count, [](const Cand& x, const Cand& y) {
        if (x.logit != y.logit) return x.logit > y.logit;
        return x.id < y.id;
    });
    if (complete) {
        // exact host normalizer for the full row
        float mx = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < count; ++i) mx = std::max(mx, c[i].logit);
        double z = 0;
        for (std::size_t i = 0; i < count; ++i) z += std::exp(static_cast<double>(c[i].logit - mx) * s.inv_t);
        s.m = mx;
        s.sum = static_cast<float>(z);
    }
    for (std::size_t i = 0; i < count; ++i)
        c[i].p = std::exp(static_cast<double>(c[i].logit - s.m) * s.inv_t) / static_cast<double>(s.sum);
    std::size_t n = count;
    bool closed = complete;
    if (s.top_k > 0) {
        if (s.top_k <= count) {
            n = s.top_k;
            closed = true;
        } else if (!complete) {
            return false;
        }
    }
    if (s.top_p < 1.0) {
        double z = 1.0;
        if (closed) {
            z = 0;
            for (std::size_t i = 0; i < n; ++i) z += c[i].p;
        }
        double cum = 0;
        bool found = false;
        for (std::size_t j = 0; j < n; ++j) {
            cum += c[j].p / z;
            if (cum >= s.top_p) {
                n = j + 1;
                found = true;
                break;
            }
        }
        if (found) closed = true;
        else if (!closed) return false;
    }
    if (s.min_p > 0) {
        const double thr = s.min_p * c[0].p;
        std::size_t j = 0;
        while (j < n && c[j].p >= thr) ++j;
        if (j == n && !closed) return false;
        n = std::max<std::size_t>(j, 1);
        closed = true;
    }
    if (closed) {
        double z = 0;
        for (std::size_t i = 0; i < n; ++i) z += c[i].p;
        for (std::size_t i = 0; i < n; ++i) c[i].p /= z;
    }
    s.n = n;
    s.closed = closed;
    return true;
}

// specsample target draw (whirl/spec_sample.h): race over the candidates; an open distribution
// first takes the tail with probability 1 - S (S = candidate mass) and walks it by inverse CDF.
std::uint32_t sampleRace(RowSource& src, Sampler& s) {
    const std::vector<Cand>& c = *s.cands;
    if (!s.closed) {
        double S = 0;
        for (std::size_t i = 0; i < s.n; ++i) S += c[i].p;
        if (!(s.uniform() < S)) {
            s.race = false;  // the inverse-CDF walk with the same u falls past the candidates
            const std::uint32_t t = sampleRow(src, s, std::nullopt, 0);
            s.race = true;
            return t;
        }
    }
    const std::uint64_t base = spec::raceBase(s.seed, s.pos_idx);
    std::uint32_t best = c[0].id;
    double bk = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < s.n; ++i) {
        if (!(c[i].p > 0)) continue;
        const double k = spec::raceE(base, c[i].id) / c[i].p;
        if (k < bk) {
            bk = k;
            best = c[i].id;
        }
    }
    return best;
}

double probOf(RowSource& src, Sampler& s, std::uint32_t t) {
    const std::vector<Cand>& c = *s.cands;
    for (std::size_t i = 0; i < s.n; ++i)
        if (c[i].id == t) return c[i].p;
    if (s.closed) return 0;
    float l = 0;
    l = src.logitAt(s.row, t);
    return std::exp(static_cast<double>(l - s.m) * s.inv_t) / static_cast<double>(s.sum);
}

std::uint32_t sampleRow(RowSource& src, Sampler& s, std::optional<std::uint32_t> ex, double p_ex) {
    if (s.race && !ex) return sampleRace(src, s);
    const double u = s.uniform();
    const double target = u * (1.0 - p_ex);
    double cum = 0;
    std::optional<std::uint32_t> last;
    const std::vector<Cand>& c = *s.cands;
    for (std::size_t i = 0; i < s.n; ++i) {
        if (ex && c[i].id == *ex) continue;
        cum += c[i].p;
        last = c[i].id;
        if (cum > target) return c[i].id;
    }
    if (s.closed) return last ? *last : c[0].id;
    // open distribution and the draw fell in the tail: walk the rest of the vocab
    s.full_used += 1;
    const float* row_host = src.loadFullRow(s.row);
    const std::uint32_t V = src.nVocab();
    std::uint8_t* in_cands = src.inCands();
    std::fill(in_cands, in_cands + V, std::uint8_t{0});
    for (std::size_t i = 0; i < s.n; ++i) in_cands[c[i].id] = 1;
    for (std::uint32_t j = 0; j < V; ++j) {
        if (in_cands[j]) continue;
        if (ex && j == *ex) continue;
        cum += std::exp(static_cast<double>(row_host[j] - s.m) * s.inv_t) / static_cast<double>(s.sum);
        last = j;
        if (cum > target) return j;
    }
    return last ? *last : c[0].id;
}

}  // namespace whirl::sampling
