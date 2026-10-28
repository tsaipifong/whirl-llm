// WHIRL unit tests of module spec.sample (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-spec-sample-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/spec_sample.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>  // K32GetProcessMemoryInfo (arch025-surv testPlatformProcSurvivors)
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
// speculative sampling (whirl/spec_sample.h): race-coupled draws; the race is an exact draw from p
// (chi-square over many trials, closed and open targets), the draft matches it with probability
// close to sum min(p, q), and whole replies are identical for a seed whatever the drafts / draft
// counts / draft-head state
namespace specsample_test {
namespace sp = whirl::spec;

// Wilson-Hilferty upper quantile of chi-square(df) at z standard deviations
double chi2Crit(double df, double z) {
    const double a = 2.0 / (9.0 * df);
    return df * std::pow(1.0 - a + z * std::sqrt(a), 3.0);
}

// filtered distribution over a row of logits (ids = token id of each logit), the target filter rule
sp::QDist filtered(const std::vector<float>& lg, const std::vector<std::uint32_t>& ids, float inv_t, std::uint32_t top_k, float top_p,
                   float min_p) {
    float m = -1e30f;
    for (float v : lg) m = std::max(m, v);
    float sum = 0;
    for (float v : lg) sum += std::exp((v - m) * inv_t);
    std::vector<std::size_t> ord(lg.size());
    for (std::size_t i = 0; i < ord.size(); ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t b) { return lg[a] != lg[b] ? lg[a] > lg[b] : ids[a] < ids[b]; });
    const std::size_t K = std::min<std::size_t>(sp::draftCands(top_k), ord.size());
    std::vector<std::uint32_t> ci(K);
    std::vector<float> cl(K);
    for (std::size_t i = 0; i < K; ++i) {
        ci[i] = ids[ord[i]];
        cl[i] = lg[ord[i]];
    }
    return sp::draftDist(ci, cl, m, sum, inv_t, top_k, top_p, min_p);
}

// target race draw over a closed distribution, as Engine::sampleRace
std::uint32_t raceDraw(const sp::QDist& pt, std::uint64_t seed, std::uint64_t pos) { return sp::sampleRace(pt, sp::raceBase(seed, pos)); }

// one case: race target against p (chi-square), race-coupled draft acceptance against sum min(p, q)
// and against the greedy draft p(argmax q)
void runCase(const char* name, const std::vector<float>& tl, const std::vector<float>& dl, const std::vector<std::uint32_t>& dsub,
             float inv_t, std::uint32_t top_k, float top_p, float min_p, int trials) {
    std::vector<std::uint32_t> all(tl.size());
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<std::uint32_t>(i);
    const sp::QDist pt = filtered(tl, all, inv_t, top_k, top_p, min_p);  // target p (closed)
    const sp::QDist q = filtered(dl, dsub, inv_t, top_k, top_p, min_p);  // draft q over the subset
    std::vector<double> cnt(tl.size(), 0);
    double acc = 0, exact = 0;
    const std::uint32_t qtop = q.id[0];
    for (int i = 0; i < trials; ++i) {
        const std::uint64_t seed = 0x1234567ull + static_cast<std::uint64_t>(i) * 7919ull;
        const std::uint32_t y = raceDraw(pt, seed, 5);  // target (emitted)
        const std::uint32_t d = raceDraw(q, seed, 5);   // device draft
        cnt[y] += 1;
        acc += d == y;
        exact += y == qtop;
    }
    double chi = 0;
    for (std::size_t t = 0; t < tl.size(); ++t) {
        const double e = pt.of(static_cast<std::uint32_t>(t)) * trials;
        if (e <= 0) {
            CHECK(cnt[t] == 0);  // nothing outside the target's support
            continue;
        }
        chi += (cnt[t] - e) * (cnt[t] - e) / e;
    }
    const double df = std::max(1.0, static_cast<double>(pt.n) - 1.0);
    const double crit = chi2Crit(df, 3.719);  // upper tail ~1e-4
    double ov = 0;
    for (std::uint32_t i = 0; i < q.n; ++i) ov += std::min(static_cast<double>(pt.of(q.id[i])), static_cast<double>(q.p[i]));
    std::printf("  specsample %-22s support p %2u q %2u  chi2 %7.2f (crit %.1f, df %.0f)  accept race %.3f, sum min(p,q) %.3f, greedy draft %.3f\n",
                name, pt.n, q.n, chi, crit, df, acc / trials, ov, exact / trials);
    CHECK(chi < crit);
    CHECK(acc / trials <= ov + 4.0 * std::sqrt(ov * (1 - ov) / trials) + 1e-3);  // no coupling beats sum min(p, q)
    CHECK(acc / trials >= 0.85 * ov);                                              // close to it
    CHECK(acc >= exact);                                                           // better than the greedy draft
}

// open target (Engine::sampleRace): the first `nc` tokens are the candidates, the rest a tail; u < S
// races over the candidates, else the inverse-CDF walk over the tail
void runOpen(const std::vector<float>& tl, float inv_t, std::size_t nc, int trials) {
    std::vector<std::uint32_t> all(tl.size());
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<std::uint32_t>(i);
    const sp::QDist pt = filtered(tl, all, inv_t, 0, 1.f, 0.f);  // sorted, whole vocab
    double S = 0;
    for (std::size_t i = 0; i < nc; ++i) S += pt.p[i];
    sp::QDist cand = pt;
    cand.n = static_cast<std::uint32_t>(nc);
    std::vector<double> cnt(tl.size(), 0);
    for (int i = 0; i < trials; ++i) {
        const std::uint64_t seed = 0xBEEFull + static_cast<std::uint64_t>(i) * 104729ull;
        const double u = sp::uniform(seed, 9, sp::u_sample);
        std::uint32_t y = pt.id[pt.n - 1];
        if (u < S) {
            y = raceDraw(cand, seed, 9);
        } else {
            double cum = S;
            for (std::uint32_t j = static_cast<std::uint32_t>(nc); j < pt.n; ++j) {
                cum += pt.p[j];
                if (cum > u) {
                    y = pt.id[j];
                    break;
                }
            }
        }
        cnt[y] += 1;
    }
    double chi = 0;
    for (std::uint32_t j = 0; j < pt.n; ++j) {
        const double e = pt.p[j] * trials;
        chi += (cnt[pt.id[j]] - e) * (cnt[pt.id[j]] - e) / e;
    }
    const double crit = chi2Crit(static_cast<double>(pt.n) - 1.0, 3.719);
    std::printf("  specsample open target, %zu candidates (mass %.3f): chi2 %7.2f (crit %.1f)\n", nc, S, chi, crit);
    CHECK(chi < crit);
}

// Toy LM for the reply test: target logits of the next token from a hash of the prefix; the draft
// head = target + a perturbation keyed by `dstate` (stands for the draft head's cache / history
// state, which differs between runs in the server).
std::uint64_t mix(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
std::vector<float> toyRow(const std::vector<std::uint32_t>& pre, std::uint32_t dstate, bool draft) {
    std::uint64_t h = 0x51ED27ull;
    for (std::uint32_t t : pre) h = mix(h ^ (t + 0x9E3779B97F4A7C15ull));
    std::vector<float> lg(24);
    for (std::uint32_t v = 0; v < 24; ++v) {
        lg[v] = static_cast<float>(mix(h + v) >> 40) * (5.0f / 16777216.0f);  // [0, 5)
        if (draft) lg[v] += static_cast<float>(mix(h * 31 + v * 7 + dstate * 1000003ull) >> 40) * (1.5f / 16777216.0f);
    }
    return lg;
}

// One reply of n tokens. nd_mode < 0: no drafts; otherwise speculative cycles whose draft count comes
// from a generator keyed by nd_mode (0..4 drafts, like the timing-trained allocation), drafts raced
// from the draft head (dstate) on the drafted prefix, accepted iff equal to the target's race draw.
std::vector<std::uint32_t> toyReply(std::uint64_t seed, int nd_mode, std::uint32_t dstate, float inv_t, std::uint32_t top_k,
                                    float top_p, float min_p, std::size_t n, double* acc_rate) {
    std::vector<std::uint32_t> all(24), out;
    for (std::uint32_t i = 0; i < 24; ++i) all[i] = i;
    std::uint64_t cyc = 0;
    double drafted = 0, accepted = 0;
    while (out.size() < n) {
        const std::uint32_t nd =
            nd_mode < 0 ? 0 : static_cast<std::uint32_t>(mix(cyc++ * 977 + static_cast<std::uint64_t>(nd_mode)) % 5);
        const std::uint64_t n_gen = out.size();
        std::vector<std::uint32_t> dr, pre = out;
        for (std::uint32_t r = 0; r < nd; ++r) {
            const sp::QDist q = filtered(toyRow(pre, dstate, true), all, inv_t, top_k, top_p, min_p);
            dr.push_back(raceDraw(q, seed, n_gen + r));
            pre.push_back(dr.back());
        }
        drafted += nd;
        for (std::uint32_t acc = 0;; ++acc) {
            const sp::QDist pt = filtered(toyRow(out, 0, false), all, inv_t, top_k, top_p, min_p);
            const std::uint32_t t = raceDraw(pt, seed, n_gen + acc);
            out.push_back(t);
            if (acc < nd && t == dr[acc]) {
                accepted += 1;
                continue;
            }
            break;
        }
    }
    out.resize(n);
    if (acc_rate) *acc_rate = drafted > 0 ? accepted / drafted : 0;
    return out;
}

void testReplies() {
    struct F {
        float inv_t;
        std::uint32_t top_k;
        float top_p, min_p;
    };
    const F fs[] = {{1.f / 0.7f, 0, 1.f, 0.f}, {1.f / 0.7f, 20, 0.95f, 0.f}, {1.f, 0, 0.8f, 0.05f}};
    int same = 0, total = 0, seed_diff = 0;
    double acc_sum = 0;
    for (const F& f : fs)
        for (std::uint64_t seed : {1234ull, 42ull, 987654321ull}) {
            const std::vector<std::uint32_t> plain = toyReply(seed, -1, 0, f.inv_t, f.top_k, f.top_p, f.min_p, 96, nullptr);
            for (int nd_mode = 0; nd_mode < 3; ++nd_mode)
                for (std::uint32_t dstate = 1; dstate <= 3; ++dstate) {
                    double ar = 0;
                    const std::vector<std::uint32_t> o = toyReply(seed, nd_mode, dstate, f.inv_t, f.top_k, f.top_p, f.min_p, 96, &ar);
                    same += o == plain;
                    ++total;
                    acc_sum += ar;
                }
            seed_diff += toyReply(seed + 1, -1, 0, f.inv_t, f.top_k, f.top_p, f.min_p, 96, nullptr) != plain;
        }
    std::printf("  specsample replies: %d / %d (3 draft-count patterns x 3 draft states x 3 seeds x 3 filters) == plain, mean acceptance %.3f\n",
                same, total, acc_sum / total);
    CHECK(same == total);
    CHECK(seed_diff == 9);         // the seed does matter
    CHECK(acc_sum / total > 0.3);  // drafts are accepted
}

void testSpecSample() {
    // stream 0 = the server sampler's Sampler::uniform
    {
        const std::uint64_t seed = 987654321ull, pos = 41;
        std::uint64_t z = seed + (pos + 1) * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        CHECK(sp::uniform(seed, pos, sp::u_sample) == static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0));
        CHECK(sp::uniform(seed, pos, 1) != sp::uniform(seed, pos, sp::u_sample));
    }
    // q filter: sorted (logit desc, id asc), top-k closes, min-p cuts, renormalized; buffer decoding
    {
        const std::vector<std::uint32_t> ids = {7, 3, 9, 1};
        const std::vector<float> lg = {1.f, 3.f, 2.f, 2.f};
        const float m = 3.f;
        float sum = 0;
        for (float v : lg) sum += std::exp(v - m);
        sum += 0.5f;  // mass outside the candidates
        const sp::QDist a = sp::draftDist(ids, lg, m, sum, 1.f, 0, 1.f, 0.f);
        CHECK(a.n == 4 && a.id[0] == 3 && a.id[1] == 1 && a.id[2] == 9 && a.id[3] == 7);
        float z = 0;
        for (std::uint32_t i = 0; i < a.n; ++i) z += a.p[i];
        CHECK(std::fabs(z - 1.f) < 1e-6f && a.of(42) == 0.f);
        const sp::QDist b = sp::draftDist(ids, lg, m, sum, 1.f, 2, 1.f, 0.f);
        CHECK(b.n == 2 && b.id[0] == 3 && b.id[1] == 1);
        const sp::QDist c = sp::draftDist(ids, lg, m, sum, 1.f, 0, 1.f, 0.5f);  // e^-1 < 0.5: the top token only
        CHECK(c.n == 1 && c.id[0] == 3 && c.p[0] == 1.f);
        std::int32_t w[sp::q_words] = {};
        w[0] = static_cast<std::int32_t>(b.n);
        for (std::uint32_t i = 0; i < b.n; ++i) {
            w[sp::q_off_ids + i] = static_cast<std::int32_t>(b.id[i]);
            std::memcpy(&w[sp::q_off_p + i], &b.p[i], 4);
        }
        const sp::QDist r = sp::qFromWords(w);
        CHECK(r.n == b.n && r.id[1] == b.id[1] && r.p[1] == b.p[1]);
    }
    // distribution equality on toy rows: V = 24, draft head = perturbed target, draft vocabulary
    // subset = even tokens plus 1 and 3 (q = 0 on the other odd tokens)
    std::vector<float> tl(24), dl, dlfull(24);
    for (int i = 0; i < 24; ++i) {
        tl[i] = 2.5f * std::sin(0.7f * i) + 0.1f * i;
        dlfull[i] = tl[i] + 0.8f * std::cos(1.3f * i);
    }
    std::vector<std::uint32_t> sub, allv(24);
    for (std::uint32_t i = 0; i < 24; ++i) {
        allv[i] = i;
        if (i % 2 == 0 || i == 1 || i == 3) {
            sub.push_back(i);
            dl.push_back(dlfull[i]);
        }
    }
    const int N = 200000;
    runCase("t0.7 top-k20 top-p0.95", tl, dl, sub, 1.f / 0.7f, 20, 0.95f, 0.f, N);
    runCase("t0.7 top-k5", tl, dl, sub, 1.f / 0.7f, 5, 1.f, 0.f, N);
    runCase("t1.0 top-p0.8", tl, dl, sub, 1.f, 0, 0.8f, 0.f, N);
    runCase("t1.0 min-p0.1", tl, dl, sub, 1.f, 0, 1.f, 0.1f, N);
    runCase("t1.5 open, full vocab", tl, dlfull, allv, 1.f / 1.5f, 0, 1.f, 0.f, N);
    runCase("t1.5 open, subset", tl, dl, sub, 1.f / 1.5f, 0, 1.f, 0.f, N);
    runOpen(tl, 1.f, 6, N);
    runOpen(tl, 1.f / 1.5f, 3, N);
    testReplies();
}


}  // namespace specsample_test

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_spec_sample() {
    static const whirl::test::Suite s{"spec.sample",
                                      {
                                          {"testSpecSample", [] { specsample_test::testSpecSample(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_spec_sample()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
