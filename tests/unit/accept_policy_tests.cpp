// WHIRL unit tests of module accept_policy (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-accept_policy-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/relax_accept.h"

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
#include <limits>
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
void testRelaxAccept() {
    using whirl::relax::Params;
    Params p;
    p.on = true;
    p.k = 2;
    p.alpha = 0.5f;
    const std::int32_t ids[3] = {7, 3, 9};
    const float lg[3] = {1.0f, 2.0f, 2.0f - 0.6f};  // ranks: 3, 9, 7; p9/p3 = e^-0.6 = 0.549
    CHECK(whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 3));     // argmax
    CHECK(whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 9));     // 2nd, ratio 0.549 >= 0.5
    CHECK(!whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 7));    // 3rd
    CHECK(!whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 42));   // not a candidate
    p.alpha = 0.6f;
    CHECK(!whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 9));    // ratio below alpha
    p.alpha = 0.5f;
    p.k = 1;
    CHECK(!whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 9) && whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 3));
    const std::int32_t tid[2] = {5, 4};
    const float tl[2] = {1.0f, 1.0f};  // tie: lower id is the argmax
    CHECK(!whirl::relax::acceptGreedy(p, tid, tl, 1.0f, 5) && whirl::relax::acceptGreedy(p, tid, tl, 1.0f, 4));
    // repetition guard (6-gram)
    {
        const std::uint32_t h[8] = {1, 2, 3, 4, 5, 6, 9, 1};
        const std::uint32_t t1[5] = {2, 3, 4, 5, 6};  // ... 1 2 3 4 5 6 repeats
        const std::uint32_t t2[5] = {2, 3, 4, 5, 7};
        CHECK(whirl::relax::extendsRepeat(h, t1) && !whirl::relax::extendsRepeat(h, t2));
    }
    // typical: min(eps, delta * e^-H)
    CHECK(whirl::relax::acceptTypical(p, 0.1, 0.0) && !whirl::relax::acceptTypical(p, 0.08, 0.0));
    CHECK(whirl::relax::acceptTypical(p, 0.3 * std::exp(-3.0) + 1e-9, 3.0) && !whirl::relax::acceptTypical(p, 0.01, 3.0));
}

// v0252-f1: the one-pass rule acceptRank (rank = entries of the whole row strictly ahead of the draft, ties to the lower
// id) against acceptGreedy on the row's EXACT top-K (lowest ids at ties): the same verdict for every token of every row
void testRelaxRank() {
    using whirl::relax::Params;
    Params p;
    p.on = true;
    p.k = 2;
    p.alpha = 0.5f;
    CHECK(whirl::relax::acceptRank(p, 0, 2.0f, 2.0f));          // the argmax
    CHECK(whirl::relax::acceptRank(p, 1, 2.0f - 0.6f, 2.0f));   // 2nd, ratio 0.549 >= 0.5
    CHECK(!whirl::relax::acceptRank(p, 2, 1.0f, 2.0f));         // 3rd
    CHECK(!whirl::relax::acceptRank(p, 1, 2.0f - 0.8f, 2.0f));  // 2nd, ratio 0.449 < 0.5
    CHECK(!whirl::relax::acceptRank(p, -1, 2.0f, 2.0f));        // no rank (corrupt row): rejected
    CHECK(!whirl::relax::acceptRank(p, std::numeric_limits<std::int32_t>::max(), 2.0f, 2.0f));  // a row without a draft
    CHECK(!whirl::relax::acceptRank(p, 0, std::nanf(""), 2.0f));  // NaN draft logit: rejected
    p.k = 1;
    CHECK(whirl::relax::acceptRank(p, 0, 2.0f, 2.0f) && !whirl::relax::acceptRank(p, 1, 2.0f, 2.0f));  // k 1 = exact
    p.k = 40;  // topk() caps at 16
    CHECK(whirl::relax::acceptRank(p, 15, 2.0f, 2.0f) && !whirl::relax::acceptRank(p, 16, 2.0f, 2.0f));
    // equivalence on random rows with many ties (values on a grid of 7), V 12..40, k 1..6, alpha 0.05..0.9
    std::uint32_t g = 0xF1u, n = 0, same = 0, kept = 0;
    for (int it = 0; it < 3000; ++it) {
        g = g * 1664525u + 1013904223u;
        const std::uint32_t V = 12 + (g >> 8) % 29;
        p.k = 1 + (g >> 16) % 6;
        p.alpha = 0.05f + 0.85f * static_cast<float>((g >> 4) % 100) / 100.0f;
        std::vector<float> L(V);
        for (float& l : L) {
            g = g * 1664525u + 1013904223u;
            l = static_cast<float>(static_cast<int>((g >> 8) % 7)) * 0.4f;
        }
        std::vector<std::uint32_t> o(V);
        for (std::uint32_t i = 0; i < V; ++i) o[i] = i;
        std::sort(o.begin(), o.end(), [&](std::uint32_t a, std::uint32_t b) { return L[a] != L[b] ? L[a] > L[b] : a < b; });
        const std::uint32_t K = p.topk();
        std::vector<std::int32_t> ids(K);
        std::vector<float> vals(K);
        for (std::uint32_t j = 0; j < K; ++j) ids[j] = static_cast<std::int32_t>(o[j]), vals[j] = L[o[j]];
        const float m = L[o[0]];
        for (std::uint32_t d = 0; d < V; ++d) {
            std::int32_t rank = 0;
            for (std::uint32_t i = 0; i < V; ++i) rank += (L[i] > L[d] || (L[i] == L[d] && i < d)) ? 1 : 0;
            const bool a = whirl::relax::acceptGreedy(p, ids, vals, m, d), b = whirl::relax::acceptRank(p, rank, L[d], m);
            same += a == b;
            kept += b;
            ++n;
        }
    }
    CHECK_EQ(same, n);
    CHECK(kept > 1000);
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_accept_policy() {
    static const whirl::test::Suite s{"accept_policy",
                                      {
                                          {"testRelaxAccept", [] { testRelaxAccept(); }},
                                          {"testRelaxRank", [] { testRelaxRank(); }},  // v0252-f1
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_accept_policy()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
