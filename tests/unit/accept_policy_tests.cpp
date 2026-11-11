// WHIRL unit tests of module accept_policy (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-accept_policy-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/relax_accept.h"
#include "../../kernels/spec_accept.h"  // v0252-e1p1 (feat026-acc port): the device accept decision, host side

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
#include <type_traits>
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

// v0252-e1p1 (port of feat026-acc specacc_test, 4ac9da6): kernels/spec_accept.h (the accept decision the device kernel
// spec_accept_rows makes) against an independent copy of the host path of a greedy slot without relaxacc - int0252's
// SpecCycle::acceptSlot (src/server/engine_decode.cpp): sampling::Acceptor::greedy with relax off (the longest prefix of
// drafts equal to the verify rows' argmax, chosen = the argmax of the first row not kept), then the end-of-turn cut and the
// max_tokens cut - on 20,000 random cycles: MTP / literal drafts, the device p-min draft count (stale / -1), matching
// prefixes of every length, end-of-turn drafts, ignore_eos, max_tokens budgets; plus the next-cycle rows against the host's
// slot update (pend = dr[0..acc) + chosen at pos + 1 .., next = chosen at pos + acc + 1).
namespace specacc_test {
struct HostRef {
    std::uint32_t nd_dev = 0, acc0 = 0, acc = 0, chosen = 0;
};
HostRef hostLoop(const std::int32_t* ctl, std::uint32_t nd_k, bool is_ng, const std::uint32_t* ng_d, bool ignore_eos,
                 std::uint32_t im_end, std::uint32_t endoftext, std::uint32_t max_tokens, std::uint32_t n_gen) {
    HostRef h;
    const std::uint32_t nd_dev = is_ng ? nd_k : (nd_k > 0 ? std::min(nd_k, static_cast<std::uint32_t>(std::max(0, ctl[48]))) : 0);
    std::uint32_t dr[16] = {};
    for (std::uint32_t r = 0; r < nd_k; ++r) dr[r] = is_ng ? ng_d[r] : static_cast<std::uint32_t>(ctl[16 + r]);
    std::uint32_t acc = 0;
    while (acc < nd_dev && static_cast<std::uint32_t>(ctl[acc]) == dr[acc]) ++acc;
    std::uint32_t chosen = static_cast<std::uint32_t>(ctl[acc]);
    h.acc0 = acc;
    if (!ignore_eos) {
        for (std::uint32_t r = 0; r < acc; ++r) {
            if (dr[r] == im_end || dr[r] == endoftext) {
                chosen = dr[r];
                acc = r;
                break;
            }
        }
    }
    {
        const std::uint32_t budget = max_tokens > n_gen ? max_tokens - n_gen : 0;
        if (budget >= 1 && acc + 1 > budget) {
            chosen = dr[budget - 1];
            acc = budget - 1;
        }
    }
    h.nd_dev = nd_dev;
    h.acc = acc;
    h.chosen = chosen;
    return h;
}

void testSpecAccept() {
    std::uint64_t st = 0x9e3779b97f4a7c15ull;
    auto rnd = [&](std::uint32_t n) {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        return static_cast<std::uint32_t>(st % n);
    };
    const std::uint32_t V = 1000, IM_END = 998, EOT = 999;
    std::uint64_t n_eos = 0, n_budget = 0, n_full = 0, tab_ok = 0, bad = 0;
    const int iters = 20000;
    for (int it = 0; it < iters; ++it) {
        const int A = 1 + static_cast<int>(rnd(16));
        std::vector<std::int32_t> ctl_all(16 * 64, 0);
        SpecAccArgs a{};
        a.n = A;
        a.n_vocab = static_cast<int>(V - (it % 4 == 0 ? 10 : 0));  // some tokens past n_vocab: lookupId -> 0
        a.stop0 = IM_END;
        a.stop1 = (it % 5 == 0) ? 0xffffffffu : EOT;
        std::vector<HostRef> want(A);
        std::vector<int> ids(16);
        for (int i = 0; i < 16; ++i) ids[i] = i;
        for (int i = 15; i > 0; --i) std::swap(ids[i], ids[rnd(static_cast<std::uint32_t>(i) + 1)]);
        int row = 0;
        for (int k = 0; k < A; ++k) {
            const int sq = ids[k];
            std::int32_t* ctl = ctl_all.data() + sq * 64;
            // the engine's row budget: one verify runs sum(nd + 1) <= spec_.vrows <= max_verify_rows (32) rows (SpecNextTab /
            // RowTab hold WHIRL_SA_MAX_ROWS rows); one row is left for every sequence still to come
            const int nd_cap = std::min(WHIRL_SA_MAX_ND, WHIRL_SA_MAX_ROWS - row - (A - k));
            const std::uint32_t nd = nd_cap > 0 ? rnd(static_cast<std::uint32_t>(nd_cap) + 1) : 0;
            const bool lit = rnd(3) == 0;
            const bool ign = rnd(4) == 0;
            const std::uint32_t n_gen = rnd(50);
            const std::uint32_t max_tok = rnd(3) == 0 ? n_gen + rnd(6) : n_gen + 100;
            a.seq[k] = sq;
            a.nd[k] = static_cast<int>(nd);
            a.lit[k] = lit ? 1 : 0;
            a.dev[k] = 1;
            a.eos[k] = ign ? 0 : 1;
            a.budget[k] = static_cast<int>(max_tok > n_gen ? max_tok - n_gen : 0);
            a.pos[k] = static_cast<int>(rnd(100000));
            a.row0[k] = row;
            row += static_cast<int>(nd) + 1;
            a.kvbase[k] = static_cast<int>(rnd(1u << 20));
            ctl[48] = static_cast<std::int32_t>(rnd(17)) - 1;  // p-min kept count (may be stale / -1)
            std::uint32_t d[16] = {};
            const std::uint32_t match = rnd(nd + 2);
            for (std::uint32_t r = 0; r < nd; ++r) {
                d[r] = rnd(V);
                if (rnd(8) == 0) d[r] = rnd(2) ? IM_END : EOT;
            }
            for (std::uint32_t r = 0; r < 16; ++r) ctl[r] = static_cast<std::int32_t>(rnd(V));
            for (std::uint32_t r = 0; r < nd && r < match; ++r) ctl[r] = static_cast<std::int32_t>(d[r]);
            std::uint32_t litd[16] = {};
            for (std::uint32_t r = 0; r < nd; ++r) {
                if (lit) {
                    a.lit_d[k][r] = static_cast<int>(d[r]);
                    litd[r] = d[r];
                    ctl[16 + r] = static_cast<std::int32_t>(rnd(V));  // stale MTP drafts: must be ignored
                } else {
                    ctl[16 + r] = static_cast<std::int32_t>(d[r]);
                }
            }
            want[k] = hostLoop(ctl, nd, lit, litd, ign, IM_END, a.stop1, max_tok, n_gen);
        }
        CHECK(row <= WHIRL_SA_MAX_ROWS);
        SpecAccResult res[WHIRL_SA_MAX_SEG];
        for (int k = 0; k < A; ++k) {
            std::int32_t* ctl = ctl_all.data() + a.seq[k] * 64;
            std::int32_t low[50];
            std::memcpy(low, ctl, sizeof(low));
            const SpecAccResult r = spec_accept_one(a, k, ctl);
            const bool same = static_cast<std::uint32_t>(r.nd_dev) == want[k].nd_dev && static_cast<std::uint32_t>(r.acc0) == want[k].acc0 &&
                              static_cast<std::uint32_t>(r.acc) == want[k].acc && static_cast<std::uint32_t>(r.chosen) == want[k].chosen &&
                              (r.flags & WHIRL_SA_F_DEV) != 0;
            if (!same && bad++ < 5)
                std::printf("FAIL spec_accept it %d k %d: dev nd %d acc0 %d acc %d chosen %d vs host %u %u %u %u\n", it, k, r.nd_dev, r.acc0,
                            r.acc, r.chosen, want[k].nd_dev, want[k].acc0, want[k].acc, want[k].chosen);
            n_eos += (r.flags & WHIRL_SA_F_EOS) != 0;
            n_budget += (r.flags & WHIRL_SA_F_BUDGET) != 0;
            n_full += r.acc0 == r.nd_dev && r.nd_dev > 0;
            spec_store(a, k, r, ctl);
            const bool stored = ctl[WHIRL_SA_CTL_ACC] == r.acc && ctl[WHIRL_SA_CTL_ACC0] == r.acc0 && ctl[WHIRL_SA_CTL_CHOSEN] == r.chosen &&
                                ctl[WHIRL_SA_CTL_NDDEV] == r.nd_dev && ctl[WHIRL_SA_CTL_FLAGS] == r.flags &&
                                ctl[WHIRL_SA_CTL_NPOS] == a.pos[k] + r.acc + 1 && ctl[WHIRL_SA_CTL_NROW] == a.row0[k] + r.acc &&
                                ctl[WHIRL_SA_CTL_NPEND] == r.acc + 1 && r.pos == a.pos[k] &&  // v0252-e1p1b
                                std::memcmp(low, ctl, sizeof(low)) == 0;  // words < 50 untouched
            if (!stored) bad += 1;
            res[k] = r;
        }
        // next-cycle rows vs the host's slot update (pend = dr[0..acc), chosen; pend_pos = pos + 1; next = chosen;
        // pos += acc + 1) and the tables Context::mtpBatchStepEx / verifyBatchEnqueue build (lookupId, kv_base)
        SpecNextTab t{};
        spec_next_rows(a, ctl_all.data(), res, &t);
        int n = 0;
        bool ok = true;
        auto look = [&](std::uint32_t tok) { return static_cast<std::int32_t>(tok < static_cast<std::uint32_t>(a.n_vocab) ? tok : 0); };
        for (int k = 0; k < A; ++k) {
            const std::int32_t* ctl = ctl_all.data() + a.seq[k] * 64;
            const std::uint32_t acc = want[k].acc;
            for (std::uint32_t j = 0; j <= acc; ++j, ++n) {
                const std::uint32_t tok =
                    j < acc ? (a.lit[k] ? static_cast<std::uint32_t>(a.lit_d[k][j]) : static_cast<std::uint32_t>(ctl[16 + j])) : want[k].chosen;
                ok = ok && t.mtp_tok[n] == look(tok) && t.mtp_pos[n] == a.pos[k] + 1 + static_cast<int>(j) && t.mtp_base[n] == a.kvbase[k];
            }
            ok = ok && t.mtp_last[k] == n - 1 && t.ver_tok[k] == look(want[k].chosen) &&
                 t.ver_pos[k] == a.pos[k] + static_cast<int>(acc) + 1 && t.ver_base[k] == a.kvbase[k];
        }
        ok = ok && t.n_mtp == n;
        tab_ok += ok ? 1 : 0;
    }
    CHECK_EQ(bad, 0u);
    CHECK_EQ(tab_ok, static_cast<std::uint64_t>(iters));
    // the random cycles reached every branch
    CHECK(n_eos > 100);
    CHECK(n_budget > 100);
    CHECK(n_full > 100);
    // v0252-e1p1b (E1): a.pos[k] < 0 takes the position from the NPOS word of the sequence's last accept (read before
    // spec_store writes the new one), NPEND = acc + 1, and a host-decided sequence (spec_copy_one) resolves it the same way
    {
        SpecAccArgs a{};
        a.n = 2;
        a.n_vocab = 1000;
        a.stop0 = 998;
        a.stop1 = 999;
        std::vector<std::int32_t> cc(2 * 64, 0);
        for (int k = 0; k < 2; ++k) {
            a.seq[k] = k;
            a.nd[k] = 3;
            a.dev[k] = k == 0 ? 1 : 0;
            a.eos[k] = 1;
            a.budget[k] = 100;
            a.pos[k] = -1;
            a.row0[k] = 4 * k;
        }
        std::int32_t* c0 = cc.data();
        std::int32_t* c1 = cc.data() + 64;
        for (int r = 0; r < 3; ++r) c0[16 + r] = 100 + r;
        c0[0] = 100;
        c0[1] = 101;
        c0[2] = 7;  // drafts 0 and 1 accepted, the trunk's token 7 after them
        c0[48] = 3;
        c0[WHIRL_SA_CTL_NPOS] = 5000;
        c1[0] = 42;
        c1[WHIRL_SA_CTL_NPOS] = 6000;
        const SpecAccResult r0 = spec_accept_one(a, 0, c0);
        const SpecAccResult r1 = spec_copy_one(a, 1, c1);
        CHECK(r0.pos == 5000 && r0.acc == 2 && r0.chosen == 7);
        CHECK(r1.pos == 6000 && r1.acc == 0 && r1.chosen == 42 && r1.flags == 0);
        spec_store(a, 0, r0, c0);
        spec_store(a, 1, r1, c1);
        CHECK(c0[WHIRL_SA_CTL_NPOS] == 5003 && c0[WHIRL_SA_CTL_NPEND] == 3 && c1[WHIRL_SA_CTL_NPOS] == 6001 && c1[WHIRL_SA_CTL_NPEND] == 1);
        const SpecAccResult res2[2] = {r0, r1};
        SpecNextTab t{};
        spec_next_rows(a, cc.data(), res2, &t);
        CHECK(t.n_mtp == 4 && t.mtp_pos[0] == 5001 && t.mtp_pos[2] == 5003 && t.ver_pos[0] == 5003 && t.mtp_pos[3] == 6001 &&
              t.ver_pos[1] == 6001 && t.ver_tok[0] == 7 && t.ver_tok[1] == 42);
        // by value (a.pos >= 0) the NPOS word is not read
        a.pos[0] = 70;
        c0[WHIRL_SA_CTL_NPOS] = -9;
        CHECK(spec_accept_one(a, 0, c0).pos == 70);
    }
    static_assert(WHIRL_SA_CTL_NPEND == 57 && WHIRL_SA_CTL_NPEND < WHIRL_SA_CTL_WORDS);
    // the kernel argument the launch copies: no padding, the expected size
    static_assert(sizeof(SpecAccArgs) == (4 + 9 * 16 + 16 * 15) * 4);
    static_assert(std::has_unique_object_representations_v<SpecAccArgs>);
    // the control words the kernel writes (50 .. 56) lie past every word the verify uses (rows 0..15, drafts 16..31,
    // probs, nd 48, stop 49) and inside the 64-word area
    static_assert(WHIRL_SA_CTL_ACC == 50 && WHIRL_SA_CTL_ACC0 == 56 && WHIRL_SA_CTL_WORDS == 64);
}
}  // namespace specacc_test

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_accept_policy() {
    static const whirl::test::Suite s{"accept_policy",
                                      {
                                          {"testRelaxAccept", [] { testRelaxAccept(); }},
                                          {"testRelaxRank", [] { testRelaxRank(); }},  // v0252-f1
                                          {"testSpecAccept", [] { specacc_test::testSpecAccept(); }},  // v0252-e1p1 (feat026-acc)
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_accept_policy()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
