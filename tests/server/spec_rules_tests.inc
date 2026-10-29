// arch025-R12cli route B: pins of the speculative-decoding rules shared by the CLI specDecode and the server Engine
// (src/spec/spec_rules.h). Included by tests/server/server_tests.cpp (needs CHECK / CHECK_EQ, TestServer, g_log).
//
// Each rule is pinned on both sides:
//   server side  the shared function == a verbatim copy of the engine code before the change (f68a9aa engine.cpp,
//                quoted with file:line) over a grid / random inputs: the server's decisions do not change.
//   CLI side     the rule the CLI loop applies (cliRule* below: the call decode.cpp makes) == the owner-chosen
//                expectation. D1 / D2 are unified to the SERVER rule (owner 2026-10-28), so their CLI tests were
//                written to the server expectation first and failed against the pre-change CLI code (kept below as
//                old_cli*, the f68a9aa decode.cpp lines); D3 / D8 / D14 / D6 stay what the CLI did.
// SPDX-License-Identifier: Apache-2.0

#include "spec/spec_rules.h"

#include <bit>
#include <random>

namespace r12b {

namespace sr = whirl::spec::rules;
namespace q35 = whirl::qwen35;

// ---- verbatim pre-change code (f68a9aa) ----------------------------------------------------------------------------

// engine.cpp:1317-1322 Engine::relaxRepeats (sl.cache_tokens -> hist, sl.next -> next)
bool old_srv_relaxRepeats(std::span<const std::uint32_t> hist, std::uint32_t next, std::span<const std::uint32_t> dr, std::uint32_t acc) {
    std::array<std::uint32_t, q35::max_ng_drafts + 2> tail{};
    tail[0] = next;
    for (std::uint32_t r = 0; r <= acc; ++r) tail[r + 1] = dr[r];
    return whirl::relax::extendsRepeat(std::span<const std::uint32_t>(hist), std::span<const std::uint32_t>(tail.data(), acc + 2));
}
// decode.cpp:221 (generated = the emitted tokens, `next` last; no prompt)
bool old_cli_relaxRepeats(std::span<const std::uint32_t> generated, std::span<const std::uint32_t> dr, std::uint32_t acc) {
    return whirl::relax::extendsRepeat(generated, std::span<const std::uint32_t>(dr.data(), acc + 1));
}
// engine.cpp:1468 nd0
std::uint32_t old_srv_nd0(std::uint32_t prev_nd, std::uint32_t cap) { return std::max(1u, std::min(prev_nd > 0 ? prev_nd : 3u, cap)); }
// engine.cpp:1491-1492 (alt_score of the one-slot MTP case)
float old_srv_altScore(const q35::DraftAccept& dacc, const q35::DraftTiming& tm, std::uint32_t nd0) {
    float alt_score = 0;
    if (auto t = tm.estimate(nd0)) alt_score = dacc.expected(nd0) / *t;
    return alt_score;
}
// decode.cpp:159-160
float old_cli_mtpScore(const q35::DraftAccept& acc_model, const q35::DraftTiming& timing, std::uint32_t nd_host) {
    const auto t = timing.estimate(nd_host);
    return t ? acc_model.expected(nd_host) / *t : 0.0f;
}
// engine.cpp:1464-1466
std::uint32_t old_srv_ngLim(std::uint32_t lit_cap, std::uint32_t vrows, std::uint32_t Au, std::uint32_t ngram_max, bool fused) {
    std::uint32_t lim = std::min({q35::max_ng_drafts, lit_cap / Au, vrows / Au > 0 ? vrows / Au - 1 : 0u});
    if (ngram_max > 0) lim = std::min(lim, ngram_max);
    if (!fused) lim = std::min(lim, 1u);
    return lim;
}
// decode.cpp:92-94 (before the snapshot-set step)
std::uint32_t old_cli_ngMax(std::uint32_t ngram_max, bool fused) {
    std::uint32_t ng_max = ngram_max > 0 ? std::min(ngram_max, q35::max_ng_drafts) : q35::max_ng_drafts;
    if (!fused) ng_max = std::min<std::uint32_t>(ng_max, 1);
    return ng_max;
}
// engine.cpp:1401 (a2 of verifyRoom)
std::uint64_t old_srv_a2(std::uint32_t pos, std::uint32_t ctx) { return ctx > pos + 2ull ? ctx - (pos + 2ull) : 0; }
// decode.cpp:152 (the ctx part of room, ng_max = lim)
std::size_t old_cli_room(std::uint32_t lim, std::uint32_t p, std::uint32_t max_ctx) {
    return std::min<std::size_t>(lim, max_ctx > p + 2 ? max_ctx - (p + 2) : 0);
}
// decode.cpp:75
std::uint32_t old_cli_nDraft(std::uint32_t drafts, bool fused) { return !fused ? 1 : std::min(drafts, q35::max_drafts); }
// decode.cpp:138 / engine.cpp:1432 (the context half)
bool old_cli_loop(std::uint32_t p, std::uint32_t n_draft, std::uint32_t max_ctx) { return p + n_draft + 1 < max_ctx; }
bool old_srv_room(std::uint32_t pos, std::uint32_t nd_room, std::uint32_t ctx) { return pos + nd_room + 1 < ctx; }
// decode.cpp:203 (relaxacc branch)
float old_cli_synth(float slope, std::uint32_t nd_host, bool ng) {
    return 100.0f * (1.0f + slope * static_cast<float>(nd_host) + (ng ? 0.0f : 0.03f * static_cast<float>(nd_host)));
}
// engine.cpp:1710-1716 synthMs (nd_of: the slots' draft counts)
float old_srv_synth(std::span<const std::uint32_t> nd_of, float slope, std::uint32_t steps) {
    float rows = 0;
    for (std::uint32_t d : nd_of) rows += static_cast<float>(d + 1);
    const float a = static_cast<float>(nd_of.size());
    return 100.0f * (a + slope * (rows - a) + 0.03f * static_cast<float>(steps));
}

// ---- the CLI's rule calls (what src/cli/decode.cpp does after the change) ------------------------------------------

// D1: hist = prompt + first + every token pushed after it (decode.cpp `hist`); the trunk holds hist[0..n-1), next = hist.back()
bool cliRuleRepeats(std::span<const std::uint32_t> prompt, std::span<const std::uint32_t> generated, std::span<const std::uint32_t> dr,
                    std::uint32_t acc) {
    std::vector<std::uint32_t> hist(prompt.begin(), prompt.end());
    hist.insert(hist.end(), generated.begin(), generated.end());
    return sr::relaxRepeats(std::span<const std::uint32_t>(hist.data(), hist.size() - 1), hist.back(), dr, acc);
}
// D2: the draft count an n-gram proposal is scored against. prev_auto = the last automatic pick (mtp_auto only),
// this_cycle = the draft count the CLI computed for this cycle before the n-gram search (pre-change rule), cap = n_draft
std::uint32_t cliRuleAltDrafts(std::uint32_t prev_auto, std::uint32_t /*this_cycle*/, std::uint32_t cap) {
    return sr::ngramAltDrafts(prev_auto, cap);
}

// ---- tests -----------------------------------------------------------------------------------------------------------

std::vector<std::uint32_t> randToks(std::mt19937& g, std::size_t n, std::uint32_t vocab) {
    std::vector<std::uint32_t> v(n);
    for (auto& x : v) x = g() % vocab;
    return v;
}

// D1 server side: the shared guard == Engine::relaxRepeats over random histories (small vocab: repeats happen)
void testRulesD1Server() {
    std::mt19937 g(1201);
    std::uint32_t hits = 0;
    for (int it = 0; it < 4000; ++it) {
        const auto hist = randToks(g, g() % 5000, 3 + g() % 6);
        const auto dr = randToks(g, q35::max_ng_drafts, 3 + g() % 6);
        const std::uint32_t next = g() % 8;
        const std::uint32_t acc = g() % q35::max_drafts;
        const bool want = old_srv_relaxRepeats(hist, next, dr, acc);
        hits += want ? 1 : 0;
        CHECK_EQ(sr::relaxRepeats(hist, next, dr, acc), want);
    }
    CHECK(hits > 100);
    CHECK(hits < 3900);
}

// D1 CLI side, server expectation: a draft that repeats a 6-gram of the PROMPT is not kept by the relaxed rule
// (the pre-change CLI looked at the generated tokens only and kept it).
void testRulesD1Cli() {
    const std::vector<std::uint32_t> prompt = {500, 11, 12, 13, 14, 15, 16, 501, 502};
    const std::vector<std::uint32_t> generated = {700, 11, 12, 13};  // `next` = 13
    const std::array<std::uint32_t, 4> dr = {14, 15, 16, 900};       // next + dr[0..2] = 11 12 13 14 15 16 again
    CHECK(old_srv_relaxRepeats(std::span<const std::uint32_t>(std::vector<std::uint32_t>{500, 11, 12, 13, 14, 15, 16, 501, 502, 700, 11, 12}), 13, dr, 2));
    CHECK(!old_cli_relaxRepeats(generated, dr, 2));  // the difference D1 (documents the pre-change CLI)
    CHECK(cliRuleRepeats(prompt, generated, dr, 2));  // server expectation
    CHECK(!cliRuleRepeats(prompt, generated, dr, 1)); // 12 13 14 15 16 is only a 5-gram
    // random: the CLI call == the server guard on prompt + trunk
    std::mt19937 g(1151);
    for (int it = 0; it < 2000; ++it) {
        const auto pr = randToks(g, g() % 3000, 4);
        const auto gen = randToks(g, 1 + g() % 3000, 4);
        const auto d = randToks(g, q35::max_ng_drafts, 4);
        const std::uint32_t a = g() % q35::max_drafts;
        std::vector<std::uint32_t> trunk(pr);
        trunk.insert(trunk.end(), gen.begin(), gen.end() - 1);
        CHECK_EQ(cliRuleRepeats(pr, gen, d, a), old_srv_relaxRepeats(trunk, gen.back(), d, a));
    }
}

// D2 server side: nd0 and the MTP score unchanged
void testRulesD2Server() {
    for (std::uint32_t prev = 0; prev <= 16; ++prev)
        for (std::uint32_t cap = 0; cap <= 16; ++cap) CHECK_EQ(sr::ngramAltDrafts(prev, cap), old_srv_nd0(prev, cap));
    q35::DraftAccept acc;
    q35::DraftTiming tm;
    for (std::uint32_t nd = 0; nd <= q35::max_drafts; ++nd) CHECK_EQ(std::bit_cast<std::uint32_t>(sr::mtpScore(acc, tm, nd)), std::bit_cast<std::uint32_t>(old_srv_altScore(acc, tm, nd)));
    std::mt19937 g(7);
    for (int it = 0; it < 500; ++it) {
        const std::uint32_t nd = 1 + g() % q35::max_drafts;
        acc.update(nd, g() % (nd + 1));
        tm.update(nd, 20.0f + static_cast<float>(g() % 1000) / 10.0f);
        for (std::uint32_t k = 0; k <= q35::max_drafts; ++k) {
            const float a = sr::mtpScore(acc, tm, k);
            CHECK_EQ(std::bit_cast<std::uint32_t>(a), std::bit_cast<std::uint32_t>(old_srv_altScore(acc, tm, k)));
            CHECK_EQ(std::bit_cast<std::uint32_t>(a), std::bit_cast<std::uint32_t>(old_cli_mtpScore(acc, tm, k)));  // same formula
        }
    }
}

// D2 CLI side, server expectation: the n-gram proposal is scored against the previous automatic pick (none yet: 3,
// non-auto: always 3) within the cap, not against this cycle's freshly computed draft count.
void testRulesD2Cli() {
    CHECK_EQ(cliRuleAltDrafts(5, 7, 8), 5u);   // mtp_auto: last pick 5, this cycle's pick would be 7
    CHECK_EQ(cliRuleAltDrafts(0, 4, 8), 3u);   // first cycle / non-auto (prev stays 0): 3, not n_draft 4
    CHECK_EQ(cliRuleAltDrafts(0, 2, 2), 2u);   // 3 capped by n_draft 2
    CHECK_EQ(cliRuleAltDrafts(9, 9, 6), 6u);   // capped
    CHECK_EQ(cliRuleAltDrafts(0, 1, 1), 1u);   // unfused: cap 1
    for (std::uint32_t prev = 0; prev <= q35::max_drafts; ++prev)
        for (std::uint32_t cap = 1; cap <= q35::max_drafts; ++cap) CHECK_EQ(cliRuleAltDrafts(prev, cap, cap), old_srv_nd0(prev, cap));
}

// D3: the shared n-gram cap == the server's; for one sequence (lit_cap = max_ng_drafts before the snapshot step,
// max_small_batch rows) == the CLI's ng_max; the context room == both old expressions
void testRulesD3() {
    for (std::uint32_t lit = 0; lit <= 16; ++lit)
        for (std::uint32_t vrows : {0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u})
            for (std::uint32_t A = 1; A <= 16; ++A)
                for (std::uint32_t ngmax : {0u, 1u, 2u, 7u, 15u, 16u, 40u})
                    for (bool fused : {false, true}) CHECK_EQ(sr::ngramDraftCap(lit, vrows, A, ngmax, fused), old_srv_ngLim(lit, vrows, A, ngmax, fused));
    for (std::uint32_t ngmax = 0; ngmax <= 40; ++ngmax)
        for (bool fused : {false, true})
            CHECK_EQ(sr::ngramDraftCap(q35::max_ng_drafts, q35::max_small_batch, 1, ngmax, fused), old_cli_ngMax(ngmax, fused));
    for (std::uint32_t ctx : {0u, 1u, 2u, 3u, 100u, 8192u, 262144u})
        for (std::uint32_t pos : {0u, 1u, 2u, 50u, 97u, 98u, 99u, 100u, 101u, 8189u, 8190u, 8191u, 262142u})
            for (std::uint32_t lim : {0u, 1u, 15u}) {
                CHECK_EQ(sr::ctxRoom(pos, ctx), old_srv_a2(pos, ctx));
                CHECK_EQ(std::min<std::size_t>(lim, sr::ctxRoom(pos, ctx)), old_cli_room(lim, pos, ctx));
            }
}

// D4: pickDrafts takes the caller's cycle counter (it changes the pick: the 32-cycle exploration); the server's
// non-relaxacc counter n_cycles_ runs on across requests (pinned through the trace lines: the second request's first
// cycle number follows the first request's last), the CLI's starts at 0 every call. Neither changes here.
void testRulesD4() {
    q35::DraftAccept acc;
    q35::DraftTiming tm;
    for (std::uint32_t nd = 1; nd <= 8; ++nd) tm.update(nd, 30.0f + 3.0f * static_cast<float>(nd));
    for (int i = 0; i < 50; ++i) acc.update(4, 4);
    const q35::DraftAccept* accs[1] = {&acc};
    std::set<std::uint32_t> picks;
    for (std::uint32_t c = 0; c < 128; ++c) picks.insert(q35::pickDrafts(accs, tm, 8, c, 4));
    CHECK(picks.size() >= 2);  // the counter matters
#ifndef WHIRL_ISO_NO_SERVER  // arch025-iso4: tests/iso/spec.rules has no Engine / TestServer; the server half runs in whirl-server-tests
    const auto cycleNums = [] {
        std::vector<std::uint32_t> v;
        std::lock_guard<std::mutex> lk(g_log.mu);
        for (const auto& l : g_log.lines) {
            const std::size_t a = l.find("trace | cycle ");
            if (a != std::string::npos) v.push_back(static_cast<std::uint32_t>(std::strtoul(l.c_str() + a + 14, nullptr, 10)));
        }
        return v;
    };
    ServerSetup su = baseSetup(true);
    su.mc.parallel = 1;
    su.eo.mtp_auto = true;
    su.eo.ngram = false;
    su.eo.trace_nd = true;
    TestServer srv(su);
    g_log.clear();
    const auto p1 = makePrompt(300, 61), p2 = makePrompt(300, 62);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p1, 64))), decode(srv.model().greedyReference(p1, 64, true)));
    const auto c1 = cycleNums();
    g_log.clear();
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p2, 64))), decode(srv.model().greedyReference(p2, 64, true)));
    const auto c2 = cycleNums();
    CHECK(!c1.empty());
    CHECK(!c2.empty());
    if (!c1.empty() && !c2.empty()) {
        CHECK_EQ(*std::min_element(c1.begin(), c1.end()), 0u);
        CHECK_EQ(*std::min_element(c2.begin(), c2.end()), *std::max_element(c1.begin(), c1.end()) + 1);
    }
#endif
}

// D8: the context stop, each loop with its own room (CLI: n_draft after the cap; server: draftRoom() = n_draft)
void testRulesD8() {
    for (std::uint32_t ctx : {0u, 1u, 2u, 16u, 4096u, 8192u})
        for (std::uint32_t pos = 0; pos <= 8200; pos += (pos < 30 || (pos > 4070 && pos < 4100) || pos > 8170) ? 1 : 97)
            for (std::uint32_t room = 0; room <= 16; ++room) {
                CHECK_EQ(sr::decodeRoom(pos, room, ctx), old_cli_loop(pos, room, ctx));
                CHECK_EQ(sr::decodeRoom(pos, room, ctx), old_srv_room(pos, room, ctx));
            }
}

// D14 (required equivalence): the CLI's per-cycle MTP draft cap == the server's draft cap for one slot
void testRulesD14() {
    for (std::uint32_t drafts = 1; drafts <= 64; ++drafts)
        for (bool fused : {false, true}) {
            CHECK_EQ(sr::soloDraftCap(drafts, fused), old_cli_nDraft(drafts, fused));
            CHECK_EQ(sr::soloDraftCap(drafts, fused), whirl::server::slotDraftCap(q35::max_drafts, drafts, fused, false, 1, q35::max_small_batch));
        }
}

// D6: relaxacc's synthetic cycle cost: bit-identical to the CLI's one-sequence formula and the server's synthMs
void testRulesD6() {
    std::mt19937 g(42);
    for (float slope : {0.0f, 0.015f, 0.12f, 0.2f, 1.0f}) {
        for (std::uint32_t nd = 0; nd <= q35::max_ng_drafts; ++nd)
            for (bool ng : {false, true}) {
                const std::uint32_t one[1] = {nd};
                const float s = sr::synthCycleMs(1.0f, static_cast<float>(nd + 1), slope, ng ? 0 : nd);
                CHECK_EQ(std::bit_cast<std::uint32_t>(s), std::bit_cast<std::uint32_t>(old_cli_synth(slope, nd, ng)));
                CHECK_EQ(std::bit_cast<std::uint32_t>(s), std::bit_cast<std::uint32_t>(old_srv_synth(one, slope, ng ? 0 : nd)));
            }
        for (int it = 0; it < 300; ++it) {
            std::vector<std::uint32_t> nd_of(1 + g() % 16);
            for (auto& x : nd_of) x = g() % 11;
            const std::uint32_t steps = g() % 11;
            float rows = 0;
            for (std::uint32_t d : nd_of) rows += static_cast<float>(d + 1);
            CHECK_EQ(std::bit_cast<std::uint32_t>(sr::synthCycleMs(static_cast<float>(nd_of.size()), rows, slope, steps)),
                     std::bit_cast<std::uint32_t>(old_srv_synth(nd_of, slope, steps)));
        }
    }
}

// ---- arch025-lane1: the last two rules both loops still wrote out (verbatim pre-change code, 3eef475) --------------

// decode.cpp:167-169 (WHIRL_MTP_ADAPT, one sequence)
std::uint32_t old_cli_adapt(float acc_ema, std::uint32_t adapt, std::uint32_t n_draft) {
    const std::uint32_t want = static_cast<std::uint32_t>(std::ceil(acc_ema + static_cast<float>(adapt)));
    return std::clamp<std::uint32_t>(want, 1, n_draft);
}
// engine.cpp:1345-1350 Engine::pickDrafts (mtp_adapt branch; nd = draftCap(A))
std::uint32_t old_srv_adapt(std::span<const float> emas, std::uint32_t adapt, std::uint32_t nd) {
    std::uint32_t want = 1;
    for (float e : emas) want = std::max(want, static_cast<std::uint32_t>(std::ceil(e + static_cast<float>(adapt))));
    return std::min(nd, want);
}
// decode.cpp:249 / engine.cpp:1804 (acc_ema after a cycle with acc accepted drafts)
float old_ema(float ema, std::uint32_t acc) { return 0.7f * ema + 0.3f * static_cast<float>(acc); }

// D16: WHIRL_MTP_ADAPT's draft count: one rule (the largest want over the slots, at least 1, within the cap); for one
// sequence == the CLI's clamp
void testRulesD16() {
    std::mt19937 g(1606);
    for (std::uint32_t adapt = 1; adapt <= 4; ++adapt)
        for (std::uint32_t cap = 1; cap <= q35::max_drafts; ++cap)
            for (float e = 0.0f; e <= 16.0f; e += 0.37f) {
                const float one[1] = {e};
                CHECK_EQ(sr::adaptDrafts(one, adapt, cap), old_cli_adapt(e, adapt, cap));
                CHECK_EQ(sr::adaptDrafts(one, adapt, cap), old_srv_adapt(one, adapt, cap));
            }
    for (int it = 0; it < 2000; ++it) {
        std::vector<float> emas(1 + g() % 16);
        for (auto& x : emas) x = static_cast<float>(g() % 1700) / 100.0f;
        const std::uint32_t adapt = 1 + g() % 4, cap = g() % (q35::max_drafts + 1);
        CHECK_EQ(sr::adaptDrafts(emas, adapt, cap), old_srv_adapt(emas, adapt, cap));
    }
}

// D17: the acceptance EMA both loops keep (bit-identical)
void testRulesD17() {
    std::mt19937 g(1707);
    for (int it = 0; it < 5000; ++it) {
        const float e = static_cast<float>(g() % 1700000) / 100000.0f;
        const std::uint32_t acc = g() % (q35::max_ng_drafts + 1);
        CHECK_EQ(std::bit_cast<std::uint32_t>(sr::accEma(e, acc)), std::bit_cast<std::uint32_t>(old_ema(e, acc)));
    }
}

}  // namespace r12b
