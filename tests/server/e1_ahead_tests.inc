// v0252-e1p1c (PLAN V1-93, E1 step P1c): the one-cycle lookahead of the decode cycle (WHIRL_E1=2, EngineOptions::e1 == 2;
// design progress/v0252-e1study.HANDOFF.md sections 3 / 4 / 8 and progress/v0252-e1p1.PRIOR_ART.md "P1c"). Cycle n + 1 of one
// greedy MTP request is submitted before the host has read cycle n (SpecCycle::aheadCycle); the host reads cycle n from a ring
// of two pinned control-word areas the accept kernel writes (SampleBuffers::aheadSubmit / aheadWait). Included by
// server_tests.cpp (after its helpers). The mock model runs every call at once, so here cycle n + 1 has ALWAYS finished when
// the host reads cycle n: a host that read the wrong ring entry, the device's control area or a host copy of a position would
// see cycle n + 1's words and change the text; the mock counts a device position above the host's bound (E1Counts::disagree).
//   e1_ahead_equal     WHIRL_E1=2 vs 0, fixed draft counts (Ornith-like 1, Swift-like 4), WHIRL_SPEC_ACC host / gpu-zc /
//                      check: per request the same text (= the greedy reference), completion tokens, drafted and accepted
//                      counts; the lookahead ran (ring cycles, cycles submitted ahead = the mock's lookahead verify segments),
//                      WHIRL_SPEC_ACC host was raised to gpu-zc; verify calls = the synchronous arm's + the discarded cycles;
//                      a request that ends by max_tokens never discards a cycle (the budget drain rule)
//   e1_ahead_end       the end of a request inside the lookahead window - end-of-turn token, stop string, client gone - then a
//                      request that reuses the prefix: same text as the synchronous arm and the reference, the gen-end
//                      checkpoint is reused to the same position (cached tokens), no state inconsistency (poison, hidden rows)
//   e1_ahead_order     E1 rules R-a .. R-e read off the Backend's call log: per lookahead cycle exactly one eventRecord and
//                      one eventSync, the record of cycle n + 1 before the wait for cycle n, no event query, no download, no
//                      stream sync, no readCtl / readCtlPinned in the steady state
//   e1_ahead_drain     the drain rules: max_tokens within two cycles' output (never ahead, never a discard), two concurrent
//                      requests, n-gram cycles, a sampled request, the end of the context (position upper bound)
//   e1_ahead_fallback  device and host disagree: an out-of-range result (gpu-zc) and a check-mode disagreement on a word the
//                      next cycle did not consume fall back to the host (same text, lookahead goes on afterwards); a device
//                      decision that is in range is followed (gpu-zc: same text); a check-mode disagreement on a consumed word
//                      with the next cycle already submitted fails the request loudly (HTTP 500), the server stays up
//   e1_ahead_env       WHIRL_E1=0|1|2 in the env table, docs/usage.md and the parser
// SPDX-License-Identifier: Apache-2.0

namespace e1ahead {

using e1test::ReqOut;
using e1test::reqOut;

struct Arm {
    std::vector<ReqOut> out;
    std::uint64_t verifies = 0, rows = 0;
    Engine::Stats es{};
    MockModel::E1Counts e1{};
};

ServerSetup setup(std::uint32_t e1, std::uint32_t nd, std::uint32_t spec_acc = 0) {
    ServerSetup su = baseSetup();
    su.mc.parallel = 1;
    su.mc.pool_tokens = 131072;
    su.eo.mtp_auto = false;
    su.eo.mtp_adapt = 0;
    su.eo.n_draft = nd;
    su.eo.spec_acc = spec_acc;
    su.eo.e1 = e1;
    su.eo.ngram = false;  // (the n-gram cases turn it on)
    return su;
}

void printAhead(const std::string& what, const Engine::Stats& es, const MockModel::E1Counts& e) {
    std::printf("  %s: ring cycles %llu, submitted ahead %llu, discarded %llu; mock lookahead verify segs %llu, disagree %llu\n", what.c_str(),
                static_cast<unsigned long long>(es.acc_ring), static_cast<unsigned long long>(es.acc_ahead),
                static_cast<unsigned long long>(es.acc_discard), static_cast<unsigned long long>(e.ahead_segs),
                static_cast<unsigned long long>(e.disagree));
}

std::string stopBody(std::span<const std::uint32_t> prompt, std::uint32_t max_tokens, const std::string& stop) {
    std::string js;
    appendJsonStr(js, stop);
    return std::format("{{\"prompt\":{},\"max_tokens\":{},\"temperature\":0,\"ignore_eos\":true,\"stop\":[{}]}}", idsJson(prompt), max_tokens, js);
}

}  // namespace e1ahead

void testE1AheadEqual() {
    struct Case {
        const char* name;
        std::uint32_t nd;
        double eos_rate;
        std::uint32_t spec_acc;
    };
    const Case cases[] = {{"ornith-like nd 1", 1, 0.0, 0}, {"swift-like nd 4", 4, 0.0, 0}, {"eos nd 3", 3, 0.04, 0},
                          {"gpu-zc nd 2", 2, 0.02, 3}, {"check nd 4", 4, 0.02, 4}};
    for (const Case& c : cases) {
        e1ahead::Arm arms[2];
        std::vector<std::int64_t> ref_n;  // the greedy reference's token count per request (0: its first token ends the turn)
        for (std::uint32_t arm = 0; arm < 2; ++arm) {
            const std::uint32_t e1 = arm == 0 ? 0u : 2u;
            ServerSetup su = e1ahead::setup(e1, c.nd, c.spec_acc);
            su.mc.eos_rate = c.eos_rate;
            g_log.clear();
            TestServer srv(su);
            e1ahead::Arm& a = arms[arm];
            const std::vector<std::uint32_t> p0 = makePrompt(300, 901), p1 = makePrompt(420, 902), p2 = makePrompt(260, 903);
            const std::pair<const std::vector<std::uint32_t>*, std::uint32_t> reqs[] = {{&p0, 120}, {&p1, 90}, {&p2, 9}, {&p0, 120}, {&p2, 13}, {&p1, 200}};
            const bool ign = c.eos_rate == 0.0;
            for (const auto& [p, mt] : reqs) {
                const HttpResult r = srv.post("/v1/completions", e1test::greedyBody(*p, mt, ign));
                CHECK_EQ(r.status, 200);
                const e1ahead::ReqOut o = e1ahead::reqOut(r);
                const std::vector<std::uint32_t> ref = srv.model().greedyReference(*p, mt, ign);
                const std::string want = decode(ref);
                if (o.text != want) std::printf("  e1_ahead case %s E1=%u: a reply differs from the greedy reference\n", c.name, e1);
                CHECK_EQ(o.text, want);
                if (arm == 0) ref_n.push_back(static_cast<std::int64_t>(ref.size()));
                a.out.push_back(o);
            }
            a.verifies = srv.model().verifyCalls();
            a.rows = srv.model().rowsRun();
            a.e1 = srv.model().e1Counts();
            a.es = srv.engine().stats();
            CHECK_EQ(srv.model().poison(), 0u);
            CHECK_EQ(srv.model().hidMismatch(), 0u);
            CHECK_EQ(a.es.acc_fallback, 0u);
            CHECK_EQ(a.es.acc_mismatch, 0u);
            CHECK_EQ(a.es.tab_mismatch, 0u);
            CHECK_EQ(a.e1.disagree, 0u);
            e1ahead::printAhead(std::string(c.name) + (e1 ? " E1=2" : " E1=0"), a.es, a.e1);
            if (e1 == 0) {
                CHECK(a.es.acc_ring == 0 && a.es.acc_ahead == 0 && a.es.acc_discard == 0 && a.e1.ahead_segs == 0);
                CHECK_EQ(g_log.count("decode: WHIRL_E1="), 0u);
            } else {
                CHECK_EQ(g_log.count("decode: WHIRL_E1=2"), 1u);
                CHECK_EQ(g_log.count("decode: WHIRL_E1=1"), 0u);
                if (c.spec_acc != 4) CHECK_EQ(g_log.count("decode: WHIRL_SPEC_ACC=3 (gpu-zc)"), 1u);
                else CHECK_EQ(g_log.count("decode: WHIRL_SPEC_ACC=4 (check)"), 1u);
                CHECK(a.es.acc_ring > 0);
                CHECK(a.es.acc_ahead > 0);
                // a pipeline starts with a cycle the host knows exactly: cycles read through the ring = pipeline starts
                // + cycles submitted ahead - submitted cycles dropped at the end of a request (SpecCycle::aheadCycle
                // counts ahead at the submit, ring at the read, discard at the drop), so the starts are ring + discard -
                // ahead. (Until v0252-e1p1c-fix this line read ring > ahead, which is false when every pipeline ends
                // inside the window: "eos nd 3" has 4 starts and 4 discards, ring 84 = ahead 84.)
                CHECK(a.es.acc_ring + a.es.acc_discard > a.es.acc_ahead);
                if (ign) CHECK(a.es.acc_ring > a.es.acc_ahead);  // no discard: one ring cycle more per pipeline
                CHECK_EQ(a.e1.ahead_segs, a.es.acc_ahead);    // the model saw exactly those cycles with a position bound
                CHECK(a.e1.verify_segs >= a.es.acc_ring);     // every ring cycle took its inputs from the device
                CHECK(a.e1.npend_dev > 0 && a.e1.pos_dev > 0 && a.e1.padded_rows > 0);
                CHECK(g_log.count("; lookahead: ring cycles ") >= 1);  // the per-request stats line carries the counts
                if (ign) CHECK_EQ(a.es.acc_discard, 0u);      // max_tokens ends: the budget rule drained first
                else CHECK(a.es.acc_discard <= std::size(reqs));
                if (c.spec_acc == 4) CHECK(a.es.acc_checked > 0);
            }
        }
        const bool same = arms[0].out == arms[1].out;
        if (!same) std::printf("  e1_ahead case %s: text / tokens / drafted / accepted differ between E1=0 and E1=2\n", c.name);
        CHECK(same);
        // every count was reported (reqOut leaves -1 for a missing field) and the reply has exactly the greedy reference's
        // tokens. (Until v0252-e1p1c-fix this line demanded tokens > 0: wrong for a request whose first token ends the
        // turn - "eos nd 3", prompt p2 (seed 903, eos_rate 0.04): the reference is empty and both arms answer 0 tokens,
        // 0 drafted, 0 accepted, "" - the WHIRL_E1=0 arm too, so the value does not come from the lookahead.)
        CHECK_EQ(arms[1].out.size(), ref_n.size());
        std::size_t replies = 0;
        for (std::size_t i = 0; i < arms[1].out.size() && i < ref_n.size(); ++i) {
            const e1ahead::ReqOut& o = arms[1].out[i];
            const e1ahead::ReqOut& o0 = arms[0].out[i];
            const bool ok = o.tokens == ref_n[i] && o.drafted >= 0 && o.accepted >= 0 && o.accepted <= o.drafted;
            if (!ok)
                std::printf("  e1_ahead case %s request %zu: reference %lld tokens | E1=2 tokens %lld drafted %lld accepted %lld text %zu bytes | E1=0 tokens %lld drafted %lld accepted %lld text %zu bytes\n",
                            c.name, i, static_cast<long long>(ref_n[i]), static_cast<long long>(o.tokens), static_cast<long long>(o.drafted), static_cast<long long>(o.accepted),
                            o.text.size(), static_cast<long long>(o0.tokens), static_cast<long long>(o0.drafted), static_cast<long long>(o0.accepted), o0.text.size());
            CHECK(ok);
            if (o.tokens > 0) ++replies;
        }
        CHECK(replies >= 4);  // (an empty reply is the exception: at most the two short p2 requests of a case)
        // the same verify cycles, plus one per lookahead cycle dropped at the end of a request (its nd + 1 rows)
        CHECK_EQ(arms[1].verifies, arms[0].verifies + arms[1].es.acc_discard);
        CHECK_EQ(arms[1].rows, arms[0].rows + arms[1].es.acc_discard * (c.nd + 1));
    }
}

void testE1AheadEnd() {
    // ---- end-of-turn token and stop string: R1 ends inside the window, R2 extends R1's prompt + reply, R3 is exactly it
    std::uint64_t discards = 0;
    for (const bool by_stop : {false, true}) {
        for (std::uint32_t seed : {911u, 912u, 913u}) {
            struct Out {
                std::string t1, t2, t3, fr1;
                std::uint64_t n1 = 0, c2 = 0, c3 = 0;
            };
            Out o[2];
            for (std::uint32_t arm = 0; arm < 2; ++arm) {
                ServerSetup su = e1ahead::setup(arm == 0 ? 0u : 2u, 3);
                su.mc.eos_rate = by_stop ? 0.0 : 0.03;
                TestServer srv(su);
                const std::vector<std::uint32_t> p = makePrompt(300, seed);
                const std::vector<std::uint32_t> ref = srv.model().greedyReference(p, 200, by_stop);
                std::vector<std::uint32_t> kept = ref;  // the reply's tokens a follow-up prompt carries
                HttpResult r1;
                if (by_stop) {
                    const std::string full = decode(ref);
                    const std::string stop = full.substr(full.size() / 2, 5);
                    const std::size_t end = full.find(stop) + stop.size();
                    std::size_t want = 0;
                    while (want < ref.size() && decode(std::span<const std::uint32_t>(ref.data(), want)).size() < end) ++want;
                    kept.resize(want);
                    r1 = srv.post("/v1/completions", e1ahead::stopBody(p, 200, stop));
                    CHECK_EQ(usageCompletion(r1), static_cast<std::uint64_t>(want));
                } else {
                    r1 = srv.post("/v1/completions", e1test::greedyBody(p, 200, false));
                    CHECK_EQ(completionText(r1), decode(ref));
                }
                CHECK_EQ(r1.status, 200);
                o[arm].t1 = completionText(r1);
                o[arm].fr1 = finishReason(r1);
                o[arm].n1 = usageCompletion(r1);
                // R2: the prompt, the reply, eight more tokens (a chat follow-up); R3: exactly prompt + reply
                std::vector<std::uint32_t> p2 = p;
                p2.insert(p2.end(), kept.begin(), kept.end());
                const std::vector<std::uint32_t> p3 = p2;
                const std::vector<std::uint32_t> more = makePrompt(8, seed + 50);
                p2.insert(p2.end(), more.begin(), more.end());
                const HttpResult r2 = srv.post("/v1/completions", completionBody(p2, 40));
                CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(p2, 40, true)));
                o[arm].t2 = completionText(r2);
                o[arm].c2 = cachedTokens(r2);
                const HttpResult r3 = srv.post("/v1/completions", completionBody(p3, 24));
                CHECK_EQ(completionText(r3), decode(srv.model().greedyReference(p3, 24, true)));
                o[arm].t3 = completionText(r3);
                o[arm].c3 = cachedTokens(r3);
                CHECK_EQ(srv.model().poison(), 0u);
                CHECK_EQ(srv.model().hidMismatch(), 0u);
                CHECK_EQ(srv.model().e1Counts().disagree, 0u);
                const Engine::Stats es = srv.engine().stats();
                CHECK_EQ(es.acc_fallback, 0u);
                if (arm == 1) {
                    discards += es.acc_discard;
                    CHECK(es.acc_ring > 0);
                }
            }
            CHECK_EQ(o[0].t1, o[1].t1);
            CHECK_EQ(o[0].fr1, o[1].fr1);
            CHECK_EQ(o[0].n1, o[1].n1);
            CHECK_EQ(o[0].t2, o[1].t2);
            CHECK_EQ(o[0].t3, o[1].t3);
            // the follow-up restarts from the same place: the gen-end checkpoint of a request that ended inside the window
            // is as usable as the synchronous one (its hidden row is the kept copy of the request's last cycle)
            CHECK_EQ(o[0].c2, o[1].c2);
            if (!by_stop) CHECK(o[1].c2 >= 300u);
            // exactly prompt + reply: the lookahead's gen-end checkpoint has no logits row (the next cycle's verify
            // overwrote it), so this one request may restart from an earlier checkpoint - never from a later one
            CHECK(o[1].c3 <= o[0].c3);
        }
    }
    std::printf("  e1_ahead_end: lookahead cycles discarded at the end of a request: %llu\n", static_cast<unsigned long long>(discards));
    CHECK(discards >= 1);  // at least one request ended with the next cycle already submitted
    // ---- the client goes away while the request decodes (a lookahead cycle in flight), then the prompt comes again
    for (std::uint32_t arm = 0; arm < 2; ++arm) {
        ServerSetup su = e1ahead::setup(arm == 0 ? 0u : 2u, 2);
        su.mc.cycle_us = 1000;
        TestServer srv(su);
        g_log.clear();
        const std::vector<std::uint32_t> p = makePrompt(64, 921);
        const auto a = test::rawConnect("127.0.0.1", srv.port());
        test::rawSend(a, rawPost(srv.port(), completionBody(p, 6000, false)));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        test::rawClose(a);
        CHECK(waitIdle(srv, 5.0));
        CHECK(g_log.count("client disconnected") >= 1);
        // the slot is consistent: the same prompt again, and the prompt with the first 30 reply tokens
        const std::vector<std::uint32_t> ref = srv.model().greedyReference(p, 60, true);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 60))), decode(ref));
        std::vector<std::uint32_t> p2 = p;
        p2.insert(p2.end(), ref.begin(), ref.begin() + 30);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p2, 30))), decode(srv.model().greedyReference(p2, 30, true)));
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
        CHECK_EQ(srv.model().e1Counts().disagree, 0u);
        if (arm == 1) CHECK(srv.engine().stats().acc_ring > 0);
    }
}

void testE1AheadOrder() {
    using Call = NullBackend::Call;
    ServerSetup su = e1ahead::setup(2, 2);
    TestServer srv(su);
    const std::vector<std::uint32_t> p = makePrompt(300, 931);
    // warm: the first request allocates what a request allocates once
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 20))), decode(srv.model().greedyReference(p, 20, true)));
    const Engine::Stats es0 = srv.engine().stats();
    const std::uint64_t rc0 = srv.model().readCtlCalls(), rp0 = srv.model().readCtlPinnedCalls(), vf0 = srv.model().verifyCalls();
    srv.ops().recordThreads(true);
    const std::vector<std::uint32_t> q = makePrompt(280, 932);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(q, 160))), decode(srv.model().greedyReference(q, 160, true)));
    CHECK(waitIdle(srv, 5.0));
    const auto calls = srv.ops().callThreads();
    srv.ops().recordThreads(false);
    const Engine::Stats es = srv.engine().stats();
    const std::uint64_t ring = es.acc_ring - es0.acc_ring, ahead = es.acc_ahead - es0.acc_ahead;
    CHECK(ring > 8 && ahead > 6);
    CHECK_EQ(es.acc_discard, 0u);  // (max_tokens end)
    // the window of the lookahead: from its first eventRecord to its last eventSync
    std::size_t first = calls.size(), last = 0;
    for (std::size_t i = 0; i < calls.size(); ++i) {
        if (calls[i].first == Call::event_record && first == calls.size()) first = i;
        if (calls[i].first == Call::event_sync) last = i;
    }
    CHECK(first < last);
    std::uint64_t n_rec = 0, n_sync = 0, covered = 0, uncovered = 0, n_copy = 0, bad = 0, early = 0;
    for (std::size_t i = first; i <= last && i < calls.size(); ++i) {
        const Call c = calls[i].first;
        if (c == Call::event_record) n_rec += 1;
        else if (c == Call::copy_async) n_copy += 1;
        else if (c == Call::event_sync) {
            if (n_rec < n_sync + 1) early += 1;        // a wait for a cycle that was not submitted
            else if (n_rec >= n_sync + 2) covered += 1;  // R-d: the next cycle was submitted and recorded before this wait
            else uncovered += 1;
            n_sync += 1;
        } else if (c == Call::event_done || c == Call::event_done_checked || c == Call::download || c == Call::stream_sync) {
            bad += 1;  // R-a: no query; R-e: no blocking read, no stream sync inside the pipeline
        }
    }
    std::printf("  e1_ahead_order: ring %llu ahead %llu | records %llu syncs %llu covered %llu uncovered %llu copies %llu bad %llu\n",
                static_cast<unsigned long long>(ring), static_cast<unsigned long long>(ahead), static_cast<unsigned long long>(n_rec),
                static_cast<unsigned long long>(n_sync), static_cast<unsigned long long>(covered), static_cast<unsigned long long>(uncovered),
                static_cast<unsigned long long>(n_copy), static_cast<unsigned long long>(bad));
    CHECK_EQ(n_rec, ring);       // R-b: exactly one eventRecord per lookahead cycle
    CHECK_EQ(n_sync, ring);      // one host wait per cycle
    CHECK_EQ(early, 0u);
    CHECK_EQ(covered, ahead);    // R-d
    CHECK_EQ(uncovered, ring - ahead);
    CHECK(uncovered <= 2);       // one pipeline: only its last cycle is waited for with nothing behind it
    CHECK_EQ(bad, 0u);
    // per cycle two device copies (the hidden rows to the MTP input and to the ring entry's keep); the first cycle's
    // copies precede its record
    CHECK_EQ(n_copy, 2 * (n_rec - 1));
    // R-c / R-e: no D2H copy command for the control words at all (gpu-zc in the synchronous cycles too)
    CHECK_EQ(srv.model().readCtlCalls(), rc0);
    CHECK_EQ(srv.model().readCtlPinnedCalls(), rp0);
    CHECK(srv.model().verifyCalls() - vf0 >= ring);
}

void testE1AheadDrain() {
    // ---- max_tokens: within two cycles' largest output (2 * (nd + 1) = 8) nothing is ever submitted ahead; no request that
    // ends by max_tokens drops a cycle
    {
        ServerSetup su = e1ahead::setup(2, 3);
        TestServer srv(su);
        std::uint32_t seed = 940;
        for (std::uint32_t mt : {1u, 2u, 3u, 5u, 8u}) {
            const std::vector<std::uint32_t> p = makePrompt(200 + 7 * mt, seed++);
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, mt))), decode(srv.model().greedyReference(p, mt, true)));
        }
        Engine::Stats es = srv.engine().stats();
        CHECK_EQ(es.acc_ahead, 0u);
        CHECK_EQ(es.acc_discard, 0u);
        for (std::uint32_t mt : {9u, 12u, 17u, 33u, 64u}) {
            const std::vector<std::uint32_t> p = makePrompt(200 + 7 * mt, seed++);
            const HttpResult r = srv.post("/v1/completions", completionBody(p, mt));
            CHECK_EQ(completionText(r), decode(srv.model().greedyReference(p, mt, true)));
            CHECK_EQ(usageCompletion(r), static_cast<std::uint64_t>(mt));
        }
        es = srv.engine().stats();
        CHECK(es.acc_ahead > 0);
        CHECK_EQ(es.acc_discard, 0u);
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().e1Counts().disagree, 0u);
    }
    // ---- two concurrent requests, n-gram cycles (markov text), then a sampled request: same replies as E1=0
    struct Case {
        const char* name;
        std::uint32_t markov, n_req;
    };
    const Case cases[] = {{"two concurrent", 0, 2}, {"n-gram cycles", 40, 1}};
    for (const Case& c : cases) {
        std::string sampled[2];
        std::size_t no_ngram[2] = {0, 0};  // requests whose stats line says "n-gram drafts in 0 of N cycles"
        for (std::uint32_t arm = 0; arm < 2; ++arm) {
            ServerSetup su = baseSetup();
            su.mc.markov = c.markov;
            su.mc.pool_tokens = 131072;
            su.eo.ngram = c.markov > 0;
            su.eo.e1 = arm == 0 ? 0u : 2u;
            g_log.clear();
            TestServer srv(su);
            std::vector<std::vector<std::uint32_t>> prompts;
            for (std::uint32_t i = 0; i < c.n_req; ++i) prompts.push_back(makePrompt(200 + 150 * i, 950 + i));
            std::vector<std::string> got(c.n_req);
            std::vector<std::thread> th;
            for (std::uint32_t i = 0; i < c.n_req; ++i)
                th.emplace_back([&, i] { got[i] = completionText(srv.post("/v1/completions", completionBody(prompts[i], 120))); });
            for (auto& t : th) t.join();
            for (std::uint32_t i = 0; i < c.n_req; ++i) CHECK_EQ(got[i], decode(srv.model().greedyReference(prompts[i], 120, true)));
            // one more greedy request alone (the lookahead may run), then a sampled one (never ahead)
            const std::vector<std::uint32_t> solo = makePrompt(240, 960);
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(solo, 100))), decode(srv.model().greedyReference(solo, 100, true)));
            const Engine::Stats es1 = srv.engine().stats();
            const std::string sb = std::format(
                "{{\"prompt\":{},\"max_tokens\":40,\"temperature\":1.0,\"top_k\":8,\"seed\":11,\"ignore_eos\":true}}", idsJson(prompts[0]));
            sampled[arm] = completionText(srv.post("/v1/completions", sb));
            const Engine::Stats es2 = srv.engine().stats();
            CHECK_EQ(es2.acc_ring, es1.acc_ring);  // the sampled request decided every cycle on the host
            CHECK_EQ(es2.acc_ahead, es1.acc_ahead);
            const MockModel::E1Counts e = srv.model().e1Counts();
            e1ahead::printAhead(std::string(c.name) + (arm ? " E1=2" : " E1=0"), es2, e);
            CHECK_EQ(e.disagree, 0u);
            CHECK_EQ(srv.model().poison(), 0u);
            CHECK_EQ(srv.model().hidMismatch(), 0u);
            CHECK_EQ(es2.acc_fallback, 0u);
            if (arm == 0) CHECK(es2.acc_ring == 0 && e.ahead_segs == 0);
            no_ngram[arm] = g_log.count("n-gram drafts in 0 of");
        }
        CHECK_EQ(sampled[0], sampled[1]);
        // the n-gram drafter still gets its cycles with the lookahead on: no more requests without one than with E1=0
        if (c.markov > 0) CHECK(no_ngram[1] <= no_ngram[0]);
    }
    // ---- the end of the context: the host's position is an upper bound while a cycle is in flight; the request ends by
    // length at the same token as with E1=0, no ContextTooLong
    {
        std::string text[2], fr[2];
        std::uint64_t n[2] = {0, 0};
        for (std::uint32_t arm = 0; arm < 2; ++arm) {
            ServerSetup su = e1ahead::setup(arm == 0 ? 0u : 2u, 4);
            su.mc.slot_ctx = 640;
            TestServer srv(su);
            const std::vector<std::uint32_t> p = makePrompt(500, 970);
            const HttpResult r = srv.post("/v1/completions", completionBody(p, 400));
            CHECK_EQ(r.status, 200);
            text[arm] = completionText(r);
            fr[arm] = finishReason(r);
            n[arm] = usageCompletion(r);
            CHECK_EQ(text[arm], decode(srv.model().greedyReference(p, static_cast<std::uint32_t>(n[arm]), true)));
            CHECK_EQ(srv.model().poison(), 0u);
            CHECK_EQ(srv.model().e1Counts().disagree, 0u);
            if (arm == 1) CHECK(srv.engine().stats().acc_ahead > 0);
        }
        CHECK_EQ(text[0], text[1]);
        CHECK_EQ(fr[0], fr[1]);
        CHECK_EQ(fr[1], std::string("length"));
        CHECK_EQ(n[0], n[1]);
        CHECK(n[1] > 100 && n[1] < 400);
    }
}

void testE1AheadFallback() {
    const std::vector<std::uint32_t> p0 = makePrompt(300, 981), p1 = makePrompt(350, 982);
    // ---- a device result out of range (every 3rd accept writes ACC0 = nd + 5; ACC / CHOSEN / NPOS / NPEND stay right):
    // gpu-zc falls back to the host's decision, check mode reports the disagreement; the words the next cycle consumed
    // agree with the host, so the request goes on with the same text and the lookahead resumes
    for (const std::uint32_t mode : {0u, 4u}) {
        ServerSetup su = e1ahead::setup(2, 3, mode);
        su.mc.spec_acc_corrupt = 3;
        g_log.clear();
        TestServer srv(su);
        for (const auto* p : {&p0, &p1})
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(*p, 150))), decode(srv.model().greedyReference(*p, 150, true)));
        const Engine::Stats es = srv.engine().stats();
        e1ahead::printAhead(mode == 4 ? "out of range, check" : "out of range, gpu-zc", es, srv.model().e1Counts());
        CHECK(srv.model().specCorrupted() > 0);
        if (mode == 4) {
            CHECK(es.acc_mismatch > 0);
            CHECK_EQ(es.acc_fallback, 0u);
        } else {
            CHECK(es.acc_fallback > 0);
            CHECK_EQ(g_log.count("spec-acc fallback | slot"), 1u);
        }
        CHECK(es.acc_ring > 0 && es.acc_ahead > 0);  // the lookahead went on after the fallbacks
        CHECK_EQ(g_log.count("E1 lookahead | slot"), 0u);
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
        CHECK_EQ(srv.model().e1Counts().disagree, 0u);
    }
    // ---- a device decision that is in range but one draft shorter than the rule (every 5th accept): gpu-zc follows the
    // device (an exact prefix is exact: same text, the cycles go on)
    {
        ServerSetup su = e1ahead::setup(2, 4, 3);
        su.mc.spec_acc_short = 5;
        TestServer srv(su);
        for (const auto* p : {&p0, &p1})
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(*p, 150))), decode(srv.model().greedyReference(*p, 150, true)));
        const Engine::Stats es = srv.engine().stats();
        CHECK(srv.model().specShorted() > 0);
        CHECK_EQ(es.acc_fallback, 0u);
        CHECK(es.acc_ahead > 0);
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
    }
    // ---- the same device decision in check mode: the host's own decision differs on words the next cycle (already
    // submitted) consumed; the DeltaNet state was replayed with the device's count and cannot be taken back - the request
    // fails loudly, the server stays up
    {
        ServerSetup su = e1ahead::setup(2, 4, 4);
        su.mc.spec_acc_short = 5;
        g_log.clear();
        TestServer srv(su);
        const HttpResult r = srv.post("/v1/completions", completionBody(p0, 300));
        CHECK_EQ(r.status, 500);
        CHECK(r.body.find("E1Diverged") != std::string::npos);
        CHECK(g_log.count("E1 lookahead | slot") >= 1);
        const Engine::Stats es = srv.engine().stats();
        CHECK(es.acc_mismatch > 0);
        CHECK_EQ(srv.get("/health").status, 200);
        CHECK(waitIdle(srv, 5.0));
        CHECK_EQ(srv.model().poison(), 0u);
    }
}

void testE1AheadEnv() {
#ifdef WHIRL_SOURCE_DIR
    auto slurp = [](const char* rel) {
        std::ifstream in(std::string(WHIRL_SOURCE_DIR) + rel, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    const std::string rel = slurp("/src/release/release.cpp");
    const std::string doc = slurp("/docs/usage.md");
    const std::string sm = slurp("/src/server/server_main.cpp");
    CHECK(!rel.empty() && !doc.empty() && !sm.empty());
    CHECK(rel.find("\"E1=0|1|2\"") != std::string::npos);
    CHECK_EQ(rel.find("\"E1=0|1\""), std::string::npos);
    CHECK(doc.find("`WHIRL_E1=0\\|1\\|2`") != std::string::npos);
    CHECK(doc.find("one cycle ahead") != std::string::npos);
    CHECK(sm.find("is not 0, 1 or 2; using 0") != std::string::npos);
#else
    CHECK(false);  // whirl-server-tests is built with WHIRL_SOURCE_DIR (cmake/server.cmake): the source checks must run
#endif
    const std::pair<const char*, std::uint32_t> good[] = {{"0", 0}, {"1", 1}, {"2", 2}, {" 2 ", 2}, {"\t2", 2}};
    for (const auto& [s, v] : good) {
        const auto r = parseE1(s);
        CHECK(r.has_value() && *r == v);
    }
    for (const char* s : {"", "3", "-1", "on", "true", "02", "2x", "yes", "ahead"}) {
        if (parseE1(s).has_value()) std::printf("  parseE1(\"%s\") accepted\n", s);
        CHECK(!parseE1(s).has_value());
    }
    // a model without the E1 path: WHIRL_E1=2 is turned off at startup like 1, said once, no device accept forced
    {
        ServerSetup su = e1ahead::setup(2, 2);
        su.mc.replay = false;
        g_log.clear();
        TestServer srv(su);
        const std::vector<std::uint32_t> p = makePrompt(200, 991);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 50))), decode(srv.model().greedyReference(p, 50, true)));
        CHECK_EQ(g_log.count("WHIRL_E1=2: the model has no E1 path (DeltaNet snapshot mode); E1 off"), 1u);
        CHECK_EQ(g_log.count("decode: WHIRL_E1="), 0u);
        CHECK_EQ(srv.model().specAccepts(), 0u);
        CHECK_EQ(srv.engine().stats().acc_ring, 0u);
    }
}
