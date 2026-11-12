// v0252-e1p1b (PLAN V1-93, E1 step P1b): the synchronous decode cycle that takes the next cycle's inputs from the device
// (WHIRL_E1, EngineOptions::e1; design progress/v0252-e1study.HANDOFF.md sections 3 / 8, P1b row). Included by server_tests.cpp.
// The mock model reads every device input the way the model's kernels do (MockModel::e1Counts in mock_backend.h): VSeg::dev ->
// the CHOSEN / NPOS words, MSeg::e1_rows -> the padded step 0 from the argmax words and row ACC, setPending(npend_dev) -> the
// NPEND word, SpecAccArgs::pos < 0 -> the NPOS word; a stale or missing device word changes the text.
//   e1_inputs_sync      WHIRL_E1=1 vs 0 with a FIXED draft count (mtp_auto off, n_draft 1 = Ornith-like and 2..4 = Swift-like:
//                       the auto policy trains on wall-clock cycle times, src/spec/policy.h DraftTiming::update, so its
//                       counters move with timing), WHIRL_SPEC_ACC host / gpu-zc / check: per request the same text (= the
//                       greedy reference), completion tokens, drafted and accepted counts; in total the same verify calls and
//                       verify rows; E1=1 ran the device path (verify segments, padded MTP step-0 segments with draft-area
//                       rows, NPEND replays, NPOS accepts) with the host copy equal to the control words on every E1 cycle
//                       (disagree 0) and raised WHIRL_SPEC_ACC host to gpu; E1=0 ran none of it. Requests end by max_tokens
//                       inside a draft run and by end-of-turn drafts, and a repeated prompt restores the gen-end checkpoint an
//                       E1 request wrote (commitSeq with the NPEND word)
//   e1_inputs_fallback  E1=1 where the E1 path does not apply: snapshot mode (the Radeon 8060S / WHIRL_GDN_REPLAY=0) and a model
//                       without the E1 kernels turn E1 off at startup with one warning (no device accept forced); two
//                       concurrent requests, n-gram cycles and a sampled request keep the host inputs on those cycles - same
//                       text as the reference / as E1=0, disagree 0
//   e1_inputs_env       WHIRL_E1 in the env table and docs/usage.md, read through config.settings (settings::e1), parseE1
//                       (v0252-e1p1c: the values are 0|1|2; 2 = the lookahead, tests/server/e1_ahead_tests.inc)
// SPDX-License-Identifier: Apache-2.0

namespace e1test {

struct ReqOut {
    std::string text;
    std::int64_t tokens = -1, drafted = -1, accepted = -1;
    bool operator==(const ReqOut&) const = default;
};

ReqOut reqOut(const HttpResult& r) {
    ReqOut o;
    o.text = completionText(r);
    const json::Value v = parseBody(r);
    if (const json::Value* u = v.get("usage"))
        if (const json::Value* t = u->get("completion_tokens")) o.tokens = t->asInt();
    if (const json::Value* tm = v.get("timings")) {
        if (const json::Value* d = tm->get("draft_n")) o.drafted = d->asInt();
        if (const json::Value* a = tm->get("draft_n_accepted")) o.accepted = a->asInt();
    }
    return o;
}

std::string greedyBody(std::span<const std::uint32_t> prompt, std::uint32_t max_tokens, bool ignore_eos) {
    return std::format("{{\"prompt\":{},\"max_tokens\":{},\"temperature\":0,\"ignore_eos\":{}}}", idsJson(prompt), max_tokens,
                       ignore_eos ? "true" : "false");
}

void printCounts(const char* what, const MockModel::E1Counts& e) {
    std::printf("  %s: E1 verify segs %llu, MTP step-0 segs %llu (draft-area rows %llu), NPEND replays %llu, NPOS accepts %llu, "
                "disagree %llu\n",
                what, static_cast<unsigned long long>(e.verify_segs), static_cast<unsigned long long>(e.mtp_segs),
                static_cast<unsigned long long>(e.padded_rows), static_cast<unsigned long long>(e.npend_dev),
                static_cast<unsigned long long>(e.pos_dev), static_cast<unsigned long long>(e.disagree));
}

}  // namespace e1test

void testE1InputsSync() {
    struct Case {
        const char* name;
        std::uint32_t nd;        // fixed MTP drafts per cycle (MTP_DRAFTS)
        double eos_rate;         // end-of-turn drafts inside a draft run
        std::uint32_t spec_acc;  // WHIRL_SPEC_ACC of both arms
    };
    const Case cases[] = {{"ornith-like nd 1", 1, 0.0, 0}, {"swift-like nd 4", 4, 0.0, 0}, {"eos nd 3", 3, 0.04, 0},
                          {"gpu-zc nd 2", 2, 0.02, 3}, {"check nd 4", 4, 0.02, 4}};
    for (const Case& c : cases) {
        struct Arm {
            std::vector<e1test::ReqOut> out;
            std::uint64_t verifies = 0, rows = 0, accepts = 0;
            MockModel::E1Counts e1{};
        };
        Arm arms[2];
        for (std::uint32_t e1 : {0u, 1u}) {
            ServerSetup su = baseSetup();
            su.mc.parallel = 1;
            su.mc.eos_rate = c.eos_rate;
            su.mc.pool_tokens = 131072;
            su.eo.mtp_auto = false;
            su.eo.mtp_adapt = 0;
            su.eo.n_draft = c.nd;
            su.eo.spec_acc = c.spec_acc;
            su.eo.e1 = e1;
            g_log.clear();
            TestServer srv(su);
            Arm& a = arms[e1];
            // sequential requests: long runs, max_tokens inside a draft run, the first prompt again (gen-end checkpoint restore)
            const std::vector<std::uint32_t> p0 = makePrompt(300, 701), p1 = makePrompt(420, 702), p2 = makePrompt(260, 703);
            const std::pair<const std::vector<std::uint32_t>*, std::uint32_t> reqs[] = {{&p0, 120}, {&p1, 90}, {&p2, 9}, {&p0, 120}, {&p2, 13}};
            const bool ign = c.eos_rate == 0.0;
            for (const auto& [p, mt] : reqs) {
                const e1test::ReqOut o = e1test::reqOut(srv.post("/v1/completions", e1test::greedyBody(*p, mt, ign)));
                const std::string want = decode(srv.model().greedyReference(*p, mt, ign));
                if (o.text != want) std::printf("  e1_inputs case %s E1=%u: a reply differs from the greedy reference\n", c.name, e1);
                CHECK_EQ(o.text, want);
                a.out.push_back(o);
            }
            a.verifies = srv.model().verifyCalls();
            a.rows = srv.model().rowsRun();
            a.accepts = srv.model().specAccepts();
            a.e1 = srv.model().e1Counts();
            const Engine::Stats es = srv.engine().stats();
            CHECK_EQ(srv.model().poison(), 0u);
            CHECK_EQ(srv.model().hidMismatch(), 0u);
            CHECK_EQ(es.acc_fallback, 0u);
            CHECK_EQ(es.acc_mismatch, 0u);
            CHECK_EQ(es.tab_mismatch, 0u);
            e1test::printCounts((std::string(c.name) + (e1 ? " E1=1" : " E1=0")).c_str(), a.e1);
            if (e1 == 0) {
                CHECK(a.e1.verify_segs == 0 && a.e1.mtp_segs == 0 && a.e1.npend_dev == 0 && a.e1.pos_dev == 0);
                CHECK_EQ(g_log.count("decode: WHIRL_E1=1"), 0u);
                CHECK_EQ(es.acc_e1, 0u);
            } else {
                CHECK_EQ(es.acc_e1, a.e1.verify_segs);  // the engine's count (stats line "E1 device-input cycles") = the model's
                CHECK(a.e1.verify_segs > 0);
                CHECK(a.e1.mtp_segs > 0);
                CHECK(a.e1.padded_rows > 0);  // a padded step 0 had rows past the accepted count
                CHECK(a.e1.npend_dev > 0);
                CHECK(a.e1.pos_dev > 0);
                CHECK_EQ(a.e1.disagree, 0u);
                CHECK(a.accepts > 0);  // the device accept ran (WHIRL_SPEC_ACC host raised to gpu)
                CHECK_EQ(g_log.count("decode: WHIRL_E1=1"), 1u);
                if (c.spec_acc == 0) CHECK_EQ(g_log.count("decode: WHIRL_SPEC_ACC=2 (gpu)"), 1u);
                CHECK(g_log.count("; E1 device-input cycles ") >= 1);  // the per-request stats line carries the count
                // an E1 cycle follows the cycle that left the inputs: every request's first cycle takes host inputs
                CHECK(a.e1.verify_segs < a.verifies);
            }
        }
        const bool same = arms[0].out == arms[1].out;
        if (!same) std::printf("  e1_inputs case %s: text / tokens / drafted / accepted differ between E1=0 and E1=1\n", c.name);
        CHECK(same);
        for (const e1test::ReqOut& o : arms[1].out) CHECK(o.tokens > 0 && o.drafted >= 0 && o.accepted >= 0);
        CHECK_EQ(arms[0].verifies, arms[1].verifies);  // the same verify cycles
        CHECK_EQ(arms[0].rows, arms[1].rows);          // the same verify (and prefill) rows
    }
}

void testE1InputsFallback() {
    const std::vector<std::uint32_t> p = makePrompt(320, 801);
    // snapshot mode / no E1 kernels: E1 off at startup, said once; the accept mode is not raised
    for (const bool replay : {false, true}) {
        ServerSetup su = baseSetup();
        su.mc.replay = replay;
        su.mc.e1_inputs = !replay ? true : false;  // replay on: a model without the E1 kernels
        su.eo.e1 = 1;
        g_log.clear();
        TestServer srv(su);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 70))), decode(srv.model().greedyReference(p, 70, true)));
        const MockModel::E1Counts e = srv.model().e1Counts();
        CHECK(e.verify_segs == 0 && e.mtp_segs == 0 && e.npend_dev == 0 && e.pos_dev == 0);
        CHECK_EQ(srv.model().specAccepts(), 0u);
        CHECK_EQ(g_log.count(replay ? "WHIRL_E1=1: the model has no E1 path (no set_rows_dp / rmsnorm_q8_rows_dv / spec_accept_rows kernel); E1 off"
                                    : "WHIRL_E1=1: the model has no E1 path (DeltaNet snapshot mode); E1 off"),
                 1u);
        CHECK_EQ(g_log.count("decode: WHIRL_E1=1"), 0u);
        CHECK_EQ(srv.model().poison(), 0u);
    }
    // two concurrent requests (A = 2 cycles: host inputs), n-gram cycles (markov text) and a sampled request: same replies
    struct Case {
        const char* name;
        std::uint32_t markov;
        std::uint32_t n_req;
    };
    const Case cases[] = {{"two concurrent", 0, 2}, {"n-gram cycles", 40, 1}};
    for (const Case& c : cases) {
        std::vector<std::string> ref_sampled(2);
        for (std::uint32_t e1 : {0u, 1u}) {
            ServerSetup su = baseSetup();
            su.mc.markov = c.markov;
            su.mc.pool_tokens = 131072;
            su.eo.ngram = c.markov > 0;
            su.eo.e1 = e1;
            TestServer srv(su);
            std::vector<std::vector<std::uint32_t>> prompts;
            for (std::uint32_t i = 0; i < c.n_req; ++i) prompts.push_back(makePrompt(200 + 150 * i, 810 + i));
            std::vector<std::string> got(c.n_req);
            std::vector<std::thread> th;
            for (std::uint32_t i = 0; i < c.n_req; ++i)
                th.emplace_back([&, i] { got[i] = completionText(srv.post("/v1/completions", completionBody(prompts[i], 80))); });
            for (auto& t : th) t.join();
            for (std::uint32_t i = 0; i < c.n_req; ++i) CHECK_EQ(got[i], decode(srv.model().greedyReference(prompts[i], 80, true)));
            const std::string sb = std::format(
                "{{\"prompt\":{},\"max_tokens\":40,\"temperature\":1.0,\"top_k\":8,\"seed\":11,\"ignore_eos\":true}}", idsJson(prompts[0]));
            ref_sampled[e1] = completionText(srv.post("/v1/completions", sb));
            const MockModel::E1Counts e = srv.model().e1Counts();
            e1test::printCounts((std::string(c.name) + (e1 ? " E1=1" : " E1=0")).c_str(), e);
            CHECK_EQ(e.disagree, 0u);
            CHECK_EQ(srv.model().poison(), 0u);
            CHECK_EQ(srv.model().hidMismatch(), 0u);
            if (e1 == 0) CHECK(e.verify_segs == 0 && e.npend_dev == 0);
        }
        CHECK_EQ(ref_sampled[0], ref_sampled[1]);  // the sampled request decides on the host in both arms
    }
}

void testE1InputsEnv() {
#ifdef WHIRL_SOURCE_DIR
    auto slurp = [](const char* rel) {
        std::ifstream in(std::string(WHIRL_SOURCE_DIR) + rel, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    const std::string rel = slurp("/src/release/release.cpp");
    const std::string doc = slurp("/docs/usage.md");
    const std::string sm = slurp("/src/server/server_main.cpp");
    const std::string rs = slurp("/src/config/run_settings.h");
    CHECK(!rel.empty() && !doc.empty() && !sm.empty() && !rs.empty());
    CHECK(rel.find("\"E1=0|1|2\"") != std::string::npos);  // v0252-e1p1c: 2 = the one-cycle lookahead (e1_ahead_env)
    CHECK(doc.find("`WHIRL_E1=0\\|1\\|2`") != std::string::npos);
    // the one read point (config.settings) and the one parser (v0252-e1def: server_main reaches it through chooseE1,
    // engine_options.h, which also applies the device profile's default when the variable is not set)
    CHECK(rs.find("envGet(\"E1\")") != std::string::npos);
    CHECK(sm.find("settings::e1()") != std::string::npos && sm.find("chooseE1(") != std::string::npos);
    CHECK_EQ(sm.find("parseE1("), std::string::npos);
    CHECK(slurp("/src/server/engine_options.h").find("parseE1(*env)") != std::string::npos);
    CHECK_EQ(sm.find("\"E1\""), std::string::npos);
#else
    CHECK(false);  // whirl-server-tests is built with WHIRL_SOURCE_DIR (cmake/server.cmake): the source checks must run
#endif
    const std::pair<const char*, std::uint32_t> good[] = {{"0", 0}, {"1", 1}, {" 1 ", 1}, {"\t0", 0}, {"2", 2}};
    for (const auto& [s, v] : good) {
        const auto r = parseE1(s);
        CHECK(r.has_value() && *r == v);
    }
    for (const char* s : {"", "3", "-1", "on", "true", "01", "1x", "yes"}) {
        if (parseE1(s).has_value()) std::printf("  parseE1(\"%s\") accepted\n", s);
        CHECK(!parseE1(s).has_value());
    }
}
