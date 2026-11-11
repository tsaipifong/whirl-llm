// v0252-e1p1 (PLAN V1-93, E1 step P1a): the port of feat026-acc (33c5da1, tests of 4ac9da6 / feat026-acc-r1 c797b27) onto
// int0252, where the decode cycle is SpecCycle (src/server/engine_decode.cpp: sampleRows reads the verify's control words,
// acceptSlot decides per slot). Included by server_tests.cpp; written from the feat026-acc design and review before the port.
//   spec_acc_modes        every WHIRL_SPEC_ACC mode gives the greedy reference text; the device accept decision
//                         (kernels/spec_accept.h, run by the mock on the host) agrees with the host path on every slot cycle
//                         and the device's next-cycle rows equal the host's (check mode); a sampled request has the same text
//                         for its seed in every mode; without the kernel a device mode falls back to the pinned read
//   spec_acc_reads        which read of the control words each mode makes (the E1 rules of progress/v0252-e1study.HANDOFF.md
//                         section 4): host = the pageable readCtl only; pinned / gpu / check = the pinned read only; gpu-zc =
//                         NO readCtl and NO pinned read (the accept kernel wrote the words into pinned host memory: no D2H
//                         copy command per cycle); the device modes run the accept kernel once per decode cycle
//   spec_acc_r1_counters  (feat026-acc-r1 review C5) out-of-range device results: modes 2/3 fall back to the host and count it
//                         as a fallback ("spec-acc fallback |"), never as a disagreement; mode 4 counts a disagreement
//                         ("spec-acc check |"); one "spec-acc stats |" line per finished request in modes >= 2
//   spec_acc_r1_env       (feat026-acc-r1 review A4) the env table row and docs/usage.md row with all five names; the server
//                         reads WHIRL_SPEC_ACC through config.settings (settings::specAcc) and parses it with parseSpecAcc
//                         (names and digits, a bad value -> nullopt), the name table round trip
// SPDX-License-Identifier: Apache-2.0

void testSpecAccModes() {
    struct Case {
        const char* name;
        bool replay;
        std::uint32_t markov;
        double eos_rate;
        std::uint32_t parallel;
    };
    const Case cases[] = {{"mtp", true, 0, 0.0, 4}, {"no-replay", false, 0, 0.0, 4}, {"ngram", true, 40, 0.0, 4},
                          {"eos", true, 0, 0.03, 4}, {"one-slot", true, 0, 0.02, 1}};
    for (const Case& c : cases) {
        std::string sampled_ref;
        for (std::uint32_t mode : {0u, 1u, 2u, 3u, 4u}) {
            ServerSetup su = baseSetup();
            su.mc.replay = c.replay;
            su.mc.markov = c.markov;
            su.mc.eos_rate = c.eos_rate;
            su.mc.parallel = c.parallel;
            su.mc.pool_tokens = 131072;
            su.eo.spec_acc = mode;
            TestServer srv(su);
            const std::uint32_t n_req = c.parallel == 1 ? 3 : 6;
            std::vector<std::vector<std::uint32_t>> prompts;
            std::vector<std::uint32_t> mt;
            for (std::uint32_t i = 0; i < n_req; ++i) {
                prompts.push_back(makePrompt(150 + 211 * i % 900, 300 + i));
                mt.push_back(i % 3 == 2 ? 7 + i : 90);  // some requests end by max_tokens inside a draft run
            }
            std::vector<std::string> got(n_req);
            std::vector<std::thread> th;
            for (std::uint32_t i = 0; i < n_req; ++i)
                th.emplace_back([&, i] {
                    const bool stream = i % 2 == 1;
                    const std::string body = std::format("{{\"prompt\":{},\"max_tokens\":{},\"temperature\":0,\"stream\":{}}}",
                                                         idsJson(prompts[i]), mt[i], stream ? "true" : "false");
                    auto r = srv.post("/v1/completions", body);
                    got[i] = stream ? streamText(r, false) : completionText(r);
                });
            for (auto& t : th) t.join();
            std::uint32_t ok = 0;
            for (std::uint32_t i = 0; i < n_req; ++i)
                if (got[i] == decode(srv.model().greedyReference(prompts[i], mt[i], false))) ++ok;
            if (ok != n_req) std::printf("  spec_acc case %s mode %u: %u / %u replies equal the reference\n", c.name, mode, ok, n_req);
            CHECK_EQ(ok, n_req);
            // a sampled request (host decision in every mode): the same text for its seed
            const std::string sb = std::format(
                "{{\"prompt\":{},\"max_tokens\":40,\"temperature\":1.0,\"top_k\":8,\"seed\":9,\"ignore_eos\":true}}", idsJson(prompts[0]));
            const std::string st = completionText(srv.post("/v1/completions", sb));
            if (mode == 0) sampled_ref = st;
            else CHECK_EQ(st, sampled_ref);
            const Engine::Stats es = srv.engine().stats();
            if (mode >= 2) CHECK(srv.model().specAccepts() > 0);
            else CHECK_EQ(srv.model().specAccepts(), 0u);
            if (mode == 2 || mode == 3) CHECK(es.acc_dev > 0);
            if (mode == 4) {
                CHECK(es.acc_checked > 0);
                CHECK(es.tab_checked > 0);
                CHECK_EQ(es.acc_dev, 0u);
            }
            if (mode < 2) CHECK(es.acc_dev == 0 && es.acc_checked == 0 && es.tab_checked == 0);
            if (es.acc_mismatch != 0 || es.tab_mismatch != 0)
                std::printf("  spec_acc case %s mode %u: acc mismatches %llu / %llu, row-table mismatches %llu / %llu\n", c.name, mode,
                            static_cast<unsigned long long>(es.acc_mismatch), static_cast<unsigned long long>(es.acc_checked),
                            static_cast<unsigned long long>(es.tab_mismatch), static_cast<unsigned long long>(es.tab_checked));
            CHECK_EQ(es.acc_mismatch, 0u);
            CHECK_EQ(es.tab_mismatch, 0u);
            CHECK_EQ(es.acc_fallback, 0u);
            CHECK_EQ(srv.model().poison(), 0u);
            CHECK_EQ(srv.model().hidMismatch(), 0u);
        }
    }
    // without the kernel, a device mode falls back to the pinned read + host loop (and says so once at startup)
    {
        ServerSetup su = baseSetup();
        su.mc.spec_acc_kernel = false;
        su.eo.spec_acc = 2;
        g_log.clear();
        TestServer srv(su);
        const auto p = makePrompt(300, 411);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 60))), decode(srv.model().greedyReference(p, 60, true)));
        CHECK_EQ(srv.model().specAccepts(), 0u);
        CHECK_EQ(srv.engine().stats().acc_dev, 0u);
        CHECK(srv.model().readCtlPinnedCalls() > 0);
        CHECK_EQ(srv.model().readCtlCalls(), 0u);
        CHECK_EQ(g_log.count("WHIRL_SPEC_ACC=2: no spec_accept_rows kernel; pinned read + host accept"), 1u);
        CHECK_EQ(g_log.count("decode: WHIRL_SPEC_ACC=1 (pinned)"), 1u);
    }
}

// which read each mode makes per decode cycle (no other path reads the control words: SpecCycle::sampleRows is the one
// reader, through SampleBuffers::readCycleCtl)
void testSpecAccReads() {
    for (std::uint32_t mode : {0u, 1u, 2u, 3u, 4u}) {
        ServerSetup su = baseSetup();
        su.eo.spec_acc = mode;
        g_log.clear();
        TestServer srv(su);
        for (std::uint32_t i = 0; i < 2; ++i) {
            const auto p = makePrompt(260 + 40 * i, 900 + i);
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 50))), decode(srv.model().greedyReference(p, 50, true)));
        }
        const std::uint64_t rc = srv.model().readCtlCalls(), rp = srv.model().readCtlPinnedCalls(), sa = srv.model().specAccepts();
        std::printf("  spec_acc_reads mode %u: readCtl %llu, pinned read %llu, accept kernel %llu\n", mode, static_cast<unsigned long long>(rc),
                    static_cast<unsigned long long>(rp), static_cast<unsigned long long>(sa));
        switch (mode) {
        case 0:  // host: the pageable read only, no kernel, no pinned buffer use
            CHECK(rc > 0);
            CHECK_EQ(rp, 0u);
            CHECK_EQ(sa, 0u);
            CHECK_EQ(g_log.count("decode: WHIRL_SPEC_ACC="), 0u);  // the default says nothing
            break;
        case 1:  // pinned: the pinned read only, no kernel
            CHECK_EQ(rc, 0u);
            CHECK(rp > 0);
            CHECK_EQ(sa, 0u);
            break;
        case 2:
        case 4:  // gpu / check: one accept kernel per cycle, then the pinned read
            CHECK_EQ(rc, 0u);
            CHECK(sa > 0);
            CHECK_EQ(rp, sa);
            break;
        case 3:  // gpu-zc: one accept kernel per cycle and NO read at all (the kernel wrote the pinned words)
            CHECK_EQ(rc, 0u);
            CHECK_EQ(rp, 0u);
            CHECK(sa > 0);
            break;
        }
        if (mode >= 1) CHECK_EQ(g_log.count(std::format("decode: WHIRL_SPEC_ACC={} ({})", mode, specAccName(mode))), 1u);
        CHECK_EQ(srv.model().poison(), 0u);
    }
}

// mode 2 / 3 / 4 with a corrupted device result (MockConfig::spec_acc_corrupt: every Nth specAccept writes an
// out-of-range accept count for its first device-decided sequence): the text stays the greedy reference in every mode
// (the host decides); modes 2/3 count it as an out-of-range fallback and warn "spec-acc fallback |", never as a
// disagreement; mode 4 counts a disagreement and warns "spec-acc check |", never a fallback.
void testSpecAccR1Counters() {
    for (std::uint32_t mode : {2u, 3u, 4u}) {
        ServerSetup su = baseSetup();
        su.mc.pool_tokens = 131072;
        su.mc.spec_acc_corrupt = 3;
        su.eo.spec_acc = mode;
        TestServer srv(su);
        g_log.clear();
        std::uint32_t ok = 0;
        const std::uint32_t n_req = 4;
        for (std::uint32_t i = 0; i < n_req; ++i) {
            const auto p = makePrompt(200 + 97 * i, 500 + i);
            const std::string body = std::format("{{\"prompt\":{},\"max_tokens\":80,\"temperature\":0}}", idsJson(p));
            if (completionText(srv.post("/v1/completions", body)) == decode(srv.model().greedyReference(p, 80, false))) ++ok;
        }
        CHECK_EQ(ok, n_req);
        const Engine::Stats es = srv.engine().stats();
        CHECK(srv.model().specCorrupted() > 0);
        if (mode == 2 || mode == 3) {
            if (es.acc_mismatch != 0)
                std::printf("  spec_acc_r1 mode %u: out-of-range device results counted as disagreements (%llu)\n", mode,
                            static_cast<unsigned long long>(es.acc_mismatch));
            CHECK_EQ(es.acc_mismatch, 0u);  // C5: not a disagreement
            CHECK(g_log.count("spec-acc fallback |") >= 1);
            CHECK_EQ(g_log.count("spec-acc check |"), 0u);
            CHECK(es.acc_fallback > 0);
            CHECK(es.acc_fallback <= srv.model().specCorrupted());
        } else {
            CHECK(es.acc_mismatch > 0);
            CHECK(g_log.count("spec-acc check |") >= 1);
            CHECK_EQ(g_log.count("spec-acc fallback |"), 0u);
            CHECK_EQ(es.acc_fallback, 0u);
            CHECK(es.acc_mismatch <= srv.model().specCorrupted());
        }
        // one stats line per finished request, with the two counters apart (the line may be written after the reply is
        // sent, so the last request's line may still be on its way: at least n_req - 1 lines, at most n_req)
        const std::size_t nst = g_log.count("spec-acc stats |");
        CHECK(nst + 1 >= n_req && nst <= n_req);
        const std::size_t fb0 = g_log.count("out of range (host decided) 0;"), ck0 = g_log.count("disagree 0;");
        if (mode == 2 || mode == 3) {
            CHECK(fb0 < nst);    // some line shows a fallback
            CHECK_EQ(ck0, nst);  // and never a check disagreement
        } else {
            CHECK_EQ(fb0, nst);
            CHECK(ck0 < nst);
        }
        CHECK_EQ(srv.model().poison(), 0u);
    }
    // no corruption: both counters stay 0; mode 0 prints no stats line
    for (std::uint32_t mode : {0u, 2u}) {
        ServerSetup su = baseSetup();
        su.eo.spec_acc = mode;
        TestServer srv(su);
        g_log.clear();
        for (std::uint32_t i = 0; i < 2; ++i) {
            const auto p = makePrompt(300 + 50 * i, 611 + i);
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 60))), decode(srv.model().greedyReference(p, 60, true)));
        }
        const Engine::Stats es = srv.engine().stats();
        CHECK_EQ(es.acc_fallback, 0u);
        CHECK_EQ(es.acc_mismatch, 0u);
        const std::size_t nst = g_log.count("spec-acc stats |");
        if (mode == 0) CHECK_EQ(nst, 0u);
        else CHECK(nst >= 1 && nst <= 2);
    }
}

// A4: the env table, the user docs, the server's read and the parser
void testSpecAccR1Env() {
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
    CHECK(rel.find("\"SPEC_ACC=") != std::string::npos);
    CHECK(doc.find("`WHIRL_SPEC_ACC=") != std::string::npos);
    for (const char* n : {"host", "pinned", "gpu", "gpu-zc", "check"}) {
        // every name in both places, on the row itself
        const std::size_t r0 = rel.find("\"SPEC_ACC=");
        const std::size_t d0 = doc.find("`WHIRL_SPEC_ACC=");
        CHECK(r0 != std::string::npos && rel.find(n, r0) != std::string::npos && rel.find(n, r0) < rel.find('\n', r0));
        CHECK(d0 != std::string::npos && doc.find(n, d0) != std::string::npos && doc.find(n, d0) < doc.find('\n', d0));
    }
    // the one read point (config.settings, the env gate's allowed scope) and the one parser; never a plain number read
    // (WHIRL_SPEC_ACC=gpu was "not a number" -> mode 0 before feat026-acc-r1)
    CHECK(rs.find("envGet(\"SPEC_ACC\")") != std::string::npos);
    CHECK(sm.find("settings::specAcc()") != std::string::npos && sm.find("parseSpecAcc(") != std::string::npos);
    CHECK_EQ(sm.find("\"SPEC_ACC\""), std::string::npos);
#else
    CHECK(false);  // whirl-server-tests is built with WHIRL_SOURCE_DIR (cmake/server.cmake): the source checks must run
#endif
    const std::pair<const char*, std::uint32_t> good[] = {
        {"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"host", 0}, {"pinned", 1}, {"gpu", 2}, {"gpu-zc", 3},
        {"gpu_zc", 3}, {"zc", 3}, {"check", 4}, {"GPU", 2}, {"Gpu-ZC", 3}, {"CHECK", 4}, {" gpu ", 2}};
    for (const auto& [s, v] : good) {
        const auto r = parseSpecAcc(s);
        if (!r || *r != v) std::printf("  parseSpecAcc(\"%s\") wrong\n", s);
        CHECK(r.has_value() && *r == v);
    }
    for (const char* s : {"", "5", "9", "-1", "gpu zc", "gpuzc2", "hostx", "2x", "0x2", "4294967298", "pinned1"}) {
        if (parseSpecAcc(s).has_value()) std::printf("  parseSpecAcc(\"%s\") accepted\n", s);
        CHECK(!parseSpecAcc(s).has_value());
    }
    // the name table the engine logs and the parser agree
    for (std::uint32_t m = 0; m < 5; ++m) CHECK_EQ(parseSpecAcc(specAccName(m)).value_or(99), m);
    CHECK_EQ(std::string(specAccName(7)), std::string("?"));
}
