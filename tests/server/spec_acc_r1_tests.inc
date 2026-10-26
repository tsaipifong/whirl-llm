// feat026-acc-r1 (review progress\feat026-pidle-acc.review.md A4 acc part, C5): included by server_tests.cpp after
// testSpecAccModes. Written from the review findings before the fix:
//   A4: WHIRL_SPEC_ACC is not in the env table (release.cpp k_env) nor in docs/usage.md, and the names the design used
//       (host / pinned / gpu / gpu-zc / check) are rejected: server_main parsed it with envU32, so WHIRL_SPEC_ACC=gpu
//       warned "not a number" and ran mode 0.
//   C5: stat_acc_mismatch_ counted two different events: a device result out of range in modes 2/3 (the host decides,
//       output unchanged) and a mode-4 disagreement between the device and the host loop.
// The parts guarded by WHIRL_SPEC_ACC_R1 use the fixed API (parseSpecAcc, Stats::acc_fallback); without it (the pre-fix
// source, built by scratch\feat026_acc_r1\run_r1.ps1 stage A) only the behaviour checks compile, and they must FAIL.

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
        } else {
            CHECK(es.acc_mismatch > 0);
            CHECK(g_log.count("spec-acc check |") >= 1);
            CHECK_EQ(g_log.count("spec-acc fallback |"), 0u);
        }
#ifdef WHIRL_SPEC_ACC_R1
        if (mode == 2 || mode == 3) {
            CHECK(es.acc_fallback > 0);
            CHECK(es.acc_fallback <= srv.model().specCorrupted());
        } else {
            CHECK_EQ(es.acc_fallback, 0u);
            CHECK(es.acc_mismatch <= srv.model().specCorrupted());
        }
        // one stats line per finished request, with the two counters apart (the line is written after the reply is
        // sent, so the last request's line may still be on its way: at least n_req - 1 lines, at most n_req)
        const std::size_t nst = g_log.count("spec-acc stats |");
        CHECK(nst + 1 >= n_req && nst <= n_req);
        const std::size_t fb0 = g_log.count("out of range (host decided) 0;"), ck0 = g_log.count("disagree 0;");
        if (mode == 2 || mode == 3) {
            CHECK(fb0 < nst);   // some line shows a fallback
            CHECK_EQ(ck0, nst);  // and never a check disagreement
        } else {
            CHECK_EQ(fb0, nst);
            CHECK(ck0 < nst);
        }
#endif
        CHECK_EQ(srv.model().poison(), 0u);
    }
#ifdef WHIRL_SPEC_ACC_R1
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
#endif
}

// A4: the env table, the user docs and the parser
void testSpecAccR1Env() {
#ifdef WHIRL_SOURCE_DIR
    auto slurp = [](const char* rel) {
        std::ifstream in(std::string(WHIRL_SOURCE_DIR) + rel, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    const std::string rel = slurp("/src/release/release.cpp");
    const std::string doc = slurp("/docs/usage.md");
    const std::string sm = slurp("/src/server/server_main.cpp");
    CHECK(!rel.empty() && !doc.empty() && !sm.empty());
    CHECK(rel.find("\"SPEC_ACC=") != std::string::npos);
    CHECK(doc.find("`WHIRL_SPEC_ACC=") != std::string::npos);
    for (const char* n : {"host", "pinned", "gpu", "gpu-zc", "check"}) {
        // every name in both places
        const std::size_t r0 = rel.find("\"SPEC_ACC=");
        const std::size_t d0 = doc.find("`WHIRL_SPEC_ACC=");
        CHECK(r0 != std::string::npos && rel.find(n, r0) != std::string::npos && rel.find(n, r0) < rel.find('\n', r0));
        CHECK(d0 != std::string::npos && doc.find(n, d0) != std::string::npos && doc.find(n, d0) < doc.find('\n', d0));
    }
    // the server must not parse it as a plain number (WHIRL_SPEC_ACC=gpu was "not a number" -> mode 0)
    CHECK_EQ(sm.find("envU32(\"SPEC_ACC\""), std::string::npos);
#endif
#ifdef WHIRL_SPEC_ACC_R1
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
#endif
}
