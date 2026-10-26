// feat026-hostgap: WHIRL_CYCLE_SYNC (EngineOptions::cycle_sync) against the mock engine. Included by server_tests.cpp.
// SPDX-License-Identifier: Apache-2.0
//
// What is checked, per request kind (greedy, sampled, relaxacc) x WHIRL_SPEC_ACC (0 host, 1 pinned, 3 gpu-zc) x
// WHIRL_CYCLE_SYNC (0..3), one slot, fixed draft count (no wall-clock policy, so every run has the same cycles):
// - the same text in every setting (greedy: also the independent greedy reference);
// - the same decode cycle count in every WHIRL_CYCLE_SYNC setting of one WHIRL_SPEC_ACC mode;
// - the sync count (counting hooks: NullBackend::blockingCalls() = download / streamSync / eventSync,
//   MockModel::hostWaits() = readCtl / readCtlPinned / syncStream): fold (bit 0) removes exactly one blocking
//   wait per folded readback, i.e. one per decode cycle of a sampled request (2 -> 1 wait per cycle) and one per
//   drafted relaxacc cycle; spin (bit 1) adds no blocking wait (it polls) and polls once per cycle with
//   WHIRL_SPEC_ACC < 2, never with >= 2; greedy cycles keep their one wait.

namespace hostgap_test {

struct Run {
    std::string text;
    bool ref_ok = true;  // greedy: equals the independent greedy reference
    std::uint64_t waits = 0;  // blocking calls + model waits, whole run (prefill, decode, finish, engine stop)
    Engine::Stats st;
};

Run runOne(int kind, std::uint32_t spec_acc, std::uint32_t cycle_sync, const std::vector<std::uint32_t>& prompt,
           std::uint32_t max_tokens) {
    ServerSetup su = baseSetup();
    su.mc.parallel = 1;
    su.mc.pool_tokens = 65536;
    su.eo.mtp_auto = false;  // fixed draft count: the cycles do not depend on wall-clock timing
    su.eo.n_draft = 3;
    su.eo.ngram = false;     // MTP drafts only (relaxacc applies to MTP drafts)
    su.eo.relax.on = kind == 2;
    su.eo.spec_acc = spec_acc;
    su.eo.cycle_sync = cycle_sync;
    TestServer srv(su);
    std::string body;
    if (kind == 1)
        body = std::format("{{\"prompt\":{},\"max_tokens\":{},\"temperature\":1.0,\"top_k\":8,\"seed\":9,\"ignore_eos\":true}}",
                           idsJson(prompt), max_tokens);
    else
        body = std::format("{{\"prompt\":{},\"max_tokens\":{},\"temperature\":0,\"ignore_eos\":true}}", idsJson(prompt), max_tokens);
    auto r = srv.post("/v1/completions", body);
    CHECK_EQ(r.status, 200);
    Run out;
    out.text = completionText(r);
    if (kind == 0) out.ref_ok = out.text == decode(srv.model().greedyReference(prompt, max_tokens, true));
    srv.stopEngine();  // every step of the request (finishJob, its syncs) is done
    out.st = srv.engine().stats();
    out.waits = srv.ops().blockingCalls() + srv.model().hostWaits();
    CHECK_EQ(srv.model().poison(), 0u);
    CHECK_EQ(srv.model().hidMismatch(), 0u);
    return out;
}

}  // namespace hostgap_test

void testCycleSyncModes() {
    using namespace hostgap_test;
    const char* kinds[] = {"greedy", "sampled", "relaxacc"};
    const auto prompt = makePrompt(420, 6031);
    const std::uint32_t mt = 70;
    for (int kind = 0; kind < 3; ++kind) {
        std::string text_ref;
        for (std::uint32_t acc : {0u, 1u, 3u}) {
            std::array<Run, 4> r;
            for (std::uint32_t cs = 0; cs < 4; ++cs) r[cs] = runOne(kind, acc, cs, prompt, mt);
            const std::uint64_t cyc = r[0].st.decode_cycles;
            CHECK(cyc > 0);
            for (std::uint32_t cs = 0; cs < 4; ++cs) {
                const Run& x = r[cs];
                if (text_ref.empty()) text_ref = x.text;
                if (x.text != text_ref)
                    std::printf("  cycle_sync %s acc %u cs %u: text differs from acc 0 cs 0\n", kinds[kind], acc, cs);
                CHECK_EQ(x.text, text_ref);
                CHECK_EQ(x.st.decode_cycles, cyc);
                // spin: one poll per cycle when the wait is outside readCycleCtl (WHIRL_SPEC_ACC < 2), else none
                const bool spin = (cs & 2u) != 0 && acc < 2;
                CHECK_EQ(x.st.cyc_spin, spin ? cyc : 0u);
                // fold: one blocking wait less per folded readback, nothing else changes
                const std::uint64_t base = r[0].waits;
                CHECK(x.ref_ok);
                CHECK_EQ(x.waits + ((cs & 1u) ? x.st.cyc_folded : 0u), base);
                if (!(cs & 1u)) CHECK_EQ(x.st.cyc_folded, 0u);
            }
            const std::uint64_t folded = r[1].st.cyc_folded;
            if (kind == 0) CHECK_EQ(folded, 0u);
            if (kind == 1) CHECK_EQ(folded, cyc);  // every cycle of a sampled request reads its candidates
            if (kind == 2) CHECK(folded > 0 && folded <= cyc);
            CHECK_EQ(r[3].st.cyc_folded, folded);
            std::printf("  cycle_sync %-8s acc %u: %llu decode cycles, waits per run cs0 %llu cs1 %llu cs2 %llu cs3 %llu "
                        "(folded %llu: %.2f fewer blocking waits per cycle)\n",
                        kinds[kind], acc, static_cast<unsigned long long>(cyc), static_cast<unsigned long long>(r[0].waits),
                        static_cast<unsigned long long>(r[1].waits), static_cast<unsigned long long>(r[2].waits),
                        static_cast<unsigned long long>(r[3].waits), static_cast<unsigned long long>(folded),
                        static_cast<double>(folded) / static_cast<double>(cyc));
        }
    }
}

// the per-cycle sync count of a greedy cycle: one wait in every setting (two run lengths, the difference is decode only)
void testCycleSyncPerCycle() {
    using namespace hostgap_test;
    for (int kind : {0, 1}) {
        for (std::uint32_t cs : {0u, 1u, 3u}) {
            const auto prompt = makePrompt(300, 7101);
            const Run a = runOne(kind, 0, cs, prompt, 30);
            const Run b = runOne(kind, 0, cs, prompt, 90);
            const std::uint64_t dc = b.st.decode_cycles - a.st.decode_cycles;
            const std::int64_t dw = static_cast<std::int64_t>(b.waits) - static_cast<std::int64_t>(a.waits);
            const double per = dc > 0 ? static_cast<double>(dw) / static_cast<double>(dc) : 0.0;
            std::printf("  cycle_sync per cycle: %s cs %u: %+lld waits over %llu more cycles = %.3f per cycle\n",
                        kind == 0 ? "greedy" : "sampled", cs, static_cast<long long>(dw), static_cast<unsigned long long>(dc), per);
            CHECK(dc > 0);
            if (kind == 0) CHECK_EQ(dw, static_cast<std::int64_t>(dc));  // greedy: the control-word read only
            // sampled, fold on: the candidates ride on the same wait; off: one more blocking download per cycle
            // (escalations to a wider top-k add the same downloads in both, so compare the two directly below)
        }
        if (kind == 1) {
            const auto prompt = makePrompt(300, 7101);
            const Run a0 = runOne(1, 0, 0, prompt, 30), b0 = runOne(1, 0, 0, prompt, 90);
            const Run a1 = runOne(1, 0, 1, prompt, 30), b1 = runOne(1, 0, 1, prompt, 90);
            const std::int64_t d0 = static_cast<std::int64_t>(b0.waits) - static_cast<std::int64_t>(a0.waits);
            const std::int64_t d1 = static_cast<std::int64_t>(b1.waits) - static_cast<std::int64_t>(a1.waits);
            const std::int64_t dc = static_cast<std::int64_t>(b0.st.decode_cycles - a0.st.decode_cycles);
            CHECK_EQ(d0 - d1, dc);  // exactly one blocking wait less per sampled decode cycle
        }
    }
}
