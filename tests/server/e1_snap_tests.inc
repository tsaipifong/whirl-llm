// v0252-e1s8 (PLAN V1-93 P2a): the E1 one-cycle lookahead on a model in DeltaNet SNAPSHOT mode (the Radeon 8060S / gfx1151
// verify: snapshot sets, no replay) through the select mode of kernels/gdn_select.h. Included by server_tests.cpp after
// e1_ahead_tests.inc (it reuses that file's suite with e1ahead::g_snap = true).
//   e1_snap_rule   the select rule alone (device pick / out, host settle) against a buffer simulation over random cycles: every
//                  cycle reads the true state, a select cycle never writes the buffer it read (the state after the cycle
//                  before survives it), after settle the spare holds that state and the live / snapshot tables hold the
//                  select cycle's outputs exactly where a synchronous cycle would have left them
//   e1_snap_ahead  the whole e1_ahead suite (equal, end, order, drain, fallback, len) on a snapshot-mode model with the select
//                  mode (MockConfig::snap_select: real buffer tables, the device rule from gdn_select.h)
//   e1_snap_gate   the switch on a snapshot-mode model: without the select mode WHIRL_E1 stays refused (as before); with it
//                  WHIRL_E1=2 runs select verifies, settles each one the host read with a cycle behind it, gives the state
//                  back at every end inside the window; WHIRL_E1=1 runs no select verify (the host restores as before);
//                  WHIRL_E1=0 calls none of it; replies = the greedy reference in every arm
// SPDX-License-Identifier: Apache-2.0

#include "../../kernels/gdn_select.h"

void testE1SnapRule() {
    // a buffer world: integer "states"; buffer contents change only by device writes
    std::mt19937 rng(20261125);
    std::uint64_t cases = 0, selects = 0, ends = 0;
    for (int m = 1; m <= 6; ++m) {
        for (int trial = 0; trial < 300; ++trial) {
            std::vector<std::int64_t> mem(static_cast<std::size_t>(m) + 2, -1);  // buffer contents
            // handles: sets[0 .. m), live, spare = indices into mem
            std::vector<int> sets(static_cast<std::size_t>(m));
            for (int r = 0; r < m; ++r) sets[r] = r;
            int live = m, spare = m + 1;
            std::int64_t truth = 1000;  // the true state value
            mem[live] = truth;
            std::int64_t next_val = 1;
            // cycle 0 is synchronous (in place): input live, snapshots -> sets, final -> live; then a pipeline of select cycles
            std::vector<std::int64_t> outs(static_cast<std::size_t>(m) + 1);
            auto runPlain = [&](int nd) {
                const std::int64_t in = mem[live];
                CHECK_EQ(in, truth);
                for (int r = 0; r <= nd; ++r) outs[r] = in * 100 + (next_val++);  // state after row r
                for (int r = 0; r < nd; ++r) mem[sets[r]] = outs[r];
                mem[live] = outs[nd];
            };
            const int nd0 = m;
            runPlain(nd0);
            int prev_nd = nd0;
            int acc_prev = static_cast<int>(rng() % (prev_nd + 1));
            const int cycles = 2 + static_cast<int>(rng() % 6);
            for (int k = 0; k < cycles; ++k) {
                // the select cycle is submitted before the host knows acc_prev: candidates P = [sets[0 .. prev_nd), live]
                const int nd = 1 + static_cast<int>(rng() % prev_nd);  // nd <= m of the verify before
                std::vector<int> P(sets.begin(), sets.begin() + prev_nd);
                P.push_back(live);
                const int c = gdn_sel_pick(acc_prev + 1, prev_nd);
                CHECK_EQ(c, acc_prev);
                const int in_buf = P[c];
                const std::int64_t want_in = outs[acc_prev];  // the state after row acc_prev of the cycle before
                CHECK_EQ(mem[in_buf], want_in);
                std::vector<std::int64_t> o(static_cast<std::size_t>(nd) + 1);
                for (int r = 0; r <= nd; ++r) o[r] = mem[in_buf] * 100 + (next_val++);
                std::vector<int> written;
                for (int r = 0; r < nd; ++r) {
                    const int j = gdn_sel_out(r, c);
                    CHECK(j <= prev_nd && j != c);
                    mem[P[j]] = o[r];
                    written.push_back(P[j]);
                }
                mem[spare] = o[nd];
                written.push_back(spare);
                for (int w : written) CHECK(w != in_buf);  // the input survives the cycle
                // all outputs distinct buffers
                std::vector<int> ws = written;
                std::sort(ws.begin(), ws.end());
                CHECK(std::adjacent_find(ws.begin(), ws.end()) == ws.end());
                selects += 1;
                // the host reads the cycle before (acc_prev) and settles (over the candidates only: sets past prev_nd stay)
                gdn_sel_settle(sets.data(), prev_nd, live, spare, c);
                CHECK_EQ(mem[spare], want_in);  // the spare holds the state after the cycle before
                for (int r = 0; r < nd; ++r) CHECK_EQ(mem[sets[r]], o[r]);  // snapshot set r = the select cycle's row r
                CHECK_EQ(mem[live], o[nd]);                                 // live = its final state
                // the request may end here (the select cycle dropped): dropPending swaps live and spare
                if (rng() % 7 == 0) {
                    std::swap(live, spare);
                    CHECK_EQ(mem[live], want_in);
                    ends += 1;
                    break;
                }
                outs = o;
                outs.resize(static_cast<std::size_t>(m) + 1);
                truth = 0;
                acc_prev = static_cast<int>(rng() % (nd + 1));
                prev_nd = nd;
                // sometimes the next cycle is synchronous instead (drain): the host restores snapshot acc_prev as before
                if (rng() % 5 == 0) {
                    if (acc_prev < nd) std::swap(live, sets[acc_prev]);
                    truth = outs[acc_prev];
                    CHECK_EQ(mem[live], truth);
                    const int ndp = 1 + static_cast<int>(rng() % m);
                    runPlain(ndp);
                    prev_nd = ndp;
                    acc_prev = static_cast<int>(rng() % (ndp + 1));
                }
            }
            cases += 1;
        }
    }
    // the pick clamps (a corrupt word never indexes past the candidates)
    CHECK_EQ(gdn_sel_pick(0, 3), 0);
    CHECK_EQ(gdn_sel_pick(-7, 3), 0);
    CHECK_EQ(gdn_sel_pick(99, 3), 3);
    CHECK_EQ(gdn_sel_out(0, 0), 1);
    CHECK_EQ(gdn_sel_out(2, 1), 3);
    CHECK_EQ(gdn_sel_out(0, 3), 0);
    std::printf("  e1_snap_rule: %llu sequences, %llu select cycles, %llu ends inside the window\n", static_cast<unsigned long long>(cases),
                static_cast<unsigned long long>(selects), static_cast<unsigned long long>(ends));
    CHECK(selects > 1000 && ends > 50);
}

void testE1SnapAhead() {
    e1ahead::g_snap = true;
    struct Restore {
        ~Restore() { e1ahead::g_snap = false; }
    } restore;
    std::printf("  [snapshot mode + select] e1_ahead_equal\n");
    testE1AheadEqual();
    std::printf("  [snapshot mode + select] e1_ahead_end\n");
    testE1AheadEnd();
    std::printf("  [snapshot mode + select] e1_ahead_order\n");
    testE1AheadOrder();
    std::printf("  [snapshot mode + select] e1_ahead_drain\n");
    testE1AheadDrain();
    std::printf("  [snapshot mode + select] e1_ahead_fallback\n");
    testE1AheadFallback();
    std::printf("  [snapshot mode + select] e1_ahead_len\n");
    testE1AheadLen();
}

void testE1SnapGate() {
#ifdef WHIRL_SOURCE_DIR
    {
        std::ifstream in(std::string(WHIRL_SOURCE_DIR) + "/docs/usage.md", std::ios::binary);
        const std::string doc((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(doc.find("On the Radeon 8060S (DeltaNet snapshot sets instead of replay) a value set here runs too") != std::string::npos);
        CHECK(doc.find("`decode: E1 in DeltaNet snapshot mode`") != std::string::npos);
        CHECK_EQ(doc.find("DeltaNet replay mode only (the Radeon 8060S verifies with snapshot sets and keeps `0`"), std::string::npos);
    }
#endif
    // ---- v0252-e1s8 round 3: the gfx1151 DEFAULT, built as server_main builds it (chooseE1(no env, e1Default(moe, gfx1151), site)
    // and e1DefaultCtxMax(choice, e1CtxMax(gfx1151))): MoE precise / balance start the lookahead with the lines an explicit
    // WHIRL_E1=2 prints plus the boundary; dense (Swift) and fast print none of them
    {
        const std::uint32_t lim = qwen35::e1CtxMax(hip::Arch::gfx1151);
        CHECK_EQ(lim, 8192u);
        // the limit is per cycle on the slot's position (prompt + generated so far): E1 while pos < 8192
        CHECK(e1LenAllows(lim, 8191) && !e1LenAllows(lim, 8192) && e1LenAllows(0, 1u << 30));
        struct Arm {
            bool moe, fast;
        };
        for (const Arm arm : {Arm{true, false}, Arm{false, false}, Arm{true, true}}) {
            E1Site site{.mtp = true, .dflash = false, .relax = arm.fast, .fast = arm.fast, .model_ok = true};
            const E1Choice c = chooseE1(std::nullopt, qwen35::e1Default(arm.moe, hip::Arch::gfx1151), site);
            ServerSetup su = e1ahead::setup(c.e1, 1);
            su.mc.replay = false;
            su.mc.snap_select = true;
            su.mc.snap_sets = 1;
            su.eo.e1_ctx_max = e1DefaultCtxMax(c, lim);
            g_log.clear();
            TestServer srv(su);
            const std::vector<std::uint32_t> p = makePrompt(300, 1997);
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 80))), decode(srv.model().greedyReference(p, 80, true)));
            std::printf("  e1_snap_gate default gfx1151 %s%s: e1 %u, limit %u, ring %llu\n", arm.moe ? "MoE" : "dense", arm.fast ? " fast" : "", c.e1,
                        su.eo.e1_ctx_max, static_cast<unsigned long long>(srv.engine().stats().acc_ring));
            if (arm.moe && !arm.fast) {
                CHECK(c.e1 == 2u && c.by_default);
                CHECK_EQ(su.eo.e1_ctx_max, 8192u);
                CHECK_EQ(g_log.count("decode: WHIRL_E1=2 by default below 8192 tokens of context"), 1u);
                CHECK_EQ(g_log.count("decode: WHIRL_SPEC_ACC=3 (gpu-zc)"), 1u);
                CHECK_EQ(g_log.count("decode: E1 in DeltaNet snapshot mode"), 1u);
                CHECK(srv.engine().stats().acc_ring > 0);
            } else {
                CHECK_EQ(c.e1, 0u);
                CHECK_EQ(g_log.count("decode: WHIRL_E1="), 0u);
                CHECK_EQ(g_log.count("decode: E1 in DeltaNet snapshot mode"), 0u);
                CHECK_EQ(g_log.count("decode: WHIRL_SPEC_ACC="), 0u);
                CHECK_EQ(srv.engine().stats().acc_ring, 0u);
            }
        }
    }
    // ---- snapshot mode WITHOUT the select mode (a code object without it): refused as before, said once
    {
        ServerSetup su = e1ahead::setup(2, 1);
        su.mc.replay = false;
        su.mc.snap_select = false;
        su.mc.snap_sets = 1;
        g_log.clear();
        TestServer srv(su);
        const std::vector<std::uint32_t> p = makePrompt(200, 1991);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 50))), decode(srv.model().greedyReference(p, 50, true)));
        CHECK_EQ(g_log.count("WHIRL_E1=2: the model has no E1 path (DeltaNet snapshot mode); E1 off"), 1u);
        CHECK_EQ(srv.engine().stats().acc_ring, 0u);
        CHECK_EQ(srv.model().selCounts().selects, 0u);
    }
    // ---- with the select mode: Ornith-like (1 draft, 1 snapshot set) and Swift-like (4 drafts), E1 = 0 / 1 / 2, requests
    // that end by end-of-turn inside the window and by max_tokens
    for (const std::uint32_t nd : {1u, 4u}) {
        std::string text[3][3];
        for (const std::uint32_t e1 : {0u, 1u, 2u}) {
            ServerSetup su = e1ahead::setup(e1, nd);
            su.mc.replay = false;
            su.mc.snap_select = true;
            su.mc.snap_sets = nd;  // exactly the sets the drafts need (Ornith: 1)
            su.mc.eos_rate = 0.03;
            g_log.clear();
            TestServer srv(su);
            std::uint32_t i = 0;
            for (const std::uint32_t seed : {1993u, 1994u, 1995u}) {
                const std::vector<std::uint32_t> p = makePrompt(240 + 30 * i, seed);
                const HttpResult r = srv.post("/v1/completions", e1test::greedyBody(p, 160, false));
                CHECK_EQ(r.status, 200);
                text[e1][i] = completionText(r);
                CHECK_EQ(text[e1][i], decode(srv.model().greedyReference(p, 160, false)));
                ++i;
            }
            const Engine::Stats es = srv.engine().stats();
            const MockModel::SelCounts sc = srv.model().selCounts();
            std::printf("  e1_snap_gate nd %u E1=%u: ring %llu ahead %llu discard %llu | selects %llu settles %llu given back %llu bad %llu\n", nd, e1,
                        static_cast<unsigned long long>(es.acc_ring), static_cast<unsigned long long>(es.acc_ahead),
                        static_cast<unsigned long long>(es.acc_discard), static_cast<unsigned long long>(sc.selects),
                        static_cast<unsigned long long>(sc.settles), static_cast<unsigned long long>(sc.given_back),
                        static_cast<unsigned long long>(sc.bad_settles));
            CHECK_EQ(srv.model().poison(), 0u);
            CHECK_EQ(srv.model().hidMismatch(), 0u);
            CHECK_EQ(sc.bad_settles, 0u);
            CHECK_EQ(srv.model().e1Counts().disagree, 0u);
            CHECK_EQ(g_log.count("has no E1 path"), 0u);
            // the startup proof line of the snapshot-mode path (what the GPU smoke greps on the 8060S)
            CHECK_EQ(g_log.count("decode: E1 in DeltaNet snapshot mode"), e1 == 0 ? 0u : 1u);
            if (e1 == 2) {
                CHECK_EQ(g_log.count("decode: WHIRL_E1=2"), 1u);
                CHECK(es.acc_ahead > 0);
                // every cycle submitted ahead is a select verify; each is settled when the host reads the cycle before it
                CHECK_EQ(sc.selects, es.acc_ahead);
                CHECK_EQ(sc.settles, es.acc_ahead);
                // every end inside the window gave the state of the last read cycle back
                CHECK_EQ(sc.given_back, es.acc_discard);
            } else {
                CHECK_EQ(sc.selects, 0u);
                CHECK_EQ(sc.settles, 0u);
                CHECK_EQ(es.acc_ring, 0u);
                if (e1 == 1) CHECK_EQ(g_log.count("decode: WHIRL_E1=1"), 1u);
                else CHECK_EQ(g_log.count("decode: WHIRL_E1="), 0u);
            }
        }
        for (std::uint32_t i = 0; i < 3; ++i) {
            CHECK_EQ(text[0][i], text[1][i]);
            CHECK_EQ(text[0][i], text[2][i]);
        }
    }
}
