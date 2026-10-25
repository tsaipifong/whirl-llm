// feat026-pidle: PRECISE-IDLE tests (included by server_tests.cpp after the TAIL-CK tests; uses its helpers).
//
// The mock runs with sched_grid 1024 (a prefill forward that starts off the cold grid changes the state) and
// decode_mark (a prefill forward over history a verify wrote changes the state, as a prefill over decode-produced
// history is not bitwise the cold one on the GPU): a turn that resumes from a state the cold run does not have changes
// the text against the greedy reference (= a cold run), and decPrefills() counts the forwards that read decode rows.

namespace pidle_test {

using whirl::server::pidle::Plan;
using whirl::server::pidle::Sched;

ServerSetup preciseSetup(bool precise, bool idle, std::uint32_t idle_ms = 0) {
    ServerSetup su = baseSetup();
    su.mc.parallel = 1;
    su.mc.slot_ctx = 32768;
    su.mc.pool_tokens = 65536;
    su.mc.sched_grid = 1024;
    su.mc.decode_mark = precise;  // balance resumes from gen-end by design (numerically equivalent, not bitwise)
    su.eo.n_ck = 3;               // room for the tail checkpoint (cache::ckpt_keep_prefix_min)
    su.eo.cold_resume = precise;
    su.eo.precise_idle = idle;
    su.eo.precise_idle_ms = idle_ms;
    return su;
}

bool waitUntil(const std::function<bool()>& f, int ms) {
    const auto t_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < t_end) {
        if (f()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return f();
}

// turn 1 (prompt n1, reply of r tokens), then turn 2 = turn 1 + reply + `extra` new tokens
struct Turns {
    std::vector<std::uint32_t> p1, out1, p2;
};
Turns makeTurns(TestServer& srv, std::uint32_t n1, std::uint32_t r, std::uint32_t extra, std::uint64_t seed) {
    Turns t;
    t.p1 = makePrompt(n1, seed);
    auto r1 = srv.post("/v1/completions", completionBody(t.p1, r));
    CHECK_EQ(r1.status, 200);
    t.out1 = srv.model().greedyReference(t.p1, r, true);
    CHECK_EQ(completionText(r1), decode(t.out1));
    std::vector<std::uint32_t> h = t.p1;
    h.insert(h.end(), t.out1.begin(), t.out1.end());
    t.p2 = makePrompt(static_cast<std::uint32_t>(h.size() + extra), seed + 1, h);
    return t;
}

// the schedule points of a walk in [0, lim] (include/whirl/prefill_schedule.h)
std::vector<std::size_t> points(std::size_t n, std::size_t sys, std::size_t think, std::span<const std::uint32_t> imgs, std::size_t lim) {
    std::vector<std::size_t> v;
    for (std::size_t p = 0; p <= lim && p <= n; ++p)
        if (whirl::sched::onSchedule(n, sys, think, imgs, p)) v.push_back(p);
    return v;
}

}  // namespace pidle_test

// The pure part (server/precise_idle.h): the checkpoint point, the start choice, the chunks; and the lemma the
// bitwise claim rests on, checked numerically: for every continuation P of the n cached tokens (its own think split
// at its end, image ends after n), if the checkpoint point is on P's schedule, P's schedule points up to it equal the
// idle schedule's (so the idle prefill ran exactly P's cold chunks there).
void testPidlePlan() {
    using namespace pidle_test;
    using whirl::cache::CkptKey;
    const std::span<const std::uint32_t> noimg;
    CHECK_EQ(whirl::server::pidle::target(Sched{5000, 0, noimg}), 4096u);
    CHECK_EQ(whirl::server::pidle::target(Sched{3040, 0, noimg}), 2048u);
    CHECK_EQ(whirl::server::pidle::target(Sched{1000, 0, noimg}), 0u);
    CHECK_EQ(whirl::server::pidle::target(Sched{1}), 0u);
    // with a system split at 2500: the grid restarts there
    CHECK_EQ(whirl::server::pidle::target(Sched{6000, 2500, noimg}), 5572u);
    {
        // kinds and positions: 'prefix' 2048 on the schedule is the start; gen-end / prompt-end / off-schedule /
        // invalid / at or past the point are not
        std::vector<CkptKey> cks(6);
        cks[0] = {true, 2048, 1, "prefix", false};
        cks[1] = {true, 3000, 2, "prompt-end", true};
        cks[2] = {true, 5000, 3, "gen-end", true};
        cks[3] = {true, 1500, 4, "prefix", false};   // off the 1024 grid
        cks[4] = {false, 3072, 5, "prefix", false};  // invalid
        cks[5] = {true, 4096, 6, "prefix", false};   // = the point: nothing left to do from there
        const Plan pl = whirl::server::pidle::plan(cks, Sched{5000, 0, noimg});
        CHECK(pl.ok());
        CHECK_EQ(pl.ck, 0);
        CHECK_EQ(pl.from, 2048u);
        CHECK_EQ(pl.to, 4096u);
        std::vector<CkptKey> only_ge(1);
        only_ge[0] = {true, 5000, 1, "gen-end", true};
        CHECK(!whirl::server::pidle::plan(only_ge, Sched{5000, 0, noimg}).ok());
        std::vector<CkptKey> sys1(1);
        sys1[0] = {true, 2500, 1, "system", false};  // the system split is a point
        const Plan ps = whirl::server::pidle::plan(sys1, Sched{6000, 2500, noimg});
        CHECK(ps.ok());
        CHECK_EQ(ps.from, 2500u);
        CHECK_EQ(ps.to, 5572u);
    }
    // the chunks from the start to the point: whole schedule chunks, the last one ends at the point
    {
        const Sched s{5000, 0, noimg};
        std::vector<std::size_t> ends;
        for (std::size_t cur = 2048; cur < 4096;) {
            cur += whirl::server::pidle::chunkLen(s, cur, 4096);
            ends.push_back(cur);
        }
        CHECK(ends == std::vector<std::size_t>({3072, 4096}));
    }
    // the lemma (randomized over lengths, system split, image ends before and after n, continuation length)
    std::uint64_t z = 12345, cases = 0, on = 0;
    auto rnd = [&](std::uint64_t m) {
        z ^= z << 13;
        z ^= z >> 7;
        z ^= z << 17;
        return z % m;
    };
    for (int it = 0; it < 4000; ++it) {
        const std::size_t n = 1200 + rnd(9000);
        const std::size_t sys = rnd(3) == 0 ? 2048 + rnd(n - 1100) : 0;
        std::vector<std::uint32_t> imgs_v;  // image ends of the cached tokens (sorted, inside n)
        for (std::size_t k = 0, b = 1024 + rnd(3000); k < 3 && b < n; ++k, b += 500 + rnd(4000))
            if (rnd(2) == 0) imgs_v.push_back(static_cast<std::uint32_t>(b));
        const Sched s{n, sys, imgs_v};
        const std::size_t to = whirl::server::pidle::target(s);
        if (to == 0) continue;
        for (int c = 0; c < 6; ++c) {
            const std::size_t n2 = n + 1 + rnd(c < 3 ? 300 : 3000);
            std::vector<std::uint32_t> imgs2 = imgs_v;
            if (rnd(3) == 0) imgs2.push_back(static_cast<std::uint32_t>(n + rnd(n2 - n)));
            std::erase_if(imgs2, [&](std::uint32_t b) { return b >= n2; });
            const std::size_t think2 = rnd(2) == 0 ? n2 - 1 : 0;
            ++cases;
            if (!whirl::sched::onSchedule(n2, sys, think2, imgs2, to)) continue;
            ++on;
            CHECK(points(n2, sys, think2, imgs2, to) == points(n, sys, 0, imgs_v, to));
        }
    }
    CHECK(on > cases / 2);  // most continuations do have the point (a short new message after an image end may not)
    if (g_verbose) std::printf("  pidle lemma: %llu continuations, %llu with the point on their schedule\n", (unsigned long long)cases,
                               (unsigned long long)on);
}

// Idle trigger, checkpoint used, bitwise: precise, a 3000-token prompt with a 2000-token reply. Without PRECISE-IDLE
// turn 2 resumes at turn 1's tail checkpoint (2048) and prefills 3350 tokens; with it, the reply is re-prefilled while
// idle and turn 2 resumes at 4096 (the start of the last chunk of prompt + reply). Its text equals a cold run, and no
// prefill forward of the server read decode-written history.
void testPidleEngine() {
    using namespace pidle_test;
    for (const bool idle : {true, false}) {
        g_log.clear();
        TestServer srv(preciseSetup(true, idle));
        ServerSetup su_nc = preciseSetup(true, false);
        su_nc.eo.no_prefix_cache = true;
        TestServer cold(su_nc);
        const Turns t = makeTurns(srv, 3000, 2000, 350, 41);
        if (idle) CHECK(waitUntil([&] { return srv.engine().pidleStats().done >= 1; }, 20000));
        else std::this_thread::sleep_for(std::chrono::milliseconds(150));
        auto r2 = srv.post("/v1/completions", completionBody(t.p2, 40));
        auto r2c = cold.post("/v1/completions", completionBody(t.p2, 40));
        CHECK_EQ(r2.status, 200);
        CHECK_EQ(completionText(r2), completionText(r2c));
        CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(t.p2, 40, true)));
        // (turn 2's own reply may already be in its re-prefill: counters from turn 1 on are >=)
        const auto st = srv.engine().pidleStats();
        if (idle) {
            CHECK_EQ(cachedTokens(r2), 4096u);
            CHECK(st.started >= 1u);
            CHECK(st.done >= 1u);
            CHECK(st.rows >= 2048u);
            CHECK_EQ(st.cancelled, 0u);
            CHECK_EQ(g_log.count("precise idle: checkpoint 'prefix' @4096 kept"), 1u);
        } else {
            CHECK_EQ(cachedTokens(r2), coldTail(t.p1.size()));  // as before: turn 1's tail checkpoint
            CHECK_EQ(st.started, 0u);
            CHECK_EQ(g_log.count("precise idle"), 0u);
        }
        CHECK_EQ(srv.model().decPrefills(), 0u);
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
        if (g_verbose) std::printf("  pidle %d: turn 2 cached %llu\n", idle ? 1 : 0, (unsigned long long)cachedTokens(r2));
    }
}

// The mock's decode_mark guard itself (so the test above means something): a precise server that resumed turn 2 at
// the reply end (WHIRL_COLD_RESUME=0 with the mark on) reads decode rows and its text differs from the cold run.
void testPidleGuardBites() {
    using namespace pidle_test;
    ServerSetup su = preciseSetup(true, false);
    su.eo.cold_resume = false;  // longest checkpoint: gen-end
    TestServer srv(su);
    const Turns t = makeTurns(srv, 3000, 2000, 350, 43);
    auto r2 = srv.post("/v1/completions", completionBody(t.p2, 40));
    // gen-end = prompt + reply - 1 (the last sampled token is not forwarded before the reply ends)
    CHECK_EQ(cachedTokens(r2), t.p1.size() + t.out1.size() - 1);
    CHECK(srv.model().decPrefills() >= 1u);
    CHECK(completionText(r2) != decode(srv.model().greedyReference(t.p2, 40, true)));
}

// No change in balance / fast (cold_resume off): even with precise_idle on, nothing runs, turn 2 resumes from gen-end
// as before, and the idle-on and idle-off servers give the same text and the same reuse. sched_grid 0 here: balance's
// gen-end resume is off the cold grid by design (numerically equivalent on the GPU), so the grid guard would flag it;
// without the guard the mock's resumed text equals the reference.
void testPidleBalanceUnchanged() {
    using namespace pidle_test;
    g_log.clear();
    std::vector<std::string> texts;
    std::vector<std::uint64_t> cached;
    for (const bool idle : {true, false}) {
        ServerSetup su = preciseSetup(false, idle);
        su.mc.sched_grid = 0;
        TestServer srv(su);
        const Turns t = makeTurns(srv, 3000, 2000, 350, 45);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        auto r2 = srv.post("/v1/completions", completionBody(t.p2, 40));
        CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(t.p2, 40, true)));
        CHECK_EQ(cachedTokens(r2), t.p1.size() + t.out1.size() - 1);  // gen-end
        texts.push_back(completionText(r2));
        cached.push_back(cachedTokens(r2));
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        const auto st = srv.engine().pidleStats();
        CHECK_EQ(st.started + st.skipped + st.done + st.cancelled + st.rows, 0u);
        CHECK_EQ(srv.model().poison(), 0u);
    }
    CHECK_EQ(texts[0], texts[1]);
    CHECK_EQ(cached[0], cached[1]);
    CHECK_EQ(g_log.count("precise idle"), 0u);
}

// Cancel on arrival: the re-prefill (5 chunks of 1024 rows, 0.3 ms per row) is running when turn 2 arrives. It stops
// after the forward in flight, keeps a checkpoint at the last chunk it finished, and turn 2 equals the cold run.
void testPidleCancel() {
    using namespace pidle_test;
    g_log.clear();
    ServerSetup su = preciseSetup(true, true);
    su.mc.prefill_us_per_row = 300;
    TestServer srv(su);
    const Turns t = makeTurns(srv, 3000, 5000, 300, 47);
    CHECK(waitUntil([&] { return srv.engine().pidleStats().started >= 1; }, 20000));
    auto r2 = srv.post("/v1/completions", completionBody(t.p2, 40));
    CHECK_EQ(r2.status, 200);
    CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(t.p2, 40, true)));
    const auto st = srv.engine().pidleStats();
    CHECK_EQ(st.cancelled, 1u);
    const std::uint64_t cached = cachedTokens(r2);
    // at least the chunk that was running when the request came finished (the loop looks at the queue between chunks)
    CHECK(cached > coldTail(t.p1.size()) && cached < 7168u && cached % 1024 == 0);
    CHECK_EQ(g_log.count(std::format("precise idle: a request arrived; stopped at {} of 2048 .. 7168 (checkpoint kept there)", cached)), 1u);
    CHECK_EQ(g_log.count("precise idle: checkpoint 'prefix' @7168"), 0u);
    CHECK_EQ(srv.model().decPrefills(), 0u);
    CHECK_EQ(srv.model().poison(), 0u);
    if (g_verbose) std::printf("  pidle cancel: %llu rows before the request, turn 2 cached %llu\n", (unsigned long long)st.rows,
                               (unsigned long long)cached);
}

// Checkpoint used only when bitwise-safe: a next prompt that continues only part of the reply, so that 4096 is not a
// point of its schedule (4250 tokens: the cold run cuts 3072 .. 4250 as one chunk), must not resume at 4096 although
// its tokens match there; a regenerate of turn 1 (its prompt-end went with the rewritten rows) resumes at the tail.
// Both equal the cold run.
void testPidleOnlyWhenSafe() {
    using namespace pidle_test;
    TestServer srv(preciseSetup(true, true));
    const Turns t = makeTurns(srv, 3000, 2000, 350, 49);
    CHECK(waitUntil([&] { return srv.engine().pidleStats().done >= 1; }, 20000));
    std::vector<std::uint32_t> part(t.p1);
    part.insert(part.end(), t.out1.begin(), t.out1.begin() + 1200);  // 4200 tokens of turn 1 + reply
    const auto p_off = makePrompt(4250, 51, part);
    auto r = srv.post("/v1/completions", completionBody(p_off, 32));
    CHECK_EQ(completionText(r), decode(srv.model().greedyReference(p_off, 32, true)));
    CHECK_EQ(cachedTokens(r), coldTail(t.p1.size()));  // 2048, not 4096
    CHECK_EQ(srv.model().decPrefills(), 0u);
    CHECK_EQ(srv.model().poison(), 0u);
    // a regenerate of turn 1 in a fresh server after the idle work
    TestServer srv2(preciseSetup(true, true));
    const Turns t2 = makeTurns(srv2, 3000, 2000, 350, 49);
    CHECK(waitUntil([&] { return srv2.engine().pidleStats().done >= 1; }, 20000));
    auto rr = srv2.post("/v1/completions", completionBody(t2.p1, 2000));
    CHECK_EQ(completionText(rr), decode(t2.out1));
    CHECK_EQ(cachedTokens(rr), coldTail(t2.p1.size()));
    CHECK_EQ(srv2.model().decPrefills(), 0u);
}
