// feat026-pidle: PRECISE-IDLE tests (included by server_tests.cpp after the TAIL-CK tests; uses its helpers).
//
// The mock runs with sched_grid 1024 (a prefill forward that starts off the cold grid changes the state) and
// decode_mark (a prefill forward over history a verify wrote changes the state, as a prefill over decode-produced
// history is not bitwise the cold one on the GPU): a turn that resumes from a state the cold run does not have changes
// the text against the greedy reference (= a cold run), and decPrefills() counts the forwards that read decode rows.

namespace pidle_test {

using whirl::server::pidle::Plan;
using whirl::server::pidle::Sched;

// n_ck: checkpoints per slot. 4 = whirl-server's default with --parallel 1 (server_main n_ckpt_default); 3 = the
// smallest set that keeps the tail checkpoint (cache::ckpt_keep_prefix_min). feat026-pidle-r1 (review A1): with 3 the
// idle checkpoint pushes prompt-end out of the set by itself, which hid a missing invalidation; the tests run both.
ServerSetup preciseSetup(bool precise, bool idle, std::uint32_t idle_ms = 0, std::uint32_t n_ck = 4) {
    ServerSetup su = baseSetup();
    su.mc.parallel = 1;
    su.mc.slot_ctx = 32768;
    su.mc.pool_tokens = 65536;
    su.mc.sched_grid = 1024;
    su.mc.decode_mark = precise;  // balance resumes from gen-end by design (numerically equivalent, not bitwise)
    su.eo.n_ck = n_ck;
    su.eo.cold_resume = precise;
    su.eo.precise_idle = idle;
    su.eo.precise_idle_ms = idle_ms;
    return su;
}

std::vector<Engine::SlotViewForTest> slotsView(TestServer& srv) {
    std::vector<Engine::SlotViewForTest> v;
    srv.onEngine([&] { v = srv.engine().slotsForTest(); });
    return v;
}

std::vector<std::uint32_t> encChat(const std::string& s) {
    const auto ids = tok().encode(s, true);
    return std::vector<std::uint32_t>(ids.begin(), ids.end());
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
    for (const std::uint32_t nck : {4u, 3u})
    for (const bool idle : {true, false}) {
        g_log.clear();
        TestServer srv(preciseSetup(true, idle, 0, nck));
        ServerSetup su_nc = preciseSetup(true, false, 0, nck);
        su_nc.eo.no_prefix_cache = true;
        TestServer cold(su_nc);
        const Turns t = makeTurns(srv, 3000, 2000, 350, 41);
        if (idle) {
            CHECK(waitUntil([&] { return srv.engine().pidleStats().done >= 1; }, 20000));
            // feat026-pidle-r1 (review C2): the slot's cached tokens are the rows its KV holds bitwise (0 .. 4096,
            // the idle checkpoint), as after any other rewrite (startJob: cache_tokens cut to the reuse point)
            const auto v = slotsView(srv);
            CHECK_EQ(v[0].tokens, 4096u);
            CHECK_EQ(v[0].max_ck, 4096u);
            CHECK(v[0].pages * qwen35::kv_page >= 4096u);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
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
    // a regenerate of turn 1 in a fresh server after the idle work. feat026-pidle-r1 (review A1): with 4 checkpoints
    // (whirl-server's --parallel 1 default) the idle checkpoint @4096 goes into a free place, so prompt-end @3000 would
    // still be in the set: only the invalidation keeps the regenerate from resuming there over rows [2048, 3000) that
    // the idle prefill rewrote with its own chunk shape (on the GPU not the bits turn 1's cold prefill left)
    for (const std::uint32_t nck : {4u, 3u}) {
        TestServer srv2(preciseSetup(true, true, 0, nck));
        const Turns t2 = makeTurns(srv2, 3000, 2000, 350, 49);
        CHECK(waitUntil([&] { return srv2.engine().pidleStats().done >= 1; }, 20000));
        auto rr = srv2.post("/v1/completions", completionBody(t2.p1, 2000));
        CHECK_EQ(completionText(rr), decode(t2.out1));
        CHECK_EQ(cachedTokens(rr), coldTail(t2.p1.size()));
        CHECK_EQ(srv2.model().decPrefills(), 0u);
        if (g_verbose) std::printf("  pidle regenerate n_ck %u: cached %llu\n", nck, (unsigned long long)cachedTokens(rr));
    }
}

// feat026-pidle-r1 (review A2): the copy on write of the re-prefill finds no page (every page of the pool is the
// slot's and shared, no other slot to evict). The idle job must not throw out of runLoop (KvPoolFull ended the server
// as "fatal"): it is skipped before any checkpoint is invalidated, and the next turn resumes at turn 1's tail.
void testPidleNoPageNoThrow() {
    using namespace pidle_test;
    g_log.clear();
    ServerSetup su = preciseSetup(true, true, 300);
    su.mc.pool_tokens = 5120;  // 20 pages: turn 1 (3000 + 1900 tokens) fills the pool
    TestServer srv(su);
    const Turns t = makeTurns(srv, 3000, 1900, 60, 53);
    srv.onEngine([&] { srv.engine().holdPagesForTest(true); });  // the slot's pages count as shared: cow needed
    CHECK(waitUntil([&] {
        const auto st = srv.engine().pidleStats();
        return st.skipped + st.started >= 1;
    }, 5000));
    srv.onEngine([&] { srv.engine().holdPagesForTest(false); });
    const auto st = srv.engine().pidleStats();
    CHECK_EQ(st.started, 0u);
    CHECK_EQ(st.skipped, 1u);
    CHECK_EQ(g_log.count("precise idle: skipped: needs "), 1u);
    CHECK_EQ(srv.get("/health").status, 200);
    auto r2 = srv.post("/v1/completions", completionBody(t.p2, 40));
    CHECK_EQ(r2.status, 200);
    CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(t.p2, 40, true)));
    CHECK_EQ(cachedTokens(r2), coldTail(t.p1.size()));  // turn 1's checkpoints were left as they were
    CHECK_EQ(srv.model().poison(), 0u);
}

// feat026-pidle-r1 (review C3): --parallel 2, conversation A idle in one slot, B's reply just ended in the other, and
// the pool too full for B's re-prefill. The speculative background job must never evict A's cache (it would, through
// the slot evictor of cowPages / ensurePages): B's re-prefill is skipped, A's and B's next turns resume at their tails.
void testPidleNoEvict() {
    using namespace pidle_test;
    g_log.clear();
    ServerSetup su = preciseSetup(true, true, 300);
    su.mc.parallel = 2;
    su.mc.pool_tokens = 9216;  // 36 pages: A (3032 tokens, 12 pages) + B (4900 tokens, 20 pages) + 4 free
    TestServer srv(su);
    const auto pa = makePrompt(3000, 61);
    auto ra = srv.post("/v1/completions", completionBody(pa, 32));
    const auto outa = srv.model().greedyReference(pa, 32, true);
    CHECK_EQ(completionText(ra), decode(outa));
    const Turns tb = makeTurns(srv, 3000, 1900, 60, 63);
    srv.onEngine([&] { srv.engine().holdPagesForTest(true); });  // B's rewrite needs copy on write: 9 pages, 4 free
    CHECK(waitUntil([&] {
        const auto st = srv.engine().pidleStats();
        return st.skipped >= 2 || st.started >= 1;  // A: nothing to do (3032 tokens); B: no free pages
    }, 5000));
    srv.onEngine([&] { srv.engine().holdPagesForTest(false); });
    const auto st = srv.engine().pidleStats();
    CHECK_EQ(st.started, 0u);
    CHECK_EQ(g_log.count("kv pool full: evicting idle slot"), 0u);
    CHECK_EQ(srv.engine().stats().evictions, 0u);
    std::vector<std::uint32_t> ha = pa;
    ha.insert(ha.end(), outa.begin(), outa.end());
    const auto pa2 = makePrompt(static_cast<std::uint32_t>(ha.size() + 200), 62, ha);
    auto ra2 = srv.post("/v1/completions", completionBody(pa2, 24));
    CHECK_EQ(completionText(ra2), decode(srv.model().greedyReference(pa2, 24, true)));
    CHECK_EQ(cachedTokens(ra2), coldTail(pa.size()));
    auto rb2 = srv.post("/v1/completions", completionBody(tb.p2, 24));
    CHECK_EQ(completionText(rb2), decode(srv.model().greedyReference(tb.p2, 24, true)));
    CHECK_EQ(cachedTokens(rb2), coldTail(tb.p1.size()));
    CHECK_EQ(srv.model().poison(), 0u);
}

// feat026-pidle-r1 (review A3): a reply to a prompt that opens a thinking block ("<think>\n" at its end). The chat
// template drops the reasoning from the history, so the next prompt does not hold the reply tokens and an idle
// checkpoint in them is never used; the re-prefill would only cost GPU time and invalidate think-open / gen-end (a
// regenerate would prefill again from the tail). It is skipped, and the regenerate resumes at think-open (N - 1).
// The reference is a cold server (no prefix cache): the cold run of a think-open prompt splits at N - 1, which the
// mock's sched_grid guard marks (off the 1024 grid), so the never-splitting greedyReference is not the cold text here.
void testPidleThinkSkipped() {
    using namespace pidle_test;
    g_log.clear();
    TestServer srv(preciseSetup(true, true));
    ServerSetup su_nc = preciseSetup(true, false);
    su_nc.eo.no_prefix_cache = true;
    TestServer cold(su_nc);
    std::vector<std::uint32_t> p1 = makePrompt(2990, 65);
    const auto gen = encChat("<|im_start|>assistant\n<think>\n");
    p1.insert(p1.end(), gen.begin(), gen.end());
    CHECK_EQ(ChatTokens(tok()).thinkOpenSplit(p1), p1.size() - 1);
    auto r1 = srv.post("/v1/completions", completionBody(p1, 2000));
    const std::string t_cold = completionText(cold.post("/v1/completions", completionBody(p1, 2000)));
    CHECK_EQ(completionText(r1), t_cold);
    CHECK(waitUntil([&] {
        const auto st = srv.engine().pidleStats();
        return st.skipped + st.started >= 1;
    }, 5000));
    const auto st = srv.engine().pidleStats();
    CHECK_EQ(st.started, 0u);
    CHECK_EQ(st.skipped, 1u);
    CHECK_EQ(g_log.count("precise idle: skipped: the prompt opens a thinking block"), 1u);
    auto rr = srv.post("/v1/completions", completionBody(p1, 2000));
    CHECK_EQ(completionText(rr), t_cold);
    CHECK_EQ(cachedTokens(rr), p1.size() - 1);  // think-open of turn 1, not the tail
    CHECK_EQ(srv.model().decPrefills(), 0u);
    CHECK_EQ(srv.model().poison(), 0u);
}

// feat026-pidle-r1, diagnostic found while writing the test above: a --precise thinking turn whose prompt resumes from
// an earlier 'prefix' checkpoint (the tail of a previous prompt) and ends with "<think>\n" must equal the cold run,
// whose last chunk is split at the think-open point (N - 1). Idle off (not about PRECISE-IDLE: the COLD-RESUME path).
void testPidleThinkResumeCold() {
    using namespace pidle_test;
    TestServer srv(preciseSetup(true, false));
    ServerSetup su_nc = preciseSetup(true, false);
    su_nc.eo.no_prefix_cache = true;
    TestServer cold(su_nc);
    const auto pa = makePrompt(3000, 67);
    auto ra = srv.post("/v1/completions", completionBody(pa, 16));
    CHECK_EQ(ra.status, 200);
    std::vector<std::uint32_t> p2 = makePrompt(3200, 68, pa);
    const auto gen = encChat("<|im_start|>assistant\n<think>\n");
    p2.insert(p2.end(), gen.begin(), gen.end());
    auto r2 = srv.post("/v1/completions", completionBody(p2, 64));
    const std::string t_cold = completionText(cold.post("/v1/completions", completionBody(p2, 64)));
    if (g_verbose) std::printf("  think resume: cached %llu of %zu, text == cold: %d\n", (unsigned long long)cachedTokens(r2), p2.size(),
                               completionText(r2) == t_cold ? 1 : 0);
    CHECK(cachedTokens(r2) > 0);
    CHECK_EQ(completionText(r2), t_cold);
}

// ---------------------------------------------------------------------------
// feat026-pidle-r2: thinking replies (research/pidle/2026-10-26-think-template-reuse.md). With thinking on the next
// prompt holds the reply only when the client sends the reasoning back (reasoning_content); the prompt's history
// tells: an earlier assistant turn with a non-empty think block -> re-prefill; empty blocks only / first turn -> skip;
// a variant-a request with preserve_thinking=false -> skip (the template strips the earlier blocks).

namespace pidle_test {

// "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content + "<|im_end|>\n<|im_start|>user\n"
// as the templates render an earlier assistant turn (reasoning "" = the empty block "<think>\n\n</think>\n\n")
std::vector<std::uint32_t> pastTurn(const std::string& reasoning, const std::string& content) {
    return encChat("<|im_end|>\n<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content +
                   "<|im_end|>\n<|im_start|>user\n");
}

// a thinking-mode completion prompt of about n tokens: 1000 tokens of turn 1, the earlier assistant turn, filler,
// the generation prompt "<|im_end|>\n<|im_start|>assistant\n<think>\n"
std::vector<std::uint32_t> thinkPrompt(std::uint32_t n, std::uint64_t seed, const std::vector<std::uint32_t>& past) {
    std::vector<std::uint32_t> p = makePrompt(1000, seed);
    p.insert(p.end(), past.begin(), past.end());
    p = makePrompt(n, seed + 1, p);
    const auto gen = encChat("<|im_end|>\n<|im_start|>assistant\n<think>\n");
    p.insert(p.end(), gen.begin(), gen.end());
    return p;
}

std::string chatChoices(const HttpResult& r) {
    const std::size_t a = r.body.find("\"choices\""), b = r.body.find("\"usage\"");
    return a == std::string::npos || b == std::string::npos || b < a ? std::string("<none>") : r.body.substr(a, b - a);
}

std::uint64_t promptTokens(const HttpResult& r) {
    const json::Value v = parseBody(r);
    const json::Value* u = v.get("usage");
    const json::Value* p = u ? u->get("prompt_tokens") : nullptr;
    return p && p->kind() == json::Value::Kind::integer ? static_cast<std::uint64_t>(p->asInt()) : 0;
}

}  // namespace pidle_test

// ChatTokens::reasoningInHistory on rendered text (the real vocabulary: "\n\n", "</think>", "assistant" are one token)
void testPidleReasoningInHistory() {
    using namespace pidle_test;
    const ChatTokens ct(tok());
    CHECK(ct.think_end != ChatTokens::none && ct.nl2 != ChatTokens::none && ct.assistant != ChatTokens::none);
    const std::string head = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi";
    const std::string gen = "<|im_end|>\n<|im_start|>assistant\n<think>\n";
    const std::string past_r =
        "<|im_end|>\n<|im_start|>assistant\n<think>\nI weigh the options.\n</think>\n\nDone.<|im_end|>\n<|im_start|>user\nnext";
    const std::string past_e = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nDone.<|im_end|>\n<|im_start|>user\nnext";
    // first turn: only the generation prompt's own block
    CHECK(!ct.reasoningInHistory(encChat(head + gen)));
    // the client sent the reasoning back
    CHECK(ct.reasoningInHistory(encChat(head + past_r + gen)));
    // the client dropped it: empty blocks only (one or two earlier turns)
    CHECK(!ct.reasoningInHistory(encChat(head + past_e + gen)));
    CHECK(!ct.reasoningInHistory(encChat(head + past_e + past_e + gen)));
    // any earlier turn with reasoning counts
    CHECK(ct.reasoningInHistory(encChat(head + past_e + past_r + gen)));
    // not a thinking-mode prompt: false whatever the history holds
    CHECK(!ct.reasoningInHistory(encChat(head + past_r + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")));
    // a past turn rendered without any block (preserve_thinking=false) is no reasoning
    CHECK(!ct.reasoningInHistory(encChat(head + "<|im_end|>\n<|im_start|>assistant\nDone.<|im_end|>\n<|im_start|>user\nnext" + gen)));
}

// Thinking on, the client sends the reasoning back: turn 1's history carries a non-empty think block, so its reply
// is re-prefilled while idle and turn 2 (= turn 1 + the reply as generated + new tokens: the FULL-prefix case)
// resumes at the idle checkpoint 4096 instead of an earlier point of turn 1; its text equals a cold server.
void testPidleThinkRoundTrip() {
    using namespace pidle_test;
    g_log.clear();
    TestServer srv(preciseSetup(true, true));
    TestServer ref(preciseSetup(true, false));  // the same server without the idle job: the reply tokens
    ServerSetup su_nc = preciseSetup(true, false);
    su_nc.eo.no_prefix_cache = true;
    TestServer cold(su_nc);
    const std::vector<std::uint32_t> p1 = thinkPrompt(2990, 71, pastTurn("I weigh the options.", "Done."));
    CHECK(ChatTokens(tok()).reasoningInHistory(p1));
    auto r1 = srv.post("/v1/completions", completionBody(p1, 2000));
    auto r1r = ref.post("/v1/completions", completionBody(p1, 2000));
    CHECK_EQ(completionText(r1), completionText(r1r));
    CHECK_EQ(completionText(r1), completionText(cold.post("/v1/completions", completionBody(p1, 2000))));
    CHECK(waitUntil([&] { return srv.engine().pidleStats().done >= 1; }, 20000));
    CHECK_EQ(g_log.count("precise idle: thinking reply kept: the history carries reasoning"), 1u);
    CHECK_EQ(g_log.count("precise idle: skipped"), 0u);
    const std::vector<std::uint32_t> held = slotsView(ref)[0].ids;  // turn 1 + reply (the rows the KV holds)
    CHECK(held.size() > p1.size() + 1000 && std::equal(p1.begin(), p1.end(), held.begin()));
    const auto v = slotsView(srv);
    CHECK_EQ(v[0].max_ck, 4096u);
    const std::vector<std::uint32_t> p2 = makePrompt(static_cast<std::uint32_t>(held.size() + 350), 72, held);
    auto r2 = srv.post("/v1/completions", completionBody(p2, 40));
    auto r2r = ref.post("/v1/completions", completionBody(p2, 40));
    const std::string t_cold = completionText(cold.post("/v1/completions", completionBody(p2, 40)));
    CHECK_EQ(completionText(r2), t_cold);
    CHECK_EQ(completionText(r2r), t_cold);
    CHECK_EQ(cachedTokens(r2), 4096u);           // the idle checkpoint
    CHECK(cachedTokens(r2r) < cachedTokens(r2));  // without it: an earlier point of turn 1
    if (g_verbose)
        std::printf("  think round trip: p1 %zu, held %zu, turn 2 cached %llu (without idle %llu)\n", p1.size(), held.size(),
                    (unsigned long long)cachedTokens(r2), (unsigned long long)cachedTokens(r2r));
    CHECK_EQ(srv.model().decPrefills(), 0u);
    CHECK_EQ(srv.model().poison(), 0u);
    CHECK_EQ(srv.model().hidMismatch(), 0u);
}

// Thinking on, the client drops the reasoning: the history has only empty think blocks, so the reply is not
// re-prefilled; turn 2 (= turn 1's prompt up to "<think>", the reply re-rendered as "\n\n</think>\n\n" + content, new
// tokens) resumes where the same server without PRECISE-IDLE resumes (at most turn 1's think-open point N - 1), and
// equals a cold server.
void testPidleThinkDropped() {
    using namespace pidle_test;
    g_log.clear();
    TestServer srv(preciseSetup(true, true));
    ServerSetup su_nc = preciseSetup(true, false);
    su_nc.eo.no_prefix_cache = true;
    TestServer cold(su_nc);
    const std::vector<std::uint32_t> p1 = thinkPrompt(2990, 73, pastTurn("", "Done."));
    CHECK(!ChatTokens(tok()).reasoningInHistory(p1));
    auto r1 = srv.post("/v1/completions", completionBody(p1, 2000));
    CHECK_EQ(completionText(r1), completionText(cold.post("/v1/completions", completionBody(p1, 2000))));
    CHECK(waitUntil([&] {
        const auto st = srv.engine().pidleStats();
        return st.skipped + st.started >= 1;
    }, 5000));
    const auto st = srv.engine().pidleStats();
    CHECK_EQ(st.started, 0u);
    CHECK_EQ(st.skipped, 1u);
    CHECK_EQ(g_log.count("precise idle: skipped: the prompt opens a thinking block and its history carries no reasoning"), 1u);
    std::vector<std::uint32_t> p2(p1.begin(), p1.end() - 1);
    const auto re = encChat("\n\n</think>\n\nThe answer.<|im_end|>\n<|im_start|>user\n");
    p2.insert(p2.end(), re.begin(), re.end());
    p2 = makePrompt(static_cast<std::uint32_t>(p2.size() + 300), 74, p2);
    // the same two turns on the same server without the idle job: the reuse must be the same (nothing was rewritten)
    TestServer ref(preciseSetup(true, false));
    CHECK_EQ(ref.post("/v1/completions", completionBody(p1, 2000)).status, 200);
    auto r2 = srv.post("/v1/completions", completionBody(p2, 40));
    auto r2r = ref.post("/v1/completions", completionBody(p2, 40));
    const std::string t_cold = completionText(cold.post("/v1/completions", completionBody(p2, 40)));
    CHECK_EQ(completionText(r2), t_cold);
    CHECK_EQ(completionText(r2r), t_cold);
    CHECK_EQ(cachedTokens(r2), cachedTokens(r2r));
    CHECK(cachedTokens(r2) > 0 && cachedTokens(r2) <= p1.size() - 1);  // at most turn 1's think-open point
    if (g_verbose)
        std::printf("  think dropped: p1 %zu, turn 2 cached %llu (without idle %llu)\n", p1.size(),
                    (unsigned long long)cachedTokens(r2), (unsigned long long)cachedTokens(r2r));
    CHECK_EQ(srv.model().decPrefills(), 0u);
    CHECK_EQ(srv.model().poison(), 0u);
}

// The chat path: preserve_thinking=false on variant a (Swift) strips the earlier think blocks, so every reply is
// skipped (thinking on or off); variant b (Ornith) ignores the flag and keeps the blocks (the reasoning sent back:
// kept); preserve_thinking=true with reasoning_content: kept; no reasoning_content: skipped. A skipped request,
// repeated, resumes at its think-open point (prompt_tokens - 1) / at most its prompt-end (thinking off).
void testPidlePreserveThinking() {
    using namespace pidle_test;
    std::string long1;
    for (int i = 0; i < 2600; ++i) long1 += std::format("w{} ", i % 97);
    struct Case {
        chat::TemplateKind kind;
        bool think;
        const char* preserve;  // chat_template_kwargs preserve_thinking ("" = not given)
        bool rc;               // the earlier assistant turn carries reasoning_content
        bool kept;
        const char* log;
    };
    const Case cases[] = {
        {chat::TemplateKind::a, true, "false", true, false, "precise idle: skipped: preserve_thinking=false"},
        {chat::TemplateKind::a, false, "false", true, false, "precise idle: skipped: preserve_thinking=false"},
        {chat::TemplateKind::b, true, "false", true, true, "precise idle: thinking reply kept"},
        {chat::TemplateKind::a, true, "true", true, true, "precise idle: thinking reply kept"},
        {chat::TemplateKind::a, true, "", false, false, "its history carries no reasoning"},
    };
    for (const Case& c : cases) {
        g_log.clear();
        ServerSetup su = preciseSetup(true, true);
        su.eo.tmpl = c.kind;
        TestServer srv(su);
        ServerSetup su_nc = preciseSetup(true, false);
        su_nc.eo.tmpl = c.kind;
        su_nc.eo.no_prefix_cache = true;
        TestServer cold(su_nc);
        std::string kw = std::format("\"enable_thinking\":{}", c.think ? "true" : "false");
        if (*c.preserve) kw += std::format(",\"preserve_thinking\":{}", c.preserve);
        const std::string rc = c.rc ? R"(,"reasoning_content":"I weigh the options.")" : "";
        const std::string body = std::format(
            R"({{"messages":[{{"role":"user","content":"{}"}},{{"role":"assistant","content":"Done."{}}},)"
            R"({{"role":"user","content":"next"}}],"max_tokens":1500,"temperature":0,"ignore_eos":true,)"
            R"("chat_template_kwargs":{{{}}}}})",
            long1, rc, kw);
        auto r1 = srv.post("/v1/chat/completions", body);
        CHECK_EQ(r1.status, 200);
        CHECK_EQ(chatChoices(r1), chatChoices(cold.post("/v1/chat/completions", body)));
        const std::uint64_t np = promptTokens(r1);
        CHECK(np > 2600);
        CHECK(waitUntil([&] {
            const auto st = srv.engine().pidleStats();
            return st.skipped >= 1 || st.done >= 1;
        }, 20000));
        const auto st = srv.engine().pidleStats();
        CHECK_EQ(g_log.count(c.log), 1u);
        CHECK_EQ(st.done, c.kept ? 1u : 0u);
        CHECK_EQ(st.skipped, c.kept ? 0u : 1u);
        auto rr = srv.post("/v1/chat/completions", body);
        CHECK_EQ(chatChoices(rr), chatChoices(r1));
        if (!c.kept && c.think) CHECK_EQ(cachedTokens(rr), np - 1);  // turn 1's think-open checkpoint
        if (!c.kept && !c.think) CHECK(cachedTokens(rr) >= coldTail(np) && cachedTokens(rr) <= np);
        if (g_verbose)
            std::printf("  preserve kind %c think %d preserve '%s' rc %d: prompt %llu, done %llu skipped %llu, repeat cached %llu\n",
                        c.kind == chat::TemplateKind::a ? 'a' : 'b', c.think ? 1 : 0, c.preserve, c.rc ? 1 : 0,
                        (unsigned long long)np, (unsigned long long)st.done, (unsigned long long)st.skipped,
                        (unsigned long long)cachedTokens(rr));
        CHECK_EQ(srv.model().decPrefills(), 0u);
        CHECK_EQ(srv.model().poison(), 0u);
    }
}

// feat026-pidle-r2 (review r1-batch-b): a throw while the idle job is being set up (after its checkpoints were
// invalidated, before it is active; here: the checkpoint load) drops the slot's cache, so the slot never claims rows
// its KV does not hold; the server goes on and the next turn equals the greedy reference (= a cold run).
void testPidleLoadThrow() {
    using namespace pidle_test;
    g_log.clear();
    TestServer srv(preciseSetup(true, true));
    srv.onEngine([&] { srv.engine().failPidleLoadForTest(); });
    const Turns t = makeTurns(srv, 3000, 2000, 350, 81);
    CHECK(waitUntil([&] { return g_log.count("precise idle: stopped: test: checkpoint load failed") >= 1; }, 20000));
    const auto v = slotsView(srv);
    CHECK_EQ(v[0].tokens, 0u);  // the cache went (not 4999 tokens over checkpoints invalidated above 2048)
    CHECK_EQ(v[0].max_ck, 0u);
    const auto st = srv.engine().pidleStats();
    CHECK_EQ(st.done, 0u);
    auto r2 = srv.post("/v1/completions", completionBody(t.p2, 40));
    CHECK_EQ(r2.status, 200);
    CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(t.p2, 40, true)));
    if (g_verbose) std::printf("  load throw: turn 2 cached %llu\n", (unsigned long long)cachedTokens(r2));
    CHECK_EQ(srv.model().decPrefills(), 0u);
    CHECK_EQ(srv.model().poison(), 0u);
}

// feat026-pidle-r3 (item 1): Swift preserve_thinking=false strips the think blocks of the earlier turns, but not of
// the assistant messages after the last REAL user query (a tool response is not one). So a reply that ends in a tool
// call is rendered again with its think block: the next prompt holds the reply as generated. A final answer is not.
void testPidleToolEndRender() {
    using namespace pidle_test;
    const auto render = [](const std::string& msgs, bool preserve) {
        const std::string kw = std::string("\"chat_template_kwargs\":{\"enable_thinking\":true,\"preserve_thinking\":") +
                               (preserve ? "true" : "false") + "}";
        chat::ChatRequest req = chat::parseChatRequest(json::parse("{" + msgs + "," + kw + "}"));
        return chat::renderChat(chat::TemplateKind::a, req.messages, req.tools, req.options);
    };
    const std::string q = R"({"role":"user","content":"find the file"})";
    const auto call = [](const char* rc, const char* id) {
        return std::string(R"({"role":"assistant","content":"","reasoning_content":")") + rc +
               R"(","tool_calls":[{"id":")" + id + R"(","type":"function","function":{"name":"f","arguments":"{\"a\":\"x\"}"}}]})";
    };
    const auto tool = [](const char* id) {
        return std::string(R"({"role":"tool","tool_call_id":")") + id + R"(","content":"ok"})";
    };
    const std::string t1 = render("\"messages\":[" + q + "]", false);
    CHECK(t1.size() > 8 && t1.compare(t1.size() - 8, 8, "<think>\n") == 0);
    const std::string reply1 = "plan one\n</think>\n\n<tool_call>\n<function=f>\n<parameter=a>\nx\n</parameter>\n</function>\n</tool_call>";
    const std::string t2 = render("\"messages\":[" + q + "," + call("plan one", "c1") + "," + tool("c1") + "]", false);
    CHECK(t2.rfind(t1 + reply1 + "<|im_end|>\n", 0) == 0);  // step 1 = turn 1 + the reply as generated
    // step 2 of the same user turn: still after the last real query
    const std::string reply2 = "plan two\n</think>\n\n<tool_call>\n<function=f>\n<parameter=a>\nx\n</parameter>\n</function>\n</tool_call>";
    const std::string t3 = render("\"messages\":[" + q + "," + call("plan one", "c1") + "," + tool("c1") + "," +
                                      call("plan two", "c2") + "," + tool("c2") + "]", false);
    CHECK(t3.rfind(t2 + reply2 + "<|im_end|>\n", 0) == 0);
    // a final answer: the next prompt has a new user query, the block of the answer is stripped (preserve false) /
    // kept (preserve true)
    const std::string ans = "think it\n</think>\n\nDone.";
    const std::string fin = R"({"role":"assistant","content":"Done.","reasoning_content":"think it"})";
    const std::string nxt = R"({"role":"user","content":"next"})";
    const std::string tf1 = render("\"messages\":[" + q + "]", false);
    const std::string tf2f = render("\"messages\":[" + q + "," + fin + "," + nxt + "]", false);
    const std::string tf2t = render("\"messages\":[" + q + "," + fin + "," + nxt + "]", true);
    CHECK(tf2f.rfind(tf1 + ans + "<|im_end|>\n", 0) != 0);
    CHECK(tf2t.rfind(tf1 + ans + "<|im_end|>\n", 0) == 0);
}

// feat026-pidle-r3 (item 1) on the engine: a reply that ends in a tool call is not skipped for preserve_thinking=false
// (thinking off: the block is empty in every prompt; thinking on: an assistant message after the last user query
// carries reasoning, as from the second tool step on). A reply that does not end in a tool call still is.
void testPidleToolEnd() {
    using namespace pidle_test;
    std::string long1;
    for (int i = 0; i < 2600; ++i) long1 += std::format("w{} ", i % 97);
    struct Case {
        bool think, tool_end, kept;
    };
    const Case cases[] = {{false, true, true}, {true, true, true}, {true, false, false}, {false, false, false}};
    for (const Case& c : cases) {
        g_log.clear();
        TestServer srv(preciseSetup(true, true));
        srv.onEngine([&] { srv.engine().toolEndForTest(c.tool_end); });
        const std::string body = std::format(
            R"({{"messages":[{{"role":"user","content":"{}"}},{{"role":"assistant","content":"","reasoning_content":"I weigh the options.",)"
            R"("tool_calls":[{{"id":"c1","type":"function","function":{{"name":"f","arguments":"{{}}"}}}}]}},)"
            R"({{"role":"tool","tool_call_id":"c1","content":"ok"}}],"max_tokens":1500,"temperature":0,"ignore_eos":true,)"
            R"("chat_template_kwargs":{{"enable_thinking":{},"preserve_thinking":false}}}})",
            long1, c.think ? "true" : "false");
        auto r1 = srv.post("/v1/chat/completions", body);
        CHECK_EQ(r1.status, 200);
        CHECK(waitUntil([&] {
            const auto st = srv.engine().pidleStats();
            return st.skipped >= 1 || st.done >= 1;
        }, 20000));
        const auto st = srv.engine().pidleStats();
        CHECK_EQ(st.done, c.kept ? 1u : 0u);
        CHECK_EQ(st.skipped, c.kept ? 0u : 1u);
        CHECK_EQ(g_log.count("precise idle: skipped: preserve_thinking=false"), c.kept ? 0u : 1u);
        CHECK_EQ(srv.model().decPrefills(), 0u);
        CHECK_EQ(srv.model().poison(), 0u);
    }
}

// feat026-pidle-r3 (item 2, mutant N8): the page budget counts freeable pages, not reclaimable ones. Every free page
// becomes a quarantined page another holder keeps (it never becomes free): the job is skipped by the budget check
// ("needs"), not started and failed later in the copy on write.
void testPidleBudgetFreeable() {
    using namespace pidle_test;
    g_log.clear();
    ServerSetup su = preciseSetup(true, true, 300);
    su.mc.pool_tokens = 12288;  // 48 pages: turn 1 (3000 + 1900 tokens) holds 20
    TestServer srv(su);
    const Turns t = makeTurns(srv, 3000, 1900, 60, 91);
    srv.onEngine([&] {
        srv.engine().quarantineFreeHeldForTest();  // reclaimable() 28, freeable() 0
        srv.engine().holdPagesForTest(true);       // the rewrite needs copy on write
    });
    CHECK(waitUntil([&] {
        const auto st = srv.engine().pidleStats();
        return st.skipped + st.started >= 1;
    }, 5000));
    srv.onEngine([&] { srv.engine().holdPagesForTest(false); });
    const auto st = srv.engine().pidleStats();
    CHECK_EQ(st.started, 0u);
    CHECK_EQ(st.skipped, 1u);
    CHECK_EQ(g_log.count("precise idle: skipped: needs "), 1u);
    CHECK_EQ(g_log.count("for the copy on write"), 0u);
    auto r2 = srv.post("/v1/completions", completionBody(t.p2, 40));
    CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(t.p2, 40, true)));
    CHECK_EQ(cachedTokens(r2), coldTail(t.p1.size()));
    CHECK_EQ(srv.model().poison(), 0u);
}

// feat026-pidle-r3 (item 3): taking the pages up to `to` fails part way (two free pages are left after the budget
// check): the pages already taken go back, the slot keeps exactly the pages of the rows it still has.
void testPidleEnsureGivesBack() {
    using namespace pidle_test;
    g_log.clear();
    TestServer srv(preciseSetup(true, true, 300));
    const Turns t = makeTurns(srv, 3000, 1900, 60, 93);
    srv.onEngine([&] {
        srv.engine().holdPagesForTest(true);  // copy on write: the pages after row from - 1 are released, then re-taken
        srv.engine().stealFreePagesForTest(2);
    });
    CHECK(waitUntil([&] { return g_log.count("precise idle: skipped: no free KV page for rows up to") >= 1; }, 5000));
    const auto v = slotsView(srv);
    CHECK_EQ(v[0].pages, PagePool::pagesFor(v[0].tokens));  // no page without rows
    std::size_t free_in = 0;
    srv.onEngine([&] { free_in = srv.engine().freePagesForTest(); });
    CHECK_EQ(free_in, 2u);  // the two pages taken were given back
    srv.onEngine([&] { srv.engine().holdPagesForTest(false); });
    auto r2 = srv.post("/v1/completions", completionBody(t.p2, 40));
    CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(t.p2, 40, true)));
    CHECK_EQ(srv.model().poison(), 0u);
}
