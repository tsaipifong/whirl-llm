// v0252-swe1 (PLAN V1-93 / V1-136): the E1 one-cycle lookahead for DENSE models (Swift: automatic draft count) in DeltaNet
// snapshot mode (gfx1151 / Radeon 8060S, the select verify of kernels/gdn_select.h: a cycle behind a cycle in flight takes at
// most that cycle's draft count m). "Clamp instead of drain": when the policy asks for more than m, the cycle behind is
// submitted with m drafts; the growth is paid with a drain (the next cycle starts a new pipeline at the policy's count) only
// once the clamps have cost more than one drain (spec::rules::aheadDrafts / aheadClampLoss, EngineOptions::e1_grow). Included by
// server_tests.cpp after e1_fast_tests.inc (it reuses e1ahead / e1fast / e1test).
//   e1_swift_rule   the pure rules: clamp value, drain at e1_grow 0 (the old rule), debt accumulation and reset, m 0, a
//                   missing timing estimate, an exploration-up wish (scored lower) never drains, the line format
//   e1_swift_ahead  snapshot mode + select, automatic draft count: WHIRL_E1=2 (e1_grow 0 = old drain / default / no growth
//                   drain) against 0 - text and tokens equal to the greedy reference; clamps happen and replace the drains;
//                   with no growth drain a request runs ONE pipeline; replay mode (gfx1201) never clamps; fast (relaxacc on the
//                   device): runs, is repeatable, check mode 0 mismatches
//   e1_swift_log    the per-request line "spec: cycles by drafts ...; verify ms by drafts ...; lookahead clamped C, growth drains
//                   G": its counts add up to the request's verify cycles (the MTP line), E1=0 and E1=2
// SPDX-License-Identifier: Apache-2.0

namespace e1swift {

// "req N | slot S | spec: cycles by drafts 1:3 4:120; verify ms by drafts 1:0.12 4:0.30; lookahead clamped 5, growth drains 1"
struct SpecLine {
    std::uint64_t cycles = 0, clamped = 0, grown = 0;
    std::size_t ms_items = 0, cyc_items = 0;
    bool ok = false;
};

SpecLine parseSpecLine(const std::string& l) {
    SpecLine s;
    const std::size_t a = l.find("spec: cycles by drafts ");
    const std::size_t b = l.find("; verify ms by drafts ");
    const std::size_t c = l.find("; lookahead clamped ");
    const std::size_t d = l.find(", growth drains ");
    if (a == std::string::npos || b == std::string::npos || c == std::string::npos || d == std::string::npos || !(a < b && b < c && c < d)) return s;
    std::istringstream cyc(l.substr(a + 23, b - (a + 23)));
    std::string item;
    while (cyc >> item) {
        const std::size_t k = item.find(':');
        if (k == std::string::npos) return s;
        s.cycles += std::stoull(item.substr(k + 1));
        s.cyc_items += 1;
    }
    std::istringstream ms(l.substr(b + 22, c - (b + 22)));
    while (ms >> item) {
        if (item.find(':') == std::string::npos) return s;
        s.ms_items += 1;
    }
    s.clamped = std::stoull(l.substr(c + 20, d - (c + 20)));
    s.grown = std::stoull(l.substr(d + 16));
    s.ok = true;
    return s;
}

// the request's verify cycles from "MTP: drafted .. (N verify cycles)"
std::uint64_t mtpCycles(const std::string& l) {
    const std::size_t e = l.find(" verify cycles)");
    const std::size_t s = l.rfind('(', e);
    if (e == std::string::npos || s == std::string::npos) return ~0ull;
    return std::stoull(l.substr(s + 1, e - s - 1));
}

std::vector<std::string> linesWith(std::string_view needle) {
    std::vector<std::string> out;
    std::lock_guard<std::mutex> lk(g_log.mu);
    for (const auto& l : g_log.lines)
        if (l.find(needle) != std::string::npos) out.push_back(l);
    return out;
}

// a Swift-like server: one slot, automatic draft count up to `cap` (the cost model of qwen35::pickDrafts), no n-gram drafts
ServerSetup setup(std::uint32_t e1, std::uint32_t cap, bool snap, std::uint32_t spec_acc = 0) {
    e1ahead::g_snap = snap;
    ServerSetup su = e1ahead::setup(e1, cap, spec_acc);
    e1ahead::g_snap = false;
    su.eo.mtp_auto = true;
    su.eo.n_draft = cap;
    su.mc.draft_miss = 0.15;  // acceptance about 0.85 per draft: the policy's best count sits near the cap and moves
    return su;
}

}  // namespace e1swift

void testE1SwiftRule() {
    namespace sr = spec::rules;
    // ---- aheadDrafts: the policy's wish `want` behind a cycle in flight with m drafts
    for (std::uint32_t m = 0; m <= 8; ++m)
        for (std::uint32_t want = 0; want <= 10; ++want)
            for (const float debt : {0.0f, 0.005f, 0.019f, 0.5f})
                for (const float loss : {0.0f, 0.001f, 0.01f, 0.02f, 0.3f})
                    for (const float grow : {0.0f, 0.02f, 1e9f}) {
                        const sr::AheadDrafts a = sr::aheadDrafts(want, m, debt, loss, grow);
                        if (want <= m) {
                            // no growth: the policy's count, the debt cleared
                            CHECK(a.nd == want && !a.drain && a.debt == 0.0f);
                        } else if (m == 0) {
                            CHECK(a.drain && a.nd == want && a.debt == 0.0f);  // nothing to clamp to
                        } else if (debt + loss >= grow) {
                            // the clamps cost one drain: drain (the next cycle starts a pipeline at `want`), debt paid
                            CHECK(a.drain && a.nd == want && a.debt == 0.0f);
                        } else {
                            // clamp: exactly the in-flight count, the loss added to the debt
                            CHECK(!a.drain && a.nd == m);
                            CHECK_EQ(a.debt, debt + loss);
                        }
                        if (grow == 0.0f && want > m) CHECK(a.drain);  // e1_grow 0 = the rule before v0252-swe1
                    }
    // NaN / infinite loss (no timing estimate) drains; a negative loss adds nothing
    CHECK(sr::aheadDrafts(4, 3, 0.0f, std::numeric_limits<float>::infinity(), 0.02f).drain);
    CHECK(sr::aheadDrafts(4, 3, 0.0f, std::numeric_limits<float>::quiet_NaN(), 0.02f).drain);
    {
        const sr::AheadDrafts a = sr::aheadDrafts(4, 3, 0.01f, -0.5f, 0.02f);
        CHECK(!a.drain && a.nd == 3u && a.debt == 0.01f);
    }
    // a sequence of wishes: debt grows by each loss and drains exactly when it reaches grow
    {
        float debt = 0;
        int drains = 0, clamps = 0;
        for (int i = 0; i < 100; ++i) {
            const sr::AheadDrafts a = sr::aheadDrafts(5, 4, debt, 0.25f, 1.0f);
            debt = a.debt;
            a.drain ? ++drains : ++clamps;
        }
        // 0.25 per clamp, 1.0 per drain (exact in binary): 3 clamps (0.75), then the 4th wish reaches 1.0 and drains
        CHECK_EQ(drains, 25);
        CHECK_EQ(clamps, 75);
    }
    // ---- aheadClampLoss: the fraction of a cycle the clamp gives away, 1 - score(m) / score(want), score = E(nd) / t(nd)
    {
        qwen35::DraftAccept acc(0.8f);
        qwen35::DraftTiming tm;
        CHECK(std::isinf(sr::aheadClampLoss(acc, tm, 3, 4)));  // no estimate yet: drain
        for (std::uint32_t nd = 1; nd <= 8; ++nd)
            for (int i = 0; i < 4; ++i) tm.update(nd, 90.0f + 1.0f * static_cast<float>(nd));
        for (std::uint32_t m = 1; m < 8; ++m)
            for (std::uint32_t w = m + 1; w <= 8; ++w) {
                const float s_m = sr::mtpScore(acc, tm, m), s_w = sr::mtpScore(acc, tm, w);
                const float l = sr::aheadClampLoss(acc, tm, m, w);
                CHECK(l >= 0.0f && l <= 1.0f);
                if (s_w > s_m) CHECK(std::fabs(l - (1.0f - s_m / s_w)) < 1e-6f);
                else CHECK_EQ(l, 0.0f);  // the policy's own exploration-up wish scores lower: costs nothing, never drains
            }
        // a high acceptance makes the clamp expensive (one draft less at 0.95 is > 2 % of a cycle), a poor one cheap
        qwen35::DraftAccept hi(0.95f), lo(0.2f);
        CHECK(sr::aheadClampLoss(hi, tm, 3, 4) > 0.02f);
        CHECK(sr::aheadClampLoss(lo, tm, 3, 4) < sr::aheadClampLoss(hi, tm, 3, 4));
        // at acceptance 0.2 a 4th draft adds less than its verify row costs: the policy scores 4 below 3, the wish costs nothing
        CHECK(sr::mtpScore(lo, tm, 4) < sr::mtpScore(lo, tm, 3));
        CHECK_EQ(sr::aheadClampLoss(lo, tm, 3, 4), 0.0f);
    }
    // ---- the per-request line's text (counts and mean ms per draft count with at least one cycle)
    {
        std::array<std::uint32_t, qwen35::max_ng_drafts + 1> n{};
        std::array<double, qwen35::max_ng_drafts + 1> ms{};
        CHECK_EQ(sr::cyclesByDrafts(n, ms), std::string("cycles by drafts -; verify ms by drafts -"));
        n[1] = 3;
        ms[1] = 0.36;
        n[4] = 120;
        ms[4] = 36.0;
        CHECK_EQ(sr::cyclesByDrafts(n, ms), std::string("cycles by drafts 1:3 4:120; verify ms by drafts 1:0.12 4:0.30"));
    }
}

void testE1SwiftAhead() {
    const std::uint32_t cap = 4;
    std::vector<std::uint32_t> ps[3] = {makePrompt(300, 2101), makePrompt(420, 2102), makePrompt(260, 2103)};
    const std::uint32_t mts[3] = {400, 300, 350};
    struct Arm {
        const char* name;
        std::uint32_t e1;
        float grow;  // EngineOptions::e1_grow
        bool snap;
    };
    // (default e1_grow: EngineOptions{}.e1_grow)
    const float def = EngineOptions{}.e1_grow;
    CHECK(def > 0.0f && def < 1.0f);
    const Arm arms[] = {{"snapshot E1=0", 0, def, true},          {"snapshot E1=2 grow 0 (drain as before)", 2, 0.0f, true},
                        {"snapshot E1=2 default", 2, def, true}, {"snapshot E1=2 no growth drain", 2, 1e9f, true},
                        {"replay E1=2 (gfx1201)", 2, def, false}};
    std::vector<e1test::ReqOut> outs[std::size(arms)];
    Engine::Stats st[std::size(arms)]{};
    for (std::size_t ai = 0; ai < std::size(arms); ++ai) {
        const Arm& a = arms[ai];
        ServerSetup su = e1swift::setup(a.e1, cap, a.snap);
        su.eo.e1_grow = a.grow;
        g_log.clear();
        TestServer srv(su);
        for (int i = 0; i < 3; ++i) {
            const HttpResult r = srv.post("/v1/completions", e1test::greedyBody(ps[i], mts[i], true));
            CHECK_EQ(r.status, 200);
            const e1test::ReqOut o = e1test::reqOut(r);
            CHECK_EQ(o.text, decode(srv.model().greedyReference(ps[i], mts[i], true)));
            outs[ai].push_back(o);
        }
        const Engine::Stats es = srv.engine().stats();
        st[ai] = es;
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
        CHECK_EQ(srv.model().e1Counts().disagree, 0u);
        CHECK_EQ(es.acc_fallback, 0u);
        const std::uint64_t starts = es.acc_ring + es.acc_discard - es.acc_ahead;
        std::printf("  e1_swift_ahead %s: ring %llu, ahead %llu, pipeline starts %llu, clamped %llu, growth drains %llu, max drafts %u\n", a.name,
                    static_cast<unsigned long long>(es.acc_ring), static_cast<unsigned long long>(es.acc_ahead), static_cast<unsigned long long>(starts),
                    static_cast<unsigned long long>(es.acc_clamp), static_cast<unsigned long long>(es.acc_grow), srv.model().maxSegDrafts());
        CHECK(srv.model().maxSegDrafts() <= cap);
        if (a.e1 == 0) {
            CHECK(es.acc_ring == 0 && es.acc_clamp == 0 && es.acc_grow == 0);
            continue;
        }
        CHECK(es.acc_ring > 0 && es.acc_ahead > 0);
        if (!a.snap) {
            // replay mode (gfx1201): no select verify, no candidate limit - never clamped, never a growth drain
            CHECK_EQ(es.acc_clamp, 0u);
            CHECK_EQ(es.acc_grow, 0u);
            continue;
        }
        const MockModel::SelCounts sc = srv.model().selCounts();
        CHECK_EQ(sc.selects, es.acc_ahead);  // (a select with more drafts than its candidates throws BadBatch in the mock)
        CHECK_EQ(sc.bad_settles, 0u);
        if (a.grow == 0.0f) {
            // the rule before v0252-swe1: every growth behind a cycle in flight drains
            CHECK_EQ(es.acc_clamp, 0u);
            CHECK(es.acc_grow > 0);
            CHECK(starts > 3);
        } else if (a.grow > 1.0f) {
            // no growth drain at all: the policy's wishes are clamped, and each request runs ONE pipeline (max_tokens ends
            // it by the budget rule; ignore_eos: no end inside the window)
            CHECK(es.acc_clamp > 0);
            CHECK_EQ(es.acc_grow, 0u);
            CHECK_EQ(starts, 3u);
            CHECK_EQ(es.acc_discard, 0u);
        } else {
            CHECK(es.acc_clamp > 0);  // the default clamps
        }
    }
    // greedy exact (precise / balance): the same tokens in every arm, the reference's
    for (std::size_t ai = 1; ai < std::size(arms); ++ai)
        for (int i = 0; i < 3; ++i) {
            CHECK_EQ(outs[ai][i].text, outs[0][i].text);
            CHECK_EQ(outs[ai][i].tokens, outs[0][i].tokens);
        }
    // the clamp replaced drains: fewer pipeline starts than the old rule, more cycles submitted ahead per ring cycle
    const auto starts = [&](std::size_t i) { return st[i].acc_ring + st[i].acc_discard - st[i].acc_ahead; };
    CHECK(starts(3) < starts(1));

    // ---- fast (relaxacc decided on the device): the path runs and is repeatable; check mode: device = host rule. Output is
    // NOT compared with E1=0 (fewer drafts in a clamped cycle -> other relaxed keeps: fast may change its output)
    {
        std::vector<e1test::ReqOut> fo[3];
        Engine::Stats fs[3]{};
        const std::uint32_t acc_modes[3] = {0, 0, 4};
        for (int run = 0; run < 3; ++run) {
            ServerSetup su = e1fast::setup(2, cap, acc_modes[run], true, true);
            su.eo.mtp_auto = true;
            su.mc.draft_miss = 0.2;
            g_log.clear();
            TestServer srv(su);
            for (int i = 0; i < 3; ++i) {
                const HttpResult r = srv.post("/v1/completions", e1test::greedyBody(ps[i], mts[i], true));
                CHECK_EQ(r.status, 200);
                fo[run].push_back(e1test::reqOut(r));
            }
            fs[run] = srv.engine().stats();
            CHECK_EQ(srv.model().poison(), 0u);
            CHECK_EQ(srv.model().hidMismatch(), 0u);
            CHECK_EQ(fs[run].acc_mismatch, 0u);
            CHECK_EQ(fs[run].acc_fallback, 0u);
            CHECK(fs[run].acc_ahead > 0);
            CHECK(srv.model().relaxCounts().launches > 0);
            CHECK_EQ(g_log.count("decode: E1 with relaxacc decided on the GPU"), 1u);
            if (acc_modes[run] == 4) CHECK(fs[run].acc_checked > 0);
            std::printf("  e1_swift_ahead fast run %d (spec_acc %u): ring %llu, ahead %llu, clamped %llu, growth drains %llu, checked %llu\n", run, acc_modes[run],
                        static_cast<unsigned long long>(fs[run].acc_ring), static_cast<unsigned long long>(fs[run].acc_ahead),
                        static_cast<unsigned long long>(fs[run].acc_clamp), static_cast<unsigned long long>(fs[run].acc_grow),
                        static_cast<unsigned long long>(fs[run].acc_checked));
        }
        // the fast policy sees a synthetic cycle cost (no wall clock): the same server settings give the same reply
        CHECK(fo[0] == fo[1]);
        CHECK(fs[0].acc_clamp == fs[1].acc_clamp && fs[0].acc_grow == fs[1].acc_grow);
        CHECK(fs[0].acc_clamp + fs[0].acc_grow > 0);  // growth behind a cycle in flight happened (exploration every 32 cycles)
    }
}

void testE1SwiftLog() {
    for (const std::uint32_t e1 : {0u, 2u}) {
        ServerSetup su = e1swift::setup(e1, 4, true);
        g_log.clear();
        TestServer srv(su);
        const std::vector<std::uint32_t> p0 = makePrompt(300, 2201), p1 = makePrompt(350, 2202);
        for (const auto* p : {&p0, &p1, &p0}) CHECK_EQ(srv.post("/v1/completions", e1test::greedyBody(*p, 200, true)).status, 200);
        for (int i = 0; i < 500 && e1swift::linesWith("spec: cycles by drafts").size() < 3; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const std::vector<std::string> sl = e1swift::linesWith("spec: cycles by drafts");
        const std::vector<std::string> ml = e1swift::linesWith("MTP: drafted");
        CHECK_EQ(sl.size(), 3u);
        CHECK_EQ(ml.size(), 3u);
        std::uint64_t clamped = 0, grown = 0;
        for (std::size_t i = 0; i < sl.size() && i < ml.size(); ++i) {
            const e1swift::SpecLine s = e1swift::parseSpecLine(sl[i]);
            if (i == 0) std::printf("  e1_swift_log E1=%u: %s\n", e1, sl[i].c_str());
            CHECK(s.ok);
            CHECK(sl[i].find("req ") != std::string::npos && sl[i].find(" | slot ") != std::string::npos);
            CHECK_EQ(s.cycles, e1swift::mtpCycles(ml[i]));  // the counts add up to the request's verify cycles
            CHECK(s.cycles > 0);
            CHECK_EQ(s.ms_items, s.cyc_items);
            clamped += s.clamped;
            grown += s.grown;
        }
        const Engine::Stats es = srv.engine().stats();
        CHECK_EQ(clamped, es.acc_clamp);  // per-request counts add up to the engine totals
        CHECK_EQ(grown, es.acc_grow);
        if (e1 == 0) CHECK(clamped == 0 && grown == 0);
    }
}

// v0252-swe1 round 2: WHIRL_E1_GROW (EngineOptions::e1_grow, the fraction of a cycle one growth drain costs; 0 = every growth
// behind a cycle in flight drains, the rule before v0252-swe1 - a second fast screen runs it as an arm on one exe): the parser,
// the env table / docs / server_main lines, and the two ends of the knob on the server path
void testE1SwiftGrowEnv() {
#ifdef WHIRL_SOURCE_DIR
    auto slurp = [](const char* rel) {
        std::ifstream in(std::string(WHIRL_SOURCE_DIR) + rel, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    const std::string rel = slurp("/src/release/release.cpp");
    const std::string doc = slurp("/docs/usage.md");
    const std::string sm = slurp("/src/server/server_main.cpp");
    CHECK(rel.find("\"E1_GROW=<fraction>\"") != std::string::npos);
    CHECK(doc.find("`WHIRL_E1_GROW=<fraction>`") != std::string::npos);
    CHECK(sm.find("settings::e1Grow()") != std::string::npos);
    CHECK(sm.find("WHIRL_E1_GROW={} is not a number >= 0; using {}") != std::string::npos);
    CHECK(sm.find("decode: WHIRL_E1_GROW={}") != std::string::npos);
#else
    CHECK(false);
#endif
    const std::pair<const char*, float> good[] = {{"0", 0.0f}, {"0.02", 0.02f}, {" 0.5 ", 0.5f}, {"1e9", 1e9f}, {"\t0.25", 0.25f}, {"1", 1.0f}};
    for (const auto& [s, v] : good) {
        const auto r = parseE1Grow(s);
        CHECK(r.has_value() && *r == v);
    }
    for (const char* s : {"", "-1", "-0.01", "abc", "nan", "inf", "0.02x", "1e99", "0x10", " "}) {
        if (parseE1Grow(s).has_value()) std::printf("  parseE1Grow(\"%s\") accepted\n", s);
        CHECK(!parseE1Grow(s).has_value());
    }
    // the knob's two ends on the server path (snapshot + select, automatic count): 0 -> no clamp, every growth drains;
    // a value past any debt -> clamps, no growth drain (e1_swift_ahead covers the arms in detail)
    for (const float g : {0.0f, 1e9f}) {
        ServerSetup su = e1swift::setup(2, 4, true);
        su.eo.e1_grow = g;
        TestServer srv(su);
        const std::vector<std::uint32_t> p = makePrompt(320, 2301);
        CHECK_EQ(completionText(srv.post("/v1/completions", e1test::greedyBody(p, 300, true))), decode(srv.model().greedyReference(p, 300, true)));
        const Engine::Stats es = srv.engine().stats();
        if (g == 0.0f) CHECK(es.acc_clamp == 0 && es.acc_grow > 0);
        else CHECK(es.acc_clamp > 0 && es.acc_grow == 0);
    }
}
