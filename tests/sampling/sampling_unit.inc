// Sampling unit tests (arch025-X3: moved verbatim from tests/server/server_tests.cpp:608-869 at 9baa686).
// Included by tests/server/server_tests.cpp (whirl-server-tests: sampler, acceptor_fixed / _rules / _random) and by
// tests/sampling/sampling_tests.cpp (whirl-sampling-tests: the same tests built against src/sampling alone, ARCH-1 8.5).
// The includer provides, before this file: CHECK / CHECK_EQ, Sampler and Cand (= whirl::sampling::*), acase:: and
// oldref:: (sampling_oldref.inc, sampling_cases.h inside namespace whirl::server), <set>, <string>, <array>, <span>.
// SPDX-License-Identifier: Apache-2.0

void testSampler() {
    std::vector<Cand> cands(8);
    Sampler s;
    s.cands = &cands;
    s.inv_t = 1;
    s.top_k = 3;
    s.top_p = 1;
    s.min_p = 0;
    const float logits[8] = {1, 5, 3, 3, 0, -1, 2, 4};
    for (std::uint32_t i = 0; i < 8; ++i) cands[i] = {i, logits[i], 0};
    s.m = 5;
    double z = 0;
    for (float l : logits) z += std::exp(static_cast<double>(l - 5));
    s.sum = static_cast<float>(z);
    CHECK(filterCands(s, 8, false));
    CHECK(s.n == 3 && s.closed);
    CHECK(cands[0].id == 1 && cands[1].id == 7 && (cands[2].id == 2));
    double tot = 0;
    for (std::size_t i = 0; i < s.n; ++i) tot += cands[i].p;
    CHECK(std::fabs(tot - 1.0) < 1e-9);
    // top_k larger than the candidates and an open set: escalation
    s.top_k = 20;
    for (std::uint32_t i = 0; i < 8; ++i) cands[i] = {i, logits[i], 0};
    CHECK(!filterCands(s, 8, false));
    // arch025-X12 (X1 survivor x12-e1a-topk-boundary): top_k == the candidate count on an open set closes the
    // distribution with every candidate (no full-row escalation)
    s.top_k = 8;
    for (std::uint32_t i = 0; i < 8; ++i) cands[i] = {i, logits[i], 0};
    CHECK(filterCands(s, 8, false));
    CHECK(s.n == 8 && s.closed);
    // arch025-X12 (X1 survivor x12-e1a-minp-strict): min_p keeps a candidate whose p equals min_p * p(top)
    // (min_p = 1 and two tied top logits: both stay)
    {
        const float tl[4] = {5, 5, 3, 1};
        std::vector<Cand> tc(4);
        for (std::uint32_t i = 0; i < 4; ++i) tc[i] = {i, tl[i], 0};
        Sampler t;
        t.cands = &tc;
        t.inv_t = 1;
        t.top_k = 0;
        t.top_p = 1;
        t.min_p = 1;
        CHECK(filterCands(t, 4, true));
        CHECK(t.n == 2 && t.closed);
    }
    // uniforms keyed by (seed, index)
    Sampler a, b;
    a.seed = b.seed = 7;
    a.pos_idx = b.pos_idx = 3;
    CHECK(a.uniform() == b.uniform());
    b.pos_idx = 4;
    CHECK(a.uniform() != b.uniform());
    CHECK(a.uniform() >= 0 && a.uniform() < 1);
}

// ---------------------------------------------------------------------------
// arch025-E1a: Acceptor (src/sampling) == the pre-move engine code, on fixed draft / target rows and seeds.
// The expected values below were recorded by running the cases on the OLD code (sampling_oldref.inc, a verbatim
// copy of engine.h:73-110 and engine.cpp:717-780, 822-888, 1455-1462, 1868-1925 at arch025-next eeb79c2) with
// scratch\a25_e1a\capture.exe before anything was moved; the same table must hold after the move.

// sampling::RowSource over the case's host rows; reads are traced like the old engine's device downloads
struct FakeRows final : sampling::RowSource {
    const std::vector<std::vector<float>>* rows = nullptr;
    std::vector<std::uint8_t> marks;
    acase::Trace* tr = nullptr;
    std::uint32_t V = 0;
    void prepareRow(Sampler& s, std::uint32_t r) override {
        acase::hostPrepare(s, r, *rows, [](Sampler& x, std::size_t n, bool c) { return sampling::filterCands(x, n, c); }, *tr);
    }
    const float* loadFullRow(std::uint32_t gr) override {
        tr->add(acase::ev_full, gr, 0);
        return (*rows)[gr].data();
    }
    float logitAt(std::uint32_t gr, std::uint32_t t) override {
        tr->add(acase::ev_logit, gr, t);
        return (*rows)[gr][t];
    }
    std::uint32_t nVocab() const override { return V; }
    std::uint8_t* inCands() override { return marks.data(); }
};

acase::Result runNewIn(const acase::Case& c, acase::Inputs& in) {
    acase::Result res;
    FakeRows src;
    src.rows = &in.rows;
    src.marks.assign(c.V, 0);
    src.tr = &res.tr;
    src.V = c.V;
    std::vector<Cand> cands(std::max(c.V, acase::kb));
    Sampler sm;
    sm.temp = c.temp;
    sm.inv_t = c.temp > 0 ? static_cast<float>(1.0 / c.temp) : 1.0f;
    sm.top_k = c.top_k;
    sm.top_p = c.top_p;
    sm.min_p = c.min_p;
    sm.seed = c.seed;
    sm.race = c.race;
    sm.cands = &cands;
    std::uint32_t relaxed = 0;
    double gap = 0;
    const sampling::Acceptor acceptor(in.rp);
    const sampling::Drafts d{std::span<const std::uint32_t>(in.dr.data(), in.dr.size()), c.nd, c.row0};
    // Engine::relaxRepeats with the slot's next token and cache_tokens
    const sampling::RepeatGuard repeats = [&](std::uint32_t a) {
        std::array<std::uint32_t, qwen35::max_ng_drafts + 2> tail{};
        tail[0] = in.next;
        for (std::uint32_t r = 0; r <= a; ++r) tail[r + 1] = in.dr[r];
        return relax::extendsRepeat(std::span<const std::uint32_t>(in.hist), std::span<const std::uint32_t>(tail.data(), a + 2));
    };
    const sampling::Accepted ac =
        c.greedy ? acceptor.greedy(in.ctl.data() + qwen35::ctl_rows, d, c.relax,
                                   sampling::RelaxRows{in.relax_host.data(), acase::max_rows_t, in.rp.topk()}, repeats, relaxed, gap)
                 : acceptor.sample(src, sm, d, c.relax, c.n_gen, repeats, relaxed);
    res.acc = ac.acc;
    res.chosen = ac.chosen;
    res.relaxed = relaxed;
    res.gap_bits = acase::bits(gap);
    res.pos_idx = sm.pos_idx;
    res.full_used = sm.full_used;
    res.big_used = sm.big_used;
    for (std::uint32_t i = 0; i <= res.acc; ++i) {
        Sampler u;
        u.seed = c.seed;
        u.pos_idx = c.n_gen + i;
        res.draws.push_back(acase::bits(u.uniform()));
    }
    return res;
}
acase::Result runNew(const acase::Case& c) { return acase::runResolved(c, runNewIn); }

struct AcceptExpect {
    const char* name;
    std::uint32_t acc, chosen, relaxed;
    std::uint64_t gap_bits, pos_idx;
    std::uint32_t full_used, big_used;
    std::uint64_t trace_hash;
    std::vector<std::uint64_t> draws;
};

// recorded on the pre-move code (scratch\a25_e1a\expected.txt, 2026-10-25)
const std::vector<AcceptExpect>& acceptExpected() {
    static const std::vector<AcceptExpect> e = {
        {"greedy_all_match", 3, 28, 0, 0x0000000000000000ull, 0, 0, 0, 0x14650fb0739d0383ull, {0x3fd95fbb374f2c4eull, 0x3fe85a64dc00ab7bull, 0x3fe0c43407fc177bull, 0x3fe1c3eeaab30755ull}},
        {"greedy_reject_1", 1, 16, 0, 0x0000000000000000ull, 0, 0, 0, 0x14650fb0739d0383ull, {0x3fd95fbb374f2c4eull, 0x3fe85a64dc00ab7bull}},
        {"greedy_reject_0", 0, 6, 0, 0x0000000000000000ull, 0, 0, 0, 0x14650fb0739d0383ull, {0x3fd95fbb374f2c4eull}},
        {"greedy_no_drafts", 0, 54, 0, 0x0000000000000000ull, 0, 0, 0, 0x14650fb0739d0383ull, {0x3fd95fbb374f2c4eull}},
        {"relax_greedy_rank1", 3, 41, 2, 0x3fc0e56080000000ull, 0, 0, 0, 0x14650fb0739d0383ull, {0x3fd95fbb374f2c4eull, 0x3fe85a64dc00ab7bull, 0x3fe0c43407fc177bull, 0x3fe1c3eeaab30755ull}},
        {"relax_greedy_rank3", 4, 23, 3, 0x3fa9581100000000ull, 0, 0, 0, 0x14650fb0739d0383ull, {0x3fd95fbb374f2c4eull, 0x3fe85a64dc00ab7bull, 0x3fe0c43407fc177bull, 0x3fe1c3eeaab30755ull, 0x3fe6a9c1e2c01989ull}},
        {"relax_greedy_far", 1, 57, 1, 0x3fa6872800000000ull, 0, 0, 0, 0x14650fb0739d0383ull, {0x3fd95fbb374f2c4eull, 0x3fe85a64dc00ab7bull}},
        {"relax_greedy_repeat", 1, 2, 1, 0x3f7eb85000000000ull, 0, 0, 0, 0x14650fb0739d0383ull, {0x3fd95fbb374f2c4eull, 0x3fe85a64dc00ab7bull}},
        {"sample_closed_topk", 1, 11, 0, 0x0000000000000000ull, 21, 0, 2, 0x66e0b0fdd3156487ull, {0x3fbd67c34f6a2a30ull, 0x3fef99504bda8630ull}},
        {"sample_topp", 0, 30, 0, 0x0000000000000000ull, 21, 1, 1, 0x47fe0c7eaf8e5030ull, {0x3fe545e33bf7f406ull}},
        {"sample_minp", 0, 39, 0, 0x0000000000000000ull, 22, 1, 1, 0x47fe0c7eaf8e5030ull, {0x3fd60b59c5ea4748ull}},
        {"sample_open_tail", 0, 24, 0, 0x0000000000000000ull, 23, 1, 0, 0x0fb3d3c281563996ull, {0x3fd9b75720072950ull}},
        {"sample_open_tail_b", 0, 43, 0, 0x0000000000000000ull, 0, 1, 0, 0xa59039122b4385f6ull, {0x3fe5d5ea5fd7ce0cull}},
        {"sample_escalate_topk", 0, 16, 0, 0x0000000000000000ull, 24, 0, 1, 0x47fe0c7eaf8e5030ull, {0x3fed89574c81bac9ull}},
        {"specsample_closed", 2, 46, 0, 0x0000000000000000ull, 27, 0, 3, 0xdfe5090601b09cafull, {0x3fedc97a524aca41ull, 0x3feb8329339ad37bull, 0x3fe7ef828545b800ull}},
        {"specsample_topp", 0, 12, 0, 0x0000000000000000ull, 26, 1, 1, 0x47fe0c7eaf8e5030ull, {0x3fdae54e1f7ab804ull}},
        {"specsample_open", 0, 15, 0, 0x0000000000000000ull, 27, 1, 0, 0x0fb3d3c281563996ull, {0x3fd3c1ac84edb6c2ull}},
        {"specsample_open_b", 0, 60, 0, 0x0000000000000000ull, 28, 1, 0, 0xdaa2066a564cdfc6ull, {0x3fedb9ea18198051ull}},
        {"specsample_minp", 0, 22, 0, 0x0000000000000000ull, 29, 1, 1, 0x47fe0c7eaf8e5030ull, {0x3fc040b9dc918150ull}},
        {"specsample_no_drafts", 0, 37, 0, 0x0000000000000000ull, 30, 1, 0, 0x0fb3d3c281563996ull, {0x3fe388eea59b03bfull}},
        {"typical_closed", 4, 50, 4, 0x0000000000000000ull, 35, 0, 5, 0x4f840a14e2b94b30ull, {0x3fd82adf0b38cce2ull, 0x3fd0439ace23450aull, 0x3fe2b35fc9016ecaull, 0x3fec05399b7ae104ull, 0x3fe080aaac70a7c6ull}},
        {"typical_topp", 4, 59, 4, 0x0000000000000000ull, 36, 5, 5, 0x31d9ec81294a4ef4ull, {0x3fdc66b2091fcfb8ull, 0x3fdc84cfae5a6feaull, 0x3fe55ca8c2e34d42ull, 0x3fed6973a34de399ull, 0x3fef0ffd3e892faaull}},
        {"typical_open_F2_02", 0, 46, 0, 0x0000000000000000ull, 33, 1, 0, 0xd223b5d048d838aaull, {0x3fe7e0b9b7ce6965ull}},
        {"typical_open_race", 0, 55, 0, 0x0000000000000000ull, 34, 1, 0, 0xa59039122b4385f6ull, {0x3feb38b858a7a633ull}},
        {"typical_far", 0, 47, 0, 0x0000000000000000ull, 35, 0, 1, 0x47fe0c7eaf8e5030ull, {0x3fdadc7f85086cd0ull}},
        {"sample_t_match", 4, 28, 0, 0x0000000000000000ull, 44, 5, 5, 0x834bc0cf99e49854ull, {0x3fd17a564ed562d2ull, 0x3fee8ccbaf14db13ull, 0x3fe36a629689298bull, 0x3fd4319b0f01ee66ull, 0x3feaea97a9fe5082ull}},
        {"sample_t_open", 2, 9, 0, 0x0000000000000000ull, 43, 2, 0, 0xc4a1a9f4ec925f17ull, {0x3fa609756bcb6960ull, 0x3fd6415f9ea5af10ull, 0x3fcc85d0b63575ecull}},
        {"specsample_t_match", 4, 37, 0, 0x0000000000000000ull, 46, 0, 5, 0xfe6e9cc2b5cd315cull, {0x3f95efb74bfb9100ull, 0x3fd544e98de425b0ull, 0x3fe9362924b58baaull, 0x3fcb38348ed71918ull, 0x3fe0fc5c865248c3ull}},
        {"specsample_t_open", 3, 51, 0, 0x0000000000000000ull, 46, 2, 0, 0xd65e1d748f433445ull, {0x3fa0196f8d4ab2b0ull, 0x3fed3c3f6655cd41ull, 0x3fcb8e0f91f6b85cull, 0x3fa37ce8f0abbb10ull}},
        {"specsample_t_minp", 3, 7, 0, 0x0000000000000000ull, 47, 4, 4, 0x5ad34310bcd7ae08ull, {0x3f8314663cf52f80ull, 0x3fe172766f5d8333ull, 0x3fe922decdf99f29ull, 0x3fe84a3b7aa47f8bull}},
        {"typical_t_open", 1, 41, 0, 0x0000000000000000ull, 46, 2, 0, 0x71eb3408b2ddeb0cull, {0x3fc414499d119280ull, 0x3fe570836d21e63eull}},
        {"typical_t_race", 4, 18, 4, 0x0000000000000000ull, 50, 5, 5, 0xe0a4f091a4d6782cull, {0x3fcd6b1d03f31e98ull, 0x3fedb9a47cb51293ull, 0x3fd69d0c84003ab0ull, 0x3fe742df9f54dd12ull, 0x3fb432ff66d751e8ull}},
        {"typical_repeat", 1, 44, 1, 0x0000000000000000ull, 37, 0, 2, 0xe157db1d7a337617ull, {0x3fecd9bb06f00448ull, 0x3fe5ac3e22af6700ull}},
    };
    return e;
}

bool sameResult(const acase::Result& a, const acase::Result& b) {
    return a.acc == b.acc && a.chosen == b.chosen && a.relaxed == b.relaxed && a.gap_bits == b.gap_bits && a.pos_idx == b.pos_idx &&
           a.full_used == b.full_used && a.big_used == b.big_used && a.tr.ev == b.tr.ev && a.draws == b.draws;
}

// fixed cases: exact accept count, chosen token, relaxacc counters, device-read order (with the pos_idx each row
// prepare saw) and the uniform draw at every output index, as recorded on the old code
void testAcceptorFixed() {
    const auto& cs = acase::cases();
    const auto& ex = acceptExpected();
    CHECK_EQ(cs.size(), ex.size());
    for (std::size_t i = 0; i < cs.size() && i < ex.size(); ++i) {
        const acase::Case& c = cs[i];
        const AcceptExpect& e = ex[i];
        CHECK_EQ(std::string(c.name), std::string(e.name));
        const acase::Result r = runNew(c);
        const bool ok = r.acc == e.acc && r.chosen == e.chosen && r.relaxed == e.relaxed && r.gap_bits == e.gap_bits &&
                        r.pos_idx == e.pos_idx && r.full_used == e.full_used && r.big_used == e.big_used &&
                        r.tr.hash() == e.trace_hash && r.draws == e.draws;
        if (!ok)
            std::fprintf(stderr, "  acceptor case %s: acc %u/%u chosen %u/%u relaxed %u/%u pos %llu/%llu trace %016llx/%016llx\n", c.name,
                         r.acc, e.acc, r.chosen, e.chosen, r.relaxed, e.relaxed, static_cast<unsigned long long>(r.pos_idx),
                         static_cast<unsigned long long>(e.pos_idx), static_cast<unsigned long long>(r.tr.hash()),
                         static_cast<unsigned long long>(e.trace_hash));
        CHECK(ok);
        // and the verbatim old code still gives the table (the reference itself did not drift)
        const acase::Result o = acase::runOld(c);
        CHECK(sameResult(o, r));
    }
    // the table exercises every rule and both outcomes
    std::set<std::string> rules;
    std::uint32_t any_acc = 0, any_rej = 0, any_relaxed = 0, any_full = 0, any_logit = 0;
    for (std::size_t i = 0; i < cs.size(); ++i) {
        rules.insert(sampling::ruleName(sampling::acceptRule(cs[i].greedy, cs[i].relax, cs[i].race)));
        const acase::Result r = runNew(cs[i]);
        any_acc += r.acc > 0;
        any_rej += r.acc < cs[i].nd;
        any_relaxed += r.relaxed > 0;
        for (std::uint64_t ev : r.tr.ev) {
            any_full += (ev >> 56) == acase::ev_full;
            any_logit += (ev >> 56) == acase::ev_logit;
        }
    }
    CHECK_EQ(rules.size(), std::size_t{5});
    CHECK(any_acc > 0 && any_rej > 0 && any_relaxed > 0 && any_full > 0 && any_logit > 0);
}

// the rule table: specsample and relaxacc are exclusive in the engine (spec_on_ needs !relax.on); a relaxacc slot with
// the race sampler is still typical acceptance
void testAcceptorRules() {
    using sampling::AcceptRule;
    CHECK(sampling::acceptRule(true, false, false) == AcceptRule::greedy);
    CHECK(sampling::acceptRule(true, false, true) == AcceptRule::greedy);
    CHECK(sampling::acceptRule(true, true, false) == AcceptRule::relaxacc_greedy);
    CHECK(sampling::acceptRule(false, false, false) == AcceptRule::sample);
    CHECK(sampling::acceptRule(false, false, true) == AcceptRule::specsample);
    CHECK(sampling::acceptRule(false, true, false) == AcceptRule::relaxacc_typical);
    CHECK(sampling::acceptRule(false, true, true) == AcceptRule::relaxacc_typical);
    CHECK_EQ(std::string(sampling::ruleName(AcceptRule::specsample)), std::string("specsample"));
}

// random cases (fixed generator seed): the new Acceptor == the verbatim old code field by field, read trace included
void testAcceptorRandom() {
    std::uint32_t g = 0xE1A0u;
    std::vector<std::string> drafts;
    drafts.reserve(4000);
    std::uint32_t same = 0, n = 0, accepted = 0;
    for (int it = 0; it < 4000; ++it) {
        acase::Case c{};
        c.name = "random";
        c.greedy = acase::lcg(g) % 3 == 0;
        c.relax = acase::lcg(g) % 2 == 0;
        c.race = !c.greedy && acase::lcg(g) % 2 == 0;
        const double temps[4] = {0.5, 0.8, 1.0, 1.6};
        c.temp = c.greedy ? 0 : temps[acase::lcg(g) % 4];
        const std::uint32_t tks[4] = {0, 0, 5, 20};
        c.top_k = tks[acase::lcg(g) % 4];
        const double tps[4] = {1, 1, 0.9, 0.97};
        c.top_p = tps[acase::lcg(g) % 4];
        const double mps[3] = {0, 0, 0.05};
        c.min_p = mps[acase::lcg(g) % 3];
        c.seed = acase::lcg(g);
        c.n_gen = acase::lcg(g) % 1000;
        c.nd = acase::lcg(g) % 5;
        c.row0 = acase::lcg(g) % (acase::max_rows_t - c.nd);
        c.V = 32 + acase::lcg(g) % 64;
        c.lseed = acase::lcg(g);
        const float spreads[8] = {0.2f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        c.spread = spreads[acase::lcg(g) % 8];
        std::string ds;
        const char kinds[6] = {'0', '1', '2', 'x', 't', 't'};
        for (std::uint32_t r = 0; r < c.nd; ++r) ds += kinds[acase::lcg(g) % 6];
        drafts.push_back(ds);
        c.drafts = drafts.back().c_str();
        c.rep_hist = c.nd >= 2 && acase::lcg(g) % 4 == 0;
        const acase::Result a = acase::runOld(c), b = runNew(c);
        same += sameResult(a, b);
        accepted += b.acc;
        ++n;
    }
    CHECK_EQ(same, n);
    CHECK(accepted > 1000);
}
