// v0252-e1fast (PLAN V1-93 P3): the E1 one-cycle lookahead for FAST mode - the relaxacc accept decision on the device
// (kernels/spec_accept_relax.h, kernel spec_accept_relax_rows, gfx1151 set). Included by server_tests.cpp after
// e1_snap_tests.inc (it uses e1ahead::setup / applyMode / g_snap).
//   e1_fast_rule   the shared rule alone: d* (the alpha test as d >= d*) == the host's std::exp(d) >= alpha around d* and over
//                  random d, and spec_relax_row == relax::acceptRank on random rank words; the ring repetition guard ==
//                  relax::extendsRepeat over random histories (window 4096 and small windows, ring wrap); a chain of cycles
//                  whose ring is kept ONLY by spec_relax_append (as on the device) decides as sampling::Acceptor::greedy +
//                  the EOS / max_tokens cuts with the host's cache_tokens: acc before / after the cuts, chosen, NPOS, the
//                  relaxed count and relax_gap, with guard refusals, EOS and budget cuts all seen
//   e1_fast_ahead  WHIRL_E1=2 vs 0 in fast (relaxacc on, runner-up drafts the relaxed rule keeps), replay and snapshot+select
//                  mode, draft counts 1 / 2 / 3, WHIRL_SPEC_ACC host / gpu-zc / check, with and without a repetitive (markov)
//                  text: per request the same text, tokens, drafted / accepted counts and relaxacc log line (relaxed count
//                  and mean gap); the lookahead ran (ring, ahead), the device decided relaxed sequences (mock relaxed
//                  launches, drafts kept by the relaxed rule), the guard refused keeps somewhere, check mode found no
//                  disagreement, an end inside the window was dropped and the following requests are unchanged
//   e1_fast_gate   the switch: no default for fast (chooseE1); a model without the kernel keeps fast cycles synchronous (one
//                  startup line, no ring, same text); with it the proof line; WHIRL_E1=1 decides on the device without the
//                  ring; a sampled request in fast never looks ahead; with relaxacc off (precise / balance) the relaxed
//                  path is never asked for (setSpecRelax 0 calls); docs/usage.md names the fast case and the proof line
// SPDX-License-Identifier: Apache-2.0

#include "../../kernels/spec_accept_relax.h"
#include "sampling/acceptor.h"
#include "spec/spec_rules.h"

#include <bit>
#include <random>

namespace e1fast {

std::vector<std::string> relaxLines() {
    std::vector<std::string> out;
    std::lock_guard<std::mutex> lk(g_log.mu);
    for (const auto& l : g_log.lines) {
        const std::size_t at = l.find("relaxacc: ");
        if (at != std::string::npos && l.find("accepted drafts kept by the relaxed rule") != std::string::npos) out.push_back(l.substr(at));
    }
    return out;
}

ServerSetup setup(std::uint32_t e1, std::uint32_t nd, std::uint32_t spec_acc, bool snap, bool relax_dev) {
    e1ahead::g_snap = snap;
    ServerSetup su = e1ahead::setup(e1, nd, spec_acc);
    e1ahead::g_snap = false;
    su.mc.mtp_miss_runner_up = true;  // a wrong draft is the row's runner-up: inside k 6 / alpha 0.05 (the 8060S Ornith arm)
    su.mc.draft_miss = 0.4;
    su.mc.relax_rank = true;
    su.mc.relax_dev = relax_dev;
    su.eo.relax.on = true;
    su.eo.relax.k = 6;
    su.eo.relax.alpha = 0.05f;
    return su;
}

}  // namespace e1fast

// v0252-e1fast (green g1 build failure, whirl/hip.h LaunchArgNoPadding): SpecRelaxArgs is a by-value kernel argument - every
// byte is a named integer field (the launch log digests every byte; a double member has no unique object representation), the
// layout below is what kernels/gfx1151/spec_accept_relax.hip reads (kt specaccx passes it by value and compares every result word)
static_assert(std::has_unique_object_representations_v<SpecRelaxArgs>, "SpecRelaxArgs: no padding, no float member");
static_assert(sizeof(SpecRelaxArgs) == 8 + 2 * WHIRL_SA_MAX_SEG * 4 + 4 * 4, "SpecRelaxArgs: 152 bytes");
static_assert(offsetof(SpecRelaxArgs, on) == 8 && offsetof(SpecRelaxArgs, next) == 8 + WHIRL_SA_MAX_SEG * 4 &&
              offsetof(SpecRelaxArgs, topk) == 8 + 2 * WHIRL_SA_MAX_SEG * 4 && offsetof(SpecRelaxArgs, ngram) == sizeof(SpecRelaxArgs) - 4,
              "SpecRelaxArgs field offsets");

void testE1FastRule() {
    std::mt19937_64 rng(20261125);
    const double inf = std::numeric_limits<double>::infinity();
    // ---- (1) d*: "d >= d*" is the host's "std::exp(d) >= alpha"
    std::uint64_t n_d = 0;
    for (const double alpha : {0.05, 0.1, 0.2, 0.5, 0.05f * 1.0, 0.999, 1.0, 1e-30, 0.0, -1.0, 2.0}) {
        const double ds = spec_relax_dstar(alpha);
        const auto same = [&](double d) {
            if ((d >= ds) != (std::exp(d) >= alpha)) std::printf("  e1_fast_rule: alpha %.17g d %.17g d* %.17g\n", alpha, d, ds);
            CHECK_EQ(d >= ds, std::exp(d) >= alpha);
            ++n_d;
        };
        if (std::isfinite(ds)) {
            double x = ds;
            for (int i = 0; i < 300; ++i) x = std::nextafter(x, -inf);
            for (int i = 0; i < 600; ++i, x = std::nextafter(x, inf)) same(x);
        }
        std::uniform_real_distribution<double> u(-60.0, 1.0);
        for (int i = 0; i < 100000; ++i) same(u(rng));
        for (const double d : {0.0, -0.0, -inf, inf, -745.2, -708.0, std::log(alpha > 0 ? alpha : 1.0)}) same(d);
    }
    CHECK(std::isinf(spec_relax_dstar(0.0)) && spec_relax_dstar(0.0) < 0);
    CHECK(std::isinf(spec_relax_dstar(std::nan(""))) && spec_relax_dstar(std::nan("")) > 0);
    // the rank words of a row (rank, row max, draft logit as float bits) against relax::acceptRank
    std::uint64_t n_row = 0, n_row_yes = 0;
    for (const std::uint32_t k : {1u, 4u, 6u}) {
        for (const float alpha : {0.05f, 0.1f, 0.5f}) {
            relax::Params p;
            p.on = true;
            p.k = k;
            p.alpha = alpha;
            SpecRelaxArgs r{};
            r.dstar_bits = spec_relax_dbits(spec_relax_dstar(static_cast<double>(p.alpha)));
            r.topk = static_cast<int>(p.topk());
            r.stride = 1;
            std::uniform_real_distribution<float> um(-8.0f, 40.0f), ug(0.0f, 6.0f);
            for (int i = 0; i < 40000; ++i) {
                const float m = um(rng);
                float ld = m - ug(rng);
                if (i % 3 == 0) {  // near the boundary: m + ln(alpha), a few float steps either way
                    ld = m + static_cast<float>(std::log(static_cast<double>(alpha)));
                    for (int s = static_cast<int>(rng() % 9) - 4; s != 0; s += s > 0 ? -1 : 1) ld = std::nextafter(ld, s > 0 ? 1e30f : -1e30f);
                }
                const int rank = static_cast<int>(rng() % 10) - 1;
                const int words[3] = {rank, spec_relax_bits(m), spec_relax_bits(ld)};
                float g = 0;
                const bool dev = spec_relax_row(r, words, 0, &g);
                CHECK_EQ(dev, relax::acceptRank(p, rank, ld, m));
                if (dev) {
                    CHECK_EQ(g, m - ld);
                    ++n_row_yes;
                }
                ++n_row;
            }
        }
    }
    // ---- (1b) the boundary itself (mut2 2026-11-25: mutant e1fast-dstar-strict `>= d*` -> `> d*` survived (1), whose float pairs
    // never land exactly on d*). The device keeps a row iff d = double(l_d) - double(m) >= d*, and (1) proves exp(d) >= alpha iff
    // d >= d* for the host's exp; here the comparison itself: a row whose d EQUALS the threshold is kept, one ulp below is not,
    // for thresholds a float pair reaches exactly (l_d - m with m = 0 or a power of two), and d* itself is where the host's exp
    // first reaches alpha (exp(d*) >= alpha > exp(d* - 1 ulp)).
    long long n_edge = 0;
    for (const float alpha : {0.05f, 0.1f, 0.2f, 0.5f}) {
        const double ds = spec_relax_dstar(static_cast<double>(alpha));
        CHECK(std::exp(ds) >= static_cast<double>(alpha) && std::exp(std::nextafter(ds, -inf)) < static_cast<double>(alpha));
    }
    for (const float ld : {-3.0f, -2.9957323f, -0.5f, -1.0e-3f, -7.25f})
        for (const float m : {0.0f, 1.0f, 0.25f, 12.0f}) {
            const double d = static_cast<double>(ld) - static_cast<double>(m);
            SpecRelaxArgs r{};
            r.topk = 6;
            r.stride = 1;
            const int words[3] = {0, spec_relax_bits(m), spec_relax_bits(ld)};
            float g = 0;
            r.dstar_bits = spec_relax_dbits(d);  // threshold exactly at d: kept
            CHECK(spec_relax_row(r, words, 0, &g));
            r.dstar_bits = spec_relax_dbits(std::nextafter(d, inf));  // threshold one ulp above d: refused
            CHECK(!spec_relax_row(r, words, 0, &g));
            r.dstar_bits = spec_relax_dbits(std::nextafter(d, -inf));  // one ulp below: kept
            CHECK(spec_relax_row(r, words, 0, &g));
            n_edge += 1;
        }
    CHECK_EQ(n_edge, 20LL);
    // ---- (2) the ring guard == relax::extendsRepeat (hist = the last positions of a history, tail = next + drafts)
    std::uint64_t n_guard = 0, n_guard_yes = 0;
    for (int trial = 0; trial < 1600; ++trial) {
        const bool big = trial % 8 == 0;
        const int W = big ? WHIRL_SR_WIN : 4 + static_cast<int>(rng() % 60);
        const int len = big ? static_cast<int>(rng() % 9000) : static_cast<int>(rng() % 20000);
        // half the trials on 2..5 tokens (6-grams repeat often), half on up to 31 (rarely)
        const std::uint32_t V = 2 + static_cast<std::uint32_t>(trial % 2 == 0 ? rng() % 4 : rng() % 30);
        std::vector<std::uint32_t> hist(static_cast<std::size_t>(len));
        for (auto& t : hist) t = static_cast<std::uint32_t>(rng() % V);
        std::vector<std::int32_t> ring(WHIRL_SR_HIST, -7);
        for (int i = 0; i < len; ++i) ring[static_cast<std::size_t>(i & (WHIRL_SR_HIST - 1))] = static_cast<std::int32_t>(hist[static_cast<std::size_t>(i)]);
        SpecAccArgs a{};
        a.n = 1;
        a.lit[0] = 1;
        for (int r = 0; r < 4; ++r) a.lit_d[0][r] = static_cast<int>(rng() % V);
        const std::uint32_t next = static_cast<std::uint32_t>(rng() % V);
        const std::int32_t ctl[WHIRL_SA_CTL_WORDS] = {};
        const SpecRelaxScalarGuard g{ring.data(), len, len < W ? len : W, static_cast<int>(next), &a, 0, ctl, WHIRL_SR_NGRAM};
        for (int acc = 0; acc < 4; ++acc) {
            std::vector<std::uint32_t> tail{next};
            for (int r = 0; r <= acc; ++r) tail.push_back(static_cast<std::uint32_t>(a.lit_d[0][r]));
            const bool want = relax::extendsRepeat(hist, tail, WHIRL_SR_NGRAM, static_cast<std::size_t>(W));
            CHECK_EQ(g(acc), want);
            n_guard += 1;
            n_guard_yes += want ? 1 : 0;
        }
    }
    // ---- (3) a chain of cycles: the ring kept by spec_relax_append only (the device), the host by cache_tokens
    std::uint64_t n_cyc = 0, n_relaxed = 0, n_guarded = 0, n_eos = 0, n_budget = 0, n_hostin = 0;
    relax::Params p;
    p.on = true;
    p.k = 6;
    p.alpha = 0.05f;
    const sampling::Acceptor acceptor(p);
    constexpr int S = 32;  // rank-word stride (max rows)
    for (int seq = 0; seq < 400; ++seq) {
        const std::uint32_t V = 3 + static_cast<std::uint32_t>(rng() % 12);
        std::vector<std::uint32_t> cache(static_cast<std::size_t>(rng() % 6000));
        for (auto& t : cache) t = static_cast<std::uint32_t>(rng() % V);
        std::vector<std::int32_t> ring(WHIRL_SR_HIST, -7);
        const auto hostWrite = [&] {  // what the engine writes before a cycle on host inputs: the window
            const std::size_t n = cache.size(), lo = n > WHIRL_SR_WIN ? n - WHIRL_SR_WIN : 0;
            for (std::size_t i = lo; i < n; ++i) ring[i & (WHIRL_SR_HIST - 1)] = static_cast<std::int32_t>(cache[i]);
        };
        std::uint32_t next = static_cast<std::uint32_t>(rng() % V);
        std::int32_t ctl[WHIRL_SA_CTL_WORDS] = {};
        SpecRelaxArgs r{};
        r.dstar_bits = spec_relax_dbits(spec_relax_dstar(static_cast<double>(p.alpha)));
        r.topk = static_cast<int>(p.topk());
        r.stride = S;
        r.win = WHIRL_SR_WIN;
        r.ngram = WHIRL_SR_NGRAM;
        r.on[0] = 1;
        bool host_in = true;
        for (int cyc = 0; cyc < 40; ++cyc) {
            const int nd = 1 + static_cast<int>(rng() % 4);
            const int row0 = static_cast<int>(rng() % 3);
            for (int q = 0; q <= nd; ++q) ctl[WHIRL_SA_CTL_ROWS + q] = static_cast<std::int32_t>(rng() % V);
            for (int q = 0; q < nd; ++q)
                ctl[WHIRL_SA_CTL_DRAFTS + q] = rng() % 2 ? ctl[WHIRL_SA_CTL_ROWS + q] : static_cast<std::int32_t>(rng() % V);
            ctl[WHIRL_SA_CTL_ND] = rng() % 5 == 0 ? static_cast<std::int32_t>(rng() % (nd + 1)) : nd;
            std::int32_t rank[3 * S] = {};
            for (int q = 0; q <= nd; ++q) {
                const float m = 5.0f + static_cast<float>(rng() % 1000) / 37.0f;
                const float ld = m - static_cast<float>(rng() % 400) / 100.0f;
                rank[row0 + q] = static_cast<std::int32_t>(rng() % 9);
                rank[S + row0 + q] = spec_relax_bits(m);
                rank[2 * S + row0 + q] = spec_relax_bits(ld);
            }
            SpecAccArgs a{};
            a.n = 1;
            a.n_vocab = static_cast<int>(V);
            a.stop0 = rng() % 3 == 0 ? 1u : 0xffffffffu;
            a.stop1 = 0xffffffffu;
            a.seq[0] = 0;
            a.nd[0] = nd;
            a.dev[0] = 1;
            a.eos[0] = a.stop0 != 0xffffffffu ? 1 : 0;
            a.budget[0] = rng() % 4 == 0 ? 1 + static_cast<int>(rng() % 4) : 1 << 30;
            a.row0[0] = row0;
            if (rng() % 8 == 0) host_in = true;  // a drained cycle before: the next one takes host inputs
            if (host_in) {
                hostWrite();
                a.pos[0] = static_cast<int>(cache.size());
                r.next[0] = static_cast<int>(next);
                n_hostin += 1;
            } else {
                a.pos[0] = -1;  // the NPOS / CHOSEN words of the last accept (device inputs)
                r.next[0] = -12345;
            }
            // device
            const int pos = spec_pos(a, 0, ctl), dnext = spec_relax_next(a, r, 0, ctl);
            CHECK_EQ(pos, static_cast<int>(cache.size()));
            CHECK_EQ(static_cast<std::uint32_t>(dnext), next);
            SpecRelaxScalarGuard g{ring.data(), pos, pos < WHIRL_SR_WIN ? pos : WHIRL_SR_WIN, dnext, &a, 0, ctl, WHIRL_SR_NGRAM};
            SpecRelaxOut ro{};
            const SpecAccResult dv = spec_accept_relax_one(a, r, 0, ctl, rank, g, ro);
            // host: Acceptor::greedy (relax, one-pass rank layout) + the engine's EOS / max_tokens cuts
            std::array<std::uint32_t, qwen35::max_ng_drafts> dr{};
            for (int q = 0; q < nd; ++q) dr[static_cast<std::size_t>(q)] = static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_DRAFTS + q]);
            const std::uint32_t nd_dev = std::min<std::uint32_t>(static_cast<std::uint32_t>(nd), static_cast<std::uint32_t>(std::max(0, ctl[WHIRL_SA_CTL_ND])));
            std::uint32_t relaxed = 0;
            double gap = 0;
            const sampling::RepeatGuard rep = [&](std::uint32_t q) {
                return spec::rules::relaxRepeats(std::span<const std::uint32_t>(cache), next, std::span<const std::uint32_t>(dr.data(), dr.size()), q);
            };
            const sampling::Accepted h =
                acceptor.greedy(ctl + WHIRL_SA_CTL_ROWS, sampling::Drafts{std::span<const std::uint32_t>(dr.data(), dr.size()), nd_dev, static_cast<std::uint32_t>(row0)},
                                true, sampling::RelaxRows{reinterpret_cast<const std::uint8_t*>(rank), S, 0, true}, rep, relaxed, gap);
            std::uint32_t acc = h.acc, chosen = h.chosen;
            if (a.eos[0])
                for (std::uint32_t q = 0; q < acc; ++q)
                    if (dr[q] == a.stop0) {
                        chosen = dr[q];
                        acc = q;
                        n_eos += 1;
                        break;
                    }
            const std::uint32_t budget = static_cast<std::uint32_t>(a.budget[0]);
            if (budget >= 1 && acc + 1 > budget) {
                chosen = dr[budget - 1];
                acc = budget - 1;
                n_budget += 1;
            }
            CHECK_EQ(static_cast<std::uint32_t>(dv.acc0), h.acc);
            CHECK_EQ(static_cast<std::uint32_t>(dv.acc), acc);
            CHECK_EQ(static_cast<std::uint32_t>(dv.chosen), chosen);
            CHECK_EQ(static_cast<std::uint32_t>(dv.nd_dev), nd_dev);
            CHECK_EQ(static_cast<std::uint32_t>(std::popcount(ro.mask)), relaxed);
            if (relaxed <= WHIRL_SR_GAP_N) {
                double dg = 0;
                for (int q = 0; q < ro.n_gap; ++q) dg += static_cast<double>(ro.gap[q]);
                CHECK_EQ(dg, gap);
            }
            n_relaxed += relaxed;
            n_guarded += ro.guard > 0 ? 1 : 0;
            // the device stores and appends, the host appends cache_tokens
            spec_store(a, 0, dv, ctl);
            spec_relax_store(ro, ctl);
            spec_relax_append(ring.data(), pos, dnext, a, 0, ctl, dv.acc);
            cache.push_back(next);
            cache.insert(cache.end(), dr.begin(), dr.begin() + acc);
            next = chosen;
            CHECK_EQ(ctl[WHIRL_SA_CTL_NPOS], static_cast<std::int32_t>(cache.size()));
            CHECK_EQ(static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_CHOSEN]), next);
            host_in = false;
            n_cyc += 1;
        }
    }
    // ---- (4) the kernel's private copies of the spec_accept.h helpers (sr_*: no code shared with spec_accept_rows) = the originals
    std::uint64_t n_copy = 0;
    for (int it = 0; it < 20000; ++it) {
        SpecAccArgs a{};
        const int A = 1 + static_cast<int>(rng() % WHIRL_SA_MAX_SEG);
        a.n = A;
        a.n_vocab = 1000 - (rng() % 4 == 0 ? 10 : 0);
        a.stop0 = 998;
        a.stop1 = rng() % 5 == 0 ? 0xffffffffu : 999u;
        std::vector<std::int32_t> all(static_cast<std::size_t>(WHIRL_SA_MAX_SEG) * WHIRL_SA_CTL_WORDS);
        for (auto& v : all) v = static_cast<std::int32_t>(rng() % 1000);
        int row = 0;
        for (int k = 0; k < A; ++k) {
            a.seq[k] = k;
            const int nd = static_cast<int>(rng() % 4);
            a.nd[k] = nd;
            a.lit[k] = rng() % 3 == 0 ? 1 : 0;
            a.dev[k] = rng() % 6 == 0 ? 0 : 1;
            a.eos[k] = rng() % 4 == 0 ? 0 : 1;
            a.budget[k] = rng() % 3 == 0 ? static_cast<int>(rng() % 5) : 100;
            a.pos[k] = rng() % 4 == 0 ? -1 : static_cast<int>(rng() % 100000);
            a.row0[k] = row;
            row += nd + 1;
            a.kvbase[k] = static_cast<int>(rng() % 4096);
            std::int32_t* cw = all.data() + k * WHIRL_SA_CTL_WORDS;
            cw[WHIRL_SA_CTL_ND] = static_cast<std::int32_t>(rng() % 6) - 1;
            const int match = static_cast<int>(rng() % (nd + 2));
            for (int r = 0; r < nd; ++r) {
                std::uint32_t d = static_cast<std::uint32_t>(rng() % 1000);
                if (rng() % 8 == 0) d = rng() % 2 ? 998u : 999u;
                if (a.lit[k]) a.lit_d[k][r] = static_cast<int>(d);
                else cw[WHIRL_SA_CTL_DRAFTS + r] = static_cast<std::int32_t>(d);
                if (r < match) cw[WHIRL_SA_CTL_ROWS + r] = static_cast<std::int32_t>(d);
            }
        }
        std::vector<std::int32_t> c1 = all, c2 = all;
        SpecAccResult r1[WHIRL_SA_MAX_SEG], r2[WHIRL_SA_MAX_SEG];
        for (int k = 0; k < A; ++k) {
            std::int32_t* w1 = c1.data() + k * WHIRL_SA_CTL_WORDS;
            std::int32_t* w2 = c2.data() + k * WHIRL_SA_CTL_WORDS;
            r1[k] = a.dev[k] ? spec_accept_one(a, k, w1) : spec_copy_one(a, k, w1);
            r2[k] = a.dev[k] ? sr_accept_one(a, k, w2) : sr_copy_one(a, k, w2);
            CHECK(std::memcmp(&r1[k], &r2[k], sizeof(SpecAccResult)) == 0);
            CHECK_EQ(spec_draft(a, k, w1, 0), sr_draft(a, k, w2, 0));
            spec_store(a, k, r1[k], w1);
            sr_store(a, k, r2[k], w2);
        }
        CHECK(c1 == c2);
        SpecNextTab t1{}, t2{};
        spec_next_rows(a, c1.data(), r1, &t1);
        sr_next_rows(a, c2.data(), r2, &t2);
        CHECK(std::memcmp(&t1, &t2, sizeof(SpecNextTab)) == 0);
        n_copy += 1;
    }
    CHECK(n_copy == 20000);
    std::printf("  e1_fast_rule: d* %llu values, rank rows %llu (%llu kept), guard %llu (%llu repeats), chain %llu cycles (%llu on host "
                "inputs): %llu relaxed keeps, %llu guard refusals, %llu EOS cuts, %llu budget cuts\n",
                static_cast<unsigned long long>(n_d), static_cast<unsigned long long>(n_row), static_cast<unsigned long long>(n_row_yes),
                static_cast<unsigned long long>(n_guard), static_cast<unsigned long long>(n_guard_yes), static_cast<unsigned long long>(n_cyc),
                static_cast<unsigned long long>(n_hostin), static_cast<unsigned long long>(n_relaxed), static_cast<unsigned long long>(n_guarded),
                static_cast<unsigned long long>(n_eos), static_cast<unsigned long long>(n_budget));
    CHECK(n_row_yes > 1000 && n_guard_yes > 500 && n_guard_yes + 500 < n_guard);  // both answers of the guard are exercised
    CHECK(n_relaxed > 1000 && n_guarded > 100 && n_eos > 50 && n_budget > 100 && n_hostin > 500);
}

void testE1FastAhead() {
    struct Case {
        const char* name;
        std::uint32_t nd;
        bool snap;
        std::uint32_t spec_acc;
        std::uint32_t markov;
        double eos_rate;
    };
    const Case cases[] = {{"ornith-like nd 1 snapshot", 1, true, 0, 0, 0.0},       {"nd 1 snapshot markov", 1, true, 0, 24, 0.0},
                          {"nd 3 snapshot markov", 3, true, 0, 24, 0.0},           {"nd 1 snapshot check", 1, true, 4, 24, 0.0},
                          {"nd 2 replay gpu-zc eos", 2, false, 3, 0, 0.03},        {"nd 1 replay markov", 1, false, 0, 24, 0.0},
                          {"nd 2 snapshot eos markov", 2, true, 0, 24, 0.03}};
    std::uint64_t guarded = 0, discard = 0;
    for (const Case& c : cases) {
        std::vector<e1test::ReqOut> out[2];
        std::vector<std::string> lines[2];
        for (std::uint32_t arm = 0; arm < 2; ++arm) {
            const std::uint32_t e1 = arm == 0 ? 0u : 2u;
            ServerSetup su = e1fast::setup(e1, c.nd, c.spec_acc, c.snap, true);
            su.mc.markov = c.markov;
            su.mc.runner_up_markov = c.markov > 0;  // the relaxed keeps repeat with the text: the repetition guard refuses some
            su.mc.eos_rate = c.eos_rate;
            g_log.clear();
            TestServer srv(su);
            const std::vector<std::uint32_t> p0 = makePrompt(300, 1901), p1 = makePrompt(420, 1902), p2 = makePrompt(260, 1903);
            const std::pair<const std::vector<std::uint32_t>*, std::uint32_t> reqs[] = {{&p0, 160}, {&p1, 90}, {&p2, 9}, {&p0, 160}, {&p2, 13}, {&p1, 220}};
            const bool ign = c.eos_rate == 0.0;
            for (const auto& [p, mt] : reqs) {
                const HttpResult r = srv.post("/v1/completions", e1test::greedyBody(*p, mt, ign));
                CHECK_EQ(r.status, 200);
                out[arm].push_back(e1test::reqOut(r));
            }
            for (int i = 0; i < 500 && e1fast::relaxLines().size() < std::size(reqs); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
            CHECK(e1fast::relaxLines().size() >= std::size(reqs));
            lines[arm] = e1fast::relaxLines();
            const Engine::Stats es = srv.engine().stats();
            const MockModel::RelaxCounts rc = srv.model().relaxCounts();
            const MockModel::E1Counts e = srv.model().e1Counts();
            std::printf("  e1_fast_ahead %s E1=%u: ring %llu, ahead %llu, discard %llu, checked %llu, mismatch %llu | relaxed launches %llu, seqs %llu, "
                        "relaxed keeps %llu, guard refusals %llu\n",
                        c.name, e1, static_cast<unsigned long long>(es.acc_ring), static_cast<unsigned long long>(es.acc_ahead),
                        static_cast<unsigned long long>(es.acc_discard), static_cast<unsigned long long>(es.acc_checked),
                        static_cast<unsigned long long>(es.acc_mismatch), static_cast<unsigned long long>(rc.launches),
                        static_cast<unsigned long long>(rc.seqs), static_cast<unsigned long long>(rc.relaxed), static_cast<unsigned long long>(rc.guarded));
            CHECK_EQ(srv.model().poison(), 0u);
            CHECK_EQ(srv.model().hidMismatch(), 0u);
            CHECK_EQ(e.disagree, 0u);
            CHECK_EQ(es.acc_fallback, 0u);
            CHECK_EQ(es.acc_mismatch, 0u);
            CHECK_EQ(rc.orphan, 0u);
            if (e1 == 0) {
                CHECK(es.acc_ring == 0 && rc.launches == 0);
            } else {
                CHECK(es.acc_ring > 0);
                CHECK(es.acc_ahead > 0);
                CHECK(rc.launches > 0 && rc.seqs > 0);
                CHECK(rc.relaxed > 0);
                CHECK_EQ(g_log.count("decode: E1 with relaxacc decided on the GPU"), 1u);
                if (c.spec_acc == 4) CHECK(es.acc_checked > 0);
                guarded += rc.guarded;
                discard += es.acc_discard;
            }
        }
        CHECK(out[0] == out[1]);
        CHECK(lines[0] == lines[1]);
        for (std::size_t i = 0; i < out[0].size() && i < out[1].size(); ++i)
            if (!(out[0][i] == out[1][i])) std::printf("  e1_fast_ahead %s: request %zu differs between E1=0 and E1=2\n", c.name, i);
        CHECK_EQ(lines[0].size(), std::size(out[0]));
        bool kept_relaxed = false;
        for (const std::string& l : lines[1]) kept_relaxed = kept_relaxed || l.rfind("relaxacc: 0 of", 0) != 0;
        CHECK(kept_relaxed);
    }
    std::printf("  e1_fast_ahead: guard refusals on the device path %llu, cycles dropped at an end inside the window %llu\n",
                static_cast<unsigned long long>(guarded), static_cast<unsigned long long>(discard));
    CHECK(guarded > 0);
    CHECK(discard > 0);
}

void testE1FastGate() {
#ifdef WHIRL_SOURCE_DIR
    {
        std::ifstream in(std::string(WHIRL_SOURCE_DIR) + "/docs/usage.md", std::ios::binary);
        const std::string doc((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(doc.find("In fast mode an explicitly set `WHIRL_E1=2` runs too on the Radeon 8060S") != std::string::npos);
        CHECK(doc.find("`decode: E1 with relaxacc decided on the GPU`") != std::string::npos);
        // v0252-e1fast: the fast default in the env table (and WHIRL_E1=0 turns it off)
        CHECK(doc.find("in fast mode `2` for MoE models on the Radeon 8060S below `131329`") != std::string::npos);
        CHECK_EQ(doc.find("in fast mode (no default yet)"), std::string::npos);
        std::ifstream rin(std::string(WHIRL_SOURCE_DIR) + "/src/release/release.cpp", std::ios::binary);
        const std::string rel((std::istreambuf_iterator<char>(rin)), std::istreambuf_iterator<char>());
        CHECK(rel.find("and in fast mode for MoE models on the Radeon 8060S below 131329 (relaxed acceptance decided on the GPU; WHIRL_E1=0 turns it off)") !=
              std::string::npos);
    }
#endif
    // ---- no default for fast (the main session's screen decides; this pins today's chooser)
    for (const bool moe : {true, false}) {
        const E1Site site{.mtp = true, .dflash = false, .relax = true, .fast = true, .model_ok = true};
        const E1Choice c = chooseE1(std::nullopt, qwen35::e1Default(moe, hip::Arch::gfx1151), site);
        CHECK_EQ(c.e1, 0u);
        CHECK_EQ(chooseE1(std::string("2"), qwen35::e1Default(moe, hip::Arch::gfx1151), site).e1, 2u);
    }
    // ---- fast's own default (device::Profile::e1_moe_fast through qwen35::e1DefaultFast; 0 on both rows today) applies only
    // where the device decides relaxacc (E1Site::relax_dev); the precise / balance default never leaks into fast
    {
        E1Site site{.mtp = true, .dflash = false, .relax = true, .fast = true, .model_ok = true};
        // v0252-e1fast: screen e1fast-orn 20261125-1545 KEEP -> the gfx1151 row is 2 (MoE), dense 0, the R9700 0
        CHECK_EQ(qwen35::e1DefaultFast(true, hip::Arch::gfx1151), 2u);
        CHECK_EQ(qwen35::e1DefaultFast(false, hip::Arch::gfx1151), 0u);
        CHECK_EQ(qwen35::e1DefaultFast(true, hip::Arch::gfx1201), 0u);
        CHECK_EQ(qwen35::e1CtxMaxFast(hip::Arch::gfx1151), 131329u);
        site.relax_dev = true;
        site.fast_def = 0;
        CHECK_EQ(chooseE1(std::nullopt, 2u, site).e1, 0u);  // fast_def 0: the balance default (2) is not fast's
        site.fast_def = qwen35::e1DefaultFast(true, hip::Arch::gfx1151);
        const E1Choice on = chooseE1(std::nullopt, 0u, site);
        CHECK(on.e1 == 2u && on.by_default);
        CHECK_EQ(e1DefaultCtxMax(on, 99999u), 99999u);
        site.relax_dev = false;                              // a code object without spec_accept_relax_rows (gfx1201)
        CHECK_EQ(chooseE1(std::nullopt, 0u, site).e1, 0u);
        site.relax_dev = true;
        site.dflash = true;                                  // DFlash2 drafts: never
        CHECK_EQ(chooseE1(std::nullopt, 0u, site).e1, 0u);
        site.dflash = false;
        site.spec_acc_env = 2u;                              // an explicit WHIRL_SPEC_ACC=gpu wins over the default
        const E1Choice y = chooseE1(std::nullopt, 0u, site);
        CHECK(y.e1 == 0u && y.yielded);
        // precise / balance (fast false) are unchanged by the new fields
        E1Site pb{.mtp = true, .dflash = false, .relax = false, .fast = false, .model_ok = true};
        pb.fast_def = 2;
        pb.relax_dev = true;
        CHECK_EQ(chooseE1(std::nullopt, 2u, pb).e1, 2u);
        CHECK_EQ(chooseE1(std::nullopt, 0u, pb).e1, 0u);
    }
#ifdef WHIRL_SOURCE_DIR
    {
        std::ifstream in(std::string(WHIRL_SOURCE_DIR) + "/src/server/server_main.cpp", std::ios::binary);
        const std::string sm((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(sm.find(".fast_def = qwen35::e1DefaultFast(model.cfg.moe, model.arch)") != std::string::npos);
        CHECK(sm.find(".relax_dev = backend->specRelaxOk()") != std::string::npos);
        CHECK(sm.find("if (nreq.mode == numerics::Mode::fast) eo.e1_ctx_max = e1DefaultCtxMax(e1c, qwen35::e1CtxMaxFast(model.arch));") !=
              std::string::npos);
    }
#endif
    const std::vector<std::uint32_t> p = makePrompt(300, 1997);
    std::string ref_text;
    {
        ServerSetup su = e1fast::setup(0, 1, 0, true, true);
        su.mc.markov = 24;
        TestServer srv(su);
        ref_text = completionText(srv.post("/v1/completions", completionBody(p, 120)));
        CHECK_EQ(srv.model().relaxCounts().launches, 0u);
    }
    // ---- the shipped fast DEFAULT on gfx1151, built as server_main builds it (chooseE1 with fast_def = e1DefaultFast, the limit
    // e1CtxMaxFast): MoE fast starts the lookahead with the relaxed rule on the GPU and the boundary line; dense fast prints none
    for (const bool moe : {true, false}) {
        E1Site site{.mtp = true, .dflash = false, .relax = true, .fast = true, .model_ok = true};
        site.fast_def = qwen35::e1DefaultFast(moe, hip::Arch::gfx1151);
        site.relax_dev = true;
        const E1Choice c = chooseE1(std::nullopt, qwen35::e1Default(moe, hip::Arch::gfx1151), site);
        ServerSetup su = e1fast::setup(c.e1, 1, 0, true, true);
        su.mc.markov = 24;
        su.eo.e1_ctx_max = e1DefaultCtxMax(c, qwen35::e1CtxMaxFast(hip::Arch::gfx1151));
        g_log.clear();
        TestServer srv(su);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 120))), ref_text);
        if (moe) {
            CHECK(c.e1 == 2u && c.by_default);
            CHECK_EQ(su.eo.e1_ctx_max, 131329u);
            CHECK_EQ(g_log.count("decode: WHIRL_E1=2 by default below 131329 tokens of context"), 1u);
            CHECK_EQ(g_log.count("decode: E1 with relaxacc decided on the GPU"), 1u);
            CHECK(srv.engine().stats().acc_ring > 0);
        } else {
            CHECK_EQ(c.e1, 0u);
            CHECK_EQ(g_log.count("decode: WHIRL_E1="), 0u);
            CHECK_EQ(srv.engine().stats().acc_ring, 0u);
        }
    }
    // ---- a model without spec_accept_relax_rows: fast cycles stay synchronous, said once
    {
        ServerSetup su = e1fast::setup(2, 1, 0, true, false);
        su.mc.markov = 24;
        g_log.clear();
        TestServer srv(su);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 120))), ref_text);
        CHECK_EQ(g_log.count("decode: E1: relaxacc cycles run the ordinary way (no spec_accept_relax_rows kernel)"), 1u);
        CHECK_EQ(g_log.count("decode: E1 with relaxacc decided on the GPU"), 0u);
        CHECK_EQ(srv.engine().stats().acc_ring, 0u);
        CHECK_EQ(srv.model().relaxCounts().launches, 0u);
    }
    // ---- with it: the proof line; WHIRL_E1=2 runs the ring, WHIRL_E1=1 decides on the device one cycle at a time
    for (const std::uint32_t e1 : {1u, 2u}) {
        ServerSetup su = e1fast::setup(e1, 1, 0, true, true);
        su.mc.markov = 24;
        g_log.clear();
        TestServer srv(su);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 120))), ref_text);
        CHECK_EQ(g_log.count("decode: E1 with relaxacc decided on the GPU"), 1u);
        const MockModel::RelaxCounts rc = srv.model().relaxCounts();
        CHECK(rc.launches > 0 && rc.seqs > 0);
        if (e1 == 2) CHECK(srv.engine().stats().acc_ring > 0);
        else CHECK_EQ(srv.engine().stats().acc_ring, 0u);
        // a sampled request in fast (typical acceptance on the host) never looks ahead and is never decided on the device
        const std::uint64_t ring0 = srv.engine().stats().acc_ring, seqs0 = rc.seqs;
        const std::string body = std::format("{{\"prompt\":{},\"max_tokens\":60,\"temperature\":0.8,\"seed\":3,\"ignore_eos\":true}}", idsJson(p));
        CHECK_EQ(srv.post("/v1/completions", body).status, 200);
        CHECK_EQ(srv.engine().stats().acc_ring, ring0);
        CHECK_EQ(srv.model().relaxCounts().seqs, seqs0);
        CHECK_EQ(srv.model().relaxCounts().orphan, 0u);
    }
    // ---- relaxacc off (precise / balance): the relaxed path is never asked for, the lookahead runs as before
    for (const bool snap : {true, false}) {
        ServerSetup su = e1fast::setup(2, 1, 0, snap, true);
        su.eo.relax.on = false;
        g_log.clear();
        TestServer srv(su);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 120))), decode(srv.model().greedyReference(p, 120, true)));
        CHECK_EQ(srv.model().relaxCounts().launches, 0u);
        CHECK_EQ(srv.model().relaxCounts().seqs, 0u);
        CHECK_EQ(g_log.count("decode: E1 with relaxacc decided on the GPU"), 0u);
        CHECK_EQ(g_log.count("relaxacc cycles run the ordinary way"), 0u);
        CHECK(srv.engine().stats().acc_ring > 0);
    }
}
