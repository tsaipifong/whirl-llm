// arch025-E1a: fixed acceptance cases for the Acceptor tests (server_tests.cpp) and the capture harness
// (scratch\a25_e1a\capture.cpp, which ran them on the pre-move code in sampling_oldref.inc to record the
// expected values). A case = fixed verify-row logits + drafts + sampler settings + seed; the result is the
// accept count, the chosen token, the relaxacc counters, the trace of device reads (row prepares with the
// pos_idx they saw, full-row loads, single-logit reads) and the uniform draws at every output index used.
// Include inside namespace whirl::server, after sampling_oldref.inc.
// SPDX-License-Identifier: Apache-2.0

namespace acase {

struct Case {
    const char* name;
    bool greedy;
    bool relax;      // relaxacc applies to these drafts
    bool race;       // specsample race sampler
    double temp;
    std::uint32_t top_k;
    double top_p;
    double min_p;
    std::uint64_t seed;
    std::uint32_t n_gen;
    std::uint32_t nd;    // drafts verified
    std::uint32_t row0;  // first verify row
    std::uint32_t V;
    std::uint32_t lseed;      // logits generator
    float spread;             // logit scale (small = flat / open rows)
    const char* drafts;       // per draft: '0'..'3' = that rank of the row, 'x' = a low-ranked token
    bool rep_hist;            // relaxacc: history that makes the repeat guard fire on draft 1
};

// ks / kb: the device top-k sizes of the fake (the engine uses k_small 256 / k_big 16384)
constexpr std::uint32_t ks = 4, kb = 16, max_rows_t = 8;

inline std::uint32_t lcg(std::uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}

inline std::vector<std::vector<float>> makeRows(const Case& c) {
    std::vector<std::vector<float>> rows(c.row0 + c.nd + 1, std::vector<float>(c.V));
    std::uint32_t s = c.lseed;
    for (auto& r : rows)
        for (auto& l : r) l = static_cast<float>(static_cast<int>(lcg(s) % 2001) - 1000) / 1000.0f * c.spread;
    return rows;
}

// tokens of row r by (logit desc, id asc), the device top-k order
inline std::vector<std::uint32_t> order(const std::vector<float>& L) {
    std::vector<std::uint32_t> o(L.size());
    for (std::uint32_t i = 0; i < o.size(); ++i) o[i] = i;
    std::sort(o.begin(), o.end(), [&](std::uint32_t a, std::uint32_t b) { return L[a] != L[b] ? L[a] > L[b] : a < b; });
    return o;
}

struct Trace {
    std::vector<std::uint64_t> ev;  // (kind << 56) | (a << 28) | b
    void add(std::uint64_t kind, std::uint64_t a, std::uint64_t b) { ev.push_back((kind << 56) | (a << 28) | b); }
    std::uint64_t hash() const {
        std::uint64_t h = 1469598103934665603ull;
        for (std::uint64_t e : ev)
            for (int i = 0; i < 8; ++i) h = (h ^ ((e >> (8 * i)) & 0xff)) * 1099511628211ull;
        return h;
    }
};
constexpr std::uint64_t ev_prep = 1, ev_full = 2, ev_logit = 3;

// Engine::prepareRow over host rows: top-ks candidates with the full row's (max, sum), then top-kb, then the
// full row (the same escalation as the engine); S = old or new Sampler, F = its filterCands
template <class S, class F>
void hostPrepare(S& s, std::uint32_t r, const std::vector<std::vector<float>>& rows, F filter, Trace& tr) {
    tr.add(ev_prep, r, s.pos_idx);
    const std::uint32_t gr = s.row_base + r;
    s.row = gr;
    const std::vector<float>& L = rows[gr];
    const std::vector<std::uint32_t> o = order(L);
    s.m = L[o[0]];
    double z = 0;
    for (float l : L) z += std::exp(static_cast<double>(l - s.m) * s.inv_t);
    s.sum = static_cast<float>(z);
    auto& c = *s.cands;
    for (std::uint32_t j = 0; j < ks; ++j) c[j] = {o[j], L[o[j]], 0};
    if (filter(s, ks, false)) return;
    s.big_used += 1;
    for (std::uint32_t j = 0; j < kb; ++j) c[j] = {o[j], L[o[j]], 0};
    if (filter(s, kb, false)) return;
    s.full_used += 1;
    const std::uint32_t V = static_cast<std::uint32_t>(L.size());
    for (std::uint32_t j = 0; j < V; ++j) c[j] = {j, L[j], 0};
    filter(s, V, true);
}

struct Inputs {
    std::vector<std::vector<float>> rows;
    std::array<std::uint32_t, 15> dr{};  // qwen35::max_ng_drafts
    std::vector<std::int32_t> ctl;       // argmax of rows row0.. (ctl_rows = 0)
    std::vector<std::uint8_t> relax_host;
    std::vector<std::uint32_t> hist;     // the slot's cache_tokens
    std::uint32_t next = 0;
    whirl::relax::Params rp;
};

inline Inputs makeInputs(const Case& c) {
    Inputs in;
    in.rows = makeRows(c);
    in.ctl.assign(64, 0);
    for (std::uint32_t r = 0; r <= c.nd; ++r) in.ctl[r] = static_cast<std::int32_t>(order(in.rows[c.row0 + r])[0]);
    for (std::uint32_t r = 0; r < c.nd; ++r) {
        const std::vector<std::uint32_t> o = order(in.rows[c.row0 + r]);
        const char d = c.drafts[r];
        in.dr[r] = d == 't' ? 0 : d == 'x' ? o[c.V - 1 - r] : o[static_cast<std::uint32_t>(d - '0')];
    }
    in.rp.on = c.relax;
    in.rp.k = 4;
    in.rp.alpha = 0.1f;
    in.rp.eps = 0.09f;
    in.rp.delta = 0.3f;
    const std::uint32_t rk = in.rp.topk();
    in.relax_host.assign(static_cast<std::size_t>(max_rows_t) * rk * 8 + max_rows_t * 8, 0);
    for (std::uint32_t gr = 0; gr < in.rows.size() && gr < max_rows_t; ++gr) {
        const std::vector<float>& L = in.rows[gr];
        const std::vector<std::uint32_t> o = order(L);
        auto* ids = reinterpret_cast<std::int32_t*>(in.relax_host.data()) + gr * rk;
        auto* vals = reinterpret_cast<float*>(in.relax_host.data() + static_cast<std::size_t>(max_rows_t) * rk * 4) + gr * rk;
        auto* st = reinterpret_cast<float*>(in.relax_host.data() + static_cast<std::size_t>(max_rows_t) * rk * 8);
        for (std::uint32_t j = 0; j < rk; ++j) {
            ids[j] = static_cast<std::int32_t>(o[j]);
            vals[j] = L[o[j]];
        }
        double z = 0;
        for (float l : L) z += std::exp(static_cast<double>(l - L[o[0]]));
        st[2 * gr] = L[o[0]];
        st[2 * gr + 1] = static_cast<float>(z);
    }
    in.next = 7;
    if (c.rep_hist) {
        // history ends in ... a b c d e and earlier holds "next dr0 dr1"-completing 6-gram: next, dr0, dr1 after
        // [h0 h1 h2] occurs before, so keeping dr1 would repeat [h0 h1 h2 next dr0 dr1]
        const std::uint32_t h[3] = {11, 12, 13};
        in.hist = {h[0], h[1], h[2], in.next, in.dr[0], in.dr[1], 50, 51, h[0], h[1], h[2]};
    } else {
        in.hist = {1, 2, 3, 4, 5, 6, 7, 8};
    }
    return in;
}

struct Result {
    std::uint32_t acc = 0, chosen = 0, relaxed = 0;
    std::uint64_t gap_bits = 0;
    std::uint64_t pos_idx = 0;
    std::uint32_t full_used = 0, big_used = 0;
    Trace tr;
    std::vector<std::uint64_t> draws;  // bits of the uniform at each output index n_gen .. n_gen + acc
};

inline std::uint64_t bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, 8);
    return b;
}

// a device read of the old engine (address, bytes) -> a full-row or a single-logit event
inline void logRead(const std::vector<std::vector<float>>& rows, std::uint64_t p, std::uint64_t n, Trace& tr) {
    for (std::uint32_t r = 0; r < rows.size(); ++r) {
        const auto b = reinterpret_cast<std::uint64_t>(rows[r].data());
        if (p >= b && p < b + rows[r].size() * 4) {
            if (n == 4) tr.add(ev_logit, r, (p - b) / 4);
            else tr.add(ev_full, r, 0);
            return;
        }
    }
    tr.add(0xff, 0, 0);
}

// draft 't': the token the run itself gives after drafts 0..r-1 (a run with nd = r), so sampled drafts
// that match the target draw are accepted
template <class Core>
Result runResolved(const Case& c, Core core) {
    Inputs in = makeInputs(c);
    for (std::uint32_t r = 0; r < c.nd; ++r) {
        if (c.drafts[r] != 't') continue;
        Case t = c;
        t.nd = r;
        in.dr[r] = core(t, in).chosen;
    }
    return core(c, in);
}

// the old code (sampling_oldref.inc)
inline Result runOldIn(const Case& c, Inputs& in) {
    Result res;
    oldref::OldEngine e;
    e.m_.c.n_vocab = c.V;
    e.opt_.relax = in.rp;
    e.row_host_.assign(c.V, 0.0f);
    e.in_cands_.assign(c.V, 0);
    e.relax_host_ = in.relax_host;
    e.rows = &in.rows;
    e.max_rows = max_rows_t;
    e.ops_.on = [&](oldref::DevPtr p, std::uint64_t n) { logRead(in.rows, p, n, res.tr); };
    e.prep = [&](oldref::Sampler& s, std::uint32_t r) {
        hostPrepare(s, r, in.rows, [](oldref::Sampler& x, std::size_t n, bool comp) { return oldref::filterCands(x, n, comp); }, res.tr);
    };
    std::vector<oldref::Cand> cands(std::max(c.V, kb));
    oldref::OldSlot sl;
    sl.greedy = c.greedy;
    sl.sm.temp = c.temp;
    sl.sm.inv_t = c.temp > 0 ? static_cast<float>(1.0 / c.temp) : 1.0f;
    sl.sm.top_k = c.top_k;
    sl.sm.top_p = c.top_p;
    sl.sm.min_p = c.min_p;
    sl.sm.seed = c.seed;
    sl.sm.race = c.race;
    sl.sm.cands = &cands;
    sl.n_gen = c.n_gen;
    sl.next = in.next;
    sl.cache_tokens = in.hist;
    e.accept(sl, in.ctl.data(), c.row0, c.nd, in.dr, c.relax, res.acc, res.chosen);
    res.relaxed = sl.relaxed;
    res.gap_bits = bits(sl.relax_gap);
    res.pos_idx = sl.sm.pos_idx;
    res.full_used = sl.sm.full_used;
    res.big_used = sl.sm.big_used;
    for (std::uint32_t i = 0; i <= res.acc; ++i) {
        oldref::Sampler u;
        u.seed = c.seed;
        u.pos_idx = c.n_gen + i;
        res.draws.push_back(bits(u.uniform()));
    }
    return res;
}
inline Result runOld(const Case& c) { return runResolved(c, runOldIn); }

// spread: 0.3 = flat rows (open distributions, tail walks), 4 = peaked rows
inline const std::vector<Case>& cases() {
    static const std::vector<Case> cs = {
        // name                  greedy relax  race   temp top_k top_p min_p seed n_gen nd row0 V  lseed spread drafts rep
        {"greedy_all_match",       true, false, false, 0,   0,  1,    0,    0,  10,  3, 0, 64, 101, 4.0f, "000", false},
        {"greedy_reject_1",        true, false, false, 0,   0,  1,    0,    0,  10,  3, 1, 64, 102, 4.0f, "010", false},
        {"greedy_reject_0",        true, false, false, 0,   0,  1,    0,    0,  10,  2, 0, 64, 103, 4.0f, "x0", false},
        {"greedy_no_drafts",       true, false, false, 0,   0,  1,    0,    0,  10,  0, 0, 64, 104, 4.0f, "", false},
        {"relax_greedy_rank1",     true, true,  false, 0,   0,  1,    0,    0,  10,  3, 0, 64, 105, 1.0f, "101", false},
        {"relax_greedy_rank3",     true, true,  false, 0,   0,  1,    0,    0,  10,  4, 2, 64, 106, 0.5f, "3210", false},
        {"relax_greedy_far",       true, true,  false, 0,   0,  1,    0,    0,  10,  3, 0, 64, 107, 4.0f, "1x0", false},
        {"relax_greedy_repeat",    true, true,  false, 0,   0,  1,    0,    0,  10,  3, 0, 64, 108, 0.5f, "112", true},
        {"sample_closed_topk",     false, false, false, 0.8, 8, 1,    0,    5,  20,  4, 0, 64, 109, 2.0f, "0000", false},
        {"sample_topp",            false, false, false, 1.0, 0, 0.9,  0,    6,  21,  4, 1, 64, 110, 2.0f, "0101", false},
        {"sample_minp",            false, false, false, 0.7, 0, 1,    0.05, 7,  22,  3, 0, 64, 111, 2.0f, "000", false},
        {"sample_open_tail",       false, false, false, 1.0, 0, 1,    0,    8,  23,  3, 0, 64, 112, 0.3f, "000", false},
        {"sample_open_tail_b",     false, false, false, 1.5, 0, 1,    0,    9,  0,   4, 2, 64, 113, 0.3f, "0123", false},
        {"sample_escalate_topk",   false, false, false, 1.0, 12, 1,   0,   10,  24,  3, 0, 64, 114, 1.0f, "000", false},
        {"specsample_closed",      false, false, true,  0.8, 8, 1,    0,   11,  25,  4, 0, 64, 115, 2.0f, "0000", false},
        {"specsample_topp",        false, false, true,  1.0, 0, 0.95, 0,   12,  26,  4, 1, 64, 116, 1.0f, "1000", false},
        {"specsample_open",        false, false, true,  1.0, 0, 1,    0,   13,  27,  4, 0, 64, 117, 0.3f, "0000", false},
        {"specsample_open_b",      false, false, true,  1.2, 0, 1,    0,   14,  28,  3, 3, 64, 118, 0.6f, "010", false},
        {"specsample_minp",        false, false, true,  0.9, 0, 1,    0.1, 15,  29,  4, 0, 64, 119, 1.5f, "0000", false},
        {"specsample_no_drafts",   false, false, true,  0.9, 0, 1,    0,   16,  30,  0, 0, 64, 120, 1.5f, "", false},
        {"typical_closed",         false, true,  false, 0.8, 8, 1,    0,   17,  31,  4, 0, 64, 121, 1.0f, "1111", false},
        {"typical_topp",           false, true,  false, 1.0, 0, 0.9,  0,   18,  32,  4, 1, 64, 122, 0.7f, "2101", false},
        {"typical_open_F2_02",     false, true,  false, 1.0, 0, 1,    0,   19,  33,  4, 0, 64, 123, 0.3f, "x3x2", false},
        {"typical_open_race",      false, true,  true,  1.0, 0, 1,    0,   20,  34,  4, 2, 64, 124, 0.3f, "1x10", false},
        {"typical_far",            false, true,  false, 0.8, 0, 0.9,  0,   21,  35,  3, 0, 64, 125, 4.0f, "xxx", false},
        {"sample_t_match",         false, false, false, 0.8, 0,  0.95, 0,  23,  40,  4, 0, 64, 127, 1.0f, "tttt", false},
        {"sample_t_open",          false, false, false, 1.0, 0,  1,    0,  24,  41,  4, 1, 64, 128, 0.3f, "tt0t", false},
        {"specsample_t_match",     false, false, true,  0.8, 8,  1,    0,  25,  42,  4, 0, 64, 129, 1.0f, "tttt", false},
        {"specsample_t_open",      false, false, true,  1.0, 0,  1,    0,  26,  43,  4, 2, 64, 130, 0.3f, "ttt1", false},
        {"specsample_t_minp",      false, false, true,  1.1, 0,  1,  0.02, 27,  44,  3, 0, 64, 131, 0.8f, "ttt", false},
        {"typical_t_open",         false, true,  false, 1.0, 0,  1,    0,  28,  45,  4, 0, 64, 132, 0.3f, "txtt", false},
        {"typical_t_race",         false, true,  true,  0.9, 0,  0.98, 0,  29,  46,  4, 1, 64, 133, 0.6f, "t2tx", false},
        {"typical_repeat",         false, true,  false, 1.0, 8, 1,    0,   22,  36,  3, 0, 64, 126, 0.5f, "112", true},
    };
    return cs;
}

}  // namespace acase
