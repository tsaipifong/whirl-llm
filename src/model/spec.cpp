// MTP draft-count cost model and prompt-lookup (n-gram) drafting.
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (Ngram, NgramPolicy,
// DraftAccept, DraftTiming, pickDrafts); CycleCost, SlotAccept and allocDrafts are the
// per-slot draft allocation of T4-1 (not wired into the engine yet).

#include "whirl/model.h"

#include <algorithm>
#include <cmath>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;

u64 keyS(const std::vector<u32>& t, std::size_t end) {
    return (static_cast<u64>(t[end - 2]) * 0x9E3779B97F4A7C15ull) ^ (static_cast<u64>(t[end - 1]) * 0xC2B2AE3D27D4EB4Full) ^
           (static_cast<u64>(t[end]) * 0x165667B19E3779F9ull);
}

u64 keyL(const std::vector<u32>& t, std::size_t end) {
    u64 h = 0xcbf29ce484222325ull;
    for (std::size_t i = end + 1 - Ngram::long_len; i < end + 1; ++i) h = (h ^ t[i]) * 0x100000001b3ull;
    return h;
}

std::size_t matchLen(const std::vector<u32>& t, std::size_t q) {
    const std::size_t n = t.size();
    std::size_t m = 0;
    while (m <= q && m < Ngram::max_match && t[q - m] == t[n - 1 - m]) m += 1;
    return m;
}

}  // namespace

void DraftAccept::updateRate(u32 nd_dev, u32 acc, float r) {
    const u32 reached = std::min(acc + 1, nd_dev);
    for (u32 k = 0; k < reached; ++k) {
        const float hit = k < acc ? 1.0f : 0.0f;
        alpha[k] = (1 - r) * alpha[k] + r * hit;
    }
}

float DraftAccept::expected(u32 nd) const {
    float e = 1, p = 1;
    for (u32 k = 0; k < nd; ++k) {
        p *= alpha[k];
        e += p;
    }
    return e;
}

void DraftTiming::update(u32 nd, float ms) {
    if (nd > max_ng_drafts) return;
    if (skip_first && !seen[nd]) {
        seen[nd] = true;
        return;
    }
    t[nd] = n[nd] == 0 ? ms : 0.8f * t[nd] + 0.2f * ms;
    n[nd] += 1;
}

std::optional<float> DraftTiming::estimate(u32 nd) const {
    if (n[nd] > 0) return t[nd];
    // weighted least squares t = a + b nd over the measured points
    float sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    u32 pts = 0, only = 0;
    for (u32 i = 0; i < max_ng_drafts + 1; ++i) {
        if (n[i] == 0) continue;
        const float w = static_cast<float>(std::min<u32>(n[i], 8));
        const float xx = static_cast<float>(i);
        sw += w;
        sx += w * xx;
        sy += w * t[i];
        sxx += w * xx * xx;
        sxy += w * xx * t[i];
        pts += 1;
        only = i;
    }
    if (pts == 0) return std::nullopt;
    const float xf = static_cast<float>(nd);
    if (pts == 1) {
        const float t0 = t[only];
        return t0 * (1 + prior_slope * (xf - static_cast<float>(only)));
    }
    const float den = sw * sxx - sx * sx;
    float b = den > 0 ? (sw * sxy - sx * sy) / den : 0;
    const float a0 = (sy - b * sx) / sw;
    b = std::max(b, 0.01f * std::max(a0, 0.1f));
    const float a = (sy - b * sx) / sw;
    return std::max(a + b * xf, 0.05f);
}

u32 pickDrafts(std::span<const DraftAccept* const> accepts, const DraftTiming& timing, u32 max, u32 cycle, u32 prev) {
    if (max <= 1) return std::max<u32>(max, 1);
    u32 best = std::min<u32>(max, 3);
    float best_score = -1, prev_score = -1;
    for (u32 nd = 1; nd <= max; ++nd) {
        const auto t = timing.estimate(nd);
        if (!t) return std::min<u32>(max, 3);
        float e = 0;
        for (const DraftAccept* a : accepts) e += a->expected(nd);
        const float score = e / *t;
        if (nd == prev) prev_score = score;
        if (score > best_score) {
            best_score = score;
            best = nd;
        }
    }
    if (prev_score >= 0.97f * best_score) best = prev;
    if (cycle % 32 == 31) {
        const bool up = (cycle / 32) % 2 == 0;
        if (up && best < max) return best + 1;
        if (!up && best > 1) return best - 1;
    }
    return best;
}

// ---- T4-1: cycle cost, per-slot acceptance, per-slot draft allocation

std::array<double, CycleCost::n_par> CycleCost::feat(const CycleFeat& x) {
    // scaled so every parameter has a similar range: rows / 16, steps / 8, rows * ctx / 1024
    return {1.0, x.rows / 16.0, x.steps / 8.0, x.rows > 16 ? 1.0 : 0.0, x.row_ctx / 1024.0};
}

void CycleCost::reset() {
    th.fill(0);
    for (u32 i = 0; i < n_par; ++i) {
        P[i].fill(0);
        P[i][i] = 1e4;
    }
    n_obs = 0;
}

void CycleCost::rls(const std::array<double, n_par>& z, double y) {
    std::array<double, n_par> pz{};
    for (u32 i = 0; i < n_par; ++i)
        for (u32 j = 0; j < n_par; ++j) pz[i] += P[i][j] * z[j];
    double den = forget;
    for (u32 i = 0; i < n_par; ++i) den += z[i] * pz[i];
    double err = y;
    for (u32 i = 0; i < n_par; ++i) err -= th[i] * z[i];
    for (u32 i = 0; i < n_par; ++i) th[i] += pz[i] / den * err;
    // P = (P - pz pz^T / den) / forget, kept symmetric; bounded so unexcited directions cannot wind up
    for (u32 i = 0; i < n_par; ++i)
        for (u32 j = i; j < n_par; ++j) {
            const double v = (P[i][j] - pz[i] * pz[j] / den) / forget;
            P[i][j] = P[j][i] = v;
        }
    for (u32 i = 0; i < n_par; ++i)
        if (P[i][i] > 1e6) {
            const double f = 1e6 / P[i][i];
            for (u32 j = 0; j < n_par; ++j) {
                P[i][j] *= std::sqrt(f);
                P[j][i] = P[i][j];
            }
            P[i][i] = 1e6;
        }
}

void CycleCost::prior(const DraftTiming& tm, u32 slots, float ctx_k) {
    reset();
    for (u32 nd = 1; nd < max_ng_drafts + 1; ++nd) {
        if (tm.n[nd] == 0) continue;
        CycleFeat x;
        x.rows = static_cast<float>(slots * (nd + 1));
        x.steps = static_cast<float>(nd);
        x.row_ctx = x.rows * ctx_k;
        const u32 w = std::min<u32>(tm.n[nd], 4);
        for (u32 i = 0; i < w; ++i) rls(feat(x), tm.t[nd]);
    }
    n_obs = 0;  // a prior is not a sample
}

void CycleCost::observe(const CycleFeat& x, float ms) {
    rls(feat(x), ms);
    n_obs += 1;
}

float CycleCost::predict(const CycleFeat& x) const {
    const auto z = feat(x);
    double t = 0;
    for (u32 i = 0; i < n_par; ++i) t += th[i] * z[i];
    return static_cast<float>(std::max(t, 0.05));
}

void SlotAccept::observe(u32 nd_dev, u32 acc) {
    const float r = cycles < 16 ? 0.2f : 0.1f;
    cycles += 1;
    nd_dev = std::min(nd_dev, max_ng_drafts);
    if (nd_dev == 0) return;
    acc = std::min(acc, nd_dev);
    a.updateRate(nd_dev, acc, r);
    const u32 reached = std::min(acc + 1, nd_dev);
    for (u32 k = 0; k < reached; ++k) n[k] = std::min(n[k] + 1, n_cap);
    // positions past the last observed one drift towards it; a fully accepted chain did not
    // break, so it is censored (no 0.95 per position) rather than an upper bound
    const bool full = acc == nd_dev;
    const u32 last = reached - 1;
    float target = a.alpha[last];
    for (u32 k = reached; k < max_ng_drafts; ++k) {
        if (!full) target *= drift_decay;
        a.alpha[k] += drift_rate * (target - a.alpha[k]);
    }
}

DraftAccept SlotAccept::effective(const DraftAccept& pool) const {
    DraftAccept e;
    for (u32 k = 0; k < max_ng_drafts; ++k) {
        const float nk = static_cast<float>(n[k]);
        e.alpha[k] = (nk * a.alpha[k] + pool_weight * pool.alpha[k]) / (nk + pool_weight);
    }
    return e;
}

namespace {

struct AllocEval {
    float score = 0;
    float t = 0;
    std::vector<float> e;  // expected tokens per slot
};

AllocEval allocEval(std::span<const AllocSlot> slots, const CycleCost& cost, const std::vector<u32>& d) {
    AllocEval r;
    r.e.resize(slots.size());
    CycleFeat x;
    float sum_e = 0;
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const AllocSlot& s = slots[i];
        const bool ng = s.ng_rows > 0;
        const float rows = ng ? static_cast<float>(s.ng_rows) : static_cast<float>(d[i] + 1);
        x.rows += rows;
        x.row_ctx += rows * s.ctx_k;
        if (!ng) x.steps = std::max(x.steps, static_cast<float>(d[i]));
        r.e[i] = ng ? s.ng_e : s.acc->expected(d[i]);
        sum_e += r.e[i];
    }
    r.t = cost.predict(x);
    r.score = sum_e / r.t;
    return r;
}

}  // namespace

AllocResult allocDrafts(std::span<const AllocSlot> slots, const CycleCost& cost, AllocBudget budget, u32 uniform,
                        std::span<const u32> prev, float x) {
    const std::size_t n = slots.size();
    AllocResult res;
    res.d.assign(n, 0);
    u32 n_mtp = 0, ng_rows = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (slots[i].ng_rows > 0) {
            ng_rows += slots[i].ng_rows;
            continue;
        }
        n_mtp += 1;
        res.d[i] = std::clamp<u32>(uniform, 1, std::max<u32>(slots[i].cap, 1));
    }
    const std::vector<u32> u = res.d;
    const AllocEval eu = allocEval(slots, cost, u);
    res.score = eu.score;
    if (n_mtp < 2 || cost.samples() < 16) return res;

    // acceptances so close that the uniform count is already the best choice
    u32 max_cap = 1;
    for (const AllocSlot& s : slots)
        if (s.ng_rows == 0) max_cap = std::max(max_cap, s.cap);
    bool close = true;
    for (u32 dd = 1; dd <= max_cap && close; ++dd) {
        float lo = 1e30f, hi = 0;
        for (const AllocSlot& s : slots) {
            if (s.ng_rows > 0) continue;
            const float e = s.acc->expected(dd);
            lo = std::min(lo, e);
            hi = std::max(hi, e);
        }
        close = hi <= 1.01f * lo;
    }
    if (close) return res;

    const u32 base_rows = ng_rows + n_mtp;
    const u32 rows_free = budget.rows > base_rows ? budget.rows - base_rows : 0;
    const u32 d_max = std::min(rows_free, budget.snaps);
    const auto feasible = [&](const std::vector<u32>& d, const AllocEval& ev) {
        u32 sum = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (slots[i].ng_rows > 0) continue;
            if (d[i] < 1 || d[i] > slots[i].cap) return false;
            sum += d[i];
        }
        if (sum > d_max) return false;
        for (std::size_t i = 0; i < n; ++i)  // per-slot throughput floor against u
            if (ev.e[i] / ev.t < (1 - x) * eu.e[i] / eu.t) return false;
        return true;
    };

    std::vector<u32> cur = u;
    float cur_score = eu.score;
    std::vector<u32> cand;
    for (res.rounds = 0; res.rounds < 16;) {
        std::vector<u32> best_d;
        float best_s = cur_score * (1 + 1e-6f);
        const auto tryMove = [&](std::size_t i, int di, std::size_t j, int dj) {
            cand = cur;
            if (static_cast<int>(cand[i]) + di < 1) return;
            cand[i] = static_cast<u32>(static_cast<int>(cand[i]) + di);
            if (dj != 0) {
                if (static_cast<int>(cand[j]) + dj < 1) return;
                cand[j] = static_cast<u32>(static_cast<int>(cand[j]) + dj);
            }
            const AllocEval ev = allocEval(slots, cost, cand);
            if (ev.score > best_s && feasible(cand, ev)) {
                best_s = ev.score;
                best_d = cand;
            }
        };
        for (std::size_t i = 0; i < n; ++i) {
            if (slots[i].ng_rows > 0) continue;
            tryMove(i, +1, i, 0);
            tryMove(i, -1, i, 0);
            for (std::size_t j = 0; j < n; ++j)
                if (j != i && slots[j].ng_rows == 0) tryMove(i, +1, j, -1);
        }
        res.rounds += 1;
        if (best_d.empty()) break;
        cur = best_d;
        cur_score = best_s;
    }

    // keep the previous allocation while it is within 2% of the best (and still allowed)
    if (prev.size() == n) {
        const std::vector<u32> p(prev.begin(), prev.end());
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i)
            if ((slots[i].ng_rows > 0) != (p[i] == 0)) ok = false;
        if (ok) {
            const AllocEval ev = allocEval(slots, cost, p);
            if (feasible(p, ev) && ev.score >= 0.98f * cur_score) {
                res.d = p;
                res.score = ev.score;
                res.uniform = p == u;
                return res;
            }
        }
    }
    // 1% hysteresis against the uniform allocation
    if (cur_score <= 1.01f * eu.score) return res;
    res.d = cur;
    res.score = cur_score;
    res.uniform = false;
    return res;
}

void Ngram::reset() {
    map_s.clear();
    map_l.clear();
    prev_s.clear();
    prev_l.clear();
    ntoks.clear();
    indexed = 2;
    cur_q = 0;
    cur_n = 0;
    gen_from = 0;
    gen_cr = false;
}

// Drafts after the best earlier occurrence of the suffix of toks (the longest
// matching suffix, the latest among equals; at least min_match tokens).
Ngram::Match Ngram::lookup(std::span<const u32> toks, u32 min_match, std::span<u32> out) {
    const std::size_t n = toks.size();
    if (n < 4) return {};
    if (gen_from == 0) gen_from = n;
    // normalized history (the last token is re-read every call)
    if (ntoks.size() >= n) ntoks.resize(n - 1);
    while (ntoks.size() < n) {
        const std::size_t i = ntoks.size();
        const u32 v = nt(toks[i]);
        if (i >= gen_from && v != toks[i]) gen_cr = true;
        ntoks.push_back(v);
    }
    const std::vector<u32>& t = ntoks;
    if (prev_s.size() < n) {
        prev_s.resize(n, none);
        prev_l.resize(n, none);
    }
    for (; indexed < n - 1; ++indexed) {
        const std::size_t i = indexed;
        {
            auto [it, inserted] = map_s.try_emplace(keyS(t, i), static_cast<u32>(i));
            prev_s[i] = inserted ? none : it->second;
            it->second = static_cast<u32>(i);
        }
        if (i + 1 >= long_len) {
            auto [it, inserted] = map_l.try_emplace(keyL(t, i), static_cast<u32>(i));
            prev_l[i] = inserted ? none : it->second;
            it->second = static_cast<u32>(i);
        }
    }
    std::size_t best_q = 0, best_m = 0;
    // previous source, advanced by what was emitted since
    if (cur_n > 0 && n > cur_n) {
        const std::size_t q = cur_q + (n - cur_n);
        if (q < n - 1) {
            best_m = matchLen(t, q);
            best_q = q;
        }
    }
    if (n >= long_len && best_m < max_match) {
        auto it = map_l.find(keyL(t, n - 1));
        u32 q = it == map_l.end() ? none : it->second;
        for (u32 tries = 0; q != none && tries < max_cands; ++tries) {
            const std::size_t m = matchLen(t, q);
            if (m > best_m) {
                best_m = m;
                best_q = q;
                if (m >= max_match) break;
            }
            q = prev_l[q];
        }
    }
    if (best_m < long_len) {
        auto it = map_s.find(keyS(t, n - 1));
        u32 q = it == map_s.end() ? none : it->second;
        for (u32 tries = 0; q != none && tries < max_cands; ++tries) {
            const std::size_t m = matchLen(t, q);
            if (m > best_m) {
                best_m = m;
                best_q = q;
            }
            q = prev_s[q];
        }
    }
    if (best_m < std::max<u32>(min_match, 3)) return {};
    std::size_t kk = 0;
    while (kk < out.size() && best_q + 1 + kk < n) {
        // vision: never draft an image placeholder (ids >= image_id_base)
        if (toks[best_q + 1 + kk] >= image_id_base) break;
        out[kk] = gen_cr ? toks[best_q + 1 + kk] : t[best_q + 1 + kk];
        kk += 1;
    }
    cur_q = best_q;
    cur_n = n;
    return {kk, static_cast<u32>(best_m)};
}

void NgramPolicy::reset() {
    ng.reset();
    acc = priorAcc();
    timing = priorTiming();
    n_pend = 0;
}

// Score the pending proposals whose outcome toks now decides.
void NgramPolicy::observe(std::span<const u32> toks) {
    u32 i = 0;
    while (i < n_pend) {
        const Pending& p = pend[i];
        u32 a = 0;
        while (a < p.n && p.at + a < toks.size() && toks[p.at + a] == p.toks[a]) a += 1;
        if (a == p.n || p.at + a < toks.size()) {
            acc[p.b].updateRate(p.n, a, acc_rate);
            n_pend -= 1;
            pend[i] = pend[n_pend];
        } else {
            i += 1;
        }
    }
}

Ngram::Match NgramPolicy::propose(std::span<const u32> toks, u32 min_match, std::span<u32> out) {
    observe(toks);
    const Ngram::Match m = ng.lookup(toks, min_match, out);
    if (m.n == 0) return m;
    if (n_pend == pend.size()) {
        // oldest first out
        for (std::size_t j = 1; j < pend.size(); ++j) pend[j - 1] = pend[j];
        n_pend -= 1;
    }
    Pending& p = pend[n_pend];
    p.at = toks.size();
    p.n = static_cast<u32>(m.n);
    p.b = bucket(m.mlen);
    std::copy(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(m.n), p.toks.begin());
    n_pend += 1;
    return m;
}

// Drafts to take from an n-gram proposal of n tokens with matched length mlen
// (0: draft with the MTP head instead, whose best choice scores mtp_score).
u32 NgramPolicy::choose(std::size_t n, u32 mlen, float mtp_score, const DraftTiming& mtp_timing) const {
    const DraftAccept& a = acc[bucket(mlen)];
    float best = mtp_score;
    u32 best_k = 0;
    // before any n-gram cycle is timed: the MTP cycle with one draft plus verify rows
    const auto t1 = mtp_timing.estimate(1);
    for (u32 kk = 1; kk <= n; ++kk) {
        float t;
        if (auto e = timing.estimate(kk)) {
            t = *e;
        } else if (t1) {
            t = *t1 * (1 + verify_slope * static_cast<float>(kk - 1));
        } else {
            return 0;
        }
        const float s = a.expected(kk) / t;
        if (s > best) {
            best = s;
            best_k = kk;
        }
    }
    return best_k;
}

}  // namespace whirl::qwen35
