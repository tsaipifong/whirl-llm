// feat026-moep: CPU tests of the multi-fragment MoE prefill expert GEMM schedule (no GPU). They replay, with the
// index maps the kernels use (kernels/moep_map.h) and the variant tables (whirl/kernels_abi.h kMoepGu / kMoepDn):
//   * moe_route / moe_tiles tile lists for token tile bn (expert counts -> (expert, first pos, count));
//   * every 1-D grid block -> (tile, row block), every wave -> (row fragment, interleaved token fragments), every
//     lane / accumulator element -> stored (pos, row): each routed (pos, row) is stored exactly once, nothing else;
//   * the fused combine's last-arriver counters under a random block order: each (token, row block) is finalized
//     exactly once, only after all K of the token's expert rows of that row block were stored, counters end at 0;
//   * the LDS / register-tile constraints of each variant (the kernels' static_asserts, the 64 KiB LDS of a gfx11 WG).
// Routings: skewed (one hot expert), Zipf(1.0) and uniform; 17 / 200 / 2048 tokens; Ornith E 2048, F 512, R 256, K 8.
// v0251-a7port: moved from the source's standalone program (wt_026_moep_p25 tests/kernels/unit/moep_sched_tests.cpp) into
// the module kernels.moe unit file (built alone as whirl-unit-kernels-moe-tests and together in whirl-tests); checks
// unchanged, counted by tests/test_check.h.
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/kernels_abi.h"
#include "whirl/moe_skip.h"  // v0252-f5 moeskip host reference

#include "../../kernels/moep_map.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using whirl::test::check;

struct Tile {
    int e, pos, cnt;
};
// moe_tiles (kernels/gfx1151/moe.hip:483-498): experts in order, positions contiguous per expert
std::vector<Tile> tilesFor(const std::vector<int>& ids, int R, int bn) {
    std::vector<int> cnt(static_cast<std::size_t>(R), 0);
    for (const int e : ids) ++cnt[static_cast<std::size_t>(e)];
    std::vector<Tile> t;
    int o = 0;
    for (int r = 0; r < R; ++r) {
        for (int s = 0; s < cnt[static_cast<std::size_t>(r)]; s += bn) t.push_back({r, o + s, std::min(bn, cnt[static_cast<std::size_t>(r)] - s)});
        o += cnt[static_cast<std::size_t>(r)];
    }
    return t;
}
// moe_route's perm (any order within an expert is valid; a stable order here)
std::vector<int> permFor(const std::vector<int>& ids, int R) {
    std::vector<int> p(ids.size());
    std::iota(p.begin(), p.end(), 0);
    std::stable_sort(p.begin(), p.end(), [&](int a, int b) { return ids[static_cast<std::size_t>(a)] < ids[static_cast<std::size_t>(b)]; });
    (void)R;
    return p;
}
int crow(int lane, int v) { return 2 * v + lane / 16; }  // CROW, kernels/gfx1151/gemm_prefill.hip:109

std::vector<int> routing(std::mt19937_64& rng, int n, int K, int R, int kind) {
    std::vector<int> ids(static_cast<std::size_t>(n) * K);
    std::vector<double> wgt(static_cast<std::size_t>(R), 1.0);
    if (kind == 1) {
        std::vector<int> rank(static_cast<std::size_t>(R));
        std::iota(rank.begin(), rank.end(), 0);
        std::shuffle(rank.begin(), rank.end(), rng);
        for (int e = 0; e < R; ++e) wgt[static_cast<std::size_t>(e)] = 1.0 / (rank[static_cast<std::size_t>(e)] + 1.0);
    }
    std::discrete_distribution<int> d(wgt.begin(), wgt.end());
    for (int t = 0; t < n; ++t)
        for (int k = 0; k < K; ++k) {
            // skewed: every 7th pair goes to the hot expert 3; duplicates within a token are re-drawn (top-k is distinct)
            int e = kind == 0 && (t * K + k) % 7 == 0 ? 3 : d(rng);
            auto dup = [&](int x) {
                for (int j = 0; j < k; ++j)
                    if (ids[static_cast<std::size_t>(t) * K + j] == x) return true;
                return false;
            };
            while (dup(e)) e = d(rng);
            ids[static_cast<std::size_t>(t) * K + k] = e;
        }
    return ids;
}

// stores of one grouped GEMM launch: rows per tile = rows_e, block layout (BM rows, WM x WN waves, TM row fragments
// per wave, FPW token fragments per wave), gu = gate + up pairing (only the gate row index is stored, BM = GR)
void coverage(const std::string& name, const std::vector<Tile>& tl, int pairs, int R, int rows_e, int BM, int WM, int TM, int BN, bool gu) {
    const int WN = 8 / WM, NF = BN / 16, FPW = NF / WN;
    const int nrb = moep::nrb(rows_e, BM);
    const int grid = moep::maxTiles(pairs, BN, R) * nrb;
    std::vector<unsigned char> cov(static_cast<std::size_t>(pairs) * rows_e, 0);
    bool extra = false;
    for (int bx = 0; bx < grid; ++bx) {
        const int ti = moep::blockTile(bx, nrb), rb = moep::blockRb(bx, nrb);
        if (ti >= static_cast<int>(tl.size())) continue;
        const Tile& t = tl[static_cast<std::size_t>(ti)];
        const int live = moep::liveFrags(t.cnt), m0 = rb * BM;
        for (int wid = 0; wid < 8; ++wid) {
            const int wm = wid % WM, wn = wid / WM;
            if (wn >= live) continue;
            for (int f = 0; f < FPW; ++f) {
                const int j = moep::frag(wn, WN, f);
                if (j >= live) continue;
                for (int lane = 0; lane < 32; ++lane) {
                    const int tok = j * 16 + lane % 16;
                    if (tok >= t.cnt) continue;
                    for (int i = 0; i < TM; ++i)
                        for (int v = 0; v < 8; ++v) {
                            const int row = m0 + (gu ? wm * 16 : (wm * TM + i) * 16) + crow(lane, v);
                            if (row >= rows_e) continue;
                            if (gu && i > 0) continue;
                            const std::size_t at = static_cast<std::size_t>(t.pos + tok) * rows_e + row;
                            if (cov[at] == 255) extra = true;
                            ++cov[at];
                        }
                }
            }
        }
    }
    const bool once = std::all_of(cov.begin(), cov.end(), [](unsigned char c) { return c == 1; });
    check(once && !extra, name + ": every routed (pos, row) stored exactly once");
}

void testMoepSchedule() {
    namespace kn = whirl::kernels;
    const int E = 2048, F = 512, R = 256, K = 8;
    std::mt19937_64 rng(20261026);
    // variant constraints (the kernels' static_asserts; LDS of one workgroup <= 64 KiB)
    for (const auto& v : kn::kMoepGu) {
        const int WM = v.gr / 16, WN = 8 / WM, NF = v.bn / 16, BM = 2 * v.gr, GPR = v.bk / 8;
        check(v.gr % 16 == 0 && 8 % WM == 0 && NF % WN == 0 && NF / WN >= 1 && BM * GPR % 256 == 0 && v.bn * GPR % 256 == 0 && v.bk % 16 == 0,
              std::string("gu variant layout ") + v.name);
        check((BM + v.bn) * (v.bk + 8) * 2 <= 65536, std::string("gu variant LDS ") + v.name);
        check(v.bn >= 64, std::string("gu variant bn >= 64 (64-token list buffer) ") + v.name);
    }
    for (const auto& v : kn::kMoepDn) {
        const int WN = 8 / v.wm, TM = v.bm / 16 / v.wm, NF = v.bn / 16, GPR = v.bk / 8;
        check(8 % v.wm == 0 && v.bm % (16 * v.wm) == 0 && TM >= 1 && NF % WN == 0 && NF / WN >= 1 && v.bm * GPR % 256 == 0 && v.bn * GPR % 256 == 0,
              std::string("dn variant layout ") + v.name);
        check((v.bm + v.bn) * (v.bk + 8) * 2 + v.bn * 4 + 4 <= 65536, std::string("dn variant LDS (incl. fused list) ") + v.name);
        check(v.bm >= kn::kMoepMinBm && v.bn >= 32, std::string("dn variant bm / bn bounds ") + v.name);
    }
    const char* kinds[] = {"skewed", "zipf", "uniform"};
    for (const int n : {17, 200, 2048})
        for (int kind = 0; kind < 3; ++kind) {
            const std::vector<int> ids = routing(rng, n, K, R, kind);
            const int pairs = n * K;
            const std::string tag = std::string(" [") + kinds[kind] + " n=" + std::to_string(n) + "]";
            for (const auto& v : kn::kMoepGu) {
                const auto tl = tilesFor(ids, R, v.bn);
                check(static_cast<int>(tl.size()) <= moep::maxTiles(pairs, v.bn, R), std::string("gu tiles <= maxTiles ") + v.name + tag);
                coverage(std::string("gu ") + v.name + tag, tl, pairs, R, F, v.gr, v.gr / 16, 1, v.bn, true);
            }
            const std::vector<int> perm = permFor(ids, R);
            std::vector<int> inv(perm.size());
            for (std::size_t q = 0; q < perm.size(); ++q) inv[static_cast<std::size_t>(perm[q])] = static_cast<int>(q);
            for (const auto& v : kn::kMoepDn) {
                const auto tl = tilesFor(ids, R, v.bn);
                coverage(std::string("dn ") + v.name + tag, tl, pairs, R, E, v.bm, v.wm, v.bm / 16 / v.wm, v.bn, false);
                // fused combine counters under a random block order
                const int nrb = moep::nrb(E, v.bm);
                std::vector<int> order(static_cast<std::size_t>(moep::maxTiles(pairs, v.bn, R) * nrb));
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), rng);
                std::vector<int> cnt(static_cast<std::size_t>(n) * nrb, 0), fin(static_cast<std::size_t>(n) * nrb, 0);
                std::vector<unsigned char> written(static_cast<std::size_t>(pairs) * nrb, 0);
                bool early = false;
                for (const int bx : order) {
                    const int ti = moep::blockTile(bx, nrb), rb = moep::blockRb(bx, nrb);
                    if (ti >= static_cast<int>(tl.size())) continue;
                    const Tile& t = tl[static_cast<std::size_t>(ti)];
                    for (int q = 0; q < t.cnt; ++q) written[static_cast<std::size_t>(t.pos + q) * nrb + rb] = 1;  // Y stores, then fence
                    for (int q = 0; q < t.cnt; ++q) {
                        const int tok = perm[static_cast<std::size_t>(t.pos + q)] / K;
                        int& c = cnt[static_cast<std::size_t>(moep::cntSlot(tok, rb, nrb))];
                        if (c++ == K - 1) {
                            c = 0;
                            ++fin[static_cast<std::size_t>(moep::cntSlot(tok, rb, nrb))];
                            for (int k = 0; k < K; ++k)
                                early = early || !written[static_cast<std::size_t>(inv[static_cast<std::size_t>(tok) * K + k]) * nrb + rb];
                        }
                    }
                }
                check(std::all_of(fin.begin(), fin.end(), [](int f) { return f == 1; }), std::string("fused combine: each (token, row block) once ") + v.name + tag);
                check(!early, std::string("fused combine: only after all K expert rows ") + v.name + tag);
                check(std::all_of(cnt.begin(), cnt.end(), [](int c) { return c == 0; }), std::string("fused combine: counters end at 0 ") + v.name + tag);
            }
        }
}


// v0252-f5 (F5 moeskip, from feat027-moeskip-r1 4b6474b testMoeSkip parts 2-3): the host reference of the moeskip entry
// points (include/whirl/moe_skip.h) - the rule every kernel and the GPU kernel tests (family moeskip) are held to:
//   * applyRow on random sorted router rows (2000 rows, K 8, thr 0.06, min_keep 1..3): the kept slots are a prefix, the
//     dropped ones id -1 / weight 0, the kept weights sum to 1 and keep their ratios, the first dropped slot was below thr and
//     every kept slot past min_keep at or above it, at least min_keep kept; nothing dropped -> ids / weights bitwise unchanged;
//     thr 0 never drops; thr 1 = static top-min_keep; min_keep clamped to [1, K]; the test distribution drops some, not all;
//   * keptPrefix;
//   * route with dropped pairs (37 tokens, K 4, R 16, bn 8): tiles cover exactly the kept positions, grouped by expert, the
//     dropped pairs parked after n_kept with inv -1 (perm there = a valid dropped pair); no -1 = moe_route's grouping;
//   * combine (k ascending, inv -1 skipped) == the direct sum over the kept experts, bitwise (NaN tail never read);
//   * mutant control: the rule without the renormalization fails the sum check.
void testMoeSkipRef() {
    namespace ms = whirl::moeskip;
    std::uint32_t seed = 12345u;
    auto rnd = [&seed] {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>((seed >> 8) & 0xFFFF) / 65536.0f;
    };
    const int K = 8;
    int total = 0, dropped = 0;
    bool ok_prefix = true, ok_sum = true, ok_ratio = true, ok_thr = true, ok_min = true, ok_same = true, ok_ids = true, mutant_caught = false;
    for (int it = 0; it < 2000; ++it) {
        float v[K];
        v[0] = 4.f * rnd();
        for (int k = 1; k < K; ++k) v[k] = v[k - 1] - 1.5f * rnd();  // sorted descending (moe_topk order)
        float w0[K];
        int ids0[K];
        {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += std::exp(v[k] - v[0]);
            for (int k = 0; k < K; ++k) {
                w0[k] = std::exp(v[k] - v[0]) / s;  // moe_topk's weights
                ids0[k] = 100 + k;
            }
        }
        const ms::Params p{0.06f, 1 + it % 3};
        float w[K];
        int ids[K];
        std::copy(w0, w0 + K, w);
        std::copy(ids0, ids0 + K, ids);
        const int keep = ms::applyRow(ids, w, K, p);
        total += K;
        dropped += K - keep;
        double s = 0;
        for (int k = 0; k < K; ++k) {
            if (k < keep && (w[k] <= 0.f || ids[k] != ids0[k])) ok_prefix = false;
            if (k >= keep && (w[k] != 0.f || ids[k] != -1)) ok_prefix = false;
            s += w[k];
        }
        if (std::fabs(s - 1.0) > 1e-5) ok_sum = false;
        for (int k = 1; k < keep; ++k)
            if (std::fabs(w[k] / w[0] - w0[k] / w0[0]) > 1e-4 * (w0[k] / w0[0]) + 1e-7) ok_ratio = false;
        if (keep < K && w0[keep] >= p.thr) ok_thr = false;
        for (int k = p.min_keep; k < keep; ++k)
            if (w0[k] < p.thr) ok_thr = false;
        if (keep < p.min_keep) ok_min = false;
        if (keep == K && (std::memcmp(w, w0, sizeof w) != 0 || std::memcmp(ids, ids0, sizeof ids) != 0)) ok_same = false;
        if (ms::keptPrefix(ids, K) != keep) ok_ids = false;
        if (keep < K) {
            // mutant: drop without renormalizing (the kernel mutant of the CPU chain) -> the weights no longer sum to 1
            double sm = 0;
            for (int k = 0; k < keep; ++k) sm += w0[k];
            if (std::fabs(sm - 1.0) > 1e-5) mutant_caught = true;
        }
    }
    check(ok_prefix, "moeskip ref: kept slots a prefix (ids kept), dropped id -1 / weight 0");
    check(ok_sum, "moeskip ref: kept weights renormalized (sum 1)");
    check(ok_ratio, "moeskip ref: kept weight ratios preserved");
    check(ok_thr, "moeskip ref: first dropped slot below thr, every kept slot past min_keep at or above it");
    check(ok_min, "moeskip ref: at least min_keep kept");
    check(ok_same, "moeskip ref: nothing dropped -> ids / weights bitwise unchanged (moe_topk's)");
    check(ok_ids, "moeskip ref: keptPrefix == the kept count");
    check(dropped > 0 && dropped < total, "moeskip ref: the test distribution drops some slots, not all");
    check(mutant_caught, "moeskip ref: mutant (no renormalization) is caught by the sum check");
    {
        float w[K] = {0.3f, 0.2f, 0.15f, 0.1f, 0.1f, 0.07f, 0.05f, 0.03f}, x[K];
        int ids[K], y[K];
        for (int k = 0; k < K; ++k) ids[k] = k;
        auto run = [&](float thr, int mk) {
            std::copy(w, w + K, x);
            std::copy(ids, ids + K, y);
            return ms::applyRow(y, x, K, ms::Params{thr, mk});
        };
        check(run(0.f, 1) == K && std::memcmp(x, w, sizeof w) == 0, "moeskip ref: thr 0 drops nothing");
        check(run(1.f, 1) == 1 && x[0] == 1.f && y[1] == -1, "moeskip ref: thr 1 keeps exactly min_keep (1)");
        check(run(1.f, 6) == 6 && y[6] == -1 && y[5] == 5, "moeskip ref: thr 1 min 6 = static top 6");
        check(run(0.5f, 0) == 1, "moeskip ref: min_keep clamped to 1");
        check(run(0.5f, 99) == K, "moeskip ref: min_keep clamped to K");
        check(run(0.06f, 1) == 6 && y[6] == -1 && y[7] == -1, "moeskip ref: thr 0.06 drops the 0.05 / 0.03 slots");
        const int ids_a[8] = {5, 9, 2, 7, -1, -1, -1, -1}, ids_b[8] = {5, 9, 2, 7, 1, 3, 4, 6}, ids_c[8] = {5, -1, -1, -1, -1, -1, -1, -1};
        check(ms::keptPrefix(ids_a, 8) == 4 && ms::keptPrefix(ids_b, 8) == 8 && ms::keptPrefix(ids_c, 8) == 1, "moeskip ref: keptPrefix cases");
    }
    // prefill grouping + combine with dropped pairs
    {
        const int R = 16, Kp = 4, n = 37, C = 5, bn = 8;
        std::uint32_t sd = 777u;
        auto nxt = [&sd] {
            sd = sd * 1664525u + 1013904223u;
            return sd >> 8;
        };
        std::vector<int> ids(static_cast<std::size_t>(n) * Kp), ids_full;
        std::vector<float> w(ids.size());
        for (int t = 0; t < n; ++t) {
            std::vector<int> pick;
            while (static_cast<int>(pick.size()) < Kp) {
                const int e = static_cast<int>(nxt() % R);
                if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
            }
            const int keep = 1 + static_cast<int>(nxt() % Kp);
            float s = 0;
            for (int k = 0; k < Kp; ++k) {
                ids[static_cast<std::size_t>(t) * Kp + k] = k < keep ? pick[static_cast<std::size_t>(k)] : -1;
                w[static_cast<std::size_t>(t) * Kp + k] = k < keep ? 1.f + static_cast<float>(nxt() % 100) / 100.f : 0.f;
                s += w[static_cast<std::size_t>(t) * Kp + k];
            }
            for (int k = 0; k < Kp; ++k) w[static_cast<std::size_t>(t) * Kp + k] /= s;
        }
        ids_full = ids;
        for (int& e : ids_full)
            if (e < 0) e = 0;
        const ms::Route rf = ms::route(ids_full, R, bn);
        bool full_ok = rf.n_kept == n * Kp;
        for (int p = 0; p < n * Kp; ++p)
            if (rf.inv[static_cast<std::size_t>(p)] < 0 || rf.perm[static_cast<std::size_t>(rf.inv[static_cast<std::size_t>(p)])] != p) full_ok = false;
        check(full_ok, "moeskip ref: route without -1 = moe_route's grouping (every pair kept, inv = perm^-1)");
        const ms::Route r = ms::route(ids, R, bn);
        int kept = 0;
        for (int e : ids) kept += e >= 0;
        check(r.n_kept == kept, "moeskip ref: route n_kept = the kept pairs");
        bool ok = true;
        std::vector<int> covered(static_cast<std::size_t>(n) * Kp, 0);
        for (const auto& tl : r.tiles) {
            if (tl.count < 1 || tl.count > bn || tl.first < 0 || tl.first + tl.count > r.n_kept) ok = false;
            for (int q = tl.first; q < tl.first + tl.count && q < r.n_kept; ++q) {
                const int p = r.perm[static_cast<std::size_t>(q)];
                if (p < 0 || ids[static_cast<std::size_t>(p)] != tl.expert) ok = false;
                ++covered[static_cast<std::size_t>(q)];
            }
        }
        for (int q = 0; q < n * Kp; ++q) {
            const int p = r.perm[static_cast<std::size_t>(q)];
            if (p < 0 || p >= n * Kp) ok = false;
            if (q < r.n_kept && covered[static_cast<std::size_t>(q)] != 1) ok = false;
            if (q >= r.n_kept && (covered[static_cast<std::size_t>(q)] != 0 || ids[static_cast<std::size_t>(p)] >= 0)) ok = false;
        }
        for (int p = 0; p < n * Kp; ++p) {
            const int q = r.inv[static_cast<std::size_t>(p)];
            if (ids[static_cast<std::size_t>(p)] < 0 ? q != -1 : (q < 0 || q >= r.n_kept || r.perm[static_cast<std::size_t>(q)] != p)) ok = false;
        }
        check(ok, "moeskip ref: route with dropped pairs (tiles = kept positions by expert, dropped parked after n_kept, inv -1)");
        std::vector<float> yexp(static_cast<std::size_t>(R) * n * C);
        for (float& y : yexp) y = static_cast<float>(static_cast<int>(nxt() % 2001) - 1000) / 250.f;
        auto yOf = [&](int e, int t, int c) { return yexp[(static_cast<std::size_t>(e) * n + t) * C + c]; };
        std::vector<float> yd(static_cast<std::size_t>(n) * Kp * C, std::nanf(""));  // tail positions: NaN, never read
        for (int q = 0; q < r.n_kept; ++q) {
            const int p = r.perm[static_cast<std::size_t>(q)];
            for (int c = 0; c < C; ++c) yd[static_cast<std::size_t>(q) * C + c] = yOf(ids[static_cast<std::size_t>(p)], p / Kp, c);
        }
        bool cmb = true;
        for (int t = 0; t < n; ++t)
            for (int c = 0; c < C; ++c) {
                const float got = ms::combine(0.5f, r.inv.data() + static_cast<std::size_t>(t) * Kp, w.data() + static_cast<std::size_t>(t) * Kp, Kp,
                                              [&](int q) { return yd[static_cast<std::size_t>(q) * C + c]; }, true, 0.25f, 2.f);
                float acc = 0.f;
                for (int k = 0; k < Kp; ++k) {
                    const int e = ids[static_cast<std::size_t>(t) * Kp + k];
                    if (e >= 0) acc += w[static_cast<std::size_t>(t) * Kp + k] * yOf(e, t, c);
                }
                acc += 0.25f * 2.f;
                if (!(got == 0.5f + acc)) cmb = false;
            }
        check(cmb, "moeskip ref: combine (inv -1 skipped, NaN tail never read) == the direct sum over the kept experts (bitwise)");
    }
}

}  // namespace

// tests/unit/unit_tests.cpp ORDER lists all modules
const whirl::test::Suite& unitSuite_kernels_moe() {
    static const whirl::test::Suite s{"kernels.moe",
                                      {
                                          {"testMoepSchedule", [] { testMoepSchedule(); }},
                                          {"testMoeSkipRef", [] { testMoeSkipRef(); }},  // v0252-f5
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_kernels_moe()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
