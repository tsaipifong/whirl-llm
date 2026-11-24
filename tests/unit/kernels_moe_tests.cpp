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
// v0252-8smoe2 (C): the coalesced f16 epilogue of gemm_moepdhs_* (kernels/gfx1151/moe_prefill_mt.hip moep_store_h16_staged,
// maps kernels/moep_map.h dhs*). CPU replay with the kernel's own maps, every kMoepDn variant, Ornith shapes and the three
// routings: (1) the staging slice holds element (token lane % 16, row i * 16 + CROW(lane, v)) of every lane / accumulator
// exactly once and fits in the A tile (all 8 waves); (2) the read chunks cover the slice exactly once, 8 rows each,
// 16-byte aligned; (3) through the whole grid every routed (pos, row) of moe_yd is stored exactly once, by the lane whose
// accumulator holds it (the value gemm_moepdh stores there), including rows_e % 8 != 0 (scalar tail). Fails to build before
// the change (no moep::dhs*).
void testMoepDhsMap() {
    namespace kn = whirl::kernels;
    for (const kn::MoepDnVariant& v : kn::kMoepDn) {
        const int WM = v.wm, WN = 8 / WM, TM = v.bm / 16 / WM, SLD = moep::dhsLd(TM), LD = v.bk + 8;
        const std::string tag = std::string(" ") + v.name;
        check(SLD % 8 == 0 && 8 * 16 * SLD <= v.bm * LD, "dhs slice: 16-byte rows, 8 slices fit in As" + tag);
        // (1) + (2): one wave's slice
        std::vector<int> slot(static_cast<std::size_t>(16 * SLD), -1);
        bool w_ok = true;
        for (int lane = 0; lane < 32; ++lane)
            for (int i = 0; i < TM; ++i)
                for (int vv = 0; vv < 8; ++vv) {
                    const int o = moep::dhsWrite(lane, i, vv, SLD);
                    const int tok = lane % 16, row = i * 16 + crow(lane, vv);
                    if (o < 0 || o >= 16 * SLD || slot[static_cast<std::size_t>(o)] != -1 || o != tok * SLD + row) w_ok = false;
                    else slot[static_cast<std::size_t>(o)] = tok * 1024 + row;
                }
        check(w_ok, "dhs write map: every accumulator element once at (token, row)" + tag);
        std::vector<int> seen(static_cast<std::size_t>(16 * TM * 16), 0);
        bool r_ok = moep::dhsChunks(TM) == 16 * TM * 16 / 8;
        for (int c = 0; c < moep::dhsChunks(TM); ++c) {
            int tk = -1, r8 = -1;
            moep::dhsRead(c, TM, &tk, &r8);
            if (tk < 0 || tk >= 16 || r8 < 0 || r8 + 8 > TM * 16 || r8 % 8 != 0 || (tk * SLD + r8) % 8 != 0) r_ok = false;
            else
                for (int q = 0; q < 8; ++q) {
                    ++seen[static_cast<std::size_t>(tk * TM * 16 + r8 + q)];
                    if (slot[static_cast<std::size_t>(tk * SLD + r8 + q)] != tk * 1024 + r8 + q) r_ok = false;
                }
        }
        r_ok = r_ok && std::all_of(seen.begin(), seen.end(), [](int x) { return x == 1; });
        check(r_ok, "dhs read map: every staged element read once, as its own (token, row)" + tag);
        // (3): the whole grid (rows_e 2048 = Ornith E, and 2040: the scalar tail)
        for (const int rows_e : {2048, 2040})
            for (const int kind : {0, 1, 2}) {
                std::mt19937_64 rng(7u + static_cast<unsigned>(kind));
                const int n = 2048, K = 8, R = 256, pairs = n * K;
                const auto tl = tilesFor(routing(rng, n, K, R, kind), R, v.bn);
                const int nrb = moep::nrb(rows_e, v.bm), grid = moep::maxTiles(pairs, v.bn, R) * nrb;
                std::vector<unsigned char> cov(static_cast<std::size_t>(pairs) * rows_e, 0);
                for (int bx = 0; bx < grid; ++bx) {
                    const int ti = moep::blockTile(bx, nrb), rb = moep::blockRb(bx, nrb);
                    if (ti >= static_cast<int>(tl.size())) continue;
                    const Tile& t = tl[static_cast<std::size_t>(ti)];
                    const int live = moep::liveFrags(t.cnt), m0 = rb * v.bm;
                    for (int wid = 0; wid < 8; ++wid) {
                        const int wm = wid % WM, wn = wid / WM;
                        for (int f = 0; f < v.bn / 16 / WN; ++f) {
                            const int j = moep::frag(wn, WN, f);
                            if (j >= live) continue;
                            for (int c = 0; c < moep::dhsChunks(TM); ++c) {  // lanes c, c + 32, ...: each chunk once
                                int tk, r8;
                                moep::dhsRead(c, TM, &tk, &r8);
                                const int tt = j * 16 + tk, row0 = m0 + wm * TM * 16 + r8;
                                if (tt >= t.cnt || row0 >= rows_e) continue;
                                for (int q = 0; q < 8 && row0 + q < rows_e; ++q) ++cov[static_cast<std::size_t>(t.pos + tt) * rows_e + row0 + q];
                            }
                        }
                    }
                }
                check(std::all_of(cov.begin(), cov.end(), [](unsigned char x) { return x == 1; }),
                      "dhs grid: every routed (pos, row) stored exactly once, rows_e " + std::to_string(rows_e) + " routing " + std::to_string(kind) + tag);
            }
    }
}

// v0252-8smoe2 (E): the fused-dequant A fetch of the gemm_moepfq* twins (kernels/gfx1151/moe_prefill_fq.hip MoepFqFetch, maps
// moep::fqRow / fqSb / fqIb). For each twin's (A rows, BK): the lane slots cover every (row, sub-block) of the chunk once, and
// the LDS column of each decoded half8 (sb * 32 + j * 8) is the column dec_group_h<QT_MXFP4> fills with that group in the base
// kernel (group g = (k0 + col) / 8 reads sub-block ((g & 31) * 8) >> 5 of the super-block, value offset (g * 8) & 31 =
// j * 8; j order = low nibbles of bytes 0-7, of 8-15, high of 0-7, high of 8-15), for every chunk k0 of K = 512 / 2048.
// With mx_inject's decode (testMxInject: bitwise == the LUT math) the A tile is the base kernel's tile, hence its bits.
// Fails to build before the change (no moep::fq*).
void testMoepFqMap() {
    struct Shape {
        const char* name;
        int bma, bk;
    };
    for (const Shape s : {Shape{"gu g64n64k64 / g64n128k64", 128, 64}, Shape{"gu g128n128k32", 256, 32}, Shape{"dn m128 k64", 128, 64}})
        for (const int ncols : {512, 2048}) {
            const int SBR = s.bk / 32;
            bool ok = s.bma * SBR % 256 == 0;
            std::vector<int> cov(static_cast<std::size_t>(s.bma * s.bk), 0);
            for (int sl = 0; sl < s.bma * SBR; ++sl) {
                const int r = moep::fqRow(sl, s.bk), sb = moep::fqSb(sl, s.bk);
                if (r < 0 || r >= s.bma || sb < 0 || sb >= SBR) {
                    ok = false;
                    continue;
                }
                for (int j = 0; j < 4; ++j)
                    for (int q = 0; q < 8; ++q) ++cov[static_cast<std::size_t>(r * s.bk + sb * 32 + j * 8 + q)];
                for (int k0 = 0; k0 < ncols; k0 += s.bk) {
                    const int ib = moep::fqIb(k0, sb);
                    for (int j = 0; j < 4; ++j) {
                        const int col = sb * 32 + j * 8, g = (k0 + col) / 8;  // dec_group_h's group of this LDS column
                        const int p = (g & 31) * 8;
                        if ((g >> 5) != (k0 >> 8) || (p >> 5) != ib || (p & 31) != j * 8) ok = false;
                    }
                }
            }
            ok = ok && std::all_of(cov.begin(), cov.end(), [](int x) { return x == 1; });
            check(ok, std::string("fq fetch map: every A element once, at dec_group_h's column, ") + s.name + " ncols " + std::to_string(ncols));
        }
}

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

// v0252-8sdet (PLAN V1-140, COMB 6): the order-independent fixed-point MoE combine of gemm_moepdi_* (kernels/moep_map.h
// fx*; kernels/gfx1151/moe_prefill_mt.hip EPI 6). CPU replay with the kernel's own helpers:
//   (1) fxLimit(K): integral, K * lim <= INT32_MAX (no overflow of the final i32 sum for any input), tight (the next
//       float up breaks it), for K 1..32;
//   (2) fxQuant: round to nearest even on the 2^-kFxShift grid (|q * 2^-S - a| <= 2^-(S+1) inside the range), symmetric
//       clamp to +-lim (+-inf too), NaN -> 0;
//   (3) the whole down grid (bn 128 tiles of the KEPT pairs, moeskip-style dropped slots with inv -1, Ornith K 8, 3
//       routings): i32 adds mod 2^32, last-arriver counters with the kept-slot count (fxKept), finalize x += fxValue(sum)
//       + sg * ysh by the last block; under 4 random block orders AND a different position order inside each expert,
//       x is bitwise identical; every (token, row block) finalized once, after all its kept addends; counters and sums end
//       at 0; |fxValue(sum) - sum of the addends| <= kept * 2^-(S+1) + |value| * 2^-24 (clamp-free addends);
//   (4) mutant control: the same addends summed in f32 in those orders (COMB 5's arithmetic) differ somewhere.
// Fails to build before the change (no moep::fx*).
void testMoepFixSum() {
    constexpr double kI32 = 2147483647.0;
    const float qstep = 1.f / moep::fxScale();
    check(moep::fxScale() == static_cast<float>(1 << moep::kFxShift) && qstep * moep::fxScale() == 1.f, "fx: scale 2^kFxShift");
    // (1)
    bool lim_ok = true;
    for (int K = 1; K <= 32; ++K) {
        const float lim = moep::fxLimit(K);
        if (!(lim > 0.f) || lim != std::floor(lim) || static_cast<double>(lim) * K > kI32 ||
            static_cast<double>(std::nextafter(lim, 1e30f)) * K <= kI32)
            lim_ok = false;
    }
    check(lim_ok, "fx limit: integral, K * lim <= INT32_MAX and tight, K 1..32");
    // (2)
    const float lim8 = moep::fxLimit(8);
    bool q_ok = moep::fxQuant(0.f, lim8) == 0 && moep::fxQuant(-0.f, lim8) == 0;
    q_ok = q_ok && moep::fxQuant(0.5f * qstep, lim8) == 0 && moep::fxQuant(1.5f * qstep, lim8) == 2 && moep::fxQuant(2.5f * qstep, lim8) == 2 &&
           moep::fxQuant(-0.5f * qstep, lim8) == 0 && moep::fxQuant(-1.5f * qstep, lim8) == -2 && moep::fxQuant(-2.5f * qstep, lim8) == -2 &&
           moep::fxQuant(0.75f * qstep, lim8) == 1 && moep::fxQuant(-0.75f * qstep, lim8) == -1;  // nearest, ties to even
    q_ok = q_ok && moep::fxQuant(1e30f, lim8) == static_cast<int>(lim8) && moep::fxQuant(-1e30f, lim8) == -static_cast<int>(lim8) &&
           moep::fxQuant(INFINITY, lim8) == static_cast<int>(lim8) && moep::fxQuant(-INFINITY, lim8) == -static_cast<int>(lim8) &&
           moep::fxQuant(NAN, lim8) == 0;
    {
        std::mt19937_64 rng(11);
        std::uniform_real_distribution<double> ex(-20.0, 13.9);  // |a| up to ~1.5e4 < lim8 / 2^14 = 16384
        for (int i = 0; i < 200000 && q_ok; ++i) {
            const float a = static_cast<float>((rng() & 1 ? 1.0 : -1.0) * std::exp2(ex(rng)));
            const int q = moep::fxQuant(a, lim8);
            if (std::fabs(static_cast<double>(q) * qstep - a) > 0.5 * qstep) q_ok = false;
        }
    }
    check(q_ok, "fx quant: nearest-even on the 2^-14 grid, error <= half a step, clamp +-lim (inf too), NaN -> 0");
    // (3) + (4)
    const int n = 300, K = 8, R = 64, rows_e = 256, BM = 128, bn = 128, nrb = moep::nrb(rows_e, BM);
    bool det = true, once = true, ends0 = true, err_ok = true, f32_differs = false, kept_ok = true;
    for (const int kind : {0, 1, 2}) {
        std::mt19937_64 rng(23u + static_cast<unsigned>(kind));
        const std::vector<int> ids = routing(rng, n, K, R, kind);
        // moeskip-style drops (fast): slot k >= 1 dropped with probability 1/4; slot 0 always kept (min keep 1)
        std::vector<int> keep(ids.size(), 1);
        for (std::size_t p = 0; p < ids.size(); ++p)
            if (p % K != 0 && rng() % 4 == 0) keep[p] = 0;
        // addends a(pair, row) = w * y: log-uniform magnitudes 2^-20 .. 2^12, random sign
        std::uniform_real_distribution<double> ex(-20.0, 12.0);
        std::vector<float> add(ids.size() * rows_e);
        for (float& a : add) a = static_cast<float>((rng() & 1 ? 1.0 : -1.0) * std::exp2(ex(rng)));
        std::vector<float> x0(static_cast<std::size_t>(n) * rows_e), ysh(x0.size()), sg(static_cast<std::size_t>(n));
        for (float& v : x0) v = static_cast<float>(std::ldexp(static_cast<double>(rng() % 2000000) - 1e6, -12));
        for (float& v : ysh) v = static_cast<float>(std::ldexp(static_cast<double>(rng() % 2000000) - 1e6, -16));
        for (float& v : sg) v = static_cast<float>((rng() % 1000) / 1000.0);
        std::vector<float> first, first32;
        for (int order = 0; order < 4; ++order) {
            // positions: kept pairs grouped by expert, any order inside an expert (order 0 stable, then shuffled); inv -1 for dropped
            std::vector<int> perm, inv(ids.size(), -1);
            std::vector<Tile> tl;
            for (int e = 0; e < R; ++e) {
                std::vector<int> ps;
                for (std::size_t p = 0; p < ids.size(); ++p)
                    if (ids[p] == e && keep[p]) ps.push_back(static_cast<int>(p));
                if (order > 0) std::shuffle(ps.begin(), ps.end(), rng);
                for (std::size_t s = 0; s < ps.size(); s += bn)
                    tl.push_back({e, static_cast<int>(perm.size() + s), static_cast<int>(std::min<std::size_t>(bn, ps.size() - s))});
                for (const int p : ps) {
                    inv[static_cast<std::size_t>(p)] = static_cast<int>(perm.size());
                    perm.push_back(p);
                }
            }
            for (int t = 0; t < n; ++t) {
                int c = 0;
                for (int k = 0; k < K; ++k) c += keep[static_cast<std::size_t>(t) * K + k];
                if (moep::fxKept(inv.data(), t, K) != c) kept_ok = false;
            }
            std::vector<int> blocks(tl.size() * nrb);
            std::iota(blocks.begin(), blocks.end(), 0);
            if (order > 0) std::shuffle(blocks.begin(), blocks.end(), rng);
            std::vector<std::uint32_t> acc(static_cast<std::size_t>(n) * rows_e, 0u);
            std::vector<int> cnt(static_cast<std::size_t>(n) * nrb, 0), fin(cnt.size(), 0), arrived(cnt.size(), 0);
            std::vector<float> x = x0, x32 = x0;
            for (const int bx : blocks) {
                const int ti = moep::blockTile(bx, nrb), rb = moep::blockRb(bx, nrb);
                const Tile& t = tl[static_cast<std::size_t>(ti)];
                const int m0 = rb * BM;
                for (int tt = 0; tt < t.cnt; ++tt) {
                    const int p = perm[static_cast<std::size_t>(t.pos + tt)], tok = p / K;
                    for (int r = m0; r < m0 + BM && r < rows_e; ++r) {
                        const float a = add[static_cast<std::size_t>(p) * rows_e + r];
                        acc[static_cast<std::size_t>(tok) * rows_e + r] += static_cast<std::uint32_t>(moep::fxQuant(a, moep::fxLimit(K)));
                        x32[static_cast<std::size_t>(tok) * rows_e + r] += a;  // COMB 5's arithmetic (mutant control)
                    }
                }
                for (int tt = 0; tt < t.cnt; ++tt) {
                    const int tok = perm[static_cast<std::size_t>(t.pos + tt)] / K;
                    const std::size_t c = static_cast<std::size_t>(moep::cntSlot(tok, rb, nrb));
                    ++arrived[c];
                    if (cnt[c]++ == moep::fxKept(inv.data(), tok, K) - 1) {
                        cnt[c] = 0;
                        ++fin[c];
                        if (arrived[c] != moep::fxKept(inv.data(), tok, K)) once = false;
                        for (int r = m0; r < m0 + BM && r < rows_e; ++r) {
                            const std::size_t o = static_cast<std::size_t>(tok) * rows_e + r;
                            const int s = static_cast<int>(acc[o]);
                            acc[o] = 0u;
                            const float v = moep::fxValue(s);
                            double ref = 0.0;
                            int kept = 0;
                            for (int k = 0; k < K; ++k)
                                if (keep[static_cast<std::size_t>(tok) * K + k]) {
                                    ref += add[(static_cast<std::size_t>(tok) * K + k) * rows_e + r];
                                    ++kept;
                                }
                            if (std::fabs(static_cast<double>(v) - ref) > kept * 0.5 * qstep + std::fabs(ref) * std::ldexp(1.0, -23)) err_ok = false;
                            x[o] += v + sg[static_cast<std::size_t>(tok)] * ysh[o];
                        }
                    }
                }
            }
            for (std::size_t c = 0; c < cnt.size(); ++c)
                if (fin[c] != 1 || cnt[c] != 0) once = false;
            for (const std::uint32_t a : acc)
                if (a != 0u) ends0 = false;
            if (order == 0) {
                first = x;
                first32 = x32;
            } else {
                if (std::memcmp(first.data(), x.data(), x.size() * sizeof(float)) != 0) det = false;
                if (std::memcmp(first32.data(), x32.data(), x32.size() * sizeof(float)) != 0) f32_differs = true;
            }
        }
    }
    check(kept_ok, "fx kept: fxKept(inv) = the kept slots of each token");
    check(once, "fx grid: every (token, row block) finalized exactly once, after all its kept addends; counters end at 0");
    check(ends0, "fx grid: the i32 sums are left zeroed");
    check(err_ok, "fx grid: |fxValue(sum) - sum of the addends| <= kept * 2^-15 + |sum| * 2^-23");
    check(det, "fx grid: x bitwise identical under 4 block orders and position orders (order independence)");
    check(f32_differs, "fx grid mutant control: f32 sums of the same addends in those orders differ (the check can see order dependence)");
}

// v0252-8sdet round 4: the buffer contract behind clearing moe_yd's fixed-point region only when it may be dirty. One MoE
// layer = the whole down grid with the product layout (i32 sums [n][rows_e] at word 0, counters [n][nrb] right after, the
// kernel's own moep::fx* helpers, last arriver swaps the sum to 0 and resets the counter), blocks in a random order,
// moeskip-style dropped slots (inv -1). Checked: (1) after every layer every word of the region is 0 again; (2) layer 2 on
// the SAME buffer without a clear gives x bitwise equal to layer 2 on a freshly cleared buffer, for a second layer with
// fewer rows (the counters move into former sum words) and with the same rows; (3) an "aborted forward" (2 of 3 layers
// run, the rest never launched) followed by a new forward without a clear == with a clear; (4) mutant control: a buffer
// left dirty (one stale counter) gives different x - the check can see a missing clear.
namespace {
struct FxIn {
    int n;
    std::vector<int> ids, keep;
    std::vector<float> add, x0;
};
FxIn fxInput(std::mt19937_64& rng, int n, int K, int R, int rows_e) {
    FxIn in;
    in.n = n;
    in.ids = routing(rng, n, K, R, 1);
    in.keep.assign(in.ids.size(), 1);
    for (std::size_t p = 0; p < in.ids.size(); ++p)
        if (p % K != 0 && rng() % 4 == 0) in.keep[p] = 0;
    std::uniform_real_distribution<double> ex(-16.0, 8.0);
    in.add.resize(in.ids.size() * rows_e);
    for (float& a : in.add) a = static_cast<float>((rng() & 1 ? 1.0 : -1.0) * std::exp2(ex(rng)));
    in.x0.resize(static_cast<std::size_t>(n) * rows_e);
    for (float& v : in.x0) v = static_cast<float>(std::ldexp(static_cast<double>(rng() % 2000000) - 1e6, -12));
    return in;
}
// one layer on buf (word layout as ffn.cpp: sums then counters); returns x
std::vector<float> fxLayer(std::vector<std::uint32_t>& buf, const FxIn& in, int K, int R, int rows_e, int BM, std::mt19937_64& rng) {
    const int n = in.n, nrb = moep::nrb(rows_e, BM), bn = 128;
    std::vector<int> perm, inv(in.ids.size(), -1);
    std::vector<Tile> tl;
    for (int e = 0; e < R; ++e) {
        std::vector<int> ps;
        for (std::size_t p = 0; p < in.ids.size(); ++p)
            if (in.ids[p] == e && in.keep[p]) ps.push_back(static_cast<int>(p));
        for (std::size_t s = 0; s < ps.size(); s += bn)
            tl.push_back({e, static_cast<int>(perm.size() + s), static_cast<int>(std::min<std::size_t>(bn, ps.size() - s))});
        for (const int p : ps) {
            inv[static_cast<std::size_t>(p)] = static_cast<int>(perm.size());
            perm.push_back(p);
        }
    }
    std::vector<int> blocks(tl.size() * nrb);
    std::iota(blocks.begin(), blocks.end(), 0);
    std::shuffle(blocks.begin(), blocks.end(), rng);
    std::uint32_t* sum = buf.data();
    std::uint32_t* cnt = buf.data() + static_cast<std::size_t>(n) * rows_e;
    std::vector<float> x = in.x0;
    const float lim = moep::fxLimit(K);
    for (const int bx : blocks) {
        const int ti = moep::blockTile(bx, nrb), rb = moep::blockRb(bx, nrb), m0 = rb * BM;
        const Tile& t = tl[static_cast<std::size_t>(ti)];
        for (int tt = 0; tt < t.cnt; ++tt) {
            const int p = perm[static_cast<std::size_t>(t.pos + tt)], tok = p / K;
            for (int r = m0; r < m0 + BM && r < rows_e; ++r)
                sum[static_cast<std::size_t>(tok) * rows_e + r] += static_cast<std::uint32_t>(moep::fxQuant(in.add[static_cast<std::size_t>(p) * rows_e + r], lim));
        }
        for (int tt = 0; tt < t.cnt; ++tt) {
            const int tok = perm[static_cast<std::size_t>(t.pos + tt)] / K;
            std::uint32_t& c = cnt[moep::cntSlot(tok, rb, nrb)];
            if (static_cast<int>(c++) == moep::fxKept(inv.data(), tok, K) - 1) {
                c = 0u;
                for (int r = m0; r < m0 + BM && r < rows_e; ++r) {
                    const std::size_t o = static_cast<std::size_t>(tok) * rows_e + r;
                    const int s = static_cast<int>(sum[o]);
                    sum[o] = 0u;
                    x[o] += moep::fxValue(s);
                }
            }
        }
    }
    return x;
}
}  // namespace

void testMoepFixReuse() {
    const int K = 8, R = 64, rows_e = 256, BM = 128, nrb = moep::nrb(rows_e, BM);
    const int nmax = 300;
    const std::size_t words = static_cast<std::size_t>(nmax) * rows_e + static_cast<std::size_t>(nmax) * nrb;
    std::mt19937_64 rng(97);
    const FxIn a = fxInput(rng, nmax, K, R, rows_e), b = fxInput(rng, 200, K, R, rows_e), b2 = fxInput(rng, nmax, K, R, rows_e);
    auto allZero = [](const std::vector<std::uint32_t>& v) { return std::all_of(v.begin(), v.end(), [](std::uint32_t w) { return w == 0u; }); };
    bool zero_after = true, reuse_ok = true, abort_ok = true;
    for (const FxIn* second : {&b, &b2}) {
        std::vector<std::uint32_t> dirty_free(words, 0u);  // one clear at the start, none between the layers
        fxLayer(dirty_free, a, K, R, rows_e, BM, rng);
        zero_after = zero_after && allZero(dirty_free);
        const std::vector<float> x_reused = fxLayer(dirty_free, *second, K, R, rows_e, BM, rng);
        zero_after = zero_after && allZero(dirty_free);
        std::vector<std::uint32_t> fresh(words, 0u);
        const std::vector<float> x_fresh = fxLayer(fresh, *second, K, R, rows_e, BM, rng);
        reuse_ok = reuse_ok && std::memcmp(x_reused.data(), x_fresh.data(), x_fresh.size() * sizeof(float)) == 0;
    }
    {
        // an aborted forward: layers 1 and 2 of 3 ran (every launched kernel runs to its end), layer 3 never launched
        std::vector<std::uint32_t> buf(words, 0u);
        fxLayer(buf, a, K, R, rows_e, BM, rng);
        fxLayer(buf, b2, K, R, rows_e, BM, rng);
        const std::vector<float> x_next = fxLayer(buf, b, K, R, rows_e, BM, rng);
        std::vector<std::uint32_t> fresh(words, 0u);
        const std::vector<float> x_clear = fxLayer(fresh, b, K, R, rows_e, BM, rng);
        abort_ok = std::memcmp(x_next.data(), x_clear.data(), x_clear.size() * sizeof(float)) == 0 && allZero(buf);
    }
    bool mutant_differs = false;
    {
        std::vector<std::uint32_t> stale(words, 0u);
        stale[static_cast<std::size_t>(b.n) * rows_e] = 1u;  // one stale counter: the finalize of that (token, row block) moves
        stale[5] = 77u;                                      // and one stale sum
        const std::vector<float> xs = fxLayer(stale, b, K, R, rows_e, BM, rng);
        std::vector<std::uint32_t> fresh(words, 0u);
        const std::vector<float> xf = fxLayer(fresh, b, K, R, rows_e, BM, rng);
        mutant_differs = std::memcmp(xs.data(), xf.data(), xf.size() * sizeof(float)) != 0;
    }
    check(zero_after, "fx reuse: the whole region is zero after every layer (sums swapped to 0, counters reset)");
    check(reuse_ok, "fx reuse: layer 2 on the same buffer without a clear == on a cleared buffer, bitwise (fewer rows and same rows)");
    check(abort_ok, "fx reuse: a forward after an aborted one (2 of 3 layers ran) without a clear == with a clear, bitwise");
    check(mutant_differs, "fx reuse mutant control: a dirty buffer (stale counter + sum) changes x");
}

}  // namespace

// tests/unit/unit_tests.cpp ORDER lists all modules
const whirl::test::Suite& unitSuite_kernels_moe() {
    static const whirl::test::Suite s{"kernels.moe",
                                      {
                                          {"testMoepSchedule", [] { testMoepSchedule(); }},
                                          {"testMoeSkipRef", [] { testMoeSkipRef(); }},  // v0252-f5
                                          {"testMoepDhsMap", [] { testMoepDhsMap(); }},  // v0252-8smoe2 (C)
                                          {"testMoepFqMap", [] { testMoepFqMap(); }},    // v0252-8smoe2 (E)
                                          {"testMoepFixSum", [] { testMoepFixSum(); }},  // v0252-8sdet (COMB 6)
                                          {"testMoepFixReuse", [] { testMoepFixReuse(); }},  // v0252-8sdet round 4
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_kernels_moe()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
