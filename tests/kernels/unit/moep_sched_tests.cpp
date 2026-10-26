// feat026-moep: CPU tests of the multi-fragment MoE prefill expert GEMM schedule (no GPU). They replay, with the
// index maps the kernels use (kernels/moep_map.h) and the variant tables (whirl/kernels_abi.h kMoepGu / kMoepDn):
//   * moe_route / moe_tiles tile lists for token tile bn (expert counts -> (expert, first pos, count));
//   * every 1-D grid block -> (tile, row block), every wave -> (row fragment, interleaved token fragments), every
//     lane / accumulator element -> stored (pos, row): each routed (pos, row) is stored exactly once, nothing else;
//   * the fused combine's last-arriver counters under a random block order: each (token, row block) is finalized
//     exactly once, only after all K of the token's expert rows of that row block were stored, counters end at 0;
//   * the LDS / register-tile constraints of each variant (the kernels' static_asserts, the 64 KiB LDS of a gfx11 WG).
// Routings: skewed (one hot expert), Zipf(1.0) and uniform; 17 / 200 / 2048 tokens; Ornith E 2048, F 512, R 256, K 8.
// SPDX-License-Identifier: Apache-2.0
//
// Run: whirl-moep-sched-tests.exe (exit 0 = all passed).

#include "whirl/kernels_abi.h"

#include "../../../kernels/moep_map.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

int g_pass = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL %s\n", what.c_str());
    }
}

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

}  // namespace

int main() {
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
    std::printf("moep_sched_tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
