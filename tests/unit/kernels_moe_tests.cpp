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
#include "whirl/moe_shx_ref.h"  // v0252-f4 moeshx host reference

#include "../../kernels/moep_map.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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


// v0252-f4 (F4 moeshx, from feat028 7ef2cae testMoeShx part 2): the host numerics reference of moe_gus_mxfp4w /
// moe_downs_mxfp4w (include/whirl/moe_shx_ref.h, the kernels' f32 order) against a double-precision oracle. Fast tolerance
// (q8_1-class activations): per output |y - oracle| <= kShxTol x scale, scale = |x0| + sum |terms|; the moeshx form vs the
// balance form (shared expert summed separately): only the f32 order differs -> |fused - separate| <= 1e-5 x scale, and
// the fused RMS error is no larger than the balance one (+1 %). moeskip ids (-1) are dropped. Mutant: the shared weight
// ignored must fail the tolerance (the test sees a missing shared expert). Shapes: E 256 F 256 K 4 of 8 and Ornith's
// E 2048 F 512 K 8 (of 16 here).
constexpr double kShxTol = 1e-2;
whirl::moeshx_ref::Problem shxProblem(int E, int F, int K, int R, int n, unsigned seed, bool skips) {
    namespace mr = whirl::moeshx_ref;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> byte(0, 255), ex(119, 124);
    mr::Problem p;
    p.s = mr::Shapes{E, F, K, R};
    p.n = n;
    auto rows = [&](int nrows, int ncols) {
        const std::uint64_t rb = mr::mxRowBytes(ncols);
        std::vector<std::uint8_t> v(static_cast<std::size_t>(nrows) * rb);
        for (int r = 0; r < nrows; ++r)
            for (std::uint64_t sb = 0; sb < rb / 136; ++sb) {
                std::uint8_t* b = v.data() + static_cast<std::size_t>(r) * rb + sb * 136;
                for (int i = 0; i < 8; ++i) b[i] = static_cast<std::uint8_t>(ex(rng));
                for (int i = 8; i < 136; ++i) b[i] = static_cast<std::uint8_t>(byte(rng));
            }
        return v;
    };
    p.gate = rows(R * F, E);
    p.up = rows(R * F, E);
    p.down = rows(R * E, F);
    p.sg = rows(F, E);
    p.su = rows(F, E);
    p.sd = rows(E, F);
    p.h.resize(static_cast<std::size_t>(n) * E);
    for (float& v : p.h) v = nd(rng);
    p.x0.resize(p.h.size());
    for (float& v : p.x0) v = 0.1f * nd(rng);
    p.ids.resize(static_cast<std::size_t>(n) * K);
    p.w.resize(p.ids.size());
    p.shw.resize(static_cast<std::size_t>(n));
    std::uniform_real_distribution<float> uw(0.05f, 1.f), us(0.1f, 0.9f);
    for (int t = 0; t < n; ++t) {
        std::vector<int> perm(static_cast<std::size_t>(R));
        for (int i = 0; i < R; ++i) perm[static_cast<std::size_t>(i)] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            p.ids[static_cast<std::size_t>(t) * K + k] = perm[static_cast<std::size_t>(k)];
            sum += p.w[static_cast<std::size_t>(t) * K + k] = uw(rng);
        }
        for (int k = 0; k < K; ++k) p.w[static_cast<std::size_t>(t) * K + k] /= sum;
        if (skips && K > 2) {  // moeskip: the last two slots dropped (id -1, weight 0), kept ones renormalized
            p.ids[static_cast<std::size_t>(t) * K + K - 1] = -1;
            p.ids[static_cast<std::size_t>(t) * K + K - 2] = -1;
            float kept = 0;
            for (int k = 0; k < K - 2; ++k) kept += p.w[static_cast<std::size_t>(t) * K + k];
            for (int k = 0; k < K; ++k)
                p.w[static_cast<std::size_t>(t) * K + k] = k < K - 2 ? p.w[static_cast<std::size_t>(t) * K + k] / kept : 0.f;
        }
        p.shw[static_cast<std::size_t>(t)] = us(rng);
    }
    return p;
}
struct ShxErr {
    double max_rel = 0, rms = 0;  // max |y - ref| / scale, RMS of (y - ref) / scale
};
ShxErr shxErr(const std::vector<float>& y, const std::vector<double>& ref, const std::vector<double>& sc) {
    ShxErr e;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double r = std::fabs(static_cast<double>(y[i]) - ref[i]) / sc[i];
        e.max_rel = std::max(e.max_rel, r);
        e.rms += r * r;
    }
    e.rms = std::sqrt(e.rms / static_cast<double>(y.size()));
    return e;
}
void testMoeShxNumerics() {
    namespace mr = whirl::moeshx_ref;
    struct Case {
        int E, F, K, R, n;
        bool skips;
    };
    for (const Case cs : {Case{256, 256, 4, 8, 3, false}, Case{256, 256, 4, 8, 3, true}, Case{2048, 512, 8, 16, 2, false},
                          Case{2048, 512, 8, 16, 1, true}}) {
        const mr::Problem p = shxProblem(cs.E, cs.F, cs.K, cs.R, cs.n, 1234u + static_cast<unsigned>(cs.E + cs.K + cs.n + (cs.skips ? 7 : 0)), cs.skips);
        std::vector<double> sc;
        const std::vector<double> ref = mr::oracle(p, &sc);
        const std::vector<float> fused = mr::q8Forward(p, false), sep = mr::q8Forward(p, true);
        const ShxErr ef = shxErr(fused, ref, sc), es = shxErr(sep, ref, sc);
        double fs = 0;
        for (std::size_t i = 0; i < fused.size(); ++i) fs = std::max(fs, std::fabs(static_cast<double>(fused[i]) - sep[i]) / sc[i]);
        const std::string tag = " E " + std::to_string(cs.E) + " F " + std::to_string(cs.F) + " K " + std::to_string(cs.K) + " n " +
                                std::to_string(cs.n) + (cs.skips ? " (moeskip)" : "");
        std::printf("  moeshx%s: fused vs oracle max %.3e rms %.3e; balance form max %.3e rms %.3e; fused vs balance %.3e\n", tag.c_str(),
                    ef.max_rel, ef.rms, es.max_rel, es.rms, fs);
        check(ef.max_rel <= kShxTol, "moeshx fused within the fast tolerance" + tag);
        check(fs <= 1e-5, "moeshx fused == balance form within f32 order" + tag);
        check(ef.rms <= es.rms * 1.01 + 1e-7, "moeshx fused RMS <= balance RMS" + tag);
        mr::Problem m = p;
        for (float& v : m.shw) v = 0.f;
        check(shxErr(mr::q8Forward(m, false), ref, sc).max_rel > kShxTol, "moeshx mutant (shared weight ignored) fails" + tag);
    }
}

}  // namespace

// tests/unit/unit_tests.cpp ORDER lists all modules
const whirl::test::Suite& unitSuite_kernels_moe() {
    static const whirl::test::Suite s{"kernels.moe",
                                      {
                                          {"testMoepSchedule", [] { testMoepSchedule(); }},
                                          {"testMoeShxNumerics", [] { testMoeShxNumerics(); }},  // v0252-f4
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_kernels_moe()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
