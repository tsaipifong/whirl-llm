// whirl-kernel-test family "ws4r" (v0252-ws4r97, PLAN V1-87 phase 1): gfx1201 attn_ws4r* (4-wave pipelined split-K
// decode / verify attention, kernels/attn_ws4r.hip) bitwise == the shipped attn_wsplit1 / attn_wsplit2 (_q8).
// SPDX-License-Identifier: Apache-2.0
// Written BEFORE the kernels (test-first): against a code object without them the presence rows fail; against the stub
// / mutant objects (kernels/attn_ws4r.hip WS4R_MUT) the bitwise rows fail.
//
// Pairs (B = new, A = reference, same arguments, grid, 128 threads, AwGroups):
//   attn_ws4r1[_q8], attn_ws4r1b[_q8]  == attn_wsplit1[_q8]  (groups of <= 16 columns)
//   attn_ws4r2[_q8], attn_ws4r2b[_q8]  == attn_wsplit2[_q8]  (groups of <= 32 columns)
// Compared per (row, split, head): m and l bitwise everywhere; acc (256 floats) bitwise where m > -inf. Unwritten entries
// are NaN-filled first, so a kernel writing what the reference does not (or not writing) shows as a mismatch.
// Inputs, for f16 and q8 (int8 K and V + f16 scales; q8h uses the same kernels) KV:
//   pools   "rand" (random f16 / int8 bytes and scales, as family ws4), "model" (Gaussian rows with per-channel scales and
//           outlier channels, q8 quantized as kv_st32 does: scale = f16(amax / 127), q = round(x / scale)), "edge" (f16:
//           +-0, denormals, the largest finite values; int8: -127 / 0 / 127 bursts; scales: denormal, 0, 2^7)
//   heads   24 / 4 (Swift 27B, GQA 6) and 16 / 2 (Ornith 35B-A3B, GQA 8)
//   rows    decode (one row per group, 8 groups, two sequences' shuffled page tables), verify (1..8 rows at consecutive
//           positions grouped as the host does: same awPer, <= 16 / grp rows (pair 1) or <= 32 / grp (pair 2); plus groups
//           the host never builds: rows with different split sizes and gapped / shuffled positions)
//   contexts 1 .. 256k incl. chunk (32), split (64), page (256) boundaries, random
//   grids   64 splits (the R9700 default) and 16 / 128 / 256
//   window  MTP draft window (W 1024 threshold 2048; W 512; a window reaching the sink)
//   repeat  50 launches of the same batch == the first (timing-dependent LDS races)
// No real Swift KV dump is used (no KV dump facility in WHIRL yet; progress/v0252-ws4r97.HANDOFF.md).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "kt.h"

namespace kt {
namespace {

int rH = 24, rNkv = 4;
constexpr int rD = 256, rQStr = 2 * rD;
const float rSc = 1.0f / 16.0f;
bool r_grp8 = false;

enum class RF { f16, q8 };
const char* rfname(RF f) { return f == RF::f16 ? "f16" : "q8"; }
const char* rfsuf(RF f) { return f == RF::f16 ? "" : "_q8"; }
enum class RPool { rand, model, edge };
const char* rpname(RPool p) { return p == RPool::rand ? "rand" : p == RPool::model ? "model" : "edge"; }

// the new kernels and their references
struct Pair {
    const char* nb;  // new (suffix added per format)
    const char* na;  // reference
    int maxcols;     // columns per group the reference takes (16 or 32)
};
const Pair kPairs[] = {{"attn_ws4r1", "attn_wsplit1", 16}, {"attn_ws4r1b", "attn_wsplit1", 16}, {"attn_ws4r2", "attn_wsplit2", 32}, {"attn_ws4r2b", "attn_wsplit2", 32}};

struct XS {
    std::uint64_t s;
    std::uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    double unit() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    double gauss() {
        const double u1 = std::max(1e-300, unit()), u2 = unit();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

std::uint16_t f2hBits(float f) { return f2h(f); }

// paged pool: pages; ptab two sequences (tables [0, pages), [pages, 2 pages)), each a different shuffle
struct RPoolBuf {
    RF f;
    Buf k, v, ks, vs, ptab;
    RPoolBuf(Ctx& c, RF fmt, RPool kind, int np) : f(fmt) {
        const std::size_t rows = static_cast<std::size_t>(np) * wk::kKvPage, nel = rows * rNkv * rD;
        XS x{c.rng() | 1};
        if (f == RF::f16) {
            std::vector<std::uint16_t> h(nel);
            std::vector<float> chs(rD);
            for (int which = 0; which < 2; ++which) {
                for (int d = 0; d < rD; ++d) chs[static_cast<std::size_t>(d)] = static_cast<float>(std::exp(0.7 * x.gauss()) * ((x.next() % 16) == 0 ? 20.0 : 1.0));
                for (std::size_t i = 0; i < nel; ++i) {
                    const std::uint64_t r = x.next();
                    if (kind == RPool::rand) {
                        const std::uint32_t b = static_cast<std::uint32_t>(r);
                        const std::uint32_t e = 9 + (b >> 10) % 8;
                        h[i] = static_cast<std::uint16_t>((b & 0x8000u) | (e << 10) | (b & 0x3ffu));
                    } else if (kind == RPool::model) {
                        h[i] = f2hBits(static_cast<float>(x.gauss()) * chs[i % rD] * 0.25f);
                    } else {  // edge: +-0, denormals, largest finite, ordinary values
                        const std::uint32_t b = static_cast<std::uint32_t>(r), sel = (b >> 16) % 8, s = b & 0x8000u;
                        h[i] = static_cast<std::uint16_t>(sel == 0 ? s : sel == 1 ? (s | (b & 0x3ffu)) : sel == 2 ? (s | 0x7bffu) : sel == 3 ? (s | 0x7800u | (b & 0x3ffu))
                                                                                                                : (s | ((13 + (b >> 10) % 4) << 10) | (b & 0x3ffu)));
                    }
                }
                (which ? v : k) = Buf(h);
            }
        } else {
            std::vector<std::uint8_t> b(nel);
            std::vector<std::uint16_t> s(nel / 32);
            std::vector<float> row(32), chs(rD);
            for (int which = 0; which < 2; ++which) {
                for (int d = 0; d < rD; ++d) chs[static_cast<std::size_t>(d)] = static_cast<float>(std::exp(0.7 * x.gauss()) * ((x.next() % 16) == 0 ? 20.0 : 1.0));
                for (std::size_t g0 = 0; g0 < nel; g0 += 32) {
                    if (kind == RPool::model) {  // kv_st32's quantizer on Gaussian rows
                        float amax = 0.f;
                        for (int j = 0; j < 32; ++j) {
                            row[static_cast<std::size_t>(j)] = static_cast<float>(x.gauss()) * chs[(g0 + j) % rD] * 0.25f;
                            amax = std::max(amax, std::fabs(row[static_cast<std::size_t>(j)]));
                        }
                        const std::uint16_t sh = f2hBits(amax / 127.f);
                        const float sc = h2f(sh);
                        s[g0 / 32] = sh;
                        for (int j = 0; j < 32; ++j) {
                            const float qv = sc > 0.f ? std::nearbyint(row[static_cast<std::size_t>(j)] / sc) : 0.f;
                            b[g0 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::clamp(qv, -127.f, 127.f)));
                        }
                    } else {
                        const std::uint64_t r = x.next();
                        for (int j = 0; j < 32; ++j) {
                            std::uint8_t y = static_cast<std::uint8_t>(x.next());
                            if (kind == RPool::edge) {
                                const int sel = static_cast<int>((r >> (2 * (j % 16))) & 3);
                                y = sel == 0 ? 0x81 : sel == 1 ? 0x7f : sel == 2 ? 0 : y;
                            }
                            if (y == 0x80) y = 0x81;  // int8 -127 .. 127 as kv_st32 stores
                            b[g0 + j] = y;
                        }
                        const std::uint32_t rs = static_cast<std::uint32_t>(r >> 32);
                        if (kind == RPool::edge) {
                            const int sel = static_cast<int>(rs % 5);
                            s[g0 / 32] = static_cast<std::uint16_t>(sel == 0 ? 0x0001 : sel == 1 ? 0x0000 : sel == 2 ? 0x5800 : ((6 + rs % 4) << 10) | ((rs >> 8) & 0x3ffu));
                        } else {
                            s[g0 / 32] = static_cast<std::uint16_t>(((6 + rs % 4) << 10) | ((rs >> 8) & 0x3ffu));
                        }
                    }
                }
                (which ? v : k) = Buf(b);
                (which ? vs : ks) = Buf(s);
            }
        }
        std::vector<int> pt(static_cast<std::size_t>(2 * np));
        std::iota(pt.begin(), pt.begin() + np, 0);
        std::iota(pt.begin() + np, pt.end(), 0);
        std::shuffle(pt.begin(), pt.begin() + np, c.rng);
        std::shuffle(pt.begin() + np, pt.end(), c.rng);
        ptab = Buf(pt);
    }
    wk::KvArgs args(DevPtr kvb, int win) const {
        wk::KvArgs a{};
        a.k = k.p();
        a.v = v.p();
        a.ks = f == RF::f16 ? 0 : ks.p();
        a.vs = f == RF::f16 ? 0 : vs.p();
        a.ptab = ptab.p();
        a.kvbase = kvb;
        a.tab0 = 0;
        a.win = win;
        return a;
    }
};

struct RParts {
    std::vector<float> ml, acc;
};

// one launch: rows pos (kvbase tbs per row), groups (first, count), ns splits
RParts rrun(Ctx& c, hip::Function fn, const RPoolBuf& p, const Buf& dq, const std::vector<int>& pos, const std::vector<int>& tbs,
            const std::vector<std::pair<int, int>>& grps, int ns, int win) {
    const int n = static_cast<int>(pos.size());
    Buf dpos(pos), dkvb(tbs);
    Buf ml(static_cast<std::size_t>(n) * ns * rH * 2 * 4), acc(static_cast<std::size_t>(n) * ns * rH * rD * 4);
    ml.fill(0x7f);
    acc.fill(0x7f);
    wk::AwGroups g{};
    for (std::size_t i = 0; i < grps.size(); ++i) {
        g.first[i] = grps[i].first;
        g.count[i] = grps[i].second;
    }
    hip::launch(fn, {static_cast<unsigned>(rNkv), static_cast<unsigned>(ns), static_cast<unsigned>(grps.size())}, {128, 1, 1}, 0, c.s, dq.p(),
                p.args(dkvb.p(), win), ml.p(), acc.p(), rH, rNkv, rQStr, dpos.p(), rSc, DevPtr{0}, g);
    c.sync();
    return {ml.down<float>(static_cast<std::size_t>(n) * ns * rH * 2), acc.down<float>(static_cast<std::size_t>(n) * ns * rH * rD)};
}

// bitwise: every ml pair; acc where the reference's m > -inf; NaN-filled (unwritten) entries compared as bits too
std::pair<std::size_t, std::size_t> rcmp(const RParts& a, const RParts& b) {
    std::size_t bad = 0, cnt = 0;
    const std::size_t nml = std::min(a.ml.size(), b.ml.size());
    for (std::size_t i = 0; i + 1 < nml; i += 2) {
        cnt += 2;
        const bool d = std::memcmp(&a.ml[i], &b.ml[i], 8) != 0;
        bad += d;
        const float mref = b.ml[i];
        if (!d && !std::isnan(mref) && !(std::isinf(mref) && mref < 0)) {
            cnt += rD;
            for (int j = 0; j < rD; ++j) bad += std::memcmp(&a.acc[i / 2 * rD + j], &b.acc[i / 2 * rD + j], 4) != 0;
        }
    }
    return {bad, cnt};
}

int awPerH(int n_pos, int ns) { return (n_pos + ns * 64 - 1) / (ns * 64) * 64; }

// host grouping (src/ops/attention.cpp group()): consecutive rows of one sequence at consecutive positions with the
// same awPer, at most maxq rows
std::vector<std::pair<int, int>> hostGroups(const std::vector<int>& pos, const std::vector<int>& tbs, int ns, int maxq) {
    std::vector<std::pair<int, int>> g;
    for (int r = 0; r < static_cast<int>(pos.size()); ++r) {
        if (!g.empty()) {
            auto& b = g.back();
            const int r0 = b.first;
            if (b.second < maxq && tbs[static_cast<std::size_t>(r)] == tbs[static_cast<std::size_t>(r0)] && pos[static_cast<std::size_t>(r)] == pos[static_cast<std::size_t>(r0)] + b.second &&
                awPerH(pos[static_cast<std::size_t>(r)] + 1, ns) == awPerH(pos[static_cast<std::size_t>(r0)] + 1, ns)) {
                b.second += 1;
                continue;
            }
        }
        g.push_back({r, 1});
    }
    return g;
}

struct Acc {
    std::size_t bad = 0, cnt = 0, cases = 0;
    std::string first_bad;
    void add(const std::pair<std::size_t, std::size_t>& r, const std::string& what) {
        bad += r.first;
        cnt += r.second;
        cases += 1;
        if ((r.first || r.second == 0) && first_bad.empty()) first_bad = what + " (" + std::to_string(r.first) + " of " + std::to_string(r.second) + ")";
    }
};

void addResult(Ctx& c, const std::string& name, const Acc& a) {
    Result res;
    res.name = name + ", " + std::to_string(a.cases) + " cases" + (a.first_bad.empty() ? "" : "; first bad: " + a.first_bad);
    res.kind = Kind::invariant;
    res.n = a.cnt;
    res.mismatches = a.bad;
    res.pass = a.bad == 0 && a.cnt > 0 && a.cases > 0;
    c.rep.add(res);
}

void ws4rPass(Ctx& c) {
    const char* tag0 = r_grp8 ? " [grp 8: 16 / 2 heads]" : " [grp 6: 24 / 4 heads]";
    const int grp = rH / rNkv;
    const int maxpos = (c.quick || r_grp8) ? 70000 : 262144;
    const int pages = (maxpos + wk::kKvPage - 1) / wk::kKvPage;
    std::vector<int> ctxs = {1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 2048, 4095, 4096, 4097, 8191, 32767, 32768, 32769, 65536, 65613};
    if (!c.quick && !r_grp8)
        for (int x : {98304, 131071, 131072, 131073, 196613, 262143, 262144}) ctxs.push_back(x);
    for (int i = 0; i < 10; ++i) ctxs.push_back(1 + static_cast<int>(c.rng() % static_cast<std::uint64_t>(maxpos)));
    const std::vector<float> q = c.randn(static_cast<std::size_t>(16) * rH * rQStr);
    Buf dq(q);
    for (RF f : {RF::q8, RF::f16}) {
        for (RPool pk : {RPool::rand, RPool::model, RPool::edge}) {
            if (r_grp8 && pk == RPool::edge) continue;  // the edge pool once (GQA 6)
            const std::string tag = std::string(" [kv ") + rfname(f) + ", pool " + rpname(pk) + "]" + tag0;
            RPoolBuf p(c, f, pk, pages);
            for (const Pair& pr : kPairs) {
                const std::string nb = std::string(pr.nb) + rfsuf(f), na = std::string(pr.na) + rfsuf(f);
                const hip::Function fb = c.fnOpt(nb), fa = c.fnOpt(na);
                if (!fa) {
                    c.rep.skip("ws4r", na, "reference kernel not in this code object");
                    continue;
                }
                if (!fb) continue;  // the presence row already failed
                const int maxq = std::max(1, pr.maxcols / grp);
                Acc dec, ver, odd, grid, win;
                // (1) decode: one row per group, batches of 8, alternating page tables, 64 splits
                for (std::size_t i = 0; i < ctxs.size(); i += 8) {
                    std::vector<int> pos, tbs;
                    std::vector<std::pair<int, int>> gs;
                    for (std::size_t j = i; j < std::min(ctxs.size(), i + 8); ++j) {
                        gs.push_back({static_cast<int>(pos.size()), 1});
                        pos.push_back(ctxs[j] - 1);
                        tbs.push_back((j % 2) ? pages : 0);
                    }
                    dec.add(rcmp(rrun(c, fb, p, dq, pos, tbs, gs, 64, 0), rrun(c, fa, p, dq, pos, tbs, gs, 64, 0)), "ctx batch at " + std::to_string(ctxs[i]));
                }
                // (2) verify: R rows at consecutive positions ending at L - 1, host grouping, two sequences (second ends 1000
                // earlier); every context at 1 / maxq rows, a subset at 1..8 rows
                const std::vector<int> vsub = {33, 257, 4097, 32769, ctxs.back(), std::min(maxpos, 65613), std::min(maxpos, 131072)};
                for (int L : ctxs) {
                    const bool sub = std::find(vsub.begin(), vsub.end(), L) != vsub.end();
                    for (int R : {1, 2, 3, 4, 5, 6, 8}) {
                        if (!sub && R != 1 && R != maxq) continue;
                        std::vector<int> pos, tbs;
                        for (int r = std::max(0, L - R); r < L; ++r) {
                            pos.push_back(r);
                            tbs.push_back(0);
                        }
                        if (L > 1100 && R >= 3)
                            for (int r = std::max(0, L - 1000 - R); r < L - 1000; ++r) {
                                pos.push_back(r);
                                tbs.push_back(pages);
                            }
                        const auto gs = hostGroups(pos, tbs, 64, maxq);
                        ver.add(rcmp(rrun(c, fb, p, dq, pos, tbs, gs, 64, 0), rrun(c, fa, p, dq, pos, tbs, gs, 64, 0)),
                                "ctx " + std::to_string(L) + " rows " + std::to_string(R));
                    }
                }
                // (3) groups the host never builds: maxq rows with gapped / shuffled positions and different split sizes
                for (int L : {300, 4097, 65613, 131000}) {
                    if (L > maxpos) continue;
                    std::vector<int> pos = {L - 1, L - 5, L - 40, L - 33, L - 2, L - 64, L - 31, L - 100};
                    pos.resize(static_cast<std::size_t>(std::min<int>(maxq, 8)));
                    std::shuffle(pos.begin(), pos.end(), c.rng);
                    const std::vector<int> tbs(pos.size(), 0);
                    std::vector<std::pair<int, int>> gs;
                    gs.push_back({0, static_cast<int>(pos.size())});
                    odd.add(rcmp(rrun(c, fb, p, dq, pos, tbs, gs, 64, 0), rrun(c, fa, p, dq, pos, tbs, gs, 64, 0)), "gapped ctx " + std::to_string(L));
                    std::vector<int> spl = {4095, 4096 + 4000};  // awPer 64 vs 128 at 64 splits, one group
                    spl.resize(static_cast<std::size_t>(std::min<int>(maxq, 2)));
                    const std::vector<int> tb2(spl.size(), 0);
                    std::vector<std::pair<int, int>> g2;
                    g2.push_back({0, static_cast<int>(spl.size())});
                    odd.add(rcmp(rrun(c, fb, p, dq, spl, tb2, g2, 64, 0), rrun(c, fa, p, dq, spl, tb2, g2, 64, 0)), "mixed split size");
                }
                // (4) other grids: 16 / 128 / 256 splits
                for (int ns : {16, 128, 256})
                    for (int L : {65, 1024, 4097, 32769, 65613}) {
                        for (int R : {1, maxq}) {
                            std::vector<int> pos, tbs;
                            for (int r = std::max(0, L - R); r < L; ++r) {
                                pos.push_back(r);
                                tbs.push_back(pages);
                            }
                            const auto gs = hostGroups(pos, tbs, ns, maxq);
                            grid.add(rcmp(rrun(c, fb, p, dq, pos, tbs, gs, ns, 0), rrun(c, fa, p, dq, pos, tbs, gs, ns, 0)),
                                     "grid " + std::to_string(ns) + " ctx " + std::to_string(L) + " rows " + std::to_string(R));
                        }
                    }
                // (5) MTP draft window
                for (int w : {(1024 / 64) | (2 << 16), (512 / 64), (131072 / 64) | (1 << 16)})
                    for (int L : {2049, 4100, 9001, 65613})
                        for (int R : {1, maxq}) {
                            std::vector<int> pos, tbs;
                            for (int r = std::max(0, L - R); r < L; ++r) {
                                pos.push_back(r);
                                tbs.push_back(0);
                            }
                            const auto gs = hostGroups(pos, tbs, 64, maxq);
                            win.add(rcmp(rrun(c, fb, p, dq, pos, tbs, gs, 64, w), rrun(c, fa, p, dq, pos, tbs, gs, 64, w)),
                                    "window " + std::to_string(w) + " ctx " + std::to_string(L) + " rows " + std::to_string(R));
                        }
                const std::string base = nb + " == " + na;
                addResult(c, base + " decode (1 row per group, 64 splits)" + tag, dec);
                addResult(c, base + " verify (1..8 rows, host groups, 64 splits)" + tag, ver);
                addResult(c, base + " gapped / mixed-split groups" + tag, odd);
                addResult(c, base + " grids 16 / 128 / 256" + tag, grid);
                addResult(c, base + " draft window" + tag, win);
                // (6) 50 repeated launches == the first (rand pool only: races do not depend on values)
                if (pk == RPool::rand) {
                    std::vector<int> pos, tbs;
                    const int Lr = std::min(131072, maxpos);
                    for (int r = Lr - maxq; r < Lr; ++r) {
                        pos.push_back(r);
                        tbs.push_back(0);
                    }
                    const auto gs = hostGroups(pos, tbs, 64, maxq);
                    const RParts f0 = rrun(c, fb, p, dq, pos, tbs, gs, 64, 0);
                    Acc rep;
                    for (int it = 0; it < 50; ++it) rep.add(rcmp(rrun(c, fb, p, dq, pos, tbs, gs, 64, 0), f0), "launch " + std::to_string(it));
                    addResult(c, nb + " 50 repeated launches == first" + tag, rep);
                }
            }
        }
    }
}

}  // namespace

void testAttnWs4r(Ctx& c) {
    c.rep.family = "ws4r";
    if (c.k.caps.attn_group1) {  // gfx1151: the kernels are gfx1201 only (requiredKernelAbsentOk "ws4r attn_ws4r")
        c.rep.skip("ws4r", "attn_ws4r", "kernel not in this code object (gfx1201 only)");
        return;
    }
    // presence rows (the RED proof against a code object without the kernels)
    for (const Pair& pr : kPairs)
        for (RF f : {RF::f16, RF::q8}) {
            const std::string nm = std::string(pr.nb) + rfsuf(f);
            Result r;
            r.name = nm + " (v0252-ws4r97) in the gfx1201 code object";
            r.kind = Kind::invariant;
            r.n = 1;
            r.pass = static_cast<bool>(c.fnOpt(nm));
            r.mismatches = r.pass ? 0 : 1;
            if (!r.pass) r.note = "missing from the code object";
            c.rep.add(r);
        }
    ws4rPass(c);
    rH = 16, rNkv = 2, r_grp8 = true;
    ws4rPass(c);
    rH = 24, rNkv = 4, r_grp8 = false;
}

}  // namespace kt
