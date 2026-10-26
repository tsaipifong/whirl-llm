// whirl-kernel-test families "ws4" (bitwise) and "ws4bench" (timing, on request): gfx1151 attn_ws4
// (4-wave pipelined split-K decode attention, kernels/gfx1151/attn_ws4.hip) against attn_wsplit1.
// SPDX-License-Identifier: Apache-2.0
// ws4: for f16 / q8 / q4 KV (synthetic pool, shuffled pages, two sequences with their own page
// tables through kvbase), one query per group (the gfx1151 host's only kind): the partials
// (m, l, acc of every split) of attn_ws4[_q8|_q4] are bitwise
// those of attn_wsplit1[_q8|_q4] for contexts 1 .. 256k incl. chunk / page / split boundaries
// and random positions, with the MTP draft window, for a smaller grid (split rule independent of
// gridDim.y), and over 200 repeated launches with changing batches.
// ws4bench: kernel time and KV GB/s at 32k / 64k / 96k / 128k (WHIRL_KT_BENCH_L) for
// attn_wsplit1 / attn_ws4 / attn_dq4 per format, plus attn_combine.

#include <algorithm>
#include <array>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "gfx1151/attn_ws4_index.h"
#include "kt.h"

namespace kt {
namespace {

// heads: the 27B (24 q / 4 kv heads, GQA 6) by default; the second pass of ws4 / ws4g uses the
// qwen35moe shape (16 / 2, GQA 8: one 8-column group per row, tiles aligned with rows)
int kH = 24, kNkv = 4;
constexpr int kD = 256, kQStr = 2 * kD, kNs = 64;
// the GQA-8 pass: reduced context list (the 27B pass covers 256k), result names tagged
bool g_grp8 = false;
const char* grpTag() { return g_grp8 ? " [grp 8: 16 / 2 heads]" : ""; }
const float kSc = 1.0f / 16.0f;

enum class F { f16, q8, q4 };
const char* fname(F f) { return f == F::f16 ? "f16" : f == F::q8 ? "q8" : "q4"; }
const char* fsuf(F f) { return f == F::f16 ? "" : f == F::q8 ? "_q8" : "_q4"; }

struct XorShift {
    std::uint64_t s;
    std::uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

// synthetic paged pool of `pages` pages; ptab: two sequences (tables [0, pages), [pages, 2 pages)),
// each a different shuffle of the pool
struct Pool {
    F f;
    int pages;
    Buf k, v, ks, vs, ptab, kvbase;
    Pool(Ctx& c, F fmt, int np) : f(fmt), pages(np) {
        const std::size_t rows = static_cast<std::size_t>(np) * wk::kKvPage, nel = rows * kNkv * kD;
        XorShift x{c.rng() | 1};
        if (f == F::f16) {
            // f16 values: random sign, exponent 2^-6 .. 2^1, random mantissa (no inf / nan)
            std::vector<std::uint16_t> h(nel);
            for (int which = 0; which < 2; ++which) {
                for (std::size_t i = 0; i < nel; i += 4) {
                    const std::uint64_t r = x.next();
                    for (int j = 0; j < 4; ++j) {
                        const std::uint32_t b = static_cast<std::uint32_t>(r >> (16 * j));
                        const std::uint32_t e = 9 + (b >> 10) % 8;
                        h[i + j] = static_cast<std::uint16_t>((b & 0x8000u) | (e << 10) | (b & 0x3ffu));
                    }
                }
                (which ? v : k) = Buf(h);
            }
        } else {
            const std::size_t nb = f == F::q8 ? nel : nel / 2;
            std::vector<std::uint8_t> b(nb);
            for (int which = 0; which < 2; ++which) {
                for (std::size_t i = 0; i < nb; i += 8) {
                    std::uint64_t r = x.next();
                    for (int j = 0; j < 8 && i + j < nb; ++j) {
                        std::uint8_t y = static_cast<std::uint8_t>(r >> (8 * j));
                        if (f == F::q8 && y == 0x80) y = 0x81;  // int8 -127 .. 127 as kv_st32 stores
                        b[i + j] = y;
                    }
                }
                (which ? v : k) = Buf(b);
            }
            // scales 2^-9 .. 2^-6 (f16, random mantissa)
            std::vector<std::uint16_t> s(nel / 32);
            for (int which = 0; which < 2; ++which) {
                for (auto& y : s) {
                    const std::uint32_t r = static_cast<std::uint32_t>(x.next());
                    y = static_cast<std::uint16_t>(((6 + r % 4) << 10) | ((r >> 8) & 0x3ffu));
                }
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
    wk::KvArgs args(DevPtr kvb, int win = 0) const {
        wk::KvArgs a;
        a.k = k.p();
        a.v = v.p();
        a.ks = f == F::f16 ? 0 : ks.p();
        a.vs = f == F::f16 ? 0 : vs.p();
        a.ptab = ptab.p();
        a.kvbase = kvb;
        a.tab0 = 0;
        a.win = win;
        return a;
    }
};

struct Partials {
    std::vector<float> ml, acc;
};

// launch fn over rows (one per group): positions pos[r], page table offset tbs[r]
Partials run(Ctx& c, hip::Function fn, const Pool& p, const Buf& dq, const std::vector<int>& pos, const std::vector<int>& tbs, int ns, int win) {
    const int n = static_cast<int>(pos.size());
    Buf dpos(pos), dkvb(tbs);
    Buf ml(static_cast<std::size_t>(n) * ns * kH * 2 * 4), acc(static_cast<std::size_t>(n) * ns * kH * kD * 4);
    ml.fill(0x7f);  // unwritten entries: NaN pattern (compared only where written)
    acc.fill(0x7f);
    wk::AwGroups g{};
    for (int r = 0; r < n; ++r) {
        g.first[r] = r;
        g.count[r] = 1;
    }
    hip::launch(fn, {static_cast<unsigned>(kNkv), static_cast<unsigned>(ns), static_cast<unsigned>(n)}, {128, 1, 1}, 0, c.s, dq.p(), p.args(dkvb.p(), win), ml.p(),
                acc.p(), kH, kNkv, kQStr, dpos.p(), kSc, DevPtr{0}, g);
    c.sync();
    return {ml.down<float>(static_cast<std::size_t>(n) * ns * kH * 2), acc.down<float>(static_cast<std::size_t>(n) * ns * kH * kD)};
}

// bitwise compare per (row, split, head): ml both; acc where the split carries keys (m > -inf).
// nsa / nsb: grid splits of a / b (layouts), compared over the first min(nsa, nsb) splits; splits
// beyond in the larger grid must be empty
std::pair<std::size_t, std::size_t> cmpParts(const Partials& a, int nsa, const Partials& b, int nsb, int n) {
    std::size_t bad = 0, cnt = 0;
    const int nsm = std::max(nsa, nsb);
    for (int r = 0; r < n; ++r)
        for (int s = 0; s < nsm; ++s)
            for (int h = 0; h < kH; ++h) {
                const bool ina = s < nsa, inb = s < nsb;
                const float* ma = ina ? &a.ml[((static_cast<std::size_t>(r) * nsa + s) * kH + h) * 2] : nullptr;
                const float* mb = inb ? &b.ml[((static_cast<std::size_t>(r) * nsb + s) * kH + h) * 2] : nullptr;
                if (!ina || !inb) {
                    const float* m1 = ina ? ma : mb;
                    cnt += 1;
                    bad += !(std::isinf(m1[0]) && m1[0] < 0);
                    continue;
                }
                cnt += 2;
                const bool d = std::memcmp(ma, mb, 8) != 0;
                bad += d;
                if (!d && !(std::isinf(ma[0]) && ma[0] < 0)) {
                    const float* aa = &a.acc[((static_cast<std::size_t>(r) * nsa + s) * kH + h) * kD];
                    const float* ab = &b.acc[((static_cast<std::size_t>(r) * nsb + s) * kH + h) * kD];
                    cnt += kD;
                    for (int i = 0; i < kD; ++i) bad += std::memcmp(aa + i, ab + i, 4) != 0;
                }
            }
    return {bad, cnt};
}
void compare(Ctx& c, const std::string& name, const Partials& a, int nsa, const Partials& b, int nsb, int n) {
    const auto [bad, cnt] = cmpParts(a, nsa, b, nsb, n);
    Result res;
    res.name = name;
    res.kind = Kind::invariant;
    res.n = cnt;
    res.mismatches = bad;
    res.pass = bad == 0 && cnt > 0;
    c.rep.add(res);
}

}  // namespace

// host checks of the shared index header (the properties the T1 emulator verified): every
// (key, dim) of a chunk staged exactly once for K (per format) and V, Vt / Sc addresses distinct
// and in range, Vt rows 16-byte aligned, and the split ranges tiling [0, n_pos) (with and without
// the draft window) for 64 splits
void ws4IndexChecks(Ctx& c) {
    std::size_t bad = 0, n = 0;
    for (int F = 0; F < 3; ++F) {
        std::vector<int> hit(ws4::kKey * ws4::kD, 0);
        for (int tv = 0; tv < 128; ++tv)
            for (int p = 0; p < ws4::k_pieces(F); ++p)
                for (int i = 0; i < ws4::k_len(F); ++i) hit[static_cast<std::size_t>(ws4::k_key(F, tv, p) * ws4::kD + ws4::k_d0(F, tv) + i)] += 1;
        for (int h : hit) bad += h != 1;
        n += hit.size();
    }
    {
        std::vector<int> hit(ws4::kKey * ws4::kD, 0), vt(ws4::kVtSize, 0);
        for (int tv = 0; tv < 128; ++tv)
            for (int r = 0; r < 8; ++r)
                for (int i = 0; i < 8; ++i) {
                    const int key = ws4::v_kb(tv) * 8 + r, d = ws4::v_d8(tv) + i;
                    hit[static_cast<std::size_t>(key * ws4::kD + d)] += 1;
                    const int a = ws4::vt_addr(d, ws4::vt_col(ws4::v_kb(tv), r));
                    if (a < 0 || a >= ws4::kVtSize) bad += 1;
                    else vt[static_cast<std::size_t>(a)] += 1;
                }
        for (int h : hit) bad += h != 1;
        for (int d = 0; d < ws4::kD; ++d) bad += (ws4::vt_addr(d, 0) % 8) != 0;
        int used = 0;
        for (int h : vt) {
            bad += h > 1;
            used += h;
        }
        bad += used != ws4::kKey * ws4::kD;
        n += hit.size() + ws4::kD + 1;
    }
    {
        std::vector<int> sc(ws4::kScSize, 0);
        for (int kt = 0; kt < 2; ++kt)
            for (int lane = 0; lane < 32; ++lane)
                for (int v = 0; v < 8; ++v) sc[static_cast<std::size_t>(ws4::sc_addr(kt, lane, v))] += 1;
        for (int h : sc) bad += h != 1;
        n += sc.size();
    }
    for (int win : {0, (1024 / 64) | (2 << 16), (512 / 64)}) {
        for (int np : {1, 31, 64, 65, 256, 257, 2047, 2048, 4095, 4097, 9000, 65613, 131072, 262144}) {
            std::vector<int> cov(static_cast<std::size_t>(np), 0);
            const ws4::Range s0 = ws4::block_range(np, 0, ws4::kNs, win);
            const bool active = win != 0 && s0.b0 == 0 && s0.b1 == ws4::kSink && np > ws4::kSink && ws4::block_range(np, 1, ws4::kNs, win).b0 > ws4::kSink;
            for (int s = 0; s < ws4::kNs; ++s) {
                const ws4::Range r = ws4::block_range(np, s, ws4::kNs, win);
                bad += (r.b0 % 32) != 0;
                for (int k = r.b0; k < r.b1; ++k) cov[static_cast<std::size_t>(k)] += 1;
                for (int k = r.b0; k < r.b1; k += 32) bad += (k / wk::kKvPage) != (std::min(k + 31, r.b1 - 1) / wk::kKvPage);  // no page straddle
            }
            const int w0 = active ? ws4::block_range(np, 1, ws4::kNs, win).b0 : 0;
            for (int k = 0; k < np; ++k) bad += cov[static_cast<std::size_t>(k)] != ((active && k >= ws4::kSink && k < w0) ? 0 : 1);
            n += static_cast<std::size_t>(np);
        }
    }
    Result r;
    r.name = "attn_ws4 index header: staging coverage, Vt / Sc addresses, split tiling";
    r.n = n;
    r.mismatches = bad;
    r.pass = bad == 0;
    c.rep.add(r);
}

static void ws4Pass(Ctx& c);
static void ws4gPass(Ctx& c);

void testAttnWs4(Ctx& c) {
    c.rep.family = "ws4";
    ws4IndexChecks(c);
    ws4Pass(c);
    kH = 16, kNkv = 2, g_grp8 = true;
    ws4Pass(c);
    kH = 24, kNkv = 4, g_grp8 = false;
}

static void ws4Pass(Ctx& c) {
    if (!c.fnOpt("attn_ws4")) {
        c.rep.skip("ws4", "attn_ws4", "kernel not in this code object");
        return;
    }
    const int maxpos = (c.quick || g_grp8) ? 70000 : 262144;
    const int pages = (maxpos + wk::kKvPage - 1) / wk::kKvPage;
    std::vector<int> ctxs = {1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 4095, 4096, 4097, 8191, 32767, 32768, 32769, 65613};
    if (!c.quick && !g_grp8)
        for (int x : {98304, 131071, 131072, 131073, 196613, 262143, 262144}) ctxs.push_back(x);
    for (int i = 0; i < 24; ++i) ctxs.push_back(1 + static_cast<int>(c.rng() % static_cast<std::uint64_t>(maxpos)));
    const std::vector<float> q = c.randn(static_cast<std::size_t>(8) * kH * kQStr);
    Buf dq(q);
    // WHIRL_KT_WS4_BASE: a code object built before attn_wsplit_impl was split into the shared
    // aw_* device functions; its attn_wsplit1* must give the same bits as this build's
    std::vector<char> base_img;
    hip::Module base_mod;
    bool have_base = false;
    if (const char* e = std::getenv("WHIRL_KT_WS4_BASE")) {
        if (std::FILE* fp = std::fopen(e, "rb")) {
            std::fseek(fp, 0, SEEK_END);
            base_img.resize(static_cast<std::size_t>(std::ftell(fp)));
            std::fseek(fp, 0, SEEK_SET);
            if (std::fread(base_img.data(), 1, base_img.size(), fp) != base_img.size()) base_img.clear();
            std::fclose(fp);
        }
        if (!base_img.empty()) {
            base_mod = hip::Module::loadData(base_img.data(), c.mod.arch());
            have_base = true;
        }
    }
    // WHIRL_KT_AD8S_HSACO (feat026-attndec8s): a candidate code object (-DWHIRL_ATTNDEC8S=<id>); its attn_ws4*
    // must give the bits of this build's attn_wsplit1* (== this build's attn_ws4*, checked in the same pass)
    std::vector<char> cand_img;
    hip::Module cand_mod;
    bool have_cand = false;
    if (const char* e = std::getenv("WHIRL_KT_AD8S_HSACO")) {
        if (std::FILE* fp = std::fopen(e, "rb")) {
            std::fseek(fp, 0, SEEK_END);
            cand_img.resize(static_cast<std::size_t>(std::ftell(fp)));
            std::fseek(fp, 0, SEEK_SET);
            if (std::fread(cand_img.data(), 1, cand_img.size(), fp) != cand_img.size()) cand_img.clear();
            std::fclose(fp);
        }
        if (!cand_img.empty()) {
            cand_mod = hip::Module::loadData(cand_img.data(), c.mod.arch());
            have_cand = true;
            std::printf("   candidate attn_ws4* from %s\n", e);
        }
    }
    for (F f : {F::f16, F::q8, F::q4}) {
        const std::string tag = std::string(" [kv ") + fname(f) + "]" + grpTag();
        const auto fold = c.fnOpt(std::string("attn_wsplit1") + fsuf(f));
        if (!fold) {
            c.rep.skip("ws4", std::string("attn_ws4") + fsuf(f), "attn_wsplit1 reference not in this code object");
            continue;
        }
        Pool p(c, f, pages);
        std::vector<std::pair<std::string, hip::Function>> news;
        {
            const std::string nm = std::string("attn_ws4") + fsuf(f);
            if (auto fn = c.fnOpt(nm)) news.push_back({nm, fn});
        }
        if (news.empty()) continue;
        if (have_base)
            if (auto fb = base_mod.getFunctionOpt((std::string("attn_wsplit1") + fsuf(f)).c_str()))
                news.push_back({std::string("base attn_wsplit1") + fsuf(f), fb});
        std::vector<std::pair<std::string, hip::Function>> cands;  // also through the draft-window / repeat checks
        if (have_cand)
            if (auto fc = cand_mod.getFunctionOpt((std::string("attn_ws4") + fsuf(f)).c_str())) {
                news.push_back({std::string("cand attn_ws4") + fsuf(f), fc});
                cands.push_back(news.back());
            }
        // contexts in batches of 8 rows (one per group), alternating the two sequences' page tables
        for (std::size_t i = 0; i < ctxs.size(); i += 8) {
            std::vector<int> pos, tbs;
            std::string lst;
            for (std::size_t j = i; j < std::min(ctxs.size(), i + 8); ++j) {
                pos.push_back(ctxs[j] - 1);
                tbs.push_back((j % 2) ? pages : 0);
                lst += (lst.empty() ? "" : ",") + std::to_string(ctxs[j]);
            }
            const Partials ref = run(c, fold, p, dq, pos, tbs, kNs, 0);
            for (const auto& [nm, fn] : news) {
                const Partials got = run(c, fn, p, dq, pos, tbs, kNs, 0);
                compare(c, nm + " == attn_wsplit1" + fsuf(f) + " (ctx " + lst + ")" + tag, got, kNs, ref, kNs, static_cast<int>(pos.size()));
            }
        }
        const auto& [nm0, fn0] = news[0];
        // MTP draft window (W 1024, threshold 2048; and a window reaching the sink)
        {
            const std::vector<int> pos = {2047, 4099, 9000, 65612}, tbs = {0, pages, 0, pages};
            for (int win : {(1024 / 64) | (2 << 16), (512 / 64) | (0 << 16), (131072 / 64) | (1 << 16)}) {
                const Partials ref = run(c, fold, p, dq, pos, tbs, kNs, win);
                const Partials got = run(c, fn0, p, dq, pos, tbs, kNs, win);
                compare(c, nm0 + " == attn_wsplit1" + fsuf(f) + " (draft window arg " + std::to_string(win) + ")" + tag, got, kNs, ref, kNs, 4);
                for (const auto& [cn, cf] : cands) {
                    const Partials gc = run(c, cf, p, dq, pos, tbs, kNs, win);
                    compare(c, cn + " == attn_wsplit1" + fsuf(f) + " (draft window arg " + std::to_string(win) + ")" + tag, gc, kNs, ref, kNs, 4);
                }
            }
        }
        // split rule independent of the grid (gridDim.y 16 vs 64, contexts <= 1024)
        {
            const std::vector<int> pos = {0, 63, 64, 500, 1023}, tbs = {0, pages, 0, pages, 0};
            const Partials a = run(c, fn0, p, dq, pos, tbs, 16, 0);
            const Partials b = run(c, fn0, p, dq, pos, tbs, kNs, 0);
            compare(c, nm0 + " grid 16 splits == grid 64 splits (ctx <= 1024)" + tag, a, 16, b, kNs, 5);
        }
        // 200 launches with changing batches (1 .. 8 rows): each == its first run
        {
            std::vector<std::vector<int>> batches;
            std::vector<Partials> firsts;
            for (int b = 0; b < 8; ++b) {
                std::vector<int> pos;
                for (int r = 0; r <= b; ++r) pos.push_back(static_cast<int>(c.rng() % static_cast<std::uint64_t>(std::min(maxpos, 140000))));
                batches.push_back(pos);
            }
            std::size_t bad = 0, cnt = 0;
            for (int it = 0; it < 200; ++it) {
                const int b = it < 8 ? it : static_cast<int>(c.rng() % 8);
                const std::vector<int>& pos = batches[static_cast<std::size_t>(b)];
                const std::vector<int> tbs(pos.size(), 0);
                const Partials got = run(c, fn0, p, dq, pos, tbs, kNs, 0);
                if (it < 8) {
                    firsts.push_back(got);
                    continue;
                }
                const Partials& f0 = firsts[static_cast<std::size_t>(b)];
                cnt += got.ml.size() + got.acc.size();
                bad += std::memcmp(got.ml.data(), f0.ml.data(), got.ml.size() * 4) != 0;
                for (std::size_t i = 0; i < got.ml.size(); i += 2) {
                    if (std::isinf(got.ml[i]) && got.ml[i] < 0) continue;
                    bad += std::memcmp(&got.acc[i / 2 * kD], &f0.acc[i / 2 * kD], kD * 4) != 0;
                }
            }
            Result res;
            res.name = nm0 + " 192 repeated launches (batches of 1..8 rows) == first" + tag;
            res.kind = Kind::invariant;
            res.n = cnt;
            res.mismatches = bad;
            res.pass = bad == 0;
            c.rep.add(res);
        }
    }
}

void benchAttnWs4(Ctx& c) {
    c.rep.family = "ws4bench";
    std::vector<int> lens = {32768, 65536, 98304, 131072};
    if (const char* e = std::getenv("WHIRL_KT_BENCH_L")) lens = {std::atoi(e)};
    const int iters = std::getenv("WHIRL_KT_BENCH_ITERS") ? std::atoi(std::getenv("WHIRL_KT_BENCH_ITERS")) : 20;
    const std::vector<float> q = c.randn(static_cast<std::size_t>(kH) * kQStr);
    Buf dq(q);
    const int maxl = *std::max_element(lens.begin(), lens.end());
    // WHIRL_KT_BENCH_HSACO: attn_ws4* from this code object instead (kernel development)
    std::vector<char> image;
    hip::Module xmod;
    bool have_x = false;
    if (const char* e = std::getenv("WHIRL_KT_BENCH_HSACO")) {
        if (std::FILE* fp = std::fopen(e, "rb")) {
            std::fseek(fp, 0, SEEK_END);
            image.resize(static_cast<std::size_t>(std::ftell(fp)));
            std::fseek(fp, 0, SEEK_SET);
            if (std::fread(image.data(), 1, image.size(), fp) != image.size()) image.clear();
            std::fclose(fp);
        }
        if (!image.empty()) {
            xmod = hip::Module::loadData(image.data(), c.mod.arch());
            have_x = true;
            std::printf("   attn_ws4* from %s\n", e);
        }
    }
    std::vector<F> fmts = {F::q8, F::f16, F::q4};
    if (const char* e = std::getenv("WHIRL_KT_BENCH_FMT")) {
        const std::string fs = e;
        fmts = {fs == "f16" ? F::f16 : fs == "q4" ? F::q4 : F::q8};
    }
    for (F f : fmts) {
        Pool p(c, f, (maxl + wk::kKvPage - 1) / wk::kKvPage);
        const double row_bytes = f == F::f16 ? 2.0 * kD * 2 : f == F::q8 ? 2.0 * (kD + kD / 32 * 2) : 2.0 * (kD / 2 + kD / 32 * 2);
        std::vector<std::pair<std::string, unsigned>> names = {{std::string("attn_wsplit1") + fsuf(f), 128},
                                                               {std::string("attn_ws4") + fsuf(f), 128}};
        if (const char* e = std::getenv("WHIRL_KT_BENCH_KERNELS")) names.push_back({e, 128});
        if (f == F::q4) names.push_back({"attn_dq4", 256});
        for (const int L : lens) {
            Buf dpos(std::vector<int>{L - 1}), dkvb(std::vector<int>{0});
            Buf ml(static_cast<std::size_t>(kNs) * kH * 2 * 4), acc(static_cast<std::size_t>(kNs) * kH * kD * 4), out(kH * kD * 4);
            wk::AwGroups g{};
            g.first[0] = 0;
            g.count[0] = 1;
            for (const auto& [nm, threads] : names) {
                const auto fn = have_x && nm.rfind("attn_ws4", 0) == 0 ? xmod.getFunctionOpt(nm.c_str()) : c.fnOpt(nm);
                if (!fn) {
                    std::printf("   %s: not in this code object\n", nm.c_str());
                    continue;
                }
                auto launch = [&]() {
                    hip::launch(fn, {static_cast<unsigned>(kNkv), kNs, 1}, {threads, 1, 1}, 0, c.s, dq.p(), p.args(dkvb.p()), ml.p(), acc.p(), kH, kNkv, kQStr, dpos.p(), kSc,
                                DevPtr{0}, g);
                };
                for (int i = 0; i < 3; ++i) launch();
                c.sync();
                hip::Event e0 = hip::eventCreate(true), e1 = hip::eventCreate(true);
                hip::eventRecord(e0, c.s);
                for (int i = 0; i < iters; ++i) launch();
                hip::eventRecord(e1, c.s);
                hip::eventSync(e1);
                const double us = 1000.0 * hip::eventElapsedMs(e0, e1) / iters;
                hip::eventDestroy(e0);
                hip::eventDestroy(e1);
                const double bytes = static_cast<double>(L) * kNkv * row_bytes;
                std::printf("   ws4bench %-4s L %6d %-20s %8.1f us  %6.1f GB/s\n", fname(f), L, nm.c_str(), us, bytes / us / 1e3);
                std::fflush(stdout);
            }
            if (f == F::q8 && L == lens[0]) {  // attn_combine (24 heads, 64 splits), for the fusion estimate
                auto launch = [&]() {
                    hip::launch(c.k.attn_combine, {static_cast<unsigned>(kH), 1, 1}, {256, 1, 1}, 0, c.s, ml.p(), acc.p(), dq.p(), out.p(), kH, kD, kQStr, kNs);
                };
                for (int i = 0; i < 3; ++i) launch();
                c.sync();
                hip::Event e0 = hip::eventCreate(true), e1 = hip::eventCreate(true);
                hip::eventRecord(e0, c.s);
                for (int i = 0; i < iters; ++i) launch();
                hip::eventRecord(e1, c.s);
                hip::eventSync(e1);
                std::printf("   ws4bench attn_combine (24 heads, 64 splits): %8.1f us per launch (back to back)\n",
                            1000.0 * hip::eventElapsedMs(e0, e1) / iters);
                hip::eventDestroy(e0);
                hip::eventDestroy(e1);
            }
        }
    }
}


// ---------------------------------------------------------------------------
// ws4g (bitwise): attn_ws4g<CG> multi-row verify (one KV read for up to 8 rows x 6 GQA heads per
// block, column groups on grid.x beyond) against the single-row kernels: for f16 / q8 KV, every
// row's partials (m, l, acc of every split with keys) of a grouped launch are bitwise those of
// attn_ws4 (one row per group) and of attn_wsplit1 (the old path), for 1 .. 32 rows at
// consecutive positions ending at contexts 1 .. 256k incl. chunk / page / split boundaries and
// random ones, shuffled page tables (two sequences, one per group), shuffled and gapped
// positions within a group, the MTP draft window, every CG for the same rows, and 100 repeated
// launches with changing groupings. Rows are grouped as the host must: equal split rule
// (awPer and draft-window ranges; ws4grp).
// ws4gbench: kernel time of 1 / 2 / 4 / 8 / 16 / 32 verify rows (consecutive positions ending at
// L - 1) for attn_ws4g vs the single-row attn_ws4 and the old per-row attn_wsplit1 groups.
namespace {

int kGrp() { return kH / kNkv; }

// natural tile count for a group of `rows` rows (16-column tiles, at most 3 = 8 rows per block)
int ws4gCg(int rows) { return std::min(3, (rows * kGrp() + 15) / 16); }

// host grouping rule: consecutive rows with the same split ranges (per and draft window)
std::vector<std::pair<int, int>> ws4grp(const std::vector<int>& pos, int win, int maxc = 32) {
    std::vector<std::pair<int, int>> g;
    auto key = [&](int p) {
        const int n = p + 1;
        return std::array<int, 3>{ws4::block_range(n, 0, ws4::kNs, win).b1 == ws4::kSink ? ws4::block_range(n, 0, ws4::kNs, win).b0 - 1 : 0,
                                  ws4::block_range(n, 1, ws4::kNs, win).b0, ws4::block_range(n, 2, ws4::kNs, win).b0};
    };
    for (int r = 0; r < static_cast<int>(pos.size()); ++r) {
        if (!g.empty() && g.back().second < maxc && key(pos[static_cast<std::size_t>(r)]) == key(pos[static_cast<std::size_t>(g.back().first)]))
            g.back().second += 1;
        else
            g.push_back({r, 1});
    }
    return g;
}

// grouped launch: 256 threads, grid (n_kv x column groups, ns, groups); kvbase per row
Partials runG(Ctx& c, hip::Function fn, int cg, const Pool& p, const Buf& dq, const std::vector<int>& pos, const std::vector<int>& tbs,
              const std::vector<std::pair<int, int>>& grps, int ns, int win) {
    const int n = static_cast<int>(pos.size());
    Buf dpos(pos), dkvb(tbs);
    Buf ml(static_cast<std::size_t>(n) * ns * kH * 2 * 4), acc(static_cast<std::size_t>(n) * ns * kH * kD * 4);
    ml.fill(0x7f);
    acc.fill(0x7f);
    wk::AwGroups g{};
    const int rpb = ws4::rows_per_block(cg, kGrp());
    int ncg = 1;
    for (std::size_t i = 0; i < grps.size(); ++i) {
        g.first[i] = grps[i].first;
        g.count[i] = grps[i].second;
        ncg = std::max(ncg, (grps[i].second + rpb - 1) / rpb);
    }
    hip::launch(fn, {static_cast<unsigned>(kNkv * ncg), static_cast<unsigned>(ns), static_cast<unsigned>(grps.size())}, {256, 1, 1}, 0, c.s, dq.p(),
                p.args(dkvb.p(), win), ml.p(), acc.p(), kH, kNkv, kQStr, dpos.p(), kSc, DevPtr{0}, g);
    c.sync();
    return {ml.down<float>(static_cast<std::size_t>(n) * ns * kH * 2), acc.down<float>(static_cast<std::size_t>(n) * ns * kH * kD)};
}

}  // namespace

void testAttnWs4g(Ctx& c) {
    c.rep.family = "ws4g";
    ws4gPass(c);
    kH = 16, kNkv = 2, g_grp8 = true;
    ws4gPass(c);
    kH = 24, kNkv = 4, g_grp8 = false;
}

static void ws4gPass(Ctx& c) {
    {
        std::size_t bad = 0;
        for (int kb = 0; kb < 4; ++kb)
            for (int r = 0; r < 8; ++r) bad += ws4::vt_key(ws4::vt_col(kb, r)) != kb * 8 + r;
        Result res;
        res.name = "attn_ws4g index: vt_key inverts vt_col";
        res.n = 32;
        res.mismatches = bad;
        res.pass = bad == 0;
        c.rep.add(res);
    }
    if (!c.fnOpt("attn_ws4g3")) {
        c.rep.skip("ws4g", "attn_ws4g3", "kernel not in this code object");
        return;
    }
    const int maxpos = (c.quick || g_grp8) ? 70000 : 262144;
    const int pages = (maxpos + wk::kKvPage - 1) / wk::kKvPage;
    std::vector<int> ctxs = {1, 2, 3, 8, 31, 32, 33, 40, 63, 64, 65, 255, 256, 257, 4095, 4096, 4097, 4100, 8191, 32767, 32768, 32769, 65613};
    if (!c.quick && !g_grp8)
        for (int x : {98304, 131071, 131072, 131073, 196613, 262143, 262144}) ctxs.push_back(x);
    for (int i = 0; i < 8; ++i) ctxs.push_back(1 + static_cast<int>(c.rng() % static_cast<std::uint64_t>(maxpos)));
    const std::vector<float> q = c.randn(static_cast<std::size_t>(32) * kH * kQStr);
    Buf dq(q);
    // WHIRL_KT_WS4G_HSACO: attn_ws4g* from this code object instead (mutation check of the test:
    // a build with -DWS4G_NOTAIL=1 must fail)
    std::vector<char> ximg;
    hip::Module xmod;
    bool have_x = false;
    if (const char* e = std::getenv("WHIRL_KT_WS4G_HSACO")) {
        if (std::FILE* fp = std::fopen(e, "rb")) {
            std::fseek(fp, 0, SEEK_END);
            ximg.resize(static_cast<std::size_t>(std::ftell(fp)));
            std::fseek(fp, 0, SEEK_SET);
            if (std::fread(ximg.data(), 1, ximg.size(), fp) != ximg.size()) ximg.clear();
            std::fclose(fp);
        }
        if (!ximg.empty()) {
            xmod = hip::Module::loadData(ximg.data(), c.mod.arch());
            have_x = true;
            std::printf("   attn_ws4g* from %s\n", e);
        }
    }
    for (F f : {F::f16, F::q8}) {
        const std::string tag = std::string(" [kv ") + fname(f) + "]" + grpTag();
        const auto fold = c.fnOpt(std::string("attn_wsplit1") + fsuf(f));
        const auto fsolo = c.fnOpt(std::string("attn_ws4") + fsuf(f));
        hip::Function fg[4];
        for (int cg = 1; cg <= 3; ++cg) {
            const std::string nm = std::string("attn_ws4g") + std::to_string(cg) + fsuf(f);
            fg[cg] = have_x ? xmod.getFunctionOpt(nm.c_str()) : c.fnOpt(nm);
        }
        if (!fold || !fsolo || !fg[1] || !fg[2] || !fg[3]) {
            c.rep.skip("ws4g", std::string("attn_ws4g") + fsuf(f), "kernels not in this code object");
            continue;
        }
        Pool p(c, f, pages);
        std::size_t bad_s = 0, bad_o = 0, cnt = 0, cases = 0;
        std::string first_bad;
        // one case: rows `pos` (two groups' worth possible), grouped by the host rule, kvbase per group
        auto check = [&](const std::vector<int>& pos, int win, int cg_force, const std::string& what) {
            const auto grps = ws4grp(pos, win);
            std::vector<int> tbs(pos.size(), 0);
            for (std::size_t gi = 0; gi < grps.size(); ++gi)
                for (int r = 0; r < grps[gi].second; ++r) tbs[static_cast<std::size_t>(grps[gi].first + r)] = (gi % 2) ? pages : 0;
            int maxc = 1;
            for (const auto& gg : grps) maxc = std::max(maxc, gg.second);
            const int cg = cg_force ? cg_force : ws4gCg(maxc);
            const Partials got = runG(c, fg[cg], cg, p, dq, pos, tbs, grps, kNs, win);
            const Partials solo = run(c, fsolo, p, dq, pos, tbs, kNs, win);
            const Partials old = run(c, fold, p, dq, pos, tbs, kNs, win);
            const auto r1 = cmpParts(got, kNs, solo, kNs, static_cast<int>(pos.size()));
            const auto r2 = cmpParts(got, kNs, old, kNs, static_cast<int>(pos.size()));
            bad_s += r1.first;
            bad_o += r2.first;
            cnt += r1.second;
            cases += 1;
            if ((r1.first || r2.first || r1.second == 0) && first_bad.empty())
                first_bad = what + " cg " + std::to_string(cg) + " rows " + std::to_string(pos.size()) + " (vs ws4 " + std::to_string(r1.first) +
                            ", vs wsplit1 " + std::to_string(r2.first) + ")";
            if ((r1.first || r2.first) && c.rep.verbose)
                std::printf("   ws4g mismatch: %s cg %d rows %zu: vs ws4 %zu, vs wsplit1 %zu\n", what.c_str(), cg, pos.size(), r1.first, r2.first);
        };
        auto consec = [&](int L, int R) {  // R rows at consecutive positions ending at L - 1 (>= 0)
            std::vector<int> pos;
            for (int r = std::max(0, L - R); r < L; ++r) pos.push_back(r);
            return pos;
        };
        // (1) every row count 1..32 at a few contexts, natural CG
        for (int L : {33, 4097, 65613, 131072})
            for (int R = 1; R <= 32; ++R) check(consec(std::min(L, maxpos), R), 0, 0, "ctx " + std::to_string(L));
        // (2) all contexts at 1, 2, 3, 5, 8, 9, 16, 32 rows; natural CG
        for (int L : ctxs)
            for (int R : {1, 2, 3, 5, 8, 9, 16, 32}) check(consec(L, R), 0, 0, "ctx " + std::to_string(L));
        // (3) the same rows through every CG (column groups on grid.x for CG < natural)
        for (int L : {65, 257, 4096, 32769, 131071})
            for (int R : {2, 5, 8, 13})
                for (int cg = 1; cg <= 3; ++cg) check(consec(std::min(L, maxpos), R), 0, cg, "cg sweep ctx " + std::to_string(L));
        // (4) shuffled / gapped positions within a group, two groups (two sequences) per launch
        for (int L : {300, 4096, 65613, 131000}) {
            std::vector<int> pos = consec(L, 8);
            std::shuffle(pos.begin(), pos.end(), c.rng);
            check(pos, 0, 0, "shuffled ctx " + std::to_string(L));
            const std::vector<int> gap = {L - 1, L - 5, L - 40, L - 33, L - 2, L - 64, L - 31, L - 100};
            check(gap, 0, 0, "gapped ctx " + std::to_string(L));
            std::vector<int> two = consec(L, 8), b = consec(L - 1000, 6);
            two.insert(two.end(), b.begin(), b.end());
            check(two, 0, 0, "two groups ctx " + std::to_string(L));
        }
        // (5) MTP draft window (W 1024 threshold 2048; W 512; W reaching the sink)
        for (int win : {(1024 / 64) | (2 << 16), (512 / 64), (131072 / 64) | (1 << 16)})
            for (int L : {2049, 4100, 9001, 65613})
                for (int R : {3, 8, 32}) check(consec(L, R), win, 0, "window " + std::to_string(win) + " ctx " + std::to_string(L));
        Result res;
        res.name = "attn_ws4g grouped (1..32 rows) == attn_ws4 single-row, " + std::to_string(cases) + " cases" + tag +
                   (first_bad.empty() ? "" : "; first bad: " + first_bad);
        res.kind = Kind::invariant;
        res.n = cnt;
        res.mismatches = bad_s;
        res.pass = bad_s == 0 && cnt > 0;
        c.rep.add(res);
        res.name = "attn_ws4g grouped (1..32 rows) == attn_wsplit1 single-row (old path), " + std::to_string(cases) + " cases" + tag;
        res.mismatches = bad_o;
        res.pass = bad_o == 0 && cnt > 0;
        c.rep.add(res);
        // (6) 100 repeated launches with changing groupings: each == its first run
        {
            std::vector<std::vector<int>> sets;
            std::vector<Partials> firsts;
            for (int b = 0; b < 6; ++b) {
                const int L = 1 + static_cast<int>(c.rng() % static_cast<std::uint64_t>(std::min(maxpos, 140000)));
                sets.push_back(consec(L, 1 + static_cast<int>(c.rng() % 32)));
            }
            std::size_t bad = 0, n = 0;
            for (int it = 0; it < 106; ++it) {
                const int b = it < 6 ? it : static_cast<int>(c.rng() % 6);
                const std::vector<int>& pos = sets[static_cast<std::size_t>(b)];
                const auto grps = ws4grp(pos, 0);
                int maxc = 1;
                for (const auto& gg : grps) maxc = std::max(maxc, gg.second);
                const int cg = ws4gCg(maxc);
                const Partials got = runG(c, fg[cg], cg, p, dq, pos, std::vector<int>(pos.size(), 0), grps, kNs, 0);
                if (it < 6) {
                    firsts.push_back(got);
                    continue;
                }
                const Partials& f0 = firsts[static_cast<std::size_t>(b)];
                n += got.ml.size();
                bad += std::memcmp(got.ml.data(), f0.ml.data(), got.ml.size() * 4) != 0;
                for (std::size_t i = 0; i < got.ml.size(); i += 2) {
                    if (std::isinf(got.ml[i]) && got.ml[i] < 0) continue;
                    bad += std::memcmp(&got.acc[i / 2 * kD], &f0.acc[i / 2 * kD], kD * 4) != 0;
                }
            }
            Result r2;
            r2.name = "attn_ws4g 100 repeated launches (1..32 rows) == first" + tag;
            r2.kind = Kind::invariant;
            r2.n = n;
            r2.mismatches = bad;
            r2.pass = bad == 0;
            c.rep.add(r2);
        }
    }
}

void benchAttnWs4g(Ctx& c) {
    c.rep.family = "ws4gbench";
    std::vector<int> lens = {32768, 65536, 131072};
    if (const char* e = std::getenv("WHIRL_KT_BENCH_L")) lens = {std::atoi(e)};
    const int iters = std::getenv("WHIRL_KT_BENCH_ITERS") ? std::atoi(std::getenv("WHIRL_KT_BENCH_ITERS")) : 20;
    std::vector<int> rowset = {1, 2, 4, 8, 16, 32};
    const bool old = std::getenv("WHIRL_KT_BENCH_OLD") != nullptr;  // also time the old per-row attn_wsplit1 groups
    const std::vector<float> q = c.randn(static_cast<std::size_t>(32) * kH * kQStr);
    Buf dq(q);
    const int maxl = *std::max_element(lens.begin(), lens.end());
    std::vector<F> fmts = {F::q8, F::f16};
    if (const char* e = std::getenv("WHIRL_KT_BENCH_FMT")) fmts = {std::string(e) == "f16" ? F::f16 : F::q8};
    for (F f : fmts) {
        Pool p(c, f, (maxl + wk::kKvPage - 1) / wk::kKvPage);
        const double row_bytes = f == F::f16 ? 2.0 * kD * 2 : 2.0 * (kD + kD / 32 * 2);
        for (const int L : lens) {
            Buf ml(static_cast<std::size_t>(32) * kNs * kH * 2 * 4), acc(static_cast<std::size_t>(32) * kNs * kH * kD * 4);
            double t1 = 0;
            auto timeit = [&](const std::function<void()>& launch) {
                for (int i = 0; i < 3; ++i) launch();
                c.sync();
                hip::Event e0 = hip::eventCreate(true), e1 = hip::eventCreate(true);
                hip::eventRecord(e0, c.s);
                for (int i = 0; i < iters; ++i) launch();
                hip::eventRecord(e1, c.s);
                hip::eventSync(e1);
                const double us = 1000.0 * hip::eventElapsedMs(e0, e1) / iters;
                hip::eventDestroy(e0);
                hip::eventDestroy(e1);
                return us;
            };
            const double bytes = static_cast<double>(L) * kNkv * row_bytes;
            for (int R : rowset) {
                std::vector<int> pos;
                for (int r = L - R; r < L; ++r) pos.push_back(r);
                Buf dpos(pos), dkvb(std::vector<int>(32, 0));
                const auto kva = p.args(dkvb.p());
                if (R == 1) {
                    wk::AwGroups g{};
                    g.first[0] = 0;
                    g.count[0] = 1;
                    const auto fn = c.fnOpt(std::string("attn_ws4") + fsuf(f));
                    t1 = timeit([&]() {
                        hip::launch(fn, {static_cast<unsigned>(kNkv), kNs, 1}, {128, 1, 1}, 0, c.s, dq.p(), kva, ml.p(), acc.p(), kH, kNkv, kQStr, dpos.p(), kSc, DevPtr{0}, g);
                    });
                    std::printf("   ws4gbench %-4s L %6d rows  1 attn_ws4 (single row)      %8.1f us  %6.1f GB/s\n", fname(f), L, t1, bytes / t1 / 1e3);
                }
                const auto grps = ws4grp(pos, 0);
                int maxc = 1;
                for (const auto& gg : grps) maxc = std::max(maxc, gg.second);
                const int cg = ws4gCg(maxc), rpb = ws4::rows_per_block(cg, kGrp());
                wk::AwGroups g{};
                for (std::size_t i = 0; i < grps.size(); ++i) {
                    g.first[i] = grps[i].first;
                    g.count[i] = grps[i].second;
                }
                const auto fn = c.fnOpt(std::string("attn_ws4g") + std::to_string(cg) + fsuf(f));
                const unsigned gx = static_cast<unsigned>(kNkv * ((maxc + rpb - 1) / rpb));
                const double us = timeit([&]() {
                    hip::launch(fn, {gx, kNs, static_cast<unsigned>(grps.size())}, {256, 1, 1}, 0, c.s, dq.p(), kva, ml.p(), acc.p(), kH, kNkv, kQStr,
                                dpos.p(), kSc, DevPtr{0}, g);
                });
                std::printf("   ws4gbench %-4s L %6d rows %2d attn_ws4g%d (%zu grp)           %8.1f us  x%.2f of 1 row  +%.1f us/extra row\n", fname(f), L, R, cg,
                            grps.size(), us, us / t1, R > 1 ? (us - t1) / (R - 1) : 0.0);
                if (old && R <= 8) {
                    wk::AwGroups go{};
                    for (int r = 0; r < R; ++r) {
                        go.first[r] = r;
                        go.count[r] = 1;
                    }
                    const auto fo = c.fnOpt(std::string("attn_wsplit1") + fsuf(f));
                    const double uo = timeit([&]() {
                        hip::launch(fo, {static_cast<unsigned>(kNkv), kNs, static_cast<unsigned>(R)}, {128, 1, 1}, 0, c.s, dq.p(), kva, ml.p(), acc.p(), kH, kNkv, kQStr, dpos.p(), kSc,
                                    DevPtr{0}, go);
                    });
                    std::printf("   ws4gbench %-4s L %6d rows %2d attn_wsplit1 (per-row groups)  %8.1f us\n", fname(f), L, R, uo);
                }
                std::fflush(stdout);
            }
        }
    }
}
}  // namespace kt
