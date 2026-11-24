// whirl-kernel-test --bench ws4s: the HIP-free cell table (kt_ws4s.h). v0252-8sws, PLAN V1-88 on gfx1151.
// SPDX-License-Identifier: Apache-2.0

#include "kt_ws4s.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "../../kernels/gfx1151/attn_ws4_index.h"  // ws4::block_range, ws4::kNs (the kernel's split rule)
#include "whirl/device_profile.h"
#include "whirl/kernels_abi.h"  // kernels::kWs4SplitCounts / ws4SplitName (the product _sp entries)
#include "whirl/ops/attention.h"  // ops::ws4SplitCount (the product rule, WHIRL_KT_WS4S_TABLE=1)

namespace kt::ws4s {

static_assert(kShip == ws4::kNs, "the shipping split constant of attn_ws4 / attn_ws4g");
static_assert(kHd == ws4::kD, "head dim of attn_ws4");

namespace {

std::vector<std::string> splitList(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char ch : s) {
        if (ch == ',' || ch == ' ' || ch == ';') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

bool parseInt(const std::string& t, int lo, int hi, int& v) {
    if (t.empty() || t.size() > 9) return false;
    for (const char ch : t)
        if (ch < '0' || ch > '9') return false;
    v = std::atoi(t.c_str());
    return v >= lo && v <= hi;
}

// src/ops/attention.cpp chooseAttnDecode, ws4 path: the group key of a row at position pp (the draft-window sink split and the
// first two split starts of its n_pos), with the split constant ns
std::array<int, 3> groupKey(int pp, int win, int ns) {
    const int np = pp + 1;
    const ws4::Range r0 = ws4::block_range(np, 0, ns, win);
    return {r0.b1 == ws4::kSink ? r0.b0 - 1 : 0, ws4::block_range(np, 1, ns, win).b0, ws4::block_range(np, 2, ns, win).b0};
}

}  // namespace

Opts optsFromEnv(const std::function<const char*(const char*)>& getenv_fn, std::string* err) {
    Opts o;
    o.models = {"ornith", "swift"};
    o.kvs = {"q8h", "f16"};
    o.keys = {8192, 16384, 32768, 49152, 65536, 98304, 131072};
    o.rowsets = {{"1", 1, false}, {"2", 2, false}, {"3", 3, false}, {"d1", 1, true}};
    o.splits = {8, 16, 32, 128};
    std::string e;
    if (const char* v = getenv_fn("WHIRL_KT_WS4S_MODELS")) {
        o.models.clear();
        for (const std::string& t : splitList(v)) {
            const bool known = std::any_of(std::begin(kShapes), std::end(kShapes), [&](const Shape& s) { return t == s.model; });
            if (!known) e += "WHIRL_KT_WS4S_MODELS: unknown model '" + t + "' (ornith, swift); ";
            else o.models.push_back(t);
        }
    }
    if (const char* v = getenv_fn("WHIRL_KT_WS4S_KV")) {
        o.kvs.clear();
        for (const std::string& t : splitList(v)) {
            if (t == "q8h" || t == "f16") o.kvs.push_back(t);
            else e += "WHIRL_KT_WS4S_KV: unknown KV kind '" + t + "' (q8h, f16); ";
        }
    }
    if (const char* v = getenv_fn("WHIRL_KT_WS4S_L")) {
        o.keys.clear();
        for (const std::string& t : splitList(v)) {
            int x = 0;
            if (!parseInt(t, 1, 262144, x)) e += "WHIRL_KT_WS4S_L: bad key count '" + t + "' (1..262144); ";
            else o.keys.push_back(x);
        }
    }
    if (const char* v = getenv_fn("WHIRL_KT_WS4S_ROWS")) {
        o.rowsets.clear();
        for (const std::string& t : splitList(v)) {
            int x = 0;
            if (t == "d1") o.rowsets.push_back({"d1", 1, true});
            else if (parseInt(t, 1, kMaxRows, x)) o.rowsets.push_back({t, x, false});
            else e += "WHIRL_KT_WS4S_ROWS: bad row set '" + t + "' (1..32 or d1); ";
        }
    }
    if (const char* v = getenv_fn("WHIRL_KT_WS4S_S")) {
        o.splits.clear();
        for (const std::string& t : splitList(v)) {
            int x = 0;
            if (!parseInt(t, 1, 256, x) || !nsBuilt(x)) e += "WHIRL_KT_WS4S_S: '" + t + "' has no kernel entry (8 16 32 64 128); ";
            else o.splits.push_back(x);
        }
        std::sort(o.splits.begin(), o.splits.end());
        o.splits.erase(std::unique(o.splits.begin(), o.splits.end()), o.splits.end());
    }
    if (const char* v = getenv_fn("WHIRL_KT_WS4S_AA")) o.aa = std::string(v) != "" && std::string(v) != "0";
    if (const char* v = getenv_fn("WHIRL_KT_WS4S_TABLE")) o.table = std::string(v) != "" && std::string(v) != "0";
    if (o.models.empty() || o.kvs.empty() || o.keys.empty() || o.rowsets.empty() || o.splits.empty()) e += "an empty list; ";
    if (err) *err = e;
    return o;
}

Plan plan(int heads, int nkv, const std::vector<int>& pos, int win, int ns, int max_ctx) {
    Plan p;
    const int n = static_cast<int>(pos.size());
    if (n < 1 || nkv < 1 || ns < 1) return p;
    const int grp = heads / nkv;
    // ops: n_split = min(ws4::kNs, ceil(max_ctx / fd_chunk 64)) -> here min(ns, ..)
    p.gy = static_cast<unsigned>(std::min(ns, (max_ctx + 63) / 64));
    // ops: known = rows_known && n > 1; one sequence (row_base all equal), rows at consecutive positions grouped while the key matches
    const bool known = n > 1;
    int r = 0, widest = 1;
    while (r < n) {
        int c = 1;
        if (known) {
            const auto k0 = groupKey(pos[static_cast<std::size_t>(r)], win, ns);
            while (r + c < n && c < kMaxRows && pos[static_cast<std::size_t>(r + c)] == pos[static_cast<std::size_t>(r)] + c &&
                   groupKey(pos[static_cast<std::size_t>(r + c)], win, ns) == k0)
                c += 1;
        }
        p.groups.push_back({r, c});
        widest = std::max(widest, c);
        r += c;
    }
    p.gz = static_cast<unsigned>(p.groups.size());
    if (widest == 1) {
        p.multi = false;
        p.cg = 0;
        p.gx = static_cast<unsigned>(nkv);
        p.block = 128;
    } else {
        p.multi = true;
        p.cg = std::min(3, (widest * grp + 15) / 16);
        const int rpb = ws4::rows_per_block(p.cg, grp);
        p.gx = static_cast<unsigned>(nkv * ((widest + rpb - 1) / rpb));
        p.block = 256;
    }
    return p;
}

std::string kernelName(bool q8, const Plan& p, int ns) {
    // v0252-8sws round 2: q8 at a product split constant = the product entry (attn_ws4[g<cg>]_q8_sp<S>)
    for (const std::uint32_t sp : whirl::kernels::kWs4SplitCounts)
        if (q8 && static_cast<int>(sp) == ns) return whirl::kernels::ws4SplitName(p.multi ? p.cg : 0, sp);
    std::string n = p.multi ? "attn_ws4g" + std::to_string(p.cg) : std::string("attn_ws4");
    if (q8) n += "_q8";
    if (ns != kShip) n += "_ns" + std::to_string(ns);
    return n;
}

bool nsBuilt(int ns) {
    if (ns == kShip) return true;
    return std::any_of(std::begin(kBenchNs), std::end(kBenchNs), [ns](int x) { return x == ns; });
}

bool gridOk(int heads, int nkv, const std::vector<int>& pos, int win, int ns) {
    if (!nsBuilt(ns) || pos.empty() || static_cast<int>(pos.size()) > kMaxRows || nkv < 1 || heads % nkv != 0) return false;
    if (win != 0 && ns < 2) return false;
    for (const int x : pos)
        if (x < 0) return false;
    const Plan p = plan(heads, nkv, pos, win, ns);
    // every key of every row lies in one of the gy splits (block_range of the row's own n_pos; a group uses its largest)
    for (const int x : pos) {
        const int np = x + 1;
        int covered = 0;
        for (int s = 0; s < static_cast<int>(p.gy); ++s) {
            const ws4::Range rg = ws4::block_range(np, s, ns, win);
            if (rg.b0 < np) covered += std::max(0, std::min(rg.b1, np) - rg.b0);
        }
        // the draft window skips the keys between the sink and the window start (by design), so count only the attended keys
        const ws4::Range r0 = ws4::block_range(np, 0, ns, win), r1 = ws4::block_range(np, 1, ns, win);
        const bool win_on = win != 0 && r0.b0 == 0 && r0.b1 == ws4::kSink && r1.b0 > ws4::kSink;
        const int want = win_on ? ws4::kSink + (np - r1.b0) : np;
        if (covered != want) return false;
    }
    return true;
}

bool samePartition(const std::vector<int>& pos, int win, int s, int t) {
    for (const int x : pos) {
        const int np = x + 1;
        std::vector<std::array<int, 2>> a, b;
        for (int i = 0; i < s; ++i) {
            const ws4::Range r = ws4::block_range(np, i, s, win);
            if (r.b0 < np && r.b1 > r.b0) a.push_back({r.b0, r.b1});
        }
        for (int i = 0; i < t; ++i) {
            const ws4::Range r = ws4::block_range(np, i, t, win);
            if (r.b0 < np && r.b1 > r.b0) b.push_back({r.b0, r.b1});
        }
        if (a != b) return false;
    }
    return true;
}

std::vector<Cell> cells(const Opts& o) {
    std::vector<Cell> out;
    for (const std::string& m : o.models) {
        const Shape* sh = nullptr;
        for (const Shape& s : kShapes)
            if (m == s.model) sh = &s;
        if (!sh) continue;
        std::vector<int> keys = o.keys;
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        for (const std::string& kv : o.kvs)
            for (const int L : keys)
                for (const RowSet& rs : o.rowsets) {
                    if (rs.rows > L || rs.rows < 1 || rs.rows > kMaxRows) continue;
                    if (rs.draft && L < kDraftMinKeys) continue;
                    Cell base;
                    base.model = sh->model;
                    base.kv = kv;
                    base.heads = sh->heads;
                    base.nkv = sh->nkv;
                    base.keys = L;
                    base.rows = rs.rows;
                    base.rows_tag = rs.tag;
                    base.win = rs.draft ? kDraftWin : 0;
                    for (int r = L - rs.rows; r < L; ++r) base.pos.push_back(r);
                    if (!gridOk(base.heads, base.nkv, base.pos, base.win, kShip)) continue;
                    base.pa = plan(base.heads, base.nkv, base.pos, base.win, kShip);
                    base.kernel_a = kernelName(kv == "q8h", base.pa, kShip);
                    std::vector<int> ss = o.splits;
                    if (o.table) {  // B = the product rule's count (gfx1151 profile, flag auto, every product entry); 64 -> no cell
                        static const whirl::kernels::AttnKernels full = [] {
                            whirl::kernels::AttnKernels k;
                            static char tags[8];
                            for (int i = 0; i < 2; ++i) {
                                k.attn_ws4_sp[i] = &tags[4 * i];
                                for (int cg = 0; cg < 3; ++cg) k.attn_ws4g_sp[i][cg] = &tags[4 * i + 1 + cg];
                            }
                            return k;
                        }();
                        const int t = static_cast<int>(whirl::ops::ws4SplitCount(static_cast<std::uint32_t>(base.nkv), kv == "q8h", whirl::ops::kAttnLsplitAuto,
                                                                                 whirl::device::k_profile_gfx1151.attn_ws4_split_blocks, full));
                        // round 3: the measured limits of the step (ops::ws4SplitForStep): over a limit -> 64 -> no cell (skipped)
                        std::vector<std::int32_t> rp(base.pos.begin(), base.pos.end()), rb(rp.size(), 0);
                        whirl::ops::AttnDecodeProblem pr;
                        pr.n = static_cast<std::uint32_t>(rp.size());
                        pr.n_head = static_cast<std::uint32_t>(base.heads);
                        pr.n_head_kv = static_cast<std::uint32_t>(base.nkv);
                        pr.max_ctx = static_cast<std::uint32_t>(kMaxCtx);
                        pr.kv_q8 = kv == "q8h";
                        pr.rows_known = true;
                        pr.row_pos = std::span<const std::int32_t>(rp);
                        pr.row_base = std::span<const std::int32_t>(rb);
                        pr.win = base.win;
                        const int tl = static_cast<int>(whirl::ops::ws4SplitForStep(pr, static_cast<std::uint32_t>(t), whirl::device::k_profile_gfx1151.ws4_split_limits));
                        ss = tl != kShip ? std::vector<int>{tl} : std::vector<int>{};
                    }
                    if (o.aa && std::find(ss.begin(), ss.end(), kShip) == ss.end()) {
                        ss.push_back(kShip);
                        std::sort(ss.begin(), ss.end());
                    }
                    for (const int s : ss) {
                        if (!gridOk(base.heads, base.nkv, base.pos, base.win, s)) continue;
                        const bool same = samePartition(base.pos, base.win, s, kShip);
                        if (same && !o.aa) continue;
                        Cell c = base;
                        c.s = s;
                        c.same_partition = same;
                        c.pb = plan(c.heads, c.nkv, c.pos, c.win, s);
                        // B keeps A's kernel kind and CG (only the split constant differs); a grouping change -> no cell
                        if (c.pb.multi != c.pa.multi || c.pb.cg != c.pa.cg || c.pb.gz != c.pa.gz) continue;
                        // S == 64 (A/A control, Opts::aa): B = the bench instantiation <A>_ns64 (same split constant, the
                        // bench-only code path), so the control also covers what the bench entries themselves cost
                        c.kernel_b = s == kShip ? c.kernel_a + "_ns64" : kernelName(kv == "q8h", c.pb, s);
                        // bench-only entries exist for CG <= kMaxBenchCg; CG 3 only as a q8 product entry (_sp<S>)
                        if (c.pb.multi && c.pb.cg > kMaxBenchCg && c.kernel_b.find("_sp") == std::string::npos) continue;
                        c.name = c.model + "." + kv + "kv.L" + std::to_string(L) + ".rows" + rs.tag + ".s" + std::to_string(s) + "_vs_s" + std::to_string(kShip) +
                                 (o.table ? ".table" : "");
                        out.push_back(std::move(c));
                    }
                }
    }
    return out;
}

}  // namespace kt::ws4s
