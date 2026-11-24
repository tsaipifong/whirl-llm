// whirl-kernel-test --bench dq4s: the HIP-free cell table (kt_dq4s.h). v0252-8sq5, PLAN V1-109.
// SPDX-License-Identifier: Apache-2.0

#include "kt_dq4s.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "whirl/ops/attention.h"  // ops::dq4Splits: S_ship (the grid the fast plan launches today)

namespace kt::dq4s {

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

int sShip(const std::vector<int>& pos, int win) {
    std::vector<std::int32_t> p(pos.begin(), pos.end());
    return static_cast<int>(whirl::ops::dq4Splits(std::span<const std::int32_t>(p.data(), p.size()), win, kFull));
}

}  // namespace

Opts optsFromEnv(const std::function<const char*(const char*)>& getenv_fn, std::string* err) {
    Opts o;
    o.models = {"ornith", "swift"};
    o.keys = {8192, 16384, 32768, 49152, 65536, 98304, 131072};
    o.rowsets = {{"1", 1, false}, {"2", 2, false}, {"3", 3, false}, {"d1", 1, true}};
    o.splits = {8, 16, 32, 64, 128};
    std::string e;
    if (const char* v = getenv_fn("WHIRL_KT_DQ4S_MODELS")) {
        o.models.clear();
        for (const std::string& t : splitList(v)) {
            const bool known = std::any_of(std::begin(kShapes), std::end(kShapes), [&](const Shape& s) { return t == s.model; });
            if (!known) e += "WHIRL_KT_DQ4S_MODELS: unknown model '" + t + "' (ornith, swift); ";
            else o.models.push_back(t);
        }
    }
    if (const char* v = getenv_fn("WHIRL_KT_DQ4S_L")) {
        o.keys.clear();
        for (const std::string& t : splitList(v)) {
            int x = 0;
            if (!parseInt(t, 1, 262144, x)) e += "WHIRL_KT_DQ4S_L: bad key count '" + t + "' (1..262144); ";
            else o.keys.push_back(x);
        }
    }
    if (const char* v = getenv_fn("WHIRL_KT_DQ4S_ROWS")) {
        o.rowsets.clear();
        for (const std::string& t : splitList(v)) {
            int x = 0;
            if (t == "d1") o.rowsets.push_back({"d1", 1, true});
            else if (parseInt(t, 1, kMaxRows, x)) o.rowsets.push_back({t, x, false});
            else e += "WHIRL_KT_DQ4S_ROWS: bad row set '" + t + "' (1..32 or d1); ";
        }
    }
    if (const char* v = getenv_fn("WHIRL_KT_DQ4S_S")) {
        o.splits.clear();
        for (const std::string& t : splitList(v)) {
            int x = 0;
            if (!parseInt(t, 1, kFull, x)) e += "WHIRL_KT_DQ4S_S: bad split count '" + t + "' (1..256); ";
            else o.splits.push_back(x);
        }
        std::sort(o.splits.begin(), o.splits.end());
        o.splits.erase(std::unique(o.splits.begin(), o.splits.end()), o.splits.end());
    }
    if (const char* v = getenv_fn("WHIRL_KT_DQ4S_AA")) o.aa = std::string(v) != "" && std::string(v) != "0";
    if (const char* v = getenv_fn("WHIRL_KT_DQ4S_TABLE")) o.table = std::string(v) != "" && std::string(v) != "0";
    if (o.models.empty() || o.keys.empty() || o.rowsets.empty() || o.splits.empty()) e += "an empty list; ";
    if (err) *err = e;
    return o;
}

// src/ops/attention.cpp dq4Per (kernels/gfx1151/attn.hip dq_per, DQ_NW 8, DQ_MAXS 128)
int per(int n_keys, int n_split) {
    const int ns = std::min(n_split, kMaxSplits);
    return std::max(8 * 32, (n_keys + ns * 64 - 1) / (ns * 64) * 64);
}

// src/ops/attention.cpp dq4Part
std::array<int, 3> part(int n_pos, int win, int n_split) {
    if (win != 0 && n_split > 1) {
        const int w = (win & 0xffff) * 64, min_pos = (win >> 16) * 1024;
        const int w0 = std::max(256, (n_pos - w) & ~63);
        if (n_pos >= min_pos && w0 > 256) {
            const int p = per(n_pos - w0, n_split - 1);
            return {1, p, 1 + (n_pos - w0 + p - 1) / p};
        }
    }
    const int p = per(n_pos, n_split);
    return {0, p, (n_pos + p - 1) / p};
}

int splitsMirror(const std::vector<int>& row_pos, int win, int full) {
    for (int s = 8; s < full; s *= 2) {
        bool same = true;
        for (const int p : row_pos) {
            const auto a = part(p + 1, win, s), b = part(p + 1, win, full);
            if (a[0] != b[0] || a[1] != b[1] || a[2] > s) {
                same = false;
                break;
            }
        }
        if (same) return s;
    }
    return full;
}

// v0252-8sq5 round 2: the product check (ops::dq4GridOk, the same one ops::dq4TableSplits applies) + the AwGroups row bound
bool gridOk(const std::vector<int>& row_pos, int win, int s) {
    if (s < 1 || s > kFull || row_pos.empty() || static_cast<int>(row_pos.size()) > kMaxRows) return false;
    std::vector<std::int32_t> p(row_pos.begin(), row_pos.end());
    return whirl::ops::dq4GridOk(std::span<const std::int32_t>(p), win, static_cast<std::uint32_t>(s), kFull);
}

bool samePartition(const std::vector<int>& row_pos, int win, int s, int t) {
    for (const int p : row_pos) {
        const auto a = part(p + 1, win, s), b = part(p + 1, win, t);
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
        for (const int L : keys)
            for (const RowSet& rs : o.rowsets) {
                if (rs.rows > L || rs.rows < 1 || rs.rows > kMaxRows) continue;
                if (rs.draft && L < kDraftMinKeys) continue;
                Cell base;
                base.model = sh->model;
                base.heads = sh->heads;
                base.nkv = sh->nkv;
                base.keys = L;
                base.rows = rs.rows;
                base.rows_tag = rs.tag;
                base.win = rs.draft ? kDraftWin : 0;
                for (int r = L - rs.rows; r < L; ++r) base.pos.push_back(r);
                base.s_ship = sShip(base.pos, base.win);
                if (!gridOk(base.pos, base.win, base.s_ship)) continue;
                std::vector<int> ss = o.splits;
                if (o.table) {  // B = the gfx1151 profile table's count (0 = the table does not decide this row set: no cell)
                    std::vector<std::int32_t> p(base.pos.begin(), base.pos.end()), z(p.size(), 0);
                    const int t = static_cast<int>(whirl::ops::dq4TableSplits(std::span<const std::int32_t>(p), std::span<const std::int32_t>(z), base.win,
                                                                               kFull, whirl::ops::kDq4SplitsAuto,
                                                                               whirl::device::k_profile_gfx1151.dq4_split_table));
                    ss = t != 0 ? std::vector<int>{t} : std::vector<int>{};
                }
                for (const int s : ss) {
                    if (!gridOk(base.pos, base.win, s)) continue;
                    const bool same = samePartition(base.pos, base.win, s, base.s_ship);
                    if (same && !o.aa) continue;
                    Cell c = base;
                    c.s = s;
                    c.same_partition = same;
                    c.name = c.model + ".q4kv.L" + std::to_string(L) + ".rows" + rs.tag + ".s" + std::to_string(s) + "_vs_s" + std::to_string(c.s_ship) +
                             (o.table ? ".table" : "");
                    out.push_back(std::move(c));
                }
            }
    }
    return out;
}

double t975(int df) {
    static const double t[] = {0,      12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
                               2.120,  2.110,  2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042};
    if (df < 1) return t[1];
    if (df <= 30) return t[df];
    return 1.96 + 2.4 / df;
}

GeoCi geoCi(const std::vector<double>& ratios) {
    GeoCi c;
    std::vector<double> l;
    for (const double r : ratios)
        if (r > 0 && std::isfinite(r)) l.push_back(std::log(r));
    c.n = static_cast<int>(l.size());
    if (c.n == 0) return c;
    double m = 0;
    for (const double x : l) m += x;
    m /= c.n;
    c.g = c.lo = c.hi = std::exp(m);
    if (c.n < 2) return c;
    double ss = 0;
    for (const double x : l) ss += (x - m) * (x - m);
    const double h = t975(c.n - 1) * std::sqrt(ss / (c.n - 1)) / std::sqrt(static_cast<double>(c.n));
    c.lo = std::exp(m - h);
    c.hi = std::exp(m + h);
    return c;
}

std::string geoLine(const GeoCi& c) {
    char b[256];
    std::snprintf(b, sizeof(b), "B/A geometric mean %.4f, 95%% interval [%.4f, %.4f] (B time %+.1f %% [%+.1f %%, %+.1f %%] vs A, %d pairs)", c.g, c.lo, c.hi,
                  100.0 * (c.g - 1), 100.0 * (c.lo - 1), 100.0 * (c.hi - 1), c.n);
    return b;
}

}  // namespace kt::dq4s
