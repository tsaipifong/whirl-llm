// whirl-kernel-test --bench dq4s (v0252-8sq5, PLAN V1-109): the cell table of the attn_dq4o split-count bench, HIP-free so
// it is unit-tested on the CPU (tests\kernels\unit\kt_bench_tests.cpp, target whirl-kt-bench-tests, which links whirl_ops).
// SPDX-License-Identifier: Apache-2.0
//
// One cell = one model shape (Ornith 16 / 2, Swift 24 / 4 heads, head dim 256) x keys L x a row set the engine launches
// (rows at positions L - R .. L - 1: R 1 = plain decode step, 2 = MTP verify of 1 draft per cycle (the 8060S Ornith
// default, spec::mtpDefaults), 3 = 2 drafts (the k4a10d2 relax arm), more on request; "d1" = the MTP draft row: one row
// with the default draft window, WHIRL_DRAFT_WINDOW 16384 from 65536 keys) x a split count S. The pair is S against
// S_ship = ops::dq4Splits(row_pos, win, 256) - the grid the fast plan launches today (8060S profile
// attn_dq4_dynamic_splits on, dq4_dyn on, dq4_min_keys 0; fd_splits 256 with q4 KV, loader.cpp fdSplitsOf).
// The key partition mirror (per / part) is kept in step with src\ops\attention.cpp dq4Per / dq4Part and pinned by the unit
// test: dq4sSplitsMirror == ops::dq4Splits over a sweep of positions and windows.
#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace kt::dq4s {

inline constexpr int kHd = 256;
inline constexpr int kFull = 256;      // fd_splits with q4 KV (src\model\loader.cpp fdSplitsOf)
inline constexpr int kMaxSplits = 128;  // the kernel's DQ_MAXS: dq_per never uses more active splits
inline constexpr int kMaxRows = 32;    // kernels::kMaxVerifyRows (AwGroups)
// qwen35::draftWindowArg(16384, 65536): W / 64 | (threshold / 1024) << 16 (src\model\qwen35\context.h)
inline constexpr int kDraftWin = (16384 / 64) | ((65536 / 1024) << 16);
inline constexpr int kDraftMinKeys = 65536;

struct Shape {
    const char* model;
    int heads, nkv;
};
inline constexpr Shape kShapes[] = {{"ornith", 16, 2}, {"swift", 24, 4}};

struct RowSet {
    std::string tag;  // "1", "2", "3", "d1"
    int rows = 1;
    bool draft = false;  // the MTP draft row (window on)
};

struct Opts {
    std::vector<std::string> models;  // subset of kShapes' names
    std::vector<int> keys;
    std::vector<RowSet> rowsets;
    std::vector<int> splits;
    bool aa = false;  // also cells whose partition equals S_ship's (S == S_ship = the A/A control)
    // v0252-8sq5 round 2 (WHIRL_KT_DQ4S_TABLE=1): B = the count of the gfx1151 profile table (ops::dq4TableSplits, auto) instead
    // of the S list, one cell per (keys, rows), name suffix ".table"; A stays S_ship = today's ops::dq4Splits
    bool table = false;
};
// defaults: both models, keys 8192 16384 32768 49152 65536 98304 131072, rows 1 2 3 d1, splits 8 16 32 64 128.
// Env (getenv_fn, so the test can inject): WHIRL_KT_DQ4S_MODELS (ornith,swift), WHIRL_KT_DQ4S_L, WHIRL_KT_DQ4S_ROWS
// (e.g. 1,2,d1), WHIRL_KT_DQ4S_S, WHIRL_KT_DQ4S_AA=1, WHIRL_KT_DQ4S_TABLE=1 (round 2: B = the gfx1151 table's count). Lists are comma / space separated; a bad entry -> err set, the
// returned opts unusable (the bench reports a FAIL and runs nothing).
Opts optsFromEnv(const std::function<const char*(const char*)>& getenv_fn, std::string* err);

// mirrors of ops dq4Per / dq4Part (src\ops\attention.cpp:27-43): keys per split, and a row's {window branch, per, splits
// carrying keys} for n_pos keys under grid n_split
int per(int n_keys, int n_split);
std::array<int, 3> part(int n_pos, int win, int n_split);
// the dq4Splits rule on the mirror (pinned against ops::dq4Splits by the unit test)
int splitsMirror(const std::vector<int>& row_pos, int win, int full);
// a grid the kernel takes for these rows: 1 <= s <= kFull, s >= 8 when a window is on (dq4Splits' floor keeps the window
// branch), every row's keyed splits fit in s, rows 1..kMaxRows
bool gridOk(const std::vector<int>& row_pos, int win, int s);
// every row has the same partition under s and t (bitwise the same partials; only empty splits differ)
bool samePartition(const std::vector<int>& row_pos, int win, int s, int t);

struct Cell {
    std::string name;   // <model>.q4kv.L<keys>.rows<tag>.s<S>_vs_s<S_ship>
    std::string model;
    int heads = 0, nkv = 0, keys = 0, rows = 0, win = 0;
    std::string rows_tag;
    std::vector<int> pos;  // row positions (L - R .. L - 1)
    int s = 0, s_ship = 0;
    bool same_partition = false;  // A/A-like (only with Opts::aa)
};
// cells in run order: model, keys, row set, S ascending. A draft row set is only made at keys >= kDraftMinKeys (below
// the threshold the kernel ignores the window: the same launch as rows 1). Cells whose grid fails gridOk are never made.
std::vector<Cell> cells(const Opts& o);

// geometric mean of paired ratios B / A with a 95 % interval (mean of ln r +- t(0.975, n - 1) sd / sqrt(n)); n < 2 ->
// lo = hi = g
struct GeoCi {
    double g = 0, lo = 0, hi = 0;
    int n = 0;
};
GeoCi geoCi(const std::vector<double>& ratios);
// t(0.975, df) (df >= 1; table to 30, then 1.96 + 2.4 / df)
double t975(int df);
// the judge line: "B/A geometric mean g, 95% interval [lo, hi] (B time +x % [lo %, hi %] vs A, N pairs)"
std::string geoLine(const GeoCi& c);

}  // namespace kt::dq4s
