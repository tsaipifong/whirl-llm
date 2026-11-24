// whirl-kernel-test --bench ws4s (v0252-8sws, PLAN V1-88 on gfx1151): the cell table of the attn_ws4 / attn_ws4g split-count
// bench, HIP-free so it is unit-tested on the CPU (tests\kernels\unit\kt_bench_tests.cpp, target whirl-kt-bench-tests, which
// links whirl_ops).
// SPDX-License-Identifier: Apache-2.0
//
// The gfx1151 balance (q8h KV) and precise (f16 KV) decode / verify attention is attn_ws4 (one row) / attn_ws4g<CG> (a group of
// consecutive verify rows), whose split rule is the COMPILE-TIME constant ws4::kNs = 64 (kernels\gfx1151\attn_ws4_index.h):
// the key ranges come from ws4::block_range(n_pos, split, kNs, win), gridDim.y only strides the partials. Another split count
// therefore needs another kernel instantiation: the bench-only entries <shipping name>_ns<S> (S in kBenchNs and the A/A
// control 64, CG 1 / 2 only; kernels\gfx1151\attn_ws4.hip BENCH_WS4_NS / BENCH_WS4G_NS), the same impl template with NS = S.
// The shipping entries are untouched (code-object comparison in progress\v0252-8sws.HANDOFF.md).
//
// One cell = model shape (Ornith 16 / 2, Swift 24 / 4 heads, head dim 256) x KV kind (q8h = balance, the _q8 kernels; f16 =
// the control, also the precise kernel) x keys L x row set (rows at L - R .. L - 1: "1" plain decode step, "2" / "3" MTP verify
// of 1 / 2 drafts, "d1" the MTP draft row with the default draft window from 65536 keys) x split count S. The pair is S
// against kShip = 64 (what ops::chooseAttnDecode launches today for the same rows: pinned by the unit test).
#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace kt::ws4s {

inline constexpr int kHd = 256;
inline constexpr int kShip = 64;          // ws4::kNs (static_assert in kt_ws4s.cpp)
inline constexpr int kMaxCtx = 262144;    // max_ctx handed to ops: the plan's grid.y = min(64, max_ctx / 64) = 64 for any context >= 4096
inline constexpr int kBenchNs[] = {8, 16, 32, 128};  // the bench-only kernel entries (<name>_ns<S>)
inline constexpr int kMaxBenchCg = 2;     // bench-only attn_ws4g entries exist for CG 1 and 2 (the q8 product _sp entries for CG 1..3)
inline constexpr int kMaxRows = 32;       // kernels::kMaxVerifyRows (AwGroups)
// qwen35::draftWindowArg(16384, 65536): W / 64 | (threshold / 1024) << 16 (src\model\qwen35\context.h), as kt_dq4s.h
inline constexpr int kDraftWin = (16384 / 64) | ((65536 / 1024) << 16);
inline constexpr int kDraftMinKeys = 65536;

struct Shape {
    const char* model;
    int heads, nkv;
};
inline constexpr Shape kShapes[] = {{"ornith", 16, 2}, {"swift", 24, 4}};
inline constexpr const char* kKvKinds[] = {"q8h", "f16"};

struct RowSet {
    std::string tag;  // "1", "2", "3", "d1"
    int rows = 1;
    bool draft = false;
};

struct Opts {
    std::vector<std::string> models, kvs;
    std::vector<int> keys;
    std::vector<RowSet> rowsets;
    std::vector<int> splits;
    // also S == 64: the A/A control, B = the bench entry <A>_ns64 (split constant 64 through the bench instantiation; for
    // attn_ws4 / attn_ws4_q8 its own helper instantiations, for attn_ws4g the shipping impl) - shows what the bench path costs
    bool aa = false;
    // v0252-8sws round 2 (WHIRL_KT_WS4S_TABLE=1): B = the count the product rule launches (ops::ws4SplitCount on the gfx1151
    // profile, flag auto, every product entry present) instead of the S list; one cell per (model, kv, keys, rows) where it is
    // not 64, name suffix ".table"; A stays today's 64
    bool table = false;
};
// defaults: models ornith, swift; kv q8h, f16; keys 8192 16384 32768 49152 65536 98304 131072; rows 1 2 3 d1; splits 8 16 32 128.
// Env (getenv_fn, so the test can inject): WHIRL_KT_WS4S_MODELS, WHIRL_KT_WS4S_KV (q8h,f16), WHIRL_KT_WS4S_L, WHIRL_KT_WS4S_ROWS
// (e.g. 1,2,d1; 1..32), WHIRL_KT_WS4S_S (only 8 16 32 64 128: the counts with a kernel entry), WHIRL_KT_WS4S_AA=1. Lists are
// comma / space separated; a bad entry -> err set, the returned opts unusable (the bench reports a FAIL and runs nothing).
Opts optsFromEnv(const std::function<const char*(const char*)>& getenv_fn, std::string* err);

// the launch ops::chooseAttnDecode makes on the gfx1151 ws4 path for these rows (one sequence, consecutive positions known
// to the host), with the split constant ns in place of ws4::kNs (ns == 64: today's plan, pinned against ops by the unit test)
struct Plan {
    bool multi = false;  // attn_ws4g (a group wider than one row), else attn_ws4
    int cg = 0;          // attn_ws4g column tiles (1..3), 0 for attn_ws4
    unsigned gx = 0, gy = 0, gz = 0, block = 0;
    std::vector<std::array<int, 2>> groups;  // {first, count}
};
Plan plan(int heads, int nkv, const std::vector<int>& pos, int win, int ns, int max_ctx = kMaxCtx);
// kernel entry name: attn_ws4[_q8] / attn_ws4g<cg>[_q8], + "_ns<S>" when ns != 64 (the A/A control entry <name>_ns64 is named
// by cells())
std::string kernelName(bool q8, const Plan& p, int ns);
// a split count with a kernel entry (64 or kBenchNs)
bool nsBuilt(int ns);
// a launch the bench may make: nsBuilt(ns), rows 1..kMaxRows at positions >= 0, ns > 1 when a window is on, and the ns ranges of
// every row's n_pos cover [0, n_pos) with at most gy splits (cells() also drops a CG the entries lack: CG 3 only for the q8
// product entries _sp8 / _sp16)
bool gridOk(int heads, int nkv, const std::vector<int>& pos, int win, int ns);
// every row has the same key ranges under s and t (bitwise the same partials; only empty splits differ)
bool samePartition(const std::vector<int>& pos, int win, int s, int t);

struct Cell {
    std::string name;  // <model>.<kv>kv.L<keys>.rows<tag>.s<S>_vs_s64
    std::string model, kv;
    int heads = 0, nkv = 0, keys = 0, rows = 0, win = 0;
    std::string rows_tag;
    std::vector<int> pos;
    int s = 0, s_ship = kShip;
    bool same_partition = false;
    Plan pa, pb;                   // A = plan(.., 64) (today), B = plan(.., S)
    std::string kernel_a, kernel_b;
};
// cells in run order: model, kv, keys, row set, S ascending. A draft row set only at keys >= kDraftMinKeys (below it the kernel
// ignores the window: the rows-1 launch). Cells whose grid fails gridOk are never made; S == 64 only with Opts::aa.
std::vector<Cell> cells(const Opts& o);

}  // namespace kt::ws4s
