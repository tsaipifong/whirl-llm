// arch025-N2a: numerics::plan decides every execution flag (Plan::flags) and bitsKey().
// SPDX-License-Identifier: Apache-2.0
//
// 1. Plan table: the S0b-2 GPU dumps (12 cli cells: 2 cards x 2 product models x 3 modes, and 56 env cells: one
//    WHIRL_* override on Swift balance, both cards), field by field: every Flags field with a dump key, the numerics
//    log line, q8dec, lossy_override and the override list. Caps from the two device::Profile rows plus the kernel
//    presence of each card's code object (the fakes below); model facts from the two product models.
// 2. bitsKey reflection: the flag table is complete (layout size), every field and item is in the key, notes are not.
// 3. Scope: every WHIRL_DUMP_EXEC key is a plan flag, the Request / plan itself, or listed with the module that
//    decides it (KV: N3; tune results and per-forward state: need the GPU dump, N2b compares them with S0b-2).
// 4. Front ends: the CLI and server knob rules (cli main.cpp:1030-1048, server_main.cpp:1040-1120), table cells.
// The "every bit-affecting env changes bitsKey and is kept by the tier fingerprint" test is in tests/cache (it needs
// cache::fingerprintEnvKept); the EnvSnapshot name cross-check is in tests/model.

#include "whirl/device_profile.h"
#include "whirl/numerics.h"

#include <cstdio>
#include <format>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_p = 0, g_f = 0;
#define NCHECK(cond, ...)                                                                                            \
    do {                                                                                                             \
        if (cond) ++g_p;                                                                                             \
        else {                                                                                                       \
            ++g_f;                                                                                                   \
            std::printf("FAIL %s:%d: %s %s\n", __FILE__, __LINE__, #cond, std::format(__VA_ARGS__).c_str());       \
        }                                                                                                            \
    } while (0)

#include "numerics_plan_s0b2.inc"

namespace nu = whirl::numerics;
namespace dev = whirl::device;

// Kernel presence of each card's code object (kernels::Caps / the kernel table; what loadOrTune reads from m.k).
// Checked against the S0b-2 log lines: the 8060S has no fp8 WMMA (fp8 / moefp8 "no fp8 WMMA on this GPU") and no
// gdn_wprep (Q4_RELAXED / GDN_STRIX=0 turn gdnwmma off there); both have q4 KV (kvq4 enabled in fast on the R9700).
struct CardFake {
    const char* name;
    const dev::Profile& prof;
    bool k_gemm8, k_gemm8_moe, k_gemm8t, k_gdn_wmma, k_kv_q8v, k_kv_q8h, k_kv_q4, k_gdn_sx, k_spec_sample, k_gdn_v0;
};
const CardFake k_cards[] = {
    {"r9700", dev::k_profile_gfx1201, true, true, true, true, true, true, true, false, true, true},
    {"8060s", dev::k_profile_gfx1151, false, false, false, false, false, true, true, true, true, true},
};
const CardFake& card(std::string_view n) {
    for (const CardFake& c : k_cards)
        if (n == c.name) return c;
    throw std::logic_error("card");
}

// Swift-1.5 27B MXFP4-A (dense) and Ornith-1.5-35B-A3B MXFP4 (MoE, MXFP4 experts): both have MXFP4 tensors, an MTP
// head and the strix DeltaNet shape (d_state 128, head V 128, V/K head ratio 3 and 2).
nu::Target target(const CardFake& c, std::string_view model, bool kv_explicit) {
    nu::Target t;
    t.gfx1151 = c.prof.numerics_8060s;
    t.has_mxfp4 = true;
    t.moe = model == "moe";
    t.moe_mxfp4 = t.moe;
    t.k_gemm8 = c.k_gemm8;
    t.k_gemm8_moe = c.k_gemm8_moe;
    t.k_gdn_wmma = c.k_gdn_wmma;
    t.k_kv_q8v = c.k_kv_q8v;
    t.k_kv_q8h = c.k_kv_q8h;
    t.mtp = true;
    t.k_spec_sample = c.k_spec_sample;
    t.k_kv_q4 = c.k_kv_q4;
    t.kv_explicit = kv_explicit;
    t.fast_kv_q8h = t.gfx1151 && t.k_kv_q8h;  // tune.cpp:936
    t.strix_profile = c.prof.gdn_strix;
    t.k_gdn_sx = c.k_gdn_sx;
    t.gdn_strix_shape = true;
    t.k_gemm8t = c.k_gemm8t;
    t.k_gdn_v0 = c.k_gdn_v0;
    return t;
}

std::vector<std::pair<std::string, std::string>> splitEnv(std::string_view s) {
    std::vector<std::pair<std::string, std::string>> o;
    std::size_t p = 0;
    while (p < s.size()) {
        std::size_t e = s.find(';', p);
        if (e == std::string_view::npos) e = s.size();
        const std::string_view it = s.substr(p, e - p);
        const std::size_t eq = it.find('=');
        o.emplace_back(std::string(it.substr(0, eq)), std::string(it.substr(eq + 1)));
        p = e + 1;
    }
    return o;
}

std::map<std::string, std::string> splitKv(std::string_view s) {
    std::map<std::string, std::string> o;
    std::istringstream in{std::string(s)};
    std::string l;
    while (std::getline(in, l)) {
        const std::size_t eq = l.find('=');
        if (eq != std::string::npos) o[l.substr(0, eq)] = l.substr(eq + 1);
    }
    return o;
}

// env names a cell sets that no plan field reads (checkGemvBitwise only: not a flag, prints a warning)
const std::set<std::string> k_not_plan_env = {"GEMV_BITWISE_FLOAT"};

void testPlanTable() {
    std::size_t n_cells = 0, n_fields = 0;
    for (const S0b2Cell& c : k_s0b2_cells) {
        ++n_cells;
        nu::Request r = nu::parseMode(c.mode, "command line");
        r.load_overrides = true;
        bool kv_explicit = false;
        for (const auto& [n, v] : splitEnv(c.env)) {
            if (k_not_plan_env.count(n)) continue;
            nu::Raw* f = r.env.find(n);
            NCHECK(f != nullptr, "{} {}: env {} has no Request field", c.card, c.cell, n);
            if (f) *f = v;
            // loader.cpp:524 kvModeOf: a format name -> explicit (Target::kv_explicit, tune.cpp:935)
            if (n == "KV" && v != "auto") kv_explicit = true;
        }
        const nu::Plan p = nu::plan(r, target(card(c.card), c.model, kv_explicit));
        const auto want = splitKv(c.kv);
        std::size_t n_over = 0;
        for (const auto& [k, v] : want) {
            ++n_fields;
            std::string got;
            bool known = true;
            if (k == "num_plan") got = nu::logLine(p);
            else if (k == "num_plan.q8dec") got = p.q8dec ? "1" : "0";
            else if (k == "num_plan.q8dec_default") got = p.q8dec_default ? "1" : "0";
            else if (k == "num_plan.lossy_override") got = p.lossy_override ? "1" : "0";
            else if (k.rfind("num_plan.override[", 0) == 0) {
                const std::size_t i = std::stoul(k.substr(18));
                got = i < p.overrides.size() ? p.overrides[i] : "<none>";
                ++n_over;
            } else {
                known = false;
                for (const nu::FlagInfo& fi : nu::flagTable())
                    if (k == fi.dump_key) {
                        got = nu::flagValue(p.flags, fi);
                        known = true;
                    }
            }
            NCHECK(known, "{} {}: key {} not decided by the plan", c.card, c.cell, k);
            NCHECK(got == v, "{} {}: {} = '{}', S0b-2 dump '{}'", c.card, c.cell, k, got, v);
        }
        NCHECK(p.overrides.size() == n_over, "{} {}: {} overrides, dump {}", c.card, c.cell, p.overrides.size(), n_over);
        // every flag with a dump key is in every cell (nothing compared by omission)
        for (const nu::FlagInfo& fi : nu::flagTable())
            if (*fi.dump_key) NCHECK(want.count(fi.dump_key) == 1, "{} {}: dump has no {}", c.card, c.cell, fi.dump_key);
    }
    NCHECK(n_cells == 68, "cells {}", n_cells);
    std::printf("  numerics plan table: %zu S0b-2 cells, %zu fields compared\n", n_cells, n_fields);
}

// arch025-N2b: the front ends call numerics::applyFront on the plan loadOrTune made (Front::none), after the load.
// It must give exactly what plan() gives with the same front and runtime values in the Request (the N2a table cells
// above use that form), for every card x model x mode and front, over runtime values drawn from each knob's parse
// classes (unset, "", "0", "1", numbers, prefix numbers, junk). gdn_v0 with and without the kernel.
void testApplyFrontEqualsPlan() {
    static const char* const vals[] = {nullptr, "", "0", "1", "2", "7", "16", "32", "100", "4096", "65536", "8192x", "99", "abc", "-1", "4294967296"};
    constexpr std::size_t n_vals = sizeof(vals) / sizeof(vals[0]);
    std::uint64_t rng = 0x9e3779b97f4a7c15ull;
    auto next = [&rng] {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };
    std::size_t n = 0;
    for (const CardFake& c : k_cards)
        for (const char* model : {"swift", "moe"})
            for (const char* mode : {"precise", "balance", "fast"})
                for (nu::Front fr : {nu::Front::none, nu::Front::cli, nu::Front::server})
                    for (int k_v0 = 0; k_v0 < 2; ++k_v0)
                        for (int it = 0; it < 400; ++it) {
                            nu::Request r = nu::parseMode(mode, "command line");
                            nu::Target t = target(c, model, false);
                            t.k_gdn_v0 = k_v0 != 0;
                            nu::RuntimeValues rv;
                            for (const auto& e : nu::RuntimeValues::names()) {
                                const char* v = vals[(it == 0) ? 0 : next() % n_vals];
                                if (v != nullptr) rv.*e.field = std::string(v);
                            }
                            const nu::Plan base = nu::plan(r, t);
                            nu::Plan wired = base;
                            nu::applyFront(wired, fr, rv, t.k_gdn_v0);
                            r.front = fr;
                            r.runtime = rv;
                            const nu::Plan ref = nu::plan(r, t);
                            NCHECK(wired.flags == ref.flags, "applyFront {} {} {} front {} v0 {} #{}", c.name, model, mode, static_cast<int>(fr),
                                   k_v0, it);
                            NCHECK(wired.bitsKey() == ref.bitsKey() && nu::logLine(wired) == nu::logLine(ref), "applyFront key {} {} {} #{}",
                                   c.name, model, mode, it);
                            ++n;
                        }
    std::printf("  numerics applyFront == plan(front): %zu cases\n", n);
}

// ---- bitsKey reflection

constexpr std::size_t layoutSize() {
    std::size_t off = 0, al = 1;
    auto add = [&](std::size_t sz, std::size_t a) {
        off = (off + a - 1) / a * a + sz;
        al = a > al ? a : al;
    };
#define WHIRL_NF_SZ_B add(sizeof(bool), alignof(bool));
#define WHIRL_NF_SZ_U add(sizeof(std::uint32_t), alignof(std::uint32_t));
#define WHIRL_NF_SZ_F add(sizeof(float), alignof(float));
#define WHIRL_NF_SZ(k, n, d, key) WHIRL_NF_SZ_##k
    WHIRL_NUMERICS_FLAGS(WHIRL_NF_SZ)
#undef WHIRL_NF_SZ
#undef WHIRL_NF_SZ_B
#undef WHIRL_NF_SZ_U
#undef WHIRL_NF_SZ_F
    return (off + al - 1) / al * al;
}
// a member added to Flags outside WHIRL_NUMERICS_FLAGS (so missing from flagTable and bitsKey) changes the size
static_assert(sizeof(nu::Flags) == layoutSize(), "Flags has a member outside WHIRL_NUMERICS_FLAGS");
static_assert(sizeof(nu::EnvValues) == (nu::EnvValues::n_fields + 13) * sizeof(nu::Raw), "EnvValues field not in names()");
static_assert(sizeof(nu::RuntimeValues) == nu::RuntimeValues::n_fields * sizeof(nu::Raw), "RuntimeValues field not in names()");

void testBitsKeyReflection() {
    const auto& tab = nu::flagTable();
    std::set<std::string> names, keys;
    for (const nu::FlagInfo& fi : tab) {
        NCHECK(names.insert(fi.name).second, "duplicate flag {}", fi.name);
        if (*fi.dump_key) NCHECK(keys.insert(fi.dump_key).second, "duplicate dump key {}", fi.dump_key);
    }
    // each field alone changes the key, and the key shows it as name=value
    const nu::Plan base = nu::plan(nu::parseMode("balance", "t"), target(k_cards[0], "swift", false));
    const std::string k0 = base.bitsKey();
    for (const nu::FlagInfo& fi : tab) {
        nu::Plan p = base;
        switch (fi.kind) {
            case nu::FlagKind::b: p.flags.*fi.b = !(p.flags.*fi.b); break;
            case nu::FlagKind::u: p.flags.*fi.u += 1; break;
            case nu::FlagKind::f: p.flags.*fi.f += 0.25f; break;
        }
        NCHECK(!(p.flags == base.flags), "{}: operator== misses it", fi.name);
        const std::string k = p.bitsKey();
        NCHECK(k != k0, "{} not in bitsKey", fi.name);
        NCHECK(k.find(std::format("{}={};", fi.name, nu::flagValue(p.flags, fi))) != std::string::npos, "{}", fi.name);
    }
    for (std::size_t i = 0; i < nu::n_items; ++i) {
        nu::Plan p = base;
        p.items[i].enabled = !p.items[i].enabled;
        NCHECK(p.bitsKey() != k0, "item {} not in bitsKey", nu::itemName(static_cast<nu::Item>(i)));
    }
    {
        nu::Plan p = base;
        p.q8dec = !p.q8dec;
        NCHECK(p.bitsKey() != k0, "q8dec");
        p = base;
        p.mode = nu::Mode::fast;
        NCHECK(p.bitsKey() != k0, "mode");
    }
    {
        // descriptions are not bits
        nu::Plan p = base;
        p.items[0].note = "other";
        p.overrides.push_back("x");
        p.source = "elsewhere";
        p.lossy_override = !p.lossy_override;
        NCHECK(p.bitsKey() == k0, "notes changed bitsKey");
    }
    // default Flags == the Model member initializers recorded in the dump (S0b-2 cli precise cells: no item on, so the
    // fields the mode does not touch keep their initializers); spot-check a few against model.h by value
    const nu::Flags d{};
    NCHECK(d.use_graph && !d.no_fuse && d.head_chunk == 248320 && d.gemv_max == 16 && d.fp8_mask == 7 && d.relax_k == 4 &&
               d.draft_window == 16384 && d.draft_window_min == 65536 && d.gv_group && d.gv_nmax == 0 && d.gdn_strix_min == 256,
           "defaults");
}

// ---- scope: every dump key has an owner
void testFlagScope() {
    // decided outside numerics::plan, with the module that decides it (and why it is not a Plan flag)
    static const std::vector<std::pair<std::string, std::string>> elsewhere = {
        {"kv_choice.", "KV format: numerics::chooseKv (N3)"},
        {"kv_q8", "KV: chooseKv (N3)"},
        {"kv_rot", "KV: chooseKv (N3)"},
        {"kv_kf16", "KV: chooseKv (N3)"},
        {"kv_q4", "KV: chooseKv (N3)"},
        {"kv_mode", "KV: WHIRL_KV (N3)"},
        {"fd_splits", "KV: resolveFdSplits (N3)"},
        {"num_req.", "the Request itself"},
        {"tune_mask", "needs GPU dump: tune result"},
        {"tune_cold", "needs GPU dump: tune result"},
        {"tune", "needs GPU dump: tune table hash"},
        {"gemv_r", "needs GPU dump: tune rows (applyGemvEnv: WHIRL_GEMV_R)"},
        {"gemv_w[", "needs GPU dump: tune rows (applyGemvEnv: WHIRL_GEMV_W)"},
        {"gemv_w_head", "needs GPU dump: tune rows (applyGemvEnv: WHIRL_GEMV_WH)"},
        {"mm_class", "per forward"},
        {"out_h16", "per forward"},
        {"small_max", "per forward"},
        {"head_phase", "per forward"},
        {"decode_graph", "per forward (captured graph)"},
        {"gdn_replay", "per forward"},
        {"mtp_batch", "per forward"},
        {"mrope_on", "per forward (vision)"},
        {"rpos_rows", "per forward (vision)"},
        {"keep_hidden", "per forward"},
        {"all_logits", "per forward"},
        {"max_ctx", "size"},
        {"max_batch", "size"},
        {"arch", "device"},
        {"allocations", "size"},
        {"alloc_total", "size"},
        {"type_bytes[", "model file"},
        {"mx_fold_lossy", "model file (weight folding count)"},
        {"gc_w_bytes", "size"},
        {"gc_u_bytes", "size"},
        {"gc_m_bytes", "size"},
        {"x16_bytes", "size"},
        {"dq_rows", "size"},
        {"pool_pages", "size"},
        {"seq_pages", "size"},
        {"snap_sets", "size"},
        {"seqs", "size"},
        {"snap_rows", "size"},
        {"ff_scratch", "size"},
        {"mtp", "model file (MTP head)"},
        {"draft_vocab", "speculative drafting (WHIRL_DRAFT_VOCAB, front end)"},
        {"draft_p_min", "speculative drafting"},
        {"draft_n_min", "speculative drafting"},
        {"draft_head", "speculative drafting"},
        {"draft_d2", "speculative drafting"},
        {"draft_rows", "speculative drafting"},
        {"draft_map", "speculative drafting"},
        {"dflash", "speculative drafting (--draft dflash)"},
        {"mtp_skipped", "speculative drafting"},
    };
    std::set<std::string> flag_keys;
    for (const nu::FlagInfo& fi : nu::flagTable())
        if (*fi.dump_key) flag_keys.insert(fi.dump_key);
    std::size_t n = 0, n_gpu = 0;
    for (const char* const* k = k_s0b2_dump_keys; *k; ++k) {
        ++n;
        const std::string key = *k;
        if (flag_keys.count(key) || key.rfind("num_plan", 0) == 0) continue;
        bool owned = false;
        for (const auto& [pre, why] : elsewhere)
            if (key == pre || ((pre.back() == '.' || pre.back() == '[') && key.rfind(pre, 0) == 0)) {
                owned = true;
                if (why.rfind("needs GPU dump", 0) == 0) ++n_gpu;
            }
        NCHECK(owned, "dump key {} is neither a plan flag nor listed with its owner", key);
    }
    // every flag with a dump key is a key of the dump (no flag compared against nothing)
    for (const std::string& fk : flag_keys) {
        bool found = false;
        for (const char* const* k = k_s0b2_dump_keys; *k; ++k) found = found || fk == *k;
        NCHECK(found, "flag dump key {} not in the S0b-2 dump", fk);
    }
    std::printf("  numerics flag scope: %zu dump keys, %zu plan flags, %zu need the GPU dump (N2b)\n", n, flag_keys.size(), n_gpu);
}

// ---- front ends (table cells from the code: cli main.cpp:1030-1048 / :1340, server_main.cpp:1040-1120)
void testFrontEnds() {
    const nu::Target t = target(k_cards[0], "swift", false);
    auto run = [&](nu::Front fr, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode("balance", "t");
        r.front = fr;
        for (const auto& [n, v] : vals)
            for (const auto& e : nu::RuntimeValues::names())
                if (std::string_view(n) == e.name) r.runtime.*e.field = std::string(v);
        return nu::plan(r, t).flags;
    };
    using F = nu::Front;
    const nu::Flags d = nu::plan(nu::parseMode("balance", "t"), t).flags;
    // none: runtime values ignored
    NCHECK(run(F::none, {{"NO_FUSE", "1"}, {"GDN_SEQ", "1"}}) == d, "front none");
    // presence knobs: any value, "0" included
    for (F fr : {F::cli, F::server}) {
        const nu::Flags a = run(fr, {{"NO_FUSE", "0"}, {"FLOAT_GEMV", "0"}, {"NAIVE_ATTN", "0"}, {"GDN_SEQ", "0"}, {"GDN_V0", "0"}});
        NCHECK(a.no_fuse && a.float_gemv && a.naive_attn && !a.gdn_chunked && a.gdn_v0, "presence {}", static_cast<int>(fr));
        // "== 0" knobs: only "0" turns them off
        const nu::Flags b = run(fr, {{"ATTN_WIDE", "0"}, {"ATTN_DQ4", "1"}, {"DQ4_DYN", ""}, {"ATTN_WS4", "0"}});
        NCHECK(!b.attn_wide && b.attn_dq4 && b.dq4_dyn && !b.attn_ws4, "==0 {}", static_cast<int>(fr));
    }
    // GDN_V0 needs the kernel
    {
        nu::Request r = nu::parseMode("balance", "t");
        r.front = F::cli;
        r.runtime.gdn_v0 = "1";
        nu::Target t2 = t;
        t2.k_gdn_v0 = false;
        NCHECK(!nu::plan(r, t2).flags.gdn_v0, "gdn_v0 without kernel");
    }
    // CLI: stoul prefix parse, a rejected value leaves the default; GEMV_MAX clamped to 1..16
    NCHECK(run(F::cli, {{"DBG", "12abc"}}).dbg_flags == 12, "cli DBG prefix");
    NCHECK(run(F::cli, {{"DBG", "x"}}).dbg_flags == 0, "cli DBG bad");
    NCHECK(run(F::cli, {{"GEMV_MAX", "0"}}).gemv_max == 1 && run(F::cli, {{"GEMV_MAX", "99"}}).gemv_max == 16 &&
               run(F::cli, {{"GEMV_MAX", "8"}}).gemv_max == 8,
           "cli GEMV_MAX clamp");
    NCHECK(!run(F::cli, {{"NO_GRAPH", "0"}}).use_graph, "cli NO_GRAPH presence");
    NCHECK(run(F::cli, {{"DRAFT_WINDOW", "4096"}, {"DRAFT_WINDOW_MIN", "8192x"}}).draft_window == 4096 &&
               run(F::cli, {{"DRAFT_WINDOW_MIN", "8192x"}}).draft_window_min == 8192,
           "cli DRAFT_WINDOW");
    NCHECK(run(F::cli, {{"MOE_BN", "64"}}).moe_bn_force == 64 && run(F::cli, {{"MOE_BN", "z"}}).moe_bn_force == 0, "cli MOE_BN");
    NCHECK(!run(F::cli, {{"GV_GROUP", "0"}}).gv_group && run(F::cli, {{"GV_GROUP", ""}}).gv_group && run(F::cli, {{"GV_NMAX", "5"}}).gv_nmax == 5,
           "cli GV");
    // the CLI does not apply the server-only knobs
    NCHECK(run(F::cli, {{"WIDE_VERIFY", "0"}, {"HEAD_CHUNK", "64"}}) == d, "cli ignores server knobs");
    // server: graph never replayed; strict number parse; server-only knobs; CLI-only knobs ignored
    NCHECK(!run(F::server, {}).use_graph, "server use_graph");
    NCHECK(run(F::server, {{"DRAFT_WINDOW", "4096"}}).draft_window == 4096 && run(F::server, {{"DRAFT_WINDOW", "4096x"}}).draft_window == 16384,
           "server strict DRAFT_WINDOW");
    NCHECK(run(F::server, {{"MOE_BN", "64x"}}).moe_bn_force == 64 && run(F::server, {{"MOE_BN", "z"}}).moe_bn_force == 0, "server MOE_BN strtoul");
    NCHECK(!run(F::server, {{"WIDE_VERIFY", "0"}}).wide_verify && run(F::server, {{"WIDE_VERIFY", "1"}}).wide_verify, "server WIDE_VERIFY");
    NCHECK(run(F::server, {{"HEAD_CHUNK", "100"}}).head_chunk == 96 && run(F::server, {{"HEAD_CHUNK", "1"}}).head_chunk == 32 &&
               run(F::server, {{"HEAD_CHUNK", "bad"}}).head_chunk == 248320,
           "server HEAD_CHUNK");
    {
        nu::Flags s = run(F::server, {{"DBG", "3"}, {"GEMV_MAX", "4"}, {"NO_GRAPH", "1"}, {"GV_GROUP", "0"}, {"GV_NMAX", "2"}});
        nu::Flags e = run(F::server, {});
        NCHECK(s == e, "server ignores CLI-only knobs");
    }
}

// ---- load-time env rules not in the S0b-2 cells (tune.cpp / loader.cpp parse rules, cells from the code)
void testLoadRules() {
    const nu::Target t = target(k_cards[0], "swift", false);
    auto fl = [&](std::vector<std::pair<const char*, const char*>> vals, const char* mode = "balance", const nu::Target* tt = nullptr) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, tt ? *tt : t);
    };
    NCHECK(fl({{"FP8_MASK", "0x5"}}).flags.fp8_mask == 5 && fl({{"FP8_MASK", "bad"}}).flags.fp8_mask == 7 &&
               fl({{"FP8_MASK", "300"}}).flags.fp8_mask == 44,
           "FP8_MASK stoul base 0, catch -> 7, uint8 cast");
    NCHECK(fl({{"ATTN_SEG", "1000"}}).flags.attn_seg == 768 && fl({{"ATTN_SEG", "x"}}).flags.attn_seg == 0, "ATTN_SEG /256*256");
    NCHECK(fl({{"MOE_WIDE", ""}}).flags.moe_wide && !fl({{"MOE_MXW", "0"}}).flags.moe_mxw && !fl({{"MOE_RBF", "0"}}).flags.moe_rbf,
           "loader flags: envFlag (\"\" is on)");
    NCHECK(!fl({{"G8T", "0"}}).flags.g8t && fl({}).flags.g8t, "G8T");
    NCHECK(!fl({}, "precise").flags.g8t, "g8t needs fp8_prefill");
    // Q8DEC: "" keeps the mode's choice; "1" in precise = int8 decode (lossy override)
    NCHECK(!fl({{"Q8DEC", ""}}).flags.prec_dec && fl({{"Q8DEC", "0"}}).flags.prec_dec, "Q8DEC");
    {
        const nu::Plan p = fl({{"Q8DEC", "1"}}, "precise");
        NCHECK(!p.flags.prec_dec && p.q8dec && p.lossy_override, "Q8DEC=1 precise");
    }
    // relaxacc parameters (fast dense): parsed only when relaxacc is on; a bad value throws as loadOrTune did
    {
        const nu::Plan p = fl({{"RELAX_K", "2"}, {"RELAX_ALPHA", "0.5"}}, "fast");
        NCHECK(p.flags.relax_on && p.flags.relax_k == 2 && p.flags.relax_alpha == 0.5f, "relax params");
        NCHECK(fl({{"RELAX_K", "2"}}, "balance").flags.relax_k == 4, "relax params only with relaxacc");
        bool threw = false;
        try {
            (void)fl({{"RELAX_K", "x"}}, "fast");
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        NCHECK(threw, "RELAX_K=x throws (std::stoul)");
    }
    // strix: the 8060S row, GDN_STRIX_MIN floor 32, needs the shape and the item
    {
        const nu::Target s = target(k_cards[1], "swift", false);
        NCHECK(fl({{"GDN_STRIX_MIN", "8"}}, "balance", &s).flags.gdn_strix_min == 32, "strix min floor");
        NCHECK(!fl({}, "precise", &s).flags.gdn_strix, "strix needs gdnwmma");
        nu::Target s2 = s;
        s2.gdn_strix_shape = false;
        NCHECK(!fl({}, "balance", &s2).flags.gdn_strix, "strix needs the shape");
        NCHECK(fl({{"GDN_STRIX_MIN", "512"}}, "balance", &t).flags.gdn_strix_min == 256, "strix min only with strix");
    }
    // dense non-MXFP4 on the R9700 (KG-1): gdn_wmma / h16 and h16_dq
    {
        nu::Target q = t;
        q.has_mxfp4 = false;
        const nu::Plan p = fl({}, "balance", &q);
        NCHECK(p.flags.gdn_wmma && p.flags.ffn_h16 && p.flags.h16_dq, "dense quant relaxed");
        nu::Target qs = target(k_cards[1], "swift", false);
        qs.has_mxfp4 = false;
        const nu::Plan ps = fl({}, "balance", &qs);
        NCHECK(!ps.flags.gdn_wmma && !ps.flags.ffn_h16 && fl({{"Q4_RELAXED", "1"}}, "balance", &qs).flags.ffn_h16, "8060s non-MXFP4");
    }
    // FP8=0 also turns moefp8 off and wins over MOE_FP8 (tune.cpp:854, :941-944)
    {
        const nu::Target m = target(k_cards[0], "moe", false);
        const nu::Plan p = fl({{"FP8", "0"}, {"MOE_FP8", "1"}}, "balance", &m);
        NCHECK(!p.flags.moe_fp8 && !p.on(nu::Item::moefp8), "FP8=0 over MOE_FP8");
    }
}

// ---- arch025-n2b-r1: unit cells for the two N2b mutants only the oracle killed (tools/impact mutants n2b-q8dec-empty,
// n2b-g8t-kernels). Decision inputs taken from the oracle's mismatch lines.
// audit-fastonly-overrides FO-5 (2026-10-26): WHIRL_<SWITCH>=0 must not turn the switch ON. runtimeFlags tests only
// presence (numerics.cpp:437-450 cli, :457-460 server: `if (v.naive_attn) F.naive_attn = true;`), so WHIRL_NAIVE_ATTN=0
// enables the naive attention (scores buffer max_batch x n_head x max_ctx x 4 B, attn.cpp:133) and labels the run
// +overrides; same for NO_FUSE, FLOAT_GEMV, GDN_SEQ, NO_GRAPH. Expected to FAIL on 799aa49 / e87d58f.
void testSwitchZeroIsOff() {
    for (const nu::Front fr : {nu::Front::cli, nu::Front::server}) {
        const char* fn = fr == nu::Front::cli ? "cli" : "server";
        auto withZero = [&](nu::Raw nu::RuntimeValues::*field) {
            nu::RuntimeValues v;
            v.*field = std::string("0");
            nu::Plan p;
            nu::applyFront(p, fr, v, true);
            return p.flags;
        };
        const nu::Flags d{};
        NCHECK(!withZero(&nu::RuntimeValues::naive_attn).naive_attn, "FO-5 {}: WHIRL_NAIVE_ATTN=0 turned naive attention on", fn);
        NCHECK(!withZero(&nu::RuntimeValues::no_fuse).no_fuse, "FO-5 {}: WHIRL_NO_FUSE=0 turned no_fuse on", fn);
        NCHECK(!withZero(&nu::RuntimeValues::float_gemv).float_gemv, "FO-5 {}: WHIRL_FLOAT_GEMV=0 turned float_gemv on", fn);
        NCHECK(withZero(&nu::RuntimeValues::gdn_seq).gdn_chunked == d.gdn_chunked, "FO-5 {}: WHIRL_GDN_SEQ=0 turned chunked DeltaNet off", fn);
        if (fr == nu::Front::cli)
            NCHECK(withZero(&nu::RuntimeValues::no_graph).use_graph == d.use_graph, "FO-5 cli: WHIRL_NO_GRAPH=0 turned the graph off");
    }
}

void testN2bR1() {
    const nu::Target t = target(k_cards[0], "swift", false);
    // n2b-q8dec-empty: WHIRL_Q8DEC= (empty) must not decide prec_dec (tune.cpp:955). Needs a mode whose default is f16 decode
    // (precise: prec_dec true); in balance prec_dec is false either way (oracle: "front 2: prec_dec: old 1 new 0", precise cell).
    for (bool items : {false, true}) {
        nu::Request r = nu::parseMode("precise", "t");
        r.load_overrides = items;
        *r.env.find("Q8DEC") = std::string();
        const nu::Plan p = nu::plan(r, t);
        NCHECK(p.flags.prec_dec, "Q8DEC='' keeps precise f16 decode (items {})", items);
        NCHECK(!p.lossy_override, "Q8DEC='' is no lossy override (items {})", items);
    }
    // n2b-g8t-kernels: g8t needs the whole fragment-tiled set (Model::useTiledFp8): fp8 GEMMs present, tiled GEMM absent -> off
    // (oracle: "refused g8t: old 0 new 1"). The default card has both, so g8t is on.
    NCHECK(nu::plan(nu::parseMode("balance", "t"), t).flags.g8t, "g8t on with k_gemm8 + k_gemm8t");
    for (bool items : {false, true}) {
        nu::Target t2 = t;
        t2.k_gemm8 = true;
        t2.k_gemm8t = false;
        nu::Request r = nu::parseMode("balance", "t");
        r.load_overrides = items;
        NCHECK(!nu::plan(r, t2).flags.g8t, "g8t needs k_gemm8t (items {})", items);
    }
}

// audit-fastonly-overrides (2026-10-26): a fast-only item (fast_only_items: kvq4, relaxacc, headq, moeskip, a8, a4)
// never reaches precise or balance through a WHIRL_* override (owner mode standard 2026-10-26; the moeskip-r1 class:
// WHIRL_MOESKIP_THR in precise reached moe_f.hip with id -1). Same semantics as numerics::applyFastOnlyOverride
// (feat027-i8qk-r2 e982759 / feat028-moefast97-r1 7ef2cae): plan untouched, a note in the load log.
// Expected to FAIL on 799aa49: FO-1 (WHIRL_KV=q4 enables kvq4 via applyOverride, numerics.cpp:394-395, and chooseKv
// returns the forced q4, numerics.cpp:684-688) and FO-2 (WHIRL_RELAX=1 enables relaxacc + relax_on,
// numerics.cpp:402-412). FO-3 is the general invariant; FO-4 pins that every env name is classified.
void testFastOnlyOverrides() {
    auto planOf = [](const char* mode, const nu::Target& t, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    auto fastOnlyOn = [](const nu::Plan& p) {
        std::uint32_t m = 0;
        for (const nu::ItemState& s : p.items)
            if (s.enabled) m |= nu::bit(s.item);
        return m & nu::fast_only_items;
    };
    for (const CardFake& c : k_cards)
        for (const char* model : {"swift", "moe"})
            for (const char* mode : {"precise", "balance"}) {
                const std::string cell = std::format("{} {} {}", c.name, model, mode);
                // FO-1: WHIRL_KV=q4 (4-bit KV is fast-only)
                {
                    const nu::Target t = target(c, model, true);
                    const nu::Plan p = planOf(mode, t, {{"KV", "q4"}});
                    NCHECK(!p.on(nu::Item::kvq4), "FO-1 {}: WHIRL_KV=q4 must not enable kvq4", cell);
                    const nu::KvCaps caps{c.k_kv_q8h, c.k_kv_q8v, c.k_kv_q4, c.prof.numerics_8060s};
                    const nu::KvChoice kc = nu::chooseKv(nu::parseMode(mode, "t"), std::string_view(model) == "moe", caps, whirl::KvFormat::q4);
                    NCHECK(kc.kv != whirl::KvFormat::q4, "FO-1 {}: chooseKv(forced q4) gave q4 ({})", cell, kc.why);
                }
                // FO-2: WHIRL_RELAX=1 (relaxed acceptance is fast-only)
                {
                    const nu::Target t = target(c, model, false);
                    const nu::Plan p = planOf(mode, t, {{"RELAX", "1"}});
                    NCHECK(!p.on(nu::Item::relaxacc) && !p.flags.relax_on, "FO-2 {}: WHIRL_RELAX=1 must not enable relaxacc", cell);
                    const nu::Plan pk = planOf(mode, t, {{"RELAX", "1"}, {"RELAX_K", "16"}, {"RELAX_ALPHA", "0.01"}});
                    NCHECK(!pk.flags.relax_on && pk.flags.relax_k == nu::Flags{}.relax_k, "FO-2 {}: relax params must stay default", cell);
                }
                // FO-3: no single override (every EnvValues name and switch, typical values) turns a fast-only item on
                {
                    const nu::Target t = target(c, model, false);
                    std::vector<std::pair<std::string, std::string>> probes = {
                        {"KV", "q4"}, {"RELAX", "1"}, {"Q4_RELAXED", "1"}, {"FP8", "1"}, {"MOE_FP8", "1"}, {"SPEC_SAMPLE", "1"},
                        {"Q8DEC", "1"}, {"GDN_WMMA", "1"}, {"FFN_H16", "1"}, {"GEMMHQ", "1"}};
                    for (const auto& e : nu::EnvValues::names()) probes.emplace_back(e.name, "1");
                    for (std::size_t i = 0; nu::k_numerics_switches[i] != nullptr; ++i) probes.emplace_back(nu::k_numerics_switches[i], "1");
                    for (const auto& [n, v] : probes) {
                        const nu::Target tt = n == "KV" ? target(c, model, true) : t;
                        nu::Plan p;
                        try {
                            p = planOf(mode, tt, {{n.c_str(), v.c_str()}});
                        } catch (...) {
                            continue;  // a rejected value is not an enable
                        }
                        NCHECK(fastOnlyOn(p) == 0, "FO-3 {}: WHIRL_{}={} enabled fast-only items 0x{:x}", cell, n, v, fastOnlyOn(p));
                    }
                }
            }
    // FO-4: every WHIRL_* value the plan reads is classified (a new override must say which class it is and, for
    // fast-only, go through the fast-only path). Classes: B bitwise-equal A/B, L lossy item allowed in balance,
    // F fast-only, S numerics A/B switch (labelled +overrides), P relaxacc parameter (only with relaxacc).
    const std::map<std::string, char> k_class = {
        {"FP8", 'L'},      {"FP8_MASK", 'L'},   {"G8T", 'B'},       {"GDN_WMMA", 'L'},    {"ACT_FUSE", 'B'},    {"GEMMH", 'B'},
        {"GEMMHQ", 'B'},   {"ATTN_KX", 'B'},    {"ATTN_KG", 'B'},   {"ATTN_DQF", 'B'},    {"ATTN_SEG", 'B'},    {"GDN_BA", 'B'},
        {"GDN_IN2", 'B'},  {"Q4_RELAXED", 'L'}, {"FFN_H16", 'L'},   {"GDN_STRIX", 'L'},   {"GDN_STRIX_MIN", 'L'}, {"KV", 'F'},
        {"Q8DEC", 'L'},    {"RELAX", 'F'},      {"RELAX_K", 'P'},   {"RELAX_ALPHA", 'P'}, {"RELAX_EPS", 'P'},   {"RELAX_DELTA", 'P'},
        {"SPEC_SAMPLE", 'L'}, {"MOE_FP8", 'L'}, {"MOE_MXW", 'B'},   {"MOE_WIDE", 'B'},    {"MOE_RBF", 'B'},
    };
    for (const auto& e : nu::EnvValues::names()) NCHECK(k_class.count(e.name) == 1, "FO-4: WHIRL_{} has no override class", e.name);
    for (std::size_t i = 0; nu::k_numerics_switches[i] != nullptr; ++i)
        NCHECK(k_class.count(nu::k_numerics_switches[i]) == 0, "FO-4: switch WHIRL_{} is also an item override", nu::k_numerics_switches[i]);
}

}  // namespace

void testNumericsPlan(int& pass, int& fail) {
    testPlanTable();
    testApplyFrontEqualsPlan();  // arch025-N2b (replaces testUnwiredCallerUnchanged: no caller passes Target::gdn_strix now)
    testBitsKeyReflection();
    testFlagScope();
    testFrontEnds();
    testLoadRules();
    testN2bR1();  // arch025-n2b-r1
    testSwitchZeroIsOff();  // audit-fastonly-overrides FO-5 (fails before the fix)
    testFastOnlyOverrides();  // audit-fastonly-overrides FO-1..4 (FO-1 / FO-2 / FO-3 fail before the fix)
    pass += g_p;
    fail += g_f;
}
