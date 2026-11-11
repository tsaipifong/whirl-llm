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
#include "whirl/kernels_abi.h"  // v0251-a7port: kMoepDefault / kMoepGu / kMoepDn (testMoepDefault)
#include "whirl/numerics.h"

#include <cstdio>
#include <cstdlib>  // v0251-a7pol: std::strtoul (testMoepCombMode)
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
    std::size_t n_cells = 0, n_fields = 0, n_repin = 0;
    // the value of one S0b-2 dump key in a plan ("" + known = false: not decided by the plan)
    auto keyOf = [](const nu::Plan& p, const std::string& k, bool& known) {
        known = true;
        if (k == "num_plan") return nu::logLine(p);
        if (k == "num_plan.q8dec") return std::string(p.q8dec ? "1" : "0");
        if (k == "num_plan.q8dec_default") return std::string(p.q8dec_default ? "1" : "0");
        if (k == "num_plan.lossy_override") return std::string(p.lossy_override ? "1" : "0");
        if (k.rfind("num_plan.override[", 0) == 0) {
            const std::size_t i = std::stoul(k.substr(18));
            return i < p.overrides.size() ? p.overrides[i] : std::string("<none>");
        }
        for (const nu::FlagInfo& fi : nu::flagTable())
            if (k == fi.dump_key) return nu::flagValue(p.flags, fi);
        known = false;
        return std::string();
    };
    for (const S0b2Cell& c : k_s0b2_cells) {
        ++n_cells;
        // bug-fastonly (audit F-A, finding M2): the 10 env_RELAX* cells (balance, 5 per card) pin the old behaviour,
        // WHIRL_RELAX=1 -> relaxacc on in balance. RE-PIN AFTER THIS BUGFIX (expected change; a new S0b-2 GPU dump, then
        // regenerate numerics_plan_s0b2.inc). Until then these cells check the fix instead: the plan equals the same cell
        // without the RELAX* values (a fast-only refusal leaves the plan untouched), the refusal note is in Plan::ignored,
        // and the dump differs from it only in the relaxacc keys. Not part of the 0.2.5 work-equivalence claim.
        const bool repin = std::string_view(c.cell).rfind("env_RELAX", 0) == 0 && std::string_view(c.mode) != "fast";
        nu::Request r = nu::parseMode(c.mode, "command line");
        r.load_overrides = true;
        nu::Request r0 = r;  // repin: without RELAX*
        bool kv_explicit = false;
        for (const auto& [n, v] : splitEnv(c.env)) {
            if (k_not_plan_env.count(n)) continue;
            nu::Raw* f = r.env.find(n);
            NCHECK(f != nullptr, "{} {}: env {} has no Request field", c.card, c.cell, n);
            if (f) *f = v;
            if (nu::Raw* f0 = r0.env.find(n); f0 && n.rfind("RELAX", 0) != 0) *f0 = v;
            // loader.cpp:524 kvModeOf: a format name -> explicit (Target::kv_explicit, tune.cpp:935)
            if (n == "KV" && v != "auto") kv_explicit = true;
        }
        const nu::Target tc = target(card(c.card), c.model, kv_explicit);
        const nu::Plan p = nu::plan(r, tc);
        const auto want = splitKv(c.kv);
        std::size_t n_over = 0;
        if (repin) {
            ++n_repin;
            const nu::Plan p0 = nu::plan(r0, tc);
            static const std::set<std::string> k_relax_keys = {"num_plan", "num_plan.lossy_override", "num_plan.override[0]", "relax.on",
                                                               "relax.k", "relax.alpha", "relax.eps", "relax.delta"};
            for (const auto& [k, v] : want) {
                ++n_fields;
                bool known = false, known0 = false;
                const std::string got = keyOf(p, k, known), got0 = keyOf(p0, k, known0);
                NCHECK(known, "{} {}: key {} not decided by the plan", c.card, c.cell, k);
                NCHECK(got == got0, "{} {} (re-pin): {} = '{}', without RELAX* '{}'", c.card, c.cell, k, got, got0);
                NCHECK(got == v || k_relax_keys.count(k) == 1, "{} {} (re-pin): {} = '{}' differs from the dump '{}' outside the relaxacc keys",
                       c.card, c.cell, k, got, v);
            }
            NCHECK(p.overrides == p0.overrides && p.bitsKey() == p0.bitsKey(), "{} {} (re-pin): plan != plan without RELAX*", c.card, c.cell);
            NCHECK(p.ignored.size() == 1 && p.ignored[0].rfind("WHIRL_RELAX=1 ignored: relaxacc is a fast-only item (" + std::string(c.mode) +
                                                                   " never takes it)",
                                                               0) == 0,
                   "{} {} (re-pin): refusal note '{}'", c.card, c.cell, p.ignored.empty() ? std::string("<none>") : p.ignored[0]);
            for (const nu::FlagInfo& fi : nu::flagTable())
                if (*fi.dump_key) NCHECK(want.count(fi.dump_key) == 1, "{} {}: dump has no {}", c.card, c.cell, fi.dump_key);
            continue;
        }
        for (const auto& [k, v] : want) {
            ++n_fields;
            bool known = false;
            const std::string got = keyOf(p, k, known);
            if (k.rfind("num_plan.override[", 0) == 0) ++n_over;
            NCHECK(known, "{} {}: key {} not decided by the plan", c.card, c.cell, k);
            NCHECK(got == v, "{} {}: {} = '{}', S0b-2 dump '{}'", c.card, c.cell, k, got, v);
        }
        NCHECK(p.overrides.size() == n_over, "{} {}: {} overrides, dump {}", c.card, c.cell, p.overrides.size(), n_over);
        NCHECK(p.ignored.empty(), "{} {}: unexpected refusal note '{}'", c.card, c.cell, p.ignored.empty() ? std::string() : p.ignored[0]);
        // every flag with a dump key is in every cell (nothing compared by omission)
        for (const nu::FlagInfo& fi : nu::flagTable())
            if (*fi.dump_key) NCHECK(want.count(fi.dump_key) == 1, "{} {}: dump has no {}", c.card, c.cell, fi.dump_key);
    }
    NCHECK(n_repin == 10, "bug-fastonly re-pin cells {} (expected the 10 env_RELAX* balance cells)", n_repin);
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
        {"gemv_k", "needs GPU dump: the gfx1151 gemvk table (defaultGemvK, applyGemvEnv / applyGemvKEnv: WHIRL_GEMV_K)"},
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
    // arch025-lane1 leg 9: the CLI-only column == the names whirl-server skipped (729c080 src/server/server_main.cpp:707,
    // verbatim), in table order; every other name is sent by both front ends
    {
        std::size_t n_cli = 0;
        for (const auto& e : nu::RuntimeValues::names()) {
            const std::string_view n = e.name;
            const bool old_skip = n == "DBG" || n == "GEMV_MAX" || n == "NO_GRAPH" || n == "GV_GROUP" || n == "GV_NMAX";
            NCHECK(e.cli_only == old_skip, "RuntimeValues::names {} cli_only {}", n, e.cli_only);
            n_cli += e.cli_only ? 1 : 0;
        }
        NCHECK(n_cli == 5, "RuntimeValues::names: 5 CLI-only names (got {})", n_cli);
    }
    // on / off knobs (bug-fastonly, audit F-D; was "presence: any value, 0 included"): on for any value but "0" / "";
    // GDN_V0 is still a presence knob (not in the F-D list; "0" turns it on)
    for (F fr : {F::cli, F::server}) {
        const nu::Flags a = run(fr, {{"NO_FUSE", "0"}, {"FLOAT_GEMV", "0"}, {"NAIVE_ATTN", "0"}, {"GDN_SEQ", "0"}, {"GDN_V0", "0"}});
        NCHECK(!a.no_fuse && !a.float_gemv && !a.naive_attn && a.gdn_chunked && a.gdn_v0, "=0 is off {}", static_cast<int>(fr));
        const nu::Flags a1 = run(fr, {{"NO_FUSE", "1"}, {"FLOAT_GEMV", "yes"}, {"NAIVE_ATTN", "1"}, {"GDN_SEQ", "1"}});
        NCHECK(a1.no_fuse && a1.float_gemv && a1.naive_attn && !a1.gdn_chunked, "=1 is on {}", static_cast<int>(fr));
        const nu::Flags a0 = run(fr, {{"NO_FUSE", ""}, {"FLOAT_GEMV", ""}, {"NAIVE_ATTN", ""}, {"GDN_SEQ", ""}});
        NCHECK(!a0.no_fuse && !a0.float_gemv && !a0.naive_attn && a0.gdn_chunked, "='' is off {}", static_cast<int>(fr));
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
    NCHECK(run(F::cli, {{"NO_GRAPH", "0"}}).use_graph && run(F::cli, {{"NO_GRAPH", ""}}).use_graph && !run(F::cli, {{"NO_GRAPH", "1"}}).use_graph,
           "cli NO_GRAPH (bug-fastonly: =0 / '' off, was presence)");
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
    // v0251-a11port (feat026-router, was loader.cpp envFlag("MOE_RFUSE", false); t = the dense model on the R9700 card,
    // where the default is off): the loader's envFlag rule, "0" = off, any other value (also "") = on; a plain value, no
    // A/B switch note
    NCHECK(!fl({}).flags.moe_rfuse && fl({{"MOE_RFUSE", "1"}}).flags.moe_rfuse && fl({{"MOE_RFUSE", ""}}).flags.moe_rfuse &&
               !fl({{"MOE_RFUSE", "0"}}).flags.moe_rfuse && !fl({{"MOE_RFUSE", "1"}}, "precise").lossy_override,
           "MOE_RFUSE: envFlag rule (moe_rfuse)");
    // v0252-c3port (feat026-moedec-r1 d53542f: envFlag rule, "true" / "yes" on, never lossy): WHIRL_MOE_FUSE /
    // WHIRL_GEMVH_F32 -> moe_fuse / gemvh_f32, default off. Fails before the port (loadFlags does not read them).
    NCHECK(!fl({}).flags.moe_fuse && fl({{"MOE_FUSE", "1"}}).flags.moe_fuse && fl({{"MOE_FUSE", "true"}}).flags.moe_fuse &&
               fl({{"MOE_FUSE", ""}}).flags.moe_fuse && !fl({{"MOE_FUSE", "0"}}).flags.moe_fuse &&
               !fl({{"MOE_FUSE", "1"}}, "precise").lossy_override,
           "MOE_FUSE: envFlag rule (moe_fuse)");
    NCHECK(!fl({}).flags.gemvh_f32 && fl({{"GEMVH_F32", "1"}}).flags.gemvh_f32 && fl({{"GEMVH_F32", "yes"}}).flags.gemvh_f32 &&
               fl({{"GEMVH_F32", ""}}).flags.gemvh_f32 && !fl({{"GEMVH_F32", "0"}}).flags.gemvh_f32 &&
               !fl({{"GEMVH_F32", "1"}}, "precise").lossy_override,
           "GEMVH_F32: envFlag rule (gemvh_f32)");
    NCHECK(!fl({{"G8T", "0"}}).flags.g8t && fl({}).flags.g8t, "G8T");
    // feat026-moep (was loader.cpp envGet; t = the R9700 card, where the default is off): variant names -> index + 1,
    // unknown / "" / "0" / unset -> 0 (off), MOEP_COMB strtoul
    NCHECK(fl({}).flags.moep_gu == 0 && fl({}).flags.moep_dn == 0 && fl({}).flags.moep_comb == 0 &&
               fl({{"MOEP_GU", "g64n64k64"}}).flags.moep_gu == 1 && fl({{"MOEP_GU", "g128n128k32"}}).flags.moep_gu == 6 &&
               fl({{"MOEP_GU", "nope"}}).flags.moep_gu == 0 && fl({{"MOEP_GU", ""}}).flags.moep_gu == 0 &&
               fl({{"MOEP_DN", "m128w4n32k64"}}).flags.moep_dn == 1 && fl({{"MOEP_DN", "m128w4n256k32"}}).flags.moep_dn == 6 &&
               fl({{"MOEP_COMB", "2"}}).flags.moep_comb == 2 && fl({{"MOEP_COMB", "x"}}).flags.moep_comb == 0 &&
               fl({{"MOEP_COMB", "5z"}}, "fast").flags.moep_comb == 5,  // v0251-a7pol: 5 is fast-only (testMoepCombMode)
           "MOEP_GU / _DN / _COMB: the loader's parse rules");
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
        // v0251-a7fix: WHIRL_MOEP_COMB 0-2 are bitwise, 3 / 4 are f16 intermediates (L); 5 (atomic scatter) is fast-only
        // through the plan's refusal (v0251-a7pol B26-41, testMoepCombMode), so the variable's class stays 'L'.
        {"MOEP_GU", 'B'},  {"MOEP_DN", 'B'},   {"MOEP_COMB", 'L'},
        {"MOE_RFUSE", 'B'},  // v0251-a11port: fused router, bit-identical to moe_logits_f32 + moe_topk (kt moemx, golden)
        {"ATTN_KGVAR", 'B'},  // v0252-c1port: attn_kgv / attn_kgk, designed bit-identical to attn_kg (kt attn, golden)
        // v0252-c3port (feat026-moedec): fused precise MoE decode / gemvh on f32 input, bit-identical (kt moefuse, golden)
        {"MOE_FUSE", 'B'},   {"GEMVH_F32", 'B'},
        {"ATTN_WS4R", 'B'},  // v0252-ws4r97b: gfx1201 attn_ws4r1 / 2[b], bit-identical to attn_wsplit1 / 2 (kt ws4r, golden)
    };
    for (const auto& e : nu::EnvValues::names()) NCHECK(k_class.count(e.name) == 1, "FO-4: WHIRL_{} has no override class", e.name);
    // v0251-a6port: WHIRL_GEMV_K (EnvSnapshot, config.cpp) is class B (every gemvk variant bitwise == gemvq). FO-3
    // safety by construction: the plan never reads it (not an EnvValues name, not a switch), so no value of it can
    // enable a fast-only item; pinned here so a later move into the plan must be classified again.
    {
        const std::map<std::string, char> k_snapshot_class = {{"GEMV_K", 'B'}};
        for (const auto& [n, cls] : k_snapshot_class) {
            NCHECK(cls == 'B', "FO-4: WHIRL_{} class {} (gemvk is bitwise: B)", n, cls);
            bool in_plan = false;
            for (const auto& e : nu::EnvValues::names()) in_plan = in_plan || n == e.name;
            for (std::size_t i = 0; nu::k_numerics_switches[i] != nullptr; ++i) in_plan = in_plan || n == nu::k_numerics_switches[i];
            NCHECK(!in_plan, "FO-4: WHIRL_{} is read by the numerics plan (classify it in k_class, probe it in FO-3)", n);
        }
    }
    for (std::size_t i = 0; nu::k_numerics_switches[i] != nullptr; ++i)
        NCHECK(k_class.count(nu::k_numerics_switches[i]) == 0, "FO-4: switch WHIRL_{} is also an item override", nu::k_numerics_switches[i]);
}

// bug-fastonly: the refusal's text and log path (Plan::ignored, chooseKv why), and fast unchanged (WHIRL_RELAX / WHIRL_KV=q4 still
// apply there as before: applyOverride, chooseKv's forced format)
void testFastOnlyRefusal() {
    auto planOf = [](const char* mode, const nu::Target& t, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    const CardFake& r9 = card("r9700");
    for (const char* model : {"swift", "moe"}) {
        const bool moe = std::string_view(model) == "moe";
        const nu::Target t = target(r9, model, false), tk = target(r9, model, true);
        for (const char* mode : {"precise", "balance"}) {
            const nu::Plan base = planOf(mode, t, {});
            const nu::Plan pr = planOf(mode, t, {{"RELAX", "1"}, {"RELAX_K", "x"}});  // the parameters are not parsed either
            NCHECK(pr.bitsKey() == base.bitsKey() && pr.overrides == base.overrides && nu::logLine(pr) == nu::logLine(base),
                   "refusal {} {}: RELAX=1 changed the plan", model, mode);
            NCHECK(pr.ignored == std::vector<std::string>{std::format("WHIRL_RELAX=1 ignored: relaxacc is a fast-only item ({} never takes it)", mode)},
                   "refusal {} {}: RELAX note", model, mode);
            NCHECK(planOf(mode, t, {{"RELAX", "0"}}).ignored.empty(), "refusal {} {}: RELAX=0 is no request", model, mode);
            const nu::Plan base_k = planOf(mode, tk, {});
            const nu::Plan pk = planOf(mode, tk, {{"KV", "q4"}});
            NCHECK(pk.bitsKey() == base_k.bitsKey() && pk.overrides == base_k.overrides, "refusal {} {}: KV=q4 changed the plan", model, mode);
            NCHECK(pk.ignored == std::vector<std::string>{std::format("WHIRL_KV=q4 ignored: kvq4 is a fast-only item ({} never takes it)", mode)},
                   "refusal {} {}: KV note", model, mode);
            // chooseKv: forced q4 -> the mode's own choice, not a debug override; other forced formats unchanged (F-E: not in scope)
            const nu::KvCaps caps{true, true, true, false};
            const nu::Request rq = nu::forModel(nu::parseMode(mode, "t"), moe);
            const nu::KvChoice own = nu::chooseKv(rq, moe, caps, std::nullopt), f4 = nu::chooseKv(rq, moe, caps, whirl::KvFormat::q4);
            NCHECK(f4.kv == own.kv && !f4.debug && f4.why == std::format("WHIRL_KV=q4 ignored: kvq4 is a fast-only item ({} never takes it); {}", mode, own.why),
                   "refusal {} {}: chooseKv(q4) {} / {}", model, mode, whirl::kvFormatName(f4.kv), f4.why);
            for (const whirl::KvFormat k : {whirl::KvFormat::f16, whirl::KvFormat::q8, whirl::KvFormat::q8h, whirl::KvFormat::q8v}) {
                const nu::KvChoice f = nu::chooseKv(rq, moe, caps, k);
                NCHECK(f.kv == k && f.debug, "refusal {} {}: forced {} still a debug override", model, mode, whirl::kvFormatName(k));
            }
        }
        // fast: unchanged
        const nu::Plan fr = planOf("fast", t, {{"RELAX", "1"}, {"RELAX_K", "6"}});
        NCHECK(fr.on(nu::Item::relaxacc) && fr.flags.relax_on && fr.flags.relax_k == 6 && fr.ignored.empty(), "fast {}: RELAX=1 applies", model);
        NCHECK(!planOf("fast", t, {{"RELAX", "0"}}).flags.relax_on, "fast {}: RELAX=0 turns it off", model);
        const nu::Plan fk = planOf("fast", tk, {{"KV", "q4"}});
        NCHECK(fk.ignored.empty(), "fast {}: KV=q4 no refusal", model);
        const nu::KvChoice ff = nu::chooseKv(nu::forModel(nu::parseMode("fast", "t"), moe), moe, nu::KvCaps{true, true, true, false}, whirl::KvFormat::q4);
        NCHECK(ff.kv == whirl::KvFormat::q4 && ff.debug, "fast {}: chooseKv(forced q4) = q4 debug override", model);
    }
}

// v0252-f2 (DISCOVERY v4 row F2): fast MoE relaxed acceptance by a named arm. RA-1 the arm table and the pre-screen
// defaults; RA-2 fast MoE takes the arm (8060S always, R9700 only with F1's one-pass check); RA-3 the env still
// overrides it; RA-4 precise / balance and dense fast plans are unchanged by any arm (bitsKey, log line, overrides,
// ignored, relax_drafts); RA-5 the draft-count default. Fails before the port (stub: no arm is known).
void testRelaxMoeArms() {
    auto planOf = [](const char* mode, const nu::Target& t, std::vector<std::pair<const char*, const char*>> vals, bool lo = true) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = lo;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    const nu::ItemState& (*item)(const nu::Plan&) = [](const nu::Plan& p) -> const nu::ItemState& {
        return p.items[static_cast<std::size_t>(nu::Item::relaxacc)];
    };
    // RA-1
    struct Want {
        const char* name;
        bool on;
        std::uint32_t k;
        float alpha;
        std::uint32_t drafts;
    };
    const Want k_want[] = {{"off", false, 4, 0.1f, 0}, {"k4a10", true, 4, 0.1f, 0}, {"k6a05", true, 6, 0.05f, 0}, {"k4a10d2", true, 4, 0.1f, 2}};
    NCHECK(nu::relax_moe_arms.size() == std::size(k_want), "RA-1 arm count {}", nu::relax_moe_arms.size());
    for (const Want& w : k_want) {
        const nu::RelaxArm* a = nu::relaxArm(w.name);
        NCHECK(a != nullptr, "RA-1 arm {} unknown", w.name);
        if (a == nullptr) continue;
        NCHECK(a->name == w.name && a->on == w.on && a->k == w.k && a->alpha == w.alpha && a->drafts == w.drafts, "RA-1 arm {} values", w.name);
    }
    NCHECK(nu::relaxArm("") == nullptr && nu::relaxArm("k4") == nullptr, "RA-1 unknown names");
    // v0252-f2def: the shipped arms (8060S G4a / T1 q7 job 96: k6a05 R 0.9851, +9.1 % chat; R9700 G3a / T1 btl
    // gpu-confirm: k4a10d2 R 0.9915, +17.9 % chat). Re-pin here when a later screen picks another arm.
    NCHECK(nu::relaxMoeDefaultName(true) == "k6a05" && nu::relaxMoeDefaultName(true) == nu::relax_moe_default_8060s, "RA-1 8060S default '{}'",
           nu::relaxMoeDefaultName(true));
    NCHECK(nu::relaxMoeDefaultName(false) == "k4a10d2" && nu::relaxMoeDefaultName(false) == nu::relax_moe_default_r9700, "RA-1 R9700 default '{}'",
           nu::relaxMoeDefaultName(false));
    for (const std::string_view n : {nu::relax_moe_default_8060s, nu::relax_moe_default_r9700})
        NCHECK(nu::relaxArm(n) != nullptr, "RA-1 default {} is not an arm", n);

    for (const CardFake& c : k_cards) {
        const bool g51 = c.prof.numerics_8060s;
        for (const bool rank : {false, true}) {
            nu::Target t0 = target(c, "moe", false);
            t0.k_relax_rank = rank;
            const nu::Plan base = planOf("fast", t0, {});
            // today's behaviour with no arm: MoE fast skips relaxacc
            NCHECK(!base.on(nu::Item::relaxacc) && !base.flags.relax_on && base.relax_drafts == 0, "RA-2 {} base", c.name);
            for (const Want& w : k_want) {
                nu::Target t = t0;
                t.relax_moe_arm = w.name;
                const std::string cell = std::format("{} rank {} arm {}", c.name, rank, w.name);
                const nu::Plan p = planOf("fast", t, {});
                const bool takes = w.on && (g51 || rank);
                // RA-2
                NCHECK(p.on(nu::Item::relaxacc) == takes && p.flags.relax_on == takes, "RA-2 {}: on {}", cell, p.flags.relax_on);
                if (takes) {
                    NCHECK(p.flags.relax_k == w.k && p.flags.relax_alpha == w.alpha && p.relax_drafts == w.drafts, "RA-2 {}: k {} alpha {} drafts {}",
                           cell, p.flags.relax_k, p.flags.relax_alpha, p.relax_drafts);
                    NCHECK(item(p).note.starts_with(std::format("MoE arm {}: ", w.name)), "RA-2 {}: note '{}'", cell, item(p).note);
                    NCHECK(item(p).note.find(std::format("[k {}, alpha {:.2f}", w.k, w.alpha)) != std::string::npos, "RA-2 {}: params note", cell);
                    // flags do not depend on load_overrides (only the notes do)
                    NCHECK(planOf("fast", t, {}, false).flags == p.flags, "RA-2 {}: flags differ without load_overrides", cell);
                } else if (w.on) {
                    NCHECK(p.relax_drafts == 0 && item(p).note.find("one-pass relaxacc check (F1)") != std::string::npos, "RA-2 {}: R9700 without F1 '{}'",
                           cell, item(p).note);
                } else {
                    NCHECK(p.bitsKey() == base.bitsKey() && nu::logLine(p) == nu::logLine(base) && p.relax_drafts == 0, "RA-2 {}: off arm changed the plan", cell);
                }
                // RA-3: the env over the arm
                const nu::Plan pk = planOf("fast", t, {{"RELAX_K", "3"}, {"RELAX_ALPHA", "0.2"}});
                if (takes) NCHECK(pk.flags.relax_k == 3 && pk.flags.relax_alpha == 0.2f, "RA-3 {}: env K / ALPHA", cell);
                const nu::Plan p0 = planOf("fast", t, {{"RELAX", "0"}});
                NCHECK(!p0.flags.relax_on && !p0.on(nu::Item::relaxacc) && p0.relax_drafts == 0, "RA-3 {}: RELAX=0 turns it off", cell);
                const nu::Plan p1 = planOf("fast", t, {{"RELAX", "1"}});
                NCHECK(p1.flags.relax_on && p1.on(nu::Item::relaxacc), "RA-3 {}: RELAX=1 forces it", cell);
                const Want& base_w = w.on ? w : k_want[1];  // forced with no arm: today's k 4 / alpha 0.1, no drafts
                NCHECK(p1.flags.relax_k == base_w.k && p1.flags.relax_alpha == base_w.alpha && p1.relax_drafts == (w.on ? w.drafts : 0u),
                       "RA-3 {}: forced k {} alpha {} drafts {}", cell, p1.flags.relax_k, p1.flags.relax_alpha, p1.relax_drafts);
                // RA-5
                NCHECK(nu::mtpDraftsDefault(1, p) == (takes && w.drafts > 0 ? w.drafts : 1u), "RA-5 {}: drafts default", cell);
                NCHECK(nu::mtpDraftsDefault(1, p0) == 1u, "RA-5 {}: RELAX=0 keeps the device default", cell);
                // RA-4: precise / balance (with and without WHIRL_RELAX=1) and dense fast never see the arm
                for (const char* mode : {"precise", "balance"})
                    for (const bool force : {false, true}) {
                        std::vector<std::pair<const char*, const char*>> env;
                        if (force) env = {{"RELAX", "1"}, {"RELAX_K", "6"}};
                        const nu::Plan a = planOf(mode, t, env), b = planOf(mode, t0, env);
                        NCHECK(a.bitsKey() == b.bitsKey() && nu::logLine(a) == nu::logLine(b) && a.overrides == b.overrides && a.ignored == b.ignored &&
                                   a.relax_drafts == 0 && !a.flags.relax_on && nu::propsJson(a, "f16") == nu::propsJson(b, "f16"),
                               "RA-4 {} {} force {}: the arm changed the plan", cell, mode, force);
                    }
                nu::Target td = target(c, "swift", false), td0 = td;
                td.relax_moe_arm = w.name;
                td.k_relax_rank = rank;
                for (const char* mode : {"precise", "balance", "fast"}) {
                    const nu::Plan a = planOf(mode, td, {}), b = planOf(mode, td0, {});
                    NCHECK(a.bitsKey() == b.bitsKey() && nu::logLine(a) == nu::logLine(b) && a.relax_drafts == 0, "RA-4 {} swift {}: the arm changed a dense plan",
                           cell, mode);
                }
            }
        }
    }
    // RA-6 (v0252-f2def): the shipped default as the loader builds it (relax_moe_arm = relaxMoeDefaultName(gfx1151),
    // k_relax_rank = F1's argmax_rank_rows_to is in the code object): Ornith (MoE) fast takes k4a10d2 on the R9700
    // (2 MTP drafts) and k6a05 on the 8060S (the device's draft count); without F1 the R9700 still skips it; precise /
    // balance and every Swift (dense) mode equal the plan with no arm; WHIRL_RELAX=0 turns it off and keeps the
    // device's draft count. Fails while both defaults are "off".
    for (const CardFake& c : k_cards) {
        const bool g51 = c.prof.numerics_8060s;
        const Want& w = g51 ? k_want[2] : k_want[3];
        for (const bool rank : {false, true}) {
            nu::Target t = target(c, "moe", false), tn = t;
            t.relax_moe_arm = nu::relaxMoeDefaultName(g51);
            t.k_relax_rank = tn.k_relax_rank = rank;
            const std::string cell = std::format("RA-6 {} rank {}", c.name, rank);
            const bool takes = g51 || rank;
            const nu::Plan p = planOf("fast", t, {});
            NCHECK(p.on(nu::Item::relaxacc) == takes && p.flags.relax_on == takes, "{}: Ornith fast relaxacc on {} (want {})", cell, p.flags.relax_on, takes);
            if (takes) {
                NCHECK(p.flags.relax_k == w.k && p.flags.relax_alpha == w.alpha && p.relax_drafts == w.drafts, "{}: arm {} k {} alpha {} drafts {}", cell,
                       w.name, p.flags.relax_k, p.flags.relax_alpha, p.relax_drafts);
                NCHECK(item(p).note.starts_with(std::format("MoE arm {}: ", w.name)), "{}: note '{}'", cell, item(p).note);
                NCHECK(nu::logLine(p).find("relaxacc") != std::string::npos, "{}: log line has no relaxacc: {}", cell, nu::logLine(p));
            } else {
                NCHECK(item(p).note.find("one-pass relaxacc check (F1)") != std::string::npos, "{}: R9700 without F1 '{}'", cell, item(p).note);
            }
            NCHECK(nu::mtpDraftsDefault(1, p) == (takes && w.drafts > 0 ? w.drafts : 1u), "{}: drafts default {}", cell, nu::mtpDraftsDefault(1, p));
            const nu::Plan p0 = planOf("fast", t, {{"RELAX", "0"}}), p0n = planOf("fast", tn, {{"RELAX", "0"}});
            NCHECK(!p0.flags.relax_on && nu::mtpDraftsDefault(1, p0) == 1u && p0.bitsKey() == p0n.bitsKey(), "{}: WHIRL_RELAX=0", cell);
            for (const char* mode : {"precise", "balance"}) {
                const nu::Plan a = planOf(mode, t, {}), b = planOf(mode, tn, {});
                NCHECK(a.bitsKey() == b.bitsKey() && nu::logLine(a) == nu::logLine(b) && a.overrides == b.overrides && a.ignored == b.ignored &&
                           a.relax_drafts == 0 && !a.flags.relax_on && nu::propsJson(a, "f16") == nu::propsJson(b, "f16"),
                       "{} Ornith {}: the shipped arm changed the plan", cell, mode);
            }
            nu::Target td = target(c, "swift", false), td0 = td;
            td.relax_moe_arm = nu::relaxMoeDefaultName(g51);
            td.k_relax_rank = td0.k_relax_rank = rank;
            for (const char* mode : {"precise", "balance", "fast"}) {
                const nu::Plan a = planOf(mode, td, {}), b = planOf(mode, td0, {});
                NCHECK(a.bitsKey() == b.bitsKey() && nu::logLine(a) == nu::logLine(b) && a.relax_drafts == 0 && nu::propsJson(a, "f16") == nu::propsJson(b, "f16"),
                       "{} Swift {}: the shipped arm changed a dense plan", cell, mode);
            }
        }
    }
}

}  // namespace

// v0251-a7port: the Radeon 8060S MoE prefill default (kernels::kMoepDefault = g64n64k64 + m128w4n128k64 + moe_combine4,
// plan flags = variant index + 1) on the gfx1151 card with MXFP4 routed experts, in every mode; off on the R9700, on a
// dense model and on non-MXFP4 experts; WHIRL_MOEP_GU / _DN / _COMB override it ("0" / "" = off)
static void testMoepDefault() {
    namespace kn = whirl::kernels;
    NCHECK(std::string(kn::kMoepGu[kn::kMoepDefault.gu].name) == "g64n64k64" && std::string(kn::kMoepDn[kn::kMoepDefault.dn].name) == "m128w4n128k64" &&
               kn::kMoepDefault.comb == 1u,
           "kMoepDefault = the screened combination (v0251-judge6 / PLAN V1-11)");
    auto fl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t).flags;
    };
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const nu::Target t = target(c, model, false);
                const nu::Flags f = fl(t, mode, {});
                const bool on = std::string_view(c.name) == "8060s" && std::string_view(model) == "moe";
                // v0252-moedf16: the R9700 MoE MXFP4 balance / fast default is COMB 3 (f16 moe_yd on the fp8 kind), no moep
                const bool ydh = std::string_view(c.name) == "r9700" && std::string_view(model) == "moe" && std::string_view(mode) != "precise";
                NCHECK(on    ? f.moep_gu == 1 && f.moep_dn == 3 && f.moep_comb == 1
                       : ydh ? f.moep_gu == 0 && f.moep_dn == 0 && f.moep_comb == 3
                             : f.moep_gu == 0 && f.moep_dn == 0 && f.moep_comb == 0,
                       "moep default {} {} {}: gu {} dn {} comb {}", c.name, model, mode, f.moep_gu, f.moep_dn, f.moep_comb);
            }
    const nu::Target t8 = target(card("8060s"), "moe", false);
    for (const char* mode : {"precise", "balance", "fast"}) {
        const nu::Flags g0 = fl(t8, mode, {{"MOEP_GU", "0"}}), d0 = fl(t8, mode, {{"MOEP_DN", ""}}), c0 = fl(t8, mode, {{"MOEP_COMB", "0"}});
        NCHECK(g0.moep_gu == 0 && g0.moep_dn == 3 && g0.moep_comb == 1, "8060s moe {}: WHIRL_MOEP_GU=0 turns the gate + up variant off", mode);
        NCHECK(d0.moep_gu == 1 && d0.moep_dn == 0 && d0.moep_comb == 1, "8060s moe {}: WHIRL_MOEP_DN= (empty) turns the down variant off", mode);
        NCHECK(c0.moep_gu == 1 && c0.moep_dn == 3 && c0.moep_comb == 0, "8060s moe {}: WHIRL_MOEP_COMB=0 = moe_combine", mode);
        const nu::Flags all0 = fl(t8, mode, {{"MOEP_GU", "0"}, {"MOEP_DN", "0"}, {"MOEP_COMB", "0"}});
        NCHECK(all0.moep_gu == 0 && all0.moep_dn == 0 && all0.moep_comb == 0, "8060s moe {}: WHIRL_MOEP_*=0 = the shipped 0.2.5 path", mode);
        const nu::Flags other = fl(t8, mode, {{"MOEP_GU", "g64n128k64"}, {"MOEP_DN", "m128w4n32k64"}, {"MOEP_COMB", "2"}});
        NCHECK(other.moep_gu == 2 && other.moep_dn == 1 && other.moep_comb == 2, "8060s moe {}: another variant by name", mode);
        NCHECK(fl(t8, mode, {{"MOEP_GU", "bogus"}}).moep_gu == 0, "8060s moe {}: an unknown name is off", mode);
    }
    // the env still works where the default is off (the R9700 table has no moep kernels: ops::chooseMoe keeps its plan)
    NCHECK(fl(target(card("r9700"), "moe", false), "balance", {{"MOEP_GU", "g64n64k64"}}).moep_gu == 1, "r9700: MOEP_GU by name");
    // gfx1151 with non-MXFP4 routed experts: off
    nu::Target tq = t8;
    tq.moe_mxfp4 = false;
    NCHECK(fl(tq, "balance", {}).moep_gu == 0 && fl(tq, "balance", {}).moep_dn == 0 && fl(tq, "balance", {}).moep_comb == 0,
           "8060s, non-MXFP4 experts: off");
}

// v0251-a7pol (B26-41): WHIRL_MOEP_COMB by the MODE (owner mode standard 2026-10-26), in the bug-fastonly refusal form:
// 5 (atomic scatter, f32 order varies run to run) is fast-only; 3 / 4 (f16 expert outputs = balance item f16
// intermediates) never in precise, also with WHIRL_Q8DEC=1 (which clears prec_dec in precise). A refused value leaves
// the mode's own combine (the 8060S default 1 or 0), bitsKey / overrides / logLine as without it, and one Plan::ignored
// note. Expected to FAIL on cfce7ea: every refusal row (the flag took the value; no note).
static void testMoepCombMode() {
    auto planOf = [](const char* mode, const nu::Target& t, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    const std::string atomic = "the atomic-scatter MoE combine (f32 order varies run to run) is fast-only";
    const std::string f16 = "f16 MoE expert outputs are a balance item";
    for (const CardFake& c : k_cards)
        for (const char* model : {"swift", "moe"}) {
            const nu::Target t = target(c, model, false);
            for (const char* q8 : {"", "1"}) {  // "": the mode's decode activations; "1": WHIRL_Q8DEC=1 (prec_dec off in precise)
                std::vector<std::pair<const char*, const char*>> extra;
                if (*q8) extra.emplace_back("Q8DEC", q8);
                for (const char* mode : {"precise", "balance", "fast"}) {
                    const std::string cell = std::format("B26-41 {} {} {} q8dec='{}'", c.name, model, mode, q8);
                    const nu::Plan base = planOf(mode, t, extra);
                    for (const char* v : {"0", "1", "2", "3", "4", "5", "5z"}) {
                        auto vals = extra;
                        vals.emplace_back("MOEP_COMB", v);
                        const nu::Plan p = planOf(mode, t, vals);
                        const unsigned want = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
                        const bool precise = std::string_view(mode) == "precise", fast = std::string_view(mode) == "fast";
                        const bool refuse = (want == 5 && !fast) || ((want == 3 || want == 4) && precise);
                        if (refuse) {
                            const std::string why = want == 5 ? atomic + std::format(" ({} never takes it)", mode) : f16 + " (precise never takes them)";
                            NCHECK(p.flags.moep_comb == base.flags.moep_comb, "{}: WHIRL_MOEP_COMB={} must keep the mode's combine {} (got {})", cell, v,
                                   base.flags.moep_comb, p.flags.moep_comb);
                            NCHECK(p.bitsKey() == base.bitsKey() && p.overrides == base.overrides && nu::logLine(p) == nu::logLine(base),
                                   "{}: WHIRL_MOEP_COMB={} changed the plan", cell, v);
                            NCHECK(p.ignored == std::vector<std::string>{std::format("WHIRL_MOEP_COMB={} ignored: {}", v, why)}, "{}: WHIRL_MOEP_COMB={} note ({})",
                                   cell, v, p.ignored.empty() ? std::string("none") : p.ignored.front());
                        } else {
                            NCHECK(p.flags.moep_comb == want && p.ignored.empty(), "{}: WHIRL_MOEP_COMB={} must apply (comb {}, {} notes)", cell, v,
                                   p.flags.moep_comb, p.ignored.size());
                        }
                    }
                }
            }
        }
    // the default plan is untouched: 8060S MoE MXFP4 = 1 / 3 / 1 in every mode (testMoepDefault), no notes without the variable
    const nu::Target t8 = target(card("8060s"), "moe", false);
    for (const char* mode : {"precise", "balance", "fast"}) {
        const nu::Plan d = planOf(mode, t8, {});
        NCHECK(d.flags.moep_gu == 1 && d.flags.moep_dn == 3 && d.flags.moep_comb == 1 && d.ignored.empty(), "B26-41 8060s moe {}: default plan changed", mode);
    }
}

// v0252-moedf16 (PLAN V1-76): on the R9700 (gfx1201 code object) with MXFP4 routed experts, balance and fast default to
// moep_comb 3 = the down projection's output moe_yd stored f16 + moe_combine4h (balance item h16, f16 intermediates);
// precise stays 0 (f32 moe_yd, bitsKey unchanged), the 8060S plan and dense models unchanged. A/B: WHIRL_MOEP_COMB=0 (A, the
// f32 path) / 3 (B); WHIRL_MOEP_COMB=3 in precise is refused (B26-41 form) and keeps 0; =5 in balance keeps the default 3.
static void testMoeYdhDefault() {
    auto planOf = [](const char* mode, const nu::Target& t, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    const nu::Target t97 = target(card("r9700"), "moe", false), t8 = target(card("8060s"), "moe", false);
    const nu::Target s97 = target(card("r9700"), "swift", false);
    for (const char* mode : {"balance", "fast"}) {
        const nu::Plan d = planOf(mode, t97, {});
        NCHECK(d.flags.moep_comb == 3 && d.flags.moep_gu == 0 && d.flags.moep_dn == 0 && d.ignored.empty(), "moedf16 r9700 moe {}: default comb 3 (got {})",
               mode, d.flags.moep_comb);
        const nu::Plan a = planOf(mode, t97, {{"MOEP_COMB", "0"}});
        NCHECK(a.flags.moep_comb == 0 && a.ignored.empty() && a.bitsKey() != d.bitsKey(), "moedf16 r9700 moe {}: WHIRL_MOEP_COMB=0 = the A arm (f32 moe_yd)",
               mode);
        NCHECK(planOf(mode, t97, {{"MOEP_COMB", "3"}}).bitsKey() == d.bitsKey(), "moedf16 r9700 moe {}: WHIRL_MOEP_COMB=3 = the default", mode);
        NCHECK(planOf(mode, s97, {}).flags.moep_comb == 0, "moedf16 r9700 swift {}: dense model unchanged", mode);
    }
    // balance: the fast-only 5 is refused back to the balance default 3
    NCHECK(planOf("balance", t97, {{"MOEP_COMB", "5"}}).flags.moep_comb == 3, "moedf16 r9700 moe balance: COMB=5 refused -> 3");
    // precise: f32 moe_yd, also with WHIRL_Q8DEC=1 and with WHIRL_MOEP_COMB=3 (refused)
    for (const bool q8 : {false, true}) {
        std::vector<std::pair<const char*, const char*>> ex;
        if (q8) ex.emplace_back("Q8DEC", "1");
        const nu::Plan p = planOf("precise", t97, ex);
        NCHECK(p.flags.moep_comb == 0, "moedf16 r9700 moe precise q8dec={}: comb 0 (got {})", q8, p.flags.moep_comb);
        ex.emplace_back("MOEP_COMB", "3");
        const nu::Plan p3 = planOf("precise", t97, ex);
        NCHECK(p3.flags.moep_comb == 0 && p3.bitsKey() == p.bitsKey() && p3.ignored.size() == 1, "moedf16 r9700 moe precise q8dec={}: COMB=3 refused", q8);
    }
    // the 8060S: unchanged in every mode (kMoepDefault 1 / 3 / 1)
    for (const char* mode : {"precise", "balance", "fast"}) {
        const nu::Plan d = planOf(mode, t8, {});
        NCHECK(d.flags.moep_gu == 1 && d.flags.moep_dn == 3 && d.flags.moep_comb == 1, "moedf16 8060s moe {}: plan changed", mode);
    }
}

// v0251-a11port (feat026-router): the fused MoE router is the Radeon 8060S default (Target::gfx1151) for MoE models in
// every mode (bit-identical); v0252-c5port: the R9700 default too (its code object has the kernel since then), so on for
// MoE models on both cards and off on dense models; the 8060S rows are the 0.2.5.1 ones (per-card selector check below).
// WHIRL_MOE_RFUSE=0 gives the 0.2.5 path everywhere, "1" turns the flag on anywhere. Not labelled +overrides (no A/B
// switch: it is the default), FO-3 / FO-4 below. The R9700 MoE rows fail before the C5 port (the 0.2.5.1 default).
static void testRfuseDefault() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const nu::Target t = target(c, model, false);
                const bool on = std::string_view(model) == "moe";
                const nu::Plan p = pl(t, mode, {});
                NCHECK(p.flags.moe_rfuse == on, "moe_rfuse default {} {} {}: {} (want {})", c.name, model, mode, p.flags.moe_rfuse, on);
                // per-card selector check (v0252-c5port, DISCOVERY 2e C5): the 8060S rows equal the 0.2.5.1 rule
                // (Target::gfx1151 && Target::moe), so the gfx1151 plan does not move with the R9700 default
                if (std::string_view(c.name) == "8060s")
                    NCHECK(p.flags.moe_rfuse == (t.gfx1151 && t.moe), "moe_rfuse 8060s {} {}: the 0.2.5.1 default moved", model, mode);
                const nu::Plan off = pl(t, mode, {{"MOE_RFUSE", "0"}});
                NCHECK(!off.flags.moe_rfuse, "moe_rfuse {} {} {}: WHIRL_MOE_RFUSE=0 = the 0.2.5 path", c.name, model, mode);
                const nu::Plan one = pl(t, mode, {{"MOE_RFUSE", "1"}});
                NCHECK(one.flags.moe_rfuse, "moe_rfuse {} {} {}: WHIRL_MOE_RFUSE=1 on", c.name, model, mode);
                // the flag changes nothing else in the plan: its bitsKey is the off plan's with moe_rfuse=1
                nu::Plan a = off;
                a.flags.moe_rfuse = true;
                NCHECK(a.bitsKey() == one.bitsKey() && nu::logLine(off) == nu::logLine(one) && !one.lossy_override && one.overrides == off.overrides,
                       "moe_rfuse {} {} {}: WHIRL_MOE_RFUSE changes only the moe_rfuse flag", c.name, model, mode);
            }
}

// v0252-c1port (C1, feat026-attnkg97 / -97k): WHIRL_ATTN_KGVAR is a load-time variable of class B (an EnvValues field,
// not an A/B switch: var 1 attn_kgv and var 3 attn_kgk are designed bit-identical to attn_kg); never lossy, no override
// note, and a valid value changes only the attn_kg_var flag.
// v0252-c1acc (REVIEW-v0252-c1 C-1 / C-3): the default is kAttnKgVarAuto on the R9700 card (gfx1201: ops::kAttnKgAuto
// picks the variant per NP x KV path) and 0 on the 8060S (no attn_kg there), every model x mode; "auto" and a whole
// decimal 0 / 1 / 3 override it; any other value (2, 4, "", "x", "-1", "3x") keeps the default and Plan::ignored carries
// exactly one note naming it (before c1acc it silently became 0). Fails before c1acc (default 0 everywhere; the
// test-first stub parseKgVar accepts nothing).
static void testKgVarFlag() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    const std::uint32_t A = nu::kAttnKgVarAuto;
    NCHECK(A == 255, "kAttnKgVarAuto {}", A);
    const std::pair<const char*, std::uint32_t> valid[] = {{"1", 1}, {"3", 3}, {"03", 3}, {"0", 0}, {"auto", A}};
    const char* const invalid[] = {"2", "4", "", "x", "-1", "3x", "AUTO", " 3"};
    for (const auto& [v, want] : valid) NCHECK(nu::parseKgVar(v) == want, "parseKgVar('{}')", v);
    for (const char* v : invalid) NCHECK(!nu::parseKgVar(v).has_value(), "parseKgVar('{}') must be invalid", v);
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const nu::Target t = target(c, model, false);
                const std::uint32_t dflt = t.gfx1151 ? 0u : A;
                const nu::Plan d = pl(t, mode, {});
                NCHECK(d.flags.attn_kg_var == dflt, "attn_kg_var default {} {} {}: {} (want {})", c.name, model, mode, d.flags.attn_kg_var, dflt);
                for (const auto& [v, want] : valid) {
                    const nu::Plan p = pl(t, mode, {{"ATTN_KGVAR", v}});
                    NCHECK(p.flags.attn_kg_var == want, "attn_kg_var {} {} {}: WHIRL_ATTN_KGVAR='{}' -> {} (want {})", c.name, model, mode, v,
                           p.flags.attn_kg_var, want);
                    nu::Plan a = d;
                    a.flags.attn_kg_var = want;
                    NCHECK(a.bitsKey() == p.bitsKey() && nu::logLine(p) == nu::logLine(d) && p.overrides == d.overrides &&
                               p.lossy_override == d.lossy_override && p.ignored.empty(),
                           "attn_kg_var {} {} {}: WHIRL_ATTN_KGVAR='{}' changes only the attn_kg_var flag", c.name, model, mode, v);
                }
                for (const char* v : invalid) {
                    const nu::Plan p = pl(t, mode, {{"ATTN_KGVAR", v}});
                    NCHECK(p.flags.attn_kg_var == dflt, "attn_kg_var {} {} {}: invalid WHIRL_ATTN_KGVAR='{}' -> {} (want the default {})", c.name, model,
                           mode, v, p.flags.attn_kg_var, dflt);
                    std::size_t notes = 0;
                    for (const std::string& n : p.ignored) notes += n.rfind(std::string("WHIRL_ATTN_KGVAR=") + v + " ignored", 0) == 0;
                    NCHECK(notes == 1 && p.ignored.size() == 1 && p.bitsKey() == d.bitsKey() && p.overrides == d.overrides,
                           "attn_kg_var {} {} {}: invalid WHIRL_ATTN_KGVAR='{}': one Plan::ignored note, nothing else changes", c.name, model, mode, v);
                }
            }
}

// v0252-c3port (feat026-moedec, gfx1201 precise decode): WHIRL_MOE_FUSE / WHIRL_GEMVH_F32 were off by default on both
// cards, both models and every mode; "1" turns only that flag on (bitsKey = the off plan's with the flag set, same log
// line, no +overrides, never lossy); "0" = off. The kernels exist only in the gfx1201 code object (ops::chooseMoeFuse /
// ops::gemvhF32 return the old path on gfx1151). Fails before the port (loadFlags does not read the two names).
// v0252-c3bdef (PLAN V1-36; q7 9505 c3bgold=ok, q7 9507 screen KEEP: Ornith precise decode plain +4.26 % / MTP +5.93 %,
// hash diffs 0): moe_fuse is the default on the R9700 card (gfx1201 code object, Target::gfx1151 false) for MoE models,
// every mode (the flag is bitwise; ops::chooseMoeFuse takes the fused launch only on the precise f32 decode chain, see
// ops_moe_cases.inc), and stays off on dense models and on the 8060S (the 0.2.5.1 / v0252-c3port rule: the gfx1151 plan
// does not move). gemvh_f32 stays off everywhere (C3 screen: Swift -10 %). WHIRL_MOE_FUSE=0 / =1 still win both ways and
// change only the moe_fuse flag; the default is not an override (no +overrides, never lossy). The R9700 MoE default rows
// fail before c3bdef (default off).
// v0252-ws4r97b (PLAN V1-87 phase 2): WHIRL_ATTN_WS4R is a load-time variable of class B (an EnvValues field: the gfx1201
// attn_ws4r1 / attn_ws4r2 kernels are bit-identical to attn_wsplit1 / attn_wsplit2, kt ws4r). The flag attn_ws4r is
// kAttnWs4rAuto (255) on every card, model and mode: device::Profile::attn_ws4r_q8 / _f16 decide (gfx1201 q8 -> b, f16 ->
// plain; gfx1151 off). auto and a whole decimal 0 / 1 / 2 override it (0 = attn_wsplit, 1 = plain, 2 = b) and change
// only that flag (no +overrides, never lossy); any other value keeps auto with exactly one Plan::ignored note. Fails
// before phase 2 (the test-first stub parseWs4r accepts nothing).
static void testWs4rFlag() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    const std::uint32_t A = nu::kAttnWs4rAuto;
    NCHECK(A == 255, "kAttnWs4rAuto {}", A);
    const std::pair<const char*, std::uint32_t> valid[] = {{"0", 0}, {"1", 1}, {"2", 2}, {"02", 2}, {"auto", A}};
    const char* const invalid[] = {"3", "255", "", "x", "-1", "2b", "AUTO", " 1", "b"};
    for (const auto& [v, want] : valid) NCHECK(nu::parseWs4r(v) == want, "parseWs4r('{}')", v);
    for (const char* v : invalid) NCHECK(!nu::parseWs4r(v).has_value(), "parseWs4r('{}') must be invalid", v);
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const nu::Target t = target(c, model, false);
                const nu::Plan d = pl(t, mode, {});
                NCHECK(d.flags.attn_ws4r == A, "attn_ws4r default {} {} {}: {} (want auto)", c.name, model, mode, d.flags.attn_ws4r);
                NCHECK(!d.lossy_override && d.ignored.empty(), "attn_ws4r default {} {} {}: the default is not an override", c.name, model, mode);
                for (const auto& [v, want] : valid) {
                    const nu::Plan p = pl(t, mode, {{"ATTN_WS4R", v}});
                    NCHECK(p.flags.attn_ws4r == want, "attn_ws4r {} {} {}: WHIRL_ATTN_WS4R='{}' -> {} (want {})", c.name, model, mode, v, p.flags.attn_ws4r,
                           want);
                    nu::Plan a = d;
                    a.flags.attn_ws4r = want;
                    NCHECK(a.bitsKey() == p.bitsKey() && nu::logLine(p) == nu::logLine(d) && p.overrides == d.overrides &&
                               p.lossy_override == d.lossy_override && p.ignored.empty(),
                           "attn_ws4r {} {} {}: WHIRL_ATTN_WS4R='{}' changes only the attn_ws4r flag", c.name, model, mode, v);
                }
                for (const char* v : invalid) {
                    const nu::Plan p = pl(t, mode, {{"ATTN_WS4R", v}});
                    NCHECK(p.flags.attn_ws4r == A, "attn_ws4r {} {} {}: invalid WHIRL_ATTN_WS4R='{}' -> {} (want auto)", c.name, model, mode, v,
                           p.flags.attn_ws4r);
                    std::size_t notes = 0;
                    for (const std::string& n : p.ignored) notes += n.rfind(std::string("WHIRL_ATTN_WS4R=") + v + " ignored", 0) == 0;
                    NCHECK(notes == 1 && p.ignored.size() == 1 && p.bitsKey() == d.bitsKey() && p.overrides == d.overrides,
                           "attn_ws4r {} {} {}: invalid WHIRL_ATTN_WS4R='{}': one Plan::ignored note, nothing else changes", c.name, model, mode, v);
                }
            }
}

static void testMoedecDefault() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const nu::Target t = target(c, model, false);
                const bool fuse_on = std::string_view(c.name) == "r9700" && std::string_view(model) == "moe";
                const nu::Plan d = pl(t, mode, {});
                NCHECK(d.flags.moe_fuse == fuse_on && !d.flags.gemvh_f32, "moedec default {} {} {}: moe_fuse {} (want {}), gemvh_f32 {} (want off)",
                       c.name, model, mode, d.flags.moe_fuse, fuse_on, d.flags.gemvh_f32);
                // per-card selector check: the 8060S rows keep the v0252-c3port (= 0.2.5.1 plan) rule, off
                if (std::string_view(c.name) == "8060s")
                    NCHECK(t.gfx1151 && !d.flags.moe_fuse, "moedec 8060s {} {}: the gfx1151 default moved", model, mode);
                NCHECK(!d.lossy_override && d.ignored.empty(), "moedec default {} {} {}: the default is not an override", c.name, model, mode);
                for (const char* name : {"MOE_FUSE", "GEMVH_F32"}) {
                    const bool fuse = std::string_view(name) == "MOE_FUSE";
                    const nu::Plan off = pl(t, mode, {{name, "0"}});
                    const nu::Plan one = pl(t, mode, {{name, "1"}});
                    NCHECK(!(fuse ? off.flags.moe_fuse : off.flags.gemvh_f32), "moedec {} {} {}: WHIRL_{}=0 off", c.name, model, mode, name);
                    NCHECK(fuse ? one.flags.moe_fuse : one.flags.gemvh_f32, "moedec {} {} {}: WHIRL_{}=1 on", c.name, model, mode, name);
                    nu::Plan a = off;
                    (fuse ? a.flags.moe_fuse : a.flags.gemvh_f32) = true;
                    // the default plan equals the "=1" plan where the default is on (R9700 MoE moe_fuse), else the "=0" one
                    const nu::Plan& same = (fuse && fuse_on) ? one : off;
                    NCHECK(same.bitsKey() == d.bitsKey() && a.bitsKey() == one.bitsKey() && nu::logLine(off) == nu::logLine(one) &&
                               nu::logLine(d) == nu::logLine(off) && !one.lossy_override && one.overrides == off.overrides &&
                               d.overrides == off.overrides,
                           "moedec {} {} {}: WHIRL_{} changes only its flag", c.name, model, mode, name);
                }
            }
}

void testNumericsPlan(int& pass, int& fail) {
    testPlanTable();
    testApplyFrontEqualsPlan();  // arch025-N2b (replaces testUnwiredCallerUnchanged: no caller passes Target::gdn_strix now)
    testBitsKeyReflection();
    testFlagScope();
    testFrontEnds();
    testLoadRules();
    testMoepDefault();  // v0251-a7port (fails before the port: the plan has no gfx1151 default)
    testMoepCombMode();  // v0251-a7pol B26-41 (the refusal rows fail on cfce7ea)
    testMoeYdhDefault();  // v0252-moedf16 (fails before the change: the R9700 balance / fast comb is 0)
    testRfuseDefault();  // v0251-a11port (fails before the port: the plan has no gfx1151 default)
    testKgVarFlag();  // v0252-c1port; v0252-c1acc auto default + invalid-value note (fails before c1acc)
    testWs4rFlag();  // v0252-ws4r97b (fails before phase 2: the stub parseWs4r accepts nothing)
    testMoedecDefault();  // v0252-c3port (the "=1 on" rows fail before the port); v0252-c3bdef (the R9700 MoE default rows fail before c3bdef)
    testN2bR1();  // arch025-n2b-r1
    testSwitchZeroIsOff();  // audit-fastonly-overrides FO-5 (fails before the fix)
    testFastOnlyOverrides();  // audit-fastonly-overrides FO-1..4 (FO-1 / FO-2 / FO-3 fail before the fix)
    testFastOnlyRefusal();  // bug-fastonly: refusal notes, fast unchanged
    testRelaxMoeArms();  // v0252-f2 (fails before the port: no arm is known)
    pass += g_p;
    fail += g_f;
}
