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
    // v0252-f5 (moeskip): both code objects have moe_topk_skip / moe_skip_apply / moe_route_skip / moe_combine_skip; the
    // per-card fast default rule as shipped (moeskip_fast_default_r9700 / _8060s)
    t.k_moe_skip = true;
    t.moeskip_default = t.gfx1151 ? nu::moeskip_fast_default_8060s : nu::moeskip_fast_default_r9700;
    t.attn_sx_profile = c.prof.attn_sx_balance;  // v0252-strixc2 (tune.cpp numericsTarget)
    t.attn_i4qk_profile = c.prof.attn_i4qk_fast;  // v0252-8sfp (tune.cpp numericsTarget)
    t.attn_i4pf_profile = c.prof.attn_i4_qlds_pf;  // v0252-8sattn3 (tune.cpp numericsTarget)
    t.moep_comb_fast_skip_profile = c.prof.moep_comb_fast_skip;  // v0252-8smoe (A, tune.cpp numericsTarget)
    t.moep_guw_profile = c.prof.moep_guw;  // v0252-8smoe (B, tune.cpp numericsTarget)
    t.moep_dhs_profile = c.prof.moep_dhs;  // v0252-8smoe2 (C, tune.cpp numericsTarget)
    t.moep_dna_profile = c.prof.moep_dna;  // v0252-8smoe2 (D)
    t.moep_fq_profile = c.prof.moep_fq;    // v0252-8smoe2 (E)
    t.gemvh_grp_profile = c.prof.gemvh_group_dense;  // v0252-8sdec (tune.cpp numericsTarget)
    t.dq4_order_profile = c.prof.attn_dq4_row_order;  // v0252-8sq4 (F8 order, tune.cpp numericsTarget)
    t.dq4_min_keys_profile = c.prof.dq4_min_keys;     // v0252-8sq4 (F8 split)
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

// v0252-ornkv2 (PLAN V1-113): the S0b-2 num_plan of an 8060S MoE balance / fast cell -> the same line with kvq8 enabled
// (int8 KV for MoE models on gfx1151; fast keeps q4 through kvq4, which wins in chooseKv): "kvq8 (MoE models keep f16
// KV)" leaves the skipped list and "kvq8 (<q8h note>)" enters the enabled list in item order (before specsample).
// false = the old text is missing. (From the v0252-ornkv port, which did it on both cards.)
bool ornkvRepin(std::string& l) {
    const std::string old_sk = "kvq8 (MoE models keep f16 KV)";
    const std::size_t at = l.find(old_sk);
    if (at == std::string::npos) return false;
    if (l.compare(at + old_sk.size(), 2, ", ") == 0) l.erase(at, old_sk.size() + 2);           // first of several
    else if (at >= 2 && l.compare(at - 2, 2, ", ") == 0) l.erase(at - 2, old_sk.size() + 2);   // after another
    else l.replace(at, old_sk.size(), "none");                                                 // the only one
    const std::size_t ss = l.find(", specsample (");
    if (ss == std::string::npos) return false;
    l.insert(ss, ", kvq8 (q8h KV (int8 + f16 scale / 32, Hadamard-rotated q/k))");
    return true;
}

// v0252-ornkv2: Ornith (MoE) balance on the 8060S: kvq8 on by default (q8h KV), the cache identity carries it (bitsKey
// and logLine - the KV-tier fingerprint's numerics field - differ from the f16 plan, so a tier entry written with f16 KV
// is never restored as q8h, nor the reverse; kv_name is in the fingerprint as well, tests/cache/fingerprint_cases.inc),
// WHIRL_KV=f16 turns it off as on a dense model (the way back to f16), WHIRL_KV=q8h (the screen's B arm) is the new
// default plan, precise unchanged; fast: kvq8 listed as enabled (as dense fast), the format stays q4 (contract_tests
// testKvChoiceMoe8060s). The R9700: unchanged (kvq8 skipped, WHIRL_KV=f16 no override).
// (the Request and the Target live in this helper's frame: testOrnkvPlan in one piece used 18584 bytes of stack, over
// MSVC /analyze's 16 KiB, C6262.) kv: nullptr = WHIRL_KV unset, else its value (Target::kv_explicit as the loader sets it)
nu::Plan ornkvPlan(const char* cn, const char* mode, const char* kv) {
    nu::Request r = nu::parseMode(mode, "command line");
    r.load_overrides = true;
    if (kv != nullptr) *r.env.find("KV") = kv;
    return nu::plan(r, target(card(cn), "moe", kv != nullptr));
}

void testOrnkvPlan8060s() {
    const nu::Plan pb = ornkvPlan("8060s", "balance", nullptr);
    NCHECK(pb.on(nu::Item::kvq8), "8060s Ornith balance: kvq8 off");
    NCHECK(nu::logLine(pb).find("MoE models keep f16 KV") == std::string::npos, "8060s Ornith balance: old skip note");
    NCHECK(pb.bitsKey().find("kvq8=1;") != std::string::npos, "8060s Ornith balance bitsKey without kvq8=1");
    const nu::Plan pf = ornkvPlan("8060s", "balance", "f16");
    NCHECK(!pf.on(nu::Item::kvq8) && pf.overrides.size() == 1 && pf.overrides[0] == "WHIRL_KV=f16: kvq8 off" && !pf.lossy_override,
           "8060s Ornith balance WHIRL_KV=f16: kvq8 not turned off ({} overrides)", pf.overrides.size());
    NCHECK(pf.bitsKey() != pb.bitsKey() && nu::logLine(pf) != nu::logLine(pb), "8060s Ornith balance: f16 and q8h plans share a cache key");
    const nu::Plan pq = ornkvPlan("8060s", "balance", "q8h");
    NCHECK(pq.on(nu::Item::kvq8) && pq.overrides.empty() && pq.bitsKey() == pb.bitsKey(), "8060s Ornith balance WHIRL_KV=q8h != default");
    NCHECK(!ornkvPlan("8060s", "precise", nullptr).on(nu::Item::kvq8), "8060s Ornith precise: kvq8 on");
    const nu::Plan pfa = ornkvPlan("8060s", "fast", nullptr);
    NCHECK(pfa.on(nu::Item::kvq8) && pfa.on(nu::Item::kvq4), "8060s Ornith fast: kvq8 + kvq4 listed (the format stays q4)");
}

void testOrnkvPlanR9700() {
    const nu::Plan pb = ornkvPlan("r9700", "balance", nullptr);
    NCHECK(!pb.on(nu::Item::kvq8) && nu::logLine(pb).find("kvq8 (MoE models keep f16 KV)") != std::string::npos &&
               pb.bitsKey().find("kvq8=0;") != std::string::npos,
           "r9700 Ornith balance: the plan changed (f16 KV must stay)");
    const nu::Plan pf = ornkvPlan("r9700", "balance", "f16");
    NCHECK(!pf.on(nu::Item::kvq8) && pf.overrides.empty() && pf.bitsKey() == pb.bitsKey(), "r9700 Ornith balance WHIRL_KV=f16: plan changed");
    NCHECK(!ornkvPlan("r9700", "fast", nullptr).on(nu::Item::kvq8), "r9700 Ornith fast: kvq8 on");
}

void testOrnkvPlan() {
    testOrnkvPlan8060s();
    testOrnkvPlanR9700();
}

// env names a cell sets that no plan field reads (checkGemvBitwise only: not a flag, prints a warning)
const std::set<std::string> k_not_plan_env = {"GEMV_BITWISE_FLOAT"};

void testPlanTable() {
    std::size_t n_cells = 0, n_fields = 0, n_repin = 0, n_ornkv = 0, n_f5 = 0, n_b5 = 0;
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
        auto want = splitKv(c.kv);
        // v0252-ornkv2: the S0b-2 dump predates int8 KV for MoE models on the 8060S; its MoE cells without WHIRL_KV
        // (balance and fast) expect kvq8 moved from "skipped (MoE models keep f16 KV)" to "enabled" - one declared text
        // edit of num_plan, no other key (the KV format itself is chooseKv's, contract_tests testKvChoiceMoe8060s; the
        // moep flags have no dump key). The R9700 MoE cells are compared unchanged.
        if (std::string_view(c.card) == "8060s" && std::string_view(c.model) == "moe" && std::string_view(c.mode) != "precise" && !kv_explicit) {
            ++n_ornkv;
            NCHECK(ornkvRepin(want["num_plan"]), "{} {} (ornkv2 re-pin): S0b-2 num_plan has no 'kvq8 (MoE models keep f16 KV)'", c.card, c.cell);
        }
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
        for (const auto& [k, v0] : want) {
            ++n_fields;
            bool known = false;
            const std::string got = keyOf(p, k, known);
            if (k.rfind("num_plan.override[", 0) == 0) ++n_over;
            NCHECK(known, "{} {}: key {} not decided by the plan", c.card, c.cell, k);
            // v0252-f5 (moeskip opt-in, fast_optin_items): a plain --fast no longer requests moeskip, so its old "skipped"
            // entry left the fast log line. Description only: the item was off and stays off (no flag, item state or bitsKey
            // change); every other character of the dump's line is still compared. RE-PIN at the next S0b-2 GPU dump.
            std::string v = v0;
            if (k == "num_plan" && std::string_view(c.mode) == "fast") {
                static const std::string old_skip = "moeskip (not yet implemented (balance behaviour)), ";
                const std::size_t at = v.find(old_skip);
                NCHECK(at != std::string::npos, "{} {}: the dump's fast line has no '{}' (re-pinned already? drop this rule)", c.card, c.cell, old_skip);
                if (at != std::string::npos) {
                    v.erase(at, old_skip.size());
                    ++n_f5;
                }
            }
            // v0252-b5 round 3 (PLAN V1-116): moeskip k6 is the Ornith (MoE, MXFP4 experts) fast default on both cards, so the
            // two plain MoE fast cells gain one enabled entry, "moeskip (... [static top 6] ...)", that the S0b-2 dump predates.
            // Declared re-pin: that entry (and nothing else) is taken out of the plan's line before the compare; the flags
            // moe_skip_thr / moe_skip_min / moep_comb have no dump key (testMoeSkipDefault pins them). RE-PIN at the next dump.
            std::string got_cmp = got;
            if (k == "num_plan" && std::string_view(c.mode) == "fast" && std::string_view(c.model) == "moe" && std::string_view(c.cell).rfind("cli_", 0) == 0) {
                const std::size_t at = got_cmp.find(", moeskip (");
                const std::size_t top = got_cmp.find("[static top 6]", at == std::string::npos ? 0 : at);
                const std::size_t end = top == std::string::npos ? std::string::npos : got_cmp.find(')', top);
                NCHECK(at != std::string::npos && end != std::string::npos, "{} {} (b5 re-pin): the MoE fast line has no enabled 'moeskip (... [static top 6] ...)': {}",
                       c.card, c.cell, got);
                if (at != std::string::npos && end != std::string::npos) {
                    got_cmp.erase(at, end + 1 - at);
                    ++n_b5;
                }
            }
            NCHECK(got_cmp == v, "{} {}: {} = '{}', S0b-2 dump '{}'", c.card, c.cell, k, got, v);
        }
        NCHECK(p.overrides.size() == n_over, "{} {}: {} overrides, dump {}", c.card, c.cell, p.overrides.size(), n_over);
        NCHECK(p.ignored.empty(), "{} {}: unexpected refusal note '{}'", c.card, c.cell, p.ignored.empty() ? std::string() : p.ignored[0]);
        // every flag with a dump key is in every cell (nothing compared by omission)
        for (const nu::FlagInfo& fi : nu::flagTable())
            if (*fi.dump_key) NCHECK(want.count(fi.dump_key) == 1, "{} {}: dump has no {}", c.card, c.cell, fi.dump_key);
    }
    NCHECK(n_repin == 10, "bug-fastonly re-pin cells {} (expected the 10 env_RELAX* balance cells)", n_repin);
    NCHECK(n_ornkv == 2, "ornkv2 re-pin cells {} (expected cli_moe_balance / cli_moe_fast of the 8060s)", n_ornkv);
    NCHECK(n_f5 == 4, "v0252-f5 fast cells without the old moeskip entry {} (expected the 4 plain fast cells)", n_f5);
    NCHECK(n_b5 == 2, "v0252-b5 MoE fast cells with the default moeskip k6 entry {} (expected cli_moe_fast of both cards)", n_b5);
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
    // v0252-8sdec: WHIRL_GEMVH_GRP -> gemvh_grp, the envFlag rule, never lossy (default off on this R9700 target)
    NCHECK(!fl({}).flags.gemvh_grp && fl({{"GEMVH_GRP", "1"}}).flags.gemvh_grp && fl({{"GEMVH_GRP", ""}}).flags.gemvh_grp &&
               !fl({{"GEMVH_GRP", "0"}}).flags.gemvh_grp && !fl({{"GEMVH_GRP", "1"}}, "precise").lossy_override,
           "GEMVH_GRP: envFlag rule (gemvh_grp)");
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

// v0252-kgcrash (PLAN V1-127): AMD_SERIALIZE_KERNEL with bit 2 ("wait for completion after enqueue", values 2 / 3) makes the
// HIP runtime corrupt the host heap inside hipGraphLaunch (scratch/v0252_graphfix/iso_graph, a WHIRL-free repro). The CLI then
// turns the decode graph off through the WHIRL_NO_GRAPH path and says so once; an explicit WHIRL_NO_GRAPH wins; unset / 0 / 1 /
// unparsable leave everything as today. Fails to build before the change (no serializeKernelNoGraph).
void testSerializeKernelNoGraph() {
    struct Row {
        nu::Raw amd, no_graph;
        bool note, graph;
    };
    const Row rows[] = {
        {std::nullopt, std::nullopt, false, true}, {std::string("0"), std::nullopt, false, true},
        {std::string("1"), std::nullopt, false, true}, {std::string("2"), std::nullopt, true, false},
        {std::string("3"), std::nullopt, true, false}, {std::string("garbage"), std::nullopt, false, true},
        {std::string(""), std::nullopt, false, true}, {std::string("6"), std::nullopt, true, false},
        {std::string("0x2"), std::nullopt, true, false}, {std::string("3x"), std::nullopt, true, false},
        {std::string("3"), std::string("0"), false, true},   // explicit WHIRL_NO_GRAPH=0 wins (graph on, as today)
        {std::string("3"), std::string("1"), false, false},  // explicit WHIRL_NO_GRAPH=1: off as today, no extra line
        {std::nullopt, std::string("1"), false, false},     {std::string("0"), std::string("0"), false, true},
    };
    for (const Row& r : rows) {
        const std::string tag = std::format("AMD_SERIALIZE_KERNEL={} WHIRL_NO_GRAPH={}", r.amd ? "'" + *r.amd + "'" : "unset",
                                            r.no_graph ? "'" + *r.no_graph + "'" : "unset");
        nu::RuntimeValues v;
        v.no_graph = r.no_graph;
        const std::optional<std::string> note = nu::serializeKernelNoGraph(r.amd, v);
        NCHECK(note.has_value() == r.note, "kgcrash {}: note {}", tag, note.has_value());
        if (note) {
            NCHECK(note->find("AMD_SERIALIZE_KERNEL=" + *r.amd) != std::string::npos, "kgcrash {}: the line names the variable: {}", tag, *note);
            NCHECK(note->find("graph") != std::string::npos && note->find('\n') == std::string::npos, "kgcrash {}: one line on the graph: {}",
                   tag, *note);
            NCHECK(v.no_graph && *v.no_graph == "1", "kgcrash {}: takes the WHIRL_NO_GRAPH path", tag);
        } else {
            NCHECK(v.no_graph == r.no_graph, "kgcrash {}: WHIRL_NO_GRAPH unchanged", tag);
        }
        for (const nu::Front fr : {nu::Front::cli, nu::Front::server}) {
            nu::Plan p;
            nu::applyFront(p, fr, v, true);
            const bool want = fr == nu::Front::server ? false : r.graph;
            NCHECK(p.flags.use_graph == want, "kgcrash {} front {}: use_graph {}", tag, fr == nu::Front::cli ? "cli" : "server", p.flags.use_graph);
        }
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
        {"MOEP_GUW", 'B'},  // v0252-8smoe: the routing-aware wide moep gate + up variant, every variant the same bits (kt moemx)
        // v0252-8smoe2: C coalesced f16 epilogue and E fused-dequant twins store the base kernels' bits (kt moemx); D picks the
        // COMB 5 atomic down variant (the atomic class is fast-only through MOEP_COMB; its variable is 'L' like MOEP_COMB)
        {"MOEP_DHS", 'B'}, {"MOEP_DNA", 'L'}, {"MOEP_FQ", 'B'},
        {"MOE_RFUSE", 'B'},  // v0251-a11port: fused router, bit-identical to moe_logits_f32 + moe_topk (kt moemx, golden)
        {"ATTN_KGVAR", 'B'},  // v0252-c1port: attn_kgv / attn_kgk, designed bit-identical to attn_kg (kt attn, golden)
        // v0252-c3port (feat026-moedec): fused precise MoE decode / gemvh on f32 input, bit-identical (kt moefuse, golden)
        {"MOE_FUSE", 'B'},   {"GEMVH_F32", 'B'},
        {"GEMVH_GRP", 'B'},  // v0252-8sdec: grouped precise gemvh, bit-identical to gemvh per matrix (kt gemvhg, golden)
        {"ATTN_WS4R", 'B'},  // v0252-ws4r97b: gfx1201 attn_ws4r1 / 2[b], bit-identical to attn_wsplit1 / 2 (kt ws4r, golden)
        // v0252-l2split: another decode split partition (another f32 combine order): balance / fast only, precise ignores it
        {"ATTN_LSPLIT", 'L'},
        {"MOESKIP", 'F'},     // v0252-f5: moeskip, fast-only (applyFastOnlyOverride; drops routed experts)
        {"ATTN_SX", 'L'},  // v0252-strixc2: attn_sx<NP> (strix fattn-rdna35), not bitwise: balance only (testAttnSxFlag)
        {"ATTN_I4QK", 'F'},  // v0252-8sfp: int4 Q.K^T prefill attention, not bitwise: fast only (testAttnI4qkFlag)
        {"W4A4_SIM", 'F'},  // v0252-w4a4k2: W4A4 fake-quant prefill GEMM arm (research), output changes: fast only (testW4a4SimFlag)
        // v0252-8sq4 (F8): q4 KV decode attention switches, fast only (q4 KV exists only there; testDq4F8Flags). ORDER keeps the
        // partials (attn_dq4o), MINKEYS changes the partition (another combine order)
        {"DQ4_ORDER", 'F'}, {"DQ4_MINKEYS", 'F'},
        {"DQ4_SPLITS", 'F'},  // v0252-8sq5: the q4 decode split count (profile table / today's rule / forced), fast only (testDq4SplitsFlag)
        // v0252-8sattn3: the int4 prefill attention with Q codes in LDS + K / V prefetch, bit-identical to the int4 kernel but
        // only reachable where the fast-only attn_i4qk is on (testAttnI4pfFlag)
        {"ATTN_I4PF", 'F'},
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

// v0252-b5 round 3, item (e) (PLAN V1-97): the fast DENSE relaxacc arm per GPU, the same mechanism as the MoE arms
// (Target::relax_dense_arm = the name the loader fills, relaxDenseDefaultName(gfx1151); "" = the Flags defaults k 4 /
// alpha 0.1, every hand-built Target). Shipped: the R9700 (gfx1201) takes k8a02 = K 8, alpha 0.02 for Swift fast; the
// 8060S keeps "" (its own screen is queued).
//  RD-1 the arm and the shipped names; RD-2 dense fast with the arm: k / alpha / plan line, drafts untouched, not an
//  override; without an arm today's plan; RD-3 WHIRL_RELAX_K / _ALPHA still override (K=4 ALPHA=0.1 = the plan before the
//  arm: the A arm), WHIRL_RELAX=0 off; RD-4 precise / balance (also with WHIRL_RELAX=1) and a MoE model never see it.
// Fails to build before the change (no relax_dense_arm / relaxDenseArm / relaxDenseDefaultName).
void testRelaxDenseArmCard(const CardFake& c) {
    auto planOf = [](const char* mode, const nu::Target& t, std::vector<std::pair<const char*, const char*>> vals, bool lo = true) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = lo;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    const bool g51 = c.prof.numerics_8060s;
    // v0252-8sdec (PLAN V1-143): both cards ship k8a02 for dense fast (the 8060S joined; its screen progress/8sdec-screen.json)
    const std::uint32_t wk = 8u;
    const float wa = 0.02f;
    (void)g51;
    const nu::Target t0 = target(c, "swift", false);  // no arm: the plan before round 3
    NCHECK(t0.relax_dense_arm.empty(), "RD-2 {}: the hand-built target has a dense arm", c.name);
    nu::Target t = t0;
    t.relax_dense_arm = nu::relaxDenseDefaultName(g51);  // as the loader fills it (tune.cpp numericsTarget)
    {
        const nu::Plan base = planOf("fast", t0, {}), p = planOf("fast", t, {});
        NCHECK(base.flags.relax_on && base.flags.relax_k == 4u && base.flags.relax_alpha == 0.1f, "RD-2 {} no arm: k {} alpha {}", c.name, base.flags.relax_k,
               base.flags.relax_alpha);
        NCHECK(p.on(nu::Item::relaxacc) && p.flags.relax_on && p.flags.relax_k == wk && p.flags.relax_alpha == wa && p.relax_drafts == 0 && !p.lossy_override &&
                   p.overrides.empty() && p.ignored.empty(),
               "RD-2 {} Swift fast: k {} alpha {} (want {} / {})", c.name, p.flags.relax_k, p.flags.relax_alpha, wk, wa);
        NCHECK(nu::logLine(p).find(std::format("[k {}, alpha {:.2f}", wk, wa)) != std::string::npos, "RD-2 {}: plan line {}", c.name, nu::logLine(p));
        NCHECK(p.bitsKey() != base.bitsKey(), "RD-2 {}: the arm moves the fast plan (k / alpha)", c.name);
        NCHECK(planOf("fast", t, {}, false).flags == p.flags, "RD-2 {}: flags differ without load_overrides", c.name);
        // RD-3
        NCHECK(planOf("fast", t, {{"RELAX_K", "4"}, {"RELAX_ALPHA", "0.1"}}).flags == base.flags,
               "RD-3 {}: WHIRL_RELAX_K=4 WHIRL_RELAX_ALPHA=0.1 = the plan before the arm (the A arm)", c.name);
    }
    {
        const nu::Flags k3 = planOf("fast", t, {{"RELAX_K", "3"}}).flags, a2 = planOf("fast", t, {{"RELAX_ALPHA", "0.2"}}).flags;
        NCHECK(k3.relax_k == 3u && k3.relax_alpha == wa && a2.relax_k == wk && a2.relax_alpha == 0.2f, "RD-3 {}: env K / ALPHA over the arm", c.name);
        const nu::Plan p0 = planOf("fast", t, {{"RELAX", "0"}});
        NCHECK(!p0.flags.relax_on && !p0.on(nu::Item::relaxacc) && p0.bitsKey() == planOf("fast", t0, {{"RELAX", "0"}}).bitsKey(), "RD-3 {}: WHIRL_RELAX=0", c.name);
    }
    // RD-4
    for (const char* mode : {"precise", "balance"})
        for (const bool force : {false, true}) {
            std::vector<std::pair<const char*, const char*>> env;
            if (force) env = {{"RELAX", "1"}, {"RELAX_K", "6"}};
            const nu::Plan a = planOf(mode, t, env), b = planOf(mode, t0, env);
            NCHECK(a.bitsKey() == b.bitsKey() && nu::logLine(a) == nu::logLine(b) && a.overrides == b.overrides && a.ignored == b.ignored && !a.flags.relax_on,
                   "RD-4 {} {} force {}: the dense arm changed the plan", c.name, mode, force);
        }
    nu::Target tm0 = target(c, "moe", false), tm = tm0;
    tm.relax_dense_arm = "k8a02";
    tm0.relax_moe_arm = tm.relax_moe_arm = nu::relaxMoeDefaultName(g51);
    tm0.k_relax_rank = tm.k_relax_rank = true;
    for (const char* mode : {"precise", "balance", "fast"})
        NCHECK(planOf(mode, tm, {}).bitsKey() == planOf(mode, tm0, {}).bitsKey(), "RD-4 {} moe {}: the dense arm changed a MoE plan", c.name, mode);
}
void testRelaxDenseArm() {
    // RD-1
    const nu::RelaxArm* a = nu::relaxDenseArm("k8a02");
    NCHECK(a != nullptr && a->name == "k8a02" && a->on && a->k == 8u && a->alpha == 0.02f && a->drafts == 0u, "RD-1 arm k8a02 = K 8, alpha 0.02, the device's drafts");
    NCHECK(nu::relaxDenseArm("") == nullptr && nu::relaxDenseArm("k4a10d2") == nullptr && nu::relaxArm("k8a02") == nullptr, "RD-1 the dense and MoE arm tables are separate");
    NCHECK(nu::relaxDenseDefaultName(false) == "k8a02" && nu::relaxDenseDefaultName(false) == nu::relax_dense_default_r9700, "RD-1 R9700 dense default '{}'",
           nu::relaxDenseDefaultName(false));
    // v0252-8sdec (PLAN V1-143): the 8060S ships the R9700's kept arm k8a02 too (A arm = WHIRL_RELAX_K=4 WHIRL_RELAX_ALPHA=0.1)
    NCHECK(nu::relaxDenseDefaultName(true) == "k8a02" && nu::relax_dense_default_8060s == "k8a02", "RD-1 8060S dense default '{}' (want k8a02)",
           nu::relaxDenseDefaultName(true));
    for (const CardFake& c : k_cards) testRelaxDenseArmCard(c);
    // the rule is the name: a target that names the arm takes it on any card (the loader decides per card)
    nu::Target t8 = target(card("8060s"), "swift", false);
    t8.relax_dense_arm = "k8a02";
    nu::Request r = nu::parseMode("fast", "t");
    r.load_overrides = true;
    NCHECK(nu::plan(r, t8).flags.relax_k == 8u, "RD-2 a named arm applies");
}

}  // namespace

// v0251-a7port: the Radeon 8060S MoE prefill default (kernels::kMoepDefault = g64n64k64 + m128w4n128k64 + moe_combine4,
// plan flags = variant index + 1) on the gfx1151 card with MXFP4 routed experts, in every mode; off on the R9700, on a
// dense model and on non-MXFP4 experts; WHIRL_MOEP_GU / _DN / _COMB override it ("0" / "" = off).
// v0252-ornkv2 (PLAN V1-94): the 8060S balance combine default is 3 (f16 moe_yd + moe_combine4h); GU / DN as before,
// precise and fast keep kMoepDefault.comb (1). The balance rows fail before the change.
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
                // v0252-ornkv2: balance 3. v0252-b5 round 3 (PLAN V1-116 / V1-95): the default fast plan has moeskip k6 on, whose
                // prefill path needs the non-f16 combine: 1 here (and 3 once WHIRL_MOESKIP=0 turns moeskip off, below)
                // v0252-8smoe (A, PLAN V1-95): the fast moeskip plan takes the 8060S row device::Profile::moep_comb_fast_skip = 5
                const unsigned c8 = std::string_view(mode) == "balance" ? 3u : std::string_view(mode) == "fast" ? 5u : 1u;
                // v0252-moedf16: the R9700 MoE MXFP4 balance default is COMB 3 (f16 moe_yd on the fp8 kind), no moep; fast the
                // same only with moeskip off - the default fast plan (moeskip k6 on) takes the f32 moe_yd, COMB 0
                const bool r97m = std::string_view(c.name) == "r9700" && std::string_view(model) == "moe";
                const bool ydh = r97m && std::string_view(mode) == "balance";
                NCHECK(on    ? f.moep_gu == 1 && f.moep_dn == 3 && f.moep_comb == c8
                       : ydh ? f.moep_gu == 0 && f.moep_dn == 0 && f.moep_comb == 3
                             : f.moep_gu == 0 && f.moep_dn == 0 && f.moep_comb == 0,
                       "moep default {} {} {}: gu {} dn {} comb {}", c.name, model, mode, f.moep_gu, f.moep_dn, f.moep_comb);
                // v0252-8smoe (B, PLAN V1-147): the routing-aware wide gate + up tile; v0252-8sopen: opt-in (off on every plan)
                const unsigned guw = 0u;
                NCHECK(f.moep_guw == guw, "moep_guw default {} {} {}: {} (want {})", c.name, model, mode, f.moep_guw, guw);
                // fast with moeskip turned off: the f16 combine on both cards' MoE plans (item (d) on the 8060S, moedf16 on the R9700)
                if (std::string_view(mode) == "fast") {
                    const nu::Flags z = fl(t, mode, {{"MOESKIP", "0"}});
                    const unsigned want = std::string_view(model) == "moe" ? 3u : 0u;
                    NCHECK(z.moep_comb == want && z.moep_gu == f.moep_gu && z.moep_dn == f.moep_dn, "moep default {} {} fast, WHIRL_MOESKIP=0: comb {} (want {})",
                           c.name, model, z.moep_comb, want);
                }
            }
    const nu::Target t8 = target(card("8060s"), "moe", false);
    for (const char* mode : {"precise", "balance", "fast"}) {
        const nu::Flags g0 = fl(t8, mode, {{"MOEP_GU", "0"}}), d0 = fl(t8, mode, {{"MOEP_DN", ""}}), c0 = fl(t8, mode, {{"MOEP_COMB", "0"}});
        // v0252-ornkv2: the mode's combine default; v0252-8smoe (A): fast (moeskip on) 5
        const unsigned cd = std::string_view(mode) == "balance" ? 3u : std::string_view(mode) == "fast" ? 5u : 1u;
        NCHECK(g0.moep_gu == 0 && g0.moep_dn == 3 && g0.moep_comb == cd, "8060s moe {}: WHIRL_MOEP_GU=0 turns the gate + up variant off", mode);
        NCHECK(d0.moep_gu == 1 && d0.moep_dn == 0 && d0.moep_comb == cd, "8060s moe {}: WHIRL_MOEP_DN= (empty) turns the down variant off", mode);
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
    // v0252-8sdet (PLAN V1-140): 6 = the fixed-point (i32) combine fused into the down projection, fast-only like 5
    const std::string fix = "the fixed-point MoE combine (i32 sums, 4-bit-level) is fast-only";
    for (const CardFake& c : k_cards)
        for (const char* model : {"swift", "moe"}) {
            const nu::Target t = target(c, model, false);
            for (const char* q8 : {"", "1"}) {  // "": the mode's decode activations; "1": WHIRL_Q8DEC=1 (prec_dec off in precise)
                std::vector<std::pair<const char*, const char*>> extra;
                if (*q8) extra.emplace_back("Q8DEC", q8);
                for (const char* mode : {"precise", "balance", "fast"}) {
                    const std::string cell = std::format("B26-41 {} {} {} q8dec='{}'", c.name, model, mode, q8);
                    const nu::Plan base = planOf(mode, t, extra);
                    for (const char* v : {"0", "1", "2", "3", "4", "5", "5z", "6"}) {
                        auto vals = extra;
                        vals.emplace_back("MOEP_COMB", v);
                        const nu::Plan p = planOf(mode, t, vals);
                        const unsigned want = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
                        const bool precise = std::string_view(mode) == "precise", fast = std::string_view(mode) == "fast";
                        const bool refuse = ((want == 5 || want == 6) && !fast) || ((want == 3 || want == 4) && precise);
                        if (refuse) {
                            const std::string why = want == 5   ? atomic + std::format(" ({} never takes it)", mode)
                                                    : want == 6 ? fix + std::format(" ({} never takes it)", mode)
                                                                : f16 + " (precise never takes them)";
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
    // the default plan is untouched: 8060S MoE MXFP4 = 1 / 3 / 1 in precise and fast, 1 / 3 / 3 in balance (v0252-ornkv2;
    // testMoepDefault), no notes without the variable
    const nu::Target t8 = target(card("8060s"), "moe", false);
    for (const char* mode : {"precise", "balance", "fast"}) {
        const nu::Plan d = planOf(mode, t8, {});
        // v0252-8smoe (A): fast (moeskip on) 5
        const unsigned cd = std::string_view(mode) == "balance" ? 3u : std::string_view(mode) == "fast" ? 5u : 1u;
        NCHECK(d.flags.moep_gu == 1 && d.flags.moep_dn == 3 && d.flags.moep_comb == cd && d.ignored.empty(), "B26-41 8060s moe {}: default plan changed", mode);
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
    // v0252-b5 round 3 (PLAN V1-116): in fast the rule holds where moeskip is off (WHIRL_MOESKIP=0: "skip0" below); the default
    // fast plan has moeskip k6 on and takes the f32 moe_yd (COMB 0: ops::chooseMoe refuses the moeskip prefill path with the
    // f16 combines), WHIRL_MOEP_COMB=3 still wins there (the screen's other arm)
    for (const char* mode : {"balance", "fast"}) {
        const bool fast = std::string_view(mode) == "fast";
        std::vector<std::pair<const char*, const char*>> skip0;
        if (fast) skip0.emplace_back("MOESKIP", "0");
        auto with = [&](const char* n, const char* v) {
            auto e = skip0;
            e.emplace_back(n, v);
            return e;
        };
        const nu::Plan d = planOf(mode, t97, skip0);
        NCHECK(d.flags.moep_comb == 3 && d.flags.moep_gu == 0 && d.flags.moep_dn == 0 && d.ignored.empty(), "moedf16 r9700 moe {}: default comb 3 (got {})",
               mode, d.flags.moep_comb);
        const nu::Plan a = planOf(mode, t97, with("MOEP_COMB", "0"));
        NCHECK(a.flags.moep_comb == 0 && a.ignored.empty() && a.bitsKey() != d.bitsKey(), "moedf16 r9700 moe {}: WHIRL_MOEP_COMB=0 = the A arm (f32 moe_yd)",
               mode);
        NCHECK(planOf(mode, t97, with("MOEP_COMB", "3")).bitsKey() == d.bitsKey(), "moedf16 r9700 moe {}: WHIRL_MOEP_COMB=3 = the default", mode);
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
    // the 8060S (v0252-ornkv2, PLAN V1-94): balance takes COMB 3 too (gemm_moepdh f16 moe_yd + moe_combine4h on the moep
    // GU / DN defaults; screen KEEP: Ornith balance prefill 8k +1.00 %, 32k +1.14 %); precise and fast keep kMoepDefault
    // 1 / 3 / 1 (fast was not screened). A/B: WHIRL_MOEP_COMB=1 (A, the f32 moe_yd) / 3 (B). Fails before the change.
    for (const char* mode : {"precise", "balance", "fast"}) {
        const bool bal = std::string_view(mode) == "balance", fst = std::string_view(mode) == "fast";
        const nu::Plan d = planOf(mode, t8, {});
        const unsigned want = bal ? 3u : fst ? 5u : 1u;  // v0252-8smoe (A): fast (moeskip on) 5
        NCHECK(d.flags.moep_gu == 1 && d.flags.moep_dn == 3 && d.flags.moep_comb == want && d.ignored.empty(),
               "ornkv2 8060s moe {}: default comb {} (want {})", mode, d.flags.moep_comb, want);
        const nu::Plan a = planOf(mode, t8, {{"MOEP_COMB", "1"}});
        NCHECK(a.flags.moep_comb == 1 && a.ignored.empty() && (a.bitsKey() != d.bitsKey()) == (bal || fst),
               "ornkv2 8060s moe {}: WHIRL_MOEP_COMB=1 = the A arm (f32 moe_yd)", mode);
        NCHECK(planOf(mode, target(card("8060s"), "swift", false), {}).flags.moep_comb == 0, "ornkv2 8060s swift {}: dense model unchanged", mode);
    }
    // (the fast rows with moeskip on / off, both cards: testMoeSkipDefault, v0252-b5 round 3)
    {
        const nu::Plan d = planOf("balance", t8, {});
        NCHECK(planOf("balance", t8, {{"MOEP_COMB", "3"}}).bitsKey() == d.bitsKey(), "ornkv2 8060s moe balance: WHIRL_MOEP_COMB=3 = the default");
        NCHECK(planOf("balance", t8, {{"MOEP_COMB", "5"}}).flags.moep_comb == 3, "ornkv2 8060s moe balance: COMB=5 refused -> 3");
        // a balance list without the item h16 (f16 intermediates): the f32 moe_yd combine
        NCHECK(planOf("balance:gdnwmma,specsample", t8, {}).flags.moep_comb == 1, "ornkv2 8060s moe balance without h16: comb 1");
        // non-MXFP4 routed experts: no moep, combine 0
        nu::Target tq = t8;
        tq.moe_mxfp4 = false;
        NCHECK(planOf("balance", tq, {}).flags.moep_comb == 0, "ornkv2 8060s non-MXFP4 experts balance: comb 0");
        // precise: f32 moe_yd, also with WHIRL_Q8DEC=1 and with WHIRL_MOEP_COMB=3 (refused)
        for (const bool q8 : {false, true}) {
            std::vector<std::pair<const char*, const char*>> ex;
            if (q8) ex.emplace_back("Q8DEC", "1");
            const nu::Plan p = planOf("precise", t8, ex);
            ex.emplace_back("MOEP_COMB", "3");
            const nu::Plan p3 = planOf("precise", t8, ex);
            NCHECK(p.flags.moep_comb == 1 && p3.flags.moep_comb == 1 && p3.bitsKey() == p.bitsKey() && p3.ignored.size() == 1,
                   "ornkv2 8060s moe precise q8dec={}: comb 1, COMB=3 refused", q8);
        }
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

// v0252-strixc2 (PLAN V1-09): the gfx1151 balance prefill attention attn_sx (strix-llama.cpp fattn-rdna35, NOT bitwise).
// The plan flag attn_sx is on exactly in balance mode on the 8060S card (device::Profile::attn_sx_balance -> Target::
// attn_sx_profile) with the item h16 (f16 intermediates) in the request, both models; never in precise or fast (precise
// stays bit-identical, fast keeps its own path), never on the R9700 card (no kernel). WHIRL_ATTN_SX=0 turns it off and
// changes nothing else (bitsKey = the default plan's with attn_sx false, same log line, no +overrides, never lossy); "1"
// changes nothing in balance; a non-"0" value in precise / fast leaves the flag off and adds exactly one Plan::ignored
// note. A custom balance item list without h16 leaves it off. Fails before the port (loadFlags never sets attn_sx).
// v0252-8sdec (PLAN V1-142): gemvh_grp (the precise grouped gemvh) defaults on for DENSE models on a card whose profile
// row says so (gemvh_group_dense), every mode (bitwise: only precise rows can take the kernel); off on MoE models;
// WHIRL_GEMVH_GRP=0 / =1 win both ways and never mark the plan lossy. The other flags do not move. Fails to build before
// the change (no Target::gemvh_grp_profile / Flags::gemvh_grp).
// v0252-8sopen: no shipped row sets it any more (screens 8sdec-screen2 / -bands OPEN, ADR-0019), so the default is off on
// both cards and WHIRL_GEMVH_GRP=1 is the opt-in (on in every mode, precise included, on the 8060S dense plan); the
// profile rule itself is still pinned through a target with gemvh_grp_profile forced on.
// v0252-8sgrp: the gfx1151 row is on again (screens 8sdec-grp2 20261124-0314 / -screen2 / -bands / -grp2-long: precise
// tg 8192 +2.36 % +- 2.00, bits SAME every round), and the profile default is PRECISE ONLY (the screened cell): balance /
// fast plans keep the flag off (their bitsKey does not move); "0" -> off and any other value -> on in every mode as before;
// MoE stays off (Ornith not screened); gfx1201 stays off.
static void testGemvhGrpDefault() {
    auto planOf = [](const char* mode, const nu::Target& t, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    for (const CardFake& c : k_cards)
        for (const char* model : {"swift", "moe"})
            for (const char* mode : {"precise", "balance", "fast"}) {
                const nu::Target t = target(c, model, false);
                const bool is_8060s = std::string_view(c.name) == "8060s", precise = std::string_view(mode) == "precise";
                // v0252-8sgrp: default on only for 8060S dense precise
                const bool want = is_8060s && std::string_view(model) == "swift" && precise;
                NCHECK(t.gemvh_grp_profile == is_8060s, "gemvh_grp {} {} {}: profile row {} (want {})", c.name, model, mode, t.gemvh_grp_profile, is_8060s);
                const nu::Plan p = planOf(mode, t, {});
                NCHECK(p.flags.gemvh_grp == want, "gemvh_grp {} {} {}: {} (want {})", c.name, model, mode, p.flags.gemvh_grp, want);
                const nu::Plan off = planOf(mode, t, {{"GEMVH_GRP", "0"}}), on = planOf(mode, t, {{"GEMVH_GRP", "1"}});
                NCHECK(!off.flags.gemvh_grp && on.flags.gemvh_grp && !off.lossy_override && !on.lossy_override, "gemvh_grp {} {} {}: env 0 / 1", c.name,
                       model, mode);
                nu::Flags a = p.flags, b = off.flags;
                a.gemvh_grp = b.gemvh_grp = false;
                NCHECK(a == b, "gemvh_grp {} {} {}: another flag moved", c.name, model, mode);
                nu::Target t0 = t;
                t0.gemvh_grp_profile = false;
                NCHECK(!planOf(mode, t0, {}).flags.gemvh_grp, "gemvh_grp {} {} {}: on without the profile row", c.name, model, mode);
                // the profile rule (a row with gemvh_group_dense on): dense precise on, MoE / balance / fast off, "0" still wins
                nu::Target t1 = t;
                t1.gemvh_grp_profile = true;
                const bool want1 = std::string_view(model) == "swift" && precise;
                const nu::Plan p1 = planOf(mode, t1, {});
                NCHECK(p1.flags.gemvh_grp == want1 && !planOf(mode, t1, {{"GEMVH_GRP", "0"}}).flags.gemvh_grp && !p1.lossy_override,
                       "gemvh_grp {} {} {}: profile row on -> {} (want {}), env 0 off", c.name, model, mode, p1.flags.gemvh_grp, want1);
            }
    // v0252-8sgrp: the three switch states on the 8060S dense MXFP4 plan in precise (the cell the grouped gemvh serves)
    const nu::Target ts = target(card("8060s"), "swift", false);
    const nu::Plan pu = planOf("precise", ts, {}), p0 = planOf("precise", ts, {{"GEMVH_GRP", "0"}});
    const bool st_unset = pu.flags.gemvh_grp, st_0 = p0.flags.gemvh_grp, st_1 = planOf("precise", ts, {{"GEMVH_GRP", "1"}}).flags.gemvh_grp,
               st_x = planOf("precise", ts, {{"GEMVH_GRP", "yes"}}).flags.gemvh_grp;
    NCHECK(st_unset && !st_0 && st_1 && st_x, "gemvh_grp 8060s swift precise: unset {} / 0 {} / 1 {} / yes {} (want on / off / on / on)", st_unset, st_0,
           st_1, st_x);
    // the default 8060S precise bitsKey differs from the WHIRL_GEMVH_GRP=0 one only in gemvh_grp (1 vs 0); the log line is the same
    std::string ku = pu.bitsKey(), k0 = p0.bitsKey();
    const std::size_t iu = ku.find("gemvh_grp=1;"), i0 = k0.find("gemvh_grp=0;");
    NCHECK(iu != std::string::npos && i0 == iu && ku.replace(iu, 12, "gemvh_grp=0;") == k0 && nu::logLine(pu) == nu::logLine(p0),
           "gemvh_grp 8060s swift precise: bitsKey / logLine of the default vs =0");
    // balance / fast plans on the 8060S do not move with the new default
    for (const char* m : {"balance", "fast"})
        NCHECK(!planOf(m, ts, {}).flags.gemvh_grp && planOf(m, ts, {}).bitsKey() == planOf(m, ts, {{"GEMVH_GRP", "0"}}).bitsKey(),
               "gemvh_grp 8060s swift {}: default on or bitsKey moved", m);
}

static void testAttnSxFlag() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    // v0252-b5 round 3, item (h): the DEFAULT is the MoE class only (Ornith screen KEEP: +1.60 % +-0.63 / +1.37 % +-1.25; Swift's
    // screen is still OPEN, +1.16 % +-2.02, so a dense model stays on the old kernel by default). WHIRL_ATTN_SX=0 forces the old
    // kernel, =1 forces attn_sx on wherever it may run (8060S balance with the item h16 and MXFP4 tensors: Swift too - the
    // arm of the next Swift screen on this exe); any other value keeps the default. 12 cells: 3 modes x 2 cards x 2 models.
    std::size_t on_cells = 0, forced_cells = 0;
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const nu::Target t = target(c, model, false);
                const bool can = std::string_view(mode) == "balance" && c.prof.attn_sx_balance;  // where the kernel may run at all
                const bool want = can && std::string_view(model) == "moe";
                const nu::Plan d = pl(t, mode, {});
                NCHECK(d.flags.attn_sx == want, "attn_sx default {} {} {}: {} (want {})", c.name, model, mode, d.flags.attn_sx, want);
                on_cells += d.flags.attn_sx;
                // "0": off, nothing else moves
                const nu::Plan z = pl(t, mode, {{"ATTN_SX", "0"}});
                nu::Plan a = d;
                a.flags.attn_sx = false;
                NCHECK(!z.flags.attn_sx && a.bitsKey() == z.bitsKey() && nu::logLine(z) == nu::logLine(d) && z.overrides == d.overrides &&
                           z.lossy_override == d.lossy_override && z.ignored.empty(),
                       "attn_sx {} {} {}: WHIRL_ATTN_SX=0 turns only the attn_sx flag off", c.name, model, mode);
                // "1": on wherever it may run (also for the dense model), nothing else moves; both arms differ in bitsKey there
                {
                    const nu::Plan f = pl(t, mode, {{"ATTN_SX", "1"}});
                    a.flags.attn_sx = can;
                    NCHECK(f.flags.attn_sx == can && a.bitsKey() == f.bitsKey() && nu::logLine(f) == nu::logLine(d) && f.overrides == d.overrides &&
                               f.lossy_override == d.lossy_override,
                           "attn_sx {} {} {}: WHIRL_ATTN_SX=1 forces only the attn_sx flag on where it may run ({}, want {})", c.name, model, mode,
                           f.flags.attn_sx, can);
                    NCHECK((f.bitsKey() != z.bitsKey()) == can, "attn_sx {} {} {}: the arms WHIRL_ATTN_SX=0 / =1 must differ exactly where it may run", c.name,
                           model, mode);
                    forced_cells += f.flags.attn_sx;
                }
                for (const char* v : {"1", "x", ""}) {
                    const nu::Plan o = pl(t, mode, {{"ATTN_SX", v}});
                    if (std::string_view(v) != "1")
                        NCHECK(o.flags.attn_sx == want && o.bitsKey() == d.bitsKey() && o.overrides == d.overrides && o.lossy_override == d.lossy_override,
                               "attn_sx {} {} {}: WHIRL_ATTN_SX='{}' keeps the plan's choice", c.name, model, mode, v);
                    std::size_t notes = 0;
                    for (const std::string& n : o.ignored) notes += n.rfind(std::string("WHIRL_ATTN_SX=") + v + " ignored", 0) == 0;
                    const std::size_t want_notes = std::string_view(mode) == "balance" ? 0 : 1;
                    NCHECK(notes == want_notes && o.ignored.size() == want_notes, "attn_sx {} {} {}: WHIRL_ATTN_SX='{}': {} ignored notes (want {})",
                           c.name, model, mode, v, notes, want_notes);
                }
            }
    NCHECK(on_cells == 1, "attn_sx: on by default in exactly 1 cell (8060S balance, the MoE model), got {}", on_cells);
    NCHECK(forced_cells == 2, "attn_sx: WHIRL_ATTN_SX=1 turns it on in exactly 2 cells (8060S balance, both models), got {}", forced_cells);
    // a custom balance list without h16: off (also with WHIRL_ATTN_SX=1); with h16: the default of the model class, on with =1 (8060S)
    for (const char* model : {"swift", "moe"}) {
        const bool moe = std::string_view(model) == "moe";
        const nu::Target t = target(card("8060s"), model, false);
        const nu::Plan no = pl(t, "balance:fp8,kvq8", {});
        const nu::Plan yes = pl(t, "balance:h16", {});
        NCHECK(!no.flags.attn_sx && yes.flags.attn_sx == moe, "attn_sx {}: custom balance without h16 -> off ({}), with h16 -> the class default ({})", model,
               no.flags.attn_sx, yes.flags.attn_sx);
        NCHECK(!pl(t, "balance:fp8,kvq8", {{"ATTN_SX", "1"}}).flags.attn_sx && pl(t, "balance:h16", {{"ATTN_SX", "1"}}).flags.attn_sx,
               "attn_sx {}: WHIRL_ATTN_SX=1 needs the item h16", model);
        nu::Target tq = t;  // no MXFP4 tensor (h16 is MXFP4-only on this GPU): off
        tq.has_mxfp4 = false;
        NCHECK(!pl(tq, "balance", {}).flags.attn_sx && !pl(tq, "balance", {{"ATTN_SX", "1"}}).flags.attn_sx,
               "attn_sx {}: a model without MXFP4 tensors -> off (also with WHIRL_ATTN_SX=1)", model);
    }
}

// v0252-8sfp (PLAN V1-144): the gfx1151 FAST prefill attention with int4 Q.K^T (attn_prefill_wmma_i4 / _s_i4, NOT bitwise).
// The plan flag attn_i4qk is on exactly in fast mode on the 8060S card (device::Profile::attn_i4qk_fast -> Target::
// attn_i4qk_profile), both models (MXFP4 tensors); never in precise or balance (their plans and bits stay exactly as they
// are), never on the R9700 card (no kernel). WHIRL_ATTN_I4QK=0 turns it off and changes nothing else (bitsKey = the default
// plan's with attn_i4qk false, same log line, no +overrides, never lossy); any other value keeps the plan's choice; a
// non-"0" value in precise / balance adds exactly one Plan::ignored note. Every OTHER flag of every plan is the one it
// was before the flag existed (the plan with attn_i4qk cleared == the WHIRL_ATTN_I4QK=0 plan). Fails before the port.
// v0252-8smoe (B, PLAN V1-147): WHIRL_MOEP_GUW / device::Profile::moep_guw - the routing-aware wide moep gate + up tile.
// Default = the card's row on the 8060S MoE MXFP4 plan in balance / fast (2 = g64n128k64), 0 elsewhere (precise, R9700,
// dense, non-MXFP4 experts); the variable names a kMoepGu variant in any mode (class B: same bits), "0" / "" / unknown = 0.
// WHIRL_MOEP_GUW=0 is the A arm: only moep_guw changes (bitsKey differs exactly where the default is on).
// v0252-8sopen: no shipped row sets it any more (screen 8smoe-screen OPEN, ADR-0019): the shipped plans are all off and
// WHIRL_MOEP_GUW=g64n128k64 is the opt-in (on in balance / fast on the 8060S MoE MXFP4 plan); the profile rule above is
// pinned through targets with moep_guw_profile forced to 2.
static void testMoepGuwFlag() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    NCHECK(std::string(whirl::kernels::kMoepGu[1].name) == "g64n128k64" && whirl::kernels::kMoepGu[1].bn == 128, "kMoepGu[1] = g64n128k64 (bn 128)");
    // v0252-8sopen: the shipped rows (both 0) and the three switch states on the 8060S MoE MXFP4 plan
    for (const CardFake& c : k_cards) NCHECK(c.prof.moep_guw == 0u, "moep_guw {}: profile row {} (want 0 since v0252-8sopen)", c.name, c.prof.moep_guw);
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const nu::Target ts = target(c, model, false);
                const nu::Plan d = pl(ts, mode, {}), z = pl(ts, mode, {{"MOEP_GUW", "0"}}), o = pl(ts, mode, {{"MOEP_GUW", whirl::kernels::kMoepGu[1].name}});
                NCHECK(d.flags.moep_guw == 0u && z.flags.moep_guw == 0u && o.flags.moep_guw == 2u && z.bitsKey() == d.bitsKey() && o.bitsKey() != d.bitsKey() &&
                           !o.lossy_override,
                       "moep_guw shipped {} {} {}: unset {} / 0 {} / g64n128k64 {} (want 0 / 0 / 2)", c.name, model, mode, d.flags.moep_guw, z.flags.moep_guw,
                       o.flags.moep_guw);
            }
    std::size_t on_cells = 0;
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                nu::Target t = target(c, model, false);
                t.moep_guw_profile = 2u;  // a row with the wide tile on (the rule; no shipped row since v0252-8sopen)
                const bool want = std::string_view(mode) != "precise" && std::string_view(model) == "moe" && c.prof.numerics_8060s;
                const nu::Plan d = pl(t, mode, {});
                NCHECK(d.flags.moep_guw == (want ? 2u : 0u), "moep_guw profile rule {} {} {}: {}", c.name, model, mode, d.flags.moep_guw);
                on_cells += want;
                const nu::Plan z = pl(t, mode, {{"MOEP_GUW", "0"}});
                nu::Flags a = d.flags;
                a.moep_guw = 0;
                NCHECK(z.flags == a && z.overrides == d.overrides && z.ignored == d.ignored && nu::logLine(z) == nu::logLine(d),
                       "moep_guw {} {} {}: WHIRL_MOEP_GUW=0 turns only moep_guw off", c.name, model, mode);
                NCHECK((z.bitsKey() != d.bitsKey()) == want, "moep_guw {} {} {}: the arms default / =0 must differ exactly where it is on", c.name, model,
                       mode);
                for (const char* off : {"", "bogus", "g64n128"})
                    NCHECK(pl(t, mode, {{"MOEP_GUW", off}}).flags.moep_guw == 0u, "moep_guw {} {} {}: '{}' = off", c.name, model, mode, off);
                NCHECK(pl(t, mode, {{"MOEP_GUW", "g64n128k64"}}).flags.moep_guw == 2u && pl(t, mode, {{"MOEP_GUW", "g64n256k32"}}).flags.moep_guw == 5u,
                       "moep_guw {} {} {}: a variant by name (any mode: bitwise)", c.name, model, mode);
                if (want)
                    NCHECK(pl(t, mode, {{"MOEP_GUW", "g64n128k64"}}).bitsKey() == d.bitsKey(), "moep_guw {} {} {}: =g64n128k64 is the default", c.name, model,
                           mode);
            }
    NCHECK(on_cells == 2, "moep_guw on in {} cells (want 2: 8060S moe balance / fast)", on_cells);
    nu::Target tq = target(card("8060s"), "moe", false);
    tq.moep_guw_profile = 2u;
    tq.moe_mxfp4 = false;
    NCHECK(pl(tq, "balance", {}).flags.moep_guw == 0u, "moep_guw: non-MXFP4 routed experts off");
}

// v0252-8smoe2 (PLAN V1-147, levers C / D / E): device::Profile moep_dhs / moep_dna / moep_fq and WHIRL_MOEP_DHS / _DNA / _FQ.
// The row (Target::*_profile) applies on the 8060S MoE MXFP4 plan: C in balance / fast, E in every mode (v0252-8sflip), D in
// fast; 0 elsewhere. The
// variables override it in any mode: DHS / FQ "0" / "" off, any other value on; DNA a kMoepDn name ("0" / "" / unknown off).
// Tests follow the profile row: the shipped rows are read from k_cards, so a default flip is the row + the device pin only.
// Each "=0" arm changes only its own flag; each switch changes bitsKey exactly where it changes its flag. Fails to build before
// the change (no Flags::moep_dhs / moep_dna / moep_fq, EnvValues fields, Target::*_profile).
static void testMoep2Flags() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    NCHECK(std::string(whirl::kernels::kMoepDn[3].name) == "m128w8n128k64", "kMoepDn[3] = m128w8n128k64");
    std::size_t on_c = 0, on_d = 0, on_e = 0;
    for (const bool forced : {false, true})  // the shipped rows, then rows with every lever on (the rule)
        for (const char* mode : {"precise", "balance", "fast"})
            for (const CardFake& c : k_cards)
                for (const char* model : {"swift", "moe"}) {
                    nu::Target t = target(c, model, false);
                    if (forced) {
                        t.moep_dhs_profile = t.moep_fq_profile = true;
                        t.moep_dna_profile = 4u;
                    }
                    const bool moep = std::string_view(model) == "moe" && c.prof.numerics_8060s;
                    const bool bf = moep && std::string_view(mode) != "precise", fa = moep && std::string_view(mode) == "fast";
                    // v0252-8sflip: E's row applies in every mode (precise too: the twins store the same bits, screen
                    // 8smoe2-e-precise equal output); C's row where the f16 epilogue can run (balance / fast: precise
                    // never takes ydh)
                    const bool want_c = bf && t.moep_dhs_profile, want_e = moep && t.moep_fq_profile;
                    const std::uint32_t want_d = fa ? t.moep_dna_profile : 0u;
                    const nu::Plan d = pl(t, mode, {});
                    const std::string cell = std::format("{} {} {} {}", forced ? "forced" : "shipped", c.name, model, mode);
                    NCHECK(d.flags.moep_dhs == want_c && d.flags.moep_dna == want_d && d.flags.moep_fq == want_e, "moep2 default {}: {} {} {}", cell,
                           d.flags.moep_dhs, d.flags.moep_dna, d.flags.moep_fq);
                    on_c += forced && want_c;
                    on_d += forced && want_d != 0;
                    on_e += forced && want_e;
                    struct Sw {
                        const char* name;
                        const char* on;
                        bool def_on;
                    };
                    for (const Sw& s : {Sw{"MOEP_DHS", "1", want_c}, Sw{"MOEP_DNA", "m128w8n128k64", want_d != 0}, Sw{"MOEP_FQ", "1", want_e}}) {
                        const nu::Plan z = pl(t, mode, {{s.name, "0"}}), o = pl(t, mode, {{s.name, s.on}});
                        nu::Flags fz = d.flags, fo = d.flags;
                        if (std::string_view(s.name) == "MOEP_DHS") fz.moep_dhs = false, fo.moep_dhs = true;
                        if (std::string_view(s.name) == "MOEP_DNA") fz.moep_dna = 0u, fo.moep_dna = 4u;
                        if (std::string_view(s.name) == "MOEP_FQ") fz.moep_fq = false, fo.moep_fq = true;
                        NCHECK(z.flags == fz && o.flags == fo && z.ignored == d.ignored && o.ignored == d.ignored && nu::logLine(z) == nu::logLine(d) &&
                                   !o.lossy_override,
                               "moep2 {} WHIRL_{}: =0 / ={} change only their own flag", cell, s.name, s.on);
                        NCHECK((z.bitsKey() != d.bitsKey()) == s.def_on && (o.bitsKey() != d.bitsKey()) == !s.def_on,
                               "moep2 {} WHIRL_{}: bitsKey differs exactly where the flag changes", cell, s.name);
                        for (const char* off : {"", "0"}) {
                            const nu::Plan x = pl(t, mode, {{s.name, off}});
                            NCHECK(x.flags == fz, "moep2 {} WHIRL_{}='{}' = off", cell, s.name, off);
                        }
                    }
                    for (const char* bogus : {"bogus", "m128w8", "1"})
                        NCHECK(pl(t, mode, {{"MOEP_DNA", bogus}}).flags.moep_dna == 0u, "moep2 {} WHIRL_MOEP_DNA='{}' = off", cell, bogus);
                    NCHECK(pl(t, mode, {{"MOEP_DNA", "m128w4n64k64"}}).flags.moep_dna == 2u, "moep2 {}: DNA by name", cell);
                }
    NCHECK(on_c == 2 && on_e == 3 && on_d == 1,
           "moep2 rule cells C {} D {} E {} (want 2 / 1 / 3: 8060S moe balance+fast / fast / precise+balance+fast)", on_c, on_d, on_e);
    // v0252-8sflip (PLAN V1-147): the SHIPPED defaults, pinned by value (not through the row): 8060S MoE MXFP4 plan - E on in
    // precise / balance / fast, C on in balance / fast (off in precise: no f16 epilogue there); "0" = the previous kernels,
    // "1" = on; R9700 and the dense model never by default. D stays off.
    for (const char* mode : {"precise", "balance", "fast"}) {
        const bool prec = std::string_view(mode) == "precise";
        const nu::Target t8 = target(card("8060s"), "moe", false);
        const nu::Plan d = pl(t8, mode, {});
        NCHECK(d.flags.moep_fq && d.flags.moep_dhs == !prec && d.flags.moep_dna == 0u, "moep2 shipped 8060s moe {}: fq {} dhs {} dna {}", mode,
               d.flags.moep_fq, d.flags.moep_dhs, d.flags.moep_dna);
        const nu::Plan z = pl(t8, mode, {{"MOEP_FQ", "0"}, {"MOEP_DHS", "0"}}), o = pl(t8, mode, {{"MOEP_FQ", "1"}, {"MOEP_DHS", "1"}});
        NCHECK(!z.flags.moep_fq && !z.flags.moep_dhs && o.flags.moep_fq && o.flags.moep_dhs, "moep2 shipped 8060s moe {}: =0 off, =1 on", mode);
        for (const auto& [cn, model] : {std::pair{"r9700", "moe"}, std::pair{"r9700", "swift"}, std::pair{"8060s", "swift"}}) {
            const nu::Plan x = pl(target(card(cn), model, false), mode, {});
            NCHECK(!x.flags.moep_fq && !x.flags.moep_dhs && x.flags.moep_dna == 0u, "moep2 shipped {} {} {}: off", cn, model, mode);
        }
    }
    nu::Target tq = target(card("8060s"), "moe", false);
    tq.moep_dhs_profile = tq.moep_fq_profile = true;
    tq.moep_dna_profile = 4u;
    tq.moe_mxfp4 = false;
    const nu::Plan q = pl(tq, "fast", {});
    NCHECK(!q.flags.moep_dhs && q.flags.moep_dna == 0u && !q.flags.moep_fq, "moep2: non-MXFP4 routed experts off");
}

// v0252-8sq4 (PLAN V1-140 / V1-109, F8): device::Profile attn_dq4_row_order / dq4_min_keys and WHIRL_DQ4_ORDER / _MINKEYS.
// The rows (Target::dq4_*_profile) apply in fast only, on both cards and both models (the flags matter only on the attn_dq4
// path, ops::chooseAttnDecode); the variables override them in fast: ORDER "0" / "" off, other values on; MINKEYS a whole
// decimal 1 .. 1048576, anything else 0. Outside fast the flags stay off and a non-"0" value is noted once (Plan::ignored).
// Each switch changes only its own flag, and bitsKey exactly where the flag changes. The rows are read from k_cards (the
// shipped rows, then forced on): a default flip is the profile line plus the device pins. Fails to build before the change
// (no Flags::dq4_order / dq4_min_keys, EnvValues fields, Target::dq4_*_profile, parseDq4MinKeys).
static void testDq4F8Flags() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    NCHECK(nu::parseDq4MinKeys("1024") == 1024u && nu::parseDq4MinKeys("1") == 1u && nu::parseDq4MinKeys("1048576") == 1048576u, "parseDq4MinKeys values");
    for (const char* bad : {"", "0", "x", "1k", "-1", "1048577", "99999999", " 512"})
        NCHECK(nu::parseDq4MinKeys(bad) == 0u, "parseDq4MinKeys('{}') = 0", bad);
    std::size_t on_o = 0, on_m = 0;
    for (const bool forced : {false, true})
        for (const char* mode : {"precise", "balance", "fast"})
            for (const CardFake& c : k_cards)
                for (const char* model : {"swift", "moe"}) {
                    nu::Target t = target(c, model, false);
                    if (forced) {
                        t.dq4_order_profile = true;
                        t.dq4_min_keys_profile = 1024u;
                    }
                    const bool fast = std::string_view(mode) == "fast";
                    const bool want_o = fast && t.dq4_order_profile;
                    const std::uint32_t want_m = fast ? t.dq4_min_keys_profile : 0u;
                    const nu::Plan d = pl(t, mode, {});
                    const std::string cell = std::format("{} {} {} {}", forced ? "forced" : "shipped", c.name, model, mode);
                    NCHECK(d.flags.dq4_order == want_o && d.flags.dq4_min_keys == want_m, "dq4 F8 default {}: {} {}", cell, d.flags.dq4_order,
                           d.flags.dq4_min_keys);
                    on_o += forced && want_o;
                    on_m += forced && want_m != 0;
                    struct Sw {
                        const char* name;
                        const char* on;
                    };
                    for (const Sw& s : {Sw{"DQ4_ORDER", "1"}, Sw{"DQ4_MINKEYS", "2048"}}) {
                        const bool order = std::string_view(s.name) == "DQ4_ORDER";
                        const nu::Plan z = pl(t, mode, {{s.name, "0"}}), o = pl(t, mode, {{s.name, s.on}});
                        nu::Flags fz = d.flags, fo = d.flags;
                        if (fast) {
                            if (order) {
                                fz.dq4_order = false;
                                fo.dq4_order = true;
                            } else {
                                fz.dq4_min_keys = 0u;
                                fo.dq4_min_keys = 2048u;
                            }
                        }
                        NCHECK(z.flags == fz && o.flags == fo && z.ignored == d.ignored && nu::logLine(z) == nu::logLine(d) && !o.lossy_override,
                               "dq4 F8 {} WHIRL_{}: =0 / ={} change only their own flag", cell, s.name, s.on);
                        NCHECK((z.bitsKey() != d.bitsKey()) == !(z.flags == d.flags), "dq4 F8 {} WHIRL_{}=0: bitsKey differs exactly where a flag changes", cell,
                               s.name);
                        NCHECK((o.bitsKey() != d.bitsKey()) == !(o.flags == d.flags), "dq4 F8 {} WHIRL_{}={}: bitsKey differs exactly where a flag changes",
                               cell, s.name, s.on);
                        std::size_t notes = 0;
                        for (const std::string& n : o.ignored) notes += n.rfind(std::string("WHIRL_") + s.name + "=" + s.on + " ignored", 0) == 0;
                        NCHECK(notes == (fast ? 0u : 1u) && o.ignored.size() == d.ignored.size() + (fast ? 0u : 1u), "dq4 F8 {} WHIRL_{}={}: {} ignored notes",
                               cell, s.name, s.on, notes);
                        NCHECK(pl(t, mode, {{s.name, ""}}).flags == fz, "dq4 F8 {} WHIRL_{}='' = off", cell, s.name);
                    }
                    for (const char* bogus : {"x", "1k", "-1"})
                        NCHECK(pl(t, mode, {{"DQ4_MINKEYS", bogus}}).flags.dq4_min_keys == 0u, "dq4 F8 {} WHIRL_DQ4_MINKEYS='{}' = 0", cell, bogus);
                }
    // forced rows: fast on both cards and both models (4 cells each)
    NCHECK(on_o == 4 && on_m == 4, "dq4 F8 rule cells order {} min keys {} (want 4 / 4: fast x 2 cards x 2 models)", on_o, on_m);
    // v0252-8sflip2 (pinned, not read from the rows): the SHIPPED defaults. 8060S fast (both models) unset -> dq4_order on,
    // "0" / "" -> off (the previous kernel attn_dq4, the reference arm of later screens), "1" -> on; precise / balance never on
    // (q4 KV is fast-only); the R9700 never on by default (no attn_dq4o in its object). dq4_min_keys stays 0 everywhere.
    std::size_t pinned = 0;
    for (const CardFake& c : k_cards)
        for (const char* model : {"swift", "moe"})
            for (const char* mode : {"precise", "balance", "fast"}) {
                const nu::Target t = target(c, model, false);
                const bool fast = std::string_view(mode) == "fast", s8 = std::string_view(c.name) == "8060s";
                const std::string cell = std::format("shipped pin {} {} {}", c.name, model, mode);
                const nu::Plan d = pl(t, mode, {});
                NCHECK(d.flags.dq4_order == (fast && s8) && d.flags.dq4_min_keys == 0u, "dq4 F8 {}: unset order {} min keys {}", cell,
                       d.flags.dq4_order, d.flags.dq4_min_keys);
                NCHECK(!pl(t, mode, {{"DQ4_ORDER", "0"}}).flags.dq4_order && !pl(t, mode, {{"DQ4_ORDER", ""}}).flags.dq4_order,
                       "dq4 F8 {}: WHIRL_DQ4_ORDER=0 / '' = off", cell);
                NCHECK(pl(t, mode, {{"DQ4_ORDER", "1"}}).flags.dq4_order == fast, "dq4 F8 {}: WHIRL_DQ4_ORDER=1 = on in fast only", cell);
                if (fast && s8) {
                    const nu::Plan z = pl(t, mode, {{"DQ4_ORDER", "0"}});
                    NCHECK(z.bitsKey() != d.bitsKey(), "dq4 F8 {}: the default 8060S fast key differs from the WHIRL_DQ4_ORDER=0 key", cell);
                    pinned += 1;
                }
            }
    NCHECK(pinned == 2, "dq4 F8 shipped pin: {} 8060S fast cells (want 2: swift, moe)", pinned);
}

// v0252-8sq5 (PLAN V1-109): flag dq4_splits, the q4 decode / verify split count (ops::dq4TableSplits). Fast only (q4 KV exists
// only there): fast = kDq4SplitsAuto (the device profile's measured table; a row that is off = today's rule), precise /
// balance = 0 (today's rule, never read: no q4 path) whatever WHIRL_DQ4_SPLITS says - one Plan::ignored note, the plan and
// bitsKey unchanged. WHIRL_DQ4_SPLITS in fast: "auto" = the table, "0" / "" = today's rule (ops::dq4Splits: the A arm of the
// screen), 1..256 = that count (ops falls back to today's rule where the grid is not valid), anything else = auto + a note.
static void testDq4SplitsFlag() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    const std::uint32_t A = nu::kDq4SplitsAuto;
    static_assert(nu::kDq4SplitsAuto == 65535u, "kDq4SplitsAuto");
    NCHECK(nu::parseDq4Splits("auto") == A && nu::parseDq4Splits("0") == 0u && nu::parseDq4Splits("") == 0u && nu::parseDq4Splits("1") == 1u &&
               nu::parseDq4Splits("8") == 8u && nu::parseDq4Splits("256") == 256u,
           "parseDq4Splits values");
    for (const char* bad : {"257", "x", "-1", "8x", " 8", "AUTO", "65535", "1024"})
        NCHECK(!nu::parseDq4Splits(bad).has_value(), "parseDq4Splits('{}') refused", bad);
    std::size_t cells = 0;
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const nu::Target t = target(c, model, false);
                const bool fast = std::string_view(mode) == "fast";
                const std::string cell = std::format("{} {} {}", c.name, model, mode);
                const nu::Plan d = pl(t, mode, {});
                NCHECK(d.flags.dq4_splits == (fast ? A : 0u), "dq4_splits default {}: {}", cell, d.flags.dq4_splits);
                for (const char* v : {"0", "", "auto", "1", "16", "256", "257", "x"}) {
                    const nu::Plan p = pl(t, mode, {{"DQ4_SPLITS", v}});
                    nu::Flags want = d.flags;
                    const std::optional<std::uint32_t> pv = nu::parseDq4Splits(v);
                    if (fast && pv) want.dq4_splits = *pv;
                    NCHECK(p.flags == want, "dq4_splits {} WHIRL_DQ4_SPLITS='{}': only its own flag, fast only ({})", cell, v, p.flags.dq4_splits);
                    NCHECK((p.bitsKey() != d.bitsKey()) == !(p.flags == d.flags), "dq4_splits {} '{}': bitsKey differs exactly where the flag changes", cell, v);
                    // a note: outside fast for any value but "0" / ""; in fast for a value that does not parse
                    const bool noted = fast ? !pv.has_value() : (std::string_view(v) != "0" && std::string_view(v) != "");
                    NCHECK(!p.lossy_override, "dq4_splits {} '{}': no lossy override", cell, v);
                    if (!noted) NCHECK(nu::logLine(p) == nu::logLine(d), "dq4_splits {} '{}': log line unchanged", cell, v);
                    std::size_t notes = 0;
                    for (const std::string& n : p.ignored) notes += n.rfind(std::string("WHIRL_DQ4_SPLITS=") + v + " ignored", 0) == 0;
                    NCHECK(notes == (noted ? 1u : 0u) && p.ignored.size() == d.ignored.size() + (noted ? 1u : 0u), "dq4_splits {} '{}': {} notes", cell, v, notes);
                    // precise / balance: plan and bitsKey never move
                    if (!fast) NCHECK(p.flags == d.flags && p.bitsKey() == d.bitsKey(), "dq4_splits {} '{}': precise / balance moved", cell, v);
                }
                cells += 1;
            }
    NCHECK(cells == 12, "dq4_splits cells {}", cells);
    // the default 8060S Ornith fast plan: bitsKey carries dq4_splits=65535 (auto: the profile table); WHIRL_DQ4_SPLITS=0 (today's
    // rule, the screen's A arm) has another key
    for (const CardFake& c : k_cards) {
        if (std::string_view(c.name) != "8060s") continue;
        const nu::Target t = target(c, "moe", false);
        const nu::Plan d = pl(t, "fast", {}), z = pl(t, "fast", {{"DQ4_SPLITS", "0"}});
        NCHECK(d.bitsKey().find("dq4_splits=65535;") != std::string::npos && z.bitsKey().find("dq4_splits=0;") != std::string::npos &&
                   d.bitsKey() != z.bitsKey(),
               "8060s Ornith fast: default key auto, WHIRL_DQ4_SPLITS=0 key 0");
    }
}

static void testAttnI4qkFlag() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    std::size_t on_cells = 0;
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const nu::Target t = target(c, model, false);
                const bool want = std::string_view(mode) == "fast" && c.prof.attn_i4qk_fast;
                const nu::Plan d = pl(t, mode, {});
                NCHECK(d.flags.attn_i4qk == want, "attn_i4qk default {} {} {}: {} (want {})", c.name, model, mode, d.flags.attn_i4qk, want);
                on_cells += d.flags.attn_i4qk;
                // "0": off, nothing else moves except its second form (v0252-8sattn3: attn_i4pf rides on attn_i4qk; since the
                // round-3 default flip the 8060S fast plan has it on, and WHIRL_ATTN_I4QK=0 takes it off with the int4 form)
                const nu::Plan z = pl(t, mode, {{"ATTN_I4QK", "0"}});
                nu::Plan a = d;
                a.flags.attn_i4qk = false;
                a.flags.attn_i4pf = false;
                NCHECK(!z.flags.attn_i4qk && a.bitsKey() == z.bitsKey() && nu::logLine(z) == nu::logLine(d) && z.overrides == d.overrides &&
                           z.lossy_override == d.lossy_override && z.ignored.empty(),
                       "attn_i4qk {} {} {}: WHIRL_ATTN_I4QK=0 turns only the attn_i4qk flag off", c.name, model, mode);
                NCHECK((d.bitsKey() != z.bitsKey()) == want, "attn_i4qk {} {} {}: the arms default / =0 must differ exactly where it runs", c.name, model,
                       mode);
                for (const char* v : {"1", "x", ""}) {
                    const nu::Plan o = pl(t, mode, {{"ATTN_I4QK", v}});
                    NCHECK(o.flags.attn_i4qk == want && o.bitsKey() == d.bitsKey() && o.overrides == d.overrides && o.lossy_override == d.lossy_override,
                           "attn_i4qk {} {} {}: WHIRL_ATTN_I4QK='{}' keeps the plan's choice", c.name, model, mode, v);
                    std::size_t notes = 0;
                    for (const std::string& n : o.ignored) notes += n.rfind(std::string("WHIRL_ATTN_I4QK=") + v + " ignored", 0) == 0;
                    const std::size_t want_notes = std::string_view(mode) == "fast" ? 0 : 1;
                    NCHECK(notes == want_notes && o.ignored.size() == want_notes, "attn_i4qk {} {} {}: WHIRL_ATTN_I4QK='{}': {} ignored notes (want {})",
                           c.name, model, mode, v, notes, want_notes);
                }
            }
    NCHECK(on_cells == 2, "attn_i4qk: on by default in exactly 2 cells (8060S fast, both models), got {}", on_cells);
    // a model without MXFP4 tensors: off
    for (const char* model : {"swift", "moe"}) {
        nu::Target tq = target(card("8060s"), model, false);
        tq.has_mxfp4 = false;
        NCHECK(!pl(tq, "fast", {}).flags.attn_i4qk, "attn_i4qk {}: a model without MXFP4 tensors -> off", model);
    }
}

// v0252-w4a4k2 (PLAN V1-144, kill-screen K2): WHIRL_W4A4_SIM, the W4A4 fake-quant research arm of the fast gfx1151 dense
// prefill GEMM (numerics flag w4a4_sim, bits nu::w4a4_sim_w / _a / _am). OFF by default in every plan. Honoured only in fast,
// on the 8060S card (gfx1151 object: the only one with the w4s kernels), on a DENSE model with MXFP4 tensors (Swift; the
// MoE expert path is not simulated); precise / balance never take it (one Plan::ignored note each, plan and bitsKey
// untouched), nor the R9700, nor a MoE model (one note). Values: "w" weights, "a" activations (absmax / 7), "am" activations
// (MSE-best step), "wa" / "1" both, "wam" both with the MSE step; "0" / "" off without a note; anything else off with one
// note. A value that turns it on changes exactly w4a4_sim (bitsKey differs, every other flag equal). Fails to build before the
// change (no EnvValues::w4a4_sim / Flags::w4a4_sim / parseW4a4Sim).
static void testW4a4SimFlag() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    NCHECK(nu::w4a4_sim_w == 1u && nu::w4a4_sim_a == 2u && nu::w4a4_sim_am == 4u, "w4a4_sim bits 1 / 2 / 4");
    const std::pair<const char*, std::uint32_t> vals[] = {{"w", 1u}, {"a", 2u}, {"am", 4u}, {"wa", 3u}, {"1", 3u}, {"wam", 5u}};
    for (const auto& [s, b] : vals) NCHECK(nu::parseW4a4Sim(s) == std::optional<std::uint32_t>(b), "parseW4a4Sim({}) = {}", s, b);
    for (const char* s : {"0", ""}) NCHECK(nu::parseW4a4Sim(s) == std::optional<std::uint32_t>(0u), "parseW4a4Sim('{}') = 0", s);
    for (const char* s : {"2", "x", "aw", "W", "m", "wm", " w", "wa "}) NCHECK(!nu::parseW4a4Sim(s), "parseW4a4Sim('{}') = invalid", s);
    std::size_t on_cells = 0;
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const nu::Target t = target(c, model, false);
                const nu::Plan d = pl(t, mode, {});
                NCHECK(d.flags.w4a4_sim == 0u, "w4a4_sim {} {} {}: off by default ({})", c.name, model, mode, d.flags.w4a4_sim);
                const bool can = std::string_view(mode) == "fast" && c.prof.numerics_8060s && std::string_view(model) == "swift";
                for (const auto& [s, b] : vals) {
                    const nu::Plan o = pl(t, mode, {{"W4A4_SIM", s}});
                    nu::Flags a = o.flags;
                    a.w4a4_sim = 0;
                    NCHECK(o.flags.w4a4_sim == (can ? b : 0u) && a == d.flags && o.lossy_override == d.lossy_override && o.overrides == d.overrides,
                           "w4a4_sim {} {} {}: '{}' -> {} (want {}), nothing else moves", c.name, model, mode, s, o.flags.w4a4_sim, can ? b : 0u);
                    NCHECK((o.bitsKey() != d.bitsKey()) == can, "w4a4_sim {} {} {}: '{}' changes bitsKey exactly where it is taken", c.name, model, mode, s);
                    std::size_t notes = 0;
                    for (const std::string& n : o.ignored) notes += n.rfind(std::string("WHIRL_W4A4_SIM=") + s + " ignored", 0) == 0;
                    NCHECK(notes == (can ? 0u : 1u) && o.ignored.size() == d.ignored.size() + notes, "w4a4_sim {} {} {}: '{}': {} notes (want {})", c.name,
                           model, mode, s, notes, can ? 0 : 1);
                    on_cells += o.flags.w4a4_sim == b && can;
                }
                for (const char* s : {"0", ""}) {
                    const nu::Plan z = pl(t, mode, {{"W4A4_SIM", s}});
                    NCHECK(z.flags == d.flags && z.bitsKey() == d.bitsKey() && z.ignored == d.ignored, "w4a4_sim {} {} {}: '{}' = off, no note", c.name,
                           model, mode, s);
                }
                const nu::Plan bad = pl(t, mode, {{"W4A4_SIM", "bogus"}});
                std::size_t bn = 0;
                for (const std::string& n : bad.ignored) bn += n.rfind("WHIRL_W4A4_SIM=bogus ignored", 0) == 0;
                NCHECK(bad.flags == d.flags && bn == 1, "w4a4_sim {} {} {}: an invalid value = off + one note", c.name, model, mode);
            }
    NCHECK(on_cells == 6, "w4a4_sim taken in {} value-cells (want 6: 8060S fast swift x 6 values)", on_cells);
    nu::Target tq = target(card("8060s"), "swift", false);
    tq.has_mxfp4 = false;
    NCHECK(pl(tq, "fast", {{"W4A4_SIM", "1"}}).flags.w4a4_sim == 0u, "w4a4_sim: a model without MXFP4 tensors -> off");
}

// v0252-8sattn3 (PLAN V1-148): the plan flag attn_i4pf picks the second int4 form of the fast gfx1151 prefill attention
// (attn_prefill_wmma_i4p / _s_i4p: int4 Q codes in LDS + K / V register prefetch, bit-identical to attn_prefill_wmma_i4 /
// _s_i4). It can only be on where attn_i4qk is on (fast, 8060S, MXFP4). Round 3 (default flip): device::Profile::
// attn_i4_qlds_pf is TRUE in the gfx1151 row and false in the gfx1201 row, so a default 8060S fast plan takes it (both
// models); WHIRL_ATTN_I4PF "0" / "" off (the previous kernel, the reference arm), any other value on, unset = the row.
// WHIRL_ATTN_I4QK=0 turns both off. A non-"0" WHIRL_ATTN_I4PF in precise / balance is noted once and changes nothing.
// Fails before the flip: the gfx1151 row is false and the default plan has attn_i4pf off.
static void testAttnI4pfFlag() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    for (const CardFake& c : k_cards)
        NCHECK(c.prof.attn_i4_qlds_pf == (std::string_view(c.name) == "8060s"), "attn_i4pf: device row {} = {} (want on for the 8060S only)", c.name,
               c.prof.attn_i4_qlds_pf);
    std::size_t on_default = 0, on_env = 0, on_prof = 0;
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const nu::Target t = target(c, model, false);
                const bool i4 = std::string_view(mode) == "fast" && c.prof.attn_i4qk_fast;  // where the int4 form runs
                const bool want = i4 && c.prof.attn_i4_qlds_pf;                             // the shipped default
                nu::Target t0 = t;  // the row off (the plan before the flip)
                t0.attn_i4pf_profile = false;
                const nu::Plan d = pl(t, mode, {}), d0 = pl(t0, mode, {});
                NCHECK(d.flags.attn_i4pf == want, "attn_i4pf default {} {} {}: {} (want {})", c.name, model, mode, d.flags.attn_i4pf, want);
                NCHECK(!d0.flags.attn_i4pf, "attn_i4pf row-off {} {} {}: on", c.name, model, mode);
                on_default += d.flags.attn_i4pf;
                nu::Plan da = d;
                da.flags.attn_i4pf = false;
                NCHECK(da.bitsKey() == d0.bitsKey() && nu::logLine(d) == nu::logLine(d0) && d.overrides == d0.overrides && d.lossy_override == d0.lossy_override,
                       "attn_i4pf default {} {} {}: only the attn_i4pf flag moves", c.name, model, mode);
                // "0" and "": off (the previous kernel), nothing else moves; no note for "0" (a non-"0" value, "" too, is noted once
                // in precise / balance - the WHIRL_DQ4_ORDER rule)
                for (const char* v : {"0", ""}) {
                    const nu::Plan z = pl(t, mode, {{"ATTN_I4PF", v}});
                    const std::size_t z_notes = std::string_view(v) == "0" || std::string_view(mode) == "fast" ? 0 : 1;
                    NCHECK(!z.flags.attn_i4pf && z.bitsKey() == d0.bitsKey() && nu::logLine(z) == nu::logLine(d0) && z.overrides == d0.overrides &&
                               z.ignored.size() == z_notes,
                           "attn_i4pf {} {} {}: WHIRL_ATTN_I4PF='{}' = the previous kernel only", c.name, model, mode, v);
                    NCHECK((d.bitsKey() != z.bitsKey()) == want, "attn_i4pf {} {} {}: the arms default / ='{}' differ exactly where it runs", c.name,
                           model, mode, v);
                }
                // the profile row forced on: exactly the int4 cells
                nu::Target tp = t;
                tp.attn_i4pf_profile = true;
                const nu::Plan p = pl(tp, mode, {});
                NCHECK(p.flags.attn_i4pf == i4, "attn_i4pf profile-on {} {} {}: {} (want {})", c.name, model, mode, p.flags.attn_i4pf, i4);
                on_prof += p.flags.attn_i4pf;
                // any other value with the row off: on exactly where the int4 form runs; elsewhere one ignored note in
                // precise / balance and nothing else
                for (const char* v : {"1", "x"}) {
                    const nu::Plan o = pl(t0, mode, {{"ATTN_I4PF", v}});
                    NCHECK(o.flags.attn_i4pf == i4, "attn_i4pf {} {} {}: WHIRL_ATTN_I4PF='{}' -> {} (want {})", c.name, model, mode, v, o.flags.attn_i4pf, i4);
                    if (std::string_view(v) == "1") on_env += o.flags.attn_i4pf;
                    nu::Plan oa = o;
                    oa.flags.attn_i4pf = false;
                    NCHECK(oa.bitsKey() == d0.bitsKey() && o.overrides == d0.overrides && o.lossy_override == d0.lossy_override,
                           "attn_i4pf {} {} {}: WHIRL_ATTN_I4PF='{}' moves only the attn_i4pf flag", c.name, model, mode, v);
                    std::size_t notes = 0;
                    for (const std::string& n : o.ignored) notes += n.rfind(std::string("WHIRL_ATTN_I4PF=") + v + " ignored", 0) == 0;
                    const std::size_t want_notes = std::string_view(mode) == "fast" ? 0 : 1;
                    NCHECK(notes == want_notes && o.ignored.size() == want_notes, "attn_i4pf {} {} {}: WHIRL_ATTN_I4PF='{}': {} ignored notes (want {})",
                           c.name, model, mode, v, notes, want_notes);
                    // with the shipped row: on where the int4 form runs
                    NCHECK(pl(t, mode, {{"ATTN_I4PF", v}}).flags.attn_i4pf == i4, "attn_i4pf {} {} {}: WHIRL_ATTN_I4PF='{}' on the shipped row", c.name,
                           model, mode, v);
                }
                // WHIRL_ATTN_I4QK=0 turns the int4 form off, and with it this one
                NCHECK(!pl(t, mode, {{"ATTN_I4QK", "0"}}).flags.attn_i4pf && !pl(t, mode, {{"ATTN_I4PF", "1"}, {"ATTN_I4QK", "0"}}).flags.attn_i4pf,
                       "attn_i4pf {} {} {}: needs attn_i4qk", c.name, model, mode);
            }
    NCHECK(on_default == 2 && on_env == 2 && on_prof == 2,
           "attn_i4pf: on in exactly 2 cells (8060S fast, both models) by default {} / by env {} / by profile {}", on_default, on_env, on_prof);
    for (const char* model : {"swift", "moe"}) {
        nu::Target tq = target(card("8060s"), model, false);
        tq.has_mxfp4 = false;
        NCHECK(!pl(tq, "fast", {}).flags.attn_i4pf && !pl(tq, "fast", {{"ATTN_I4PF", "1"}}).flags.attn_i4pf,
               "attn_i4pf {}: a model without MXFP4 tensors -> off", model);
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

// v0252-l2split (PLAN V1-88): numerics flag attn_lsplit = the decode / verify split count rule (ops::lsplitCount). By MODE:
// precise 0 (today's split, bit-identical, golden unchanged) on every card, model and value of WHIRL_ATTN_LSPLIT (the variable
// is ignored there with one Plan::ignored note); balance / fast "auto" (device::Profile::attn_lsplit_blocks); WHIRL_ATTN_LSPLIT
// = auto / 0..4096 overrides only that flag in balance / fast; an invalid value keeps auto with one note. Fails before
// v0252-l2split (no flag, no parseLsplit).
static void testLsplitFlag() {
    auto pl = [&](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals) {
        nu::Request r = nu::parseMode(mode, "t");
        r.load_overrides = true;
        for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
        return nu::plan(r, t);
    };
    const std::uint32_t A = nu::kAttnLsplitAuto;
    NCHECK(A == 0xFFFFFFFFu, "kAttnLsplitAuto {}", A);
    NCHECK(nu::Flags{}.attn_lsplit == 0, "Flags default attn_lsplit {} (want 0: today's split)", nu::Flags{}.attn_lsplit);
    const std::pair<const char*, std::uint32_t> valid[] = {{"0", 0}, {"64", 64}, {"128", 128}, {"0192", 192}, {"4096", 4096}, {"auto", A}};
    const char* const invalid[] = {"4097", "", "x", "-1", "64b", "AUTO", " 64", "10000", "99999"};
    for (const auto& [v, want] : valid) NCHECK(nu::parseLsplit(v) == want, "parseLsplit('{}')", v);
    for (const char* v : invalid) NCHECK(!nu::parseLsplit(v).has_value(), "parseLsplit('{}') must be invalid", v);
    for (const char* mode : {"precise", "balance", "fast"})
        for (const CardFake& c : k_cards)
            for (const char* model : {"swift", "moe"}) {
                const bool precise = std::string_view(mode) == "precise";
                const nu::Target t = target(c, model, false);
                const nu::Plan d = pl(t, mode, {});
                const std::uint32_t want_d = precise ? 0u : A;
                NCHECK(d.flags.attn_lsplit == want_d, "attn_lsplit default {} {} {}: {} (want {})", c.name, model, mode, d.flags.attn_lsplit, want_d);
                NCHECK(!d.lossy_override && d.ignored.empty(), "attn_lsplit default {} {} {}: the default is not an override", c.name, model, mode);
                for (const auto& [v, want] : valid) {
                    const nu::Plan p = pl(t, mode, {{"ATTN_LSPLIT", v}});
                    if (precise) {
                        // precise: never another split; the variable is reported once and changes nothing else
                        NCHECK(p.flags.attn_lsplit == 0 && p.bitsKey() == d.bitsKey() && !p.lossy_override && p.ignored.size() == 1 &&
                                   p.ignored[0].rfind(std::string("WHIRL_ATTN_LSPLIT=") + v + " ignored: precise", 0) == 0,
                               "attn_lsplit precise {} {}: WHIRL_ATTN_LSPLIT='{}' -> {} ({} notes)", c.name, model, v, p.flags.attn_lsplit, p.ignored.size());
                        continue;
                    }
                    NCHECK(p.flags.attn_lsplit == want, "attn_lsplit {} {} {}: WHIRL_ATTN_LSPLIT='{}' -> {} (want {})", c.name, model, mode, v,
                           p.flags.attn_lsplit, want);
                    nu::Plan a = d;
                    a.flags.attn_lsplit = want;
                    NCHECK(a.bitsKey() == p.bitsKey() && nu::logLine(p) == nu::logLine(d) && p.overrides == d.overrides &&
                               p.lossy_override == d.lossy_override && p.ignored.empty(),
                           "attn_lsplit {} {} {}: WHIRL_ATTN_LSPLIT='{}' changes only the attn_lsplit flag", c.name, model, mode, v);
                }
                for (const char* v : invalid) {
                    const nu::Plan p = pl(t, mode, {{"ATTN_LSPLIT", v}});
                    NCHECK(p.flags.attn_lsplit == want_d, "attn_lsplit {} {} {}: invalid WHIRL_ATTN_LSPLIT='{}' -> {} (want the default)", c.name, model,
                           mode, v, p.flags.attn_lsplit);
                    std::size_t notes = 0;
                    for (const std::string& n : p.ignored) notes += n.rfind(std::string("WHIRL_ATTN_LSPLIT=") + v + " ignored", 0) == 0;
                    NCHECK(notes == 1 && p.ignored.size() == 1 && p.bitsKey() == d.bitsKey() && p.overrides == d.overrides,
                           "attn_lsplit {} {} {}: invalid WHIRL_ATTN_LSPLIT='{}': one Plan::ignored note, nothing else changes", c.name, model, mode, v);
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

// v0252-f5 (F5 moeskip, ported from feat027-moeskip-r1 4b6474b; DISCOVERY v4 F5 :206, rule :192): the selector side of the
// plan. moeskip drops routed MoE slots whose renormalized router weight is below a threshold (fast-only, opt-in):
//  S1 the item: fast-only, implemented, opt-in (fast_optin_items), never a balance item; the shipped per-card default pins
//     (both ON since v0252-b5 round 3, PLAN V1-116; they were off until the G5 / G6 screens) and the default arm (static
//     top 6: thr 1 / min 6);
//  S2 shipped state: a plain --fast requests it for the MoE model with MXFP4 experts on both cards (static top 6, not an
//     override: in the log line and /props); a dense model's plan never mentions it (moe_skip_thr 0);
//  S3 the per-card fast default rule (Target::moeskip_default, what flipping a constant does): plain fast with an MXFP4
//     MoE model requests and enables it with the default threshold (not an override); a dense model is not requested
//     (plan unchanged); precise / balance and an explicit fast list ignore the rule;
//  S4 explicit requests (fast:moeskip, fast:all, a list with it): on, flags = the defaults, the note says what it does;
//  S5 skip reasons (dense, experts not MXFP4, no kernels);
//  S6 WHIRL_MOESKIP: precise / balance ignore it (plan and bitsKey unchanged, one Plan::ignored note; "0" no note); in fast
//     T > 0 = on with that threshold (a user override where the mode did not request it), "T:M" also sets the top-M floor,
//     "1:M" = static top-M, "0" = off, "" = nothing; a malformed value = the plan without it plus one note; not applicable
//     (dense) = off;
//  S7 f16 decode activations (WHIRL_Q8DEC=0: prec_dec) keep moe_skip_thr 0 (the precise decode experts never see id -1);
//  S8 precise / balance plans do not depend on the moeskip inputs of the target at all;
//  S9 parseMoeSkip.
// The Plan / Target values of this test live in several parts (one function each) so no frame sums over MSVC /analyze's 16 KB
// C6262 limit (v0252-f5fix2; same class as the v0252-f5fix test-stack fix in tests/ops/ops_moe_cases.inc). The checks are unchanged.
static nu::Plan msPlan(const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals = {}) {
    nu::Request r = nu::parseMode(mode, "t");
    r.load_overrides = true;
    for (const auto& [n, v] : vals) *r.env.find(n) = std::string(v);
    return nu::plan(r, t);
}
static const nu::ItemState& msSt(const nu::Plan& p) { return p.items[static_cast<std::size_t>(nu::Item::moeskip)]; }
struct MsTargets {
    nu::Target r97, r97d, s8, s8d;
};
static const MsTargets& msT() {
    static const MsTargets t{target(card("r9700"), "moe", false), target(card("r9700"), "swift", false), target(card("8060s"), "moe", false),
                             target(card("8060s"), "swift", false)};
    return t;
}
static void testMoeSkipPlanS1S2() {
    using Item [[maybe_unused]] = nu::Item;
    [[maybe_unused]] auto pl = [](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals = {}) { return msPlan(t, mode, std::move(vals)); };
    [[maybe_unused]] auto st = msSt;
    [[maybe_unused]] const nu::Target& r97 = msT().r97;
    [[maybe_unused]] const nu::Target& r97d = msT().r97d;
    [[maybe_unused]] const nu::Target& s8 = msT().s8;
    [[maybe_unused]] const nu::Target& s8d = msT().s8d;
    // S1
    NCHECK((nu::fast_only_items & nu::bit(Item::moeskip)) != 0 && (nu::balance_items & nu::bit(Item::moeskip)) == 0, "S1 fast-only");
    NCHECK((nu::implemented_items & nu::bit(Item::moeskip)) != 0 && nu::itemFromName("moeskip") == Item::moeskip, "S1 implemented");
    NCHECK((nu::fast_optin_items & nu::bit(Item::moeskip)) != 0 && (nu::fast_optin_items & ~nu::fast_only_items) == 0, "S1 opt-in set");
    // v0252-b5 round 3 (PLAN V1-116; the G5 / G6 screens re-judged KEEP for k6): the per-card fast default rule is ON on both
    // cards and the values without a WHIRL_MOESKIP value are the screened arm k6 = static top 6 (thr 1, min 6)
    NCHECK(nu::moeskip_fast_default_r9700 && nu::moeskip_fast_default_8060s, "S1 shipped: the per-card fast default rule is on on both cards (k6, PLAN V1-116)");
    NCHECK(nu::moeskip_default_thr == 1.f && nu::moeskip_default_min == 6u, "S1 default = static top 6 (thr 1 / min 6: the screened arm k6)");
    NCHECK(r97.k_moe_skip && s8.k_moe_skip && r97.moeskip_default && s8.moeskip_default && !r97d.moe && !s8d.moe, "S1 target fakes");
    // S2 (the shipped state): a plain --fast requests it for the MoE model with MXFP4 experts only; a dense model's plan
    // never mentions it
    for (const CardFake& c : k_cards)
        for (const char* model : {"swift", "moe"}) {
            const nu::Plan p = pl(target(c, model, false), "fast");
            if (std::string_view(model) == "moe")
                NCHECK(st(p).requested && p.on(Item::moeskip) && p.flags.moe_skip_thr == 1.f && p.flags.moe_skip_min == 6u &&
                           nu::logLine(p).find("moeskip (") != std::string::npos && nu::logLine(p).find("[static top 6]") != std::string::npos &&
                           nu::propsJson(p, "f16").find("moeskip") != std::string::npos && !p.lossy_override && p.overrides.empty(),
                       "S2 {} {}: plain fast runs moeskip k6 (static top 6) by default", c.name, model);
            else
                NCHECK(!st(p).requested && !p.on(Item::moeskip) && p.flags.moe_skip_thr == 0.f && p.flags.moe_skip_min == 1u &&
                           nu::logLine(p).find("moeskip") == std::string::npos && nu::propsJson(p, "f16").find("moeskip") == std::string::npos,
                       "S2 {} {}: plain fast does not request moeskip on a dense model", c.name, model);
        }
}

static void testMoeSkipPlanS3() {
    using Item [[maybe_unused]] = nu::Item;
    [[maybe_unused]] auto pl = [](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals = {}) { return msPlan(t, mode, std::move(vals)); };
    [[maybe_unused]] auto st = msSt;
    [[maybe_unused]] const nu::Target& r97 = msT().r97;
    [[maybe_unused]] const nu::Target& r97d = msT().r97d;
    [[maybe_unused]] const nu::Target& s8 = msT().s8;
    [[maybe_unused]] const nu::Target& s8d = msT().s8d;
    // S3 (the rule switched on in the target, as the loader fills it when a constant is true)
    for (const nu::Target* base : {&r97, &s8}) {
        nu::Target on = *base;
        on.moeskip_default = true;
        const nu::Plan p = pl(on, "fast");
        NCHECK(st(p).requested && p.on(Item::moeskip) && p.flags.moe_skip_thr == nu::moeskip_default_thr && p.flags.moe_skip_min == nu::moeskip_default_min &&
                   p.flags.moe_skip_min == 6u && !p.lossy_override && p.overrides.empty() && nu::modeLabel(p) == "fast",
               "S3 {} moe plain fast: moeskip on by the default rule (not an override)", base->gfx1151 ? "8060s" : "r9700");
        NCHECK(nu::logLine(p).find("fast runs as balance") == std::string::npos, "S3 moeskip is a fast item that applies");
        for (const char* mode : {"precise", "balance"}) {
            const nu::Plan a = pl(on, mode), b = pl(*base, mode);
            NCHECK(!a.on(Item::moeskip) && a.flags.moe_skip_thr == 0.f && a.bitsKey() == b.bitsKey() && nu::logLine(a) == nu::logLine(b),
                   "S3 {}: the fast default rule does not reach it", mode);
        }
        const nu::Plan pc = pl(on, "fast:kvq4");
        NCHECK(!st(pc).requested && pc.flags.moe_skip_thr == 0.f, "S3 an explicit fast list is taken as written");
    }
    for (const nu::Target* base : {&r97d, &s8d}) {
        nu::Target dn = *base;
        dn.moeskip_default = true;
        const nu::Plan pd = pl(dn, "fast"), p0 = pl(*base, "fast");
        NCHECK(!st(pd).requested && nu::logLine(pd) == nu::logLine(p0) && pd.bitsKey() == p0.bitsKey(), "S3 dense model: not requested, plan unchanged");
    }
}

static void testMoeSkipPlanS4S5() {
    using Item [[maybe_unused]] = nu::Item;
    [[maybe_unused]] auto pl = [](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals = {}) { return msPlan(t, mode, std::move(vals)); };
    [[maybe_unused]] auto st = msSt;
    [[maybe_unused]] const nu::Target& r97 = msT().r97;
    [[maybe_unused]] const nu::Target& r97d = msT().r97d;
    [[maybe_unused]] const nu::Target& s8 = msT().s8;
    [[maybe_unused]] const nu::Target& s8d = msT().s8d;
    // S4
    for (const nu::Target* t : {&r97, &s8})
        for (const char* spec : {"fast:moeskip", "fast:all", "fast:kvq4,relaxacc,moeskip"}) {
            const nu::Plan p = pl(*t, spec);
            NCHECK(p.on(Item::moeskip) && p.flags.moe_skip_thr == nu::moeskip_default_thr && p.flags.moe_skip_min == 6u &&
                       st(p).note.find("renormalized weight") != std::string::npos && st(p).note.find("[static top 6]") != std::string::npos,
                   "S4 {}: on (note '{}')", spec, st(p).note);
        }
    // S5
    {
        nu::Target q4m = r97;
        q4m.moe_mxfp4 = false;
        nu::Target nok = r97;
        nok.k_moe_skip = false;
        const struct {
            const nu::Target* t;
            const char* why;
        } cases[] = {{&r97d, "dense model"}, {&s8d, "dense model"}, {&q4m, "experts are not MXFP4"}, {&nok, "no moe_topk_skip"}};
        for (const auto& c : cases) {
            const nu::Plan p = pl(*c.t, "fast:moeskip");
            NCHECK(!p.on(Item::moeskip) && p.flags.moe_skip_thr == 0.f && st(p).note.find(c.why) != std::string::npos, "S5 skipped: '{}' (note '{}')", c.why,
                   st(p).note);
        }
    }
}

static void testMoeSkipPlanS6() {
    using Item [[maybe_unused]] = nu::Item;
    [[maybe_unused]] auto pl = [](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals = {}) { return msPlan(t, mode, std::move(vals)); };
    [[maybe_unused]] auto st = msSt;
    [[maybe_unused]] const nu::Target& r97 = msT().r97;
    [[maybe_unused]] const nu::Target& r97d = msT().r97d;
    [[maybe_unused]] const nu::Target& s8 = msT().s8;
    [[maybe_unused]] const nu::Target& s8d = msT().s8d;
    // S6
    for (const char* mode : {"precise", "balance"})
        for (const nu::Target* t : {&r97, &s8}) {
            const nu::Plan base = pl(*t, mode);
            for (const char* v : {"0.05", "1:6", "abc"}) {
                const nu::Plan one = pl(*t, mode, {{"MOESKIP", v}});
                NCHECK(!one.on(Item::moeskip) && one.flags.moe_skip_thr == 0.f && one.bitsKey() == base.bitsKey() && nu::logLine(one) == nu::logLine(base) &&
                           !one.lossy_override && one.overrides.empty(),
                       "S6 {}: WHIRL_MOESKIP={} ignored", mode, v);
                NCHECK(one.ignored.size() == 1 && one.ignored[0] == std::format("WHIRL_MOESKIP={} ignored: moeskip is a fast-only item ({} never takes it)", v, mode),
                       "S6 {}: one refusal note for {} (got {})", mode, v, one.ignored.empty() ? std::string("<none>") : one.ignored[0]);
            }
            const nu::Plan zero = pl(*t, mode, {{"MOESKIP", "0"}});
            NCHECK(zero.ignored.empty() && zero.bitsKey() == base.bitsKey(), "S6 {}: WHIRL_MOESKIP=0 no note", mode);
        }
    for (const nu::Target* t : {&r97, &s8}) {
        // (a user override where the mode did not request it: a target whose fast default rule is off, as before v0252-b5 round 3)
        nu::Target off = *t;
        off.moeskip_default = false;
        const nu::Plan p1 = pl(off, "fast", {{"MOESKIP", "0.08"}});
        NCHECK(p1.on(Item::moeskip) && p1.flags.moe_skip_thr == 0.08f && p1.flags.moe_skip_min == 1u && nu::modeLabel(p1) == "fast+overrides" &&
                   nu::logLine(p1).find("WHIRL_MOESKIP=0.08: moeskip on (lossy, user-requested)") != std::string::npos &&
                   st(p1).note.find("[tau 0.080, top 1 kept]") != std::string::npos,
               "S6 fast: WHIRL_MOESKIP=0.08 on (override; note '{}')", st(p1).note);
        // (with the shipped default the same value retunes the mode's item: testMoeSkipDefaultEnv)
        const nu::Plan p2 = pl(*t, "fast", {{"MOESKIP", "0.03:2"}});
        NCHECK(p2.on(Item::moeskip) && p2.flags.moe_skip_thr == 0.03f && p2.flags.moe_skip_min == 2u && st(p2).note.find("[tau 0.030, top 2 kept]") != std::string::npos,
               "S6 fast: WHIRL_MOESKIP=0.03:2 (note '{}')", st(p2).note);
        const nu::Plan p3 = pl(*t, "fast", {{"MOESKIP", "1:6"}});
        NCHECK(p3.on(Item::moeskip) && p3.flags.moe_skip_thr == 1.f && p3.flags.moe_skip_min == 6u && st(p3).note.find("[static top 6]") != std::string::npos,
               "S6 fast: WHIRL_MOESKIP=1:6 = static top 6 (note '{}')", st(p3).note);
        const nu::Plan p4 = pl(*t, "fast:moeskip", {{"MOESKIP", "0.08"}});
        NCHECK(p4.on(Item::moeskip) && p4.flags.moe_skip_thr == 0.08f && !p4.lossy_override && p4.overrides.empty(),
               "S6 fast:moeskip + WHIRL_MOESKIP=0.08: the mode's item retuned, not an override");
        const nu::Plan p0 = pl(*t, "fast:moeskip", {{"MOESKIP", "0"}});
        NCHECK(!p0.on(Item::moeskip) && p0.flags.moe_skip_thr == 0.f && !p0.lossy_override && st(p0).note == "off by WHIRL_MOESKIP=0",
               "S6 fast:moeskip + WHIRL_MOESKIP=0 off");
        const nu::Plan pe = pl(*t, "fast", {{"MOESKIP", ""}});
        NCHECK(pe.bitsKey() == pl(*t, "fast").bitsKey() && pe.overrides.empty() && pe.ignored.empty(), "S6 empty value: nothing");
        for (const char* bad : {"abc", "1.5", "-0.1", "0.05:0", "0.05:65", "0.05:x", "0.05:2.5", "nan", "0x1", "0.05:", ":2", "inf"}) {
            const nu::Plan pb = pl(*t, "fast", {{"MOESKIP", bad}}), ref = pl(*t, "fast");
            NCHECK(pb.bitsKey() == ref.bitsKey() && pb.overrides == ref.overrides && pb.ignored.size() == 1 &&
                       pb.ignored[0].rfind(std::string("WHIRL_MOESKIP=") + bad + " ignored: not T or T:M", 0) == 0,
                   "S6 fast: malformed WHIRL_MOESKIP={} -> the plan without it and one note (got {})", bad, pb.ignored.empty() ? std::string("<none>") : pb.ignored[0]);
        }
        const nu::Plan pd = pl(t->gfx1151 ? s8d : r97d, "fast", {{"MOESKIP", "0.05"}});
        NCHECK(!pd.on(Item::moeskip) && pd.flags.moe_skip_thr == 0.f, "S6 dense: not applicable, off");
    }
}

static void testMoeSkipPlanS7S9() {
    using Item [[maybe_unused]] = nu::Item;
    [[maybe_unused]] auto pl = [](const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals = {}) { return msPlan(t, mode, std::move(vals)); };
    [[maybe_unused]] auto st = msSt;
    [[maybe_unused]] const nu::Target& r97 = msT().r97;
    [[maybe_unused]] const nu::Target& r97d = msT().r97d;
    [[maybe_unused]] const nu::Target& s8 = msT().s8;
    [[maybe_unused]] const nu::Target& s8d = msT().s8d;
    // S7
    for (const nu::Target* t : {&r97, &s8}) {
        const nu::Plan p = pl(*t, "fast:moeskip", {{"Q8DEC", "0"}});
        NCHECK(p.flags.prec_dec && p.flags.moe_skip_thr == 0.f, "S7 f16 decode activations: moe_skip_thr 0");
    }
    // S8
    for (const CardFake& c : k_cards)
        for (const char* model : {"swift", "moe"})
            for (const char* mode : {"precise", "balance"}) {
                nu::Target a = target(c, model, false), b = a;
                b.k_moe_skip = !a.k_moe_skip;
                b.moeskip_default = true;
                const nu::Plan pa = pl(a, mode), pb = pl(b, mode);
                NCHECK(pa.bitsKey() == pb.bitsKey() && nu::logLine(pa) == nu::logLine(pb) && pa.flags.moe_skip_thr == 0.f && pa.flags.moe_skip_min == 1u,
                       "S8 {} {} {}: precise / balance independent of the moeskip target inputs", c.name, model, mode);
            }
    // S9
    {
        const auto a = nu::parseMoeSkip("0.05"), b = nu::parseMoeSkip("0.08:2"), c = nu::parseMoeSkip("1:6"), z = nu::parseMoeSkip("0"),
                   o = nu::parseMoeSkip("1"), m64 = nu::parseMoeSkip("0.5:64");
        NCHECK(a && a->thr == 0.05f && !a->min_keep && b && b->thr == 0.08f && b->min_keep == 2u && c && c->thr == 1.f && c->min_keep == 6u && z &&
                   z->thr == 0.f && o && o->thr == 1.f && m64 && m64->min_keep == 64u,
               "S9 parseMoeSkip valid values");
        for (const char* bad : {"", "abc", "1.5", "-0.1", "0.05:0", "0.05:65", "0.05:x", "0.05:2.5", "nan", "0x1", "0.05:", ":2", "inf", " 0.05", "0.05 "})
            NCHECK(!nu::parseMoeSkip(bad), "S9 parseMoeSkip('{}') rejects", bad);
    }
}

// v0252-b5 round 3 (PLAN V1-116 moeskip k6 = the Ornith fast default on both cards; V1-95 / V1-94 the 8060S fast combine;
// the combine rule decided 2026-11-13): for {r9700, 8060s} x {swift, moe} x {precise, balance, fast}
//  D1 only MoE + fast has the item on, by default (not an override), with thr 1 / min 6 (static top 6);
//  D2 precise / balance plans (bitsKey, log line) equal those of a target whose default rule is off; a dense fast plan too;
//  D3 the MoE prefill combine: balance 3 on both cards; fast with moeskip ON = the non-f16 combine (8060S kMoepDefault.comb
//     = 1, R9700 0: ops::chooseMoe refuses the moeskip prefill path with the f16 moe_yd), fast with moeskip OFF
//     (WHIRL_MOESKIP=0, or f16 decode activations) = 3 on both cards (item (d) on the 8060S);
//  D4 an explicitly set WHIRL_MOEP_COMB (0 / 1 / 3) wins in fast with moeskip on and off, and in balance; both screen arms
//     are reachable on one exe and differ in bitsKey ("k6 + the non-f16 combine" = no variable, "COMB 3" = WHIRL_MOEP_COMB=3);
//  D5 WHIRL_MOESKIP=0 gives the fast plan of a target whose default rule is off; a "T" without ":M" keeps its own meaning
//     (top 1 kept), "1:6" is the default arm.
// Small helpers return Flags / strings so no frame holds many Plans (MSVC /analyze C6262).
static nu::Flags msFlags(const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals = {}) {
    return msPlan(t, mode, std::move(vals)).flags;
}
static std::string msKey(const nu::Target& t, const char* mode, std::vector<std::pair<const char*, const char*>> vals = {}) {
    return msPlan(t, mode, std::move(vals)).bitsKey();
}
static void testMoeSkipDefaultCell(const CardFake& c, const char* model) {
    using Item = nu::Item;
    const bool g51 = c.prof.numerics_8060s, moe = std::string_view(model) == "moe";
    // the combine of the fast moeskip plan: v0252-8smoe (A) the card's row device::Profile::moep_comb_fast_skip (8060S 5,
    // the atomic scatter), else the non-f16 combine of the card (R9700 0)
    const std::uint32_t comb_skip = c.prof.moep_comb_fast_skip != 0 ? c.prof.moep_comb_fast_skip : g51 ? whirl::kernels::kMoepDefault.comb : 0u;
    const std::string cell = std::format("b5 moeskip default {} {}", c.name, model);
    NCHECK(comb_skip == (g51 ? 5u : 0u), "{}: the fast moeskip combine of the card", cell);
    const nu::Target t = target(c, model, false);
    nu::Target t0 = t;  // the default rule off (the state before round 3)
    t0.moeskip_default = false;
    NCHECK(t.moeskip_default, "{}: the target fake follows the shipped constants (on)", cell);
    // D2
    for (const char* mode : {"precise", "balance"}) {
        const nu::Plan a = msPlan(t, mode);
        NCHECK(!a.on(Item::moeskip) && !msSt(a).requested && a.flags.moe_skip_thr == 0.f && a.flags.moe_skip_min == 1u, "{} {}: moeskip not in the plan", cell, mode);
        NCHECK(a.bitsKey() == msKey(t0, mode) && nu::logLine(a) == nu::logLine(msPlan(t0, mode)), "{} {}: the plan moved with the fast default rule", cell, mode);
    }
    // D1
    {
        const nu::Plan f = msPlan(t, "fast");
        if (!moe) {
            NCHECK(!msSt(f).requested && f.flags.moe_skip_thr == 0.f && f.flags.moep_comb == 0u && f.bitsKey() == msKey(t0, "fast") &&
                       nu::logLine(f) == nu::logLine(msPlan(t0, "fast")),
                   "{} fast: a dense plan moved", cell);
            return;
        }
        NCHECK(msSt(f).requested && f.on(Item::moeskip) && f.flags.moe_skip_thr == 1.f && f.flags.moe_skip_min == 6u && !f.lossy_override &&
                   f.overrides.empty() && f.ignored.empty() && nu::modeLabel(f) == "fast" && msSt(f).note.find("[static top 6]") != std::string::npos,
               "{} fast: moeskip k6 on by default (thr {}, min {}, note '{}')", cell, f.flags.moe_skip_thr, f.flags.moe_skip_min, msSt(f).note);
        // D3
        NCHECK(f.flags.moep_comb == comb_skip, "{} fast, moeskip on: comb {} (want the non-f16 {})", cell, f.flags.moep_comb, comb_skip);
        NCHECK(f.bitsKey() != msKey(t0, "fast"), "{} fast: the default plan equals the plan without moeskip", cell);
    }
    NCHECK(msFlags(t, "balance").moep_comb == 3u, "{} balance: comb 3", cell);
    NCHECK(msFlags(t, "precise").moep_comb == (g51 ? whirl::kernels::kMoepDefault.comb : 0u), "{} precise: the f32 combine", cell);
}
static void testMoeSkipDefaultEnv(const CardFake& c) {
    using Item = nu::Item;
    const bool g51 = c.prof.numerics_8060s;
    // v0252-8smoe (A): the fast moeskip combine = the card's row (8060S 5); comb_f32 = the non-f16 combine without moeskip
    const std::uint32_t comb_f32 = g51 ? whirl::kernels::kMoepDefault.comb : 0u;
    const std::uint32_t comb_skip = g51 ? 5u : 0u;
    const std::string cell = std::format("b5 moeskip default {} moe", c.name);
    const nu::Target t = target(c, "moe", false);
    nu::Target t0 = t;
    t0.moeskip_default = false;
    // D5 / D3: WHIRL_MOESKIP=0 = the fast plan without the default rule, with the f16 combine (3) on both cards
    {
        const nu::Plan z = msPlan(t, "fast", {{"MOESKIP", "0"}});
        NCHECK(!z.on(Item::moeskip) && z.flags.moe_skip_thr == 0.f && z.flags.moe_skip_min == 1u && !z.lossy_override && msSt(z).note == "off by WHIRL_MOESKIP=0",
               "{} fast WHIRL_MOESKIP=0: off (note '{}')", cell, msSt(z).note);
        NCHECK(z.flags.moep_comb == 3u, "{} fast WHIRL_MOESKIP=0: comb {} (want 3: the f16 moe_yd combine where moeskip is off)", cell, z.flags.moep_comb);
        NCHECK(z.flags == msFlags(t0, "fast"), "{} fast WHIRL_MOESKIP=0: flags differ from the plan of a target without the default rule", cell);
    }
    // f16 decode activations (prec_dec): no moeskip, so the mode's f16 combine
    {
        const nu::Flags q = msFlags(t, "fast", {{"Q8DEC", "0"}});
        NCHECK(q.prec_dec && q.moe_skip_thr == 0.f && q.moep_comb == 3u, "{} fast WHIRL_Q8DEC=0: thr {} comb {} (want 0 / 3)", cell, q.moe_skip_thr, q.moep_comb);
    }
    // D5: "T" without ":M" keeps the top 1; "1:6" = the default arm; the value retunes the mode's item (not an override)
    {
        const nu::Plan a = msPlan(t, "fast", {{"MOESKIP", "0.05"}});
        NCHECK(a.on(Item::moeskip) && a.flags.moe_skip_thr == 0.05f && a.flags.moe_skip_min == 1u && a.flags.moep_comb == comb_skip && !a.lossy_override &&
                   a.overrides.empty() && msSt(a).note.find("[tau 0.050, top 1 kept]") != std::string::npos,
               "{} fast WHIRL_MOESKIP=0.05: tau 0.05, top 1 kept, the non-f16 combine (min {}, comb {}, note '{}')", cell, a.flags.moe_skip_min,
               a.flags.moep_comb, msSt(a).note);
        const nu::Flags one = msFlags(t, "fast", {{"MOESKIP", "1"}});
        NCHECK(one.moe_skip_thr == 1.f && one.moe_skip_min == 1u, "{} fast WHIRL_MOESKIP=1: static top 1 (min {})", cell, one.moe_skip_min);
        NCHECK(msKey(t, "fast", {{"MOESKIP", "1:6"}}) == msKey(t, "fast"), "{} fast WHIRL_MOESKIP=1:6 != the default plan", cell);
    }
    // D4: an explicit WHIRL_MOEP_COMB wins
    for (const char* v : {"0", "1", "3"}) {
        const std::uint32_t want = static_cast<std::uint32_t>(v[0] - '0');
        const nu::Plan on = msPlan(t, "fast", {{"MOEP_COMB", v}});
        NCHECK(on.flags.moep_comb == want && on.flags.moe_skip_thr == 1.f && on.flags.moe_skip_min == 6u && on.ignored.empty(),
               "{} fast WHIRL_MOEP_COMB={} (moeskip on): comb {}", cell, v, on.flags.moep_comb);
        NCHECK(msFlags(t, "fast", {{"MOESKIP", "0"}, {"MOEP_COMB", v}}).moep_comb == want, "{} fast WHIRL_MOESKIP=0 WHIRL_MOEP_COMB={}", cell, v);
        NCHECK(msFlags(t, "balance", {{"MOEP_COMB", v}}).moep_comb == want, "{} balance WHIRL_MOEP_COMB={}", cell, v);
    }
    NCHECK(msKey(t, "fast", {{"MOEP_COMB", "3"}}) != msKey(t, "fast"), "{} fast: the arms 'k6 + non-f16 combine' and 'WHIRL_MOEP_COMB=3' share a bitsKey", cell);
    NCHECK(msKey(t, "fast", {{"MOESKIP", "0"}}) != msKey(t, "fast"), "{} fast: the arms 'k6' and 'WHIRL_MOESKIP=0' share a bitsKey", cell);
    // a fast list without the item h16 (f16 intermediates) and without moeskip: the f32 combine
    NCHECK(msFlags(t, "fast:kvq4").moep_comb == comb_f32 && msFlags(t, "fast:kvq4").moe_skip_thr == 0.f, "{} fast:kvq4: comb without h16", cell);
    // v0252-8smoe (A): WHIRL_MOEP_COMB=1 is the 0.2.5.1 arm of the fast moeskip plan (the 8060S default is 5 now: another bitsKey)
    // (the R9700 default stays 0: WHIRL_MOEP_COMB=0 is its default there)
    NCHECK(!g51 || msKey(t, "fast", {{"MOEP_COMB", "1"}}) != msKey(t, "fast"), "{} fast: WHIRL_MOEP_COMB=1 vs the default", cell);
    NCHECK(msKey(t, "fast", {{"MOEP_COMB", g51 ? "5" : "0"}}) == msKey(t, "fast"), "{} fast: WHIRL_MOEP_COMB={} = the default", cell, g51 ? 5 : 0);
    // fast without moeskip keeps its own rule (3, item h16): the new row is the moeskip plan's only
    NCHECK(msFlags(t, "fast", {{"MOESKIP", "0"}}).moep_comb == 3u, "{} fast WHIRL_MOESKIP=0: comb 3", cell);
}
static void testMoeSkipDefault() {
    for (const CardFake& c : k_cards) {
        for (const char* model : {"swift", "moe"}) testMoeSkipDefaultCell(c, model);
        testMoeSkipDefaultEnv(c);
    }
}

static void testMoeSkipPlan() {
    testMoeSkipDefault();  // v0252-b5 round 3 (fails before the change: both default constants false, thr 0.05 / min 1, fast comb by the mode alone)
    testMoeSkipPlanS1S2();
    testMoeSkipPlanS3();
    testMoeSkipPlanS4S5();
    testMoeSkipPlanS6();
    testMoeSkipPlanS7S9();
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
    testMoeYdhDefault();  // v0252-moedf16 (fails before the change: the R9700 balance / fast comb is 0); v0252-ornkv2 8060S balance 3
    testOrnkvPlan();  // v0252-ornkv2: 8060S Ornith balance kvq8 on, cache key, WHIRL_KV=f16 way back (fails before the change)
    testRfuseDefault();  // v0251-a11port (fails before the port: the plan has no gfx1151 default)
    testKgVarFlag();  // v0252-c1port; v0252-c1acc auto default + invalid-value note (fails before c1acc)
    testWs4rFlag();  // v0252-ws4r97b (fails before phase 2: the stub parseWs4r accepts nothing)
    testLsplitFlag();  // v0252-l2split (fails to build before the change: no attn_lsplit / parseLsplit)
    testAttnSxFlag();  // v0252-strixc2 (fails before the port: loadFlags never sets attn_sx)
    testAttnI4qkFlag();  // v0252-8sfp (fails before the port: loadFlags never sets attn_i4qk)
    testW4a4SimFlag();  // v0252-w4a4k2 (fails to build before the change: no Flags::w4a4_sim / parseW4a4Sim)
    testAttnI4pfFlag();  // v0252-8sattn3 (port: fails without the flag; round 3: fails while the gfx1151 row is off)
    testMoepGuwFlag();  // v0252-8smoe (B; fails to build before the change: no Flags::moep_guw / EnvValues::moep_guw)
    testDq4F8Flags();  // v0252-8sq4 (fails to build before the change: no Flags::dq4_order / dq4_min_keys, parseDq4MinKeys)
    testDq4SplitsFlag();  // v0252-8sq5 (fails to build before the change: no Flags::dq4_splits, kDq4SplitsAuto, parseDq4Splits)
    testMoep2Flags();  // v0252-8smoe2 (C / D / E; fails to build before the change: no Flags::moep_dhs / moep_dna / moep_fq)
    testMoedecDefault();  // v0252-c3port (the "=1 on" rows fail before the port); v0252-c3bdef (the R9700 MoE default rows fail before c3bdef)
    testN2bR1();  // arch025-n2b-r1
    testSwitchZeroIsOff();  // audit-fastonly-overrides FO-5 (fails before the fix)
    testSerializeKernelNoGraph();  // v0252-kgcrash (fails to build before the change)
    testFastOnlyOverrides();  // audit-fastonly-overrides FO-1..4 (FO-1 / FO-2 / FO-3 fail before the fix)
    testFastOnlyRefusal();  // bug-fastonly: refusal notes, fast unchanged
    testRelaxMoeArms();  // v0252-f2 (fails before the port: no arm is known)
    testRelaxDenseArm();  // v0252-b5 round 3 (e) (fails to build before the change: no dense arm)
    testGemvhGrpDefault();  // v0252-8sdec (fails to build before the change: no gemvh_grp)
    testMoeSkipPlan();  // v0252-f5 (fails before the port: moeskip not implemented, not opt-in, no WHIRL_MOESKIP, no flags)
    pass += g_p;
    fail += g_f;
}
