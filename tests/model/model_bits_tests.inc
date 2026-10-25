// feat026-o6 (1): flagsOf / tierBitsKey (include/whirl/model_bits.h), included by model_tests.cpp.
// SPDX-License-Identifier: Apache-2.0
//
// (a) every numerics::Flags field is read from the Model member of the same "D exec:" dump key: one setter per field
//     changes exactly that field of flagsOf and the dump line agrees (the dump, src/model/exec_dump.cpp, is the
//     independent name -> member map); the setter table covers flagTable() exactly.
// (b) a default Model gives the default Flags (Flags defaults = Model initializers, numerics.h).
// (c) the key is a pure function of the Model: the same Model twice gives the same key; descriptions do not enter it.
// Stability across N2b and the pinned key strings: testPlanVsLoadOrTune (plan.bitsKey() == tierBitsKey for every case,
// and the base cases' keys against tests/model/bits_key.expected.txt).

namespace o6 {
namespace nu = whirl::numerics;

std::map<std::string, std::string> dumpOf(const q::Model& m) {
    std::map<std::string, std::string> o;
    const std::string d = q::execDump(m);
    for (std::size_t p = 0; p < d.size();) {
        std::size_t e = d.find('\n', p);
        if (e == std::string::npos) e = d.size();
        const std::string l = d.substr(p, e - p);
        if (l.rfind("D exec: ", 0) == 0) {
            const std::size_t eq = l.find('=');
            o[l.substr(8, eq - 8)] = l.substr(eq + 1);
        }
        p = e + 1;
    }
    return o;
}

}  // namespace o6

void testModelBitsKey() {
    namespace nu = whirl::numerics;
    {
        q::Model m;
        check(q::flagsOf(m, false) == nu::Flags{}, "o6: default Model -> default Flags");
    }
    struct Setter {
        const char* name;  // FlagInfo::name
        std::function<void(q::Model&)> set;
    };
    // a non-default value per field (gv_group / gv_nmax are process globals: restored after each case)
    const std::vector<Setter> setters = {
        {"use_graph", [](q::Model& m) { m.use_graph = false; }},
        {"no_fuse", [](q::Model& m) { m.no_fuse = true; }},
        {"dbg_flags", [](q::Model& m) { m.dbg_flags = 5; }},
        {"naive_attn", [](q::Model& m) { m.naive_attn = true; }},
        {"attn_wide", [](q::Model& m) { m.attn_wide = false; }},
        {"attn_dq4", [](q::Model& m) { m.attn_dq4 = false; }},
        {"dq4_dyn", [](q::Model& m) { m.dq4_dyn = false; }},
        {"attn_ws4", [](q::Model& m) { m.attn_ws4 = false; }},
        {"wide_verify", [](q::Model& m) { m.wide_verify = false; }},
        {"moe_wide", [](q::Model& m) { m.moe_wide = true; }},
        {"head_chunk", [](q::Model& m) { m.head_chunk = 4096; }},
        {"float_gemv", [](q::Model& m) { m.float_gemv = true; }},
        {"prec_dec", [](q::Model& m) { m.prec_dec = true; }},
        {"gemv_max", [](q::Model& m) { m.gemv_max = 8; }},
        {"fp8_prefill", [](q::Model& m) { m.fp8_prefill = false; }},
        {"fp8_mask", [](q::Model& m) { m.fp8_mask = 5; }},
        {"relax_on", [](q::Model& m) { m.relax.on = true; }},
        {"relax_k", [](q::Model& m) { m.relax.k = 7; }},
        {"relax_alpha", [](q::Model& m) { m.relax.alpha = 0.25f; }},
        {"relax_eps", [](q::Model& m) { m.relax.eps = 0.5f; }},
        {"relax_delta", [](q::Model& m) { m.relax.delta = 0.75f; }},
        {"gdn_chunked", [](q::Model& m) { m.gdn_chunked = false; }},
        {"gdn_wmma", [](q::Model& m) { m.gdn_wmma = true; }},
        {"gdn_strix", [](q::Model& m) { m.gdn_strix = true; }},
        {"gdn_strix_min", [](q::Model& m) { m.gdn_strix_min = 512; }},
        {"act_fuse", [](q::Model& m) { m.act_fuse = false; }},
        {"attn_kx_on", [](q::Model& m) { m.attn_kx_on = false; }},
        {"gdn_ba_on", [](q::Model& m) { m.gdn_ba_on = false; }},
        {"gdn_in2_on", [](q::Model& m) { m.gdn_in2_on = false; }},
        {"attn_dqf_on", [](q::Model& m) { m.attn_dqf_on = false; }},
        {"attn_seg", [](q::Model& m) { m.attn_seg = 768; }},
        {"attn_kg_on", [](q::Model& m) { m.attn_kg_on = false; }},
        {"ffn_h16", [](q::Model& m) { m.ffn_h16 = true; }},
        {"h16_dq", [](q::Model& m) { m.h16_dq = true; }},
        {"g8t", [](q::Model& m) { m.g8t = true; }},
        {"gemmh_on", [](q::Model& m) { m.gemmh_on = false; }},
        {"gemmhq_on", [](q::Model& m) { m.gemmhq_on = true; }},
        {"draft_window", [](q::Model& m) { m.draft_window = 4096; }},
        {"draft_window_min", [](q::Model& m) { m.draft_window_min = 8192; }},
        {"moe_bn_force", [](q::Model& m) { m.moe_bn_force = 64; }},
        {"moe_fp8", [](q::Model& m) { m.moe_fp8 = false; }},
        {"moe_mxw", [](q::Model& m) { m.moe_mxw = false; }},
        {"moe_rbf", [](q::Model& m) { m.moe_rbf = false; }},
        {"gv_group", [](q::Model&) { q::gv_group = false; }},
        {"gv_nmax", [](q::Model&) { q::gv_nmax = 3; }},
    };
    const auto& table = nu::flagTable();
    std::size_t covered = 0;
    for (const nu::FlagInfo& fi : table) {
        if (std::string_view(fi.name) == "gdn_v0") {  // no Model member: the caller's argument
            q::Model m;
            const nu::Flags a = q::flagsOf(m, false), b = q::flagsOf(m, true);
            check(!a.gdn_v0 && b.gdn_v0, "o6: gdn_v0 from the argument");
            check(q::tierBitsKey(m, false) != q::tierBitsKey(m, true), "o6: gdn_v0 changes the key");
            ++covered;
            continue;
        }
        const Setter* s = nullptr;
        for (const Setter& x : setters)
            if (std::string_view(x.name) == fi.name) s = &x;
        check(s != nullptr, std::string("o6: no setter for flag ") + fi.name);
        if (!s) continue;
        ++covered;
        const bool gvg = q::gv_group;
        const std::uint32_t gvn = q::gv_nmax;
        q::Model m;
        const nu::Flags before = q::flagsOf(m, false);
        const std::string key0 = q::tierBitsKey(m, false);
        s->set(m);
        const nu::Flags after = q::flagsOf(m, false);
        const auto d = o6::dumpOf(m);
        const std::string key1 = q::tierBitsKey(m, false);
        q::gv_group = gvg;
        q::gv_nmax = gvn;
        for (const nu::FlagInfo& gi : table) {
            const bool same = nu::flagValue(before, gi) == nu::flagValue(after, gi);
            check(same == (&gi != &fi), std::string("o6: setting ") + fi.name + " changed flag " + gi.name + (same ? " (not)" : ""));
        }
        const auto it = d.find(fi.dump_key);
        check(it != d.end() && it->second == nu::flagValue(after, fi),
              std::string("o6: flag ") + fi.name + " = " + nu::flagValue(after, fi) + ", dump " + fi.dump_key + " = " +
                  (it == d.end() ? std::string("<none>") : it->second));
        check(key0 != key1, std::string("o6: flag ") + fi.name + " does not change the key");
    }
    check(covered == table.size() && setters.size() + 1 == table.size(), "o6: setter table covers flagTable exactly");
    // (c) pure function of the Model; notes, overrides and the source are not in it
    {
        q::Model m;
        m.num_plan = nu::plan(nu::parseMode("balance", "test"), nu::Target{});
        const std::string k = q::tierBitsKey(m, false);
        check(k == q::tierBitsKey(m, false), "o6: key deterministic");
        m.num_plan.overrides.push_back("WHIRL_X=1: note only");
        m.num_plan.source = "command line";
        for (auto& it : m.num_plan.items) it.note += " (reworded)";
        check(k == q::tierBitsKey(m, false), "o6: descriptions not in the key");
        check(k != nu::logLine(m.num_plan), "o6: the key is not the log line");
    }
    std::printf("  o6 model bits key: %zu flags, each read from its Model member\n", covered);
}
