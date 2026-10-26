// feat026-o6-r1: review findings C5 / C6 of progress/o6-ornkv-relaxdet.review.md (A1 is in testPlanVsLoadOrTune),
// included by model_tests.cpp. Written from the findings, before the fix.
// SPDX-License-Identifier: Apache-2.0
//
// C5: every runtime switch the front ends apply (qwen35::applyRuntimeEnv) and the GEMV tune lists are recorded by
// numerics::noteSwitches as an override; a switch whose paths give the same output bits (bitwise) never sets
// lossy_override (the label stays "precise" / "balance"), a lossy one shows "+overrides", a draft-only one is lossy
// only while relaxed acceptance (relaxacc) is on.
// C6: a malformed WHIRL_GEMV_R / _W / _WH is one warning and changes nothing (applyRuntimeEnv, the server's startup),
// never an exception.

namespace o6r1 {
namespace nu = whirl::numerics;

enum class Kind { bitwise, lossy, drafts };
struct Sw {
    const char* name;
    const char* value;
    Kind kind;
};
// the expected classification (evidence: progress/feat026-o6-r1.PRIOR_ART.md 1c and the HANDOFF table)
const Sw k_new[] = {
    {"GEMV_R", "2:2", Kind::bitwise},          // BitClass q8_row: every int8 GEMV variant == the one-token gemvq
    {"GEMV_W", "2:1", Kind::bitwise},          // (checkGemvBitwise gate)
    {"GEMV_WH", "2:1", Kind::bitwise},         // output head W variant, same class
    {"HEAD_CHUNK", "4096", Kind::bitwise},     // rows per head-split launch, 32-row blocks kept (only the grid moves)
    {"DQ4_DYN", "0", Kind::bitwise},           // attention.cpp FAST-8S-1: same partials, same combine order
    {"GDN_V0", "1", Kind::bitwise},            // fused_decode.hip: gdn_step_norm == gdn_step_norm_v0 bit for bit
    {"NO_GRAPH", "1", Kind::bitwise},          // the same kernels launched directly instead of replayed
    {"WIDE_VERIFY", "0", Kind::lossy},         // verify row groups (attention grouping) differ; no bitwise gate
    {"DRAFT_WINDOW", "4096", Kind::drafts},    // draft attention only (trunk and verify see everything)
    {"DRAFT_WINDOW_MIN", "1024", Kind::drafts},
};

nu::Plan noted(const char* mode, const char* name, const char* value, bool relax) {
    nu::Plan p = nu::plan(nu::parseMode(mode, "t"), nu::Target{});
    nu::applyOverride(p, nu::Item::relaxacc, "RELAX", relax ? "1" : "0", relax, true);  // relaxacc exactly as asked
    p.lossy_override = false;  // only what noteSwitches does
    p.overrides.clear();
    const std::string n = name;
    nu::noteSwitches(p, [&](const char* k) -> std::optional<std::string> { return n == k ? std::optional<std::string>(value) : std::nullopt; });
    return p;
}

void testSwitchLabels() {
    std::size_t n = 0;
    for (const Sw& s : k_new)
        for (const char* mode : {"precise", "balance", "fast"})
            for (const bool relax : {false, true}) {
                const nu::Plan p = noted(mode, s.name, s.value, relax);
                check(p.on(nu::Item::relaxacc) == relax, std::format("o6r1 C5: relaxacc {} in {}", relax, mode));
                const std::string want = std::string("WHIRL_") + s.name + "=" + s.value + ":";
                const bool listed = p.overrides.size() == 1 && p.overrides[0].rfind(want, 0) == 0;
                check(listed, std::format("o6r1 C5: WHIRL_{}={} ({}{}) recorded as one override ({} overrides)", s.name, s.value, mode,
                                          relax ? ", relaxacc" : "", p.overrides.size()));
                const bool lossy = s.kind == Kind::lossy || (s.kind == Kind::drafts && relax);
                check(p.lossy_override == lossy, std::format("o6r1 C5: WHIRL_{}={} ({}{}): lossy_override {} (want {})", s.name, s.value, mode,
                                                             relax ? ", relaxacc" : "", p.lossy_override, lossy));
                const std::string label = nu::modeLabel(p);
                check(label == std::string(mode) + (lossy ? "+overrides" : ""),
                      std::format("o6r1 C5: WHIRL_{}={} ({}{}): label {}", s.name, s.value, mode, relax ? ", relaxacc" : "", label));
                if (listed && !lossy)
                    check(p.overrides[0].find("bitwise") != std::string::npos || p.overrides[0].find("drafts only") != std::string::npos,
                          "o6r1 C5: the override text says why the label stays: " + p.overrides[0]);
                ++n;
            }
    // every switch applyRuntimeEnv reads is in the list (or is GEMV_* / one of the 13 lossy ones already there)
    for (const auto& e : nu::RuntimeValues::names()) {
        const nu::Plan p = noted("balance", e.name, "1", false);
        check(p.overrides.size() == 1, std::string("o6r1 C5: runtime switch WHIRL_") + e.name + " is not in k_numerics_switches");
    }
    std::printf("  o6r1 C5: %zu switch / mode cells\n", n);
}

// (feat026-o6-p25: on N2b applyRuntimeEnv takes the front end; both are checked. The server's front always turns the
// graph replay off, so its Model starts with use_graph off here and the dump must not change either.)
void testGemvEnvWarns() {
    namespace nu = whirl::numerics;
    struct C {
        const char* name;
        const char* value;
    };
    const C cs[] = {{"GEMV_R", "2:x"}, {"GEMV_R", "2:2,3:"}, {"GEMV_R", "a"}, {"GEMV_W", "2:1,x:1"}, {"GEMV_WH", ":5"}, {"GEMV_W", "99999999999:1"}};
    for (const nu::Front fr : {nu::Front::cli, nu::Front::server})
        for (const C& c : cs) {
            q::Model m;
            m.env_snap = std::make_shared<const whirl::config::EnvSnapshot>(
                whirl::config::EnvSnapshot::fromValues({{c.name, whirl::config::EnvSnapshot::Value(std::string(c.value))}}));
            if (fr == nu::Front::server) m.num_plan.flags.use_graph = false;
            const std::string d0 = q::execDump(m);
            const auto r0 = m.gemv_r, wh0 = m.gemv_w_head;
            const auto w0 = m.gemv_w;
            std::vector<std::string> warn;
            std::string thrown;
            try {
                q::applyRuntimeEnv(
                    m,
                    [&](const char* k) -> std::optional<std::string> {
                        return std::string_view(k) == c.name ? std::optional<std::string>(c.value) : std::nullopt;
                    },
                    fr, &warn);
            } catch (const std::exception& e) {
                thrown = e.what();
            }
            const int fi = static_cast<int>(fr);
            check(thrown.empty(), std::format("o6r1 C6: WHIRL_{}={} (front {}) threw ({}) instead of a warning", c.name, c.value, fi, thrown));
            check(q::execDump(m) == d0 && m.gemv_r == r0 && m.gemv_w == w0 && m.gemv_w_head == wh0,
                  std::format("o6r1 C6: WHIRL_{}={} (front {}) changed the model (a bad list is not applied, not even in part)", c.name, c.value, fi));
            check(warn.size() == 1 && warn[0] == std::format("WHIRL_{}={} is not a valid nt:value list; ignored", c.name, c.value),
                  std::format("o6r1 C6: WHIRL_{}={} (front {}): one warning ({}: {})", c.name, c.value, fi, warn.size(), warn.empty() ? "" : warn[0]));
        }
    // a good list still applies, with no warning
    for (const nu::Front fr : {nu::Front::cli, nu::Front::server}) {
        q::Model m;
        m.env_snap = std::make_shared<const whirl::config::EnvSnapshot>(
            whirl::config::EnvSnapshot::fromValues({{"GEMV_R", whirl::config::EnvSnapshot::Value(std::string("2:2,3:4"))}}));
        std::vector<std::string> warn;
        q::applyRuntimeEnv(m, [](const char* k) -> std::optional<std::string> {
            return std::string_view(k) == "GEMV_R" ? std::optional<std::string>("2:2,3:4") : std::nullopt;
        }, fr, &warn);
        check(warn.empty() && m.gemv_r[2] == 2 && m.gemv_r[3] == 4, "o6r1 C6: a good WHIRL_GEMV_R list applies without a warning");
    }
    std::printf("  o6r1 C6: malformed GEMV lists warn (cli and server)\n");
}

}  // namespace o6r1
