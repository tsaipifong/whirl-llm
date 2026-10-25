// feat026-o6 (2): qwen35::applyRuntimeEnv (src/model/config.cpp), the runtime switches of the CLI and the server,
// included by model_tests.cpp.
// SPDX-License-Identifier: Apache-2.0
//
// For every runtime switch name (numerics::RuntimeValues, GDN_V0 aside: it swaps a kernel the host-only Model cannot look
// up) and a set of values: the Model members the function changes are exactly the flags numerics::plan changes for
// Front::cli and for Front::server (the plan's rules, unit-tested in numerics_plan_tests testFrontEnds), with the same
// values; a value that is not a number gives one warning and changes nothing. The CLI and the server both call this
// function (cli main.cpp applyRuntimeEnv, server_main.cpp after the draft model), so the server == CLI.

void testRuntimeEnvShared() {
    namespace nu = whirl::numerics;
    const char* values[] = {"0", "1", "", "2", "8", "12abc", "64", "100", "4096", "x"};
    std::size_t n = 0, n_warn = 0;
    for (const auto& e : nu::RuntimeValues::names()) {
        if (std::string_view(e.name) == "GDN_V0") continue;
        for (const char* v : values) {
            const bool gvg = q::gv_group;
            const std::uint32_t gvn = q::gv_nmax;
            q::Model m;
            m.env_snap = std::make_shared<const whirl::config::EnvSnapshot>(whirl::config::EnvSnapshot::fromValues({}));
            const auto before = n2a::dumpKv(q::execDump(m));  // the "D exec:" lines (no dependence on model_bits.h)
            std::vector<std::string> warn;
            const std::string name = e.name;
            q::applyRuntimeEnv(
                m, [&](const char* k) -> std::optional<std::string> { return name == k ? std::optional<std::string>(v) : std::nullopt; },
                &warn);
            const auto after = n2a::dumpKv(q::execDump(m));
            q::gv_group = gvg;
            q::gv_nmax = gvn;
            for (const nu::Front fr : {nu::Front::cli, nu::Front::server}) {
                nu::Request r0 = nu::parseMode("balance", "t");
                r0.front = fr;
                nu::Request r1 = r0;
                r1.runtime.*e.field = std::string(v);
                const nu::Flags p0 = nu::plan(r0, nu::Target{}).flags, p1 = nu::plan(r1, nu::Target{}).flags;
                for (const nu::FlagInfo& fi : nu::flagTable()) {
                    if (!*fi.dump_key) continue;  // gdn_v0 (skipped above)
                    // the server's plan has use_graph off whatever WHIRL_NO_GRAPH says (it never replays the graph)
                    if (fr == nu::Front::server && std::string_view(fi.name) == "use_graph") continue;
                    const std::string b0 = before.count(fi.dump_key) ? before.at(fi.dump_key) : "<none>";
                    const std::string a0 = after.count(fi.dump_key) ? after.at(fi.dump_key) : "<none>";
                    const bool m_changed = b0 != a0;
                    const bool p_changed = nu::flagValue(p0, fi) != nu::flagValue(p1, fi);
                    check(m_changed == p_changed && (!m_changed || a0 == nu::flagValue(p1, fi)),
                          std::format("o6 runtime env: WHIRL_{}={} flag {}: model {} -> {}, plan ({}) {} -> {}", e.name, v, fi.name, b0, a0,
                                      fr == nu::Front::cli ? "cli" : "server", nu::flagValue(p0, fi), nu::flagValue(p1, fi)));
                }
            }
            // warnings: only for a number switch with a value that is not a whole number
            static const std::vector<std::string> numbers = {"DBG", "GEMV_MAX", "DRAFT_WINDOW", "DRAFT_WINDOW_MIN", "MOE_BN", "GV_NMAX", "HEAD_CHUNK"};
            const bool is_num = std::find(numbers.begin(), numbers.end(), name) != numbers.end();
            const std::string sv = v;
            const bool bad = sv.empty() || sv.find_first_not_of("0123456789") != std::string::npos;
            check(warn.size() == ((is_num && bad) ? 1u : 0u), std::format("o6 runtime env: WHIRL_{}={}: {} warnings", e.name, v, warn.size()));
            if (!warn.empty()) {
                check(warn[0] == std::format("WHIRL_{}={} is not a number; ignored", e.name, v), "o6 runtime env: warning text " + warn[0]);
                ++n_warn;
            }
            ++n;
        }
    }
    // nothing set: nothing changes
    {
        q::Model m;
        m.env_snap = std::make_shared<const whirl::config::EnvSnapshot>(whirl::config::EnvSnapshot::fromValues({}));
        const std::string d0 = q::execDump(m);
        std::vector<std::string> warn;
        q::applyRuntimeEnv(m, [](const char*) { return std::optional<std::string>(); }, &warn);
        check(q::execDump(m) == d0 && warn.empty(), "o6 runtime env: nothing set");
    }
    std::printf("  o6 runtime env: %zu switch values (%zu warnings), model == plan(cli) == plan(server)\n", n, n_warn);
}
