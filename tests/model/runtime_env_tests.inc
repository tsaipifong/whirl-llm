// feat026-o6 (2) (owner O6), redone on arch025-N2b by feat026-o6-p25: qwen35::applyRuntimeEnv (src/model/config.cpp),
// the runtime switches of the CLI and the server, included by model_tests.cpp.
// SPDX-License-Identifier: Apache-2.0
//
// For every runtime switch name (numerics::RuntimeValues; GDN_V0 aside: it swaps a kernel the host-only Model cannot
// look up, its rule is pinned by numerics_plan_tests testFrontEnds / testApplyFrontEqualsPlan), a set of values and both
// front ends: the plan applyRuntimeEnv leaves on the Model equals numerics::plan with that front and value in the
// Request (the rules, unit-tested in numerics_plan_tests), and the Model's "D exec:" dump (what the forward reads,
// nf() since N2b) shows exactly those flags; a value that is not a number gives one warning and changes nothing. The
// CLI (cli main.cpp applyRuntimeEnv) and the server (server_main.cpp after the draft model) both call this function.

void testRuntimeEnvShared() {
    namespace nu = whirl::numerics;
    const char* values[] = {"0", "1", "", "2", "8", "12abc", "64", "100", "4096", "x"};
    static const std::vector<std::string> numbers = {"DBG", "GEMV_MAX", "DRAFT_WINDOW", "DRAFT_WINDOW_MIN", "MOE_BN", "GV_NMAX", "HEAD_CHUNK"};
    std::size_t n = 0, n_warn = 0;
    for (const auto& e : nu::RuntimeValues::names()) {
        if (std::string_view(e.name) == "GDN_V0") continue;
        for (const char* v : values)
            for (const nu::Front fr : {nu::Front::cli, nu::Front::server}) {
                const char* fname = fr == nu::Front::cli ? "cli" : "server";
                q::Model m;
                m.env_snap = std::make_shared<const whirl::config::EnvSnapshot>(whirl::config::EnvSnapshot::fromValues({}));
                m.num_plan = nu::plan(nu::parseMode("balance", "t"), nu::Target{});
                std::vector<std::string> warn;
                const std::string name = e.name;
                q::applyRuntimeEnv(
                    m, [&](const char* k) -> std::optional<std::string> { return name == k ? std::optional<std::string>(v) : std::nullopt; }, fr,
                    &warn);
                nu::Request r = nu::parseMode("balance", "t");
                r.front = fr;
                r.runtime.*e.field = std::string(v);
                const nu::Flags want = nu::plan(r, nu::Target{}).flags;
                check(m.num_plan.flags == want, std::format("o6 runtime env: WHIRL_{}={} ({}): Model plan flags != plan(front)", e.name, v, fname));
                const auto d = n2a::dumpKv(q::execDump(m));
                for (const nu::FlagInfo& fi : nu::flagTable()) {
                    if (!*fi.dump_key) continue;  // gdn_v0 (no dump key)
                    const auto it = d.find(fi.dump_key);
                    check(it != d.end() && it->second == nu::flagValue(want, fi),
                          std::format("o6 runtime env: WHIRL_{}={} ({}): dump {} = {}, plan {}", e.name, v, fname, fi.dump_key,
                                      it == d.end() ? std::string("<none>") : it->second, nu::flagValue(want, fi)));
                }
                // warnings: only for a number switch with a value that is not a whole number
                const bool is_num = std::find(numbers.begin(), numbers.end(), name) != numbers.end();
                const std::string sv = v;
                const bool bad = sv.empty() || sv.find_first_not_of("0123456789") != std::string::npos;
                check(warn.size() == ((is_num && bad) ? 1u : 0u), std::format("o6 runtime env: WHIRL_{}={} ({}): {} warnings", e.name, v, fname, warn.size()));
                if (!warn.empty()) {
                    check(warn[0] == std::format("WHIRL_{}={} is not a number; ignored", e.name, v), "o6 runtime env: warning text " + warn[0]);
                    ++n_warn;
                }
                ++n;
            }
    }
    // nothing set: the CLI changes nothing, the server only turns the graph replay off (not a switch)
    for (const nu::Front fr : {nu::Front::cli, nu::Front::server}) {
        q::Model m;
        m.env_snap = std::make_shared<const whirl::config::EnvSnapshot>(whirl::config::EnvSnapshot::fromValues({}));
        m.num_plan = nu::plan(nu::parseMode("balance", "t"), nu::Target{});
        nu::Flags f0 = m.num_plan.flags;
        std::vector<std::string> warn;
        q::applyRuntimeEnv(m, [](const char*) { return std::optional<std::string>(); }, fr, &warn);
        if (fr == nu::Front::server) f0.use_graph = false;
        check(m.num_plan.flags == f0 && warn.empty(), std::format("o6 runtime env: nothing set ({})", static_cast<int>(fr)));
    }
    std::printf("  o6 runtime env: %zu switch values x front (%zu warnings), model == plan(cli / server)\n", n, n_warn);
}
