// whirl-iso-server-args-tests: src/server/server_args.cpp built with only its declared dependencies (numerics::resolve
// of src/common/numerics.cpp, its dump::kv in src/common/dump.cpp, platform env) (ARCH-1 8.5 isolation, arch025-isoall).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/server_args.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace whirl::server;

namespace {

bool visOk(std::string_view m) { return m == "auto" || m == "gpu" || m == "cpu"; }

void testDefaults() {
    const Options o = parseServerArgs({"m.gguf"}, visOk);
    ISO_CHECK(o.path == "m.gguf" && o.host == "127.0.0.1" && o.port == 8080 && o.parallel == 4 && !o.parallel_given);
    ISO_CHECK(!o.mode_spec.has_value());
    ISO_THROWS(parseServerArgs({}, visOk), ArgError);  // missing MODEL.gguf
}

void testFlags() {
    const Options o = parseServerArgs({"m.gguf", "--port", "9001", "--parallel", "2"}, visOk);
    ISO_CHECK(o.port == 9001 && o.parallel == 2 && o.parallel_given);
    ISO_THROWS(parseServerArgs({"m.gguf", "--port", "x"}, visOk), ArgError);
    ISO_THROWS(parseServerArgs({"m.gguf", "--no-such-option"}, visOk), ArgError);
}

void testResolve() {
    Options o;
    const ResolvedConfig r = resolveServerConfig(o, std::nullopt);
    ISO_CHECK(r.opt.parallel == 4);
    Options bad;
    bad.parallel = 17;
    ISO_THROWS(resolveServerConfig(bad, std::nullopt), ArgError);
    Options d;
    ISO_THROWS(checkKvSsdDirEnv(d, std::optional<std::string>("")), ArgError);
}

// arch025-lane1 leg 8: the model name /v1/models and the startup lines show (5728296 src/server/server_main.cpp:953-958)
std::string old_name(const std::string& base, const std::optional<std::string>& alias) {
    std::string name = alias ? *alias : base;
    if (!alias && name.size() > 5) {
        std::string tail = name.substr(name.size() - 5);
        for (char& c : tail) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (tail == ".gguf") name = name.substr(0, name.size() - 5);
    }
    return name;
}

// :1543-1544 the SSD tier mode: --kv-ssd-mode, else WHIRL_KV_SSD_MODE, else "shutdown"; anything but always / off = shutdown
std::string old_ssd_mode(const std::optional<std::string>& cli, const std::optional<std::string>& env) {
    std::string ssd_mode = cli.value_or(env.value_or("shutdown"));
    if (ssd_mode != "always" && ssd_mode != "off") ssd_mode = "shutdown";
    return ssd_mode;
}

void testKvSsdMode() {
    int n = 0;
    const std::optional<std::string> vs[] = {std::nullopt, std::string("always"), std::string("off"), std::string("shutdown"), std::string(""),
                                             std::string("Always"), std::string("x")};
    for (const auto& c : vs)
        for (const auto& e : vs) {
            ISO_CHECK(kvSsdMode(c, e) == old_ssd_mode(c, e));
            ++n;
        }
    std::printf("kv ssd mode: %d cases == 5728296\n", n);
}

void testServedModelName() {
    int n = 0;
    for (const char* b : {"", "a", ".gguf", "x.gguf", "Swift-1.5-27B-MXFP4.GGUF", "m.GgUf", "model.gguf.bak", "model.bin", "ab.ggu",
                          "\xe6\xa8\xa1\xe5\x9e\x8b.gguf", "a.gguf.gguf", "     .gguf"})
        for (const std::optional<std::string>& al : {std::optional<std::string>(), std::optional<std::string>(""),
                                                     std::optional<std::string>("alias.gguf"), std::optional<std::string>("x")}) {
            ISO_CHECK(servedModelName(b, al) == old_name(b, al));
            ++n;
        }
    ISO_CHECK(servedModelName("Swift.gguf", std::nullopt) == "Swift" && servedModelName("Swift.gguf", std::string("s")) == "s");
    std::printf("served model name: %d cases == 5728296\n", n);
}

}  // namespace

// ---- leg 10: verbatim c1ca7ce src/server/server_main.cpp (ServerRun::startTier) ------------------------------------
// :1512-1514 the SSD cap (WHIRL_KV_SSD_GB read only without --kv-ssd-gb)
std::uint64_t old_ssdGb(const std::optional<std::uint64_t>& cli, const std::optional<std::uint64_t>& env, int& reads) {
    std::uint64_t ssd_gb = 64;
    if (cli) ssd_gb = *cli;
    else if (const std::optional<std::uint64_t> g = (++reads, env)) ssd_gb = *g;
    return ssd_gb;
}
// :1518-1524 the SSD directory (platform::paths::kvCacheDir faked: <base>\whirl\kvcache)
std::optional<std::string> old_ssdDir(std::uint64_t ssd_gb, const std::string& ssd_mode, const std::optional<std::string>& cli,
                                      const std::optional<std::string>& env, const char* lad, int& env_reads, int& def_calls) {
    std::optional<std::string> ssd_dir;
    if (ssd_gb > 0 && ssd_mode != "off") {
        if (cli) ssd_dir = cli;
        else if (auto v = (++env_reads, env)) ssd_dir = *v;
        else if (lad) ssd_dir = (++def_calls, std::string(lad) + "\\whirl\\kvcache");
    }
    return ssd_dir;
}
// :1517 / :1576-1580 which tier branch, and the layout warning
int old_tierStart(bool tier_pre, std::uint64_t page_row, std::uint64_t per_tok, std::uint64_t kv_ram_mb) {
    if (tier_pre && page_row == per_tok) return 0;
    else if (kv_ram_mb > 0 && tier_pre && page_row != per_tok) return 1;
    else return 2;
}
std::string old_layoutWarn(std::uint64_t page_row, std::uint64_t per_tok) {
    return std::format("kv tier: KV layout not understood ({} vs {} bytes per token); host tiers off", page_row, per_tok);
}
void testKvTierSetup() {
    std::size_t n = 0;
    const std::vector<std::optional<std::uint64_t>> gbs = {std::nullopt, 0ull, 1ull, 64ull, 1000ull, 0xffffffffffffffffull};
    for (const auto& c : gbs)
        for (const auto& e : gbs) {
            int r0 = 0, r1 = 0;
            const std::uint64_t o = old_ssdGb(c, e, r0);
            ISO_CHECK(whirl::server::kvSsdGb(c, [&] { ++r1; return e; }) == o);
            ISO_CHECK(r1 == r0);
            ++n;
        }
    ISO_CHECK(whirl::server::kv_ssd_gb_default == 64);
    const std::vector<std::optional<std::string>> dirs = {std::nullopt, std::string(), std::string("D:\\kv"), std::string("C:\\x y\\z")};
    for (const std::uint64_t gb : {0ull, 1ull, 64ull})
        for (const char* mode : {"shutdown", "always", "off", "OFF", ""})
            for (const auto& c : dirs)
                for (const auto& e : dirs)
                    for (const char* lad : {static_cast<const char*>(nullptr), "C:\\Users\\u\\AppData\\Local", ""}) {
                        int er0 = 0, dc0 = 0, er1 = 0, dc1 = 0;
                        const auto o = old_ssdDir(gb, mode, c, e, lad, er0, dc0);
                        const auto d = whirl::server::kvSsdDir(gb, std::string(mode), c, [&] { ++er1; return e; }, lad,
                                                               [&](const char* b) { ++dc1; return std::string(b) + "\\whirl\\kvcache"; });
                        ISO_CHECK(d == o);
                        ISO_CHECK(er1 == er0);
                        ISO_CHECK(dc1 == dc0);
                        ++n;
                    }
    for (const bool tp : {false, true})
        for (const std::uint64_t pr : {0ull, 1ull, 4096ull, 73728ull})
            for (const std::uint64_t pt : {0ull, 1ull, 4096ull, 73728ull})
                for (const std::uint64_t ram : {0ull, 1ull, 20480ull}) {
                    const whirl::server::TierStart ts = whirl::server::tierStart(tp, pr, pt, ram);
                    ISO_CHECK(static_cast<int>(ts) == old_tierStart(tp, pr, pt, ram));
                    ISO_CHECK(whirl::server::tierLayoutWarn(pr, pt) == old_layoutWarn(pr, pt));
                    ++n;
                }
    ISO_CHECK(static_cast<int>(whirl::server::TierStart::on) == 0 && static_cast<int>(whirl::server::TierStart::layout_unknown) == 1 &&
              static_cast<int>(whirl::server::TierStart::off) == 2);
    std::printf("kv tier setup: %zu cases == c1ca7ce\n", n);
}

// ---- leg 10 part B: verbatim c1ca7ce src/server/server_main.cpp (CLI over env; env read only without the flag) ---------
// :1002 (makeEngine) --decode-min-tps / WHIRL_DECODE_MIN_TPS (default 20 inside the env reader)
std::uint32_t old_decodeMin(const std::optional<std::uint32_t>& cli, std::uint32_t env, int& reads) { return cli ? *cli : (++reads, env); }
// :1092 (reserveRamTier) and :1168 (loadVisionEarly) --mmproj / WHIRL_MMPROJ
std::optional<std::string> old_mmproj(const std::optional<std::string>& cli, const std::optional<std::string>& env, int& reads) {
    return cli ? cli : (++reads, env);
}
// :1529-1530 (startTier) --kv-ssd-delay (seconds) / WHIRL_KV_SSD_DELAY_MS (default 300000 inside the env reader)
std::uint64_t old_ssdDelay(const std::optional<std::uint64_t>& cli, std::uint32_t env, int& reads) {
    std::uint64_t ssd_delay_ms = 0;
    if (cli) ssd_delay_ms = *cli * 1000;
    else ssd_delay_ms = (++reads, env);
    return ssd_delay_ms;
}
// :1854 (serveMain) --log-file / the default log path
std::string old_logPath(const std::optional<std::string>& cli, const std::string& def, int& calls) { return cli ? *cli : (++calls, def); }
void testCliOverEnv() {
    std::size_t n = 0;
    ISO_CHECK(whirl::server::decode_min_tps_default == 20u);
    ISO_CHECK(whirl::server::kv_ssd_delay_ms_default == 300000u);
    for (const auto& c : std::vector<std::optional<std::uint32_t>>{std::nullopt, 0u, 1u, 20u, 0xffffffffu})
        for (const std::uint32_t e : {0u, 20u, 77u, 0xffffffffu}) {
            int r0 = 0, r1 = 0;
            const std::uint32_t o = old_decodeMin(c, e, r0);
            ISO_CHECK(whirl::server::decodeMinTps(c, [&] { ++r1; return e; }) == o);
            ISO_CHECK(r1 == r0);
            ++n;
        }
    const std::vector<std::optional<std::string>> ss = {std::nullopt, std::string(), std::string("a.gguf"), std::string("C:\\m\\mmproj.gguf")};
    for (const auto& c : ss)
        for (const auto& e : ss) {
            int r0 = 0, r1 = 0;
            const auto o = old_mmproj(c, e, r0);
            ISO_CHECK(whirl::server::mmprojPath(c, [&] { ++r1; return e; }) == o);
            ISO_CHECK(r1 == r0);
            ++n;
        }
    for (const auto& c : std::vector<std::optional<std::uint64_t>>{std::nullopt, 0ull, 1ull, 300ull, 18446744073709551ull, 0xffffffffffffffffull})
        for (const std::uint32_t e : {0u, 300000u, 0xffffffffu}) {
            int r0 = 0, r1 = 0;
            const std::uint64_t o = old_ssdDelay(c, e, r0);
            ISO_CHECK(whirl::server::kvSsdDelayMs(c, [&] { ++r1; return e; }) == o);
            ISO_CHECK(r1 == r0);
            ++n;
        }
    for (const auto& c : ss)
        for (const char* d : {"whirl-server.log", "C:\\Users\\u\\AppData\\Local\\whirl\\server.log"}) {
            int r0 = 0, r1 = 0;
            const std::string o = old_logPath(c, d, r0);
            ISO_CHECK(whirl::server::logPath(c, [&] { ++r1; return std::string(d); }) == o);
            ISO_CHECK(r1 == r0);
            ++n;
        }
    std::printf("cli over env: %zu cases == c1ca7ce\n", n);
}

int main(int argc, char** argv) {
    return iso::run("server.args", {{"defaults", testDefaults}, {"flags", testFlags}, {"resolve", testResolve}, {"served_model_name", testServedModelName}, {"kv_ssd_mode", testKvSsdMode}, {"kv_tier_setup", testKvTierSetup}, {"cli_over_env", testCliOverEnv}}, argc, argv);
}
