// whirl-server's command line (arch025-C1a): the options struct, the argument loop and the numerics request, moved
// out of serveMain so the server tests can run them (an argument table test pins every flag, the defaults of v0.2.4
// and the --help body). No environment reads and no HIP here: serveMain reads WHIRL_MODE and passes the value in,
// and passes the vision mode check (vision::parseMode lives in the HIP vision module). A bad command line throws
// ArgError; serveMain prints it as before (die: "<program>: error: <msg>", exit code 2).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "whirl/numerics.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::server {

// The option list of `whirl-server --help` (between the usage line and the environment variables).
extern const char* const kHelpBody;

struct Options {
    std::string path;
    std::string host = "127.0.0.1";
    std::uint16_t port = 8080;
    std::optional<std::uint32_t> ctx;
    std::uint32_t parallel = 4;
    bool parallel_given = false;  // --parallel / -np on the command line
    std::optional<std::uint32_t> ctx_per_slot;
    std::optional<std::uint32_t> mtp_drafts;
    std::string draft = "mtp";
    std::string draft_model;  // --draft-model FILE (wins over WHIRL_DRAFT_MODEL)
    std::optional<std::uint32_t> decode_min_tps;
    std::optional<std::string> log_file;
    std::optional<std::string> alias;
    std::optional<std::uint64_t> kv_ram_mb;
    std::optional<std::string> kv_ssd_dir;
    std::optional<std::uint64_t> kv_ssd_gb;
    std::optional<std::string> kv_ssd_mode;
    std::optional<std::uint64_t> kv_ssd_delay;  // seconds (--kv-ssd-delay)
    std::string device;
    // vision: encoder, idle release seconds, weight mode, embedding cache MiB
    std::optional<std::string> mmproj;
    std::uint32_t vis_idle_s = 60;
    std::string vis_mode = "auto";
    std::uint64_t vis_cache_mb = 1024;
    bool allow_local_images = false;
    std::vector<std::string> cors_origins;  // --cors-origin (repeatable)
    std::vector<std::string> allowed_hosts;  // --allow-host (repeatable)
    std::optional<std::string> mode_spec;   // --precise / --balance[=items] / --fast[=items] / --mode M
};

// The command line resolved: the options and the numerics request (mode flags, else WHIRL_MODE, else the default).
// The KV format is chosen at load from this request and the model type (numerics::chooseKv inside Model::load).
struct ResolvedConfig {
    Options opt;
    numerics::Request nreq;
};

// A bad command line; what() is the message die() prints after "<program>: error: ".
class ArgError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// vision::parseMode(s).has_value() in the server; the tests pass their own check.
using VisModeOk = bool (*)(std::string_view);

// The argument loop over args (argv[1..], --dump-only already taken out) and the missing-model check, in the order
// serveMain ran them. Throws ArgError at the first bad argument.
Options parseServerArgs(const std::vector<std::string>& args, VisModeOk vis_mode_ok);

// numerics::resolve(opt.mode_spec, mode_env) and the --parallel range check (1..16), in that order. mode_env is the
// value of WHIRL_MODE. Throws ArgError.
ResolvedConfig resolveServerConfig(Options opt, std::optional<std::string> mode_env);

// bug-emptyrootpaths: WHIRL_KV_SSD_DIR = --kv-ssd-dir (docs/usage.md), so a set-but-empty value is refused the same
// way when no --kv-ssd-dir was given (--kv-ssd-dir wins, as at the tier setup). kv_ssd_dir_env is the value of
// WHIRL_KV_SSD_DIR (nullopt = unset). Throws ArgError; serveMain runs it after resolveServerConfig, before the model
// file check (arch025-gaptests G18: moved out of serveMain unchanged so server-tests can pin it).
void checkKvSsdDirEnv(const Options& opt, const std::optional<std::string>& kv_ssd_dir_env);

// The model name of /v1/models, /props and the startup lines (arch025-lane1 leg 8: was serveMain's, 5728296
// server_main.cpp:953-958): --alias as given, else the GGUF file name without a final ".gguf" (any case).
std::string servedModelName(const std::string& file_name, const std::optional<std::string>& alias);
// The SSD tier mode (arch025-lane1 leg 8: was ServerRun::startTier's, 5728296 server_main.cpp:1543-1544): --kv-ssd-mode,
// else WHIRL_KV_SSD_MODE, else "shutdown"; any text but "always" / "off" is "shutdown".
std::string kvSsdMode(const std::optional<std::string>& cli, const std::optional<std::string>& env);

// ---- the host tier setup's rules (arch025-lane1 leg 10: were ServerRun::startTier's, c1ca7ce server_main.cpp:1512-1524,
// :1517, :1576-1580; pinned in tests/iso/server.args). env / dir_of are callables, called only where the old code read
// the environment / the default directory (the env read order is unchanged).
inline constexpr std::uint64_t kv_ssd_gb_default = 64;
// The SSD tier cap in GiB: --kv-ssd-gb, else env() (WHIRL_KV_SSD_GB; read only without the flag), else 64.
template <class Env>
std::uint64_t kvSsdGb(const std::optional<std::uint64_t>& cli, Env&& env) {
    if (cli) return *cli;
    if (const std::optional<std::uint64_t> g = env()) return *g;
    return kv_ssd_gb_default;
}
// The SSD tier directory: none with a 0 GiB cap or mode "off"; else --kv-ssd-dir, else env() (WHIRL_KV_SSD_DIR), else
// dir_of(base) (base: %LOCALAPPDATA%, nullptr = no default directory).
template <class Env, class DirOf>
std::optional<std::string> kvSsdDir(std::uint64_t ssd_gb, std::string_view mode, const std::optional<std::string>& cli, Env&& env,
                                    const char* base, DirOf&& dir_of) {
    std::optional<std::string> ssd_dir;
    if (ssd_gb > 0 && mode != "off") {
        if (cli) ssd_dir = cli;
        else if (auto v = env()) ssd_dir = *v;
        else if (base) ssd_dir = dir_of(base);
    }
    return ssd_dir;
}
// Whether the host tiers start: on when they were set up (tier_made) and the backend's KV rows per token (page_row) are
// the bytes per token the tier layout assumes; layout_unknown (a warning, tierLayoutWarn) when a RAM tier was asked for
// and they differ; else off.
enum class TierStart { on, layout_unknown, off };
TierStart tierStart(bool tier_made, std::uint64_t page_row, std::uint64_t per_tok, std::uint64_t kv_ram_mb);
std::string tierLayoutWarn(std::uint64_t page_row, std::uint64_t per_tok);

// ---- command line over environment (arch025-lane1 leg 10 part B: were inline in server_main.cpp, c1ca7ce :1002, :1092,
// :1168, :1529-1530, :1854; pinned in tests/iso/server.args). The flag wins; env() / def() is called only without it.
inline constexpr std::uint32_t decode_min_tps_default = 20;
inline constexpr std::uint32_t kv_ssd_delay_ms_default = 300000;
// --decode-min-tps, else env() (WHIRL_DECODE_MIN_TPS, default decode_min_tps_default).
template <class Env>
std::uint32_t decodeMinTps(const std::optional<std::uint32_t>& cli, Env&& env) {
    return cli ? *cli : env();
}
// --mmproj, else env() (WHIRL_MMPROJ).
template <class Env>
std::optional<std::string> mmprojPath(const std::optional<std::string>& cli, Env&& env) {
    return cli ? cli : env();
}
// --kv-ssd-delay (seconds) in ms, else env() (WHIRL_KV_SSD_DELAY_MS, default kv_ssd_delay_ms_default).
template <class Env>
std::uint64_t kvSsdDelayMs(const std::optional<std::uint64_t>& cli, Env&& env) {
    if (cli) return *cli * 1000;
    return env();
}
// --log-file, else def() (the default log path).
template <class Def>
std::string logPath(const std::optional<std::string>& cli, Def&& def) {
    return cli ? *cli : def();
}

}  // namespace whirl::server
