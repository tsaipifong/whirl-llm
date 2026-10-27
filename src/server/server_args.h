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

}  // namespace whirl::server
