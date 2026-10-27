// whirl: GGUF inference CLI (qwen35 / qwen35moe on HIP).
// SPDX-License-Identifier: Apache-2.0
//
//   whirl chat MODEL.gguf "PROMPT" [options]     greedy generation (streams text)
//   whirl bench MODEL.gguf [options]             prefill sweep + decode plain / MTP / MTP+n-gram
//   whirl selftest MODEL.gguf                    bitwise kernel self-checks on the model's matrices
//   whirl devices                                list HIP devices
//
// Written for WHIRL from the documented behaviour of the research
// prototype's chat command (flags, environment variables, output summary);
// the decode loops (pipelined plain decode, MTP speculative decode with the
// n-gram co-drafter) follow the WHIRL Zig research prototype (gguf_cli.zig).

// arch025-C2: each command is in its own file (cmd_<name>.cpp; serve and vis-encode forward to the server / the vision
// encoder), the shared helpers in common.cpp / decode.cpp, the CLI's model building in build_model.cpp, their
// declarations in cli.h. This file: the help texts' assembly, the command table and wmain.
#include "cli/cli.h"

namespace cli {
namespace {

const char* k_help_main =
    "usage: whirl <command> [arguments]\n"
    "\n"
    "commands:\n"
    "  chat MODEL.gguf \"PROMPT\"   generate a reply to one prompt (the text streams as it is generated)\n"
    "  bench MODEL.gguf           measure prefill and decode speed\n"
    "  serve MODEL.gguf           OpenAI-compatible HTTP server (the same as whirl-server.exe)\n"
    "  devices                    list the AMD GPUs the driver reports\n"
    "  selftest MODEL.gguf        bitwise self-checks of the GPU kernels on the model's own weights\n"
    "  seqtest MODEL.gguf         check the multi-request (server) paths against single-request runs\n"
    "  golden MODEL.gguf          print per-step output hashes of fixed prompts (regression gate, developers)\n"
    "  dflash-check MODEL DRAFT DIR  DFlash2 drafter forward vs reference values (tools/dflash_ref)\n"
    "  vis-encode MMPROJ IMAGE    encode one image with a vision encoder (diagnostic)\n"
    "  help [COMMAND | env]       this text, the options of COMMAND, or every environment variable\n"
    "\n"
    "options:\n"
    "  -h, --help                 show help (also after a command: whirl chat --help)\n"
    "  -V, --version              show the version\n"
    "\n"
    "examples:\n"
    "  whirl chat Qwen3.8-27B-UD-Q4_K_M.gguf \"Explain TCP slow start in two sentences.\" --max-tokens 400\n"
    "  whirl serve Qwen3.8-27B-UD-Q4_K_M.gguf --port 8080\n"
    "\n"
    "Supported models: GGUF files of the qwen35 / qwen35moe architectures (see the README).\n"
    "The first run with a new model tunes the GPU kernels once (cached in %LOCALAPPDATA%\\whirl).\n";

// arch025-Q2 (include/whirl/leak_count.h): with WHIRL_LEAK_CHECK=1 a whole selftest / seqtest run is one unit: after it
// returns (the Model, its buffers, streams and events destroyed) no hip allocation, stream or event it made may be alive.
// A leak prints "LEAK_CHECK: <cmd>: ..." and turns the exit code into 1; a clean run prints "leak check: ok (1 units)".
// Off: nothing printed, the exit code is the command's (no HIP call added either way).
int leakChecked(const char* unit, int (*cmd)(const Args&), const Args& a) {
    const hip::leakcount::Snapshot before = hip::leakcount::snapshot();
    int rc = cmd(a);
    if (hip::leakcount::check(unit, before)) rc = 1;
    if (hip::leakcount::report() != 0) rc = 1;
    return rc;
}

// One row per command, in the order of the old `known` list (llama.cpp app/llama.cpp cmds[], git.c commands[]): the
// text of `whirl help NAME` / `whirl NAME --help` with its environment scope, and how the command runs. raw: on the
// unparsed arguments, right after the GPU preflight (serve, vis-encode parse their own options); run: on parseArgs'
// result. A command whose part is not in this build (no server / no vision) has neither: main help + usage error.
struct Command {
    const char* name;
    const char* help;  // nullptr: no text of its own (serve: printHelp prints the server's help)
    unsigned scope;    // app::envHelp scope printed under the help text
    int (*raw)(std::vector<std::string>& argv);
    int (*run)(const Args& a);
};

// nullptr: not a command
const Command* findCommand(const std::string& name) {
    static const Command t[] = {
        {"chat", k_help_chat, app::sc_chat, nullptr, cmdChat},
        {"bench", k_help_bench, app::sc_bench, nullptr, cmdBench},
        {"selftest", k_help_selftest, app::sc_selftest, nullptr, [](const Args& a) { return leakChecked("selftest", cmdSelftest, a); }},
        {"seqtest", k_help_seqtest, app::sc_seqtest, nullptr, [](const Args& a) { return leakChecked("seqtest", cmdSeqtest, a); }},
        {"golden", k_help_golden, app::sc_seqtest, nullptr, cmdGolden},
        {"dflash-check", k_help_dflash_check, app::sc_selftest, nullptr, cmdDflashCheck},
        {"devices", k_help_devices, 0, nullptr, [](const Args&) { return cmdDevices(); }},
#if WHIRL_HAVE_SERVER
        {"serve", nullptr, 0, cmdServe, nullptr},
#else
        {"serve", nullptr, 0, nullptr, nullptr},
#endif
#if WHIRL_HAVE_VISION
        {"vis-encode", k_help_vis, app::sc_vis, cmdVisEncode, nullptr},
#else
        {"vis-encode", k_help_vis, app::sc_vis, nullptr, nullptr},
#endif
    };
    for (const Command& c : t)
        if (name == c.name) return &c;
    return nullptr;
}

// help for one topic on stdout; false = unknown topic
bool printHelp(const std::string& topic) {
    if (topic.empty()) {
        out(std::string("whirl ") + app::version() + " - Windows HIP Inference for RDNA LLMs\n\n" + k_help_main + "\n" + app::exitCodeHelp());
        return true;
    }
    if (topic == "env") {
        out("Every environment variable WHIRL reads (whirl and whirl-server).\n" + app::envHelp(app::sc_all));
        return true;
    }
#if WHIRL_HAVE_SERVER
    if (topic == "serve") {
        out(whirl::server::serveHelp("whirl serve"));
        return true;
    }
#endif
    const Command* const c = findCommand(topic);
    if (c == nullptr || c->help == nullptr) return false;
    out(std::string(c->help) + app::envHelp(c->scope) + "\n" + app::exitCodeHelp());
    return true;
}

bool isHelpFlag(const std::string& x) { return x == "-h" || x == "--help" || x == "/?"; }

}  // namespace
}  // namespace cli

using namespace cli;  // wmain sees the commands as it saw the old anonymous namespace

int wmain(int argc, wchar_t** wargv) {
    whirl::platform::console::setStdoutBinary();  // arch025 PL1: _setmode(_fileno(stdout), _O_BINARY)
    app::initConsole();
    std::vector<std::string> argv = narrowArgs(argc, wargv);
    // s8-719 root fix: the GPU list helper (a child of this program, see gpu_select.h); never interactive
    if (argv.size() == 2 && argv[1] == whirl::gpusel::k_probe_arg) return whirl::gpusel::probeMain();
    if (argv.size() >= 2 && argv[1] == "--list-devices") argv[1] = "devices";  // llama.cpp's name for it
    if (argv.size() < 2) {
        std::fputs(k_help_main, stderr);
        return app::exit_usage;
    }
    const std::string cmd = argv[1];
    if (isHelpFlag(cmd) || cmd == "help") {
        const std::string topic = argv.size() > 2 ? argv[2] : std::string();
        if (printHelp(topic)) return app::exit_ok;
        std::fprintf(stderr, "whirl: no help for '%s' (commands: chat, bench, serve, devices, selftest, seqtest, vis-encode; or env)\n", topic.c_str());
        return app::exit_usage;
    }
    if (cmd == "-V" || cmd == "--version" || cmd == "version") {
        out(app::versionText("whirl"));
        return app::exit_ok;
    }
    const Command* const c = findCommand(cmd);
    if (c == nullptr) {
        std::fprintf(stderr, "whirl: unknown command '%s'\n\n%s", cmd.c_str(), k_help_main);
        return app::exit_usage;
    }
    // arch025-S0a: --dump-only = exit 0 right after load + tune (baseline dump matrix, whirl/dump.h); serve parses
    // it itself (serveMain), the other commands here
    bool dump_only = false;
    if (cmd != "serve")
        for (std::size_t i = 2; i < argv.size();) {
            if (argv[i] == "--dump-only") {
                dump_only = true;
                argv.erase(argv.begin() + static_cast<std::ptrdiff_t>(i));
            } else {
                ++i;
            }
        }
    bool wants_help = false;
    for (std::size_t i = 2; i < argv.size(); ++i) wants_help = wants_help || isHelpFlag(argv[i]);
    if (wants_help) {
        printHelp(cmd);
        return app::exit_ok;
    }
    // bug-cli-dumponly: a command that never reaches loadOrTune refuses --dump-only (before any GPU work) instead of
    // running to the end without the dump-only exit
    if (dump_only) {
        if (const std::string why = whirl::dump::dumpOnlyRefusal(cmd); !why.empty()) {
            std::fprintf(stderr, "whirl: error: %s\n  (run 'whirl %s --help' for the options)\n", why.c_str(), cmd.c_str());
            return app::exit_usage;
        }
        whirl::dump::setDumpOnly(true);
    }
    // the GPU runtime of the driver must be loadable before any HIP call
    if (const int pf = app::gpuPreflight(); pf != app::exit_ok) return pf;
    // serve / vis-encode: their own options (cmd_serve.cpp, cmd_vis.cpp)
    if (c->raw != nullptr) return c->raw(argv);
    try {
        const Args a = parseArgs(argv, 2);
        if (cmd != "devices" && !a.pos.empty()) app::requireModelFile(a.pos[0]);
        if (c->run != nullptr) return c->run(a);
        std::fputs(k_help_main, stderr);
        return app::exit_usage;
    } catch (const std::exception& e) {
        const int code = app::explain(e, "whirl");
        if (code == app::exit_usage) std::fprintf(stderr, "  (run 'whirl %s --help' for the options)\n", cmd.c_str());
        return code;
    }
}
