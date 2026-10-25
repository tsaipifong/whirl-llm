// Windows launcher (0.2.6, feat026-launcher): double-click whirl.exe -> first-run setup -> server -> tray.
// SPDX-License-Identifier: Apache-2.0
//
// Owner 2026-10-23 (internal_docs/PLAN_0.2.1.md "Windows launcher"): double-click whirl with no arguments in an
// interactive console -> first-run setup (GPU with pci:<bus>:<device>, recommended = most GPU memory; model file,
// recommended = the product MXFP4 file; mode, recommended balance; context length, recommended from the startup fit
// estimate with a margin, the maximum shown; server port), saved settings; then the server starts, the console is
// the log panel and minimises to the system tray (menu: open chat, show log, settings, quit). Started with
// arguments, by a script or without a console: never prompts, never goes to the tray (as before).
//
// This header is the decision logic only: no OS call, no HIP, no file IO, so tests/launcher drives every rule with
// injected facts. launcher_main.cpp does the IO through src/platform (shell.h, http_client.h).
// Prior art and the design reasons: progress/feat026-launcher.PRIOR_ART.md; docs/dev/launcher.md.
#pragma once

#include "whirl/gpu_select.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace whirl::launcher {

// ---------------------------------------------------------------------------
// 1. When does the launcher run?

// What the process knows about how it was started.
struct StartFacts {
    std::size_t n_args = 0;           // command-line arguments after the program name
    bool stdin_console = false;       // standard input is a console (not a pipe, file or NUL)
    bool stdout_console = false;      // standard output is a console (not redirected)
    unsigned console_processes = 0;   // processes attached to this console (GetConsoleProcessList; 0 = no console)
    bool no_launcher_env = false;     // WHIRL_NO_LAUNCHER is set to a non-empty value other than "0"
};

enum class Start {
    classic,   // today's behaviour: the usage text to stderr, exit code 2 (no arguments) / the command
    launcher,  // the first-run setup or the saved settings, then the server and the tray
};

// No arguments AND stdin and stdout are the console AND this process is the console's only process (it was created
// for us: Explorer double-click, Start menu, a shortcut; Raymond Chen 2016-01-25) AND WHIRL_NO_LAUNCHER is not set.
// Anything else (a command prompt or script: >= 2 processes; a pipe / redirect; no console at all; any argument)
// is classic: never a prompt, never the tray.
Start decideStart(const StartFacts& f);

// `whirl --setup` (redo the setup, then start): interactive only. It may be typed in a command prompt (the console
// then has two or more processes), but stdin and stdout must both be the console; otherwise an error with exit
// code 2 and no prompt (a script must never block on a question).
bool setupAllowed(const StartFacts& f);

// The value of WHIRL_NO_LAUNCHER (nullptr = unset) as StartFacts::no_launcher_env.
bool noLauncherEnv(const char* value);

// ---------------------------------------------------------------------------
// 2. Settings file (%LOCALAPPDATA%\whirl\launcher.ini, next to the tune cache and server.log)

inline constexpr int settings_version = 1;
inline constexpr std::uint16_t default_port = 8080;

struct Settings {
    std::string gpu;          // "pci:71:0" (the --device form that stays stable across reboots)
    std::string model;        // full path of the GGUF file (UTF-8)
    std::string mode = "balance";  // precise | balance | fast
    std::uint32_t ctx = 0;    // --ctx-per-slot; 0 = the server's default
    std::uint16_t port = default_port;
    std::string host = "127.0.0.1";  // not asked (only this computer); editable in the file
    bool operator==(const Settings&) const = default;
};

// "key=value" lines, UTF-8, '#' comments; unknown keys are ignored (a later version may add some).
std::string serializeSettings(const Settings& s);
// nullopt with *why set when the text is not a settings file of this version or a value is invalid.
std::optional<Settings> parseSettings(std::string_view text, std::string* why = nullptr);
// gpu and model set, mode valid, port non-zero.
bool settingsComplete(const Settings& s);
bool validMode(std::string_view m);

// ---------------------------------------------------------------------------
// 3. Questions and recommendations

// One line the user typed (nullopt: end of input, Ctrl+Z / Ctrl+C) against a numbered list of n options.
struct Answer {
    enum class Kind { recommended, index, text, invalid, eof } kind = Kind::invalid;
    std::size_t index = 0;  // kind == index: 0-based
    std::string text;       // kind == text: the trimmed line (a path or a number for free-form questions)
};
// Enter (empty / blank line) = recommended; "1".."n" = that option; with allow_text any other line is text (a
// path, a number), else invalid (ask again). Surrounding quotes of a pasted path are removed.
Answer parseAnswer(const std::optional<std::string>& line, std::size_t n_options, bool allow_text);

// GPUs as the setup lists them; recommended = gpusel::defaultGpu (most GPU memory among supported GPUs, the same
// rule as `whirl devices` [default] and the server).
struct GpuOption {
    std::string pci;     // "pci:71:0"
    std::string label;   // "AMD Radeon AI PRO R9700 (gfx1201), 31.9 GiB, pci:71:0"
    std::uint64_t fit_bytes = 0;  // memory a model can use (fit_mib, else total_mib)
    bool supported = false;
    bool recommended = false;
};
std::vector<GpuOption> gpuOptions(const std::vector<gpusel::GpuRecord>& gpus);
// Index of the recommended option (the first one if none is marked), -1 when the list is empty.
int recommendedIndex(const std::vector<GpuOption>& opts);

// Model files.
struct ModelFile {
    std::string path;
    std::uint64_t bytes = 0;
};
// A .gguf file that is a model to serve (not a vision encoder mmproj, not a DFlash drafter).
bool isModelCandidate(std::string_view filename);
// 0 = Swift-1.5 27B MXFP4-A (the recommended product file), 1 = another Swift-1.5 / Ornith-1.5 MXFP4 file,
// 2 = any other MXFP4 file, 3 = anything else. Case is ignored.
int productRank(std::string_view filename);
// Sorted for the list: files that fit in gpu_fit_bytes (bytes < fit, or fit unknown = 0) first, then productRank,
// then name; duplicates (same path, case-insensitive) removed. The first entry is the recommendation.
std::vector<ModelFile> rankModels(std::vector<ModelFile> files, std::uint64_t gpu_fit_bytes);
bool modelFits(std::uint64_t model_bytes, std::uint64_t gpu_fit_bytes);

// Modes in list order; recommended = balance (owner: 預設 balance).
const std::vector<std::string>& modeList();
inline constexpr const char* recommended_mode = "balance";

// Context length (the longest request: --ctx-per-slot).
inline constexpr std::uint32_t ctx_hard_max = 262144;   // the server's limit (server --help: up to 262144)
inline constexpr std::uint32_t ctx_default_cap = 131072;  // the server's default per request
inline constexpr std::uint32_t ctx_min = 4096;          // below this the server refuses to start
struct CtxInput {
    std::uint64_t mem_bytes = 0;     // the GPU's fit memory (VRAM / the dedicated carve-out)
    std::uint64_t need_fixed = 0;    // weights + load-time state + buffers (qwen35::estimateLoad, bufferBytes)
    std::uint64_t kv_per_token = 0;  // KV bytes per token of the mode's KV format (numerics::chooseKv)
};
struct CtxChoice {
    std::uint32_t max = 0;          // longest context that fits by the estimate (multiple of 4096, <= ctx_hard_max)
    std::uint32_t recommended = 0;  // with a margin: 80% of max, <= ctx_default_cap, multiple of 4096
    bool fits = false;              // max >= ctx_min
    bool known = false;             // the inputs were known (else recommended = 0 = the server's default)
};
// Usable memory = mem - margin (10% of mem, at least 1.5 GiB: the desktop, other programs, WDDM) - need_fixed;
// max = usable / kv_per_token. An estimate: the server's startup fit (startup_fit.h) still lowers a value that
// does not fit when it starts.
CtxChoice recommendContext(const CtxInput& in);
// "32768", "32k", "32K", "128k" -> tokens (k = 1024); nullopt for anything else or 0.
std::optional<std::uint32_t> parseCtx(std::string_view s);
// The context to use for a typed value: clamped to [ctx_min, ctx_hard_max], rounded down to a multiple of 1024.
std::uint32_t clampCtx(std::uint32_t v);

// The first free port of 8080..8099 (is_free injected), else 8080 (the server then reports the port in use).
std::uint16_t recommendPort(const std::function<bool(std::uint16_t)>& is_free, std::uint16_t first = default_port);
// "8080" -> 8080; nullopt for 0, > 65535 or not a number.
std::optional<std::uint16_t> parsePort(std::string_view s);

// ---------------------------------------------------------------------------
// 4. The server command

// Arguments after the program: {"serve", model, "--device", gpu, "--balance", "--ctx-per-slot", "N", "--port", "P"}
// (--ctx-per-slot only when ctx > 0; --host only when it is not 127.0.0.1).
std::vector<std::string> serverArgs(const Settings& s);
// One argument quoted for a Windows command line (CommandLineToArgvW / the MSVC CRT rules: quotes when it holds a
// space, tab or quote or is empty; backslashes before a quote doubled).
std::string quoteArg(std::string_view a);
// program + arguments, each quoted: the command line shown to the user ("equivalent command") and the one the
// child is started with.
std::string commandLine(std::string_view program, const std::vector<std::string>& args);

// ---------------------------------------------------------------------------
// 5. The tray menu and the chat client (pure parts)

enum class TrayItem : int { open_chat = 1, show_log = 2, settings = 3, quit = 4 };
// Menu labels in menu order (open chat, show log, settings, quit).
std::vector<std::pair<TrayItem, std::string>> trayMenu();

// The chat request of the console chat client (OpenAI /v1/chat/completions, streamed).
struct ChatMessage {
    std::string role;  // "system" | "user" | "assistant"
    std::string content;
};
std::string chatRequestBody(const std::vector<ChatMessage>& msgs, int max_tokens);
// Feeds bytes of a text/event-stream response; returns the content deltas of complete events ("data: {json}"
// lines; choices[0].delta.content), keeps an incomplete line in *pending, sets *done at "data: [DONE]".
std::vector<std::string> sseDeltas(std::string_view bytes, std::string* pending, bool* done);

}  // namespace whirl::launcher
