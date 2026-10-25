// Windows launcher: the interactive flow (setup questions, settings file, the server child, the tray, the chat
// window). The decisions are in launcher_logic.cpp; every OS call goes through src/platform (shell.h,
// http_client.h). See launcher.h and docs/dev/launcher.md.
// SPDX-License-Identifier: Apache-2.0

#include "launcher/launcher_main.h"

#include "launcher/launcher.h"
#include "platform/http_client.h"
#include "platform/shell.h"

#include "whirl/gguf.h"
#include "whirl/gpu_select.h"
#include "whirl/model.h"
#include "whirl/numerics.h"
#if WHIRL_HAVE_RELEASE
#include "release/release.h"
#endif

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <system_error>

namespace whirl::launcher {

namespace fs = std::filesystem;
namespace sh = whirl::platform::shell;

namespace {

constexpr const char* k_hf_swift = "https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF";

void say(const std::string& s) {
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fflush(stdout);
}

fs::path u8path(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }
std::string utf8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

std::string gibText(std::uint64_t bytes) { return std::format("{:.1f} GiB", static_cast<double>(bytes) / (1024.0 * 1024 * 1024)); }

std::string settingsPath() {
    const std::string lad = sh::localAppData();
    if (lad.empty()) return {};
    return utf8(u8path(lad) / "whirl" / "launcher.ini");
}

std::optional<Settings> loadSettings(const std::string& path, std::string* why) {
    std::ifstream f(u8path(path), std::ios::binary);
    if (!f) {
        *why = "no settings file yet";
        return std::nullopt;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return parseSettings(ss.str(), why);
}

bool saveSettings(const std::string& path, const Settings& s, std::string* why) {
    std::error_code ec;
    fs::create_directories(u8path(path).parent_path(), ec);
    const fs::path tmp = u8path(path + ".tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            *why = "cannot write " + utf8(tmp);
            return false;
        }
        const std::string t = serializeSettings(s);
        f.write(t.data(), static_cast<std::streamsize>(t.size()));
        if (!f) {
            *why = "cannot write " + utf8(tmp);
            return false;
        }
    }
    fs::rename(tmp, u8path(path), ec);  // replaces the old file in one step
    if (ec) {
        *why = "cannot replace " + path + ": " + ec.message();
        return false;
    }
    return true;
}

// One question: prints the prompt, reads a line; EOF -> nullopt (the caller stops without saving).
std::optional<Answer> ask(const std::string& prompt, std::size_t n_options, bool allow_text) {
    for (;;) {
        say(prompt);
        const Answer a = parseAnswer(sh::readConsoleLine(), n_options, allow_text);
        if (a.kind == Answer::Kind::eof) return std::nullopt;
        if (a.kind != Answer::Kind::invalid) return a;
        say(std::format("  Please type a number from 1 to {}, or press Enter for the recommended choice.\n", n_options));
    }
}

void waitEnter(const char* what) {
    say(what);
    (void)sh::readConsoleLine();
}

// ---- GPU

std::vector<gpusel::GpuRecord> probeGpus(std::string* why) {
    const std::wstring exe = sh::exePathWide();
    if (exe.empty()) {
        *why = "the program path is unknown";
        return {};
    }
    // the same helper child the server uses (gpu_select.h): this process never initialises the GPU runtime, so it
    // holds no GPU memory next to the server it starts (docs/guide/en/windows-hip.md 4.4)
    const std::wstring cmd = L"\"" + exe + L"\" " + std::wstring(gpusel::k_probe_arg, gpusel::k_probe_arg + std::strlen(gpusel::k_probe_arg));
    const gpusel::ChildRun r = gpusel::runChild(exe, cmd, gpusel::k_probe_timeout_ms);
    if (!r.why.empty()) *why = r.why;
    return gpusel::parseRecords(r.out);
}

// ---- model files

std::vector<ModelFile> findModels(const Settings& saved) {
    std::vector<fs::path> dirs;
    const fs::path exe_dir = u8path(sh::exePath()).parent_path();
    dirs.push_back(exe_dir);
    dirs.push_back(exe_dir / "models");
    if (!saved.model.empty()) dirs.push_back(u8path(saved.model).parent_path());
    dirs.push_back(u8path("C:\\models"));
    if (const std::string d = sh::downloadsFolder(); !d.empty()) dirs.push_back(u8path(d));
    if (const std::string up = sh::envUtf8("USERPROFILE"); !up.empty()) dirs.push_back(u8path(up) / "models");
    std::vector<ModelFile> out;
    for (const fs::path& d : dirs) {
        std::error_code ec;
        if (!fs::is_directory(d, ec)) continue;
        for (fs::directory_iterator it(d, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code e2;
            if (!it->is_regular_file(e2)) continue;
            const std::string p = utf8(it->path());
            if (!isModelCandidate(p)) continue;
            out.push_back({p, static_cast<std::uint64_t>(it->file_size(e2))});
        }
    }
    return out;
}

// ---- context estimate (the server's own pre-load functions, qwen35::estimateLoad / numerics::chooseKv)

CtxChoice estimateContext(const std::string& model, const std::string& mode, const GpuOption& gpu, const std::string& arch,
                          std::string* note) {
    CtxInput in;
    in.mem_bytes = gpu.fit_bytes;
    try {
        const gguf::File f = gguf::File::open(model);
        const qwen35::Config cfg = qwen35::Config::fromGguf(f);
        const qwen35::LoadEstimate est = qwen35::estimateLoad(f, cfg, false);
        const numerics::Request req = numerics::forModel(numerics::parseMode(mode, "launcher"), cfg.moe);
        const numerics::KvCaps caps{true, true, true, arch == "gfx1151"};
        const numerics::KvKind kv = numerics::chooseKv(req, cfg.moe, caps, std::nullopt).kv;
        in.kv_per_token = kv == numerics::KvKind::f16   ? est.kv_f16
                          : kv == numerics::KvKind::q8v ? est.kv_q8v
                          : kv == numerics::KvKind::q4  ? est.kv_q4
                                                        : est.kv_q8h;
        const std::uint64_t bufs = qwen35::bufferBytes(cfg, 2048, est.ffs, est.max_elems, qwen35::resolveFdSplits(kv == numerics::KvKind::q4));
        // weights + load-time state + prefill buffers + the MTP draft head + 256 MiB slack, + 4% (as server_main's
        // estimate), + the recurrent state of up to 4 request slots
        std::uint64_t fixed = est.weights + est.state + bufs + (est.has_mtp ? 65536ull * cfg.n_embd * 5 / 16 : 0) + (256ull << 20);
        fixed += fixed / 25 + 4 * est.state;
        in.need_fixed = fixed;
        *note = std::format("estimate: {} for the model and its buffers, {} KV cache, {:.0f} KiB per token", gibText(fixed),
                            numerics::kvKindName(kv), static_cast<double>(in.kv_per_token) / 1024.0);
    } catch (const std::exception& e) {
        *note = std::string("no estimate (") + e.what() + ")";
        return {};
    }
    return recommendContext(in);
}

bool portFree(std::uint16_t p, const std::string& host) {
#if WHIRL_HAVE_RELEASE
    try {
        app::requirePortFree(host, p);
        return true;
    } catch (const std::exception&) {
        return false;
    }
#else
    (void)p;
    (void)host;
    return true;
#endif
}

// ---- the setup questions; nullopt = cancelled (end of input) or impossible (no GPU)

struct SetupResult {
    std::optional<Settings> settings;
    int exit_code = 0;
};

SetupResult runSetup(const Settings& saved) {
    SetupResult res;
    Settings s = saved;
    say("\nWHIRL first-start setup. Press Enter to take the recommended choice [in brackets].\n\n");

    // 1. GPU
    say("Looking for AMD GPUs...\n");
    std::string why;
    const std::vector<gpusel::GpuRecord> recs = probeGpus(&why);
    const std::vector<GpuOption> gpus = gpuOptions(recs);
    if (gpus.empty()) {
        say("No AMD GPU found" + (why.empty() ? std::string() : " (" + why + ")") +
            ".\nWHIRL needs an AMD Radeon AI PRO R9700 (gfx1201) or a Radeon 8060S (gfx1151) and AMD Software: Adrenalin Edition 26.8.1 or newer.\n");
        res.exit_code = 3;
        return res;
    }
    bool any_supported = false;
    for (const GpuOption& g : gpus) any_supported = any_supported || g.supported;
    if (!any_supported) {
        say("This build of WHIRL has no GPU kernels for the GPU(s) found:\n");
        for (const GpuOption& g : gpus) say("  " + g.label + "\n");
        res.exit_code = 3;
        return res;
    }
    const int rec_gpu = recommendedIndex(gpus);
    say("GPU:\n");
    for (std::size_t i = 0; i < gpus.size(); ++i)
        say(std::format("  {}. {}{}\n", i + 1, gpus[i].label, static_cast<int>(i) == rec_gpu ? "   (recommended: most GPU memory)" : ""));
    const GpuOption* gpu = nullptr;
    std::string gpu_arch;
    for (;;) {
        const auto a = ask(std::format("GPU [{}]: ", rec_gpu + 1), gpus.size(), false);
        if (!a) return res;
        const std::size_t i = a->kind == Answer::Kind::index ? a->index : static_cast<std::size_t>(rec_gpu);
        if (!gpus[i].supported) {
            say("  This build has no GPU kernels for that GPU; choose another one.\n");
            continue;
        }
        gpu = &gpus[i];
        gpu_arch = recs[i].arch;
        break;
    }
    s.gpu = gpu->pci;

    // 2. model
    std::vector<ModelFile> models = rankModels(findModels(saved), gpu->fit_bytes);
    if (models.size() > 9) models.resize(9);
    say("\nModel file:\n");
    if (models.empty())
        say(std::format("  No model file found next to whirl.exe, in its models folder, C:\\models or Downloads.\n"
                        "  Download Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf from {}\n"
                        "  and type its full path below (or drag the file into this window).\n",
                        k_hf_swift));
    for (std::size_t i = 0; i < models.size(); ++i) {
        const bool fits = modelFits(models[i].bytes, gpu->fit_bytes);
        say(std::format("  {}. {}  ({}){}{}\n", i + 1, models[i].path, gibText(models[i].bytes), i == 0 && fits ? "   (recommended)" : "",
                        fits ? "" : "   (larger than this GPU's memory)"));
    }
    for (;;) {
        const std::string prompt = models.empty() ? std::string("Model file path: ")
                                                  : std::format("Model [1] (or type a path): ");
        const auto a = ask(prompt, models.size(), true);
        if (!a) return res;
        std::string path;
        if (a->kind == Answer::Kind::recommended && !models.empty()) path = models[0].path;
        else if (a->kind == Answer::Kind::index) path = models[a->index].path;
        else if (a->kind == Answer::Kind::text) path = a->text;
        else continue;  // Enter with no file found
        std::error_code ec;
        if (!fs::is_regular_file(u8path(path), ec)) {
            say("  No such file: " + path + "\n");
            continue;
        }
        if (!isModelCandidate(path)) {
            say("  That is not a model .gguf file (a vision encoder mmproj or a DFlash drafter is not served on its own).\n");
            continue;
        }
        s.model = utf8(fs::absolute(u8path(path), ec));
        break;
    }

    // 3. mode (before the context: the KV format, so the memory per token, depends on it)
    say("\nMode:\n"
        "  1. balance   recommended: fast, output within the published accuracy limits\n"
        "  2. precise   bit-exact reference arithmetic, slower, more memory\n"
        "  3. fast      fastest, 4-bit KV cache, small accuracy loss\n");
    {
        const auto a = ask("Mode [1]: ", modeList().size(), false);
        if (!a) return res;
        s.mode = a->kind == Answer::Kind::index ? modeList()[a->index] : std::string(recommended_mode);
    }

    // 4. context length
    std::string note;
    const CtxChoice cc = estimateContext(s.model, s.mode, *gpu, gpu_arch, &note);
    say("\nContext length (the longest single request, in tokens; 32k = 32768):\n  " + note + "\n");
    if (cc.known && !cc.fits)
        say(std::format("  The model hardly fits in this GPU's {}: the server will start with the smallest context it can, or say "
                        "what to change.\n",
                        gibText(gpu->fit_bytes)));
    for (;;) {
        const std::string rec_txt = cc.known && cc.fits ? std::format("{}", cc.recommended) : std::string("server default");
        const std::string max_txt = cc.known && cc.fits ? std::format("; maximum about {} by the estimate", cc.max) : std::string();
        const auto a = ask(std::format("Context [{}]{}: ", rec_txt, max_txt), 0, true);
        if (!a) return res;
        if (a->kind == Answer::Kind::recommended) {
            s.ctx = cc.known && cc.fits ? cc.recommended : 0;
            break;
        }
        const auto v = parseCtx(a->text);
        if (!v) {
            say("  Type a number of tokens such as 65536 or 64k.\n");
            continue;
        }
        s.ctx = clampCtx(*v);
        if (cc.known && cc.fits && s.ctx > cc.max)
            say(std::format("  Note: {} is above the estimate ({}); the server lowers it at start if it does not fit.\n", s.ctx, cc.max));
        break;
    }

    // 5. port
    const std::uint16_t rec_port = recommendPort([&](std::uint16_t p) { return portFree(p, s.host); });
    say("\nServer port (http://127.0.0.1:PORT/v1, for chat programs and editors):\n");
    for (;;) {
        const auto a = ask(std::format("Port [{}]: ", rec_port), 0, true);
        if (!a) return res;
        if (a->kind == Answer::Kind::recommended) {
            s.port = rec_port;
            break;
        }
        const auto p = parsePort(a->text);
        if (!p) {
            say("  Type a port number from 1 to 65535.\n");
            continue;
        }
        s.port = *p;
        break;
    }
    res.settings = s;
    return res;
}

// ---- tray phase

struct Run {
    std::uint32_t pid = 0;
    std::atomic<bool> quitting{false};
    std::chrono::steady_clock::time_point quit_at{};
    bool hidden_once = false;
    bool hide_failed = false;
    int exit_code = 0;
    bool exited = false;
};

}  // namespace

int launcherMain(bool force_setup) {
    sh::setConsoleTitle("WHIRL");
    const std::string ini = settingsPath();
    std::string why;
    std::optional<Settings> saved = ini.empty() ? std::nullopt : loadSettings(ini, &why);
    std::error_code ec;
    const bool usable = saved && settingsComplete(*saved) && fs::is_regular_file(u8path(saved->model), ec);
    Settings s;
    if (usable && !force_setup) {
        s = *saved;
        say("WHIRL: starting with the saved settings (" + ini + "; run `whirl --setup` to change them).\n");
    } else {
        if (saved && !usable) say("WHIRL: the saved settings are incomplete or the model file is gone; asking again.\n");
        const SetupResult r = runSetup(saved.value_or(Settings{}));
        if (!r.settings) {
            if (r.exit_code != 0) {
                waitEnter("\nPress Enter to close.");
                return r.exit_code;
            }
            say("\nSetup cancelled; nothing was saved.\n");
            return 2;
        }
        s = *r.settings;
        if (ini.empty()) {
            say("\n(LOCALAPPDATA is not set: the settings are not saved.)\n");
        } else if (!saveSettings(ini, s, &why)) {
            say("\nCould not save the settings: " + why + "\n");
        } else {
            say("\nSaved to " + ini + " (`whirl --setup` asks again).\n");
        }
    }

    const std::string exe = sh::exePath();
    const std::vector<std::string> args = serverArgs(s);
    const std::string cl = commandLine(exe, args);
    say("\nThe same server from a command prompt or a script:\n  " + cl + "\n\n");
    say(std::format("Starting the WHIRL server. When it is ready, chat programs connect to http://{}:{}/v1 .\n"
                    "Minimise this window to keep WHIRL running in the system tray (right-click the WHIRL icon: open chat, "
                    "show log, settings, quit).\n\n",
                    s.host, s.port));

    Run run;
    run.pid = sh::startChild(cl, &why);
    if (run.pid == 0) {
        say("Could not start the server: " + why + "\n");
        waitEnter("Press Enter to close.");
        return 1;
    }
    sh::setConsoleTitle(std::format("WHIRL server - {}:{} (minimise to the tray)", s.host, s.port));
    // Ctrl+C in this window goes to the server (it shuts down cleanly); closing the window: wait a moment for it
    sh::ignoreCtrlC([&run] { sh::waitChild(run.pid, 4500, nullptr); });

    sh::TrayConfig tc;
    tc.tooltip = std::format("WHIRL server {}:{}", s.host, s.port);
    for (const auto& [id, label] : trayMenu()) tc.menu.emplace_back(static_cast<int>(id), label);
    tc.double_click_id = static_cast<int>(TrayItem::show_log);
    tc.on_item = [&](int id) {
        switch (static_cast<TrayItem>(id)) {
        case TrayItem::open_chat: {
            std::string w;
            const std::string chat = commandLine(exe, {"--whirl-launcher-chat", std::format("{}:{}", s.host == "0.0.0.0" ? "127.0.0.1" : s.host, s.port)});
            if (!sh::startInNewConsole(chat, &w)) sh::messageBox("WHIRL", "Could not open the chat window: " + w);
            break;
        }
        case TrayItem::show_log:
            sh::showConsole();
            break;
        case TrayItem::settings:
            sh::messageBox("WHIRL settings",
                           "The settings file opens in your editor:\n" + ini +
                               "\n\nChanges apply at the next start: quit WHIRL and start it again. To answer the setup questions again, "
                               "run  whirl --setup  (or delete the file).");
            if (!ini.empty()) sh::openWithDefaultApp(ini);
            break;
        case TrayItem::quit:
            if (!run.quitting.exchange(true)) {
                run.quit_at = std::chrono::steady_clock::now();
                sh::showConsole();
                say("\nWHIRL: stopping the server (finishing running requests, saving the prompt cache)...\n");
                if (!sh::sendCtrlBreak(run.pid)) sh::killChild(run.pid);
            }
            break;
        }
    };
    tc.on_tick = [&]() {
        int code = 0;
        if (sh::waitChild(run.pid, 0, &code)) {
            run.exited = true;
            run.exit_code = code;
            return false;
        }
        if (run.quitting && std::chrono::steady_clock::now() - run.quit_at > std::chrono::seconds(180)) sh::killChild(run.pid);
        if (!run.hide_failed && sh::consoleMinimized()) {
            if (sh::hideConsole()) {
                if (!run.hidden_once) sh::trayBalloon("WHIRL is still running", "Right-click the WHIRL icon to open the chat, the log, or quit.");
                run.hidden_once = true;
            } else {
                run.hide_failed = true;  // a terminal that ignores hide: the window stays on the taskbar
            }
        }
        return true;
    };
    if (!sh::runTray(tc, &why)) {
        say("(No system tray: " + why + ". Close this window or press Ctrl+C to stop the server.)\n");
        while (!sh::waitChild(run.pid, 1000, &run.exit_code)) {
        }
        run.exited = true;
    }
    if (!run.exited) run.exited = sh::waitChild(run.pid, 0xFFFFFFFFu, &run.exit_code);
    if (run.quitting) return 0;
    sh::showConsole();
    say(std::format("\nThe WHIRL server stopped (exit code {}). The messages above say why.\n", run.exit_code));
    waitEnter("Press Enter to close this window.");
    return run.exit_code;
}

int chatMain(const std::string& host_port) {
    const std::size_t colon = host_port.rfind(':');
    const std::string host = colon == std::string::npos ? std::string("127.0.0.1") : host_port.substr(0, colon);
    const std::optional<std::uint16_t> port = parsePort(colon == std::string::npos ? host_port : host_port.substr(colon + 1));
    if (!port) {
        say("whirl: --whirl-launcher-chat HOST:PORT\n");
        return 2;
    }
    sh::setConsoleTitle(std::format("WHIRL chat - {}:{}", host, *port));
    say(std::format("WHIRL chat with the server at http://{}:{}  (/new = new conversation, /exit or Ctrl+Z = close)\n", host, *port));
    {
        const platform::http_client::Result h = platform::http_client::get(host, *port, "/health", 5000);
        if (!h.ok || h.status != 200)
            say("The server is not ready yet (it may still be loading the model); your first message waits for it.\n");
    }
    std::vector<ChatMessage> history;
    for (;;) {
        say("\nyou> ");
        const std::optional<std::string> line = sh::readConsoleLine();
        if (!line || *line == "/exit" || *line == "/quit") return 0;
        if (*line == "/new") {
            history.clear();
            say("(new conversation)\n");
            continue;
        }
        if (line->empty()) continue;
        history.push_back({"user", *line});
        say("\nwhirl> ");
        std::string pending, answer;
        bool done = false;
        const platform::http_client::Result r = platform::http_client::postJson(
            host, *port, "/v1/chat/completions", chatRequestBody(history, 0), 600000, [&](const std::string& chunk) {
                for (const std::string& d : sseDeltas(chunk, &pending, &done)) {
                    answer += d;
                    say(d);
                }
                return !done;
            });
        say("\n");
        if (!r.ok || r.status != 200) {
            say(std::format("(no answer: {})\n", r.ok ? std::format("HTTP {}", r.status) : r.why));
            history.pop_back();
            continue;
        }
        history.push_back({"assistant", answer});
    }
}

}  // namespace whirl::launcher
