// `whirl-server MODEL.gguf [options]`: loads a qwen35 / qwen35moe GGUF on
// the GPU and serves the OpenAI-compatible API (continuous batching over
// --parallel slots, prefix cache, shared prefix checkpoints, host tiers).
// Reimplements the WHIRL Zig research prototype's server.zig (main, buildKvArrays,
// tierFingerprint, defaultBatchDrafts).
// SPDX-License-Identifier: Apache-2.0

#include "server_main.h"

#include "backend.h"
#include "engine.h"
#include "http.h"
#include "log.h"
#include "tier/device_ops.h"
#include "tier/kv_tier.h"
#include "tier/ram_size.h"
#include "whirl/common.h"
#include "whirl/gguf.h"
#include "whirl/hip.h"
#include "whirl/model.h"
#include "whirl/tokenizer.h"
#include "whirl/vram_limit.h"
#include "release/release.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")  // timeBeginPeriod (whirl-server and whirl serve)
#endif

namespace whirl::server {

namespace {

// Graceful shutdown: Ctrl+C / Ctrl+Break / closing the console window (and logoff /
// system shutdown) stop the engine, which answers queued requests 503, lets the
// running ones finish, flushes the prefix-cache tiers (pending RAM -> SSD writes,
// bounded by Engine::shutdown_flush_ms) and returns from runLoop; main then exits.
// The handler thread waits for that (the OS ends the process about 5 s after a
// close / logoff / shutdown event regardless). A second Ctrl+C terminates at once.
std::atomic<Engine*> g_engine{nullptr};
std::atomic<bool> g_stopping{false};
HANDLE g_stopped = nullptr;  // set when runLoop has returned

BOOL WINAPI onConsoleCtrl(DWORD type) {
    Engine* e = g_engine.load();
    if (e == nullptr || g_stopping.exchange(true)) return FALSE;  // default handling: exit now
    const char* what = type == CTRL_C_EVENT ? "Ctrl+C" : type == CTRL_BREAK_EVENT ? "Ctrl+Break" : type == CTRL_CLOSE_EVENT ? "console closed" : "logoff / shutdown";
    logI("shutdown requested ({}): no new requests; finishing running ones and flushing the prefix-cache tiers", what);
    e->stop();
    const bool interactive = type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT;
    const DWORD wait_ms = interactive ? static_cast<DWORD>(Engine::shutdown_flush_ms + 60000) : 4500;
    if (g_stopped != nullptr && WaitForSingleObject(g_stopped, wait_ms) == WAIT_OBJECT_0) {
        // main returns from serveMain and the process exits normally; for close /
        // logoff / shutdown events the OS ends the process when the handler returns
        if (!interactive) Sleep(500);
        return TRUE;
    }
    logW("shutdown: the engine did not stop within {} s; exiting", wait_ms / 1000);
    return FALSE;
}

// Windows timer resolution: 1 ms while serving. At the default tick (10-16 ms) every
// short wait of the main loop (the 1 ms polls of a burst gather and of host-tier
// restores, the restore tail's 200 us poll) sleeps a whole tick. Restored on exit.
struct TimerRes {
    bool on = timeBeginPeriod(1) == TIMERR_NOERROR;
    ~TimerRes() {
        if (on) timeEndPeriod(1);
    }
};

// WHIRL_TIMER_PROBE: mean actual duration of a short sleep (diagnostics)
double sleepMs(std::chrono::microseconds d, int n) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < n; ++k) std::this_thread::sleep_for(d);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
}

constexpr std::uint32_t ctx_per_slot_default = 131072;
constexpr std::uint32_t floor_second_ctx = 65536;
constexpr std::uint32_t serve_prefill_batch = 4096;
constexpr std::size_t n_ckpt_default = 4;
constexpr std::size_t n_ckpt_default_par = 2;
constexpr std::size_t n_spe_default = 2;
// KV pool (tokens, smallest format) below which startup lowers the prefill batch / checkpoints
constexpr std::uint32_t vram_tight_pool = 16384;
constexpr std::uint32_t sys_min_default = 2048;
constexpr std::uint64_t kv_ssd_gb_default = 64;
constexpr std::uint32_t decode_min_tps_default = 20;

const char* kHelpBody =
    "Loads a qwen35 / qwen35moe GGUF model on the GPU and serves the OpenAI-compatible API\n"
    "(GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions; plus\n"
    "GET /props and GET /version for clients that detect the context length) with\n"
    "continuous batching over --parallel request slots, a prefix cache and host RAM / SSD tiers.\n"
    "Stop it with Ctrl+C (running requests finish, cached sessions are written out).\n"
    "\n"
    "options:\n"
    "  --host ADDR              address to listen on (default 127.0.0.1 = this computer only;\n"
    "                           0.0.0.0 = every network: anyone who can reach this PC can use it)\n"
    "  --port N                 TCP port (default 8080)\n"
    "  --alias NAME             model id reported by /v1/models (default: the file name without .gguf)\n"
    "  --device D               GPU: r9700 (default), 8060s, a device index, or a name / gfx substring\n"
    "  -np, --parallel N        concurrent request slots (continuous batching, 1..16; default 4, 1 on cards < 20 GiB)\n"
    "  -c, --ctx N              shared KV pool in tokens; slots take pages on demand and idle slots'\n"
    "                           prefix caches are evicted (LRU) when it is full (default: all VRAM\n"
    "                           left over, kept 768 MiB (MoE 1.5 GiB) under both the free VRAM and the\n"
    "                           Windows (WDDM) budget; WHIRL_POOL_RESERVE_MB=N: free VRAM minus N only;\n"
    "                           262144 on the Radeon 8060S)\n"
    "  --ctx-per-slot N         longest context of one request (default min(pool, 131072); up to 262144)\n"
    "  --precise | --balance | --fast\n"
    "                           numerics mode (default precise: f16/f32 activations, f16 KV; balance:\n"
    "                           fp8 MXFP4 prefill, int8 KV when f16 does not fit, ...; fast: balance +\n"
    "                           future lossy items). --balance=fp8,kvq8 picks items; --mode M; WHIRL_MODE\n"
    "  --mtp-drafts N           MTP drafts per cycle, 1..10 (fixed count; default: per model type)\n"
    "  --decode-min-tps N       while other requests prefill, keep every streaming (decoding) request\n"
    "                           at >= N tok/s by limiting prefill forwards (default 20; 0 = off:\n"
    "                           prefill forwards are not limited)\n"
    "  --kv-ram-mb N            host RAM tier of the prefix cache in MiB of pinned memory (default: 1/4\n"
    "                           of physical RAM, at least 8 GiB or one full-length session, at most\n"
    "                           32 GiB, and at most half of the RAM available at startup; 0 = no host\n"
    "                           tiers; Radeon 8060S: default 0, its KV pool already is system memory).\n"
    "                           Idle sessions are copied there and restored instead of prefilled again\n"
    "  --kv-ssd-dir PATH        SSD tier directory (default %LOCALAPPDATA%\\whirl\\kvcache)\n"
    "  --kv-ssd-gb N            SSD tier size cap in GiB (default 64; 0 = no SSD tier)\n"
    "  --mmproj FILE            vision encoder (Qwen3-VL style mmproj GGUF, F16 / BF16): image_url parts\n"
    "                           (data: URLs with base64 PNG / JPEG / ...) become image tokens; weights stay\n"
    "                           in pinned host RAM, nothing in VRAM until an image arrives\n"
    "  --vis-idle-s N           release the vision encoder after N s without images (default 60)\n"
    "  --vis-mode M             auto (default), resident (weights in VRAM), stream (layer by layer)\n"
    "  --vis-cache-mb N         host cache of image embeddings by content hash, MiB (default 1024)\n"
    "  --allow-local-images     also accept local file paths / file:// URLs as image sources\n"
    "  --cors-origin ORIGIN     web pages from ORIGIN (e.g. https://app.example.com) may call the server\n"
    "                           from a browser (CORS); repeatable; * = any page (the behaviour before\n"
    "                           v0.1.4). Default: only pages on http(s)://localhost, 127.0.0.1, [::1]\n"
    "  --log-file PATH          log file (default %LOCALAPPDATA%\\whirl\\server.log; the log also goes to\n"
    "                           the console)\n"
    "  -h, --help               this text\n"
    "  -V, --version            show the version\n";

std::string g_program = "whirl-server";

struct Options {
    std::string path;
    std::string host = "127.0.0.1";
    std::uint16_t port = 8080;
    std::optional<std::uint32_t> ctx;
    std::uint32_t parallel = 4;
    bool parallel_given = false;  // --parallel / -np on the command line
    std::optional<std::uint32_t> ctx_per_slot;
    std::optional<std::uint32_t> mtp_drafts;
    std::optional<std::uint32_t> decode_min_tps;
    std::optional<std::string> log_file;
    std::optional<std::string> alias;
    std::optional<std::uint64_t> kv_ram_mb;
    std::optional<std::string> kv_ssd_dir;
    std::optional<std::uint64_t> kv_ssd_gb;
    std::string device;
    // vision: encoder, idle release seconds, weight mode, embedding cache MiB
    std::optional<std::string> mmproj;
    std::uint32_t vis_idle_s = 60;
    std::string vis_mode = "auto";
    std::uint64_t vis_cache_mb = 1024;
    bool allow_local_images = false;
    std::vector<std::string> cors_origins;  // --cors-origin (repeatable)
    std::optional<std::string> mode_spec;   // --precise / --balance[=items] / --fast[=items] / --mode M
};

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "%s: error: %s\n  (run '%s --help' for the options)\n", g_program.c_str(), msg.c_str(), g_program.c_str());
    std::exit(app::exit_usage);
}

template <class T>
T parseNum(const std::string& v, const char* what) {
    try {
        std::size_t used = 0;
        const unsigned long long x = std::stoull(v, &used, 10);
        if (used != v.size() || x > static_cast<unsigned long long>(std::numeric_limits<T>::max())) throw std::out_of_range("");
        return static_cast<T>(x);
    } catch (const std::exception&) {
        die(std::string("invalid ") + what + " " + v);
    }
}

// "--name V" or "--name=V"
std::optional<std::string> argValue(const std::vector<std::string>& args, std::size_t& i, std::string_view name) {
    const std::string& a = args[i];
    if (a == name) {
        if (i + 1 >= args.size()) return std::nullopt;
        ++i;
        return args[i];
    }
    if (a.size() > name.size() && a.compare(0, name.size(), name) == 0 && a[name.size()] == '=')
        return a.substr(name.size() + 1);
    return std::nullopt;
}

std::optional<std::string> env(std::string_view name) { return envGet(name); }

std::uint32_t envU32(std::string_view name, std::uint32_t def) {
    if (auto v = env(name)) {
        try {
            return static_cast<std::uint32_t>(std::stoul(*v));
        } catch (const std::exception&) {
        }
    }
    return def;
}

// WHIRL_DRAFT_VOCAB (default: the embedded 64k subset on dense qwen35 models with the 248320-token
// vocabulary): keep only that vocabulary subset of the 2-bit draft head (special and byte tokens are
// always added); outputs are unchanged, only the acceptance can drop.
void applyDraftVocab(qwen35::Model& model, const Tokenizer& tok, const qwen35::DraftVocabChoice& dv) {
    using K = qwen35::DraftVocabChoice::Kind;
    if (dv.kind != K::embedded_64k && dv.kind != K::file) return;
    if (dv.kind == K::embedded_64k) {
        const bool fits = dv.by_default ? qwen35::draftVocabDefaultFits(model.cfg.moe, model.cfg.n_vocab)
                                        : model.cfg.n_vocab == qwen35::draft_vocab_embedded_n_vocab;
        if (!fits) {
            if (dv.by_default)
                logI("draft head: full vocabulary (the embedded 64k subset is for dense qwen35 models with a {}-token vocabulary)",
                     qwen35::draft_vocab_embedded_n_vocab);
            else
                logW("WHIRL_DRAFT_VOCAB=64k ignored: the embedded subset needs a {}-token vocabulary (model: {})",
                     qwen35::draft_vocab_embedded_n_vocab, model.cfg.n_vocab);
            return;
        }
        if (!model.draft_d2) {
            if (dv.by_default) logI("draft head: full vocabulary (the vocabulary subset needs the 2-bit draft head)");
            else logW("WHIRL_DRAFT_VOCAB ignored: it needs the 2-bit draft head (Q6_K output head, WHIRL_DRAFT_HEAD not q4)");
            return;
        }
    }
    std::vector<std::uint32_t> req;
    for (const TokenId t : tok.specials())
        if (t >= 0) req.push_back(static_cast<std::uint32_t>(t));
    for (int b = 0; b < 256; ++b)
        if (tok.byteToken(b) >= 0) req.push_back(static_cast<std::uint32_t>(tok.byteToken(b)));
    std::uint32_t added = 0;
    const auto ids = qwen35::draftVocabIds(dv.kind == K::file ? qwen35::readDraftVocab(dv.file) : qwen35::embeddedDraftVocab64k(),
                                           model.cfg.n_vocab, req, &added);
    const std::string label = dv.kind == K::file ? dv.file : std::string(dv.by_default ? "64k (embedded, default)" : "64k (embedded)");
    if (model.setDraftVocab(ids))
        logI("draft head: vocabulary subset {} ({} of {} rows; {} special / byte tokens added)", label, ids.size(), model.cfg.n_vocab,
             added);
    else if (dv.by_default)  // the default only: no warning for a setting the user did not make
        logI("draft head: full vocabulary (no vocabulary subset with this draft head / kernel set)");
    else
        logW("WHIRL_DRAFT_VOCAB ignored: it needs the 2-bit draft head (Q6_K output head, WHIRL_DRAFT_HEAD not q4)");
}

bool envOn(std::string_view name, bool def) {
    if (auto v = env(name)) return *v != "0";
    return def;
}

// Default max drafts per cycle by number of decoding slots (index = slots).
// wide: verify batches of up to 32 rows (otherwise 16); the cost model picks within the cap.
std::array<std::uint32_t, gdn_max_seg + 1> defaultBatchDrafts(bool moe, bool wide) {
    (void)wide;
    if (moe) return {0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    return {0, 8, 7, 4, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
}

std::uint64_t alignUp(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

double gib(std::size_t a, std::size_t b) { return static_cast<double>(a > b ? a - b : 0) / (1024.0 * 1024.0 * 1024.0); }

// Identity of everything that decides the bits of a tier entry.
std::uint64_t tierFingerprint(const qwen35::Model& m, const std::string& path, const qwen35::LoadStats& stats, bool use_mtp,
                              std::uint64_t n_gdn, const tier::Layout& lay) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    h = tier::hashStr(h, "whirl-kv-tier-1");
    const std::string base = narrow(std::filesystem::path(widen(path)).filename().wstring());
    h = tier::hashStr(h, std::format("exe {:x} model {} {} {} kv {} mtp {} gdn {} ck {} pg {} batch {}", tier::exeIdentity(),
                                     base, stats.tensors, stats.bytes, m.kvName(), use_mtp, n_gdn, lay.ck_bytes,
                                     lay.page_bytes, m.max_batch));
    // the numerics mode and its items (a --balance run's KV / DeltaNet state must never be
    // restored by a --precise process: same exe, model and KV type, other prompt numerics)
    h = tier::hashStr(h, "numerics " + whirl::numerics::logLine(m.num_plan));
#ifdef _WIN32
    std::vector<std::pair<std::string, std::string>> kv;
    if (wchar_t* blk = GetEnvironmentStringsW()) {
        for (const wchar_t* p = blk; *p; p += wcslen(p) + 1) {
            const std::string s = narrow(p);
            const std::size_t eq = s.find('=', 1);
            if (eq == std::string::npos) continue;
            std::string k = s.substr(0, eq);
            std::string up = k;
            for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            if (up.rfind("WHIRL_", 0) != 0) continue;
            static const char* skip[] = {"WHIRL_KV_RAM_MB", "WHIRL_KV_SSD_DIR", "WHIRL_KV_SSD_GB", "WHIRL_KV_SSD_DELAY_MS",
                                         "WHIRL_KV_TIER_MIN", "WHIRL_R9700_LOCK_HELD", "WHIRL_GPU_WAIT", "WHIRL_GPU_SHARE",
                                         "WHIRL_PROFILE", "WHIRL_TRACE_ND", "WHIRL_GATHER_MS", "WHIRL_EXE", "WHIRL_LOOP_LOG",
                                         "WHIRL_TIER_VERIFY", "WHIRL_TIER_MIN_GAIN", "WHIRL_DECODE_MIN_TPS",
                                         "WHIRL_TIMER_PROBE"};
            bool sk = false;
            for (const char* x : skip) sk = sk || up == x;
            if (!sk) kv.emplace_back(k, s.substr(eq + 1));
        }
        FreeEnvironmentStringsW(blk);
    }
    std::sort(kv.begin(), kv.end());
    for (const auto& [k, v] : kv) {
        h = tier::hashStr(h, k);
        h = tier::hashStr(h, "=");
        h = tier::hashStr(h, v);
        h = tier::hashStr(h, ";");
    }
#endif
    return h;
}

}  // namespace

std::string serveHelp(const char* program) {
    return std::format("{} {}\n\nusage: {} MODEL.gguf [options]\n\n", program, app::version(), program) + kHelpBody +
           app::envHelp(app::sc_serve) + "\n" + app::exitCodeHelp();
}

int serveMain(int argc, char** argv, const char* program) {
    // argv: UTF-8, argv[0] = the program (or the `serve` subcommand), options from argv[1]
    g_program = program;
    std::vector<std::string> args;
    for (int k = 1; k < argc; ++k) args.push_back(argv[k]);
    for (const std::string& a : args) {
        // help / version first: they need neither a model nor a GPU
        if (a == "-h" || a == "--help" || a == "/?") {
            const std::string h = serveHelp(program);
            std::fwrite(h.data(), 1, h.size(), stdout);
            return app::exit_ok;
        }
        if (a == "-V" || a == "--version") {
            const std::string v = app::versionText(program);
            std::fwrite(v.data(), 1, v.size(), stdout);
            return app::exit_ok;
        }
    }
    Options opt;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string arg = args[i];
        std::string mspec;
        bool mnext = false;
        bool is_mode = false;
        try {
            is_mode = numerics::modeFlag(arg, &mspec, &mnext);
        } catch (const std::exception& ex) {
            die(ex.what());
        }
        if (is_mode) {
            if (mnext) {
                if (i + 1 >= args.size()) die("missing value for --mode");
                mspec = args[++i];
            }
            if (opt.mode_spec) die("give one numerics mode (--precise, --balance, --fast or --mode M)");
            try {
                (void)numerics::parseMode(mspec, "command line");
            } catch (const std::exception& ex) {
                die(ex.what());
            }
            opt.mode_spec = mspec;
            continue;
        }
        if (auto v = argValue(args, i, "--host")) opt.host = *v;
        else if (auto v2 = argValue(args, i, "--port")) opt.port = parseNum<std::uint16_t>(*v2, "--port");
        else if (auto v3 = argValue(args, i, "--ctx")) opt.ctx = parseNum<std::uint32_t>(*v3, "--ctx");
        else if (auto v3b = argValue(args, i, "-c")) opt.ctx = parseNum<std::uint32_t>(*v3b, "--ctx");
        else if (auto v4 = argValue(args, i, "--parallel")) { opt.parallel = parseNum<std::uint32_t>(*v4, "--parallel"); opt.parallel_given = true; }
        else if (auto v4b = argValue(args, i, "-np")) { opt.parallel = parseNum<std::uint32_t>(*v4b, "--parallel"); opt.parallel_given = true; }
        else if (auto v5 = argValue(args, i, "--mtp-drafts"))
            opt.mtp_drafts = std::clamp(parseNum<std::uint32_t>(*v5, "--mtp-drafts"), 1u, qwen35::max_drafts);
        else if (auto v5b = argValue(args, i, "--decode-min-tps"))
            opt.decode_min_tps = parseNum<std::uint32_t>(*v5b, "--decode-min-tps");
        else if (auto v6 = argValue(args, i, "--ctx-per-slot")) opt.ctx_per_slot = parseNum<std::uint32_t>(*v6, "--ctx-per-slot");
        else if (auto v7 = argValue(args, i, "--log-file")) opt.log_file = *v7;
        else if (auto v8 = argValue(args, i, "--alias")) opt.alias = *v8;
        else if (auto v9 = argValue(args, i, "--kv-ram-mb")) opt.kv_ram_mb = parseNum<std::uint64_t>(*v9, "--kv-ram-mb");
        else if (auto v10 = argValue(args, i, "--kv-ssd-dir")) opt.kv_ssd_dir = *v10;
        else if (auto v11 = argValue(args, i, "--kv-ssd-gb")) opt.kv_ssd_gb = parseNum<std::uint64_t>(*v11, "--kv-ssd-gb");
        else if (auto v12 = argValue(args, i, "--device")) opt.device = *v12;
        else if (auto v13 = argValue(args, i, "--mmproj")) opt.mmproj = *v13;
        else if (auto v14 = argValue(args, i, "--vis-idle-s")) opt.vis_idle_s = parseNum<std::uint32_t>(*v14, "--vis-idle-s");
        else if (auto v15 = argValue(args, i, "--vis-mode")) {
            if (!vision::parseMode(*v15)) die("invalid --vis-mode " + *v15);
            opt.vis_mode = *v15;
        } else if (auto v16 = argValue(args, i, "--vis-cache-mb")) opt.vis_cache_mb = parseNum<std::uint64_t>(*v16, "--vis-cache-mb");
        else if (arg == "--allow-local-images") opt.allow_local_images = true;
        else if (auto v17 = argValue(args, i, "--cors-origin")) {
            if (*v17 != "*" && v17->find("://") == std::string::npos)
                die("invalid --cors-origin " + *v17 + " (give an origin such as https://app.example.com, or *)");
            opt.cors_origins.push_back(*v17);
        }
        else if (!arg.empty() && arg[0] == '-') die("unknown option " + arg);
        else if (opt.path.empty()) opt.path = arg;
        else die("unexpected argument " + arg);
    }
    if (opt.path.empty()) die("missing MODEL.gguf");
    numerics::Request nreq;
    try {
        nreq = numerics::resolve(opt.mode_spec, env("MODE"));
    } catch (const std::exception& ex) {
        die(ex.what());
    }
    if (opt.parallel < 1 || opt.parallel > gdn_max_seg) die("--parallel must be 1..16");
    // the model file and the port are checked before the (long) model load
    try {
        app::requireModelFile(opt.path);
        if (opt.mmproj) app::requireModelFile(*opt.mmproj, "vision encoder (--mmproj) file");
        app::requirePortFree(opt.host, opt.port);
    } catch (const std::exception& ex) {
        return app::explain(ex, g_program);
    }

    const char* lad = std::getenv("LOCALAPPDATA");
    const std::string log_path = opt.log_file ? *opt.log_file
                                              : (lad ? std::string(lad) + "\\whirl\\server.log" : std::string("whirl-server.log"));
    try {
        Log::open(log_path);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "warning: %s\n", ex.what());
    }

    try {
        logI("WHIRL serve starting, log file {}", log_path);
        const TimePoint t_load0 = Clock::now();
        const gguf::File f = gguf::File::open(opt.path);
        const qwen35::Config cfg = qwen35::Config::fromGguf(f);
        int dev = 0;
        if (auto v = env("HIP_DEVICE")) dev = std::stoi(*v);
        else dev = qwen35::pickDevice(opt.device);
        const hip::DeviceInfo info = hip::describeDevice(dev);
        hip::setDevice(dev);
        // integrated GPU (Radeon 8060S): the "VRAM" is system memory shared with the CPU
        const bool is_uma = info.integrated;
        const std::optional<std::uint32_t> pool_req = opt.ctx ? opt.ctx : (is_uma ? std::optional<std::uint32_t>(262144) : std::nullopt);
        const std::uint32_t cap_max = 262144;
        const std::uint32_t slot_ctx = static_cast<std::uint32_t>(alignUp(
            opt.ctx_per_slot ? *opt.ctx_per_slot : (pool_req ? std::min(*pool_req, ctx_per_slot_default) : ctx_per_slot_default),
            kv_page));
        if (slot_ctx < 1024) die(std::format("context per slot must be >= 1024 (got {})", slot_ctx));
        logI("model: {}", opt.path);
        logI("arch {}, layers {} (attn every {}), embd {}, ff {}, vocab {}{}", cfg.archName(), cfg.n_layer, cfg.interval,
             cfg.n_embd, cfg.n_ff, cfg.n_vocab, cfg.n_nextn > 0 ? ", MTP (nextn) layer" : "");
        if (cfg.moe)
            logI("experts {} (top {}, ff {}) + shared expert ff {}", cfg.n_expert, cfg.n_expert_used, cfg.n_ff_exp, cfg.n_ff);
        logI("device {}: {} ({}), {:.1f} GiB VRAM{}", dev, info.name, info.gcn_arch,
             static_cast<double>(info.total_mem) / (1024.0 * 1024.0 * 1024.0), vram::limitBytes() ? " (simulated limit)" : "");
        if (vram::limitBytes()) logI("{}: memory sizes, the WDDM budget and allocations are capped as on a card of that size", vram::limitNote());
        // small cards (< 20 GiB, e.g. a 16 GB RX 9070 XT that also drives the desktop): 1 slot,
        // q8v KV first, VRAM headroom for other apps; cards >= 20 GiB are unchanged
        vram::CardDefaults card;
        {
            vram::CardInput ci;
            ci.total = info.total_mem;
            ci.uma = is_uma;
            if (opt.parallel_given) ci.parallel_arg = opt.parallel;
            ci.parallel_default = opt.parallel;
            ci.kv_auto = env("KV").value_or("auto") == "auto" && !cfg.moe;
            ci.kv_quant_ok = nreq.has(numerics::Item::kvq8);  // precise: never prefer int8 KV
            if (env("VRAM_HEADROOM_MB")) ci.headroom_mb_env = envU32("VRAM_HEADROOM_MB", 0);
            ci.reserve_explicit = env("POOL_RESERVE_MB").has_value();
            card = vram::cardDefaults(ci);
            opt.parallel = card.parallel;
            if (!card.note.empty()) logI("{}", card.note);
        }

        const Tokenizer tok = Tokenizer::fromGguf(f);
        const std::string_view tmpl_text = f.getStringOr("tokenizer.chat_template", "");
        const chat::TemplateKind tmpl = chat::detectTemplate(tmpl_text);
        logI("chat template: GGUF tokenizer.chat_template ({} B) -> built-in Qwen3.x renderer variant '{}' ({})", tmpl_text.size(),
             chat::templateName(tmpl),
             tmpl == chat::TemplateKind::a ? "reasoning effort, merged system, preserve_thinking" : "system + tools, <think> in history");

        const hip::MemInfo mem0 = hip::memInfo();
        qwen35::LoadStats stats;
        qwen35::LoadOptions lo;
        lo.numerics = nreq;
        lo.max_batch = serve_prefill_batch;
        lo.embd_on_host = envOn("EMBD_HOST", true);  // WHIRL_EMBD_HOST=0: embedding in VRAM
        if (auto v = env("PREFILL_BATCH")) {
            try {
                lo.max_batch = std::clamp(static_cast<std::uint32_t>(std::stoul(*v)), 1u, qwen35::max_batch_limit);
            } catch (const std::exception&) {
            }
        }
        // prefix checkpoints (per slot / shared); the auto-shrink below may lower the defaults
        const bool no_pc = envOn("NO_PREFIX_CACHE", false);
        const std::size_t ck_def = opt.parallel == 1 ? n_ckpt_default : (opt.parallel <= 4 ? n_ckpt_default_par : 1);
        std::size_t n_ck = no_pc ? 0 : envU32("SERVE_CKPTS", static_cast<std::uint32_t>(ck_def));
        std::uint32_t n_spe_plan = (n_ck == 0 || no_pc) ? 0 : envU32("SYS_CKPTS", static_cast<std::uint32_t>(n_spe_default));
        // VRAM estimate before loading: when the KV pool left after weights, prefill buffers and
        // checkpoints would be small, lower the prefill batch and the checkpoint counts first
        // (only values the user did not set; nothing changes when the pool has room). Not on the
        // UMA iGPU, whose VRAM numbers do not bound what it can allocate.
        if (!is_uma) {
            const qwen35::LoadEstimate est = qwen35::estimateLoad(f, cfg, lo.embd_on_host);
            const hip::MemInfo mf = hip::memInfo();
            const hip::WddmMemInfo w = hip::wddmMemInfo();
            const bool res_explicit = env("POOL_RESERVE_MB").has_value();
            const std::uint64_t res = static_cast<std::uint64_t>(envU32("POOL_RESERVE_MB", cfg.moe ? 1536 : (w.ok ? 768 : 3072))) << 20;
            const std::uint64_t margin = static_cast<std::uint64_t>(envU32("POOL_BUDGET_MARGIN_MB", cfg.moe ? 1536 : 768)) << 20;
            std::uint64_t av = mf.free > res ? mf.free - res : 0;
            if (!res_explicit && w.ok) av = std::min(av, w.local_budget > w.local_usage + margin ? w.local_budget - w.local_usage - margin : 0);
            av = av > card.headroom ? av - card.headroom : 0;
            std::uint64_t n_gdn_c = 0;
            for (std::uint32_t li = 0; li < cfg.n_layer; ++li)
                if (!cfg.isAttn(li)) ++n_gdn_c;
            const std::uint64_t st1 = n_gdn_c * (static_cast<std::uint64_t>(cfg.d_conv - 1) * cfg.convCh() * 4 +
                                                 static_cast<std::uint64_t>(cfg.n_v_heads) * cfg.d_state * cfg.headV() * 4);
            const std::uint64_t pend1 = n_gdn_c * (static_cast<std::uint64_t>(qwen35::max_small_batch) * cfg.convCh() * 4 +
                                                   static_cast<std::uint64_t>(qwen35::max_small_batch) * cfg.n_v_heads *
                                                       (cfg.d_state + cfg.headV() + 2) * 4);
            vram::FitInput fi;
            fi.avail = av;
            fi.weights = est.weights;
            // load-time state, per-slot state + replay rows, draft head (64k rows, 2 bit), slack; and the
            // ~4% more that WDDM sees for this process than the sum of its allocations
            fi.fixed = est.state + opt.parallel * (st1 + pend1) + (est.has_mtp ? 65536ull * cfg.n_embd * 5 / 16 : 0) + (256ull << 20);
            fi.fixed += (est.weights + fi.fixed) / 25;
            const std::uint64_t b2 = qwen35::bufferBytes(cfg, 2048, est.ffs, est.max_elems);
            const std::uint64_t b4 = qwen35::bufferBytes(cfg, 4096, est.ffs, est.max_elems);
            fi.buf_per_row = (b4 - b2) / 2048;
            fi.buf_fixed = b2 - fi.buf_per_row * 2048;
            // (WHIRL_CKPT_HOST=1: checkpoints in pinned host memory, no VRAM)
            fi.ckpt_bytes = envOn("CKPT_HOST", false) ? 0 : st1 + static_cast<std::uint64_t>(cfg.n_embd) * 4 + static_cast<std::uint64_t>(cfg.n_vocab) * 4;
            fi.parallel = opt.parallel;
            const std::string kv_env = env("KV").value_or("auto");
            // KV auto: precise f16; balance / fast as before (dense: the q8h size)
            fi.kv_per_token = kv_env == "f16" ? est.kv_f16 : kv_env == "q8v" ? est.kv_q8v : (kv_env == "q8" || kv_env == "q8h") ? est.kv_q8h
                              : (cfg.moe || !nreq.has(numerics::Item::kvq8)) ? est.kv_f16 : est.kv_q8h;
            fi.pool_min_tokens = pool_req ? *pool_req : std::min<std::uint32_t>(slot_ctx, vram_tight_pool);
            fi.batch = lo.max_batch;
            fi.n_ck = static_cast<std::uint32_t>(n_ck);
            fi.n_spe = n_spe_plan;
            fi.batch_fixed = env("PREFILL_BATCH").has_value();
            fi.ck_fixed = env("SERVE_CKPTS").has_value() || env("SYS_CKPTS").has_value();
            const vram::FitPlan plan = vram::planFit(fi);
            const double g = 1024.0 * 1024.0 * 1024.0;
            if (plan.reduced) {
                lo.max_batch = plan.batch;
                n_ck = plan.n_ck;
                n_spe_plan = plan.n_spe;
                logW("VRAM is tight (about {:.2f} GiB usable for weights {:.2f} GiB + buffers + checkpoints + KV): reduced {} so the KV "
                     "pool gets about {} tokens (wanted {}); set WHIRL_PREFILL_BATCH / WHIRL_SERVE_CKPTS / WHIRL_SYS_CKPTS to choose",
                     av / g, est.weights / g, plan.note, plan.pool_tokens, fi.pool_min_tokens);
            }
            // not even the weights + smallest buffers fit (no room for any KV): stop before a load that
            // would run out of memory part way through, with the "smaller quantization" advice
            // (WHIRL_FIT_CHECK=0 loads anyway; borderline cases only warn)
            if (!plan.fits && plan.pool_tokens == 0 && envOn("FIT_CHECK", true))
                throw qwen35::ModelError("VramLimit",
                                         std::format("needs about {:.2f} GiB of VRAM before any KV cache (weights {:.2f} GiB + prefill batch {}, "
                                                     "state, checkpoints), about {:.2f} GiB usable of {:.2f} GiB{}",
                                                     plan.need_fixed / g, est.weights / g, plan.batch, av / g, mf.total / g,
                                                     vram::limitBytes() ? " (" + vram::limitNote() + ")" : std::string()));
            if (!plan.fits)
                logW("VRAM estimate: weights + buffers + checkpoints {:.2f} GiB leave about {} KV tokens of {:.2f} GiB usable "
                     "(at least {} needed); the model may not fit this GPU",
                     plan.need_fixed / g, plan.pool_tokens, av / g, fi.pool_hard_min);
        }
        std::size_t prefill_exec = 2048;
        if (auto v = env("PREFILL_CHUNK")) {
            try {
                prefill_exec = std::stoul(*v);
            } catch (const std::exception&) {
            }
            logI("prefill forward up to {} rows requested (WHIRL_PREFILL_CHUNK)", prefill_exec);
        }
        // whole schedule chunks, and a forward (+ the merged tail) must fit the prefill batch
        {
            const std::size_t mb = lo.max_batch > prefill_merge ? lo.max_batch - prefill_merge : 0;
            prefill_exec = std::max(prefill_chunk, std::min(prefill_exec / prefill_chunk * prefill_chunk, mb / prefill_chunk * prefill_chunk));
        }
        std::unique_ptr<qwen35::Model> mp = qwen35::Model::load(f, 0, stats, lo);
        qwen35::Model& model = *mp;
        {
            std::string tl;
            qwen35::loadOrTune(model, opt.path, tl);
            std::size_t p = 0;
            while (p < tl.size()) {
                std::size_t e = tl.find('\n', p);
                if (e == std::string::npos) e = tl.size();
                const std::string_view l = trimWs(std::string_view(tl).substr(p, e - p));
                if (!l.empty()) logI("{}", l);
                p = e + 1;
            }
        }
        const hip::MemInfo mem = hip::memInfo();
        logI("loaded {} tensors, {:.2f} GiB in {:.1f} s ({:.2f} GB/s)", stats.tensors,
             static_cast<double>(stats.bytes) / (1024.0 * 1024.0 * 1024.0), stats.ms / 1000.0,
             static_cast<double>(stats.bytes) / (stats.ms * 1e6));

        // pinned host memory shows up as this process's GPU "Shared Usage": declare it so monitoring
        // does not read the embedding table as VRAM spill (the RAM tier / vision add theirs below)
        const std::uint64_t embd_pinned =
            model.embd_host ? static_cast<std::uint64_t>(model.tok_embd.row_bytes) * model.tok_embd.nrows : 0;
        if (embd_pinned > 0) {
            tier::declarePinned(lad, embd_pinned);
            logI("token embedding: {:.0f} MiB in pinned host memory (counted as this process's GPU 'Shared Usage', declared; "
                 "WHIRL_EMBD_HOST=0 keeps it in VRAM)",
                 static_cast<double>(embd_pinned) / 1048576.0);
        }

        // same knobs as the CLI
        if (env("NO_FUSE")) model.no_fuse = true;
        if (env("FLOAT_GEMV")) model.float_gemv = true;
        if (env("NAIVE_ATTN")) model.naive_attn = true;
        if (env("GDN_SEQ")) model.gdn_chunked = false;
        if (auto v = env("MOE_BN")) model.moe_bn_force = static_cast<std::uint32_t>(std::strtoul(v->c_str(), nullptr, 10));
        if (auto v = env("PREFILL_BATCH"))
            model.max_batch = std::max(1u, std::min(model.max_batch, static_cast<std::uint32_t>(std::stoul(*v))));
        // WHIRL_DRAFT_VOCAB: unset = embedded 64k subset (dense qwen35, 248320 vocab), 64k = embedded,
        // 48k / <file> = frequency subset file, N = the first N rows (old experiment), off = full head
        const qwen35::DraftVocabChoice draft_vocab = qwen35::draftVocabChoice(env("DRAFT_VOCAB"), qwen35::exeDirectory());
        if (draft_vocab.kind == qwen35::DraftVocabChoice::Kind::first_n) model.draft_vocab = draft_vocab.n;
        if (auto v = env("DRAFT_WINDOW")) model.draft_window = static_cast<std::uint32_t>(std::stoul(*v));
        if (auto v = env("DRAFT_WINDOW_MIN")) model.draft_window_min = static_cast<std::uint32_t>(std::stoul(*v));
        if (env("GEMV_R") || env("GEMV_W") || env("GEMV_WH"))
            logW("WHIRL_GEMV_R / WHIRL_GEMV_W / WHIRL_GEMV_WH are not supported by whirl-server (ignored)");
        model.use_graph = false;
        const bool use_mtp = model.mtp.has_value() && envOn("MTP", true);
        const qwen35::MtpDefaults def = qwen35::mtpDefaults(model.cfg.moe, model.arch);
        std::optional<std::uint32_t> drafts_env;
        if (auto v = env("MTP_DRAFTS")) {
            std::uint32_t d = def.drafts;
            try {
                d = static_cast<std::uint32_t>(std::stoul(*v));
            } catch (const std::exception&) {
            }
            drafts_env = std::clamp(d, 1u, qwen35::max_drafts);
        }
        const std::optional<std::uint32_t> drafts_user = opt.mtp_drafts ? opt.mtp_drafts : drafts_env;
        const std::uint32_t drafts = drafts_user.value_or(def.drafts);
        float p_min = def.p_min;
        if (auto v = env("MTP_PMIN")) {
            try {
                p_min = std::stof(*v);
            } catch (const std::exception&) {
            }
        }
        const std::uint32_t n_min = envU32("MTP_NMIN", 0);
        const auto adapt_env = env("MTP_ADAPT");
        std::uint32_t adapt = def.adapt;
        if (adapt_env) {
            try {
                adapt = static_cast<std::uint32_t>(std::stoul(*adapt_env));
            } catch (const std::exception&) {
                adapt = 0;
            }
        }
        // an explicit draft count is used as is (fixed) unless WHIRL_MTP_ADAPT asks otherwise
        const bool auto_mode = adapt_env ? (*adapt_env == "auto") : (def.automatic && !drafts_user);
        if (!envOn("WIDE_VERIFY", true)) model.wide_verify = false;
        if (auto hc = env("HEAD_CHUNK")) model.head_chunk = std::max<std::uint32_t>(32, static_cast<std::uint32_t>(std::stoul(*hc)) / 32 * 32);
        auto batch_drafts = defaultBatchDrafts(model.cfg.moe, model.wideCapable());
        batch_drafts[1] = drafts;
        const auto bd_env = env("MTP_BATCH_DRAFTS");
        if (bd_env) {
            std::size_t k = 1, p = 0;
            const std::string& s = *bd_env;
            while (p < s.size() && k <= gdn_max_seg) {
                std::size_t e = s.find(',', p);
                if (e == std::string::npos) e = s.size();
                if (e > p) {
                    std::uint32_t x = 1;
                    try {
                        x = static_cast<std::uint32_t>(std::stoul(s.substr(p, e - p)));
                    } catch (const std::exception&) {
                    }
                    batch_drafts[k++] = std::min(x, qwen35::max_drafts);
                }
                p = e + 1;
            }
            for (; k <= gdn_max_seg; ++k) batch_drafts[k] = batch_drafts[k - 1];
        }
        if (env("GDN_V0")) {
            if (hip::Function fv0 = model.module.getFunctionOpt("gdn_step_norm_v0")) model.k.gdn_step_norm = fv0;
        }
        // MTP block as Q4_K from Q6_K or Q8_0 (drafts only; WHIRL_MTP_Q4=0 keeps the file's types)
        if (use_mtp && envOn("MTP_Q4", true)) model.requantMtpQ4();
        if (!envOn("ATTN_WIDE", true)) model.attn_wide = false;
        if (use_mtp && !env("MTP_FULLHEAD")) {
            const bool q4 = env("DRAFT_HEAD").value_or("") == "q4";
            model.buildDraftHeadEx(q4 ? qwen35::Model::DraftHeadKind::q4 : qwen35::Model::DraftHeadKind::d2);
            applyDraftVocab(model, tok, draft_vocab);
        }
        if (use_mtp) {
            logI("MTP speculative decoding: on, up to {} draft(s) per cycle{}, p-min {:.2f}, n-min {}, draft count {}", drafts,
                 (!model.fusedDecode() && drafts > 1) ? " (fused decode off: 1 used)" : "", p_min, n_min,
                 auto_mode ? "auto (cost model)" : (adapt > 0 ? "adaptive margin" : "fixed"));
            std::string tbl;
            for (std::uint32_t k = 1; k <= opt.parallel; ++k) tbl += (k > 1 ? ", " : "") + std::to_string(batch_drafts[k]);
            logI("MTP under load: max drafts per cycle with 1..{} decoding slots: {{ {} }}", opt.parallel, tbl);
        } else {
            logI("MTP speculative decoding: off{}", !model.mtp ? " (no nextn layer)" : " (WHIRL_MTP=0)");
        }

        // slots: KV-cache regions, recurrent state, MTP hidden rows
        model.gdn_replay = envOn("GDN_REPLAY", true);
        const hip::MemInfo mem_s0 = hip::memInfo();
        model.setupSeqs(opt.parallel, slot_ctx);
        if (model.gdn_replay) {
            logI("DeltaNet verify: replay of kept rows ({:.1f} MiB per slot), no snapshot sets",
                 static_cast<double>(model.pendLayerBytes() * model.gdn_ord[cfg.n_layer - 1]) / (1024.0 * 1024.0));
        }
        if (use_mtp && !model.gdn_replay) {
            if (!bd_env)
                for (std::uint32_t k = 2; k <= gdn_max_seg; ++k) batch_drafts[k] = std::min(batch_drafts[k], 8 / k);
            std::uint32_t need = drafts;
            for (std::uint32_t k = 1; k <= opt.parallel; ++k) need = std::max(need, k * std::min(batch_drafts[k], drafts));
            if (auto v = env("SNAP_SETS")) {
                std::uint32_t x = 0;
                try {
                    x = static_cast<std::uint32_t>(std::stoul(*v));
                } catch (const std::exception&) {
                }
                need = std::max(need, std::min(x, qwen35::gdn_max_snap));
            }
            model.ensureSnapshots(std::min(qwen35::gdn_max_snap, need));
        }

        // engine (checkpoint buffers, shared checkpoints, sampling buffers)
        const std::uint64_t conv_bytes = model.convBytes();
        const std::uint64_t ssm_bytes = model.ssmBytes();
        std::uint64_t n_gdn = 0;
        for (std::uint32_t li = 0; li < cfg.n_layer; ++li)
            if (model.ssm_state[li] != 0) ++n_gdn;
        const std::uint32_t V = cfg.n_vocab;
        std::random_device rd;
        const std::uint64_t seed0 = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();

        const std::string base = narrow(std::filesystem::path(widen(opt.path)).filename().wstring());
        std::string name = opt.alias ? *opt.alias : base;
        if (!opt.alias && name.size() > 5) {
            std::string tail = name.substr(name.size() - 5);
            for (char& c : tail) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (tail == ".gguf") name = name.substr(0, name.size() - 5);
        }
        EngineOptions eo;
        eo.model_name = name;
        eo.model_file = base;
        eo.ctx = slot_ctx;
        eo.parallel = opt.parallel;
        eo.tmpl = tmpl;
        eo.use_mtp = use_mtp;
        eo.spec_sample = model.num_plan.on(numerics::Item::specsample);
        eo.relax = model.relax;
        eo.n_draft = drafts;
        eo.p_min = p_min;
        eo.n_min = n_min;
        eo.mtp_adapt = adapt;
        eo.mtp_auto = auto_mode;
        eo.batch_drafts = batch_drafts;
        eo.no_prefix_cache = no_pc;
        eo.verify_cache = env("PREFIX_CACHE_VERIFY").has_value();
        eo.seg_prefill = envOn("SEG_PREFILL", true);
        eo.timing_reset = envOn("TIMING_RESET", true);
        eo.ngram = envOn("NGRAM", true);
        eo.ngram_min = envU32("NGRAM_MIN", def.ngram_min);
        eo.ngram_slope = def.ngram_slope;
        if (auto v = env("NGRAM_SLOPE")) eo.ngram_slope = std::strtof(v->c_str(), nullptr);
        eo.ngram_max = envU32("NGRAM_MAX", 0);
        eo.ngram_force = envOn("NGRAM_FORCE", false);
        // WHIRL_SLOT_DRAFTS: 0 / unset uniform, 1 marginal-gain swaps, 2 cost-model allocation
        if (auto v = env("SLOT_DRAFTS")) eo.slot_drafts = *v == "2" ? 2u : (*v != "0" ? 1u : 0u);
        eo.trace_nd = env("TRACE_ND").has_value();
        eo.loop_log = env("LOOP_LOG").has_value();
        eo.tier_verify = env("TIER_VERIFY").has_value();
        eo.tier_min_gain = envU32("TIER_MIN_GAIN", 512);
        if (auto v = env("GATHER_MS")) {
            try {
                eo.gather_ms = std::stod(*v);
            } catch (const std::exception&) {
            }
        }
        eo.decode_min_tps = opt.decode_min_tps ? *opt.decode_min_tps : envU32("DECODE_MIN_TPS", decode_min_tps_default);
        if (eo.decode_min_tps > 0)
            logI("decode floor: >= {:.0f} tok/s per decoding request while others prefill (--decode-min-tps)", eo.decode_min_tps);
        else
            logI("decode floor: off (--decode-min-tps 0: prefill forwards are not limited)");
        if (auto v = env("PROFILE")) eo.profile = *v == "2" ? 2 : 1;
        eo.sys_min = envU32("SYS_MIN", sys_min_default);
        eo.lcp_on = envOn("SYS_LCP", true);
        eo.n_ck = n_ck;
        eo.n_spe = (n_ck == 0 || no_pc) ? 0 : n_spe_plan;
        eo.ckpt_host = envOn("CKPT_HOST", false);
        eo.prefill_exec = prefill_exec;
        eo.seed = seed0;

        auto ops = tier::makeHipDeviceOps();
        auto backend = makeQwen35Backend(model);
        Engine engine(*backend, *ops, tok, eo);
        const hip::MemInfo mem_ck0 = hip::memInfo();
        engine.allocState();
        const hip::MemInfo mem_ck1 = hip::memInfo();

        // host tiers, VRAM side first (tier stream, pinned arena mapped for the
        // GPU): the KV pool below is sized after it
        const std::uint64_t ram_def = [&] {
            const std::uint64_t ck_stride = alignUp(n_gdn * (conv_bytes + ssm_bytes) + static_cast<std::uint64_t>(cfg.n_embd) * 4 +
                                                        static_cast<std::uint64_t>(V) * 4,
                                                    tier::block_bytes);
            const std::uint64_t full = static_cast<std::uint64_t>(slot_ctx) * model.kvBytesPerTokenFmt(false, false) + n_ck * ck_stride;
            return std::max(tier::ram_auto_min_mb, alignUp(full, 1ull << 30) >> 20);
        }();
        // size: explicit (--kv-ram-mb / WHIRL_KV_RAM_MB), else 1/4 of physical RAM
        // within [ram_def, 32 GiB] and at most half of the RAM available now;
        // integrated GPU: the KV pool already lives in system memory, so the RAM
        // tier (and the SSD tier behind it) is off unless asked for explicitly
        tier::RamTierInput rin;
        rin.floor_mb = ram_def;
        rin.integrated = info.integrated;
        if (opt.kv_ram_mb) {
            rin.explicit_mb = *opt.kv_ram_mb;
        } else if (auto v = env("KV_RAM_MB")) {
            try {
                rin.explicit_mb = std::stoull(*v);
                rin.explicit_src = "WHIRL_KV_RAM_MB";
            } catch (const std::exception&) {
                logW("kv tier: WHIRL_KV_RAM_MB=\"{}\" is not a number; using the automatic size", *v);
            }
        }
        {
            MEMORYSTATUSEX ms{};
            ms.dwLength = sizeof(ms);
            if (GlobalMemoryStatusEx(&ms)) {
                rin.total_phys = ms.ullTotalPhys;
                rin.avail_phys = ms.ullAvailPhys;
            }
        }
        const tier::RamTierSize rsz = tier::ramTierSize(rin);
        const std::uint64_t kv_ram_mb = rsz.mb;
        std::unique_ptr<tier::Tier> tier_pre;
        double tier_alloc_s = 0.0;
        if (kv_ram_mb > 0 && n_ck > 0 && !no_pc) {
            if (rsz.avail_limited) logW("kv tier: RAM tier size {} MiB ({})", kv_ram_mb, rsz.reason);
            else logI("kv tier: RAM tier size {} MiB ({})", kv_ram_mb, rsz.reason);
            if (rin.explicit_mb && rin.avail_phys > 0 && (kv_ram_mb << 20) > rin.avail_phys / 2)
                logW("kv tier: {} MiB is more than half of the {} MiB of RAM available now; the system may page", kv_ram_mb,
                     rin.avail_phys >> 20);
            const TimePoint t_alloc0 = Clock::now();
            try {
                tier_pre = tier::Tier::create(*ops, kv_ram_mb << 20, seed0 ^ 0x5bd1e995ull, dev);
            } catch (const std::exception& ex) {
                logE("kv tier: cannot allocate {} MiB of pinned host memory ({}); running without host tiers", kv_ram_mb, ex.what());
            }
            tier_alloc_s = msSince(t_alloc0) / 1000.0;
        } else if (kv_ram_mb == 0 && n_ck > 0 && !no_pc) {
            logI("kv tier: RAM tier off ({})", rsz.reason);
        }
        const hip::MemInfo mem_t1 = hip::memInfo();
        // shared KV pool: --ctx, or (R9700) the VRAM left now minus a reserve
        // The reserve is kept below both the free VRAM and this process's WDDM budget: going over the
        // budget makes Windows demote allocations to system memory (seen as GPU 'Shared Usage'), and a
        // demoted KV array is read across PCIe on every step (FIX-1: ~128k agent decode 29 -> 68 tok/s).
        // WHIRL_POOL_RESERVE_MB keeps the old free-VRAM-only rule with that reserve.
        const bool reserve_explicit = env("POOL_RESERVE_MB").has_value();
        const hip::WddmMemInfo wddm0 = hip::wddmMemInfo();
        const std::uint64_t reserve =
            static_cast<std::uint64_t>(envU32("POOL_RESERVE_MB", cfg.moe ? 1536 : (wddm0.ok ? 768 : 3072))) << 20;
        const std::uint64_t budget_margin = static_cast<std::uint64_t>(envU32("POOL_BUDGET_MARGIN_MB", cfg.moe ? 1536 : 768)) << 20;
        std::string budget_note;
        const std::uint64_t avail = [&] {
            const hip::MemInfo mf = hip::memInfo();
            std::uint64_t a = mf.free > reserve ? mf.free - reserve : 0;
            if (!reserve_explicit && wddm0.ok) {
                const std::uint64_t room = wddm0.local_budget > wddm0.local_usage + budget_margin
                                               ? wddm0.local_budget - wddm0.local_usage - budget_margin
                                               : 0;
                budget_note = std::format("WDDM budget {:.2f} GiB, used {:.2f} GiB, margin {} MiB -> room {:.2f} GiB; free VRAM {:.2f} GiB "
                                          "minus {} MiB -> {:.2f} GiB",
                                          wddm0.local_budget / 1073741824.0, wddm0.local_usage / 1073741824.0, budget_margin >> 20,
                                          room / 1073741824.0, mf.free / 1073741824.0, reserve >> 20, a / 1073741824.0);
                a = std::min(a, room);
            } else if (reserve_explicit) {
                budget_note = std::format("WHIRL_POOL_RESERVE_MB={}: free VRAM {:.2f} GiB minus the reserve, WDDM budget not applied",
                                          reserve >> 20, mf.free / 1073741824.0);
            } else {
                budget_note = std::format("WDDM budget unavailable: free VRAM {:.2f} GiB minus {} MiB", mf.free / 1073741824.0, reserve >> 20);
            }
            if (card.headroom > 0) {
                budget_note += std::format("; minus {} MiB desktop headroom (WHIRL_VRAM_HEADROOM_MB) -> {:.2f} GiB", card.headroom >> 20,
                                           (a > card.headroom ? a - card.headroom : 0) / 1073741824.0);
                a = a > card.headroom ? a - card.headroom : 0;
            }
            return a;
        }();
        if (!pool_req) logI("kv pool sizing: {}", budget_note);
        std::string kv_note;
        // precise (KV auto without the kvq8 item): always f16; a pool / context that does not fit
        // shrinks when not given explicitly, else the server refuses to start
        // (MoE models always had f16 KV: their pool sizing is unchanged)
        const bool kv_precise = model.kvAutoDense() && !model.kvQuantAuto();
        std::uint32_t slot_ctx_eff = slot_ctx;
        std::optional<std::uint32_t> precise_pool;
        if (kv_precise) {
            const numerics::PoolFit pf = numerics::serverPoolFit(avail, model.kvBytesPerTokenFmt(false, false), pool_req, opt.ctx.has_value(),
                                                                 slot_ctx, opt.ctx_per_slot.has_value(), kv_page, 8 * cap_max);
            if (pf.refuse) throw app::UserError(app::exit_vram, pf.msg);
            if (!pf.msg.empty()) logW("{}", pf.msg);
            precise_pool = pf.pool;
            slot_ctx_eff = pf.slot_ctx;
            kv_note = " (precise mode: f16 KV)";
        } else if (model.kvAutoDense()) {
            const std::uint32_t second = opt.parallel > 1 ? std::min(floor_second_ctx, slot_ctx) : 0;
            const std::uint64_t need = pool_req ? *pool_req : static_cast<std::uint64_t>(slot_ctx) + second + (opt.parallel > 1 ? 2ull : 1ull) * kv_page;
            // small-card q8v preference only where the code object has the q8v and q8h kernels
            const bool small_kv = card.prefer_q8v && model.k.caps.kv_q8v && model.k.caps.kv_q8h;
            const vram::SmallKv sk = small_kv ? vram::smallCardKv(avail, need, model.kvBytesPerTokenFmt(true, true)) : vram::SmallKv::q8h;
            if (small_kv && sk != vram::SmallKv::q8h) {
                model.setKvFormat(true, false, true);
                kv_note = sk == vram::SmallKv::q8v
                              ? " (auto: small card (< 20 GiB) prefers q8v; WHIRL_KV=f16 forces f16)"
                              : " (auto: small card (< 20 GiB) prefers q8v, with a pool below one full request rather than K in int8; "
                                "WHIRL_KV=q8 for a longer pool)";
            } else if (small_kv) {
                model.setKvFormat(true, true, false);
                kv_note = std::format(" (auto: small card (< 20 GiB), q8v would leave under {} tokens)", vram::small_card_q8v_min_tokens);
            } else if (need * model.kvBytesPerTokenFmt(false, false) <= avail) {
                kv_note = " (auto: the floor pool fits as f16)";
            } else if (model.k.caps.kv_q8v && need * model.kvBytesPerTokenFmt(true, true) <= avail) {
                model.setKvFormat(true, false, true);
                kv_note = " (auto: the floor pool (one full request + a 64k second one) does not fit as f16, fits as q8v)";
            } else {
                model.setKvFormat(true, model.k.caps.kv_q8h, false);
                kv_note = " (auto: the floor pool (one full request + a 64k second one) does not fit as f16 or q8v)";
            }
        }
        const std::uint64_t per_tok = model.kvBytesPerToken();
        const std::uint32_t pool_tokens = precise_pool ? *precise_pool : pool_req ? *pool_req : [&] {
            const std::uint64_t t = std::min(avail / per_tok, static_cast<std::uint64_t>(8) * cap_max);
            return static_cast<std::uint32_t>(t / kv_page * kv_page);
        }();
        if (pool_tokens < 4096)
            throw app::UserError(app::exit_vram,
                                 std::format("not enough GPU memory left for the KV cache: a pool of {} tokens (at least 4096 needed).\n"
                                             "  Use a smaller model or quantization, give --ctx explicitly, use --balance (int8 KV, dense models),\n"
                                             "  or close other programs that use the GPU.",
                                             pool_tokens));
        if (slot_ctx_eff != slot_ctx) engine.setCtx(slot_ctx_eff);
        {
            const std::uint64_t floor = static_cast<std::uint64_t>(slot_ctx_eff) + (opt.parallel > 1 ? std::min(floor_second_ctx, slot_ctx_eff) : 0);
            if (pool_tokens < floor)
                logW("kv pool {} tokens < {} (one {}-token request + a second one reaching {}): concurrent long requests may end "
                     "early (finish_reason length){}",
                     pool_tokens, floor, slot_ctx_eff, opt.parallel > 1 ? std::min(floor_second_ctx, slot_ctx_eff) : 0,
                     kv_precise ? "; --balance allows int8 KV (a longer pool)" : "");
        }
        try {
            model.allocKvPool(pool_tokens);
        } catch (const std::exception& ex) {
            logE("cannot allocate a KV pool of {} tokens ({:.2f} GiB): {}", pool_tokens,
                 static_cast<double>(per_tok * pool_tokens) / (1024.0 * 1024.0 * 1024.0), ex.what());
            throw;
        }
        const double ck_mb = static_cast<double>(n_gdn * (conv_bytes + ssm_bytes) + static_cast<std::uint64_t>(V) * 4 +
                                                 static_cast<std::uint64_t>(cfg.n_embd) * 4) /
                             (1024.0 * 1024.0);
        if (no_pc) logI("prefix cache: disabled (WHIRL_NO_PREFIX_CACHE)");
        else logI("prefix cache: {} checkpoint(s) per slot x {:.1f} MiB ({} DeltaNet layers of recurrent state + logits row)", n_ck, ck_mb, n_gdn);
        {
            const double st_mib = static_cast<double>(n_gdn * (conv_bytes + ssm_bytes)) / (1024.0 * 1024.0);
            const double snap_mib = st_mib * model.snap_sets;
            logI("kv pool: {} tokens ({} pages of {}), {} KV{}, {:.1f} KiB/token, {:.2f} GiB, shared by {} slots; max {} tokens per "
                 "request",
                 model.pool_pages * kv_page, model.pool_pages, kv_page, model.kvName(), kv_note, static_cast<double>(per_tok) / 1024.0,
                 static_cast<double>(per_tok * model.pool_pages * kv_page) / (1024.0 * 1024.0 * 1024.0), opt.parallel,
                 std::min<std::uint64_t>(slot_ctx_eff, static_cast<std::uint64_t>(model.pool_pages) * kv_page));
            logI("slots: {}; per slot: recurrent state {:.1f} MiB, prefix checkpoints {:.1f} MiB; shared: {} snapshot sets {:.1f} MiB",
                 opt.parallel, st_mib, ck_mb * static_cast<double>(n_ck), model.snap_sets, snap_mib);
        }
        const hip::MemInfo mem2 = hip::memInfo();
        if (tier_pre)
            logI("kv tier: VRAM taken by the tier stream / pinned arena mapping: {:.1f} MiB (before the pool was sized)",
                 static_cast<double>(mem_ck1.free > mem_t1.free ? mem_ck1.free - mem_t1.free : 0) / 1048576.0);
        logI("VRAM use: weights + buffers {:.2f} GiB (prefill batch {}, embedding {}), slots {:.2f} GiB, other {:.2f} GiB, checkpoints "
             "{:.2f} GiB{}, KV pool {:.2f} GiB, left {:.2f} GiB",
             gib(mem0.free, mem.free), model.max_batch, model.embd_host ? "in pinned host memory" : "in VRAM",
             gib(mem_s0.free, mem_ck0.free), gib(mem.free, mem_s0.free), gib(mem_ck0.free, mem_ck1.free),
             eo.ckpt_host ? " (in pinned host memory)" : "", gib(mem_ck1.free, mem2.free),
             static_cast<double>(mem2.free) / (1024.0 * 1024.0 * 1024.0));
        logI("context {} tokens per request, pool {} tokens; VRAM free {:.2f} / {:.2f} GiB (after weights {:.2f}, at start {:.2f})",
             std::min<std::uint64_t>(slot_ctx_eff, static_cast<std::uint64_t>(model.pool_pages) * kv_page), model.pool_pages * kv_page, static_cast<double>(mem2.free) / (1024.0 * 1024.0 * 1024.0),
             static_cast<double>(mem2.total) / (1024.0 * 1024.0 * 1024.0), static_cast<double>(mem.free) / (1024.0 * 1024.0 * 1024.0),
             static_cast<double>(mem0.free) / (1024.0 * 1024.0 * 1024.0));
        if (use_mtp)
            logI("prompt-lookup (n-gram) drafting: {} (WHIRL_NGRAM{}), min match {}, up to {} drafts", eo.ngram ? "on" : "off",
                 eo.ngram ? "=0 turns it off" : "=1 turns it on", eo.ngram_min,
                 eo.ngram_max > 0 ? std::min(eo.ngram_max, qwen35::max_ng_drafts) : qwen35::max_ng_drafts);
        engine.setNumerics(numerics::propsJson(model.num_plan, model.kv_q8 ? (model.kv_q4 ? "q4" : model.kv_kf16 ? "q8v" : model.kv_rot ? "q8h" : "q8") : "f16"));
        engine.initPool();
        {
            // after everything sized at startup is allocated: any growth of the non-local segment
            // since the pool was sized is VRAM demoted to system memory (pinned host buffers made
            // before that point, such as the embedding table and the RAM tier, are in the baseline)
            const hip::WddmMemInfo w = hip::wddmMemInfo();
            if (w.ok && wddm0.ok) {
                const std::uint64_t grow = w.nonlocal_usage > wddm0.nonlocal_usage ? w.nonlocal_usage - wddm0.nonlocal_usage : 0;
                const bool over = w.local_usage > w.local_budget || grow > (256ull << 20);
                const std::string msg = std::format(
                    "WDDM memory: local {:.2f} / budget {:.2f} GiB; non-local (GPU 'Shared Usage') {:.0f} MiB, of which {:.0f} MiB was there "
                    "before the KV pool (declared pinned host memory: embedding {:.0f} MiB{}); growth {:.0f} MiB",
                    w.local_usage / 1073741824.0, w.local_budget / 1073741824.0, w.nonlocal_usage / 1048576.0,
                    wddm0.nonlocal_usage / 1048576.0, embd_pinned / 1048576.0, tier_pre ? ", RAM tier" : "", grow / 1048576.0);
                if (over)
                    logW("{} -- over the WDDM budget, part of the GPU buffers may live in system memory and decode can be several "
                         "times slower; set WHIRL_POOL_RESERVE_MB larger (e.g. 3072) or give --ctx",
                         msg);
                else
                    logI("{}", msg);
            }
        }
        if (eo.n_spe > 0)
            logI("shared prefix checkpoints: {} x {:.1f} MiB in VRAM; system messages >= {} tokens are prefilled as their own chunk "
                 "run and kept (WHIRL_SYS_MIN, 0 = no split), plus checkpoints at prefixes common to sessions ({})",
                 eo.n_spe, ck_mb, eo.sys_min, eo.lcp_on ? "on; WHIRL_SYS_LCP=0 turns it off" : "off");
        else
            logI("shared prefix checkpoints: off (system message split at >= {} tokens{})", eo.sys_min, eo.sys_min == 0 ? ": off" : "");

        // host prefix-cache tiers (pinned RAM, SSD)
        std::unique_ptr<tier::Tier> tier_owned;
        {
            std::uint64_t ssd_gb = kv_ssd_gb_default;
            if (opt.kv_ssd_gb) ssd_gb = *opt.kv_ssd_gb;
            else if (auto v = env("KV_SSD_GB")) {
                try {
                    ssd_gb = std::stoull(*v);
                } catch (const std::exception&) {
                }
            }
            std::uint64_t page_row = 0;
            for (const KvArr& a : backend->kvArrays()) page_row += a.row;
            if (tier_pre && page_row == per_tok) {
                std::optional<std::string> ssd_dir;
                if (ssd_gb > 0) {
                    if (opt.kv_ssd_dir) ssd_dir = opt.kv_ssd_dir;
                    else if (auto v = env("KV_SSD_DIR")) ssd_dir = *v;
                    else if (lad) ssd_dir = std::string(lad) + "\\whirl\\kvcache";
                }
                tier::Layout lay;
                lay.ck_bytes = n_gdn * (conv_bytes + ssm_bytes) + static_cast<std::uint64_t>(cfg.n_embd) * 4 + static_cast<std::uint64_t>(V) * 4;
                lay.page_bytes = per_tok * kv_page;
                lay.page_tokens = kv_page;
                lay.tok_cap = slot_ctx_eff;
                const std::uint64_t fp = tierFingerprint(model, opt.path, stats, use_mtp, n_gdn, lay);
                const TimePoint t_tier0 = Clock::now();
                tier::Config tc;
                tc.ram_bytes = kv_ram_mb << 20;
                tc.ssd_dir = ssd_dir;
                tc.ssd_cap = ssd_gb << 30;
                tc.min_tokens = envU32("KV_TIER_MIN", 2048);
                tc.ssd_delay_ms = envU32("KV_SSD_DELAY_MS", 2000);
                // the wake hook must be in place before the IO thread starts
                engine.attachTier(tier_pre.get());
                try {
                    tier_pre->start(tc, lay, fp);
                    tier_owned = std::move(tier_pre);
                } catch (const std::exception& ex) {
                    engine.attachTier(nullptr);
                    logE("kv tier: cannot set up the host tiers ({}); running without them", ex.what());
                }
                if (tier_owned) {
                    tier::Tier& tr = *tier_owned;
                    tier::declarePinned(lad, embd_pinned + tr.pinnedBytes());
                    logI("kv tier: RAM {:.2f} GiB of pinned host memory (counted as this process's GPU 'Shared Usage'; pinned in {:.1f} s, "
                         "SSD index {:.1f} s), entries >= {} tokens; SSD {}{} (cap {} GiB, {} entries / {:.2f} GiB indexed); entry: {:.1f} "
                         "MiB per checkpoint, {:.1f} MiB per 256-token page",
                         static_cast<double>(tr.pinnedBytes()) / 1073741824.0, tier_alloc_s, msSince(t_tier0) / 1000.0, tr.cfg.min_tokens,
                         ssd_dir ? *ssd_dir : std::string("off"), (ssd_dir && !tr.hasSsd()) ? " (unavailable)" : "", ssd_gb,
                         tr.ssdEntries(), static_cast<double>(tr.ssd_bytes) / 1073741824.0,
                         static_cast<double>(lay.ck_bytes) / 1048576.0, static_cast<double>(lay.page_bytes) / 1048576.0);
                }
            } else if (kv_ram_mb > 0 && tier_pre && page_row != per_tok) {
                logW("kv tier: KV layout not understood ({} vs {} bytes per token); host tiers off", page_row, per_tok);
            } else {
                logI("kv tier: off");
            }
        }

        // ---- vision encoder (after the KV pool: the pool size does not depend on it)
        std::unique_ptr<vision::Vision> vis_store;
        std::unique_ptr<ServerVision> vis_srv;
        if (const std::optional<std::string> mmp = opt.mmproj ? opt.mmproj : env("MMPROJ")) {
            const TimePoint tv0 = Clock::now();
            const hip::MemInfo mem_v0 = hip::memInfo();
            try {
                vis_store = vision::Vision::load(*mmp);
            } catch (const std::exception& ex) {
                logE("vision: cannot load mmproj {}: {}", *mmp, ex.what());
                throw;
            }
            vision::Vision& v = *vis_store;
            v.idle_s = opt.vis_idle_s;
            if (auto x = env("VIS_IDLE_S")) {
                try {
                    v.idle_s = static_cast<std::uint32_t>(std::stoul(*x));
                } catch (const std::exception&) {
                }
            }
            v.mode = *vision::parseMode(opt.vis_mode);
            if (auto x = env("VIS_MODE")) v.mode = vision::parseMode(*x).value_or(v.mode);
            VisionConfig vc;
            {
                const TokenId ve = tok.find("<|vision_end|>");
                vc.vision_end_id = ve < 0 ? 0 : static_cast<std::uint32_t>(ve);
            }
            if (auto x = env("VIS_CKPT_MIN")) {
                try {
                    vc.ck_min = static_cast<std::uint32_t>(std::stoul(*x));
                } catch (const std::exception&) {
                }
            }
            const TokenId pad = tok.find("<|image_pad|>");
            if (pad < 0) {
                logE("vision: the model's vocabulary has no <|image_pad|> token");
                throw std::runtime_error("NoImagePadToken");
            }
            vc.image_pad_id = static_cast<std::uint32_t>(pad);
            if (!backend->hasMrope()) {
                logE("vision: this GPU's kernels have no multi-section RoPE (attn_prep_m)");
                throw std::runtime_error("MropeUnsupported");
            }
            std::uint64_t cache_mb = opt.vis_cache_mb;
            if (auto x = env("VIS_CACHE_MB")) {
                try {
                    cache_mb = std::stoull(*x);
                } catch (const std::exception&) {
                }
            }
            vc.cache_bytes = cache_mb << 20;
            vc.allow_files = opt.allow_local_images;
            if (auto x = env("VIS_DUMP")) vc.dump_prefix = *x;
            vis_srv = makeQwen35Vision(v, model);
            vc.vis = vis_srv.get();
            std::array<std::array<std::uint64_t, 2>, 16> raw{};
            const std::size_t n_lend = backend->lendScratch(raw);
            std::uint64_t lent = 0;
            for (std::size_t k = 0; k < n_lend; ++k) lent += raw[k][1];
            tier::declarePinned(lad, embd_pinned + (tier_owned ? tier_owned->pinnedBytes() : 0) + v.pinnedBytes());
            const hip::MemInfo mem_v1 = hip::memInfo();
            engine.attachVision(vc);
            logI("vision: {}: {} layers, {} pinned MiB in host RAM ({:.1f} s; VRAM change {:.1f} MiB); encoder loaded on demand, released "
                 "after {} s idle; weights {}; activations in the prefill scratch ({} buffers, {:.0f} MiB; one 4096-token image needs {:.0f} "
                 "MiB); embedding cache {} MiB; local image files {}",
                 *mmp, v.hp.n_layer, v.pinnedBytes() >> 20, msSince(tv0) / 1000.0,
                 (static_cast<double>(mem_v0.free) - static_cast<double>(mem_v1.free)) / 1048576.0, v.idle_s, vision::modeName(v.mode), n_lend,
                 static_cast<double>(lent) / 1048576.0, static_cast<double>(v.arenaBytes(v.hp.max_tokens)) / 1048576.0, cache_mb,
                 vc.allow_files ? "allowed" : "off");
        }

        logI("model ready in {:.1f} s (weights {:.1f} s)", msSince(t_load0) / 1000.0, stats.ms / 1000.0);
        const bool timer_probe = env("TIMER_PROBE").has_value();
        const double probe_1ms = timer_probe ? sleepMs(std::chrono::milliseconds(1), 50) : 0.0;
        const double probe_200us = timer_probe ? sleepMs(std::chrono::microseconds(200), 50) : 0.0;
        const TimerRes timer_res;
        if (!timer_res.on) logW("timer: 1 ms resolution not available; short waits of the main loop take one system tick");
        if (timer_probe)
            logI("timer probe | sleep 1 ms: {:.2f} ms before, {:.2f} ms after timeBeginPeriod(1) | sleep 200 us: {:.2f} / {:.2f} ms",
                 probe_1ms, sleepMs(std::chrono::milliseconds(1), 50), probe_200us, sleepMs(std::chrono::microseconds(200), 50));
        HttpOptions ho;
        ho.cors_origins = opt.cors_origins;
        HttpServer http(engine, ho);
        http.start(opt.host, opt.port);
        logI("server listening on http://{}:{} (model id \"{}\", {} slots); endpoints: GET /health, GET /v1/models, POST "
             "/v1/chat/completions, POST /v1/completions, GET /props, GET /version",
             opt.host, http.port(), name, opt.parallel);
        // main thread: continuous batching over the slots (until a console control
        // event stops the engine: graceful shutdown, see onConsoleCtrl)
        g_stopped = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        g_engine.store(&engine);
        SetConsoleCtrlHandler(nullptr, FALSE);  // an "ignore Ctrl+C" attribute inherited from the parent would hide Ctrl+C
        SetConsoleCtrlHandler(onConsoleCtrl, TRUE);
        engine.runLoop();
        http.stop();
        logI("server stopped");
        g_engine.store(nullptr);
        if (g_stopped != nullptr) SetEvent(g_stopped);
        return 0;
    } catch (const std::exception& ex) {
        logE("fatal: {}", ex.what());
        int code = app::exit_error;
        const std::string msg = app::explainText(ex, &code);
        std::fprintf(stderr, "%s: error: %s", g_program.c_str(), msg.c_str());
        std::fflush(stderr);
        return code;
    }
}

}  // namespace whirl::server
