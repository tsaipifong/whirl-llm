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
#include "whirl/common.h"
#include "whirl/gguf.h"
#include "whirl/hip.h"
#include "whirl/model.h"
#include "whirl/tokenizer.h"
#include "release/release.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
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

constexpr std::uint32_t ctx_per_slot_default = 131072;
constexpr std::uint32_t floor_second_ctx = 65536;
constexpr std::uint32_t serve_prefill_batch = 4096;
constexpr std::size_t n_ckpt_default = 4;
constexpr std::size_t n_ckpt_default_par = 2;
constexpr std::size_t n_spe_default = 2;
constexpr std::uint32_t sys_min_default = 2048;
constexpr std::uint64_t kv_ram_mb_default = 8192;
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
    "  -np, --parallel N        concurrent request slots (continuous batching, 1..16; default 4)\n"
    "  -c, --ctx N              shared KV pool in tokens; slots take pages on demand and idle slots'\n"
    "                           prefix caches are evicted (LRU) when it is full (default: all VRAM\n"
    "                           left over minus 768 MiB (MoE 1.5 GiB); 262144 on the Radeon 8060S)\n"
    "  --ctx-per-slot N         longest context of one request (default min(pool, 131072); up to 262144)\n"
    "  --mtp-drafts N           MTP drafts per cycle, 1..10 (fixed count; default: per model type)\n"
    "  --decode-min-tps N       while other requests prefill, keep every streaming (decoding) request\n"
    "                           at >= N tok/s by limiting prefill forwards (default 20; 0 = off:\n"
    "                           prefill forwards are not limited)\n"
    "  --kv-ram-mb N            host RAM tier of the prefix cache in MiB of pinned memory (default 8192,\n"
    "                           or one full-length session if more; 0 = no host tiers). Idle sessions\n"
    "                           are copied there and restored instead of prefilled again\n"
    "  --kv-ssd-dir PATH        SSD tier directory (default %LOCALAPPDATA%\\whirl\\kvcache)\n"
    "  --kv-ssd-gb N            SSD tier size cap in GiB (default 64; 0 = no SSD tier)\n"
    "  --mmproj FILE            vision encoder (Qwen3-VL style mmproj GGUF, F16 / BF16): image_url parts\n"
    "                           (data: URLs with base64 PNG / JPEG / ...) become image tokens; weights stay\n"
    "                           in pinned host RAM, nothing in VRAM until an image arrives\n"
    "  --vis-idle-s N           release the vision encoder after N s without images (default 60)\n"
    "  --vis-mode M             auto (default), resident (weights in VRAM), stream (layer by layer)\n"
    "  --vis-cache-mb N         host cache of image embeddings by content hash, MiB (default 1024)\n"
    "  --allow-local-images     also accept local file paths / file:// URLs as image sources\n"
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

bool envOn(std::string_view name, bool def) {
    if (auto v = env(name)) return *v != "0";
    return def;
}

// Default max drafts per cycle by number of decoding slots (index = slots).
std::array<std::uint32_t, gdn_max_seg + 1> defaultBatchDrafts(bool moe) {
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
                                         "WHIRL_TIER_VERIFY", "WHIRL_TIER_MIN_GAIN", "WHIRL_DECODE_MIN_TPS"};
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
        if (auto v = argValue(args, i, "--host")) opt.host = *v;
        else if (auto v2 = argValue(args, i, "--port")) opt.port = parseNum<std::uint16_t>(*v2, "--port");
        else if (auto v3 = argValue(args, i, "--ctx")) opt.ctx = parseNum<std::uint32_t>(*v3, "--ctx");
        else if (auto v3b = argValue(args, i, "-c")) opt.ctx = parseNum<std::uint32_t>(*v3b, "--ctx");
        else if (auto v4 = argValue(args, i, "--parallel")) opt.parallel = parseNum<std::uint32_t>(*v4, "--parallel");
        else if (auto v4b = argValue(args, i, "-np")) opt.parallel = parseNum<std::uint32_t>(*v4b, "--parallel");
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
        else if (!arg.empty() && arg[0] == '-') die("unknown option " + arg);
        else if (opt.path.empty()) opt.path = arg;
        else die("unexpected argument " + arg);
    }
    if (opt.path.empty()) die("missing MODEL.gguf");
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
        const bool is_uma = hip::archFor(info.gcn_arch) == hip::Arch::gfx1151;
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
        logI("device {}: {} ({}), {:.1f} GiB VRAM", dev, info.name, info.gcn_arch,
             static_cast<double>(info.total_mem) / (1024.0 * 1024.0 * 1024.0));

        const Tokenizer tok = Tokenizer::fromGguf(f);
        const std::string_view tmpl_text = f.getStringOr("tokenizer.chat_template", "");
        const chat::TemplateKind tmpl = chat::detectTemplate(tmpl_text);
        logI("chat template: GGUF tokenizer.chat_template ({} B) -> built-in Qwen3.x renderer variant '{}' ({})", tmpl_text.size(),
             chat::templateName(tmpl),
             tmpl == chat::TemplateKind::a ? "reasoning effort, merged system, preserve_thinking" : "system + tools, <think> in history");

        const hip::MemInfo mem0 = hip::memInfo();
        qwen35::LoadStats stats;
        qwen35::LoadOptions lo;
        lo.max_batch = serve_prefill_batch;
        lo.embd_on_host = envOn("EMBD_HOST", false);
        if (auto v = env("PREFILL_BATCH")) {
            try {
                lo.max_batch = std::clamp(static_cast<std::uint32_t>(std::stoul(*v)), 1u, qwen35::max_batch_limit);
            } catch (const std::exception&) {
            }
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

        // same knobs as the CLI
        if (env("NO_FUSE")) model.no_fuse = true;
        if (env("FLOAT_GEMV")) model.float_gemv = true;
        if (env("NAIVE_ATTN")) model.naive_attn = true;
        if (env("GDN_SEQ")) model.gdn_chunked = false;
        if (auto v = env("MOE_BN")) model.moe_bn_force = static_cast<std::uint32_t>(std::strtoul(v->c_str(), nullptr, 10));
        if (auto v = env("PREFILL_BATCH"))
            model.max_batch = std::max(1u, std::min(model.max_batch, static_cast<std::uint32_t>(std::stoul(*v))));
        if (auto v = env("DRAFT_VOCAB")) model.draft_vocab = static_cast<std::uint32_t>(std::stoul(*v));
        if (env("GEMV_R") || env("GEMV_W") || env("GEMV_WH"))
            logW("WHIRL_GEMV_R / WHIRL_GEMV_W / WHIRL_GEMV_WH are not supported by whirl-server (ignored)");
        model.use_graph = false;
        const bool use_mtp = model.mtp.has_value() && envOn("MTP", true);
        const qwen35::MtpDefaults def = qwen35::mtpDefaults(model.cfg.moe);
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
        auto batch_drafts = defaultBatchDrafts(model.cfg.moe);
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
        // MTP block as Q4_K (drafts only; WHIRL_MTP_Q4=0 keeps Q6_K)
        if (use_mtp && envOn("MTP_Q4", true)) model.requantMtpQ4();
        if (use_mtp && !env("MTP_FULLHEAD")) {
            const bool q4 = env("DRAFT_HEAD").value_or("") == "q4";
            model.buildDraftHeadEx(q4 ? qwen35::Model::DraftHeadKind::q4 : qwen35::Model::DraftHeadKind::d2);
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
        const bool no_pc = envOn("NO_PREFIX_CACHE", false);
        const std::size_t ck_def = opt.parallel == 1 ? n_ckpt_default : (opt.parallel <= 4 ? n_ckpt_default_par : 1);
        const std::size_t n_ck = no_pc ? 0 : envU32("SERVE_CKPTS", static_cast<std::uint32_t>(ck_def));
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
        eo.ngram_min = envU32("NGRAM_MIN", 3);
        eo.ngram_max = envU32("NGRAM_MAX", 0);
        eo.ngram_force = envOn("NGRAM_FORCE", false);
        eo.slot_drafts = envOn("SLOT_DRAFTS", false);
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
        eo.n_spe = (n_ck == 0 || no_pc) ? 0 : envU32("SYS_CKPTS", static_cast<std::uint32_t>(n_spe_default));
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
            return std::max(kv_ram_mb_default, alignUp(full, 1ull << 30) >> 20);
        }();
        std::uint64_t kv_ram_mb = ram_def;
        if (opt.kv_ram_mb) kv_ram_mb = *opt.kv_ram_mb;
        else if (auto v = env("KV_RAM_MB")) {
            try {
                kv_ram_mb = std::stoull(*v);
            } catch (const std::exception&) {
            }
        }
        std::unique_ptr<tier::Tier> tier_pre;
        if (kv_ram_mb > 0 && n_ck > 0 && !no_pc && hip::archFor(info.gcn_arch) == hip::Arch::gfx1201) {
            try {
                tier_pre = tier::Tier::create(*ops, kv_ram_mb << 20, seed0 ^ 0x5bd1e995ull, dev);
            } catch (const std::exception& ex) {
                logE("kv tier: cannot allocate {} MiB of pinned host memory ({}); running without host tiers", kv_ram_mb, ex.what());
            }
        }
        const hip::MemInfo mem_t1 = hip::memInfo();
        // shared KV pool: --ctx, or (R9700) the VRAM left now minus a reserve
        const std::uint64_t reserve = static_cast<std::uint64_t>(envU32("POOL_RESERVE_MB", cfg.moe ? 1536 : 768)) << 20;
        const std::uint64_t avail = [&] {
            const hip::MemInfo mf = hip::memInfo();
            return mf.free > reserve ? mf.free - reserve : 0;
        }();
        std::string kv_note;
        if (model.kvAutoDense()) {
            const std::uint32_t second = opt.parallel > 1 ? std::min(floor_second_ctx, slot_ctx) : 0;
            const std::uint64_t need = pool_req ? *pool_req : static_cast<std::uint64_t>(slot_ctx) + second + (opt.parallel > 1 ? 2ull : 1ull) * kv_page;
            if (need * model.kvBytesPerTokenFmt(false, false) <= avail) {
                kv_note = " (auto: the floor pool fits as f16)";
            } else if (need * model.kvBytesPerTokenFmt(true, true) <= avail) {
                model.setKvFormat(true, false, true);
                kv_note = " (auto: the floor pool (one full request + a 64k second one) does not fit as f16, fits as q8v)";
            } else {
                model.setKvFormat(true, true, false);
                kv_note = " (auto: the floor pool (one full request + a 64k second one) does not fit as f16 or q8v)";
            }
        }
        const std::uint64_t per_tok = model.kvBytesPerToken();
        const std::uint32_t pool_tokens = pool_req ? *pool_req : [&] {
            const std::uint64_t t = std::min(avail / per_tok, static_cast<std::uint64_t>(8) * cap_max);
            return static_cast<std::uint32_t>(t / kv_page * kv_page);
        }();
        if (pool_tokens < 4096)
            throw app::UserError(app::exit_vram,
                                 std::format("not enough GPU memory left for the KV cache: a pool of {} tokens (at least 4096 needed).\n"
                                             "  Use a smaller model or quantization, give --ctx explicitly, set WHIRL_KV=q8v (dense models),\n"
                                             "  or close other programs that use the GPU.",
                                             pool_tokens));
        {
            const std::uint64_t floor = static_cast<std::uint64_t>(slot_ctx) + (opt.parallel > 1 ? std::min(floor_second_ctx, slot_ctx) : 0);
            if (pool_tokens < floor)
                logW("kv pool {} tokens < {} (one {}-token request + a second one reaching {}): concurrent long requests may end "
                     "early (finish_reason length)",
                     pool_tokens, floor, slot_ctx, opt.parallel > 1 ? std::min(floor_second_ctx, slot_ctx) : 0);
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
                 static_cast<double>(per_tok * model.pool_pages * kv_page) / (1024.0 * 1024.0 * 1024.0), opt.parallel, slot_ctx);
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
             slot_ctx, model.pool_pages * kv_page, static_cast<double>(mem2.free) / (1024.0 * 1024.0 * 1024.0),
             static_cast<double>(mem2.total) / (1024.0 * 1024.0 * 1024.0), static_cast<double>(mem.free) / (1024.0 * 1024.0 * 1024.0),
             static_cast<double>(mem0.free) / (1024.0 * 1024.0 * 1024.0));
        if (use_mtp)
            logI("prompt-lookup (n-gram) drafting: {} (WHIRL_NGRAM{}), min match {}, up to {} drafts", eo.ngram ? "on" : "off",
                 eo.ngram ? "=0 turns it off" : "=1 turns it on", eo.ngram_min,
                 eo.ngram_max > 0 ? std::min(eo.ngram_max, qwen35::max_ng_drafts) : qwen35::max_ng_drafts);
        engine.initPool();
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
                lay.tok_cap = slot_ctx;
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
                    tier::declarePinned(lad, tr.pinnedBytes());
                    logI("kv tier: RAM {:.2f} GiB of pinned host memory (counted as this process's GPU 'Shared Usage', SSD index {:.1f} "
                         "s), entries >= {} tokens; SSD {}{} (cap {} GiB, {} entries / {:.2f} GiB indexed); entry: {:.1f} MiB per "
                         "checkpoint, {:.1f} MiB per 256-token page",
                         static_cast<double>(tr.pinnedBytes()) / 1073741824.0, msSince(t_tier0) / 1000.0, tr.cfg.min_tokens,
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
            tier::declarePinned(lad, (tier_owned ? tier_owned->pinnedBytes() : 0) + v.pinnedBytes());
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
        HttpServer http(engine);
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
