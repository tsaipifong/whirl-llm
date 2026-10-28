// whirl-server's command line (arch025-C1a): moved verbatim from server_main.cpp (serveMain's argument loop, Options,
// parseNum, argValue, the --help option list); die(msg) became throw ArgError(msg), which serveMain turns back into
// die(msg). See server_args.h.
// SPDX-License-Identifier: Apache-2.0

#include "server_args.h"

#include "whirl/model_types.h"  // qwen35::max_drafts, qwen35::gdn_max_seg

#include <algorithm>
#include <cctype>
#include <exception>
#include <limits>

namespace whirl::server {

const char* const kHelpBody =
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
    "  --device D, -dev D       GPU: pci:BUS:DEVICE (Task Manager > Performance > GPU > Physical location: PCI bus 71,\n"
    "                           device 0 -> pci:71:0), an index from --list-devices, r9700, 8060s, or a part of the name.\n"
    "                           Default: the GPU with the most GPU memory\n"
    "  --list-devices           list the GPUs (index, name, memory, pci:BUS:DEVICE) and exit\n"
    "  -np, --parallel N        concurrent request slots (continuous batching, 1..16; default 4, 1 on cards < 20 GiB)\n"
    "  -c, --ctx N              shared KV pool in tokens; slots take pages on demand and idle slots'\n"
    "                           prefix caches are evicted (LRU) when it is full (default: all VRAM\n"
    "                           left over, kept 768 MiB (MoE 1.5 GiB) under the free VRAM and a margin\n"
    "                           under the Windows (WDDM) budget: other programs' VRAM the budget does\n"
    "                           not show yet, plus room for growth, the larger of 2 GiB (3 GiB with a\n"
    "                           monitor on the card) and 6% of the budget; WHIRL_POOL_BUDGET_MARGIN_MB=N\n"
    "                           sets that room. Settings that do not fit are lowered ('adjusted:');\n"
    "                           WHIRL_POOL_RESERVE_MB=N: free VRAM minus N only, no WDDM budget;\n"
    "                           262144 on the Radeon 8060S)\n"
    "  --ctx-per-slot N         longest context of one request (default min(pool, 131072); up to 262144)\n"
    "  --precise | --balance | --fast\n"
    "                           numerics mode (default balance: dense fp8 MXFP4 prefill, q8h KV, int8\n"
    "                           decode activations; MoE f16 KV; --precise: f16/f32 activations,\n"
    "                           f16 KV; --fast: balance + 4-bit KV, relaxed MTP acceptance).\n"
    "                           --balance=fp8,kvq8 picks items; --mode M;\n"
    "                           WHIRL_MODE\n"
    "  --mtp-drafts N           MTP drafts per cycle, 1..10 (fixed count; default: per model type)\n"
    "  --draft mtp|dflash       draft source of the speculative decode (default mtp). dflash: the DFlash2\n"
    "                           drafter GGUF from --draft-model FILE (block 8: up to 7 drafts per cycle);\n"
    "                           the MTP block is then not loaded; the drafter's weights and context rings\n"
    "                           come out of the KV pool (about 2 GiB net of the MTP block).\n"
    "                           A model without a matching drafter (e.g. a MoE model) keeps mtp.\n"
    "                           n-gram drafts stay on in both modes; outputs equal plain greedy decoding\n"
    "  --draft-model FILE       DFlash2 drafter GGUF for --draft dflash (wins over env WHIRL_DRAFT_MODEL)\n"
    "  --decode-min-tps N       while other requests prefill, keep every streaming (decoding) request\n"
    "                           at >= N tok/s by limiting prefill forwards (default 20; 0 = off:\n"
    "                           prefill forwards are not limited)\n"
    "  --kv-ram-mb N            host RAM tier of the prefix cache in MiB of pinned memory (default: 1/4\n"
    "                           of physical RAM, at least 8 GiB or one full-length session, at most\n"
    "                           32 GiB, and at most half of the RAM available at startup; 0 = no host\n"
    "                           tiers; Radeon 8060S: OS-visible RAM / 8, at least one max-context entry,\n"
    "                           at most 16 GiB and 1/4 of the RAM available at startup, off under 32 GB\n"
    "                           of OS-visible RAM; the BIOS VRAM carve-out does not count).\n"
    "                           Idle sessions are copied there and restored instead of prefilled again\n"
    "  --kv-ssd-dir PATH        SSD tier directory (default %LOCALAPPDATA%\\whirl\\kvcache)\n"
    "  --kv-ssd-gb N            SSD tier size cap in GiB (default 64; 0 = no SSD tier)\n"
    "  --kv-ssd-mode M          shutdown (default: only reused entries are written to the SSD -- when\n"
    "                           WHIRL is closed (Ctrl+C up to 60 s; console close / logoff / OS shutdown\n"
    "                           only about 5 s), when RAM must evict them, and after about 5 min idle;\n"
    "                           a one-shot prompt is never written),\n"
    "                           always (idle sessions go to the SSD after --kv-ssd-delay), off (no SSD files)\n"
    "  --kv-ssd-delay SEC       --kv-ssd-mode always: write a session once it has been unchanged for SEC\n"
    "                           seconds (default 300; used again sooner: served from RAM, never written)\n"
    "  --mmproj FILE            vision encoder (Qwen3-VL style mmproj GGUF, F16 / BF16): image_url parts\n"
    "                           (data: URLs with base64 PNG / JPEG / ...) become image tokens; weights stay\n"
    "                           in pinned host RAM, nothing in VRAM until an image arrives\n"
    "  --vis-idle-s N           release the vision encoder after N s without images (default 60)\n"
    "  --vis-mode M             auto (default), resident (weights in VRAM), stream (layer by layer)\n"
    "  --vis-cache-mb N         host cache of image embeddings by content hash, MiB (default 1024)\n"
    "  --allow-local-images     also accept local file paths / file:// URLs as image sources\n"
    "  --cors-origin ORIGIN     web pages from ORIGIN (e.g. https://app.example.com) may call the server\n"
    "                           from a browser (CORS); repeatable; * = any page (the behaviour before\n"
    "                           v0.1.4). Default: only pages on http(s)://localhost, 127.0.0.1, [::1].\n"
    "                           A POST from any other page is refused (403)\n"
    "  --allow-host NAME        also answer requests whose Host header is NAME (repeatable; * = any).\n"
    "                           Default: localhost, 127.0.0.1, [::1] and the --host address (any IP\n"
    "                           address with --host 0.0.0.0); other names get 403 (DNS rebinding guard)\n"
    "  --log-file PATH          log file (default %LOCALAPPDATA%\\whirl\\server.log; the log also goes to\n"
    "                           the console)\n"
    "  -h, --help               this text\n"
    "  -V, --version            show the version\n";

namespace {

template <class T>
T parseNum(const std::string& v, const char* what) {
    try {
        std::size_t used = 0;
        const unsigned long long x = std::stoull(v, &used, 10);
        if (used != v.size() || x > static_cast<unsigned long long>(std::numeric_limits<T>::max())) throw std::out_of_range("");
        return static_cast<T>(x);
    } catch (const std::exception&) {
        throw ArgError(std::string("invalid ") + what + " " + v);
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

}  // namespace

Options parseServerArgs(const std::vector<std::string>& args, VisModeOk vis_mode_ok) {
    Options opt;
    bool device_given = false;  // --device / -dev seen
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string arg = args[i];
        std::string mspec;
        bool mnext = false;
        bool is_mode = false;
        try {
            is_mode = numerics::modeFlag(arg, &mspec, &mnext);
        } catch (const std::exception& ex) {
            throw ArgError(ex.what());
        }
        if (is_mode) {
            if (mnext) {
                if (i + 1 >= args.size()) throw ArgError("missing value for --mode");
                mspec = args[++i];
            }
            if (opt.mode_spec) throw ArgError("give one numerics mode (--precise, --balance, --fast or --mode M)");
            try {
                (void)numerics::parseMode(mspec, "command line");
            } catch (const std::exception& ex) {
                throw ArgError(ex.what());
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
        else if (auto v5d = argValue(args, i, "--draft-model")) opt.draft_model = *v5d;
        else if (auto v5c = argValue(args, i, "--draft")) {
            if (*v5c != "mtp" && *v5c != "dflash") throw ArgError("invalid --draft " + *v5c + " (mtp or dflash)");
            opt.draft = *v5c;
        } else if (auto v5b = argValue(args, i, "--decode-min-tps"))
            opt.decode_min_tps = parseNum<std::uint32_t>(*v5b, "--decode-min-tps");
        else if (auto v6 = argValue(args, i, "--ctx-per-slot")) opt.ctx_per_slot = parseNum<std::uint32_t>(*v6, "--ctx-per-slot");
        else if (auto v7 = argValue(args, i, "--log-file")) opt.log_file = *v7;
        else if (auto v8 = argValue(args, i, "--alias")) opt.alias = *v8;
        else if (auto v9 = argValue(args, i, "--kv-ram-mb")) opt.kv_ram_mb = parseNum<std::uint64_t>(*v9, "--kv-ram-mb");
        else if (auto v10 = argValue(args, i, "--kv-ssd-dir")) {
            // bug-emptyrootpaths: "" + "\\whirl.lock" was the drive root (as llama.cpp --slot-save-path: refused)
            if (v10->empty()) throw ArgError("--kv-ssd-dir is empty (give a folder; to turn the SSD tier off use --kv-ssd-gb 0 or --kv-ssd-mode off)");
            opt.kv_ssd_dir = *v10;
        }
        else if (auto v11 = argValue(args, i, "--kv-ssd-gb")) opt.kv_ssd_gb = parseNum<std::uint64_t>(*v11, "--kv-ssd-gb");
        else if (auto vd = argValue(args, i, "--kv-ssd-delay")) opt.kv_ssd_delay = parseNum<std::uint64_t>(*vd, "--kv-ssd-delay");
        else if (auto vm = argValue(args, i, "--kv-ssd-mode")) {
            if (*vm != "always" && *vm != "shutdown" && *vm != "off")
                throw ArgError("invalid --kv-ssd-mode " + *vm + " (expected always, shutdown or off)");
            opt.kv_ssd_mode = *vm;
        }
        else if (auto v12 = argValue(args, i, "--device")) {
            if (device_given) throw ArgError("give --device only once");  // same rule as whirl
            device_given = true;
            opt.device = *v12;
        } else if (auto v12b = argValue(args, i, "-dev")) {  // llama.cpp's short name
            if (device_given) throw ArgError("give --device only once");
            device_given = true;
            opt.device = *v12b;
        }
        else if (auto v13 = argValue(args, i, "--mmproj")) opt.mmproj = *v13;
        else if (auto v14 = argValue(args, i, "--vis-idle-s")) opt.vis_idle_s = parseNum<std::uint32_t>(*v14, "--vis-idle-s");
        else if (auto v15 = argValue(args, i, "--vis-mode")) {
            if (!vis_mode_ok(*v15)) throw ArgError("invalid --vis-mode " + *v15);
            opt.vis_mode = *v15;
        } else if (auto v16 = argValue(args, i, "--vis-cache-mb")) opt.vis_cache_mb = parseNum<std::uint64_t>(*v16, "--vis-cache-mb");
        else if (arg == "--allow-local-images") opt.allow_local_images = true;
        else if (auto v17 = argValue(args, i, "--cors-origin")) {
            if (*v17 != "*" && v17->find("://") == std::string::npos)
                throw ArgError("invalid --cors-origin " + *v17 + " (give an origin such as https://app.example.com, or *)");
            opt.cors_origins.push_back(*v17);
        }
        else if (auto v18 = argValue(args, i, "--allow-host")) {
            if (v18->empty() || v18->find_first_of(" /\\") != std::string::npos) throw ArgError("invalid --allow-host " + *v18 + " (give a host name such as myhost.lan, or *)");
            opt.allowed_hosts.push_back(*v18);
        }
        else if (!arg.empty() && arg[0] == '-') throw ArgError("unknown option " + arg);
        else if (opt.path.empty()) opt.path = arg;
        else throw ArgError("unexpected argument " + arg);
    }
    if (opt.path.empty()) throw ArgError("missing MODEL.gguf");
    return opt;
}

ResolvedConfig resolveServerConfig(Options opt, std::optional<std::string> mode_env) {
    numerics::Request nreq;
    try {
        nreq = numerics::resolve(opt.mode_spec, mode_env);
    } catch (const std::exception& ex) {
        throw ArgError(ex.what());
    }
    if (opt.parallel < 1 || opt.parallel > qwen35::gdn_max_seg) throw ArgError("--parallel must be 1..16");
    return ResolvedConfig{std::move(opt), std::move(nreq)};
}

void checkKvSsdDirEnv(const Options& opt, const std::optional<std::string>& kv_ssd_dir_env) {
    if (!opt.kv_ssd_dir && kv_ssd_dir_env && kv_ssd_dir_env->empty())
        throw ArgError("WHIRL_KV_SSD_DIR is empty (give a folder or remove the variable; to turn the SSD tier off use --kv-ssd-gb 0 or --kv-ssd-mode off)");
}


std::string servedModelName(const std::string& file_name, const std::optional<std::string>& alias) {
    std::string name = alias ? *alias : file_name;
    if (!alias && name.size() > 5) {
        std::string tail = name.substr(name.size() - 5);
        for (char& c : tail) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (tail == ".gguf") name = name.substr(0, name.size() - 5);
    }
    return name;
}

std::string kvSsdMode(const std::optional<std::string>& cli, const std::optional<std::string>& env) {
    std::string ssd_mode = cli.value_or(env.value_or("shutdown"));
    if (ssd_mode != "always" && ssd_mode != "off") ssd_mode = "shutdown";
    return ssd_mode;
}

}  // namespace whirl::server
