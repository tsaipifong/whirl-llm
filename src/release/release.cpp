// Release front-end shared by whirl.exe and whirl-server.exe (see release.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL.

#include "release/release.h"

#include "whirl/common.h"
#include "whirl/gguf.h"
#include "whirl/hip.h"
#include "whirl/model.h"
#include "whirl/version.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
// after windows.h
#include <delayimp.h>

namespace whirl::app {

namespace {

constexpr const char* k_driver_advice =
    "  Install AMD Software: Adrenalin Edition 26.8.1 or newer (https://www.amd.com/en/support),\n"
    "  restart Windows, and run the command again. Only the graphics driver is needed to run\n"
    "  WHIRL; the AMD HIP SDK is needed only to build it from source.\n";

constexpr const char* k_gpu_runtime_dll = "amdhip64_7.dll";

UINT g_saved_cp = 0;

void restoreConsole() {
    if (g_saved_cp != 0) SetConsoleOutputCP(g_saved_cp);
}

bool isConsole(DWORD std_handle) {
    HANDLE h = GetStdHandle(std_handle);
    DWORD mode = 0;
    return h != nullptr && h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode) != 0;
}

}  // namespace

// A HIP function the driver's runtime does not export (delay-load failure):
// the driver predates something WHIRL uses.
FARPROC WINAPI delayLoadFailureHook(unsigned notify, PDelayLoadInfo info) {
    if (notify == dliFailLoadLib || notify == dliFailGetProc) {
        const char* dll = info && info->szDll ? info->szDll : "?";
        if (notify == dliFailLoadLib) {
            std::fprintf(stderr, "error: the AMD GPU runtime (%s) could not be loaded.\n%s", dll, k_driver_advice);
        } else {
            char fn[128] = "?";
            if (info && info->dlp.fImportByName && info->dlp.szProcName) std::snprintf(fn, sizeof fn, "%s", info->dlp.szProcName);
            std::fprintf(stderr,
                         "error: the installed AMD graphics driver is too old: its GPU runtime (%s) has no %s.\n%s", dll, fn,
                         k_driver_advice);
        }
        std::fflush(stderr);
        ExitProcess(static_cast<UINT>(exit_gpu));
    }
    return nullptr;
}

namespace {

std::string supportedArchList() {
    std::string s;
    for (const hip::EmbeddedObject& o : hip::embeddedObjects()) {
        if (!s.empty()) s += ", ";
        s += o.arch;
    }
    return s;
}

std::string deviceListText() {
    std::string s;
    try {
        for (const hip::DeviceInfo& d : hip::listDevices())
            s += std::format("    device {}: {} ({}), {:.1f} GiB\n", d.index, d.name, d.gcn_arch,
                             static_cast<double>(d.total_mem) / (1024.0 * 1024.0 * 1024.0));
    } catch (const std::exception&) {
    }
    return s.empty() ? std::string("    (the driver reports no AMD GPU)\n") : s;
}

constexpr const char* k_vram_advice =
    "  Try a smaller context: whirl chat --ctx 8192 (or less); whirl-server --ctx N, a smaller\n"
    "  --ctx-per-slot, or fewer --parallel slots. Dense models can keep the KV cache in 8 bits:\n"
    "  set WHIRL_KV=q8v (or q8h; Radeon 8060S: q8). Close other programs that use the GPU (games,\n"
    "  browsers with hardware acceleration, other AI tools) and check the model fits in the GPU's\n"
    "  memory.\n";

}  // namespace

const char* version() { return WHIRL_VERSION_STRING; }

std::string versionText(std::string_view program) {
    return std::format(
        "{} {} (WHIRL - Windows HIP Inference for RDNA LLMs)\n"
        "GPU kernels for: {}\n"
        "Runs with the AMD graphics driver (AMD Software: Adrenalin Edition 26.8.1 or newer).\n"
        "License: Apache-2.0\n",
        program, WHIRL_VERSION_STRING, WHIRL_GPU_ARCHS_STRING);
}

void initConsole() {
    if (g_saved_cp != 0) return;
    if (!isConsole(STD_OUTPUT_HANDLE) && !isConsole(STD_ERROR_HANDLE)) return;
    const UINT cp = GetConsoleOutputCP();
    if (cp == CP_UTF8 || cp == 0) return;
    if (SetConsoleOutputCP(CP_UTF8)) {
        g_saved_cp = cp;
        std::atexit(restoreConsole);
    }
}

int gpuPreflight() {
    // the same search order the executable's import of the DLL uses
    const UINT old_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);  // no system dialog
    HMODULE h = LoadLibraryW(L"amdhip64_7.dll");
    const DWORD err = h == nullptr ? GetLastError() : 0;
    SetErrorMode(old_mode);
    if (h == nullptr) {
        std::fprintf(stderr,
                     "error: the AMD GPU runtime (%s) %s, so WHIRL cannot use the GPU.\n"
                     "  It is installed by the AMD graphics driver.\n%s",
                     k_gpu_runtime_dll, err == ERROR_MOD_NOT_FOUND ? "was not found" : "could not be loaded", k_driver_advice);
        return exit_gpu;
    }
    // keep it loaded: the delay-load stub binds to this module
    return exit_ok;
}

void requireModelFile(const std::string& path, std::string_view what) {
    std::error_code ec;
    const std::filesystem::path p(widen(path));
    if (!std::filesystem::exists(p, ec))
        throw UserError(exit_model, std::format("{} not found: {}\n  Check the path (put it in quotes if it contains spaces).", what, path));
    if (std::filesystem::is_directory(p, ec))
        throw UserError(exit_model, std::format("{} is a folder, not a file: {}\n  Give the path of the .gguf file itself.", what, path));
}

void requirePortFree(const std::string& host, std::uint16_t port) {
    if (port == 0) return;
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d) != 0) return;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICHOST;
    addrinfo* res = nullptr;
    const std::string ps = std::to_string(port);
    if (getaddrinfo(host.c_str(), ps.c_str(), &hints, &res) != 0 || res == nullptr) {
        WSACleanup();
        throw UserError(exit_usage, std::format("invalid --host {} (give a numeric address such as 127.0.0.1 or 0.0.0.0)", host));
    }
    int err = 0;
    SOCKET s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s != INVALID_SOCKET) {
        BOOL on = TRUE;
        setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&on), sizeof on);
        if (bind(s, res->ai_addr, static_cast<int>(res->ai_addrlen)) != 0 || listen(s, 1) != 0) err = WSAGetLastError();
        closesocket(s);
    }
    freeaddrinfo(res);
    WSACleanup();
    if (err == WSAEADDRINUSE || err == WSAEACCES)
        throw UserError(exit_port, std::format("port {} on {} is already in use (another whirl-server or another program is listening there).\n"
                                               "  Stop that program, or start this one on another port: --port {}",
                                               port, host, port == 65535 ? 8081 : port + 1));
    if (err == WSAEADDRNOTAVAIL)
        throw UserError(exit_usage, std::format("--host {} is not an address of this computer (use 127.0.0.1 for local use, 0.0.0.0 for all networks)", host));
}

std::string explainText(const std::exception& e, int* exit_code) {
    int code = exit_error;
    std::string msg;
    const std::string what = e.what();
    if (const auto* ue = dynamic_cast<const UserError*>(&e)) {
        code = ue->exitCode();
        msg = what + "\n";
    } else if (const auto* me = dynamic_cast<const qwen35::ModelError*>(&e)) {
        const std::string& c = me->code();
        if (c == "UnsupportedArch") {
            code = exit_model;
            msg = "this model's architecture is not supported (" + what +
                  ").\n"
                  "  WHIRL runs these GGUF architectures: qwen35 (Qwen3.5-generation dense, e.g. Qwen3.8-27B)\n"
                  "  and qwen35moe (the mixture-of-experts variant, e.g. Ornith-1.5-35B-A3B).\n"
                  "  Architectures are added one at a time, each with kernels tuned for it; for other models\n"
                  "  (Llama, Gemma, Mistral, DeepSeek, Qwen3 / Qwen2.5, ...) use llama.cpp.\n";
        } else if (c == "UnsupportedTensorType" || c == "UnsupportedMoeShape" || c == "UnsupportedTensorShape" || c == "UnsupportedConfig" || c == "MissingTensor" || c == "Truncated" || c == "OpenFailed") {
            code = exit_model;
            msg = "this GGUF file cannot be loaded (" + what +
                  ").\n"
                  "  The architecture is supported, but the file uses a tensor type / layout WHIRL does not\n"
                  "  implement, or the file is incomplete (an interrupted download?). See the README for the\n"
                  "  tested files and quantizations.\n";
        } else if (c == "VramLimit") {
            code = exit_vram;
            msg = "the model does not fit in the GPU's memory (" + what +
                  ").\n"
                  "  Use a smaller quantization of the model (e.g. Q4_K_M or a 3-bit file instead of Q6_K / Q8_0 / MXFP4),\n"
                  "  a smaller --ctx / --ctx-per-slot, or a smaller prefill batch (WHIRL_PREFILL_BATCH=1024).\n"
                  "  Close other programs that use the GPU.\n";
        } else if (c == "NoTargetGpu") {
            code = exit_gpu;
            msg = "no supported AMD GPU found (" + what + ").\n  WHIRL needs an AMD Radeon AI PRO R9700 (RDNA 4, gfx1201) or a Radeon 8060S\n  (Ryzen AI Max, RDNA 3.5, gfx1151); this build has GPU kernels for: " +
                  supportedArchList() + ".\n  GPUs reported by the driver:\n" + deviceListText() +
                  "  If the GPU is installed but not listed, update the driver:\n" + k_driver_advice;
        } else if (c == "GpuBusy") {
            code = exit_gpu;
            msg = "the GPU is still in use by another WHIRL process (waited WHIRL_GPU_WAIT seconds).\n"
                  "  Close the other whirl / whirl-server, or set WHIRL_GPU_SHARE=1 to run side by side\n"
                  "  (both then share the GPU's memory and speed).\n";
        } else if (c == "KvF16DoesNotFit") {
            code = exit_vram;
            msg = (what.size() > c.size() + 2 ? what.substr(c.size() + 2) : what) + "\n";
        } else if (c == "UnsupportedKvFormat") {
            code = exit_usage;
            msg = what + "\n";
        } else if (c == "ContextTooLong") {
            code = exit_usage;
            msg = "the prompt plus the tokens to generate do not fit in the context (" + what +
                  ").\n  Raise --ctx (and lower --max-tokens if needed), or shorten the prompt.\n";
        } else {
            msg = what + "\n";
        }
    } else if (const auto* he = dynamic_cast<const hip::Error*>(&e)) {
        const int hc = he->code();
        const std::string& op = he->op();
        if (op == "VramLimit") {
            code = exit_vram;
            msg = "out of GPU memory under the simulated VRAM limit (" + what + ").\n" + k_vram_advice +
                  "  Unset WHIRL_VRAM_LIMIT_MB (or raise it) to use all of the GPU's memory.\n";
        } else if (hc == 2 /* hipErrorOutOfMemory */ || op == "hipMalloc" || op == "hipMallocAsync") {
            code = exit_vram;
            msg = "out of GPU memory (" + what + ").\n" + k_vram_advice;
        } else if (op == "hipHostMalloc" || op == "hipHostRegister") {
            code = exit_vram;
            msg = "cannot allocate pinned host memory (" + what +
                  ").\n  Free some RAM, or give the server a smaller host cache: --kv-ram-mb 2048 (0 turns it off).\n";
        } else if (hc == 35 /* hipErrorInsufficientDriver */ || hc == 100 /* hipErrorNoDevice */ || hc == 3 /* hipErrorNotInitialized */ ||
                   hc == 209 /* hipErrorNoBinaryForGpu */ || hc == 200 /* hipErrorInvalidImage */ || hc == 218 /* hipErrorInvalidKernelFile */ ||
                   hc == 101 /* hipErrorInvalidDevice */ || hc == 98 /* hipErrorInvalidDeviceFunction */) {
            code = exit_gpu;
            msg = "the AMD GPU runtime reported a problem with the GPU or driver (" + what + ").\n  GPUs reported by the driver:\n" +
                  deviceListText() + k_driver_advice;
        } else {
            msg = what + "\n";
        }
    } else if (dynamic_cast<const gguf::Error*>(&e) != nullptr) {
        code = exit_model;
        if (what.find("bad magic") != std::string::npos)
            msg = "this file is not a GGUF model (" + what + ").\n  WHIRL loads .gguf files (for example from Hugging Face; see the README).\n";
        else
            msg = "the GGUF file is damaged or incomplete (" + what + ").\n  Download it again and compare its size with the source.\n";
    } else if (what.rfind("cannot open ", 0) == 0 || what.rfind("cannot map", 0) == 0 || what.rfind("cannot stat ", 0) == 0) {
        code = exit_model;
        msg = what + "\n  Check that the file exists and is readable (not locked by another program).\n";
    } else if (what.find("out of memory") != std::string::npos || dynamic_cast<const std::bad_alloc*>(&e) != nullptr) {
        code = exit_vram;
        msg = "out of memory (" + what + ").\n" + k_vram_advice;
    } else if (what.rfind("cannot listen on ", 0) == 0) {
        const bool in_use = what.find("10048") != std::string::npos || what.find("10013") != std::string::npos;
        code = in_use ? exit_port : exit_error;
        msg = what + (in_use ? "\n  The port is already in use: stop the other server / program, or choose another --port.\n" : "\n");
    } else if (what.rfind("unknown option", 0) == 0 || what.rfind("missing value", 0) == 0 || what.find(": missing ") != std::string::npos) {
        code = exit_usage;
        msg = what + "\n";
    } else {
        msg = what + "\n";
    }
    if (exit_code) *exit_code = code;
    return msg;
}

int explain(const std::exception& e, std::string_view program) {
    int code = exit_error;
    const std::string msg = explainText(e, &code);
    std::fprintf(stderr, "%.*s: error: %s", static_cast<int>(program.size()), program.data(), msg.c_str());
    std::fflush(stderr);
    return code;
}

std::string exitCodeHelp() {
    return "exit codes: 0 ok, 1 other error, 2 bad command line, 3 GPU / driver problem,\n"
           "            4 model file problem, 5 out of GPU memory, 6 server port in use\n";
}

// ---------------------------------------------------------------------------
// environment variables

namespace {

struct EnvDoc {
    const char* group;
    const char* name;  // without the WHIRL_ prefix, with "=VALUES" when useful
    const char* desc;
    unsigned scope;
};

constexpr unsigned C = sc_chat, B = sc_bench, T = sc_selftest, Q = sc_seqtest, S = sc_serve, V = sc_vis;
constexpr unsigned GPU = C | B | T | Q | S;
constexpr unsigned TUNED = C | B | Q | S;

constexpr EnvDoc k_env[] = {
    {"GPU and loading", "DEVICE=SPEC", "GPU to use: r9700 (default), 8060s, an index, or a name / gfx substring (= --device)", GPU | V},
    {"GPU and loading", "HIP_DEVICE=N", "device index, bypassing device matching and the one-process-per-GPU lock", GPU},
    {"GPU and loading", "GPU_SHARE=1", "do not wait for other WHIRL processes on the same GPU", GPU},
    {"GPU and loading", "GPU_WAIT=S", "seconds to wait for another WHIRL process to free the GPU (default 1800)", GPU},
    {"GPU and loading", "MODE=precise|balance|fast[:ITEMS]", "numerics mode (= --precise / --balance / --fast / --mode; default precise; "
     "ITEMS e.g. fp8,kvq8 picks lossy items)", GPU},
    {"GPU and loading", "KV=auto|f16|q8|q8h|q8v", "KV cache format (auto: f16; in balance / fast mode dense models fall back to q8v, then q8h, "
     "when f16 does not fit; MoE f16). A q8 value in precise mode is a user-requested lossy override",
     GPU},
    {"GPU and loading", "PREFILL_BATCH=N", "prefill rows per forward (default 4096, up to 16384)", C | B | T | S},
    {"GPU and loading", "MAX_CTX=N", "default context size of chat (= --ctx; default 8192)", C},
    {"GPU and loading", "CODE_OBJECT=FILE", "development: load the GPU kernels from this code object instead of the built-in one", GPU},
    {"GPU and loading", "MOE_FP8=0|1", "MoE expert prefill with fp8 activations (MXFP4 experts; default: on in balance / fast)", GPU},
    {"GPU and loading", "MOE_MXW=0", "generic MXFP4 MoE decode kernels instead of the whole-block ones", GPU},
    {"GPU and loading", "EMBD_HOST=0", "keep the token embedding table in VRAM (default: pinned host memory, declared as Shared Usage)", S},
    {"GPU and loading", "TUNE_COLD=1", "autotune: evict the cache before each timing", C | B | T},
    {"GPU and loading", "TUNE_MASK=BITS", "autotune: mask of the candidate GEMM configurations", C | B | T},

    {"Thinking", "THINK=0|1", "thinking off / on (default on; = --no-think / --think)", C | B},

    {"Speculative decoding (output always equals plain greedy)", "MTP=0", "plain decoding, no MTP drafts", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "MTP_DRAFTS=N", "fixed MTP drafts per cycle, 1..10 (default: cost model, up to 8 dense / 1 MoE)", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "MTP_ADAPT=auto|N", "draft count from the cost model, or ceil(recent accepted + N)", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "MTP_PMIN=P", "end a draft chain at a draft with probability below P (after MTP_NMIN drafts)", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "MTP_NMIN=N", "drafts always made before MTP_PMIN applies", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "MTP_BATCH_DRAFTS=d1,d2,...", "max drafts per cycle with 1, 2, ... decoding slots", S},
    {"Speculative decoding (output always equals plain greedy)", "WIDE_VERIFY=0", "server: keep one verify forward at <= 16 rows (default: up to 32 rows when n-gram drafts add rows)", S},
    {"Speculative decoding (output always equals plain greedy)", "HEAD_CHUNK=N", "server: output head rows per pass in a > 16-row verify (default: the whole vocabulary)", S},
    {"Speculative decoding (output always equals plain greedy)", "MTP_Q4=0", "keep the MTP block's Q6_K matrices (default: Q4_K copies, used for drafts only)", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "DRAFT_HEAD=q4", "Q4_K draft head instead of the 2-bit one", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "MTP_FULLHEAD=1", "drafts use the full output head", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "DRAFT_VOCAB=off|64k|48k|FILE|N", "draft head over a frequency subset of the vocabulary (default: the embedded 64k subset on dense qwen35 models with a 248320-token vocabulary; off = full head; 48k = draft_vocab\\subset_48k.bin next to the exe; FILE = uint32 ids), or the first N rows", C | S},
    {"Speculative decoding (output always equals plain greedy)", "DRAFT_WINDOW=W, DRAFT_WINDOW_MIN=N", "MTP draft attention over the first 256 + last W positions once the context reaches N (defaults W = 16384, N = 65536); W = 0: off", C | S},
    {"Speculative decoding (output always equals plain greedy)", "NGRAM=0", "no n-gram (prompt-lookup) drafts", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "NGRAM_MIN=N", "minimum matched suffix for an n-gram draft (default 3)", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "NGRAM_SLOPE=X", "prior cost of one more n-gram verify row, relative to a 1-draft cycle (default 0.015; Radeon 8060S 0.12)", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "NGRAM_MAX=N", "most n-gram drafts per cycle (default 15)", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "NGRAM_FORCE=1", "take every n-gram proposal", C | B | S},
    {"Speculative decoding (output always equals plain greedy)", "NGRAM_DEBUG=1", "log every decode cycle to stderr", C | B},

    {"Server", "NO_PREFIX_CACHE=1", "disable the prefix cache", S},
    {"Server", "PREFIX_CACHE_VERIFY=1", "diagnostic: re-prefill after every cache hit and compare", S},
    {"Server", "SERVE_CKPTS=N", "prefix checkpoints per slot (default 4 with one slot, 2 up to 4 slots, else 1)", S},
    {"Server", "SYS_MIN=N", "system messages of at least N tokens get their own checkpoint (default 2048, 0 = off)", S},
    {"Server", "SYS_CKPTS=N", "shared prefix checkpoints kept in VRAM (default 2)", S},
    {"Server", "SYS_LCP=0", "no checkpoints at prefixes common to sessions", S},
    {"Server", "CKPT_HOST=1", "keep prefix checkpoints in pinned host memory instead of VRAM", S},
    {"Server", "POOL_RESERVE_MB=N", "VRAM left free when the KV pool is sized (default 768, MoE 1536)", S},
    {"Server", "PREFILL_CHUNK=N", "most rows per merged prefill forward (multiple of 1024, default 2048)", S},
    {"Server", "SEG_PREFILL=0", "prefill each request on its own instead of several in one forward", S},
    {"Server", "GATHER_MS=MS", "window to gather a burst of new requests (default 30, 0 = off)", S},
    {"Server", "DECODE_MIN_TPS=N", "decode floor per streaming request while others prefill (= --decode-min-tps, default 20, 0 = off)", S},
    {"Server", "GDN_REPLAY=0", "DeltaNet verify with snapshot sets instead of replaying kept rows", S},
    {"Server", "SNAP_SETS=N", "minimum number of recurrent-state snapshot sets (with GDN_REPLAY=0)", S},
    {"Server", "SLOT_DRAFTS=1", "split the draft budget between slots by expected acceptance", S},
    {"Server", "TIMING_RESET=0", "keep the MTP timing table across requests (not recommended)", S},
    {"Server", "KV_RAM_MB=N", "host RAM tier of the prefix cache in MiB (= --kv-ram-mb)", S},
    {"Server", "KV_SSD_DIR=PATH", "SSD tier directory (= --kv-ssd-dir)", S},
    {"Server", "KV_SSD_GB=N", "SSD tier size cap in GiB (= --kv-ssd-gb)", S},
    {"Server", "KV_TIER_MIN=N", "smallest session copied to the host tiers, in tokens (default 2048)", S},
    {"Server", "KV_SSD_DELAY_MS=MS", "delay before a RAM tier entry is also written to the SSD (default 2000)", S},
    {"Server", "MMPROJ=FILE", "vision encoder (= --mmproj)", S},
    {"Server", "VIS_IDLE_S=N", "release the vision encoder after N s without images (= --vis-idle-s)", S},
    {"Server", "VIS_CACHE_MB=N", "image embedding cache in MiB (= --vis-cache-mb)", S},
    {"Server", "VIS_CKPT_MIN=N", "keep a shared checkpoint after images ending at >= N tokens (default 1024, 0 = off)", S},

    {"Vision", "VIS_MODE=auto|resident|stream", "encoder weights resident in VRAM or streamed layer by layer (= --vis-mode)", C | S},
    {"Vision", "VIS_DUMP=PREFIX", "diagnostic: write each image's embeddings to PREFIX.<n>.f32", C | S},
    {"Vision", "VIS_DUMP_RGB=FILE", "diagnostic: write the preprocessed image", V},
    {"Vision", "VIS_PRE=FILE", "diagnostic: preprocess this image instead", V},
    {"Vision", "VIS_PROF=1", "diagnostic: per-stage encoder timing", V},

    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "FP8=0|1", "MXFP4 prefill with fp8 activations (default: on in balance / fast; 0 also turns MOE_FP8 off)", TUNED},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "FP8_MASK=BITS", "matmul classes that use fp8 activations (default 7)", TUNED},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "G8T=0", "row-major fp8 GEMM instead of the fragment-tiled one (same bits)", TUNED},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "GDN_WMMA=0|1", "f16-WMMA DeltaNet prefill chunks (default: on for MXFP4 in balance / fast)", TUNED},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "FFN_H16=0|1", "f16 GEMM outputs into the elementwise ops (default: on for MXFP4 in balance / fast)", TUNED},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "Q4_RELAXED=1", "the MXFP4 balance-mode prefill switches (WMMA DeltaNet, h16) for other models too", TUNED},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "ACT_FUSE=0", "no fused activation in prefill (bitwise-equal alternative)", TUNED},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "GEMMH=0", "no f16-output prefill GEMM (bitwise-equal alternative)", TUNED},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "GEMMHQ=1", "f16-output GEMM for the attention projections too", TUNED},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "ATTN_KX=0", "prefill attention without the K-exchange kernel (bitwise-equal)", TUNED},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "ATTN_KG=0", "prefill attention without the GQA-grouped kernel (bitwise-equal)", TUNED},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "GDN_SEQ=1", "sequential DeltaNet prefill instead of the chunked scan", C | B | S},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "GDN_V0=1", "per-row DeltaNet decode step kernel (same values)", C | S},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "NAIVE_ATTN=1", "reference attention path (numerics comparisons)", C | B | S},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "NO_FUSE=1", "no fused decode kernels (reference path)", C | B | S},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "FLOAT_GEMV=1", "f32 GEMV instead of int8 (reference path)", C | B | S},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "NO_GRAPH=1", "no HIP graph for the plain decode step", C | B},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "GEMV_MAX=N", "most tokens per multi-token GEMV", C | B},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "GEMV_R=nt:r,...", "multi-token GEMV rows-per-block overrides", C | B},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "GEMV_W=nt:v,...", "multi-token GEMV kernel variant overrides", C | B},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "GEMV_WH=nt:v,...", "multi-token GEMV variant overrides (f16 output)", C | B},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "GV_GROUP=0", "no grouped same-input GEMV launches", C | B},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "GV_NMAX=N", "largest group of same-input GEMVs", C | B},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "MOE_BN=32|64", "grouped expert GEMM token tile", C | B | S},
    {"Numerics / speed switches (A/B tests; defaults are the tested paths)", "DBG=BITS", "1 no gdn_abconv merge, 2 scalar split attention, 4 one query per attention group", C | B},

    {"Diagnostics", "TOKENIZE_ONLY=1", "print the prompt token ids and stop", C},
    {"Diagnostics", "PRINT_IDS=1", "print the generated token ids and their FNV-1a hash", C},
    {"Diagnostics", "PROFILE=1", "per-op-class GPU time (distorts speed numbers; bench: per prefill size; server: 1 or 2)", C | B | S},
    {"Diagnostics", "TRACE_TPS=N", "print the window tok/s every N tokens (stderr)", C | B},
    {"Diagnostics", "DUMP_LOGITS=FILE", "(no MTP) next-token logits of every prompt position >= DUMP_FROM as f16 rows", C},
    {"Diagnostics", "DUMP_FROM=N", "first prompt position written by DUMP_LOGITS", C},
    {"Diagnostics", "DUMP_MOE=FILE", "selected experts of every prefill MoE block (i32)", C},
    {"Diagnostics", "TRACE_ND=1", "log the draft count of every decode cycle", S},
    {"Diagnostics", "LOOP_LOG=1", "log per-loop phase timing", S},
    {"Diagnostics", "TIMER_PROBE=1", "log the actual length of a 1 ms / 200 us sleep before and after the 1 ms timer resolution is set", S},
    {"Diagnostics", "TIER_VERIFY=1", "byte-verify every host tier spill / restore", S},
    {"Diagnostics", "TIER_MIN_GAIN=N", "restore from a host tier only when it saves at least N tokens (default 512)", S},
};

}  // namespace

std::string envHelp(unsigned scope) {
    std::string out;
    const char* group = nullptr;
    for (const EnvDoc& d : k_env) {
        if ((d.scope & scope) == 0) continue;
        if (group == nullptr || std::string_view(group) != d.group) {
            group = d.group;
            out += std::format("\n  {}:\n", group);
        }
        const std::string name = std::string("WHIRL_") + d.name;
        if (name.size() <= 30)
            out += std::format("    {:<30} {}\n", name, d.desc);
        else
            out += std::format("    {}\n    {:<30} {}\n", name, "", d.desc);
    }
    if (!out.empty()) out = "\nenvironment variables:" + out;
    return out;
}

}  // namespace whirl::app

// The delay-load failure hook (whirl.exe / whirl-server.exe link amdhip64_7.dll
// with /DELAYLOAD so that a missing or too old driver gets a message, not a crash).
extern "C" const PfnDliHook __pfnDliFailureHook2 = whirl::app::delayLoadFailureHook;
