// whirl CLI (src/cli): the declarations the command files share (arch025-C2: src/cli/main.cpp split into one file per
// command, as llama.cpp's app/llama.cpp + tools/<name>/ and git's git.c + builtin/<cmd>.c).
// SPDX-License-Identifier: Apache-2.0
//
// The include list and the using lines are the old main.cpp's, so every command file sees the declarations the single
// file saw. Everything is in the global namespace `cli` (the old file used an anonymous namespace, i.e. a uniquely named
// global namespace + using-directive): unqualified lookup inside `cli` is the old one (cli, then the global namespace with
// `using namespace whirl`). The structs are the old text; the functions are declared here and defined in common.cpp
// (arguments, text, the server's prefill forwards, hashes), build_model.cpp (the CLI's own model building: GPU choice,
// Model::load options, its after-load switches; not shared with whirl-server), decode.cpp (the decode loops) and the
// cmd_<name>.cpp files. Helpers only one file uses stay in that file's anonymous namespace.
#pragma once

#include "whirl/dump.h"
#include "whirl/common.h"
#include "whirl/dflash_policy.h"
#include "whirl/gguf.h"
#include "whirl/hip.h"
#include "whirl/gpu_select.h"
#include "whirl/hip_runtime_loader.h"
#include "whirl/leak_count.h"
#include "whirl/model.h"
#include "whirl/prefill_schedule.h"
#include "whirl/tokenizer.h"
#include "whirl/uma_mem.h"
#include "whirl/vram_limit.h"

#if WHIRL_HAVE_SERVER
#include "server/server_main.h"
#endif
#if WHIRL_HAVE_VISION
#include "whirl/vision.h"
#endif
#include "release/release.h"
#include "platform/console.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace whirl;
namespace q = whirl::qwen35;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

namespace cli {

#define WHIRL_HELP_MODES                                                                            \
    "  --precise | --balance | --fast\n"                                                            \
    "                             numerics mode (default balance: dense fp8 MXFP4 prefill, q8h KV,\n" \
    "                             int8 decode activations; MoE f16 KV; --precise: f16/f32\n" \
    "                             activations, f16 KV; --fast: balance + 4-bit KV, relaxed MTP\n" \
    "                             acceptance). --balance=fp8,kvq8 picks\n" \
    "                             items; --mode M also works; env WHIRL_MODE\n"

// the help text of each command (cmd_<name>.cpp; main.cpp's command table points at them)
extern const char* k_help_chat;
extern const char* k_help_bench;
extern const char* k_help_selftest;
extern const char* k_help_golden;
extern const char* k_help_dflash_check;
extern const char* k_help_seqtest;
extern const char* k_help_devices;
extern const char* k_help_vis;

// common.cpp
void out(const std::string& s);
std::string fmt(const char* f, ...);
double nowMs();

// Wall-clock timer around device work: synchronizes at begin and end.
struct Timer {
    double t0 = 0;
    void begin() {
        hip::sync();
        t0 = nowMs();
    }
    double end() {
        hip::sync();
        return nowMs() - t0;
    }
};

struct Args {
    std::vector<std::string> pos;
    std::vector<std::string> images;  // --image (repeatable)
    std::unordered_map<std::string, std::string> opts;
    bool has(const char* k) const { return opts.count(k) != 0; }
    const std::string* get(const char* k) const {
        auto it = opts.find(k);
        return it == opts.end() ? nullptr : &it->second;
    }
};

// common.cpp
Args parseArgs(const std::vector<std::string>& argv, std::size_t from);
std::vector<u32> parseIds(const std::string& list);
std::optional<u32> envU32(const char* name);

// build_model.cpp
void applyDraftVocab(q::Model& model, const Tokenizer& tok);

// ---------------------------------------------------------------------------
// tokenizer helpers

struct Tok {
    Tokenizer t;
    u32 im_end = 0xffffffffu, eot = 0xffffffffu, think_open = 0xffffffffu, newline = 0xffffffffu;
    u32 im_start = 0xffffffffu, system = 0xffffffffu, vision_end = 0xffffffffu;  // CLI-SCHED: forced splits

    explicit Tok(const gguf::File& f) : t(Tokenizer::fromGguf(f)) {
        auto id = [&](const char* s) {
            const TokenId i = t.find(s);
            return i < 0 ? 0xffffffffu : static_cast<u32>(i);
        };
        im_end = id("<|im_end|>");
        eot = id("<|endoftext|>");
        think_open = id("<think>");
        const std::vector<TokenId> nl = t.encode("\n");
        if (nl.size() == 1) newline = static_cast<u32>(nl[0]);
        // as the server's ChatTokens (src/server/tokens.cpp) and VisionConfig::vision_end_id
        im_start = id("<|im_start|>");
        const std::vector<TokenId> sy = t.encode("system", false);
        if (sy.size() == 1) system = static_cast<u32>(sy[0]);
        vision_end = id("<|vision_end|>");
    }

    std::vector<u32> encode(const std::string& s) const {
        const std::vector<TokenId> v = t.encode(s);
        return std::vector<u32>(v.begin(), v.end());
    }

    std::string decode(std::span<const u32> ids) const {
        std::vector<TokenId> v(ids.begin(), ids.end());
        return t.decode(v);
    }

    // A thinking-mode prompt ("...<think>\n") runs its last token as its own
    // step (like the server's prefill schedule); 0 = no split.
    std::size_t thinkOpenSplit(std::span<const u32> ids) const {
        const std::size_t n = ids.size();
        if (n >= 2 && ids[n - 2] == think_open && ids[n - 1] == newline) return n - 1;
        return 0;
    }

    // token id -> the token with the same text where CR LF is LF (identity when
    // the text has no CR LF or no such token exists)
    std::vector<u32> crlfToLf() const {
        const std::size_t n = t.size();
        std::unordered_map<std::string, u32> by_piece;
        std::vector<std::string> pieces(n);
        for (std::size_t i = 0; i < n; ++i) {
            pieces[i] = t.piece(static_cast<TokenId>(i), true);
            by_piece.emplace(pieces[i], static_cast<u32>(i));
        }
        std::vector<u32> norm(n);
        bool any = false;
        for (std::size_t i = 0; i < n; ++i) {
            norm[i] = static_cast<u32>(i);
            const std::string& p = pieces[i];
            if (p.find("\r\n") == std::string::npos) continue;
            std::string q;
            for (std::size_t j = 0; j < p.size(); ++j) {
                if (p[j] == '\r' && j + 1 < p.size() && p[j + 1] == '\n') continue;
                q.push_back(p[j]);
            }
            auto it = by_piece.find(q);
            if (it != by_piece.end()) {
                norm[i] = it->second;
                any = true;
            }
        }
        if (!any) norm.clear();
        return norm;
    }
};

// common.cpp
std::vector<std::size_t> serverForwardEnds(const Tok& tok, std::span<const u32> ids, u32 max_batch, const device::Profile& prof, bool moe);
void prefillLikeServer(q::Model& m, const Tok& tok, std::span<const u32> ids, bool mtp);

// Prints the complete UTF-8 prefix of the decoded generation as it grows.
struct Streamer {
    const Tok& tok;
    bool on;
    std::size_t printed = 0;
    void update(std::span<const u32> gen) {
        if (!on) return;
        const std::string s = tok.decode(gen);
        std::size_t end = s.size();
        while (end > printed && !isValidUtf8(std::string_view(s).substr(printed, end - printed))) end -= 1;
        if (end > printed) {
            out(s.substr(printed, end - printed));
            printed = end;
        }
    }
};

// common.cpp
u64 fnv(std::span<const u32> toks);
struct ChatOpts {
    u32 max_tokens = 64;
    u32 max_ctx = 8192;
    bool stream = true;
    float mtp_p_min = 0;
    u32 mtp_n_min = 0;
    u32 mtp_adapt = 0;
    bool mtp_auto = false;
    u32 trace_tps = 0;
    bool ngram = true;
    u32 ngram_min = 3;
    float ngram_slope = 0.015f;
    u32 ngram_max = 0;
    bool ngram_force = false;
    bool ngram_debug = false;
    bool print_text = true;
    // --draft dflash: DFlash2 walk cutoffs (WHIRL_DFLASH_PMIN / WHIRL_DFLASH_NMIN)
    float dfl_p_min = 0;
    u32 dfl_n_min = 0;
};

struct DecodeResult {
    std::vector<u32> tokens;
    double decode_ms = 0;
    u32 steps = 0;  // plain: decode steps timed (n_gen - 1)
    u32 cycles = 0, accepted = 0, drafted = 0;
    std::size_t verify_tokens = 0;
    std::array<u32, q::max_drafts> pos_tries{}, pos_hits{};
    u32 ng_cycles = 0, ng_drafted = 0, ng_accepted = 0;
    std::array<float, 3> ng_alpha{};
    double t_draft = 0, t_verify = 0;
    u32 n_draft = 0, n_min = 0;
    float p_min = 0;
};

struct TpsTrace {
    u32 every = 0;
    double t0 = 0, tl = 0;
    std::size_t nl = 0;
    void start(u32 e) {
        every = e;
        t0 = tl = nowMs();
        nl = 0;
    }
    void tick(std::size_t n) {
        if (every == 0 || n < nl + every) return;
        const double now = nowMs();
        std::fprintf(stderr, "  [tps] t=%.1f s  tokens=%zu  window %.2f tok/s\n", (now - t0) / 1000.0, n, (n - nl) / ((now - tl) / 1000.0));
        tl = now;
        nl = n;
    }
};

// decode.cpp
DecodeResult plainDecode(q::Model& model, const Tok& tok, u32 first, std::size_t n_prompt, const ChatOpts& opt);
DecodeResult specDecode(q::Model& model, const Tok& tok, u32 first, u32 n_prompt, const ChatOpts& opt, u32 drafts,
                        std::span<const u32> prompt_ids);
DecodeResult dflashDecode(q::Model& model, const Tok& tok, u32 first, u32 n_prompt, const ChatOpts& opt, std::span<const u32> prompt_ids);
void printProfile(const char* label, q::Profile& p, std::size_t n_tok);

// build_model.cpp
bool wantDFlash(const Args& a, const gguf::File& f);
bool loadDraftModelEnv(q::Model& m, const Args& a);
void applyRuntimeEnv(q::Model& m, bool gdn_v0 = false);

struct Loaded {
    std::unique_ptr<q::Model> model;
    q::LoadStats stats;
};

// build_model.cpp
int selectDevice(const Args& a, std::string* info_line);
whirl::numerics::Request modeRequest(const Args& a);
Loaded loadModel(const Args& a, const gguf::File& f, u32 max_ctx, bool ctx_explicit = true, u32 min_ctx = 0, bool no_mtp = false);
void loadLog(q::Model& m, const q::LoadStats& stats);

// common.cpp
std::string readPromptArg(const std::string& p);
std::string chatWrap(const std::string& prompt, bool think);
bool thinkFromArgs(const Args& a);

// decode.cpp
ChatOpts mtpOpts(const q::Model& model, ChatOpts base, u32* drafts_out);
ChatOpts dflashOpts(const q::Model& model, ChatOpts base);
void printSpecSummary(const DecodeResult& r, std::size_t n_prompt, double prefill_ms, const ChatOpts& opt, const char* kind = "MTP spec");

// common.cpp: the bench prompt (bench decode runs; golden / seqtest long prompts), the logits hash, the long prompts
extern const char* k_bench_text;
u64 fnvF32(std::span<const float> v);
std::vector<u32> goldenLongIds(const Tok& tok, std::size_t n);

// the commands (cmd_<name>.cpp); cmdServe / cmdVisEncode take the unparsed arguments and exist only in builds with the
// server (WHIRL_HAVE_SERVER) / the vision encoder (WHIRL_HAVE_VISION)
int cmdChat(const Args& a);
int cmdBench(const Args& a);
int cmdSeqtest(const Args& a);
int cmdGolden(const Args& a);
int cmdDflashCheck(const Args& a);
int cmdSelftest(const Args& a);
int cmdDevices();
int cmdServe(std::vector<std::string>& argv);
int cmdVisEncode(std::vector<std::string>& argv);

}  // namespace cli
