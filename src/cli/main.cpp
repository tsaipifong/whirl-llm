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

#include "whirl/common.h"
#include "whirl/dflash_policy.h"
#include "whirl/gguf.h"
#include "whirl/hip.h"
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

#include <fcntl.h>
#include <io.h>

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

#define WHIRL_HELP_MODES                                                                            \
    "  --precise | --balance | --fast\n"                                                            \
    "                             numerics mode (default balance: dense fp8 MXFP4 prefill, q8h KV,\n" \
    "                             int8 decode activations; MoE f16 KV; --precise: f16/f32\n" \
    "                             activations, f16 KV; --fast: balance + 4-bit KV, relaxed MTP\n" \
    "                             acceptance). --balance=fp8,kvq8 picks\n" \
    "                             items; --mode M also works; env WHIRL_MODE\n"

const char* k_help_chat =
    "usage: whirl chat MODEL.gguf \"PROMPT\" | @PROMPT_FILE [options]\n"
    "\n"
    "Greedy generation for one prompt, wrapped in the Qwen chat template. With an MTP head in\n"
    "the model the decode is speculative (MTP + n-gram drafts, or DFlash2 + n-gram drafts with\n"
    "--draft dflash); the output is identical to plain greedy decoding.\n"
    "\n"
    "options:\n"
    "  \"PROMPT\" | @FILE           prompt text, or @path of a UTF-8 file holding it (long prompts)\n"
    "  --max-tokens N             tokens to generate (default 64)\n"
    "  --ctx N                    context size: prompt + generated tokens (default 8192)\n"
    "  --no-think / --think       thinking off / on (default on)\n"
    "  --no-stream                print the reply once at the end instead of streaming it\n"
    "  --raw                      no chat template: the prompt text is tokenized as is\n"
    "  --tokens ID,ID,...         token ids instead of a prompt (no template)\n"
    "  --out FILE                 write the last prompt position's logits (f32) to FILE (no MTP)\n"
    "  --draft mtp|dflash         draft source (default mtp). dflash: the DFlash2 drafter GGUF in\n"
    "                             --draft-model FILE (else WHIRL_DRAFT_MODEL) drafts up to block - 1 tokens per cycle (n-gram drafts\n"
    "                             stay on); the MTP block is not loaded. Without a matching drafter: mtp\n"
    "  --mmproj MMPROJ.gguf       vision encoder (Qwen3-VL style mmproj, F16 / BF16) for --image\n"
    "  --image IMAGE              an image placed before the prompt text (repeatable; needs --mmproj)\n"
    "  --device SPEC              GPU: r9700 (default), 8060s, an index, or a name / gfx substring\n"
    WHIRL_HELP_MODES
    "  -h, --help                 this text\n";

const char* k_help_bench =
    "usage: whirl bench MODEL.gguf [options]\n"
    "\n"
    "Loads the model once, then measures prefill speed for each size (best of 2 after a warm-up\n"
    "up to 8k, one timed run above) and decode speed in each mode. The token streams of all\n"
    "decode modes must be identical; a difference is reported.\n"
    "\n"
    "options:\n"
    "  --prefill N,N,...          prefill sizes in tokens (default 2048,8192,32768)\n"
    "  --decode N                 tokens to generate per decode mode (default 256)\n"
    "  --prompt TEXT | @FILE      prompt of the decode runs (default: a zh/en coding question)\n"
    "  --modes M,M,...            decode modes: mtp-ngram (MTP + n-gram drafts), mtp, plain; with\n"
    "                             --draft dflash: dflash-ngram, dflash, plain (default: all three)\n"
    "  --draft mtp|dflash         draft source (default mtp; dflash: --draft-model, see whirl chat)\n"
    "  --draft-model FILE         DFlash2 drafter GGUF for --draft dflash (wins over env WHIRL_DRAFT_MODEL)\n"
    "  --no-think / --think       thinking off / on for the decode prompt (default on)\n"
    "  --device SPEC              GPU: r9700 (default), 8060s, an index, or a name / gfx substring\n"
    WHIRL_HELP_MODES
    "  -h, --help                 this text\n";

const char* k_help_selftest =
    "usage: whirl selftest MODEL.gguf [--device SPEC]\n"
    "\n"
    "Bitwise self-checks on the model's own matrices: int8 vs f32 GEMV error, multi-token GEMV\n"
    "== one-token GEMV, prefill GEMM invariance over all configurations, MoE token-tile\n"
    "invariance, grouped vs per-query decode attention. Prints 'selftest: ok' or FAIL (exit 1).\n"
    "\n"
    "options:\n"
    "  --device SPEC              GPU: r9700 (default), 8060s, an index, or a name / gfx substring\n"
    WHIRL_HELP_MODES
    "  -h, --help                 this text\n";

const char* k_help_golden =
    "usage: whirl golden MODEL.gguf [--suite single|pair|ngram] [--decode N] [mode] [--device SPEC]\n"
    "\n"
    "Runs fixed prompts and prints one 'G ...' line per step: top-5 ids and the FNV-1a hash\n"
    "of the logits, generated token streams. single: plain prefill (short + 4000 tokens),\n"
    "decode steps, MTP decode (fixed drafts; forced n-gram drafts). pair: two sequences on the\n"
    "server paths (segmented prefill, batched rows, batched verify, one batched MTP cycle).\n"
    "tools/golden/golden.ps1 records these under tests/golden/<arch>/ and diffs later runs.\n";
const char* k_help_dflash_check =
    "usage: whirl dflash-check MODEL.gguf DRAFT.gguf REFDIR [--device SPEC]\n"
    "\n"
    "Runs the DFlash2 drafter (DRAFT.gguf) of MODEL.gguf on reference inputs and compares each layer's\n"
    "block hidden state, the final norm and the drafts with reference values (whirl-cloud\n"
    "tools/dflash_ref: ref_drafter.py, then export_check.py writes REFDIR = meta.txt, feats.f32 in\n"
    "layer-major order, hidden_l<i>.f32, final_norm.f32, drafts.i32). Prints relative errors; exit 1\n"
    "if a layer differs by more than 3e-2 (Q8_0 weights, int8 GEMV activations).\n"
    "\n"
    "options:\n"
    "  --device SPEC              GPU: r9700 (default), 8060s, an index, or a name / gfx substring\n"
    "  -h, --help                 this text\n";

const char* k_help_seqtest =
    "usage: whirl seqtest MODEL.gguf [--decode N] [--slot-ctx N] [--device SPEC]\n"
    "\n"
    "Runs the multi-sequence (server) paths against the single-sequence reference on two\n"
    "prompts: segmented prefill, batched decode rows, batched verify with literal drafts.\n"
    "Every token must match; prints 'seqtest: ok' or FAIL (exit 1).\n"
    "\n"
    "options:\n"
    "  --decode N                 tokens generated per sequence (default 24)\n"
    "  --device SPEC              GPU: r9700 (default), 8060s, an index, or a name / gfx substring\n"
    WHIRL_HELP_MODES
    "  -h, --help                 this text\n";

const char* k_help_devices =
    "usage: whirl devices\n"
    "\n"
    "Lists the AMD GPUs the driver reports (index, name, architecture, memory, compute units),\n"
    "whether this build has GPU kernels for each of them, and which one is used by default\n"
    "(the first R9700, else the first supported GPU; --device / WHIRL_DEVICE pick another).\n";

const char* k_help_vis =
    "usage: whirl vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto|resident|stream]\n"
    "\n"
    "Encodes one image with a vision encoder (diagnostic) and reports the timing; OUT.f32\n"
    "receives the projected embeddings.\n"
    "\n"
    "options:\n"
    "  --reps N                   encode N times (timing)\n"
    "  --mode M                   auto (default), resident (weights in VRAM), stream (layer by layer)\n";

void out(const std::string& s) {
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fflush(stdout);
}

std::string fmt(const char* f, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

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

bool takesValue(const std::string& o) {
    static const char* v[] = {"--max-tokens", "--ctx", "--tokens", "--out", "--device", "--prefill", "--decode", "--prompt", "--modes", "--mmproj", "--suite",
                              "--draft", "--draft-model", "--slot-ctx", "--sched-lens"};
    for (const char* s : v)
        if (o == s) return true;
    return false;
}

Args parseArgs(const std::vector<std::string>& argv, std::size_t from) {
    Args a;
    for (std::size_t i = from; i < argv.size(); ++i) {
        const std::string& s = argv[i];
        if (s.size() > 2 && s[0] == '-' && s[1] == '-') {
            std::string mspec;
            bool mnext = false;
            if (whirl::numerics::modeFlag(s, &mspec, &mnext)) {
                if (mnext) {
                    if (i + 1 >= argv.size()) throw std::runtime_error("missing value for " + s);
                    mspec = argv[++i];
                }
                if (a.has("--mode")) throw std::runtime_error("give one numerics mode (--precise, --balance, --fast or --mode M)");
                (void)whirl::numerics::parseMode(mspec, "command line");  // reject a bad value before any GPU work
                a.opts["--mode"] = mspec;
            } else if (s == "--image") {
                if (i + 1 >= argv.size()) throw std::runtime_error("missing value for " + s);
                a.images.push_back(argv[++i]);
            } else if (takesValue(s)) {
                if (i + 1 >= argv.size()) throw std::runtime_error("missing value for " + s);
                a.opts[s] = argv[++i];
            } else {
                static const char* flags[] = {"--no-stream", "--raw", "--no-think", "--think"};
                bool known = false;
                for (const char* f : flags) known = known || s == f;
                if (!known) throw std::runtime_error("unknown option " + s);
                a.opts[s] = "";
            }
        } else {
            a.pos.push_back(s);
        }
    }
    return a;
}

std::vector<u32> parseIds(const std::string& list) {
    std::vector<u32> ids;
    std::size_t p = 0;
    while (p < list.size()) {
        std::size_t e = list.find(',', p);
        if (e == std::string::npos) e = list.size();
        std::string t = list.substr(p, e - p);
        t.erase(0, t.find_first_not_of(' '));
        if (!t.empty()) ids.push_back(static_cast<u32>(std::stoul(t)));
        p = e + 1;
    }
    return ids;
}

// WHIRL_DRAFT_VOCAB: unset = the embedded 64k subset on dense qwen35 models with the 248320-token
// vocabulary; 64k = embedded; 48k / <file> = a frequency subset file; N = the first N draft-head rows
// (old experiment); off = full head. Special and byte tokens are always added.
void applyDraftVocab(q::Model& model, const Tokenizer& tok) {
    using K = q::DraftVocabChoice::Kind;
    const q::DraftVocabChoice dv = q::draftVocabChoice(envGet("DRAFT_VOCAB"), q::exeDirectory());
    if (dv.kind == K::first_n) {
        model.draft_vocab = dv.n;
        return;
    }
    if (dv.kind == K::full) return;
    if (dv.kind == K::embedded_64k) {
        const bool fits = dv.by_default ? q::draftVocabDefaultFits(model.cfg.moe, model.cfg.n_vocab)
                                        : model.cfg.n_vocab == q::draft_vocab_embedded_n_vocab;
        if (!fits || (dv.by_default && !model.draft_d2)) {
            if (!dv.by_default) std::fprintf(stderr, "WHIRL_DRAFT_VOCAB=64k ignored: the embedded subset needs a 248320-token vocabulary\n");
            return;
        }
    }
    std::vector<u32> req;
    for (const TokenId t : tok.specials())
        if (t >= 0) req.push_back(static_cast<u32>(t));
    for (int b = 0; b < 256; ++b)
        if (tok.byteToken(b) >= 0) req.push_back(static_cast<u32>(tok.byteToken(b)));
    u32 added = 0;
    const auto ids =
        q::draftVocabIds(dv.kind == K::file ? q::readDraftVocab(dv.file) : q::embeddedDraftVocab64k(), model.cfg.n_vocab, req, &added);
    const std::string label = dv.kind == K::file ? dv.file : std::string(dv.by_default ? "64k (embedded, default)" : "64k (embedded)");
    if (model.setDraftVocab(ids))
        std::fprintf(stderr, "draft head: vocabulary subset %s (%zu of %u rows; %u special / byte tokens added)\n", label.c_str(),
                     ids.size(), model.cfg.n_vocab, added);
    else if (!dv.by_default)  // the default only: stay quiet (the user did not set WHIRL_DRAFT_VOCAB)
        std::fprintf(stderr, "WHIRL_DRAFT_VOCAB ignored: it needs the 2-bit draft head (Q6_K output head, WHIRL_DRAFT_HEAD not q4)\n");
}

std::optional<u32> envU32(const char* name) {
    if (auto v = envGet(name)) {
        try {
            return static_cast<u32>(std::stoul(*v));
        } catch (...) {
        }
    }
    return std::nullopt;
}

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

// ---------------------------------------------------------------------------
// CLI-SCHED: every CLI prefill runs the forwards whirl-server runs for the same prompt alone with
// default options (include/whirl/prefill_schedule.h: the 1024-row chunk schedule with its merged
// tail, the forced splits - system message, think-open, image ends - the forward grouping, the
// TAIL-CK forward end), so `whirl chat` and the server give the same bits for the same prompt.
// The server's knobs apply here too: WHIRL_PREFILL_CHUNK, WHIRL_SYS_MIN, WHIRL_VIS_CKPT_MIN,
// WHIRL_TAIL_CK=0.
// (image ends: the ids >= image_id_base exist only in a prompt with images)
std::vector<std::size_t> serverForwardEnds(const Tok& tok, std::span<const u32> ids, u32 max_batch, bool gfx1151, bool moe) {
    const std::size_t n = ids.size();
    const std::size_t sys = sched::sysSplit(sched::sysBoundary(ids, tok.im_start, tok.system, tok.im_end, tok.newline), n,
                                            envU32("SYS_MIN").value_or(sched::sys_min_default));
    const std::vector<std::uint32_t> imgs =
        sched::imageSplits(ids, q::image_id_base, tok.vision_end, envU32("VIS_CKPT_MIN").value_or(sched::img_split_min_default));
    const std::size_t exec = sched::prefillExecFor(envU32("PREFILL_CHUNK").value_or(sched::prefillExecDefault(gfx1151, moe)), max_batch, gfx1151);
    const std::size_t tail_min = envFlag("TAIL_CK", true) ? sched::tail_ck_min_default : 0;
    return sched::soloForwardEnds(n, sys, tok.thinkOpenSplit(ids), imgs, 0, exec, tail_min, max_batch);
}

// the CLI prefill of ids from position 0 on the server's forwards (with the MTP block over the prompt: mtp)
void prefillLikeServer(q::Model& m, const Tok& tok, std::span<const u32> ids, bool mtp) {
    std::size_t off = 0;
    for (const std::size_t e : serverForwardEnds(tok, ids, m.max_batch, m.arch == hip::Arch::gfx1151, m.cfg.moe)) {
        if (mtp)
            m.prefillMtpChunk(ids, 0, off, e - off, std::nullopt);
        else
            m.forward(ids.subspan(off, e - off), static_cast<u32>(off));
        off = e;
    }
}

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

u64 fnv(std::span<const u32> toks) {
    u64 h = 0xcbf29ce484222325ull;
    for (u32 t : toks) {
        h ^= t;
        h *= 0x100000001b3ull;
    }
    return h;
}

// ---------------------------------------------------------------------------
// decode loops

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

// Pipelined decode: step k+1 is enqueued before the host waits for token k.
DecodeResult plainDecode(q::Model& model, const Tok& tok, u32 first, std::size_t n_prompt, const ChatOpts& opt) {
    DecodeResult r;
    Streamer st{tok, opt.stream};
    constexpr int ring_n = 4;
    auto* ring = static_cast<std::int32_t*>(hip::hostMalloc(4 * ring_n));
    hip::Event events[ring_n];
    for (auto& e : events) e = hip::eventCreate();
    auto enq = [&](std::size_t kk) {
        model.decodeStep();
        const std::size_t slot = kk % ring_n;
        hip::downloadAsync(&ring[slot], model.out_tok, 4, model.stream);
        hip::eventRecord(events[slot], model.stream);
    };
    Timer timer;
    timer.begin();
    u32 n_gen = 0;
    u32 cur = first;
    std::size_t kk = 0;
    const u32 budget_ctx = opt.max_ctx - static_cast<u32>(n_prompt);
    if (opt.max_tokens > 1) enq(1);
    TpsTrace tps;
    tps.start(opt.trace_tps);
    while (n_gen < opt.max_tokens && n_gen < budget_ctx) {
        if (cur == tok.im_end || cur == tok.eot) break;
        r.tokens.push_back(cur);
        n_gen += 1;
        tps.tick(n_gen);
        if (n_gen == opt.max_tokens) break;
        if (n_gen + 1 < opt.max_tokens) enq(kk + 2);
        hip::eventSync(events[(kk + 1) % ring_n]);
        cur = static_cast<u32>(ring[(kk + 1) % ring_n]);
        kk += 1;
        st.update(r.tokens);
    }
    r.decode_ms = timer.end();
    r.steps = n_gen > 0 ? n_gen - 1 : 0;
    for (auto e : events) hip::eventDestroy(e);
    hip::hostFree(ring);
    if (!opt.stream && opt.print_text) out(tok.decode(r.tokens));
    return r;
}

// Speculative decoding with the checkpoint's MTP head (+ n-gram drafts).
DecodeResult specDecode(q::Model& model, const Tok& tok, u32 first, u32 n_prompt, const ChatOpts& opt, u32 drafts,
                        std::span<const u32> prompt_ids) {
    DecodeResult res;
    const u64 E = model.cfg.n_embd;
    constexpr u32 MD = q::max_drafts;
    Streamer st{tok, opt.stream};
    std::vector<u32>& generated = res.tokens;
    u32 next = first;
    u32 p = n_prompt;  // position of `next`
    std::array<u32, q::max_small_batch> pend{};
    pend[0] = first;
    std::size_t n_pend = 1;
    u32 pend_pos = p;
    u32 cycles = 0, accepted = 0;
    bool done = false;
    double t_draft = 0, t_verify = 0;
    std::size_t verify_tokens = 0;
    // drafts per cycle: more than 1 needs the fused decode path (multi-row snapshots)
    const u32 n_draft = !model.fusedDecode() ? 1 : std::min(drafts, MD);
    const float p_min = opt.mtp_p_min;
    const u32 n_min = std::min(opt.mtp_n_min, n_draft);
    model.draft_p_min = p_min;
    model.draft_n_min = n_min;
    model.ensureSnapshots(n_draft);
    u32 drafted = 0;
    std::array<u32, MD> pos_tries{}, pos_hits{};
    float acc_ema = static_cast<float>(n_draft);
    q::DraftAccept acc_model;
    u32 prev_nd = 0;
    q::DraftTiming timing;
    std::vector<u32> hist;
    q::NgramPolicy ngp;
    ngp.verify_slope = opt.ngram_slope;
    ngp.reset();
    if (opt.ngram) ngp.ng.norm = tok.crlfToLf();
    u32 ng_max = opt.ngram_max > 0 ? std::min(opt.ngram_max, q::max_ng_drafts) : q::max_ng_drafts;
    // the unfused decode path snapshots row 0 only: more than 1 draft would restore an unwritten set
    if (!model.fusedDecode()) ng_max = std::min<u32>(ng_max, 1);
    if (opt.ngram && ng_max > model.snap_sets) {
        // n-gram verifies need a recurrent-state snapshot set per draft: take them now
        // if they fit in free VRAM with room to spare, else draft at most what exists
        u64 per = 0;
        for (u32 i = 0; i < model.cfg.n_layer; ++i)
            if (model.ssm_state[i] != 0) per += model.convBytes() + model.ssmBytes();
        const hip::MemInfo mem = hip::memInfo();
        const u64 extra = static_cast<u64>(ng_max - model.snap_sets) * per;
        if (mem.free > extra + (1536ull << 20))
            model.ensureSnapshots(ng_max);
        else
            ng_max = std::max<u32>(model.snap_sets, 1);
    }
    u32 ng_cycles = 0, ng_drafted = 0, ng_accepted = 0;
    if (opt.ngram) {
        hist.assign(prompt_ids.begin(), prompt_ids.end());
        hist.push_back(first);
    }
    auto emit = [&](u32 t) {
        if (t == tok.im_end || t == tok.eot) return true;
        generated.push_back(t);
        return generated.size() >= opt.max_tokens;
    };
    // relaxacc (fast mode): top-K of the verify rows for the relaxed greedy rule (whirl/relax_accept.h)
    const whirl::relax::Params& rx = model.relax;
    const u32 rk = rx.topk();
    hip::Function rx_topk = nullptr;
    hip::DevPtr rx_dev = 0;
    std::vector<std::uint8_t> rx_host;
    if (rx.on) {
        rx_topk = model.module.getFunction("topk_rows");
        const std::size_t rb = static_cast<std::size_t>(q::max_small_batch) * rk * 8 + q::max_small_batch * 8;
        rx_dev = hip::malloc(rb);
        rx_host.assign(rb, 0);
    }
    u32 relaxed = 0;
    Timer timer;
    timer.begin();
    TpsTrace tps;
    tps.start(opt.trace_tps);
    done = emit(next);
    while (!done && p + n_draft + 1 < opt.max_ctx) {
        const double c0 = nowMs();
        u32 nd_host = n_draft;
        if (opt.mtp_auto) {
            const q::DraftAccept* accs[1] = {&acc_model};
            nd_host = q::pickDrafts(accs, timing, n_draft, cycles, prev_nd);
        } else if (opt.mtp_adapt > 0) {
            const u32 want = static_cast<u32>(std::ceil(acc_ema + static_cast<float>(opt.mtp_adapt)));
            nd_host = std::clamp<u32>(want, 1, n_draft);
        }
        // n-gram drafts when the history's suffix occurred before and they
        // promise more tokens per ms than the MTP drafts
        std::array<u32, q::max_ng_drafts> ngd{};
        std::size_t ng_n = 0;
        const std::size_t room = std::min<std::size_t>(ng_max, opt.max_ctx > p + 2 ? opt.max_ctx - (p + 2) : 0);
        q::Ngram::Match dbg_mt;
        float dbg_score = 0;
        if (opt.ngram && room > 0) {
            const q::Ngram::Match mt = ngp.propose(hist, opt.ngram_min, std::span<u32>(ngd.data(), room));
            dbg_mt = mt;
            if (mt.n > 0) {
                const auto t = timing.estimate(nd_host);
                const float mtp_score = t ? acc_model.expected(nd_host) / *t : 0.0f;
                dbg_score = mtp_score;
                ng_n = opt.ngram_force ? mt.n : ngp.choose(mt.n, mt.mlen, mtp_score, timing);
            }
        }
        if (ng_n > 0) {
            nd_host = static_cast<u32>(ng_n);
            // MTP KV rows only; the verify takes the drafts from the host
            model.mtpEnqueue(model.mtp_h, std::span<const u32>(pend.data(), n_pend), 0, pend_pos, std::nullopt);
        } else {
            // drafts chain through their device slots into the verify ids
            model.mtpEnqueue(model.mtp_h, std::span<const u32>(pend.data(), n_pend), 0, pend_pos, 0u);
            for (u32 dri = 1; dri < nd_host; ++dri) {
                // chained drafts: the MTP's own head-normed output stands in for the main hidden
                model.mtpEnqueue(model.h, {}, model.draftSlot(dri - 1), p + dri, dri);
            }
        }
        if (ng_n > 0)
            model.verifyEnqueueEx(next, nd_host, std::span<const u32>(ngd.data(), ng_n), p);
        else
            model.verifyEnqueueEx(next, nd_host, std::nullopt, p);
        verify_tokens += nd_host + 1;
        if (ng_n == 0) prev_nd = nd_host;
        if (rx.on && nd_host > 0 && ng_n == 0) {
            const hip::DevPtr ids = rx_dev, vals = rx_dev + static_cast<u64>(q::max_small_batch) * rk * 4,
                         stats = rx_dev + static_cast<u64>(q::max_small_batch) * rk * 8;
            hip::launch(rx_topk, hip::Dim3{nd_host, 1, 1}, hip::Dim3{1024, 1, 1}, 0, model.stream, model.logits,
                        static_cast<std::int32_t>(model.cfg.n_vocab), static_cast<std::int32_t>(rk), 1.0f, ids, vals, stats);
            hip::downloadAsync(rx_host.data(), rx_dev, rx_host.size(), model.stream);
        }
        const double c1 = nowMs();
        std::array<u32, q::max_small_batch> dr{}, outv{};
        u32 nd;
        if (ng_n > 0) {
            (void)model.readCycle(std::span<u32>(outv.data(), nd_host + 1), std::span<u32>(dr.data(), 0));
            std::copy(ngd.begin(), ngd.begin() + static_cast<std::ptrdiff_t>(ng_n), dr.begin());
            nd = nd_host;
        } else {
            nd = model.readCycle(std::span<u32>(outv.data(), nd_host + 1), std::span<u32>(dr.data(), nd_host)).nd;
        }
        const double c2 = nowMs();
        // relaxacc: the draft-count policy sees a deterministic cycle cost, not wall-clock times
        // (the kept tokens depend on the drafts; see the server engine)
        const float cyc_ms = rx.on ? 100.0f * (1.0f + opt.ngram_slope * static_cast<float>(nd_host) + (ng_n > 0 ? 0.0f : 0.03f * static_cast<float>(nd_host)))
                                   : static_cast<float>(c2 - c0);
        t_draft += c1 - c0;
        t_verify += c2 - c1;
        cycles += 1;
        drafted += nd;
        u32 acc = 0;
        if (rx.on && nd > 0 && ng_n == 0) hip::streamSync(model.stream);
        while (acc < nd) {
            if (outv[acc] == dr[acc]) {
                acc += 1;
                continue;
            }
            if (!rx.on || ng_n > 0) break;  // n-gram drafts copy the history: exact only (loops)
            const auto* rids = reinterpret_cast<const std::int32_t*>(rx_host.data()) + static_cast<std::size_t>(acc) * rk;
            const auto* rvals = reinterpret_cast<const float*>(rx_host.data() + static_cast<std::size_t>(q::max_small_batch) * rk * 4) + acc * rk;
            const auto* rst = reinterpret_cast<const float*>(rx_host.data() + static_cast<std::size_t>(q::max_small_batch) * rk * 8);
            if (!whirl::relax::acceptGreedy(rx, std::span<const std::int32_t>(rids, rk), std::span<const float>(rvals, rk), rst[2 * acc], dr[acc]) ||
                whirl::relax::extendsRepeat(std::span<const u32>(generated), std::span<const u32>(dr.data(), acc + 1)))
                break;
            acc += 1;
            relaxed += 1;
        }
        accepted += acc;
        if (opt.ngram_debug) {
            u32 hyp = 0;
            while (hyp < dbg_mt.n && hyp <= acc && hyp < nd_host + 1 && ngd[hyp] == outv[hyp]) hyp += 1;
            const u32 b = q::NgramPolicy::bucket(dbg_mt.mlen);
            const auto e1 = ngp.timing.estimate(static_cast<u32>(std::max<std::size_t>(dbg_mt.n, 1)));
            const auto e15 = timing.estimate(15);
            std::fprintf(stderr, "CY ng=%zu prop=%zu mlen=%u acc=%u nd=%u ms=%.1f a0=%.2f a3=%.2f mtp_score=%.3f ngt=%.1f ngt15=%.1f hyp>=%u\n", ng_n,
                         dbg_mt.n, dbg_mt.mlen, acc, nd_host, c2 - c0, ngp.acc[b].alpha[0], ngp.acc[b].alpha[3], dbg_score, e1 ? *e1 : -1.0f,
                         e15 ? *e15 : -1.0f, hyp);
        }
        if (ng_n > 0) {
            // n-gram cycles stay out of the MTP acceptance / cycle-time models
            ng_cycles += 1;
            ng_drafted += nd;
            ng_accepted += acc;
            ngp.timing.update(nd_host, cyc_ms);
        } else {
            for (u32 kk = 0; kk < std::min(acc + 1, nd); ++kk) pos_tries[kk] += 1;
            for (u32 kk = 0; kk < acc; ++kk) pos_hits[kk] += 1;
            acc_ema = 0.7f * acc_ema + 0.3f * static_cast<float>(acc);
            acc_model.update(nd, acc);
            timing.update(nd_host, cyc_ms);
        }
        // the verify ran nd_host + 1 rows: keep acc + 1 of them
        if (acc < nd_host) model.restoreSnapshot(acc + 1);
        // emit accepted drafts plus the model's own next token (relaxacc: the accepted drafts
        // themselves, which may differ from the rows' argmax)
        for (u32 r = 0; r < acc + 1; ++r) {
            const u32 t = r < acc ? dr[r] : outv[r];
            if (!done) done = emit(t);
            if (opt.ngram) hist.push_back(t);
        }
        hip::copyAsync(model.mtp_h, model.hn, (acc + 1) * E * 4, model.stream);
        for (u32 r = 0; r < acc; ++r) pend[r] = dr[r];
        pend[acc] = outv[acc];
        n_pend = acc + 1;
        pend_pos = p + 1;
        next = outv[acc];
        p += acc + 1;
        tps.tick(generated.size());
        st.update(generated);
    }
    res.decode_ms = timer.end();
    model.draft_p_min = 0;
    if (rx_dev) hip::free(rx_dev);
    if (rx.on)
        std::fprintf(stderr, "relaxacc: %u of %u accepted drafts kept by the relaxed rule (top-%u, alpha %.2f)\n", relaxed, accepted, rk, rx.alpha);
    if (!opt.stream && opt.print_text) out(tok.decode(generated));
    res.cycles = cycles;
    res.accepted = accepted;
    res.drafted = drafted;
    res.verify_tokens = verify_tokens;
    res.pos_tries = pos_tries;
    res.pos_hits = pos_hits;
    res.ng_cycles = ng_cycles;
    res.ng_drafted = ng_drafted;
    res.ng_accepted = ng_accepted;
    res.ng_alpha = {ngp.acc[0].alpha[0], ngp.acc[1].alpha[0], ngp.acc[2].alpha[0]};
    res.t_draft = t_draft;
    res.t_verify = t_verify;
    res.n_draft = n_draft;
    res.n_min = n_min;
    res.p_min = p_min;
    return res;
}

// Speculative decoding with DFlash2 drafts (+ n-gram drafts): one draft block per cycle against
// the drafter context of slot 0, a verify of the literal drafts (the n-gram path's bitwise verify),
// then the kept rows' captured features go into the context. Output = plain greedy decoding.
DecodeResult dflashDecode(q::Model& model, const Tok& tok, u32 first, u32 n_prompt, const ChatOpts& opt, std::span<const u32> prompt_ids) {
    DecodeResult res;
    Streamer st{tok, opt.stream};
    std::vector<u32>& generated = res.tokens;
    u32 next = first;
    u32 p = n_prompt;  // position of `next`
    const u32 max_d = !model.fusedDecode() ? 1 : std::min<u32>(model.dfr->block - 1, q::max_ng_drafts);
    u32 ng_max = opt.ngram_max > 0 ? std::min(opt.ngram_max, q::max_ng_drafts) : q::max_ng_drafts;
    if (!model.fusedDecode()) ng_max = 1;
    {
        // a recurrent-state snapshot set per draft row: take them if they fit in free VRAM with
        // room to spare, else draft at most what exists
        const u32 want = std::max(max_d, opt.ngram ? ng_max : 0u);
        if (want > model.snap_sets) {
            u64 per = 0;
            for (u32 i = 0; i < model.cfg.n_layer; ++i)
                if (model.ssm_state[i] != 0) per += model.convBytes() + model.ssmBytes();
            const hip::MemInfo mem = hip::memInfo();
            const u64 extra = static_cast<u64>(want - model.snap_sets) * per;
            if (mem.free > extra + (1536ull << 20)) model.ensureSnapshots(want);
        }
    }
    const u32 d_max = std::min(max_d, std::max<u32>(model.snap_sets, 1));
    ng_max = std::min(ng_max, std::max<u32>(model.snap_sets, 1));
    q::DraftAccept dacc;
    q::DraftTiming timing;  // DFlash2 cycle times (drafting included) by draft count
    float nd_ema = static_cast<float>(d_max);
    std::array<u32, q::max_drafts> pos_tries{}, pos_hits{};
    q::NgramPolicy ngp;
    ngp.verify_slope = opt.ngram_slope;
    ngp.reset();
    std::vector<u32> hist;
    if (opt.ngram) {
        ngp.ng.norm = tok.crlfToLf();
        hist.assign(prompt_ids.begin(), prompt_ids.end());
        hist.push_back(first);
    }
    u32 cycles = 0, accepted = 0, drafted = 0, ng_cycles = 0, ng_drafted = 0, ng_accepted = 0;
    double t_draft = 0, t_verify = 0;
    std::size_t verify_tokens = 0;
    bool done = false;
    auto emit = [&](u32 t) {
        if (t == tok.im_end || t == tok.eot) return true;
        generated.push_back(t);
        return generated.size() >= opt.max_tokens;
    };
    Timer timer;
    timer.begin();
    TpsTrace tps;
    tps.start(opt.trace_tps);
    done = emit(next);
    while (!done && p + 2 < opt.max_ctx) {
        const double c0 = nowMs();
        const u32 room = std::min<u32>(std::max(d_max, ng_max), opt.max_ctx - (p + 2));
        std::array<u32, q::max_ng_drafts> dr{};
        u32 nd = 0;
        bool is_ng = false;
        if (opt.ngram && room > 0) {
            const q::Ngram::Match mt = ngp.propose(hist, opt.ngram_min, std::span<u32>(dr.data(), std::min(room, ng_max)));
            if (mt.n > 0) {
                const u32 dn = std::min({d_max, room, static_cast<u32>(std::lround(nd_ema))});
                const auto t = timing.estimate(dn);
                const float alt = t ? dacc.expected(dn) / *t : 0.0f;
                nd = opt.ngram_force ? static_cast<u32>(mt.n) : ngp.choose(mt.n, mt.mlen, alt, timing);
                is_ng = nd > 0;
            }
        }
        if (!is_ng && room > 0) {
            const std::vector<u32> d = model.dflashDraft(0, next, p, opt.dfl_p_min, opt.dfl_n_min);
            nd_ema = 0.8f * nd_ema + 0.2f * static_cast<float>(d.size());
            nd = std::min({static_cast<u32>(d.size()), d_max, room});
            std::copy(d.begin(), d.begin() + nd, dr.begin());
        }
        model.verifyEnqueueEx(next, nd, std::span<const u32>(dr.data(), nd), p);
        verify_tokens += nd + 1;
        const double c1 = nowMs();
        std::array<u32, q::max_small_batch> outv{};
        (void)model.readCycle(std::span<u32>(outv.data(), nd + 1), std::span<u32>(outv.data(), 0));
        const double c2 = nowMs();
        t_draft += c1 - c0;
        t_verify += c2 - c1;
        cycles += 1;
        drafted += nd;
        u32 acc = 0;
        while (acc < nd && outv[acc] == dr[acc]) acc += 1;
        accepted += acc;
        if (is_ng) {
            ng_cycles += 1;
            ng_drafted += nd;
            ng_accepted += acc;
            ngp.timing.update(nd, static_cast<float>(c2 - c0));
        } else {
            for (u32 kk = 0; kk < std::min(acc + 1, nd); ++kk) pos_tries[kk] += 1;
            for (u32 kk = 0; kk < acc; ++kk) pos_hits[kk] += 1;
            if (nd > 0) dacc.update(nd, acc);
            timing.update(nd, static_cast<float>(c2 - c0));
        }
        // the verify ran nd + 1 rows: keep acc + 1 of them, and their features
        if (acc < nd) model.restoreSnapshot(acc + 1);
        // (FR-DFL P2-2: via planIngest, so a capture shorter than the verify never reads before cap_buf)
        if (const whirl::dflash::Ingest g = whirl::dflash::verifyIngest(model.cap_n, nd + 1, acc + 1, p); g.any)
            model.dflashIngest(0, model.cap_buf, model.cap_rows, g.cap_row, g.n, g.pos);
        for (u32 r = 0; r < acc + 1; ++r) {
            if (!done) done = emit(outv[r]);
            if (opt.ngram) hist.push_back(outv[r]);
        }
        next = outv[acc];
        p += acc + 1;
        tps.tick(generated.size());
        st.update(generated);
    }
    res.decode_ms = timer.end();
    if (!opt.stream && opt.print_text) out(tok.decode(generated));
    res.cycles = cycles;
    res.accepted = accepted;
    res.drafted = drafted;
    res.verify_tokens = verify_tokens;
    res.pos_tries = pos_tries;
    res.pos_hits = pos_hits;
    res.ng_cycles = ng_cycles;
    res.ng_drafted = ng_drafted;
    res.ng_accepted = ng_accepted;
    res.ng_alpha = {ngp.acc[0].alpha[0], ngp.acc[1].alpha[0], ngp.acc[2].alpha[0]};
    res.t_draft = t_draft;
    res.t_verify = t_verify;
    res.n_draft = d_max;
    res.p_min = opt.dfl_p_min;
    res.n_min = opt.dfl_n_min;
    return res;
}

void printProfile(const char* label, q::Profile& p, std::size_t n_tok) {
    p.flush();
    double total = 0;
    for (double v : p.ms) total += v;
    const double per = static_cast<double>(std::max<std::size_t>(n_tok, 1));
    out(fmt("\n  [profile %s: %.1f ms GPU, %zu tok, %.2f ms/tok]\n", label, total, n_tok, total / per));
    for (std::size_t i = 0; i < q::n_classes; ++i)
        if (p.ms[i] > 0)
            out(fmt("    %-9s %9.1f ms  %7.3f ms/tok  %5.1f%%\n", q::opClassName(static_cast<q::OpClass>(i)), p.ms[i], p.ms[i] / per, 100.0 * p.ms[i] / total));
    const double mm = p.ms[static_cast<std::size_t>(q::OpClass::matmul)];
    if (mm > 0) out(fmt("    matmul weight stream %.0f GB/s\n", static_cast<double>(p.matmul_bytes) / (mm * 1e6)));
}

// ---------------------------------------------------------------------------
// shared setup


// Runtime A/B switches shared by chat and bench (WHIRL_FLOAT_GEMV, ...).
// --draft dflash: true when --draft-model FILE (else WHIRL_DRAFT_MODEL) is a DFlash2 drafter that fits this model
// (headers only, before loading); otherwise a warning and the MTP drafts (whirl/dflash_policy.h).
std::optional<std::string> draftModelOpt(const Args& a) {
    std::optional<std::string> cli;
    if (a.get("--draft-model")) cli = *a.get("--draft-model");
    return whirl::dflash::resolveDraftModel(cli, envGet("DRAFT_MODEL"));
}
bool wantDFlash(const Args& a, const gguf::File& f) {
    const std::string mode = a.get("--draft") ? *a.get("--draft") : std::string("mtp");
    const auto src = whirl::dflash::parseSource(mode);
    if (!src) throw std::runtime_error("--draft must be mtp or dflash");
    const auto dm = draftModelOpt(a);
    if (*src == whirl::dflash::Source::dflash) {
        std::optional<std::string> cli;
        if (a.get("--draft-model")) cli = *a.get("--draft-model");
        if (const std::string note = whirl::dflash::draftModelNote(cli, envGet("DRAFT_MODEL")); !note.empty())
            std::fprintf(stderr, "note: %s\n", note.c_str());
    }
    std::string why;
    if (*src == whirl::dflash::Source::dflash && dm) why = q::dflashIncompatibleFile(f, *dm);
    const whirl::dflash::Choice c = whirl::dflash::choose(*src, dm, why, true);
    if (!c.warning.empty()) std::fprintf(stderr, "warning: %s\n", c.warning.c_str());
    return c.source == whirl::dflash::Source::dflash;
}

// WHIRL_DRAFT_MODEL (--draft dflash): DFlash2 drafter GGUF, the hidden capture of its target layers
// and the drafter context of one sequence. false: the code object has no DFlash2 kernels, so
// Model::load kept the MTP block (warning printed; the caller drafts with MTP).
bool loadDraftModelEnv(q::Model& m, const Args& a) {
    const auto dm0 = draftModelOpt(a);
    if (!m.mtp_skipped) {
        const whirl::dflash::Choice c = whirl::dflash::choose(whirl::dflash::Source::dflash, dm0, "", false);
        std::fprintf(stderr, "warning: %s (missing: %s)\n", c.warning.c_str(), m.dflashMissingKernel().c_str());
        return false;
    }
    const auto dm = draftModelOpt(a);
    if (!dm || m.dflash) return m.dflash != nullptr;
    std::vector<std::string> warn;
    const whirl::dflash::Params dp = whirl::dflash::parseParams([](const char* n) { return envGet(n); }, warn);
    for (const std::string& w : warn) std::fprintf(stderr, "warning: %s\n", w.c_str());
    const hip::MemInfo d0 = hip::memInfo();
    const gguf::File df = gguf::File::open(*dm);
    q::LoadStats ds;
    const u64 wb = m.loadDrafter(df, ds);
    const auto& dc = m.dflash->cfg;
    m.enableCapture(whirl::dflash::captureRows(dp, dc.window, m.max_batch));
    m.dflashSetup(1, m.cap_rows);
    const hip::MemInfo d1 = hip::memInfo();
    std::fprintf(stderr, "draft model %s: DFlash2, %u layers, block %u, weights %.2f GiB, capture %u rows, VRAM +%.2f GiB (MTP block not loaded)\n",
                 dm->c_str(), dc.n_layer, dc.block, static_cast<double>(wb) / (1024.0 * 1024.0 * 1024.0), m.cap_rows,
                 static_cast<double>(d0.free > d1.free ? d0.free - d1.free : 0) / (1024.0 * 1024.0 * 1024.0));
    return true;
}

void applyRuntimeEnv(q::Model& m) {
    if (envGet("FLOAT_GEMV")) m.float_gemv = true;
    q::applyGemvEnv(m);
    if (auto v = envU32("DBG")) m.dbg_flags = *v;
    if (auto v = envU32("GEMV_MAX")) m.gemv_max = std::clamp<u32>(*v, 1, q::max_small_batch);
    if (envGet("NO_GRAPH")) m.use_graph = false;
    if (envGet("NO_FUSE")) m.no_fuse = true;
    if (auto v = envU32("DRAFT_WINDOW")) m.draft_window = *v;
    if (auto v = envU32("DRAFT_WINDOW_MIN")) m.draft_window_min = *v;
    if (envGet("NAIVE_ATTN")) m.naive_attn = true;
    if (auto v = envGet("ATTN_WIDE"); v && *v == "0") m.attn_wide = false;
    if (auto v = envGet("ATTN_DQ4"); v && *v == "0") m.attn_dq4 = false;
    if (auto v = envGet("DQ4_DYN"); v && *v == "0") m.dq4_dyn = false;
    if (auto v = envGet("ATTN_WS4"); v && *v == "0") m.attn_ws4 = false;
    if (auto v = envU32("MOE_BN")) m.moe_bn_force = *v;
    if (envGet("GDN_SEQ")) m.gdn_chunked = false;
    if (auto v = envGet("GV_GROUP")) q::gv_group = *v != "0";
    if (auto v = envU32("GV_NMAX")) q::gv_nmax = *v;
}

struct Loaded {
    std::unique_ptr<q::Model> model;
    q::LoadStats stats;
};

int selectDevice(const Args& a, std::string* info_line) {
    int dev;
    if (auto v = envGet("HIP_DEVICE"))
        dev = std::stoi(*v);
    else
        dev = q::pickDevice(a.get("--device") ? *a.get("--device") : std::string());
    const hip::DeviceInfo info = hip::describeDevice(dev);
    hip::setDevice(dev);
    if (info_line) *info_line = fmt("  whirl build, device %d: %s (%s)\n", dev, info.name.c_str(), info.gcn_arch.c_str());
    return dev;
}

// numerics mode: --precise / --balance[=items] / --fast[=items] / --mode M, else WHIRL_MODE, else balance
whirl::numerics::Request modeRequest(const Args& a) {
    return whirl::numerics::resolve(a.get("--mode") ? std::optional<std::string>(*a.get("--mode")) : std::nullopt, envGet("MODE"));
}

// ctx_explicit: the context was asked for (--ctx / WHIRL_MAX_CTX, bench sizes): precise f16 KV that
// does not fit refuses instead of shrinking to at least min_ctx tokens. no_mtp: --draft dflash
// (the MTP block is not loaded; the DFlash2 drafter drafts instead)
Loaded loadModel(const Args& a, const gguf::File& f, u32 max_ctx, bool ctx_explicit = true, u32 min_ctx = 0, bool no_mtp = false) {
    Loaded L;
    q::LoadOptions lo;
    lo.numerics = modeRequest(a);
    lo.ctx_explicit = ctx_explicit;
    lo.min_ctx = min_ctx;
    lo.no_mtp = no_mtp;
    if (auto v = envU32("PREFILL_BATCH")) lo.max_batch = std::max(q::max_batch_default, std::min(q::max_batch_limit, *v));
    lo.kv_mode = q::kvModeFromEnv().value_or(q::KvMode::automatic);
    L.model = q::Model::load(f, max_ctx, L.stats, lo);
    q::Model& m = *L.model;
    if (envGet("TUNE_COLD")) m.tune_cold = true;
    if (auto v = envGet("TUNE_MASK")) m.tune_mask = std::stoull(*v, nullptr, 0);
    return L;
}

void loadLog(q::Model& m, const q::LoadStats& stats) {
    const hip::MemInfo mem = hip::memInfo();
    out(fmt("  loaded %u tensors, %.2f GiB in %.1f s (%.2f GB/s); VRAM free %.2f/%.2f GiB\n", stats.tensors,
            static_cast<double>(stats.bytes) / (1024.0 * 1024.0 * 1024.0), stats.ms / 1000.0, static_cast<double>(stats.bytes) / (stats.ms * 1e6),
            static_cast<double>(mem.free) / (1024.0 * 1024.0 * 1024.0), static_cast<double>(mem.total) / (1024.0 * 1024.0 * 1024.0)));
    if (vram::limitBytes()) out(fmt("  %s: VRAM sizes and allocations are capped (simulated card)\n", vram::limitNote().c_str()));
    out(fmt("  KV cache: %s, %u tokens\n", m.kvName(), m.max_ctx));
    if (envGet("LOAD_DEBUG")) out(fmt("  gdn beta/alpha contiguous: %u layers\n", stats.gdn_ba_contig));
}

std::string readPromptArg(const std::string& p) {
    // "@path" reads the prompt from a file (Windows caps command lines at ~32k chars)
    if (p.size() > 1 && p[0] == '@') return readFile(p.substr(1));
    return p;
}

std::string chatWrap(const std::string& prompt, bool think) {
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n" + (think ? "<think>\n" : "<think>\n\n</think>\n\n");
}

bool thinkFromArgs(const Args& a) {
    bool think = true;
    if (auto v = envGet("THINK")) think = *v != "0";
    if (a.has("--no-think")) think = false;
    if (a.has("--think")) think = true;
    return think;
}

ChatOpts mtpOpts(const q::Model& model, ChatOpts base, u32* drafts_out) {
    const q::MtpDefaults def = q::mtpDefaults(model.cfg.moe, model.arch);
    u32 drafts = def.drafts;
    const auto dv = envGet("MTP_DRAFTS");
    if (dv) {
        try {
            drafts = std::clamp<u32>(static_cast<u32>(std::stoul(*dv)), 1, q::max_drafts);
        } catch (...) {
            drafts = def.drafts;
        }
    }
    base.trace_tps = envU32("TRACE_TPS").value_or(0);
    base.mtp_p_min = def.p_min;
    base.mtp_adapt = def.adapt;
    base.ngram_min = def.ngram_min;
    base.ngram_slope = def.ngram_slope;
    // an explicit draft count is used as is (fixed) unless WHIRL_MTP_ADAPT asks otherwise
    base.mtp_auto = def.automatic && !dv;
    if (auto v = envGet("MTP_PMIN")) {
        try {
            base.mtp_p_min = std::stof(*v);
        } catch (...) {
            base.mtp_p_min = 0;
        }
    }
    if (auto v = envGet("MTP_NMIN")) base.mtp_n_min = static_cast<u32>(std::strtoul(v->c_str(), nullptr, 10));
    if (auto v = envGet("MTP_ADAPT")) {
        base.mtp_auto = *v == "auto";
        base.mtp_adapt = static_cast<u32>(std::strtoul(v->c_str(), nullptr, 10));
    }
    if (auto v = envGet("NGRAM")) base.ngram = *v != "0";
    if (auto v = envGet("NGRAM_MIN")) base.ngram_min = static_cast<u32>(std::strtoul(v->c_str(), nullptr, 10));
    if (auto v = envGet("NGRAM_SLOPE")) base.ngram_slope = std::strtof(v->c_str(), nullptr);
    if (auto v = envGet("NGRAM_MAX")) base.ngram_max = static_cast<u32>(std::strtoul(v->c_str(), nullptr, 10));
    if (auto v = envGet("NGRAM_FORCE")) base.ngram_force = *v != "0";
    if (auto v = envGet("NGRAM_DEBUG")) base.ngram_debug = *v != "0";
    *drafts_out = drafts;
    return base;
}

// WHIRL_DFLASH_DUMP=DIR (whirl chat --draft dflash, GPU phase reference check): the target features
// the capture holds after the prompt (its last chunk: use --no-think so the prompt is not split
// before its last token) as DIR/feats.npy [n_ctx][n_target_layers * n_embd] f32 (the input layout of
// whirl-cloud tools/dflash_ref/ref_drafter.py) and DIR/dump_meta.txt (n_ctx, ctx_pos0, anchor,
// anchor_pos: its --ctx-pos0 / --anchor / --anchor-pos). export_check.py then writes the REFDIR of
// whirl dflash-check.
void dflashDumpFeatures(q::Model& m, const std::string& dir, u32 anchor, u32 anchor_pos) {
    if (!m.cap_buf || !m.dflash || m.cap_n == 0) {
        std::fprintf(stderr, "warning: WHIRL_DFLASH_DUMP: no captured rows\n");
        return;
    }
    const u64 E = m.cfg.n_embd, L = m.dflash->cfg.target_layers.size(), n = m.cap_n;
    std::vector<float> lm(static_cast<std::size_t>(L * n * E));
    hip::streamSync(m.stream);
    for (u64 j = 0; j < L; ++j)
        hip::download(lm.data() + j * n * E, m.cap_buf + j * m.cap_rows * E * 4, static_cast<std::size_t>(n * E * 4));
    std::vector<float> pm(lm.size());  // position-major: row r = concat over layers
    for (u64 r = 0; r < n; ++r)
        for (u64 j = 0; j < L; ++j) std::memcpy(&pm[(r * L + j) * E], &lm[(j * n + r) * E], static_cast<std::size_t>(E * 4));
    std::filesystem::create_directories(std::filesystem::path(widen(dir)));
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(n) + ", " + std::to_string(L * E) + "), }";
    while ((10 + hdr.size() + 1) % 64 != 0) hdr += ' ';
    hdr += '\n';
    std::string npy = "\x93NUMPY";
    npy += static_cast<char>(1);
    npy += static_cast<char>(0);
    npy += static_cast<char>(hdr.size() & 0xff);
    npy += static_cast<char>((hdr.size() >> 8) & 0xff);
    npy += hdr;
    npy.append(reinterpret_cast<const char*>(pm.data()), pm.size() * 4);
    writeFile(dir + "/feats.npy", npy);
    writeFile(dir + "/dump_meta.txt", "n_ctx " + std::to_string(n) + "\nctx_pos0 " + std::to_string(anchor_pos - n) + "\nanchor " +
                                          std::to_string(anchor) + "\nanchor_pos " + std::to_string(anchor_pos) + "\n");
    std::fprintf(stderr, "WHIRL_DFLASH_DUMP: %llu rows x %llu target layers -> %s (anchor %u at %u)\n", static_cast<unsigned long long>(n),
                 static_cast<unsigned long long>(L), dir.c_str(), anchor, anchor_pos);
}

// --draft dflash: the walk cutoffs (whirl/dflash_policy.h) and the n-gram settings, with the same
// per-model defaults and environment as the MTP path (mtpOpts)
ChatOpts dflashOpts(const q::Model& model, ChatOpts base) {
    u32 unused = 0;
    base = mtpOpts(model, base, &unused);
    std::vector<std::string> warn;
    const whirl::dflash::Params dp = whirl::dflash::parseParams([](const char* n) { return envGet(n); }, warn);
    for (const std::string& w : warn) std::fprintf(stderr, "warning: %s\n", w.c_str());
    base.dfl_p_min = dp.p_min;
    base.dfl_n_min = dp.n_min;
    return base;
}

void printSpecSummary(const DecodeResult& r, std::size_t n_prompt, double prefill_ms, const ChatOpts& opt, const char* kind = "MTP spec") {
    const std::size_t n_gen = r.tokens.size();
    const bool mtp = std::string_view(kind) == "MTP spec";
    out(fmt("\n\n  prefill: %zu tok in %.0f ms (%.1f tok/s)\n", n_prompt, prefill_ms, n_prompt * 1000.0 / prefill_ms));
    out(fmt("  decode (%s, max %u min %u p-min %.2f%s): %zu tok in %.0f ms (%.2f tok/s); %u verify cycles, drafts %u/%u accepted (%.2f tok/cycle)\n",
            kind, r.n_draft, r.n_min, r.p_min, !mtp ? "" : opt.mtp_auto ? " auto" : opt.mtp_adapt > 0 ? " adaptive" : "", n_gen, r.decode_ms, n_gen * 1000.0 / r.decode_ms,
            r.cycles, r.accepted, r.drafted, r.cycles > 0 ? static_cast<double>(n_gen) / r.cycles : 0.0));
    if (r.cycles > 0) {
        std::string line = "  acceptance by draft position (accepted/reached):";
        for (u32 kk = 0; kk < r.n_draft; ++kk) {
            if (r.pos_tries[kk] == 0) break;
            line += fmt(" %u:%.0f%%(%u)", kk + 1, 100.0 * r.pos_hits[kk] / r.pos_tries[kk], r.pos_tries[kk]);
        }
        out(line + "\n");
        if (opt.ngram)
            out(fmt("  n-gram drafts: %u of %u cycles, drafts %u/%u accepted; n-gram acceptance model (<8 / 8-23 / >=24 matched): %.2f %.2f %.2f\n",
                    r.ng_cycles, r.cycles, r.ng_accepted, r.ng_drafted, r.ng_alpha[0], r.ng_alpha[1], r.ng_alpha[2]));
        out(fmt("  per cycle: host enqueue %.2f ms, wait %.2f ms, other %.2f ms; verified rows %.2f\n", r.t_draft / r.cycles, r.t_verify / r.cycles,
                (r.decode_ms - r.t_draft - r.t_verify) / r.cycles, static_cast<double>(r.verify_tokens) / r.cycles));
    }
}

// ---------------------------------------------------------------------------
// chat

int cmdChat(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("chat: missing MODEL.gguf");
    const std::string path = a.pos[0];
    ChatOpts opt;
    if (auto v = a.get("--max-tokens")) opt.max_tokens = static_cast<u32>(std::stoul(*v));
    if (auto v = envU32("MAX_CTX")) opt.max_ctx = *v;
    if (auto v = a.get("--ctx")) opt.max_ctx = static_cast<u32>(std::stoul(*v));
    opt.stream = !a.has("--no-stream");
    const bool raw = a.has("--raw");
    const bool think = thinkFromArgs(a);
    const std::string* logits_out = a.get("--out");

    gguf::File f = gguf::File::open(path);
    const q::Config cfg = q::Config::fromGguf(f);
    std::string dev_line;
    selectDevice(a, &dev_line);
    out(fmt("whirl gguf: %s\n  arch %s  layers %u (attn every %u)  embd %u  ff %u  vocab %u\n", path.c_str(), cfg.archName(), cfg.n_layer, cfg.interval,
            cfg.n_embd, cfg.n_ff, cfg.n_vocab));
    if (cfg.moe) out(fmt("  experts %u (top %u, ff %u) + shared expert ff %u\n", cfg.n_expert, cfg.n_expert_used, cfg.n_ff_exp, cfg.n_ff));
    out(dev_line);

    Tok tok(f);
    std::vector<u32> ids;
    if (auto v = a.get("--tokens")) {
        ids = parseIds(*v);
    } else {
        if (a.pos.size() < 2) throw std::runtime_error("chat: missing PROMPT");
        const std::string p = readPromptArg(a.pos[1]);
        // vision: one image placeholder per --image before the text (as llama-mtmd-cli)
        std::string marks;
        if (!raw)
            for (std::size_t i = 0; i < a.images.size(); ++i) marks += "<|vision_start|><|image_pad|><|vision_end|>";
        ids = tok.encode(raw ? p : chatWrap(marks + p, think));
    }
#if WHIRL_HAVE_VISION
    // vision: decode / preprocess the images, expand their placeholders
    std::unique_ptr<vision::Vision> vis;
    std::vector<vision::Prepared> preps;
    std::array<q::VisSpan, 16> spans{};
    q::VisMap vmap;
    // projected embeddings in pinned host memory (as the prototype CLI)
    struct Pinned {
        float* p = nullptr;
        std::size_t n = 0;
        explicit Pinned(std::size_t count) : p(static_cast<float*>(hip::hostMalloc(count * 4))), n(count) {}
        Pinned(Pinned&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
        Pinned(const Pinned&) = delete;
        ~Pinned() {
            if (p) hip::hostFree(p);
        }
    };
    std::vector<Pinned> embs;
    if (!a.images.empty()) {
        if (a.images.size() > spans.size()) throw std::runtime_error("TooManyImages");
        const std::string* mm = a.get("--mmproj");
        if (mm == nullptr) {
            out("--image needs --mmproj MMPROJ.gguf\n");
            throw std::runtime_error("MissingMmproj");
        }
        const double tv = nowMs();
        vis = vision::Vision::load(*mm);
        out(fmt("  mmproj: %s (%.0f MiB pinned, %.0f ms)\n", mm->c_str(), static_cast<double>(vis->pinnedBytes()) / 1048576.0, nowMs() - tv));
        for (const std::string& ip : a.images) {
            const std::string bytes = readFile(ip);
            preps.push_back(vis->prepare(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size())));
            const vision::Prepared& pp = preps.back();
            out(fmt("  image %s: %ux%u -> %u tokens (%ux%u)\n", ip.c_str(), pp.rgb.w, pp.rgb.h, pp.nTokens(), pp.nx, pp.ny));
        }
        std::vector<const vision::Prepared*> ptrs;
        for (const vision::Prepared& pp : preps) ptrs.push_back(&pp);
        const TokenId pad = tok.t.find("<|image_pad|>");
        if (pad < 0) throw std::runtime_error("NoImagePadToken");
        ids = vision::expand(ids, static_cast<u32>(pad), ptrs, std::span<q::VisSpan>(spans.data(), preps.size()));
        vmap.spans = std::span<const q::VisSpan>(spans.data(), preps.size());
    }
#else
    if (!a.images.empty()) throw std::runtime_error("this build has no image input (vision)");
#endif
    if (envGet("TOKENIZE_ONLY")) {
        std::string s;
        for (std::size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
        out(s + "\n");
        return 0;
    }
    if (ids.empty()) throw q::ModelError("EmptyPrompt");
    if (ids.size() + opt.max_tokens > opt.max_ctx) throw q::ModelError("ContextTooLong");

    const bool ctx_given = a.get("--ctx") != nullptr || envU32("MAX_CTX").has_value();
    bool use_dflash = logits_out == nullptr && wantDFlash(a, f);
    Loaded L = loadModel(a, f, opt.max_ctx, ctx_given, static_cast<u32>(ids.size()) + opt.max_tokens, use_dflash);
    opt.max_ctx = L.model->max_ctx;  // precise: may be lowered to what f16 KV fits
    q::Model& model = *L.model;
    {
        std::string log;
        q::loadOrTune(model, path, log);
        out(log);
    }
    if (envGet("GDN_V0")) {
        if (auto fv0 = model.module.getFunctionOpt("gdn_step_norm_v0")) model.k.gdn_step_norm = fv0;
    }
    // MTP block as Q4_K from Q6_K or Q8_0 (drafts only; default, WHIRL_MTP_Q4=0 keeps the file's types)
    if (envFlag("MTP_Q4", true)) model.requantMtpQ4();
    loadLog(model, L.stats);
#if WHIRL_HAVE_VISION
    // vision: encode the images now (borrowing the prefill scratch), then let the
    // prefill inject their rows and use multi-section RoPE
    if (vis) {
        if (auto mv = envGet("VIS_MODE")) vis->mode = vision::parseMode(*mv).value_or(vision::Mode::automatic);
        std::array<std::array<std::uint64_t, 2>, 16> lend_raw{};
        const std::size_t n_lend = model.lendScratch(lend_raw);
        std::vector<vision::Region> lend;
        for (std::size_t i = 0; i < n_lend; ++i) lend.push_back({lend_raw[i][0], lend_raw[i][1]});
        for (std::size_t i = 0; i < preps.size(); ++i) {
            const vision::Prepared& pp = preps[i];
            embs.emplace_back(static_cast<std::size_t>(pp.nTokens()) * vis->hp.proj_dim);
            const vision::EncodeStats st = vis->encode(pp, lend, model.stream, embs.back().p, embs.back().n);
            spans[i].emb = embs.back().p;
            out(fmt("  image %zu: encoded %u tokens in %.1f ms (%s%s%s)\n", i, pp.nTokens(), st.ms_total, st.resident ? "resident weights" : "streamed weights",
                    st.ms_upload > 0 ? ", incl. upload" : "", st.borrowed_extra > 0 ? ", extra VRAM" : ""));
            if (auto dp = envGet("VIS_DUMP")) {
                const Pinned& e = embs.back();
                writeFile(*dp + "." + std::to_string(i) + ".f32", std::string_view(reinterpret_cast<const char*>(e.p), e.n * 4));
            }
        }
        model.scratchClobbered();
        model.vis_single = &vmap;
    }
#endif
    out(fmt("  prompt tokens: %zu\n\n", ids.size()));

    // (CLI-SCHED: not below one schedule chunk + its merged tail, as the server's vram::min_prefill_batch)
    if (auto v = envU32("PREFILL_BATCH"))
        model.max_batch = std::max<u32>(sched::prefill_chunk + sched::prefill_merge, std::min(model.max_batch, *v));
    q::Profile prof;
    const bool profiling = envGet("PROFILE").has_value();
    if (profiling) model.prof = &prof;
    applyRuntimeEnv(model);
    if (use_dflash) use_dflash = loadDraftModelEnv(model, a);

    // MTP speculative decode is exact greedy: on whenever the checkpoint has a nextn
    // layer; WHIRL_MTP=0 turns it off.
    const bool use_mtp = model.mtp.has_value() && logits_out == nullptr && envFlag("MTP", true);
    Timer timer;
    timer.begin();
    std::vector<std::int32_t> moe_ids;
    const auto moe_dump_path = envGet("DUMP_MOE");
    if (moe_dump_path) model.moe_dump = &moe_ids;
    const u32 tps_every = envU32("TRACE_TPS").value_or(0);
    {
        // CLI-SCHED: the server's forwards for this prompt (serverForwardEnds: chunk schedule, think-open /
        // system / image splits). WHIRL_DUMP_LOGITS keeps its own teacher-forced forwards (16-row windows,
        // max_batch chunks between them; a thinking-mode prompt's last token as its own step).
        const std::size_t split = tok.thinkOpenSplit(ids);
        // WHIRL_DUMP_LOGITS=path, WHIRL_DUMP_FROM=N (no MTP): next-token logits of every
        // prompt position >= N as f16 rows (teacher-forced, 16-row forwards).
        // WHIRL_DUMP_AT=E1,E2,... instead: the WHIRL_DUMP_ROWS (default 128) positions before each
        // end Ei (rows in that order; the KL gate, tools/klgate, measures 8k..128k in one run)
        const auto dump_path = envGet("DUMP_LOGITS");
        std::vector<std::pair<std::size_t, std::size_t>> dump_win;  // [start, end) windows, sorted
        if (dump_path) {
            if (const auto at = envGet("DUMP_AT")) {
                const std::size_t rows = envU32("DUMP_ROWS").value_or(128);
                std::size_t prev = 0;
                for (std::size_t p = 0; p < at->size();) {
                    std::size_t c = at->find(',', p);
                    if (c == std::string::npos) c = at->size();
                    const std::size_t e = std::stoul(at->substr(p, c - p));
                    if (e < rows || e > ids.size() || e - rows < prev) throw std::runtime_error("WHIRL_DUMP_AT: ends must be ascending, >= WHIRL_DUMP_ROWS apart and <= the prompt length");
                    dump_win.emplace_back(e - rows, e);
                    prev = e;
                    p = c + 1;
                }
            } else {
                dump_win.emplace_back(envU32("DUMP_FROM").value_or(0), ids.size());
            }
        }
        std::vector<std::uint16_t> dump;
        std::vector<float> row_f32(dump_path ? static_cast<std::size_t>(q::max_small_batch) * model.cfg.n_vocab : 0);
        const double p0 = nowMs();
        const std::vector<std::size_t> ends = dump_path ? std::vector<std::size_t>{} : serverForwardEnds(tok, ids, model.max_batch, model.arch == hip::Arch::gfx1151, model.cfg.moe);
        std::size_t off = 0, ei = 0;
        while (off < ids.size()) {
            bool dumping = false;
            std::size_t lim = (split > 0 && off < split) ? split : ids.size();
            if (dump_path && !use_mtp)
                for (const auto& [ws, we] : dump_win) {
                    if (off >= ws && off < we) {
                        dumping = true;
                        lim = std::min(lim, we);
                        break;
                    }
                    if (off < ws) {
                        lim = std::min(lim, ws);
                        break;
                    }
                }
            const std::size_t n = !dump_path ? ends[ei++] - off
                                  : dumping  ? std::min<std::size_t>(lim - off, q::max_small_batch)
                                             : std::min<std::size_t>(lim - off, model.max_batch);
            const double c0 = nowMs();
            // all_logits needs keep_hidden (normed rows in hn); without it forward() computes only
            // the chunk's last row (into row 0) and the dump held 1 real row per 16
            model.all_logits = dumping;
            model.keep_hidden = dumping;
            if (use_mtp)
                model.prefillMtpChunk(ids, 0, off, n, std::nullopt);
            else
                model.forward(std::span<const u32>(ids).subspan(off, n), static_cast<u32>(off));
            model.all_logits = false;
            model.keep_hidden = false;
            // DFlash2 context: the chunk's captured rows (its last cap_n positions)
            if (use_dflash) model.dflashIngest(0, model.cap_buf, model.cap_rows, 0, model.cap_n, static_cast<u32>(off + n - model.cap_n));
            if (dumping) {
                const std::size_t V = model.cfg.n_vocab;
                hip::download(row_f32.data(), model.logits, n * V * 4);
                for (std::size_t i = 0; i < n * V; ++i) {
                    // f32 -> f16 bits, round to nearest even (halfs of logits are normal-range)
                    std::uint32_t u;
                    std::memcpy(&u, &row_f32[i], 4);
                    const std::uint32_t sign = (u >> 16) & 0x8000;
                    int e = static_cast<int>((u >> 23) & 0xff) - 112;
                    std::uint32_t hm = (u & 0x7fffff) >> 13;
                    const std::uint32_t rem = u & 0x1fff;
                    std::uint16_t hb;
                    if (e <= 0)
                        hb = static_cast<std::uint16_t>(sign);
                    else if (e >= 31)
                        hb = static_cast<std::uint16_t>(sign | 0x7c00);
                    else {
                        if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) {
                            hm += 1;
                            if (hm == 0x400) {
                                hm = 0;
                                e += 1;
                            }
                        }
                        hb = static_cast<std::uint16_t>(e >= 31 ? (sign | 0x7c00) : (sign | (static_cast<std::uint32_t>(e) << 10) | hm));
                    }
                    dump.push_back(hb);
                }
            }
            if (tps_every > 0) {
                hip::sync();
                const double c1 = nowMs();
                std::fprintf(stderr, "  [tps] prefill t=%.1f s  chunk %zu+%zu  %.1f tok/s\n", (c1 - p0) / 1000.0, off, n, n / ((c1 - c0) / 1000.0));
            }
            off += n;
        }
        if (dump_path) writeFile(*dump_path, std::string_view(reinterpret_cast<const char*>(dump.data()), dump.size() * 2));
    }
    model.moe_dump = nullptr;
    if (moe_dump_path) writeFile(*moe_dump_path, std::string_view(reinterpret_cast<const char*>(moe_ids.data()), moe_ids.size() * 4));
    const u32 next = model.beginDecode(static_cast<u32>(ids.size()));
    const double prefill_ms = timer.end();
    if (use_dflash) {
        if (profiling) {
            printProfile("prefill", prof, ids.size());
            prof.reset();
        }
        if (auto dir = envGet("DFLASH_DUMP")) dflashDumpFeatures(model, *dir, next, static_cast<u32>(ids.size()));
        const ChatOpts dopt = dflashOpts(model, opt);
        const DecodeResult r = dflashDecode(model, tok, next, static_cast<u32>(ids.size()), dopt, ids);
        printSpecSummary(r, ids.size(), prefill_ms, dopt, "DFlash2 spec");
        if (envGet("PRINT_IDS")) {
            std::string s = "  ids:";
            for (u32 t : r.tokens) s += " " + std::to_string(t);
            out(s + fmt("\n  hash: %016llx\n", static_cast<unsigned long long>(fnv(r.tokens))));
        }
        if (profiling) printProfile("verify (per verified token)", prof, r.verify_tokens);
        return 0;
    }
    if (use_mtp) {
        if (profiling) {
            printProfile("prefill", prof, ids.size());
            prof.reset();
        }
        // WHIRL_DRAFT_HEAD=q4 builds the older Q4_K draft head instead of the 2-bit one
        if (!envGet("MTP_FULLHEAD")) {
            const auto dh = envGet("DRAFT_HEAD");
            model.buildDraftHeadEx(dh && *dh == "q4" ? q::Model::DraftHeadKind::q4 : q::Model::DraftHeadKind::d2);
        }
        applyDraftVocab(model, tok.t);
        u32 drafts = 0;
        const ChatOpts sopt = mtpOpts(model, opt, &drafts);
        const DecodeResult r = specDecode(model, tok, next, static_cast<u32>(ids.size()), sopt, drafts, ids);
        printSpecSummary(r, ids.size(), prefill_ms, sopt);
        if (envGet("PRINT_IDS")) {
            std::string s = "  ids:";
            for (u32 t : r.tokens) s += " " + std::to_string(t);
            out(s + fmt("\n  hash: %016llx\n", static_cast<unsigned long long>(fnv(r.tokens))));
        }
        if (profiling) printProfile("verify (per verified token)", prof, r.verify_tokens);
        return 0;
    }
    if (profiling) {
        printProfile("prefill", prof, ids.size());
        prof.reset();
    }
    if (logits_out) {
        std::vector<float> lg(cfg.n_vocab);
        model.readLogits(lg);
        writeFile(*logits_out, std::string_view(reinterpret_cast<const char*>(lg.data()), lg.size() * 4));
        out(fmt("  wrote logits (%u f32) to %s; argmax %u\n", cfg.n_vocab, logits_out->c_str(), next));
    }
    opt.trace_tps = tps_every;
    const DecodeResult r = plainDecode(model, tok, next, ids.size(), opt);
    if (profiling && r.tokens.size() > 1) printProfile("decode", prof, r.tokens.size() - 1);
    out(fmt("\n\n  prefill: %zu tok in %.0f ms (%.1f tok/s)\n", ids.size(), prefill_ms, ids.size() * 1000.0 / prefill_ms));
    if (r.steps > 0)
        out(fmt("  decode:  %u tok in %.0f ms (%.2f tok/s, %.1f ms/tok)\n", r.steps, r.decode_ms, r.steps * 1000.0 / r.decode_ms, r.decode_ms / r.steps));
    if (envGet("PRINT_IDS")) {
        std::string s = "  ids:";
        for (u32 t : r.tokens) s += " " + std::to_string(t);
        out(s + fmt("\n  hash: %016llx\n", static_cast<unsigned long long>(fnv(r.tokens))));
    }
    return 0;
}

// ---------------------------------------------------------------------------
// bench: prefill sweep + decode plain / MTP / MTP + n-gram

const char* k_bench_text =
    "請用繁體中文說明下面這段 Python 程式的用途，並指出可以改進的地方。\n"
    "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n"
    "    out.extend(a[i:]); out.extend(b[j:])\n    return out\n"
    "Then rewrite it in C++20 with std::span inputs, add unit tests, and explain the complexity in English.\n";

int cmdBench(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("bench: missing MODEL.gguf");
    const std::string path = a.pos[0];
    std::vector<u32> sizes = {2048, 8192, 32768};
    if (auto v = a.get("--prefill")) sizes = parseIds(*v);
    const u32 n_dec = a.get("--decode") ? static_cast<u32>(std::stoul(*a.get("--decode"))) : 256;
    std::vector<std::string> modes = {"mtp-ngram", "mtp", "plain"};
    const bool dflash_req = a.get("--draft") && *a.get("--draft") == "dflash";
    if (dflash_req) modes = {"dflash-ngram", "dflash", "plain"};
    if (auto v = a.get("--modes")) {
        modes.clear();
        std::size_t p = 0;
        while (p < v->size()) {
            std::size_t e = v->find(',', p);
            if (e == std::string::npos) e = v->size();
            modes.push_back(v->substr(p, e - p));
            p = e + 1;
        }
    }
    const bool think = thinkFromArgs(a);
    gguf::File f = gguf::File::open(path);
    const q::Config cfg = q::Config::fromGguf(f);
    std::string dev_line;
    selectDevice(a, &dev_line);
    out(fmt("whirl bench: %s (%s)\n", path.c_str(), cfg.archName()));
    out(dev_line);
    Tok tok(f);
    const std::string dec_prompt = a.get("--prompt") ? readPromptArg(*a.get("--prompt")) : std::string(k_bench_text);
    const std::vector<u32> dec_ids = tok.encode(chatWrap(dec_prompt, think));
    u32 max_size = 0;
    for (u32 s : sizes) max_size = std::max(max_size, s);
    const u32 ctx = std::max<u32>(max_size + 64, static_cast<u32>(dec_ids.size()) + n_dec + 64);
    bool use_dflash = wantDFlash(a, f);
    if (dflash_req && !use_dflash && !a.get("--modes")) modes = {"mtp-ngram", "mtp", "plain"};
    Loaded L = loadModel(a, f, ctx, true, 0, use_dflash);
    q::Model& model = *L.model;
    {
        std::string log;
        q::loadOrTune(model, path, log);
        out(log);
    }
    if (envFlag("MTP_Q4", true)) model.requantMtpQ4();
    loadLog(model, L.stats);
    applyRuntimeEnv(model);
    if (use_dflash) {
        use_dflash = loadDraftModelEnv(model, a);
        if (!use_dflash && !a.get("--modes")) modes = {"mtp-ngram", "mtp", "plain"};
    }
    const bool has_mtp = model.mtp.has_value() && envFlag("MTP", true);

    // prefill sweep: a long mixed zh/en coding text, truncated to each size
    std::vector<u32> long_ids;
    {
        const std::size_t per = std::max<std::size_t>(1, tok.encode(std::string(k_bench_text) + "\n").size());
        std::string text;
        for (std::size_t i = 0; i < max_size / per + 2; ++i) text += std::string(k_bench_text) + "\n";
        long_ids = tok.encode(text);
    }
    out("\n  prefill (tokens, ms, tok/s)" + std::string(has_mtp ? " - with the MTP block over the prompt, as chat runs it" : "") + ":\n");
    // WHIRL_PROFILE: per-op-class GPU time of the last prefill run of each size
    // (per-op events slow the run down; the printed tok/s is then not comparable)
    q::Profile prof;
    const bool profiling = envGet("PROFILE").has_value();
    Timer timer;
    // CLI-ATTN-GAP: the prefill runs on the server's / chat's forwards (serverForwardEnds: gfx1151 dense
    // 2048 rows, WHIRL_PREFILL_CHUNK applies), as whirl chat and whirl-server run a prompt. It used to run
    // max_batch (4096) row forwards, which on gfx1151 dense cost 13% at 128k (SRV-PP-DENSE), so bench
    // under-reported chat / server prefill. WHIRL_BENCH_MAX_BATCH=1: the old max_batch forwards.
    const bool bench_max_batch = envFlag("BENCH_MAX_BATCH", false);
    out(fmt("  prefill forwards: %s\n", bench_max_batch ? "max_batch rows (WHIRL_BENCH_MAX_BATCH=1)" : "the server's schedule (as whirl chat / whirl-server)"));
    for (u32 s : sizes) {
        if (s > long_ids.size()) continue;
        const std::span<const u32> pr(long_ids.data(), s);
        double best = 1e30;
        const int reps = s <= 8192 ? 2 : 1;
        for (int rep = 0; rep < reps + 1; ++rep) {  // first run = warm-up
            model.reset();
            if (profiling) {
                prof.reset();
                model.prof = rep == reps ? &prof : nullptr;
            }
            timer.begin();
            if (bench_max_batch) {
                std::size_t off = 0;
                while (off < pr.size()) {
                    const std::size_t n = std::min<std::size_t>(pr.size() - off, model.max_batch);
                    if (has_mtp)
                        model.prefillMtpChunk(pr, 0, off, n, std::nullopt);
                    else
                        model.forward(pr.subspan(off, n), static_cast<u32>(off));
                    off += n;
                }
            } else {
                prefillLikeServer(model, tok, pr, has_mtp);
            }
            (void)model.beginDecode(s);
            const double ms = timer.end();
            if (rep > 0 || reps == 0) best = std::min(best, ms);
        }
        out(fmt("    prefill %6u tok: %9.1f ms  %8.1f tok/s\n", s, best, s * 1000.0 / best));
        if (profiling) {
            printProfile("prefill", prof, s);
            model.prof = nullptr;
        }
    }
    // decode
    out(fmt("\n  decode (%zu-token prompt, %u tokens, greedy):\n", dec_ids.size(), n_dec));
    ChatOpts opt;
    opt.max_tokens = n_dec;
    opt.max_ctx = ctx;
    opt.stream = false;
    opt.print_text = false;
    if (has_mtp && !envGet("MTP_FULLHEAD")) {
        const auto dh = envGet("DRAFT_HEAD");
        model.buildDraftHeadEx(dh && *dh == "q4" ? q::Model::DraftHeadKind::q4 : q::Model::DraftHeadKind::d2);
    }
    if (has_mtp) applyDraftVocab(model, tok.t);
    std::optional<u64> ref_hash;
    for (const std::string& mode : modes) {
        const bool dfl_mode = mode == "dflash" || mode == "dflash-ngram";
        if (dfl_mode ? !use_dflash : (mode != "plain" && !has_mtp)) continue;
        model.reset();
        model.decode_graph.reset();
        Timer pt;
        pt.begin();
        if (mode == "plain") {
            model.prefill(dec_ids, 0);
        } else if (dfl_mode) {
            // a fresh drafter context, filled chunk by chunk from the capture
            model.dflashTrim(0, 0);
            std::size_t off = 0;
            while (off < dec_ids.size()) {
                const std::size_t n = std::min<std::size_t>(dec_ids.size() - off, model.max_batch);
                model.forward(std::span<const u32>(dec_ids).subspan(off, n), static_cast<u32>(off));
                model.dflashIngest(0, model.cap_buf, model.cap_rows, 0, model.cap_n, static_cast<u32>(off + n - model.cap_n));
                off += n;
            }
        } else {
            std::size_t off = 0;
            while (off < dec_ids.size()) {
                const std::size_t n = std::min<std::size_t>(dec_ids.size() - off, model.max_batch);
                model.prefillMtpChunk(dec_ids, 0, off, n, std::nullopt);
                off += n;
            }
        }
        const u32 next = model.beginDecode(static_cast<u32>(dec_ids.size()));
        (void)pt.end();
        // the generated text goes to a scratch string: only the numbers are printed
        std::FILE* saved = stdout;
        (void)saved;
        DecodeResult r;
        if (mode == "plain") {
            ChatOpts po = opt;
            r = plainDecode(model, tok, next, dec_ids.size(), po);
            const double tps = r.steps > 0 ? r.steps * 1000.0 / r.decode_ms : 0;
            out(fmt("\n    plain (no MTP)  : %4u tok in %7.0f ms  %7.2f tok/s  hash %016llx\n", r.steps, r.decode_ms, tps,
                    static_cast<unsigned long long>(fnv(r.tokens))));
        } else if (dfl_mode) {
            ChatOpts dop = dflashOpts(model, opt);
            dop.ngram = mode == "dflash-ngram";
            r = dflashDecode(model, tok, next, static_cast<u32>(dec_ids.size()), dop, dec_ids);
            out(fmt("\n    %-16s: %4zu tok in %7.0f ms  %7.2f tok/s  %.2f tok/cycle  drafts %u/%u  hash %016llx\n",
                    mode == "dflash" ? "DFlash2" : "DFlash2 + n-gram", r.tokens.size(), r.decode_ms, r.tokens.size() * 1000.0 / r.decode_ms,
                    r.cycles ? static_cast<double>(r.tokens.size()) / r.cycles : 0.0, r.accepted, r.drafted,
                    static_cast<unsigned long long>(fnv(r.tokens))));
        } else {
            u32 drafts = 0;
            ChatOpts so = mtpOpts(model, opt, &drafts);
            so.ngram = mode == "mtp-ngram";
            r = specDecode(model, tok, next, static_cast<u32>(dec_ids.size()), so, drafts, dec_ids);
            out(fmt("\n    %-16s: %4zu tok in %7.0f ms  %7.2f tok/s  %.2f tok/cycle  hash %016llx\n", mode == "mtp" ? "MTP" : "MTP + n-gram", r.tokens.size(),
                    r.decode_ms, r.tokens.size() * 1000.0 / r.decode_ms, r.cycles ? static_cast<double>(r.tokens.size()) / r.cycles : 0.0,
                    static_cast<unsigned long long>(fnv(r.tokens))));
        }
        const u64 hsh = fnv(r.tokens);
        if (!ref_hash) ref_hash = hsh;
        if (*ref_hash != hsh) out("    WARNING: token stream differs from the first mode (greedy outputs must match)\n");
    }
    return 0;
}

// Multi-sequence (server) paths against the single-sequence reference:
// 1. solo per sequence: chunked prefill + greedy one-token forwards;
// 2. both sequences: segmented prefill (prefillSegs) + batched decode rows
//    (verifyBatchEnqueue, no drafts), then a batched verify with literal
//    drafts taken from the reference continuation (all must be accepted).
// Every token must match the solo run.
u64 fnvF32(std::span<const float> v);

// logits hash of row r of the last forward
u64 rowHash(q::Model& m, u32 r) {
    std::vector<float> lg(m.cfg.n_vocab);
    hip::download(lg.data(), m.logits + static_cast<u64>(r) * m.cfg.n_vocab * 4, lg.size() * 4);
    return fnvF32(lg);
}

std::vector<u32> goldenLongIds(const Tok& tok, std::size_t n);

int cmdSeqtest(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("seqtest: missing MODEL.gguf");
    gguf::File f = gguf::File::open(a.pos[0]);
    selectDevice(a, nullptr);
    Tok tok(f);
    // --slot-ctx N: the context per sequence (P3-6: the solo digest must not depend on it)
    // (12288: room for the chunk schedule gate's long prompts of long_n / long_n2 tokens)
    const u32 long_n = 9500, long_n2 = 8232;
    // CLI-SCHED gate: prompt lengths whose old CLI chunking (max_batch forwards) was off the server's
    // schedule; --sched-lens N,N,... adds lengths (e.g. 32800; the context per sequence grows to fit)
    std::vector<u32> sched_lens = {4000, 4100, long_n2};  // 4000: the golden suite single long4k prompt
    if (auto v = a.get("--sched-lens"))
        for (u32 x : parseIds(*v)) sched_lens.push_back(x);
    u32 slot_ctx = a.get("--slot-ctx") ? static_cast<u32>(std::stoul(*a.get("--slot-ctx"))) : 12288;
    const u32 n_gen = a.get("--decode") ? static_cast<u32>(std::stoul(*a.get("--decode"))) : 24;
    for (u32 x : sched_lens) slot_ctx = std::max<u32>(slot_ctx, (x + 64 + q::kv_page - 1) / q::kv_page * q::kv_page);
    const std::vector<u32> pa = tok.encode(chatWrap(std::string("Write a C function that reverses a singly linked list, and explain it.") + k_bench_text, false));
    const std::vector<u32> pb = tok.encode(chatWrap("請用繁體中文說明 TCP 三向交握的過程，並用 Python 寫一個簡單的 socket 伺服器。", false));
    q::LoadOptions lo;
    lo.kv_mode = q::kvModeFromEnv().value_or(q::KvMode::automatic);
    lo.numerics = modeRequest(a);
    q::LoadStats stats;
    auto mp = q::Model::load(f, 0, stats, lo);
    q::Model& m = *mp;
    {
        std::string log;
        q::loadOrTune(m, a.pos[0], log);
        out(log);
    }
    m.setupSeqs(2, slot_ctx);
    m.allocKvPool(2 * slot_ctx);
    for (u32 s = 0; s < 2; ++s) {
        std::vector<std::int32_t> phys(slot_ctx / q::kv_page);
        for (std::size_t j = 0; j < phys.size(); ++j) phys[j] = static_cast<std::int32_t>(s * phys.size() + j);
        m.mapPages(s, 0, phys);
    }
    const std::vector<u32>* prompts[2] = {&pa, &pb};
    // 1. solo reference
    // lh[s][i]: logits hash of the solo row that produced ref[s][i] (the CLI plain decode head)
    std::vector<u32> ref[2];
    std::vector<u64> lh[2];
    for (u32 s = 0; s < 2; ++s) {
        m.selectSeq(s);
        m.reset();
        m.prefill(*prompts[s], 0);
        u32 t = m.argmax();
        lh[s].push_back(rowHash(m, 0));
        u32 pos = static_cast<u32>(prompts[s]->size());
        for (u32 i = 0; i < n_gen; ++i) {
            ref[s].push_back(t);
            m.step(t, pos++);
            t = m.argmax();
            lh[s].push_back(rowHash(m, 0));
        }
        ref[s].push_back(t);
    }
    {
        u64 dg = 0xcbf29ce484222325ull;
        for (u32 s = 0; s < 2; ++s)
            for (u64 v : lh[s]) dg = (dg ^ v) * 0x100000001b3ull;
        out(fmt("  solo logits digest (slot ctx %u, max_ctx %u): %016llx\n", slot_ctx, m.max_ctx, static_cast<unsigned long long>(dg)));
    }
    // P2-1: every head input (plain decode, CLI verify rows, server batched rows) must give the
    // same logits bits as the solo row; lh_bad counts rows whose logits hash differs
    u32 lh_bad = 0, lh_total = 0;
    bool ok = true;
    // 2. segmented prefill + batched rows
    for (u32 s = 0; s < 2; ++s) {
        m.selectSeq(s);
        m.reset();
    }
    const q::PSeg segs[2] = {{0, pa, 0, 0, pa.size(), std::nullopt}, {1, pb, 0, 0, pb.size(), std::nullopt}};
    const bool segmented = m.canSegment() && pa.size() + pb.size() <= m.max_batch;
    u32 next[2];
    if (segmented) {
        m.prefillSegs(segs, false);
        for (u32 s = 0; s < 2; ++s) {
            m.selectSeq(0);
            m.segLogitsToFront(s);
            next[s] = m.argmax();
        }
    } else {
        for (u32 s = 0; s < 2; ++s) {
            m.selectSeq(s);
            m.prefill(*prompts[s], 0);
            next[s] = m.argmax();
        }
    }
    out(fmt("  prefill: %s; first tokens %s\n", segmented ? "segmented (prefillSegs)" : "solo (cannot segment)",
            next[0] == ref[0][0] && next[1] == ref[1][0] ? "match" : "DIFFER"));
    ok = ok && next[0] == ref[0][0] && next[1] == ref[1][0];
    u32 pos[2] = {static_cast<u32>(pa.size()), static_cast<u32>(pb.size())};
    const bool fused = m.fusedDecode();
    u32 matched = 0, total = 0;
    std::vector<std::int32_t> ctl(2 * q::ctl_words);
    const u32 half = n_gen / 2;
    for (u32 i = 0; i < half; ++i) {
        std::vector<q::VSeg> vs;
        for (u32 s = 0; s < (fused ? 2u : 1u); ++s) vs.push_back({s, next[s], 0, pos[s], {}});
        m.verifyBatchEnqueue(vs);
        m.readCtl(ctl);
        for (const q::VSeg& v : vs) {
            lh_total += 1;
            lh_bad += rowHash(m, v.seq) != lh[v.seq][i + 1] ? 1 : 0;
            next[v.seq] = static_cast<u32>(ctl[v.seq * q::ctl_words + q::ctl_rows]);
            pos[v.seq] += 1;
            total += 1;
            matched += next[v.seq] == ref[v.seq][i + 1] ? 1 : 0;
        }
    }
    out(fmt("  batched decode rows (%s): %u/%u tokens match the solo run\n", fused ? "2 sequences" : "1 sequence, unfused path", matched, total));
    ok = ok && matched == total;
    if (fused) {
        // verify with literal drafts = the reference continuation (all accepted)
        const u32 nd = std::min<u32>(4, n_gen - half - 1);
        std::vector<u32> dr[2];
        std::vector<q::VSeg> vs;
        for (u32 s = 0; s < 2; ++s) {
            dr[s].assign(ref[s].begin() + half + 1, ref[s].begin() + half + 1 + nd);
            vs.push_back({s, next[s], nd, pos[s], dr[s]});
        }
        m.verifyBatchEnqueue(vs);
        m.readCtl(ctl);
        u32 good = 0;
        for (u32 s = 0; s < 2; ++s)
            for (u32 r = 0; r <= nd; ++r) {
                good += static_cast<u32>(ctl[s * q::ctl_words + q::ctl_rows + r]) == ref[s][half + 1 + r] ? 1 : 0;
                lh_total += 1;
                lh_bad += rowHash(m, s * (nd + 1) + r) != lh[s][half + 1 + r] ? 1 : 0;
            }
        out(fmt("  batched verify with %u literal drafts per sequence: %u/%u rows match the solo run\n", nd, good, 2 * (nd + 1)));
        ok = ok && good == 2 * (nd + 1);
    }
    // the CLI's MTP / n-gram verify (Model::verify) over the reference continuation of sequence 0
    {
        m.selectSeq(0);
        m.reset();
        m.prefill(pa, 0);
        const u32 nv = std::min<u32>(8, n_gen);
        std::vector<u32> vo(nv);
        m.verify(std::span<const u32>(ref[0]).subspan(0, nv), static_cast<u32>(pa.size()), vo);
        u32 vbad = 0;
        for (u32 r = 0; r < nv; ++r) vbad += rowHash(m, r) != lh[0][r + 1] ? 1 : 0;
        lh_total += nv;
        lh_bad += vbad;
        out(fmt("  CLI verify rows (%u): %u/%u logits hashes differ from the solo rows\n", nv, vbad, nv));
    }
    // FR-DFL: the literal-draft verify of whirl chat's DFlash2 / n-gram decode (verifyEnqueueEx, the
    // dflashDecode path) with a DFlash2 block of 8 (7 drafts) and 16 (15 drafts): every row's logits
    // bits must equal the solo decode row, in every numerics mode (fast: 4-bit KV)
    for (const u32 nd : {7u, 15u}) {
        if (nd + 1 > n_gen || nd + 1 > q::max_small_batch) continue;
        m.selectSeq(0);
        m.reset();
        m.prefill(pa, 0);
        const u32 first = m.argmax();
        m.verifyEnqueueEx(first, nd, std::span<const u32>(ref[0]).subspan(1, nd), static_cast<u32>(pa.size()));
        std::array<u32, q::max_small_batch> vo{};
        (void)m.readCycle(std::span<u32>(vo.data(), nd + 1), std::span<u32>(vo.data(), 0));
        u32 vbad = 0, tbad = 0;
        for (u32 r = 0; r <= nd; ++r) {
            vbad += rowHash(m, r) != lh[0][r + 1] ? 1 : 0;
            tbad += vo[r] != ref[0][r + 1] ? 1 : 0;
        }
        lh_total += nd + 1;
        lh_bad += vbad;
        ok = ok && tbad == 0 && first == ref[0][0];
        out(fmt("  literal-draft verify (verifyEnqueueEx, %u drafts = %u rows): %u/%u logits hashes and %u tokens differ from the solo rows\n",
                nd, nd + 1, vbad, nd + 1, tbad));
    }
    // FIX-TEST-B gate (part of ok): the server's cold chunk schedule (include/whirl/prefill_schedule.h,
    // the functions the engine runs) on a golden long prompt of > 4096 tokens. Every execution the
    // server can choose for the same prompt must give the same last-row logits bits: forwards of 1, 2
    // or more chunks (decode floor / WHIRL_PREFILL_CHUNK / WHIRL_PREFILL_BATCH), with or without the
    // TAIL-CK tail checkpoint, and a resume from a checkpoint at a schedule point (a prefix checkpoint
    // inside the prompt, the tail checkpoint). Splits off the schedule are printed for information only;
    // the CLI's prefill is checked by the CLI-SCHED gate below.
    u32 sched_bad = 0, sched_total = 0;
    if (slot_ctx < long_n + 64) {
        out(fmt("  chunk schedule: needs --slot-ctx >= %u (got %u)\n", long_n + 64, slot_ctx));
        ok = false;
    } else {
        m.selectSeq(0);
        const std::vector<u32> lp = goldenLongIds(tok, long_n);
        const std::span<const u32> ls(lp);
        const std::size_t N = lp.size(), th = tok.thinkOpenSplit(lp);
        const std::span<const std::uint32_t> noimg;
        const std::size_t tail = sched::tailPoint(N, 0, th, noimg);
        // run forwards ending at `ends` (the first starts at 0); the last row's logits hash
        auto runEnds = [&](std::span<const u32> ids, const std::vector<std::size_t>& ends) {
            m.reset();
            std::size_t cur = 0;
            for (std::size_t e : ends) {
                m.forward(ids.subspan(cur, e - cur), static_cast<u32>(cur));
                cur = e;
            }
            return rowHash(m, 0);
        };
        auto cold = [&](std::size_t exec, std::span<const std::size_t> stops) { return sched::forwardEnds(N, 0, th, noimg, 0, exec, stops, m.max_batch); };
        // resume at a schedule point r: the run that kept the checkpoint there (no forward crosses r),
        // then the resumed request's own forwards from r (the state at r is what the checkpoint holds)
        auto resumed = [&](std::size_t r, std::size_t exec) {
            const std::array<std::size_t, 1> st = {r};
            std::vector<std::size_t> e = sched::forwardEnds(N, 0, th, noimg, 0, exec, st, m.max_batch);
            e.erase(std::upper_bound(e.begin(), e.end(), r), e.end());
            if (e.empty() || e.back() != r) e.push_back(r);  // the checkpoint holds the state after a forward that ended at r
            const std::vector<std::size_t> e2 = sched::forwardEnds(N, 0, th, noimg, r, exec, {}, m.max_batch);
            e.insert(e.end(), e2.begin(), e2.end());
            return e;
        };
        auto endsStr = [](const std::vector<std::size_t>& e) {
            std::string r;
            for (std::size_t x : e) r += (r.empty() ? "" : ",") + std::to_string(x);
            return r;
        };
        const bool g1151 = m.arch == hip::Arch::gfx1151;
        const std::size_t ex = sched::prefillExecFor(sched::prefillExecDefault(g1151, m.cfg.moe), m.max_batch, g1151),
                          ex_big = sched::prefillExecFor(16384, m.max_batch, g1151), ex2 = sched::prefillExecFor(2048, m.max_batch);
        const std::array<std::size_t, 1> tail_stop = {tail};
        const std::size_t mid = sched::schedFloor(N, 0, th, noimg, N / 3);  // a prefix checkpoint inside the prompt
        const std::vector<std::size_t> ref_ends = cold(ex, tail_stop);
        const u64 ref = runEnds(ls, ref_ends);
        out(fmt("  chunk schedule (N %zu, tail checkpoint @%zu, exec %zu): reference = cold server forwards [%s]\n", N, tail, ex,
                endsStr(ref_ends).c_str()));
        struct Var {
            const char* name;
            std::vector<std::size_t> ends;
        };
        const Var vars[] = {
            {"cold, no tail checkpoint (parallel > 4)", cold(ex, {})},
            {"cold, 1-chunk forwards (decode floor)", cold(sched::prefill_chunk, tail_stop)},
            {"cold, 2-chunk forwards (WHIRL_PREFILL_CHUNK=2048)", cold(ex2, tail_stop)},
            {"cold, largest forwards (WHIRL_PREFILL_CHUNK)", cold(ex_big, tail_stop)},
            {"resume at the tail checkpoint", resumed(tail, ex)},
            {"resume at a prefix checkpoint", resumed(mid, ex)},
            {"resume at a prefix checkpoint, 1-chunk forwards", resumed(mid, sched::prefill_chunk)},
        };
        for (const Var& v : vars) {
            bool on = !v.ends.empty() && v.ends.back() == N;
            for (std::size_t e : v.ends) on = on && sched::onSchedule(N, 0, th, noimg, e);
            const bool same = on && runEnds(ls, v.ends) == ref;
            sched_bad += same ? 0 : 1;
            sched_total += 1;
            out(fmt("    %-48s [%s]: %s\n", v.name, endsStr(v.ends).c_str(), !on ? "FAIL (a forward ends off the schedule)" : same ? "==" : "DIFFER (FAIL)"));
        }
        // information only: one extra split the server never makes (added to the reference's
        // forwards)
        const std::pair<const char*, std::size_t> info[] = {
            {"split 64 rows after a schedule point", mid + 64},
            {"split 40 rows after a schedule point", mid + 40},
            {"split at 1000 (not 64-aligned)", 1000},
            {"last 40 rows as their own forward", N - 40},
            {"last 9 rows as their own forward (decode kernels)", N - 9},
        };
        for (const auto& [name, sp] : info) {
            std::vector<std::size_t> e = ref_ends;  // the reference's forwards with one more split at sp
            if (std::find(e.begin(), e.end(), sp) == e.end()) e.insert(std::upper_bound(e.begin(), e.end(), sp), sp);
            out(fmt("    info: %-50s: %s\n", name, runEnds(ls, e) == ref ? "==" : "DIFFER"));
        }
    }
    out(fmt("  chunk schedule: %u/%u server executions differ from the cold reference\n", sched_bad, sched_total));
    ok = ok && sched_bad == 0;
    // CLI-SCHED gate (part of ok): the CLI's prefill (prefillLikeServer, the code `whirl chat` and the golden
    // suite run) must give the same last-row logits bits as the server's forwards for the same prompt
    // (the engine's cold solo request: sched::forwardEnds with the TAIL-CK stop, exec from
    // WHIRL_PREFILL_CHUNK / max_batch), without and with the MTP block. Prompts: no-think chat prompts of
    // sched_lens tokens (not 64-aligned / short tails past a max_batch boundary), a thinking-mode prompt
    // (think-open split) and one with a long system message (system split). "old CLI" (max_batch
    // forwards, the CLI before CLI-SCHED) is printed for information.
    u32 cli_bad = 0, cli_total = 0;
    {
        m.selectSeq(0);
        const std::span<const std::uint32_t> noimg;
        const bool g1151 = m.arch == hip::Arch::gfx1151;
        const std::size_t ex = sched::prefillExecFor(sched::prefillExecDefault(g1151, m.cfg.moe), m.max_batch, g1151);
        auto runEndsMtp = [&](std::span<const u32> ids, const std::vector<std::size_t>& ends, bool mtp) {
            m.reset();
            std::size_t cur = 0;
            for (std::size_t e : ends) {
                if (mtp)
                    m.prefillMtpChunk(ids, 0, cur, e - cur, std::nullopt);
                else
                    m.forward(ids.subspan(cur, e - cur), static_cast<u32>(cur));
                cur = e;
            }
            return rowHash(m, 0);
        };
        struct P {
            std::string name;
            std::vector<u32> ids;
        };
        std::vector<P> ps;
        for (u32 n : sched_lens) ps.push_back({fmt("no-think, %u tokens", n), goldenLongIds(tok, n)});
        {
            // (k sections of the bench text, ~140 tokens each: thinking ~5k tokens; system message ~2.7k
            // tokens (>= sys_min, so the server splits there) + a ~1.7k-token user turn)
            auto doc = [](int k) {
                std::string d;
                for (int i = 1; i <= k; ++i) d += fmt("Section %d.\n", i) + std::string(k_bench_text) + "\n";
                return d;
            };
            ps.push_back({"thinking mode (think-open split)", tok.encode(chatWrap("Summarize these notes.\n\n" + doc(36), true))});
            ps.push_back({"long system message (system split)",
                          tok.encode("<|im_start|>system\nYou are a careful assistant. Reference notes:\n" + doc(19) + "<|im_end|>\n<|im_start|>user\n" +
                                     doc(12) + "List the three most important points.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")});
        }
        const bool has_mtp = m.mtp.has_value();
        for (const P& pp : ps) {
            const std::span<const u32> ids(pp.ids);
            const std::size_t n = ids.size();
            if (n + 64 > slot_ctx) {
                out(fmt("    CLI-SCHED %-36s: %zu tokens do not fit --slot-ctx %u (FAIL)\n", pp.name.c_str(), n, slot_ctx));
                ++cli_bad;
                ++cli_total;
                continue;
            }
            const std::size_t th = tok.thinkOpenSplit(pp.ids);
            const std::size_t sy = sched::sysSplit(sched::sysBoundary(ids, tok.im_start, tok.system, tok.im_end, tok.newline), n, sched::sys_min_default);
            // the split prompts must really split (else the case is vacuous)
            if ((pp.name.rfind("thinking", 0) == 0 && th == 0) || (pp.name.rfind("long system", 0) == 0 && sy == 0)) {
                out(fmt("    CLI-SCHED %-36s: no forced split in the prompt (N %zu) (FAIL)\n", pp.name.c_str(), n));
                ++cli_bad;
                ++cli_total;
                continue;
            }
            std::vector<std::size_t> st;
            if (n >= sched::tail_ck_min_default) {
                const std::size_t t = sched::tailPoint(n, sy, th, noimg);
                if (t > 0 && t != sy && t != th) st.push_back(t);
            }
            const std::vector<std::size_t> srv_ends = sched::forwardEnds(n, sy, th, noimg, 0, ex, st, m.max_batch);
            // the CLI before CLI-SCHED: max_batch forwards, the think-open token as its own step
            std::vector<std::size_t> old_ends;
            for (std::size_t off = 0; off < n;) {
                const std::size_t lim = (th > 0 && off < th) ? th : n;
                off += std::min<std::size_t>(lim - off, m.max_batch);
                old_ends.push_back(off);
            }
            for (int mt = 0; mt < (has_mtp ? 2 : 1); ++mt) {
                const bool mtp = mt == 1;
                if (mtp && pp.name.rfind("no-think", 0) == 0 && n != long_n2) continue;  // MTP: one length + the split prompts
                const u64 srv = runEndsMtp(ids, srv_ends, mtp);
                m.reset();
                prefillLikeServer(m, tok, ids, mtp);
                const u64 cli = rowHash(m, 0);
                const u64 old = runEndsMtp(ids, old_ends, mtp);
                cli_bad += cli == srv ? 0 : 1;
                cli_total += 1;
                std::string se;
                for (std::size_t e : srv_ends) se += (se.empty() ? "" : ",") + std::to_string(e);
                out(fmt("    CLI-SCHED %-36s%s (N %zu, sys %zu, think %zu, server forwards [%s]): CLI %s server; old CLI %s\n", pp.name.c_str(),
                        mtp ? " +MTP" : "", n, sy, th, se.c_str(), cli == srv ? "==" : "DIFFER (FAIL)", old == srv ? "==" : "differed"));
            }
        }
    }
    out(fmt("  CLI-SCHED: %u/%u CLI prefills differ from the server's forwards\n", cli_bad, cli_total));
    ok = ok && cli_bad == 0;
    out(fmt("  head logits bits: %u/%u verify / batched rows differ from the solo decode row\n", lh_bad, lh_total));
    ok = ok && lh_bad == 0;
    out(ok ? "seqtest: ok\n" : "seqtest: FAIL\n");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// golden: fixed-prompt runs whose per-step outputs print as hash lines
// ("G <case> key=value ..."); tools/golden/golden.ps1 records them under
// tests/golden/<arch>/ and diffs later runs against them (refactor gate: a
// change that must not alter output has to reproduce every line bit for bit).
// suite single: plain prefill (short + ~4k tokens) and decode steps with the
// logits hash of every step; MTP speculative decode token streams (fixed draft
// count, no timing-based adaptation; then with forced n-gram drafts).
// suite pair: two sequences as the server runs them: solo reference,
// segmented prefill, batched decode rows, batched verify with literal drafts,
// one batched MTP draft + verify cycle.

// FNV-1a 64 over the f32 bits of one logits row
u64 fnvF32(std::span<const float> v) {
    u64 h = 0xcbf29ce484222325ull;
    for (float x : v) {
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        h ^= b;
        h *= 0x100000001b3ull;
    }
    return h;
}

// top-5 ids and the hash of logits row 0
struct GoldenRow {
    std::vector<float> lg;
    std::vector<u32> idx;
    std::string line(q::Model& m) {
        lg.resize(m.cfg.n_vocab);
        m.readLogits(lg);
        idx.resize(lg.size());
        for (u32 i = 0; i < idx.size(); ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](u32 x, u32 y) { return lg[x] > lg[y] || (lg[x] == lg[y] && x < y); });
        return fmt("top5=%u,%u,%u,%u,%u lh=%016llx", idx[0], idx[1], idx[2], idx[3], idx[4], static_cast<unsigned long long>(fnvF32(lg)));
    }
};

std::string idList(std::span<const u32> t) {
    std::string s;
    for (std::size_t i = 0; i < t.size(); ++i) s += (i ? "," : "") + std::to_string(t[i]);
    return s;
}

void gout(const std::string& s) { out("G " + s + "\n"); }

std::string streamLine(const char* name, std::span<const u32> t) {
    return fmt("%s.stream n=%zu h=%016llx ids=", name, t.size(), static_cast<unsigned long long>(fnv(t))) + idList(t);
}

const char* k_golden_short = "請用繁體中文說明快速排序的原理，並用 C++ 寫一個泛型的 quicksort 函式，程式碼註解用中文。";
const char* k_golden_pair_a = "Write a C function that reverses a singly linked list, and explain it.";
const char* k_golden_pair_b = "請用繁體中文說明 TCP 三向交握的過程，並用 Python 寫一個簡單的 socket 伺服器。";

// a chat prompt of exactly n tokens: numbered sections of the bench text, cut, then the assistant header
std::vector<u32> goldenLongIds(const Tok& tok, std::size_t n) {
    std::string text = "Summarize the following notes in one paragraph, then list the three most important points.\n\n";
    std::vector<u32> ids;
    for (int i = 1; ids.size() < n + 64; ++i) {
        text += fmt("Section %d.\n", i) + std::string(k_bench_text) + "\n";
        ids = tok.encode(chatWrap(text, false));
    }
    const std::vector<u32> tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    ids.resize(n - tail.size());
    ids.insert(ids.end(), tail.begin(), tail.end());
    return ids;
}

// ngram_only (suite ngram): only the forced n-gram MTP case, on a prompt that asks the model to copy a
// text (its own reply then repeats the prompt, so n-gram drafts really run: the runner fails the case
// when ng_cycles is 0). The suite single's prompt has no repetition (its n-gram case never drafts).
int goldenSingle(const Args& a, const gguf::File& f, const Tok& tok, u32 n_dec, bool ngram_only = false) {
    Loaded L = loadModel(a, f, 8192);
    q::Model& m = *L.model;
    {
        std::string log;
        q::loadOrTune(m, a.pos[0], log);
    }
    const bool mtp = m.mtp.has_value() && envFlag("MTP", true);
    if (mtp && envFlag("MTP_Q4", true)) m.requantMtpQ4();
    applyRuntimeEnv(m);
    gout(fmt("info kv=%s max_ctx=%u mtp=%d fused=%d", m.kvName(), m.max_ctx, mtp ? 1 : 0, m.fusedDecode() ? 1 : 0));
    GoldenRow gr;
    const std::vector<u32> sp = tok.encode(chatWrap(
        ngram_only ? std::string("Copy the following text word for word, twice, and add nothing else.\n\n") + k_bench_text : std::string(k_golden_short), false));
    const std::vector<u32> lp = goldenLongIds(tok, 4000);
    auto* tk = static_cast<std::int32_t*>(hip::hostMalloc(4));
    // the chat prefill (the server's forwards, CLI-SCHED), then the pipelined decode step chat runs (graph / fused path)
    auto plain = [&](const char* name, const std::vector<u32>& ids, u32 nd) {
        m.reset();
        prefillLikeServer(m, tok, ids, false);
        u32 t = m.beginDecode(static_cast<u32>(ids.size()));
        gout(fmt("%s.prefill n=%zu next=%u ", name, ids.size(), t) + gr.line(m));
        std::vector<u32> gen{t};
        for (u32 i = 1; i <= nd; ++i) {
            m.decodeStep();
            hip::downloadAsync(tk, m.out_tok, 4, m.stream);
            hip::sync();
            t = static_cast<u32>(*tk);
            gout(fmt("%s.step%02u next=%u ", name, i, t) + gr.line(m));
            gen.push_back(t);
        }
        gout(streamLine(name, gen));
    };
    if (!ngram_only) {
        plain("short", sp, n_dec);
        plain("long4k", lp, 8);
    }
    hip::hostFree(tk);
    if (ngram_only && !mtp) throw std::runtime_error("golden: suite ngram needs a model with an MTP head (and WHIRL_MTP != 0)");
    if (mtp) {
        bool head = false;
        auto spec = [&](const char* name, bool ngram) {
            m.reset();
            prefillLikeServer(m, tok, sp, true);
            const u32 next = m.beginDecode(static_cast<u32>(sp.size()));
            if (!head) {
                m.buildDraftHeadEx(q::Model::DraftHeadKind::d2);
                applyDraftVocab(m, tok.t);
                head = true;
            }
            u32 drafts = 0;
            ChatOpts o;
            o.stream = false;
            o.print_text = false;
            o.max_tokens = 64;
            o.max_ctx = m.max_ctx;
            o = mtpOpts(m, o, &drafts);
            // deterministic draft counts: no timing-based adaptation or n-gram choice
            o.mtp_auto = false;
            o.mtp_adapt = 0;
            o.trace_tps = 0;
            o.ngram = ngram;
            o.ngram_force = ngram;
            const DecodeResult r = specDecode(m, tok, next, static_cast<u32>(sp.size()), o, drafts, sp);
            gout(fmt("%s.cycles drafts=%u cycles=%u accepted=%u ng_cycles=%u ng_accepted=%u", name, drafts, r.cycles, r.accepted, r.ng_cycles,
                     r.ng_accepted));
            gout(streamLine(name, r.tokens));
        };
        if (!ngram_only) spec("mtp", false);
        spec("mtp_ngram", true);
    }
    return 0;
}

int goldenPair(const Args& a, const gguf::File& f, const Tok& tok, u32 n_dec) {
    const u32 slot_ctx = 4096;
    const std::vector<u32> pa = tok.encode(chatWrap(std::string(k_golden_pair_a) + k_bench_text, false));
    const std::vector<u32> pb = tok.encode(chatWrap(k_golden_pair_b, false));
    q::LoadOptions lo;
    lo.kv_mode = q::kvModeFromEnv().value_or(q::KvMode::automatic);
    lo.numerics = modeRequest(a);
    q::LoadStats stats;
    auto mp = q::Model::load(f, 0, stats, lo);
    q::Model& m = *mp;
    {
        std::string log;
        q::loadOrTune(m, a.pos[0], log);
    }
    const bool mtp = m.mtp.has_value() && envFlag("MTP", true);
    if (mtp && envFlag("MTP_Q4", true)) m.requantMtpQ4();
    if (mtp) {
        m.buildDraftHeadEx(q::Model::DraftHeadKind::d2);
        applyDraftVocab(m, tok.t);
    }
    applyRuntimeEnv(m);
    m.gdn_replay = envFlag("GDN_REPLAY", true);  // the server default
    m.setupSeqs(2, slot_ctx);
    m.allocKvPool(2 * slot_ctx);
    for (u32 s = 0; s < 2; ++s) {
        std::vector<std::int32_t> phys(slot_ctx / q::kv_page);
        for (std::size_t j = 0; j < phys.size(); ++j) phys[j] = static_cast<std::int32_t>(s * phys.size() + j);
        m.mapPages(s, 0, phys);
    }
    const u32 nd_mtp = std::min<u32>(q::mtpDefaults(m.cfg.moe, m.arch).drafts, 4);
    if (mtp && !m.gdn_replay) m.ensureSnapshots(2 * nd_mtp);
    const bool segmented = m.canSegment() && pa.size() + pb.size() <= m.max_batch;
    const bool fused = m.fusedDecode();
    gout(fmt("info kv=%s mtp=%d fused=%d segment=%d replay=%d", m.kvName(), mtp ? 1 : 0, fused ? 1 : 0, segmented ? 1 : 0, m.gdn_replay ? 1 : 0));
    GoldenRow gr;
    const std::vector<u32>* prompts[2] = {&pa, &pb};
    // 1. solo reference
    std::vector<u32> ref[2];
    for (u32 s = 0; s < 2; ++s) {
        m.selectSeq(s);
        m.reset();
        m.prefill(*prompts[s], 0);
        u32 t = m.argmax();
        gout(fmt("solo%u.prefill n=%zu next=%u ", s, prompts[s]->size(), t) + gr.line(m));
        u32 pos = static_cast<u32>(prompts[s]->size());
        for (u32 i = 0; i < n_dec; ++i) {
            ref[s].push_back(t);
            m.step(t, pos++);
            t = m.argmax();
        }
        ref[s].push_back(t);
        gout(streamLine(s == 0 ? "solo0" : "solo1", ref[s]));
    }
    std::vector<std::int32_t> ctl(2 * q::ctl_words);
    auto prefillBoth = [&](bool with_mtp, u32* next) {
        const char* tag = with_mtp ? "segmtp" : "seg";
        for (u32 s = 0; s < 2; ++s) {
            m.selectSeq(s);
            m.reset();
        }
        const q::PSeg segs[2] = {{0, pa, 0, 0, pa.size(), std::nullopt}, {1, pb, 0, 0, pb.size(), std::nullopt}};
        if (segmented) {
            m.prefillSegs(segs, with_mtp);
            for (u32 s = 0; s < 2; ++s) {
                m.selectSeq(0);
                m.segLogitsToFront(s);
                next[s] = m.argmax();
                gout(fmt("%s%u.prefill next=%u ", tag, s, next[s]) + gr.line(m));
            }
        } else {
            for (u32 s = 0; s < 2; ++s) {
                m.selectSeq(s);
                if (with_mtp)
                    m.prefillWithMtp(*prompts[s]);
                else
                    m.prefill(*prompts[s], 0);
                next[s] = m.argmax();
                gout(fmt("%s%u.prefill next=%u ", tag, s, next[s]) + gr.line(m));
            }
        }
    };
    // 2. segmented prefill + batched decode rows + batched verify with literal drafts
    u32 next[2];
    prefillBoth(false, next);
    u32 pos[2] = {static_cast<u32>(pa.size()), static_cast<u32>(pb.size())};
    std::vector<u32> rows[2] = {{next[0]}, {next[1]}};
    const u32 half = n_dec / 2;
    for (u32 i = 0; i < half; ++i) {
        std::vector<q::VSeg> vs;
        for (u32 s = 0; s < (fused ? 2u : 1u); ++s) vs.push_back({s, next[s], 0, pos[s], {}});
        m.verifyBatchEnqueue(vs);
        m.readCtl(ctl);
        for (const q::VSeg& v : vs) {
            next[v.seq] = static_cast<u32>(ctl[v.seq * q::ctl_words + q::ctl_rows]);
            pos[v.seq] += 1;
            rows[v.seq].push_back(next[v.seq]);
            // DeltaNet replay: the verify ran 1 row of this sequence, keep it (as the engine does)
            if (m.gdn_replay) m.setPending(v.seq, 1);
        }
    }
    for (u32 s = 0; s < 2; ++s) gout(fmt("batch%u.rows ids=", s) + idList(rows[s]));
    if (fused) {
        const u32 nd = std::min<u32>(4, n_dec - half - 1);
        std::vector<u32> dr[2];
        std::vector<q::VSeg> vs;
        for (u32 s = 0; s < 2; ++s) {
            dr[s].assign(ref[s].begin() + half + 1, ref[s].begin() + half + 1 + nd);
            vs.push_back({s, next[s], nd, pos[s], dr[s]});
        }
        m.verifyBatchEnqueue(vs);
        m.readCtl(ctl);
        for (u32 s = 0; s < 2; ++s) {
            std::vector<u32> rr;
            for (u32 r = 0; r <= nd; ++r) rr.push_back(static_cast<u32>(ctl[s * q::ctl_words + q::ctl_rows + r]));
            gout(fmt("litverify%u drafts=%s ids=", s, idList(dr[s]).c_str()) + idList(rr));
        }
    }
    // 3. segmented prefill with the MTP block + one batched MTP draft / verify cycle
    if (mtp && fused) {
        prefillBoth(true, next);
        m.draft_p_min = 0;
        m.draft_n_min = 0;
        const u32 pp[2] = {static_cast<u32>(pa.size()), static_cast<u32>(pb.size())};
        const q::MSeg ms[2] = {{0, std::span<const u32>(&next[0], 1), pp[0], pp[0]}, {1, std::span<const u32>(&next[1], 1), pp[1], pp[1]}};
        for (u32 r = 0; r < nd_mtp; ++r) m.mtpBatchStepEx(std::span<const q::MSeg>(ms, 2), r, true);
        const q::VSeg vs[2] = {{0, next[0], nd_mtp, pp[0], {}}, {1, next[1], nd_mtp, pp[1], {}}};
        m.verifyBatchEnqueue(std::span<const q::VSeg>(vs, 2));
        m.readCtl(ctl);
        for (u32 s = 0; s < 2; ++s) {
            std::vector<u32> d, rr;
            for (u32 r = 0; r < nd_mtp; ++r) d.push_back(static_cast<u32>(ctl[s * q::ctl_words + q::ctl_drafts + r]));
            for (u32 r = 0; r <= nd_mtp; ++r) rr.push_back(static_cast<u32>(ctl[s * q::ctl_words + q::ctl_rows + r]));
            gout(fmt("mtpcycle%u nd=%d drafts=%s ids=", s, ctl[s * q::ctl_words + q::ctl_nd], idList(d).c_str()) + idList(rr));
        }
    }
    return 0;
}

int cmdGolden(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("golden: missing MODEL.gguf");
    gguf::File f = gguf::File::open(a.pos[0]);
    std::string dev_line;
    selectDevice(a, &dev_line);
    out(dev_line);
    Tok tok(f);
    const u32 n_dec = a.get("--decode") ? std::clamp<u32>(static_cast<u32>(std::stoul(*a.get("--decode"))), 6, 256) : 24;
    const std::string suite = a.get("--suite") ? *a.get("--suite") : std::string("single");
    gout(fmt("header format=1 suite=%s decode=%u", suite.c_str(), n_dec));
    if (suite == "single") return goldenSingle(a, f, tok, n_dec);
    if (suite == "pair") return goldenPair(a, f, tok, n_dec);
    if (suite == "ngram") return goldenSingle(a, f, tok, n_dec, true);
    throw std::runtime_error("golden: --suite must be single, pair or ngram");
}

// whirl dflash-check MODEL DRAFT REFDIR: drafter forward vs reference values
std::string readBinFile(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("dflash-check: cannot open " + path);
    std::string s;
    char buf[1 << 16];
    for (std::size_t n; (n = std::fread(buf, 1, sizeof buf, fp)) > 0;) s.append(buf, n);
    std::fclose(fp);
    return s;
}

int cmdDflashCheck(const Args& a) {
    if (a.pos.size() < 3) throw std::runtime_error("dflash-check: usage: whirl dflash-check MODEL.gguf DRAFT.gguf REFDIR");
    gguf::File f = gguf::File::open(a.pos[0]);
    selectDevice(a, nullptr);
    Loaded L = loadModel(a, f, 4096);
    q::Model& m = *L.model;
    const gguf::File df = gguf::File::open(a.pos[1]);
    q::LoadStats ds;
    m.loadDrafter(df, ds);
    const std::string dir = a.pos[2];
    std::map<std::string, long long> meta;
    {
        const std::string t = readBinFile(dir + "/meta.txt");
        std::size_t p = 0;
        while (p < t.size()) {
            std::size_t e = t.find('\n', p);
            if (e == std::string::npos) e = t.size();
            const std::string line = t.substr(p, e - p);
            const std::size_t sp = line.find(' ');
            if (sp != std::string::npos) meta[line.substr(0, sp)] = std::stoll(line.substr(sp + 1));
            p = e + 1;
        }
    }
    for (const char* key : {"n_ctx", "ctx_pos0", "anchor", "anchor_pos"})
        if (!meta.count(key)) throw std::runtime_error(std::string("dflash-check: meta.txt has no ") + key);
    const auto& dc = m.dflash->cfg;
    const u64 E = dc.n_embd;
    const u32 n_ctx = static_cast<u32>(meta["n_ctx"]), B = dc.block;
    const std::string feats = readBinFile(dir + "/feats.f32");
    if (feats.size() != dc.target_layers.size() * n_ctx * E * 4)
        throw std::runtime_error("dflash-check: feats.f32 is not [target layers][n_ctx][n_embd] f32");
    m.dflashSetup(1, std::max<u32>(1, n_ctx), true);
    const hip::DevPtr fd = m.alloc(feats.size());
    hip::upload(fd, feats.data(), feats.size());
    m.dflashIngest(0, fd, n_ctx, 0, n_ctx, static_cast<u32>(meta["ctx_pos0"]));
    const std::vector<u32> drafts = m.dflashDraft(0, static_cast<u32>(meta["anchor"]), static_cast<u32>(meta["anchor_pos"]));
    bool ok = true;
    std::vector<float> got(static_cast<std::size_t>(B) * E);
    auto compare = [&](const std::string& name, hip::DevPtr src, const std::string& file) {
        hip::download(got.data(), src, got.size() * 4);
        const std::string rs = readBinFile(dir + "/" + file);
        if (rs.size() != got.size() * 4) throw std::runtime_error("dflash-check: " + file + " is not [block][n_embd] f32");
        const float* ref = reinterpret_cast<const float*>(rs.data());
        double num = 0, den = 0, mx = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double d = static_cast<double>(got[i]) - ref[i];
            num += d * d;
            den += static_cast<double>(ref[i]) * ref[i];
            mx = std::max(mx, std::fabs(d));
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        const bool pass = rel < 3e-2;
        ok = ok && pass;
        out(fmt("  %-12s rel. error %.3e, max abs %.3e  %s\n", name.c_str(), rel, mx, pass ? "ok" : "FAIL"));
    };
    for (u32 l = 0; l < dc.n_layer; ++l)
        compare("layer " + std::to_string(l), m.dfr->dbg + static_cast<u64>(l) * B * E * 4, "hidden_l" + std::to_string(l) + ".f32");
    compare("final norm", m.dfr->e, "final_norm.f32");
    const std::string dr = readBinFile(dir + "/drafts.i32");
    const auto* rd = reinterpret_cast<const std::int32_t*>(dr.data());
    const std::size_t n_ref = dr.size() / 4;
    std::size_t same = 0;
    while (same < n_ref && same < drafts.size() && static_cast<u32>(rd[same]) == drafts[same]) ++same;
    std::string gs, rs2;
    for (u32 t : drafts) gs += " " + std::to_string(t);
    for (std::size_t i = 0; i < n_ref; ++i) rs2 += " " + std::to_string(rd[i]);
    out(fmt("  drafts: whirl%s | reference%s | common prefix %zu of %zu\n", gs.c_str(), rs2.c_str(), same, n_ref));
    out(ok ? "dflash-check: ok\n" : "dflash-check: FAIL\n");
    return ok ? 0 : 1;
}

int cmdSelftest(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("selftest: missing MODEL.gguf");
    gguf::File f = gguf::File::open(a.pos[0]);
    selectDevice(a, nullptr);
    Loaded L = loadModel(a, f, 4096);
    q::Model& m = *L.model;
    hip::memset(m.h, 0, 4ull * m.max_batch * m.cfg.n_embd);
    hip::memset(m.ffn_g, 0, 4ull * m.max_batch * m.ff_scratch);
    bool ok = true;
    std::string log;
    // the precise checks force the precise items themselves (fp8 / MoE fp8 / h16 off), whatever
    // the load mode; the int8 and balance checks use the load mode's items
    out(fmt("  numerics at load: %s (precise checks run with fp8 / MoE fp8 / h16 off)\n", whirl::numerics::modeName(m.num_req.mode)));
    out("  int8 GEMV vs f32 GEMV (random x, one matrix per type):\n");
    ok = m.checkGemvq(log) && ok;
    ok = m.checkGemvBitwise(log) && ok;
    ok = m.checkPrefillInvariance(log) && ok;
    ok = m.checkPreciseDecode(log) && ok;
    ok = m.checkCapture(log) && ok;
    const u32 grp = std::max<u32>(1, 16 / (m.cfg.n_head / m.cfg.n_head_kv));
    ok = m.checkAttnGroups(log, 1000, grp) && ok;
    ok = m.checkAttnGroups(log, 1023, grp) && ok;
    const u32 grp2 = std::max<u32>(1, 32 / (m.cfg.n_head / m.cfg.n_head_kv));
    ok = m.checkAttnGroups(log, 1000, grp2, true) && ok;
    ok = m.checkAttnGroups(log, 1023, grp2, true) && ok;
    out(log);
    out(ok ? "selftest: ok\n" : "selftest: FAIL\n");
    return ok ? 0 : 1;
}

int cmdDevices() {
    const std::vector<hip::DeviceInfo> devs = hip::listDevices();
    if (devs.empty()) {
        std::fputs("whirl: error: the AMD graphics driver reports no GPU.\n"
                   "  WHIRL needs an AMD Radeon AI PRO R9700 (gfx1201) or a Radeon 8060S (gfx1151). If one is\n"
                   "  installed, install AMD Software:\n"
                   "  Adrenalin Edition 26.8.1 or newer (https://www.amd.com/en/support) and restart Windows.\n",
                   stderr);
        return app::exit_gpu;
    }
    auto haveKernels = [](const hip::DeviceInfo& d) {
        for (const hip::EmbeddedObject& o : hip::embeddedObjects())
            if (d.gcn_arch.rfind(o.arch, 0) == 0) return true;
        return false;
    };
    // the device used without --device / WHIRL_DEVICE (pickDevice's rule)
    int def = -1;
    for (const hip::DeviceInfo& d : devs)
        if (def < 0 && d.gcn_arch.rfind("gfx1201", 0) == 0 && haveKernels(d)) def = d.index;
    for (const hip::DeviceInfo& d : devs)
        if (def < 0 && haveKernels(d)) def = d.index;
    for (const hip::DeviceInfo& d : devs) {
        std::string tags;
        if (!haveKernels(d)) tags += "  [no GPU kernels for this architecture in this build]";
        else tags += "  [supported]";
        if (d.integrated) tags += " [integrated: memory shared with the CPU]";
        if (d.index == def) tags += " [default]";
        // integrated GPU: the carve-out (DXGI), not the runtime's total, which adds a share of system RAM (UMA-1)
        std::string mem = fmt("%.1f GiB", d.total_mem / (1024.0 * 1024.0 * 1024.0));
        if (d.integrated)
            if (const hip::UmaMem u = hip::umaMemory(d.index); u.known) mem = hip::umaSizeText(u);
        out(fmt("  device %d: %s (%s), %s, %d CUs%s\n", d.index, d.name.c_str(), d.gcn_arch.c_str(), mem.c_str(), d.compute_units,
                tags.c_str()));
    }
    return 0;
}

const char* commandHelp(const std::string& cmd) {
    if (cmd == "chat") return k_help_chat;
    if (cmd == "bench") return k_help_bench;
    if (cmd == "selftest") return k_help_selftest;
    if (cmd == "seqtest") return k_help_seqtest;
    if (cmd == "golden") return k_help_golden;
    if (cmd == "dflash-check") return k_help_dflash_check;
    if (cmd == "devices") return k_help_devices;
    if (cmd == "vis-encode") return k_help_vis;
    return nullptr;
}

unsigned commandScope(const std::string& cmd) {
    if (cmd == "chat") return app::sc_chat;
    if (cmd == "bench") return app::sc_bench;
    if (cmd == "selftest") return app::sc_selftest;
    if (cmd == "seqtest" || cmd == "golden") return app::sc_seqtest;
    if (cmd == "dflash-check") return app::sc_selftest;
    if (cmd == "vis-encode") return app::sc_vis;
    return 0;
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
    const char* h = commandHelp(topic);
    if (h == nullptr) return false;
    out(std::string(h) + app::envHelp(commandScope(topic)) + "\n" + app::exitCodeHelp());
    return true;
}

bool isHelpFlag(const std::string& x) { return x == "-h" || x == "--help" || x == "/?"; }

}  // namespace

int wmain(int argc, wchar_t** wargv) {
    _setmode(_fileno(stdout), _O_BINARY);
    app::initConsole();
    std::vector<std::string> argv;
    for (int i = 0; i < argc; ++i) argv.push_back(narrow(wargv[i]));
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
    static const char* known[] = {"chat", "bench", "selftest", "seqtest", "golden", "dflash-check", "devices", "serve", "vis-encode"};
    bool is_known = false;
    for (const char* k : known) is_known = is_known || cmd == k;
    if (!is_known) {
        std::fprintf(stderr, "whirl: unknown command '%s'\n\n%s", cmd.c_str(), k_help_main);
        return app::exit_usage;
    }
    bool wants_help = false;
    for (std::size_t i = 2; i < argv.size(); ++i) wants_help = wants_help || isHelpFlag(argv[i]);
    if (wants_help) {
        printHelp(cmd);
        return app::exit_ok;
    }
    // the GPU runtime of the driver must be loadable before any HIP call
    if (const int pf = app::gpuPreflight(); pf != app::exit_ok) return pf;
#if WHIRL_HAVE_SERVER
    if (cmd == "serve") {
        // the server parses its own options: argv[0] = program, then everything after "serve"
        std::vector<char*> sargv;
        sargv.push_back(argv[0].data());
        for (std::size_t i = 2; i < argv.size(); ++i) sargv.push_back(argv[i].data());
        sargv.push_back(nullptr);
        return whirl::server::serveMain(static_cast<int>(sargv.size() - 1), sargv.data(), "whirl serve");
    }
#endif
#if WHIRL_HAVE_VISION
    if (cmd == "vis-encode") {
        try {
            const std::span<const std::string> va = std::span<const std::string>(argv).subspan(2);
            if (va.size() >= 2) {
                app::requireModelFile(va[0], "vision encoder (mmproj) file");
                app::requireModelFile(va[1], "image file");
            }
            return vision::cliVisEncode(va);
        } catch (const std::exception& e) {
            return app::explain(e, "whirl");
        }
    }
#endif
    try {
        const Args a = parseArgs(argv, 2);
        if (cmd != "devices" && !a.pos.empty()) app::requireModelFile(a.pos[0]);
        if (cmd == "chat") return cmdChat(a);
        if (cmd == "bench") return cmdBench(a);
        if (cmd == "selftest") return cmdSelftest(a);
        if (cmd == "seqtest") return cmdSeqtest(a);
        if (cmd == "golden") return cmdGolden(a);
        if (cmd == "dflash-check") return cmdDflashCheck(a);
        if (cmd == "devices") return cmdDevices();
        std::fputs(k_help_main, stderr);
        return app::exit_usage;
    } catch (const std::exception& e) {
        const int code = app::explain(e, "whirl");
        if (code == app::exit_usage) std::fprintf(stderr, "  (run 'whirl %s --help' for the options)\n", cmd.c_str());
        return code;
    }
}
