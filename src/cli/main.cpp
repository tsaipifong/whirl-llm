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
#include "whirl/gguf.h"
#include "whirl/hip.h"
#include "whirl/model.h"
#include "whirl/tokenizer.h"

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

const char* k_help_chat =
    "usage: whirl chat MODEL.gguf \"PROMPT\" | @PROMPT_FILE [options]\n"
    "\n"
    "Greedy generation for one prompt, wrapped in the Qwen chat template. With an MTP head in\n"
    "the model the decode is speculative (MTP + n-gram drafts); the output is identical to\n"
    "plain greedy decoding.\n"
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
    "  --mmproj MMPROJ.gguf       vision encoder (Qwen3-VL style mmproj, F16 / BF16) for --image\n"
    "  --image IMAGE              an image placed before the prompt text (repeatable; needs --mmproj)\n"
    "  --device SPEC              GPU: r9700 (default), 8060s, an index, or a name / gfx substring\n"
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
    "  --modes M,M,...            decode modes: mtp-ngram (MTP + n-gram drafts), mtp, plain\n"
    "                             (default mtp-ngram,mtp,plain)\n"
    "  --no-think / --think       thinking off / on for the decode prompt (default on)\n"
    "  --device SPEC              GPU: r9700 (default), 8060s, an index, or a name / gfx substring\n"
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
    "  -h, --help                 this text\n";

const char* k_help_seqtest =
    "usage: whirl seqtest MODEL.gguf [--decode N] [--device SPEC]\n"
    "\n"
    "Runs the multi-sequence (server) paths against the single-sequence reference on two\n"
    "prompts: segmented prefill, batched decode rows, batched verify with literal drafts.\n"
    "Every token must match; prints 'seqtest: ok' or FAIL (exit 1).\n"
    "\n"
    "options:\n"
    "  --decode N                 tokens generated per sequence (default 24)\n"
    "  --device SPEC              GPU: r9700 (default), 8060s, an index, or a name / gfx substring\n"
    "  -h, --help                 this text\n";

const char* k_help_devices =
    "usage: whirl devices\n"
    "\n"
    "Lists the AMD GPUs the driver reports (index, name, architecture, memory, compute units)\n"
    "and whether this build has GPU kernels for each of them.\n";

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
    static const char* v[] = {"--max-tokens", "--ctx", "--tokens", "--out", "--device", "--prefill", "--decode", "--prompt", "--modes", "--mmproj"};
    for (const char* s : v)
        if (o == s) return true;
    return false;
}

Args parseArgs(const std::vector<std::string>& argv, std::size_t from) {
    Args a;
    for (std::size_t i = from; i < argv.size(); ++i) {
        const std::string& s = argv[i];
        if (s.size() > 2 && s[0] == '-' && s[1] == '-') {
            if (s == "--image") {
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
    u32 ngram_max = 0;
    bool ngram_force = false;
    bool ngram_debug = false;
    bool print_text = true;
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
    if (opt.ngram) ngp.ng.norm = tok.crlfToLf();
    u32 ng_max = opt.ngram_max > 0 ? std::min(opt.ngram_max, q::max_ng_drafts) : q::max_ng_drafts;
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
        t_draft += c1 - c0;
        t_verify += c2 - c1;
        cycles += 1;
        drafted += nd;
        u32 acc = 0;
        while (acc < nd && outv[acc] == dr[acc]) acc += 1;
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
            ngp.timing.update(nd_host, static_cast<float>(c2 - c0));
        } else {
            for (u32 kk = 0; kk < std::min(acc + 1, nd); ++kk) pos_tries[kk] += 1;
            for (u32 kk = 0; kk < acc; ++kk) pos_hits[kk] += 1;
            acc_ema = 0.7f * acc_ema + 0.3f * static_cast<float>(acc);
            acc_model.update(nd, acc);
            timing.update(nd_host, static_cast<float>(c2 - c0));
        }
        // the verify ran nd_host + 1 rows: keep acc + 1 of them
        if (acc < nd_host) model.restoreSnapshot(acc + 1);
        // emit accepted drafts plus the model's own next token
        for (u32 r = 0; r < acc + 1; ++r) {
            if (!done) done = emit(outv[r]);
            if (opt.ngram) hist.push_back(outv[r]);
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
void applyRuntimeEnv(q::Model& m) {
    if (envGet("FLOAT_GEMV")) m.float_gemv = true;
    q::applyGemvEnv(m);
    if (auto v = envU32("DBG")) m.dbg_flags = *v;
    if (auto v = envU32("GEMV_MAX")) m.gemv_max = std::clamp<u32>(*v, 1, q::max_small_batch);
    if (envGet("NO_GRAPH")) m.use_graph = false;
    if (envGet("NO_FUSE")) m.no_fuse = true;
    if (envGet("NAIVE_ATTN")) m.naive_attn = true;
    if (auto v = envGet("ATTN_WIDE"); v && *v == "0") m.attn_wide = false;
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

Loaded loadModel(const gguf::File& f, u32 max_ctx) {
    Loaded L;
    q::LoadOptions lo;
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
    out(fmt("  KV cache: %s, %u tokens\n", m.kvName(), m.max_ctx));
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
    const q::MtpDefaults def = q::mtpDefaults(model.cfg.moe);
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
    if (auto v = envGet("NGRAM_MAX")) base.ngram_max = static_cast<u32>(std::strtoul(v->c_str(), nullptr, 10));
    if (auto v = envGet("NGRAM_FORCE")) base.ngram_force = *v != "0";
    if (auto v = envGet("NGRAM_DEBUG")) base.ngram_debug = *v != "0";
    *drafts_out = drafts;
    return base;
}

void printSpecSummary(const DecodeResult& r, std::size_t n_prompt, double prefill_ms, const ChatOpts& opt) {
    const std::size_t n_gen = r.tokens.size();
    out(fmt("\n\n  prefill: %zu tok in %.0f ms (%.1f tok/s)\n", n_prompt, prefill_ms, n_prompt * 1000.0 / prefill_ms));
    out(fmt("  decode (MTP spec, max %u min %u p-min %.2f%s): %zu tok in %.0f ms (%.2f tok/s); %u verify cycles, drafts %u/%u accepted (%.2f tok/cycle)\n",
            r.n_draft, r.n_min, r.p_min, opt.mtp_auto ? " auto" : opt.mtp_adapt > 0 ? " adaptive" : "", n_gen, r.decode_ms, n_gen * 1000.0 / r.decode_ms,
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

    Loaded L = loadModel(f, opt.max_ctx);
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

    if (auto v = envU32("PREFILL_BATCH")) model.max_batch = std::max<u32>(1, std::min(model.max_batch, *v));
    q::Profile prof;
    const bool profiling = envGet("PROFILE").has_value();
    if (profiling) model.prof = &prof;
    applyRuntimeEnv(model);

    // MTP speculative decode is exact greedy: on whenever the checkpoint has a nextn
    // layer; WHIRL_MTP=0 turns it off.
    const bool use_mtp = model.mtp.has_value() && logits_out == nullptr && envFlag("MTP", true);
    if (auto v = envU32("DRAFT_VOCAB")) model.draft_vocab = *v;
    Timer timer;
    timer.begin();
    std::vector<std::int32_t> moe_ids;
    const auto moe_dump_path = envGet("DUMP_MOE");
    if (moe_dump_path) model.moe_dump = &moe_ids;
    const u32 tps_every = envU32("TRACE_TPS").value_or(0);
    {
        // chunks of max_batch; a thinking-mode prompt runs its last token as its own step
        const std::size_t split = tok.thinkOpenSplit(ids);
        // WHIRL_DUMP_LOGITS=path, WHIRL_DUMP_FROM=N (no MTP): next-token logits of every
        // prompt position >= N as f16 rows (teacher-forced, 16-row forwards)
        const auto dump_path = envGet("DUMP_LOGITS");
        const std::size_t dump_from = envU32("DUMP_FROM").value_or(0);
        std::vector<std::uint16_t> dump;
        std::vector<float> row_f32(dump_path ? static_cast<std::size_t>(q::max_small_batch) * model.cfg.n_vocab : 0);
        const double p0 = nowMs();
        std::size_t off = 0;
        while (off < ids.size()) {
            const bool dumping = dump_path && !use_mtp && off >= dump_from;
            std::size_t lim = (split > 0 && off < split) ? split : ids.size();
            if (dump_path && off < dump_from) lim = std::min(lim, dump_from);
            const std::size_t n = dumping ? std::min<std::size_t>(lim - off, q::max_small_batch) : std::min<std::size_t>(lim - off, model.max_batch);
            const double c0 = nowMs();
            model.all_logits = dumping;
            if (use_mtp)
                model.prefillMtpChunk(ids, 0, off, n, std::nullopt);
            else
                model.forward(std::span<const u32>(ids).subspan(off, n), static_cast<u32>(off));
            model.all_logits = false;
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
    Loaded L = loadModel(f, ctx);
    q::Model& model = *L.model;
    {
        std::string log;
        q::loadOrTune(model, path, log);
        out(log);
    }
    if (envFlag("MTP_Q4", true)) model.requantMtpQ4();
    loadLog(model, L.stats);
    applyRuntimeEnv(model);
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
    Timer timer;
    for (u32 s : sizes) {
        if (s > long_ids.size()) continue;
        const std::span<const u32> pr(long_ids.data(), s);
        double best = 1e30;
        const int reps = s <= 8192 ? 2 : 1;
        for (int rep = 0; rep < reps + 1; ++rep) {  // first run = warm-up
            model.reset();
            timer.begin();
            std::size_t off = 0;
            while (off < pr.size()) {
                const std::size_t n = std::min<std::size_t>(pr.size() - off, model.max_batch);
                hip::busy::beginRegion("bench_pf", model.stream);
                if (has_mtp)
                    model.prefillMtpChunk(pr, 0, off, n, std::nullopt);
                else
                    model.forward(pr.subspan(off, n), static_cast<u32>(off));
                hip::busy::markEnd();
                if (hip::busy::on()) {
                    char info[128];
                    std::snprintf(info, sizeof info, "\"size\":%u,\"rep\":%d,\"rows\":%zu,\"pos\":%zu", s, rep, n, off);
                    hip::busy::endRegion(info);
                }
                off += n;
            }
            (void)model.beginDecode(s);
            const double ms = timer.end();
            if (hip::busy::on()) {
                char info[160];
                std::snprintf(info, sizeof info, "\"size\":%u,\"rep\":%d,\"wall_ms\":%.2f", s, rep, ms);
                hip::busy::beginRegion("bench_pf_total", model.stream);
                hip::busy::endRegion(info);
                hip::busy::flush();
            }
            if (rep > 0 || reps == 0) best = std::min(best, ms);
        }
        out(fmt("    prefill %6u tok: %9.1f ms  %8.1f tok/s\n", s, best, s * 1000.0 / best));
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
    std::optional<u64> ref_hash;
    for (const std::string& mode : modes) {
        if (mode != "plain" && !has_mtp) continue;
        model.reset();
        model.decode_graph.reset();
        Timer pt;
        pt.begin();
        if (mode == "plain") {
            model.prefill(dec_ids, 0);
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
int cmdSeqtest(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("seqtest: missing MODEL.gguf");
    gguf::File f = gguf::File::open(a.pos[0]);
    selectDevice(a, nullptr);
    Tok tok(f);
    const u32 slot_ctx = 4096, n_gen = a.get("--decode") ? static_cast<u32>(std::stoul(*a.get("--decode"))) : 24;
    const std::vector<u32> pa = tok.encode(chatWrap(std::string("Write a C function that reverses a singly linked list, and explain it.") + k_bench_text, false));
    const std::vector<u32> pb = tok.encode(chatWrap("請用繁體中文說明 TCP 三向交握的過程，並用 Python 寫一個簡單的 socket 伺服器。", false));
    q::LoadOptions lo;
    lo.kv_mode = q::kvModeFromEnv().value_or(q::KvMode::automatic);
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
    std::vector<u32> ref[2];
    for (u32 s = 0; s < 2; ++s) {
        m.selectSeq(s);
        m.reset();
        m.prefill(*prompts[s], 0);
        u32 t = m.argmax();
        u32 pos = static_cast<u32>(prompts[s]->size());
        for (u32 i = 0; i < n_gen; ++i) {
            ref[s].push_back(t);
            m.step(t, pos++);
            t = m.argmax();
        }
        ref[s].push_back(t);
    }
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
            for (u32 r = 0; r <= nd; ++r) good += static_cast<u32>(ctl[s * q::ctl_words + q::ctl_rows + r]) == ref[s][half + 1 + r] ? 1 : 0;
        out(fmt("  batched verify with %u literal drafts per sequence: %u/%u rows match the solo run\n", nd, good, 2 * (nd + 1)));
        ok = ok && good == 2 * (nd + 1);
    }
    out(ok ? "seqtest: ok\n" : "seqtest: FAIL\n");
    return ok ? 0 : 1;
}

int cmdSelftest(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("selftest: missing MODEL.gguf");
    gguf::File f = gguf::File::open(a.pos[0]);
    selectDevice(a, nullptr);
    Loaded L = loadModel(f, 4096);
    q::Model& m = *L.model;
    hip::memset(m.h, 0, 4ull * m.max_batch * m.cfg.n_embd);
    hip::memset(m.ffn_g, 0, 4ull * m.max_batch * m.ff_scratch);
    bool ok = true;
    std::string log;
    out("  int8 GEMV vs f32 GEMV (random x, one matrix per type):\n");
    ok = m.checkGemvq(log) && ok;
    ok = m.checkGemvBitwise(log) && ok;
    ok = m.checkPrefillInvariance(log) && ok;
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
                   "  WHIRL needs an AMD Radeon AI PRO R9700 (gfx1201). If one is installed, install AMD Software:\n"
                   "  Adrenalin Edition 26.8.1 or newer (https://www.amd.com/en/support) and restart Windows.\n",
                   stderr);
        return app::exit_gpu;
    }
    for (const hip::DeviceInfo& d : devs) {
        bool have = false;
        for (const hip::EmbeddedObject& o : hip::embeddedObjects()) have = have || d.gcn_arch.rfind(o.arch, 0) == 0;
        out(fmt("  device %d: %s (%s), %.1f GiB, %d CUs%s\n", d.index, d.name.c_str(), d.gcn_arch.c_str(), d.total_mem / (1024.0 * 1024.0 * 1024.0),
                d.compute_units, have ? "" : "  [no GPU kernels for this architecture in this build]"));
    }
    return 0;
}

const char* commandHelp(const std::string& cmd) {
    if (cmd == "chat") return k_help_chat;
    if (cmd == "bench") return k_help_bench;
    if (cmd == "selftest") return k_help_selftest;
    if (cmd == "seqtest") return k_help_seqtest;
    if (cmd == "devices") return k_help_devices;
    if (cmd == "vis-encode") return k_help_vis;
    return nullptr;
}

unsigned commandScope(const std::string& cmd) {
    if (cmd == "chat") return app::sc_chat;
    if (cmd == "bench") return app::sc_bench;
    if (cmd == "selftest") return app::sc_selftest;
    if (cmd == "seqtest") return app::sc_seqtest;
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
    static const char* known[] = {"chat", "bench", "selftest", "seqtest", "devices", "serve", "vis-encode"};
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
        if (cmd == "devices") return cmdDevices();
        std::fputs(k_help_main, stderr);
        return app::exit_usage;
    } catch (const std::exception& e) {
        const int code = app::explain(e, "whirl");
        if (code == app::exit_usage) std::fprintf(stderr, "  (run 'whirl %s --help' for the options)\n", cmd.c_str());
        return code;
    }
}
