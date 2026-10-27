// whirl CLI: the helpers every command uses: output, arguments, the prompt text, the server's prefill forwards, hashes (arch025-C2)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"

namespace cli {

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

namespace {
bool takesValue(const std::string& o) {
    static const char* v[] = {"--max-tokens", "--ctx", "--tokens", "--out", "--device", "--prefill", "--decode", "--prompt", "--modes", "--mmproj", "--suite",
                              "--draft", "--draft-model", "--slot-ctx", "--sched-lens"};
    for (const char* s : v)
        if (o == s) return true;
    return false;
}
}  // namespace

Args parseArgs(const std::vector<std::string>& argv, std::size_t from) {
    Args a;
    for (std::size_t i = from; i < argv.size(); ++i) {
        const std::string& s = argv[i] == "-dev" ? std::string("--device") : argv[i];  // llama.cpp's short name
        if (s == "--device" && a.has("--device")) throw std::runtime_error("give --device only once");
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
// CLI-SCHED: every CLI prefill runs the forwards whirl-server runs for the same prompt alone with
// default options (include/whirl/prefill_schedule.h: the 1024-row chunk schedule with its merged
// tail, the forced splits - system message, think-open, image ends - the forward grouping, the
// TAIL-CK forward end), so `whirl chat` and the server give the same bits for the same prompt.
// The server's knobs apply here too: WHIRL_PREFILL_CHUNK, WHIRL_SYS_MIN, WHIRL_VIS_CKPT_MIN,
// WHIRL_TAIL_CK=0.
// (image ends: the ids >= image_id_base exist only in a prompt with images)
std::vector<std::size_t> serverForwardEnds(const Tok& tok, std::span<const u32> ids, u32 max_batch, const device::Profile& prof, bool moe) {
    const std::size_t n = ids.size();
    const std::size_t sys = sched::sysSplit(sched::sysBoundary(ids, tok.im_start, tok.system, tok.im_end, tok.newline), n,
                                            envU32("SYS_MIN").value_or(sched::sys_min_default));
    const std::vector<std::uint32_t> imgs =
        sched::imageSplits(ids, q::image_id_base, tok.vision_end, envU32("VIS_CKPT_MIN").value_or(sched::img_split_min_default));
    const std::size_t exec = sched::prefillExecFor(envU32("PREFILL_CHUNK").value_or(sched::prefillExecDefault(prof, moe)), max_batch, prof.prefill_fill_batch);
    const std::size_t tail_min = envFlag("TAIL_CK", true) ? sched::tail_ck_min_default : 0;
    return sched::soloForwardEnds(n, sys, tok.thinkOpenSplit(ids), imgs, 0, exec, tail_min, max_batch);
}

// the CLI prefill of ids from position 0 on the server's forwards (with the MTP block over the prompt: mtp)
void prefillLikeServer(q::Model& m, const Tok& tok, std::span<const u32> ids, bool mtp) {
    std::size_t off = 0;
    for (const std::size_t e : serverForwardEnds(tok, ids, m.max_batch, device::profile(m.arch), m.cfg.moe)) {
        if (mtp)
            m.prefillMtpChunk(ids, 0, off, e - off, std::nullopt);
        else
            m.forward(ids.subspan(off, e - off), static_cast<u32>(off));
        off = e;
    }
}

u64 fnv(std::span<const u32> toks) {
    u64 h = 0xcbf29ce484222325ull;
    for (u32 t : toks) {
        h ^= t;
        h *= 0x100000001b3ull;
    }
    return h;
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

const char* k_bench_text =
    "請用繁體中文說明下面這段 Python 程式的用途，並指出可以改進的地方。\n"
    "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n"
    "    out.extend(a[i:]); out.extend(b[j:])\n    return out\n"
    "Then rewrite it in C++20 with std::span inputs, add unit tests, and explain the complexity in English.\n";

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

}  // namespace cli
