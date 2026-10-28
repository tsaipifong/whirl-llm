// whirl golden: per-step output hashes of fixed prompts (the refactor gate's tool; arch025-C2: from src/cli/main.cpp)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"

namespace cli {

const char* k_help_golden =
    "usage: whirl golden MODEL.gguf [--suite single|pair|ngram] [--decode N] [mode] [--device SPEC]\n"
    "\n"
    "Runs fixed prompts and prints one 'G ...' line per step: top-5 ids and the FNV-1a hash\n"
    "of the logits, generated token streams. single: plain prefill (short + 4000 tokens),\n"
    "decode steps, MTP decode (fixed drafts; forced n-gram drafts). pair: two sequences on the\n"
    "server paths (segmented prefill, batched rows, batched verify, one batched MTP cycle).\n"
    "tools/golden/golden.ps1 records these under tests/golden/<arch>/ and diffs later runs.\n";

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

namespace {

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
    const bool mtp = m.mtp.has_value() && whirl::settings::mtp();
    if (mtp && whirl::settings::mtpQ4()) m.requantMtpQ4();
    applyRuntimeEnv(m);
    gout(fmt("info kv=%s max_ctx=%u mtp=%d fused=%d", m.kvName(), m.max_ctx, mtp ? 1 : 0, m.fusedDecode() ? 1 : 0));
    GoldenRow gr;
    const std::vector<u32> sp = tok.encode(chatWrap(
        ngram_only ? std::string("Copy the following text word for word, twice, and add nothing else.\n\n") + k_bench_text : std::string(k_golden_short), false));
    const std::vector<u32> lp = goldenLongIds(tok, 4000);
    device::PinnedBuf tk_buf = device::PinnedBuf::alloc(device::hipBackend(), 4);
    auto* tk = static_cast<std::int32_t*>(tk_buf.get());
    // the chat prefill (the server's forwards, CLI-SCHED), then the pipelined decode step chat runs (graph / fused path)
    auto plain = [&](const char* name, const std::vector<u32>& ids, u32 nd) {
        m.reset();
        prefillLikeServer(m, tok, ids, false);
        u32 t = m.beginDecode(static_cast<u32>(ids.size()));
        gout(fmt("%s.prefill n=%zu next=%u ", name, ids.size(), t) + gr.line(m));
        std::vector<u32> gen{t};
        for (u32 i = 1; i <= nd; ++i) {
            m.decodeStep();
            hip::downloadAsync(tk, m.scratch().out_tok, 4, m.stream);
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
    tk_buf.reset();
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
    lo.kv_req = q::kvRequestFromEnv();
    lo.numerics = modeRequest(a);
    q::LoadStats stats;
    auto mp = q::Model::load(f, 0, stats, lo);
    q::Model& m = *mp;
    {
        std::string log;
        q::loadOrTune(m, a.pos[0], log);
    }
    const bool mtp = m.mtp.has_value() && whirl::settings::mtp();
    if (mtp && whirl::settings::mtpQ4()) m.requantMtpQ4();
    if (mtp) {
        m.buildDraftHeadEx(q::Model::DraftHeadKind::d2);
        applyDraftVocab(m, tok.t);
    }
    applyRuntimeEnv(m);
    m.gdn_replay = whirl::settings::gdnReplay();  // the server default (shared reader, arch025-N15 rest)
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

}  // namespace

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

}  // namespace cli
