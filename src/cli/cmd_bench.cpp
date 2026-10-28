// whirl bench: prefill and decode speed (arch025-C2: from src/cli/main.cpp)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"

namespace cli {

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
    "  --device SPEC, -dev SPEC   GPU: pci:BUS:DEVICE (Task Manager > Performance > GPU > Physical location: PCI bus 71,\n"
    "                             device 0 -> pci:71:0), an index from 'whirl devices', r9700, 8060s, or a part of the\n"
    "                             name. Default: the GPU with the most GPU memory\n"
    WHIRL_HELP_MODES
    "  -h, --help                 this text\n";

// ---------------------------------------------------------------------------
// bench: prefill sweep + decode plain / MTP / MTP + n-gram


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
    const q::Config cfg = whirl::model::openArch(f).cfg;  // the architecture module, as whirl-server (arch025-N15d)
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
    if (whirl::settings::mtpQ4()) model.requantMtpQ4();
    loadLog(model, L.stats);
    applyRuntimeEnv(model);
    if (use_dflash) {
        use_dflash = loadDraftModelEnv(model, a);
        if (!use_dflash && !a.get("--modes")) modes = {"mtp-ngram", "mtp", "plain"};
    }
    const bool has_mtp = model.mtp.has_value() && whirl::settings::mtp();

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

}  // namespace cli
