// whirl chat: generate a reply to one prompt (arch025-C2: from src/cli/main.cpp)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"
#include "cli/dflash_dump.h"

namespace cli {

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
    "  --device SPEC, -dev SPEC   GPU: pci:BUS:DEVICE (Task Manager > Performance > GPU > Physical location: PCI bus 71,\n"
    "                             device 0 -> pci:71:0), an index from 'whirl devices', r9700, 8060s, or a part of the\n"
    "                             name. Default: the GPU with the most GPU memory\n"
    WHIRL_HELP_MODES
    "  -h, --help                 this text\n";

namespace {

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

}  // namespace

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
    const q::Config cfg = whirl::model::openArch(f).cfg;  // the architecture module, as whirl-server (arch025-N15d)
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
        device::PinnedBuf b;  // owner; p is its address as float*
        float* p = nullptr;
        std::size_t n = 0;
        explicit Pinned(std::size_t count)
            : b(device::PinnedBuf::alloc(device::hipBackend(), count * 4)), p(static_cast<float*>(b.get())), n(count) {}
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
    // (WHIRL_GDN_V0: applyRuntimeEnv(model, true) below, arch025-N2b)
    // MTP block as Q4_K from Q6_K or Q8_0 (drafts only; default, WHIRL_MTP_Q4=0 keeps the file's types)
    if (whirl::settings::mtpQ4()) model.requantMtpQ4();
    loadLog(model, L.stats);
#if WHIRL_HAVE_VISION
    // vision: encode the images now (borrowing the prefill scratch), then let the
    // prefill inject their rows and use multi-section RoPE
    if (vis) {
        if (auto mv = envGet("VIS_MODE")) vis->mode = vision::parseMode(*mv).value_or(vision::Mode::automatic);
        std::array<std::array<std::uint64_t, 2>, 16> lend_raw{};
        const std::size_t n_lend = model.lendScratch(lend_raw);
        const std::vector<vision::Region> lend = vision::lendRegions(lend_raw, n_lend);
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

    // (CLI-SCHED: not below one schedule chunk + its merged tail, never above the loaded batch: the server's rule,
    // arch025-N15c)
    if (auto v = whirl::settings::prefillBatch(settingsWarn)) model.max_batch = vram::prefillBatchCap(model.max_batch, *v);
    q::Profile prof;
    const bool profiling = envGet("PROFILE").has_value();
    if (profiling) model.prof = &prof;
    applyRuntimeEnv(model, true);
    if (use_dflash) use_dflash = loadDraftModelEnv(model, a);

    // MTP speculative decode is exact greedy: on whenever the checkpoint has a nextn
    // layer; WHIRL_MTP=0 turns it off.
    const bool use_mtp = model.mtp.has_value() && logits_out == nullptr && whirl::settings::mtp();
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
        const std::size_t split = tok.ct.thinkOpenSplit(ids);
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
        const std::vector<std::size_t> ends = dump_path ? std::vector<std::size_t>{} : serverForwardEnds(tok, ids, model.max_batch, device::profile(model.arch), model.cfg.moe);
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
                hip::download(row_f32.data(), model.scratch().logits, n * V * 4);
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
        // bug-emptyrootpaths: empty = no dump (create_directories("") throws in MSVC, "" + "/feats.npy" is the drive root)
        // (the rule is dflash_dump.h dflashDumpDir, pinned by whirl-tests testDFlashDumpDir; arch025-gaptests G19)
        const std::optional<std::string> dump_env = envGet("DFLASH_DUMP");
        if (dump_env && dump_env->empty()) std::fprintf(stderr, "WHIRL_DFLASH_DUMP is empty: no feature dump\n");
        if (const std::optional<std::string> dir = dflashDumpDir(dump_env)) dflashDumpFeatures(model, *dir, next, static_cast<u32>(ids.size()));
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
            model.buildDraftHeadEx(whirl::settings::draftHeadQ4() ? q::Model::DraftHeadKind::q4 : q::Model::DraftHeadKind::d2);
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

}  // namespace cli
