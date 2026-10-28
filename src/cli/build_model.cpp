// whirl CLI: the CLI's own model building (arch025-C2): the GPU choice, the numerics request, Model::load with the CLI's
// options, the load log, the after-load switches (applyRuntimeEnv: the CLI front of numerics::applyFront, the draft
// vocabulary, the DFlash2 drafter). whirl-server builds its model in src/server/server_main.cpp (buildModel) and does
// not apply these switches (0.2.5; feat026-o6 shares them in 0.2.6)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"
#include "config/device_choice.h"  // the GPU choice shared with the server (arch025-N15b)

namespace cli {

// WHIRL_DRAFT_VOCAB: unset = the embedded 64k subset on dense qwen35 models with the 248320-token
// vocabulary; 64k = embedded; 48k / <file> = a frequency subset file; N = the first N draft-head rows
// (old experiment); off = full head. Special and byte tokens are always added.
void applyDraftVocab(q::Model& model, const Tokenizer& tok) {
    using K = q::DraftVocabChoice::Kind;
    const q::DraftVocabChoice dv = q::draftVocabChoice(whirl::settings::draftVocab(), q::exeDirectory());
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

// ---------------------------------------------------------------------------
// shared setup


// Runtime A/B switches shared by chat and bench (WHIRL_FLOAT_GEMV, ...).
// --draft dflash: true when --draft-model FILE (else WHIRL_DRAFT_MODEL) is a DFlash2 drafter that fits this model
// (headers only, before loading); otherwise a warning and the MTP drafts (whirl/dflash_policy.h).
namespace {
std::optional<std::string> draftModelOpt(const Args& a) {
    std::optional<std::string> cli;
    if (a.get("--draft-model")) cli = *a.get("--draft-model");
    return whirl::dflash::resolveDraftModel(cli, whirl::settings::draftModel());
}
}  // namespace
bool wantDFlash(const Args& a, const gguf::File& f) {
    const std::string mode = a.get("--draft") ? *a.get("--draft") : std::string("mtp");
    const auto src = whirl::dflash::parseSource(mode);
    if (!src) throw std::runtime_error("--draft must be mtp or dflash");
    const auto dm = draftModelOpt(a);
    if (*src == whirl::dflash::Source::dflash) {
        std::optional<std::string> cli;
        if (a.get("--draft-model")) cli = *a.get("--draft-model");
        if (const std::string note = whirl::dflash::draftModelNote(cli, whirl::settings::draftModel()); !note.empty())
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

// arch025-N2b (R-5): the CLI's after-load knobs are numerics::applyFront(Front::cli) on the model's plan (the rules
// that were here, main.cpp:1030-1048 and the chat's WHIRL_GDN_V0 at :1340, moved there in arch025-N2a). gdn_v0: the
// chat only (the other commands never applied it); its kernel is looked up only when WHIRL_GDN_V0 is set, as before.
// applyGemvEnv sets tune rows (not a flag) and stays here.
void applyRuntimeEnv(q::Model& m, bool gdn_v0) {
    namespace nu = whirl::numerics;
    nu::RuntimeValues rv;
    for (const auto& e : nu::RuntimeValues::names()) rv.*e.field = envGet(e.name);
    if (!gdn_v0) rv.gdn_v0.reset();
    hip::Function fv0 = nullptr;
    if (rv.gdn_v0) fv0 = m.module.getFunctionOpt("gdn_step_norm_v0");
    q::applyGemvEnv(m);
    nu::applyFront(m.num_plan, nu::Front::cli, rv, fv0 != nullptr);
    if (m.nf().gdn_v0) m.k.gdn_step_norm = fv0;
}

int selectDevice(const Args& a, std::string* info_line) {
    // the GPU choice shared with the server (src/config/device_choice.h, arch025-N15b): WHIRL_HIP_DEVICE, else the
    // s8-719 restriction + pickDevice; the restriction note goes to stderr before pickDevice
    const std::string spec = a.get("--device") ? *a.get("--device") : std::string();
    const whirl::settings::DeviceChoice ch = whirl::settings::chooseDevice(spec, settingsWarn, settingsWarn);
    const int dev = ch.dev;
    const int shown = ch.shown;  // the index as `whirl devices` prints it (before the restriction)
    const std::string& gpu_note = ch.note;
    const hip::DeviceInfo info = hip::describeDevice(dev);
    hip::setDevice(dev);
    if (info_line)
        *info_line = fmt("  whirl build, device %d: %s (%s)\n", shown, info.name.c_str(), info.gcn_arch.c_str()) +
                     (gpu_note.empty() ? std::string() : "  " + gpu_note + "\n") +
                     fmt("  HIP runtime: %s\n", hip::runtime::describe(hip::runtime::load()).c_str());  // the exact DLL loaded
    return dev;
}

// numerics mode: --precise / --balance[=items] / --fast[=items] / --mode M, else WHIRL_MODE, else balance
whirl::numerics::Request modeRequest(const Args& a) {
    return whirl::numerics::resolve(a.get("--mode") ? std::optional<std::string>(*a.get("--mode")) : std::nullopt, whirl::settings::mode());
}

// ctx_explicit: the context was asked for (--ctx / WHIRL_MAX_CTX, bench sizes): precise f16 KV that
// does not fit refuses instead of shrinking to at least min_ctx tokens. no_mtp: --draft dflash
// (the MTP block is not loaded; the DFlash2 drafter drafts instead)
Loaded loadModel(const Args& a, const gguf::File& f, u32 max_ctx, bool ctx_explicit, u32 min_ctx, bool no_mtp) {
    Loaded L;
    q::LoadOptions lo;
    lo.numerics = modeRequest(a);
    lo.ctx_explicit = ctx_explicit;
    lo.min_ctx = min_ctx;
    lo.no_mtp = no_mtp;
    // (arch025-N15c: the server's rule; the floor was max_batch_default 4096, now the scheduler's 1280 rows)
    if (auto v = whirl::settings::prefillBatch(settingsWarn)) lo.max_batch = vram::prefillBatchAtLoad(*v, q::max_batch_limit);
    lo.kv_req = q::kvRequestFromEnv();
    L.model = q::Model::load(f, max_ctx, L.stats, lo);
    q::Model& m = *L.model;
    if (envGet("TUNE_COLD")) m.tuneTable().tune_cold = true;
    if (auto v = envGet("TUNE_MASK")) m.tuneTable().tune_mask = std::stoull(*v, nullptr, 0);
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

}  // namespace cli
