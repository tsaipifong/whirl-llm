// Qwen3.5-family model (GGUF architectures `qwen35` / `qwen35moe`) on HIP:
// hybrid Gated DeltaNet + gated full attention, dense SwiGLU or MoE FFN,
// optional MTP (nextn) head. Weights stay in their GGUF quant formats on the
// GPU; the kernels come from the embedded code object (kernels/).
// SPDX-License-Identifier: Apache-2.0
//
// Layer schedule: layer il is full attention when (il + 1) % interval == 0,
// otherwise Gated DeltaNet.
//
// API notes (for the server and the CLI):
// - Model is one loaded checkpoint with its activations, KV pool and
//   recurrent state. All work runs on Model::stream; nothing here is
//   thread-safe.
// - Single-sequence entry points (forward, prefill*, verify*, mtp*) act on
//   the current sequence (selectSeq). Without setupSeqs there is one
//   implicit sequence with an identity page table over the whole KV pool.
// - Device buffers are hip::DevPtr (uint64 addresses); the members are public
//   on purpose (the server reads logits / hn / mtp_h / control words and
//   copies recurrent state for its checkpoints).
// - Errors are thrown as ModelError (code() names the condition, e.g.
//   "UnsupportedArch", "UnsupportedTensorType", "ContextTooLong").
//
// Header stability: the names below mirror the research prototype's model
// interface; additions are allowed, renames are not.

#pragma once

// qwen35::Context (the execution state and the forward; arch025-lane2 N9 S3/S4) and everything the forward shares
#include "model/qwen35/context.h"
#include "whirl/device_pick.h"  // qwen35::pickDevice / restrictDevice (moved from here, arch025-lane1 leg 3)

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::config {
class EnvSnapshot;  // src/config/env_snapshot.h (arch025-N1)
}  // namespace whirl::config

// envGet / envFlag (WHIRL_<name>): src/config/env_get.h (arch025-lane2 D2.2)
#include "config/env_get.h"

namespace whirl::qwen35 {

// KV formats: whirl::KvFormat (whirl/kv_format.h). A request is std::optional<KvFormat>: nullopt (WHIRL_KV unset or
// "auto") = the numerics mode decides (numerics::chooseKv: precise f16; balance q8h on dense models, f16 on MoE;
// fast q4, q8h on dense models on the Radeon 8060S); a format is a debug override (WHIRL_KV).

// WHIRL_KV=auto|f16|q8|q8h|q8v|q4 (nullopt when unset or auto; throws std::invalid_argument
// for another value). Model::load applies it when LoadOptions::kv_req is
// nullopt, like the prototype's process-wide KV setting.
std::optional<KvFormat> kvRequestFromEnv();

// Before loading anything: empty when the DFlash2 drafter `draft` fits the target model `target`
// (both GGUF headers only), else the reason (e.g. a MoE target such as Ornith has no drafter).
std::string dflashIncompatible(const gguf::File& target, const gguf::File& draft);
// Same for the drafter at draft_path, opening it first: a file that cannot be opened / parsed gives
// the reason too (never throws). On success *opened (if given) holds the open file.
std::string dflashIncompatibleFile(const gguf::File& target, const std::string& draft_path, std::optional<gguf::File>* opened = nullptr);

// DFlash2 drafter run state (dflashSetup): per-slot context ring and the scratch of one draft block.
struct DFlashSlotCtx {
    std::vector<DevPtr> kc, vc;        // per drafter layer: [ring][n_head_kv][head_dim] f32 (RoPE'd K, V)
    DevPtr cpos = 0;                   // [ring] i32: position of each ring row, -1 = empty
    std::vector<std::int32_t> cpos_h;  // host mirror of cpos
};

struct DFlashRun {
    std::uint32_t ring = 0, block = 0, ingest_rows = 0;
    DevPtr ids = 0, pos = 0, x = 0, h = 0, h2 = 0, dyn = 0, q = 0, kb = 0, vb = 0, att = 0, o = 0, g = 0, u = 0, e = 0;
    DevPtr logits = 0, cand = 0, unary = 0, gate = 0, lat = 0, out = 0;
    DevPtr ing_g = 0, ing_k = 0, ing_v = 0, ing_pos = 0;
    DevPtr dbg = 0;  // when set: [n_layer][block][n_embd] hidden after each drafter layer (dflash-check)
    std::vector<DFlashSlotCtx> slots;
};

struct LoadOptions {
    // Prefill batch (rows per forward) the buffers are sized for.
    std::uint32_t max_batch = max_batch_default;
    // Keep token_embd in pinned host memory instead of VRAM.
    bool embd_on_host = false;
    std::optional<KvFormat> kv_req;  // nullopt: WHIRL_KV, else the numerics mode decides
    // Numerics mode (whirl/numerics.h): precise / balance (default) / fast and the lossy items.
    // Precise: no fp8 / MoE fp8 / f16-WMMA DeltaNet / h16, f16 KV (WHIRL_KV still honoured).
    whirl::numerics::Request numerics;
    // CLI KV pool (max_ctx_req > 0): the context was given explicitly (precise f16 KV that does not
    // fit then refuses instead of shrinking), and the smallest context a shrink may leave.
    bool ctx_explicit = false;
    std::uint32_t min_ctx = 0;
    // Optional external code object file (development: WHIRL_CODE_OBJECT).
    std::string code_object;
    // Skip the MTP (nextn) block: no weights, no KV layer (--draft dflash drafts with DFlash2).
    // Ignored when the code object lacks the DFlash2 kernels (Model::mtp_skipped stays false):
    // the caller then falls back to the MTP drafts.
    bool no_mtp = false;
    // The load's WHIRL_* environment (arch025-N1, src/config/env_snapshot.h). Null: Model::load reads one at its
    // start. Model::load keeps it (Model::env_snap) for loadOrTune, applyGemvEnv and the selftest checks.
    std::shared_ptr<const whirl::config::EnvSnapshot> env;
};

// LoadEstimate: whirl/model_types.h (arch025-M5: the architecture module's sizing interface returns it)
// no_mtp: the load skips the MTP (nextn) block (--draft dflash): no nextn weights, no MTP KV layer.
LoadEstimate estimateLoad(const gguf::File& f, const Config& cfg, bool embd_on_host, bool no_mtp = false);
// Bytes a DFlash2 drafter's weights take in VRAM (Model::loadDrafter: every tensor, + MXFP4 row refs).
std::uint64_t dflashWeightBytes(const gguf::File& draft);
// bufferBytes (the prefill / decode buffers of Model::load): src/model/qwen35/scratch.h (arch025-lane2 N9 S2c)
// The attention split count Model::load uses for the KV format (and WHIRL_FD_SPLITS).
std::uint32_t resolveFdSplits(KvFormat kv);
// KV bytes per token (Model::kvBytesPerTokenFmt) from the config alone.
std::uint64_t kvBytesPerTokenCfg(const Config& cfg, bool has_mtp, KvFormat kv);

// ---------------------------------------------------------------------------
// kernels (table from whirl/kernels_abi.h)

// (Kernels = kernels::KernelTable: src/model/qwen35/context.h)
// the attention variants of the KV format (KernelTable::load)
Kernels loadKernels(const hip::Module& m, KvFormat kv);
// ---------------------------------------------------------------------------
// the model

// Model = Context (src/model/qwen35/context.h: weights, KV pool, sequences, activations, tune table, the forward) +
// loading (Model::load, loadDrafter), the DFlash2 drafter forward, the KV pool management (allocKvPool .. setupSeqs),
// the draft-head builds, autotune / tune cache and the self-checks (arch025-lane2 N9 S3/S4).
class Model : public Context {
public:
    // arch025-gmsurv test seam: tests/iso/model.core defines ModelTestAccess (allocScratch on the HIP double's
    // allocation recorder, mutant m3-scratch-fd); no product code defines or uses it. A friend declaration changes
    // no member's access, so no decorated name and no product object changes (seam_objeq proof).
    friend struct ModelTestAccess;
    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    ~Model();

    // Loads weights, allocates activations and (max_ctx > 0) the KV pool for
    // one sequence. The current HIP device must be set.
    static std::unique_ptr<Model> load(const gguf::File& f, std::uint32_t max_ctx, LoadStats& stats,
                                       const LoadOptions& opt = {});
    hip::Module module;
    device::StreamH stream_own;
    // numerics mode: the request (LoadOptions), the resolved plan (loadOrTune), load-time notes
    // (precise KV shrink) for the caller's log
    whirl::numerics::Request num_req;
    std::string load_note;
    // the load's environment (LoadOptions::env; null for a Model that did not come from Model::load)
    std::shared_ptr<const whirl::config::EnvSnapshot> env_snap;
    // the KV format and why (fixed at load: numerics::chooseKv)
    whirl::numerics::KvChoice kv_choice;
    std::optional<KvFormat> kv_req;  // the load's request (LoadOptions::kv_req, else WHIRL_KV; nullopt: the mode chose)
    // Loads a DFlash2 GGUF (weights only), checks it against this target, returns the VRAM bytes.
    std::uint64_t loadDrafter(const gguf::File& f, LoadStats& stats);
    // Turns the capture on for the drafter's target layers, keeping up to `rows` rows per run().
    void enableCapture(std::uint32_t rows);
    // Capture for explicit layer-input ids (0 = embeddings, n_layer = last layer output); off: empty.
    void enableCaptureLayers(std::span<const std::uint32_t> ids, std::uint32_t rows);
    void disableCapture();
    // selftest: the capture copies exactly x for each slot (random x, no forward needed).
    bool checkCapture(std::string& log);

    // ---- DFlash2 drafter forward (src/model/dflash.cpp); needs loadDrafter and the dflash_* kernels
    std::unique_ptr<DFlashRun> dfr;
    // First kernel the drafter forward needs that the code object lacks ("" = all present). Both
    // kernel sets carry kernels/dflash.hip; an old WHIRL_CODE_OBJECT may not.
    std::string dflashMissingKernel() const;
    // Model::load with LoadOptions::no_mtp skipped the MTP block (false: kernels missing, MTP loaded)
    bool mtp_skipped = false;
    // Buffers for `slots` contexts (ring = the drafter's window, else 2048) and ingest batches of
    // up to `ingest_rows` rows; `debug` keeps every layer's block hidden for dflash-check.
    void dflashSetup(std::uint32_t slots, std::uint32_t ingest_rows, bool debug = false);
    // VRAM dflashSetup allocates per slot (the context ring of every drafter layer); 0 without a drafter (v0.2.2.2: the
    // server counts it when it re-fits --parallel after loading)
    std::uint64_t dflashSlotBytes() const;
    // Context rows for positions pos0 .. pos0 + n - 1 from target features in layer-major layout
    // (feature of row r, target layer j at feats + (j * stride + r) * n_embd floats): fc + norm,
    // then each layer's K (k_norm, RoPE) and V into the slot's ring.
    void dflashIngest(std::uint32_t slot, DevPtr feats, std::uint32_t stride, std::uint32_t row0, std::uint32_t n, std::uint32_t pos0);
    // One draft block [anchor, MASK x (block - 1)] at positions pos .. pos + block - 1 against the
    // slot's context (< pos): up to block - 1 draft tokens (greedy lattice walk; p-min / n-min).
    std::vector<std::uint32_t> dflashDraft(std::uint32_t slot, std::uint32_t anchor, std::uint32_t pos, float p_min = 0,
                                           std::uint32_t n_min = 0);
    // Forget context rows at positions >= keep (rejected drafts, rollback).
    void dflashTrim(std::uint32_t slot, std::uint32_t keep);
    std::uint64_t kvBytesPerTokenFmt(KvFormat kv) const;
    std::uint64_t kvBytesPerToken() const { return kvBytesPerTokenFmt(kv_fmt); }
    std::uint32_t kvLayers() const;
    const char* kvName() const;

    // ---- KV pool and sequences
    void allocKvPool(std::uint32_t tokens);
    // Frees the KV pool arrays so allocKvPool can run again with another size (issue #5: the server
    // shrinks the pool when the WDDM budget read after allocation is lower). Only before anything
    // used the pool; the page table of setupSeqs stays.
    void freeKvPool();
    void mapPages(std::uint32_t s, std::uint32_t first, std::span<const std::int32_t> phys);
    void setupSeqs(std::uint32_t n, std::uint32_t slot_ctx);
    void setPending(std::uint32_t s, std::uint32_t rows) { seqs[s].npend = rows; }
    void restoreSeqSnapshot(std::uint32_t s, std::uint32_t set);

    // MTP block Q6_K matrices as Q4_K (drafts only).
    void requantMtpQ4();
    enum class DraftHeadKind { q4, d2 };
    void buildDraftHeadEx(DraftHeadKind kind);
    // Keep only these rows (ascending token ids, see draftVocabIds) of the 2-bit draft head;
    // drafts map back to token ids, the trunk and its output head are untouched. False (and no
    // change) without the 2-bit head or the copy_rows_map kernel.
    bool setDraftVocab(std::span<const std::uint32_t> ids);

    // ---- prefill GEMM autotune
    void autotuneGemm(std::size_t bucket, std::uint32_t reps, std::string* log);
    std::string writeTune() const;
    bool readTune(std::string_view text, std::size_t n_buckets);
    bool checkGemvq(std::string& log);
    bool checkGemvBitwise(std::string& log);
    bool checkPrefillInvariance(std::string& log);
    // precise decode: every small-batch row (n = 1..16 / 32) bitwise == the same row in a
    // 200-row prefill GEMM, per weight type; MoE: a token alone == inside a batch
    bool checkPreciseDecode(std::string& log);
    // wide: the grouped launch is attn_wsplit2 (<= 32 columns) instead of attn_wsplit1
    bool checkAttnGroups(std::string& log, std::uint32_t p0, std::uint32_t n, bool wide = false);

    // ---- stage-bench seam (tests/bench/stage_bench_main.cpp; arch025-M3 C7, src/model/qwen35/layer_subset.cpp):
    // run() over a subset of the trunk layers. The constructor swaps the layer list and the per-layer arrays
    // (kcache / vcache / kscale / vscale, conv_state / ssm_state) for the selected layers (in `sel` order) and drops
    // the decode graph; the destructor puts everything back. Test-only.
    class LayerSubset {
    public:
        LayerSubset(Model& m, const std::vector<std::uint32_t>& sel);
        ~LayerSubset();
        LayerSubset(const LayerSubset&) = delete;
        LayerSubset& operator=(const LayerSubset&) = delete;

    private:
        Model& m_;
        std::vector<Layer> layers0_;
        std::vector<DevPtr> kc0_, vc0_, ks0_, vs0_, conv_, ssm_;
        std::span<DevPtr> conv0_, ssm0_;
    };

private:
    // Model::load's prefill / decode buffers (Scratch) and the sizing self-check (src/model/qwen35/scratch.cpp)
    void allocScratch(std::uint32_t fd_splits_);
    Mat requantQ4k(const Mat& w);
    std::vector<Mat*> tuneMats(Layer& L);
};

// ---------------------------------------------------------------------------
// prefill GEMM tune cache and runtime switches (WHIRL_FP8, WHIRL_GDN_WMMA, ...):
// reads %LOCALAPPDATA%\whirl\tune4-<file>-<size>[-gfx1151].txt, else autotunes and
// writes it. `log` receives the human-readable lines.
void loadOrTune(Model& m, const std::string& model_path, std::string& log);
// arch025-N2b: numerics::plan's inputs from a loaded Model (tune.cpp; k_gdn_v0 stays false: a front end that applies
// WHIRL_GDN_V0 looks the kernel up itself, as before)
whirl::numerics::Target numericsTarget(const Model& m);
whirl::numerics::Request numericsRequest(const Model& m, const whirl::config::EnvSnapshot& env);
// arch025-N2b-r2: the load-time flags (numerics::loaderFlags: fp8_prefill, moe_fp8, moe_mxw, moe_wide, moe_rbf) from
// m.num_req and the load's snapshot into m.num_plan.flags. Model::load calls it on every load, so the commands that
// never call loadOrTune (selftest, dflash-check) run with them; loadOrTune's plan then sets every flag again.
void applyLoadFlags(Model& m);

// Multi-token GEMV selection overrides (A/B testing), as "nt:value,..." lists:
// rows per wave (r = 1, 2, 4), WMMA variant for every type (0..8), and the
// variant of the Q6_K output head. applyGemvEnv reads WHIRL_GEMV_R /
// WHIRL_GEMV_W / WHIRL_GEMV_WH. Malformed items throw std::invalid_argument.
void parseGemvR(Model& m, std::string_view list);
void parseGemvW(Model& m, std::string_view list);
void parseGemvWHead(Model& m, std::string_view list);
void applyGemvEnv(Model& m);

// ---------------------------------------------------------------------------
// device selection: qwen35::pickDevice / restrictDevice are declared in whirl/device_pick.h (included above;
// arch025-lane1 leg 3: config.device needs them without this header)

// ---------------------------------------------------------------------------
// MTP draft-count control and n-gram (prompt-lookup) drafting: src/spec/policy.h (arch025-M4: MtpDefaults,
// mtpDefaults, the draft-head vocabulary choice, DraftAccept .. NgramPolicy). The two parts that need the model
// library stay here:

// the embedded 64k subset (data/draft_vocab/subset_64k.bin, built by whirl-cloud tools/vocab_subset);
// defined in draft_vocab_embed.cpp (whirl_model only)
std::vector<std::uint32_t> embeddedDraftVocab64k();
// draftWindowArg (KvArgs::win of the MTP draft attention window): src/model/qwen35/context.h

}  // namespace whirl::qwen35
