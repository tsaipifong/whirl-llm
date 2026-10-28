// qwen35::Context (arch025-lane2 N9 S3/S4): the execution state of a loaded Qwen3.5-family model and every forward
// function (src/model/qwen35/forward/*.cpp, context.cpp): the weights, KV pool, sequence table, activations and tune
// table (its bases), the forward's switches, recurrent-state views, MTP / draft-head / capture / vision state, and
// the device-memory ledger (alloc / allocZero / freeAlloc). class Model (include/whirl/model.h) derives from it and
// adds loading, the DFlash2 drafter, the KV pool management, autotune and the self-checks; callers keep writing
// m.forward(...), m.cfg, m.stream. The declarations moved verbatim from include/whirl/model.h (same order within
// each class); the forward sources include this header only (forward_internal.h), never whirl/model.h.
// Like llama.cpp's llama_context (state + graph execution) beside llama_model (load), see progress/arch025-lane2.PRIOR_ART.md.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/handles.h"  // src/: RAII owners of the device / pinned buffers, events and streams (arch024 S11)
#include "model/qwen35/weights.h"  // src/: Mat, Layer, ..., qwen35::Weights (arch025-M1)
#include "model/qwen35/kv.h"  // src/: qwen35::KvPool, SeqTable, Seq, kvArrays (arch025-M2)
#include "model/qwen35/scratch.h"  // src/: qwen35::Scratch (arch025-M3)
#include "model/qwen35/tune_table.h"  // src/: qwen35::TuneTable, GemvR / GemvW (arch025-M3)
#include "spec/policy.h"  // src/: the MTP / n-gram draft policy, mtpDefaults, the draft vocabulary choice (arch025-M4)
#include "whirl/gguf.h"
#include "whirl/model_types.h"
#include "whirl/device_profile.h"
#include "whirl/hip.h"
#include "whirl/kernels_abi.h"
#include "whirl/ops/act.h"
#include "whirl/ops/attention.h"
#include "whirl/ops/matmul.h"
#include "whirl/ops/moe.h"
#include "whirl/ops/mtp.h"
#include "whirl/numerics.h"
#include "whirl/relax_accept.h"
#include "whirl/value_scope.h"
#include "whirl/spec_sample.h"
#include "whirl/vismap.h"

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// ModelError: src/model/qwen35/model_error.h; DFlashW (the capture reads dflash->cfg), LoadStats: load_types.h
#include "model/qwen35/load_types.h"
#include "model/qwen35/model_error.h"

namespace whirl::qwen35 {

// ---------------------------------------------------------------------------
// constants (shared with the kernels; see kernels_abi.h)

// n_types (> every GgmlType used): src/model/qwen35/weights.h
// max_batch_default .. ctl_words, Config, VSeg / PSeg / PRows / MSeg / BatchPlan: whirl/model_types.h (arch025-M3;
// the server core includes only that header); DraftAccept, DraftTiming, pickDrafts .. NgramPolicy: src/spec/policy.h
// (arch025-M4)

// Prefill GEMM autotune buckets (n_tune: src/model/qwen35/weights.h).
constexpr std::uint32_t tune_small_max = 1024;
constexpr std::array<std::uint32_t, n_tune> tune_sizes = {512, 4096, 128, 32, 48, 64, 96, 192, 256, 384, 1024};
std::size_t tuneBucket(std::uint32_t n);

using GemmCfg = kernels::GemmCfg;
inline constexpr const auto& gemm_cfgs = kernels::kGemmCfgs;
inline constexpr const auto& gemms_cfgs = kernels::kGemmsCfgs;
inline constexpr const auto& gemm8_cfgs = kernels::kGemm8Cfgs;
inline constexpr const auto& gemm8t_cfgs = kernels::kGemm8tCfgs;
// n_choices (distinct prefill GEMM choices): src/model/qwen35/tune_table.h (arch025-lane2: tune_table.cpp checks it)

// Multi-token GEMV tables.
constexpr std::uint32_t n_gemvw = kernels::kNGemvw;
inline constexpr const auto& gemvw_rows = kernels::kGemvwRows;
// GemvR, GemvW, defaultGemvR / W / WHead: src/model/qwen35/tune_table.h (arch025-M3)

// ---------------------------------------------------------------------------
// kernel-argument structs and kernel table: the single definition lives in
// whirl/kernels_abi.h (kernels area); the model uses them under these names.

using GdnSeg = kernels::GdnSeg;
using GdnSegs = kernels::GdnSegs;
using GvArgs = kernels::GvArgs;
using KvArgs = kernels::KvArgs;
using Tok16 = kernels::Tok16;
using RowTab = kernels::RowTab;
using RowIdx = kernels::RowIdx;
using AwGroups = kernels::AwGroups;
static_assert(kv_page == kernels::kKvPage && gdn_max_seg == kernels::kGdnMaxSeg && gdn_max_snap == kernels::kGdnMaxSnap &&
              max_small_batch == kernels::kMaxSmallBatch && max_drafts == kernels::kMaxDrafts && n_types == kernels::kNTypes &&
              max_verify_rows == kernels::kMaxVerifyRows);
static_assert(ctl_rows == kernels::kCtlRows && ctl_drafts == kernels::kCtlDrafts && ctl_probs == kernels::kCtlProbs &&
              ctl_nd == kernels::kCtlNd && ctl_stop == kernels::kCtlStop && ctl_words == kernels::kCtlWords);
// ---------------------------------------------------------------------------
// configuration and weights


// Mat, AttnW, GdnW, MoeW, MtpW, Layer, MatList, rowsContiguous, concatRows: src/model/qwen35/weights.h (arch025-M1)

// ---------------------------------------------------------------------------
// load-time options (set before Model::load)

using Kernels = kernels::KernelTable;

// ---------------------------------------------------------------------------
// profiling (WHIRL_PROFILE=1)

enum class OpClass { embed, norm, quant, matmul, attn, gdn_prep, gdn_rec, gdn_norm, misc, head };
constexpr std::size_t n_classes = 10;
const char* opClassName(OpClass c);

struct Profile {
    std::array<double, n_classes> ms{};
    std::uint64_t matmul_bytes = 0;
    std::vector<device::EventH> events;  // timing events, made on first use (adopted from hip::eventCreate())
    std::vector<OpClass> classes;
    std::size_t count = 0;
    static constexpr std::size_t cap = 2048;

    Profile();
    ~Profile();
    Profile(const Profile&) = delete;
    Profile& operator=(const Profile&) = delete;
    void record(hip::Stream stream, OpClass cls);
    void flush();
    void reset();
};

// ---------------------------------------------------------------------------
// sequences and batched steps

// (Seq, one sequence / server slot: src/model/qwen35/kv.h since arch025-M2)

// ---------------------------------------------------------------------------
// the execution context

// KvArgs::win for the MTP draft attention window (moved from include/whirl/model.h): W / 64 (rounded up, <= 65535) in
// the low 16 bits, the context threshold / 1024 in the high bits; 0 (off) for W = 0.
inline std::int32_t draftWindowArg(std::uint32_t w, std::uint32_t min_ctx) {
    if (w == 0) return 0;
    const std::uint32_t lo = std::min<std::uint32_t>((w + 63) / 64, 0xffffu);
    const std::uint32_t hi = std::min<std::uint32_t>(min_ctx / 1024, 0x7fffu);
    return static_cast<std::int32_t>(lo | hi << 16);
}

// Context derives from Weights (tok_embd, embd_host, output, output_norm, layers, mtp, type_bytes, mx_fold_lossy;
// src/model/qwen35/weights.h, bound by WeightLoader in Model::load / loadDrafter, arch025-M1), KvPool and SeqTable
// (src/model/qwen35/kv.h, arch025-M2), and from Scratch and TuneTable (src/model/qwen35/scratch.h / tune_table.h,
// arch025-M3: scratch() / tuneTable(); protected so Model's members reach them, as before when they were Model's
// private bases). The launch internals are protected for the same reason (tune.cpp / dflash.cpp are Model members).
class Context : public Weights, public KvPool, public SeqTable, protected Scratch, protected TuneTable {
public:
    Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // ---- configuration
    Config cfg;
    std::uint32_t max_ctx = 0;
    std::uint32_t max_batch = max_batch_default;
    hip::Arch arch = device::k_default_arch;
    std::vector<std::int32_t> host_ids;
    Kernels k;
    std::vector<device::DevBuf> allocations;  // owners of alloc(); bytes() is each one's size
    std::uint64_t alloc_total = 0;  // bytes live through alloc() (sizing self-check in load; freeAlloc subtracts)
    Profile* prof = nullptr;
    hip::Stream stream = nullptr;  // the work stream (owned by stream_own; every kernel call reads this value)
    std::unique_ptr<hip::Graph> decode_graph;
    bool head_phase = false;
    std::uint32_t row_n = 0;
    std::array<std::int32_t, max_verify_rows> row_pos{};
    std::array<std::int32_t, max_verify_rows> row_base{};
    // most rows the GEMV / decode-fusion paths take in the current forward: max_small_batch,
    // raised to max_verify_rows inside a wide batched verify / MTP step (see wideOk())
    std::uint32_t small_max = max_small_batch;
    // research switches WHIRL_HEAD_GEMVX / PREC_GEMVH / PREC_GS (arch025-F7: config::forwardEnv of the load's
    // snapshot, set by Model::load; the defaults = not set). head_gemvx: the output head may take gemvx in a wide
    // verify; prec_gemvh false: precise rows skip gemvh; prec_gs >= 0: forced precise gemms slot.
    bool head_gemvx = false;
    bool prec_gemvh = true;
    int prec_gs = -1;
    // precise decode: the kernels every weight type of this model needs; "" = all present
    std::string precMissing() const;

    // ---- activations: qwen35::Scratch (src/model/qwen35/scratch.h, arch025-M3), a private base; outside the model
    // read them through scratch() (read-only view, like llama_context::get_logits)
    const Scratch& scratch() const { return *this; }
    // ---- prefill GEMM / GEMV tune table: qwen35::TuneTable (src/model/qwen35/tune_table.h, arch025-M3), a private base
    TuneTable& tuneTable() { return *this; }
    const TuneTable& tuneTable() const { return *this; }
    whirl::numerics::Plan num_plan;
    // arch025-N2b (R-5): the execution flags are the plan's (numerics::Flags, include/whirl/numerics.h; decided by
    // numerics::plan in loadOrTune, the front end's after-load knobs by numerics::applyFront; the five load-time ones also
    // by applyLoadFlags in Model::load, arch025-N2b-r2). Read them through nf().
    // The only other writers are the tune / selftest probes in tune.cpp (save, run a kernel, restore). Notes moved here
    // from the old members:
    //   attn_wide: verify attention: groups of up to 32 columns (attn_wsplit2) when that lets a sequence's rows share one K/V pass (WHIRL_ATTN_WIDE=0: <= 16 columns as before)
    //   attn_dq4: q4 KV decode / verify attention through attn_dq4 when the code object has it (fast mode on the 8060S; WHIRL_ATTN_DQ4=0: attn_wsplit1_q4 as before)
    //   dq4_dyn: gfx1151 attn_dq4: launch only as many splits as the rows' key partition uses (same bits as the fixed fd_splits grid, cheaper combine; FAST-8S-1; WHIRL_DQ4_DYN=0: the fixed grid)
    //   attn_ws4: gfx1151 f16 / int8 KV decode and verify attention through attn_ws4 (single-row groups) and attn_ws4g (multi-row verify groups, one KV read per split for up to 8 rows) when the code object has them; same partial bits as attn_wsplit1 one-query groups (DES-ATTN-8S T2 / T3). WHIRL_ATTN_WS4=0: the old attn_wsplit1 one-query groups (debug fallback, logged as override)
    //   wide_verify: batched verify of more than 16 rows (WHIRL_WIDE_VERIFY=0: at most 16 as before)
    //   moe_wide: wide verify on MoE models (balance / fast; WHIRL_MOE_WIDE=1): 17..32-row verify batches take the int8 decode experts, like a 1-token decode. Off by default: each extra row reads its own experts, and Ornith C=4 measured 1.7-2.4% slower with it (KG-2)
    //   head_chunk: output-head rows per range in a wide verify (two 16-token passes per range; WHIRL_HEAD_CHUNK)
    //   gv_group: Grouped same-input GEMV launches (WHIRL_GV_GROUP=0: off) and the largest token count that takes the grouped multi-token twins (WHIRL_GV_NMAX, 0: per type). Set by the CLI's applyRuntimeEnv after the load; the server keeps the defaults. (arch025-F7: were the process-wide qwen35::gv_group / gv_nmax, same initial values.)
    //   prec_dec: precise decode (numerics precise mode, P-8): no int8 activations anywhere in decode / verify / small batches. Every matmul of n <= gvMax() rows takes the f16-activation WMMA GEMM (f32 accumulation; the same kernels and per-element bits as the prefill GEMM, so a row's result does not depend on n, the batch's other rows or the tuned config), MoE experts the grouped f16 GEMM. WHIRL_Q8DEC=1 / =0 overrides.
    //   gdn_strix: GDN-8S (gfx1151, default on for dense and MXFP4 MoE models since rc6 / rc7; WHIRL_GDN_STRIX=0 turns it off): strix-llama.cpp's f16-WMMA chunked prefill (gdn_sx_prep / gdn_sx_scan) for runs of >= gdn_strix_min rows; same precision class and mode policy as gdn_wmma
    //   gdn_ba_on: n>16 GDN prefill: one [beta; alpha] GEMM + gdn_gates_ba (WHIRL_GDN_BA=0 off)
    //   gdn_in2_on: n>16 GDN prefill: one norm -> fp8 (qkv / gate) + f16 x16b ([beta; alpha]) (WHIRL_GDN_IN2=0 off)
    //   attn_dqf_on: q8 / q8h KV prefill attention: dequantize the sequence's keys into f16 rows in ffn_g / ffn_u, then the f16 attn_kg (bit-identical to attn_kg_q8; WHIRL_ATTN_DQF=0 off); dq_ptab: identity page table
    //   attn_seg: KG-3 (gfx1151): keys per range of the prefill attention key-range series (attn_prefill_wmma_s; 0 = one range when the keys fit the f16 scratch). WHIRL_ATTN_SEG (multiple of 256); f16 KV also walks its cache in ranges when set. Bit-identical either way.
    //   attn_kg_on: GQA-grouped prefill attention (f16 / q8 / q8h / q8v KV; WHIRL_ATTN_KG=0 off)
    //   h16_dq: KG-1: f16 outputs from quantized weights whatever the tuned choice (a fused quantized gemm_cN choice runs as dequant + gemm_chN, bitwise the same as gemm_cN), so whether a GEMM writes f16 never depends on autotune timings (dense non-MXFP4 models, R9700)
    //   draft_window: WHIRL_DRAFT_WINDOW / _MIN: MTP draft attention over the first 256 + the last W positions once the context reaches draft_window_min (W = 0: off); the trunk and verify always see everything
    //   moe_rbf: MXFP4 x fp8 grouped GEMM: row-block-fast gemm8r_* when present (WHIRL_MOE_RBF=0 -> gemm8_moe*)
    const whirl::numerics::Flags& nf() const { return num_plan.flags; }
    // numerics item relaxacc (fast): relaxed draft acceptance (from nf().relax_*)
    whirl::relax::Params relax() const { return {nf().relax_on, nf().relax_k, nf().relax_alpha, nf().relax_eps, nf().relax_delta}; }
    std::uint32_t fwd_pos0 = 0;
    static constexpr std::uint32_t kDqMaxKeys = ops::kDqMaxKeys;  // dq_ptab covers keys [0, kDqMaxKeys)
    bool out_h16 = false;

    // ---- per-layer state
    // (arch025-M2: kcache / vcache / kscale / vscale, mtp_kc / _vc / _ks / _vs, pool_pages and the KV format kv_fmt
    // (kv_choice.kv, fixed by Model::load through KvPool::fixFormat; read-only) are KvPool's, src/model/qwen35/kv.h)
    // flash-decoding split cap (part_ml / part_acc are sized for it): 64; q4 KV 256 (long-context
    // decode is latency bound at 64 splits, so the smaller K / V reads only pay off with more
    // blocks in flight). WHIRL_FD_SPLITS=N (64..256, multiple of 64) overrides.
    std::uint32_t fd_splits = 64;
    // (arch025-M2: seq_pages, ptab, state_tables, seqs, cur_seq, kv_off are SeqTable's, src/model/qwen35/kv.h)
    // per-layer recurrent state of the current sequence (views; swapping an
    // entry, e.g. restoreSnapshot, updates the selected Seq too)
    std::span<DevPtr> conv_state, ssm_state;
    std::array<std::vector<DevPtr>, gdn_max_snap> snap_conv, snap_ssm;
    std::uint32_t snap_sets = 0;
    DevPtr gate = 0;
    bool gdn_replay = false;
    std::vector<std::uint32_t> gdn_ord;
    std::uint32_t ctl_off = 0;
    DevPtr kvbase_buf = 0;
    const BatchPlan* bplan = nullptr;
    std::span<const PRows> psegs;
    DevPtr mtp_in = 0;
    bool mtp_batch = false;
    bool keep_hidden = false;
    bool all_logits = false;
    std::uint32_t snap_rows = 0;
    std::uint32_t draft_vocab = 0;  // WHIRL_DRAFT_VOCAB=N: the first N rows only (old experiment)
    float draft_p_min = 0;
    std::uint32_t draft_n_min = 0;
    // speculative sampling of MTP drafts (numerics item specsample, server): per sequence; a
    // sequence with on = true gets draft_sample_rows after draft_pick_rows (q per draft step in
    // draft_qbuf: [seq][max_drafts][spec::q_words] i32)
    std::vector<spec::DraftSample> draft_samp;
    DevPtr draft_qbuf = 0;
    bool draftSampleOk() const { return k.draft_sample_rows != nullptr && draft_qbuf != 0; }
    std::optional<Mat> draft_head;
    std::optional<DevPtr> draft_d2;
    // draft-head vocabulary subset (setDraftVocab): rows of draft_d2 and their token ids on the device
    std::uint32_t draft_rows = 0;
    std::optional<DevPtr> draft_map;
    std::int32_t attn_win = 0;  // KvArgs::win of the current attnBlock (set around the MTP calls only)
    // The MTP draft attention (both MTP call sites, mtpEnqueue and mtpBatchStepEx): attnBlock of the
    // MTP layer with attn_win = the draft window, restored on every exit, also when attnBlock throws
    // (OVH-C1: the engine keeps serving after a failed step, the trunk must not inherit the window).
    void draftAttn(const AttnW& a, std::uint32_t n);
    // test seam (tests/model, no GPU): called first in attnBlock with the window it will use; may throw
    void (*attn_test_hook)(void* ctx, std::int32_t win) = nullptr;
    void* attn_test_ctx = nullptr;
    // DFlash2 drafter (loadDrafter) and the target hidden capture it needs. Capture slot j of
    // cap_buf holds x at the input of layer dflash->cfg.target_layers[j] for the last cap_n rows
    // of the latest run() (layer-major: [slot][cap_rows][n_embd] f32). Off unless enabled.
    std::unique_ptr<DFlashW> dflash;
    std::vector<std::int32_t> cap_slot;  // per layer input 0..n_layer: capture slot or -1
    DevPtr cap_buf = 0;
    std::uint32_t cap_rows = 0;
    std::uint32_t cap_n = 0;
    // run(): copy x (the input of layer li) for the last min(n, cap_rows) rows into its slot.
    void captureLayerInput(std::size_t li, std::uint32_t n);
    std::vector<std::int32_t>* moe_dump = nullptr;

    // ---- vision (image prompts; see whirl/vismap.h). Text-only sequences never take
    // these paths (bitwise unchanged).
    // image maps per sequence (server slots) and for the single sequence (CLI)
    std::array<const VisMap*, gdn_max_seg> vis_seq{};
    const VisMap* vis_single = nullptr;
    // this forward's attention rows use multi-section RoPE (attn_prep_m): rows from
    // rpos_buf when rpos_rows, else every position = pos_buf[row] - rdelta
    bool mrope_on = false;
    bool rpos_rows = false;
    std::int32_t rdelta = 0;
    // [max_batch][3] i32 (allocated on first image use, freed by visRelease) and its host staging
    DevPtr rpos_buf = 0;
    device::DevBuf rpos_own;  // owns rpos_buf (visEnsure / visRelease)
    std::vector<std::int32_t> rpos_host;
    // image rows injected into x after the embedding lookup of the next run() / MTP step
    struct Inj {
        std::uint32_t row = 0, n = 0;
        const float* src = nullptr;
    };
    static constexpr std::uint32_t max_inj = 256;
    std::array<Inj, max_inj> inj{};
    std::uint32_t n_inj = 0;
    // Prefill scratch buffers that hold nothing between forwards (the vision encoder
    // borrows them on the model stream): {ptr, bytes} pairs; returns the count.
    std::size_t lendScratch(std::span<std::array<std::uint64_t, 2>> out) const;
    // After borrowed scratch was overwritten: drop the cached activation copies.
    void scratchClobbered() {
        xq_src = 0;
        x16_src = 0;
    }
    // Free the RoPE row buffer when no sequence has an image map.
    void visRelease();

    // ---- sizes
    std::uint64_t convBytes() const;
    std::uint64_t ssmBytes() const;
    std::uint64_t pendLayerBytes() const;
    void selectSeq(std::uint32_t s);
    void commitSeq(std::uint32_t s);
    void dropPending(std::uint32_t s) {
        if (s < seqs.size()) seqs[s].npend = 0;
    }
    void ensureSnapshots(std::uint32_t n);
    void reset();

    // ---- single-sequence forward
    void step(std::uint32_t token, std::uint32_t pos);
    void forward(std::span<const std::uint32_t> tokens, std::uint32_t pos0);
    void prefill(std::span<const std::uint32_t> tokens, std::uint32_t pos0);
    void prefillWithMtp(std::span<const std::uint32_t> tokens);
    void prefillMtpChunk(std::span<const std::uint32_t> tokens, std::uint32_t pos0, std::size_t off, std::size_t n,
                         std::optional<DevPtr> prev_hidden);
    bool canSegment() const;
    void prefillSegs(std::span<const PSeg> segs, bool with_mtp);
    void segLogitsToFront(std::uint32_t k);
    void decodeStep();
    std::uint32_t beginDecode(std::uint32_t n_past);
    std::uint32_t argmax();
    void readLogits(std::span<float> dst);

    // ---- speculative decoding
    std::uint32_t mtpForward(DevPtr hidden, std::span<const std::uint32_t> tokens, std::uint32_t pos0, bool want_draft);
    void mtpEnqueue(DevPtr hidden, std::span<const std::uint32_t> tokens, DevPtr dev_tok, std::uint32_t pos0,
                    std::optional<std::uint32_t> slot);
    DevPtr ctlSlot(std::uint32_t word) const;
    DevPtr draftSlot(std::uint32_t r) const { return ctlSlot(ctl_drafts + r); }
    void verify(std::span<const std::uint32_t> tokens, std::uint32_t pos0, std::span<std::uint32_t> out);
    void verifyEnqueueEx(std::uint32_t next, std::uint32_t n_draft, std::optional<std::span<const std::uint32_t>> host_drafts,
                         std::uint32_t pos0);
    std::uint32_t verifyBatchEnqueue(std::span<const VSeg> segs);
    void mtpBatchStepEx(std::span<const MSeg> segs, std::uint32_t r, bool draft);
    void readCtl(std::span<std::int32_t> dst);
    struct Cycle {
        std::uint32_t nd = 0;  // drafts kept by the device-side p-min cutoff
        std::array<float, max_small_batch> probs{};
    };
    Cycle readCycle(std::span<std::uint32_t> out, std::span<std::uint32_t> drafts);
    void argmaxRows(std::uint32_t n, std::span<std::uint32_t> out);
    void restoreSnapshot(std::uint32_t keep);
    bool fusedDecode() const;
    // batched verify / MTP steps of up to max_verify_rows rows (dense, fused decode, replay)
    bool wideOk() const;
    // wideOk() before setupSeqs decides replay (which wide batches also need)
    bool wideCapable() const;
    std::uint32_t verifyRows() const { return wideOk() ? max_verify_rows : max_small_batch; }
    bool hasTiledFp8() const;  // the whole fragment-tiled fp8 set (numerics::Target::k_gemm8t)

    // ---- self-checks and microbenchmarks (bench / selftest)
    MatList layerMats(const Layer& L) const;

    // Device memory helpers (tracked in `allocations`, freed by the destructor).
    DevPtr alloc(std::uint64_t bytes);
    DevPtr allocZero(std::uint64_t bytes);
    void freeAlloc(DevPtr p);

    // ---- matmul dispatch queries (public for tests/model: the dispatch tables, no GPU). The launch internals
    // (gemvLaunch, matmulGroup, matmul, gemmF16, gdnIn2, x16bOff, run) are private since arch025-M3.
    // The matmul problem / flags matmul() hands to ops::chooseMatmulPath and ops::chooseGemvTile (K-1; the
    // f32 / f16 dispatch of RC-R1 / PREFILL-R2 is checked there, tests/ops, no GPU).
    ops::MatShape matShape(const Mat& w) const;
    ops::MatmulFlags matmulFlags() const;
    ops::GemmFlags gemmFlags() const;
    // tokens per launch of the f32 / f16 multi-token GEMV loop: gemv_<T>_8 / _4 when present, else _1
    static std::uint32_t floatGemvStep(std::uint32_t left, bool has8, bool has4) {
        return left >= 8 && has8 ? 8 : left >= 4 && has4 ? 4 : 1;
    }

protected:

    // ---- launch internals (private since arch025-M3; were public "for tests", no caller outside the model)
    void gemvLaunch(std::span<const Mat> ws, DevPtr xin, std::span<const DevPtr> ys, std::uint32_t n, std::int32_t acc);
    void matmulGroup(std::span<const Mat> ws, DevPtr xin, std::span<const DevPtr> ys, std::uint32_t n);
    void matmul(const Mat& w, DevPtr xin, DevPtr y, std::uint32_t n, bool accumulate);
    void gemmF16(const Mat& w, DevPtr x16in, DevPtr y, std::uint32_t n, std::int32_t acc);
    bool gdnIn2(const GdnW& g, std::uint32_t n) const;
    // x16b: f16 sub-buffer of x16 past the fp8 rows (tiled fp8 pads to 16 rows), 256-aligned
    std::uint64_t x16bOff() const { return ((static_cast<std::uint64_t>((max_batch + 15) / 16 * 16) * cfg.n_embd) + 255) & ~std::uint64_t(255); }
    void run(std::uint32_t n);
    struct KvLayer {
        DevPtr k = 0, v = 0, ks = 0, vs = 0;
    };
    KvLayer kvLayer(std::size_t i) const { return {kcache[i], vcache[i], kscale[i], vscale[i]}; }
    KvLayer mtpKv() const { return {mtp_kc, mtp_vc, mtp_ks, mtp_vs}; }
    KvArgs kvArgs(const KvLayer& kv, DevPtr kvbase, std::uint32_t tab0) const;
    void mark(OpClass cls);
    hip::Function x8(hip::Function f, hip::Function ft) const { return nf().g8t ? ft : f; }
    void rmsnorm(DevPtr xin, DevPtr w, DevPtr out, std::uint32_t n, std::uint32_t count, std::uint32_t in_stride,
                 std::uint32_t out_stride);
    ops::DraftHeadProblem draftHeadProblem(const Mat& head) const;  // forward/mtp_verify.cpp (ops::chooseDraftHead)
    ops::H16Tune h16Tune() const;  // what the f16-output choice reads besides the matrix (h16Out, ops::chooseGdnH16)
    bool h16Out(const Mat& w, std::uint32_t n, std::uint8_t cls) const;
    std::optional<ops::ActIn> actCommon(std::span<const Mat> consumers, std::uint32_t n, std::uint8_t cls) const;
    void rmsnormIn(DevPtr xin, DevPtr w, DevPtr out, std::uint32_t n, std::span<const Mat> consumers, std::uint8_t cls);
    bool fused(std::uint32_t n) const;
    // most tokens of the int8 GEMV path in the current forward
    std::uint32_t gvMax() const { return small_max > max_small_batch ? small_max : nf().gemv_max; }
    void gemvKernels(std::span<const Mat> ws, std::span<const DevPtr> ys, std::uint32_t n, std::int32_t acc, DevPtr xqp, DevPtr xdp);
    void rmsnormQ8(DevPtr xin, DevPtr w, DevPtr out, std::uint32_t n);
    void elementwise(hip::Function f, DevPtr a, DevPtr b, std::uint32_t n);
    void decodeStepLaunches();
    void attnBlock(const AttnW& a, const KvLayer& lkv, std::uint32_t n);
    void prefillAttn(DevPtr q, const KvArgs& kva, DevPtr out, DevPtr pos, std::uint32_t n, std::uint32_t pos0, float scale);
    void ffnBlock(DevPtr post_norm, const Mat& gate_w, const Mat& up, const Mat& down, std::uint32_t n);
    void moeBlock(DevPtr post_norm, const Mat& sg, const Mat& su, const Mat& sd, const MoeW& mo, std::uint32_t n);
    void setTokensDev(DevPtr ids_dev, std::span<const std::uint32_t> host, DevPtr dev_src, std::uint32_t n_dev,
                      std::uint32_t pos0);
    void setTokens(DevPtr ids_dev, std::span<const std::uint32_t> tokens, std::uint32_t pos0);
    void draftHeadMatmul(const Mat& head, std::uint32_t n);
    void argmaxRowsEnqueue(std::uint32_t n);
    GdnSegs gdnSegsSingle(std::size_t i, bool conv, std::uint32_t n) const;
    GdnSegs gdnSegsBatch(const BatchPlan& bp, std::size_t i, bool conv) const;
    std::uint64_t pendSsmOff() const;
    std::uint64_t pendPtr(const Seq& sq, std::size_t i, bool conv) const;
    void gdnChunked(std::size_t i, DevPtr conv_o, DevPtr alpha_b, DevPtr beta_b, DevPtr out, std::uint32_t n, float scale);
    void gdnScan(std::size_t i, DevPtr conv_o, DevPtr alpha_b, DevPtr beta_b, DevPtr out, std::uint32_t n, float scale);
    void gdnLayer(std::size_t i, const GdnW& g, std::uint32_t n, bool gdn_in2);  // forward/gdn.cpp (arch025-F3)
    // vision helpers (forward.cpp)
    bool visAny() const;
    const VisMap* visOf(std::uint32_t seq) const;
    std::int32_t lookupId(std::uint32_t t) const { return t < cfg.n_vocab ? static_cast<std::int32_t>(t) : 0; }
    void visBegin();
    void visEnsure();
    void visRow(const VisMap* vm, std::uint32_t r, std::uint32_t pos);
    void visCommit(std::uint32_t n);
    void visRowsCur(std::uint32_t n, std::uint32_t pos0);
    void injectRows();
};

}  // namespace whirl::qwen35
