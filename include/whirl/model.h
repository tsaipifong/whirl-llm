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

#include "whirl/gguf.h"
#include "whirl/hip.h"
#include "whirl/kernels_abi.h"
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

namespace whirl {

// Environment variable lookup: WHIRL_<name>.
// Returns nullopt when it is not set.
std::optional<std::string> envGet(std::string_view name);
// envGet(name) present and not "0".
bool envFlag(std::string_view name, bool def = false);

}  // namespace whirl

namespace whirl::qwen35 {

using hip::DevPtr;
using gguf::GgmlType;

class ModelError : public std::runtime_error {
public:
    ModelError(std::string code, const std::string& detail = {})
        : std::runtime_error(detail.empty() ? code : code + ": " + detail), code_(std::move(code)) {}
    const std::string& code() const { return code_; }

private:
    std::string code_;
};

// ---------------------------------------------------------------------------
// constants (shared with the kernels; see kernels_abi.h)

constexpr std::uint32_t n_types = 40;  // > every GgmlType used (mxfp4 = 39)
constexpr std::uint32_t max_batch_default = 4096;
// Largest prefill batch a model can be loaded with (WHIRL_PREFILL_BATCH).
constexpr std::uint32_t max_batch_limit = 16384;
// Largest batch served by the multi-token int8 GEMV / small-batch kernels.
constexpr std::uint32_t max_small_batch = 16;
// Most rows of one batched verify / MTP step over several sequences (dense models with the
// wide GEMV kernels; each sequence still has at most max_small_batch rows).
constexpr std::uint32_t max_verify_rows = 32;
// Most MTP drafts per verify cycle.
constexpr std::uint32_t max_drafts = 10;
// Recurrent-state segments / snapshots per segment in the fused DeltaNet kernels.
constexpr std::uint32_t gdn_max_seg = 16;
constexpr std::uint32_t gdn_max_snap = 15;
// Paged KV cache page size (tokens).
constexpr std::uint32_t kv_page = 256;
// Most n-gram drafts per cycle.
constexpr std::uint32_t max_ng_drafts = gdn_max_snap;

// Control buffer layout (out_tok, i32 words per sequence).
constexpr std::uint32_t ctl_rows = 0;     // [0, 16): per-row argmax of a verify
constexpr std::uint32_t ctl_drafts = 16;  // [16, 32): draft tokens
constexpr std::uint32_t ctl_probs = 32;   // [32, 48): draft probabilities (f32)
constexpr std::uint32_t ctl_nd = 48;      // drafts kept this cycle (device p-min cutoff)
constexpr std::uint32_t ctl_stop = 49;    // draft chain stopped
constexpr std::uint32_t ctl_words = 64;

// Prefill GEMM autotune buckets.
constexpr std::uint32_t n_tune = 11;
constexpr std::uint32_t tune_small_max = 1024;
constexpr std::array<std::uint32_t, n_tune> tune_sizes = {512, 4096, 128, 32, 48, 64, 96, 192, 256, 384, 1024};
std::size_t tuneBucket(std::uint32_t n);

using GemmCfg = kernels::GemmCfg;
inline constexpr const auto& gemm_cfgs = kernels::kGemmCfgs;
inline constexpr const auto& gemms_cfgs = kernels::kGemmsCfgs;
inline constexpr const auto& gemm8_cfgs = kernels::kGemm8Cfgs;
inline constexpr const auto& gemm8t_cfgs = kernels::kGemm8tCfgs;
// Number of distinct prefill GEMM choices (tune values).
constexpr std::uint32_t n_choices = 2 * 24 + 8;

// Multi-token GEMV tables.
constexpr std::uint32_t n_gemvw = kernels::kNGemvw;
inline constexpr const auto& gemvw_rows = kernels::kGemvwRows;
using GemvR = std::array<std::uint8_t, max_small_batch + 1>;
using GemvW = std::array<GemvR, n_types>;
GemvR defaultGemvR();
GemvR gfx1151GemvR();
GemvW defaultGemvW();
GemvR defaultGemvWHead();

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

struct Config {
    std::uint32_t n_layer = 0;
    std::uint32_t n_embd = 0;
    std::uint32_t n_ff = 0;  // MoE: the shared expert's width
    std::uint32_t n_head = 0;
    std::uint32_t n_head_kv = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t n_rot = 0;
    float rope_base = 0;
    float eps = 0;
    std::uint32_t interval = 4;
    std::uint32_t d_conv = 0;
    std::uint32_t d_state = 0;
    std::uint32_t n_k_heads = 0;
    std::uint32_t n_v_heads = 0;
    std::uint32_t d_inner = 0;
    std::uint32_t n_vocab = 0;
    std::uint32_t n_nextn = 0;  // MTP blocks after the trunk layers
    bool moe = false;
    std::uint32_t n_expert = 0;
    std::uint32_t n_expert_used = 0;
    std::uint32_t n_ff_exp = 0;
    // vision: multi-section RoPE sections (rope.dimension_sections; image prompts only)
    std::array<std::uint32_t, 4> rope_sections = {0, 0, 0, 0};

    // Throws ModelError("UnsupportedArch") unless general.architecture is
    // qwen35 or qwen35moe.
    static Config fromGguf(const gguf::File& f);
    const char* archName() const { return moe ? "qwen35moe" : "qwen35"; }
    bool isAttn(std::uint32_t il) const { return (il + 1) % interval == 0; }
    std::uint32_t convCh() const { return 2 * n_k_heads * d_state + d_inner; }
    std::uint32_t headV() const { return d_inner / n_v_heads; }
};

struct Mat {
    DevPtr ptr = 0;
    // Prefill GEMM choice per batch bucket (tuneBucket): [0, 24) fused dequant
    // with gemm_cfgs[c]; [24, 48) dequantize to f16 first, then f16 GEMM with
    // gemm_cfgs[c - 24]; [48, 56) small-batch GEMM gemms_cfgs[c - 48].
    std::array<std::uint8_t, n_tune> tune = {};
    GgmlType ty = GgmlType::f32;
    // MXFP4 only: per-row reference exponent (u8) for the fp8 prefill GEMM.
    DevPtr ref = 0;
    std::uint32_t ncols = 0;
    std::uint32_t nrows = 0;
    std::uint64_t row_bytes = 0;
};

// True when b's rows directly follow a's in one allocation (same type and row
// layout), so [a; b] can be read as one (a.nrows + b.nrows)-row matrix.
inline bool rowsContiguous(const Mat& a, const Mat& b) {
    return a.ptr != 0 && a.ty == b.ty && a.ncols == b.ncols && a.row_bytes == b.row_bytes && a.ref == 0 && b.ref == 0 &&
           b.ptr == a.ptr + static_cast<std::uint64_t>(a.nrows) * a.row_bytes;
}

// [a; b] as one matrix (a's tune); only valid when rowsContiguous(a, b).
inline Mat concatRows(const Mat& a, const Mat& b) {
    Mat r = a;
    r.nrows = a.nrows + b.nrows;
    return r;
}

struct AttnW {
    Mat q, k, v, o;
    DevPtr q_norm = 0, k_norm = 0;
};

struct GdnW {
    Mat qkv, gate, beta, alpha, out;
    DevPtr conv = 0, dt = 0, a = 0, norm = 0;
};

// Routed experts of a qwen35moe FFN (the shared expert is the layer's
// ffn_gate / ffn_up / ffn_down).
struct MoeW {
    Mat router;          // [n_expert][n_embd] f32
    DevPtr sh_gate = 0;  // shared-expert gate vector [n_embd] f32 (0 = none)
    Mat gate, up, down;  // 3D expert tensors: expert e owns rows [e*rows_e, (e+1)*rows_e)
};

// The nextn (MTP) block.
struct MtpW {
    AttnW attn;
    DevPtr attn_norm = 0, post_norm = 0;
    Mat ffn_gate, ffn_up, ffn_down;
    Mat eh_proj;
    DevPtr enorm = 0, hnorm = 0, head_norm = 0;
    std::optional<MoeW> moe;
};

enum class LayerKind { attn, gdn };

struct Layer {
    DevPtr attn_norm = 0, post_norm = 0;
    Mat ffn_gate, ffn_up, ffn_down;
    LayerKind kind = LayerKind::gdn;
    AttnW attn;  // kind == attn
    GdnW gdn;    // kind == gdn
    std::optional<MoeW> moe;
};

struct MatList {
    std::array<Mat, 8> items{};
    std::size_t len = 0;
    std::span<const Mat> slice() const { return {items.data(), len}; }
    void add(const Mat& m) { items[len++] = m; }
};

// ---------------------------------------------------------------------------
// load-time options (set before Model::load)

// KV formats: f16; q8 = int8 + f16 scale per 32 values (K and V); q8h = q8
// with Hadamard-rotated q / k; q8v = K f16, V q8. auto: f16 when it fits
// (dense models fall back to q8v, then q8h); MoE always f16.
enum class KvMode { automatic, f16, q8, q8h, q8v };

// WHIRL_KV=auto|f16|q8|q8h|q8v (nullopt when unset; throws std::invalid_argument
// for another value). Model::load applies it when LoadOptions::kv_mode is
// automatic, like the prototype's process-wide KV setting.
std::optional<KvMode> kvModeFromEnv();

struct LoadOptions {
    // Prefill batch (rows per forward) the buffers are sized for.
    std::uint32_t max_batch = max_batch_default;
    // Keep token_embd in pinned host memory instead of VRAM.
    bool embd_on_host = false;
    KvMode kv_mode = KvMode::automatic;
    // MXFP4 prefill with fp8 activations (lossy; WHIRL_FP8=0 keeps f16).
    bool fp8_default = true;
    // Optional external code object file (development: WHIRL_CODE_OBJECT).
    std::string code_object;
};

// Grouped same-input GEMV launches (WHIRL_GV_GROUP=0: off) and the largest
// token count that takes the grouped multi-token twins (WHIRL_GV_NMAX).
extern bool gv_group;
extern std::uint32_t gv_nmax;

// ---------------------------------------------------------------------------
// kernels (table from whirl/kernels_abi.h)

using Kernels = kernels::KernelTable;
// q8: q8 KV kernels; rot: Hadamard-rotated q/k (q8h); kf16: q8v.
Kernels loadKernels(const hip::Module& m, bool q8, bool rot, bool kf16);
// ---------------------------------------------------------------------------
// profiling (WHIRL_PROFILE=1)

enum class OpClass { embed, norm, quant, matmul, attn, gdn_prep, gdn_rec, gdn_norm, misc, head };
constexpr std::size_t n_classes = 10;
const char* opClassName(OpClass c);

struct Profile {
    std::array<double, n_classes> ms{};
    std::uint64_t matmul_bytes = 0;
    std::vector<hip::Event> events;
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

struct LoadStats {
    std::uint64_t bytes = 0;
    std::uint32_t tensors = 0;
    // GDN layers whose ssm_beta / ssm_alpha were loaded into one block (beta rows then alpha rows)
    std::uint32_t gdn_ba_contig = 0;
    double ms = 0;
};

// ---------------------------------------------------------------------------
// sequences and batched steps

// One sequence (a server slot): recurrent state, KV page-table offset, MTP
// hidden rows and replay buffer.
struct Seq {
    // per-layer state pointers (0 for attention layers); views into tables
    // owned by the Model, shared with Model::conv_state / ssm_state while selected
    std::span<DevPtr> conv;
    std::span<DevPtr> ssm;
    DevPtr hid = 0;               // normed trunk hidden rows feeding the next MTP step
    std::uint32_t kv_base = 0;    // offset of this sequence's page table
    DevPtr pend = 0;              // DeltaNet replay inputs (gdn_replay)
    std::uint32_t npend = 0;
};

// Batched verify segment: sequence `seq` runs [next, drafts 0 .. nd) at pos ..
struct VSeg {
    std::uint32_t seq = 0, next = 0, nd = 0, pos = 0;
    std::span<const std::uint32_t> drafts = {};  // empty: drafts from the control area (MTP)
};
// One sequence's prefill chunk in a segmented prefill.
struct PSeg {
    std::uint32_t seq = 0;
    std::span<const std::uint32_t> tokens;
    std::uint32_t pos0 = 0;
    std::size_t off = 0;
    std::size_t n = 0;
    std::optional<DevPtr> prev_hidden;
};
struct PRows {
    std::uint32_t seq = 0, r0 = 0, n = 0, pos0 = 0;
};
// Batched MTP step segment.
struct MSeg {
    std::uint32_t seq = 0;
    std::span<const std::uint32_t> pend = {};
    std::uint32_t pend_pos = 0;
    std::uint32_t pos = 0;
};

struct BatchPlan {
    std::span<const VSeg> segs;
    std::array<std::uint32_t, gdn_max_seg> snap_base{};
};

// ---------------------------------------------------------------------------
// the model

class Model {
public:
    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    ~Model();

    // Loads weights, allocates activations and (max_ctx > 0) the KV pool for
    // one sequence. The current HIP device must be set.
    static std::unique_ptr<Model> load(const gguf::File& f, std::uint32_t max_ctx, LoadStats& stats,
                                       const LoadOptions& opt = {});

    // ---- configuration
    Config cfg;
    std::uint32_t max_ctx = 0;
    std::uint32_t max_batch = max_batch_default;
    hip::Arch arch = hip::Arch::gfx1201;
    std::vector<std::int32_t> host_ids;
    Kernels k;
    hip::Module module;
    Mat tok_embd;
    void* embd_host = nullptr;
    Mat output;
    DevPtr output_norm = 0;
    std::vector<Layer> layers;
    std::vector<DevPtr> allocations;
    std::array<std::uint64_t, n_types> type_bytes{};
    Profile* prof = nullptr;
    hip::Stream stream = nullptr;
    std::unique_ptr<hip::Graph> decode_graph;
    bool use_graph = true;
    bool head_phase = false;
    bool no_fuse = false;
    std::uint32_t row_n = 0;
    std::array<std::int32_t, max_verify_rows> row_pos{};
    std::array<std::int32_t, max_verify_rows> row_base{};
    // most rows the GEMV / decode-fusion paths take in the current forward: max_small_batch,
    // raised to max_verify_rows inside a wide batched verify / MTP step (see wideOk())
    std::uint32_t small_max = max_small_batch;
    std::uint32_t dbg_flags = 0;
    std::uint64_t tune_mask = ~0ull;
    bool tune_cold = false;
    bool naive_attn = false;
    // verify attention: groups of up to 32 columns (attn_wsplit2) when that lets a
    // sequence's rows share one K/V pass (WHIRL_ATTN_WIDE=0: <= 16 columns as before)
    bool attn_wide = true;
    // batched verify of more than 16 rows (WHIRL_WIDE_VERIFY=0: at most 16 as before)
    bool wide_verify = true;
    // output-head rows per range in a wide verify (two 16-token passes per range; WHIRL_HEAD_CHUNK)
    std::uint32_t head_chunk = 248320;
    bool float_gemv = false;
    std::uint32_t gemv_max = max_small_batch;
    GemvR gemv_r = defaultGemvR();
    GemvW gemv_w = defaultGemvW();
    GemvR gemv_w_head = defaultGemvWHead();

    // ---- activations
    DevPtr x = 0, h = 0, qf = 0, kv_k = 0, kv_v = 0, attn_out = 0, qkv = 0, conv_out = 0, z = 0, beta = 0, alpha = 0, ba_buf = 0,
           gdn_out = 0, ffn_g = 0, ffn_u = 0, logits = 0, scores = 0, ids = 0, pos_buf = 0, out_tok = 0;
    DevPtr part_ml = 0, part_acc = 0;
    DevPtr x16 = 0, xq = 0, xd = 0;
    DevPtr xq_src = 0;
    std::uint32_t xq_n = 0;
    DevPtr x16_src = 0;
    std::uint32_t x16_count = 0;
    bool x16_fp8 = false;
    DevPtr sx8 = 0;
    bool fp8_prefill = true;
    std::uint8_t fp8_mask = 7;
    std::uint8_t mm_class = 1;
    std::uint64_t mx_fold_lossy = 0;
    DevPtr w16 = 0;
    DevPtr gc_w = 0, gc_u = 0, gc_m = 0, gc_g = 0;
    std::uint32_t fwd_pos0 = 0;
    bool gdn_chunked = true;
    bool gdn_wmma = false;
    bool act_fuse = true;
    bool attn_kx_on = true;
    bool gdn_ba_on = true;   // n>16 GDN prefill: one [beta; alpha] GEMM + gdn_gates_ba (WHIRL_GDN_BA=0 off)
    bool gdn_in2_on = true;  // n>16 GDN prefill: one norm -> fp8 (qkv / gate) + f16 x16b ([beta; alpha]) (WHIRL_GDN_IN2=0 off)
    std::uint64_t x16_bytes = 0;  // x16 allocation size (x16b sub-buffer bound check)
    bool attn_kg_on = true;  // GQA-grouped prefill attention (f16 / q8 / q8h / q8v KV; WHIRL_ATTN_KG=0 off)
    bool ffn_h16 = false;
    bool out_h16 = false;
    bool g8t = false;
    bool gemmh_on = true;
    bool gemmhq_on = false;

    // ---- per-layer state
    std::vector<DevPtr> kcache, vcache, kscale, vscale;
    bool kv_q8 = true;
    bool kv_rot = false;
    bool kv_kf16 = false;
    KvMode kv_mode = KvMode::automatic;
    std::uint32_t pool_pages = 0;
    std::uint32_t seq_pages = 0;
    DevPtr ptab = 0;
    // per-layer recurrent state of the current sequence (views; swapping an
    // entry, e.g. restoreSnapshot, updates the selected Seq too)
    std::span<DevPtr> conv_state, ssm_state;
    std::deque<std::vector<DevPtr>> state_tables;  // backing storage of the views
    std::array<std::vector<DevPtr>, gdn_max_snap> snap_conv, snap_ssm;
    std::uint32_t snap_sets = 0;
    DevPtr gate = 0;
    std::vector<Seq> seqs;
    bool gdn_replay = false;
    std::vector<std::uint32_t> gdn_ord;
    std::uint32_t cur_seq = 0;
    std::uint32_t kv_off = 0;
    std::uint32_t ctl_off = 0;
    DevPtr kvbase_buf = 0;
    const BatchPlan* bplan = nullptr;
    std::span<const PRows> psegs;
    DevPtr mtp_in = 0;
    bool mtp_batch = false;
    std::optional<MtpW> mtp;
    DevPtr mtp_kc = 0, mtp_vc = 0, mtp_ks = 0, mtp_vs = 0;
    DevPtr hn = 0, mtp_cat = 0, mtp_h = 0, mtp_ids = 0;
    bool keep_hidden = false;
    bool all_logits = false;
    std::uint32_t snap_rows = 0;
    std::uint32_t draft_vocab = 0;  // WHIRL_DRAFT_VOCAB=N: the first N rows only (old experiment)
    // WHIRL_DRAFT_WINDOW / _MIN: MTP draft attention over the first 256 + the last W positions once
    // the context reaches draft_window_min (W = 0: off); the trunk and verify always see everything
    std::uint32_t draft_window = 16384;
    std::uint32_t draft_window_min = 65536;
    float draft_p_min = 0;
    std::uint32_t draft_n_min = 0;
    std::optional<Mat> draft_head;
    std::optional<DevPtr> draft_d2;
    // draft-head vocabulary subset (setDraftVocab): rows of draft_d2 and their token ids on the device
    std::uint32_t draft_rows = 0;
    std::optional<DevPtr> draft_map;
    std::int32_t attn_win = 0;  // KvArgs::win of the current attnBlock (set around the MTP calls only)
    std::uint32_t moe_bn_force = 0;
    std::vector<std::int32_t>* moe_dump = nullptr;
    std::uint32_t ff_scratch = 0;
    DevPtr moe_logits = 0, moe_ids = 0, moe_w = 0, moe_sg = 0, moe_g = 0, moe_u = 0, moe_xq = 0, moe_xd = 0, moe_ysh = 0,
           moe_perm = 0, moe_inv = 0, moe_tiles = 0, moe_ntiles = 0, moe_x16 = 0, moe_yg = 0, moe_yu = 0, moe_yd = 0;
    // MXFP4 routed experts (Ornith MXFP4): per-pos fp8 activation scales [max_batch * K];
    // prefill experts on the MXFP4 x fp8 grouped GEMM (WHIRL_MOE_FP8=0 -> f16 gemm_moe);
    // decode experts on the whole-block kernels (WHIRL_MOE_MXW=0 -> generic moe_gu / moe_down).
    DevPtr moe_sx = 0;
    bool moe_fp8 = true;
    bool moe_mxw = true;

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
    std::uint64_t kvBytesPerTokenFmt(bool q8, bool kf16) const;
    std::uint64_t kvBytesPerToken() const { return kvBytesPerTokenFmt(kv_q8, kv_kf16); }
    std::uint32_t kvLayers() const;
    const char* kvName() const;
    bool kvAutoDense() const;
    std::uint64_t pendLayerBytes() const;

    // ---- KV pool and sequences
    void setKvFormat(bool q8, bool rot, bool kf16);
    void allocKvPool(std::uint32_t tokens);
    void mapPages(std::uint32_t s, std::uint32_t first, std::span<const std::int32_t> phys);
    void setupSeqs(std::uint32_t n, std::uint32_t slot_ctx);
    void selectSeq(std::uint32_t s);
    DevPtr seqCtl(std::uint32_t s, std::uint32_t word) const;
    void commitSeq(std::uint32_t s);
    void setPending(std::uint32_t s, std::uint32_t rows) { seqs[s].npend = rows; }
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
    void prefillWithMtpFrom(std::span<const std::uint32_t> tokens, std::uint32_t pos0, DevPtr prev_hidden);
    void prefillMtpChunk(std::span<const std::uint32_t> tokens, std::uint32_t pos0, std::size_t off, std::size_t n,
                         std::optional<DevPtr> prev_hidden);
    bool canSegment() const;
    void prefillSegs(std::span<const PSeg> segs, bool with_mtp);
    void segLogitsToFront(std::uint32_t k);
    void decodeStep();
    std::uint32_t beginDecode(std::uint32_t n_past);
    std::uint32_t argmax();
    void readLogits(std::span<float> dst);
    void readHidden(std::span<float> dst);

    // ---- speculative decoding
    std::uint32_t mtpForward(DevPtr hidden, std::span<const std::uint32_t> tokens, std::uint32_t pos0, bool want_draft);
    void mtpEnqueue(DevPtr hidden, std::span<const std::uint32_t> tokens, DevPtr dev_tok, std::uint32_t pos0,
                    std::optional<std::uint32_t> slot);
    DevPtr ctlSlot(std::uint32_t word) const;
    DevPtr draftSlot(std::uint32_t r) const { return ctlSlot(ctl_drafts + r); }
    void verify(std::span<const std::uint32_t> tokens, std::uint32_t pos0, std::span<std::uint32_t> out);
    void verifyEnqueue(std::uint32_t next, std::uint32_t n_draft, std::uint32_t pos0);
    void verifyEnqueueEx(std::uint32_t next, std::uint32_t n_draft, std::optional<std::span<const std::uint32_t>> host_drafts,
                         std::uint32_t pos0);
    std::uint32_t verifyBatchEnqueue(std::span<const VSeg> segs);
    void mtpBatchStep(std::span<const MSeg> segs, std::uint32_t r) { mtpBatchStepEx(segs, r, true); }
    void mtpBatchStepEx(std::span<const MSeg> segs, std::uint32_t r, bool draft);
    void readCtl(std::span<std::int32_t> dst);
    struct Draft {
        std::uint32_t tok;
        float p;
    };
    Draft readDraft(std::uint32_t r);
    struct Cycle {
        std::uint32_t nd = 0;  // drafts kept by the device-side p-min cutoff
        std::array<float, max_small_batch> probs{};
    };
    Cycle readCycle(std::span<std::uint32_t> out, std::span<std::uint32_t> drafts);
    void argmaxRows(std::uint32_t n, std::span<std::uint32_t> out);
    void restoreSnapshot(std::uint32_t keep);
    void restoreSeqSnapshot(std::uint32_t s, std::uint32_t set);
    bool fusedDecode() const;
    // batched verify / MTP steps of up to max_verify_rows rows (dense, fused decode, replay)
    bool wideOk() const;
    // wideOk() before setupSeqs decides replay (which wide batches also need)
    bool wideCapable() const;
    std::uint32_t verifyRows() const { return wideOk() ? max_verify_rows : max_small_batch; }

    // MTP block Q6_K matrices as Q4_K (drafts only).
    void requantMtpQ4();
    enum class DraftHeadKind { q4, d2 };
    void buildDraftHead() { buildDraftHeadEx(DraftHeadKind::d2); }
    void buildDraftHeadEx(DraftHeadKind kind);
    // Keep only these rows (ascending token ids, see draftVocabIds) of the 2-bit draft head;
    // drafts map back to token ids, the trunk and its output head are untouched. False (and no
    // change) without the 2-bit head or the copy_rows_map kernel.
    bool setDraftVocab(std::span<const std::uint32_t> ids);

    // ---- prefill GEMM autotune
    void autotuneGemm(std::size_t bucket, std::uint32_t reps, std::string* log);
    std::string writeTune() const;
    bool readTune(std::string_view text, std::size_t n_buckets);
    bool useTiledFp8();
    std::uint32_t moeTile(std::uint32_t n) const;
    bool moeFp8(const MoeW& mo) const;
    hip::Function moeGu(GgmlType ty) const;
    hip::Function moeDown(GgmlType ty) const;

    // ---- self-checks and microbenchmarks (bench / selftest)
    MatList layerMats(const Layer& L) const;
    std::uint64_t matmulsOnly(std::uint32_t n);
    std::uint64_t matmulsOfType(GgmlType ty, std::uint32_t n);
    bool checkGemvq(std::string& log);
    bool checkGemvBitwise(std::string& log);
    bool checkPrefillInvariance(std::string& log);
    // wide: the grouped launch is attn_wsplit2 (<= 32 columns) instead of attn_wsplit1
    bool checkAttnGroups(std::string& log, std::uint32_t p0, std::uint32_t n, bool wide = false);

    // Device memory helpers (tracked in `allocations`, freed by the destructor).
    DevPtr alloc(std::uint64_t bytes);
    DevPtr allocZero(std::uint64_t bytes);
    void freeAlloc(DevPtr p);

    // ---- internals (public for tests; not part of the stable API)
    enum class ActIn { gemv, f32in, fp8, f16 };
    void gemvLaunch(std::span<const Mat> ws, DevPtr xin, std::span<const DevPtr> ys, std::uint32_t n, std::int32_t acc);
    void matmulGroup(std::span<const Mat> ws, DevPtr xin, std::span<const DevPtr> ys, std::uint32_t n);
    void matmul(const Mat& w, DevPtr xin, DevPtr y, std::uint32_t n, bool accumulate);
    void gemmF16(const Mat& w, DevPtr x16in, DevPtr y, std::uint32_t n, std::int32_t acc);
    bool gdnIn2(const GdnW& g, std::uint32_t n) const;
    // x16b: f16 sub-buffer of x16 past the fp8 rows (tiled fp8 pads to 16 rows), 256-aligned
    std::uint64_t x16bOff() const { return ((static_cast<std::uint64_t>((max_batch + 15) / 16 * 16) * cfg.n_embd) + 255) & ~std::uint64_t(255); }
    void run(std::uint32_t n);

private:
    struct KvLayer {
        DevPtr k = 0, v = 0, ks = 0, vs = 0;
    };
    KvLayer kvLayer(std::size_t i) const { return {kcache[i], vcache[i], kscale[i], vscale[i]}; }
    KvLayer mtpKv() const { return {mtp_kc, mtp_vc, mtp_ks, mtp_vs}; }
    KvArgs kvArgs(const KvLayer& kv, DevPtr kvbase, std::uint32_t tab0) const;
    void mark(OpClass cls);
    hip::Function x8(hip::Function f, hip::Function ft) const { return g8t ? ft : f; }
    void rmsnorm(DevPtr xin, DevPtr w, DevPtr out, std::uint32_t n, std::uint32_t count, std::uint32_t in_stride,
                 std::uint32_t out_stride);
    ActIn actIn(const Mat& w, std::uint32_t n, std::uint8_t cls) const;
    bool h16Out(const Mat& w, std::uint32_t n, std::uint8_t cls) const;
    std::optional<ActIn> actCommon(std::span<const Mat> consumers, std::uint32_t n, std::uint8_t cls) const;
    void rmsnormIn(DevPtr xin, DevPtr w, DevPtr out, std::uint32_t n, std::span<const Mat> consumers, std::uint8_t cls);
    bool fused(std::uint32_t n) const;
    // most tokens of the int8 GEMV path in the current forward
    std::uint32_t gvMax() const { return small_max > max_small_batch ? small_max : gemv_max; }
    void gemvKernels(std::span<const Mat> ws, std::span<const DevPtr> ys, std::uint32_t n, std::int32_t acc, DevPtr xqp, DevPtr xdp);
    bool floatGemvN(const Mat& w, std::uint32_t n) const;
    bool gdnAbFusable(const GdnW& g) const;
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
    Mat requantQ4k(const Mat& w);
    void argmaxRowsEnqueue(std::uint32_t n);
    GdnSegs gdnSegsSingle(std::size_t i, bool conv, std::uint32_t n) const;
    GdnSegs gdnSegsBatch(const BatchPlan& bp, std::size_t i, bool conv) const;
    std::uint64_t pendSsmOff() const;
    std::uint64_t pendPtr(const Seq& sq, std::size_t i, bool conv) const;
    void gdnChunked(std::size_t i, DevPtr conv_o, DevPtr alpha_b, DevPtr beta_b, DevPtr out, std::uint32_t n, float scale);
    void gdnScan(std::size_t i, DevPtr conv_o, DevPtr alpha_b, DevPtr beta_b, DevPtr out, std::uint32_t n, float scale);
    void benchMat(const Mat& w, std::uint32_t n);
    std::vector<Mat*> tuneMats(Layer& L);
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

    friend struct Loader;
};

// ---------------------------------------------------------------------------
// prefill GEMM tune cache and runtime switches (WHIRL_FP8, WHIRL_GDN_WMMA, ...):
// reads %LOCALAPPDATA%\whirl\tune4-<file>-<size>[-gfx1151].txt, else autotunes and
// writes it. `log` receives the human-readable lines.
void loadOrTune(Model& m, const std::string& model_path, std::string& log);

// Multi-token GEMV selection overrides (A/B testing), as "nt:value,..." lists:
// rows per wave (r = 1, 2, 4), WMMA variant for every type (0..8), and the
// variant of the Q6_K output head. applyGemvEnv reads WHIRL_GEMV_R /
// WHIRL_GEMV_W / WHIRL_GEMV_WH. Malformed items throw std::invalid_argument.
void parseGemvR(Model& m, std::string_view list);
void parseGemvW(Model& m, std::string_view list);
void parseGemvWHead(Model& m, std::string_view list);
void applyGemvEnv(Model& m);

// ---------------------------------------------------------------------------
// device selection

// HIP device for `spec` (index, "r9700" / "8060s" alias, or a substring of
// the name / gcnArchName); empty: WHIRL_DEVICE, else the first R9700, else the
// first device with an embedded code object (a Radeon 8060S on its own).
// Takes the per-GPU process mutex (WHIRL_GPU_SHARE=1 skips it).
int pickDevice(std::string_view spec);

// ---------------------------------------------------------------------------
// MTP draft-count control and n-gram (prompt-lookup) drafting

struct MtpDefaults {
    std::uint32_t drafts;
    float p_min;
    std::uint32_t adapt;
    bool automatic;
};
MtpDefaults mtpDefaults(bool moe);

struct DraftAccept {
    std::array<float, max_ng_drafts> alpha;
    explicit DraftAccept(float a = 0.75f) { alpha.fill(a); }
    void update(std::uint32_t nd_dev, std::uint32_t acc) { updateRate(nd_dev, acc, 0.1f); }
    void updateRate(std::uint32_t nd_dev, std::uint32_t acc, float r);
    float expected(std::uint32_t nd) const;
};

struct DraftTiming {
    std::array<float, max_ng_drafts + 1> t{};
    std::array<std::uint32_t, max_ng_drafts + 1> n{};
    float prior_slope = 0.12f;
    bool skip_first = false;
    std::array<bool, max_ng_drafts + 1> seen{};
    void update(std::uint32_t nd, float ms);
    std::optional<float> estimate(std::uint32_t nd) const;
};

// ---- draft-head vocabulary subset (T5-1b): WHIRL_DRAFT_VOCAB = 48k | 64k | <file> | off | N
// File for a WHIRL_DRAFT_VOCAB value: "48k" / "64k" / ... -> <dir>/subset_48k.bin, anything else
// that is not "off" or a plain number (the old first-N-rows experiment) is a path; else nullopt.
std::optional<std::string> draftVocabFile(std::string_view spec, const std::string& dir);
// uint32 little-endian token ids (throws on a missing file or a size that is not a multiple of 4)
std::vector<std::uint32_t> readDraftVocab(const std::string& path);
// Checks a subset (strictly ascending, every id < n_rows; throws otherwise) and adds the missing
// `required` ids (special and byte tokens), counting them in *added.
std::vector<std::uint32_t> draftVocabIds(std::span<const std::uint32_t> ids, std::uint32_t n_rows,
                                         std::span<const std::uint32_t> required, std::uint32_t* added = nullptr);
// directory of the running executable ("" if unknown)
std::string exeDirectory();
// What WHIRL_DRAFT_VOCAB selects. Unset = the embedded 64k subset by default (only applied to a
// model draftVocabDefaultFits accepts); "64k" = the embedded subset explicitly; "off" / "0" / "" =
// the full vocabulary; N = the first N rows (old experiment); "48k" etc. = <dir>/draft_vocab/
// subset_48k.bin; anything else = a file path.
struct DraftVocabChoice {
    enum class Kind { full, first_n, embedded_64k, file } kind = Kind::full;
    std::uint32_t n = 0;
    std::string file;
    bool by_default = false;  // WHIRL_DRAFT_VOCAB unset
};
DraftVocabChoice draftVocabChoice(const std::optional<std::string>& env_value, const std::string& dir);
// vocabulary size of the qwen35 tokenizer the embedded subset was built for
inline constexpr std::uint32_t draft_vocab_embedded_n_vocab = 248320;
// the default (embedded 64k) subset is used only for dense qwen35 models with that vocabulary
inline bool draftVocabDefaultFits(bool moe, std::uint32_t n_vocab) { return !moe && n_vocab == draft_vocab_embedded_n_vocab; }
// the embedded 64k subset (data/draft_vocab/subset_64k.bin, built by whirl-cloud tools/vocab_subset);
// defined in draft_vocab_embed.cpp (whirl_model only)
std::vector<std::uint32_t> embeddedDraftVocab64k();
// KvArgs::win for the MTP draft attention window: W / 64 (rounded up, <= 65535) in the low 16
// bits, the context threshold / 1024 in the high bits; 0 (off) for W = 0.
inline std::int32_t draftWindowArg(std::uint32_t w, std::uint32_t min_ctx) {
    if (w == 0) return 0;
    const std::uint32_t lo = std::min<std::uint32_t>((w + 63) / 64, 0xffffu);
    const std::uint32_t hi = std::min<std::uint32_t>(min_ctx / 1024, 0x7fffu);
    return static_cast<std::int32_t>(lo | hi << 16);
}

// Draft count in [1, max] maximizing sum(E) / T (see the prototype notes).
std::uint32_t pickDrafts(std::span<const DraftAccept* const> accepts, const DraftTiming& timing, std::uint32_t max,
                         std::uint32_t cycle, std::uint32_t prev);

// ---- per-slot draft allocation (T4-1): pure model code, not wired into the engine yet

// One verify cycle as the cost model sees it.
struct CycleFeat {
    float rows = 0;    // R: verify rows of all sequences
    float steps = 0;   // S: sequential MTP draft steps (the longest MTP draft)
    float row_ctx = 0; // sum over sequences of rows_i * context_i, in k tokens
};

// Cycle time for one number of decoding slots: T ~ a + b R + c S + e [R > 16] + f sum(rows_i ctx_i),
// fitted by recursive least squares with forgetting factor 0.98.
class CycleCost {
public:
    static constexpr std::uint32_t n_par = 5;
    static constexpr double forget = 0.98;
    CycleCost() { reset(); }
    void reset();
    // seed from the uniform-allocation timing of `slots` decoding slots (DraftTiming points or its prior)
    void prior(const DraftTiming& tm, std::uint32_t slots, float ctx_k = 0);
    void observe(const CycleFeat& x, float ms);
    float predict(const CycleFeat& x) const;
    std::uint32_t samples() const { return n_obs; }
    const std::array<double, n_par>& params() const { return th; }

private:
    static std::array<double, n_par> feat(const CycleFeat& x);
    void rls(const std::array<double, n_par>& z, double y);
    std::array<double, n_par> th{};
    std::array<std::array<double, n_par>, n_par> P{};
    std::uint32_t n_obs = 0;
};

// Per-slot acceptance with the T4-1 additions on top of DraftAccept: faster start (r = 0.2 for 16
// cycles), unobserved positions drift towards the last observed rate x 0.95 (rate 0.02; a fully
// accepted chain is censored, not a ceiling, so it drifts without the 0.95), observation counts
// for shrinkage towards a global pool.
struct SlotAccept {
    static constexpr std::uint32_t n_cap = 32;
    static constexpr float pool_weight = 4.0f;
    static constexpr float drift_rate = 0.02f;
    static constexpr float drift_decay = 0.95f;
    DraftAccept a;
    std::array<std::uint32_t, max_ng_drafts> n{};
    std::uint32_t cycles = 0;
    explicit SlotAccept(float a0 = 0.75f) : a(a0) {}
    void observe(std::uint32_t nd_dev, std::uint32_t acc);
    // alpha_eff[k] = (n_k alpha_k + 4 alpha_pool[k]) / (n_k + 4)
    DraftAccept effective(const DraftAccept& pool) const;
};

struct AllocSlot {
    const DraftAccept* acc = nullptr; // MTP slot: acceptance (alpha_eff); unused for n-gram slots
    std::uint32_t cap = 1;            // max drafts of this slot
    float ctx_k = 0;                  // context length, k tokens
    std::uint32_t ng_rows = 0;        // > 0: n-gram slot with this many verify rows (fixed)
    float ng_e = 0;                   // n-gram slot: expected tokens of its fixed draft
};

struct AllocBudget {
    std::uint32_t rows = 16;           // verify rows available in total
    std::uint32_t snaps = 0xffffffffu; // snapshot sets free for drafts (0xffffffff: replay, no limit)
};

struct AllocResult {
    std::vector<std::uint32_t> d;     // drafts per slot (0 for n-gram slots)
    float score = 0;                  // sum(E) / T (tokens per ms)
    std::uint32_t rounds = 0;         // local-search rounds used
    bool uniform = true;              // true: the uniform allocation u was returned
};

// Per-slot draft counts maximizing sum(E_i) / T, starting from the uniform count `uniform`
// (capped per slot): local search over +1 / -1 / swap moves, at most 16 rounds, subject to
// sum(d) <= min(rows free, snapshots free), 1 <= d_i <= cap_i and each slot keeping at least
// (1 - x) of its uniform-allocation throughput. Keeps `prev` when it scores >= 0.98 of the best,
// and returns u unless the best beats it by more than 1%, with < 2 MTP slots, < 16 cost samples,
// or acceptances so close that u is already optimal.
AllocResult allocDrafts(std::span<const AllocSlot> slots, const CycleCost& cost, AllocBudget budget, std::uint32_t uniform,
                        std::span<const std::uint32_t> prev, float x = 0.03f);

class Ngram {
public:
    static constexpr std::uint32_t none = 0xffffffffu;
    static constexpr std::size_t long_len = 12;
    static constexpr std::uint32_t max_cands = 16;
    static constexpr std::uint32_t max_match = 256;
    struct Match {
        std::size_t n = 0;
        std::uint32_t mlen = 0;
    };
    // token id -> the same text with CR LF as LF (empty: none)
    std::vector<std::uint32_t> norm;

    void reset();
    Match lookup(std::span<const std::uint32_t> toks, std::uint32_t min_match, std::span<std::uint32_t> out);

private:
    std::uint32_t nt(std::uint32_t t) const { return t < norm.size() ? norm[t] : t; }
    std::unordered_map<std::uint64_t, std::uint32_t> map_s, map_l;
    std::vector<std::uint32_t> prev_s, prev_l, ntoks;
    std::size_t indexed = 2;
    std::size_t cur_q = 0, cur_n = 0;
    std::size_t gen_from = 0;
    bool gen_cr = false;
};

class NgramPolicy {
public:
    Ngram ng;
    std::array<DraftAccept, 3> acc = priorAcc();
    DraftTiming timing = priorTiming();
    static constexpr float verify_slope = 0.015f;
    static constexpr float acc_rate = 0.2f;

    static std::uint32_t bucket(std::uint32_t mlen) { return mlen >= 24 ? 2 : (mlen >= 8 ? 1 : 0); }
    void reset();
    void observe(std::span<const std::uint32_t> toks);
    Ngram::Match propose(std::span<const std::uint32_t> toks, std::uint32_t min_match, std::span<std::uint32_t> out);
    std::uint32_t choose(std::size_t n, std::uint32_t mlen, float mtp_score, const DraftTiming& mtp_timing) const;

private:
    struct Pending {
        std::size_t at = 0;
        std::uint32_t n = 0, b = 0;
        std::array<std::uint32_t, max_ng_drafts> toks{};
    };
    static std::array<DraftAccept, 3> priorAcc() { return {DraftAccept(0.35f), DraftAccept(0.7f), DraftAccept(0.85f)}; }
    static DraftTiming priorTiming() {
        DraftTiming t;
        t.prior_slope = verify_slope;
        t.skip_first = true;
        return t;
    }
    std::array<Pending, 16> pend{};
    std::uint32_t n_pend = 0;
};

}  // namespace whirl::qwen35
