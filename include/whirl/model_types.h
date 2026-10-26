// The qwen35 model types the server core sees (arch025-M3, R-8c): the shared constants, the model configuration and
// the batched-step segments, without the Model class (include/whirl/model.h,
// which includes this header). whirl_server_core (cmake/server.cmake) includes this instead of model.h
// (tools/gates/pending_scan.py rule include, blocking since M3), like llama.h's opaque `struct llama_context;`.
// SPDX-License-Identifier: Apache-2.0
//
// The declarations were moved verbatim from model.h; the draft policy part moved on to src/spec/policy.h (arch025-M4);
// LoadEstimate followed in arch025-M5 (the sizing interface of src/model/arch_registry.h returns it).

#pragma once

#include "whirl/hip.h"
#include "whirl/kv_format.h"  // KvFormat (LoadEstimate::kvPerToken)

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace whirl::gguf {
class File;
}  // namespace whirl::gguf

namespace whirl::qwen35 {

using hip::DevPtr;

class Model;  // include/whirl/model.h

// ---------------------------------------------------------------------------
// constants (shared with the kernels; model.h checks them against kernels_abi.h)

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

// ---------------------------------------------------------------------------
// configuration

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
    // qwen35 or qwen35moe, and ModelError("UnsupportedConfig") when the
    // hyper-parameters are zero, inconsistent or outside what the kernels
    // handle (e.g. nextn_predict_layers >= block_count, head_dim > 256).
    static Config fromGguf(const gguf::File& f);
    // Checks the type (known, non-empty) and ne[] of every tensor Model::load
    // reads against this config, before any GPU work: ModelError("MissingTensor"),
    // ("UnsupportedTensorType") or ("UnsupportedTensorShape"). Host-only.
    void validateTensors(const gguf::File& f) const;
    const char* archName() const { return moe ? "qwen35moe" : "qwen35"; }
    bool isAttn(std::uint32_t il) const { return (il + 1) % interval == 0; }
    std::uint32_t convCh() const { return 2 * n_k_heads * d_state + d_inner; }
    std::uint32_t headV() const { return d_inner / n_v_heads; }
    // The DeltaNet (GDN) kernels are written for a 4-tap conv, a 128-wide key/state dimension and
    // value columns in groups of 32 (conv state rows, gdn_seq_128's compile-time DK, grid.y =
    // headV / 32). Anything else would read / write past the state buffers or skip columns.
    bool gdnShapeSupported() const { return d_conv == 4 && d_state == 128 && headV() >= 32 && headV() % 32 == 0; }
};

// (moved verbatim from model.h, arch025-M5)
// VRAM Model::load will take, estimated from the GGUF header before any
// allocation (pre-load fit check, server auto-shrink of the prefill batch).
// buffers(B) follows the allocation list of Model::load exactly for B a
// multiple of 64; weights are the tensor bytes (+ MXFP4 row refs) that go to VRAM.
struct LoadEstimate {
    std::uint64_t weights = 0;  // tensor bytes uploaded to VRAM
    std::uint64_t state = 0;    // DeltaNet conv + ssm state allocated at load
    std::uint64_t ffs = 0;      // widest layer-matrix side (ffn_g / ffn_u scratch)
    std::uint64_t max_elems = 0;  // largest layer matrix (w16 scratch)
    bool embd_on_host = false;  // token_embd goes to pinned host memory
    bool has_mtp = false;       // nextn layer present (one more KV layer)
    std::uint64_t kv_f16 = 0, kv_q8v = 0, kv_q8h = 0, kv_q4 = 0;  // KV bytes per token by format
    // KV bytes per token of the format a pre-load estimate chose (numerics::kvEstimate); q8 has q8h's rows
    // (arch025-N3-r1: was the same ternary at loader.cpp's UMA check and the server's dGPU / UMA estimates)
    std::uint64_t kvPerToken(KvFormat kv) const {
        return kv == KvFormat::f16 ? kv_f16 : kv == KvFormat::q8v ? kv_q8v : kv == KvFormat::q4 ? kv_q4 : kv_q8h;
    }
};

// ---------------------------------------------------------------------------
// batched steps

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
// MTP draft-count control and n-gram (prompt-lookup) drafting (DraftAccept .. NgramPolicy): src/spec/policy.h
// (arch025-M4)

}  // namespace whirl::qwen35
