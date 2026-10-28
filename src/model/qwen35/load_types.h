// qwen35 load types (arch025-lane2 N9 S2c): what WeightLoader (weight_loader.h) fills besides qwen35::Weights - the
// DFlash2 drafter's config and weights (DFlashConfig, DFlashLayerW, DFlashW) and the load statistics (LoadStats).
// Moved verbatim from include/whirl/model.h, which includes this header where the DFlash2 types used to be, so
// weight_loader.h no longer includes the Model class.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "model/qwen35/weights.h"  // Mat, DevPtr
#include "whirl/gguf.h"
#include "whirl/model_types.h"     // Config

#include <cstdint>
#include <vector>

namespace whirl::qwen35 {

// ---- DFlash2 block-diffusion drafter (arch "dflash"; z-lab DFlash2, llama.cpp names):
// loaded next to the target (WHIRL_DRAFT_MODEL), drafts with `--draft dflash`.
// Most drafter layers accepted (FR-DFL P2-1; the official DFlash2 drafters have 5).
inline constexpr std::uint32_t kDFlashMaxLayers = 64;
struct DFlashConfig {
    std::uint32_t n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0;
    std::uint32_t block = 0, conv_k = 0, conv_group = 0, sel_rank = 0, sel_topk = 0, window = 0;
    std::uint32_t mask_id = 0;
    float rope_base = 0, eps = 0;
    // target layer ids, llama.cpp convention: id il = the input of target layer il
    // (= the output of layer il - 1; HF layers 5/19/33/47/61 -> 6/20/34/48/62)
    std::vector<std::uint32_t> target_layers;
    static DFlashConfig fromGguf(const gguf::File& f);
    // Throws ModelError unless it fits the target (hidden size, layer ids, vocabulary, tensors).
    void validate(const Config& target, const gguf::File& f) const;
};

struct DFlashLayerW {
    DevPtr attn_norm = 0, ffn_norm = 0, q_norm = 0, k_norm = 0;
    Mat q, k, v, o, gate, up, down;
    Mat attn_conv_proj, ffn_conv_proj;  // dynamic-convolution kernel projections
    DevPtr attn_conv_base = 0, ffn_conv_base = 0;  // [2][K][n_embd] f32 static kernels
};

struct DFlashW {
    DFlashConfig cfg;
    Mat fc;                     // [n_embd, n_target_layers * n_embd]
    DevPtr enc_norm = 0;        // enc.output_norm (after fc)
    DevPtr out_norm = 0;        // output_norm (decoder)
    Mat sel_prev, sel_next;     // selector codebooks [n_vocab, rank]
    Mat sel_hidden;             // [rank, n_embd]
    std::vector<DFlashLayerW> layers;
    std::uint64_t bytes = 0;    // weights in VRAM
};

struct LoadStats {
    std::uint64_t bytes = 0;
    std::uint32_t tensors = 0;
    // GDN layers whose ssm_beta / ssm_alpha were loaded into one block (beta rows then alpha rows)
    std::uint32_t gdn_ba_contig = 0;
    double ms = 0;
};

}  // namespace whirl::qwen35
