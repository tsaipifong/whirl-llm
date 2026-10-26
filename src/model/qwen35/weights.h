// qwen35 weights (arch025-M1, R-8a): the weight types of a Qwen3.5-family checkpoint and qwen35::Weights, the
// tensors Model::load binds (embeddings, output head, trunk layers, MTP block) and the load statistics they carry.
// SPDX-License-Identifier: Apache-2.0
//
// Model derives from Weights (include/whirl/model.h), so every reader keeps its spelling (m.layers, tok_embd, ...).
// The binding itself (file reads, MXFP4 repack, name -> field, type / kernel / shape checks) is WeightLoader
// (weight_loader.h, weights.cpp). The device memory behind the pointers stays in the Model's allocation ledger
// (Model::alloc / allocations): GDN layers allocate their recurrent state between weight uploads, and the ledger's
// order is the process's free order at teardown (alloc log, invariant I2), see progress\arch025-M1.PRIOR_ART.md.

#pragma once

#include "device/handles.h"
#include "whirl/gguf.h"
#include "whirl/hip.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace whirl::qwen35 {

using hip::DevPtr;
using gguf::GgmlType;

constexpr std::uint32_t n_types = 40;  // > every GgmlType used (mxfp4 = 39)
// Prefill GEMM autotune buckets (tune_sizes, tuneBucket: include/whirl/model.h).
constexpr std::uint32_t n_tune = 11;

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

// The bound tensors of one checkpoint (Model::load through WeightLoader) and what the binding counted.
struct Weights {
    Mat tok_embd;
    device::PinnedBuf embd_host;  // token_embd in pinned host memory (embd_on_host), else empty
    Mat output;
    DevPtr output_norm = 0;
    std::vector<Layer> layers;
    std::optional<MtpW> mtp;
    // bytes uploaded per GGUF type (WeightLoader; a DFlash2 drafter's tensors are counted too)
    std::array<std::uint64_t, n_types> type_bytes{};
    // MXFP4 repack (WeightLoader::uploadMx, kernels::repackMxfp4Row): folded blocks that were rounded (tune.cpp log)
    std::uint64_t mx_fold_lossy = 0;
};

}  // namespace whirl::qwen35
