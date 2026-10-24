// KV page layout of one attention layer (arch024-S11c, F-6): which pool arrays a KV layer has and their bytes per
// token row, per KV format. The one definition behind the server's KV arrays (ServerModel::kvArrays, the order of a
// tier entry's page: K, V, K scales, V scales) and the model's KV bytes per token (Model::kvBytesPerTokenFmt,
// kvBytesPerTokenCfg). Before, the two were written out separately; the S11b mutant Q2 (V scales dropped from
// kvArrays only) made them disagree, and the server then silently turned the RAM tier off (page_row != per_tok).
// Pure (no HIP), so whirl-model-tests checks every format.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Same idea as llama.cpp's one ggml_row_size() for the K / V views and the size report.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace whirl::kv {

// One pool array of a KV layer.
enum class Part : std::uint8_t { k, v, k_scale, v_scale };

struct Row {
    Part part = Part::k;
    std::uint64_t bytes = 0;  // per token
};

// The arrays of one KV layer in page order (K, V, then K scales, V scales where the format has them).
struct LayerRows {
    std::array<Row, 4> a{};
    std::size_t n = 0;
    std::uint64_t total() const {
        std::uint64_t s = 0;
        for (std::size_t i = 0; i < n; ++i) s += a[i].bytes;
        return s;
    }
};

// elems = n_head_kv * head_dim. Formats (Model flags): f16 (!q8), q8 / q8h (q8), q8v (q8 && kf16: K f16, V int8),
// q4 (q8 && q4: 4-bit K / V). int8 / 4-bit rows carry one f16 scale per 32 values in their own array.
inline LayerRows layerRows(std::uint64_t elems, bool q8, bool kf16, bool q4) {
    LayerRows r;
    const bool kq8 = q8 && !kf16;
    const std::uint64_t rk = q4 ? elems / 2 : elems * (kq8 ? 1 : 2);
    const std::uint64_t rv = q4 ? elems / 2 : elems * (q8 ? 1 : 2);
    const std::uint64_t rs = elems / 32 * 2;
    r.a[r.n++] = {Part::k, rk};
    r.a[r.n++] = {Part::v, rv};
    if (kq8) r.a[r.n++] = {Part::k_scale, rs};
    if (q8) r.a[r.n++] = {Part::v_scale, rs};
    return r;
}

// KV bytes per token of one layer as the model sizes the pool (the formula of Model::kvBytesPerTokenFmt; q4 is
// counted with its scales, as the q4 format always has q8 set).
inline std::uint64_t layerBytes(std::uint64_t elems, bool q8, bool kf16, bool q4) {
    if (q4) return 2 * (elems / 2 + elems / 32 * 2);
    const std::uint64_t q = elems + elems / 32 * 2;
    return (q8 && !kf16 ? q : elems * 2) + (q8 ? q : elems * 2);
}

}  // namespace whirl::kv
