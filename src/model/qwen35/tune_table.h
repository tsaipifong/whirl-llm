// qwen35::TuneTable: the prefill GEMM tune switches and the multi-token GEMV selection tables of a Model
// (arch025-M3, R-8c), and the tune cache text (tune4-*.txt) as pure functions over the matrices (tuneText /
// applyTuneText, src/model/qwen35/tune_table.cpp; the choices themselves live in each Mat::tune).
// SPDX-License-Identifier: Apache-2.0
//
// A private base of qwen35::Model (autotune, the GEMV dispatch and the selftests read the members by name as before);
// outside the model through Model::tuneTable(). The declarations moved verbatim from include/whirl/model.h.

#pragma once

#include "model/qwen35/weights.h"  // Mat, n_types
#include "whirl/model_types.h"     // max_small_batch

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace whirl::qwen35 {

using GemvR = std::array<std::uint8_t, max_small_batch + 1>;
using GemvW = std::array<GemvR, n_types>;
GemvR defaultGemvR();  // the gfx1201 row of device::Profile::gemv_r
GemvW defaultGemvW();
GemvR defaultGemvWHead();

struct TuneTable {
    std::uint64_t tune_mask = ~0ull;
    bool tune_cold = false;
    GemvR gemv_r = defaultGemvR();
    GemvW gemv_w = defaultGemvW();
    GemvR gemv_w_head = defaultGemvWHead();
};

// The tune table as text (Model::writeTune): one line "bucket type rows cols choice" per distinct (bucket, type, rows,
// cols), in the order of `mats` (every layer's tuneMats), buckets [0, n_tune).
std::string tuneText(std::span<Mat* const> mats);
// Apply buckets [0, n_buckets) of such a text to `mats` (Model::readTune). False when a line is malformed or a choice is
// out of range (nothing applied), or when a matrix shape is missing (stale cache: the matrices before it, in order,
// already took their choices).
bool applyTuneText(std::string_view text, std::size_t n_buckets, std::span<Mat* const> mats);

}  // namespace whirl::qwen35
