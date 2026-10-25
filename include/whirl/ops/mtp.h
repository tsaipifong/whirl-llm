// MTP drafting and speculative-verify choice (ARCH-1 §6.5, arch025-F6): the draft head (rows, the 2-bit head's row
// map), whether a chained draft gates its big kernels, the row limit of a batched verify / MTP step, and which batched
// verifies the unfused decode path can run. Pure functions; src/model/forward/mtp_verify.cpp asks here and launches.
// SPDX-License-Identifier: Apache-2.0
//
// Tests: tests/ops/ops_tests.cpp (testMtpChoice).

#pragma once

#include <cstddef>
#include <cstdint>

namespace whirl::ops {

// the draft head over the vocabulary subset (the 2-bit head's draft_rows rows, mapped back by draft_map) or the
// first draft_vocab rows only (0 = the whole head)
struct DraftHeadProblem {
    std::uint32_t head_rows = 0;    // rows of the head matrix (the draft head, else the output head)
    std::uint32_t draft_rows = 0;   // rows of the 2-bit head's vocabulary subset
    std::uint32_t draft_vocab = 0;  // WHIRL_DRAFT_VOCAB / the model's draft vocabulary
    bool d2 = false;                // the 2-bit draft head is built (Model::draft_d2)
    bool map = false;               // the subset's row -> token map exists (Model::draft_map)
};
struct DraftHeadPlan {
    std::uint32_t nrows = 0;  // head rows the draft head matmul and draft_pick read
    bool map = false;         // draft_pick maps rows through draft_map
};
DraftHeadPlan chooseDraftHead(const DraftHeadProblem& p);

// a chained draft after a possible device-side cutoff (device token, no host tokens, slot > 0, draft_p_min > 0)
// gates its big kernels on the control word
constexpr bool draftGated(std::uint32_t n_dev, std::size_t n_host, bool has_slot, std::uint32_t slot, float draft_p_min) {
    return n_dev > 0 && n_host == 0 && has_slot && slot > 0 && draft_p_min > 0;
}

// the small-batch limit while a batched verify / MTP step of n rows runs (wide verifies: 17..max_verify_rows)
constexpr std::uint32_t verifySmallMax(std::uint32_t n, std::uint32_t max_small_batch, std::uint32_t max_verify_rows) {
    return n > max_small_batch ? max_verify_rows : max_small_batch;
}

// without the fused decode kernels a batched verify is one sequence with at most one draft (several sequences need the
// segment-aware fused DeltaNet kernels; the unfused path writes only snapshot 0)
constexpr bool unfusedVerifyOk(std::size_t n_segs, std::uint32_t nd0) { return n_segs <= 1 && nd0 <= 1; }

}  // namespace whirl::ops
