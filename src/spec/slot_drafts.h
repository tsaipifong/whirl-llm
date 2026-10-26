// Per-slot MTP draft allocation that the server engine runs every decode cycle (src/server/engine.* until arch025-E1b):
// the WHIRL_SLOT_DRAFTS=2 planner over qwen35::allocDrafts (DraftPlanSlot, planSlotDrafts), the WHIRL_SLOT_DRAFTS=1
// marginal-gain swaps (splitDraftsByGain, the body of Engine::splitDrafts) and the draft cap for A decoding slots
// (slotDraftCap, the body of Engine::draftCap). Pure functions of values and acceptance models: the engine reads its
// slots / options / model and passes them in (the shape of llama.cpp's server_slot::get_n_draft_max feeding
// common_speculative's draft parameters; progress\arch025-E1b.PRIOR_ART.md).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "spec/policy.h"  // DraftAccept, CycleCost, allocDrafts

#include <cstdint>
#include <span>
#include <vector>

namespace whirl::server {

// One decoding slot as the per-slot draft planner (WHIRL_SLOT_DRAFTS=2) sees it.
struct DraftPlanSlot {
    const qwen35::DraftAccept* acc = nullptr;  // MTP slot: effective acceptance; unused for n-gram slots
    std::uint32_t pos = 0;                     // context length
    std::uint32_t ng_rows = 0;                 // > 0: n-gram slot with this many verify rows
    float ng_e = 0;                            // n-gram slot: expected tokens
    std::uint32_t d_prev = 0;                  // MTP drafts in its last MTP cycle (0: none)
};

// MTP draft counts of the MTP slots (in order, n-gram slots skipped) from qwen35::allocDrafts:
// `uniform` drafts each unless the cost model finds a better split within `rows` verify rows and
// `snaps` snapshot sets; every MTP slot gets 1..cap. Pure (no engine state), for tests.
std::vector<std::uint32_t> planSlotDrafts(std::span<const DraftPlanSlot> slots, const qwen35::CycleCost& cost,
                                          std::uint32_t uniform, std::uint32_t cap, std::uint32_t rows, std::uint32_t snaps);

// WHIRL_SLOT_DRAFTS=1 (with automatic draft counts, two or more MTP slots, nd > 0): moves drafts between the MTP slots
// by marginal gain (acc[k]->expected) while the verify stays within `rows` rows and every slot within `cap` drafts.
// On entry nd_m holds nd for every slot (Engine::splitDrafts fills it); nd_m.size() == acc.size().
void splitDraftsByGain(std::span<const qwen35::DraftAccept* const> acc, std::uint32_t nd, std::uint32_t rows,
                       std::uint32_t cap, std::span<std::uint32_t> nd_m);

// Largest uniform MTP draft count for A decoding slots: the per-batch-size setting `batch_drafts` capped by n_draft
// (1 without the fused decode), lowered until A * (nd + 1) verify rows fit in `vrows` and, without DeltaNet replay,
// A * nd snapshot sets fit in qwen35::gdn_max_snap.
std::uint32_t slotDraftCap(std::uint32_t batch_drafts, std::uint32_t n_draft, bool fused, bool replay, std::uint32_t A,
                           std::uint32_t vrows);

}  // namespace whirl::server
