// Server engine, the engine's main-thread state groups (arch025-engine S3, module server.engine_types; moved verbatim
// out of engine.h): BatchStats (arch025-N4c), SpecState and FloorState (arch025-N4f). Engine holds them at the
// positions of the members they replaced (destruction order unchanged).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "server/decode_floor.h"      // DecodeFloor
#include "server/engine_api.h"        // TimePoint
#include "server/engine_common.h"     // gdn_max_seg
#include "server/engine_slot.h"       // max_small_batch
#include "spec/policy.h"              // qwen35::DraftTiming, qwen35::CycleCost, qwen35::DraftAccept

#include <array>
#include <cstdint>
#include <memory>

namespace whirl::server {

struct CycleProf;

// The engine's batch statistics (main thread; the "batch |" status line, Stats' DFlash2 counters). arch025-N4c: the
// former Engine members stat_t0_ .. stat_dfl_gap_, same types, initial values and order.
struct BatchStats {
    TimePoint t0{};
    std::uint64_t tokens = 0, cycles = 0, rows = 0, slots = 0, drafts = 0,
                  lit_drafts = 0,  // FR-SRV P2-5: DFlash2 / n-gram drafts verified (all slots)
                  prefill_tok = 0, seg_batches = 0, seg_chunks = 0;
    std::uint64_t dfl_cycles = 0, dfl_drafted = 0, dfl_accepted = 0, dfl_cold = 0, dfl_gap = 0;
};

// The decode cycle's draft-policy state (main thread). arch025-N4f: the former Engine members timing_ .. policy_cycles_,
// same types, initial values and order (Engine keeps it where they were: destruction order unchanged).
struct SpecState {
    std::array<qwen35::DraftTiming, gdn_max_seg + 1> timing{};
    // WHIRL_SLOT_DRAFTS=2: cycle time per number of decoding slots, global acceptance pool
    std::array<qwen35::CycleCost, gdn_max_seg + 1> ccost{};
    std::array<bool, gdn_max_seg + 1> ccost_primed{};
    qwen35::DraftAccept dpool{};
    // DFlash2 cycle times by decoding slots (drafting included), keyed by the largest draft count
    std::array<qwen35::DraftTiming, gdn_max_seg + 1> dtiming{};
    std::unique_ptr<CycleProf> prof;  // CycleProf is complete where Engine is destroyed (engine.cpp, engine_internal.h)
    std::uint32_t n_cycles = 0;
    // most rows of one batched verify (ServerModel::verifyRows: 16, or 32 with the wide GEMV path)
    std::uint32_t vrows = max_small_batch;
    std::uint32_t prev_nd = 0;
    std::uint32_t policy_cycles = 0;  // cycles since the draft-policy state was last reset
};

// The decode floor bookkeeping (main thread). arch025-N4f: the former Engine members floor_ .. stat_floor_periods_,
// same types, initial values and order.
struct FloorState {
    DecodeFloor floor;
    TimePoint epoch{};
    TimePoint log_t{};
    std::uint64_t waits = 0, periods = 0;
};

}  // namespace whirl::server
