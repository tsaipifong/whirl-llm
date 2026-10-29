// Server engine, the WHIRL_PROFILE decode-cycle profile (arch025-engine S2, moved out of engine_internal.h): per
// number of decoding slots the cycle / row / draft / token counts and the wall, host, gap, enqueue and post-readCtl
// times the decode cycle adds up (engine_decode.cpp), printed and reset by print() as the "profile |" lines with the
// model's per-op-class GPU times (ServerModel::profileTake). No Engine (progress/arch025-engine.DESIGN.md, layer L1).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "backend.h"
#include "engine_api.h"
#include "whirl/model_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace whirl::server {

// WHIRL_PROFILE decode-cycle profile (index = decoding slots)

struct CycleProf {
    static constexpr std::size_t n = qwen35::gdn_max_seg + 1;  // = engine.h's gdn_max_seg + 1
    std::array<std::uint64_t, n> cycles{}, rows{}, drafts{}, tokens{};
    std::array<double, n> wall_ms{}, host_ms{}, gap_ms{}, post_ms{}, enq_ms{};
    std::optional<TimePoint> last_end;
    std::optional<TimePoint> t_ready;
    bool events = true;

    void print(ServerModel& m);
};

}  // namespace whirl::server
