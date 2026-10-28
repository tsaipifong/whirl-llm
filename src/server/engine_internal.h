// Server engine, the types engine.cpp and its split files (engine_decode / _pages / _prefill / _queue / _tier .cpp)
// share (arch025-N4f; were engine.cpp's own): one definition, so a KvPoolFull thrown in one file is the type another
// file's handler sees. Not part of the engine's interface (engine.h).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine.h"
#include "log.h"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace whirl::server {

struct KvPoolFull : std::runtime_error {
    KvPoolFull() : std::runtime_error("KvPoolFull") {}
};

// WHIRL_PROFILE decode-cycle profile (index = decoding slots)

struct CycleProf {
    static constexpr std::size_t n = gdn_max_seg + 1;
    std::array<std::uint64_t, n> cycles{}, rows{}, drafts{}, tokens{};
    std::array<double, n> wall_ms{}, host_ms{}, gap_ms{}, post_ms{}, enq_ms{};
    std::optional<TimePoint> last_end;
    std::optional<TimePoint> t_ready;
    bool events = true;

    void print(ServerModel& m) {
        for (std::size_t a = 1; a < n; ++a) {
            if (cycles[a] == 0) continue;
            const double c = static_cast<double>(cycles[a]);
            std::array<double, ServerModel::n_prof_classes> pm{}, pv{};
            m.profileTake(a, pm, pv);
            double tm = 0, tv = 0;
            for (double v : pm) tm += v;
            for (double v : pv) tv += v;
            logI("profile | slots {} | {} cycles, {:.2f} rows, {:.2f} drafts/slot, {:.2f} tok/cycle | wall {:.2f} ms/cycle "
                 "(pre-launch host {:.2f}, gap before cycle {:.2f}) | GPU mtp {:.2f} + verify {:.2f} ms | {:.1f} tok/s in "
                 "cycles",
                 a, cycles[a], static_cast<double>(rows[a]) / c, static_cast<double>(drafts[a]) / c / static_cast<double>(a),
                 static_cast<double>(tokens[a]) / c, wall_ms[a] / c, host_ms[a] / c, gap_ms[a] / c, tm / c, tv / c,
                 static_cast<double>(tokens[a]) * 1000.0 / (wall_ms[a] + gap_ms[a]));
            for (std::size_t i = 0; i < ServerModel::n_prof_classes; ++i) {
                if (pm[i] > 0 || pv[i] > 0)
                    logI("profile | slots {} |   {:<9} mtp {:>6.3f}  verify {:>6.3f} ms/cycle", a, m.profileClassName(i),
                         pm[i] / c, pv[i] / c);
            }
            logI("profile | slots {} |   host: enqueue {:.3f} ms, after readCtl {:.3f} ms per cycle", a, enq_ms[a] / c,
                 post_ms[a] / c);
            cycles[a] = rows[a] = drafts[a] = tokens[a] = 0;
            wall_ms[a] = host_ms[a] = gap_ms[a] = post_ms[a] = enq_ms[a] = 0;
        }
        last_end.reset();
    }
};

}  // namespace whirl::server
