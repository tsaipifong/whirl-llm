// Small helpers that several WHIRL modules wrote out the same way (arch025-N6, lane 1 leg 6; DESIGN 3.6 / 3.8): each
// copy was read and found identical before it was replaced (scratch/a25_lane1b/leg6/apply_n6.py lists them).
// Header-only, no dependency beyond the standard library.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>

namespace whirl {

// a - b, or 0 when b >= a (was src/common/vram_limit.cpp and src/tier/kv_tier.cpp)
constexpr std::uint64_t satSub(std::uint64_t a, std::uint64_t b) { return a > b ? a - b : 0; }
// x rounded up to a multiple of a (a > 0) (was src/server/server_main.cpp, src/tier/kv_tier.cpp, src/vision/vision.cpp)
constexpr std::uint64_t alignUp(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }
// seconds since the Unix epoch (was src/server/stream.cpp and src/tier/kv_tier.cpp)
inline std::int64_t unixNow() { return static_cast<std::int64_t>(std::time(nullptr)); }
// milliseconds of the steady clock since t0 (was src/server/engine.h, src/server/tier_bridge.cpp, src/tier/kv_tier.cpp;
// src/vision/vision.cpp's msSince(double seconds) is a different function and stays there)
inline double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
// the steady clock in milliseconds (was src/cli/common.cpp; src/model/tune.cpp has the same text)
inline double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace whirl
