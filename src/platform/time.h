// Platform layer: clocks and timer resolution (arch025 PL1). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is time_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
// Note: inside whirl::platform the name `time` is this namespace; platform code calls std::time, never bare time().
#pragma once

namespace whirl::platform::time {

// PL1 moves here (same calls, same order, same log text):
//   - the 1 ms timer resolution of whirl-server / whirl serve: timeBeginPeriod / timeEndPeriod (struct TimerRes,
//     winmm.lib, <timeapi.h>) (src/server/server_main.cpp)
//   - the local wall-clock time of log lines: GetLocalTime (src/server/log.cpp)
// std::chrono stays where it is used (not platform code).

}  // namespace whirl::platform::time
