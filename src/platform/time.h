// Platform layer: clocks and timer resolution (arch025 PL1). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is time_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
// Note: inside whirl::platform the name `time` is this namespace; platform code calls std::time, never bare time().
#pragma once

namespace whirl::platform::time {

// timeBeginPeriod(1) == TIMERR_NOERROR / timeEndPeriod(1) (winmm). whirl-server / whirl serve hold 1 ms while serving
// (struct TimerRes in src/server/server_main.cpp keeps the RAII and the order).
bool beginPeriod1ms();
void endPeriod1ms();

// The local wall-clock time (GetLocalTime), the fields the log line prints.
struct LocalTime {
    unsigned short year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0, milliseconds = 0;
};
LocalTime localTime();

// std::chrono stays where it is used (not platform code).

}  // namespace whirl::platform::time
