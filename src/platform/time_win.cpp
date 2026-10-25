// Platform layer, clocks and timer resolution: the Windows implementation of platform/time.h.
// SPDX-License-Identifier: Apache-2.0
//
// arch025 PL1: timeBeginPeriod / timeEndPeriod of src/server/server_main.cpp (struct TimerRes) and GetLocalTime of
// src/server/log.cpp, unchanged. winmm is linked by cmake/platform.cmake (was #pragma comment in server_main.cpp).

#include "platform/time.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <timeapi.h>

namespace whirl::platform::time {

bool beginPeriod1ms() { return timeBeginPeriod(1) == TIMERR_NOERROR; }

void endPeriod1ms() { timeEndPeriod(1); }

LocalTime localTime() {
    SYSTEMTIME st;
    GetLocalTime(&st);
    LocalTime t;
    t.year = st.wYear;
    t.month = st.wMonth;
    t.day = st.wDay;
    t.hour = st.wHour;
    t.minute = st.wMinute;
    t.second = st.wSecond;
    t.milliseconds = st.wMilliseconds;
    return t;
}

}  // namespace whirl::platform::time
