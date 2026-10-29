// whirl-iso-platform-time-tests: the platform clock layer (src/platform/time_win.cpp) built with nothing but its own header and winmm
// (ARCH-1 8.5 isolation, arch025-iso4). No unit test existed. The checks pin the 1 ms timer-resolution pair and the wall-clock fields.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "platform/time.h"

#include <chrono>
#include <ctime>

namespace {

namespace pt = whirl::platform::time;

void testTimerResolution() {
    ISO_CHECK(pt::beginPeriod1ms());  // timeBeginPeriod(1) == TIMERR_NOERROR
    pt::endPeriod1ms();               // balanced
    ISO_CHECK(pt::beginPeriod1ms());
    pt::endPeriod1ms();
}

void testLocalTime() {
    const pt::LocalTime t = pt::localTime();
    ISO_CHECK(t.year >= 2024 && t.year < 2200);
    ISO_CHECK(t.month >= 1 && t.month <= 12 && t.day >= 1 && t.day <= 31);
    ISO_CHECK(t.hour <= 23 && t.minute <= 59 && t.second <= 60 && t.milliseconds <= 999);
    // the fields agree with the C library's local time (same minute, or the next one when the call straddled a minute change)
    std::time_t now = std::time(nullptr);
    std::tm lt{};
    localtime_s(&lt, &now);
    ISO_CHECK(t.year == lt.tm_year + 1900 && t.month == lt.tm_mon + 1);
    const int d = (lt.tm_hour * 60 + lt.tm_min) - (t.hour * 60 + t.minute);
    ISO_CHECK(d >= 0 && d <= 1 || d == -1439 /* midnight */);
}

}  // namespace

int main(int argc, char** argv) { return iso::run("platform.time", {{"timer_resolution", testTimerResolution}, {"local_time", testLocalTime}}, argc, argv); }
