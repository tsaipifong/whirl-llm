// WHIRL unit tests of module platform.console (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-platform-console-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "platform/console.h"
#include "platform/time.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>  // K32GetProcessMemoryInfo (arch025-surv testPlatformProcSurvivors)
#include <fcntl.h>  // _O_TEXT / _O_BINARY (arch025-mutcat2 setStdoutBinary check)
#include <io.h>     // _setmode, _fileno
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
static void testPlatformConsoleTime() {
#ifdef _WIN32
    namespace pc = whirl::platform::console;
    namespace pt = whirl::platform::time;
    CHECK(pc::kCodePageUtf8 == CP_UTF8);
    CHECK(pc::outputCodePage() == GetConsoleOutputCP());
    auto oldIsConsole = [](DWORD std_handle) {
        HANDLE h = GetStdHandle(std_handle);
        DWORD mode = 0;
        return h != nullptr && h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode) != 0;
    };
    CHECK(pc::isConsole(pc::Stream::out) == oldIsConsole(STD_OUTPUT_HANDLE));
    CHECK(pc::isConsole(pc::Stream::err) == oldIsConsole(STD_ERROR_HANDLE));
    // arch025-mutcat2: setStdoutBinary puts stdout in binary mode (the server / CLI stream raw UTF-8 and "\n" bytes);
    // _setmode returns the mode it replaces, so the probe reads the mode and puts it back
    {
        std::fflush(stdout);
        const int orig = _setmode(_fileno(stdout), _O_TEXT);
        CHECK(_setmode(_fileno(stdout), orig) == _O_TEXT);  // the probe's own restore (int6 cpu_analyze C6031: use the result)
        pc::setStdoutBinary();
        const int was = _setmode(_fileno(stdout), orig);  // back to the mode the program had
        CHECK(was == _O_BINARY);
    }
    // arch025-mutcat2: setOutputCodePage sets the OUTPUT code page (not the input one). Only with a console attached
    // (GetConsoleOutputCP() == 0 without one); the old value is put back.
    if (const unsigned old_out = GetConsoleOutputCP(); old_out != 0) {
        const unsigned old_in = GetConsoleCP();
        const unsigned want = old_out == 437 ? 850 : 437;
        CHECK(pc::setOutputCodePage(want));
        CHECK(GetConsoleOutputCP() == want);
        SetConsoleOutputCP(old_out);
        SetConsoleCP(old_in);
    } else {
        std::printf("  (no console: setOutputCodePage not checked)\n");
    }
    // local time: the same fields as GetLocalTime (retry once across a second boundary)
    bool same = false;
    for (int k = 0; k < 2 && !same; ++k) {
        SYSTEMTIME a;
        GetLocalTime(&a);
        const pt::LocalTime t = pt::localTime();
        SYSTEMTIME b;
        GetLocalTime(&b);
        same = a.wSecond == b.wSecond && t.year == a.wYear && t.month == a.wMonth && t.day == a.wDay && t.hour == a.wHour &&
               t.minute == a.wMinute && t.second == a.wSecond && t.milliseconds >= a.wMilliseconds &&
               t.milliseconds <= b.wMilliseconds;
    }
    CHECK(same);
    // timer resolution: begin succeeds (per process since Windows 10 2004) and is paired with end
    const bool on = pt::beginPeriod1ms();
    CHECK(on);
    if (on) pt::endPeriod1ms();
#endif
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_platform_console() {
    static const whirl::test::Suite s{"platform.console",
                                      {
                                          {"testPlatformConsoleTime", [] { testPlatformConsoleTime(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_platform_console()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
