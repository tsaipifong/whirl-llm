// WHIRL unit tests of module platform.env (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-platform-env-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "platform/env.h"

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
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
static void testPlatformEnv() {
    namespace pe = whirl::platform::env;
#ifdef _WIN32
    CHECK(!pe::get("WHIRL_PL1U_X"));
    CHECK(pe::set("WHIRL_PL1U_X", "v=1") == 0);
    CHECK(pe::get("WHIRL_PL1U_X") == std::optional<std::string>("v=1"));
    CHECK(pe::get("whirl_pl1u_x") == std::optional<std::string>("v=1"));  // names are case-insensitive on Windows
    // bug-lappdata: getUtf8 is get for ASCII values (set, any case, unset below) and the UTF-8 of the wide value else
    CHECK(pe::getUtf8("WHIRL_PL1U_X") == std::optional<std::string>("v=1"));
    CHECK(pe::getUtf8("whirl_pl1u_x") == std::optional<std::string>("v=1"));
    CHECK(!pe::getUtf8("WHIRL_PL1U_NONE"));
    CHECK(_wputenv_s(L"WHIRL_PL1U_W", L"a蔡é\U0001F600") == 0);
    CHECK(pe::getUtf8("WHIRL_PL1U_W") == std::optional<std::string>("a\xe8\x94\xa1\xc3\xa9\xf0\x9f\x98\x80"));
    CHECK(_wputenv_s(L"WHIRL_PL1U_W", L"") == 0);
    CHECK(!pe::getUtf8("WHIRL_PL1U_W"));
    // block(): the same entries in the same order as the v0.2.4 GetEnvironmentStringsW loop
    std::vector<std::wstring> ref;
    if (wchar_t* blk = GetEnvironmentStringsW()) {
        for (const wchar_t* p = blk; *p; p += wcslen(p) + 1) ref.emplace_back(p);
        FreeEnvironmentStringsW(blk);
    }
    const std::vector<std::wstring> got = pe::block();
    CHECK(!got.empty());
    CHECK(got == ref);
    CHECK(std::find(got.begin(), got.end(), std::wstring(L"WHIRL_PL1U_X=v=1")) != got.end());
    CHECK(pe::set("WHIRL_PL1U_X", "") == 0);  // "" deletes
    CHECK(!pe::get("WHIRL_PL1U_X"));
    const std::vector<std::wstring> after = pe::block();
    CHECK(std::find(after.begin(), after.end(), std::wstring(L"WHIRL_PL1U_X=v=1")) == after.end());
    CHECK(after.size() + 1 == got.size());
#endif
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_platform_env() {
    static const whirl::test::Suite s{"platform.env",
                                      {
                                          {"testPlatformEnv", [] { testPlatformEnv(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_platform_env()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
