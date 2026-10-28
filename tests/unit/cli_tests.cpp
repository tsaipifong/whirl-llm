// WHIRL unit tests of module cli (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-cli-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "cli/dflash_dump.h"

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
// arch025-gaptests G19 (X4 report): bug-emptyrootpaths (37ad538) (c), an empty WHIRL_DFLASH_DUMP means no dump. The
// dump itself runs only in a GPU whirl chat --draft dflash, so the rule cmd_chat.cpp applies is pinned here.
static void testDFlashDumpDir() {
    using OS = std::optional<std::string>;
    CHECK(!cli::dflashDumpDir(std::nullopt));         // unset: no dump
    CHECK(!cli::dflashDumpDir(OS("")));               // set but empty: no dump (before: "" -> create_directories("") threw)
    CHECK(cli::dflashDumpDir(OS("D:\\dump")) == OS("D:\\dump"));  // a folder: that folder, unchanged
    CHECK(cli::dflashDumpDir(OS(" ")) == OS(" "));    // only the empty string is "no dump" (as --kv-ssd-dir "")
    CHECK(cli::dflashDumpDir(OS(".")) == OS("."));
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_cli() {
    static const whirl::test::Suite s{"cli",
                                      {
                                          {"testDFlashDumpDir", [] { testDFlashDumpDir(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_cli()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
