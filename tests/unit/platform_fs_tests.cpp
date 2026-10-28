// WHIRL unit tests of module platform.fs (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-platform-fs-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/common.h"
#include "platform/fs.h"
#include "platform/fs_fake.h"

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
// arch025 PL2: MappedFile's CreateFileW keeps the access / share / disposition / flags of v0.2.4 0e35407
// (util.cpp:31: GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL), recorded by FakeFs.
void testFsMappedOpen() {
    namespace pfs = whirl::platform::fs;
    const std::filesystem::path dir = whirl::test::testTmpBase() / L"unit_fs_mapped";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string path = whirl::narrow((dir / L"m.bin").wstring());
    const std::string empty = whirl::narrow((dir / L"empty.bin").wstring());
    whirl::writeFile(path, "0123456789");
    whirl::writeFile(empty, "");
    pfs::fake::begin();
    {
        whirl::MappedFile m(path);
        CHECK(m.size() == 10 && std::memcmp(m.data(), "0123456789", 10) == 0);
        whirl::MappedFile e(empty);
        CHECK(e.size() == 0);
    }
    bool threw = false;
    try {
        whirl::MappedFile x(whirl::narrow((dir / L"missing.bin").wstring()));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    const auto opens = pfs::fake::end();
    CHECK_EQ(opens.size(), static_cast<std::size_t>(3));
    for (const auto& o : opens) {
        CHECK_EQ(o.access, 0x80000000u);    // GENERIC_READ
        CHECK_EQ(o.share, 0x1u);            // FILE_SHARE_READ
        CHECK_EQ(o.disposition, 3u);        // OPEN_EXISTING
        CHECK_EQ(o.flags, 0x80u);           // FILE_ATTRIBUTE_NORMAL (no NO_BUFFERING / OVERLAPPED / WRITE_THROUGH)
    }
    if (opens.size() == 3) CHECK(opens[0].ok && opens[1].ok && !opens[2].ok);
    // off again: nothing recorded
    { whirl::MappedFile m(path); }
    CHECK(pfs::fake::snapshot().empty());
    std::filesystem::remove_all(dir, ec);
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_platform_fs() {
    static const whirl::test::Suite s{"platform.fs",
                                      {
                                          {"testFsMappedOpen", [] { testFsMappedOpen(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_platform_fs()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
