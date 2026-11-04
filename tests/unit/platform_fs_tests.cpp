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

// PATH-256 (v0.2.5.1 PLAN V1-22, pathlen256.DISCOVERY.md v3.1 2c #5 / #6): a file the user gives is never refused for
// its length; when its open / write fails, the message adds the full path's length in UTF-16 units and, from 260 on,
// the 259 limit of Windows without long path support (English, then Traditional Chinese). Written from the design before
// the implementation; every expected text is literal.
static void testPathLengthNote() {
    namespace pfs = whirl::platform::fs;
    const std::string zh_long = "\xef\xbc\x8cWindows \xe6\x9c\xaa\xe9\x96\x8b\xe9\x95\xb7\xe8\xb7\xaf\xe5\xbe\x91\xe6\x94\xaf\xe6\x8f\xb4"
                                "\xe6\x99\x82\xe4\xb8\x8a\xe9\x99\x90 259 \xe5\xad\x97\xe5\x85\x83";  // ，Windows 未開長路徑支援時上限 259 字元
    const auto zh = [](std::size_t n) {  // " / 路徑長 N 字元（UTF-16）"
        return " / \xe8\xb7\xaf\xe5\xbe\x91\xe9\x95\xb7 " + std::to_string(n) + " \xe5\xad\x97\xe5\x85\x83\xef\xbc\x88UTF-16\xef\xbc\x89";
    };
    const std::string p40 = "C:\\" + std::string(37, 'a');
    CHECK(pfs::pathLengthNote(p40) == " (the path is 40 characters long in UTF-16" + zh(40) + ")");
    const std::string p259 = "C:\\" + std::string(256, 'a');
    CHECK(pfs::pathLengthNote(p259) == " (the path is 259 characters long in UTF-16" + zh(259) + ")");
    const std::string p260 = "C:\\" + std::string(257, 'a');
    CHECK(pfs::pathLengthNote(p260) ==
          " (the path is 260 characters long in UTF-16; Windows without long path support allows at most 259" + zh(260) + zh_long + ")");
    // the unit is UTF-16: 100 Chinese characters after C:\ are 103 characters (303 bytes)
    std::string z = "C:\\";
    for (int i = 0; i < 100; ++i) z += "\xe4\xb8\xad";
    CHECK(pfs::pathLengthNote(z).find("the path is 103 characters long in UTF-16") != std::string::npos);
    // \\?\ is not counted; a relative path is measured as its full path
    CHECK(pfs::pathLengthNote("\\\\?\\" + p40).find("the path is 40 characters") != std::string::npos);
    std::error_code ec;
    const std::size_t cwd = std::filesystem::current_path(ec).wstring().size();
    CHECK(pfs::pathLengthNote("rel\\x.txt").find("the path is " + std::to_string(cwd + 10) + " characters") != std::string::npos);
    // the open / write failures carry it (readFile: --image, @file, server local images; writeFile: --out, debug dumps;
    // MappedFile: the GGUF file); a path of 300 characters (a string only: nothing is created)
    const std::string base = whirl::narrow((whirl::test::testTmpBase() / L"pln").wstring());
    const std::string p300 = base + "\\" + std::string(300 - base.size() - 1, 'f');
    const auto what = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const std::string r = what([&] { whirl::readFile(p300); });
    CHECK(r.rfind("cannot open " + p300 + " (the path is 300 characters long in UTF-16; Windows without long path support allows at most 259", 0) == 0);
    const std::string w = what([&] { whirl::writeFile(base + "\\no-such-folder\\out.bin", "x"); });
    CHECK(w.rfind("cannot write " + base + "\\no-such-folder\\out.bin (the path is " + std::to_string(base.size() + 23) + " characters long in UTF-16 / ", 0) == 0);
    const std::string mp = what([&] { whirl::MappedFile m(p300); });
    CHECK(mp.rfind("cannot open " + p300 + " (the path is 300 characters long in UTF-16;", 0) == 0);
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_platform_fs() {
    static const whirl::test::Suite s{"platform.fs",
                                      {
                                          {"testFsMappedOpen", [] { testFsMappedOpen(); }},
                                          {"testPathLengthNote", [] { testPathLengthNote(); }},  // PATH-256 2c #5
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_platform_fs()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
