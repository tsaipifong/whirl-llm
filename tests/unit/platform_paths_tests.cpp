// WHIRL unit tests of module platform.paths (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-platform-paths-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "platform/paths.h"
#include "whirl/common.h"

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
static void testPlatformPaths() {
    namespace pp = whirl::platform::paths;
    struct Row {
        const char* base;
        const char* whirl_dir;
        const char* log;
        const char* kvcache;
    };
    const Row rows[] = {
        {"C:\\Users\\u\\AppData\\Local", "C:\\Users\\u\\AppData\\Local\\whirl", "C:\\Users\\u\\AppData\\Local\\whirl\\server.log",
         "C:\\Users\\u\\AppData\\Local\\whirl\\kvcache"},
        // set but empty: the pure joins as before (callers never pass "": localAppData() is nullopt then); the log path
        // is the unset one (bug-emptyrootpaths: was "\\whirl\\server.log", the drive root)
        {"", "\\whirl", "whirl-server.log", "\\whirl\\kvcache"},
        {"D:\\x\\", "D:\\x\\\\whirl", "D:\\x\\\\whirl\\server.log", "D:\\x\\\\whirl\\kvcache"},  // no normalising
        {"C:\\Users\\\xe8\x94\xa1\\AppData\\Local", "C:\\Users\\\xe8\x94\xa1\\AppData\\Local\\whirl",
         "C:\\Users\\\xe8\x94\xa1\\AppData\\Local\\whirl\\server.log", "C:\\Users\\\xe8\x94\xa1\\AppData\\Local\\whirl\\kvcache"},
    };
    for (const Row& r : rows) {
        CHECK(pp::whirlDir(r.base) == r.whirl_dir);
        CHECK(pp::serverLogPath(std::optional<std::string>(r.base)) == r.log);
        CHECK(pp::kvCacheDir(r.base) == r.kvcache);
    }
    CHECK(pp::serverLogPath(std::nullopt) == "whirl-server.log");  // LOCALAPPDATA unset: the working folder
    // tune cache (src/model/tune.cpp): the file written, then the read order (tune4 in whirl, whirl-r9700; then the
    // tune3- / tune- tables of the research prototype)
    const std::string b = "C:\\Users\\u\\AppData\\Local";
    const std::string stem = "Swift-1.5-27B-MXFP4.gguf-16000000000-gfx1151r2.txt";
    CHECK(pp::tuneCachePath(b, stem) == "C:\\Users\\u\\AppData\\Local\\whirl\\tune4-Swift-1.5-27B-MXFP4.gguf-16000000000-gfx1151r2.txt");
    const std::vector<std::string> r4 = {
        "C:\\Users\\u\\AppData\\Local\\whirl\\tune4-Swift-1.5-27B-MXFP4.gguf-16000000000-gfx1151r2.txt",
        "C:\\Users\\u\\AppData\\Local\\whirl-r9700\\tune4-Swift-1.5-27B-MXFP4.gguf-16000000000-gfx1151r2.txt"};
    CHECK(pp::tuneReadPaths4(b, stem) == r4);
    const std::vector<std::string> rold = {
        "C:\\Users\\u\\AppData\\Local\\whirl\\tune3-Swift-1.5-27B-MXFP4.gguf-16000000000-gfx1151r2.txt",
        "C:\\Users\\u\\AppData\\Local\\whirl-r9700\\tune3-Swift-1.5-27B-MXFP4.gguf-16000000000-gfx1151r2.txt",
        "C:\\Users\\u\\AppData\\Local\\whirl\\tune-Swift-1.5-27B-MXFP4.gguf-16000000000-gfx1151r2.txt",
        "C:\\Users\\u\\AppData\\Local\\whirl-r9700\\tune-Swift-1.5-27B-MXFP4.gguf-16000000000-gfx1151r2.txt"};
    CHECK(pp::tuneReadPathsOld(b, stem) == rold);
    CHECK(pp::tuneCachePath("", "m.txt") == "\\whirl\\tune4-m.txt");
#ifdef _WIN32
    // localAppData() == both v0.2.4 reads (tune.cpp _dupenv_s, server_main.cpp std::getenv), set and unset
    auto oldDupenv = []() -> std::string {
        char* buf = nullptr;
        std::size_t len = 0;
        if (_dupenv_s(&buf, &len, "LOCALAPPDATA") == 0 && buf != nullptr) {
            std::string v(buf);
            std::free(buf);
            return v;
        }
        return {};
    };
    std::optional<std::string> saved;
    if (const char* v = std::getenv("LOCALAPPDATA")) saved = v;
    for (const char* val : {"C:\\pl1\\lad", "D:\\x y\\Local", static_cast<const char*>(nullptr)}) {
        _putenv_s("LOCALAPPDATA", val != nullptr ? val : "");  // "" deletes the variable (MS docs)
        const std::optional<std::string> got = pp::localAppData();
        const char* old_srv = std::getenv("LOCALAPPDATA");
        CHECK(got.has_value() == (old_srv != nullptr));
        CHECK(got.value_or(std::string()) == oldDupenv());
        if (val != nullptr) CHECK(got && *got == val);
        else CHECK(!got);
        // the server's default log path: v0.2.4 expression on the getenv pointer vs paths::serverLogPath
        const std::string old_log = old_srv ? std::string(old_srv) + "\\whirl\\server.log" : std::string("whirl-server.log");
        CHECK(pp::serverLogPath(got) == old_log);
    }
    _putenv_s("LOCALAPPDATA", saved ? saved->c_str() : "");
    // bug-lappdata: localAppData() is the UTF-16 value as UTF-8, the narrow read is the ANSI code page; they agree for
    // ASCII only (a non-ASCII user folder of the machine running the tests is not an error here)
    const bool saved_ascii = !saved || std::all_of(saved->begin(), saved->end(), [](char c) { return (static_cast<unsigned char>(c) & 0x80) == 0; });
    if (saved_ascii) CHECK(pp::localAppData() == saved);
#endif
}

#ifdef _WIN32
// bug-lappdata (2026-10-26): LOCALAPPDATA was read through the CRT narrow table (the ANSI code page) and the path text
// then went through widen() (UTF-8), so a non-ASCII user folder gave a mangled tune-cache / server-log / kvcache path in
// a process whose ANSI code page is not UTF-8. The child is this test program (--child-lappdata, helper mode in main())
// started with a fixed environment: SystemRoot, PATH and the LOCALAPPDATA of the case only.
static std::string hexBytes(const std::string& s) {
    std::string h;
    char b[3];
    for (unsigned char c : s) {
        std::snprintf(b, sizeof b, "%02x", c);
        h += b;
    }
    return h;
}

// helper mode: print what localAppData() and the old narrow read (_dupenv_s) return, then create <base>\whirl the way
// tune.cpp creates the tune folder (path text -> widen -> file system)
static int childLappdata() {
    namespace pp = whirl::platform::paths;
    const std::optional<std::string> got = pp::localAppData();
    std::optional<std::string> old;
    char* buf = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&buf, &len, "LOCALAPPDATA") == 0 && buf != nullptr) {
        old = std::string(buf);
        std::free(buf);
    }
    std::printf("new=%s:%s old=%s:%s", got ? "set" : "unset", hexBytes(got.value_or(std::string())).c_str(),
                old ? "set" : "unset", hexBytes(old.value_or(std::string())).c_str());
    if (got && !got->empty()) {
        std::error_code ec;
        std::filesystem::create_directories(whirl::widen(pp::whirlDir(*got)), ec);
        std::printf(" mkdir=%d", ec ? 0 : 1);
    }
    std::fflush(stdout);
    return 0;
}

struct LadChild {
    bool started = false;
    DWORD exit_code = ~0u;
    std::string out;
};

static LadChild runLadChild(const std::wstring& self, const std::optional<std::wstring>& lad, const std::filesystem::path& out_file) {
    LadChild r;
    // the environment block, sorted by name: LOCALAPPDATA (when given), PATH (DLL search), SystemRoot
    std::wstring blk;
    auto add = [&blk](const wchar_t* n, const std::wstring& v) {
        blk += n;
        blk += L'=';
        blk += v;
        blk += L'\0';
    };
    if (lad) add(L"LOCALAPPDATA", *lad);
    if (const wchar_t* v = _wgetenv(L"PATH")) add(L"PATH", v);
    if (const wchar_t* v = _wgetenv(L"SystemRoot")) add(L"SystemRoot", v);
    blk += L'\0';
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE f = CreateFileW(out_file.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return r;
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nullptr;
    si.hStdOutput = f;
    si.hStdError = f;
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"" + self + L"\" --child-lappdata";
    r.started = CreateProcessW(self.c_str(), cmd.data(), nullptr, nullptr, TRUE, CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
                               blk.data(), nullptr, &si, &pi) != 0;
    CloseHandle(f);
    if (!r.started) return r;
    if (WaitForSingleObject(pi.hProcess, 20000) != WAIT_OBJECT_0) TerminateProcess(pi.hProcess, 99);
    GetExitCodeProcess(pi.hProcess, &r.exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    std::ifstream in(out_file, std::ios::binary);
    r.out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return r;
}

static void testLocalAppDataChild() {
    namespace fs = std::filesystem;
    namespace pp = whirl::platform::paths;
    wchar_t self[1024];
    const DWORD n = GetModuleFileNameW(nullptr, self, 1024);
    CHECK(n > 0 && n < 1024);
    if (n == 0 || n >= 1024) return;
    const fs::path root = whirl::test::testTmpBase() / L"lappdata";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    const fs::path out_file = root / L"child_out.txt";
    // non-ASCII user folders: CJK + Latin-1 + a character outside the BMP (none of the last two in code page 950), and
    // CJK that code page 950 holds (the narrow read is then valid Big5, not '?', and still not UTF-8)
    for (const wchar_t* user : {L"u蔡é\U0001F600", L"測試"}) {
        const fs::path users = root / L"users";
        fs::remove_all(users, ec);
        const std::wstring lad = (users / user / L"AppData" / L"Local").wstring();
        const std::string want = whirl::narrow(lad);
        const LadChild c = runLadChild(self, lad, out_file);
        CHECK(c.started && c.exit_code == 0);
        CHECK(c.out.rfind("new=set:" + hexBytes(want) + " ", 0) == 0);
        if (c.out.rfind("new=set:" + hexBytes(want) + " ", 0) != 0) std::printf("  lappdata child: %s (want %s)\n", c.out.c_str(), hexBytes(want).c_str());
        CHECK(c.out.find(" mkdir=1") != std::string::npos);
        CHECK(fs::is_directory(fs::path(lad) / L"whirl", ec));  // the folder the user owns, not a mangled sibling
        std::size_t entries = 0;
        for (const fs::directory_entry& e : fs::directory_iterator(users, ec)) {
            ++entries;
            CHECK(e.path().filename().wstring() == user);
        }
        CHECK(entries == 1);
        // the path text of every user goes back to the same UTF-16 folder
        CHECK(whirl::widen(pp::kvCacheDir(want)) == lad + L"\\whirl\\kvcache");
        CHECK(whirl::widen(pp::serverLogPath(std::optional<std::string>(want))) == lad + L"\\whirl\\server.log");
        CHECK(whirl::widen(pp::tuneCachePath(want, "m.gguf-1.txt")) == lad + L"\\whirl\\tune4-m.gguf-1.txt");
    }
    // ASCII: the new read is byte for byte the old narrow read (paths of ASCII users do not change)
    {
        const std::wstring lad = (root / L"ascii user" / L"AppData" / L"Local").wstring();
        const std::string h = hexBytes(whirl::narrow(lad));
        const LadChild c = runLadChild(self, lad, out_file);
        CHECK(c.started && c.exit_code == 0);
        CHECK(c.out.rfind("new=set:" + h + " old=set:" + h + " mkdir=1", 0) == 0);
    }
    // bug-emptyrootpaths: set but empty (a parent's environment block holds "LOCALAPPDATA=": the narrow read sees it as
    // set) is unset for WHIRL, as llama.cpp fs_get_cache_directory (.empty()); was "new=set:" and every
    // <base>\whirl\... path then started at the drive root
    {
        const LadChild c = runLadChild(self, std::wstring(), out_file);
        CHECK(c.started && c.exit_code == 0);
        CHECK(c.out == "new=unset: old=set:");
        if (c.out != "new=unset: old=set:") std::printf("  lappdata child (empty): %s (want new=unset: old=set:)\n", c.out.c_str());
    }
    {
        const LadChild c = runLadChild(self, std::nullopt, out_file);
        CHECK(c.started && c.exit_code == 0);
        CHECK(c.out == "new=unset: old=unset:");
    }
    fs::remove_all(root, ec);
}

#endif  // _WIN32 (bug-lappdata helpers)

}  // namespace

namespace {
// the old single main()'s helper modes of these tests (the test runs this program as the child)
static int childModes(int argc, char** argv) {
#ifdef _WIN32
    if (argc == 2 && std::string(argv[1]) == "--child-lappdata") return childLappdata();  // bug-lappdata helper mode
#endif
    (void)argc;
    (void)argv;
    return -1;
}
}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_platform_paths() {
    static const whirl::test::Suite s{"platform.paths",
                                      {
                                          {"testPlatformPaths", [] { testPlatformPaths(); }},
#if defined(_WIN32)
                                          {"testLocalAppDataChild", [] { testLocalAppDataChild(); }},
#endif
                                      },
                                      childModes};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_platform_paths()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
