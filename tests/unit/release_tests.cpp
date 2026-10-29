// WHIRL unit tests of module release (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-release-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#if WHIRL_TEST_DLL_DEPS
#include "release/dll_deps.h"
#endif
#include "whirl/small_helpers.h"  // lowerAscii (arch025-N6, lane 1 leg 11)

#include <cctype>

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
#if WHIRL_TEST_DLL_DEPS
#include "hip/win_text.h"  // narrowUtf8 / fileExists (arch025-N6, lane 1 leg 11; after <windows.h>)
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
#if WHIRL_TEST_DLL_DEPS
// Issue #2: amdhip64_7.dll present but its import rocm_kpack.dll missing (error 126) must not be
// reported as "amdhip64_7.dll was not found". Optional real case: WHIRL_TEST_HIP3684 = path of a copy
// of the ROCm 10.1 driver's amdhip64_7.dll (10.0.3684) in a folder without rocm_kpack.dll.
void testDllDeps() {
    namespace dd = whirl::app::dlldeps;
    auto has = [](const std::vector<std::string>& v, const char* name) {
        for (const std::string& s : v)
            if (_stricmp(s.c_str(), name) == 0) return true;
        return false;
    };
    wchar_t self[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, self, MAX_PATH);
    CHECK(n > 0 && n < MAX_PATH);
    const std::vector<std::string> imp = dd::peImports(self);
    CHECK(has(imp, "KERNEL32.dll"));
    CHECK(dd::peImports(L"C:\\no\\such\\file_whirl_test.dll").empty());
    CHECK(!dd::findDll(L"kernel32.dll").empty());
    CHECK(dd::findDll(L"no_such_dll_whirl_test.dll").empty());

    // the DLL itself is missing: the old wording, with the driver advice
    dd::Diagnosis none = dd::diagnose(L"no_such_dll_whirl_test.dll");
    CHECK(none.found_at.empty());
    std::string m = dd::message126("amdhip64_7.dll", none, "ADVICE\n");
    CHECK(m.find("amdhip64_7.dll) was not found") != std::string::npos);
    CHECK(m.find("ADVICE") != std::string::npos);

    // present, rocm_kpack.dll missing: names it and says where to put it, no driver advice
    dd::Diagnosis k;
    k.found_at = L"C:\\Windows\\System32\\amdhip64_7.dll";
    k.missing = {"rocm_kpack.dll"};
    m = dd::message126("amdhip64_7.dll", k, "ADVICE\n");
    CHECK(m.find("was found at C:\\Windows\\System32\\amdhip64_7.dll") != std::string::npos);
    CHECK(m.find("missing: rocm_kpack.dll") != std::string::npos);
    CHECK(m.find("next to whirl.exe") != std::string::npos);
    CHECK(m.find("was not found") == std::string::npos);
    CHECK(m.find("ADVICE") == std::string::npos);

    // rocm_kpack.dll present, but the Visual C++ runtime it needs is missing
    dd::Diagnosis v;
    v.found_at = L"C:\\Windows\\System32\\amdhip64_7.dll";
    v.missing = {"VCRUNTIME140.dll", "MSVCP140.dll"};
    m = dd::message126("amdhip64_7.dll", v, "ADVICE\n");
    CHECK(m.find("missing: VCRUNTIME140.dll, MSVCP140.dll") != std::string::npos);
    CHECK(m.find("from the WHIRL zip") != std::string::npos);  // 0.2.2 ships them app-locally
    CHECK(m.find("vc_redist.x64.exe") != std::string::npos);

    // present, nothing identifiable: still says "found", plus the driver advice
    dd::Diagnosis u;
    u.found_at = L"C:\\x\\amdhip64_7.dll";
    m = dd::message126("amdhip64_7.dll", u, "ADVICE\n");
    CHECK(m.find("was found at C:\\x\\amdhip64_7.dll") != std::string::npos);
    CHECK(m.find("ADVICE") != std::string::npos);

    // a present DLL whose imports all load: nothing missing
    dd::Diagnosis ok = dd::diagnose(self);
    CHECK(!ok.found_at.empty());
    CHECK(ok.missing.empty());

    if (const char* p = std::getenv("WHIRL_TEST_HIP3684"); p && *p) {
        const std::wstring wp(p, p + std::strlen(p));
        CHECK(has(dd::peImports(wp), "rocm_kpack.dll"));
        const dd::Diagnosis r = dd::diagnose(wp);
        CHECK(!r.found_at.empty());
        CHECK(has(r.missing, "rocm_kpack.dll"));
        std::printf("dll_deps: real 3684 case checked (%s)\n", p);
    }

    // ---- error 127 (0.2.2): a shipped DLL shadows the version a (future) driver expects.
    // The message must say which file to rename/remove or to use the matching WHIRL release,
    // and must never tell the user to update or install the driver.
    auto no_driver_advice = [](const std::string& s) {
        return s.find("update the driver") == std::string::npos && s.find("Install AMD Software") == std::string::npos &&
               s.find("newer driver") == std::string::npos;
    };
    dd::Shadow127 s;
    s.found_at = L"C:\\Windows\\System32\\amdhip64_7.dll";
    s.app_dir = L"C:\\whirl";
    s.bundled = {"rocm_kpack.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"};
    s.other_kpack = L"C:\\Windows\\System32\\rocm_kpack.dll";
    m = dd::message127("amdhip64_7.dll", s);
    CHECK(m.find("error 127") != std::string::npos);
    CHECK(m.find("at C:\\Windows\\System32\\amdhip64_7.dll") != std::string::npos);
    CHECK(m.find("C:\\whirl\\rocm_kpack.dll") != std::string::npos);
    CHECK(m.find("rename or remove rocm_kpack.dll in C:\\whirl") != std::string::npos);
    CHECK(m.find("WHIRL release that matches your driver") != std::string::npos);
    CHECK(m.find("own copy: C:\\Windows\\System32\\rocm_kpack.dll") != std::string::npos);
    CHECK(m.find("also rename msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll") != std::string::npos);
    CHECK(no_driver_advice(m));
    // kpack shipped, no driver copy found, no VC runtime next to it
    s.bundled = {"rocm_kpack.dll"};
    s.other_kpack.clear();
    m = dd::message127("amdhip64_7.dll", s);
    CHECK(m.find("rename or remove rocm_kpack.dll") != std::string::npos);
    CHECK(m.find("own copy") == std::string::npos);
    CHECK(m.find("also rename") == std::string::npos);
    CHECK(no_driver_advice(m));
    // only the VC++ runtime shipped (kpack removed by the user): names those files
    s.bundled = {"msvcp140.dll", "vcruntime140.dll"};
    m = dd::message127("amdhip64_7.dll", s);
    CHECK(m.find("rename or remove msvcp140.dll, vcruntime140.dll in C:\\whirl") != std::string::npos);
    CHECK(m.find("WHIRL release that matches your driver") != std::string::npos);
    CHECK(no_driver_advice(m));
    // nothing shipped present
    s.bundled.clear();
    m = dd::message127("amdhip64_7.dll", s);
    CHECK(m.find("None of the DLLs WHIRL ships") != std::string::npos);
    CHECK(m.find("WHIRL release that matches your driver") != std::string::npos);
    CHECK(no_driver_advice(m));

    // inspect127 on a real folder: finds the shipped files that exist, ignores the rest
    {
        wchar_t tmp[MAX_PATH];
        CHECK(GetTempPathW(MAX_PATH, tmp) > 0);
        const std::wstring dir = std::wstring(tmp) + L"whirl_test_127_" + std::to_wstring(GetCurrentProcessId());
        CHECK(CreateDirectoryW(dir.c_str(), nullptr) != 0);
        const std::wstring files[] = {dir + L"\\rocm_kpack.dll", dir + L"\\vcruntime140.dll", dir + L"\\amdhip64_7.dll"};
        for (const std::wstring& f : files) {
            HANDLE h = CreateFileW(f.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            CHECK(h != INVALID_HANDLE_VALUE);
            if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        }
        const dd::Shadow127 r = dd::inspect127(dir + L"\\amdhip64_7.dll", dir + L"\\");
        CHECK(r.found_at == dir + L"\\amdhip64_7.dll");
        CHECK(r.app_dir == dir);
        CHECK(r.bundled.size() == 2 && has(r.bundled, "rocm_kpack.dll") && has(r.bundled, "vcruntime140.dll"));
        // the runtime's own folder is the app folder here, so its kpack is ours, not "the driver's copy"
        CHECK(r.other_kpack.find(dir) == std::wstring::npos);
        for (const std::wstring& f : files) DeleteFileW(f.c_str());
        RemoveDirectoryW(dir.c_str());
    }
    CHECK(!dd::exeDir().empty());

    // Optional real case: WHIRL_TEST_KPACK127_DIR = folder with the ROCm 10.1 driver's amdhip64_7.dll
    // and a rocm_kpack.dll that lacks its exports (a stand-in for a mismatched kpack). The loader must
    // report 127 and the message must point at that folder's rocm_kpack.dll.
    if (const char* p = std::getenv("WHIRL_TEST_KPACK127_DIR"); p && *p) {
        const std::wstring dir(p, p + std::strlen(p));
        const std::wstring hip = dir + L"\\amdhip64_7.dll";
        const UINT old_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);
        HMODULE h = LoadLibraryExW(hip.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        const DWORD e = h ? 0 : GetLastError();
        SetErrorMode(old_mode);
        CHECK(h == nullptr && e == ERROR_PROC_NOT_FOUND);
        if (h) FreeLibrary(h);
        const dd::Shadow127 r = dd::inspect127(hip, dir);
        CHECK(has(r.bundled, "rocm_kpack.dll"));
        const std::string msg = dd::message127("amdhip64_7.dll", r);
        CHECK(msg.find("rename or remove rocm_kpack.dll in " + std::string(p)) != std::string::npos);
        CHECK(no_driver_advice(msg));
        std::printf("dll_deps: real 127 case checked (%s, LoadLibrary error %lu)\n%s", p, e, msg.c_str());
    }
}
#endif

// arch025-N6 (lane 1 leg 11): the shared helpers against verbatim copies of every old body they replaced
namespace n6old {
// src/common/gpu_select.cpp
std::string lower_gpu_select(std::string_view s) {
    std::string r(s);
    for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}
// src/common/numerics.cpp
std::string lower_numerics(std::string_view s) {
    std::string o(s);
    for (char& c : o)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return o;
}
// src/hip/hip_runtime.cpp
std::string lower_hip_runtime(std::string_view s) {
    std::string r(s);
    for (auto& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}
// src/server/http.cpp
std::string lower_http(std::string_view s) {
    std::string r(s);
    for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}
// src/release/dll_deps.cpp
std::string lower_dll_deps(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}
#if WHIRL_TEST_DLL_DEPS
// src/hip/runtime_loader.cpp (src/release/dll_deps.cpp had the same text)
std::string narrowUtf8_old(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<std::size_t>(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}
// src/hip/runtime_loader.cpp (src/release/dll_deps.cpp had the same text)
bool fileExists_old(const std::wstring& p) {
    const DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) == 0;
}
#endif
}  // namespace n6old

void testSharedHelpersN6() {
    // lowerAscii: every single byte, all 256 bytes in one string, embedded NUL, words the callers lower
    std::vector<std::string> in;
    for (int b = 0; b < 256; ++b) in.emplace_back(1, static_cast<char>(b));
    std::string all;
    for (int b = 0; b < 256; ++b) all.push_back(static_cast<char>(b));
    in.push_back(all);
    for (const char* w : {"", "GFX1201", "gfx1151", "AMD Radeon(TM) 8060S Graphics", "Rocm_KPack.DLL", "API-MS-Win-Core",
                          "http://LOCALHOST:8080", "GPU-0A1B2C3D-UUID", "Balance", " Fast\t", "\xC3\x89t\xC3\xA9"})
        in.emplace_back(w);
    in.emplace_back("A\0Z", 3);
    int lower_same = 0;
    for (const std::string& s : in) {
        const std::string want = n6old::lower_numerics(s);
        const bool same = whirl::lowerAscii(s) == want && n6old::lower_gpu_select(s) == want &&
                          n6old::lower_hip_runtime(s) == want && n6old::lower_http(s) == want && n6old::lower_dll_deps(s) == want;
        lower_same += same ? 1 : 0;
        CHECK(same);
    }
    CHECK_EQ(lower_same, static_cast<int>(in.size()));
#if WHIRL_TEST_DLL_DEPS
    // narrowUtf8: empty, ASCII, BMP, a surrogate pair, lone surrogates (replaced, flags 0), embedded NUL, U+FFFF, long
    const std::vector<std::wstring> wide = {L"", L"C:\\Program Files\\AMD\\ROCm\\bin\\amdhip64_7.dll", L"\u4E2D\u6587\\\u8DEF\u5F91",
                                            std::wstring(L"\xD83D\xDE00"), std::wstring(L"a\xD800" L"b"), std::wstring(L"\xDC00"),
                                            std::wstring(L"x\0y", 3), std::wstring(L"\xFFFF"), std::wstring(300, L'\u00E9')};
    for (const std::wstring& w : wide) CHECK(whirl::hip::win::narrowUtf8(w) == n6old::narrowUtf8_old(w));
    CHECK(whirl::hip::win::narrowUtf8(L"\u4E2D") == "\xE4\xB8\xAD");
    // fileExists: this program (a file), its folder (a directory), a missing file, "" and trailing separators
    wchar_t self[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, self, MAX_PATH);
    CHECK(n > 0 && n < MAX_PATH);
    const std::wstring exe(self, n);
    const std::wstring dir = exe.substr(0, exe.find_last_of(L"\\/"));
    for (const std::wstring& p : {exe, dir, dir + L"\\", dir + L"\\no_such_file_whirl_n6.dll", std::wstring(), exe + L"\\"})
        CHECK(whirl::hip::win::fileExists(p) == n6old::fileExists_old(p));
    CHECK(whirl::hip::win::fileExists(exe));
    CHECK(!whirl::hip::win::fileExists(dir));
#endif
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_release() {
    static const whirl::test::Suite s{"release",
                                      {
#if WHIRL_TEST_DLL_DEPS
                                          {"testDllDeps", [] { testDllDeps(); }},
#endif
                                          {"testSharedHelpersN6", [] { testSharedHelpersN6(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_release()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
