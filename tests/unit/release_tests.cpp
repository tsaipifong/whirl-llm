// WHIRL unit tests of module release (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-release-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#if WHIRL_TEST_DLL_DEPS
#include "release/dll_deps.h"
#endif

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

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_release() {
    static const whirl::test::Suite s{"release",
                                      {
#if WHIRL_TEST_DLL_DEPS
                                          {"testDllDeps", [] { testDllDeps(); }},
#endif
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_release()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
