// whirl-hiprt-tests: unit tests of the HIP runtime loader (ARCH-1 D-1,
// include/whirl/hip_runtime_loader.h). CPU only: fake runtimes built from
// tests/hiprt/fake_hiprt.cpp, copied into temporary folders.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL.
//   whirl-hiprt-tests            fake / absent DLLs only (never loads the driver's runtime)
//   WHIRL_TEST_REAL_HIPRT=1      also loads this machine's real runtime through load() and checks
//                                that a runtime which does not import rocm_kpack.dll never loads it

#include "hip/hip_api.h"
#include "release/dll_deps.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {

int g_pass = 0, g_fail = 0;

void check(bool ok, const char* what, int line, const std::string& detail = {}) {
    if (ok) {
        ++g_pass;
        return;
    }
    ++g_fail;
    std::printf("FAIL line %d: %s %s\n", line, what, detail.c_str());
}
#define CHECK(c) check((c), #c, __LINE__)
#define CHECKD(c, d) check((c), #c, __LINE__, (d))

namespace rt = whirl::hip::runtime;
namespace dd = whirl::app::dlldeps;

std::string n8(const std::wstring& w) {
    std::string s;
    for (wchar_t c : w) s += c < 128 ? static_cast<char>(c) : '?';
    return s;
}

struct TempDirs {
    std::wstring root;
    std::vector<std::wstring> files, dirs;
    TempDirs() {
        wchar_t t[MAX_PATH];
        GetTempPathW(MAX_PATH, t);
        root = std::wstring(t) + L"whirl_hiprt_" + std::to_wstring(GetCurrentProcessId());
        CreateDirectoryW(root.c_str(), nullptr);
        dirs.push_back(root);
    }
    std::wstring dir(const wchar_t* name) {
        std::wstring d = root + L"\\" + name;
        CreateDirectoryW(d.c_str(), nullptr);
        dirs.push_back(d);
        return d;
    }
    bool copy(const std::wstring& from, const std::wstring& to) {
        files.push_back(to);
        return CopyFileW(from.c_str(), to.c_str(), FALSE) != 0;
    }
    ~TempDirs() {
        for (const auto& f : files) {
            while (HMODULE h = GetModuleHandleW(f.c_str())) FreeLibrary(h);
            DeleteFileW(f.c_str());
        }
        for (auto it = dirs.rbegin(); it != dirs.rend(); ++it) RemoveDirectoryW(it->c_str());
    }
};

bool loaded(const std::wstring& path) { return GetModuleHandleW(path.c_str()) != nullptr; }

void unload(const std::wstring& path) {
    while (HMODULE h = GetModuleHandleW(path.c_str())) FreeLibrary(h);
}

bool allSlots(const rt::Api& api) {
    const void* const* s = reinterpret_cast<const void* const*>(&api);
    for (std::size_t i = 0; i < rt::symbolCount(); ++i)
        if (!s[i]) return false;
    return true;
}

// the HIP functions the delay-load import table of the v0.2.2 whirl.exe named (dumpbin /imports,
// progress/arch2 evidence): the table must resolve every one of them, under the same exported name
const char* const k_v022_imports[] = {
    "hipDeviceSynchronize", "hipGraphLaunch", "hipSetDevice", "hipGetDevice", "hipGetDeviceCount", "hipGetDevicePropertiesR0600",
    "hipGetLastError", "hipGetErrorString", "hipStreamCreate", "hipStreamCreateWithFlags", "hipStreamDestroy",
    "hipStreamSynchronize", "hipStreamWaitEvent", "hipEventCreateWithFlags", "hipEventCreate", "hipEventRecord", "hipEventDestroy",
    "hipEventSynchronize", "hipEventElapsedTime", "hipEventQuery", "hipMalloc", "hipHostMalloc", "hipFree", "hipHostFree",
    "hipMemcpy", "hipModuleGetGlobal", "hipMemcpyAsync", "hipMemset", "hipMemGetInfo", "hipModuleUnload", "hipModuleGetFunction",
    "hipModuleLoadData", "hipModuleLaunchKernel", "hipStreamBeginCapture", "hipStreamEndCapture", "hipGraphDestroy",
    "hipGraphInstantiateWithFlags", "hipGraphExecDestroy"};

void testTable() {
    std::set<std::string> names;
    for (std::size_t i = 0; i < rt::symbolCount(); ++i) names.insert(rt::symbolName(i));
    CHECK(names.size() == rt::symbolCount());  // no duplicates
    CHECK(rt::symbolCount() == sizeof(rt::Api) / sizeof(void*));
    CHECK(rt::symbolName(rt::symbolCount()) == nullptr);
    for (const char* n : k_v022_imports) CHECKD(names.count(n) == 1, n);
    // exactly the functions v0.2.2 imported: hipMemsetAsync (no caller, dropped by the 0.2.2 linker) is not
    // required at start-up (REVIEW-ARCH-D1 P3-1)
    CHECK(names.size() == std::size(k_v022_imports) && names.count("hipMemsetAsync") == 0);
}

void testFakes(const std::wstring& fakes) {
    const std::wstring full = fakes + L"\\whirl_fake_hiprt.dll", drop = fakes + L"\\whirl_fake_hiprt_drop.dll",
                       withdep = fakes + L"\\whirl_fake_hiprt_dep.dll", dep = fakes + L"\\whirl_fake_dep.dll";
    bool have = true;
    for (const auto& f : {full, drop, withdep, dep}) {
        const bool e = GetFileAttributesW(f.c_str()) != INVALID_FILE_ATTRIBUTES;
        CHECKD(e, n8(f));
        have = have && e;
    }
    // arch6b: without the fakes every probe below fails and the API slots stay null; calling one crashed the test
    // (0xC0000005, empty log) in a saved bin that held only the *.exe. Stop here: the FAIL lines above name the files.
    if (!have) {
        std::printf("FAIL: fake runtimes missing; copy the build's hiprt_fakes folder next to whirl-hiprt-tests.exe\n");
        return;
    }
    TempDirs t;
    const std::wstring empty = t.dir(L"empty");

    // absent everywhere: 126, nothing found, no fallback path invented
    {
        const rt::LoadResult r = rt::probe(empty, L"whirl_no_such_hiprt_t2.dll");
        CHECK(!r.ok && r.error == ERROR_MOD_NOT_FOUND && r.path.empty() && !r.app_local && r.missing_symbol.empty());
        CHECKD(rt::failureText(r).find("error 126") != std::string::npos, rt::failureText(r));
    }
    // a complete fake in the executable's folder: loaded from there, every slot points into it
    {
        const std::wstring a = t.dir(L"t3"), p = a + L"\\whirl_t3.dll";
        CHECK(t.copy(full, p));
        rt::Api api;
        const rt::LoadResult r = rt::probeTable(a, L"whirl_t3.dll", &api, true);
        CHECK(r.ok && r.app_local && r.path == p && r.error == 0 && r.missing_symbol.empty());
        CHECK(allSlots(api));
        int n = 0;
        CHECK(api.hipGetDeviceCount(&n) == hipSuccess && n == 7);
        CHECK(std::strcmp(api.hipGetErrorString(hipSuccess), "fake") == 0);
        CHECKD(rt::describe(r) == n8(p), rt::describe(r));  // exact path logged (the fake has no version resource)
        CHECK(loaded(p));
        unload(p);
        // keep = false unloads again
        const rt::LoadResult r2 = rt::probe(a, L"whirl_t3.dll");
        CHECK(r2.ok && !loaded(p));
    }
    // a HIP function missing (old driver): named, module unloaded, not reported as a load error
    {
        const std::wstring a = t.dir(L"t4"), p = a + L"\\whirl_t4.dll";
        CHECK(t.copy(drop, p));
        const rt::LoadResult r = rt::probe(a, L"whirl_t4.dll", true);
        const std::string last = rt::symbolName(rt::symbolCount() - 1);
        CHECKD(!r.ok && r.error == 0 && r.missing_symbol == last && r.path == p, r.missing_symbol);
        CHECK(!loaded(p));
        CHECKD(rt::failureText(r).find("has no " + last) != std::string::npos, rt::failureText(r));
        // the user message (gpuPreflight): a copy in the executable's folder -> remove or replace that file,
        // named by its path; no "driver too old" text or driver advice (REVIEW-ARCH-D1 P2-2)
        const std::string m = dd::messageMissingSymbol("amdhip64_7.dll", r.missing_symbol, r.app_local, r.path, "ADVICE\n");
        CHECKD(r.app_local && m.find(n8(p)) != std::string::npos && m.find("has no " + last) != std::string::npos &&
                   m.find("Remove that file") != std::string::npos && m.find("replace it") != std::string::npos &&
                   m.find("ADVICE") == std::string::npos && m.find("too old") == std::string::npos,
               m);
    }
    // fixed order: the executable's folder before the search path; a file found there that fails is the
    // result (no fallback to a working copy further down); not in the folder -> the search path, exact path
    {
        const std::wstring a = t.dir(L"t5app"), b = t.dir(L"t5path");
        CHECK(t.copy(full, a + L"\\whirl_t5.dll") && t.copy(drop, b + L"\\whirl_t5.dll"));
        CHECK(t.copy(drop, a + L"\\whirl_t6.dll") && t.copy(full, b + L"\\whirl_t6.dll"));
        CHECK(t.copy(drop, b + L"\\whirl_t5b.dll"));
        std::wstring old(32767, L'\0');
        old.resize(GetEnvironmentVariableW(L"PATH", old.data(), 32767));
        SetEnvironmentVariableW(L"PATH", (b + L";" + old).c_str());
        const rt::LoadResult r5 = rt::probe(a, L"whirl_t5.dll");
        CHECKD(r5.ok && r5.app_local && r5.path == a + L"\\whirl_t5.dll", n8(r5.path));
        const rt::LoadResult r6 = rt::probe(a, L"whirl_t6.dll");
        CHECKD(!r6.ok && r6.app_local && r6.path == a + L"\\whirl_t6.dll" && !r6.missing_symbol.empty(), n8(r6.path));
        const rt::LoadResult r5b = rt::probe(empty, L"whirl_t5b.dll");
        CHECKD(!r5b.ok && !r5b.app_local && r5b.path == b + L"\\whirl_t5b.dll" && !r5b.missing_symbol.empty(), n8(r5b.path));
        // found by the search path (System32 for the real driver): the driver wording and advice, unchanged
        {
            const std::string m = dd::messageMissingSymbol("amdhip64_7.dll", r5b.missing_symbol, r5b.app_local, r5b.path, "ADVICE\n");
            CHECKD(m == "error: the installed AMD graphics driver is too old: its GPU runtime (amdhip64_7.dll) has no " +
                            r5b.missing_symbol + ".\nADVICE\n",
                   m);
        }
        const rt::LoadResult r5c = rt::probe(empty, L"whirl_t5.dll");  // search path only: the broken copy in b
        CHECKD(!r5c.ok && !r5c.app_local && r5c.path == b + L"\\whirl_t5.dll", n8(r5c.path));
        SetEnvironmentVariableW(L"PATH", old.c_str());
    }
    // 126 with the DLL present: its import is missing (issue #2: rocm_kpack.dll); the dll_deps diagnosis
    // gpuPreflight prints names the missing import
    {
        const std::wstring a = t.dir(L"t7"), p = a + L"\\whirl_t7.dll";
        CHECK(t.copy(withdep, p));
        const rt::LoadResult r = rt::probe(a, L"whirl_t7.dll");
        CHECK(!r.ok && r.error == ERROR_MOD_NOT_FOUND && r.app_local && r.path == p);
        const dd::Diagnosis d = dd::diagnose(r.path);
        CHECK(d.found_at == p && d.missing.size() == 1 && d.missing[0] == "whirl_fake_dep.dll");
        const std::string m = dd::message126("amdhip64_7.dll", d, "ADVICE\n");
        CHECKD(m.find("whirl_fake_dep.dll") != std::string::npos && m.find("was found at") != std::string::npos, m);
    }
    // 127: the import is present but lacks the function (a shipped DLL shadowing another version)
    {
        const std::wstring a = t.dir(L"t8"), p = a + L"\\whirl_t8.dll";
        CHECK(t.copy(withdep, p) && t.copy(full, a + L"\\whirl_fake_dep.dll"));
        const rt::LoadResult r = rt::probe(a, L"whirl_t8.dll");
        CHECK(!r.ok && r.error == ERROR_PROC_NOT_FOUND && r.app_local && r.path == p);
        const dd::Shadow127 s = dd::inspect127(r.path, a);
        CHECK(s.found_at == p);
        unload(a + L"\\whirl_fake_dep.dll");
    }
    // the import next to the runtime in the executable's folder is used (the ROCm 10.1 case: runtime + kpack app-local)
    {
        const std::wstring a = t.dir(L"t9"), p = a + L"\\whirl_t9.dll";
        CHECK(t.copy(withdep, p) && t.copy(dep, a + L"\\whirl_fake_dep.dll"));
        rt::Api api;
        const rt::LoadResult r = rt::probeTable(a, L"whirl_t9.dll", &api, true);
        int n = 0;
        CHECK(r.ok && api.hipGetDeviceCount && api.hipGetDeviceCount(&n) == hipSuccess && n == 3);
        unload(p);
        unload(a + L"\\whirl_fake_dep.dll");
    }
}

// optional: this machine's real runtime through the process-wide load()
void testReal() {
    const rt::LoadResult& r = rt::load();
    std::printf("real runtime: ok=%d %s app_local=%d\n", r.ok ? 1 : 0, rt::describe(r).c_str(), r.app_local ? 1 : 0);
    CHECKD(r.ok, rt::failureText(r));
    if (!r.ok) return;
    CHECK(!r.version.empty());
    CHECK(&rt::rt() == &rt::g_api && allSlots(rt::g_api));
    int n = -1;
    CHECK(rt::rt().hipGetDeviceCount(&n) == hipSuccess && n >= 0);
    bool imports_kpack = false;
    for (const std::string& s : dd::peImports(r.path)) imports_kpack = imports_kpack || _stricmp(s.c_str(), "rocm_kpack.dll") == 0;
    const bool kpack_loaded = GetModuleHandleW(L"rocm_kpack.dll") != nullptr;
    std::printf("real runtime: imports rocm_kpack.dll=%d, rocm_kpack.dll loaded=%d\n", imports_kpack ? 1 : 0, kpack_loaded ? 1 : 0);
    if (!imports_kpack) CHECK(!kpack_loaded);
}

}  // namespace

int main() {
    testTable();
    testFakes(rt::exeDir() + L"\\hiprt_fakes");
    if (const char* e = std::getenv("WHIRL_TEST_REAL_HIPRT"); e && std::strcmp(e, "1") == 0) testReal();
    std::printf("hiprt tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
