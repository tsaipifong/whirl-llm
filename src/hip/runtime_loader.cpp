// HIP runtime loader (see include/whirl/hip_runtime_loader.h and src/hip/hip_api.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL from the Win32 documentation (LoadLibraryExW,
// GetProcAddress, SearchPathW, VS_FIXEDFILEINFO).

#include "hip_api.h"

#include <cstring>
#include <format>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace whirl::hip::runtime {

// every table entry has exactly the SDK's signature (unevaluated: no link-time reference to amdhip64)
#define WHIRL_HIP_API_CHECK(ret, name, params)                                                       \
    static_assert(std::is_same_v<decltype(static_cast<ret(*) params>(&::name)), decltype(Api::name)>, \
                  #name ": table type differs from the HIP SDK declaration");
WHIRL_HIP_API_LIST(WHIRL_HIP_API_CHECK)
#undef WHIRL_HIP_API_CHECK

Api g_api;
std::atomic<bool> g_ready{false};

namespace {

#define WHIRL_HIP_API_NAME(ret, name, params) #name,
constexpr const char* k_names[] = {WHIRL_HIP_API_LIST(WHIRL_HIP_API_NAME)};
#undef WHIRL_HIP_API_NAME
static_assert(sizeof(Api) == sizeof(k_names) / sizeof(k_names[0]) * sizeof(void*), "Api must hold only the listed pointers");

std::string narrowUtf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<std::size_t>(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

std::wstring modulePath(HMODULE h) {
    std::wstring p(1024, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(h, p.data(), static_cast<DWORD>(p.size()));
        if (n == 0) return {};
        if (n < p.size()) {
            p.resize(n);
            return p;
        }
        p.resize(p.size() * 2);
    }
}

// file version from the module's own VERSIONINFO resource (no version.lib import)
std::string fileVersion(HMODULE h) {
    HRSRC r = FindResourceW(h, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(16) /* RT_VERSION */);
    if (!r) return {};
    HGLOBAL g = LoadResource(h, r);
    const DWORD size = SizeofResource(h, r);
    const auto* p = g ? static_cast<const unsigned char*>(LockResource(g)) : nullptr;
    if (!p) return {};
    constexpr DWORD k_sig = 0xFEEF04BD;
    for (DWORD off = 0; off + sizeof(VS_FIXEDFILEINFO) <= size; off += 4) {
        DWORD sig = 0;
        std::memcpy(&sig, p + off, sizeof sig);
        if (sig != k_sig) continue;
        VS_FIXEDFILEINFO fi;
        std::memcpy(&fi, p + off, sizeof fi);
        return std::format("{}.{}.{}.{}", HIWORD(fi.dwFileVersionMS), LOWORD(fi.dwFileVersionMS), HIWORD(fi.dwFileVersionLS),
                           LOWORD(fi.dwFileVersionLS));
    }
    return {};
}

bool fileExists(const std::wstring& p) {
    const DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

std::wstring searchPath(const std::wstring& dll) {
    wchar_t buf[MAX_PATH];
    const DWORD n = SearchPathW(nullptr, dll.c_str(), nullptr, MAX_PATH, buf, nullptr);
    return n > 0 && n < MAX_PATH ? std::wstring(buf, n) : std::wstring();
}

// resolves every entry; returns the first missing name, or nullptr
const char* resolve(HMODULE h, Api& api) {
    void** slot = reinterpret_cast<void**>(&api);
    for (std::size_t i = 0; i < std::size(k_names); ++i) {
        FARPROC f = GetProcAddress(h, k_names[i]);
        if (!f) return k_names[i];
        slot[i] = reinterpret_cast<void*>(f);
    }
    return nullptr;
}

}  // namespace

std::wstring exeDir() {
    std::wstring p = modulePath(nullptr);
    const std::size_t s = p.find_last_of(L"\\/");
    return s == std::wstring::npos ? std::wstring() : p.substr(0, s);
}

LoadResult probeTable(const std::wstring& app_dir, const std::wstring& dll, Api* out, bool keep) {
    LoadResult r;
    const UINT old_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);  // no system dialog
    HMODULE h = nullptr;
    // step 1: the executable's folder
    const std::wstring local = app_dir.empty() ? std::wstring() : app_dir + L"\\" + dll;
    if (!local.empty() && fileExists(local)) {
        r.app_local = true;
        r.path = local;
        h = LoadLibraryExW(local.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    } else {
        // step 2: the standard search order (System32, Windows, current folder, PATH)
        h = LoadLibraryExW(dll.c_str(), nullptr, 0);
    }
    r.error = h == nullptr ? GetLastError() : 0;
    SetErrorMode(old_mode);
    if (h == nullptr) {
        if (!r.app_local) r.path = searchPath(dll);
        return r;
    }
    if (!r.app_local) r.path = modulePath(h);
    r.version = fileVersion(h);
    Api api;
    if (const char* miss = resolve(h, api)) {
        r.missing_symbol = miss;
        FreeLibrary(h);
        return r;
    }
    r.ok = true;
    if (out) *out = api;
    if (!keep) FreeLibrary(h);
    return r;
}

LoadResult probe(const std::wstring& app_dir, const std::wstring& dll, bool keep) {
    return probeTable(app_dir, dll, nullptr, keep);
}

const LoadResult& load() {
    static std::once_flag once;
    static LoadResult result;
    std::call_once(once, [] {
        Api api;
        result = probeTable(exeDir(), k_dll, &api, true);  // kept loaded for the process lifetime
        if (result.ok) {
            g_api = api;
            g_ready.store(true, std::memory_order_release);
        }
    });
    return result;
}

const Api& loadOrThrow() {
    const LoadResult& r = load();
    if (!r.ok) throw std::runtime_error(failureText(r));
    return g_api;
}

std::string describe(const LoadResult& r) {
    std::string s = r.path.empty() ? std::string(k_dll_name) : narrowUtf8(r.path);
    if (!r.version.empty()) s += " (" + r.version + ")";
    return s;
}

std::string failureText(const LoadResult& r) {
    if (r.ok) return {};
    if (!r.missing_symbol.empty())
        return std::format("the AMD GPU runtime {} has no {} (the graphics driver is too old)", describe(r), r.missing_symbol);
    const std::string where = r.path.empty() ? std::string(k_dll_name) + " (not found)" : narrowUtf8(r.path);
    const char* why = r.error == ERROR_MOD_NOT_FOUND    ? ": the DLL or one of the DLLs it imports is missing"
                      : r.error == ERROR_PROC_NOT_FOUND ? ": a DLL it imports lacks a function it needs"
                                                        : "";
    return std::format("the AMD GPU runtime could not be loaded: {}, LoadLibrary error {}{}", where, r.error, why);
}

std::size_t symbolCount() { return std::size(k_names); }

const char* symbolName(std::size_t i) { return i < std::size(k_names) ? k_names[i] : nullptr; }

}  // namespace whirl::hip::runtime
