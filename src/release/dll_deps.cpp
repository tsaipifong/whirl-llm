// Load-failure diagnosis for a DLL (see dll_deps.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL.

#include "release/dll_deps.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace whirl::app::dlldeps {

namespace {

std::string narrowUtf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<std::size_t>(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

std::wstring widenAscii(const std::string& s) { return std::wstring(s.begin(), s.end()); }

std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

bool isApiSet(const std::string& dll) {
    const std::string l = lower(dll);
    return l.rfind("api-ms-", 0) == 0 || l.rfind("ext-ms-", 0) == 0;
}

template <class T>
bool rd(const std::vector<unsigned char>& b, std::size_t off, T& v) {
    if (off > b.size() || b.size() - off < sizeof(T)) return false;
    std::memcpy(&v, b.data() + off, sizeof(T));
    return true;
}

void addUnique(std::vector<std::string>& v, const std::string& s) {
    const std::string l = lower(s);
    for (const std::string& x : v)
        if (lower(x) == l) return;
    v.push_back(s);
}

// true when `name` loads; with `loaded_err` the GetLastError of a failure
bool tryLoad(const std::wstring& name, DWORD* err) {
    const UINT old_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);
    HMODULE h = LoadLibraryW(name.c_str());
    *err = h == nullptr ? GetLastError() : 0;
    SetErrorMode(old_mode);
    return h != nullptr;  // kept loaded: the process exits right after the diagnosis
}

}  // namespace

std::vector<std::string> peImports(const std::wstring& path) {
    std::vector<std::string> out;
    std::ifstream f(path, std::ios::binary);
    if (!f) return out;
    const std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::uint16_t mz = 0;
    std::uint32_t pe = 0, sig = 0;
    if (!rd(b, 0, mz) || mz != 0x5a4d || !rd(b, 0x3c, pe) || !rd(b, pe, sig) || sig != 0x00004550) return out;
    std::uint16_t nsec = 0, opt_size = 0, magic = 0;
    if (!rd(b, pe + 6, nsec) || !rd(b, pe + 20, opt_size)) return out;
    const std::size_t opt = static_cast<std::size_t>(pe) + 24;
    if (!rd(b, opt, magic) || magic != 0x20b) return out;  // PE32+ only
    struct Sec {
        std::uint32_t va, vs, raw, rawsize;
    };
    std::vector<Sec> secs;
    for (std::uint16_t i = 0; i < nsec; ++i) {
        const std::size_t s = opt + opt_size + 40u * i;
        Sec x{};
        if (!rd(b, s + 8, x.vs) || !rd(b, s + 12, x.va) || !rd(b, s + 16, x.rawsize) || !rd(b, s + 20, x.raw)) return out;
        secs.push_back(x);
    }
    auto rva2off = [&](std::uint32_t rva, std::size_t& off) {
        for (const Sec& s : secs) {
            const std::uint32_t span = std::max(s.vs, s.rawsize);
            if (rva >= s.va && rva - s.va < span) {
                off = static_cast<std::size_t>(rva - s.va) + s.raw;
                return off < b.size();
            }
        }
        return false;
    };
    auto cstr = [&](std::size_t off) {
        std::string s;
        while (off < b.size() && b[off] != 0 && s.size() < 260) s += static_cast<char>(b[off++]);
        return s;
    };
    // data directory 1: imports (20-byte descriptors, name RVA at +12);
    // 13: delay imports (32-byte descriptors, name RVA at +4)
    const struct {
        int idx;
        std::size_t size, name_at;
    } dirs[] = {{1, 20, 12}, {13, 32, 4}};
    for (const auto& d : dirs) {
        std::uint32_t rva = 0;
        if (!rd(b, opt + 112 + 8u * d.idx, rva) || rva == 0) continue;
        std::size_t off = 0;
        if (!rva2off(rva, off)) continue;
        for (int n = 0; n < 4096; ++n, off += d.size) {
            std::uint32_t nrva = 0;
            if (!rd(b, off + d.name_at, nrva) || nrva == 0) break;
            std::size_t noff = 0;
            if (!rva2off(nrva, noff)) break;
            const std::string name = cstr(noff);
            if (!name.empty()) addUnique(out, name);
        }
    }
    return out;
}

std::wstring findDll(const std::wstring& dll) {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = SearchPathW(nullptr, dll.c_str(), nullptr, static_cast<DWORD>(std::size(buf)), buf, nullptr);
    if (n == 0 || n >= std::size(buf)) return {};
    return std::wstring(buf, n);
}

Diagnosis diagnose(const std::wstring& dll) {
    Diagnosis d;
    const bool is_path = dll.find_first_of(L"\\/") != std::wstring::npos;
    if (is_path) {
        if (GetFileAttributesW(dll.c_str()) != INVALID_FILE_ATTRIBUTES) d.found_at = dll;
    } else {
        d.found_at = findDll(dll);
    }
    if (d.found_at.empty()) return d;
    for (const std::string& imp : peImports(d.found_at)) {
        if (isApiSet(imp)) continue;
        DWORD err = 0;
        if (tryLoad(widenAscii(imp), &err)) continue;
        const std::wstring at = findDll(widenAscii(imp));
        if (at.empty()) {
            addUnique(d.missing, imp);
            continue;
        }
        // present but not loadable: one level further down
        bool sub_missing = false;
        for (const std::string& sub : peImports(at)) {
            if (isApiSet(sub)) continue;
            DWORD e2 = 0;
            if (tryLoad(widenAscii(sub), &e2)) continue;
            if (findDll(widenAscii(sub)).empty()) {
                addUnique(d.missing, sub);
                sub_missing = true;
            }
        }
        if (!sub_missing) addUnique(d.unloadable, imp);
    }
    return d;
}

std::string message126(const std::string& dll_name, const Diagnosis& d, const char* driver_advice) {
    if (d.found_at.empty())
        return std::format("error: the AMD GPU runtime ({}) was not found, so WHIRL cannot use the GPU.\n"
                           "  It is installed by the AMD graphics driver.\n{}",
                           dll_name, driver_advice);
    const std::string at = narrowUtf8(d.found_at);
    if (d.missing.empty() && d.unloadable.empty())
        return std::format("error: the AMD GPU runtime ({}) was found at {}, but it could not be loaded\n"
                           "  (a DLL it needs is missing), so WHIRL cannot use the GPU.\n{}",
                           dll_name, at, driver_advice);
    std::string list;
    for (const std::string& m : d.missing) list += (list.empty() ? "" : ", ") + m;
    for (const std::string& m : d.unloadable) list += (list.empty() ? "" : ", ") + m + " (present, does not load)";
    std::string s = std::format("error: the AMD GPU runtime ({}) was found at {},\n"
                                "  but it cannot be loaded because a DLL it needs is missing: {}.\n",
                                dll_name, at, list);
    bool kpack = false, vcrt = false;
    for (const std::string& m : d.missing) {
        const std::string l = lower(m);
        if (l == "rocm_kpack.dll") kpack = true;
        if (l.rfind("vcruntime140", 0) == 0 || l.rfind("msvcp140", 0) == 0) vcrt = true;
    }
    if (kpack)
        s += "  rocm_kpack.dll is part of AMD's ROCm runtime; the AMD Software: Adrenalin Edition for ROCm\n"
             "  driver installs a GPU runtime that needs it but does not install it. WHIRL ships it in its zip:\n"
             "  put rocm_kpack.dll from the WHIRL zip next to whirl.exe and whirl-server.exe, or unzip WHIRL\n"
             "  again into a new folder.\n";
    if (vcrt)
        s += "  These are the Microsoft Visual C++ runtime. WHIRL ships them in its zip: put msvcp140.dll,\n"
             "  vcruntime140.dll and vcruntime140_1.dll from the WHIRL zip next to whirl.exe and\n"
             "  whirl-server.exe (or unzip WHIRL again into a new folder), or install the Visual C++\n"
             "  Redistributable (x64) from https://aka.ms/vs/17/release/vc_redist.x64.exe, and run the\n"
             "  command again.\n";
    if (!kpack && !vcrt) s += driver_advice;
    return s;
}

namespace {

bool fileExists(const std::wstring& p) {
    const DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

std::wstring dirOf(const std::wstring& path) {
    const std::size_t k = path.find_last_of(L"\\/");
    return k == std::wstring::npos ? std::wstring() : path.substr(0, k);
}

bool sameDir(const std::wstring& a, const std::wstring& b) {
    if (a.empty() || b.empty()) return false;
    wchar_t fa[MAX_PATH * 2], fb[MAX_PATH * 2];
    const DWORD na = GetFullPathNameW(a.c_str(), static_cast<DWORD>(std::size(fa)), fa, nullptr);
    const DWORD nb = GetFullPathNameW(b.c_str(), static_cast<DWORD>(std::size(fb)), fb, nullptr);
    if (na == 0 || nb == 0 || na >= std::size(fa) || nb >= std::size(fb)) return _wcsicmp(a.c_str(), b.c_str()) == 0;
    std::wstring x(fa, na), y(fb, nb);
    while (!x.empty() && (x.back() == L'\\' || x.back() == L'/')) x.pop_back();
    while (!y.empty() && (y.back() == L'\\' || y.back() == L'/')) y.pop_back();
    return _wcsicmp(x.c_str(), y.c_str()) == 0;
}

}  // namespace

std::wstring exeDir() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(std::size(buf)));
    if (n == 0 || n >= std::size(buf)) return {};
    return dirOf(std::wstring(buf, n));
}

Shadow127 inspect127(const std::wstring& dll, const std::wstring& app_dir) {
    Shadow127 s;
    s.app_dir = app_dir;
    while (!s.app_dir.empty() && (s.app_dir.back() == L'\\' || s.app_dir.back() == L'/')) s.app_dir.pop_back();
    const bool is_path = dll.find_first_of(L"\\/") != std::wstring::npos;
    s.found_at = is_path ? (fileExists(dll) ? dll : std::wstring()) : findDll(dll);
    if (!s.app_dir.empty())
        for (const char* b : k_bundled_dlls)
            if (fileExists(s.app_dir + L"\\" + widenAscii(b))) s.bundled.emplace_back(b);
    // the driver's own rocm_kpack.dll, if any: next to its runtime, or in System32
    std::vector<std::wstring> dirs;
    if (!s.found_at.empty()) dirs.push_back(dirOf(s.found_at));
    wchar_t sys[MAX_PATH];
    const UINT n = GetSystemDirectoryW(sys, MAX_PATH);
    if (n > 0 && n < MAX_PATH) dirs.emplace_back(sys, n);
    for (const std::wstring& d : dirs) {
        if (d.empty() || sameDir(d, s.app_dir)) continue;
        const std::wstring k = d + L"\\rocm_kpack.dll";
        if (fileExists(k)) {
            s.other_kpack = k;
            break;
        }
    }
    return s;
}

std::string message127(const std::string& dll_name, const Shadow127& s) {
    auto has = [&](const char* name) {
        for (const std::string& b : s.bundled)
            if (lower(b) == name) return true;
        return false;
    };
    const bool kpack = has("rocm_kpack.dll");
    std::string vc_list;
    for (const std::string& b : s.bundled)
        if (lower(b) != "rocm_kpack.dll") vc_list += (vc_list.empty() ? "" : ", ") + b;
    const std::string dir = s.app_dir.empty() ? std::string("the WHIRL folder") : narrowUtf8(s.app_dir);
    const std::string at = s.found_at.empty() ? std::string() : " at " + narrowUtf8(s.found_at);

    std::string m = std::format("error: the AMD GPU runtime ({}){} could not be loaded: Windows error 127, a function it\n"
                                "  needs is missing from one of the DLLs it uses. WHIRL cannot use the GPU.\n",
                                dll_name, at);
    const char* releases = "https://github.com/tsaipifong/whirl-llm/releases";
    if (kpack) {
        m += std::format("  The likely cause: the rocm_kpack.dll that comes with WHIRL ({}\\rocm_kpack.dll) is not the\n"
                         "  version your AMD driver's GPU runtime expects. WHIRL ships it for the Adrenalin Edition for\n"
                         "  ROCm 10.1 driver, whose runtime needs it but does not install it; it is found before the\n"
                         "  driver's own files because it sits next to whirl.exe.\n",
                         dir);
        if (!s.other_kpack.empty())
            m += std::format("  Your driver has its own copy: {}\n", narrowUtf8(s.other_kpack));
        m += std::format("  To fix it, do one of these and run the command again:\n"
                         "    - rename or remove rocm_kpack.dll in {}\n"
                         "    - or use the WHIRL release that matches your driver: {}\n",
                         dir, releases);
        if (!vc_list.empty())
            m += std::format("  If the error stays, also rename {} in that folder;\n"
                             "  then the Visual C++ runtime installed in Windows is used instead of WHIRL's copy.\n",
                             vc_list);
    } else if (!vc_list.empty()) {
        m += std::format("  The likely cause: the Visual C++ runtime that comes with WHIRL ({} in {})\n"
                         "  is older than what your AMD driver's files need; it is found first because it sits next to\n"
                         "  whirl.exe. To fix it, do one of these and run the command again:\n"
                         "    - rename or remove {} in {} (the Visual C++ runtime installed in\n"
                         "      Windows is then used; install the Visual C++ Redistributable (x64) from\n"
                         "      https://aka.ms/vs/17/release/vc_redist.x64.exe if it is missing)\n"
                         "    - or use the WHIRL release that matches your driver: {}\n",
                         vc_list, dir, vc_list, dir, releases);
    } else {
        m += std::format("  None of the DLLs WHIRL ships is in {}, so another copy of a DLL the runtime uses\n"
                         "  is probably being found first. Check that no other program's rocm_kpack.dll, msvcp140.dll or\n"
                         "  vcruntime140*.dll is in that folder or on PATH,\n"
                         "  or use the WHIRL release that matches your driver: {}\n",
                         dir, releases);
    }
    return m;
}

}  // namespace whirl::app::dlldeps
