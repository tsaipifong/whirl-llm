// Platform layer, process environment: the Windows implementation of platform/env.h.
// SPDX-License-Identifier: Apache-2.0
//
// arch025 PL1: the _dupenv_s reads of src/model/config.cpp (envGet, since N1 in src/config/env_snapshot.cpp),
// src/common/vram_limit.cpp, src/kernels_host/kernels_abi.cpp and src/model/tune.cpp (LOCALAPPDATA), the
// _putenv_s of src/server/server_main.cpp and the GetEnvironmentStringsW loop of src/cache/fingerprint.cpp,
// each with the calls it had in v0.2.4.

#include "platform/env.h"

#include <cstdlib>
#include <cwchar>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace whirl::platform::env {

std::optional<std::string> get(const std::string& name) {
    // _dupenv_s keeps MSVC quiet about getenv
    char* buf = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&buf, &len, name.c_str()) == 0 && buf != nullptr) {
        std::string v(buf);
        std::free(buf);
        return v;
    }
    return std::nullopt;
}

std::optional<std::string> getUtf8(const std::string& name) {
    // bug-lappdata: the wide table, as llama.cpp's common_get_path_from_env (_wgetenv); PRIOR_ART bug-lappdata
    std::wstring wname;
    if (!name.empty()) {
        const int nw = MultiByteToWideChar(CP_UTF8, 0, name.data(), static_cast<int>(name.size()), nullptr, 0);
        wname.resize(static_cast<std::size_t>(nw));
        MultiByteToWideChar(CP_UTF8, 0, name.data(), static_cast<int>(name.size()), wname.data(), nw);
    }
    wchar_t* buf = nullptr;
    std::size_t len = 0;
    if (_wdupenv_s(&buf, &len, wname.c_str()) != 0 || buf == nullptr) return std::nullopt;
    const std::size_t wl = wcslen(buf);
    std::string v;
    if (wl > 0) {
        const int n = WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(wl), nullptr, 0, nullptr, nullptr);
        v.resize(static_cast<std::size_t>(n));
        WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(wl), v.data(), n, nullptr, nullptr);
    }
    std::free(buf);
    return v;
}

int set(const char* name, const char* value) { return _putenv_s(name, value); }

std::vector<std::wstring> block() {
    std::vector<std::wstring> out;
    if (wchar_t* blk = GetEnvironmentStringsW()) {
        for (const wchar_t* p = blk; *p; p += wcslen(p) + 1) out.emplace_back(p);
        FreeEnvironmentStringsW(blk);
    }
    return out;
}

}  // namespace whirl::platform::env
