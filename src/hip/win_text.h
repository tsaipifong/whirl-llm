// Win32 text / file helpers of the HIP runtime loader (src/hip/runtime_loader.cpp) and the DLL load-failure diagnosis
// (src/release/dll_deps.cpp): one copy (arch025-N6, lane 1 leg 11; both files had the same text). Header-only and inline
// because whirl_hiprt links no other WHIRL library (its tests link only it) and dll_deps.cpp is built into
// whirl-hiprt-tests too; the platform layer's whirl::narrow (src/common/util.cpp) is in the core library.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <string>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace whirl::hip::win {

// UTF-8 of a UTF-16 string (WideCharToMultiByte CP_UTF8, flags 0: an unpaired surrogate becomes U+FFFD); "" for ""
inline std::string narrowUtf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<std::size_t>(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

// p names an existing file (not a directory)
inline bool fileExists(const std::wstring& p) {
    const DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

}  // namespace whirl::hip::win
