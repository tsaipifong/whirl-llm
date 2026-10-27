// Platform layer: UTF-8 <-> UTF-16 conversion of command lines and paths (arch025 PL4). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is args_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
//
// One function per Win32 call. whirl::widen / whirl::narrow (src/common/util.cpp) keep their "ask for the length,
// then convert" logic and call these twice each; whirl::narrowArgs converts wmain's argv for both executables
// (src/cli/main.cpp, src/server/whirl_server.cpp). Flags stay 0 (no MB_ERR_INVALID_CHARS / WC_ERR_INVALID_CHARS:
// invalid input is replaced, as in v0.2.4); the strict variant of the tier files is fs::utf8ToWideStrict (PL2).
#pragma once

namespace whirl::platform::args {

// MultiByteToWideChar(CP_UTF8, 0, s, n, out, cap): the number of UTF-16 units written (out = nullptr, cap = 0: needed)
int wideFromUtf8(const char* s, int n, wchar_t* out, int cap);
// WideCharToMultiByte(CP_UTF8, 0, s, n, out, cap, nullptr, nullptr): the number of bytes written (or needed)
int utf8FromWide(const wchar_t* s, int n, char* out, int cap);

}  // namespace whirl::platform::args
