// Platform layer: console (arch025 PL1). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is console_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
#pragma once

namespace whirl::platform::console {

inline constexpr unsigned kCodePageUtf8 = 65001;  // CP_UTF8 (static_assert in console_win.cpp)

enum class Stream { out, err };

// GetStdHandle(STD_OUTPUT_HANDLE / STD_ERROR_HANDLE) is a console (GetConsoleMode succeeds).
bool isConsole(Stream s);
// GetConsoleOutputCP / SetConsoleOutputCP (returns its BOOL != 0).
unsigned outputCodePage();
bool setOutputCodePage(unsigned cp);
// _setmode(_fileno(stdout), _O_BINARY): the CLI writes UTF-8 bytes unchanged (no LF -> CRLF). Result ignored, as before.
void setStdoutBinary();

// The saved-code-page / atexit-restore logic stays in the caller (src/release/release.cpp initConsole).
// Console Ctrl+C handling (SetConsoleCtrlHandler) is proc (PL4), not here.

}  // namespace whirl::platform::console
