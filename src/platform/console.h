// Platform layer: console (arch025 PL1). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is console_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
#pragma once

namespace whirl::platform::console {

// PL1 moves here (same code page and console mode before / after, same restore order at exit):
//   - UTF-8 output code page and its restore: GetConsoleOutputCP / SetConsoleOutputCP, GetStdHandle, GetConsoleMode
//     (src/release/release.cpp)
//   - binary stdout of the CLI: _setmode(_fileno(stdout), _O_BINARY) (src/cli/main.cpp, <io.h>)
// Console Ctrl+C handling (SetConsoleCtrlHandler) is proc (PL4), not here.

}  // namespace whirl::platform::console
