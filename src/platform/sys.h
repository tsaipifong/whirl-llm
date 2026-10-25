// Platform layer: machine facts (arch025 PL4). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is sys_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
#pragma once

namespace whirl::platform::sys {

// PL4 moves here (same numbers, same rounding):
//   - physical RAM: GlobalMemoryStatusEx (src/server/server_main.cpp; the figure src/tier/ram_size.h consumes)
// Command-line conversion of wmain (UTF-16 argv -> UTF-8: WideCharToMultiByte / MultiByteToWideChar in
// src/common/util.cpp) is "args", also PL4; it lives here until a third use asks for its own header.

}  // namespace whirl::platform::sys
