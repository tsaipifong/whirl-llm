// Platform layer: machine facts (arch025 PL4). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is sys_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
// The command-line conversion of wmain (UTF-16 <-> UTF-8) is platform/args.h.
#pragma once

#include <cstdint>

namespace whirl::platform::sys {

// GlobalMemoryStatusEx (dwLength = sizeof): ullTotalPhys / ullAvailPhys in bytes; false when the call fails (outputs
// untouched). The server's RAM-tier sizing (src/server/server_main.cpp -> tier::RamTierInput, src/tier/ram_size.h).
bool memoryStatus(std::uint64_t* total_phys, std::uint64_t* avail_phys);

}  // namespace whirl::platform::sys
