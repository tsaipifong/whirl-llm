// Platform layer: process environment (arch025 PL1). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is env_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
#pragma once

namespace whirl::platform::env {

// PL1 moves here (call order and returned strings unchanged; fingerprint string bitwise identical):
//   - the full environment block read of the SSD fingerprint: GetEnvironmentStringsW / FreeEnvironmentStringsW
//     (src/cache/fingerprint.cpp)
//   - the single-variable reads with _dupenv_s (src/model/config.cpp, src/common/vram_limit.cpp,
//     src/kernels_host/kernels_abi.cpp)
//   - the one environment write: _putenv_s("GPU_RESOURCE_CACHE_SIZE", ...) before the HIP runtime loads
//     (src/server/server_main.cpp)
// Reads in src/cli/main.cpp and src/server/server_main.cpp stay counted by env_read_check (C1a / C2), not here.

// True when the variable exists in the process environment (even when empty). First PL1 step: used by the test-only
// NullBackend leak ledger (WHIRL_LEAKCHECK_QUIET, bug-nullbackendenv).
bool isSet(const char* name);

}  // namespace whirl::platform::env
