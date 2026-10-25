// Platform layer: well-known folders and path text (arch025 PL1). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is paths_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
#pragma once

namespace whirl::platform::paths {

// PL1 moves here (every resulting path string unchanged, pinned by a table test):
//   - the tune cache folder from LOCALAPPDATA (src/model/tune.cpp)
//   - the default cache / log folders of the server and the release front-end (src/server/server_main.cpp,
//     src/release/release.cpp)
// The executable's own folder (GetModuleFileNameW) is proc (PL4), not here.

}  // namespace whirl::platform::paths
