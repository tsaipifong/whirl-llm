// Platform layer: well-known folders and path text (arch025 PL1). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is paths_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace whirl::platform::paths {

// %LOCALAPPDATA% as the CRT's narrow environment holds it (env::get; read at the call, nullopt when unset).
// Not SHGetKnownFolderPath and not _wgetenv: v0.2.4 read the narrow variable, and every path below must stay byte
// for byte the same (a changed tune path re-tunes, a changed kvcache path misses the SSD tier; PRIOR_ART PL1).
std::optional<std::string> localAppData();

// Path text, pure functions (no file system access). `base` is localAppData()'s value. Pinned by the table test
// testPlatformPaths (tests/unit/unit_tests.cpp) with the literal strings of v0.2.4.
std::string whirlDir(const std::string& base);                          // <base>\whirl
std::string serverLogPath(const std::optional<std::string>& base);      // <base>\whirl\server.log, else whirl-server.log
std::string kvCacheDir(const std::string& base);                        // <base>\whirl\kvcache
// tune cache (src/model/tune.cpp): stem = "<model file>-<size><suffix>.txt"
std::string tuneCachePath(const std::string& base, const std::string& stem);             // <base>\whirl\tune4-<stem>
std::vector<std::string> tuneReadPaths4(const std::string& base, const std::string& stem);  // whirl, whirl-r9700 tune4-
std::vector<std::string> tuneReadPathsOld(const std::string& base, const std::string& stem);  // tune3- / tune- x 2 dirs

}  // namespace whirl::platform::paths
