// Platform layer: well-known folders and path text (arch025 PL1). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is paths_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::platform::paths {

// %LOCALAPPDATA% as UTF-8 (env::getUtf8: the CRT's wide environment, read at the call; nullopt when unset AND when set
// but empty -- bug-emptyrootpaths: "" + "\whirl\..." was a drive-root path; llama.cpp treats empty as unset). Every caller hands the path text to widen() (UTF-8), so a non-ASCII user folder now reaches the
// file system unchanged (bug-lappdata; v0.2.4 read the narrow table, ANSI code page). ASCII values: byte for byte the
// old read, so no path of an ASCII user changes (a changed tune path re-tunes, a changed kvcache path misses the SSD
// tier). Still not SHGetKnownFolderPath: tests and users point LOCALAPPDATA at another folder (PRIOR_ART PL1).
std::optional<std::string> localAppData();

// Path text, pure functions (no file system access). `base` is localAppData()'s value. Pinned by the table test
// testPlatformPaths (tests/unit/unit_tests.cpp) with the literal strings of v0.2.4.
std::string whirlDir(const std::string& base);                          // <base>\whirl
std::string serverLogPath(const std::optional<std::string>& base);      // <base>\whirl\server.log, else (unset or "") whirl-server.log
std::string kvCacheDir(const std::string& base);                        // <base>\whirl\kvcache
// tune cache (src/model/tune.cpp): stem = "<model file>-<size><suffix>.txt"
std::string tuneCachePath(const std::string& base, const std::string& stem);             // <base>\whirl\tune4-<stem>
std::vector<std::string> tuneReadPaths4(const std::string& base, const std::string& stem);  // whirl, whirl-r9700 tune4-
std::vector<std::string> tuneReadPathsOld(const std::string& base, const std::string& stem);  // tune3- / tune- x 2 dirs

// ---- PATH-256 (v0.2.5.1 PLAN V1-22, progress/pathlen256.DISCOVERY.md v3.1 2b-1, 2c #3 / #8) ----
// The tune cache file name of v0.2.5.1: "tune5-<p24>-<h16><suffix>.txt", at most path_budget::kTune5NameMax (61)
// characters whatever the model's name. name = the model file's name as the file system stores it (UTF-8; tune.cpp asks
// FindFirstFileW, the typed name when that fails), size = its size in bytes, suffix = device::profile().tune_cache_suffix.
// p24 = the first 24 characters (code points) of name, each one outside [A-Za-z0-9._-] replaced by '_', lower-cased
// (ASCII, for people); h16 = fnv1a64(name bytes, a NUL byte, size in decimal) as 16 lower-case hex digits (the key: the
// same name and size as the tune4 name of v0.2.5).
std::string tune5Name(const std::string& name, std::uint64_t size, const std::string& suffix);
std::string tune5Path(const std::string& base, const std::string& tune5_name);  // <base>\whirl\<tune5 name>
std::string tune5LedgerPath(const std::string& base);                          // <base>\whirl\tune5-imported.txt
// FNV-1a 64 (offset basis 0xcbf29ce484222325, prime 0x100000001b3) of the bytes
std::uint64_t fnv1a64(std::string_view bytes);
// localAppData() returns nullopt also when LOCALAPPDATA is longer than path_budget::kLocalAppDataMax UTF-16 units (full
// path): WHIRL then keeps no file there (no tune cache, no default SSD folder, no pinned note; the server log goes to
// the console only). This is the warning for that case (English + Traditional Chinese line), else nullopt.
std::optional<std::string> localAppDataWarning();
// The server's default log file (no --log-file): <LOCALAPPDATA>\whirl\server.log; LOCALAPPDATA unset or empty:
// "whirl-server.log" in the working folder (as v0.2.4); nullopt = no log file, the console only: LOCALAPPDATA set but
// over the budget, or unset with <working folder>\whirl-server.log longer than path_budget::kPathMax UTF-16 units.
std::optional<std::string> serverLogDefault();

}  // namespace whirl::platform::paths
