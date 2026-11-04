// Platform layer: the path length budget (PATH-256, v0.2.5.1 PLAN V1-22). See progress/pathlen256.DISCOVERY.md v3.1
// (2a budget, 2c refusals) and docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is path_budget_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
//
// Sir Pifeng 2026-11-03: every path shorter than 256 (「必須重構目錄名稱與檔案名稱，確保路徑長度短於256」), the install
// folder at most 150 (17:44 「安裝資料夾 ≤150」), an over-long --kv-ssd-dir refused (「八、改」).
//
// Every limit WHIRL puts on a path is in this file, each with its unit. A limit has exactly one unit, chosen by the API
// that reads the path:
//   - UTF-8 bytes: ONLY the install folder (and so the exe path). The AMD runtime reads the exe path with
//     GetModuleFileNameA, which returns UTF-8 under the manifest's activeCodePage (res/whirl.manifest): an English
//     letter or digit is 1 byte, a Chinese character 3, a character outside the BMP 4.
//   - UTF-16 units: every path WHIRL opens itself (UTF-8 text widened, then the W API: models, --kv-ssd-dir, the files
//     under %LOCALAPPDATA%, ...). Windows measures those in UTF-16 code units: every letter or Chinese character is 1,
//     a character outside the BMP 2.
// A path is measured after GetFullPathNameW (a relative path is resolved against the working folder) and without a
// \\?\ or \\?\UNC\ prefix (\\?\UNC\srv\share -> \\srv\share). User files (models, --image, @file, --out, --log-file,
// debug outputs) have NO WHIRL limit: v0.2.5.1 opens every path v0.2.5 could; --kv-ssd-dir is the one user path
// refused up front.
// The release-zip and development-tree limits are checked by tools outside the product (release_check R7, the
// cpu_pathlen gate); they are listed here so every number has one home.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace whirl::platform::path_budget {

// ---- the limits -------------------------------------------------------------------------------------------------
inline constexpr std::size_t kPathMax = 255;           // UTF-16 units: the longest path WHIRL makes (shorter than 256)
inline constexpr std::size_t kInstallDirMax = 150;     // UTF-8 bytes: the folder of whirl.exe / whirl-server.exe / whirl-vis.exe
inline constexpr std::size_t kLocalAppDataMax = 187;   // UTF-16 units: %LOCALAPPDATA% = kPathMax - kLadPartMax
inline constexpr std::size_t kKvSsdDirMax = 217;       // UTF-16 units: --kv-ssd-dir / WHIRL_KV_SSD_DIR = kPathMax - kKvEntryName
inline constexpr std::size_t kZipEntryMax = 104;       // bytes (entry names ASCII only): kPathMax - kInstallDirMax - 1 (release_check R7)
inline constexpr std::size_t kDevExeMax = 150;         // UTF-8 bytes: an .exe in a development tree (cpu_pathlen; the measured-safe exe path)
inline constexpr std::size_t kDevFileBelow = 240;      // UTF-16 units: any other file of a development tree stays below this (cpu_pathlen)

// ---- the fixed parts WHIRL appends to a base (ASCII: units = bytes) ----------------------------------------------
inline constexpr std::size_t kTuneSuffixMax = 10;      // device::profile().tune_cache_suffix ("-gfx1151r2")
inline constexpr std::size_t kTune5NameMax = 61;       // "tune5-" 6 + 24 + "-" 1 + 16 hex + suffix 10 + ".txt" 4
inline constexpr std::size_t kLadPartMax = 68;         // "\whirl\" 7 + kTune5NameMax: the longest WHIRL path under %LOCALAPPDATA%
inline constexpr std::size_t kKvEntryName = 38;        // "\<16 hex>-<16 hex>.wkv" (an SSD tier entry, src/tier/kv_tier.cpp)
inline constexpr std::size_t kKvLockName = 11;         // "\whirl.lock" (the SSD tier's folder lock)
inline constexpr std::size_t kServerLogPart = 17;      // "\whirl\server.log" (under %LOCALAPPDATA%)
inline constexpr std::size_t kPinnedPartMax = 28;      // "\whirl\pinned\" 14 + a process id (<= 10 digits) + ".txt" 4
inline constexpr std::size_t kCwdLogName = 17;         // "\whirl-server.log" (%LOCALAPPDATA% unset: in the working folder)

static_assert(kLocalAppDataMax + kLadPartMax == kPathMax);
static_assert(kKvSsdDirMax + kKvEntryName == kPathMax);
static_assert(kInstallDirMax + 1 + kZipEntryMax == kPathMax);
static_assert(kLadPartMax >= kServerLogPart && kLadPartMax >= kPinnedPartMax && kLadPartMax >= 14 + kKvEntryName);

// ---- measuring --------------------------------------------------------------------------------------------------
// UTF-8 bytes of UTF-16 text (what WideCharToMultiByte(CP_UTF8) makes: 1 / 2 / 3 bytes per BMP character, 4 per
// surrogate pair; an unpaired surrogate becomes U+FFFD, 3 bytes).
std::size_t utf8Bytes(std::wstring_view w);
// UTF-16 units of UTF-8 text (what MultiByteToWideChar(CP_UTF8) makes: 1 per character in the BMP, 2 outside it; an
// invalid byte becomes U+FFFD, 1 unit).
std::size_t utf16Units(std::string_view utf8);
// The path without a long-path prefix: \\?\C:\x -> C:\x, \\?\UNC\srv\share\x -> \\srv\share\x; other text unchanged.
std::wstring stripLongPrefix(std::wstring_view p);
// GetFullPathNameW (a relative path against this process's working folder), then stripLongPrefix; "" when the call
// fails or p is empty.
std::wstring fullPath(const std::wstring& p);
// The folder of a full path: C:\a\b.exe -> C:\a, C:\b.exe -> C:\, \\srv\share\b.exe -> \\srv\share; "" without a separator.
std::wstring folderOf(std::wstring_view full);

// ---- refusals and warnings: an English line, then a Traditional Chinese line; each states its unit ------------
// The install folder of exe_path (as GetModuleFileNameW returns it; may start with \\?\): the refusal text when the
// folder is longer than kInstallDirMax UTF-8 bytes, else nullopt. prog: "whirl", "whirl-server" or "whirl-vis".
std::optional<std::string> installDirRefusal(const std::wstring& exe_path, std::string_view prog);
// The same for this process (platform::proc::exePath()); nullopt also when the path cannot be read. The three
// programs call it as the first action of their main function, before anything can load the AMD runtime
// (src/cli/main.cpp, src/server/whirl_server.cpp, src/vision/vis_main.cpp), and exit 2 with this text.
std::optional<std::string> installDirRefusal(std::string_view prog);
// --kv-ssd-dir / WHIRL_KV_SSD_DIR (name: how the user gave it): the refusal text when the folder's full path is longer
// than kKvSsdDirMax UTF-16 units, else nullopt. dir: UTF-8.
std::optional<std::string> kvSsdDirRefusal(std::string_view name, const std::string& dir);
// %LOCALAPPDATA% (UTF-8, as platform::paths reads it) as the base of WHIRL's own files: the warning text when its full
// path is longer than kLocalAppDataMax UTF-16 units (WHIRL then keeps no file there), else nullopt.
std::optional<std::string> localAppDataWarning(const std::string& lad);
// The full path of <cwd>\whirl-server.log fits kPathMax UTF-16 units (the server log when %LOCALAPPDATA% is unset).
bool cwdLogFits(const std::wstring& cwd_full);

}  // namespace whirl::platform::path_budget
