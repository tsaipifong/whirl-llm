// Platform layer, well-known folders: the Windows implementation of platform/paths.h.
// SPDX-License-Identifier: Apache-2.0
//
// arch025 PL1: LOCALAPPDATA of src/model/tune.cpp (_dupenv_s) and src/server/server_main.cpp (std::getenv: the same
// CRT narrow table) and the path text built from it, character for character as in v0.2.4. bug-lappdata (2026-10-26):
// LOCALAPPDATA is read from the wide table and converted to UTF-8 (the same bytes for ASCII values).

#include "platform/paths.h"

#include "platform/env.h"

namespace whirl::platform::paths {

// bug-lappdata: UTF-16 -> UTF-8 (was env::get, the ANSI code page)
// bug-emptyrootpaths: set but empty (a parent's environment block) is unset, as llama.cpp fs_get_cache_directory
// (.empty()); "" + "\\whirl\\..." was a path at the root of the current drive
std::optional<std::string> localAppData() {
    std::optional<std::string> v = env::getUtf8("LOCALAPPDATA");
    if (v && v->empty()) return std::nullopt;
    return v;
}

std::string whirlDir(const std::string& base) { return base + "\\whirl"; }

std::string serverLogPath(const std::optional<std::string>& base) {
    return base && !base->empty() ? *base + "\\whirl\\server.log" : std::string("whirl-server.log");
}

std::string kvCacheDir(const std::string& base) { return base + "\\whirl\\kvcache"; }

std::string tuneCachePath(const std::string& base, const std::string& stem) { return whirlDir(base) + "\\tune4-" + stem; }

std::vector<std::string> tuneReadPaths4(const std::string& base, const std::string& stem) {
    std::vector<std::string> out;
    // tables written by the research prototype are read as well (same format)
    for (const char* d : {"\\whirl\\", "\\whirl-r9700\\"}) out.push_back(base + d + "tune4-" + stem);
    return out;
}

std::vector<std::string> tuneReadPathsOld(const std::string& base, const std::string& stem) {
    std::vector<std::string> out;
    for (const char* pfx : {"tune3-", "tune-"})
        for (const char* d : {"\\whirl\\", "\\whirl-r9700\\"}) out.push_back(base + d + pfx + stem);
    return out;
}

// PATH-256 test-first stubs (v0251-patha commit 1): declared in paths.h, no behaviour yet; the next commit implements them
std::string tune5Name(const std::string&, std::uint64_t, const std::string&) { return std::string(); }
std::string tune5Path(const std::string&, const std::string&) { return std::string(); }
std::string tune5LedgerPath(const std::string&) { return std::string(); }
std::uint64_t fnv1a64(std::string_view) { return 0; }
std::optional<std::string> localAppDataWarning() { return std::nullopt; }
std::optional<std::string> serverLogDefault() { return serverLogPath(localAppData()); }

}  // namespace whirl::platform::paths
