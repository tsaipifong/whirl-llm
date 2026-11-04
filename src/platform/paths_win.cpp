// Platform layer, well-known folders: the Windows implementation of platform/paths.h.
// SPDX-License-Identifier: Apache-2.0
//
// arch025 PL1: LOCALAPPDATA of src/model/tune.cpp (_dupenv_s) and src/server/server_main.cpp (std::getenv: the same
// CRT narrow table) and the path text built from it, character for character as in v0.2.4. bug-lappdata (2026-10-26):
// LOCALAPPDATA is read from the wide table and converted to UTF-8 (the same bytes for ASCII values).

#include "platform/paths.h"

#include "platform/env.h"
#include "platform/path_budget.h"

#include <cstdio>

namespace whirl::platform::paths {

namespace {
// bug-lappdata: UTF-16 -> UTF-8 (was env::get, the ANSI code page)
// bug-emptyrootpaths: set but empty (a parent's environment block) is unset, as llama.cpp fs_get_cache_directory
// (.empty()); "" + "\\whirl\\..." was a path at the root of the current drive
std::optional<std::string> rawLocalAppData() {
    std::optional<std::string> v = env::getUtf8("LOCALAPPDATA");
    if (v && v->empty()) return std::nullopt;
    return v;
}
}  // namespace

// PATH-256 (2c #3): a LOCALAPPDATA over path_budget::kLocalAppDataMax UTF-16 units (full path) is not a base for WHIRL's
// files (every caller treats nullopt as "no folder": no tune cache, no default SSD folder, no pinned note)
std::optional<std::string> localAppData() {
    std::optional<std::string> v = rawLocalAppData();
    if (v && path_budget::localAppDataWarning(*v)) return std::nullopt;
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

// ---- PATH-256 (v0.2.5.1 PLAN V1-22, pathlen256.DISCOVERY.md v3.1 2b-1, 2c #3 / #8) ----
std::uint64_t fnv1a64(std::string_view bytes) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (const unsigned char c : bytes) {
        h ^= c;
        h *= 0x100000001b3ull;
    }
    return h;
}

std::string tune5Name(const std::string& name, std::uint64_t size, const std::string& suffix) {
    // p24: the first 24 code points; an ASCII letter / digit / . _ - lower-cased, anything else (a multi-byte UTF-8
    // character, a stray byte) one '_'
    std::string p24;
    std::size_t i = 0;
    for (int cps = 0; cps < 24 && i < name.size(); ++cps) {
        const unsigned char c = static_cast<unsigned char>(name[i]);
        std::size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        if (len > name.size() - i) len = name.size() - i;
        const bool keep = len == 1 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-');
        p24 += keep ? static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c) : '_';
        i += len;
    }
    std::string key = name;
    key.push_back('\0');
    key += std::to_string(size);
    char h16[17];
    std::snprintf(h16, sizeof h16, "%016llx", static_cast<unsigned long long>(fnv1a64(key)));
    return "tune5-" + p24 + "-" + h16 + suffix + ".txt";
}

std::string tune5Path(const std::string& base, const std::string& tune5_name) { return whirlDir(base) + "\\" + tune5_name; }

std::string tune5LedgerPath(const std::string& base) { return whirlDir(base) + "\\tune5-imported.txt"; }

std::optional<std::string> localAppDataWarning() {
    const std::optional<std::string> v = rawLocalAppData();
    return v ? path_budget::localAppDataWarning(*v) : std::nullopt;
}

std::optional<std::string> serverLogDefault() {
    const std::optional<std::string> v = rawLocalAppData();
    if (v) {
        if (path_budget::localAppDataWarning(*v)) return std::nullopt;  // 2c #3: the console only
        return serverLogPath(v);
    }
    // 2c #8: unset -> <working folder>\whirl-server.log, while that fits kPathMax
    if (!path_budget::cwdLogFits(path_budget::fullPath(L"."))) return std::nullopt;
    return serverLogPath(std::nullopt);
}

}  // namespace whirl::platform::paths
