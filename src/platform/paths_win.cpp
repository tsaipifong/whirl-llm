// Platform layer, well-known folders: the Windows implementation of platform/paths.h.
// SPDX-License-Identifier: Apache-2.0
//
// arch025 PL1: LOCALAPPDATA of src/model/tune.cpp (_dupenv_s) and src/server/server_main.cpp (std::getenv: the same
// CRT narrow table) and the path text built from it, character for character as in v0.2.4.

#include "platform/paths.h"

#include "platform/env.h"

namespace whirl::platform::paths {

std::optional<std::string> localAppData() { return env::get("LOCALAPPDATA"); }

std::string whirlDir(const std::string& base) { return base + "\\whirl"; }

std::string serverLogPath(const std::optional<std::string>& base) {
    return base ? *base + "\\whirl\\server.log" : std::string("whirl-server.log");
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

}  // namespace whirl::platform::paths
