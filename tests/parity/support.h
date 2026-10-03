// Helpers for the C++ parity drivers: child processes (Windows, idle
// priority, captured stdout/stderr), SHA-256, small string utilities.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::parity {

struct ProcResult {
    int exit_code = -1;  // -1: could not start
    std::string out;
    std::string err;
};

// Runs argv[0] with the given arguments (UTF-8), quoting them for the Windows
// command line. Runs at IDLE priority unless idle = false.
ProcResult run(const std::vector<std::string>& argv, bool idle = true);

// Lower-case hex SHA-256 (FIPS 180-4).
std::string sha256Hex(std::string_view data);

std::vector<std::string> splitLines(std::string_view s);
std::string basename(std::string_view path);
std::string joinPath(std::string_view dir, std::string_view name);
void makeDirs(const std::string& dir);
bool isDirectory(const std::string& path);
bool fileExists(const std::string& path);
// Regular files directly inside dir (non-recursive), full paths, sorted by name.
std::vector<std::string> listFiles(const std::string& dir);

// Sets the current process to IDLE priority (builds/benchmarks keep priority).
void lowerOwnPriority();

}  // namespace whirl::parity
