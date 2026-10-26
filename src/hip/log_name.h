// Per-process debug log file names (bug:log-pid-names), shared by the launch log (launch_log.cpp) and the allocation
// log (alloc_log.cpp).
// SPDX-License-Identifier: Apache-2.0
//
// A name by PID alone (launch_<pid>.txt) was overwritten when Windows gave a later process the same PID
// (arch025 S0b-1 8060S: the moe balance pair table, PID 16468, was replaced 77 s later by moe fast single). Now:
//   <dir>\<prefix><pid>_<start>.txt          e.g. launch_16468_20261025T134651112.txt
//   <dir>\<prefix><pid>_<start>_<n>.txt      n = 2, 3, ... only if that name already exists
// <start> is this process's creation time (GetProcessTimes, UTC, YYYYMMDDTHHMMSSmmm), the PID + start-time key that
// Windows itself uses for a unique process (RM_UNIQUE_PROCESS). The file is created exclusively (fopen "wbx", C11
// 7.21.5.3; CREATE_NEW): an existing file is never opened, so no writer can truncate another one's table.
// Readers accept the old and the new names: ^<prefix>(\d+)(?:_(\d{8}T\d{9}))?(?:_(\d+))?\.txt$ (tools/impact/*.py,
// progress/tools/lt_import.py). Only reached with WHIRL_LAUNCH_LOG / WHIRL_ALLOC_LOG set.

#pragma once

#include <cerrno>
#include <cstdio>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace whirl::hip::logname {

inline constexpr int kMaxTries = 1000;

inline unsigned long processId() {
#ifdef _WIN32
    return GetCurrentProcessId();
#else
    return 0;
#endif
}

// this process's creation time, UTC, "YYYYMMDDTHHMMSSmmm" ("0" if unknown)
inline std::string processStart() {
#ifdef _WIN32
    FILETIME c{}, e{}, k{}, u{};
    SYSTEMTIME st{};
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u) && FileTimeToSystemTime(&c, &st)) {
        char b[32];
        std::snprintf(b, sizeof b, "%04u%02u%02uT%02u%02u%02u%03u", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute,
                      st.wSecond, st.wMilliseconds);
        return b;
    }
#endif
    return "0";
}

// creates <dir>\<prefix><pid>_<start>[_<n>]<ext> exclusively and returns it open for binary writing (nullptr if no
// name could be created); *stem gets the chosen name without <ext>. pid / start are parameters for the unit tests.
inline std::FILE* openUnique(const std::string& dir, const std::string& prefix, unsigned long pid, const std::string& start,
                             const char* ext, std::string* stem) {
    const std::string base = prefix + std::to_string(pid) + "_" + start;
    for (int i = 1; i <= kMaxTries; ++i) {
        std::string s = i == 1 ? base : base + "_" + std::to_string(i);
        const std::string path = dir + "\\" + s + ext;
        errno = 0;
        std::FILE* f = std::fopen(path.c_str(), "wbx");
        if (f != nullptr) {
            if (stem != nullptr) *stem = std::move(s);
            return f;
        }
        if (errno != EEXIST) return nullptr;  // folder missing, no access: nothing to try
    }
    return nullptr;
}

// the same for this process
inline std::FILE* openUnique(const std::string& dir, const std::string& prefix, const char* ext, std::string* stem) {
    static const std::string start = processStart();
    return openUnique(dir, prefix, processId(), start, ext, stem);
}

// creates <dir>\<stem><ext> exclusively (the RAW dump next to a table); nullptr if it exists
inline std::FILE* openExclusive(const std::string& dir, const std::string& stem, const char* ext) {
    return std::fopen((dir + "\\" + stem + ext).c_str(), "wbx");
}

}  // namespace whirl::hip::logname
