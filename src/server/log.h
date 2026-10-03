// Server log: "yyyy-mm-dd HH:MM:SS.mmm L message" lines on stderr and
// appended to a log file. Thread-safe.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <format>
#include <string>
#include <string_view>

namespace whirl::server {

class Log {
public:
    // Opens (creates directories, appends to) the log file. Throws on failure.
    static void open(const std::string& path);
    static void close();
    // Writes one line (level 'I', 'W' or 'E').
    static void line(char level, std::string_view msg);
    // Mirror to stderr (default true; tests turn it off).
    static void setConsole(bool on);
    // Optional hook receiving every line (tests).
    using Hook = void (*)(void* ctx, std::string_view line);
    static void setHook(Hook h, void* ctx);
};

template <class... Args>
void logI(std::format_string<Args...> f, Args&&... args) {
    Log::line('I', std::format(f, std::forward<Args>(args)...));
}
template <class... Args>
void logW(std::format_string<Args...> f, Args&&... args) {
    Log::line('W', std::format(f, std::forward<Args>(args)...));
}
template <class... Args>
void logE(std::format_string<Args...> f, Args&&... args) {
    Log::line('E', std::format(f, std::forward<Args>(args)...));
}

}  // namespace whirl::server
