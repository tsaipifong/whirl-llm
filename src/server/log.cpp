// SPDX-License-Identifier: Apache-2.0

#include "log.h"

#include "platform/time.h"
#include "whirl/common.h"

#include <cstdio>
#include <filesystem>
#include <mutex>
#include <stdexcept>

namespace whirl::server {

namespace {

std::mutex g_mutex;
std::FILE* g_file = nullptr;
bool g_console = true;
Log::Hook g_hook = nullptr;
void* g_hook_ctx = nullptr;

std::string stamp() {
    // arch025 PL1: GetLocalTime is platform::time::localTime (same fields, same format)
    const platform::time::LocalTime st = platform::time::localTime();
    return std::format("{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}", st.year, st.month, st.day, st.hour, st.minute,
                       st.second, st.milliseconds);
}

}  // namespace

void Log::open(const std::string& path) {
    const std::filesystem::path p(widen(path));
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    std::FILE* f = _wfopen(p.c_str(), L"ab");
    if (!f) throw std::runtime_error("cannot open log file " + path);
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_file) std::fclose(g_file);
    g_file = f;
}

void Log::close() {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_file) std::fclose(g_file);
    g_file = nullptr;
}

void Log::setConsole(bool on) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_console = on;
}

void Log::setHook(Hook h, void* ctx) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_hook = h;
    g_hook_ctx = ctx;
}

void Log::line(char level, std::string_view msg) {
    if (msg.size() > 16000) msg = msg.substr(0, 16000);
    std::string l = stamp();
    l += ' ';
    l += level;
    l += ' ';
    l += msg;
    l += '\n';
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_console) {
        std::fwrite(l.data(), 1, l.size(), stderr);
        std::fflush(stderr);
    }
    if (g_file) {
        std::fwrite(l.data(), 1, l.size(), g_file);
        std::fflush(g_file);
    }
    if (g_hook) g_hook(g_hook_ctx, std::string_view(l.data(), l.size() - 1));
}

}  // namespace whirl::server
