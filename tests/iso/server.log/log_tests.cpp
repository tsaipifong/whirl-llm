// whirl-iso-server-log-tests: module server.log (src/server/log.{h,cpp}; registered by arch025-reg SC-14, manifest
// arch025-regb2 / PLAN N4-reg b2), built with only its declared dependencies (platform.time for the stamp, platform.fs
// for the file, common.base's widen) - no engine, no other server module (ARCH-1 8.5 isolation).
// The checks pin the line format that tools/gates/server_log_scan.ps1 and startup_lines.py read, from log.cpp:
// - "yyyy-mm-dd HH:MM:SS.mmm L message": 23-character stamp, a space, the level, a space, the message (the hook gets the
//   line without its '\n'); logI / logW / logE write levels I / W / E through std::format;
// - a message is cut to 16000 bytes; no hook after setHook(nullptr); open() appends to the file (creating the
//   directory), close() stops writing it; open() on a path that cannot be opened throws.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/log.h"

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace s = whirl::server;

std::vector<std::string> g_lines;
void hook(void*, std::string_view l) { g_lines.emplace_back(l); }

bool stampOk(const std::string& l) {
    // yyyy-mm-dd HH:MM:SS.mmm
    if (l.size() < 23) return false;
    const char* pat = "dddd-dd-dd dd:dd:dd.ddd";
    for (int i = 0; i < 23; ++i) {
        if (pat[i] == 'd' ? !std::isdigit(static_cast<unsigned char>(l[i])) : l[i] != pat[i]) return false;
    }
    return true;
}

void testFormat() {
    g_lines.clear();
    s::Log::setConsole(false);
    s::Log::setHook(hook, nullptr);
    s::logI("req {} | {}", 7, "hello");
    s::logW("w{}", 1);
    s::logE("e");
    s::Log::setHook(nullptr, nullptr);
    s::logI("not seen");
    ISO_CHECK(g_lines.size() == 3);
    if (g_lines.size() != 3) return;
    for (const std::string& l : g_lines) ISO_CHECK(stampOk(l) && l[23] == ' ' && l[25] == ' ' && l.back() != '\n');
    ISO_CHECK(g_lines[0].substr(24) == "I req 7 | hello");
    ISO_CHECK(g_lines[1].substr(24) == "W w1");
    ISO_CHECK(g_lines[2].substr(24) == "E e");
}

void testLongMessageCut() {
    g_lines.clear();
    s::Log::setHook(hook, nullptr);
    s::Log::line('I', std::string(20000, 'x'));
    s::Log::setHook(nullptr, nullptr);
    ISO_CHECK(g_lines.size() == 1 && g_lines[0].size() == 26 + 16000);
}

void testFile() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "whirl_iso_server_log" / "sub";
    std::error_code ec;
    fs::remove_all(dir.parent_path(), ec);
    const fs::path f = dir / "server.log";
    s::Log::open(f.string());  // creates the directory
    s::logI("first");
    s::Log::close();
    s::logI("after close");
    s::Log::open(f.string());  // appends
    s::logW("second");
    s::Log::close();
    std::ifstream in(f);
    std::stringstream ss;
    ss << in.rdbuf();
    in.close();
    const std::string t = ss.str();
    ISO_CHECK(t.find(" I first\n") != std::string::npos && t.find(" W second\n") != std::string::npos);
    ISO_CHECK(t.find("after close") == std::string::npos);
    ISO_CHECK(t.find(" I first\n") < t.find(" W second\n"));
    fs::remove_all(dir.parent_path(), ec);
    bool threw = false;
    try {
        s::Log::open((dir.parent_path() / "missing" / "dir" / "<bad>:name").string());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ISO_CHECK(threw);
    fs::remove_all(dir.parent_path(), ec);
}

// PATH-256 (pathlen256.DISCOVERY.md v3.1 2c #5): --log-file is never refused for its length; when it cannot be opened the
// message adds the path's length in UTF-16 units (a folder of that name: the open fails). Written before the fix.
void testOpenFailLength() {
    namespace fs = std::filesystem;
    const fs::path d = fs::temp_directory_path() / "whirl_iso_server_log_len" / "a-folder.log";
    std::error_code ec;
    fs::create_directories(d, ec);
    const std::string p = d.string();  // ASCII temp folder on this machine: characters = UTF-16 units
    std::string what;
    try {
        s::Log::open(p);
    } catch (const std::runtime_error& e) {
        what = e.what();
    }
    ISO_CHECK(what.rfind("cannot open log file " + p + " (the path is " + std::to_string(p.size()) + " characters long in UTF-16 / ", 0) == 0);
    fs::remove_all(d.parent_path(), ec);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.log", {{"format", testFormat}, {"long_message_cut", testLongMessageCut}, {"file", testFile}, {"open_fail_length", testOpenFailLength}},
                    argc, argv);
}
