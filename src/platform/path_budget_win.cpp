// Platform layer, the path length budget: the Windows implementation of platform/path_budget.h.
// SPDX-License-Identifier: Apache-2.0
//
// PATH-256 (v0.2.5.1 PLAN V1-22; progress/pathlen256.DISCOVERY.md v3.1 2a, 2c). A path is measured after
// GetFullPathNameW (a relative path against the working folder; WHIRL never changes its working folder, so the one
// measurement at startup is what the later open sees) and without a \\?\ / \\?\UNC\ prefix. The Chinese lines of the
// messages are written as UTF-8 escapes (no non-ASCII text in string literals: a build without /utf-8 would re-read it
// in the ANSI code page); their text is in the comments next to them.

#include "platform/path_budget.h"

#include "platform/proc.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace whirl::platform::path_budget {

namespace {

std::string num(std::size_t n) { return std::to_string(n); }

bool isHigh(unsigned c) { return c >= 0xD800 && c <= 0xDBFF; }
bool isLow(unsigned c) { return c >= 0xDC00 && c <= 0xDFFF; }

std::wstring wideOf(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) return {};
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

// UTF-16 units of the full path of a UTF-8 path (the typed text without its prefix when GetFullPathNameW cannot read it)
std::size_t fullUnits(const std::string& utf8) {
    const std::wstring w = wideOf(utf8);
    const std::wstring f = fullPath(w);
    return (f.empty() ? stripLongPrefix(w) : f).size();
}

}  // namespace

std::size_t utf8Bytes(std::wstring_view w) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        const unsigned c = static_cast<unsigned>(w[i]);
        if (c < 0x80) {
            n += 1;
        } else if (c < 0x800) {
            n += 2;
        } else if (isHigh(c) && i + 1 < w.size() && isLow(static_cast<unsigned>(w[i + 1]))) {
            n += 4;  // a surrogate pair: one character outside the BMP
            ++i;
        } else {
            n += 3;  // the rest of the BMP, and an unpaired surrogate (WideCharToMultiByte writes U+FFFD)
        }
    }
    return n;
}

// the count MultiByteToWideChar(CP_UTF8, 0) gives, the flags of whirl::widen (platform/args_win.cpp)
std::size_t utf16Units(std::string_view utf8) {
    if (utf8.empty()) return 0;
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    return n > 0 ? static_cast<std::size_t>(n) : 0;
}

std::wstring stripLongPrefix(std::wstring_view p) {
    if (p.size() >= 8 && p.substr(0, 4) == L"\\\\?\\" && (p[4] == L'U' || p[4] == L'u') && (p[5] == L'N' || p[5] == L'n') &&
        (p[6] == L'C' || p[6] == L'c') && p[7] == L'\\')
        return L"\\\\" + std::wstring(p.substr(8));  // \\?\UNC\srv\share -> \\srv\share
    if (p.size() >= 4 && p.substr(0, 4) == L"\\\\?\\") return std::wstring(p.substr(4));
    return std::wstring(p);
}

std::wstring fullPath(const std::wstring& p) {
    if (p.empty()) return {};
    for (int attempt = 0; attempt < 2; ++attempt) {
        const DWORD need = GetFullPathNameW(p.c_str(), 0, nullptr, nullptr);  // the size with the NUL
        if (need == 0) return {};
        std::wstring out(need, L'\0');
        const DWORD got = GetFullPathNameW(p.c_str(), need, out.data(), nullptr);  // the length without the NUL
        if (got == 0) return {};
        if (got < need) {
            out.resize(got);
            return stripLongPrefix(out);
        }
        // the working folder changed between the two calls: ask again once
    }
    return {};
}

std::wstring folderOf(std::wstring_view full) {
    const std::size_t s = full.find_last_of(L"\\/");
    if (s == std::wstring_view::npos) return {};
    if (s == 2 && full[1] == L':') return std::wstring(full.substr(0, 3));  // C:\b.exe -> C:\ (the root keeps its separator)
    return std::wstring(full.substr(0, s));
}

std::optional<std::string> installDirRefusal(const std::wstring& exe_path, std::string_view prog) {
    const std::size_t n = utf8Bytes(folderOf(stripLongPrefix(exe_path)));
    if (n <= kInstallDirMax) return std::nullopt;
    const std::string p(prog);
    // <prog>: the install folder path is N bytes in UTF-8 (...); the limit is 150. Move ...
    // <prog>：安裝資料夾路徑長 N 位元組（UTF-8：英數字每字 1、中文每字 3），上限 150。請把 WHIRL 資料夾移到較短的路徑，例如 C:\whirl。
    return p + ": the install folder path is " + num(n) +
           " bytes in UTF-8 (English letters and digits count 1, Chinese characters count 3); the limit is " + num(kInstallDirMax) +
           ". Move the WHIRL folder to a shorter path, for example C:\\whirl.\n" + p +
           "\xef\xbc\x9a\xe5\xae\x89\xe8\xa3\x9d\xe8\xb3\x87\xe6\x96\x99\xe5\xa4\xbe\xe8\xb7\xaf\xe5\xbe\x91\xe9\x95\xb7 " + num(n) +
           " \xe4\xbd\x8d\xe5\x85\x83\xe7\xb5\x84\xef\xbc\x88UTF-8\xef\xbc\x9a\xe8\x8b\xb1\xe6\x95\xb8\xe5\xad\x97\xe6\xaf\x8f\xe5\xad\x97 "
           "1\xe3\x80\x81\xe4\xb8\xad\xe6\x96\x87\xe6\xaf\x8f\xe5\xad\x97 3\xef\xbc\x89\xef\xbc\x8c\xe4\xb8\x8a\xe9\x99\x90 " +
           num(kInstallDirMax) +
           "\xe3\x80\x82\xe8\xab\x8b\xe6\x8a\x8a WHIRL \xe8\xb3\x87\xe6\x96\x99\xe5\xa4\xbe\xe7\xa7\xbb\xe5\x88\xb0\xe8\xbc\x83\xe7\x9f\xad\xe7"
           "\x9a\x84\xe8\xb7\xaf\xe5\xbe\x91\xef\xbc\x8c\xe4\xbe\x8b\xe5\xa6\x82 C:\\whirl\xe3\x80\x82\n";
}

std::optional<std::string> installDirRefusal(std::string_view prog) {
    const std::wstring exe = proc::exePath();
    if (exe.empty()) return std::nullopt;
    return installDirRefusal(exe, prog);
}

std::optional<std::string> kvSsdDirRefusal(std::string_view name, const std::string& dir) {
    const std::size_t n = fullUnits(dir);
    if (n <= kKvSsdDirMax) return std::nullopt;
    const std::string nm(name);
    // <name> is N characters long (UTF-16: ...); the limit is 217.
    // <name> 長 N 字元（UTF-16：中英文每字都算 1），上限 217。
    return nm + " is " + num(n) + " characters long (UTF-16: every letter or Chinese character counts 1); the limit is " + num(kKvSsdDirMax) +
           ".\n" + nm + " \xe9\x95\xb7 " + num(n) +
           " \xe5\xad\x97\xe5\x85\x83\xef\xbc\x88UTF-16\xef\xbc\x9a\xe4\xb8\xad\xe8\x8b\xb1\xe6\x96\x87\xe6\xaf\x8f\xe5\xad\x97\xe9\x83\xbd\xe7\xae\x97 "
           "1\xef\xbc\x89\xef\xbc\x8c\xe4\xb8\x8a\xe9\x99\x90 " +
           num(kKvSsdDirMax) + "\xe3\x80\x82";
}

std::optional<std::string> localAppDataWarning(const std::string& lad) {
    const std::size_t n = fullUnits(lad);
    if (n <= kLocalAppDataMax) return std::nullopt;
    // warning: LOCALAPPDATA is N characters long (UTF-16: ...); the limit is 187, so WHIRL keeps no files there (...).
    // 警告：LOCALAPPDATA 長 N 字元（UTF-16：中英文每字都算 1），上限 187，所以 WHIRL 不在那裡存檔（不存調校快取、不用預設 SSD 快取、server log 只輸出到主控台）。
    return "warning: LOCALAPPDATA is " + num(n) + " characters long (UTF-16: every letter or Chinese character counts 1); the limit is " +
           num(kLocalAppDataMax) +
           ", so WHIRL keeps no files there (no tune cache, no default SSD cache, the server log goes to the console only).\n"
           "\xe8\xad\xa6\xe5\x91\x8a\xef\xbc\x9aLOCALAPPDATA \xe9\x95\xb7 " +
           num(n) +
           " \xe5\xad\x97\xe5\x85\x83\xef\xbc\x88UTF-16\xef\xbc\x9a\xe4\xb8\xad\xe8\x8b\xb1\xe6\x96\x87\xe6\xaf\x8f\xe5\xad\x97\xe9\x83\xbd\xe7\xae\x97 "
           "1\xef\xbc\x89\xef\xbc\x8c\xe4\xb8\x8a\xe9\x99\x90 " +
           num(kLocalAppDataMax) +
           "\xef\xbc\x8c\xe6\x89\x80\xe4\xbb\xa5 WHIRL \xe4\xb8\x8d\xe5\x9c\xa8\xe9\x82\xa3\xe8\xa3\xa1\xe5\xad\x98\xe6\xaa\x94\xef\xbc\x88\xe4\xb8\x8d"
           "\xe5\xad\x98\xe8\xaa\xbf\xe6\xa0\xa1\xe5\xbf\xab\xe5\x8f\x96\xe3\x80\x81\xe4\xb8\x8d\xe7\x94\xa8\xe9\xa0\x90\xe8\xa8\xad SSD \xe5\xbf\xab"
           "\xe5\x8f\x96\xe3\x80\x81server log \xe5\x8f\xaa\xe8\xbc\xb8\xe5\x87\xba\xe5\x88\xb0\xe4\xb8\xbb\xe6\x8e\xa7\xe5\x8f\xb0\xef\xbc\x89\xe3\x80\x82\n";
}

bool cwdLogFits(const std::wstring& cwd_full) { return stripLongPrefix(cwd_full).size() + kCwdLogName <= kPathMax; }

}  // namespace whirl::platform::path_budget
