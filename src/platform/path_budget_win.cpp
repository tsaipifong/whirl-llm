// Platform layer, the path length budget: the Windows implementation of platform/path_budget.h.
// SPDX-License-Identifier: Apache-2.0
//
// PATH-256 test-first stub (v0251-patha commit 1): the declarations of path_budget.h with no behaviour yet, so the
// tests of tests/iso/platform.path_budget compile and fail on their checks. The implementation is the next commit.

#include "platform/path_budget.h"

namespace whirl::platform::path_budget {

std::size_t utf8Bytes(std::wstring_view) { return 0; }
std::size_t utf16Units(std::string_view) { return 0; }
std::wstring stripLongPrefix(std::wstring_view p) { return std::wstring(p); }
std::wstring fullPath(const std::wstring& p) { return p; }
std::wstring folderOf(std::wstring_view full) { return std::wstring(full); }
std::optional<std::string> installDirRefusal(const std::wstring&, std::string_view) { return std::nullopt; }
std::optional<std::string> installDirRefusal(std::string_view) { return std::nullopt; }
std::optional<std::string> kvSsdDirRefusal(std::string_view, const std::string&) { return std::nullopt; }
std::optional<std::string> localAppDataWarning(const std::string&) { return std::nullopt; }
bool cwdLogFits(const std::wstring&) { return true; }

}  // namespace whirl::platform::path_budget
