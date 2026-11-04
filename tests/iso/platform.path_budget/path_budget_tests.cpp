// whirl-iso-platform-path-budget-tests: PATH-256 P1 (v0.2.5.1 PLAN V1-22; progress/pathlen256.DISCOVERY.md v3.1 2a, 2c,
// 2d-1): the path budget of src/platform/path_budget.h built with nothing but its own files and its declared
// dependency (platform.proc: the exe path). Written from the design text before the implementation (test first): every
// limit with its unit, the two counting rules (UTF-8 bytes only for the install folder, UTF-16 units for every path WHIRL
// opens itself), the \\?\ / \\?\UNC\ prefix, relative paths against a known working folder, and the refusal / warning
// texts (English line + Traditional Chinese line, both naming the unit). Expected numbers are literal; the UTF-8 /
// UTF-16 counts were computed independently (scratch/v0251_patha/vectors.py -> vectors.out).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "platform/path_budget.h"

#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>

namespace {

namespace pb = whirl::platform::path_budget;
namespace fs = std::filesystem;

const wchar_t kZh = L'\u4e2d';  // 中: 1 UTF-16 unit, 3 UTF-8 bytes

std::wstring rep(wchar_t c, std::size_t n) { return std::wstring(n, c); }

bool has(const std::optional<std::string>& m, const char* s) { return m && m->find(s) != std::string::npos; }

// the limits of the design (2a), literal: a changed number must change this test too
void testLimits() {
    ISO_CHECK(pb::kPathMax == 255);
    ISO_CHECK(pb::kInstallDirMax == 150);
    ISO_CHECK(pb::kLocalAppDataMax == 187);
    ISO_CHECK(pb::kKvSsdDirMax == 217);
    ISO_CHECK(pb::kZipEntryMax == 104);
    ISO_CHECK(pb::kDevExeMax == 150);
    ISO_CHECK(pb::kDevFileBelow == 240);
    ISO_CHECK(pb::kTune5NameMax == 61);
    ISO_CHECK(pb::kLadPartMax == 68);
    ISO_CHECK(pb::kKvEntryName == 38);
    ISO_CHECK(pb::kKvLockName == 11);
    ISO_CHECK(pb::kServerLogPart == 17);
    ISO_CHECK(pb::kPinnedPartMax == 28);
    ISO_CHECK(pb::kCwdLogName == 17);
    // the fixed parts are the real strings
    ISO_CHECK(std::string("\\whirl\\").size() + pb::kTune5NameMax == pb::kLadPartMax);
    ISO_CHECK(std::string("\\0123456789abcdef-0123456789abcdef.wkv").size() == pb::kKvEntryName);
    ISO_CHECK(std::string("\\whirl.lock").size() == pb::kKvLockName);
    ISO_CHECK(std::string("\\whirl\\server.log").size() == pb::kServerLogPart);
    ISO_CHECK(std::string("\\whirl\\pinned\\4294967295.txt").size() == pb::kPinnedPartMax);
    ISO_CHECK(std::string("\\whirl-server.log").size() == pb::kCwdLogName);
    ISO_CHECK(std::string("tune5-").size() + 24 + 1 + 16 + pb::kTuneSuffixMax + 4 == pb::kTune5NameMax);
    ISO_CHECK(std::string("-gfx1151r2").size() == pb::kTuneSuffixMax);
}

// UTF-8 bytes (the install folder's unit): ASCII 1, U+00E9 2, CJK 3, a surrogate pair 4, an unpaired surrogate 3 (U+FFFD)
void testUtf8Bytes() {
    ISO_CHECK(pb::utf8Bytes(L"") == 0);
    ISO_CHECK(pb::utf8Bytes(L"C:\\" + rep(L'a', 147)) == 150);
    ISO_CHECK(pb::utf8Bytes(L"C:\\" + rep(L'a', 148)) == 151);
    ISO_CHECK(pb::utf8Bytes(rep(kZh, 50)) == 150);
    ISO_CHECK(pb::utf8Bytes(rep(kZh, 51)) == 153);
    ISO_CHECK(pb::utf8Bytes(L"\u00e9") == 2);
    ISO_CHECK(pb::utf8Bytes(L"a\U0001F600b") == 6);
    ISO_CHECK(pb::utf8Bytes(std::wstring(1, static_cast<wchar_t>(0xD800))) == 3);
    ISO_CHECK(pb::utf8Bytes(std::wstring(1, static_cast<wchar_t>(0xDC00)) + L"a") == 4);
}

// UTF-16 units (the unit of every path WHIRL opens): CJK 1 each although 3 bytes in UTF-8
void testUtf16Units() {
    std::string zh217, zh218;
    for (int i = 0; i < 217; ++i) zh217 += "\xe4\xb8\xad";  // 中
    zh218 = zh217 + "\xe4\xb8\xad";
    ISO_CHECK(zh217.size() == 651);  // the byte count is not the unit
    ISO_CHECK(pb::utf16Units(zh217) == 217);
    ISO_CHECK(pb::utf16Units(zh218) == 218);
    ISO_CHECK(pb::utf16Units("") == 0);
    ISO_CHECK(pb::utf16Units("C:\\abc") == 6);
    ISO_CHECK(pb::utf16Units("a\xf0\x9f\x98\x80" "b") == 4);  // a U+1F600 b: a surrogate pair counts 2
    ISO_CHECK(pb::utf16Units("\xff") == 1);                    // invalid byte -> U+FFFD
}

void testPrefix() {
    ISO_CHECK(pb::stripLongPrefix(L"\\\\?\\C:\\x\\y") == L"C:\\x\\y");
    ISO_CHECK(pb::stripLongPrefix(L"\\\\?\\UNC\\srv\\share\\x") == L"\\\\srv\\share\\x");
    ISO_CHECK(pb::stripLongPrefix(L"C:\\x") == L"C:\\x");
    ISO_CHECK(pb::stripLongPrefix(L"\\\\srv\\share\\x") == L"\\\\srv\\share\\x");
    ISO_CHECK(pb::stripLongPrefix(L"") == L"");
    ISO_CHECK(pb::folderOf(L"C:\\a\\b.exe") == L"C:\\a");
    ISO_CHECK(pb::folderOf(L"C:\\b.exe") == L"C:\\");
    ISO_CHECK(pb::folderOf(L"\\\\srv\\share\\b.exe") == L"\\\\srv\\share");
    ISO_CHECK(pb::folderOf(L"b.exe") == L"");
}

// relative paths are measured after GetFullPathNameW against the working folder; \\?\ is not counted
void testFullPath() {
    std::error_code ec;
    const fs::path saved = fs::current_path(ec);
    const fs::path root = fs::temp_directory_path(ec) / L"whirl-pb-iso";
    fs::create_directories(root / L"a" / L"b", ec);
    fs::current_path(root / L"a" / L"b", ec);
    ISO_CHECK(!ec);
    // the working folder as Windows keeps it (no 8.3 expansion), and its parent
    const std::wstring b = fs::current_path(ec).wstring();
    const std::wstring a = b.substr(0, b.find_last_of(L'\\'));
    ISO_CHECK(pb::fullPath(L"..\\x") == a + L"\\x");
    ISO_CHECK(pb::fullPath(L"x") == b + L"\\x");
    ISO_CHECK(pb::fullPath(L"\\\\?\\C:\\x\\y") == L"C:\\x\\y");
    ISO_CHECK(pb::fullPath(L"") == L"");
    // a short typed folder whose full path is over the limit is refused (the length is the full path's)
    const std::wstring cwd = fs::current_path(ec).wstring();
    if (cwd.size() + 1 + 5 <= pb::kKvSsdDirMax) {
        // name = (218 - cwd - 1) characters: <cwd>\<name> is 218 units
        const std::wstring name(pb::kKvSsdDirMax + 1 - cwd.size() - 1, L'k');
        const std::string n8(name.size(), 'k');
        ISO_CHECK(pb::fullPath(name).size() == pb::kKvSsdDirMax + 1);
        ISO_CHECK(has(pb::kvSsdDirRefusal("--kv-ssd-dir", n8), "is 218 characters long"));
        const std::string n7 = n8.substr(1);
        ISO_CHECK(!pb::kvSsdDirRefusal("--kv-ssd-dir", n7));
    } else {
        std::printf("  (relative kv folder case not run: working folder %zu characters)\n", cwd.size());
    }
    fs::current_path(saved, ec);
    fs::remove_all(root, ec);
}

// 2c #2: the install folder, UTF-8 bytes of the folder of the exe path; 150 passes, 151 is refused
void testInstallDir() {
    const std::wstring f150 = L"C:\\" + rep(L'i', 147);
    const std::wstring f151 = L"C:\\" + rep(L'i', 148);
    ISO_CHECK(!pb::installDirRefusal(f150 + L"\\whirl.exe", "whirl"));
    ISO_CHECK(!pb::installDirRefusal(f150 + L"\\whirl-server.exe", "whirl-server"));  // the exe name is not counted
    const std::optional<std::string> m = pb::installDirRefusal(f151 + L"\\whirl.exe", "whirl");
    ISO_CHECK(m.has_value());
    ISO_CHECK(m && m->rfind("whirl: the install folder path is 151 bytes in UTF-8 (English letters and digits count 1, Chinese characters count 3); the limit is 150. Move the WHIRL folder to a shorter path, for example C:\\whirl.\n", 0) == 0);
    ISO_CHECK(has(m, "whirl\xef\xbc\x9a\xe5\xae\x89\xe8\xa3\x9d\xe8\xb3\x87\xe6\x96\x99\xe5\xa4\xbe\xe8\xb7\xaf\xe5\xbe\x91\xe9\x95\xb7 151 \xe4\xbd\x8d\xe5\x85\x83\xe7\xb5\x84"));  // whirl：安裝資料夾路徑長 151 位元組
    ISO_CHECK(has(m, "\xe4\xb8\x8a\xe9\x99\x90 150"));  // 上限 150
    ISO_CHECK(has(m, "UTF-8"));
    const std::optional<std::string> ms = pb::installDirRefusal(f151 + L"\\whirl-server.exe", "whirl-server");
    ISO_CHECK(ms && ms->rfind("whirl-server: the install folder path is 151 bytes", 0) == 0);
    ISO_CHECK(has(ms, "whirl-server\xef\xbc\x9a"));  // whirl-server：
    // the unit is bytes: 49 Chinese characters after C:\ = 150 bytes pass, 50 (53 characters) = 153 bytes are refused
    ISO_CHECK(!pb::installDirRefusal(L"C:\\" + rep(kZh, 49) + L"\\whirl.exe", "whirl"));
    const std::optional<std::string> mz = pb::installDirRefusal(L"C:\\" + rep(kZh, 50) + L"\\whirl.exe", "whirl");
    ISO_CHECK(has(mz, "is 153 bytes in UTF-8"));
    // \\?\ and \\?\UNC\ are not part of the folder's length
    ISO_CHECK(!pb::installDirRefusal(L"\\\\?\\" + f150 + L"\\whirl.exe", "whirl"));
    ISO_CHECK(has(pb::installDirRefusal(L"\\\\?\\" + f151 + L"\\whirl.exe", "whirl"), "is 151 bytes"));
    const std::wstring unc150 = L"\\\\srv\\share\\" + rep(L'u', 150 - 12);
    ISO_CHECK(unc150.size() == 150);
    ISO_CHECK(!pb::installDirRefusal(L"\\\\?\\UNC\\srv\\share\\" + rep(L'u', 150 - 12) + L"\\whirl.exe", "whirl"));
    ISO_CHECK(has(pb::installDirRefusal(L"\\\\?\\UNC\\srv\\share\\" + rep(L'u', 151 - 12) + L"\\whirl.exe", "whirl"), "is 151 bytes"));
    ISO_CHECK(!pb::installDirRefusal(L"C:\\whirl.exe", "whirl"));
    // this process (the test program's own folder is short)
    ISO_CHECK(!pb::installDirRefusal("whirl"));
}

// 2c #4 (「八、改」): --kv-ssd-dir / WHIRL_KV_SSD_DIR, UTF-16 units of the full path; 217 passes, 218 is refused
void testKvSsdDir() {
    const std::string d217 = "C:\\" + std::string(214, 'k');
    const std::string d218 = "C:\\" + std::string(215, 'k');
    ISO_CHECK(!pb::kvSsdDirRefusal("--kv-ssd-dir", d217));
    const std::optional<std::string> m = pb::kvSsdDirRefusal("--kv-ssd-dir", d218);
    ISO_CHECK(m && m->rfind("--kv-ssd-dir is 218 characters long (UTF-16: every letter or Chinese character counts 1); the limit is 217.\n", 0) == 0);
    ISO_CHECK(has(m, "--kv-ssd-dir \xe9\x95\xb7 218 \xe5\xad\x97\xe5\x85\x83\xef\xbc\x88UTF-16\xef\xbc\x9a"));  // --kv-ssd-dir 長 218 字元（UTF-16：
    ISO_CHECK(has(m, "\xe4\xb8\x8a\xe9\x99\x90 217"));                                                       // 上限 217
    ISO_CHECK(has(pb::kvSsdDirRefusal("WHIRL_KV_SSD_DIR", d218), "WHIRL_KV_SSD_DIR is 218 characters long"));
    // the unit is UTF-16: 214 Chinese characters after C:\ = 217 units (645 bytes) pass, 215 are refused
    std::string z214 = "C:\\";
    for (int i = 0; i < 214; ++i) z214 += "\xe4\xb8\xad";
    ISO_CHECK(!pb::kvSsdDirRefusal("--kv-ssd-dir", z214));
    ISO_CHECK(has(pb::kvSsdDirRefusal("--kv-ssd-dir", z214 + "\xe4\xb8\xad"), "is 218 characters long"));
    // \\?\ is not counted
    ISO_CHECK(!pb::kvSsdDirRefusal("--kv-ssd-dir", "\\\\?\\" + d217));
    ISO_CHECK(has(pb::kvSsdDirRefusal("--kv-ssd-dir", "\\\\?\\" + d218), "is 218 characters"));
    // the longest allowed folder keeps every SSD tier file under 256
    ISO_CHECK(pb::kKvSsdDirMax + pb::kKvEntryName <= pb::kPathMax && pb::kKvSsdDirMax + pb::kKvLockName <= pb::kPathMax);
}

// 2c #3: %LOCALAPPDATA% longer than 187 UTF-16 units -> WHIRL keeps no file there, one warning (both languages)
void testLocalAppData() {
    ISO_CHECK(!pb::localAppDataWarning("C:\\" + std::string(184, 'l')));
    const std::optional<std::string> w = pb::localAppDataWarning("C:\\" + std::string(185, 'l'));
    ISO_CHECK(has(w, "LOCALAPPDATA is 188 characters long (UTF-16: every letter or Chinese character counts 1); the limit is 187"));
    ISO_CHECK(has(w, "LOCALAPPDATA \xe9\x95\xb7 188 \xe5\xad\x97\xe5\x85\x83"));  // LOCALAPPDATA 長 188 字元
    ISO_CHECK(has(w, "\xe4\xb8\x8a\xe9\x99\x90 187"));                           // 上限 187
    std::string z184 = "C:\\";
    for (int i = 0; i < 184; ++i) z184 += "\xe4\xb8\xad";
    ISO_CHECK(!pb::localAppDataWarning(z184));  // 187 units although 555 bytes
    ISO_CHECK(pb::localAppDataWarning(z184 + "\xe4\xb8\xad").has_value());
    // the longest allowed %LOCALAPPDATA% keeps every WHIRL file under it below 256
    ISO_CHECK(pb::kLocalAppDataMax + pb::kLadPartMax <= pb::kPathMax);
    // 2c #8: %LOCALAPPDATA% unset: <cwd>\whirl-server.log must fit 255
    ISO_CHECK(pb::cwdLogFits(L"C:\\" + rep(L'c', 235)));   // 238 + 17 = 255
    ISO_CHECK(!pb::cwdLogFits(L"C:\\" + rep(L'c', 236)));  // 239 + 17 = 256
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("platform.path_budget",
                    {{"limits", testLimits},
                     {"utf8_bytes", testUtf8Bytes},
                     {"utf16_units", testUtf16Units},
                     {"prefix", testPrefix},
                     {"full_path", testFullPath},
                     {"install_dir", testInstallDir},
                     {"kv_ssd_dir", testKvSsdDir},
                     {"local_app_data", testLocalAppData}},
                    argc, argv);
}
