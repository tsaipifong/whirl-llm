// whirl-iso-platform-fs-tests: src/platform/fs_win.cpp (+ the fs_fake recorder) built alone (ARCH-1 8.5 isolation,
// arch025-isoall). Uses a file in the current directory (iso_fs_test.tmp), removed at the end.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "platform/fs.h"
#include "platform/fs_fake.h"

#include <cstdint>
#include <cstring>
#include <string>

using namespace whirl::platform;

namespace {

const wchar_t* kPath = L"iso_fs_test.tmp";

void testWriteRead() {
    fs::remove(kPath);
    fs::Handle h = fs::open(kPath, fs::kGenericRead | fs::kGenericWrite, fs::kShareNone, fs::kOpenAlways, fs::kAttrNormal);
    ISO_CHECK(fs::valid(h));
    const char msg[] = "isolation";
    std::uint32_t n = 0;
    ISO_CHECK(fs::write(h, msg, 9, &n) && n == 9);
    ISO_CHECK(fs::writeAt(h, "X", 1, 20, &n) && n == 1);
    std::int64_t sz = 0;
    ISO_CHECK(fs::size(h, &sz) && sz == 21);
    char buf[16] = {};
    ISO_CHECK(fs::readAt(h, buf, 9, 0, &n) && n == 9);
    ISO_CHECK(std::memcmp(buf, msg, 9) == 0);
    ISO_CHECK(fs::close(h));
    ISO_CHECK(fs::remove(kPath));
    fs::Handle g = fs::open(kPath, fs::kGenericRead, fs::kShareRead, fs::kOpenExisting, fs::kAttrNormal);
    ISO_CHECK(!fs::valid(g));
    ISO_CHECK(fs::lastError() == fs::kErrFileNotFound);
}

void testFakeRecorder() {
    fs::fake::begin();
    fs::Handle h = fs::open(kPath, fs::kGenericWrite, fs::kShareNone, fs::kOpenAlways, fs::kAttrNormal);
    const bool ok = fs::valid(h);
    if (ok) fs::close(h);
    const auto opens = fs::fake::end();
    fs::remove(kPath);
    ISO_CHECK(ok);
    ISO_CHECK(opens.size() == 1);
    if (opens.size() == 1) {
        ISO_CHECK(opens[0].path == kPath);
        ISO_CHECK(opens[0].access == fs::kGenericWrite && opens[0].disposition == fs::kOpenAlways && opens[0].ok);
    }
}

void testUtf8Strict() {
    std::wstring w;
    ISO_CHECK(fs::utf8ToWideStrict("a\xC3\xA9", &w) && w == L"a\x00E9");
    const char bad[] = {'a', static_cast<char>(0xFF), 0};
    ISO_CHECK(!fs::utf8ToWideStrict(bad, &w));
}

// arch025-d7 (D7.1): longPathForm leaves short and already-prefixed paths alone and gives a long path the \\?\ form
void testLongPathForm() {
    const std::wstring shortp = L"C:\\whirl\\whirl-server.exe";
    ISO_CHECK(fs::longPathForm(shortp) == shortp);
    const std::wstring seg(120, L'q');
    const std::wstring longp = L"C:\\" + seg + L"\\" + seg + L"\\" + seg + L"\\whirl-server.exe";
    ISO_CHECK(fs::longPathForm(longp) == L"\\\\?\\" + longp);
    const std::wstring pre = L"\\\\?\\" + longp;
    ISO_CHECK(fs::longPathForm(pre) == pre);
    const std::wstring unc = L"\\\\server\\share\\" + seg + L"\\" + seg + L"\\x.exe";
    ISO_CHECK(fs::longPathForm(unc) == L"\\\\?\\UNC\\" + unc.substr(2));
    // made absolute and normalised (\\?\ turns normalisation off): forward slashes and ".." resolved
    const std::wstring messy = L"C:/" + seg + L"/../" + seg + L"/" + seg + L"/" + seg + L"/a.exe";
    ISO_CHECK(fs::longPathForm(messy) == L"\\\\?\\C:\\" + seg + L"\\" + seg + L"\\" + seg + L"\\a.exe");
}

// arch025-lpath (1): shortPathForm takes the long-path prefix off when the plain path is short, and only then
void testShortPathForm() {
    ISO_CHECK(fs::shortPathForm(L"C:\\whirl\\x.exe") == L"C:\\whirl\\x.exe");
    ISO_CHECK(fs::shortPathForm(L"\\\\?\\C:\\whirl\\x.exe") == L"C:\\whirl\\x.exe");
    ISO_CHECK(fs::shortPathForm(L"\\\\?\\UNC\\srv\\share\\x.exe") == L"\\\\srv\\share\\x.exe");
    ISO_CHECK(fs::shortPathForm(L"\\\\srv\\share\\x.exe") == L"\\\\srv\\share\\x.exe");
    ISO_CHECK(fs::shortPathForm(L"\\\\.\\COM1") == L"\\\\.\\COM1");  // a device path is not a file path
    ISO_CHECK(fs::shortPathForm(L"\\\\?\\Volume{1}\\x") == L"\\\\?\\Volume{1}\\x");
    ISO_CHECK(fs::shortPathForm(L"") == L"");
    const std::wstring seg(120, L'q');
    const std::wstring longp = L"\\\\?\\C:\\" + seg + L"\\" + seg + L"\\" + seg + L"\\x.exe";  // plain form is 370 characters
    ISO_CHECK(fs::shortPathForm(longp) == longp);
    const std::wstring edge = L"C:\\" + std::wstring(240, L'e');  // 243 characters: plain form < 248
    ISO_CHECK(fs::shortPathForm(L"\\\\?\\" + edge) == edge);
    const std::wstring edge2 = L"C:\\" + std::wstring(245, L'e');  // 248 characters
    ISO_CHECK(fs::shortPathForm(L"\\\\?\\" + edge2) == L"\\\\?\\" + edge2);
}

// arch025-lpath (2): longPathFormDir is the long-path form when the files under the directory need it
void testLongPathFormDir() {
    const std::wstring d200 = L"C:\\" + std::wstring(197, L'd');  // 200 characters
    ISO_CHECK(fs::longPathFormDir(d200, 20) == d200);              // 200 + 1 + 20 < 248
    ISO_CHECK(fs::longPathFormDir(d200, 60) == L"\\\\?\\" + d200);  // its files would not be openable
    const std::wstring seg(120, L'q');
    const std::wstring longd = L"C:\\" + seg + L"\\" + seg + L"\\" + seg;
    ISO_CHECK(fs::longPathFormDir(longd, 1) == L"\\\\?\\" + longd);
    ISO_CHECK(fs::longPathFormDir(L"\\\\?\\" + longd, 40) == L"\\\\?\\" + longd);
    ISO_CHECK(fs::longPathFormDir(L"", 40) == L"");
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("platform.fs", {{"write_read", testWriteRead}, {"fake_recorder", testFakeRecorder}, {"utf8_strict", testUtf8Strict},
                                   {"long_path_form", testLongPathForm},
                                   {"short_path_form", testShortPathForm}, {"long_path_form_dir", testLongPathFormDir}},
                    argc, argv);
}
