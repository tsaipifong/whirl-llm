// whirl-iso-platform-fs-tests: src/platform/fs_win.cpp (+ the fs_fake recorder) built alone (ARCH-1 8.5 isolation,
// arch025-isoall). Uses a file in the current directory (iso_fs_test.tmp), removed at the end.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "platform/fs.h"
#include "platform/fs_fake.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

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

// (arch025-lpath2: shortPathForm, the prefix strip of arch025-lpath (1), is gone with its test: review F1)

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

// arch025-lpath2 (review F2): a long path that already carries a prefix, or a forward-slash spelling of one, comes back
// with exactly one prefix, as Rust std get_long_path does: \\?\ and \??\ input unchanged; otherwise the prefix follows
// the GetFullPathNameW result (\\?\ and \??\ as they are, \\.\ replaced by \\?\, \\ -> \\?\UNC\). cdd3519 made
// \\?\C:\??\C:\..., \\?\UNC\?\C:\... and \\?\UNC\.\C:\... of them, and a --kv-ssd-dir spelled so lost its SSD tier
// ("cannot lock the SSD directory", Windows error 123 / 53). Without a prefix Win32 parses \\.\ and //?/ paths and they
// fail past MAX_PATH (scratch\a25_lpath2\probe_out.txt), so \\.\ is replaced, not kept.
void testLongPathFormPrefixed() {
    const auto fwd = [](std::wstring s) {
        std::replace(s.begin(), s.end(), L'\\', L'/');
        return s;
    };
    const std::wstring seg(120, L'q');
    const std::wstring longp = L"C:\\" + seg + L"\\" + seg + L"\\" + seg + L"\\whirl.lock";  // 376 characters
    const std::wstring vb = L"\\\\?\\" + longp;
    ISO_CHECK(fs::longPathForm(L"\\??\\" + longp) == L"\\??\\" + longp);  // NT prefix: as given
    ISO_CHECK(fs::longPathForm(L"//?/" + fwd(longp)) == vb);  // //?/ is parsed by Win32 (not verbatim): one \\?\ prefix
    ISO_CHECK(fs::longPathForm(L"\\\\?/" + fwd(longp)) == vb);
    ISO_CHECK(fs::longPathForm(L"//./" + fwd(longp)) == vb);  // device form: \\.\ replaced by \\?\ (same file)
    ISO_CHECK(fs::longPathForm(L"\\\\.\\" + longp) == vb);
    ISO_CHECK(fs::longPathForm(L"\\\\.\\C:\\" + seg + L"\\..\\" + seg + L"\\" + seg + L"\\" + seg + L"\\whirl.lock") == vb);
    const std::wstring unc = L"\\\\srv\\share\\" + seg + L"\\" + seg + L"\\whirl.lock";
    ISO_CHECK(fs::longPathForm(L"//srv/share/" + seg + L"/" + seg + L"/whirl.lock") == L"\\\\?\\UNC\\" + unc.substr(2));
    // no result carries a second prefix
    const std::vector<std::wstring> ins = {L"\\??\\" + longp, L"//?/" + fwd(longp), L"\\\\?/" + fwd(longp), L"//./" + fwd(longp),
                                           L"\\\\.\\" + longp, L"\\\\?\\" + longp, longp, fwd(longp), unc};
    for (const std::wstring& in : ins) {
        const std::wstring r = fs::longPathForm(in);
        const bool one = r.find(L"\\\\?\\", 1) == std::wstring::npos && r.find(L"\\??\\", 1) == std::wstring::npos &&
                         r.find(L"\\\\.\\", 1) == std::wstring::npos && r.find(L"UNC\\?") == std::wstring::npos &&
                         r.find(L"UNC\\.") == std::wstring::npos;
        if (!one) std::printf("  long_path_form_prefixed: %zu-character input -> second prefix in the result\n", in.size());
        ISO_CHECK(one);
    }
    // short spellings stay byte-identical
    for (const wchar_t* s : {L"\\??\\C:\\w\\whirl.lock", L"//?/C:/w/whirl.lock", L"//./C:/w/whirl.lock", L"\\\\.\\C:\\w\\whirl.lock",
                             L"\\\\?/C:/w/whirl.lock"})
        ISO_CHECK(fs::longPathForm(s) == s);
    // the directory form of the SSD tier (longPathFormDir(dir, 48), Tier::start) for the spellings the review measured
    const std::wstring d300 = L"C:\\" + seg + L"\\" + seg + L"\\" + std::wstring(55, L'r');  // 300 characters
    ISO_CHECK(d300.size() == 300);
    ISO_CHECK(fs::longPathFormDir(L"\\??\\" + d300, 48) == L"\\??\\" + d300);
    ISO_CHECK(fs::longPathFormDir(L"//?/" + fwd(d300), 48) == L"\\\\?\\" + d300);
    ISO_CHECK(fs::longPathFormDir(L"//./" + fwd(d300), 48) == L"\\\\?\\" + d300);
    ISO_CHECK(fs::longPathFormDir(L"\\\\.\\" + d300, 48) == L"\\\\?\\" + d300);
    const std::wstring d200 = L"C:\\" + std::wstring(197, L'd');  // 200: "//?/" + it is 204, its files reach 253 characters
    ISO_CHECK(fs::longPathFormDir(L"//?/" + fwd(d200), 48) == L"\\\\?\\" + d200);  // 6a9f6be opened it, cdd3519 did not
}

// arch025-lpath (review D8) kept as a test: short paths are never changed. 20000 random strings over path characters
// (separators, dots, colons, wildcards, spaces, non-ASCII): longPathForm leaves every one under 248 characters as it is,
// longPathFormDir(dir, 48) every one under 199 (199 + 1 + 48 = 248). Deterministic seed.
void testShortPathIdentity() {
    std::mt19937_64 rng(20261101);
    const std::wstring alpha = L"abcXYZ019 .\\/:?*~$\x00e9\x4e2d";
    std::size_t changed_form = 0, changed_dir = 0;
    for (int i = 0; i < 20000; ++i) {
        const std::size_t len = static_cast<std::size_t>(rng() % 248);
        std::wstring s;
        for (std::size_t k = 0; k < len; ++k) s += alpha[static_cast<std::size_t>(rng() % alpha.size())];
        if (fs::longPathForm(s) != s) ++changed_form;
        const std::wstring d = s.substr(0, std::min<std::size_t>(s.size(), 198));
        if (fs::longPathFormDir(d, 48) != d) ++changed_dir;
    }
    std::printf("short path identity: 20000 random strings: longPathForm changed %zu, longPathFormDir changed %zu\n", changed_form, changed_dir);
    ISO_CHECK(changed_form == 0);
    ISO_CHECK(changed_dir == 0);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("platform.fs", {{"write_read", testWriteRead}, {"fake_recorder", testFakeRecorder}, {"utf8_strict", testUtf8Strict},
                                   {"long_path_form", testLongPathForm},
                                   {"long_path_form_dir", testLongPathFormDir},
                                   {"long_path_form_prefixed", testLongPathFormPrefixed}, {"short_path_identity", testShortPathIdentity}},
                    argc, argv);
}
