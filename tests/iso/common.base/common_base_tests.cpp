// whirl-iso-common-base-tests: the module common.base (include/whirl/common.h, src/common/util.cpp, include/whirl/small_helpers.h) built with
// only its declared dependencies (the platform args / fs layer) and run (ARCH-1 8.5 isolation, arch025-iso4). No unit test file existed; the
// checks pin the UTF-8 / wide conversions, whole-file read / write, the memory mapping and the small numeric helpers.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "whirl/common.h"
#include "whirl/small_helpers.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

namespace {

namespace fs = std::filesystem;

std::string tmpFile(const char* name) {
    const fs::path d = fs::temp_directory_path() / "whirl_iso_common_base";
    fs::create_directories(d);
    return (d / name).string();
}

void testWidenNarrow() {
    ISO_CHECK(whirl::widen("abc") == L"abc" && whirl::narrow(L"abc") == "abc");
    const std::string s = "caf\xC3\xA9 \xE4\xB8\xAD\xE6\x96\x87 \xF0\x9F\x98\x80";  // e-acute, two CJK, an emoji (4-byte)
    const std::wstring w = whirl::widen(s);
    ISO_CHECK(w.size() == 4 + 1 + 2 + 1 + 2);  // the emoji is a surrogate pair
    ISO_CHECK(whirl::narrow(w) == s);
    ISO_CHECK(whirl::widen("").empty() && whirl::narrow(L"").empty());
    const wchar_t* argv[] = {L"whirl.exe", L"--model", L"müll.gguf"};
    const std::vector<std::string> a = whirl::narrowArgs(3, argv);
    ISO_CHECK(a.size() == 3 && a[0] == "whirl.exe" && a[2] == "m\xC3\xBCll.gguf");
}

void testUtf8() {
    std::string o;
    whirl::appendUtf8(o, 0x41);
    whirl::appendUtf8(o, 0xE9);
    whirl::appendUtf8(o, 0x4E2D);
    whirl::appendUtf8(o, 0x1F600);
    ISO_CHECK(o == "A\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80");
    ISO_CHECK(whirl::isValidUtf8(o) && whirl::isValidUtf8("") && whirl::isValidUtf8("plain ascii"));
    ISO_CHECK(!whirl::isValidUtf8("\xC0\x80"));          // overlong NUL
    ISO_CHECK(!whirl::isValidUtf8("\xED\xA0\x80"));      // a surrogate
    ISO_CHECK(!whirl::isValidUtf8("\xF4\x90\x80\x80"));  // above U+10FFFF
    ISO_CHECK(!whirl::isValidUtf8("\xE4\xB8"));          // truncated
    ISO_CHECK(!whirl::isValidUtf8("\x80"));              // stray continuation
    ISO_CHECK(whirl::utf8LenFromLead('a') == 1 && whirl::utf8LenFromLead(0xC3) == 2 && whirl::utf8LenFromLead(0xE4) == 3 &&
              whirl::utf8LenFromLead(0xF0) == 4 && whirl::utf8LenFromLead(0x80) == 1);
}

void testFiles() {
    const std::string p = tmpFile("rw.bin");
    const std::string data("hello\0world\n", 12);
    whirl::writeFile(p, data);
    ISO_CHECK(whirl::readFile(p) == data);
    {
        whirl::MappedFile m(p);
        ISO_CHECK(m.size() == data.size() && m.path() == p);
        ISO_CHECK(std::string(reinterpret_cast<const char*>(m.data()), m.size()) == data);
        whirl::MappedFile moved(std::move(m));
        ISO_CHECK(moved.size() == data.size() && m.data() == nullptr);
    }
    ISO_CHECK(whirl::test::throws([&] { (void)whirl::readFile(tmpFile("does_not_exist.bin")); }));
    ISO_CHECK(whirl::test::throws([&] { whirl::MappedFile m(tmpFile("does_not_exist.bin")); }));
    // a path with non-ASCII characters goes through widen()
    // (built as a string: std::filesystem would convert a narrow path through the ANSI code page)
    const std::string rw = tmpFile("rw.bin");
    const std::string u = rw.substr(0, rw.size() - 6) + "\xC3\xA9\xE4\xB8\xAD.txt";
    whirl::writeFile(u, "x");
    ISO_CHECK(whirl::readFile(u) == "x");
    std::error_code ec;
    fs::remove_all(fs::temp_directory_path() / "whirl_iso_common_base", ec);
}

void testSmallHelpers() {
    static_assert(whirl::satSub(5, 3) == 2 && whirl::satSub(3, 5) == 0 && whirl::satSub(3, 3) == 0);
    static_assert(whirl::alignUp(0, 16) == 0 && whirl::alignUp(1, 16) == 16 && whirl::alignUp(16, 16) == 16 && whirl::alignUp(17, 16) == 32);
    const std::int64_t u = whirl::unixNow();
    ISO_CHECK(u > 1700000000);  // after 2023
    const auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ISO_CHECK(whirl::msSince(t0) >= 15.0);
    const double a = whirl::nowMs();
    const double b = whirl::nowMs();
    ISO_CHECK(b >= a);
    const double s0 = whirl::nowSeconds();
    ISO_CHECK(whirl::nowSeconds() >= s0);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("common.base", {{"widen_narrow", testWidenNarrow}, {"utf8", testUtf8}, {"files", testFiles}, {"small_helpers", testSmallHelpers}},
                    argc, argv);
}
