// whirl-iso-platform-args-tests: src/platform/args_win.cpp built alone (ARCH-1 8.5 isolation, arch025-isoall).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "platform/args.h"

#include <cstring>
#include <string>

using namespace whirl::platform;

namespace {

void testRoundTrip() {
    const char* s = "h\xC3\xA9llo \xE4\xB8\xAD";  // "héllo 中": 6 + 1 UTF-16 units, 10 bytes
    const int n = static_cast<int>(std::strlen(s));
    const int need = args::wideFromUtf8(s, n, nullptr, 0);
    ISO_CHECK(need == 7);
    std::wstring w(static_cast<std::size_t>(need), L'\0');
    ISO_CHECK(args::wideFromUtf8(s, n, w.data(), need) == need);
    ISO_CHECK(w[1] == L'\x00E9' && w[6] == L'\x4E2D');
    const int back = args::utf8FromWide(w.data(), need, nullptr, 0);
    ISO_CHECK(back == n);
    std::string u(static_cast<std::size_t>(back), '\0');
    ISO_CHECK(args::utf8FromWide(w.data(), need, u.data(), back) == back);
    ISO_CHECK(u == s);
}

void testInvalidReplaced() {
    // flags 0: an invalid byte becomes U+FFFD (v0.2.4 behaviour)
    const char bad[] = {'a', static_cast<char>(0xFF), 'b'};
    wchar_t out[8] = {};
    const int k = args::wideFromUtf8(bad, 3, out, 8);
    ISO_CHECK(k == 3);
    ISO_CHECK(out[0] == L'a' && out[1] == L'\xFFFD' && out[2] == L'b');
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("platform.args", {{"round_trip", testRoundTrip}, {"invalid_replaced", testInvalidReplaced}}, argc, argv);
}
