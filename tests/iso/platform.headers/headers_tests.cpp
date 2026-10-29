// whirl-iso-platform-headers-tests: the module platform.headers (src/platform/headers_win.cpp, the compile-time proof that the src/platform
// headers stay platform-free) built with only the platform headers and run (ARCH-1 8.5 isolation, arch025-iso4). The proof is the #error
// in headers_win.cpp and in this file: each includes every platform header FIRST and stops the build when <windows.h> / Winsock came in with
// them. The run-time cases pin that the layer's marker symbol is there and that the headers declare the layer's namespaces.
// SPDX-License-Identifier: Apache-2.0
#include "platform/args.h"
#include "platform/console.h"
#include "platform/env.h"
#include "platform/fs.h"
#include "platform/fs_fake.h"
#include "platform/net.h"
#include "platform/net_fake.h"
#include "platform/paths.h"
#include "platform/proc.h"
#include "platform/sys.h"
#include "platform/time.h"

#if defined(_WINDOWS_) || defined(_WINSOCKAPI_) || defined(_WINSOCK2API_) || defined(_INC_WINDOWS)
#error "a src/platform header includes <windows.h> / Winsock; headers must stay platform-free (ADR-0004 #5)"
#endif

#include "iso/iso_test.h"

#include <cstring>

namespace whirl::platform {
extern const char kLayerName[];
}

namespace {

void testLayerMarker() { ISO_CHECK(std::strcmp(whirl::platform::kLayerName, "win32") == 0); }

void testHeadersPlatformFree() {
    // the preprocessor checks above ran at compile time; a header that leaked windows.h would not have compiled
#if defined(_WINDOWS_) || defined(_WINSOCKAPI_) || defined(_INC_WINDOWS)
    ISO_CHECK(false);
#else
    ISO_CHECK(true);
#endif
    ISO_CHECK(whirl::platform::net::kInvalid == ~whirl::platform::net::Socket(0));  // a header constant is usable without Winsock
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("platform.headers", {{"layer_marker", testLayerMarker}, {"headers_platform_free", testHeadersPlatformFree}}, argc, argv);
}
