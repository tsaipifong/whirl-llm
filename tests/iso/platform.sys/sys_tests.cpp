// whirl-iso-platform-sys-tests: src/platform/sys_win.cpp built alone (ARCH-1 8.5 isolation, arch025-isoall).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "platform/sys.h"

#include <cstdint>

namespace {

void testMemoryStatus() {
    std::uint64_t total = 0, avail = 0;
    ISO_CHECK(whirl::platform::sys::memoryStatus(&total, &avail));
    ISO_CHECK(total > (1ull << 30));  // more than 1 GiB of physical RAM
    ISO_CHECK(avail > 0 && avail <= total);
}

}  // namespace

int main(int argc, char** argv) { return iso::run("platform.sys", {{"memory_status", testMemoryStatus}}, argc, argv); }
