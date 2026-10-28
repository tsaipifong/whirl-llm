// whirl-iso-platform-proc-tests: src/platform/proc_win.cpp built alone (ARCH-1 8.5 isolation, arch025-isoall).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "platform/proc.h"

#include <cstdint>
#include <string>

using namespace whirl::platform;

namespace {

void testExePath() {
    const std::wstring p = proc::exePath();
    ISO_CHECK(p.size() > 4);
    ISO_CHECK(p.find(L"whirl-iso-platform-proc-tests") != std::wstring::npos);
    ISO_CHECK(p.substr(p.size() - 4) == L".exe");
}

void testIds() {
    ISO_CHECK(proc::currentId() != 0);
    std::uint64_t ws = 0;
    ISO_CHECK(proc::workingSetBytes(proc::currentProcess(), &ws));
    ISO_CHECK(ws > 0);
}

void testEvent() {
    proc::Handle ev = proc::createEvent();
    ISO_CHECK(ev != nullptr);
    ISO_CHECK(proc::wait(ev, 0) == proc::kWaitTimeout);  // not signaled
    ISO_CHECK(proc::setEvent(ev));
    ISO_CHECK(proc::wait(ev, 0) == proc::kWaitObject0);
    ISO_CHECK(proc::wait(ev, 0) == proc::kWaitObject0);  // manual reset: stays signaled
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("platform.proc", {{"exe_path", testExePath}, {"ids", testIds}, {"event", testEvent}}, argc, argv);
}
