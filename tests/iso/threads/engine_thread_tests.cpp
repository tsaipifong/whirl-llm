// whirl-iso-threads-tests: src/server/engine_thread.cpp built with only its declared dependencies (the server log
// src/server/log.cpp, whirl::common of src/common/util.cpp, platform time / fs / args) (ARCH-1 8.5 isolation,
// arch025-isoall).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/engine_thread.h"

#include <stdexcept>
#include <thread>

using whirl::server::EngineThread;

namespace {

void testOwner() {
    EngineThread t;
    ISO_CHECK(!t.running());
    t.enter();
    ISO_CHECK(t.running() && t.owner() == std::this_thread::get_id());
    const auto v0 = EngineThread::violations();
    t.check("iso_same_thread");
    ISO_CHECK(EngineThread::violations() == v0);
    t.leave();
    ISO_CHECK(!t.running());
}

void testViolation() {
    EngineThread t;
    t.enter();
    EngineThread::resetLoggedForTests();
    EngineThread::setThrowing(true);
    const auto v0 = EngineThread::violations();
    bool threw = false;
    std::thread other([&] {
        try {
            t.check("iso_other_thread");
        } catch (const std::logic_error&) {
            threw = true;
        }
    });
    other.join();
    EngineThread::setThrowing(false);
    ISO_CHECK(threw);
    ISO_CHECK(EngineThread::violations() == v0 + 1);
    t.leave();
}

}  // namespace

int main(int argc, char** argv) { return iso::run("threads", {{"owner", testOwner}, {"violation", testViolation}}, argc, argv); }
