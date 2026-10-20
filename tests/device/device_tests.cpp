// Unit tests of the device module (src/device), CPU only, no GPU, no HIP:
//   NullBackend (ARCH-1 D-3): host-memory semantics, counts, copy log, fault injection.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Run: whirl-device-tests.exe (exit 0 = all passed).

#include "device/null_backend.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace {

int g_pass = 0, g_fail = 0;

void check(bool ok, const char* what, int line) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL %s:%d: %s\n", __FILE__, line, what);
    }
}
#define CHECK(c) check((c), #c, __LINE__)

template <class F>
bool throws(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

using whirl::device::DevPtr;
using whirl::device::NullBackend;

void testNullBackendMemory() {
    NullBackend b;
    const DevPtr d = b.malloc(1000);
    void* h = b.hostMalloc(4096);
    CHECK(d != 0 && d % 4096 == 0 && reinterpret_cast<std::uintptr_t>(h) % 4096 == 0);
    CHECK(b.liveAllocations() == 2 && b.liveHost() == 1);
    CHECK(b.isHost(reinterpret_cast<DevPtr>(h)) && b.isHost(reinterpret_cast<DevPtr>(h) + 4095));
    CHECK(!b.isHost(reinterpret_cast<DevPtr>(h) + 4096) && !b.isHost(d));
    // zero-filled
    unsigned char z[1000];
    b.download(z, d, sizeof z);
    bool zero = true;
    for (unsigned char c : z) zero = zero && c == 0;
    CHECK(zero);
    // host -> device -> host round trip through copyAsync / download
    std::memset(h, 0x5a, 1000);
    b.copyAsync(d, reinterpret_cast<DevPtr>(h), 1000, nullptr);
    b.download(z, d, sizeof z);
    CHECK(z[0] == 0x5a && z[999] == 0x5a);
    CHECK(b.copyCalls() == 1 && b.copiedBytes() == 1000);
    b.copyAsync(d, d, 0, nullptr);  // empty copy: counted as a call, no bytes
    CHECK(b.copyCalls() == 2 && b.copiedBytes() == 1000);
    b.free(d);
    b.hostFree(h);
    b.free(0);
    b.hostFree(nullptr);  // null frees are no-ops, like the HIP runtime
    CHECK(b.clean() && b.liveHost() == 0);
}

void testNullBackendHandles() {
    NullBackend b;
    auto s = b.streamCreateNonBlocking();
    auto e1 = b.eventCreate();
    auto e2 = b.eventCreate();
    CHECK(b.liveStreams() == 1 && b.liveEvents() == 2 && !b.clean());
    CHECK(s != nullptr && e1 != e2);
    b.eventRecord(e1, s);
    CHECK(b.eventDone(e1) && b.eventDoneChecked(e2));
    b.eventDestroy(e1);
    b.eventDestroy(e2);
    b.streamDestroy(s);
    CHECK(b.clean());
}

void testNullBackendCopyLog() {
    NullBackend b;
    void* h = b.hostMalloc(256);
    const DevPtr d = b.malloc(256);
    b.copyAsync(d, reinterpret_cast<DevPtr>(h), 64, nullptr);  // not recorded yet
    b.recordCopies(true);
    b.copyAsync(d, reinterpret_cast<DevPtr>(h), 128, nullptr);   // pinned -> device (a restore)
    b.copyAsync(reinterpret_cast<DevPtr>(h), d, 32, nullptr);    // device -> pinned (a spill)
    b.copyAsync(d + 128, d, 16, nullptr);                        // device -> device
    b.copyAsync(d, d, 0, nullptr);                               // empty: not logged
    const auto log = b.copies();
    CHECK(log.size() == 3);
    if (log.size() == 3) {
        CHECK(log[0].bytes == 128 && log[0].from_host && !log[0].to_host);
        CHECK(log[1].bytes == 32 && !log[1].from_host && log[1].to_host);
        CHECK(log[2].bytes == 16 && !log[2].from_host && !log[2].to_host);
    }
    b.recordCopies(false);
    b.copyAsync(d, d + 64, 8, nullptr);
    CHECK(b.copies().size() == 3);
    b.recordCopies(true);  // a new recording starts empty
    CHECK(b.copies().empty());
    b.free(d);
    b.hostFree(h);
    CHECK(b.clean());
}

void testNullBackendFaults() {
    NullBackend b;
    void* h = b.hostMalloc(64);
    const DevPtr d = b.malloc(64);
    // copy call n throws (counted like copyCalls)
    b.failCopyCall(b.copyCalls() + 2);
    CHECK(!throws([&] { b.copyAsync(d, d, 8, nullptr); }));
    CHECK(throws([&] { b.copyAsync(d, d, 8, nullptr); }));
    CHECK(!throws([&] { b.copyAsync(d, d, 8, nullptr); }));
    b.failCopyCall(0);
    // the 2nd copy from pinned host memory throws, once; device-sourced copies do not count
    b.failHostToDevCopy(2);
    CHECK(b.hostCopyFailArmed());
    CHECK(!throws([&] { b.copyAsync(d, reinterpret_cast<DevPtr>(h), 8, nullptr); }));
    CHECK(!throws([&] { b.copyAsync(d, d, 8, nullptr); }));
    CHECK(throws([&] { b.copyAsync(d, reinterpret_cast<DevPtr>(h), 8, nullptr); }));
    CHECK(!b.hostCopyFailArmed());
    CHECK(!throws([&] { b.copyAsync(d, reinterpret_cast<DevPtr>(h), 8, nullptr); }));
    // event query errors: eventDoneChecked throws, eventDone still reads as complete
    auto e = b.eventCreate();
    b.failEventQueries(true);
    CHECK(throws([&] { b.eventDoneChecked(e); }) && b.eventDone(e));
    b.failEventQueries(false);
    CHECK(!throws([&] { b.eventDoneChecked(e); }));
    // eventSync held until released
    b.holdEventSync();
    std::atomic<bool> passed{false};
    std::thread t([&] {
        b.eventSync(e);
        passed = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(!passed.load());
    b.releaseEventSync();
    t.join();
    CHECK(passed.load());
    b.eventDestroy(e);
    b.free(d);
    b.hostFree(h);
    CHECK(b.clean());
}

}  // namespace

int main() {
    testNullBackendMemory();
    testNullBackendHandles();
    testNullBackendCopyLog();
    testNullBackendFaults();
    std::printf("device tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
