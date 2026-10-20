// Unit tests of the device module (src/device), CPU only, no GPU, no HIP:
//   NullBackend (ARCH-1 D-3): host-memory semantics, counts, copy log, fault injection;
//   its memory model (D-5): memInfo / wddmMemInfo / localMemSnap / umaMemory return what the test set.
//   device::Profile (D-4): the per-family table, field by field (a changed value must be a deliberate edit here).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Run: whirl-device-tests.exe (exit 0 = all passed).

#include "device/null_backend.h"
#include "whirl/device_profile.h"

#include <array>
#include <chrono>
#include <string_view>
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

void testProfiles() {
    using whirl::hip::Arch;
    namespace d = whirl::device;
    const d::Profile& a = d::profile(Arch::gfx1201);
    const d::Profile& b = d::profile(Arch::gfx1151);
    CHECK(&a == &d::k_profile_gfx1201 && &b == &d::k_profile_gfx1151);
    CHECK(a.arch == Arch::gfx1201 && b.arch == Arch::gfx1151);
    // R9700 (gfx1201): discrete, the reference behaviour
    CHECK(!a.uma_no_vram_fit_check && !a.gemvh_grid_by_rows && !a.h16_from_candidates && !a.gemms_tie_larger_bm);
    CHECK(!a.attn_dq4_dynamic_splits && !a.tune_fast && !a.gdn_strix && !a.numerics_8060s && !a.mtp_costly_verify_rows);
    CHECK(std::string_view(a.tune_cache_suffix).empty());
    CHECK(std::string_view(a.tune_time_hint) == "2 minutes (a few seconds for MXFP4 models)");
    CHECK(a.prefill_exec_moe == 2048 && !a.prefill_fill_batch && a.vision_attn_threads == 128);
    const std::array<std::uint8_t, 17> ra = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    CHECK(a.gemv_r == ra);
    // Radeon 8060S (gfx1151): UMA iGPU
    CHECK(b.uma_no_vram_fit_check && b.gemvh_grid_by_rows && b.h16_from_candidates && b.gemms_tie_larger_bm);
    CHECK(b.attn_dq4_dynamic_splits && b.tune_fast && b.gdn_strix && b.numerics_8060s && b.mtp_costly_verify_rows);
    CHECK(std::string_view(b.tune_cache_suffix) == "-gfx1151r2");
    CHECK(std::string_view(b.tune_time_hint) == "8 minutes on the Radeon 8060S (less for MXFP4 models)");
    CHECK(b.prefill_exec_moe == 4096 && b.prefill_fill_batch && b.vision_attn_threads == 256);
    const std::array<std::uint8_t, 17> rb = {1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    CHECK(b.gemv_r == rb);
    // constexpr: the field reads fold like the arch comparisons they replaced
    static_assert(d::profile(Arch::gfx1151).prefill_exec_moe == 4096 && !d::profile(Arch::gfx1201).tune_fast);
}

void testNullBackendMemoryModel() {
    NullBackend b;
    whirl::device::Backend& be = b;  // through the interface the server uses
    CHECK(be.memInfo().free == 0 && be.memInfo().total == 0 && !be.wddmMemInfo().ok && !be.umaMemory(0).integrated);
    b.mem.info = {3ull << 30, 32ull << 30};
    b.mem.wddm.ok = true;
    b.mem.wddm.local_budget = 30ull << 30;
    b.mem.local.hip_free = 5ull << 30;
    b.mem.local.cls = whirl::hip::MemClass::integrated;
    b.mem.uma.integrated = b.mem.uma.known = true;
    b.mem.uma.dedicated = 64ull << 30;
    CHECK(be.memInfo().free == (3ull << 30) && be.memInfo().total == (32ull << 30));
    CHECK(be.wddmMemInfo().ok && be.wddmMemInfo().local_budget == (30ull << 30));
    CHECK(be.localMemSnap().hip_free == (5ull << 30) && be.localMemSnap().cls == whirl::hip::MemClass::integrated);
    const whirl::hip::UmaMem u = be.umaMemory(1);
    CHECK(u.integrated && u.known && u.dedicated == (64ull << 30));
}

}  // namespace

int main() {
    testNullBackendMemory();
    testNullBackendHandles();
    testNullBackendCopyLog();
    testNullBackendFaults();
    testNullBackendMemoryModel();
    testProfiles();
    std::printf("device tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
