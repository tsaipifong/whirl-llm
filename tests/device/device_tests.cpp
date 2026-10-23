// Unit tests of the device module (src/device), CPU only, no GPU, no HIP:
//   NullBackend (ARCH-1 D-3): host-memory semantics, counts, copy log, fault injection;
//   its memory model (D-5): memInfo / wddmMemInfo / localMemSnap / umaMemory return what the test set.
//   device::Profile (D-4): the per-family table, field by field (a changed value must be a deliberate edit here).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Run: whirl-device-tests.exe (exit 0 = all passed).

#include "device/handles.h"
#include "device/null_backend.h"
#include "whirl/device_profile.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <string_view>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

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
    // REVIEW-ARCH3 P3-3: a value with no row throws instead of getting the gfx1201 row (a new enumerator without a
    // case does not compile: tools/impact/profile_switch_check.ps1)
    bool threw = false;
    try {
        (void)d::profile(static_cast<Arch>(7));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
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

// arch024 M1: every handle releases on destruction, a moved-from handle releases nothing, move-assign releases
// the old value of the target first, reset() releases once; NullBackend::clean() at the end
void testHandles() {
    using whirl::device::DevBuf;
    using whirl::device::EventH;
    using whirl::device::PinnedBuf;
    using whirl::device::StreamH;
    NullBackend b;
    {
        DevBuf d = DevBuf::alloc(b, 1000);
        PinnedBuf h = PinnedBuf::alloc(b, 4096);
        EventH e = EventH::create(b);
        StreamH s = StreamH::create(b);
        CHECK(d && h && e && s && d.bytes() == 1000 && h.bytes() == 4096);
        CHECK(b.liveAllocations() == 2 && b.liveEvents() == 1 && b.liveStreams() == 1 && b.isHost(h.addr()));
        // move construction: the source is empty, nothing released
        DevBuf d2 = std::move(d);
        EventH e2 = std::move(e);
        CHECK(!d && !e && d.get() == 0 && e.get() == nullptr && d.bytes() == 0);  // NOLINT(bugprone-use-after-move)
        CHECK(d2 && e2 && b.liveAllocations() == 2 && b.liveEvents() == 1);
        // move-assign over a live handle releases the old one first
        DevBuf d3 = DevBuf::alloc(b, 64);
        CHECK(b.liveAllocations() == 3);
        d3 = std::move(d2);
        CHECK(b.liveAllocations() == 2 && d3.bytes() == 1000);
        EventH e3 = EventH::create(b);
        e3 = std::move(e2);
        CHECK(b.liveEvents() == 1 && e3);
        // self move-assign keeps the value
        DevBuf& alias = d3;
        d3 = std::move(alias);
        CHECK(d3 && b.liveAllocations() == 2);
        // reset releases once, a second reset calls nothing
        h.reset();
        h.reset();
        CHECK(!h && b.liveAllocations() == 1 && b.liveHost() == 0);
        s.reset();
        CHECK(b.liveStreams() == 0);
        // handles in a vector: push_back / erase / resize move them, no double release
        std::vector<EventH> v;
        for (int i = 0; i < 5; ++i) v.push_back(EventH::create(b));
        CHECK(b.liveEvents() == 6);
        v.erase(v.begin() + 1);
        CHECK(b.liveEvents() == 5);
        v.resize(20);
        CHECK(b.liveEvents() == 5);
        v.clear();
        CHECK(b.liveEvents() == 1);
        // waitIfPending on an empty handle calls nothing (it has no backend)
        EventH none;
        none.waitIfPending();
    }
    CHECK(b.clean());
}

// arch024 M1 F1 harness: NullBackend tracking mode, stream-order semantics
void testTracking() {
    using whirl::device::EventH;
    using whirl::device::StreamH;
    // (a) off (default): eventDone always true, no sync counted, no violation
    {
        NullBackend b;
        CHECK(!b.tracking());
        const DevPtr d = b.malloc(256), h = reinterpret_cast<DevPtr>(b.hostMalloc(256));
        StreamH s = StreamH::create(b);
        EventH e = EventH::create(b);
        b.copyAsync(h, d, 256, s.get());
        b.eventRecord(e.get(), s.get());
        CHECK(b.eventDone(e.get()) && b.eventDoneChecked(e.get()));
        e.waitIfPending();
        b.free(d);
        CHECK(b.syncCalls() == 0 && b.pendingFreeViolations() == 0);
        b.hostFree(reinterpret_cast<void*>(h));
    }
    // (d) "if (!eventDone) eventSync" really syncs while tracking; a later event covers earlier work on its stream
    {
        NullBackend b;
        b.trackPending(true);
        const DevPtr d = b.malloc(256), h = reinterpret_cast<DevPtr>(b.hostMalloc(256));
        StreamH s = StreamH::create(b);
        EventH e1 = EventH::create(b), e2 = EventH::create(b), never = EventH::create(b);
        CHECK(b.eventDone(never.get()));  // never recorded: done
        b.copyAsync(h, d, 256, s.get());
        b.eventRecord(e1.get(), s.get());
        b.copyAsync(h, d, 128, s.get());
        b.eventRecord(e2.get(), s.get());
        CHECK(!b.eventDone(e1.get()) && !b.eventDone(e2.get()));
        e2.waitIfPending();
        CHECK(b.syncCalls() == 1 && b.eventDone(e1.get()) && b.eventDone(e2.get()));
        e1.waitIfPending();  // already done: no second sync
        CHECK(b.syncCalls() == 1);
        b.free(d);
        b.hostFree(reinterpret_cast<void*>(h));
        CHECK(b.pendingFreeViolations() == 0);
    }
    // an earlier event does not cover later work; freeing a pending source is one violation (with its stream)
    {
        NullBackend b;
        b.trackPending(true);
        const DevPtr d = b.malloc(256), h = reinterpret_cast<DevPtr>(b.hostMalloc(256));
        StreamH s = StreamH::create(b);
        EventH e1 = EventH::create(b);
        b.copyAsync(h, d, 256, s.get());
        b.eventRecord(e1.get(), s.get());
        b.copyAsync(h, d, 64, s.get());
        e1.waitIfPending();
        b.free(d);
        CHECK(b.pendingFreeViolations() == 1);
        const auto v = b.violations();
        CHECK(v.size() == 1 && v[0].seq == 2 && v[0].stream != 0 && !v[0].host && v[0].order == 1);
        b.completeAll();
        b.hostFree(reinterpret_cast<void*>(h));
        CHECK(b.pendingFreeViolations() == 1);
    }
    // (e) streamWaitEvent is GPU-side ordering: a free right after it is still a violation; an event recorded later
    // on the waiting stream carries the dependency, so syncing it clears the first stream's work up to ev
    {
        NullBackend b;
        b.trackPending(true);
        const DevPtr d = b.malloc(256), h = reinterpret_cast<DevPtr>(b.hostMalloc(256));
        const DevPtr d2 = b.malloc(256), d3 = b.malloc(256);
        StreamH tier = StreamH::create(b), main = StreamH::create(b);
        EventH ev = EventH::create(b), later = EventH::create(b);
        b.copyAsync(h, d, 256, tier.get());
        b.eventRecord(ev.get(), tier.get());
        b.streamWaitEvent(main.get(), ev.get());
        b.free(d);
        CHECK(b.pendingFreeViolations() == 1);
        b.copyAsync(d3, d2, 256, tier.get());  // tier work after ev
        b.eventRecord(later.get(), main.get());
        later.waitIfPending();
        CHECK(b.eventDone(ev.get()));
        b.hostFree(reinterpret_cast<void*>(h));  // covered through the dependency
        CHECK(b.pendingFreeViolations() == 1);
        b.free(d2);  // tier work after ev is not covered
        CHECK(b.pendingFreeViolations() == 2);
        b.free(d3);
        CHECK(b.pendingFreeViolations() == 3);
    }
    // (f) streamSync clears every mark of that stream (and only that stream); complete() moves time, not a host sync
    {
        NullBackend b;
        b.trackPending(true);
        const DevPtr a = b.malloc(64), c = b.malloc(64), x = b.malloc(64), y = b.malloc(64);
        StreamH s1 = StreamH::create(b), s2 = StreamH::create(b);
        EventH e = EventH::create(b);
        b.copyAsync(c, a, 64, s1.get());
        b.copyAsync(c, a, 64, s1.get());
        b.copyAsync(y, x, 64, s2.get());
        b.eventRecord(e.get(), s2.get());
        b.streamSync(s1.get());
        CHECK(b.syncCalls() == 1);
        b.free(a);
        b.free(c);
        CHECK(b.pendingFreeViolations() == 0);
        CHECK(!b.eventDone(e.get()));
        b.complete(e.get());
        CHECK(b.syncCalls() == 1 && b.eventDone(e.get()));
        b.free(x);
        b.free(y);
        CHECK(b.pendingFreeViolations() == 0);
    }
    // the null stream is its own stream; download leaves no mark (synchronous)
    {
        NullBackend b;
        b.trackPending(true);
        const DevPtr a = b.malloc(64), c = b.malloc(64);
        unsigned char buf[64];
        b.download(buf, a, sizeof buf);
        b.free(a);
        CHECK(b.pendingFreeViolations() == 0);
        const DevPtr a2 = b.malloc(64);
        b.copyAsync(c, a2, 64, nullptr);
        b.free(a2);
        CHECK(b.pendingFreeViolations() == 1 && b.violations()[0].stream == 0);
        b.streamSync(nullptr);
        b.free(c);
        CHECK(b.pendingFreeViolations() == 1);
        // turning tracking off clears its state
        b.trackPending(false);
        CHECK(b.pendingFreeViolations() == 0 && b.syncCalls() == 0);
    }
}

// arch024 M1 (A4), S6a: the thread of every Backend call is recorded, in call order
void testReleaseThreads() {
    using whirl::device::DevBuf;
    using whirl::device::EventH;
    using Call = NullBackend::Call;
    NullBackend b;
    b.recordThreads(true);
    DevBuf d = DevBuf::alloc(b, 64);
    EventH e = EventH::create(b);
    std::thread::id other;
    std::thread t([&] {
        other = std::this_thread::get_id();
        b.setDevice(0);
        b.eventSync(e.get());
        d.reset();
    });
    t.join();
    e.reset();
    const auto all = b.callThreads();
    std::vector<std::pair<Call, std::thread::id>> r;
    for (const auto& c : all)
        if (NullBackend::isRelease(c.first)) r.push_back(c);
    CHECK(r.size() == 2);
    CHECK(r.size() == 2 && r[0].first == Call::free && r[0].second == other);
    CHECK(r.size() == 2 && r[1].first == Call::event_destroy && r[1].second == std::this_thread::get_id());
    // every call, in order: malloc, eventCreate (this thread), setDevice, eventSync, free (the other), eventDestroy
    const std::vector<Call> want = {Call::malloc, Call::event_create, Call::set_device, Call::event_sync, Call::free,
                                    Call::event_destroy};
    CHECK(all.size() == want.size());
    for (std::size_t i = 0; i < std::min(all.size(), want.size()); ++i) {
        CHECK(all[i].first == want[i]);
        const bool on_other = i >= 2 && i <= 4;
        CHECK(all[i].second == (on_other ? other : std::this_thread::get_id()));
    }
    CHECK(std::string(NullBackend::callName(Call::uma_memory)) == "umaMemory");
    CHECK(std::string(NullBackend::callName(Call::copy_async)) == "copyAsync");
    b.recordThreads(false);
    b.setDevice(0);
    CHECK(b.callThreads().size() == all.size());  // off: nothing more recorded
    CHECK(b.clean());
}

}  // namespace

int main() {
    testNullBackendMemory();
    testNullBackendHandles();
    testNullBackendCopyLog();
    testNullBackendFaults();
    testNullBackendMemoryModel();
    testProfiles();
    testHandles();
    testTracking();
    testReleaseThreads();
    std::printf("device tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
