// Host-memory device backend (ARCH-1 D-3): device::Backend without a GPU, for
// the unit tests of the host-side logic (server engine, prefix-cache tiers).
// Device addresses are host addresses; copies are memmove; streams and events
// are counted handles; every call is cheap and deterministic.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. No HIP. Replaces the two test mocks that had grown apart
// (tests/server/tier_tests.cpp and tests/server/mock_backend.h MockDeviceOps).
//
// Observation: live allocation / stream / event counts (a leak check after a
// test), copy counts and bytes, and an optional log of the copies (direction
// and size), e.g. "a restore copies N pages of B bytes from pinned RAM".
// Fault injection: copy call n throws; the nth copy from pinned host memory
// (a tier restore) throws; event queries report an asynchronous device error;
// eventSync can be held closed so a test controls when an IO thread proceeds.

#pragma once

#include "device/backend.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace whirl::device {

class NullBackend final : public Backend {
public:
    NullBackend() = default;
    NullBackend(const NullBackend&) = delete;
    NullBackend& operator=(const NullBackend&) = delete;
    ~NullBackend() override;

    // ---- device::Backend
    void copyAsync(DevPtr dst, DevPtr src, std::uint64_t bytes, Stream s) override;
    void download(void* dst, DevPtr src, std::uint64_t bytes) override;
    Stream streamCreateNonBlocking() override;
    void streamDestroy(Stream s) override;
    void streamSync(Stream) override {}
    void streamWaitEvent(Stream, Event) override {}
    Event eventCreate() override;
    void eventDestroy(Event e) override;
    void eventRecord(Event, Stream) override {}
    bool eventDone(Event) override { return true; }  // like the HIP backend: a query error reads as complete
    bool eventDoneChecked(Event) override;
    void eventSync(Event) override;
    void* hostMalloc(std::uint64_t bytes) override;  // zero-filled, 4 KiB aligned
    void hostFree(void* p) override;
    DevPtr malloc(std::uint64_t bytes) override;     // zero-filled, 4 KiB aligned
    void free(DevPtr p) override;
    void setDevice(int) override {}
    const char* lastErrorString() const override { return "null backend"; }
    hip::MemInfo memInfo() override { return mem.info; }
    hip::WddmMemInfo wddmMemInfo() override { return mem.wddm; }
    hip::LocalMemSnap localMemSnap() override { return mem.local; }
    hip::UmaMem umaMemory(int) override { return mem.uma; }

    // ---- memory model: the values memInfo / wddmMemInfo / localMemSnap / umaMemory return (a test sets them)
    struct Memory {
        hip::MemInfo info;
        hip::WddmMemInfo wddm;
        hip::LocalMemSnap local;
        hip::UmaMem uma;
    } mem;

    // ---- observation
    std::size_t liveAllocations() const;  // device + pinned host
    long liveHost() const { return live_host_.load(); }
    long liveStreams() const { return live_streams_.load(); }
    long liveEvents() const { return live_events_.load(); }
    bool clean() const { return liveAllocations() == 0 && liveStreams() == 0 && liveEvents() == 0; }
    std::uint64_t copyCalls() const { return copy_calls_.load(); }  // every copyAsync call, also 0 bytes
    std::uint64_t copiedBytes() const { return copied_.load(); }
    bool isHost(DevPtr p) const;  // inside a hostMalloc block
    struct Copy {
        std::uint64_t bytes = 0;
        bool from_host = false, to_host = false;
    };
    void recordCopies(bool on);
    std::vector<Copy> copies() const;  // recorded copies (non-empty ones), in call order

    // ---- fault injection
    // copyAsync call number n (as counted by copyCalls(), so "copyCalls() + 2" is the second call from now) throws; 0: off
    void failCopyCall(std::uint64_t n) { fail_call_ = n; }
    // the nth non-empty copy (from now) whose source is pinned host memory throws (once); 0: off
    void failHostToDevCopy(std::uint64_t nth) {
        host_copies_ = 0;
        fail_host_copy_ = nth;
    }
    bool hostCopyFailArmed() const { return fail_host_copy_.load() != 0; }
    // eventDoneChecked throws (an asynchronous failure of the work before the event)
    void failEventQueries(bool on) { event_error_ = on; }
    // eventSync blocks until released (deterministic hold of an IO thread)
    void holdEventSync();
    void releaseEventSync();

private:
    mutable std::mutex mu_;
    std::set<void*> live_;
    std::map<std::uintptr_t, std::uint64_t> host_;  // pinned host blocks (start -> bytes)
    std::atomic<long> live_host_{0}, live_streams_{0}, live_events_{0};
    std::atomic<std::uint64_t> copy_calls_{0}, copied_{0}, fail_call_{0};
    std::atomic<std::uint64_t> fail_host_copy_{0}, host_copies_{0};
    std::atomic<bool> event_error_{false}, record_{false};
    std::vector<Copy> log_;
    std::mutex gate_m_;
    std::condition_variable gate_cv_;
    bool gate_open_ = true;
};

}  // namespace whirl::device
