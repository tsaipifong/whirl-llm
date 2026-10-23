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
//
// Tracking mode (arch024 M1, invariant F1 "never free what the GPU may still read"), off by default: with
// trackPending(true) the backend models stream order. Each stream (the null stream is stream 0) has an issued
// work counter and a completion mark; copyAsync marks its source and destination allocation with
// {stream, issued}; eventRecord(ev, s) captures s's counter (and what s waited for); eventSync / complete
// move the completion mark of that stream up to the event (so a later event covers earlier work on the same
// stream), streamSync to the newest work. streamWaitEvent orders the GPU side only: it moves no host-side
// mark, but events recorded on that stream later carry the dependency. eventDone reports pending work as
// not done. free / hostFree of an allocation with a pending mark is counted in pendingFreeViolations()
// (never thrown: free runs from destructors). With tracking off every call behaves as before.
// recordThreads(true) logs the thread of every Backend call (its Call kind, in call order), e.g. "the tier IO
// thread only calls setDevice and eventSync" (arch024 S6a). lastErrorString is not logged (const, no device work).

#pragma once

#include "device/backend.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
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
    void streamSync(Stream s) override;
    void streamWaitEvent(Stream s, Event e) override;
    Event eventCreate() override;
    void eventDestroy(Event e) override;
    void eventRecord(Event e, Stream s) override;
    bool eventDone(Event e) override;  // like the HIP backend: a query error reads as complete
    bool eventDoneChecked(Event e) override;
    void eventSync(Event e) override;
    void* hostMalloc(std::uint64_t bytes) override;  // zero-filled, 4 KiB aligned
    void hostFree(void* p) override;
    DevPtr malloc(std::uint64_t bytes) override;     // zero-filled, 4 KiB aligned
    void free(DevPtr p) override;
    void setDevice(int) override { noteThread(Call::set_device); }
    const char* lastErrorString() const override { return "null backend"; }
    hip::MemInfo memInfo() override {
        noteThread(Call::mem_info);
        return mem.info;
    }
    hip::WddmMemInfo wddmMemInfo() override {
        noteThread(Call::wddm_mem_info);
        return mem.wddm;
    }
    hip::LocalMemSnap localMemSnap() override {
        noteThread(Call::local_mem_snap);
        return mem.local;
    }
    hip::UmaMem umaMemory(int) override {
        noteThread(Call::uma_memory);
        return mem.uma;
    }
    hip::GpuOthers gpuOthers(int) override {
        noteThread(Call::gpu_others);
        return mem.others;
    }
    std::uint64_t pinnedHostBytes() override {
        noteThread(Call::pinned_host_bytes);
        return mem.pinned;
    }

    // ---- memory model: the values memInfo / wddmMemInfo / localMemSnap / umaMemory / gpuOthers / pinnedHostBytes
    // return (a test sets them)
    struct Memory {
        hip::MemInfo info;
        hip::WddmMemInfo wddm;
        hip::LocalMemSnap local;
        hip::UmaMem uma;
        hip::GpuOthers others;
        std::uint64_t pinned = 0;
    } mem;

    // ---- observation
    std::size_t liveAllocations() const;  // device + pinned host
    long liveHost() const { return live_host_.load(); }
    long liveStreams() const { return live_streams_.load(); }
    long liveEvents() const { return live_events_.load(); }
    bool clean() const { return liveAllocations() == 0 && liveStreams() == 0 && liveEvents() == 0; }
    std::uint64_t copyCalls() const { return copy_calls_.load(); }  // every copyAsync call, also 0 bytes
    std::uint64_t copiedBytes() const { return copied_.load(); }
    // bytes of device allocations alive (the memory governor test's "WDDM local usage")
    std::uint64_t deviceBytes() const { return dev_bytes_.load(); }
    bool isHost(DevPtr p) const;  // inside a hostMalloc block
    struct Copy {
        std::uint64_t bytes = 0;
        bool from_host = false, to_host = false;
    };
    void recordCopies(bool on);
    std::vector<Copy> copies() const;  // recorded copies (non-empty ones), in call order

    // ---- tracking mode (F1; see the header comment). Turning it on or off clears its state.
    void trackPending(bool on);
    bool tracking() const { return track_.load(); }
    void complete(Event e);  // the work before e finished (a test moves time forward)
    void completeAll();      // all work on every stream finished
    std::uint64_t syncCalls() const { return sync_calls_.load(); }  // eventSync + streamSync while tracking
    struct Violation {
        std::uint64_t stream = 0, seq = 0;  // the pending work: stream id (0: null stream), its work number
        std::uint64_t order = 0;            // release call number (1 = first free / hostFree while tracking)
        bool host = false;                  // hostFree
    };
    std::size_t pendingFreeViolations() const;
    std::vector<Violation> violations() const;
    // ---- threads of the Backend calls
    enum class Call {
        copy_async, download, stream_create, stream_destroy, stream_sync, stream_wait_event, event_create,
        event_destroy, event_record, event_done, event_done_checked, event_sync, host_malloc, host_free, malloc,
        free, set_device, mem_info, wddm_mem_info, local_mem_snap, uma_memory, gpu_others, pinned_host_bytes
    };
    // free / hostFree / eventDestroy / streamDestroy
    static bool isRelease(Call c) {
        return c == Call::free || c == Call::host_free || c == Call::event_destroy || c == Call::stream_destroy;
    }
    static const char* callName(Call c);
    void recordThreads(bool on);
    std::vector<std::pair<Call, std::thread::id>> callThreads() const;  // in call order

    // ---- fault injection
    // copyAsync call number n (as counted by copyCalls(), so "copyCalls() + 2" is the second call from now) throws; 0: off
    void failCopyCall(std::uint64_t n) { fail_call_ = n; }
    // the nth non-empty copy (from now) whose source is pinned host memory throws (once); 0: off
    void failHostToDevCopy(std::uint64_t nth) {
        host_copies_ = 0;
        fail_host_copy_ = nth;
    }
    bool hostCopyFailArmed() const { return fail_host_copy_.load() != 0; }
    // the nth device malloc (from now) throws (once); 0: off
    void failMalloc(std::uint64_t nth) {
        mallocs_ = 0;
        fail_malloc_ = nth;
    }
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
    std::atomic<std::uint64_t> fail_malloc_{0}, mallocs_{0};
    std::map<std::uintptr_t, std::uint64_t> dev_;  // device allocations (start -> bytes)
    std::atomic<std::uint64_t> dev_bytes_{0};
    std::atomic<bool> event_error_{false}, record_{false};
    std::vector<Copy> log_;
    // tracking mode (under mu_)
    using Mark = std::pair<std::uint64_t, std::uint64_t>;  // {stream id, work number}
    struct StreamState {
        std::uint64_t issued = 0, done = 0;
        std::vector<Mark> deps;  // work of other streams this stream waited for (GPU side)
    };
    struct EventRec {
        Mark at{0, 0};
        std::vector<Mark> deps;
    };
    static std::uint64_t idOf(Stream s) { return s == nullptr ? 0 : static_cast<std::uint64_t>(*static_cast<int*>(s)); }
    std::uintptr_t allocOf(std::uintptr_t p) const;  // start of the allocation holding p (0: none)
    void advance(const Mark& m, const std::vector<Mark>& deps);
    void checkRelease(std::uintptr_t start, bool host);
    void noteThread(Call c);
    bool doneLocked(Event e);  // eventDone's answer with tracking on, under mu_
    std::atomic<bool> track_{false}, rec_threads_{false};
    std::atomic<std::uint64_t> sync_calls_{0};
    std::atomic<int> next_id_{0};
    std::map<std::uint64_t, StreamState> streams_;
    std::map<void*, EventRec> events_;
    std::map<std::uintptr_t, std::vector<Mark>> marks_;
    std::vector<Violation> violations_;
    std::uint64_t releases_ = 0;
    std::vector<std::pair<Call, std::thread::id>> threads_;
    std::mutex gate_m_;
    std::condition_variable gate_cv_;
    bool gate_open_ = true;
};

}  // namespace whirl::device
