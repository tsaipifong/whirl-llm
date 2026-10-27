// SPDX-License-Identifier: Apache-2.0
// Host-memory device backend (see null_backend.h). Written for WHIRL.

#include "device/null_backend.h"
#include "platform/env.h"

#include <algorithm>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <new>
#include <stdexcept>

namespace whirl::device {

#ifdef WHIRL_LEAKCHECK
// safe025: leak check at the end of every NullBackend's life. Most tests build their own NullBackend, so the destructor
// is the end of the test; the tier tests share a few (review C2c), there a leak is found but not tied to one test.
// Anything still alive then was never freed through the backend (the VRAM / pinned-memory leak of the real backend).
// The ledger prints once at process exit and turns a clean exit into code 97.
namespace {
std::atomic<long> g_leaks{0};
std::atomic<long> g_backends{0};
struct LeakReport {
    ~LeakReport() {
        if (!platform::env::isSet("WHIRL_LEAKCHECK_QUIET"))
            std::fprintf(stderr, "LEAKCHECK: %ld NullBackend instances checked, %ld leaked\n", g_backends.load(), g_leaks.load());
        // _Exit skips the CRT's stream flush: flush first, or a test log redirected to a file (fully buffered) loses its
        // tail - the failing test names and the 'N passed, M failed' summary (safe025 review A1).
        if (g_leaks.load() != 0) {
            std::fflush(nullptr);
            std::_Exit(97);
        }
    }
} g_report;
}  // namespace

NullBackend::~NullBackend() {
    ++g_backends;
    if (allow_leak_.load()) return;
    const std::size_t a = liveAllocations();
    const long h = live_host_.load(), s = live_streams_.load(), e = live_events_.load();
    if (a != 0 || h != 0 || s != 0 || e != 0) {
        ++g_leaks;
        std::fprintf(stderr, "LEAKCHECK: NullBackend destroyed with %zu allocations (%ld pinned host), %ld streams, %ld events alive\n", a, h, s, e);
    }
}
long NullBackend::leaksFound() { return g_leaks.load(); }
#else
NullBackend::~NullBackend() = default;
long NullBackend::leaksFound() { return 0; }
#endif

bool NullBackend::isHost(DevPtr p) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = host_.upper_bound(static_cast<std::uintptr_t>(p));
    if (it == host_.begin()) return false;
    --it;
    return p >= it->first && p < it->first + it->second;
}

void NullBackend::copyAsync(DevPtr dst, DevPtr src, std::uint64_t bytes, Stream s) {
    noteThread(Call::copy_async);
    if (const std::uint64_t n = ++copy_calls_; n == fail_call_.load()) throw std::runtime_error("injected copyAsync failure");
    if (bytes == 0) return;
    const bool from_host = isHost(src);
    if (const std::uint64_t nth = fail_host_copy_.load(); nth != 0 && from_host && ++host_copies_ == nth) {
        fail_host_copy_ = 0;
        throw std::runtime_error("injected copyAsync failure (pinned host source)");
    }
    std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), static_cast<std::size_t>(bytes));
    copied_ += bytes;
    if (track_.load()) {
        std::lock_guard<std::mutex> lk(mu_);
        StreamState& st = streams_[idOf(s)];
        const Mark m{idOf(s), ++st.issued};
        // the source too: a spill is the GPU still reading the checkpoint / page buffers
        for (const std::uintptr_t a : {allocOf(static_cast<std::uintptr_t>(dst)), allocOf(static_cast<std::uintptr_t>(src))})
            if (a != 0) marks_[a].push_back(m);
    }
    if (record_.load()) {
        const bool to_host = isHost(dst);
        std::lock_guard<std::mutex> lk(mu_);
        log_.push_back({bytes, from_host, to_host});
    }
}

void NullBackend::download(void* dst, DevPtr src, std::uint64_t bytes) {
    noteThread(Call::download);
    std::memcpy(dst, reinterpret_cast<const void*>(src), static_cast<std::size_t>(bytes));
}

Stream NullBackend::streamCreateNonBlocking() {
    noteThread(Call::stream_create);
    live_streams_ += 1;
    return new int(++next_id_);  // the stream id of tracking mode (0 is the null stream)
}

void NullBackend::streamDestroy(Stream s) {
    noteThread(Call::stream_destroy);
    if (s != nullptr) {
        std::lock_guard<std::mutex> lk(rec_mu_);
        dead_streams_.insert(idOf(s));
    }
    live_streams_ -= 1;
    delete static_cast<int*>(s);
}

void NullBackend::streamSync(Stream s) {
    noteThread(Call::stream_sync);
    if (!track_.load()) return;
    std::lock_guard<std::mutex> lk(mu_);
    sync_calls_ += 1;
    StreamState& st = streams_[idOf(s)];
    const std::vector<Mark> deps = st.deps;
    advance({idOf(s), st.issued}, deps);
}

void NullBackend::streamWaitEvent(Stream s, Event e) {
    noteThread(Call::stream_wait_event);
    if (!track_.load()) return;
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = events_.find(e);
    if (it == events_.end()) return;  // never recorded: nothing to wait for
    StreamState& st = streams_[idOf(s)];
    st.deps.push_back(it->second.at);
    st.deps.insert(st.deps.end(), it->second.deps.begin(), it->second.deps.end());
}

void NullBackend::eventRecord(Event e, Stream s) {
    noteThread(Call::event_record);
    {
        std::lock_guard<std::mutex> lk(rec_mu_);
        rec_stream_[e] = idOf(s);
    }
    if (!track_.load()) return;
    std::lock_guard<std::mutex> lk(mu_);
    const StreamState& st = streams_[idOf(s)];
    events_[e] = EventRec{{idOf(s), st.issued}, st.deps};
}

void NullBackend::noteEventUse(Event e) {
    std::lock_guard<std::mutex> lk(rec_mu_);
    const auto it = rec_stream_.find(e);
    if (it != rec_stream_.end() && it->second != 0 && dead_streams_.count(it->second) != 0) dead_stream_uses_ += 1;
}

bool NullBackend::eventDone(Event e) {
    noteThread(Call::event_done);
    noteEventUse(e);
    if (!track_.load()) return true;
    std::lock_guard<std::mutex> lk(mu_);
    return doneLocked(e);
}

bool NullBackend::doneLocked(Event e) {
    const auto it = events_.find(e);
    if (it == events_.end()) return true;  // never recorded
    // done when its own stream got there and so did everything that stream waited for before the record
    if (it->second.at.second > streams_[it->second.at.first].done) return false;
    for (const Mark& d : it->second.deps)
        if (d.second > streams_[d.first].done) return false;
    return true;
}

Event NullBackend::eventCreate() {
    noteThread(Call::event_create);
    live_events_ += 1;
    return new int(0);
}

void NullBackend::eventDestroy(Event e) {
    noteThread(Call::event_destroy);
    {
        std::lock_guard<std::mutex> lk(rec_mu_);
        rec_stream_.erase(e);
    }
    if (track_.load()) {
        std::lock_guard<std::mutex> lk(mu_);
        events_.erase(e);
    }
    live_events_ -= 1;
    delete static_cast<int*>(e);
}

bool NullBackend::eventDoneChecked(Event e) {
    noteThread(Call::event_done_checked);
    noteEventUse(e);
    if (event_error_.load()) throw std::runtime_error("injected asynchronous device error");
    if (!track_.load()) return true;
    std::lock_guard<std::mutex> lk(mu_);
    return doneLocked(e);
}

void NullBackend::eventSync(Event e) {
    noteThread(Call::event_sync);
    noteEventUse(e);
    {
        std::unique_lock<std::mutex> lk(gate_m_);
        gate_cv_.wait(lk, [&] { return gate_open_; });
    }
    if (!track_.load()) return;
    std::lock_guard<std::mutex> lk(mu_);
    sync_calls_ += 1;
    if (const auto it = events_.find(e); it != events_.end()) {
        const EventRec r = it->second;
        advance(r.at, r.deps);
    }
}

void NullBackend::complete(Event e) {
    std::lock_guard<std::mutex> lk(mu_);
    if (const auto it = events_.find(e); it != events_.end()) {
        const EventRec r = it->second;
        advance(r.at, r.deps);
    }
}

void NullBackend::completeAll() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [id, st] : streams_) st.done = st.issued;
}

void NullBackend::advance(const Mark& m, const std::vector<Mark>& deps) {
    StreamState& st = streams_[m.first];
    st.done = std::max(st.done, m.second);
    // deps are flattened when streamWaitEvent adds them (an event's own deps come along), so one level suffices
    for (const Mark& d : deps) {
        StreamState& ds = streams_[d.first];
        ds.done = std::max(ds.done, d.second);
    }
}

std::uintptr_t NullBackend::allocOf(std::uintptr_t p) const {
    for (const auto* m : {&dev_, &host_}) {
        auto it = m->upper_bound(p);
        if (it == m->begin()) continue;
        --it;
        if (p >= it->first && p < it->first + std::max<std::uint64_t>(it->second, 1)) return it->first;
    }
    return 0;
}

void NullBackend::checkRelease(std::uintptr_t start, bool host) {
    // under mu_, tracking on
    const std::uint64_t order = ++releases_;
    const auto it = marks_.find(start);
    if (it == marks_.end()) return;
    for (const Mark& m : it->second)
        if (m.second > streams_[m.first].done) {
            violations_.push_back({m.first, m.second, order, host});
            break;  // one violation per release
        }
    marks_.erase(it);
}

void NullBackend::trackPending(bool on) {
    std::lock_guard<std::mutex> lk(mu_);
    track_ = on;
    sync_calls_ = 0;
    streams_.clear();
    events_.clear();
    marks_.clear();
    violations_.clear();
    releases_ = 0;
}

std::size_t NullBackend::pendingFreeViolations() const {
    std::lock_guard<std::mutex> lk(mu_);
    return violations_.size();
}

std::vector<NullBackend::Violation> NullBackend::violations() const {
    std::lock_guard<std::mutex> lk(mu_);
    return violations_;
}

void NullBackend::noteThread(Call c) {
    if (!rec_threads_.load()) return;
    std::lock_guard<std::mutex> lk(mu_);
    threads_.push_back({c, std::this_thread::get_id()});
}

const char* NullBackend::callName(Call c) {
    static constexpr const char* names[] = {"copyAsync", "download", "streamCreateNonBlocking", "streamDestroy",
                                            "streamSync", "streamWaitEvent", "eventCreate", "eventDestroy",
                                            "eventRecord", "eventDone", "eventDoneChecked", "eventSync",
                                            "hostMalloc", "hostFree", "malloc", "free", "setDevice", "memInfo",
                                            "wddmMemInfo", "localMemSnap", "umaMemory", "gpuOthers",
                                            "pinnedHostBytes"};
    static_assert(std::size(names) == static_cast<std::size_t>(Call::pinned_host_bytes) + 1);
    return names[static_cast<std::size_t>(c)];
}

void NullBackend::recordThreads(bool on) {
    std::lock_guard<std::mutex> lk(mu_);
    rec_threads_ = on;
    if (on) threads_.clear();
}

std::vector<std::pair<NullBackend::Call, std::thread::id>> NullBackend::callThreads() const {
    std::lock_guard<std::mutex> lk(mu_);
    return threads_;
}

void NullBackend::holdEventSync() {
    std::lock_guard<std::mutex> lk(gate_m_);
    gate_open_ = false;
}

void NullBackend::releaseEventSync() {
    {
        std::lock_guard<std::mutex> lk(gate_m_);
        gate_open_ = true;
    }
    gate_cv_.notify_all();
}

void* NullBackend::hostMalloc(std::uint64_t bytes) {
    noteThread(Call::host_malloc);
    void* p = _aligned_malloc(bytes ? static_cast<std::size_t>(bytes) : 1, 4096);
    if (p == nullptr) throw std::bad_alloc();
    std::memset(p, 0, static_cast<std::size_t>(bytes));
    std::lock_guard<std::mutex> lk(mu_);
    live_.insert(p);
    host_[reinterpret_cast<std::uintptr_t>(p)] = bytes;
    live_host_ += 1;
    return p;
}

void NullBackend::hostFree(void* p) {
    if (p == nullptr) return;  // like hipHostFree(nullptr)
    noteThread(Call::host_free);
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (track_.load()) checkRelease(reinterpret_cast<std::uintptr_t>(p), true);
        live_.erase(p);
        host_.erase(reinterpret_cast<std::uintptr_t>(p));
    }
    live_host_ -= 1;
    _aligned_free(p);
}

DevPtr NullBackend::malloc(std::uint64_t bytes) {
    noteThread(Call::malloc);
    if (const std::uint64_t nth = fail_malloc_.load(); nth != 0 && ++mallocs_ == nth) {
        fail_malloc_ = 0;
        throw std::runtime_error("injected malloc failure");
    }
    void* p = _aligned_malloc(bytes ? static_cast<std::size_t>(bytes) : 1, 4096);
    if (p == nullptr) throw std::bad_alloc();
    std::memset(p, 0, static_cast<std::size_t>(bytes));
    std::lock_guard<std::mutex> lk(mu_);
    live_.insert(p);
    dev_[reinterpret_cast<std::uintptr_t>(p)] = bytes;
    dev_bytes_.fetch_add(bytes);
    return reinterpret_cast<DevPtr>(p);
}

void NullBackend::free(DevPtr p) {
    if (p == 0) return;  // like hipFree(nullptr)
    noteThread(Call::free);
    void* q = reinterpret_cast<void*>(p);
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (track_.load()) checkRelease(static_cast<std::uintptr_t>(p), false);
        live_.erase(q);
        if (const auto it = dev_.find(reinterpret_cast<std::uintptr_t>(q)); it != dev_.end()) {
            dev_bytes_.fetch_sub(it->second);
            dev_.erase(it);
        }
    }
    _aligned_free(q);
}

std::size_t NullBackend::liveAllocations() const {
    std::lock_guard<std::mutex> lk(mu_);
    return live_.size();
}

void NullBackend::recordCopies(bool on) {
    std::lock_guard<std::mutex> lk(mu_);
    record_ = on;
    if (on) log_.clear();
}

std::vector<NullBackend::Copy> NullBackend::copies() const {
    std::lock_guard<std::mutex> lk(mu_);
    return log_;
}

}  // namespace whirl::device
