// SPDX-License-Identifier: Apache-2.0
// Host-memory device backend (see null_backend.h). Written for WHIRL.

#include "device/null_backend.h"

#include <cstring>
#include <malloc.h>
#include <new>
#include <stdexcept>

namespace whirl::device {

NullBackend::~NullBackend() = default;

bool NullBackend::isHost(DevPtr p) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = host_.upper_bound(static_cast<std::uintptr_t>(p));
    if (it == host_.begin()) return false;
    --it;
    return p >= it->first && p < it->first + it->second;
}

void NullBackend::copyAsync(DevPtr dst, DevPtr src, std::uint64_t bytes, Stream) {
    if (const std::uint64_t n = ++copy_calls_; n == fail_call_.load()) throw std::runtime_error("injected copyAsync failure");
    if (bytes == 0) return;
    const bool from_host = isHost(src);
    if (const std::uint64_t nth = fail_host_copy_.load(); nth != 0 && from_host && ++host_copies_ == nth) {
        fail_host_copy_ = 0;
        throw std::runtime_error("injected copyAsync failure (pinned host source)");
    }
    std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), static_cast<std::size_t>(bytes));
    copied_ += bytes;
    if (record_.load()) {
        const bool to_host = isHost(dst);
        std::lock_guard<std::mutex> lk(mu_);
        log_.push_back({bytes, from_host, to_host});
    }
}

void NullBackend::download(void* dst, DevPtr src, std::uint64_t bytes) {
    std::memcpy(dst, reinterpret_cast<const void*>(src), static_cast<std::size_t>(bytes));
}

Stream NullBackend::streamCreateNonBlocking() {
    live_streams_ += 1;
    return new int(0);
}

void NullBackend::streamDestroy(Stream s) {
    live_streams_ -= 1;
    delete static_cast<int*>(s);
}

Event NullBackend::eventCreate() {
    live_events_ += 1;
    return new int(0);
}

void NullBackend::eventDestroy(Event e) {
    live_events_ -= 1;
    delete static_cast<int*>(e);
}

bool NullBackend::eventDoneChecked(Event) {
    if (event_error_.load()) throw std::runtime_error("injected asynchronous device error");
    return true;
}

void NullBackend::eventSync(Event) {
    std::unique_lock<std::mutex> lk(gate_m_);
    gate_cv_.wait(lk, [&] { return gate_open_; });
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
    {
        std::lock_guard<std::mutex> lk(mu_);
        live_.erase(p);
        host_.erase(reinterpret_cast<std::uintptr_t>(p));
    }
    live_host_ -= 1;
    _aligned_free(p);
}

DevPtr NullBackend::malloc(std::uint64_t bytes) {
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
    void* q = reinterpret_cast<void*>(p);
    {
        std::lock_guard<std::mutex> lk(mu_);
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
