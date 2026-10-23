// RAII handles over device::Backend (arch024 M1): device buffers, pinned host buffers, events and streams that
// release themselves. Move-only; every handle carries the Backend that made it (no global hip:: calls), so the
// same code runs against the HIP backend and against NullBackend in tests.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Shape after llama.cpp's ggml-cpp.h deleters and ggml_cuda_pool_alloc (MIT), with a
// Backend pointer per handle and moves that empty the source (progress/arch024-S2.PRIOR_ART.md).
//
// Rules:
// - The destructor and reset() release through the Backend (free / hostFree / eventDestroy / streamDestroy).
//   They never wait: a buffer the GPU may still read is released only after its fence was waited for
//   (invariant F1; EventH::waitIfPending is the one place that waits).
// - A moved-from handle is empty; destroying or resetting an empty handle calls nothing.
// - Move-assign releases what the target held first, then takes over.
// - No release(): ownership never leaves a handle as a raw value.

#pragma once

#include "device/backend.h"

#include <cstdint>
#include <utility>

namespace whirl::device {

namespace detail {

// One owned raw value plus its Backend; Free(Backend&, T) releases it.
template <class T, void (*Free)(Backend&, T)>
class Handle {
public:
    Handle() noexcept = default;
    Handle(Backend& b, T v) noexcept : b_(&b), v_(v) {}
    ~Handle() { reset(); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& o) noexcept : b_(o.b_), v_(o.v_) {
        o.b_ = nullptr;
        o.v_ = T{};
    }
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) {
            reset();
            b_ = std::exchange(o.b_, nullptr);
            v_ = std::exchange(o.v_, T{});
        }
        return *this;
    }

    T get() const noexcept { return v_; }
    Backend* backend() const noexcept { return b_; }
    explicit operator bool() const noexcept { return v_ != T{}; }
    // Releases the value (if any) and leaves the handle empty.
    void reset() noexcept {
        if (b_ != nullptr && v_ != T{}) Free(*b_, v_);
        b_ = nullptr;
        v_ = T{};
    }

private:
    Backend* b_ = nullptr;
    T v_{};
};

inline void freeDev(Backend& b, DevPtr p) { b.free(p); }
inline void freeHost(Backend& b, void* p) { b.hostFree(p); }
inline void destroyEvent(Backend& b, Event e) { b.eventDestroy(e); }
inline void destroyStream(Backend& b, Stream s) { b.streamDestroy(s); }

}  // namespace detail

// Device memory: Backend::malloc / free.
class DevBuf {
public:
    DevBuf() noexcept = default;
    static DevBuf alloc(Backend& b, std::uint64_t bytes) { return DevBuf(b, b.malloc(bytes), bytes); }
    DevPtr get() const noexcept { return h_.get(); }
    std::uint64_t bytes() const noexcept { return h_ ? bytes_ : 0; }
    explicit operator bool() const noexcept { return static_cast<bool>(h_); }
    void reset() noexcept {
        h_.reset();
        bytes_ = 0;
    }

private:
    DevBuf(Backend& b, DevPtr p, std::uint64_t n) noexcept : h_(b, p), bytes_(n) {}
    detail::Handle<DevPtr, &detail::freeDev> h_;
    std::uint64_t bytes_ = 0;
};

// Pinned host memory: Backend::hostMalloc / hostFree.
class PinnedBuf {
public:
    PinnedBuf() noexcept = default;
    static PinnedBuf alloc(Backend& b, std::uint64_t bytes) { return PinnedBuf(b, b.hostMalloc(bytes), bytes); }
    void* get() const noexcept { return h_.get(); }
    DevPtr addr() const noexcept { return reinterpret_cast<DevPtr>(h_.get()); }
    std::uint64_t bytes() const noexcept { return h_ ? bytes_ : 0; }
    explicit operator bool() const noexcept { return static_cast<bool>(h_); }
    void reset() noexcept {
        h_.reset();
        bytes_ = 0;
    }

private:
    PinnedBuf(Backend& b, void* p, std::uint64_t n) noexcept : h_(b, p), bytes_(n) {}
    detail::Handle<void*, &detail::freeHost> h_;
    std::uint64_t bytes_ = 0;
};

// Event without timing: Backend::eventCreate / eventDestroy.
class EventH {
public:
    EventH() noexcept = default;
    static EventH create(Backend& b) { return EventH(b, b.eventCreate()); }
    Event get() const noexcept { return h_.get(); }
    explicit operator bool() const noexcept { return static_cast<bool>(h_); }
    void reset() noexcept { h_.reset(); }
    // F1: the host waits for the work before this event (if any is still pending). Every "wait before free"
    // goes through here (govFreeCache's hand-written line before arch024).
    void waitIfPending() const {
        if (Backend* b = h_.backend(); b != nullptr && h_ && !b->eventDone(h_.get())) b->eventSync(h_.get());
    }

private:
    EventH(Backend& b, Event e) noexcept : h_(b, e) {}
    detail::Handle<Event, &detail::destroyEvent> h_;
};

// Non-blocking stream: Backend::streamCreateNonBlocking / streamDestroy.
class StreamH {
public:
    StreamH() noexcept = default;
    static StreamH create(Backend& b) { return StreamH(b, b.streamCreateNonBlocking()); }
    Stream get() const noexcept { return h_.get(); }
    explicit operator bool() const noexcept { return static_cast<bool>(h_); }
    void reset() noexcept { h_.reset(); }

private:
    StreamH(Backend& b, Stream s) noexcept : h_(b, s) {}
    detail::Handle<Stream, &detail::destroyStream> h_;
};

}  // namespace whirl::device
