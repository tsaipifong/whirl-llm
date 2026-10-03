// Device memory / stream / event operations used by the server engine and
// the host prefix-cache tiers, behind an interface so that the scheduling,
// prefix-cache and tier logic can run against a host-memory mock in tests
// (no GPU). The production implementation forwards to whirl/hip.h.
// SPDX-License-Identifier: Apache-2.0
//
// Addresses are plain uint64 values (hip::DevPtr): device pointers or
// pinned host pointers; copyAsync infers the direction (hipMemcpyDefault).

#pragma once

#include <cstdint>
#include <memory>

namespace whirl::tier {

using DevPtr = std::uint64_t;
using Stream = void*;  // opaque (hipStream_t for the HIP implementation)
using Event = void*;   // opaque (hipEvent_t)

class DeviceOps {
public:
    virtual ~DeviceOps() = default;

    // Any address pair (device or pinned host), asynchronous on stream s.
    virtual void copyAsync(DevPtr dst, DevPtr src, std::uint64_t bytes, Stream s) = 0;
    // Synchronous device -> host copy (orders after all prior work on the
    // legacy null stream, like hipMemcpy).
    virtual void download(void* dst, DevPtr src, std::uint64_t bytes) = 0;

    virtual Stream streamCreateNonBlocking() = 0;
    virtual void streamDestroy(Stream s) = 0;
    virtual void streamSync(Stream s) = 0;
    virtual void streamWaitEvent(Stream s, Event e) = 0;

    // Events without timing.
    virtual Event eventCreate() = 0;
    virtual void eventDestroy(Event e) = 0;
    virtual void eventRecord(Event e, Stream s) = 0;
    // true when complete (also for a never-recorded event and on query errors)
    virtual bool eventDone(Event e) = 0;
    virtual void eventSync(Event e) = 0;

    // Pinned host memory (throws on failure).
    virtual void* hostMalloc(std::uint64_t bytes) = 0;
    virtual void hostFree(void* p) = 0;

    // Device memory (throws on failure).
    virtual DevPtr malloc(std::uint64_t bytes) = 0;
    virtual void free(DevPtr p) = 0;

    // Makes `device` current on the calling thread (IO threads).
    virtual void setDevice(int device) = 0;

    // Description of the last failing operation (for log lines).
    virtual const char* lastErrorString() const = 0;
};

// Forwarding implementation over whirl/hip.h (src/tier/device_ops_hip.cpp).
std::unique_ptr<DeviceOps> makeHipDeviceOps();

}  // namespace whirl::tier
