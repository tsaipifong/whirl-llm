// DeviceOps over the whirl HIP runtime wrapper.
// SPDX-License-Identifier: Apache-2.0

#include "device_ops.h"

#include "whirl/hip.h"

#include <string>

namespace whirl::tier {

namespace {

class HipDeviceOps final : public DeviceOps {
public:
    void copyAsync(DevPtr dst, DevPtr src, std::uint64_t bytes, Stream s) override {
        hip::copyAnyAsync(dst, src, bytes, s);
    }
    void download(void* dst, DevPtr src, std::uint64_t bytes) override { hip::download(dst, src, bytes); }
    Stream streamCreateNonBlocking() override { return hip::streamCreate(true); }
    void streamDestroy(Stream s) override { hip::streamDestroy(s); }
    void streamSync(Stream s) override { hip::streamSync(s); }
    void streamWaitEvent(Stream s, Event e) override { hip::streamWaitEvent(s, e); }
    Event eventCreate() override { return hip::eventCreate(false); }
    void eventDestroy(Event e) override { hip::eventDestroy(e); }
    void eventRecord(Event e, Stream s) override { hip::eventRecord(e, s); }
    bool eventDone(Event e) override {
        try {
            return hip::eventDone(e);
        } catch (const hip::Error&) {
            return true;
        }
    }
    void eventSync(Event e) override { hip::eventSync(e); }
    void* hostMalloc(std::uint64_t bytes) override { return hip::hostMalloc(bytes); }
    void hostFree(void* p) override { hip::hostFree(p); }
    DevPtr malloc(std::uint64_t bytes) override { return hip::malloc(bytes); }
    void free(DevPtr p) override { hip::free(p); }
    void setDevice(int device) override { hip::setDevice(device); }
    const char* lastErrorString() const override {
        thread_local std::string s;
        s = std::string(hip::lastErrorOp()) + ": " + hip::errorString(hip::lastErrorCode());
        return s.c_str();
    }
};

}  // namespace

std::unique_ptr<DeviceOps> makeHipDeviceOps() { return std::make_unique<HipDeviceOps>(); }

}  // namespace whirl::tier
