// device::Backend over the whirl HIP runtime wrapper (ARCH-1 D-2; formerly tier::DeviceOps).
// SPDX-License-Identifier: Apache-2.0

#include "device/backend.h"

#include "whirl/device_mem.h"
#include "whirl/hip.h"
#include "whirl/uma_mem.h"

#include <string>

namespace whirl::device {

namespace {

class HipBackend final : public Backend {
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
    bool eventDoneChecked(Event e) override { return hip::eventDone(e); }  // throws hip::Error
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
    hip::MemInfo memInfo() override { return hip::memInfo(); }
    hip::WddmMemInfo wddmMemInfo() override { return hip::wddmMemInfo(); }
    hip::LocalMemSnap localMemSnap() override { return hip::localMemSnap(); }
    hip::UmaMem umaMemory(int device) override { return hip::umaMemory(device); }
    hip::GpuOthers gpuOthers(int device) override { return hip::gpuOthers(device); }
    std::uint64_t pinnedHostBytes() override { return hip::pinnedHostBytes(); }
};

}  // namespace

std::unique_ptr<Backend> makeHipBackend() { return std::make_unique<HipBackend>(); }

}  // namespace whirl::device
