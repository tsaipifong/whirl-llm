// whirl-iso-contract-device-tests: the module contract.device (src/device/backend.h, src/device/backend_hip.cpp, device::Backend over the HIP layer) built with its
// declared dependency, the HIP layer, as the link double tests/iso/fake_hip.cpp (ARCH-1 8.5 isolation, arch025-iso4). No test existed (the CPU tests run on the null
// backend). The double throws "iso fake hip: <op> reached the device runtime" from every call that would reach a GPU, so each case is a forwarding pin: the
// Backend call reaches exactly the HIP-layer function of that name, with no GPU involved. Releasing nothing (null handles) is a no-op, as it is in the product.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "device/backend.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

namespace dev = whirl::device;

// the HIP-layer op a Backend call forwarded to ("" = it did not reach the HIP layer, or threw something else)
std::string reaches(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::logic_error& e) {
        const std::string w = e.what();
        const std::string head = "iso fake hip: ", tail = " reached the device runtime";
        if (w.rfind(head, 0) == 0 && w.size() > head.size() + tail.size()) return w.substr(head.size(), w.size() - head.size() - tail.size());
    }
    return {};
}

void testForwarding() {
    dev::Backend& b = dev::hipBackend();
    char host[8];
    void* const h = host;  // a non-null stream / event / host pointer
    dev::Stream s = h;
    dev::Event e = h;
    ISO_CHECK(reaches([&] { b.copyAsync(1, 2, 8, s); }) == "copyAnyAsync");
    ISO_CHECK(reaches([&] { b.download(host, 2, 8); }) == "download");
    ISO_CHECK(reaches([&] { (void)b.streamCreateNonBlocking(); }) == "streamCreate");
    ISO_CHECK(reaches([&] { b.streamDestroy(s); }) == "streamDestroy");
    ISO_CHECK(reaches([&] { b.streamSync(s); }) == "streamSync");
    ISO_CHECK(reaches([&] { b.streamWaitEvent(s, e); }) == "streamWaitEvent");
    ISO_CHECK(reaches([&] { (void)b.eventCreate(); }) == "eventCreate");
    ISO_CHECK(reaches([&] { b.eventDestroy(e); }) == "eventDestroy");
    ISO_CHECK(reaches([&] { b.eventRecord(e, s); }) == "eventRecord");
    ISO_CHECK(reaches([&] { (void)b.eventDone(e); }) == "eventDone");  // only a hip::Error reads as "complete"; anything else propagates
    ISO_CHECK(reaches([&] { (void)b.eventDoneChecked(e); }) == "eventDone");
    ISO_CHECK(reaches([&] { b.eventSync(e); }) == "eventSync");
    ISO_CHECK(reaches([&] { (void)b.hostMalloc(8); }) == "hostMalloc");
    ISO_CHECK(reaches([&] { b.hostFree(host); }) == "hostFree");
    ISO_CHECK(reaches([&] { (void)b.malloc(8); }) == "malloc");
    ISO_CHECK(reaches([&] { b.free(1); }) == "free");
    ISO_CHECK(reaches([&] { b.setDevice(1); }) == "setDevice");
    ISO_CHECK(reaches([&] { (void)b.memInfo(); }) == "memInfo");
    ISO_CHECK(reaches([&] { (void)b.wddmMemInfo(); }) == "wddmMemInfo");
    ISO_CHECK(reaches([&] { (void)b.localMemSnap(); }) == "localMemSnap");
    ISO_CHECK(reaches([&] { (void)b.umaMemory(0); }) == "umaMemory");
    ISO_CHECK(reaches([&] { (void)b.gpuOthers(0); }) == "gpuOthers");
}

void testNoOpsAndText() {
    dev::Backend& b = dev::hipBackend();
    b.streamDestroy(nullptr);  // releasing a null handle reaches nothing (the HIP layer ignores it)
    b.eventDestroy(nullptr);
    b.hostFree(nullptr);
    b.free(0);
    ISO_CHECK(b.pinnedHostBytes() == 0);                                         // forwards to hip::pinnedHostBytes
    ISO_CHECK(std::string(b.lastErrorString()) == ": iso fake hip error 0");     // "<op>: <error text>" of the last failing HIP call
}

void testFactories() {
    ISO_CHECK(&dev::hipBackend() == &dev::hipBackend());  // the process's one backend
    const std::unique_ptr<dev::Backend> own = dev::makeHipBackend();
    ISO_CHECK(own != nullptr && own.get() != &dev::hipBackend());  // makeHipBackend gives a fresh forwarding object
    ISO_CHECK(reaches([&] { (void)own->malloc(8); }) == "malloc");
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("contract.device", {{"forwarding", testForwarding}, {"no_ops_and_text", testNoOpsAndText}, {"factories", testFactories}}, argc, argv);
}
