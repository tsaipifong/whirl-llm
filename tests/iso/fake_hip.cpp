// Link double of the device runtime (src/hip, whirl_hip) for the isolation programs (ARCH-1 8.5; forward, model.weights;
// arch025-lane2 leg 3). The forward sources and their declared dependencies call the HIP layer through
// include/whirl/hip.h (+ device_mem.h / uma_mem.h); the real layer needs the HIP SDK headers and a GPU, so the isolated
// program links these definitions instead. Every call that would reach the GPU throws (no test here may launch,
// allocate or synchronize); releasing nothing (null handles, an empty Graph) is a no-op, as the destructors of a
// host-only object do it. Only the functions the declared sources reference are defined: anything else the forward
// starts to call is LNK2019 here, which is the point (iso_check.py call_* mutants).
// SPDX-License-Identifier: Apache-2.0
#include "whirl/device_mem.h"
#include "whirl/hip.h"
#include "whirl/uma_mem.h"

#include <stdexcept>
#include <string>

namespace whirl::hip {

namespace {

[[noreturn]] void gpu(const char* op) { throw std::logic_error(std::string("iso fake hip: ") + op + " reached the device runtime"); }

}  // namespace

const char* lastErrorOp() { return ""; }
int lastErrorCode() { return 0; }
std::string errorString(int code) { return "iso fake hip error " + std::to_string(code); }
DeviceInfo describeDevice(int) { gpu("describeDevice"); }
void setDevice(int) { gpu("setDevice"); }
int getDevice() { gpu("getDevice"); }
DevPtr malloc(std::size_t) { gpu("malloc"); }
void free(DevPtr p) {
    if (p) gpu("free");
}
void* hostMalloc(std::size_t) { gpu("hostMalloc"); }
void hostFree(void* p) {
    if (p) gpu("hostFree");
}
std::uint64_t pinnedHostBytes() { return 0; }
void upload(DevPtr, const void*, std::size_t) { gpu("upload"); }
void download(void*, DevPtr, std::size_t) { gpu("download"); }
void copyAsync(DevPtr, DevPtr, std::size_t, Stream) { gpu("copyAsync"); }
void copyAnyAsync(DevPtr, DevPtr, std::size_t, Stream) { gpu("copyAnyAsync"); }
void memset(DevPtr, int, std::size_t) { gpu("memset"); }
void sync() { gpu("sync"); }
MemInfo memInfo() { gpu("memInfo"); }
WddmMemInfo wddmMemInfo() { gpu("wddmMemInfo"); }
GpuOthers gpuOthers(int) { gpu("gpuOthers"); }
LocalMemSnap localMemSnap() { gpu("localMemSnap"); }
UmaMem umaMemory(int) { gpu("umaMemory"); }
Stream streamCreate(bool) { gpu("streamCreate"); }
void streamDestroy(Stream s) {
    if (s) gpu("streamDestroy");
}
void streamSync(Stream) { gpu("streamSync"); }
void streamWaitEvent(Stream, Event) { gpu("streamWaitEvent"); }
Event eventCreate(bool) { gpu("eventCreate"); }
void eventDestroy(Event e) {
    if (e) gpu("eventDestroy");
}
void eventRecord(Event, Stream) { gpu("eventRecord"); }
void eventSync(Event) { gpu("eventSync"); }
bool eventDone(Event) { gpu("eventDone"); }
float eventElapsedMs(Event, Event) { gpu("eventElapsedMs"); }

Graph::Graph(Graph&& o) noexcept : graph_(o.graph_), exec_(o.exec_) { o.graph_ = o.exec_ = nullptr; }
Graph& Graph::operator=(Graph&& o) noexcept {
    graph_ = o.graph_;
    exec_ = o.exec_;
    o.graph_ = o.exec_ = nullptr;
    return *this;
}
Graph::~Graph() {
    if (exec_ || graph_) std::terminate();  // an empty Graph only: nothing here captures one
}
void Graph::beginCapture(Stream) { gpu("Graph::beginCapture"); }
Graph Graph::endCapture(Stream) { gpu("Graph::endCapture"); }
void Graph::launch(Stream) const { gpu("Graph::launch"); }

Function Module::getFunction(const char*) const { gpu("Module::getFunction"); }
Function Module::getFunctionOpt(const char*) const { gpu("Module::getFunctionOpt"); }
DevPtr Module::getGlobal(const char*, std::size_t*) const { gpu("Module::getGlobal"); }

void launchRaw(Function, Dim3, Dim3, unsigned, Stream, void**) { gpu("launchRaw"); }

}  // namespace whirl::hip
