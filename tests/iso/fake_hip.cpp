// Link double of the device runtime (src/hip, whirl_hip) for the isolation programs (ARCH-1 8.5; forward, model.weights;
// arch025-lane2 leg 3; model.core, leg 4). The forward sources and their declared dependencies call the HIP layer through
// include/whirl/hip.h (+ device_mem.h / uma_mem.h); the real layer needs the HIP SDK headers and a GPU, so the isolated
// program links these definitions instead. Every call that would reach the GPU throws (no test here may launch,
// allocate or synchronize); releasing nothing (null handles, an empty Graph) is a no-op, as the destructors of a
// host-only object do it. Only the functions the declared sources reference are defined: anything else the forward
// starts to call is LNK2019 here, which is the point (iso_check.py call_* mutants).
// SPDX-License-Identifier: Apache-2.0
#include "whirl/device_mem.h"
#include "whirl/hip.h"
#include "whirl/uma_mem.h"

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace whirl::hip {

// arch025-n8b3: one recorded launch (the test programs that read the recorder declare the same struct)
namespace isofake {
struct Launch {
    Function f;
    Dim3 grid, block;
};
}  // namespace isofake

namespace {

[[noreturn]] void gpu(const char* op) { throw std::logic_error(std::string("iso fake hip: ") + op + " reached the device runtime"); }

// arch025-gmsurv: the opt-in allocation recorder (isofake::recordAllocs). Off (the default) every device call throws
// as before. On: malloc hands out fake addresses (never dereferenced: nothing here maps memory) and records the
// requested sizes in order; free / memset accept an address it handed out (also after recording stopped, so a Model
// destroyed later releases its ledger). Only the functions an allocation-ledger test needs; launches, uploads and
// syncs still throw.
bool g_record = false;
std::vector<std::uint64_t> g_sizes;
std::set<DevPtr> g_live;
DevPtr g_next = 0x100000;

bool handedOut(DevPtr p) { return g_live.count(p) != 0; }

// arch025-n8b3: the opt-in launch recorder (isofake::recordLaunches) and fake kernel handles (isofake::fakeFunctions), for
// the per-file forward manifests (forward.attn / forward.ffn ...: code that only launches is observed through what it
// launches; the host-side form of the launch table of tools/impact/work_equal.py, not a GPU result). Off (the default)
// launchRaw and Module::getFunction throw as before. Recording: launchRaw records the kernel handle, grid and block and
// returns (no argument is read). Fake functions: getFunction / getFunctionOpt return a distinct non-null handle per name
// (never called: launchRaw only records it).
bool g_rec_launch = false;
bool g_fake_fn = false;
std::vector<isofake::Launch> g_launches;
std::vector<std::string> g_fn_names;

Function fakeFunction(const char* name) {
    for (std::size_t i = 0; i < g_fn_names.size(); ++i)
        if (g_fn_names[i] == name) return reinterpret_cast<Function>(0x7000 + 0x10 * i);
    g_fn_names.emplace_back(name);
    return reinterpret_cast<Function>(0x7000 + 0x10 * (g_fn_names.size() - 1));
}

}  // namespace

namespace isofake {
void recordAllocs(bool on) {
    g_record = on;
    if (on) g_sizes.clear();
}
std::vector<std::uint64_t> recordedSizes() { return g_sizes; }
void recordLaunches(bool on) {
    g_rec_launch = on;
    if (on) g_launches.clear();
}
std::vector<isofake::Launch> recordedLaunches() { return g_launches; }
void fakeFunctions(bool on) { g_fake_fn = on; }
}  // namespace isofake

const char* lastErrorOp() { return ""; }
int lastErrorCode() { return 0; }
std::string errorString(int code) { return "iso fake hip error " + std::to_string(code); }
DeviceInfo describeDevice(int) { gpu("describeDevice"); }
void setDevice(int) { gpu("setDevice"); }
int getDevice() { gpu("getDevice"); }
DevPtr malloc(std::size_t n) {
    if (!g_record) gpu("malloc");
    const DevPtr p = g_next;
    g_next += (static_cast<DevPtr>(n) + 0xFFFF) / 0x10000 * 0x10000 + 0x10000;  // distinct, increasing, never overlapping
    g_sizes.push_back(n);
    g_live.insert(p);
    return p;
}
void free(DevPtr p) {
    if (p && handedOut(p)) {
        g_live.erase(p);
        return;
    }
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
void memset(DevPtr p, int, std::size_t) {
    if (!handedOut(p)) gpu("memset");
}
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

Function Module::getFunction(const char* name) const {
    if (!g_fake_fn) gpu("Module::getFunction");
    return fakeFunction(name);
}
Function Module::getFunctionOpt(const char* name) const {
    if (!g_fake_fn) gpu("Module::getFunctionOpt");
    return fakeFunction(name);
}
DevPtr Module::getGlobal(const char*, std::size_t*) const { gpu("Module::getGlobal"); }

void launchRaw(Function f, Dim3 grid, Dim3 block, unsigned, Stream, void**) {
    if (!g_rec_launch) gpu("launchRaw");
    g_launches.push_back({f, grid, block});
}

// arch025-lane2 leg 4 (model.core): the code-object loading of the kernel table (Model::loadKernels), the gfx target
// names and the UMA check of the loader. Loading a code object reaches the GPU; an unloaded Module (every Model member
// before Model::load) is released as a no-op; archName / archFor are the HIP layer's (hip_runtime.cpp) and a test of the
// layers above has no device to name.
Module& Module::operator=(Module&& o) noexcept {
    std::swap(handle_, o.handle_);
    std::swap(arch_, o.arch_);
    return *this;
}
Module::~Module() {
    if (handle_) std::terminate();  // never loaded here
}
Module Module::loadEmbedded() { gpu("Module::loadEmbedded"); }
Module Module::loadData(const void*, Arch) { gpu("Module::loadData"); }
Arch archFor(std::string_view) { gpu("archFor"); }
const char* archName(Arch) { gpu("archName"); }
bool umaCheckOn() { gpu("umaCheckOn"); }

}  // namespace whirl::hip
