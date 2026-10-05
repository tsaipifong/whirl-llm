// HIP runtime wrapper: devices, memory, streams, events, graphs, embedded
// code objects and kernel launch.
// SPDX-License-Identifier: Apache-2.0
//
// The header is free of HIP SDK includes so that every other translation
// unit compiles with plain MSVC; only src/hip/hip_runtime.cpp sees
// <hip/hip_runtime_api.h>. Device pointers are carried as uint64_t
// addresses (DevPtr) so offsets are plain arithmetic.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace whirl::hip {

using DevPtr = std::uint64_t;
using Stream = void*;    // hipStream_t (nullptr = legacy null stream)
using Event = void*;     // hipEvent_t
using Function = void*;  // hipFunction_t

// Thrown by every wrapper on a non-zero hipError_t.
class Error : public std::runtime_error {
public:
    Error(std::string op, int code, std::string message);
    const std::string& op() const { return op_; }
    int code() const { return code_; }

private:
    std::string op_;
    int code_;
};

// Last failing operation name / status, kept for diagnostics.
const char* lastErrorOp();
int lastErrorCode();
std::string errorString(int code);

struct DeviceInfo {
    int index = -1;
    std::string name;      // marketing name, e.g. "AMD Radeon AI PRO R9700"
    std::string gcn_arch;  // e.g. "gfx1201" or "gfx1151"
    std::size_t total_mem = 0;
    int compute_units = 0;
    int warp_size = 0;
    int clock_khz = 0;
    int pci_bus = 0;
    int pci_device = 0;
    bool integrated = false;
    std::size_t shared_mem_per_block = 0;
    int max_threads_per_block = 0;
};

int deviceCount();
DeviceInfo describeDevice(int index);
std::vector<DeviceInfo> listDevices();
// Device selection by index ("1"), gfx arch ("gfx1151", prefix match) or a
// case-insensitive substring of the name ("R9700"). Returns -1 if none.
int findDevice(std::string_view spec);
void setDevice(int index);
int getDevice();

// Code object family. gfx1151 (RDNA 3.5) has its own build; every other
// device uses the gfx1201 build.
enum class Arch { gfx1201, gfx1151 };
Arch archFor(std::string_view gcn_arch);
const char* archName(Arch a);

// ---- memory
DevPtr malloc(std::size_t bytes);
void free(DevPtr p);
void* hostMalloc(std::size_t bytes);  // pinned host memory
void hostFree(void* p);
void upload(DevPtr dst, const void* src, std::size_t bytes);
void download(void* dst, DevPtr src, std::size_t bytes);
void uploadAsync(DevPtr dst, const void* src, std::size_t bytes, Stream s);
void downloadAsync(void* dst, DevPtr src, std::size_t bytes, Stream s);
void copyAsync(DevPtr dst, DevPtr src, std::size_t bytes, Stream s);  // device to device
// Any pointer pair (device or pinned host); the runtime infers the direction.
void copyAnyAsync(DevPtr dst, DevPtr src, std::size_t bytes, Stream s);
void memset(DevPtr dst, int value, std::size_t bytes);
void memsetAsync(DevPtr dst, int value, std::size_t bytes, Stream s);
void sync();
struct MemInfo {
    std::size_t free = 0, total = 0;
};
MemInfo memInfo();
// This process's WDDM video memory numbers for the current device (DXGI
// IDXGIAdapter3::QueryVideoMemoryInfo). Going over local_budget makes the OS
// demote allocations to system memory (they then show up as non-local /
// "Shared Usage" and every access crosses PCIe). ok = false off Windows or if
// the adapter cannot be matched by LUID.
struct WddmMemInfo {
    bool ok = false;
    std::uint64_t local_budget = 0, local_usage = 0, nonlocal_budget = 0, nonlocal_usage = 0;
};
WddmMemInfo wddmMemInfo();

// ---- streams and events
// Blocking streams order with the legacy null stream; non-blocking ones do not.
Stream streamCreate(bool non_blocking = false);
void streamDestroy(Stream s);
void streamSync(Stream s);
void streamWaitEvent(Stream s, Event e);
Event eventCreate(bool timing = true);
void eventDestroy(Event e);
void eventRecord(Event e, Stream s);
void eventSync(Event e);
bool eventDone(Event e);  // true when complete (also for a never-recorded event)
float eventElapsedMs(Event start, Event stop);

// Captured launch sequence replayed with one hipGraphLaunch. Capture uses
// the thread-local capture mode.
class Graph {
public:
    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&& o) noexcept;
    Graph& operator=(Graph&& o) noexcept;
    ~Graph();

    static void beginCapture(Stream s);
    static Graph endCapture(Stream s);
    void launch(Stream s) const;
    bool valid() const { return exec_ != nullptr; }

private:
    void reset();
    void* graph_ = nullptr;
    void* exec_ = nullptr;
};

// ---- code objects
struct EmbeddedObject {
    const char* arch;
    const unsigned char* data;
    const unsigned long long* size;
};
std::span<const EmbeddedObject> embeddedObjects();  // generated at build time
const EmbeddedObject* embeddedObjectFor(Arch a);

class Module {
public:
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&& o) noexcept : handle_(o.handle_), arch_(o.arch_) { o.handle_ = nullptr; }
    Module& operator=(Module&& o) noexcept;
    ~Module();

    // Loads the embedded code object matching the current device.
    static Module loadEmbedded();
    static Module loadData(const void* image, Arch arch);

    Function getFunction(const char* name) const;     // throws if missing
    Function getFunctionOpt(const char* name) const;  // nullptr if missing
    DevPtr getGlobal(const char* name, std::size_t* bytes = nullptr) const;
    Arch arch() const { return arch_; }
    bool loaded() const { return handle_ != nullptr; }

private:
    void* handle_ = nullptr;
    Arch arch_ = Arch::gfx1201;
};

struct Dim3 {
    unsigned x = 1, y = 1, z = 1;
};

void launchRaw(Function f, Dim3 grid, Dim3 block, unsigned shared_bytes, Stream s, void** params);
std::uint64_t launchCount();

// Launches `f` with by-value arguments. Each argument type must match the
// kernel's parameter type exactly (DevPtr / pointers for pointers, int32_t,
// uint32_t, float, uint64_t, or a trivially copyable struct).
template <class... Args>
void launch(Function f, Dim3 grid, Dim3 block, unsigned shared_bytes, Stream s, const Args&... args) {
    static_assert((std::is_trivially_copyable_v<Args> && ...), "kernel arguments must be trivially copyable");
    std::tuple<Args...> copy(args...);
    void* params[sizeof...(Args) > 0 ? sizeof...(Args) : 1] = {};
    std::apply([&](auto&... a) {
        std::size_t i = 0;
        ((params[i++] = static_cast<void*>(&a)), ...);
    }, copy);
    launchRaw(f, grid, block, shared_bytes, s, params);
}

// Runs the smoke kernel on the current device; returns an empty string on
// success, otherwise a description of the failure.
std::string smokeTest();

}  // namespace whirl::hip
