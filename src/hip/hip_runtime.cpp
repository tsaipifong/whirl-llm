// HIP runtime wrapper (see include/whirl/hip.h).
// SPDX-License-Identifier: Apache-2.0
//
// Ported from the project's own prototype runtime (runtime.zig: memory,
// streams, events, graphs, embedded code objects, launch); device
// description and selection are written from the HIP runtime API docs.

#include "whirl/hip.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <utility>

namespace whirl::hip {

namespace {

thread_local const char* g_last_op = "";
thread_local int g_last_code = 0;
std::atomic<std::uint64_t> g_launch_count{0};

void check(hipError_t st, const char* op) {
    if (st == hipSuccess) return;
    g_last_op = op;
    g_last_code = static_cast<int>(st);
    throw Error(op, static_cast<int>(st), hipGetErrorString(st));
}

void* ptr(DevPtr p) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(p)); }
hipStream_t strm(Stream s) { return static_cast<hipStream_t>(s); }
hipEvent_t evt(Event e) { return static_cast<hipEvent_t>(e); }

std::string lower(std::string_view s) {
    std::string r(s);
    for (auto& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

}  // namespace

Error::Error(std::string op, int code, std::string message)
    : std::runtime_error(op + " failed: " + message + " (hipError " + std::to_string(code) + ")"),
      op_(std::move(op)),
      code_(code) {}

const char* lastErrorOp() { return g_last_op; }
int lastErrorCode() { return g_last_code; }
std::string errorString(int code) { return hipGetErrorString(static_cast<hipError_t>(code)); }

// ---------------------------------------------------------------------------
// devices

int deviceCount() {
    int n = 0;
    hipError_t st = hipGetDeviceCount(&n);
    if (st == hipErrorNoDevice) return 0;
    check(st, "hipGetDeviceCount");
    return n;
}

DeviceInfo describeDevice(int index) {
    hipDeviceProp_t p{};
    check(hipGetDeviceProperties(&p, index), "hipGetDeviceProperties");
    DeviceInfo d;
    d.index = index;
    d.name = p.name;
    d.gcn_arch = p.gcnArchName;
    d.total_mem = p.totalGlobalMem;
    d.compute_units = p.multiProcessorCount;
    d.warp_size = p.warpSize;
    d.clock_khz = p.clockRate;
    d.pci_bus = p.pciBusID;
    d.pci_device = p.pciDeviceID;
    d.integrated = p.integrated != 0;
    d.shared_mem_per_block = p.sharedMemPerBlock;
    d.max_threads_per_block = p.maxThreadsPerBlock;
    return d;
}

std::vector<DeviceInfo> listDevices() {
    std::vector<DeviceInfo> out;
    const int n = deviceCount();
    for (int i = 0; i < n; ++i) out.push_back(describeDevice(i));
    return out;
}

int findDevice(std::string_view spec) {
    const auto devs = listDevices();
    if (!spec.empty() && std::all_of(spec.begin(), spec.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        const int i = std::stoi(std::string(spec));
        return i < static_cast<int>(devs.size()) ? i : -1;
    }
    const std::string s = lower(spec);
    for (const auto& d : devs)
        if (lower(d.gcn_arch).rfind(s, 0) == 0) return d.index;
    for (const auto& d : devs)
        if (lower(d.name).find(s) != std::string::npos) return d.index;
    return -1;
}

void setDevice(int index) { check(hipSetDevice(index), "hipSetDevice"); }

int getDevice() {
    int d = 0;
    check(hipGetDevice(&d), "hipGetDevice");
    return d;
}

Arch archFor(std::string_view gcn_arch) {
    return gcn_arch.rfind("gfx1151", 0) == 0 ? Arch::gfx1151 : Arch::gfx1201;
}

const char* archName(Arch a) { return a == Arch::gfx1151 ? "gfx1151" : "gfx1201"; }

// ---------------------------------------------------------------------------
// memory

DevPtr malloc(std::size_t bytes) {
    void* p = nullptr;
    check(hipMalloc(&p, std::max<std::size_t>(bytes, 1)), "hipMalloc");
    return reinterpret_cast<std::uintptr_t>(p);
}

void free(DevPtr p) {
    if (p) (void)hipFree(ptr(p));
}

void* hostMalloc(std::size_t bytes) {
    void* p = nullptr;
    check(hipHostMalloc(&p, std::max<std::size_t>(bytes, 1), 0), "hipHostMalloc");
    return p;
}

void hostFree(void* p) {
    if (p) (void)hipHostFree(p);
}

void upload(DevPtr dst, const void* src, std::size_t bytes) {
    check(hipMemcpy(ptr(dst), src, bytes, hipMemcpyHostToDevice), "hipMemcpy H2D");
}

void download(void* dst, DevPtr src, std::size_t bytes) {
    check(hipMemcpy(dst, ptr(src), bytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");
}

void uploadAsync(DevPtr dst, const void* src, std::size_t bytes, Stream s) {
    check(hipMemcpyAsync(ptr(dst), src, bytes, hipMemcpyHostToDevice, strm(s)), "hipMemcpyAsync H2D");
}

void downloadAsync(void* dst, DevPtr src, std::size_t bytes, Stream s) {
    check(hipMemcpyAsync(dst, ptr(src), bytes, hipMemcpyDeviceToHost, strm(s)), "hipMemcpyAsync D2H");
}

void copyAsync(DevPtr dst, DevPtr src, std::size_t bytes, Stream s) {
    check(hipMemcpyAsync(ptr(dst), ptr(src), bytes, hipMemcpyDeviceToDevice, strm(s)), "hipMemcpyAsync D2D");
}

void copyAnyAsync(DevPtr dst, DevPtr src, std::size_t bytes, Stream s) {
    check(hipMemcpyAsync(ptr(dst), ptr(src), bytes, hipMemcpyDefault, strm(s)), "hipMemcpyAsync default");
}

void memset(DevPtr dst, int value, std::size_t bytes) {
    check(hipMemset(ptr(dst), value, bytes), "hipMemset");
}

void memsetAsync(DevPtr dst, int value, std::size_t bytes, Stream s) {
    check(hipMemsetAsync(ptr(dst), value, bytes, strm(s)), "hipMemsetAsync");
}

void sync() { check(hipDeviceSynchronize(), "hipDeviceSynchronize"); }

MemInfo memInfo() {
    MemInfo m;
    check(hipMemGetInfo(&m.free, &m.total), "hipMemGetInfo");
    return m;
}

// ---------------------------------------------------------------------------
// streams and events

Stream streamCreate(bool non_blocking) {
    hipStream_t s = nullptr;
    if (non_blocking)
        check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "hipStreamCreateWithFlags");
    else
        check(hipStreamCreate(&s), "hipStreamCreate");
    return s;
}

void streamDestroy(Stream s) {
    if (s) (void)hipStreamDestroy(strm(s));
}

void streamSync(Stream s) { check(hipStreamSynchronize(strm(s)), "hipStreamSynchronize"); }

void streamWaitEvent(Stream s, Event e) { check(hipStreamWaitEvent(strm(s), evt(e), 0), "hipStreamWaitEvent"); }

Event eventCreate(bool timing) {
    hipEvent_t e = nullptr;
    if (timing)
        check(hipEventCreate(&e), "hipEventCreate");
    else
        check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "hipEventCreateWithFlags");
    return e;
}

void eventDestroy(Event e) {
    if (e) (void)hipEventDestroy(evt(e));
}

void eventRecord(Event e, Stream s) { check(hipEventRecord(evt(e), strm(s)), "hipEventRecord"); }

void eventSync(Event e) { check(hipEventSynchronize(evt(e)), "hipEventSynchronize"); }

bool eventDone(Event e) {
    const hipError_t st = hipEventQuery(evt(e));
    if (st == hipSuccess) return true;
    if (st == hipErrorNotReady) return false;
    check(st, "hipEventQuery");
    return false;
}

float eventElapsedMs(Event start, Event stop) {
    float ms = 0;
    check(hipEventElapsedTime(&ms, evt(start), evt(stop)), "hipEventElapsedTime");
    return ms;
}

// ---------------------------------------------------------------------------
// graphs

Graph::Graph(Graph&& o) noexcept : graph_(o.graph_), exec_(o.exec_) { o.graph_ = o.exec_ = nullptr; }

Graph& Graph::operator=(Graph&& o) noexcept {
    if (this != &o) {
        reset();
        graph_ = std::exchange(o.graph_, nullptr);
        exec_ = std::exchange(o.exec_, nullptr);
    }
    return *this;
}

Graph::~Graph() { reset(); }

void Graph::reset() {
    if (exec_) (void)hipGraphExecDestroy(static_cast<hipGraphExec_t>(exec_));
    if (graph_) (void)hipGraphDestroy(static_cast<hipGraph_t>(graph_));
    graph_ = exec_ = nullptr;
}

void Graph::beginCapture(Stream s) {
    check(hipStreamBeginCapture(strm(s), hipStreamCaptureModeThreadLocal), "hipStreamBeginCapture");
}

Graph Graph::endCapture(Stream s) {
    Graph g;
    hipGraph_t graph = nullptr;
    check(hipStreamEndCapture(strm(s), &graph), "hipStreamEndCapture");
    g.graph_ = graph;
    hipGraphExec_t exec = nullptr;
    check(hipGraphInstantiateWithFlags(&exec, graph, 0), "hipGraphInstantiate");
    g.exec_ = exec;
    return g;
}

void Graph::launch(Stream s) const {
    check(hipGraphLaunch(static_cast<hipGraphExec_t>(exec_), strm(s)), "hipGraphLaunch");
}

// ---------------------------------------------------------------------------
// code objects

const EmbeddedObject* embeddedObjectFor(Arch a) {
    for (const auto& o : embeddedObjects())
        if (std::strcmp(o.arch, archName(a)) == 0) return &o;
    return nullptr;
}

Module& Module::operator=(Module&& o) noexcept {
    if (this != &o) {
        if (handle_) (void)hipModuleUnload(static_cast<hipModule_t>(handle_));
        handle_ = std::exchange(o.handle_, nullptr);
        arch_ = o.arch_;
    }
    return *this;
}

Module::~Module() {
    if (handle_) (void)hipModuleUnload(static_cast<hipModule_t>(handle_));
}

Module Module::loadData(const void* image, Arch arch) {
    Module m;
    hipModule_t h = nullptr;
    check(hipModuleLoadData(&h, image), "hipModuleLoadData");
    m.handle_ = h;
    m.arch_ = arch;
    return m;
}

Module Module::loadEmbedded() {
    const DeviceInfo d = describeDevice(getDevice());
    const Arch a = archFor(d.gcn_arch);
    const EmbeddedObject* o = embeddedObjectFor(a);
    if (!o)
        throw Error("Module::loadEmbedded", static_cast<int>(hipErrorNoBinaryForGpu),
                    std::string("no embedded code object for ") + archName(a));
    return loadData(o->data, a);
}

Function Module::getFunction(const char* name) const {
    hipFunction_t f = nullptr;
    const hipError_t st = hipModuleGetFunction(&f, static_cast<hipModule_t>(handle_), name);
    if (st != hipSuccess) {
        (void)hipGetLastError();
        g_last_op = "hipModuleGetFunction";
        g_last_code = static_cast<int>(st);
        throw Error(std::string("hipModuleGetFunction(") + name + ")", static_cast<int>(st), hipGetErrorString(st));
    }
    return f;
}

Function Module::getFunctionOpt(const char* name) const {
    hipFunction_t f = nullptr;
    if (hipModuleGetFunction(&f, static_cast<hipModule_t>(handle_), name) != hipSuccess) {
        (void)hipGetLastError();  // clear the not-found status
        return nullptr;
    }
    return f;
}

DevPtr Module::getGlobal(const char* name, std::size_t* bytes) const {
    hipDeviceptr_t p = nullptr;
    std::size_t n = 0;
    check(hipModuleGetGlobal(&p, &n, static_cast<hipModule_t>(handle_), name), "hipModuleGetGlobal");
    if (bytes) *bytes = n;
    return reinterpret_cast<std::uintptr_t>(p);
}

void launchRaw(Function f, Dim3 grid, Dim3 block, unsigned shared_bytes, Stream s, void** params) {
    g_launch_count.fetch_add(1, std::memory_order_relaxed);
    check(hipModuleLaunchKernel(static_cast<hipFunction_t>(f), grid.x, grid.y, grid.z, block.x, block.y, block.z,
                                shared_bytes, strm(s), params, nullptr),
          "hipModuleLaunchKernel");
}

std::uint64_t launchCount() { return g_launch_count.load(std::memory_order_relaxed); }

// ---------------------------------------------------------------------------
// smoke test

std::string smokeTest() {
    try {
        Module m = Module::loadEmbedded();
        Function f = m.getFunction("whirl_smoke_axpy");
        if (m.getFunctionOpt("whirl_no_such_kernel") != nullptr) return "getFunctionOpt returned a missing kernel";
        std::size_t gbytes = 0;
        const DevPtr marker = m.getGlobal("whirl_smoke_marker", &gbytes);
        int marker_val = 0;
        download(&marker_val, marker, sizeof marker_val);
        if (gbytes != sizeof(int) || marker_val != 0x57484952) return "device global mismatch";

        constexpr int n = 1000;
        std::vector<float> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = static_cast<float>(i);
            y[i] = 1.0f;
        }
        const DevPtr dx = malloc(n * sizeof(float));
        const DevPtr dy = malloc(n * sizeof(float));
        Stream s = streamCreate();
        Event e0 = eventCreate(), e1 = eventCreate();
        upload(dx, x.data(), n * sizeof(float));
        upload(dy, y.data(), n * sizeof(float));
        eventRecord(e0, s);
        launch(f, Dim3{(n + 255) / 256}, Dim3{256}, 0, s, dy, dx, 2.0f, static_cast<std::int32_t>(n));
        eventRecord(e1, s);
        // the same kernel replayed through a captured graph: y = 1*x + y
        Graph::beginCapture(s);
        launch(f, Dim3{(n + 255) / 256}, Dim3{256}, 0, s, dy, dx, 1.0f, static_cast<std::int32_t>(n));
        Graph g = Graph::endCapture(s);
        g.launch(s);
        streamSync(s);
        (void)eventElapsedMs(e0, e1);
        download(y.data(), dy, n * sizeof(float));
        free(dx);
        free(dy);
        eventDestroy(e0);
        eventDestroy(e1);
        streamDestroy(s);
        for (int i = 0; i < n; ++i)
            if (y[i] != 3.0f * static_cast<float>(i) + 1.0f) return "wrong result at index " + std::to_string(i);
        return "";
    } catch (const std::exception& ex) {
        return ex.what();
    }
}

}  // namespace whirl::hip
