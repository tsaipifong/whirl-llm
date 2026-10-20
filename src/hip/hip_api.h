// The HIP function table filled by the runtime loader (ARCH-1 D-1, see
// include/whirl/hip_runtime_loader.h). Only src/hip/*.cpp include this (it
// needs <hip/hip_runtime_api.h> for the types); they call rt().hipX(...) where
// they used to call the imported hipX(...).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL.
#pragma once

#include "whirl/hip_runtime_loader.h"

#include <hip/hip_runtime_api.h>

#include <atomic>

// X(return type, exported name, parameter list). The exported name is the
// symbol the DLL exports: hipGetDeviceProperties is a macro for
// hipGetDevicePropertiesR0600 in the HIP 6+ headers, so rt().hipGetDeviceProperties
// expands to the right member. Each entry is checked against the SDK
// declaration at compile time (runtime_loader.cpp).
#define WHIRL_HIP_API_LIST(X)                                                                                         \
    X(hipError_t, hipDeviceSynchronize, (void))                                                                       \
    X(hipError_t, hipSetDevice, (int))                                                                                \
    X(hipError_t, hipGetDevice, (int*))                                                                               \
    X(hipError_t, hipGetDeviceCount, (int*))                                                                          \
    X(hipError_t, hipGetDevicePropertiesR0600, (hipDeviceProp_tR0600*, int))                                          \
    X(hipError_t, hipGetLastError, (void))                                                                            \
    X(const char*, hipGetErrorString, (hipError_t))                                                                   \
    X(hipError_t, hipMalloc, (void**, size_t))                                                                        \
    X(hipError_t, hipFree, (void*))                                                                                   \
    X(hipError_t, hipHostMalloc, (void**, size_t, unsigned int))                                                      \
    X(hipError_t, hipHostFree, (void*))                                                                               \
    X(hipError_t, hipMemcpy, (void*, const void*, size_t, hipMemcpyKind))                                             \
    X(hipError_t, hipMemcpyAsync, (void*, const void*, size_t, hipMemcpyKind, hipStream_t))                           \
    X(hipError_t, hipMemset, (void*, int, size_t))                                                                    \
    X(hipError_t, hipMemGetInfo, (size_t*, size_t*))                                                                  \
    X(hipError_t, hipStreamCreate, (hipStream_t*))                                                                    \
    X(hipError_t, hipStreamCreateWithFlags, (hipStream_t*, unsigned int))                                             \
    X(hipError_t, hipStreamDestroy, (hipStream_t))                                                                    \
    X(hipError_t, hipStreamSynchronize, (hipStream_t))                                                                \
    X(hipError_t, hipStreamWaitEvent, (hipStream_t, hipEvent_t, unsigned int))                                        \
    X(hipError_t, hipEventCreate, (hipEvent_t*))                                                                      \
    X(hipError_t, hipEventCreateWithFlags, (hipEvent_t*, unsigned))                                                   \
    X(hipError_t, hipEventDestroy, (hipEvent_t))                                                                      \
    X(hipError_t, hipEventRecord, (hipEvent_t, hipStream_t))                                                          \
    X(hipError_t, hipEventSynchronize, (hipEvent_t))                                                                  \
    X(hipError_t, hipEventQuery, (hipEvent_t))                                                                        \
    X(hipError_t, hipEventElapsedTime, (float*, hipEvent_t, hipEvent_t))                                              \
    X(hipError_t, hipStreamBeginCapture, (hipStream_t, hipStreamCaptureMode))                                         \
    X(hipError_t, hipStreamEndCapture, (hipStream_t, hipGraph_t*))                                                    \
    X(hipError_t, hipGraphInstantiateWithFlags, (hipGraphExec_t*, hipGraph_t, unsigned long long))                    \
    X(hipError_t, hipGraphLaunch, (hipGraphExec_t, hipStream_t))                                                      \
    X(hipError_t, hipGraphExecDestroy, (hipGraphExec_t))                                                              \
    X(hipError_t, hipGraphDestroy, (hipGraph_t))                                                                      \
    X(hipError_t, hipModuleLoadData, (hipModule_t*, const void*))                                                     \
    X(hipError_t, hipModuleUnload, (hipModule_t))                                                                     \
    X(hipError_t, hipModuleGetFunction, (hipFunction_t*, hipModule_t, const char*))                                   \
    X(hipError_t, hipModuleGetGlobal, (hipDeviceptr_t*, size_t*, hipModule_t, const char*))                           \
    X(hipError_t, hipModuleLaunchKernel,                                                                              \
      (hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, \
       hipStream_t, void**, void**))

namespace whirl::hip::runtime {

struct Api {
#define WHIRL_HIP_API_FIELD(ret, name, params) ret(*name) params = nullptr;
    WHIRL_HIP_API_LIST(WHIRL_HIP_API_FIELD)
#undef WHIRL_HIP_API_FIELD
};

// Loads the runtime and returns its table, or throws std::runtime_error.
const Api& loadOrThrow();

// The process-wide table, filled by load() (the store of g_ready publishes it).
extern Api g_api;
extern std::atomic<bool> g_ready;

// The table every HIP call goes through: one acquire load (a plain load on
// x64) on the hot path, the same cost class as the import thunk it replaces.
inline const Api& rt() { return g_ready.load(std::memory_order_acquire) ? g_api : loadOrThrow(); }

// probe() with the resolved table (unit tests call through a fake DLL's table).
LoadResult probeTable(const std::wstring& app_dir, const std::wstring& dll, Api* out, bool keep);

}  // namespace whirl::hip::runtime
