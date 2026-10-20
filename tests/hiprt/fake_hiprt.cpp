// A fake HIP runtime for the loader tests (tests/hiprt/hiprt_tests.cpp): exports
// the HIP function names src/hip/hip_api.h resolves, as stubs. No HIP, no GPU.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL.
//   (default)               all names; hipGetDeviceCount reports 7 devices, hipGetErrorString "fake"
//   WHIRL_FAKE_DROP_LAST    without the last table entry (hipModuleLaunchKernel): "driver too old"
//   WHIRL_FAKE_WITH_DEP     hipGetDeviceCount calls whirl_fake_dep.dll (an import that can be
//                           missing: error 126, or present without the function: error 127)

#define WHIRL_FAKE_EXPORT extern "C" __declspec(dllexport)

#if WHIRL_FAKE_WITH_DEP
extern "C" __declspec(dllimport) int whirl_fake_dep_fn(void);
#endif

WHIRL_FAKE_EXPORT int hipGetDeviceCount(int* n) {
#if WHIRL_FAKE_WITH_DEP
    *n = whirl_fake_dep_fn();
#else
    *n = 7;
#endif
    return 0;
}
WHIRL_FAKE_EXPORT const char* hipGetErrorString(int) { return "fake"; }

#define WHIRL_FAKE_STUB(name) \
    WHIRL_FAKE_EXPORT int name(void) { return 0; }
WHIRL_FAKE_STUB(hipDeviceSynchronize)
WHIRL_FAKE_STUB(hipSetDevice)
WHIRL_FAKE_STUB(hipGetDevice)
WHIRL_FAKE_STUB(hipGetDevicePropertiesR0600)
WHIRL_FAKE_STUB(hipGetLastError)
WHIRL_FAKE_STUB(hipMalloc)
WHIRL_FAKE_STUB(hipFree)
WHIRL_FAKE_STUB(hipHostMalloc)
WHIRL_FAKE_STUB(hipHostFree)
WHIRL_FAKE_STUB(hipMemcpy)
WHIRL_FAKE_STUB(hipMemcpyAsync)
WHIRL_FAKE_STUB(hipMemset)
WHIRL_FAKE_STUB(hipMemsetAsync)
WHIRL_FAKE_STUB(hipMemGetInfo)
WHIRL_FAKE_STUB(hipStreamCreate)
WHIRL_FAKE_STUB(hipStreamCreateWithFlags)
WHIRL_FAKE_STUB(hipStreamDestroy)
WHIRL_FAKE_STUB(hipStreamSynchronize)
WHIRL_FAKE_STUB(hipStreamWaitEvent)
WHIRL_FAKE_STUB(hipEventCreate)
WHIRL_FAKE_STUB(hipEventCreateWithFlags)
WHIRL_FAKE_STUB(hipEventDestroy)
WHIRL_FAKE_STUB(hipEventRecord)
WHIRL_FAKE_STUB(hipEventSynchronize)
WHIRL_FAKE_STUB(hipEventQuery)
WHIRL_FAKE_STUB(hipEventElapsedTime)
WHIRL_FAKE_STUB(hipStreamBeginCapture)
WHIRL_FAKE_STUB(hipStreamEndCapture)
WHIRL_FAKE_STUB(hipGraphInstantiateWithFlags)
WHIRL_FAKE_STUB(hipGraphLaunch)
WHIRL_FAKE_STUB(hipGraphExecDestroy)
WHIRL_FAKE_STUB(hipGraphDestroy)
WHIRL_FAKE_STUB(hipModuleLoadData)
WHIRL_FAKE_STUB(hipModuleUnload)
WHIRL_FAKE_STUB(hipModuleGetFunction)
WHIRL_FAKE_STUB(hipModuleGetGlobal)
#if !WHIRL_FAKE_DROP_LAST
WHIRL_FAKE_STUB(hipModuleLaunchKernel)
#endif
