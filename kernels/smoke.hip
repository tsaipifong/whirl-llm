// Smoke-test kernels used by `whirl-tool devices --smoke` to prove that an
// embedded code object loads and runs on a device.
// SPDX-License-Identifier: Apache-2.0

// y[i] = a * x[i] + y[i]
extern "C" __global__ void whirl_smoke_axpy(float* y, const float* x, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// Device global read back through hipModuleGetGlobal.
extern "C" __device__ int whirl_smoke_marker = 0x57484952;  // "WHIR"
