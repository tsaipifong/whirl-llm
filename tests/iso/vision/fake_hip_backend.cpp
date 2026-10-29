// Link double of the HIP device backend factory (src/device/backend_hip.cpp, device::hipBackend) for the isolation program of vision (ARCH-1 8.5,
// arch025-iso4). The vision host code reaches the device only through device::Backend (src/device/backend.h); the real factory returns the HIP backend
// (a GPU), the double returns the host-memory null backend (src/device/null_backend.cpp, the CPU tests' backend), so nothing here can reach a GPU.
// SPDX-License-Identifier: Apache-2.0
#include "device/backend.h"
#include "device/null_backend.h"

namespace whirl::device {

Backend& hipBackend() {
    static NullBackend b;
    return b;
}

}  // namespace whirl::device
