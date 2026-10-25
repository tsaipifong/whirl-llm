// Platform layer: this process and named kernel objects (arch025 PL4). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is proc_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
#pragma once

namespace whirl::platform::proc {

// PL4 moves here (exeIdentity string and mutex names bitwise identical, pinned by tests):
//   - the executable's path: GetModuleFileNameW (src/model/spec.cpp exeDirectory, src/tier/kv_tier.cpp exeIdentity)
//   - the per-GPU named mutex of GPU_WAIT: CreateMutexW / WaitForSingleObject (src/model/device.cpp)
//   - Ctrl+C / shutdown: SetConsoleCtrlHandler, CreateEventW / SetEvent / WaitForSingleObject, Sleep
//     (src/server/server_main.cpp)
//   - GetCurrentProcess and the working set K32GetProcessMemoryInfo (<psapi.h>) (src/server/server_main.cpp);
//     GetCurrentProcessId of the SSD directory lock / owner file (src/tier/kv_tier.cpp)
// The OVERLAPPED events of the tier IO thread (CreateEventW / ResetEvent) belong to fs (PL2).
// The HIP runtime's own LoadLibrary / DXGI / PDH code stays in src/hip/ (ADR-0004 #5, whitelisted).

}  // namespace whirl::platform::proc
