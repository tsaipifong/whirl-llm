// Platform layer: this process and named kernel objects (arch025 PL4). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only (a handle
// is the bits of a Win32 HANDLE as void*); the Windows implementation is proc_win.cpp, chosen by cmake/platform.cmake.
// No virtual interface.
//
// One function per Win32 call (the PL2 / PL3 recipe): the callers keep their logic and so their call order; every
// argument is the one v0.2.4 0e35407 passed. The callers:
//   - the executable's path: src/spec/policy.cpp exeDirectory, src/tier/kv_tier.cpp exeIdentity (the SSD-cache
//     entries are keyed by it: tests/server/tier_tests.cpp testExeIdentityPinned)
//   - the per-GPU named mutex of GPU_WAIT (src/model/device.cpp lockGpu; the name is gpusel::lockMutexName, pinned
//     by tests/unit/unit_tests.cpp testPlatformProc)
//   - Ctrl+C / shutdown: the console control handler and the "runLoop returned" event (src/server/server_main.cpp)
//   - the process id of the SSD directory lock and the pinned-memory declaration (src/tier/kv_tier.cpp), the working
//     set of the WDDM-LOG status line (src/server/server_main.cpp)
// The OVERLAPPED events of the tier IO thread (CreateEventW / ResetEvent) belong to fs (PL2).
// The HIP runtime's own LoadLibrary / DXGI / PDH code stays in src/hip/ (ADR-0004 #5, whitelisted).
#pragma once

#include <cstdint>
#include <string>

namespace whirl::platform::proc {

using Handle = void*;  // a Win32 HANDLE

// WaitForSingleObject results the callers test for (proc_win.cpp static_asserts them against <windows.h>)
inline constexpr std::uint32_t kWaitObject0 = 0x00000000u;   // WAIT_OBJECT_0
inline constexpr std::uint32_t kWaitAbandoned = 0x00000080u; // WAIT_ABANDONED
inline constexpr std::uint32_t kWaitTimeout = 0x00000102u;   // WAIT_TIMEOUT (tests only)

// Console control events (SetConsoleCtrlHandler's dwCtrlType)
inline constexpr int kCtrlC = 0;      // CTRL_C_EVENT
inline constexpr int kCtrlBreak = 1;  // CTRL_BREAK_EVENT
inline constexpr int kCtrlClose = 2;  // CTRL_CLOSE_EVENT

// GetModuleFileNameW(nullptr, ...): the whole path of this executable, in the form it was started with (it can start
// with \\?\). The first try is v0.2.4's (nSize 1023 into a 1024-character buffer), so a path of up to 1022 characters
// is the same string as v0.2.4 (exeIdentity is an SSD-cache key); a longer one is read again into a larger buffer
// (arch025-d7, X4 K7: v0.2.4 cut it to 1022 characters). "" when the call returns 0 or the path does not fit 64 Ki.
std::wstring exePath();

std::uint32_t currentId();  // GetCurrentProcessId
Handle currentProcess();    // GetCurrentProcess (the pseudo handle)
// K32GetProcessMemoryInfo(h, &PROCESS_MEMORY_COUNTERS, sizeof): the working set in bytes; false when the call fails
// (*bytes untouched).
bool workingSetBytes(Handle h, std::uint64_t* bytes);

// CreateMutexW(nullptr, FALSE, name): nullptr on failure. Never closed by its callers (held until the process exits).
Handle createMutex(const wchar_t* name);
// CreateEventW(nullptr, TRUE (manual reset), FALSE (not signaled), nullptr): nullptr on failure.
Handle createEvent();
bool setEvent(Handle ev);                        // SetEvent
std::uint32_t wait(Handle h, std::uint32_t ms);  // WaitForSingleObject: its result as is
void sleepMs(std::uint32_t ms);                  // Sleep

// Console control handler: return true = handled (TRUE), false = pass on (FALSE). Called on the OS's handler thread.
using CtrlHandler = bool (*)(int event);
// SetConsoleCtrlHandler(nullptr, FALSE): clears an "ignore Ctrl+C" attribute inherited from the parent.
bool restoreCtrlC();
// SetConsoleCtrlHandler(<trampoline>, TRUE): the trampoline calls fn(event) and returns its result as BOOL. One
// handler per process (a second call replaces the function the trampoline calls and registers the trampoline again).
bool addCtrlHandler(CtrlHandler fn);

}  // namespace whirl::platform::proc
