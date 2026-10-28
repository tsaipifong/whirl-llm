// Platform layer, this process: the Windows implementation of platform/proc.h.
// SPDX-License-Identifier: Apache-2.0
//
// arch025 PL4: the process / named-object calls of src/spec/policy.cpp, src/tier/kv_tier.cpp, src/model/device.cpp and
// src/server/server_main.cpp moved here one function per call, each with the arguments it had in v0.2.4 0e35407.

#include "platform/proc.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>  // K32GetProcessMemoryInfo (kernel32)

#include <atomic>

namespace pproc = whirl::platform::proc;
static_assert(WAIT_OBJECT_0 == pproc::kWaitObject0);
static_assert(WAIT_ABANDONED == pproc::kWaitAbandoned);
static_assert(WAIT_TIMEOUT == pproc::kWaitTimeout);
static_assert(CTRL_C_EVENT == pproc::kCtrlC);
static_assert(CTRL_BREAK_EVENT == pproc::kCtrlBreak);
static_assert(CTRL_CLOSE_EVENT == pproc::kCtrlClose);
static_assert(sizeof(HANDLE) == sizeof(pproc::Handle));
static_assert(sizeof(DWORD) == sizeof(std::uint32_t));

namespace whirl::platform::proc {

namespace {

std::atomic<CtrlHandler> g_ctrl{nullptr};

BOOL WINAPI ctrlTrampoline(DWORD type) {
    const CtrlHandler fn = g_ctrl.load();
    if (fn == nullptr) return FALSE;
    return fn(static_cast<int>(type)) ? TRUE : FALSE;
}

}  // namespace

std::wstring exePath() {
    // arch025-d7 (D7.1, X4 K7): grow until the path fits (GetModuleFileNameW returns nSize, truncated, when it does
    // not). The first try is v0.2.4's (nSize 1023), so a path of up to 1022 characters is the same string as before.
    std::wstring buf(1024, L'\0');
    DWORD size = 1024 - 1;
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), size);
        if (n == 0) return {};
        if (n < size) {
            buf.resize(n);
            return buf;
        }
        if (size >= 0x10000u) return {};  // past the 32767-character Windows limit: never a cut path
        size = size * 2 + 2;
        buf.assign(static_cast<std::size_t>(size) + 1, L'\0');
    }
}

std::uint32_t currentId() { return GetCurrentProcessId(); }

Handle currentProcess() { return GetCurrentProcess(); }

bool workingSetBytes(Handle h, std::uint64_t* bytes) {
    PROCESS_MEMORY_COUNTERS pmc{};
    if (!K32GetProcessMemoryInfo(static_cast<HANDLE>(h), &pmc, sizeof(pmc))) return false;
    *bytes = pmc.WorkingSetSize;
    return true;
}

Handle createMutex(const wchar_t* name) { return CreateMutexW(nullptr, FALSE, name); }

Handle createEvent() { return CreateEventW(nullptr, TRUE, FALSE, nullptr); }

bool setEvent(Handle ev) { return SetEvent(static_cast<HANDLE>(ev)) != 0; }

std::uint32_t wait(Handle h, std::uint32_t ms) { return WaitForSingleObject(static_cast<HANDLE>(h), ms); }

void sleepMs(std::uint32_t ms) { Sleep(ms); }

bool restoreCtrlC() { return SetConsoleCtrlHandler(nullptr, FALSE) != 0; }

bool addCtrlHandler(CtrlHandler fn) {
    g_ctrl.store(fn);
    return SetConsoleCtrlHandler(ctrlTrampoline, TRUE) != 0;
}

}  // namespace whirl::platform::proc
