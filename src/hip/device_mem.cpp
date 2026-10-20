// Device-local memory accounting per device class (see include/whirl/device_mem.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written from the PDH documentation (PdhOpenQueryW, PdhAddEnglishCounterW,
// PdhGetFormattedCounterValue) and the "GPU Process Memory" counter set's instance
// names (pid_<pid>_luid_0x<high>_0x<low>_phys_0).

#include "whirl/device_mem.h"

#include "whirl/hip.h"

#include "hip_api.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <pdh.h>
#endif

namespace whirl::hip {

using runtime::rt;  // every HIP call goes through the loader's table (ARCH-1 D-1)

namespace {

#ifdef _WIN32
// this process's "GPU Process Memory" counters on the current adapter
bool readPdh(std::uint64_t& dedicated, std::uint64_t& shared) {
    int dev = 0;
    if (rt().hipGetDevice(&dev) != hipSuccess) return false;
    hipDeviceProp_t prop{};
    if (rt().hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
    LUID luid{};
    static_assert(sizeof(luid) == sizeof(prop.luid));
    std::memcpy(&luid, prop.luid, sizeof(luid));
    PDH_HQUERY q = nullptr;
    if (PdhOpenQueryW(nullptr, 0, &q) != ERROR_SUCCESS) return false;
    bool ok = true;
    std::uint64_t v[2] = {0, 0};
    const wchar_t* names[2] = {L"Dedicated Usage", L"Shared Usage"};
    PDH_HCOUNTER c[2] = {nullptr, nullptr};
    for (int k = 0; k < 2 && ok; ++k) {
        wchar_t path[256];
        std::swprintf(path, 256, L"\\GPU Process Memory(pid_%lu_luid_0x%08lX_0x%08lX_phys_0)\\%ls", GetCurrentProcessId(),
                      static_cast<unsigned long>(luid.HighPart), static_cast<unsigned long>(luid.LowPart), names[k]);
        ok = PdhAddEnglishCounterW(q, path, 0, &c[k]) == ERROR_SUCCESS;
    }
    ok = ok && PdhCollectQueryData(q) == ERROR_SUCCESS;
    for (int k = 0; k < 2 && ok; ++k) {
        PDH_FMT_COUNTERVALUE fv{};
        ok = PdhGetFormattedCounterValue(c[k], PDH_FMT_LARGE, nullptr, &fv) == ERROR_SUCCESS;
        if (ok) v[k] = static_cast<std::uint64_t>(fv.largeValue);
    }
    PdhCloseQuery(q);
    if (!ok) return false;
    dedicated = v[0];
    shared = v[1];
    return true;
}
#endif

}  // namespace

LocalMemSnap localMemSnap() {
    LocalMemSnap s;
    s.cls = describeDevice(getDevice()).integrated ? MemClass::integrated : MemClass::discrete;
    s.hip_free = memInfo().free;
#ifdef _WIN32
    s.pdh_ok = readPdh(s.dedicated, s.shared);
#endif
    return s;
}

}  // namespace whirl::hip
