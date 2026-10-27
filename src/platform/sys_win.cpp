// Platform layer, machine facts: the Windows implementation of platform/sys.h.
// SPDX-License-Identifier: Apache-2.0
//
// arch025 PL4: the GlobalMemoryStatusEx call of src/server/server_main.cpp (RAM-tier sizing), arguments as in v0.2.4.

#include "platform/sys.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace whirl::platform::sys {

bool memoryStatus(std::uint64_t* total_phys, std::uint64_t* avail_phys) {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return false;
    *total_phys = ms.ullTotalPhys;
    *avail_phys = ms.ullAvailPhys;
    return true;
}

}  // namespace whirl::platform::sys
