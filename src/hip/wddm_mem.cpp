// WDDM video memory budget / usage of this process (see include/whirl/hip.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written from the DXGI documentation (CreateDXGIFactory1, IDXGIAdapter1::GetDesc1,
// IDXGIAdapter3::QueryVideoMemoryInfo); the HIP adapter is matched by its LUID.

#include "whirl/hip.h"

#include <hip/hip_runtime_api.h>

#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dxgi1_4.h>
#endif

namespace whirl::hip {

WddmMemInfo wddmMemInfo() {
    WddmMemInfo r;
#ifdef _WIN32
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return r;
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return r;
    IDXGIFactory1* fac = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&fac))) || !fac) return r;
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* ad = nullptr;
        if (fac->EnumAdapters1(i, &ad) == DXGI_ERROR_NOT_FOUND || !ad) break;
        DXGI_ADAPTER_DESC1 d{};
        bool match = SUCCEEDED(ad->GetDesc1(&d)) && sizeof(d.AdapterLuid) == sizeof(prop.luid) &&
                     std::memcmp(&d.AdapterLuid, prop.luid, sizeof(prop.luid)) == 0;
        if (match) {
            IDXGIAdapter3* a3 = nullptr;
            if (SUCCEEDED(ad->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&a3))) && a3) {
                DXGI_QUERY_VIDEO_MEMORY_INFO lo{}, nl{};
                if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &lo)) &&
                    SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nl))) {
                    r.ok = true;
                    r.local_budget = lo.Budget;
                    r.local_usage = lo.CurrentUsage;
                    r.nonlocal_budget = nl.Budget;
                    r.nonlocal_usage = nl.CurrentUsage;
                }
                a3->Release();
            }
        }
        ad->Release();
        if (match) break;
    }
    fac->Release();
#endif
    return r;
}

}  // namespace whirl::hip
