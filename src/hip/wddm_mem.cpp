// WDDM video memory budget / usage of this process (see include/whirl/hip.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written from the DXGI documentation (CreateDXGIFactory1, IDXGIAdapter1::GetDesc1,
// IDXGIAdapter3::QueryVideoMemoryInfo); the HIP adapter is matched by its LUID.

#include "whirl/hip.h"
#include "whirl/vram_limit.h"

#include <hip/hip_runtime_api.h>

#include <chrono>
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
#include <dxgi1_4.h>
#include <pdh.h>
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
                    // WHIRL_VRAM_LIMIT_MB: the simulated card's budget is smaller by the VRAM it lacks
                    r.local_budget = vram::clampBudget(lo.Budget, prop.totalGlobalMem, vram::limitBytes());
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

GpuOthers gpuOthers() {
    GpuOthers r;
#ifdef _WIN32
    const auto t0 = std::chrono::steady_clock::now();
    int dev = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&dev) == hipSuccess && hipGetDeviceProperties(&prop, dev) == hipSuccess) {
        LUID luid{};
        static_assert(sizeof(luid) == sizeof(prop.luid));
        std::memcpy(&luid, prop.luid, sizeof(luid));
        // display attached: a monitor on one of this adapter's outputs
        IDXGIFactory1* fac = nullptr;
        if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&fac))) && fac) {
            for (UINT i = 0;; ++i) {
                IDXGIAdapter1* ad = nullptr;
                if (fac->EnumAdapters1(i, &ad) == DXGI_ERROR_NOT_FOUND || !ad) break;
                DXGI_ADAPTER_DESC1 d{};
                const bool match = SUCCEEDED(ad->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, &luid, sizeof(luid)) == 0;
                if (match) {
                    IDXGIOutput* out = nullptr;
                    r.display_known = true;
                    r.display = ad->EnumOutputs(0, &out) != DXGI_ERROR_NOT_FOUND && out != nullptr;
                    if (out) out->Release();
                }
                ad->Release();
                if (match) break;
            }
            fac->Release();
        }
        PDH_HQUERY q = nullptr;
        if (PdhOpenQueryW(nullptr, 0, &q) == ERROR_SUCCESS) {
            wchar_t pa[192], pp[192];
            std::swprintf(pa, 192, L"\\GPU Adapter Memory(luid_0x%08lX_0x%08lX_phys_0)\\Dedicated Usage",
                          static_cast<unsigned long>(luid.HighPart), static_cast<unsigned long>(luid.LowPart));
            std::swprintf(pp, 192, L"\\GPU Process Memory(pid_%lu_luid_0x%08lX_0x%08lX_phys_0)\\Dedicated Usage", GetCurrentProcessId(),
                          static_cast<unsigned long>(luid.HighPart), static_cast<unsigned long>(luid.LowPart));
            PDH_HCOUNTER ca = nullptr, cp = nullptr;
            bool ok = PdhAddEnglishCounterW(q, pa, 0, &ca) == ERROR_SUCCESS;
            // this process may have no counter instance yet (nothing allocated on the adapter): own = 0
            const bool own_ok = PdhAddEnglishCounterW(q, pp, 0, &cp) == ERROR_SUCCESS;
            ok = ok && PdhCollectQueryData(q) == ERROR_SUCCESS;
            PDH_FMT_COUNTERVALUE fv{};
            if (ok && PdhGetFormattedCounterValue(ca, PDH_FMT_LARGE, nullptr, &fv) == ERROR_SUCCESS) {
                r.adapter_dedicated = static_cast<std::uint64_t>(fv.largeValue);
                PDH_FMT_COUNTERVALUE fp{};
                if (own_ok && PdhGetFormattedCounterValue(cp, PDH_FMT_LARGE, nullptr, &fp) == ERROR_SUCCESS)
                    r.own_dedicated = static_cast<std::uint64_t>(fp.largeValue);
                r.others = r.adapter_dedicated > r.own_dedicated ? r.adapter_dedicated - r.own_dedicated : 0;
                r.ok = true;
            }
            PdhCloseQuery(q);
        }
    }
    r.cost_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
#endif
    return r;
}

}  // namespace whirl::hip
