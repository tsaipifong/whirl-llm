// WDDM video memory budget / usage of this process (see include/whirl/hip.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written from the DXGI documentation (CreateDXGIFactory1, IDXGIAdapter1::GetDesc1,
// IDXGIAdapter3::QueryVideoMemoryInfo); the HIP adapter is matched by its LUID.

#include "whirl/hip.h"
#include "whirl/vram_limit.h"

#include "hip_api.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

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
#include <pdhmsg.h>
#endif

namespace whirl::hip {

using runtime::rt;  // every HIP call goes through the loader's table (ARCH-1 D-1)

WddmMemInfo wddmMemInfo() {
    WddmMemInfo r;
#ifdef _WIN32
    int dev = 0;
    if (rt().hipGetDevice(&dev) != hipSuccess) return r;
    hipDeviceProp_t prop{};
    if (rt().hipGetDeviceProperties(&prop, dev) != hipSuccess) return r;
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

GpuOthers gpuOthers(int dev) {
    GpuOthers r;
#ifdef _WIN32
    const auto t0 = std::chrono::steady_clock::now();
    hipDeviceProp_t prop{};
    if (rt().hipGetDeviceProperties(&prop, dev) == hipSuccess) {
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

GpuPinned gpuPinnedNow() {
    GpuPinned r;
#ifdef _WIN32
    PDH_HQUERY q = nullptr;
    if (PdhOpenQueryW(nullptr, 0, &q) == ERROR_SUCCESS) {
        PDH_HCOUNTER c = nullptr;
        if (PdhAddEnglishCounterW(q, L"\\GPU Adapter Memory(*)\\Shared Usage", 0, &c) == ERROR_SUCCESS &&
            PdhCollectQueryData(q) == ERROR_SUCCESS) {
            DWORD bytes = 0, n = 0;
            if (PdhGetFormattedCounterArrayW(c, PDH_FMT_LARGE, &bytes, &n, nullptr) == PDH_MORE_DATA && bytes > 0) {
                std::vector<unsigned char> buf(bytes);
                auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
                if (PdhGetFormattedCounterArrayW(c, PDH_FMT_LARGE, &bytes, &n, items) == ERROR_SUCCESS) {
                    for (DWORD i = 0; i < n; ++i) {
                        if (items[i].FmtValue.CStatus != PDH_CSTATUS_VALID_DATA && items[i].FmtValue.CStatus != PDH_CSTATUS_NEW_DATA) continue;
                        const auto v = static_cast<std::uint64_t>(std::max<LONGLONG>(items[i].FmtValue.largeValue, 0));
                        r.max_shared = std::max(r.max_shared, v);
                        r.sum_shared += v;
                        ++r.adapters;
                    }
                    r.ok = r.adapters > 0;
                }
            }
        }
        PdhCloseQuery(q);
    }
    int ndev = 0;
    if (rt().hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) ndev = 1;
    r.devs.assign(static_cast<std::size_t>(ndev), GpuPinnedDev{});
    IDXGIFactory1* fac = nullptr;
    if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&fac))) && fac) {
        for (int d = 0; d < ndev; ++d) {
            hipDeviceProp_t prop{};
            if (rt().hipGetDeviceProperties(&prop, d) != hipSuccess) continue;
            for (UINT i = 0;; ++i) {
                IDXGIAdapter1* ad = nullptr;
                if (fac->EnumAdapters1(i, &ad) == DXGI_ERROR_NOT_FOUND || !ad) break;
                DXGI_ADAPTER_DESC1 desc{};
                const bool match = SUCCEEDED(ad->GetDesc1(&desc)) && sizeof(desc.AdapterLuid) == sizeof(prop.luid) &&
                                   std::memcmp(&desc.AdapterLuid, prop.luid, sizeof(prop.luid)) == 0;
                if (match) {
                    GpuPinnedDev& g = r.devs[static_cast<std::size_t>(d)];
                    g.shared_system = desc.SharedSystemMemory;
                    // this process's Shared Usage there: an integrated GPU reports a non-local usage of 0 (REVIEW-S8-719 C4)
                    PDH_HQUERY qp = nullptr;
                    if (PdhOpenQueryW(nullptr, 0, &qp) == ERROR_SUCCESS) {
                        wchar_t pp[192];
                        std::swprintf(pp, 192, L"\\GPU Process Memory(pid_%lu_luid_0x%08lX_0x%08lX_phys_0)\\Shared Usage",
                                      GetCurrentProcessId(), static_cast<unsigned long>(desc.AdapterLuid.HighPart),
                                      static_cast<unsigned long>(desc.AdapterLuid.LowPart));
                        PDH_HCOUNTER cp = nullptr;
                        PDH_FMT_COUNTERVALUE fp{};
                        if (PdhAddEnglishCounterW(qp, pp, 0, &cp) == ERROR_SUCCESS && PdhCollectQueryData(qp) == ERROR_SUCCESS &&
                            PdhGetFormattedCounterValue(cp, PDH_FMT_LARGE, nullptr, &fp) == ERROR_SUCCESS && fp.largeValue > 0)
                            g.own_shared = static_cast<std::uint64_t>(fp.largeValue);
                        PdhCloseQuery(qp);
                    }
                    IDXGIAdapter3* a3 = nullptr;
                    if (SUCCEEDED(ad->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&a3))) && a3) {
                        DXGI_QUERY_VIDEO_MEMORY_INFO nl{};
                        if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nl))) {
                            g.nonlocal_budget = nl.Budget;
                            g.nonlocal_usage = nl.CurrentUsage;
                        }
                        a3->Release();
                    }
                }
                ad->Release();
                if (match) break;
            }
        }
        fac->Release();
    }
#endif
    return r;
}

}  // namespace whirl::hip
