// Integrated GPU carve-out (see include/whirl/uma_mem.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written from the DXGI documentation (CreateDXGIFactory1, IDXGIAdapter1::GetDesc1,
// DXGI_ADAPTER_DESC1::DedicatedVideoMemory / SharedSystemMemory); the HIP device is matched to
// its DXGI adapter by LUID, as in wddm_mem.cpp.

#include "whirl/uma_mem.h"

#include "hip_api.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dxgi.h>
#endif

namespace whirl::hip {

using runtime::rt;  // every HIP call goes through the loader's table (ARCH-1 D-1)

namespace {

std::mutex g_mu;
bool g_have = false;
UmaMem g_last;

}  // namespace

UmaMem umaMemory(int device) {
    UmaMem r;
    hipDeviceProp_t prop{};
    if (rt().hipGetDeviceProperties(&prop, device) != hipSuccess) return r;
    r.integrated = prop.integrated != 0;
    if (!r.integrated) return r;
#ifdef _WIN32
    IDXGIFactory1* fac = nullptr;
    if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&fac))) && fac) {
        for (UINT i = 0;; ++i) {
            IDXGIAdapter1* ad = nullptr;
            if (fac->EnumAdapters1(i, &ad) == DXGI_ERROR_NOT_FOUND || !ad) break;
            DXGI_ADAPTER_DESC1 d{};
            const bool match = SUCCEEDED(ad->GetDesc1(&d)) && sizeof(d.AdapterLuid) == sizeof(prop.luid) &&
                               std::memcmp(&d.AdapterLuid, prop.luid, sizeof(prop.luid)) == 0;
            if (match) {
                r.known = true;
                r.dedicated = d.DedicatedVideoMemory;
                r.shared = d.SharedSystemMemory;
            }
            ad->Release();
            if (match) break;
        }
        fac->Release();
    }
#endif
    // test hook: a simulated carve-out size (GiB), e.g. WHIRL_FAKE_DEDICATED_GB=0.5
    if (const char* v = std::getenv("WHIRL_FAKE_DEDICATED_GB"); v && *v) {
        char* end = nullptr;
        const double gb = std::strtod(v, &end);
        if (end != v && gb >= 0) {
            r.known = true;
            r.fake = true;
            r.dedicated = static_cast<std::uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
        }
    }
    std::lock_guard<std::mutex> lk(g_mu);
    g_last = r;
    g_have = true;
    return r;
}

const UmaMem* lastUmaMemory() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_have ? &g_last : nullptr;
}

}  // namespace whirl::hip
