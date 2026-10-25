// Integrated GPU (Radeon 8060S, Ryzen AI Max) memory: the dedicated carve-out and the
// pre-load "does the model fit in it" check (UMA-1, GitHub issue #7).
// SPDX-License-Identifier: Apache-2.0
//
// Why this exists (measured on the owner's ROG Flow Z13, 64 GB carve-out + 64 GB OS RAM,
// HIP runtime 7.2 on Windows, progress/uma1.log):
//  - hipDeviceProp_t::totalGlobalMem / hipMemGetInfo report 99.7 GiB: the carve-out plus a
//    credit for the GART aperture in system RAM (ROCm CLR rocclr/device/pal/paldevice.cpp,
//    globalMemSize_ for APUs). The carve-out alone is DXGI_ADAPTER_DESC1::DedicatedVideoMemory
//    (63.8 GiB); SharedSystemMemory is 47.7 GiB; the Vulkan device-local heap (llama.cpp:
//    114,327 MiB) is about the sum of both.
//  - hipMalloc past the carve-out does succeed (the excess shows as GPU "Shared Usage"), but GPU
//    copies of that memory ran at 65-75 GB/s against 209-233 GB/s inside the carve-out, and with a
//    0.5 GB carve-out the runtime itself fails (hipError 719 from hipMemGetInfo /
//    hipModuleLoadData, "PAL failed to finalize a command buffer", issue #7). So WHIRL keeps
//    weights, buffers and the KV cache inside the carve-out and refuses before loading when they
//    do not fit, naming the setting to change.
// The DXGI read needs only hipGetDeviceProperties (for the adapter LUID): no device memory call,
// so it runs before anything that would fail with a tiny carve-out.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <optional>
#include <string>
#include <string_view>

namespace whirl::hip {

struct UmaMem {
    bool integrated = false;       // the device is an integrated GPU (hipDeviceProp_t::integrated)
    bool known = false;            // dedicated / shared below are valid (DXGI matched, or faked)
    bool fake = false;             // dedicated comes from WHIRL_FAKE_DEDICATED_GB (test hook)
    std::uint64_t dedicated = 0;   // carve-out (DXGI DedicatedVideoMemory)
    std::uint64_t shared = 0;      // DXGI SharedSystemMemory (0 if unknown)
};

// The device's carve-out. For a discrete GPU: integrated = false and nothing else is read.
// WHIRL_FAKE_DEDICATED_GB=N (GiB, may be fractional) replaces the dedicated size of an
// integrated GPU, so the refusal path can be tested without changing the BIOS.
// Remembers the result for lastUmaMemory().
UmaMem umaMemory(int device);
// The last umaMemory() result of this process for an integrated GPU (the hipError 719 hint),
// nullptr if none was read.
const UmaMem* lastUmaMemory();

// Carve-out WHIRL does not plan to use: the desktop / runtime part of it. Measured: with 63.83 GiB
// dedicated, hipMalloc started to spill to Shared Usage at 62.32 GiB (0.16 GiB in use by others).
inline constexpr std::uint64_t uma_reserve = 1536ull << 20;

// WHIRL_UMA_CHECK=0: no refusal and no KV pool cap (load anyway; unsupported). Read on every call (once per load /
// sizing estimate); defined in src/hip/uma_mem.cpp (platform::env::get, arch025 PL1b)
bool umaCheckOn();

inline std::uint64_t umaUsable(std::uint64_t dedicated) { return dedicated > uma_reserve ? dedicated - uma_reserve : 0; }

// What the load needs in the carve-out.
struct UmaNeed {
    std::uint64_t fixed = 0;         // weights + state + buffers + checkpoints + slack
    std::uint64_t weights = 0;       // part of fixed, for the message
    std::uint64_t kv_per_token = 0;  // in the KV format the mode uses
    std::uint32_t min_tokens = 0;    // context the load cannot do without (one request / --ctx)
    std::uint32_t want_tokens = 0;   // the default (or requested) KV pool
    std::uint64_t at(std::uint32_t tokens) const { return fixed + kv_per_token * tokens + uma_reserve; }
};

// "dedicated 63.8 GiB (+ shared 47.7 GiB)" for device lists and the startup log.
inline std::string umaSizeText(const UmaMem& u) {
    constexpr double g = 1024.0 * 1024.0 * 1024.0;
    std::string s = std::format("dedicated {:.1f} GiB", static_cast<double>(u.dedicated) / g);
    if (u.shared) s += std::format(" (+ shared {:.1f} GiB)", static_cast<double>(u.shared) / g);
    if (u.fake) s += " [WHIRL_FAKE_DEDICATED_GB]";
    return s;
}

// The setting to change, in the words the user sees in the vendor tools.
inline const char* umaSettingHint() {
    return "raise the GPU memory (UMA frame buffer size) in Armoury Crate or the BIOS\n"
           "  (on other laptops: \"Variable Graphics Memory\" in AMD Software, or the BIOS iGPU memory setting),\n"
           "  then restart Windows";
}

// Whole GB to ask for (BIOS / Armoury Crate sizes are whole GB).
inline std::uint64_t umaGbFor(std::uint64_t bytes) {
    return static_cast<std::uint64_t>(std::ceil(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)));
}

// Refusal text when the load cannot fit the carve-out (empty: it fits).
inline std::string umaRefusal(const UmaMem& u, const UmaNeed& n, std::string_view gpu_name) {
    if (!u.integrated || !u.known) return {};
    const std::uint64_t need_min = n.at(n.min_tokens);
    if (u.dedicated >= need_min) return {};
    constexpr double g = 1024.0 * 1024.0 * 1024.0;
    const std::uint32_t want = std::max(n.want_tokens, n.min_tokens);
    std::string s = std::format(
        "the {} has {:.1f} GB of dedicated GPU memory (the UMA carve-out set in Armoury Crate / the BIOS){}; this model "
        "needs about {:.1f} GB of it (weights {:.1f} GB + buffers and state {:.1f} GB + KV cache for {} tokens {:.1f} GB + "
        "{:.1f} GB kept for the desktop).\n"
        "  To run it, {}:\n"
        "  at least {} GB{}. A smaller context (--ctx-per-slot) or fewer server slots (--parallel) needs less.\n"
        "  (The system memory WHIRL reports beyond the carve-out is not used for the model: it is slower, and with a small\n"
        "  carve-out the GPU runtime fails with hipError 719. WHIRL_UMA_CHECK=0 skips this check.)\n",
        gpu_name, static_cast<double>(u.dedicated) / g, u.fake ? " [simulated: WHIRL_FAKE_DEDICATED_GB]" : "", need_min / g,
        n.weights / g, (n.fixed - std::min(n.fixed, n.weights)) / g, n.min_tokens,
        static_cast<double>(n.kv_per_token) * n.min_tokens / g, uma_reserve / g, umaSettingHint(), umaGbFor(need_min),
        want > n.min_tokens ? std::format(", or {} GB for the full {}-token context", umaGbFor(n.at(want)), want) : std::string());
    return s;
}

// KV pool tokens that fit the carve-out after the fixed part (page multiple), or nullopt when
// the wanted pool fits. Never below min_tokens (umaRefusal stops the load before that).
inline std::optional<std::uint32_t> umaPoolCap(const UmaMem& u, const UmaNeed& n, std::uint32_t page) {
    if (!u.integrated || !u.known || n.kv_per_token == 0) return std::nullopt;
    if (u.dedicated >= n.at(n.want_tokens)) return std::nullopt;
    const std::uint64_t room = umaUsable(u.dedicated) > n.fixed ? umaUsable(u.dedicated) - n.fixed : 0;
    std::uint64_t t = room / n.kv_per_token / page * page;
    t = std::max<std::uint64_t>(t, n.min_tokens);
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(t, n.want_tokens));
}

// Extra lines for a hipError 719 on an integrated GPU (empty otherwise).
inline std::string uma719Hint(const UmaMem* u) {
    if (!u || !u->integrated) return {};
    std::string s = "  On a Radeon 8060S / Ryzen AI Max this error usually means the dedicated GPU memory (UMA carve-out) is too small";
    if (u->known) s += std::format(": it is {:.1f} GB now", static_cast<double>(u->dedicated) / (1024.0 * 1024.0 * 1024.0));
    s += ".\n  The README lists the size each model needs. To fix it, ";
    s += umaSettingHint();
    s += ".\n";
    return s;
}

}  // namespace whirl::hip
