// Default size of the server's pinned-RAM tier of the prefix cache.
// SPDX-License-Identifier: Apache-2.0
//
// Pure function (no OS calls), so the host tests can run it for any machine:
// the server passes the physical / available RAM it read at startup.
//
//   explicit size (--kv-ram-mb / WHIRL_KV_RAM_MB)  -> used as given (0 = off)
//   integrated GPU (UMA)                          -> off (the GPU's memory is
//                                                    system RAM already)
//   otherwise                                      -> 1/4 of physical RAM (nearest GiB),
//       at least the floor (8 GiB, or one full-length session + checkpoints
//       if more), at most 32 GiB (the floor wins over the cap), and never
//       more than half of the RAM available at startup (so the pinned arena
//       does not push the machine into paging); below 1 GiB the tier is off.
// Automatic sizes are whole GiB.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

namespace whirl::tier {

constexpr std::uint64_t ram_auto_min_mb = 8192;    // floor without a session size
constexpr std::uint64_t ram_auto_cap_mb = 32768;   // cap of the 1/4-of-RAM rule
constexpr std::uint64_t ram_auto_off_mb = 1024;    // availability cap below this: off

struct RamTierInput {
    std::optional<std::uint64_t> explicit_mb;  // --kv-ram-mb or WHIRL_KV_RAM_MB
    const char* explicit_src = "--kv-ram-mb";
    std::uint64_t total_phys = 0;  // bytes (GlobalMemoryStatusEx ullTotalPhys); 0 = unknown
    std::uint64_t avail_phys = 0;  // bytes (ullAvailPhys); 0 = unknown
    std::uint64_t floor_mb = ram_auto_min_mb;  // max(8 GiB, one full session + checkpoints)
    bool integrated = false;
};

struct RamTierSize {
    std::uint64_t mb = 0;
    bool avail_limited = false;  // the availability cap lowered the size
    std::string reason;
};

inline RamTierSize ramTierSize(const RamTierInput& in) {
    constexpr std::uint64_t mib = 1ull << 20;
    constexpr std::uint64_t gib_mb = 1024;
    const auto gib = [](std::uint64_t mb) {
        char b[32];
        if (mb % 1024 == 0) std::snprintf(b, sizeof b, "%llu GiB", static_cast<unsigned long long>(mb / 1024));
        else std::snprintf(b, sizeof b, "%.1f GiB", static_cast<double>(mb) / 1024.0);
        return std::string(b);
    };
    RamTierSize r;
    if (in.explicit_mb) {
        r.mb = *in.explicit_mb;
        r.reason = std::string(in.explicit_src) + (r.mb == 0 ? " 0: host tiers off" : " " + std::to_string(r.mb));
        return r;
    }
    if (in.integrated) {
        r.reason = "integrated GPU (shared system memory): off by default; --kv-ram-mb N turns it on";
        return r;
    }
    const std::uint64_t floor_mb = std::max(in.floor_mb, ram_auto_min_mb);
    std::uint64_t mb;
    if (in.total_phys == 0) {
        mb = floor_mb;
        r.reason = "physical RAM unknown: the minimum (" + gib(floor_mb) + ")";
    } else {
        const std::uint64_t quarter = (in.total_phys / 4 / mib + gib_mb / 2) / gib_mb * gib_mb;  // nearest GiB
        const std::uint64_t capped = std::min(quarter, ram_auto_cap_mb);
        mb = std::max(capped, floor_mb);
        r.reason = "auto: 1/4 of " + gib((in.total_phys + mib / 2) / mib) + " RAM";
        if (quarter > ram_auto_cap_mb) r.reason += ", capped at " + gib(ram_auto_cap_mb);
        if (capped < floor_mb) r.reason += ", raised to the minimum " + gib(floor_mb) + " (8 GiB or one full-length session)";
    }
    if (in.avail_phys > 0) {
        const std::uint64_t half = (in.avail_phys / 2 / mib) / gib_mb * gib_mb;
        if (half < mb) {
            r.avail_limited = true;
            if (half < ram_auto_off_mb) {
                r.reason += "; only " + gib(in.avail_phys / mib) + " RAM available at startup: off (half of it would be under 1 GiB)";
                mb = 0;
            } else {
                r.reason += "; limited to " + gib(half) + " = half of the " + gib(in.avail_phys / mib) + " RAM available at startup (wanted " +
                            gib(mb) + ")";
                mb = half;
            }
        }
    }
    r.mb = mb;
    return r;
}

}  // namespace whirl::tier
