// Startup fit of explicit settings to the measured resources (v0.2.2.1).
// SPDX-License-Identifier: Apache-2.0
//
// Owner, 2026-10-21: 「使用者的確也可能設置了很爛的配置而沒有考量實際資源，所以我們系統先查而調整反而是好事，
// 再將調整結果寫在log」. When a setting the user gave does not fit what the server measured at startup (system RAM,
// the WDDM budget with its margin, other programs' VRAM), the server lowers it to what fits and logs one
// "adjusted:" line per setting, instead of refusing or overcommitting. These are the decisions, free of HIP so
// tests/unit can drive them with injected values; server_main.cpp applies them in this order (each only when
// still needed):
//   1. RAM tier (--kv-ram-mb / WHIRL_KV_RAM_MB): pinned before anything else, checked against system RAM;
//   2. --parallel: each slot's recurrent state and prefix checkpoints are fixed VRAM; fewer slots when one full
//      request no longer fits next to them: in the pre-load estimate, and again after the weights are loaded with
//      the measured VRAM / budget (slotsThatFit), before any slot is allocated;
//   3. prefill batch / checkpoint counts set by the user (WHIRL_PREFILL_BATCH, WHIRL_SERVE_CKPTS, WHIRL_SYS_CKPTS);
//   4. -c / --ctx (the shared KV pool) and 5. --ctx-per-slot: numerics::serverPoolFit with "not explicit", so it
//      lowers instead of refusing; the per-request context reserves no memory by itself, so it is capped last.
// Below the minimum (one slot, a 4096-token pool) the server still refuses to start.

#pragma once

#include "whirl/vram_limit.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace whirl::startfit {

// 1. RAM tier. Windows lets the GPUs lock only about half of the physical RAM in total (all processes, all
// adapters; a hipHostMalloc buffer is mapped to every GPU). Past that limit the pinned allocations still succeed,
// but the driver cannot make them resident when work is submitted (AMD_LOG_LEVEL=1: "PAL failed to submit CMD!
// result:-5") and the first request fails with hipError 719 (kvram719: --kv-ram-mb 24064 on a 63.6 GiB PC with
// another WHIRL server pinning ~10 GiB; a standalone hipHostMalloc probe fails the same way, without WHIRL).
// So the RAM tier, explicit or automatic, is lowered to what the limit leaves: pinnedRoomMb(total RAM, the GPU-pinned
// memory of all processes now). An explicit size above half of the RAM available now is also lowered: to the default
// size (auto_mb; 0 = none), never above half of the available RAM (whole 512 MiB).
inline constexpr std::uint64_t pinned_reserve_mb = 2048;  // pinned after the tier (HIP staging, vision, checkpoints) + slack
inline constexpr std::uint64_t pinned_room_unknown = ~0ull;
// MiB the RAM tier may still pin: total_phys / 2 - pinned_now - pinned_reserve_mb, whole 512 MiB, 0 when none is
// left; pinned_room_unknown when total_phys is unknown (0).
std::uint64_t pinnedRoomMb(std::uint64_t total_phys_bytes, std::uint64_t pinned_now_bytes);
struct RamFit {
    std::uint64_t mb = 0;
    bool half = false;    // lowered because it was more than half of the available RAM
    bool pinned = false;  // lowered to the GPU-pinned room (pinned_room_mb)
};
// nullopt: unchanged. explicit_size: the user gave the size (--kv-ram-mb / WHIRL_KV_RAM_MB).
std::optional<RamFit> ramTierFit(std::uint64_t mb, std::uint64_t avail_phys_bytes, std::uint64_t auto_mb, bool explicit_size,
                                 std::uint64_t pinned_room_mb = pinned_room_unknown);

// the pool the plan must reach: max(pool_min_tokens, pool_hard_min)
std::uint64_t wantTokens(const vram::FitInput& fi);

// 2. --parallel: with the plan for `fi` (fi.parallel slots) short of wantTokens, remove slots one at a time
// (each frees per_slot_bytes of fixed VRAM) until it fits or one slot is left. Only when the user gave --parallel
// (given); otherwise the plan is returned unchanged.
struct ParallelFit {
    std::uint32_t parallel = 1;
    vram::FitInput fi;
    vram::FitPlan plan;
    std::uint64_t pool_requested = 0;  // the plan's pool with the requested slots
};
ParallelFit fitParallel(const vram::FitInput& fi, std::uint64_t per_slot_bytes, bool given);

// 3. user-set prefill batch / checkpoint counts: when the plan is still short, plan again with them free; the
// result is used only if it gives a larger pool.
struct FixedFit {
    bool changed = false;
    bool batch_changed = false, ck_changed = false;
    vram::FitInput fi;
    vram::FitPlan plan;
};
FixedFit fitFixed(const vram::FitInput& fi, const vram::FitPlan& plan);

// 2b. After the weights are loaded: how many slots leave a pool of at least target_tokens, given the VRAM usable now
// (avail), the fixed VRAM per slot and the shared fixed VRAM. Never below 1 and never above `parallel`.
std::uint32_t slotsThatFit(std::uint32_t parallel, std::uint64_t avail, std::uint64_t per_slot, std::uint64_t shared,
                           std::uint64_t per_token, std::uint64_t target_tokens);

// The "startup" object of GET /props and GET /health (effective values after the startup fit).
struct StartupInfo {
    std::uint32_t ctx_per_request = 0, pool_tokens = 0, parallel = 0;
    std::uint64_t kv_ram_mb = 0;          // the RAM tier in use (0: none)
    std::uint64_t wddm_margin_mib = 0;    // the margin the KV pool was sized with (0: UMA / no WDDM numbers)
    std::uint64_t other_processes_mib = 0;  // other processes' VRAM at startup
    bool display_attached = false;
    std::vector<std::string> adjusted;    // one entry per setting the server lowered
};
std::string startupJson(const StartupInfo& s);

}  // namespace whirl::startfit
