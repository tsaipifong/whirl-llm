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
#include <string_view>
#include <vector>

namespace whirl::startfit {

// 1. RAM tier. A 24064 MiB RAM tier failed with hipError 719 on the first request on the R9700 test machine (0.2.2 too;
// 20480 MiB works; cause not known yet, task_063b1573), so every RAM tier, explicit or automatic, is capped at
// ram_tier_safe_mb until the cause is known. Then (v0.2.2.2: the cap first, so asking for more never gives less) an explicit
// size above half of the RAM available now is lowered to half of it (whole 512 MiB). auto_mb: unused (kept for callers).
inline constexpr std::uint64_t ram_tier_safe_mb = 20480;
// GPU-pinned room for the RAM tier (kvram719 / s8-719). The RAM tier is pinned host memory mapped to every HIP GPU, and
// Windows limits how much of it the GPUs can keep resident: past the limit the allocations still succeed, but the first
// request fails with hipError 719 (AMD_LOG_LEVEL=1: "PAL failed to submit CMD! result:-5"). pinnedRoomMb takes the
// smallest room of: each HIP GPU (a discrete GPU's DXGI non-local budget / the number of HIP GPUs; an integrated GPU,
// which reports a non-local budget of 0, its SharedSystemMemory), minus this process's pinned memory there and
// pinned_reserve_mb; and the machine (half of the RAM minus every program's GPU-pinned memory, largest adapter, minus
// pinned_machine_reserve_mb). Measured: kvram719 (R9700 box, 64 GB carve-out) and s8-719 (8060S at 96 GB with the
// R9700 attached: 4096 MiB worked, 8192 failed; the R9700's 15435 MiB budget / 2 was the limit). Every RAM tier,
// explicit or automatic, is lowered to that room after the 20480 MiB cap, with an "adjusted:" line for an explicit
// size (a server that answers with a smaller tier instead of a server whose every request fails); a room below one
// max-context entry turns the automatic tier off (ramTierFit entry_mb); an explicit size is only clamped.
// Machine reserve 2048 (REVIEW-S8-719 A2; was 6144, carried over from kvram719): the largest value every recorded data
// point admits. It still catches kvram719's failure (another server had 10890 MiB pinned: 32587 - 12178 - 2048 = 18361
// < 20480 failed) and admits s23_ram12288_vis (8704 answered at 10161 MiB 8060S Shared Usage: 16203 - 1441 - 2048 =
// 12714); 6144 would have cut that to 8192 and lowered single-R9700 32 GB-RAM hosts' automatic tier with no failure
// on record.
inline constexpr std::uint64_t pinned_reserve_mb = 2048;
inline constexpr std::uint64_t pinned_machine_reserve_mb = 2048;
inline constexpr std::uint64_t pinned_room_unknown = ~0ull;
struct PinnedGpu {                     // one HIP GPU's adapter, as this process sees it (DXGI)
    std::uint64_t nonlocal_budget = 0;  // bytes: DXGI non-local budget; 0 on an integrated GPU (the 8060S reports 0)
    std::uint64_t own_nonlocal = 0;     // bytes: this process's pinned memory on that adapter so far (DXGI non-local usage;
                                        // on an integrated GPU, which reports 0 there, PDH process Shared Usage)
    std::uint64_t shared_system = 0;    // bytes: DXGI SharedSystemMemory of that adapter ("shared GPU memory")
    std::uint64_t adapter_shared = 0;   // bytes: PDH "GPU Adapter Memory" Shared Usage of that adapter, every process
                                        // (this one included; 0: unknown) (REVIEW-S8-719 C3)
};
struct PinnedState {
    std::uint64_t total_phys = 0;  // bytes; 0 = unknown
    std::uint64_t pinned_max = 0;  // bytes: GPU Shared Usage of the largest adapter (all processes, this one included)
    std::vector<PinnedGpu> gpus;   // every HIP GPU a pinned buffer is mapped to (empty = unknown)
    std::uint64_t later = 0;       // bytes this process pins after the measurement (the mmproj; REVIEW-S8-719-3 N3-4)
};
// MiB the RAM tier may still pin, whole 512 MiB (0: none); pinned_room_unknown when nothing is known.
std::uint64_t pinnedRoomMb(const PinnedState& s);
// Bytes the vision encoder pins for an mmproj file of file_bytes (PinnedState::later). The file size alone is not an
// upper bound (REVIEW-B0 N1): Vision::load pads its layout (vision.cpp: 256-byte / 4-KiB alignment, n_ff_pad, f32
// position embedding and biases), so Swift's 884.6 MiB mmproj pinned 887 MiB (REVIEW-S8-719-3 N3-4).
std::uint64_t mmprojPinnedBytes(std::uint64_t file_bytes);
// The reason text of a RAM tier lowered to the pinned room (log and "adjusted:" line). It must match none of the error
// patterns the gates and repro scripts scan server logs for (golden.ps1, kvram719_tier.ps1, fix719.ps1: "hipError",
// "hip ... 719", "| failed:", "PAL failed"): a healthy, adjusted server must not read as a failed one (REVIEW-S8-719 A1).
std::string pinnedWhy(std::uint64_t room_mb, std::size_t n_gpus);
// What is appended to pinnedWhy when the pinned room turns the tier off (mb 0): below_entry (the room is below one
// max-context entry, automatic size only) or no room at all. With the RAM tier the SSD tier behind it is off too, and
// the text says so (REVIEW-S8-719-2 C-1).
std::string pinnedOffWhy(bool below_entry, std::uint64_t entry_mb);
struct RamFit {
    std::uint64_t mb = 0;
    bool half = false;    // lowered because it was more than half of the available RAM
    bool safe = false;    // capped at ram_tier_safe_mb
    bool pinned = false;  // lowered to the GPU-pinned room (pinned_room_mb)
    bool below_entry = false;  // automatic size: the pinned room is below one max-context entry (entry_mb): off (mb 0)
};
// nullopt: unchanged. explicit_size: the user gave the size (--kv-ram-mb / WHIRL_KV_RAM_MB).
// entry_mb: one max-context entry (MiB); an automatic tier the pinned room lowers below it is turned off (FR-CACHE P3-4,
// ram_size.h). An explicit size is clamped to the room and otherwise used as given (monotonic in the request).
std::optional<RamFit> ramTierFit(std::uint64_t mb, std::uint64_t avail_phys_bytes, std::uint64_t auto_mb, bool explicit_size,
                                 std::uint64_t pinned_room_mb = pinned_room_unknown, std::uint64_t entry_mb = 0);
// fix024-ramdip: one sample of the system memory status (server_main.cpp fills it from GlobalMemoryStatusEx; tests
// inject it). ramForTier picks the RAM the tier sizing (tier::ramTierSize, ramTierFit) counts as available: Windows'
// Available (standby + free + zero) plus the modified list (pages written but not yet on disk; reusable once the
// writer flushes them, which pinning makes Windows do first). Available alone fell to 233-587 MiB for minutes while
// 70 GB of GGUF were written (S0b) and the one startup sample turned both host tiers off for the server's life.
// Linux's MemAvailable counts dirty file pages the same way; llama.cpp never gates its prompt cache on the startup
// Available at all (server-task.cpp). File pages mapped in another program's working set are still not counted.
struct RamStatus {
    std::uint64_t total_phys = 0;    // ullTotalPhys (bytes; 0 = unknown)
    std::uint64_t avail_phys = 0;    // ullAvailPhys: standby + free + zero (bytes; 0 = unknown)
    std::uint64_t modified = 0;      // PDH \Memory\Modified Page List Bytes (bytes; 0 = none or unknown)
};
struct RamForTier {
    std::uint64_t avail = 0;  // bytes the sizing counts as available
};
RamForTier ramForTier(const RamStatus& s);
// The texts of the RAM tier's startup lines (server_main.cpp), from the fit. rf: ramTierFit's result (nullopt: unchanged or
// no host tiers); asked_mb: the size before the fit; src: where an explicit size came from (--kv-ram-mb / WHIRL_KV_RAM_MB);
// size_reason: tier::RamTierSize::reason; pinned_why: pinnedWhy() ("" when the room was not measured); avail_mb: RAM
// available now.
struct RamTierText {
    std::string pinned_why;    // pinnedWhy, plus pinnedOffWhy when the room turns the tier off
    std::string adjusted_why;  // explicit size: the reason of the "adjusted:" line ("" when unchanged)
    std::string off_reason;    // what "kv tier: RAM tier off (...)" names
    std::string size_reason;   // what "kv tier: RAM tier size N MiB (...)" names
};
RamTierText ramTierText(const std::optional<RamFit>& rf, std::uint64_t asked_mb, bool explicit_size, std::string_view src,
                        std::string_view size_reason, std::string pinned_why, std::uint64_t entry_mb, std::uint64_t avail_mb);

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
