// KV pool vs the WDDM video memory budget after the pool is allocated (issue #5).
// SPDX-License-Identifier: Apache-2.0
//
// The server sizes the KV pool from the WDDM local budget it reads before allocating the pool. That
// budget can be lower a moment later: in issue #5 (R9700, 12 GiB RAM tier) it read 31.02 GiB when the
// pool was sized and 29.55 GiB right after the pool was allocated, so the server started 0.37 GiB over
// the budget and Windows began moving its GPU memory to system RAM. The server therefore reads the
// budget again once the pool exists and shrinks the pool (whole pages) until local usage + margin fits.
//
// Header free of HIP: the arithmetic is unit-tested on the CPU with injected budget values.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace whirl::poolbudget {

struct Trim {
    std::uint32_t pool_tokens = 0;  // the pool to keep (== the input when nothing changes)
    bool shrink = false;            // pool_tokens < the input
    bool at_floor = false;          // the fit needs a pool below min_tokens: kept at min_tokens, still over
    std::uint64_t over = 0;         // local + margin - budget before the shrink (0: fits)
};

// local_usage / local_budget: WDDM numbers read with the pool allocated; margin: what must stay free
// under the budget; per_tok: KV bytes per token; page: tokens per page; min_tokens: smallest pool kept.
// Freed bytes are assumed to lower the local usage by exactly their size (WDDM counts a few percent
// more per allocation, so the real fit is a little better). local_budget 0 (unknown): no change.
Trim fitPool(std::uint32_t pool_tokens, std::uint64_t local_usage, std::uint64_t local_budget, std::uint64_t margin,
             std::uint64_t per_tok, std::uint32_t page, std::uint32_t min_tokens);

// Default WDDM budget margin (v0.2.2.1). The owner, 2026-10-21: the margin must cover the other programs that use
// the same GPU (desktop compositor, browser, Docker, other apps), looking at what they use when the server starts.
// Measured on the R9700 (rel0221): with another process holding 3.3 GiB, the budget read 31.02 GiB before WHIRL
// allocated its pool (the others not counted yet) and 28.08 GiB right after (now counted). So the margin has two parts:
//   pending = what other processes used at startup minus what the budget has dropped since startup (the part of
//             their usage the budget does not show yet; 0 once it does, so it is never counted twice);
//   room    = max(floor, 6% of the budget): room for other programs to grow later (issue #5: the budget fell
//             1.47 GiB right after startup); floor = 3072 MiB with a monitor on the card, else 2048.
// margin = pending + room. Without the other-process counters: pending = 0 (budget-only rule).
// WHIRL_POOL_BUDGET_MARGIN_MB sets the room part (pending is always counted).
inline constexpr std::uint64_t margin_floor_mb = 2048;
inline constexpr std::uint64_t margin_floor_display_mb = 3072;
inline constexpr double margin_budget_frac = 0.06;
inline constexpr std::uint64_t moe_extra_mb = 0;
struct MarginIn {
    std::uint64_t local_budget = 0;   // the budget now (0: unknown)
    std::uint64_t budget_start = 0;   // the budget read when the server started (0: unknown, treated as now)
    bool moe = false;
    bool others_known = false;        // the other-process counters were read at startup
    std::uint64_t others = 0;         // VRAM other processes used on this GPU at startup
    bool display = false;             // a monitor is attached to this GPU
    std::optional<std::uint64_t> room_override;  // WHIRL_POOL_BUDGET_MARGIN_MB (bytes)
};
struct Margin {
    std::uint64_t bytes = 0;    // pending + room, whole MiB
    std::uint64_t pending = 0;  // other processes' VRAM the budget does not show yet
    std::uint64_t room = 0;     // room for growth
    std::string why;            // for the startup log
};
Margin defaultMargin(const MarginIn& in);

// One shrink round freed `freed` bytes: true when the local usage read afterwards fell by at least half of it.
// The re-check stops otherwise, so a WDDM counter that has not caught up yet cannot make it shrink again.
bool usageFell(std::uint64_t usage_before, std::uint64_t usage_after, std::uint64_t freed);

// VRAM the vision encoder (--mmproj) adds after startup, on top of what startup allocated (REVIEW-GOV-3 question 4).
struct VisionVram {
    std::uint64_t weights = 0;    // encoder weights made resident on the first image (0 in stream mode)
    std::uint64_t act_extra = 0;  // activation buffers of the largest image that do not fit the lent prefill scratch
};
// The bytes to keep free for the encoder when the KV pool is sized: weights + act_extra + vision_fixed_mb (measured
// 12.5 MiB on the R9700 for both product models, see pool_budget.cpp); 0 when both are 0.
inline constexpr std::uint64_t vision_fixed_mb = 16;
std::uint64_t visionReserve(const VisionVram& v);
// The bytes left for the KV pool: avail minus the reserve (0 when it does not fit). reserve 0 (no --mmproj): avail.
std::uint64_t kvAvail(std::uint64_t avail, std::uint64_t reserve);

// The VRAM the KV pool may use now (arch025-N10b: one copy of the rule server_main.cpp wrote three times - buildModel's
// pre-load fit, ServerRun::planSlots' slot re-check and ServerRun::planPool's pool sizing). Free VRAM minus the reserve
// (WHIRL_POOL_RESERVE_MB, default defaultReserveMb); capped by the WDDM room (budget - usage - margin) unless the reserve
// was given or the WDDM numbers are unavailable; minus the desktop headroom (WHIRL_VRAM_HEADROOM_MB); minus the vision
// encoder's reserve (--mmproj) unless the reserve was given. Pure: the caller measures free VRAM and the WDDM numbers.
inline constexpr std::uint32_t defaultReserveMb(bool moe, bool wddm_ok) { return moe ? 1536 : (wddm_ok ? 768 : 3072); }
struct RoomIn {
    std::uint64_t free = 0;          // hip::MemInfo::free, read now
    std::uint64_t reserve = 0;       // bytes
    bool reserve_explicit = false;   // WHIRL_POOL_RESERVE_MB set: the old free-VRAM-only rule
    bool wddm_ok = false;            // hip::WddmMemInfo::ok
    std::uint64_t local_budget = 0;  // WDDM local budget / usage
    std::uint64_t local_usage = 0;
    std::uint64_t margin = 0;        // Margin::bytes for that budget
    std::uint64_t headroom = 0;      // desktop headroom
    std::uint64_t vis_reserve = 0;   // visionReserve (0: text only)
};
struct Room {
    std::uint64_t after_reserve = 0;   // free minus the reserve
    std::uint64_t wddm_room = 0;       // budget - usage - margin (0 when over)
    std::uint64_t after_headroom = 0;  // after the WDDM cap and the headroom
    std::uint64_t bytes = 0;           // the result
};
Room kvRoom(const RoomIn& in);
// The "kv pool sizing:" startup note for that computation (margin_why: Margin::why; vis_note: what the vision reserve is).
std::string kvRoomNote(const RoomIn& in, const Room& r, std::string_view margin_why, std::string_view vis_note);

}  // namespace whirl::poolbudget
