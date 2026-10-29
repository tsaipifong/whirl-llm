// Startup fit of explicit settings (see include/whirl/startup_fit.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/startup_fit.h"

#include "whirl/json.h"
#include "whirl/small_helpers.h"  // alignUp (arch025-lane1 leg 8: contextPlan / ramFloor)

#include <algorithm>
#include <format>

namespace whirl::startfit {

std::uint64_t pinnedRoomMb(const PinnedState& s) {
    constexpr std::uint64_t MiB = 1ull << 20;
    std::uint64_t room = pinned_room_unknown;
    const auto take = [&](std::uint64_t limit_b, std::uint64_t used_b, std::uint64_t reserve_mb) {
        const std::uint64_t free_mb = limit_b > used_b ? (limit_b - used_b) / MiB : 0;
        room = std::min(room, free_mb > reserve_mb ? (free_mb - reserve_mb) / 512 * 512 : 0);
    };
    // per HIP GPU: a hipHostMalloc buffer is mapped to every HIP GPU and charged on each. A discrete GPU's DXGI non-local
    // budget holds this process's pinned memory times the number of HIP GPUs (kvram719, R9700 box: 22528 MiB worked,
    // 23552 failed at 48026 / 2). An integrated GPU reports a non-local budget of 0 (the 8060S); its shared GPU memory
    // (SharedSystemMemory) is the limit there, without the division (kvram719: 22528 MiB fit the 8060S's 32587).
    // REVIEW-S8-719 C3: what is pinned on that adapter by every process (PDH GPU Adapter Memory Shared Usage), not only
    // this one; this process alone when the adapter's counter is unknown (0)
    const std::uint64_t n = std::max<std::uint64_t>(s.gpus.size(), 1);
    for (const PinnedGpu& g : s.gpus) {
        const std::uint64_t used = std::max(g.own_nonlocal, g.adapter_shared) + s.later;  // + what is pinned later (N3-4)
        if (g.nonlocal_budget > 0) take(g.nonlocal_budget / n, used, pinned_reserve_mb);
        else if (g.shared_system > 0) take(g.shared_system, used, pinned_reserve_mb);
    }
    // machine: every program's GPU-pinned memory (largest adapter) well under half of the RAM (kvram719)
    if (s.total_phys > 0) take(s.total_phys / 2, s.pinned_max + s.later, pinned_machine_reserve_mb);
    return room;
}

std::uint64_t mmprojPinnedBytes(std::uint64_t file_bytes) {
    // REVIEW-B0 N1: the padded layout is a little larger than the file (measured +0.27%: 884.6 MiB file, 887 MiB pinned);
    // + 1/32 + 32 MiB covers alignment, n_ff_pad and the f16 -> f32 position embedding with room to spare
    return file_bytes ? file_bytes + file_bytes / 32 + (32ull << 20) : 0;
}

std::string pinnedWhy(std::uint64_t room_mb, std::size_t n_gpus) {
    // no "hip" anywhere, no "failed:" (REVIEW-S8-719 A1): log scanners match hipError / hip...719 case-insensitively
    return std::format("lowered to {} MiB: the GPUs can keep only so much pinned host memory resident ({} GPU(s) map it; "
                       "a larger RAM tier makes the first request fail with GPU error 719)",
                       room_mb, n_gpus);
}

std::string pinnedOffWhy(bool below_entry, std::uint64_t entry_mb) {
    return below_entry ? std::format("; that is below one max-context entry ({} MiB), so the host tiers are off (RAM and SSD)",
                                     entry_mb)
                       : std::string("; no room left, so the host tiers are off (RAM and SSD)");
}

std::optional<RamFit> ramTierFit(std::uint64_t mb, std::uint64_t avail_phys_bytes, std::uint64_t auto_mb, bool explicit_size,
                                 std::uint64_t pinned_room_mb, std::uint64_t entry_mb) {
    (void)auto_mb;
    if (mb == 0) return std::nullopt;
    RamFit r;
    r.mb = mb;
    // v0.2.2.2: the cap first, then the half-of-available-RAM rule on the capped size, which lowers to half (whole
    // 512 MiB): asking for more never gives less (0.2.2.1: 22528 -> 20480 but 24064 -> 16384). s8-719: the GPU-pinned
    // room after the cap (explicit and automatic sizes alike; each step is a min, so an explicit size stays monotonic).
    if (r.mb > ram_tier_safe_mb) {
        r.mb = ram_tier_safe_mb;
        r.safe = true;
    }
    if (pinned_room_mb != pinned_room_unknown && r.mb > pinned_room_mb) {
        r.mb = pinned_room_mb;
        r.pinned = true;
        // FR-CACHE P3-4 (src/tier/ram_size.h): the automatic size holds one max-context entry, else off (REVIEW-S8-719
        // C2). Not an explicit size: it is clamped to the room and used as given (ram_size.h), so asking for more never
        // gives less (REVIEW-S8-719-2 X1/X2: golden's 8192 on two GPUs at 96 GB -> 4096, not off).
        if (!explicit_size && r.mb > 0 && r.mb < entry_mb) {
            r.mb = 0;
            r.below_entry = true;
        }
    }
    // the half rule is a min with half of the available RAM in whole 512 MiB, so it is monotonic at every MiB too
    // (REVIEW-S8-719-3 N3-1: comparing the ask with the unrounded half gave 8800 -> 8800 but 8822 -> 8704)
    if (explicit_size && avail_phys_bytes > 0) {
        const std::uint64_t half_mb = (avail_phys_bytes / 2 >> 20) / 512 * 512;
        if (r.mb > half_mb) {
            r.mb = half_mb;
            r.half = true;
        }
    }
    if (!r.half && !r.safe && !r.pinned) return std::nullopt;
    return r;
}

RamTierText ramTierText(const std::optional<RamFit>& rf, std::uint64_t asked_mb, bool explicit_size, std::string_view src,
                        std::string_view size_reason, std::string pinned_why, std::uint64_t entry_mb, std::uint64_t avail_mb) {
    RamTierText t;
    t.pinned_why = std::move(pinned_why);
    t.off_reason = std::string(size_reason);  // what the "RAM tier off" line names (REVIEW-S8-719 N3)
    t.size_reason = std::string(size_reason);
    if (!rf) return t;
    // the pinned room itself turned the tier off (room 0, or below one entry for the automatic size); when the half rule
    // did it after the room (REVIEW-S8-719-3 N3-2) the room was > 0, so "no room left" would be wrong: the half rule is named
    const bool room_off = rf->mb == 0 && rf->pinned && !rf->half;
    if (room_off) {
        t.pinned_why += pinnedOffWhy(rf->below_entry, entry_mb);
        t.off_reason = std::format("{} {} MiB asked: {}", explicit_size ? std::string(src) : std::string("auto"), asked_mb, t.pinned_why);
    }
    // REVIEW-S8-719-3 N3-6: the size line names the reason of the size asked; say it was lowered, and to what
    if (rf->mb > 0 && rf->mb != asked_mb)
        t.size_reason = explicit_size ? std::format("{} asked, lowered to {} MiB: see the adjusted: line", size_reason, rf->mb)
                                      : std::format("{}; lowered to {} MiB: see the line above", size_reason, rf->mb);
    if (explicit_size) {
        std::string why;
        if (rf->safe) why = std::format("capped at {} MiB while a crash with larger RAM tiers is investigated", ram_tier_safe_mb);
        if (rf->pinned) why += std::format("{}{}", why.empty() ? "" : "; ", t.pinned_why);
        if (rf->half)
            why += std::format("{}the RAM tier is pinned host memory; more than half of the {} MiB of RAM available now "
                               "would make the system page, so it is lowered to half",
                               why.empty() ? "" : "; ", avail_mb);
        if (rf->mb == 0 && rf->half) {
            why += "; that leaves 0 MiB, so the host tiers are off (RAM and SSD)";
            t.off_reason = std::format("{} {} MiB asked: {}", std::string(src), asked_mb, why);
        }
        t.adjusted_why = why;
    }
    return t;
}

std::uint64_t wantTokens(const vram::FitInput& fi) { return std::max<std::uint64_t>(fi.pool_min_tokens, fi.pool_hard_min); }

ParallelFit fitParallel(const vram::FitInput& fi, std::uint64_t per_slot_bytes, bool given) {
    ParallelFit r;
    r.fi = fi;
    r.plan = vram::planFit(fi);
    r.parallel = fi.parallel;
    r.pool_requested = r.plan.pool_tokens;
    while (given && r.parallel > 1 && r.plan.pool_tokens < wantTokens(r.fi)) {
        --r.parallel;
        r.fi.parallel = r.parallel;
        r.fi.fixed = r.fi.fixed > per_slot_bytes ? r.fi.fixed - per_slot_bytes : 0;
        r.plan = vram::planFit(r.fi);
    }
    return r;
}

FixedFit fitFixed(const vram::FitInput& fi, const vram::FitPlan& plan) {
    FixedFit r;
    r.fi = fi;
    r.plan = plan;
    if (plan.pool_tokens >= wantTokens(fi) || !(fi.batch_fixed || fi.ck_fixed)) return r;
    vram::FitInput f2 = fi;
    f2.batch_fixed = f2.ck_fixed = false;
    const vram::FitPlan p2 = vram::planFit(f2);
    if (p2.pool_tokens <= plan.pool_tokens) return r;
    r.changed = true;
    r.batch_changed = fi.batch_fixed && p2.batch != plan.batch;
    r.ck_changed = fi.ck_fixed && (p2.n_ck != plan.n_ck || p2.n_spe != plan.n_spe);
    r.fi = f2;
    r.plan = p2;
    return r;
}

std::uint32_t slotsThatFit(std::uint32_t parallel, std::uint64_t avail, std::uint64_t per_slot, std::uint64_t shared,
                           std::uint64_t per_token, std::uint64_t target_tokens) {
    if (per_token == 0) return parallel;
    std::uint32_t p = std::max<std::uint32_t>(parallel, 1);
    while (p > 1 && poolTokensWith(p, avail, per_slot, shared, per_token) < target_tokens) --p;
    return p;
}

SlotFixed slotFixed(const SlotFixedIn& in) {
    SlotFixed r;
    r.per_slot = (in.state + in.replay + in.n_ck * in.ck + in.dflash_slot) * 104 / 100;
    r.shared = (in.n_spe * in.ck + static_cast<std::uint64_t>(in.snap_sets) * in.state) * 104 / 100 + (256ull << 20);
    return r;
}

std::uint64_t poolTokensWith(std::uint32_t n, std::uint64_t avail, std::uint64_t per_slot, std::uint64_t shared,
                             std::uint64_t per_token) {
    const std::uint64_t fixed = shared + per_slot * n;
    return avail > fixed && per_token ? (avail - fixed) / per_token : 0;
}

std::string startupJson(const StartupInfo& s) {
    std::string j = std::format("{{\"ctx_per_request\":{},\"pool_tokens\":{},\"parallel\":{},\"kv_ram_mb\":{},\"wddm_margin_mib\":{},"
                                "\"other_processes_mib\":{},\"display_attached\":{},\"adjusted\":[",
                                s.ctx_per_request, s.pool_tokens, s.parallel, s.kv_ram_mb, s.wddm_margin_mib, s.other_processes_mib,
                                s.display_attached ? "true" : "false");
    for (std::size_t k = 0; k < s.adjusted.size(); ++k) {
        j += k ? ",\"" : "\"";
        json::appendEscaped(j, s.adjusted[k]);  // bug-gatenul: control bytes as \u00XX too (RFC 8259 7)
        j += "\"";
    }
    return j + "]}";
}

PreloadAdjust preloadAdjust(const PreloadIn& in) {
    PreloadAdjust o;
    vram::FitInput fi = in.fi;
    vram::FitPlan plan = in.plan;
    std::uint32_t parallel = in.parallel;
    const auto adjust = [&o](const std::string& what, const std::string& why) { o.adjusted.push_back({what, why}); };
    const double g = 1024.0 * 1024.0 * 1024.0;
    const std::uint64_t av = in.avail, margin = in.margin;
    {
        const auto want = [&] { return wantTokens(fi); };
        const std::string budget_why =
            std::format("{:.2f} GiB usable under the Windows video memory budget, keeping {:.2f} GiB free for other programs "
                        "({:.2f} GiB they use that the budget does not show yet + {:.2f} GiB room)",
                        av / g, margin / g, in.pending / g, in.room / g);
        // an explicit -c that does not fit is lowered when the pool is sized (later); plan for one request here
        if (in.ctx_given && plan.pool_tokens < want()) {
            fi.pool_min_tokens = std::min<std::uint32_t>(in.slot_ctx, in.tight_pool);
            plan = vram::planFit(fi);
        }
        // 3. --parallel given: fewer slots while even one request does not fit next to every slot's fixed memory
        const std::uint32_t par0 = parallel;
        o.fitp = fi;
        o.per_slot = (in.st1 + in.pend1) + (in.st1 + in.pend1) / 25;
        const ParallelFit pfit = fitParallel(fi, o.per_slot, in.parallel_given);
        if (pfit.parallel != par0) {
            parallel = pfit.parallel;
            fi = pfit.fi;
            plan = pfit.plan;
            adjust(std::format("--parallel {} -> {}", par0, parallel),
                   std::format("with {} slots, their recurrent state and prefix checkpoints leave a KV pool of about {} tokens, "
                               "less than one {}-token request; {}",
                               par0, pfit.pool_requested, want(), budget_why));
        }
        // 2. prefill batch / checkpoint counts set by the user: lowered only when the minimum still does not fit
        o.fitf = fi;
        o.fitf_plan = plan;
        const FixedFit ffit = fitFixed(fi, plan);
        if (ffit.changed) {
            if (ffit.batch_changed)
                adjust(std::format("WHIRL_PREFILL_BATCH {} -> {}", plan.batch, ffit.plan.batch),
                       std::format("the prefill buffers left a KV pool of about {} tokens, less than {}; {}", plan.pool_tokens,
                                   want(), budget_why));
            if (ffit.ck_changed)
                adjust(std::format("prefix checkpoints {} per slot + {} shared -> {} + {} (WHIRL_SERVE_CKPTS / WHIRL_SYS_CKPTS)",
                                   plan.n_ck, plan.n_spe, ffit.plan.n_ck, ffit.plan.n_spe),
                       std::format("they left a KV pool of about {} tokens, less than {}; {}", plan.pool_tokens, want(),
                                   budget_why));
            fi = ffit.fi;
            plan = ffit.plan;
        }
    }
    if (plan.reduced)
        o.tight = std::format("VRAM is tight (about {:.2f} GiB usable for weights {:.2f} GiB + buffers + checkpoints + KV): reduced {} so the KV "
                              "pool gets about {} tokens (wanted {}); WHIRL_PREFILL_BATCH (at least 1280) / WHIRL_SERVE_CKPTS / WHIRL_SYS_CKPTS "
                              "choose these yourself, --ctx a smaller pool",
                              av / g, in.weights / g, plan.note, plan.pool_tokens, fi.pool_min_tokens);
    // not even the weights + smallest buffers fit (no room for any KV): the caller stops before a load that would run
    // out of memory part way through, with the "smaller quantization" advice (WHIRL_FIT_CHECK=0 loads anyway)
    if (!plan.fits && plan.pool_tokens == 0) {
        o.refuse = true;
        o.refusal = std::format("needs about {:.2f} GiB of VRAM before any KV cache (weights {:.2f} GiB + prefill batch {}, "
                                "state, checkpoints), about {:.2f} GiB usable of {:.2f} GiB{}",
                                plan.need_fixed / g, in.weights / g, plan.batch, av / g, in.total / g,
                                vram::limitBytes() ? " (" + vram::limitNote() + ")" : std::string());
    }
    if (!plan.fits)
        o.short_warn = std::format("VRAM estimate: weights + buffers + checkpoints {:.2f} GiB leave about {} KV tokens of {:.2f} GiB usable "
                                   "(at least {} needed); the model may not fit this GPU",
                                   plan.need_fixed / g, plan.pool_tokens, av / g, fi.pool_hard_min);
    o.parallel = parallel;
    o.fi = fi;
    o.plan = plan;
    return o;
}

UmaFixed umaFixed(const UmaFixedIn& in) {
    UmaFixed r;
    const std::uint64_t ck = in.st1 + static_cast<std::uint64_t>(in.n_embd) * 4 + static_cast<std::uint64_t>(in.n_vocab) * 4;
    // DeltaNet snapshot sets for MTP verify: one recurrent state per draft (8 dense, 1 MoE: the server log's
    // "N snapshot sets")
    const std::uint64_t snaps = std::max<std::uint32_t>(1, in.drafts) * in.st1;
    r.slots = in.parallel * (in.st1 + in.pend1) + snaps;
    r.ck = (in.parallel * in.n_ck + in.n_spe) * ck;
    r.fixed = in.weights + in.state + in.bufs + r.slots + r.ck + (in.has_mtp ? 65536ull * in.n_embd * 5 / 16 : 0) + (256ull << 20);
    r.fixed += r.fixed / 12;  // +8%: measured fast-mode Swift fixed part 3.9% above the itemized sum
    return r;
}

// (arch025-lane1 leg 8: src/server/server_main.cpp:348-352 at 5728296)
ContextPlan contextPlan(std::optional<std::uint32_t> ctx, std::optional<std::uint32_t> ctx_per_slot, bool uma, std::uint32_t page) {
    ContextPlan p;
    p.pool_req = ctx ? ctx : (uma ? std::optional<std::uint32_t>(uma_pool_default) : std::nullopt);
    p.slot_ctx = static_cast<std::uint32_t>(alignUp(
        ctx_per_slot ? *ctx_per_slot : (p.pool_req ? std::min(*p.pool_req, ctx_per_slot_default) : ctx_per_slot_default), page));
    return p;
}

// (server_main.cpp:457 at 5728296)
std::size_t ckptDefault(std::uint32_t parallel) { return parallel == 1 ? 4 : (parallel <= 4 ? 3 : 1); }

// (server_main.cpp:459-462 at 5728296)
CkptCap capCkpts(std::size_t n_ck, std::size_t max_ck) {
    CkptCap c;
    c.n = n_ck;
    if (n_ck > max_ck) {
        c.warn = std::format("WHIRL_SERVE_CKPTS={} is above the maximum ({}); using {}", n_ck, max_ck, max_ck);
        c.n = max_ck;
    }
    return c;
}

// (server_main.cpp:898 and :1049-1050 at 5728296)
std::uint64_t ckptBytes(std::uint64_t state, std::uint32_t n_vocab, std::uint32_t n_embd) {
    return state + static_cast<std::uint64_t>(n_vocab) * 4 + static_cast<std::uint64_t>(n_embd) * 4;
}

// (server_main.cpp:1048-1055 at 5728296)
RamFloor ramFloor(std::uint32_t slot_ctx, std::uint64_t kv_bytes_per_token_f16, std::size_t n_ck, std::uint64_t ck_stride,
                  std::uint64_t auto_min_mb) {
    RamFloor r;
    const std::uint64_t full = static_cast<std::uint64_t>(slot_ctx) * kv_bytes_per_token_f16 + n_ck * ck_stride;
    r.entry_mb = alignUp(full, 1ull << 30) >> 20;
    r.floor_mb = std::max(auto_min_mb, r.entry_mb);
    return r;
}

// ---- arch025-lane1 leg 9 (729c080 src/server/server_main.cpp, moved verbatim) -----------------------------------------
// (:1086-1094, reserveRamTier)
std::string pinnedGpusText(const std::vector<PinnedGpu>& gpus) {
    std::string per;
    for (const PinnedGpu& g : gpus)
        per += std::format("{}{} MiB {} (this process {} MiB, all programs {})", per.empty() ? "" : ", ",
                           (g.nonlocal_budget > 0 ? g.nonlocal_budget : g.shared_system) >> 20,
                           g.nonlocal_budget > 0 ? "non-local budget" : "shared GPU memory", g.own_nonlocal >> 20,
                           g.adapter_shared ? std::format("{} MiB", g.adapter_shared >> 20) : std::string("unknown"));
    return per;
}

// (:1408-1410, reportPool)
double ckptMib(std::uint64_t state, std::uint32_t n_vocab, std::uint32_t n_embd) {
    return static_cast<double>(ckptBytes(state, n_vocab, n_embd)) / (1024.0 * 1024.0);
}

// (:1276-1279, planPool)
std::uint32_t defaultPoolTokens(std::uint64_t avail, std::uint64_t per_tok, std::uint64_t pool_cap, std::uint32_t kv_page) {
    const std::uint64_t t = std::min(avail / per_tok, pool_cap);
    return static_cast<std::uint32_t>(t / kv_page * kv_page);
}

// (:1280-1285, planPool)
std::string poolRefusal(std::uint32_t pool_tokens) {
    if (pool_tokens >= pool_min_tokens) return {};
    return std::format("not enough GPU memory left for the KV cache: a pool of {} tokens (at least 4096 needed).\n"
                       "  Use a smaller model or quantization, give --ctx explicitly,\n"
                       "  or close other programs that use the GPU.",
                       pool_tokens);
}

// (:1395-1403, planPool)
std::string poolFloorWarn(std::uint32_t pool_now, std::uint32_t slot_ctx, std::uint32_t parallel, bool int8_hint) {
    const std::uint32_t second = parallel > 1 ? std::min(floor_second_ctx, slot_ctx) : 0;
    const std::uint64_t floor = static_cast<std::uint64_t>(slot_ctx) + second;
    if (pool_now >= floor) return {};
    return std::format("kv pool {} tokens < {} (one {}-token request + a second one reaching {}): concurrent long requests wait in "
                       "the queue, or end early while generating (finish_reason length, finish_cause kv_pool_exhausted){}",
                       pool_now, floor, slot_ctx, second, int8_hint ? "; --balance allows int8 KV (a longer pool)" : "");
}

// (:1722-1726, runServer)
std::optional<std::string> targetNotReached(bool parallel_given, std::uint32_t ctx_eff, std::uint64_t target_ctx, std::uint32_t parallel) {
    if (!(parallel_given && ctx_eff < target_ctx)) return std::nullopt;
    return std::format("target not reached: context {} tokens per request with {} slot(s), less than {}", ctx_eff, parallel, target_ctx);
}

// (:1747-1750, runServer)
std::string startupNote(const std::vector<std::string>& adjusted) {
    std::string note;
    for (const std::string& a : adjusted)
        if (a.find("--ctx-per-slot") != std::string::npos || a.rfind("-c ", 0) == 0) note += (note.empty() ? "" : "; ") + a;
    return note.empty() ? std::string() : "adjusted at startup to fit the GPU memory: " + note;
}

PoolFitExplicit poolFitExplicit(bool uma, bool ctx_given, bool ctx_per_slot_given) {
    const bool fit_explicit = !uma;
    PoolFitExplicit r;
    r.pool = ctx_given && !fit_explicit;
    r.slot = ctx_per_slot_given && !fit_explicit;
    return r;
}

PoolAdjust poolAdjust(bool uma, std::optional<std::uint32_t> ctx, std::optional<std::uint32_t> ctx_per_slot, std::uint32_t slot_ctx,
                      std::uint32_t fit_pool, std::uint32_t fit_slot_ctx, bool fit_msg) {
    PoolAdjust r;
    const bool fit_explicit = !uma;
    const bool adj_c = fit_explicit && ctx && fit_pool < *ctx;
    const bool adj_s = fit_explicit && ctx_per_slot && fit_slot_ctx < slot_ctx;
    if (adj_c) r.ctx = std::format("-c {} -> {}", *ctx, fit_pool);
    if (adj_s) {
        r.slot = std::format("--ctx-per-slot {} -> {}", slot_ctx, fit_slot_ctx);
        r.slot_why = std::format("the KV pool holds {} tokens", fit_pool);
    }
    r.warn = fit_msg && !(adj_c || adj_s);
    return r;
}

TailCkLine tailCkLine(bool tail_ck, std::size_t n_ck, std::size_t tail_ck_slots, std::uint32_t tail_ck_min, double ck_mib) {
    TailCkLine r;
    if (!tail_ck) {
        r.warn = true;
        r.text = "prefix cache: tail checkpoint off (WHIRL_TAIL_CK=0, debug override): a second question about the same long "
                 "document re-prefills it";
    } else if (n_ck >= tail_ck_slots) {
        r.text = std::format("prefix cache: tail checkpoint on: prompts >= {} tokens keep one of the {} per-slot checkpoints ({:.1f} MiB) at the "
                             "start of their last chunk, so another question about the same document resumes there",
                             tail_ck_min, n_ck, ck_mib);
    } else {
        r.text = std::format("prefix cache: tail checkpoint off ({} checkpoint(s) per slot, needs {}; WHIRL_SERVE_CKPTS={} turns it on at "
                             "+{:.1f} MiB VRAM per slot each)",
                             n_ck, tail_ck_slots, tail_ck_slots, ck_mib);
    }
    return r;
}

std::uint32_t governorPoolMin(std::uint32_t pool_start) { return std::min<std::uint32_t>(pool_start, pool_min_tokens); }

std::optional<std::string> governorOffWhy(bool wddm_ok, bool uma, bool reserve_explicit) {
    if (!(wddm_ok && !uma)) return std::nullopt;
    return std::string(reserve_explicit ? "WHIRL_POOL_RESERVE_MB is set: the pool is never resized" : "WHIRL_GOVERNOR=0");
}

std::uint32_t targetCtx(std::uint32_t slot_ctx) { return std::min<std::uint32_t>(slot_ctx, vram_tight_pool); }

std::uint32_t fitPoolMin(std::optional<std::uint32_t> pool_req, std::uint32_t slot_ctx) { return pool_req ? *pool_req : targetCtx(slot_ctx); }

UmaTokens umaTokens(std::optional<std::uint32_t> ctx, std::optional<std::uint32_t> pool_req, std::uint32_t slot_ctx) {
    UmaTokens r;
    r.min_tokens = ctx ? *ctx : slot_ctx;
    r.want_tokens = pool_req ? *pool_req : slot_ctx;
    return r;
}

std::uint64_t ctxPerRequest(std::uint64_t slot_ctx, std::uint64_t pool_tokens) { return std::min(slot_ctx, pool_tokens); }

}  // namespace whirl::startfit
