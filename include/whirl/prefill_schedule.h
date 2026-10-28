// The server's cold prefill schedule as pure functions (FIX-TEST-B): the server engine and
// `whirl seqtest` use the same code, so the seqtest gate checks the schedule the server really runs.
//
// Schedule: chunks of prefill_chunk (1024) rows from position 0; a remainder of up to
// prefill_chunk + prefill_merge (1280) rows is one chunk (no short tail); forced splits (system
// boundary, think-open split, image ends) restart the chunking there. Checkpoints are kept, and
// reused, only at schedule points (chunk starts). Execution: consecutive whole chunks run as one
// forward of up to `exec` rows (+ a merged tail), never across a kept checkpoint.
//
// What holds (measured, docs/guide/en/kv-and-caching.md section 7): splitting a prefill into
// several forwards at schedule points gives the same logits bits as fewer, larger forwards over the
// same schedule; resuming from a checkpoint at a schedule point is such a split.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "whirl/device_profile.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace whirl::sched {

inline constexpr std::size_t prefill_chunk = 1024;
inline constexpr std::size_t prefill_merge = 256;

inline std::size_t prefillChunkLen(std::size_t remaining) {
    return remaining <= prefill_chunk + prefill_merge ? remaining : prefill_chunk;
}

// Default forward size of a solo prefill (WHIRL_PREFILL_CHUNK unset): 2048 rows; gfx1151 (Radeon 8060S)
// MoE models 4096 = the whole default prefill batch (SRV-PP): Ornith MoE 128k +3.6% over 2048 (CLI on the
// server's schedule, real 131k prompt: 983 -> 1018 tok/s; server 128k +4.2%). Dense models on gfx1151 keep
// 2048 (SRV-PP-DENSE, server, 2 rounds): 4096 cost Qwen3.8-27B 3.5% / 5% / 8% / 13% at 32k / 64k / 96k /
// 128k (128k 248.0 vs 284.6 tok/s), while MoE 64k / 96k gain 3.0% / 4.3%, so the 8060S default is per
// model. No extra memory either way (the prefill buffers are sized for the batch). Same bits (whole-chunk
// grouping, section 7 of docs/guide/en/kv-and-caching.md; seqtest gates it). gfx1201 keeps 2048 until measured there.
// (the 8060S MoE value is device::Profile::prefill_exec_moe)
inline constexpr std::size_t prefill_exec_default = 2048;
inline constexpr std::size_t prefillExecDefault(const device::Profile& p, bool moe) { return moe ? p.prefill_exec_moe : prefill_exec_default; }

// EngineOptions::prefill_exec for a prefill batch: whole schedule chunks, and a forward (+ the merged
// tail) must fit the batch; never below one chunk (the batch is >= prefill_chunk + prefill_merge,
// vram::min_prefill_batch). fill (gfx1151, SRV-PP): whole chunks up to the batch itself; the forward
// is then capped at the batch by execLen max_rows (a merged tail that does not fit after `exec` rows
// runs as the next forward). `requested`: WHIRL_PREFILL_CHUNK, default prefillExecDefault.
inline std::size_t prefillExecFor(std::size_t requested, std::uint32_t max_batch, bool fill = false) {
    const std::size_t mb = fill ? max_batch : max_batch > prefill_merge ? max_batch - prefill_merge : 0;
    return std::max(prefill_chunk, std::min(requested / prefill_chunk * prefill_chunk, mb / prefill_chunk * prefill_chunk));
}

// The forward size of whirl.exe's and whirl-server's prefill (arch025-N15c: was written out at both): the
// WHIRL_PREFILL_CHUNK request, else the device / model default, as whole chunks that fit max_batch.
inline std::size_t prefillExec(std::optional<std::size_t> requested, const device::Profile& p, bool moe, std::uint32_t max_batch) {
    return prefillExecFor(requested.value_or(prefillExecDefault(p, moe)), max_batch, p.prefill_fill_batch);
}

// next forced split after pos: the system boundary, then the think-open split, then the first image
// end (imgs sorted), else n (0 = none)
inline std::size_t schedNext(std::size_t n, std::size_t sys, std::size_t think, std::span<const std::uint32_t> imgs, std::size_t pos) {
    std::size_t nx = sys > pos ? sys : think > pos ? think : n;
    for (std::uint32_t b : imgs)
        if (b > pos) {
            nx = std::min<std::size_t>(nx, b);
            break;
        }
    return nx;
}

// the last schedule point <= lim (0 if none)
inline std::size_t schedFloor(std::size_t n, std::size_t sys, std::size_t think, std::span<const std::uint32_t> imgs, std::size_t lim) {
    std::size_t pos = 0, last = 0;
    while (pos < n && pos <= lim) {
        last = pos;
        pos += prefillChunkLen(schedNext(n, sys, think, imgs, pos) - pos);
    }
    if (pos <= lim) last = pos;
    return last;
}

inline bool onSchedule(std::size_t n, std::size_t sys, std::size_t think, std::span<const std::uint32_t> imgs, std::size_t a) {
    return a == 0 || schedFloor(n, sys, think, imgs, a) == a;
}

// TAIL-CK: the tail checkpoint position, the start of the prompt's last schedule chunk
inline std::size_t tailPoint(std::size_t n, std::size_t sys, std::size_t think, std::span<const std::uint32_t> imgs) {
    return n == 0 ? 0 : schedFloor(n, sys, think, imgs, n - 1);
}

// rows of the forward that starts at cur: whole chunks up to exec (+ merged tail), at most max_rows
// rows (the prefill batch; 0 = exec + prefill_merge), not past the next forced split, never across a
// stop (a kept checkpoint); exec <= prefill_chunk = one chunk per forward
inline std::size_t execLen(std::size_t n, std::size_t sys, std::size_t think, std::span<const std::uint32_t> imgs, std::size_t cur,
                           std::size_t exec, std::span<const std::size_t> stops, std::size_t max_rows = 0) {
    const std::size_t lim = schedNext(n, sys, think, imgs, cur);
    std::size_t len = prefillChunkLen(lim - cur);
    const std::size_t cap = max_rows > 0 ? std::min(max_rows, exec + prefill_merge) : exec + prefill_merge;
    if (exec > prefill_chunk)
        while (cur + len < lim) {
            if (std::find(stops.begin(), stops.end(), cur + len) != stops.end()) break;
            const std::size_t nx = prefillChunkLen(lim - cur - len);
            if (len + nx > cap) break;
            len += nx;
        }
    // a kept checkpoint is a forward end (stops are schedule points, so this only bites if one is not:
    // then the forward is split there, which the seqtest gate detects)
    for (std::size_t st : stops)
        if (st > cur && st < cur + len) len = st - cur;
    return len;
}

// the forward ends of a prefill of n tokens that starts (or resumes) at `from` (a schedule point)
inline std::vector<std::size_t> forwardEnds(std::size_t n, std::size_t sys, std::size_t think, std::span<const std::uint32_t> imgs,
                                            std::size_t from, std::size_t exec, std::span<const std::size_t> stops, std::size_t max_rows = 0) {
    std::vector<std::size_t> ends;
    for (std::size_t cur = from; cur < n;) {
        cur += execLen(n, sys, think, imgs, cur, exec, stops, max_rows);
        ends.push_back(cur);
    }
    return ends;
}

// ---------------------------------------------------------------------------
// The forced splits as functions of the tokens (shared by the server engine and the CLI, CLI-SCHED)

// server defaults: a leading system message of >= sys_min_default tokens is a split (WHIRL_SYS_MIN,
// 0 = off); image ends at >= img_split_min_default are splits (WHIRL_VIS_CKPT_MIN, 0 = off); a prompt
// of >= tail_ck_min_default tokens keeps the TAIL-CK checkpoint (a forward end, WHIRL_TAIL_CK=0 = off)
inline constexpr std::uint32_t sys_min_default = 2048;
inline constexpr std::uint32_t img_split_min_default = 1024;
inline constexpr std::uint32_t tail_ck_min_default = 4096;

// Prompt starting with "<|im_start|>" "system": the position after the first "<|im_end|>" "\n"
// that is followed by "<|im_start|>" (a "<|im_end|>" the system text itself mentions is not the
// end). 0 = none. `none` ids never match.
inline std::size_t sysBoundary(std::span<const std::uint32_t> toks, std::uint32_t im_start, std::uint32_t system, std::uint32_t im_end,
                               std::uint32_t nl, std::uint32_t none = 0xffffffffu) {
    if (toks.size() < 3 || im_start == none || im_end == none) return 0;
    if (toks[0] != im_start || toks[1] != system) return 0;
    for (std::size_t i = 2; i + 2 < toks.size(); ++i) {
        if (toks[i] == im_end && toks[i + 1] == nl && toks[i + 2] == im_start) return i + 2;
    }
    return 0;
}

// the system-message split the schedule uses: sysBoundary when it is >= sys_min (> 0) and inside the prompt
inline std::size_t sysSplit(std::size_t boundary, std::size_t n, std::uint32_t sys_min) {
    return (sys_min == 0 || boundary < sys_min || boundary >= n) ? 0 : boundary;
}

// vision: the image-end splits (position after the <|vision_end|> that follows a run of image ids,
// ids >= image_id_base), those >= ck_min (> 0) and inside the prompt; at most 32. A function of the
// tokens only, so every prompt with the same prefix gets the same splits there.
inline std::vector<std::uint32_t> imageSplits(std::span<const std::uint32_t> toks, std::uint32_t image_id_base, std::uint32_t vision_end_id,
                                              std::uint32_t ck_min) {
    std::vector<std::uint32_t> out;
    if (ck_min == 0) return out;
    std::size_t i = 0;
    while (i < toks.size() && out.size() < 32) {
        if (toks[i] < image_id_base) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < toks.size() && toks[j] >= image_id_base) ++j;
        const std::size_t b = (j < toks.size() && toks[j] == vision_end_id) ? j + 1 : j;
        if (b >= ck_min && b < toks.size()) out.push_back(static_cast<std::uint32_t>(b));
        i = j + 1;
    }
    return out;
}

// CLI-SCHED: the forward ends of a prompt prefilled alone on a server with default options, from
// `from` (0 = cold): forwardEnds with the TAIL-CK stop (tail_min > 0, n >= tail_min, the tail point is
// not already a forced split). `whirl chat` / bench / golden run exactly these forwards.
inline std::vector<std::size_t> soloForwardEnds(std::size_t n, std::size_t sys, std::size_t think, std::span<const std::uint32_t> imgs,
                                                std::size_t from, std::size_t exec, std::size_t tail_min, std::size_t max_rows = 0) {
    std::vector<std::size_t> stops;
    if (tail_min > 0 && n >= tail_min) {
        const std::size_t t = tailPoint(n, sys, think, imgs);
        if (t > from && t != sys && t != think && std::find(imgs.begin(), imgs.end(), static_cast<std::uint32_t>(t)) == imgs.end())
            stops.push_back(t);
    }
    return forwardEnds(n, sys, think, imgs, from, exec, stops, max_rows);
}

}  // namespace whirl::sched
