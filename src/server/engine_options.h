// Server engine, the engine's options (arch025-engine S3, module server.engine_types; moved verbatim out of engine.h):
// struct EngineOptions, filled by server_main from the command line, the numerics plan and the startup fit.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "server/engine_common.h"     // gdn_max_seg, prefill_exec_default
#include "tier/idle_flush.h"          // tier::idle_flush_ms_default
#include "whirl/chat.h"               // chat::TemplateKind
#include "whirl/prefill_schedule.h"   // sched::tail_ck_min_default
#include "whirl/relax_accept.h"       // whirl::relax::Params

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace whirl::server {

struct EngineOptions {
    std::string model_name;
    std::string model_file;  // GGUF file name without directory (GET /props model_path)
    std::uint32_t ctx = 131072;  // context per slot (cap)
    std::string numerics_json;   // GET /props "numerics" (numerics::propsJson; empty: not reported)
    std::string startup_json;    // GET /props and /health "startup": effective values and startup adjustments (v0.2.2.1)
    std::string ctx_note;        // why the context per request is what it is when startup lowered it (logged on a too-long prompt)
    std::string kv_ssd_mode = "off";  // --kv-ssd-mode as in effect (GET /props kv_ssd)
    bool kv_supersede = true;         // CACHE-1 superseded-session cut (WHIRL_KV_SUPERSEDE=0: off)
    // TIER-IDLE: --kv-ssd-mode shutdown writes reused entries to SSD after this long idle (tier/idle_flush.h);
    // no CLI flag, WHIRL_KV_IDLE_FLUSH_MS overrides it for tests (0: off)
    std::uint64_t kv_idle_flush_ms = tier::idle_flush_ms_default;
    std::uint32_t parallel = 4;
    chat::TemplateKind tmpl = chat::TemplateKind::a;
    bool use_mtp = false;
    // --draft dflash: DFlash2 drafts instead of MTP (the backend has the drafter; use_mtp is false)
    bool use_dflash = false;
    float dfl_p_min = 0;
    std::uint32_t dfl_n_min = 0;
    // a slot drafts with DFlash2 once its drafter context has an unbroken run of at least
    // min(pos, dfl_warm) rows ending at pos (after a restore whose features are gone)
    std::uint32_t dfl_warm = 64;
    std::uint32_t n_draft = 3;
    float p_min = 0;
    std::uint32_t n_min = 0;
    std::uint32_t mtp_adapt = 0;
    bool mtp_auto = false;
    std::array<std::uint32_t, gdn_max_seg + 1> batch_drafts{};
    // without DeltaNet replay: snapshot sets the n-gram / DFlash2 drafts may use (0 = all; CONC-8S
    // keeps them at the budget they had before the MTP caps under load were raised)
    std::uint32_t ng_snap_sets = 0;
    bool no_prefix_cache = false;
    bool verify_cache = false;
    // COLD-RESUME (--precise): resume only from checkpoints whose state is bitwise the cold prefill's: on the
    // new prompt's chunk schedule and produced by prefill (never 'gen-end', whose history came from the decode /
    // verify kernels). Off (balance / fast): the longest checkpoint, as before. WHIRL_COLD_RESUME=0/1 overrides.
    bool cold_resume = false;
    bool seg_prefill = true;
    bool timing_reset = true;
    bool ngram = true;
    std::uint32_t ngram_min = 3;
    float ngram_slope = 0.015f;
    std::uint32_t ngram_max = 0;
    bool ngram_force = false;
    // per-slot draft counts: 0 uniform, 1 marginal-gain swaps (splitDrafts), 2 cost model (allocDrafts)
    std::uint32_t slot_drafts = 0;
    bool trace_nd = false;
    bool loop_log = false;
    bool tier_verify = false;
    std::uint32_t tier_min_gain = 512;
    double gather_ms = 30;
    // decode floor (--decode-min-tps): while slots decode, each keeps >= this many tok/s
    // (prefill forwards limited, decode cycles interleaved); 0 = off
    double decode_min_tps = 0;
    int profile = 0;  // 0 off, 1 events, 2 cycle stats only
    std::uint32_t sys_min = 2048;
    bool lcp_on = true;
    std::size_t n_ck = 2;
    // TAIL-CK: a prompt of >= tail_ck_min tokens keeps a slot checkpoint (kind 'prefix') at the start
    // of its last schedule chunk, so the same long document with another question resumes there on
    // the second request (not only the third). Needs n_ck >= tail_ck_slots, so the tail never
    // pushes out the prompt-end / gen-end checkpoints a regenerate or the next chat turn resumes from.
    // WHIRL_TAIL_CK=0 turns it off (debug).
    bool tail_ck = true;
    std::uint32_t tail_ck_min = sched::tail_ck_min_default;
    std::size_t n_spe = 2;
    bool ckpt_host = false;
    std::size_t prefill_exec = prefill_exec_default;
    std::uint64_t seed = 0;
    // numerics item specsample (balance / fast): temperature > 0 requests sample by a race keyed by
    // (seed, output index, token) and draw MTP drafts from the draft head's filtered distribution q
    // by the same race (whirl/spec_sample.h); off (precise): inverse-CDF sampling, greedy drafts.
    // Both accept a draft iff it equals the sampled token: the reply is a function of the seed only
    // (MTP == plain bit for bit within the mode).
    bool spec_sample = false;
    // numerics item relaxacc (fast): relaxed draft acceptance (whirl/relax_accept.h); greedy
    // requests keep near-argmax drafts, sampling requests use typical acceptance (and greedy
    // drafts: specsample is not used with it)
    whirl::relax::Params relax;
};

}  // namespace whirl::server
