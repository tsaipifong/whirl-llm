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
#include <optional>
#include <string>
#include <string_view>

namespace whirl::server {

// v0252-e1p1 (port of feat026-acc-r1 c797b27: parseSpecAcc / specAccName, engine.{h,cpp} there): WHIRL_SPEC_ACC values.
// 0..4 or the names host / pinned / gpu / gpu-zc / check (case-insensitive, surrounding blanks ignored; gpu_zc and zc =
// gpu-zc). Anything else: nullopt (server_main warns and uses 0 = host). specAccName: the name of a mode ("?" outside
// 0..4); one table for the parser, the startup log line and the per-request stats line.
inline constexpr std::array<const char*, 5> spec_acc_names{"host", "pinned", "gpu", "gpu-zc", "check"};

inline const char* specAccName(std::uint32_t mode) { return mode < spec_acc_names.size() ? spec_acc_names[mode] : "?"; }

inline std::optional<std::uint32_t> parseSpecAcc(std::string_view v) {
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.remove_prefix(1);
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.remove_suffix(1);
    if (v.size() == 1 && v[0] >= '0' && v[0] <= '4') return static_cast<std::uint32_t>(v[0] - '0');
    std::string s(v);
    for (char& c : s) c = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    if (s == "gpu_zc" || s == "zc") return 3u;
    for (std::uint32_t m = 0; m < spec_acc_names.size(); ++m)
        if (s == spec_acc_names[m]) return m;
    return std::nullopt;
}

// v0252-e1p1b (E1, PLAN V1-93 P1b): WHIRL_E1 values: "0" / "1" / "2" (v0252-e1p1c: 2 = the one-cycle lookahead;
// surrounding blanks ignored); anything else nullopt (server_main warns and uses 0).
inline std::optional<std::uint32_t> parseE1(std::string_view v) {
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.remove_prefix(1);
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.remove_suffix(1);
    if (v == "0" || v == "1" || v == "2") return static_cast<std::uint32_t>(v[0] - '0');
    return std::nullopt;
}

// v0252-e1def (PLAN V1-93): the WHIRL_E1 value whirl-server starts with. Set: that value, on every card, model and mode
// (bad = not 0 / 1 / 2: server_main warns and 0 runs, as before). Not set: `def` (qwen35::e1Default: the device profile's
// value for the model class) where a decode cycle can use it at all - MTP drafts with exact acceptance (no DFlash2
// drafts, no relaxed acceptance) and the model's E1 path (Backend::e1InputsOk), and not in fast mode (its relaxed
// acceptance never qualifies, and fast with that arm turned off was never run with the lookahead) - else 0, so a default
// never turns the device accept on (EngineOptions::spec_acc) for cycles that all run the usual way, and never warns.
struct E1Site {
    bool mtp = false;       // EngineOptions::use_mtp
    bool dflash = false;    // EngineOptions::use_dflash
    bool relax = false;     // EngineOptions::relax.on
    bool fast = false;      // the numerics mode is fast (numerics::Mode::fast)
    bool model_ok = false;  // Backend::e1InputsOk
};
struct E1Choice {
    std::uint32_t e1 = 0;
    bool bad = false;         // WHIRL_E1 is set and is not 0 / 1 / 2
    bool by_default = false;  // WHIRL_E1 is not set
};
inline E1Choice chooseE1(const std::optional<std::string>& env, std::uint32_t def, const E1Site& site) {
    if (env) {
        const std::optional<std::uint32_t> m = parseE1(*env);
        return {m.value_or(0u), !m.has_value(), false};
    }
    const bool usable = site.mtp && !site.dflash && !site.relax && !site.fast && site.model_ok;
    return {usable ? def : 0u, false, true};
}

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
    // v0252-e1p1 (feat026-acc port, WHIRL_SPEC_ACC): how a decode cycle reads the verify's control words and where the
    // greedy accept decision is made. 0 host (pageable readCtl, host path: as before), 1 pinned (async copy into pinned
    // memory + stream sync), 2 gpu (spec_accept_rows decides greedy exact slots; pinned read), 3 gpu-zc (the kernel writes
    // the control words into pinned memory: no copy), 4 check (gpu, and the host path runs too and must agree: accept
    // counts, next token, next-cycle rows; the host result is used). The reply is the same in every mode (exact greedy
    // acceptance; sampling / relaxacc slots always decide on the host). SampleBuffers::readCycleCtl.
    std::uint32_t spec_acc = 0;
    // v0252-e1p1b (E1, PLAN V1-93 P1b; WHIRL_E1): 1 = a decode cycle that qualifies (one greedy exact MTP slot, DeltaNet
    // replay mode, no image / DFlash2 / profiling events) takes its inputs from the control words the last cycle's accept
    // kernel wrote (next token, positions, MTP step-0 rows, draft-head row, replay count), and leaves the next cycle's
    // there. Still one cycle at a time (P1c pipelines it); needs the device accept (Engine::allocState raises spec_acc to
    // 2 = gpu when it is host / pinned) and the model's E1 kernels (ServerModel::e1InputsOk; else off with a warning).
    // The reply is the same as with 0.
    // v0252-e1p1c (PLAN V1-93 P1c): 2 = the one-cycle lookahead on top of 1 - the next cycle of that request is submitted
    // before the host has read the last one (SpecCycle::aheadCycle; the host reads each cycle from a ring of two pinned
    // control-word areas the accept kernel writes, so Engine::allocState raises spec_acc to 3 = gpu-zc unless it is 4 =
    // check). A cycle that does not qualify, the last cycles before max_tokens and every cycle while another request is
    // active run as with 1. The reply is the same as with 0.
    // v0252-e1def: whirl-server sets it with chooseE1 (WHIRL_E1, else the device profile's default for the model class:
    // 2 for MoE models on the R9700, 0 elsewhere); the engine's own default stays 0.
    std::uint32_t e1 = 0;
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
    // v0252-n10: the TAIL-CK checkpoint written inside the solo prefill forward that contains the tail point
    // (PrefillPhase::soloLen, sched::tailInside, ServerModel::setStateCapture) instead of ending a forward there; the
    // same bits (progress/v0252-n10.DESIGN.md). whirl-server sets it from device::Profile::tail_ck_in_forward (gfx1201).
    bool tail_ck_in_forward = false;
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
