// whirl-iso-server-engine-types-tests: the header-only module server.engine_types (src/server/engine_common.h,
// engine_options.h, engine_state.h; arch025-engine S3, moved verbatim out of engine.h and engine_internal.h) compiled
// with only its declared dependency headers and linked to nothing (ARCH-1 8.5 isolation). The checks pin what the move
// must keep: the aliases' values and types, KvPoolFull's type and text (engine.cpp:578 compares err == "KvPoolFull"),
// every EngineOptions default, and the zero / max_small_batch starts of BatchStats, SpecState and FloorState
// (a17493b engine.h:63-80, 85-201, engine_internal.h:19-21).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/engine_common.h"
#include "server/engine_options.h"
#include "server/engine_state.h"
// test only: CycleProf complete, as where Engine is destroyed (SpecState's implicit constructor names ~unique_ptr
// for its exception cleanup, which needs the complete type; engine_state.h itself compiles without it: S3 hdr pin)
#include "server/engine_prof.h"

#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

namespace s = whirl::server;

void testAliases() {
    static_assert(std::is_same_v<s::Cand, whirl::sampling::Cand>);
    static_assert(std::is_same_v<s::Sampler, whirl::sampling::Sampler>);
    static_assert(std::is_same_v<decltype(s::kv_page), const std::uint32_t>);
    static_assert(std::is_same_v<decltype(s::tail_ck_slots), const std::size_t>);
    ISO_CHECK(s::kv_page == whirl::qwen35::kv_page && s::kv_page == 256u);
    ISO_CHECK(s::gdn_max_seg == whirl::qwen35::gdn_max_seg && s::gdn_max_seg == 16u);
    ISO_CHECK(s::tail_ck_slots == whirl::cache::ckpt_keep_prefix_min && s::tail_ck_slots == 3u);
    ISO_CHECK(s::prefill_exec_default == whirl::sched::prefill_exec_default && s::prefill_exec_default == 2048u);
    ISO_CHECK(s::prefillChunkLen(100) == whirl::sched::prefillChunkLen(100));
}

void testKvPoolFull() {
    static_assert(std::is_base_of_v<std::runtime_error, s::KvPoolFull>);
    bool caught = false;
    try {
        throw s::KvPoolFull();
    } catch (const std::runtime_error& e) {
        caught = std::string(e.what()) == "KvPoolFull";
    }
    ISO_CHECK(caught);
}

void testOptionsDefaults() {
    s::EngineOptions o;
    ISO_CHECK(o.model_name.empty() && o.model_file.empty() && o.numerics_json.empty() && o.startup_json.empty());
    ISO_CHECK(o.ctx_note.empty());
    ISO_CHECK(o.ctx == 131072u);
    ISO_CHECK(o.kv_ssd_mode == "off");
    ISO_CHECK(o.kv_supersede);
    ISO_CHECK(o.kv_idle_flush_ms == whirl::tier::idle_flush_ms_default && o.kv_idle_flush_ms == 300000u);
    ISO_CHECK(o.parallel == 4u);
    ISO_CHECK(o.tmpl == whirl::chat::TemplateKind::a);
    ISO_CHECK(!o.use_mtp && !o.use_dflash && o.dfl_p_min == 0 && o.dfl_n_min == 0u && o.dfl_warm == 64u);
    ISO_CHECK(o.n_draft == 3u && o.p_min == 0 && o.n_min == 0u && o.mtp_adapt == 0u && !o.mtp_auto);
    ISO_CHECK(o.batch_drafts.size() == s::gdn_max_seg + 1);
    bool zero = true;
    for (std::uint32_t v : o.batch_drafts) zero = zero && v == 0;
    ISO_CHECK(zero);
    ISO_CHECK(o.ng_snap_sets == 0u && !o.no_prefix_cache && !o.verify_cache && !o.cold_resume);
    ISO_CHECK(o.seg_prefill && o.timing_reset && o.ngram);
    ISO_CHECK(o.ngram_min == 3u && o.ngram_slope == 0.015f && o.ngram_max == 0u && !o.ngram_force);
    ISO_CHECK(o.slot_drafts == 0u && !o.trace_nd && !o.loop_log && !o.tier_verify);
    ISO_CHECK(o.tier_min_gain == 512u && o.gather_ms == 30 && o.decode_min_tps == 0 && o.profile == 0);
    ISO_CHECK(o.sys_min == 2048u && o.lcp_on && o.n_ck == 2u);
    ISO_CHECK(o.tail_ck && o.tail_ck_min == whirl::sched::tail_ck_min_default && o.tail_ck_min == 4096u);
    ISO_CHECK(o.n_spe == 2u && !o.ckpt_host && o.prefill_exec == 2048u && o.seed == 0u && !o.spec_sample);
    ISO_CHECK(!o.relax.on && o.relax.k == 4u);
}

void testStateStarts() {
    s::BatchStats b;
    ISO_CHECK(b.t0 == s::TimePoint{});
    ISO_CHECK(b.tokens == 0 && b.cycles == 0 && b.rows == 0 && b.slots == 0 && b.drafts == 0 && b.lit_drafts == 0);
    ISO_CHECK(b.prefill_tok == 0 && b.seg_batches == 0 && b.seg_chunks == 0);
    ISO_CHECK(b.dfl_cycles == 0 && b.dfl_drafted == 0 && b.dfl_accepted == 0 && b.dfl_cold == 0 && b.dfl_gap == 0);

    const s::SpecState spo;
    const s::SpecState* sp = &spo;
    ISO_CHECK(sp->timing.size() == s::gdn_max_seg + 1 && sp->dtiming.size() == s::gdn_max_seg + 1);
    bool unprimed = true;
    for (bool p : sp->ccost_primed) unprimed = unprimed && !p;
    ISO_CHECK(unprimed);
    ISO_CHECK(sp->prof == nullptr);
    ISO_CHECK(sp->n_cycles == 0u && sp->prev_nd == 0u && sp->policy_cycles == 0u);
    ISO_CHECK(sp->vrows == s::max_small_batch && sp->vrows == 16u);

    s::FloorState f;
    ISO_CHECK(f.epoch == s::TimePoint{} && f.log_t == s::TimePoint{});
    ISO_CHECK(f.waits == 0 && f.periods == 0);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.engine_types",
                    {{"aliases", testAliases},
                     {"kv_pool_full", testKvPoolFull},
                     {"options_defaults", testOptionsDefaults},
                     {"state_starts", testStateStarts}},
                    argc, argv);
}
