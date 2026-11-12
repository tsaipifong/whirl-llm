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
    ISO_CHECK(!o.tail_ck_in_forward);  // v0252-n10: off unless whirl-server sets it from the device profile (gfx1201)
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
    // v0252-e1p1c (E1 lookahead): no cycle in flight, no forced synchronous cycle, no read yet
    ISO_CHECK(sp->e1_fly == nullptr && !sp->e1_sync && sp->e1_t_read == s::TimePoint{});
    // WHIRL_E1: 0 off, 1 the synchronous device-input cycle (P1b), 2 the lookahead (P1c); nothing else
    ISO_CHECK(s::parseE1("0") == std::optional<std::uint32_t>(0u) && s::parseE1("1") == std::optional<std::uint32_t>(1u));
    ISO_CHECK(s::parseE1("2") == std::optional<std::uint32_t>(2u) && s::parseE1(" 2	") == std::optional<std::uint32_t>(2u));
    ISO_CHECK(!s::parseE1("3").has_value() && !s::parseE1("").has_value() && !s::parseE1("02").has_value());
    ISO_CHECK(s::EngineOptions{}.e1 == 0u);
    // v0252-e1def (PLAN V1-93): what whirl-server starts with. WHIRL_E1 set: that value on every card and model (a bad
    // one: 0, flagged); not set: the profile default, and only where a cycle can use it (MTP drafts, exact acceptance,
    // no DFlash2, the model's E1 path) - else 0, so the default never turns the device accept on for nothing
    const s::E1Site ok{.mtp = true, .dflash = false, .relax = false, .fast = false, .model_ok = true};
    for (const std::uint32_t def : {0u, 1u, 2u}) {
        for (const s::E1Site site : {ok, s::E1Site{}, s::E1Site{.mtp = true, .relax = true, .model_ok = true}}) {
            for (const char* v : {"0", "1", "2"}) {
                const s::E1Choice c = s::chooseE1(std::optional<std::string>(v), def, site);
                ISO_CHECK(c.e1 == static_cast<std::uint32_t>(v[0] - '0') && !c.bad && !c.by_default);
            }
            const s::E1Choice bad = s::chooseE1(std::optional<std::string>("on"), def, site);
            ISO_CHECK(bad.e1 == 0u && bad.bad && !bad.by_default);
            const s::E1Choice empty = s::chooseE1(std::optional<std::string>(""), def, site);
            ISO_CHECK(empty.e1 == 0u && empty.bad && !empty.by_default);
        }
        const s::E1Choice d = s::chooseE1(std::nullopt, def, ok);
        ISO_CHECK(d.e1 == def && !d.bad && d.by_default);
        // not usable: no MTP drafts / DFlash2 drafts / relaxed acceptance / fast mode (also with its relaxed acceptance
        // turned off) / a model without the E1 path
        for (const s::E1Site site : {s::E1Site{.mtp = false, .dflash = false, .relax = false, .fast = false, .model_ok = true},
                                     s::E1Site{.mtp = true, .dflash = true, .relax = false, .fast = false, .model_ok = true},
                                     s::E1Site{.mtp = true, .dflash = false, .relax = true, .fast = false, .model_ok = true},
                                     s::E1Site{.mtp = true, .dflash = false, .relax = true, .fast = true, .model_ok = true},
                                     s::E1Site{.mtp = true, .dflash = false, .relax = false, .fast = true, .model_ok = true},
                                     s::E1Site{.mtp = true, .dflash = false, .relax = false, .fast = false, .model_ok = false}}) {
            const s::E1Choice n = s::chooseE1(std::nullopt, def, site);
            ISO_CHECK(n.e1 == 0u && !n.bad && n.by_default);
        }
        // v0252-b5 round 3 (E1 review item C2): an explicitly set WHIRL_SPEC_ACC wins over the DEFAULT E1. The engine
        // raises the accept mode for E1 (1: host / pinned -> gpu; 2: everything but check -> gpu-zc), so a default E1
        // would make WHIRL_SPEC_ACC=0 and =3 run the same thing; the default yields (e1 0, flagged `yielded`, no warning)
        // when the explicit mode is one that E1 value would raise. WHIRL_E1 set: that value, as before (the user's order)
        for (const std::uint32_t sa : {0u, 1u, 2u, 3u, 4u}) {
            s::E1Site site = ok;
            site.spec_acc_env = sa;
            const std::uint32_t want = def == 2u ? (sa >= 3u ? 2u : 0u) : def == 1u ? (sa >= 2u ? 1u : 0u) : 0u;
            const s::E1Choice y = s::chooseE1(std::nullopt, def, site);
            ISO_CHECK(y.e1 == want && !y.bad && y.by_default && y.yielded == (def != 0u && want == 0u));
            for (const char* v : {"0", "1", "2"}) {
                const s::E1Choice c = s::chooseE1(std::optional<std::string>(v), def, site);
                ISO_CHECK(c.e1 == static_cast<std::uint32_t>(v[0] - '0') && !c.bad && !c.by_default && !c.yielded);
            }
            s::E1Site off = site;  // a site the default cannot use anyway: 0, and nothing yielded
            off.model_ok = false;
            const s::E1Choice z = s::chooseE1(std::nullopt, def, off);
            ISO_CHECK(z.e1 == 0u && z.by_default && !z.yielded);
        }
        ISO_CHECK(!d.yielded && !s::E1Site{}.spec_acc_env.has_value());
    }

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
