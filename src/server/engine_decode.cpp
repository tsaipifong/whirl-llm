// Server engine, the decode cycle (arch025-N4f, from engine.cpp): draft selection (MTP, n-gram, DFlash2), the batched
// verify, sampling and acceptance; decodeCycle's phases (arch025-E2) in call order. The DFlash2 drafter context moved
// to engine_drafter.h (arch025-engine S4, module server.drafter_ctx).
// SPDX-License-Identifier: Apache-2.0

// arch025-regb1 (N4-reg b1): the component's own header, not engine.h / engine_internal.h (no include edge back to server.engine)
#include "server/engine_cycle.h"
#include "server/engine_output.h"
#include "server/engine_prof.h"

#include "cache/tier_adapter.h"
#include "log.h"
#include "whirl/dflash_policy.h"
#include "spec/spec_rules.h"  // the speculative-decoding rules shared with the CLI (arch025-R12cli)

#include "../../kernels/spec_accept.h"  // v0252-e1p1 (feat026-acc port): the device accept decision (WHIRL_SPEC_ACC)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <limits>
#include <future>
#include <stdexcept>
#include <thread>

namespace whirl::server {

// relaxacc repetition guard: would keeping draft dr[acc] (after sl.next and dr[0..acc)) end in
// an n-gram already in the slot's recent tokens?
// (the rule is shared with the CLI: spec::rules::relaxRepeats, src/spec/spec_rules.h)
bool SpecCycle::relaxRepeats(const Slot& sl, std::span<const std::uint32_t> dr, std::uint32_t acc) const {
    return spec::rules::relaxRepeats(std::span<const std::uint32_t>(sl.cache_tokens), sl.next, dr, acc);
}

std::uint32_t SpecCycle::pickDrafts(std::span<Slot* const> act) {
    if (!opt_.use_mtp) return 0;
    const std::uint32_t A = static_cast<std::uint32_t>(act.size());
    std::uint32_t nd = draftCap(A);
    if (nd == 0) return 0;
    if (opt_.mtp_auto) {
        std::vector<const qwen35::DraftAccept*> accs;
        std::vector<qwen35::DraftAccept> eff;
        eff.reserve(act.size());
        for (Slot* sl : act) {
            if (opt_.slot_drafts == 2 && act.size() >= 2) {  // one slot: exactly as before
                eff.push_back(sl->sacc.effective(spec_.dpool));
                accs.push_back(&eff.back());
            } else {
                accs.push_back(&sl->dacc);
            }
        }
        const std::uint32_t pick =
            qwen35::pickDrafts(accs, spec_.timing[std::min(A, gdn_max_seg)], nd, opt_.relax.on ? spec_.policy_cycles : spec_.n_cycles, spec_.prev_nd);
        spec_.prev_nd = pick;
        return pick;
    }
    if (opt_.mtp_adapt > 0) nd = spec::rules::adaptDrafts(act, [](const Slot* sl) { return sl->acc_ema; }, opt_.mtp_adapt, nd);
    return nd;
}

// arch025-E2: decodeCycle's phases. Cycle holds the values one decode cycle passes from one phase to the next
// (they were decodeCycle's locals; value-initialized per cycle, never kept across cycles).
struct SpecCycle::Cycle {
    std::uint64_t E = 0;
    TimePoint t_cycle0{};
    std::vector<Slot*> act;  // the slots with context room
    std::size_t A = 0;
    std::uint32_t Au = 0;
    // n-gram drafts per slot (history = cache_tokens + next) when they promise
    // more tokens per ms than MTP / DFlash2 drafts. Literal drafts (n-gram, DFlash2) go to the
    // verify from the host: ng_n / ng_d, with dsrc telling which drafter made them.
    std::array<std::uint32_t, gdn_max_seg> ng_n{}, ng_ml{};
    std::array<std::array<std::uint32_t, qwen35::max_ng_drafts>, gdn_max_seg> ng_d{};
    std::array<std::uint8_t, gdn_max_seg> dsrc{};  // 0 MTP / none, 1 n-gram, 2 DFlash2
    std::size_t n_ng = 0;
    std::uint32_t snap_cap = 0, lit_cap = 0;
    std::uint32_t dfl_lim = 0;  // DFlash2: drafts per slot within the shared verify rows / snapshot sets
    TimePoint td0{};
    std::size_t n_dfl = 0;
    std::uint32_t dfl_max_n = 0;
    std::vector<Slot*> act_mtp;  // the MTP slots
    std::size_t n_mtp = 0;
    std::uint32_t nd = 0;
    std::array<std::uint32_t, gdn_max_seg> nd_m{};  // per-slot counts (MTP slots, in act_mtp order)
    std::uint32_t nd_max = 0;  // MTP steps = the largest
    TimePoint tc0{};
    std::size_t pa = 0;
    CycleProf* p = nullptr;
    std::array<MSeg, gdn_max_seg> msegs{};
    std::array<VSeg, gdn_max_seg> vsegs{};
    std::array<std::uint32_t, gdn_max_seg> row_of{}, snap_of{}, nd_of{};
    std::array<std::size_t, qwen35::max_drafts + 1> n_step{};
    std::uint32_t rows = 0;  // the verify's rows
    std::uint32_t rk = 0;
    bool rank1p = false;  // v0252-f1: the verify's argmax kernel ranked the greedy relaxacc drafts (no topk_rows)
    // v0252-e1p1 (feat026-acc port, WHIRL_SPEC_ACC): this cycle's device accept arguments (sampleRows; acc_dev = a device
    // mode, acc_check = mode 4, acc_all_dev = every slot greedy exact), the control words the slots read
    // (SampleBuffers::readCycleCtl) and check mode's next-cycle row comparison (acceptSlot, slot by slot)
    bool acc_dev = false, acc_check = false, acc_all_dev = false;
    SpecAccArgs aa{};
    const std::int32_t* ctl_all = nullptr;
    SpecNextTab tab{};
    std::uint32_t tab_n = 0;
    bool tab_ok = true;
    // v0252-e1p1b (E1, WHIRL_E1; layoutBatch): this cycle is an E1 cycle - its slot takes the device inputs the last cycle
    // left (Slot::e1_rows > 0) and its accept leaves the next cycle's on the device (acceptSlot)
    bool e1 = false;
    // v0252-e1p1c (E1 lookahead, aheadCycle / aheadBuild): the cycle went through the lookahead ring - its accept wrote
    // ring entry ring_e and its inputs for the next cycle were left at submit time (SampleBuffers::aheadSubmit); succ =
    // the next cycle was submitted before this one was read (it consumed this cycle's ACC / CHOSEN / NPOS / NPEND words);
    // ring_ms = the cycle's time for the draft-count policy (< 0: a synchronous cycle, its wall time)
    bool ring = false, succ = false;
    std::uint32_t ring_e = 0;
    float ring_ms = -1.0f;
};

// relaxacc (fast) applies to MTP drafts only: one rule for every site below (FR-DFL P2-3)
bool SpecCycle::relaxes(const Cycle& cy, std::size_t k) const {
    return whirl::dflash::relaxAppliesTo(static_cast<whirl::dflash::DraftKind>(cy.dsrc[k]), opt_.relax.on);
}

// the verify writes KV rows pos .. pos + drafts: stay inside the pages mapped
// for the slot and the context
std::uint32_t SpecCycle::verifyRoom(const Slot& sl, std::uint32_t lim) const {
    const std::uint64_t mapped = static_cast<std::uint64_t>(sl.pages.size()) * kv_page;
    const std::uint64_t a1 = mapped > static_cast<std::uint64_t>(sl.pos) + 1 ? mapped - (sl.pos + 1ull) : 0;
    const std::uint64_t a2 = spec::rules::ctxRoom(sl.pos, ctx_);
    return static_cast<std::uint32_t>(std::min({static_cast<std::uint64_t>(lim), a1, a2}));
}

void SpecCycle::decodeCycle(std::span<Slot* const> act_in) {
    // v0252-e1p1c (E1 lookahead, WHIRL_E1=2): a cycle is in flight, or the one decoding slot starts a pipeline
    if (opt_.e1 >= 2 && aheadCycle(act_in)) return;
    Cycle cy;
    const auto& cfg = m_.cfg();
    cy.E = cfg.n_embd;
    cy.t_cycle0 = Clock::now();
    dropOutOfRoom(cy, act_in);
    cy.A = cy.act.size();
    if (cy.A == 0) return;
    draftLimits(cy);
    ngramDrafts(cy);
    dflashDrafts(cy);
    mtpDrafts(cy);
    layoutBatch(cy);
    draftSampling(cy);
    enqueueVerify(cy);
    sampleRows(cy);
    trainCycle(cy);
    cycleStats(cy);
    const sampling::Acceptor acceptor(opt_.relax);
    for (std::size_t k = 0; k < cy.A; ++k) acceptSlot(cy, k, acceptor);
    finishCycle(cy);
}

void SpecCycle::dropOutOfRoom(Cycle& cy, std::span<Slot* const> act_in) {
    // slots out of context room end here
    const std::uint32_t nd_room = dc_.draftRoom();
    for (Slot* sl : act_in) {
        const bool room_ok = spec::rules::decodeRoom(sl->pos, nd_room, ctx_) && sc_.ensurePages(*sl, sl->pos + nd_room + 2);
        if (!room_ok) {
            if (spec::rules::decodeRoom(sl->pos, nd_room, ctx_))
            {
                logW("req {} | slot {} | kv pool exhausted at {} tokens (every page belongs to a running request); "
                     "finish_reason length, finish_cause kv_pool_exhausted",
                     sl->reqId(), sl->id, sl->pos);
                sl->pool_cut = true;
            }
            sl->finish = "length";
            sl->logits_ok = false;  // other work ran since its last cycle
            finishJob(*sl);
            continue;
        }
        cy.act.push_back(sl);
    }
}

void SpecCycle::draftLimits(Cycle& cy) {
    // replay keeps no snapshot sets: the verify row budget is the only shared limit
    cy.snap_cap = m_.gdnReplay() ? spec_.vrows : std::min(qwen35::gdn_max_snap, m_.snapSets());
    // literal drafts (n-gram, DFlash2) per slot: the snapshot budget they had before CONC-8S
    cy.lit_cap = (m_.gdnReplay() || opt_.ng_snap_sets == 0) ? cy.snap_cap : std::min(cy.snap_cap, opt_.ng_snap_sets);
    cy.Au = static_cast<std::uint32_t>(cy.A);
    if (opt_.use_dflash) {
        cy.dfl_lim = ::whirl::dflash::draftLimit(m_.dflashMaxDrafts() + 1, qwen35::max_ng_drafts, cy.lit_cap, spec_.vrows, cy.Au, m_.fusedDecode());
    }
}

void SpecCycle::ngramDrafts(Cycle& cy) {
    const qwen35::DraftTiming& dtm = spec_.dtiming[std::min<std::size_t>(cy.A, gdn_max_seg)];
    if (opt_.ngram && (opt_.use_mtp || opt_.use_dflash)) {
        const std::uint32_t lim = spec::rules::ngramDraftCap(cy.lit_cap, spec_.vrows, cy.Au, opt_.ngram_max, m_.fusedDecode());
        const qwen35::DraftTiming& tm = opt_.use_dflash ? dtm : spec_.timing[std::min<std::size_t>(cy.A, gdn_max_seg)];
        const std::uint32_t nd0 = spec::rules::ngramAltDrafts(spec_.prev_nd, draftCap(cy.Au));
        for (std::size_t k = 0; k < cy.A; ++k) {
            if (lim == 0) break;
            Slot& sl = *cy.act[k];
            sl.cache_tokens.push_back(sl.next);
            struct Pop {
                std::vector<std::uint32_t>& v;
                ~Pop() { v.pop_back(); }
            } pop{sl.cache_tokens};
            const std::uint32_t lim_k = verifyRoom(sl, lim);
            if (lim_k == 0) continue;
            const auto mt = sl.ng.propose(sl.cache_tokens, opt_.ngram_min, std::span<std::uint32_t>(cy.ng_d[k].data(), lim_k));
            if (mt.n == 0) continue;
            // the alternative's tokens per ms: MTP with nd0 drafts, or DFlash2 with its usual draft
            // count (a slot whose drafter context is not ready verifies one row)
            float alt_score = 0;
            if (opt_.use_dflash) {
                const std::uint32_t dn = dc_.dflReady(sl) ? verifyRoom(sl, std::min(cy.dfl_lim, static_cast<std::uint32_t>(std::lround(sl.dfl_nd)))) : 0;
                if (auto t = dtm.estimate(dn)) alt_score = sl.dfl_acc.expected(dn) / *t;
            } else if (opt_.slot_drafts == 2 && cy.A >= 2) {
                // against this slot's own last MTP allocation and its effective acceptance (one slot: as before)
                const std::uint32_t d = sl.d_prev > 0 ? sl.d_prev : nd0;
                if (auto t = tm.estimate(d)) alt_score = sl.sacc.effective(spec_.dpool).expected(d) / *t;
            } else {
                alt_score = spec::rules::mtpScore(sl.dacc, tm, nd0);
            }
            cy.ng_ml[k] = mt.mlen;
            cy.ng_n[k] = opt_.ngram_force ? static_cast<std::uint32_t>(mt.n) : sl.ng.choose(mt.n, mt.mlen, alt_score, tm);
            if (cy.ng_n[k] > 0) {
                cy.n_ng += 1;
                cy.dsrc[k] = 1;
            }
        }
    }
}

void SpecCycle::dflashDrafts(Cycle& cy) {
    // DFlash2 drafts for the slots without n-gram drafts (their time counts in the DFlash2 cycle time)
    cy.td0 = Clock::now();
    if (opt_.use_dflash && cy.dfl_lim > 0) {
        for (std::size_t k = 0; k < cy.A; ++k) {
            if (cy.dsrc[k] != 0) continue;
            Slot& sl = *cy.act[k];
            if (!dc_.dflReady(sl)) {
                bst_.dfl_cold += 1;
                continue;
            }
            const std::uint32_t lim_k = verifyRoom(sl, cy.dfl_lim);
            if (lim_k == 0) continue;
            const std::vector<std::uint32_t> d = m_.dflashDraft(sl.id, sl.next, sl.pos, opt_.dfl_p_min, opt_.dfl_n_min);
            sl.dfl_nd = 0.8f * sl.dfl_nd + 0.2f * static_cast<float>(d.size());
            const std::uint32_t n = std::min(lim_k, static_cast<std::uint32_t>(d.size()));
            std::copy(d.begin(), d.begin() + n, cy.ng_d[k].begin());
            cy.ng_n[k] = n;
            cy.dsrc[k] = 2;
            cy.n_dfl += 1;
            cy.dfl_max_n = std::max(cy.dfl_max_n, n);
        }
    }
}

void SpecCycle::mtpDrafts(Cycle& cy) {
    // MTP drafts for the other slots
    for (std::size_t k = 0; k < cy.A; ++k)
        if (cy.ng_n[k] == 0) cy.act_mtp.push_back(cy.act[k]);
    cy.n_mtp = cy.act_mtp.size();
    cy.nd = cy.n_mtp > 0 ? pickDrafts(cy.act_mtp) : 0;
    {
        // keep the verify within spec_.vrows rows (and the snapshot sets without replay)
        std::uint32_t ng_tot = 0;
        for (std::size_t k = 0; k < cy.A; ++k) ng_tot += cy.ng_n[k];
        while (cy.nd > 0 && (static_cast<std::uint32_t>(cy.A) + ng_tot + static_cast<std::uint32_t>(cy.n_mtp) * cy.nd > spec_.vrows ||
                          ng_tot + static_cast<std::uint32_t>(cy.n_mtp) * cy.nd > cy.snap_cap))
            cy.nd -= 1;
    }
    cy.nd_max = cy.nd;
    if (cy.n_mtp > 0) {
        std::uint32_t ng_tot = 0;
        for (std::size_t k = 0; k < cy.A; ++k) ng_tot += cy.ng_n[k];
        const std::uint32_t used = static_cast<std::uint32_t>(cy.A) + ng_tot;
        const std::uint32_t rows_free = spec_.vrows > used ? spec_.vrows - used : 0;
        const std::uint32_t snap_free = cy.snap_cap > ng_tot ? cy.snap_cap - ng_tot : 0;
        if (opt_.slot_drafts == 2 && cy.nd > 0) {
            const std::size_t ia = std::min<std::size_t>(cy.A, gdn_max_seg);
            qwen35::CycleCost& cc = spec_.ccost[ia];
            if (!spec_.ccost_primed[ia] && cc.samples() == 0) {
                float ctx_k = 0;
                for (Slot* s : cy.act) ctx_k += static_cast<float>(s->pos) / 1024.0f;
                cc.prior(spec_.timing[ia], static_cast<std::uint32_t>(cy.A), ctx_k / static_cast<float>(cy.A));
                spec_.ccost_primed[ia] = true;
            }
            std::vector<qwen35::DraftAccept> eff;
            eff.reserve(cy.A);
            std::vector<DraftPlanSlot> ps(cy.A);
            for (std::size_t k = 0; k < cy.A; ++k) {
                Slot& s = *cy.act[k];
                ps[k].pos = s.pos;
                ps[k].d_prev = s.d_prev;
                if (cy.ng_n[k] > 0) {
                    ps[k].ng_rows = cy.ng_n[k] + 1;
                    ps[k].ng_e = s.ng.acc[qwen35::NgramPolicy::bucket(cy.ng_ml[k])].expected(cy.ng_n[k]);
                } else {
                    eff.push_back(s.sacc.effective(spec_.dpool));
                    ps[k].acc = &eff.back();
                }
            }
            const std::uint32_t cap = !m_.fusedDecode() ? 1u : std::min({opt_.n_draft, qwen35::max_drafts, max_small_batch - 1});
            const std::vector<std::uint32_t> d = planSlotDrafts(ps, cc, cy.nd, cap, spec_.vrows, snap_free);
            for (std::size_t i = 0; i < cy.n_mtp; ++i) cy.nd_m[i] = d[i];
        } else {
            splitDrafts(cy.act_mtp, cy.nd, std::min(rows_free, snap_free), std::span<std::uint32_t>(cy.nd_m.data(), cy.n_mtp));
        }
        for (std::size_t i = 0; i < cy.n_mtp; ++i) cy.act_mtp[i]->d_prev = cy.nd_m[i];
        cy.nd_max = 0;
        for (std::size_t i = 0; i < cy.n_mtp; ++i) cy.nd_max = std::max(cy.nd_max, cy.nd_m[i]);
    }
}

void SpecCycle::layoutBatch(Cycle& cy) {
    cy.tc0 = Clock::now();
    cy.pa = std::min<std::size_t>(cy.A, gdn_max_seg);
    cy.p = spec_.prof.get();
    if (cy.p && cy.p->events) m_.profileStart(0, cy.pa);
    // v0252-e1p1b (E1, PLAN V1-93 P1b; WHIRL_E1): one greedy exact MTP slot (no relaxacc, no n-gram / DFlash2 drafts, no
    // image), DeltaNet replay mode, no profiling events. Engine::allocState kept opt_.e1 only with the model's E1 path and a
    // device accept mode. Any other cycle: host inputs as before (a slot's device inputs are then not used, and its accept
    // takes them back to the host: acceptSlot)
    cy.e1 = opt_.e1 != 0 && opt_.use_mtp && cy.A == 1 && cy.n_mtp == 1 && !opt_.use_dflash && m_.gdnReplay() && !(cy.p && cy.p->events) &&
            cy.act[0]->greedy && !relaxes(cy, 0) && (cy.act[0]->job == nullptr || cy.act[0]->job->vis == nullptr);
    // MTP steps r >= 1 run on the MTP slots with more than r drafts: MTP slots
    // sorted by draft count (descending) at the front of the batch
    {
        std::array<std::size_t, gdn_max_seg> ord{};
        for (std::size_t i = 0; i < cy.n_mtp; ++i) ord[i] = i;
        for (std::size_t i = 1; i < cy.n_mtp; ++i) {
            std::size_t j = i;
            while (j > 0 && cy.nd_m[ord[j - 1]] < cy.nd_m[ord[j]]) {
                std::swap(ord[j - 1], ord[j]);
                --j;
            }
        }
        for (std::size_t q = 0; q < cy.n_mtp; ++q) {
            Slot& s = *cy.act_mtp[ord[q]];
            // v0252-e1p1b (E1): the device inputs - step 0 runs the last verify's nd + 1 rows from its argmax words (no pend)
            if (cy.e1 && s.e1_rows > 0) cy.msegs[q] = MSeg{s.id, {}, s.pend_pos, s.pos, s.e1_rows};
            else cy.msegs[q] = MSeg{s.id, std::span<const std::uint32_t>(s.pend.data(), s.n_pend), s.pend_pos, s.pos};
        }
        for (std::size_t r = 0; r <= qwen35::max_drafts; ++r) {
            std::size_t c = 0;
            for (std::size_t i = 0; i < cy.n_mtp; ++i)
                if (cy.nd_m[i] > r) ++c;
            cy.n_step[r] = c;
        }
    }
    {
        // n-gram slots after the MTP slots in the MTP batch (their pend rows only)
        std::size_t ig = cy.n_mtp, im = 0;
        std::uint32_t row = 0, snap = 0;
        for (std::size_t k = 0; k < cy.A; ++k) {
            Slot& sl = *cy.act[k];
            const MSeg ms{sl.id, std::span<const std::uint32_t>(sl.pend.data(), sl.n_pend), sl.pend_pos, sl.pos};
            if (cy.ng_n[k] == 0) {
                cy.nd_of[k] = cy.nd_m[im];
                ++im;
            } else {
                cy.msegs[ig] = ms;
                ++ig;
                cy.nd_of[k] = cy.ng_n[k];
            }
            cy.vsegs[k] = VSeg{sl.id, sl.next, cy.nd_of[k], sl.pos,
                            cy.ng_n[k] > 0 ? std::span<const std::uint32_t>(cy.ng_d[k].data(), cy.ng_n[k])
                                        : std::span<const std::uint32_t>()};
            // v0252-e1p1b (E1): row 0 and the positions from the slot's control words (sl.next / sl.pos stay the host's copy:
            // the model reads neither, pos bounds its context check)
            cy.vsegs[k].dev = cy.e1 && sl.e1_rows > 0;
            cy.row_of[k] = row;
            cy.snap_of[k] = snap;
            row += cy.nd_of[k] + 1;
            snap += cy.nd_of[k];
        }
    }
}

void SpecCycle::draftSampling(Cycle& cy) {
    if (spec_on_ && cy.nd_max > 0) {
        // specsample: MTP slots of sampling requests draw their drafts from q by the target's race
        for (std::size_t k = 0; k < cy.A; ++k) {
            const Slot& sl = *cy.act[k];
            spec::DraftSample ds;
            ds.on = !sl.greedy && cy.ng_n[k] == 0;
            if (ds.on) {
                ds.top_k = sl.sm.top_k;
                ds.inv_t = sl.sm.inv_t;
                ds.top_p = static_cast<float>(sl.sm.top_p);
                ds.min_p = static_cast<float>(sl.sm.min_p);
                ds.seed = sl.sm.seed;
                ds.pos0 = sl.n_gen;
            }
            m_.setDraftSample(sl.id, ds);
        }
    }
}

void SpecCycle::enqueueVerify(Cycle& cy) {
    // v0252-btl: the MTP step r = 0 takes every slot's pend rows: the rows its LAST verify kept (acc + 1, within that
    // cycle's per-slot caps for the slots of that cycle), 1 for a slot that joined after its prefill since. After a join
    // they can be more than one batched MTP step takes (spec_.vrows = the model's verifyRows(): Context::mtpBatchStepEx threw
    // BatchTooLarge and every request of the batch failed with HTTP 500). The segments then run in consecutive groups of at
    // most spec_.vrows pend rows, each group its own draft chain r = 0 .. (a step r >= 1 reads the head rows the step just
    // before it wrote, so one group's chain ends before the next group starts). One group (the usual case): the calls are
    // exactly the ones of one batch. Like llama.cpp, which decodes a batch over n_batch in n_batch slices
    // (server-context.cpp:3231-3256; progress/v0252-btl.PRIOR_ART.md).
    if (cy.nd_max > 0 || opt_.use_mtp) {
        // no MTP drafts this cycle (draft false): still feed the pend rows to the MTP block so
        // its KV cache stays complete for later drafting cycles
        const bool draft = cy.nd_max > 0;
        if (draft) m_.setDraftCutoff(opt_.p_min, std::min(opt_.n_min, cy.nd_max));
        struct Reset {
            ServerModel& m;
            bool on;
            ~Reset() {
                if (on) m.setDraftCutoff(0, 0);
            }
        } reset{m_, draft};
        const std::uint32_t vrows = spec_.vrows;
        std::size_t groups = 0, rows_all = 0;
        // step 0's rows of a segment (v0252-e1p1b: an E1 segment's padded rows)
        const auto rows0 = [](const MSeg& ms) { return ms.e1_rows > 0 ? static_cast<std::size_t>(ms.e1_rows) : ms.pend.size(); };
        for (std::size_t b = 0, e = 0; b < cy.A; b = e) {
            std::size_t rows = 0;
            while (e < cy.A && (e == b || rows + rows0(cy.msegs[e]) <= vrows)) rows += rows0(cy.msegs[e++]);
            m_.mtpBatchStepEx(std::span<const MSeg>(cy.msegs.data() + b, e - b), 0, draft);
            // steps r >= 1: the group's MTP slots with more than r drafts (sorted by draft count, at the front of msegs)
            for (std::uint32_t r = 1; draft && r < cy.nd_max && b < cy.n_step[r]; ++r)
                m_.mtpBatchStepEx(std::span<const MSeg>(cy.msegs.data() + b, std::min(e, cy.n_step[r]) - b), r, true);
            groups += 1;
            rows_all += rows;
        }
        if (groups > 1)
            logI("batch | MTP step r = 0 split into {} groups: {} pend rows of {} decoding slots, at most {} per step (a slot joined "
                 "since the last verify)",
                 groups, rows_all, cy.A, vrows);
    }
    // relaxacc one-pass check (v0252-f1, DISCOVERY F1): greedy relaxacc slots with MTP drafts get their drafts' full-row
    // ranks from the verify's argmax kernel (sampleRows then launches no topk_rows for them); a model without the
    // kernel keeps topk_rows
    std::uint32_t rmask = 0;
    for (std::size_t k = 0; k < cy.A && k < 32; ++k)
        if (cy.act[k]->greedy && relaxes(cy, k) && cy.nd_of[k] > 0) rmask |= 1u << k;
    cy.rank1p = rmask != 0 && m_.setRelaxRank(sb_.relaxDev(), rmask, max_rows);
    if (cy.p && cy.p->events) m_.profileStart(1, cy.pa);
    cy.rows = m_.verifyBatchEnqueue(std::span<const VSeg>(cy.vsegs.data(), cy.A));
    if (cy.p) m_.profileStop();
}

void SpecCycle::sampleRows(Cycle& cy) {
    bool any_sampling = false, any_relax = false;
    cy.rk = opt_.relax.topk();
    for (std::size_t k = 0; k < cy.A; ++k) {
        Slot& sl = *cy.act[k];
        if (sl.greedy) {
            if (relaxes(cy, k) && cy.nd_of[k] > 0) {
                any_relax = true;
                if (!cy.rank1p) sb_.launchTopk(cy.row_of[k], cy.nd_of[k], cy.rk, 1.0f, sb_.relaxDev(), max_rows, cy.row_of[k]);
            }
            continue;
        }
        any_sampling = true;
        sb_.launchTopk(cy.row_of[k], cy.nd_of[k] + 1, k_small, sl.sm.inv_t, sb_.sampDev(), max_rows, cy.row_of[k]);
    }
    // v0252-e1p1 (feat026-acc port): the device accept decision of the greedy exact slots (sampling / relaxacc slots are
    // only copied: the host decides them in every mode)
    cy.acc_dev = opt_.spec_acc >= 2;
    cy.acc_check = opt_.spec_acc == 4;
    if (cy.acc_dev) {
        SpecAccArgs& aa = cy.aa;
        aa.n = static_cast<int>(cy.A);
        aa.stop0 = ct_.im_end;
        aa.stop1 = ct_.endoftext;
        cy.acc_all_dev = true;
        for (std::size_t k = 0; k < cy.A; ++k) {
            const Slot& sl = *cy.act[k];
            aa.seq[k] = static_cast<int>(sl.id);
            aa.nd[k] = static_cast<int>(cy.nd_of[k]);
            aa.lit[k] = cy.ng_n[k] > 0 ? 1 : 0;
            aa.dev[k] = sl.greedy && !relaxes(cy, k) ? 1 : 0;
            aa.eos[k] = sl.ignore_eos ? 0 : 1;
            const std::uint32_t budget = sl.max_tokens > sl.n_gen ? sl.max_tokens - sl.n_gen : 0;
            aa.budget[k] = static_cast<int>(std::min<std::uint32_t>(budget, 1u << 30));
            // v0252-e1p1b (E1): a slot on device inputs verifies at the NPOS word its last accept wrote (-1)
            aa.pos[k] = cy.e1 && sl.e1_rows > 0 ? -1 : static_cast<int>(sl.pos);
            aa.row0[k] = static_cast<int>(cy.row_of[k]);
            for (std::uint32_t r = 0; r < cy.ng_n[k] && r < static_cast<std::uint32_t>(WHIRL_SA_MAX_ND); ++r)
                aa.lit_d[k][r] = static_cast<int>(cy.ng_d[k][r]);
            cy.acc_all_dev = cy.acc_all_dev && aa.dev[k] != 0;
        }
    }
    const TimePoint t_enq = Clock::now();
    // the cycle's one read of the control words (feat026-acc's readCycleCtl seam; WHIRL_SPEC_ACC=host: readCtl as before)
    cy.ctl_all = sb_.readCycleCtl(opt_.spec_acc, slots_.size() * qwen35::ctl_words, cy.acc_dev ? &cy.aa : nullptr);
    if (cy.p) {
        cy.p->t_ready = Clock::now();
        cy.p->enq_ms[cy.pa] += std::chrono::duration<double, std::milli>(t_enq - cy.tc0).count();
    }
    if (any_sampling) ops_.download(sb_.sampHost().data(), sb_.sampDev(), SampleBuffers::sampBytes());
    if (any_relax)
        ops_.download(sb_.relaxHost().data(), sb_.relaxDev(), cy.rank1p ? static_cast<std::size_t>(max_rows) * 12 : sb_.relaxHost().size());
}

void SpecCycle::trainCycle(Cycle& cy) {
    // relaxacc: the kept tokens depend on which drafts were proposed, so the draft-count policy
    // must not see wall-clock times (run-to-run jitter would change the output). It sees a
    // deterministic cycle cost instead: a base per slot, + ngram_slope per extra verify row (the
    // per-device verify-row slope, 0.12 on the 8060S), + 3% per sequential MTP draft step.
    const auto synthMs = [&](std::uint32_t steps) {
        float rows = 0;
        for (std::size_t k = 0; k < cy.A; ++k) rows += static_cast<float>(cy.nd_of[k] + 1);
        const float a = static_cast<float>(cy.A);
        return spec::rules::synthCycleMs(a, rows, opt_.ngram_slope, steps);
    };
    // (v0252-e1p1c: a lookahead cycle overlaps its neighbours - its time runs from the read of the cycle before it)
    const float cycle_ms = opt_.relax.on ? synthMs(cy.n_ng == cy.A ? 0 : cy.nd_max)
                                         : (cy.ring_ms >= 0.0f ? cy.ring_ms : static_cast<float>(msSince(cy.tc0)));
    if (opt_.slot_drafts == 2 && opt_.use_mtp) {
        // every cycle (uniform, per-slot, with n-gram slots) trains the cost model of A slots;
        // a per-slot cycle reaches spec_.timing as the time the uniform split would have taken
        qwen35::CycleCost& cc = spec_.ccost[std::min<std::size_t>(cy.A, gdn_max_seg)];
        qwen35::CycleFeat x, xu;
        bool uni = cy.n_ng == 0;
        for (std::size_t k = 0; k < cy.A; ++k) {
            const float ctx_k = static_cast<float>(cy.act[k]->pos) / 1024.0f;
            x.rows += static_cast<float>(cy.nd_of[k] + 1);
            x.row_ctx += static_cast<float>(cy.nd_of[k] + 1) * ctx_k;
            xu.rows += static_cast<float>(cy.nd + 1);
            xu.row_ctx += static_cast<float>(cy.nd + 1) * ctx_k;
            uni = uni && cy.nd_of[k] == cy.nd;
        }
        x.steps = static_cast<float>(cy.nd_max);
        xu.steps = static_cast<float>(cy.nd);
        if (cy.nd > 0 && cy.n_ng == 0) {
            if (uni) spec_.timing[std::min<std::size_t>(cy.A, gdn_max_seg)].update(cy.nd, cycle_ms);
            else if (cc.samples() >= 16)
                spec_.timing[std::min<std::size_t>(cy.A, gdn_max_seg)].update(cy.nd, cycle_ms * cc.predict(xu) / cc.predict(x));
        }
        cc.observe(x, cycle_ms);
    } else if (cy.nd > 0 && cy.n_ng == 0) {
        // swaps keep the rows of the common count (an extra step at most)
        std::uint32_t tot = 0;
        for (std::size_t i = 0; i < cy.n_mtp; ++i) tot += cy.nd_m[i];
        if (tot == cy.nd * static_cast<std::uint32_t>(cy.n_mtp))
            spec_.timing[std::min<std::size_t>(cy.A, gdn_max_seg)].update(cy.nd, cycle_ms);
    }
    if (cy.n_ng == cy.A) {
        // n-gram cycle times (every slot drafted from its history): per slot
        std::uint32_t ng_max_n = 0;
        for (std::size_t k = 0; k < cy.A; ++k) ng_max_n = std::max(ng_max_n, cy.ng_n[k]);
        const float ms = opt_.relax.on ? cycle_ms : static_cast<float>(msSince(cy.tc0));
        for (Slot* sl : cy.act) sl->ng.timing.update(ng_max_n, ms);
    }
    if (opt_.use_dflash && cy.n_ng == 0) {
        // DFlash2 cycle times (drafting included; 0 drafts: a plain verify)
        spec_.dtiming[std::min<std::size_t>(cy.A, gdn_max_seg)].update(cy.dfl_max_n, static_cast<float>(msSince(cy.td0)));
    }
}

void SpecCycle::cycleStats(Cycle& cy) {
    if (cy.p) {
        if (cy.p->events) m_.profileFlush(cy.pa);
        cy.p->cycles[cy.pa] += 1;
        cy.p->rows[cy.pa] += cy.rows;
        cy.p->drafts[cy.pa] += cy.rows - cy.A;
        cy.p->wall_ms[cy.pa] += msSince(cy.t_cycle0);
        cy.p->host_ms[cy.pa] += msSince(cy.t_cycle0) - msSince(cy.tc0);
        if (cy.p->last_end) cy.p->gap_ms[cy.pa] += std::chrono::duration<double, std::milli>(cy.t_cycle0 - *cy.p->last_end).count();
    }
    if (opt_.trace_nd) logI("trace | cycle {} | slots {} | nd {} | n-gram {} | dflash {} | {:.2f} ms", spec_.n_cycles, cy.A, cy.nd, cy.n_ng, cy.n_dfl, msSince(cy.tc0));
    spec_.n_cycles += 1;
    spec_.policy_cycles += 1;
    bst_.cycles += 1;
    bst_.rows += cy.rows;
    bst_.slots += cy.A;
    bst_.drafts += cy.nd;
    for (std::size_t k = 0; k < cy.A; ++k) bst_.lit_drafts += cy.ng_n[k];
}

void SpecCycle::acceptSlot(Cycle& cy, std::size_t k, const sampling::Acceptor& acceptor) {
    Slot& sl = *cy.act[k];
    const std::int32_t* ctl = cy.ctl_all + static_cast<std::size_t>(sl.id) * qwen35::ctl_words;
    // v0252-e1p1 (feat026-acc port): decided on the device (the host path runs only for WHIRL_SPEC_ACC=check, to compare)
    const bool dev_k = cy.acc_dev && cy.aa.dev[k] != 0;
    bool dev_use = dev_k && !cy.acc_check;
    // v0252-e1p1b (E1): the device result is one the host agrees with (gpu / gpu-zc: in range; check: equal) - only then
    // may the next cycle take its inputs from the control words
    bool dev_good = dev_k;
    if (cy.vsegs[k].dev) sb_.accStats().e1 += 1;  // this cycle took the slot's inputs from the device
    const std::uint32_t row0 = cy.row_of[k];
    const std::uint32_t nd_k = cy.nd_of[k];
    const bool is_ng = cy.ng_n[k] > 0;  // literal drafts (n-gram or DFlash2)
    // MTP: drafts (and the device p-min cutoff) from the control area; n-gram: literal
    const std::uint32_t nd_dev =
        is_ng ? nd_k : (nd_k > 0 ? std::min(nd_k, static_cast<std::uint32_t>(std::max(0, ctl[qwen35::ctl_nd]))) : 0);
    std::array<std::uint32_t, qwen35::max_ng_drafts> dr{};
    for (std::uint32_t r = 0; r < nd_k; ++r)
        dr[r] = is_ng ? cy.ng_d[k][r] : static_cast<std::uint32_t>(ctl[qwen35::ctl_drafts + r]);
    if (cy.dsrc[k] == 1) sl.ng_cycles += 1;
    sl.cycles += 1;
    sl.drafted += nd_dev;
    if (dev_use) {
        // a device result outside the cycle's bounds (never expected: a broken copy) falls back to the host path
        const std::int32_t a0 = ctl[WHIRL_SA_CTL_ACC0], a1 = ctl[WHIRL_SA_CTL_ACC];
        if (a0 < 0 || static_cast<std::uint32_t>(a0) > nd_dev || a1 < 0 || a1 > a0 ||
            static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_NDDEV]) != nd_dev) {
            if (sb_.accStats().fallback == 0)
                logW("spec-acc fallback | slot {} | device result out of range (acc {}/{}, nd {}); host decides", sl.id, a0, a1, nd_dev);
            sb_.accStats().fallback += 1;
            dev_use = false;
            dev_good = false;
        }
    }
    // draft acceptance (src/sampling/acceptor.h): greedy / relaxacc greedy, or the sampler's draw at every
    // output index (inverse CDF / specsample race / relaxacc typical); v0252-e1p1: or the device's decision
    const sampling::Drafts drafts{std::span<const std::uint32_t>(dr.data(), dr.size()), nd_dev, row0};
    const sampling::RepeatGuard repeats = [&](std::uint32_t a) { return relaxRepeats(sl, dr, a); };
    sampling::Accepted ac;
    if (dev_use) {
        ac.acc = static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_ACC0]);
        ac.chosen = static_cast<std::uint32_t>(ctl[qwen35::ctl_rows + ac.acc]);
        sb_.accStats().dev += 1;
    } else {
        ac = sl.greedy ? acceptor.greedy(ctl + qwen35::ctl_rows, drafts, relaxes(cy, k), sampling::RelaxRows{sb_.relaxHost().data(), max_rows, cy.rk, cy.rank1p},
                                         repeats, sl.relaxed, sl.relax_gap)
                       : acceptor.sample(sb_, sl.sm, drafts, relaxes(cy, k), sl.n_gen, repeats, sl.relaxed);
    }
    std::uint32_t acc = ac.acc;
    std::uint32_t chosen = ac.chosen;
    sl.accepted += acc;
    if (cy.dsrc[k] == 0 && opt_.use_mtp) {
        // n-gram cycles stay out of the MTP acceptance models
        sl.acc_ema = spec::rules::accEma(sl.acc_ema, acc);
        sl.dacc.update(nd_dev, acc);
        if (opt_.slot_drafts == 2 && nd_dev > 0) {
            sl.sacc.observe(nd_dev, acc);
            spec_.dpool.update(nd_dev, acc);
        }
    } else if (cy.dsrc[k] == 2) {
        sl.dfl_acc.update(nd_dev, acc);
        sl.dfl_cycles += 1;
        bst_.dfl_cycles += 1;
        bst_.dfl_drafted += nd_dev;
        bst_.dfl_accepted += acc;
    }
    const std::uint32_t acc_pre = acc;
    // an accepted end-of-turn draft ends the reply: the trunk keeps only the
    // rows before it and the stop token becomes `chosen` (v0252-e1p1: on the device, the kernel made both cuts)
    if (dev_use) {
        acc = static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_ACC]);
        chosen = static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_CHOSEN]);
    } else if (!sl.ignore_eos) {
        for (std::uint32_t r = 0; r < acc; ++r) {
            if (dr[r] == ct_.im_end || dr[r] == ct_.endoftext) {
                chosen = dr[r];
                acc = r;
                break;
            }
        }
    }
    // same for max_tokens: the last emitted token is not fed to the trunk
    if (!dev_use) {
        const std::uint32_t budget = sl.max_tokens > sl.n_gen ? sl.max_tokens - sl.n_gen : 0;
        if (budget >= 1 && acc + 1 > budget) {
            chosen = dr[budget - 1];
            acc = budget - 1;
        }
    }
    // v0252-e1p1 (WHIRL_SPEC_ACC=check): the device decided too; every word it wrote must equal the host path's
    if (cy.acc_check && dev_k) {
        SpecAccStats& as = sb_.accStats();
        as.checked += 1;
        const bool same = static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_ACC0]) == acc_pre && static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_ACC]) == acc &&
                          static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_CHOSEN]) == chosen &&
                          static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_NDDEV]) == nd_dev &&
                          static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_NPOS]) == sl.pos + acc + 1 &&
                          static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_NROW]) == row0 + acc &&
                          static_cast<std::uint32_t>(ctl[WHIRL_SA_CTL_NPEND]) == acc + 1;  // v0252-e1p1b
        if (!same) {
            if (as.mismatch == 0)
                logW("spec-acc check | slot {} | device acc {}/{} chosen {} nd {} vs host acc {}/{} chosen {} nd {}", sl.id, ctl[WHIRL_SA_CTL_ACC0],
                     ctl[WHIRL_SA_CTL_ACC], ctl[WHIRL_SA_CTL_CHOSEN], ctl[WHIRL_SA_CTL_NDDEV], acc_pre, acc, chosen, nd_dev);
            as.mismatch += 1;
            dev_good = false;
        }
    }
    // v0252-e1p1b (E1): an E1 cycle with a good device result leaves the next cycle's inputs on the device - the replay
    // count is the NPEND word (qwen35::npend_dev), the hidden rows of all nd + 1 verify rows are kept below, e1_rows says so
    bool e1_next = cy.e1 && dev_good && m_.gdnReplay();
    // the verify ran nd + 1 rows of this slot: keep acc + 1 of them
    if (cy.ring) {
        // v0252-e1p1c (lookahead): the cycle's inputs for the next one were left when it was submitted (replay count = the
        // NPEND word, all nd + 1 hidden rows: aheadBuild). They stand when the host's decision is the device's words. If
        // not, nothing may have consumed them: with the next cycle already submitted the DeltaNet state was replayed with
        // the device's count and cannot be taken back (replay mode keeps no snapshot) - the request fails loudly
        const bool agree = spec::rules::aheadWordsAgree(ctl[WHIRL_SA_CTL_ACC], ctl[WHIRL_SA_CTL_CHOSEN], ctl[WHIRL_SA_CTL_NPOS],
                                                        ctl[WHIRL_SA_CTL_NPEND], sl.pos, acc, chosen);
        if (!agree && cy.succ) {
            logE("E1 lookahead | slot {} | the device accept (acc {}, next {}, pos {}, replay {}) differs from the host's (acc {}, next {}, "
                 "pos {}) and the next cycle was already submitted; the request fails",
                 sl.id, ctl[WHIRL_SA_CTL_ACC], ctl[WHIRL_SA_CTL_CHOSEN], ctl[WHIRL_SA_CTL_NPOS], ctl[WHIRL_SA_CTL_NPEND], acc, chosen,
                 sl.pos + acc + 1);
            throw std::runtime_error("E1Diverged");
        }
        e1_next = e1_next && agree;
        if (!e1_next) {
            spec_.e1_sync = true;                         // the slot's next unsubmitted cycle runs synchronously
            if (!cy.succ) m_.setPending(sl.id, acc + 1);  // nothing consumed the words: the host's count
        }
    } else if (e1_next) m_.setPending(sl.id, qwen35::npend_dev);
    else if (m_.gdnReplay()) m_.setPending(sl.id, acc + 1);
    else if (acc < nd_k) m_.restoreSeqSnapshot(sl.id, cy.snap_of[k] + acc);
    // DFlash2 context: the kept rows' features (whatever drafted them; rejected rows never enter)
    dc_.dflIngest(sl, cy.rows, row0, acc + 1, sl.pos);
    // trunk now holds next, dr[0..acc) at pos .. pos + acc
    sl.cache_tokens.push_back(sl.next);
    sl.cache_tokens.insert(sl.cache_tokens.end(), dr.begin(), dr.begin() + acc);
    const std::uint32_t n_before = sl.n_gen;
    for (std::uint32_t r = 0; r < acc; ++r)
        if (!sl.done) sl.done = emit(ct_, sl, dr[r]);
    if (!sl.done) sl.done = emit(ct_, sl, chosen);
    bst_.tokens += sl.n_gen - n_before;
    // (v0252-e1p1b, E1: all nd + 1 rows - the next MTP step 0 runs them padded; rows 0 .. acc are the same)
    // (v0252-e1p1c: a ring cycle copied all nd + 1 rows when it was submitted - by now the model's rows may be a later cycle's)
    if (opt_.use_mtp && !cy.ring)
        ops_.copyAsync(m_.seqHid(sl.id), m_.hn() + static_cast<std::uint64_t>(row0) * cy.E * 4, ((e1_next ? nd_k : acc) + 1) * cy.E * 4,
                       m_.stream());
    for (std::uint32_t r = 0; r < acc; ++r) sl.pend[r] = dr[r];
    sl.pend[acc] = chosen;
    sl.n_pend = acc + 1;
    sl.pend_pos = sl.pos + 1;
    sl.next = chosen;
    sl.pos += acc + 1;
    sl.last_acc = acc;
    sl.last_row = row0 + acc;
    sl.e1_rows = e1_next ? nd_k + 1 : 0;  // v0252-e1p1b
    sl.e1_hid = cy.ring ? sb_.aheadHid(cy.ring_e) : 0;  // v0252-e1p1c: the kept hidden rows of a ring cycle (finishJob)
    sl.logits_ok = true;
    // (v0252-e1p1c: not for a ring cycle - the device table may already be the next cycle's, and the read would block)
    if (cy.acc_check && cy.acc_all_dev && !cy.ring) {
        // v0252-e1p1 (WHIRL_SPEC_ACC=check, every slot decided on the device): this slot's rows of the device's next-cycle
        // table (read once, at the first slot) against the host's slot update just made; the table is judged at the last slot
        if (k == 0) {
            cy.tab = SpecNextTab{};
            m_.readSpecTab(cy.tab);
        }
        const std::uint32_t V = m_.cfg().n_vocab;
        const auto look = [&](std::uint32_t tok) { return static_cast<std::int32_t>(tok < V ? tok : 0); };
        for (std::uint32_t j = 0; j < sl.n_pend; ++j, ++cy.tab_n)
            cy.tab_ok = cy.tab_ok && cy.tab_n < static_cast<std::uint32_t>(WHIRL_SA_MAX_ROWS) && cy.tab.mtp_tok[cy.tab_n] == look(sl.pend[j]) &&
                        cy.tab.mtp_pos[cy.tab_n] == static_cast<std::int32_t>(sl.pend_pos + j);
        cy.tab_ok = cy.tab_ok && cy.tab.mtp_last[k] == static_cast<std::int32_t>(cy.tab_n) - 1 && cy.tab.ver_tok[k] == look(sl.next) &&
                    cy.tab.ver_pos[k] == static_cast<std::int32_t>(sl.pos);
        if (k + 1 == cy.A) {
            SpecAccStats& as = sb_.accStats();
            cy.tab_ok = cy.tab_ok && cy.tab.n_mtp == static_cast<std::int32_t>(cy.tab_n);
            as.tab_checked += 1;
            if (!cy.tab_ok) {
                if (as.tab_mismatch == 0) logW("spec-acc check | cycle {} | device next-cycle rows differ from the host's", spec_.n_cycles);
                as.tab_mismatch += 1;
            }
        }
    }
    flushOut(sl);
    if (sl.n_gen >= sl.last_log_n + 100 || msSince(sl.last_log_t) >= 5000) {
        const double el = msSince(sl.td0);
        if (opt_.use_mtp || opt_.use_dflash) {
            logI("req {} | slot {} | gen: n_gen {}, tg {:.2f} t/s, draft acceptance {:.1f}% ({:.2f} tok/cycle)",
                 sl.reqId(), sl.id, sl.n_gen, static_cast<double>(sl.n_gen) * 1000.0 / el,
                 sl.drafted > 0 ? 100.0 * sl.accepted / sl.drafted : 0.0,
                 static_cast<double>(sl.accepted + sl.cycles) / sl.cycles);
        } else {
            logI("req {} | slot {} | gen: n_gen {}, tg {:.2f} t/s", sl.reqId(), sl.id, sl.n_gen,
                 static_cast<double>(sl.n_gen) * 1000.0 / el);
        }
        sl.last_log_n = sl.n_gen;
        sl.last_log_t = Clock::now();
        // WDDM-LOG: a sharp drop against the request's early speed; the spill is named as the likely
        // cause only while one is seen (vram_watch_ state from the last status-line sample)
        if (const auto ev = sl.dec_watch.report(sl.n_gen, el, sl.cycles, vw_.vramSpilling(), prefill_tok_cum_)) {
            if (ev->warn) logW("req {} | slot {} | {}", sl.reqId(), sl.id, ev->text);
            else logI("req {} | slot {} | {}", sl.reqId(), sl.id, ev->text);
        }
    }
}

void SpecCycle::finishCycle(Cycle& cy) {
    if (cy.p) {
        std::uint64_t tk = 0;
        for (Slot* sl : cy.act) tk += sl->last_acc + 1;
        cy.p->tokens[cy.pa] += tk;
    }
    // finish after every slot of the cycle is processed (their logits rows are
    // still intact for the gen-end checkpoints)
    bool any_done = false;
    for (Slot* sl : cy.act) {
        if (sl->done || sl->st.gone) {
            any_done = true;
            finishJob(*sl);
        }
    }
    if (cy.p) {
        if (cy.p->t_ready) cy.p->post_ms[cy.pa] += msSince(*cy.p->t_ready);
        if (any_done) cy.p->last_end.reset();
        else cy.p->last_end = Clock::now();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// v0252-e1p1c (PLAN V1-93 P1c): the E1 one-cycle lookahead (WHIRL_E1=2). Design: progress/v0252-e1study.HANDOFF.md
// sections 3 / 4 / 8, progress/v0252-e1p1.PRIOR_ART.md "P1c" (TensorRT-LLM's overlap loop: launch step n + 1, then
// process step n). E1 rules: R-a no event / stream query; R-b one eventRecord per cycle; R-c no D2H copy command (the
// accept kernel writes the ring); R-d cycle n + 1 is submitted and recorded before the wait for cycle n; R-e no
// blocking read and no stream sync between the cycles of a pipeline.

std::shared_ptr<SpecCycle::Cycle> SpecCycle::aheadBuild(Slot& sl, const Cycle* fly) {
    namespace sr = spec::rules;
    // ---- the drain rules: any "no" = this cycle is not submitted through the ring
    if (!sb_.aheadOk() || !opt_.use_mtp || opt_.use_dflash || opt_.relax.on || !m_.gdnReplay()) return nullptr;
    if (sl.phase != Phase::decode || !sl.greedy || sl.done || sl.st.gone || (sl.job != nullptr && sl.job->vis != nullptr)) return nullptr;
    if (spec_.prof && spec_.prof->events) return nullptr;
    // another request is being admitted, prefilled, restored or decoded: its work goes between this slot's cycles
    for (const Slot& o : slots_)
        if (&o != &sl && o.phase != Phase::idle) return nullptr;
    // position: exact for the first cycle of a pipeline, an upper bound behind a cycle in flight (every draft kept); the
    // context and the mapped KV pages must hold a cycle at the bound
    const std::uint32_t fly_nd = fly != nullptr ? fly->nd_of[0] : 0;
    const std::uint32_t pos_b = fly != nullptr ? sr::aheadPosBound(sl.pos, fly_nd) : sl.pos;
    const std::uint32_t nd_room = dc_.draftRoom();
    if (!sr::decodeRoom(pos_b, nd_room, ctx_) || !sc_.ensurePages(sl, static_cast<std::uint64_t>(pos_b) + nd_room + 2)) return nullptr;
    // max_tokens: the budget drain rule, with the most drafts this cycle can get; the first cycle of a pipeline counts a
    // cycle of that size behind it (a pipeline of one cycle gains nothing)
    const std::uint32_t nd_cap = draftCap(1);
    const std::uint32_t fly_out = fly != nullptr ? fly_nd + 1 : 0;
    if (!sr::aheadBudgetOk(sl.max_tokens, sl.n_gen, fly != nullptr ? fly_out : nd_cap + 1, nd_cap)) return nullptr;
    auto cyp = std::make_shared<Cycle>();
    Cycle& cy = *cyp;
    cy.E = m_.cfg().n_embd;
    cy.t_cycle0 = Clock::now();
    cy.act.push_back(&sl);
    cy.A = 1;
    draftLimits(cy);
    if (fly != nullptr && opt_.ngram) {
        // the n-gram drafter's look at the host's history (the call the synchronous cycle would make with it; the first
        // cycle of a pipeline makes none): a proposal it would take ends the lookahead - the slot's next cycle runs
        // synchronously and drafts from the history then
        const std::uint32_t lim = sr::ngramDraftCap(cy.lit_cap, spec_.vrows, 1, opt_.ngram_max, m_.fusedDecode());
        if (lim > 0) {
            sl.cache_tokens.push_back(sl.next);
            struct Pop {
                std::vector<std::uint32_t>& v;
                ~Pop() { v.pop_back(); }
            } pop{sl.cache_tokens};
            const std::uint32_t lim_k = verifyRoom(sl, lim);
            if (lim_k > 0) {
                std::array<std::uint32_t, qwen35::max_ng_drafts> d{};
                const auto mt = sl.ng.propose(sl.cache_tokens, opt_.ngram_min, std::span<std::uint32_t>(d.data(), lim_k));
                if (mt.n > 0) {
                    const qwen35::DraftTiming& tm = spec_.timing[1];
                    const std::uint32_t nd0 = sr::ngramAltDrafts(spec_.prev_nd, nd_cap);
                    const std::uint32_t take =
                        opt_.ngram_force ? static_cast<std::uint32_t>(mt.n) : sl.ng.choose(mt.n, mt.mlen, sr::mtpScore(sl.dacc, tm, nd0), tm);
                    if (take > 0) {
                        spec_.e1_sync = true;
                        return nullptr;
                    }
                }
            }
        }
    }
    // ---- the cycle: MTP drafts only, the slot's inputs from the control words the cycle before it wrote
    mtpDrafts(cy);
    layoutBatch(cy);
    if (!cy.e1 || (fly == nullptr && sl.e1_rows == 0)) return nullptr;  // (layoutBatch's rule = the checks above)
    if (fly != nullptr) {
        // behind a cycle in flight the host knows neither the slot's next token nor its position: the MTP step 0 runs
        // that cycle's nd + 1 verify rows (padded) at the positions after its row 0, the verify takes row 0 and the
        // position from the control words; pos_b only bounds the model's context check
        cy.msegs[0] = MSeg{sl.id, {}, sl.pos + 1, pos_b, fly_nd + 1};
        cy.vsegs[0] = VSeg{sl.id, qwen35::tok_dev, cy.nd_of[0], pos_b, std::span<const std::uint32_t>()};
        cy.vsegs[0].dev = true;
    }
    draftSampling(cy);
    enqueueVerify(cy);
    // the accept decision of the one greedy exact sequence, on the device, into the ring entry
    cy.acc_dev = true;
    cy.acc_check = opt_.spec_acc == 4;
    cy.acc_all_dev = true;
    SpecAccArgs& aa = cy.aa;
    aa.n = 1;
    aa.stop0 = ct_.im_end;
    aa.stop1 = ct_.endoftext;
    aa.seq[0] = static_cast<int>(sl.id);
    aa.nd[0] = static_cast<int>(cy.nd_of[0]);
    aa.lit[0] = 0;
    aa.dev[0] = 1;
    aa.eos[0] = sl.ignore_eos ? 0 : 1;
    aa.budget[0] = static_cast<int>(sr::aheadBudgetLow(sl.max_tokens, sl.n_gen, fly_out));
    aa.pos[0] = -1;  // the NPOS word the sequence's last accept wrote
    aa.row0[0] = static_cast<int>(cy.row_of[0]);
    cy.ring = true;
    cy.ring_e = fly != nullptr ? sr::aheadRingNext(fly->ring_e) : 0;
    sb_.aheadSubmit(cy.ring_e, aa, sl.id, cy.row_of[0], cy.nd_of[0] + 1);
    // the next verify / commitSeq of the sequence replays by the NPEND word this cycle's accept writes
    m_.setPending(sl.id, qwen35::npend_dev);
    return cyp;
}

bool SpecCycle::aheadCycle(std::span<Slot* const> act_in) {
    std::shared_ptr<Cycle> fly = std::static_pointer_cast<Cycle>(spec_.e1_fly);
    try {
        if (fly == nullptr) {
            // the start of a pipeline: one decoding slot whose last (synchronous) cycle left its inputs on the device, and
            // no reason to run the next cycle synchronously
            if (spec_.e1_sync) {
                spec_.e1_sync = false;
                return false;
            }
            if (act_in.size() != 1 || act_in[0]->e1_rows == 0) return false;
            fly = aheadBuild(*act_in[0], nullptr);
            if (fly == nullptr) return false;
        }
        Slot& sl = *fly->act[0];
        if (std::find(act_in.begin(), act_in.end(), &sl) == act_in.end()) {
            // (never expected: the slot left the decode phase with its cycle in flight - nothing to read it for)
            spec_.e1_fly.reset();
            spec_.e1_sync = false;
            ops_.streamSync(m_.stream());
            return false;
        }
        // ---- R-d: the next cycle is submitted (and its event recorded) before the host waits for this one
        std::shared_ptr<Cycle> nxt;
        if (!spec_.e1_sync && act_in.size() == 1) nxt = aheadBuild(sl, fly.get());
        if (nxt != nullptr) {
            fly->succ = true;
            sb_.accStats().ahead += 1;
        }
        // ---- the cycle's one host wait: its own event; its control words are in its ring entry
        fly->ctl_all = sb_.aheadWait(fly->ring_e);
        const TimePoint t_read = Clock::now();
        if (fly->p) fly->p->t_ready = t_read;
        fly->ring_ms = std::chrono::duration<float, std::milli>(t_read - std::max(fly->tc0, spec_.e1_t_read)).count();
        spec_.e1_t_read = t_read;
        sb_.accStats().ring += 1;
        trainCycle(*fly);
        cycleStats(*fly);
        const sampling::Acceptor acceptor(opt_.relax);
        acceptSlot(*fly, 0, acceptor);
        if (nxt != nullptr && (sl.done || sl.st.gone)) {
            // the request ended at this cycle with the next one already submitted: that cycle is dropped. Its verify
            // already replayed this cycle's kept rows (the DeltaNet state is "through this cycle", what commitSeq leaves
            // without the lookahead); its own rows must never reach the state. Its KV rows lie past the kept prefix.
            ops_.streamSync(m_.stream());
            m_.dropPending(sl.id);
            sb_.accStats().discard += 1;
            nxt.reset();
        }
        spec_.e1_fly = nxt;
        finishCycle(*fly);
        return true;
    } catch (...) {
        // nothing stays in flight behind a failure (Engine::runLoop fails the cycle's requests)
        spec_.e1_fly.reset();
        spec_.e1_sync = false;
        try {
            ops_.streamSync(m_.stream());
        } catch (...) {
        }
        throw;
    }
}

// v0252-e1p1c-fix: the startup of the decode accept, verbatim from Engine::allocState (engine cut: a startup function of
// server.engine has at most 30 lines; it had 61 with P1a + P1b + P1c).
void SpecCycle::specAccSetup() {
    // v0252-e1p1b (E1, WHIRL_E1): the next cycle's inputs on the device need the model's E1 path (ServerModel::e1InputsOk:
    // the device-input kernels, the accept kernel, DeltaNet replay mode) and the device accept (host / pinned -> gpu)
    if (opt_.e1 != 0) {
        if (!m_.e1InputsOk()) {
            logW("WHIRL_E1={}: the model has no E1 path ({}); E1 off", opt_.e1,
                 m_.gdnReplay() ? "no set_rows_dp / rmsnorm_q8_rows_dv / spec_accept_rows kernel" : "DeltaNet snapshot mode");
            opt_.e1 = 0;
        } else if (opt_.e1 >= 2) {
            // v0252-e1p1c (the lookahead): the host reads each cycle from pinned memory the accept kernel wrote (gpu-zc);
            // check stays check (the host decides too and compares)
            if (opt_.spec_acc != 4) opt_.spec_acc = 3;
        } else if (opt_.spec_acc < 2) {
            opt_.spec_acc = 2;
        }
    }
    // v0252-e1p1 (feat026-acc port, WHIRL_SPEC_ACC >= 1): pinned control words; a device mode without the model's accept
    // kernel falls back to the pinned read + host accept (said once here)
    if (opt_.spec_acc != 0) {
        const std::uint32_t want = opt_.spec_acc;
        opt_.spec_acc = sb_.allocSpecAcc(opt_.parallel, want);
        if (want >= 2 && want <= 4 && opt_.spec_acc == 1)
            logW("WHIRL_SPEC_ACC={}: no spec_accept_rows kernel; pinned read + host accept", want);
        if (opt_.spec_acc != 0) logI("decode: WHIRL_SPEC_ACC={} ({})", opt_.spec_acc, specAccName(opt_.spec_acc));
    }
    if (opt_.e1 != 0) {
        if (opt_.spec_acc < 2) {
            opt_.e1 = 0;  // (no accept kernel after all: e1InputsOk asked for it)
        } else if (opt_.e1 >= 2 && sb_.allocAhead(opt_.parallel, max_small_batch)) {
            // v0252-e1p1c (PLAN V1-93 P1c)
            opt_.e1 = 2;
            logI("decode: WHIRL_E1=2 (one cycle ahead: the next draft cycle of a greedy request is submitted before the host reads the "
                 "last one; ring of 2 pinned control areas, {})",
                 specAccName(opt_.spec_acc));
        } else {
            opt_.e1 = 1;
            logI("decode: WHIRL_E1=1 (the next cycle's inputs from the device accept result; one cycle at a time)");
        }
    }
}

// v0252-e1p1c-fix: verbatim from Engine::stats (engine cut: a report function of server.engine has at most 30 lines)
void SpecCycle::specAccReport(EngineApi::Stats& s) const {
    // v0252-e1p1 (feat026-acc port)
    const SpecAccStats& as = sb_.accCounts();
    s.acc_dev = as.dev;
    s.acc_checked = as.checked;
    s.acc_mismatch = as.mismatch;
    s.acc_fallback = as.fallback;
    s.tab_checked = as.tab_checked;
    s.tab_mismatch = as.tab_mismatch;
    s.acc_e1 = as.e1;  // v0252-e1p1b
    s.acc_ring = as.ring;  // v0252-e1p1c
    s.acc_ahead = as.ahead;
    s.acc_discard = as.discard;
}

void SpecCycle::splitDrafts(std::span<Slot* const> act, std::uint32_t nd, std::uint32_t rows, std::span<std::uint32_t> nd_m) {
    for (auto& x : nd_m) x = nd;
    if (opt_.slot_drafts != 1 || !opt_.mtp_auto || act.size() < 2 || nd == 0) return;
    const std::uint32_t cap = std::min(opt_.n_draft, qwen35::max_drafts);
    std::array<const qwen35::DraftAccept*, gdn_max_seg> acc{};
    for (std::size_t k = 0; k < act.size(); ++k) acc[k] = &act[k]->dacc;
    splitDraftsByGain(std::span<const qwen35::DraftAccept* const>(acc.data(), act.size()), nd, rows, cap, nd_m);
}

std::uint32_t SpecCycle::draftCap(std::uint32_t A) {
    return slotDraftCap(opt_.batch_drafts[std::min(A, gdn_max_seg)], opt_.n_draft, m_.fusedDecode(), m_.gdnReplay(), A, spec_.vrows);
}

}  // namespace whirl::server
