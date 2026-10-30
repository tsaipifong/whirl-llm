// Server engine, the decode cycle (arch025-N4f, from engine.cpp): draft selection (MTP, n-gram, DFlash2), the batched
// verify, sampling and acceptance; decodeCycle's phases (arch025-E2) in call order. The DFlash2 drafter context moved
// to engine_drafter.h (arch025-engine S4, module server.drafter_ctx).
// SPDX-License-Identifier: Apache-2.0

#include "engine.h"
#include "engine_internal.h"

#include "cache/tier_adapter.h"
#include "log.h"
#include "whirl/dflash_policy.h"
#include "spec/spec_rules.h"  // the speculative-decoding rules shared with the CLI (arch025-R12cli)

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
bool Engine::relaxRepeats(const Slot& sl, std::span<const std::uint32_t> dr, std::uint32_t acc) const {
    return spec::rules::relaxRepeats(std::span<const std::uint32_t>(sl.cache_tokens), sl.next, dr, acc);
}

std::uint32_t Engine::pickDrafts(std::span<Slot* const> act) {
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
struct Engine::Cycle {
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
};

// relaxacc (fast) applies to MTP drafts only: one rule for every site below (FR-DFL P2-3)
bool Engine::relaxes(const Cycle& cy, std::size_t k) const {
    return whirl::dflash::relaxAppliesTo(static_cast<whirl::dflash::DraftKind>(cy.dsrc[k]), opt_.relax.on);
}

// the verify writes KV rows pos .. pos + drafts: stay inside the pages mapped
// for the slot and the context
std::uint32_t Engine::verifyRoom(const Slot& sl, std::uint32_t lim) const {
    const std::uint64_t mapped = static_cast<std::uint64_t>(sl.pages.size()) * kv_page;
    const std::uint64_t a1 = mapped > static_cast<std::uint64_t>(sl.pos) + 1 ? mapped - (sl.pos + 1ull) : 0;
    const std::uint64_t a2 = spec::rules::ctxRoom(sl.pos, ctx_);
    return static_cast<std::uint32_t>(std::min({static_cast<std::uint64_t>(lim), a1, a2}));
}

void Engine::decodeCycle(std::span<Slot* const> act_in) {
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

void Engine::dropOutOfRoom(Cycle& cy, std::span<Slot* const> act_in) {
    // slots out of context room end here
    const std::uint32_t nd_room = dc_.draftRoom();
    for (Slot* sl : act_in) {
        const bool room_ok = spec::rules::decodeRoom(sl->pos, nd_room, ctx_) && ensurePages(*sl, sl->pos + nd_room + 2);
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

void Engine::draftLimits(Cycle& cy) {
    // replay keeps no snapshot sets: the verify row budget is the only shared limit
    cy.snap_cap = m_.gdnReplay() ? spec_.vrows : std::min(qwen35::gdn_max_snap, m_.snapSets());
    // literal drafts (n-gram, DFlash2) per slot: the snapshot budget they had before CONC-8S
    cy.lit_cap = (m_.gdnReplay() || opt_.ng_snap_sets == 0) ? cy.snap_cap : std::min(cy.snap_cap, opt_.ng_snap_sets);
    cy.Au = static_cast<std::uint32_t>(cy.A);
    if (opt_.use_dflash) {
        cy.dfl_lim = ::whirl::dflash::draftLimit(m_.dflashMaxDrafts() + 1, qwen35::max_ng_drafts, cy.lit_cap, spec_.vrows, cy.Au, m_.fusedDecode());
    }
}

void Engine::ngramDrafts(Cycle& cy) {
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

void Engine::dflashDrafts(Cycle& cy) {
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

void Engine::mtpDrafts(Cycle& cy) {
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

void Engine::layoutBatch(Cycle& cy) {
    cy.tc0 = Clock::now();
    cy.pa = std::min<std::size_t>(cy.A, gdn_max_seg);
    cy.p = spec_.prof.get();
    if (cy.p && cy.p->events) m_.profileStart(0, cy.pa);
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
            cy.msegs[q] = MSeg{s.id, std::span<const std::uint32_t>(s.pend.data(), s.n_pend), s.pend_pos, s.pos};
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
            cy.row_of[k] = row;
            cy.snap_of[k] = snap;
            row += cy.nd_of[k] + 1;
            snap += cy.nd_of[k];
        }
    }
}

void Engine::draftSampling(Cycle& cy) {
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

void Engine::enqueueVerify(Cycle& cy) {
    if (cy.nd_max > 0) {
        m_.setDraftCutoff(opt_.p_min, std::min(opt_.n_min, cy.nd_max));
        struct Reset {
            ServerModel& m;
            ~Reset() { m.setDraftCutoff(0, 0); }
        } reset{m_};
        m_.mtpBatchStepEx(std::span<const MSeg>(cy.msegs.data(), cy.A), 0, true);
        for (std::uint32_t r = 1; r < cy.nd_max; ++r)
            m_.mtpBatchStepEx(std::span<const MSeg>(cy.msegs.data(), cy.n_step[r]), r, true);
    } else if (opt_.use_mtp) {
        // no MTP drafts this cycle: still feed the pend rows to the MTP block so
        // its KV cache stays complete for later drafting cycles
        m_.mtpBatchStepEx(std::span<const MSeg>(cy.msegs.data(), cy.A), 0, false);
    }
    if (cy.p && cy.p->events) m_.profileStart(1, cy.pa);
    cy.rows = m_.verifyBatchEnqueue(std::span<const VSeg>(cy.vsegs.data(), cy.A));
    if (cy.p) m_.profileStop();
}

void Engine::sampleRows(Cycle& cy) {
    bool any_sampling = false, any_relax = false;
    cy.rk = opt_.relax.topk();
    for (std::size_t k = 0; k < cy.A; ++k) {
        Slot& sl = *cy.act[k];
        if (sl.greedy) {
            if (relaxes(cy, k) && cy.nd_of[k] > 0) {
                any_relax = true;
                sb_.launchTopk(cy.row_of[k], cy.nd_of[k], cy.rk, 1.0f, sb_.relaxDev(), max_rows, cy.row_of[k]);
            }
            continue;
        }
        any_sampling = true;
        sb_.launchTopk(cy.row_of[k], cy.nd_of[k] + 1, k_small, sl.sm.inv_t, sb_.sampDev(), max_rows, cy.row_of[k]);
    }
    const TimePoint t_enq = Clock::now();
    m_.readCtl(std::span<std::int32_t>(sb_.ctlHost().data(), slots_.size() * qwen35::ctl_words));
    if (cy.p) {
        cy.p->t_ready = Clock::now();
        cy.p->enq_ms[cy.pa] += std::chrono::duration<double, std::milli>(t_enq - cy.tc0).count();
    }
    if (any_sampling) ops_.download(sb_.sampHost().data(), sb_.sampDev(), SampleBuffers::sampBytes());
    if (any_relax) ops_.download(sb_.relaxHost().data(), sb_.relaxDev(), sb_.relaxHost().size());
}

void Engine::trainCycle(Cycle& cy) {
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
    const float cycle_ms = opt_.relax.on ? synthMs(cy.n_ng == cy.A ? 0 : cy.nd_max) : static_cast<float>(msSince(cy.tc0));
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

void Engine::cycleStats(Cycle& cy) {
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

void Engine::acceptSlot(Cycle& cy, std::size_t k, const sampling::Acceptor& acceptor) {
    Slot& sl = *cy.act[k];
    const std::int32_t* ctl = sb_.ctlHost().data() + static_cast<std::size_t>(sl.id) * qwen35::ctl_words;
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
    // draft acceptance (src/sampling/acceptor.h): greedy / relaxacc greedy, or the sampler's draw at every
    // output index (inverse CDF / specsample race / relaxacc typical)
    const sampling::Drafts drafts{std::span<const std::uint32_t>(dr.data(), dr.size()), nd_dev, row0};
    const sampling::RepeatGuard repeats = [&](std::uint32_t a) { return relaxRepeats(sl, dr, a); };
    const sampling::Accepted ac =
        sl.greedy ? acceptor.greedy(ctl + qwen35::ctl_rows, drafts, relaxes(cy, k), sampling::RelaxRows{sb_.relaxHost().data(), max_rows, cy.rk},
                                    repeats, sl.relaxed, sl.relax_gap)
                  : acceptor.sample(sb_, sl.sm, drafts, relaxes(cy, k), sl.n_gen, repeats, sl.relaxed);
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
    // an accepted end-of-turn draft ends the reply: the trunk keeps only the
    // rows before it and the stop token becomes `chosen`
    if (!sl.ignore_eos) {
        for (std::uint32_t r = 0; r < acc; ++r) {
            if (dr[r] == ct_.im_end || dr[r] == ct_.endoftext) {
                chosen = dr[r];
                acc = r;
                break;
            }
        }
    }
    // same for max_tokens: the last emitted token is not fed to the trunk
    {
        const std::uint32_t budget = sl.max_tokens > sl.n_gen ? sl.max_tokens - sl.n_gen : 0;
        if (budget >= 1 && acc + 1 > budget) {
            chosen = dr[budget - 1];
            acc = budget - 1;
        }
    }
    // the verify ran nd + 1 rows of this slot: keep acc + 1 of them
    if (m_.gdnReplay()) m_.setPending(sl.id, acc + 1);
    else if (acc < nd_k) m_.restoreSeqSnapshot(sl.id, cy.snap_of[k] + acc);
    // DFlash2 context: the kept rows' features (whatever drafted them; rejected rows never enter)
    dc_.dflIngest(sl, cy.rows, row0, acc + 1, sl.pos);
    // trunk now holds next, dr[0..acc) at pos .. pos + acc
    sl.cache_tokens.push_back(sl.next);
    sl.cache_tokens.insert(sl.cache_tokens.end(), dr.begin(), dr.begin() + acc);
    const std::uint32_t n_before = sl.n_gen;
    for (std::uint32_t r = 0; r < acc; ++r)
        if (!sl.done) sl.done = emit(sl, dr[r]);
    if (!sl.done) sl.done = emit(sl, chosen);
    bst_.tokens += sl.n_gen - n_before;
    if (opt_.use_mtp)
        ops_.copyAsync(m_.seqHid(sl.id), m_.hn() + static_cast<std::uint64_t>(row0) * cy.E * 4, (acc + 1) * cy.E * 4,
                       m_.stream());
    for (std::uint32_t r = 0; r < acc; ++r) sl.pend[r] = dr[r];
    sl.pend[acc] = chosen;
    sl.n_pend = acc + 1;
    sl.pend_pos = sl.pos + 1;
    sl.next = chosen;
    sl.pos += acc + 1;
    sl.last_acc = acc;
    sl.last_row = row0 + acc;
    sl.logits_ok = true;
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
        if (const auto ev = sl.dec_watch.report(sl.n_gen, el, sl.cycles, vram_warn_ && vram_watch_.spilling(), prefill_tok_cum_)) {
            if (ev->warn) logW("req {} | slot {} | {}", sl.reqId(), sl.id, ev->text);
            else logI("req {} | slot {} | {}", sl.reqId(), sl.id, ev->text);
        }
    }
}

void Engine::finishCycle(Cycle& cy) {
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

void Engine::splitDrafts(std::span<Slot* const> act, std::uint32_t nd, std::uint32_t rows, std::span<std::uint32_t> nd_m) {
    for (auto& x : nd_m) x = nd;
    if (opt_.slot_drafts != 1 || !opt_.mtp_auto || act.size() < 2 || nd == 0) return;
    const std::uint32_t cap = std::min(opt_.n_draft, qwen35::max_drafts);
    std::array<const qwen35::DraftAccept*, gdn_max_seg> acc{};
    for (std::size_t k = 0; k < act.size(); ++k) acc[k] = &act[k]->dacc;
    splitDraftsByGain(std::span<const qwen35::DraftAccept* const>(acc.data(), act.size()), nd, rows, cap, nd_m);
}

std::uint32_t Engine::draftCap(std::uint32_t A) {
    return slotDraftCap(opt_.batch_drafts[std::min(A, gdn_max_seg)], opt_.n_draft, m_.fusedDecode(), m_.gdnReplay(), A, spec_.vrows);
}

}  // namespace whirl::server
