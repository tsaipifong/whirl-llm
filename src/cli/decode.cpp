// whirl CLI: the decode loops (plain, MTP speculative, DFlash2) and their options / summaries (arch025-C2)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"
#include "sampling/acceptor.h"  // draft acceptance shared with the server (arch025-E1a)
#include "spec/spec_rules.h"   // speculative-decoding rules shared with the server (arch025-R12cli)

namespace cli {

// ---------------------------------------------------------------------------
// decode loops


// Pipelined decode: step k+1 is enqueued before the host waits for token k.
DecodeResult plainDecode(q::Model& model, const Tok& tok, u32 first, std::size_t n_prompt, const ChatOpts& opt) {
    DecodeResult r;
    Streamer st{tok, opt.stream};
    constexpr int ring_n = 4;
    device::PinnedBuf ring_buf = device::PinnedBuf::alloc(device::hipBackend(), 4 * ring_n);
    auto* ring = static_cast<std::int32_t*>(ring_buf.get());
    device::EventH events[ring_n];
    for (auto& e : events) e = device::EventH::adopt(device::hipBackend(), hip::eventCreate());  // timing, as before
    auto enq = [&](std::size_t kk) {
        model.decodeStep();
        const std::size_t slot = kk % ring_n;
        hip::downloadAsync(&ring[slot], model.scratch().out_tok, 4, model.stream);
        hip::eventRecord(events[slot].get(), model.stream);
    };
    Timer timer;
    timer.begin();
    u32 n_gen = 0;
    u32 cur = first;
    std::size_t kk = 0;
    const u32 budget_ctx = opt.max_ctx - static_cast<u32>(n_prompt);
    if (opt.max_tokens > 1) enq(1);
    TpsTrace tps;
    tps.start(opt.trace_tps);
    while (n_gen < opt.max_tokens && n_gen < budget_ctx) {
        if (cur == tok.ct.im_end || cur == tok.ct.endoftext) break;
        r.tokens.push_back(cur);
        n_gen += 1;
        tps.tick(n_gen);
        if (n_gen == opt.max_tokens) break;
        if (n_gen + 1 < opt.max_tokens) enq(kk + 2);
        hip::eventSync(events[(kk + 1) % ring_n].get());
        cur = static_cast<u32>(ring[(kk + 1) % ring_n]);
        kk += 1;
        st.update(r.tokens);
    }
    r.decode_ms = timer.end();
    r.steps = n_gen > 0 ? n_gen - 1 : 0;
    for (auto& e : events) e.reset();  // events, then the ring (the order of before arch024 S11)
    ring_buf.reset();
    if (!opt.stream && opt.print_text) out(tok.decode(r.tokens));
    return r;
}

// Speculative decoding with the checkpoint's MTP head (+ n-gram drafts). The loop (single-sequence device calls,
// snapshots) is the CLI's own; the rules (draft caps, n-gram vs MTP choice, context stop, relaxacc repetition guard
// and synthetic cycle cost) are src/spec/spec_rules.h and the acceptance is sampling::Acceptor, both shared with the
// server engine (arch025-R12cli route B).
DecodeResult specDecode(q::Model& model, const Tok& tok, u32 first, u32 n_prompt, const ChatOpts& opt, u32 drafts,
                        std::span<const u32> prompt_ids) {
    DecodeResult res;
    const u64 E = model.cfg.n_embd;
    constexpr u32 MD = q::max_drafts;
    Streamer st{tok, opt.stream};
    std::vector<u32>& generated = res.tokens;
    u32 next = first;
    u32 p = n_prompt;  // position of `next`
    std::array<u32, q::max_small_batch> pend{};
    pend[0] = first;
    std::size_t n_pend = 1;
    u32 pend_pos = p;
    u32 cycles = 0, accepted = 0;
    bool done = false;
    double t_draft = 0, t_verify = 0;
    std::size_t verify_tokens = 0;
    // drafts per cycle: more than 1 needs the fused decode path (multi-row snapshots)
    const u32 n_draft = spec::rules::soloDraftCap(drafts, model.fusedDecode());
    const float p_min = opt.mtp_p_min;
    const u32 n_min = std::min(opt.mtp_n_min, n_draft);
    model.draft_p_min = p_min;
    model.draft_n_min = n_min;
    model.ensureSnapshots(n_draft);
    u32 drafted = 0;
    std::array<u32, MD> pos_tries{}, pos_hits{};
    float acc_ema = static_cast<float>(n_draft);
    q::DraftAccept acc_model;
    u32 prev_nd = 0;
    q::DraftTiming timing;
    std::vector<u32> hist;
    q::NgramPolicy ngp;
    ngp.verify_slope = opt.ngram_slope;
    ngp.reset();
    if (opt.ngram) ngp.ng.norm = chat::ChatTokens::crlfToLf(tok.t);
    // the unfused decode path snapshots row 0 only: more than 1 draft would restore an unwritten set
    u32 ng_max = spec::rules::ngramDraftCap(q::max_ng_drafts, q::max_small_batch, 1, opt.ngram_max, model.fusedDecode());
    if (opt.ngram && ng_max > model.snap_sets) {
        // n-gram verifies need a recurrent-state snapshot set per draft: take them now
        // if they fit in free VRAM with room to spare, else draft at most what exists
        u64 per = 0;
        for (u32 i = 0; i < model.cfg.n_layer; ++i)
            if (model.ssm_state[i] != 0) per += model.convBytes() + model.ssmBytes();
        const hip::MemInfo mem = hip::memInfo();
        const u64 extra = static_cast<u64>(ng_max - model.snap_sets) * per;
        if (mem.free > extra + (1536ull << 20))
            model.ensureSnapshots(ng_max);
        else
            ng_max = std::max<u32>(model.snap_sets, 1);
    }
    u32 ng_cycles = 0, ng_drafted = 0, ng_accepted = 0;
    auto emit = [&](u32 t) {
        if (t == tok.ct.im_end || t == tok.ct.endoftext) return true;
        generated.push_back(t);
        return generated.size() >= opt.max_tokens;
    };
    // relaxacc (fast mode): top-K of the verify rows for the relaxed greedy rule (whirl/relax_accept.h)
    const whirl::relax::Params rx = model.relax();
    const u32 rk = rx.topk();
    hip::Function rx_topk = nullptr;
    device::DevBuf rx_own;
    hip::DevPtr rx_dev = 0;
    std::vector<std::uint8_t> rx_host;
    if (rx.on) {
        rx_topk = model.module.getFunction("topk_rows");
        const std::size_t rb = static_cast<std::size_t>(q::max_small_batch) * rk * 8 + q::max_small_batch * 8;
        rx_own = device::DevBuf::alloc(device::hipBackend(), rb);
        rx_dev = rx_own.get();
        rx_host.assign(rb, 0);
    }
    // hist = prompt + every token the trunk holds or is about to take (`next` last): the n-gram lookup's history and
    // the relaxacc repetition guard's (the server's cache_tokens + next)
    const bool keep_hist = opt.ngram || rx.on;
    if (keep_hist) {
        hist.assign(prompt_ids.begin(), prompt_ids.end());
        hist.push_back(first);
    }
    const sampling::Acceptor acceptor(rx);
    double relax_gap = 0;  // the acceptor's statistic (server only reports it)
    u32 relaxed = 0;
    Timer timer;
    timer.begin();
    TpsTrace tps;
    tps.start(opt.trace_tps);
    done = emit(next);
    while (!done && spec::rules::decodeRoom(p, n_draft, opt.max_ctx)) {
        const double c0 = nowMs();
        // n-gram drafts first (the server's order) when the history's suffix occurred before and they promise more
        // tokens per ms than MTP with the last automatic draft count (spec::rules::ngramAltDrafts)
        std::array<u32, q::max_ng_drafts> ngd{};
        std::size_t ng_n = 0;
        const std::size_t room = std::min<std::size_t>(ng_max, spec::rules::ctxRoom(p, opt.max_ctx));
        q::Ngram::Match dbg_mt;
        float dbg_score = 0;
        if (opt.ngram && room > 0) {
            const q::Ngram::Match mt = ngp.propose(hist, opt.ngram_min, std::span<u32>(ngd.data(), room));
            dbg_mt = mt;
            if (mt.n > 0) {
                const float mtp_score = spec::rules::mtpScore(acc_model, timing, spec::rules::ngramAltDrafts(prev_nd, n_draft));
                dbg_score = mtp_score;
                ng_n = opt.ngram_force ? mt.n : ngp.choose(mt.n, mt.mlen, mtp_score, timing);
            }
        }
        // MTP draft count (cycles without n-gram drafts, as the server's MTP slots)
        u32 nd_host = n_draft;
        if (ng_n == 0) {
            if (opt.mtp_auto) {
                const q::DraftAccept* accs[1] = {&acc_model};
                nd_host = q::pickDrafts(accs, timing, n_draft, cycles, prev_nd);
                prev_nd = nd_host;
            } else if (opt.mtp_adapt > 0) {
                const float emas[1] = {acc_ema};
                nd_host = spec::rules::adaptDrafts(emas, opt.mtp_adapt, n_draft);
            }
        }
        if (ng_n > 0) {
            nd_host = static_cast<u32>(ng_n);
            // MTP KV rows only; the verify takes the drafts from the host
            model.mtpEnqueue(model.scratch().mtp_h, std::span<const u32>(pend.data(), n_pend), 0, pend_pos, std::nullopt);
        } else {
            // drafts chain through their device slots into the verify ids
            model.mtpEnqueue(model.scratch().mtp_h, std::span<const u32>(pend.data(), n_pend), 0, pend_pos, 0u);
            for (u32 dri = 1; dri < nd_host; ++dri) {
                // chained drafts: the MTP's own head-normed output stands in for the main hidden
                model.mtpEnqueue(model.scratch().h, {}, model.draftSlot(dri - 1), p + dri, dri);
            }
        }
        if (ng_n > 0)
            model.verifyEnqueueEx(next, nd_host, std::span<const u32>(ngd.data(), ng_n), p);
        else
            model.verifyEnqueueEx(next, nd_host, std::nullopt, p);
        verify_tokens += nd_host + 1;
        if (rx.on && nd_host > 0 && ng_n == 0) {
            const hip::DevPtr ids = rx_dev, vals = rx_dev + static_cast<u64>(q::max_small_batch) * rk * 4,
                         stats = rx_dev + static_cast<u64>(q::max_small_batch) * rk * 8;
            hip::launch(rx_topk, hip::Dim3{nd_host, 1, 1}, hip::Dim3{1024, 1, 1}, 0, model.stream, model.scratch().logits,
                        static_cast<std::int32_t>(model.cfg.n_vocab), static_cast<std::int32_t>(rk), 1.0f, ids, vals, stats);
            hip::downloadAsync(rx_host.data(), rx_dev, rx_host.size(), model.stream);
        }
        const double c1 = nowMs();
        std::array<u32, q::max_small_batch> dr{}, outv{};
        u32 nd;
        if (ng_n > 0) {
            (void)model.readCycle(std::span<u32>(outv.data(), nd_host + 1), std::span<u32>(dr.data(), 0));
            std::copy(ngd.begin(), ngd.begin() + static_cast<std::ptrdiff_t>(ng_n), dr.begin());
            nd = nd_host;
        } else {
            nd = model.readCycle(std::span<u32>(outv.data(), nd_host + 1), std::span<u32>(dr.data(), nd_host)).nd;
        }
        const double c2 = nowMs();
        // relaxacc: the draft-count policy sees a deterministic cycle cost, not wall-clock times
        // (the kept tokens depend on the drafts; see the server engine)
        const float cyc_ms = rx.on ? spec::rules::synthCycleMs(1.0f, static_cast<float>(nd_host + 1), opt.ngram_slope, ng_n > 0 ? 0u : nd_host)
                                   : static_cast<float>(c2 - c0);
        t_draft += c1 - c0;
        t_verify += c2 - c1;
        cycles += 1;
        drafted += nd;
        if (rx.on && nd > 0 && ng_n == 0) hip::streamSync(model.stream);
        // the server's acceptance (sampling::Acceptor::greedy): argmax rows, relaxacc for MTP drafts only (n-gram drafts
        // copy the history: exact only), the repetition guard on prompt + trunk (hist without `next`, then next + drafts)
        const bool relax_k = dflash::relaxAppliesTo(ng_n > 0 ? dflash::DraftKind::ngram : dflash::DraftKind::mtp, rx.on);
        const std::span<const u32> drs(dr.data(), dr.size());
        const sampling::RepeatGuard repeats = [&](u32 a) {
            return spec::rules::relaxRepeats(std::span<const u32>(hist.data(), hist.size() - 1), hist.back(), drs, a);
        };
        const u32 acc = acceptor
                            .greedy(reinterpret_cast<const std::int32_t*>(outv.data()), sampling::Drafts{drs, nd, 0}, relax_k,
                                    sampling::RelaxRows{rx_host.data(), q::max_small_batch, rk}, repeats, relaxed, relax_gap)
                            .acc;
        accepted += acc;
        if (opt.ngram_debug) {
            u32 hyp = 0;
            while (hyp < dbg_mt.n && hyp <= acc && hyp < nd_host + 1 && ngd[hyp] == outv[hyp]) hyp += 1;
            const u32 b = q::NgramPolicy::bucket(dbg_mt.mlen);
            const auto e1 = ngp.timing.estimate(static_cast<u32>(std::max<std::size_t>(dbg_mt.n, 1)));
            const auto e15 = timing.estimate(15);
            std::fprintf(stderr, "CY ng=%zu prop=%zu mlen=%u acc=%u nd=%u ms=%.1f a0=%.2f a3=%.2f mtp_score=%.3f ngt=%.1f ngt15=%.1f hyp>=%u\n", ng_n,
                         dbg_mt.n, dbg_mt.mlen, acc, nd_host, c2 - c0, ngp.acc[b].alpha[0], ngp.acc[b].alpha[3], dbg_score, e1 ? *e1 : -1.0f,
                         e15 ? *e15 : -1.0f, hyp);
        }
        if (ng_n > 0) {
            // n-gram cycles stay out of the MTP acceptance / cycle-time models
            ng_cycles += 1;
            ng_drafted += nd;
            ng_accepted += acc;
            ngp.timing.update(nd_host, cyc_ms);
        } else {
            for (u32 kk = 0; kk < std::min(acc + 1, nd); ++kk) pos_tries[kk] += 1;
            for (u32 kk = 0; kk < acc; ++kk) pos_hits[kk] += 1;
            acc_ema = spec::rules::accEma(acc_ema, acc);
            acc_model.update(nd, acc);
            timing.update(nd_host, cyc_ms);
        }
        // the verify ran nd_host + 1 rows: keep acc + 1 of them
        if (acc < nd_host) model.restoreSnapshot(acc + 1);
        // emit accepted drafts plus the model's own next token (relaxacc: the accepted drafts
        // themselves, which may differ from the rows' argmax)
        for (u32 r = 0; r < acc + 1; ++r) {
            const u32 t = r < acc ? dr[r] : outv[r];
            if (!done) done = emit(t);
            if (keep_hist) hist.push_back(t);
        }
        hip::copyAsync(model.scratch().mtp_h, model.scratch().hn, (acc + 1) * E * 4, model.stream);
        for (u32 r = 0; r < acc; ++r) pend[r] = dr[r];
        pend[acc] = outv[acc];
        n_pend = acc + 1;
        pend_pos = p + 1;
        next = outv[acc];
        p += acc + 1;
        tps.tick(generated.size());
        st.update(generated);
    }
    res.decode_ms = timer.end();
    model.draft_p_min = 0;
    rx_own.reset();
    if (rx.on)
        std::fprintf(stderr, "relaxacc: %u of %u accepted drafts kept by the relaxed rule (top-%u, alpha %.2f)\n", relaxed, accepted, rk, rx.alpha);
    if (!opt.stream && opt.print_text) out(tok.decode(generated));
    res.cycles = cycles;
    res.accepted = accepted;
    res.drafted = drafted;
    res.verify_tokens = verify_tokens;
    res.pos_tries = pos_tries;
    res.pos_hits = pos_hits;
    res.ng_cycles = ng_cycles;
    res.ng_drafted = ng_drafted;
    res.ng_accepted = ng_accepted;
    res.ng_alpha = {ngp.acc[0].alpha[0], ngp.acc[1].alpha[0], ngp.acc[2].alpha[0]};
    res.t_draft = t_draft;
    res.t_verify = t_verify;
    res.n_draft = n_draft;
    res.n_min = n_min;
    res.p_min = p_min;
    return res;
}

// Speculative decoding with DFlash2 drafts (+ n-gram drafts): one draft block per cycle against
// the drafter context of slot 0, a verify of the literal drafts (the n-gram path's bitwise verify),
// then the kept rows' captured features go into the context. Output = plain greedy decoding.
DecodeResult dflashDecode(q::Model& model, const Tok& tok, u32 first, u32 n_prompt, const ChatOpts& opt, std::span<const u32> prompt_ids) {
    DecodeResult res;
    Streamer st{tok, opt.stream};
    std::vector<u32>& generated = res.tokens;
    u32 next = first;
    u32 p = n_prompt;  // position of `next`
    const u32 max_d = !model.fusedDecode() ? 1 : std::min<u32>(model.dfr->block - 1, q::max_ng_drafts);
    u32 ng_max = opt.ngram_max > 0 ? std::min(opt.ngram_max, q::max_ng_drafts) : q::max_ng_drafts;
    if (!model.fusedDecode()) ng_max = 1;
    {
        // a recurrent-state snapshot set per draft row: take them if they fit in free VRAM with
        // room to spare, else draft at most what exists
        const u32 want = std::max(max_d, opt.ngram ? ng_max : 0u);
        if (want > model.snap_sets) {
            u64 per = 0;
            for (u32 i = 0; i < model.cfg.n_layer; ++i)
                if (model.ssm_state[i] != 0) per += model.convBytes() + model.ssmBytes();
            const hip::MemInfo mem = hip::memInfo();
            const u64 extra = static_cast<u64>(want - model.snap_sets) * per;
            if (mem.free > extra + (1536ull << 20)) model.ensureSnapshots(want);
        }
    }
    const u32 d_max = std::min(max_d, std::max<u32>(model.snap_sets, 1));
    ng_max = std::min(ng_max, std::max<u32>(model.snap_sets, 1));
    q::DraftAccept dacc;
    q::DraftTiming timing;  // DFlash2 cycle times (drafting included) by draft count
    float nd_ema = static_cast<float>(d_max);
    std::array<u32, q::max_drafts> pos_tries{}, pos_hits{};
    q::NgramPolicy ngp;
    ngp.verify_slope = opt.ngram_slope;
    ngp.reset();
    std::vector<u32> hist;
    if (opt.ngram) {
        ngp.ng.norm = chat::ChatTokens::crlfToLf(tok.t);
        hist.assign(prompt_ids.begin(), prompt_ids.end());
        hist.push_back(first);
    }
    u32 cycles = 0, accepted = 0, drafted = 0, ng_cycles = 0, ng_drafted = 0, ng_accepted = 0;
    double t_draft = 0, t_verify = 0;
    std::size_t verify_tokens = 0;
    bool done = false;
    auto emit = [&](u32 t) {
        if (t == tok.ct.im_end || t == tok.ct.endoftext) return true;
        generated.push_back(t);
        return generated.size() >= opt.max_tokens;
    };
    Timer timer;
    timer.begin();
    TpsTrace tps;
    tps.start(opt.trace_tps);
    done = emit(next);
    while (!done && p + 2 < opt.max_ctx) {
        const double c0 = nowMs();
        const u32 room = std::min<u32>(std::max(d_max, ng_max), opt.max_ctx - (p + 2));
        std::array<u32, q::max_ng_drafts> dr{};
        u32 nd = 0;
        bool is_ng = false;
        if (opt.ngram && room > 0) {
            const q::Ngram::Match mt = ngp.propose(hist, opt.ngram_min, std::span<u32>(dr.data(), std::min(room, ng_max)));
            if (mt.n > 0) {
                const u32 dn = std::min({d_max, room, static_cast<u32>(std::lround(nd_ema))});
                const auto t = timing.estimate(dn);
                const float alt = t ? dacc.expected(dn) / *t : 0.0f;
                nd = opt.ngram_force ? static_cast<u32>(mt.n) : ngp.choose(mt.n, mt.mlen, alt, timing);
                is_ng = nd > 0;
            }
        }
        if (!is_ng && room > 0) {
            const std::vector<u32> d = model.dflashDraft(0, next, p, opt.dfl_p_min, opt.dfl_n_min);
            nd_ema = 0.8f * nd_ema + 0.2f * static_cast<float>(d.size());
            nd = std::min({static_cast<u32>(d.size()), d_max, room});
            std::copy(d.begin(), d.begin() + nd, dr.begin());
        }
        model.verifyEnqueueEx(next, nd, std::span<const u32>(dr.data(), nd), p);
        verify_tokens += nd + 1;
        const double c1 = nowMs();
        std::array<u32, q::max_small_batch> outv{};
        (void)model.readCycle(std::span<u32>(outv.data(), nd + 1), std::span<u32>(outv.data(), 0));
        const double c2 = nowMs();
        t_draft += c1 - c0;
        t_verify += c2 - c1;
        cycles += 1;
        drafted += nd;
        u32 acc = 0;
        while (acc < nd && outv[acc] == dr[acc]) acc += 1;
        accepted += acc;
        if (is_ng) {
            ng_cycles += 1;
            ng_drafted += nd;
            ng_accepted += acc;
            ngp.timing.update(nd, static_cast<float>(c2 - c0));
        } else {
            for (u32 kk = 0; kk < std::min(acc + 1, nd); ++kk) pos_tries[kk] += 1;
            for (u32 kk = 0; kk < acc; ++kk) pos_hits[kk] += 1;
            if (nd > 0) dacc.update(nd, acc);
            timing.update(nd, static_cast<float>(c2 - c0));
        }
        // the verify ran nd + 1 rows: keep acc + 1 of them, and their features
        if (acc < nd) model.restoreSnapshot(acc + 1);
        // (FR-DFL P2-2: via planIngest, so a capture shorter than the verify never reads before cap_buf)
        if (const whirl::dflash::Ingest g = whirl::dflash::verifyIngest(model.cap_n, nd + 1, acc + 1, p); g.any)
            model.dflashIngest(0, model.cap_buf, model.cap_rows, g.cap_row, g.n, g.pos);
        for (u32 r = 0; r < acc + 1; ++r) {
            if (!done) done = emit(outv[r]);
            if (opt.ngram) hist.push_back(outv[r]);
        }
        next = outv[acc];
        p += acc + 1;
        tps.tick(generated.size());
        st.update(generated);
    }
    res.decode_ms = timer.end();
    if (!opt.stream && opt.print_text) out(tok.decode(generated));
    res.cycles = cycles;
    res.accepted = accepted;
    res.drafted = drafted;
    res.verify_tokens = verify_tokens;
    res.pos_tries = pos_tries;
    res.pos_hits = pos_hits;
    res.ng_cycles = ng_cycles;
    res.ng_drafted = ng_drafted;
    res.ng_accepted = ng_accepted;
    res.ng_alpha = {ngp.acc[0].alpha[0], ngp.acc[1].alpha[0], ngp.acc[2].alpha[0]};
    res.t_draft = t_draft;
    res.t_verify = t_verify;
    res.n_draft = d_max;
    res.p_min = opt.dfl_p_min;
    res.n_min = opt.dfl_n_min;
    return res;
}

void printProfile(const char* label, q::Profile& p, std::size_t n_tok) {
    p.flush();
    double total = 0;
    for (double v : p.ms) total += v;
    const double per = static_cast<double>(std::max<std::size_t>(n_tok, 1));
    out(fmt("\n  [profile %s: %.1f ms GPU, %zu tok, %.2f ms/tok]\n", label, total, n_tok, total / per));
    for (std::size_t i = 0; i < q::n_classes; ++i)
        if (p.ms[i] > 0)
            out(fmt("    %-9s %9.1f ms  %7.3f ms/tok  %5.1f%%\n", q::opClassName(static_cast<q::OpClass>(i)), p.ms[i], p.ms[i] / per, 100.0 * p.ms[i] / total));
    const double mm = p.ms[static_cast<std::size_t>(q::OpClass::matmul)];
    if (mm > 0) out(fmt("    matmul weight stream %.0f GB/s\n", static_cast<double>(p.matmul_bytes) / (mm * 1e6)));
}

ChatOpts mtpOpts(const q::Model& model, ChatOpts base, u32* drafts_out) {
    const q::MtpDefaults def = q::mtpDefaults(model.cfg.moe, model.arch);
    // the run settings shared with whirl-server (config.settings, arch025-N15 rest): each read at its old place
    namespace st = whirl::settings;
    const std::optional<u32> dv = st::mtpDrafts(def.drafts, q::max_drafts);
    const u32 drafts = dv.value_or(def.drafts);
    base.trace_tps = envU32("TRACE_TPS").value_or(0);
    base.mtp_p_min = def.p_min;
    base.mtp_adapt = def.adapt;
    base.ngram_min = def.ngram_min;
    base.ngram_slope = def.ngram_slope;
    // an explicit draft count is used as is (fixed) unless WHIRL_MTP_ADAPT asks otherwise
    base.mtp_auto = def.automatic && !dv;
    base.mtp_p_min = st::mtpPmin(base.mtp_p_min);
    base.mtp_n_min = st::mtpNmin(base.mtp_n_min, settingsWarn);
    if (const whirl::settings::MtpAdapt ad = whirl::settings::mtpAdapt(); ad.set) {  // (the server's rule, arch025-N15)
        base.mtp_auto = ad.automatic;
        base.mtp_adapt = ad.adapt;
    }
    base.ngram = st::ngram(base.ngram);
    base.ngram_min = st::ngramMin(base.ngram_min, settingsWarn);
    base.ngram_slope = st::ngramSlope(base.ngram_slope);
    base.ngram_max = st::ngramMax(base.ngram_max, settingsWarn);
    base.ngram_force = st::ngramForce(base.ngram_force);
    if (auto v = envGet("NGRAM_DEBUG")) base.ngram_debug = *v != "0";
    *drafts_out = drafts;
    return base;
}

// --draft dflash: the walk cutoffs (whirl/dflash_policy.h) and the n-gram settings, with the same
// per-model defaults and environment as the MTP path (mtpOpts)
ChatOpts dflashOpts(const q::Model& model, ChatOpts base) {
    u32 unused = 0;
    base = mtpOpts(model, base, &unused);
    std::vector<std::string> warn;
    const whirl::dflash::Params dp = whirl::dflash::parseParams([](const char* n) { return envGet(n); }, warn);
    for (const std::string& w : warn) std::fprintf(stderr, "warning: %s\n", w.c_str());
    base.dfl_p_min = dp.p_min;
    base.dfl_n_min = dp.n_min;
    return base;
}

void printSpecSummary(const DecodeResult& r, std::size_t n_prompt, double prefill_ms, const ChatOpts& opt, const char* kind) {
    const std::size_t n_gen = r.tokens.size();
    const bool mtp = std::string_view(kind) == "MTP spec";
    out(fmt("\n\n  prefill: %zu tok in %.0f ms (%.1f tok/s)\n", n_prompt, prefill_ms, n_prompt * 1000.0 / prefill_ms));
    out(fmt("  decode (%s, max %u min %u p-min %.2f%s): %zu tok in %.0f ms (%.2f tok/s); %u verify cycles, drafts %u/%u accepted (%.2f tok/cycle)\n",
            kind, r.n_draft, r.n_min, r.p_min, !mtp ? "" : opt.mtp_auto ? " auto" : opt.mtp_adapt > 0 ? " adaptive" : "", n_gen, r.decode_ms, n_gen * 1000.0 / r.decode_ms,
            r.cycles, r.accepted, r.drafted, r.cycles > 0 ? static_cast<double>(n_gen) / r.cycles : 0.0));
    if (r.cycles > 0) {
        std::string line = "  acceptance by draft position (accepted/reached):";
        for (u32 kk = 0; kk < r.n_draft; ++kk) {
            if (r.pos_tries[kk] == 0) break;
            line += fmt(" %u:%.0f%%(%u)", kk + 1, 100.0 * r.pos_hits[kk] / r.pos_tries[kk], r.pos_tries[kk]);
        }
        out(line + "\n");
        if (opt.ngram)
            out(fmt("  n-gram drafts: %u of %u cycles, drafts %u/%u accepted; n-gram acceptance model (<8 / 8-23 / >=24 matched): %.2f %.2f %.2f\n",
                    r.ng_cycles, r.cycles, r.ng_accepted, r.ng_drafted, r.ng_alpha[0], r.ng_alpha[1], r.ng_alpha[2]));
        out(fmt("  per cycle: host enqueue %.2f ms, wait %.2f ms, other %.2f ms; verified rows %.2f\n", r.t_draft / r.cycles, r.t_verify / r.cycles,
                (r.decode_ms - r.t_draft - r.t_verify) / r.cycles, static_cast<double>(r.verify_tokens) / r.cycles));
    }
}

}  // namespace cli
