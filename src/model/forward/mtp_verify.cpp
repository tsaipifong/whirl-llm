// Model forward, MTP drafting and speculative verify (arch025-F4, moved from src/model/forward.cpp): token
// upload (setTokens / setTokensDev / ctlSlot), the MTP block (mtpForward / mtpEnqueue / mtpBatchStepEx), DFlash2
// layer-input capture, the batched and single verify (verifyBatchEnqueue / verifyEnqueueEx / verify), the replay
// commit (commitSeq) and snapshot restore, draft-head logits and argmax / cycle readback. The draft-head and batch
// choices are in src/ops/mtp.cpp (arch025-F6).
// SPDX-License-Identifier: Apache-2.0

#include "../forward_internal.h"

namespace whirl::qwen35 {

// Rows [0, host.size()) take host tokens, the next n_dev rows the i32 tokens at
// dev_src; positions pos0 + r. Kernel args only: never blocks.
void Model::setTokensDev(DevPtr ids_dev, std::span<const u32> host, DevPtr dev_src, u32 n_dev, u32 pos0) {
    const u32 n = static_cast<u32>(host.size()) + n_dev;
    visRowsCur(n, pos0);
    Tok16 t;
    for (std::size_t i = 0; i < host.size(); ++i) t.t[i] = lookupId(host[i]);
    row_n = n;
    for (u32 r = 0; r < n; ++r) {
        row_pos[r] = static_cast<i32>(pos0 + r);
        row_base[r] = static_cast<i32>(kv_off);
    }
    hip::launch(k.set_tokens, D(1), D(32), 0, stream, ids_dev, pos_buf, dev_src, t, I(n), I(host.size()), I(pos0));
}

DevPtr Model::ctlSlot(u32 word) const { return out_tok + 4 * static_cast<u64>(ctl_off + word); }

void Model::setTokens(DevPtr ids_dev, std::span<const u32> tokens, u32 pos0) {
    if (tokens.size() <= max_small_batch) return setTokensDev(ids_dev, tokens, 0, 0, pos0);
    const std::size_t cap = host_ids.size() / 2;
    const std::size_t n = tokens.size();
    visRowsCur(static_cast<u32>(n), pos0);
    for (std::size_t i = 0; i < n; ++i) {
        host_ids[i] = lookupId(tokens[i]);
        host_ids[cap + i] = static_cast<i32>(pos0 + i);
    }
    hip::upload(ids_dev, host_ids.data(), n * 4);
    hip::upload(pos_buf, host_ids.data() + cap, n * 4);
}

// MTP block over n rows: row r pairs hidden + r*E with token tokens[r] at
// position pos0 + r. Fills the MTP KV cache; if want_draft, returns the argmax
// of the last row.
u32 Model::mtpForward(DevPtr hidden, std::span<const u32> tokens, u32 pos0, bool want_draft) {
    mtpEnqueue(hidden, tokens, 0, pos0, want_draft ? std::optional<u32>(0) : std::nullopt);
    if (!want_draft) return 0;
    i32 r = 0;
    hip::download(&r, draftSlot(0), 4);
    return static_cast<u32>(r);
}

// mtpForward without a host round trip: the last row's token may come from
// the device (dev_tok) and the draft lands in draftSlot(slot).
void Model::mtpEnqueue(DevPtr hidden, std::span<const u32> tokens, DevPtr dev_tok, u32 pos0, std::optional<u32> slot) {
    if (!mtp) throw ModelError("NoMtp");
    const MtpW& mw = *mtp;
    const u32 E = cfg.n_embd;
    const u32 n_dev = dev_tok != 0 ? 1 : 0;
    const u32 n = static_cast<u32>(tokens.size()) + n_dev;
    xq_src = 0;
    x16_src = 0;
    // chained draft after a possible device-side cutoff: gate its big kernels
    const bool gated = ops::draftGated(n_dev, tokens.size(), slot.has_value(), slot.value_or(0), draft_p_min);
    if (gated) gate = ctlSlot(ctl_stop);
    struct GateReset {
        DevPtr& g;
        ~GateReset() { g = 0; }
    } gate_reset{gate};
    if (n_dev > 0)
        setTokensDev(mtp_ids, tokens, dev_tok, n_dev, pos0);
    else
        setTokens(mtp_ids, tokens, pos0);
    hip::launch(k.get_rows[ti(tok_embd.ty)], D(8, n), D(128), 0, stream, tok_embd.ptr, tok_embd.row_bytes, mtp_ids, x, I(E));
    if (n_inj > 0) injectRows();
    rmsnorm(x, mw.enorm, mtp_cat, E, n, E, 2 * E);
    rmsnorm(hidden, mw.hnorm, mtp_cat + static_cast<u64>(E) * 4, E, n, E, 2 * E);
    matmul(mw.eh_proj, mtp_cat, x, n, false);
    if (fused(n))
        rmsnormQ8(x, mw.attn_norm, h, n);
    else
        rmsnorm(x, mw.attn_norm, h, E, n, E, E);
    draftAttn(mw.attn, n);  // drafts only: sink + last W positions (OVH-C1)
    if (mw.moe)
        moeBlock(mw.post_norm, mw.ffn_gate, mw.ffn_up, mw.ffn_down, *mw.moe, n);
    else
        ffnBlock(mw.post_norm, mw.ffn_gate, mw.ffn_up, mw.ffn_down, n);
    if (!slot) return;
    const u32 draft_slot = *slot;
    const DevPtr last = x + static_cast<u64>(n - 1) * E * 4;
    if (fused(1))
        rmsnormQ8(last, mw.head_norm, h, 1);
    else
        rmsnorm(last, mw.head_norm, h, E, 1, E, E);
    // draft head over the vocabulary subset (draft_map) or the first draft_vocab rows only
    Mat head = draft_head ? *draft_head : output;
    const ops::DraftHeadPlan dh = ops::chooseDraftHead(draftHeadProblem(head));
    head.nrows = dh.nrows;
    draftHeadMatmul(head, 1);
    const DevPtr map = dh.map ? *draft_map : DevPtr{0};
    hip::launch(k.draft_pick, D(1), D(1024), 0, stream, logits, I(head.nrows), ctlSlot(ctl_drafts), ctlSlot(ctl_probs), ctlSlot(ctl_nd),
                I(draft_slot), I(draft_n_min), draft_p_min, map);
}

// DFlash2 capture: x (the input of layer li) for the last min(n, cap_rows) rows into li's slot.
void Model::captureLayerInput(std::size_t li, u32 n) {
    if (!cap_buf || li >= cap_slot.size() || cap_slot[li] < 0 || n == 0) return;
    const u64 E = cfg.n_embd;
    const u32 c = std::min(n, cap_rows);
    hip::copyAsync(cap_buf + static_cast<u64>(cap_slot[li]) * cap_rows * E * 4, x + static_cast<u64>(n - c) * E * 4,
                   static_cast<std::size_t>(c) * E * 4, stream);
    cap_n = c;
}

// Apply sequence s's kept verify rows to its recurrent state (replay mode).
void Model::commitSeq(u32 s) {
    if (!gdn_replay || s >= seqs.size()) return;
    const Seq& sq = seqs[s];
    if (sq.npend == 0) return;
    const u32 ch = cfg.convCh();
    const float scale_g = 1.0f / std::sqrt(static_cast<float>(cfg.d_state));
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const Layer& L = layers[i];
        if (L.kind != LayerKind::gdn) continue;
        const GdnW& g = L.gdn;
        GdnSegs sc;
        sc.s[0].state = sq.conv[i];
        sc.s[0].pend = pendPtr(sq, i, true);
        sc.s[0].npend = static_cast<i32>(sq.npend);
        hip::launch(k.gdn_conv_l2, D(ch / 128), D(128), 0, stream, qkv, g.conv, conv_out, I(ch), I(2 * cfg.n_k_heads), cfg.eps, sc);
        GdnSegs ss;
        ss.s[0].state = sq.ssm[i];
        ss.s[0].pend = pendPtr(sq, i, false);
        ss.s[0].npend = static_cast<i32>(sq.npend);
        hip::launch(k.gdn_step_norm, D(cfg.n_v_heads), D(512), 0, stream, conv_out, alpha, beta, z, g.norm, gdn_out, xq, xd, I(cfg.n_k_heads),
                    scale_g, cfg.eps, I(cfg.n_v_heads), I(ch), ss);
    }
    xq_src = 0;
    seqs[s].npend = 0;
}

// Batched verify over sequences (see the header).
u32 Model::verifyBatchEnqueue(std::span<const VSeg> segs) {
    if (segs.empty() || segs.size() > gdn_max_seg) throw ModelError("BadBatch");
    // several sequences need the segment-aware fused DeltaNet kernels
    const bool unfused = !fusedDecode();
    if (unfused) {
        if (!ops::unfusedVerifyOk(segs.size(), segs[0].nd)) throw ModelError("BatchNeedsFusedDecode");
        selectSeq(segs[0].seq);
        snap_rows = segs[0].nd;
    }
    struct SnapReset {
        u32& r;
        ~SnapReset() { r = 0; }
    } snap_reset{snap_rows};
    BatchPlan plan;
    plan.segs = segs;
    RowTab tab;
    RowIdx dst;
    u32 n = 0;
    u32 snaps = 0;
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const VSeg& sg = segs[kk];
        if (sg.pos + sg.nd + 1 > max_ctx) throw ModelError("ContextTooLong");
        if (sg.nd + 1 > max_small_batch || n + sg.nd + 1 > verifyRows()) throw ModelError("BatchTooLarge");
        plan.snap_base[kk] = snaps;
        snaps += sg.nd;
        for (u32 r = 0; r < sg.nd + 1; ++r) {
            // drafts: the sequence's control area (MTP), or literal tokens (n-gram)
            if (r == 0)
                tab.tok[n] = lookupId(sg.next);
            else if (!sg.drafts.empty())
                tab.tok[n] = lookupId(sg.drafts[r - 1]);
            else
                tab.tok[n] = -static_cast<i32>(sg.seq * ctl_words + ctl_drafts + r - 1) - 1;
            tab.pos[n] = static_cast<i32>(sg.pos + r);
            tab.base[n] = static_cast<i32>(seqs[sg.seq].kv_base);
            dst.v[n] = static_cast<i32>(sg.seq * ctl_words + ctl_rows + r);
            n += 1;
        }
    }
    if (!gdn_replay) {
        if (snaps > gdn_max_snap) throw ModelError("BatchTooLarge");
        ensureSnapshots(snaps);
    }
    row_n = n;
    for (u32 ri = 0; ri < n; ++ri) {
        row_pos[ri] = tab.pos[ri];
        row_base[ri] = tab.base[ri];
    }
    // vision: RoPE rows of a batched verify (seg k: nd + 1 rows from sg.pos)
    visBegin();
    if (visAny()) {
        bool any = false;
        for (const VSeg& sg : segs)
            if (visOf(sg.seq) != nullptr) any = true;
        if (any) {
            visEnsure();
            u32 r = 0;
            for (const VSeg& sg : segs)
                for (u32 j = 0; j < sg.nd + 1; ++j) visRow(visOf(sg.seq), r++, sg.pos + j);
            n_inj = 0;  // decode rows are text
            visCommit(n);
        }
    }
    hip::launch(k.set_rows, D(1), D(32), 0, stream, ids, pos_buf, kvbase_buf, out_tok, tab, I(n));
    keep_hidden = true;
    all_logits = true;
    bplan = &plan;
    small_max = ops::verifySmallMax(n, max_small_batch, max_verify_rows);
    struct Reset {
        Model& m;
        ~Reset() {
            m.keep_hidden = false;
            m.all_logits = false;
            m.bplan = nullptr;
            m.small_max = max_small_batch;
        }
    } reset_flags{*this};
    xq_src = 0;
    x16_src = 0;
    run(n);
    // the kept rows of the previous verify are in the state now
    if (gdn_replay)
        for (const VSeg& sg : segs) seqs[sg.seq].npend = 0;
    hip::launch(k.argmax_rows_to, D(n), D(1024), 0, stream, logits, I(cfg.n_vocab), out_tok, dst);
    return n;
}

// Batched MTP draft step r over sequences (see the header).
void Model::mtpBatchStepEx(std::span<const MSeg> segs, u32 r, bool draft) {
    if (!mtp) throw ModelError("NoMtp");
    const MtpW& mw = *mtp;
    const u64 E = cfg.n_embd;
    if (segs.empty() || segs.size() > max_small_batch) throw ModelError("BadBatch");
    RowTab tab;
    RowIdx last;
    RowIdx area;
    u32 n = 0;
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const MSeg& sg = segs[kk];
        const Seq& sq = seqs[sg.seq];
        if (r == 0) {
            if (sg.pend.size() > max_small_batch || n + sg.pend.size() > verifyRows()) throw ModelError("BatchTooLarge");
            hip::copyAsync(mtp_in + static_cast<u64>(n) * E * 4, sq.hid, sg.pend.size() * E * 4, stream);
            for (std::size_t j = 0; j < sg.pend.size(); ++j) {
                tab.tok[n] = lookupId(sg.pend[j]);
                tab.pos[n] = static_cast<i32>(sg.pend_pos + j);
                tab.base[n] = static_cast<i32>(sq.kv_base);
                n += 1;
            }
        } else {
            tab.tok[n] = -static_cast<i32>(sg.seq * ctl_words + ctl_drafts + r - 1) - 1;
            tab.pos[n] = static_cast<i32>(sg.pos + r);
            tab.base[n] = static_cast<i32>(sq.kv_base);
            n += 1;
        }
        last.v[kk] = static_cast<i32>(n - 1);
        area.v[kk] = static_cast<i32>(sg.seq * ctl_words);
    }
    const DevPtr hidden = r == 0 ? mtp_in : h;
    xq_src = 0;
    x16_src = 0;
    mtp_batch = true;
    small_max = ops::verifySmallMax(n, max_small_batch, max_verify_rows);
    struct BatchReset {
        Model& m;
        ~BatchReset() {
            m.mtp_batch = false;
            m.small_max = max_small_batch;
        }
    } batch_reset{*this};
    row_n = n;
    for (u32 ri = 0; ri < n; ++ri) {
        row_pos[ri] = tab.pos[ri];
        row_base[ri] = tab.base[ri];
    }
    // vision: RoPE rows of the batched MTP step
    visBegin();
    if (visAny()) {
        bool any = false;
        for (const MSeg& sg : segs)
            if (visOf(sg.seq) != nullptr) any = true;
        if (any) {
            visEnsure();
            u32 ri = 0;
            for (const MSeg& sg : segs) {
                const std::size_t cnt = r == 0 ? sg.pend.size() : 1;
                for (std::size_t j = 0; j < cnt; ++j) {
                    const u32 p = r == 0 ? sg.pend_pos + static_cast<u32>(j) : sg.pos + r;
                    visRow(visOf(sg.seq), ri++, p);
                }
            }
            n_inj = 0;  // decode rows are text
            visCommit(n);
        }
    }
    hip::launch(k.set_rows, D(1), D(32), 0, stream, mtp_ids, pos_buf, kvbase_buf, out_tok, tab, I(n));
    hip::launch(k.get_rows[ti(tok_embd.ty)], D(8, n), D(128), 0, stream, tok_embd.ptr, tok_embd.row_bytes, mtp_ids, x, I(E));
    rmsnorm(x, mw.enorm, mtp_cat, static_cast<u32>(E), n, static_cast<u32>(E), static_cast<u32>(2 * E));
    rmsnorm(hidden, mw.hnorm, mtp_cat + E * 4, static_cast<u32>(E), n, static_cast<u32>(E), static_cast<u32>(2 * E));
    matmul(mw.eh_proj, mtp_cat, x, n, false);
    rmsnormQ8(x, mw.attn_norm, h, n);
    draftAttn(mw.attn, n);  // drafts only: sink + last W positions (OVH-C1)
    if (mw.moe)
        moeBlock(mw.post_norm, mw.ffn_gate, mw.ffn_up, mw.ffn_down, *mw.moe, n);
    else
        ffnBlock(mw.post_norm, mw.ffn_gate, mw.ffn_up, mw.ffn_down, n);
    if (!draft) return;
    // head over each seg's last row
    const u32 ns = static_cast<u32>(segs.size());
    hip::launch(k.rmsnorm_q8_rows, D(ns), D(1024), 0, stream, x, mw.head_norm, h, xq, xd, I(E), cfg.eps, last);
    xq_src = h;
    xq_n = ns;
    x16_src = 0;
    Mat head = draft_head ? *draft_head : output;
    const ops::DraftHeadPlan dh = ops::chooseDraftHead(draftHeadProblem(head));
    head.nrows = dh.nrows;
    draftHeadMatmul(head, ns);
    const DevPtr map = dh.map ? *draft_map : DevPtr{0};
    hip::launch(k.draft_pick_rows, D(ns), D(1024), 0, stream, logits, I(head.nrows), out_tok, area, I(r), I(draft_n_min), draft_p_min,
                map);
    if (draftSampleOk()) {
        // specsample: rows of sampling requests redraw draft r from the filtered draft distribution  by the race
        // coupled to the target sampler for output index pos0 + r (whirl/spec_sample.h)
        kernels::DSampArgs da;
        bool any = false;
        for (u32 kk = 0; kk < ns; ++kk) {
            const u32 sq = segs[kk].seq;
            if (sq >= draft_samp.size() || !draft_samp[sq].on) continue;
            const spec::DraftSample& ds = draft_samp[sq];
            any = true;
            da.on[kk] = 1;
            da.top_k[kk] = static_cast<i32>(std::min<std::uint32_t>(ds.top_k, 1u << 30));
            da.qoff[kk] = static_cast<i32>((static_cast<u64>(sq) * max_drafts + r) * spec::q_words);
            da.inv_t[kk] = ds.inv_t;
            da.top_p[kk] = ds.top_p;
            da.min_p[kk] = ds.min_p;
            const std::uint64_t key = spec::raceBase(ds.seed, ds.pos0 + r);
            da.key_lo[kk] = static_cast<std::uint32_t>(key);
            da.key_hi[kk] = static_cast<std::uint32_t>(key >> 32);
        }
        if (any)
            hip::launch(k.draft_sample_rows, D(ns), D(1024), 0, stream, logits, I(head.nrows), out_tok, area, I(r), map, da, draft_qbuf);
    }
}

// the draft-head problem of `head` (the draft head, else the output head)
ops::DraftHeadProblem Model::draftHeadProblem(const Mat& head) const {
    ops::DraftHeadProblem p;
    p.head_rows = head.nrows;
    p.draft_rows = draft_rows;
    p.draft_vocab = draft_vocab;
    p.d2 = draft_d2.has_value();
    p.map = draft_map.has_value();
    return p;
}

void Model::readCtl(std::span<i32> dst) { hip::download(dst.data(), out_tok, dst.size() * 4); }

// feat026-acc: accept decision on the device (see the header and kernels/spec_accept.hip)
bool Model::specAcceptEnqueue(const SpecAccArgs& a, DevPtr host) {
    if (k.spec_accept_rows == nullptr) return false;
    if (a.n <= 0 || a.n > WHIRL_SA_MAX_SEG) throw ModelError("BadBatch");
    SpecAccArgs b = a;
    for (int kk = 0; kk < b.n; ++kk) {
        if (b.seq[kk] < 0 || static_cast<std::size_t>(b.seq[kk]) >= seqs.size() || b.nd[kk] < 0 || b.nd[kk] > WHIRL_SA_MAX_ND)
            throw ModelError("BadBatch");
        b.kvbase[kk] = static_cast<i32>(seqs[b.seq[kk]].kv_base);
    }
    b.n_vocab = static_cast<i32>(cfg.n_vocab);
    if (spec_tab == 0) spec_tab = allocZero(sizeof(SpecNextTab));
    hip::launch(k.spec_accept_rows, D(1), D(64), 0, stream, out_tok, b, host, spec_tab);
    return true;
}

void Model::readCtlPinned(std::span<i32> dst) {
    hip::downloadAsync(dst.data(), out_tok, dst.size() * 4, stream);
    hip::streamSync(stream);
}

void Model::readSpecTab(SpecNextTab& t) {
    if (spec_tab == 0) {
        t = SpecNextTab{};
        return;
    }
    hip::streamSync(stream);
    hip::download(&t, spec_tab, sizeof(SpecNextTab));
}

// Draft-head logits for the n rows of h into logits: the 2-bit head when
// built, else the Q4_K / full head.
void Model::draftHeadMatmul(const Mat& head, u32 n) {
    if (draft_d2) {
        if (xq_src != h || xq_n != n) {
            const u32 cnt = n * head.ncols;
            hip::launch(k.quantize_q8, D((cnt + 255) / 256), D(256), 0, stream, h, xq, xd, I(cnt));
            xq_src = h;
            xq_n = n;
        }
        hip::launch(k.gemv_d2[n - 1], D((head.nrows + 7) / 8), D(256), 0, stream, *draft_d2, xq, xd, logits, I(head.ncols), I(head.nrows), gate);
        mark(OpClass::head);
        return;
    }
    matmul(head, h, logits, n, false);
}

void Model::argmaxRows(u32 n, std::span<u32> out) {
    argmaxRowsEnqueue(n);
    i32 buf[max_small_batch];
    hip::download(buf, out_tok, n * 4);
    for (u32 r = 0; r < n; ++r) out[r] = static_cast<u32>(buf[r]);
}

void Model::argmaxRowsEnqueue(u32 n) {
    hip::launch(k.argmax_rows, D(n), D(1024), 0, stream, logits, I(cfg.n_vocab), ctlSlot(ctl_rows));
}

void Model::verifyEnqueueEx(u32 next, u32 n_draft, std::optional<std::span<const u32>> host_drafts, u32 pos0) {
    const u32 n = n_draft + 1;
    if (pos0 + n > max_ctx) throw ModelError("ContextTooLong");
    ensureSnapshots(n_draft);
    keep_hidden = true;
    all_logits = true;
    snap_rows = n_draft;
    struct Reset {
        Model& m;
        ~Reset() {
            m.keep_hidden = false;
            m.all_logits = false;
            m.snap_rows = 0;
        }
    } reset_flags{*this};
    xq_src = 0;
    x16_src = 0;
    if (host_drafts) {
        u32 tv[max_small_batch];
        tv[0] = next;
        for (u32 i = 0; i < n_draft; ++i) tv[1 + i] = (*host_drafts)[i];
        setTokensDev(ids, std::span<const u32>(tv, n), 0, 0, pos0);
    } else {
        const u32 one[1] = {next};
        setTokensDev(ids, one, draftSlot(0), n_draft, pos0);
    }
    run(n);
    argmaxRowsEnqueue(n);
}

Model::Cycle Model::readCycle(std::span<u32> out, std::span<u32> drafts) {
    i32 buf[ctl_words];
    hip::download(buf, out_tok, sizeof buf);
    for (std::size_t r = 0; r < out.size(); ++r) out[r] = static_cast<u32>(buf[ctl_rows + r]);
    for (std::size_t r = 0; r < drafts.size(); ++r) drafts[r] = static_cast<u32>(buf[ctl_drafts + r]);
    Cycle c;
    c.nd = static_cast<u32>(std::max<i32>(0, buf[ctl_nd]));
    for (u32 r = 0; r < max_small_batch; ++r) std::memcpy(&c.probs[r], &buf[ctl_probs + r], 4);
    c.nd = std::min<u32>(c.nd, static_cast<u32>(drafts.size()));
    return c;
}

// Main-model forward over tokens at pos0 keeping every row's logits and normed
// hidden; the recurrent state after rows 0 .. len-2 is snapshotted.
void Model::verify(std::span<const u32> tokens, u32 pos0, std::span<u32> out) {
    const u32 snaps = static_cast<u32>(tokens.size() - 1);
    ensureSnapshots(snaps);
    keep_hidden = true;
    all_logits = true;
    snap_rows = snaps;
    struct Reset {
        Model& m;
        ~Reset() {
            m.keep_hidden = false;
            m.all_logits = false;
            m.snap_rows = 0;
        }
    } reset_flags{*this};
    forward(tokens, pos0);
    argmaxRows(static_cast<u32>(tokens.size()), out);
}

// Roll the recurrent state back to the snapshot taken after row keep - 1.
void Model::restoreSnapshot(u32 keep) {
    // a captured decode graph holds the old state pointers
    decode_graph.reset();
    // the unfused decode path writes only snapshot 0: any later set would be stale
    if (keep == 0 || keep > snap_sets || (keep > 1 && !fusedDecode())) throw ModelError("SnapshotNotWritten");
    const u32 kk = keep - 1;
    for (u32 i = 0; i < cfg.n_layer; ++i) {
        if (ssm_state[i] == 0) continue;
        std::swap(ssm_state[i], snap_ssm[kk][i]);
        std::swap(conv_state[i], snap_conv[kk][i]);
    }
}

}  // namespace whirl::qwen35
