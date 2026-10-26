// Model forward, prefill drivers (arch025-F4, moved from src/model/forward.cpp): prefill with MTP
// (prefillWithMtp / prefillMtpChunk), segmented multi-sequence prefill (canSegment / prefillSegs /
// segLogitsToFront), plain prefill, argmax and logits readback.
// SPDX-License-Identifier: Apache-2.0

#include "../forward_internal.h"

namespace whirl::qwen35 {

// ---------------------------------------------------------------------------
// prefill drivers

void Model::prefillWithMtp(std::span<const u32> tokens) {
    std::size_t off = 0;
    while (off < tokens.size()) {
        const std::size_t n = std::min<std::size_t>(tokens.size() - off, max_batch);
        prefillMtpChunk(tokens, 0, off, n, std::nullopt);
        off += n;
    }
}

// One chunk of an MTP prefill: tokens[off .. off + n] at positions pos0 + off
// .., then the MTP rows pairing each of their trunk hiddens with the following
// token. The row pairing a chunk's LAST hidden with the next token is not run
// in that chunk: the chunk's last hidden goes to mtp_h, and the next chunk runs
// that boundary row on its own (one row) before its forward, exactly as a
// prefix-cache resume at that position does (prev_hidden = the checkpoint's
// hidden). So the MTP KV (and every later draft) is bitwise the same whether
// the prompt was prefilled cold or resumed from a checkpoint at a chunk end:
// small-row MTP forwards (q8 activations, fused kernels) are not bitwise equal
// to the same row inside a larger batch. Exact acceptance never showed the
// difference; relaxacc (fast) keeps drafts, so the text depended on it (FIX-FS).
// The final chunk's last hidden stays in mtp_h for the first decode draft.
void Model::prefillMtpChunk(std::span<const u32> tokens, u32 pos0, std::size_t off, std::size_t n, std::optional<DevPtr> prev_hidden) {
    const u64 E = cfg.n_embd;
    if (off == 0 && prev_hidden) hip::copyAsync(mtp_h, *prev_hidden, E * 4, stream);
    // mtp_h: the previous chunk's last hidden (off > 0) or the checkpoint's (resume)
    if (off > 0 || prev_hidden) (void)mtpForward(mtp_h, tokens.subspan(off, 1), pos0 + static_cast<u32>(off), false);
    const u32 p = pos0 + static_cast<u32>(off);
    keep_hidden = true;
    try {
        forward(tokens.subspan(off, n), p);
    } catch (...) {
        keep_hidden = false;
        throw;
    }
    keep_hidden = false;
    if (n > 1) (void)mtpForward(hn, tokens.subspan(off + 1, n - 1), p + 1, false);
    hip::copyAsync(mtp_h, hn + (n - 1) * E * 4, E * 4, stream);
}

bool Model::canSegment() const {
    return seqs.size() > 1 && k.attn_prep != nullptr && !nf().no_fuse && !nf().naive_attn && cfg.head_dim == 256 && cfg.d_conv == 4;
}

// Several sequences' prefill chunks as one forward (each chunk longer than
// max_small_batch). Logits of every segment that ends its prompt land in
// logits row k; with mtp, also the MTP rows and mtp_h of each sequence.
void Model::prefillSegs(std::span<const PSeg> segs, bool with_mtp) {
    if (segs.empty() || segs.size() > max_small_batch) throw ModelError("BadBatch");
    const u64 E = cfg.n_embd;
    u32 total = 0;
    PRows rows[max_small_batch];
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const PSeg& sg = segs[kk];
        if (sg.n <= max_small_batch) throw ModelError("BadBatch");
        if (sg.pos0 + sg.off + sg.n > max_ctx) throw ModelError("ContextTooLong");
        rows[kk] = {sg.seq, total, static_cast<u32>(sg.n), sg.pos0 + static_cast<u32>(sg.off)};
        total += static_cast<u32>(sg.n);
    }
    if (total > max_batch) throw ModelError("BatchTooLarge");
    // boundary MTP rows, one row each as in prefillMtpChunk (solo order): a cached
    // prefix's last hidden (off 0) or the previous chunk's (in mtp_h) with tokens[off]
    if (with_mtp)
        for (const PSeg& sg : segs)
            if (sg.off > 0 || sg.prev_hidden) {
                selectSeq(sg.seq);
                if (sg.off == 0) hip::copyAsync(mtp_h, *sg.prev_hidden, E * 4, stream);
                (void)mtpForward(mtp_h, sg.tokens.subspan(sg.off, 1), sg.pos0 + static_cast<u32>(sg.off), false);
            }
    const std::size_t cap = host_ids.size() / 2;
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const PSeg& sg = segs[kk];
        for (std::size_t j = 0; j < sg.n; ++j) {
            host_ids[rows[kk].r0 + j] = lookupId(sg.tokens[sg.off + j]);
            host_ids[cap + rows[kk].r0 + j] = static_cast<i32>(sg.pos0 + sg.off + j);
        }
    }
    // vision: RoPE rows / image rows of the segments of sequences with images
    visBegin();
    if (visAny()) {
        bool any = false;
        for (const PSeg& sg : segs)
            if (visOf(sg.seq) != nullptr) any = true;
        if (any) {
            visEnsure();
            for (std::size_t kk = 0; kk < segs.size(); ++kk) {
                const PSeg& sg = segs[kk];
                for (std::size_t j = 0; j < sg.n; ++j)
                    visRow(visOf(sg.seq), rows[kk].r0 + static_cast<u32>(j), sg.pos0 + static_cast<u32>(sg.off + j));
            }
            visCommit(total);
        }
    }
    hip::upload(ids, host_ids.data(), static_cast<std::size_t>(total) * 4);
    hip::upload(pos_buf, host_ids.data() + cap, static_cast<std::size_t>(total) * 4);
    row_n = 0;
    xq_src = 0;
    x16_src = 0;
    psegs = std::span<const PRows>(rows, segs.size());
    keep_hidden = with_mtp;
    struct Reset {
        Model& m;
        ~Reset() {
            m.psegs = {};
            m.keep_hidden = false;
        }
    } reset_flags{*this};
    run(total);
    psegs = {};
    // logits of the segments that end their prompt (before the MTP block reuses x)
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const PSeg& sg = segs[kk];
        if (sg.off + sg.n != sg.tokens.size()) continue;
        const DevPtr last = x + (static_cast<u64>(rows[kk].r0) + rows[kk].n - 1) * E * 4;
        rmsnorm(last, output_norm, h, static_cast<u32>(E), 1, static_cast<u32>(E), static_cast<u32>(E));  // P2-1: as run()
        head_phase = true;
        matmul(output, h, logits + static_cast<u64>(kk) * cfg.n_vocab * 4, 1, false);
        head_phase = false;
    }
    if (!with_mtp) return;
    for (std::size_t kk = 0; kk < segs.size(); ++kk) {
        const PSeg& sg = segs[kk];
        selectSeq(sg.seq);
        const DevPtr hid = hn + static_cast<u64>(rows[kk].r0) * E * 4;
        const u32 p = sg.pos0 + static_cast<u32>(sg.off);
        // the row pairing the last hidden with the next token runs with the next chunk
        if (sg.n > 1) (void)mtpForward(hid, sg.tokens.subspan(sg.off + 1, sg.n - 1), p + 1, false);
        hip::copyAsync(mtp_h, hid + (sg.n - 1) * E * 4, E * 4, stream);
    }
}

void Model::segLogitsToFront(u32 kk) {
    if (kk == 0) return;
    const u64 V = cfg.n_vocab;
    hip::copyAsync(logits, logits + static_cast<u64>(kk) * V * 4, V * 4, stream);
}

void Model::prefill(std::span<const u32> tokens, u32 pos0) {
    std::size_t off = 0;
    while (off < tokens.size()) {
        const std::size_t n = std::min<std::size_t>(tokens.size() - off, max_batch);
        forward(tokens.subspan(off, n), pos0 + static_cast<u32>(off));
        off += n;
    }
}

u32 Model::argmax() {
    hip::launch(k.argmax, D(1), D(1024), 0, stream, logits, I(cfg.n_vocab), out_tok, ids, pos_buf, i32(0));
    i32 out = 0;
    hip::download(&out, out_tok, 4);
    return static_cast<u32>(out);
}

void Model::readLogits(std::span<float> dst) { hip::download(dst.data(), logits, dst.size() * 4); }

}  // namespace whirl::qwen35
