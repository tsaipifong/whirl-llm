// Model forward, the drivers (arch025-F4, moved from src/model/forward.cpp): Model::forward, the device-resident
// decode step (decodeStep / decodeStepLaunches / beginDecode), the layer loop Model::run and the prefill scratch
// lent to the vision encoder (lendScratch). Reimplements the WHIRL Zig research prototype's model/qwen35.zig.
// SPDX-License-Identifier: Apache-2.0

#include "../forward_internal.h"

namespace whirl::qwen35 {

void Model::forward(std::span<const u32> tokens, u32 pos0) {
    const u32 n = static_cast<u32>(tokens.size());
    if (n == 0 || n > max_batch) throw ModelError("BatchTooLarge");
    xq_src = 0;
    x16_src = 0;
    if (pos0 + n > max_ctx) throw ModelError("ContextTooLong");
    setTokens(ids, tokens, pos0);
    fwd_pos0 = pos0;
    run(n);
}

// Decode one token entirely from device state: the input token is ids[0] and
// its position pos_buf[0] (both written by the previous argmax), so the launch
// sequence is identical every step and never waits on the host.
void Model::decodeStep() {
    if (!use_graph || prof != nullptr) return decodeStepLaunches();
    if (!decode_graph) {
        hip::Graph::beginCapture(stream);
        try {
            decodeStepLaunches();
        } catch (...) {
            try {
                (void)hip::Graph::endCapture(stream);
            } catch (...) {
            }
            throw;
        }
        decode_graph = std::make_unique<hip::Graph>(hip::Graph::endCapture(stream));
    }
    decode_graph->launch(stream);
}

void Model::decodeStepLaunches() {
    // device positions (argmax advances pos_buf): the host row positions are not known here
    row_n = 0;
    // vision: device positions (argmax advances pos_buf): RoPE position = pos - delta
    visBegin();
    if (const VisMap* vm = visOf(cur_seq)) {
        if (k.attn_prep_m == nullptr) throw ModelError("MropeUnsupported");
        mrope_on = true;
        rdelta = vm->deltaEnd();
    }
    run(1);
    hip::launch(k.argmax, D(1), D(1024), 0, stream, logits, I(cfg.n_vocab), out_tok, ids, pos_buf, i32(1));
}

// Seed device state for decodeStep after a prefill of n_past tokens.
u32 Model::beginDecode(u32 n_past) {
    const i32 p = static_cast<i32>(n_past);
    hip::upload(pos_buf, &p, 4);
    hip::launch(k.argmax, D(1), D(1024), 0, stream, logits, I(cfg.n_vocab), out_tok, ids, pos_buf, i32(0));
    i32 out = 0;
    hip::download(&out, out_tok, 4);
    return static_cast<u32>(out);
}

void Model::run(u32 n) {
    if (gdn_replay && bplan == nullptr) {
        if (!psegs.empty()) {
            for (const PRows& ps : psegs) commitSeq(ps.seq);
        } else {
            commitSeq(cur_seq);
        }
    }
    xq_src = 0;
    x16_src = 0;
    hip::launch(k.get_rows[ti(tok_embd.ty)], D(8, n), D(128), 0, stream, tok_embd.ptr, tok_embd.row_bytes, ids, x, I(cfg.n_embd));
    if (n_inj > 0) injectRows();
    mark(OpClass::embed);

    const u32 E = cfg.n_embd;
    // DFlash2 drafter features: x at the input of its target layers (a copy on the stream; the
    // trunk's arithmetic is untouched)
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const Layer& L = layers[i];
        bool gdn_in2 = false;  // x16b holds this layer's normed input as f16 for [beta; alpha]
        if (cap_buf) captureLayerInput(i, n);
        if (fused(n)) {
            rmsnormQ8(x, L.attn_norm, h, n);
        } else if (L.kind == LayerKind::attn) {
            const Mat c[3] = {L.attn.q, L.attn.k, L.attn.v};
            rmsnormIn(x, L.attn_norm, h, n, c, 1);
        } else if (gdnIn2(L.gdn, n)) {
            // one norm: fp8 (+ sx8) into x16 for qkv / gate (registered as h's
            // conversion; h itself is left unwritten) and f16 into x16b for [beta; alpha]
            if (x16bOff() + static_cast<u64>(n) * E * 2 > x16_bytes) throw ModelError("X16bOverflow");
            hip::launch(x8(k.rmsnorm_x8h16, k.rmsnorm_x8h16t), D(n), D(256), 0, stream, x, L.attn_norm, x16, sx8, x16 + x16bOff(), I(E),
                        cfg.eps);
            xq_src = 0;
            x16_src = h;
            x16_count = n * E;
            x16_fp8 = true;
            mark(OpClass::norm);
            gdn_in2 = true;
        } else {
            const Mat c[4] = {L.gdn.qkv, L.gdn.gate, L.gdn.beta, L.gdn.alpha};
            rmsnormIn(x, L.attn_norm, h, n, c, 1);
        }
        if (L.kind == LayerKind::attn) {
            attnBlock(L.attn, kvLayer(i), n);
        } else {
            gdnLayer(i, L.gdn, n, gdn_in2);
        }
        if (L.moe)
            moeBlock(L.post_norm, L.ffn_gate, L.ffn_up, L.ffn_down, *L.moe, n);
        else
            ffnBlock(L.post_norm, L.ffn_gate, L.ffn_up, L.ffn_down, n);
    }
    if (cap_buf) captureLayerInput(layers.size(), n);  // id n_layer: the last layer's output
    if (keep_hidden) rmsnorm(x, output_norm, hn, E, n, E, E);
    if (!psegs.empty()) return;  // prefillSegs: per-segment logits from x
    if (keep_hidden && all_logits && n <= small_max) {
        head_phase = true;
        matmul(output, hn, logits, n, false);
        head_phase = false;
        return;
    }
    const DevPtr last = x + static_cast<u64>(n - 1) * E * 4;
    // FIX-NUM P2-1: every head input uses the 256-thread rmsnorm (the reduction order of the
    // verify / all-logits rows above, which the server's decode and the CLI's MTP / n-gram verify
    // take), never rmsnormQ8's 1024-thread block sum: plain decode, prompt-end logits and verify
    // rows then give the same logits bits for the same hidden row.
    rmsnorm(last, output_norm, h, E, 1, E, E);
    head_phase = true;
    matmul(output, h, logits, 1, false);
    head_phase = false;
}

// ---------------------------------------------------------------------------
// vision: lent prefill scratch, multi-section RoPE rows and image embedding rows
// (ported from the research prototype, item VIS)

std::size_t Model::lendScratch(std::span<std::array<u64, 2>> out) const {
    const u64 B = max_batch;
    const u64 f4 = 4;
    const u64 in_max = std::max<u64>(std::max<u64>(cfg.n_ff, 2ull * cfg.n_embd), std::max<u64>(cfg.d_inner, static_cast<u64>(cfg.n_head) * cfg.head_dim));
    u64 max_elems = 0;
    for (const Layer& L : layers) {
        const MatList all = layerMats(L);
        for (const Mat& w : all.slice()) max_elems = std::max<u64>(max_elems, static_cast<u64>(w.ncols) * w.nrows);
    }
    const u64 n_chunks = (B + 63) / 64;
    const std::array<u64, 2> list[] = {
        {ffn_g, B * ff_scratch * f4},
        {ffn_u, B * ff_scratch * f4},
        {qf, B * cfg.n_head * cfg.head_dim * 2 * f4},
        {w16, max_elems * 2},
        {qkv, B * cfg.convCh() * f4},
        {conv_out, B * cfg.convCh() * f4},
        {x16, B * std::max<u64>(in_max, cfg.n_embd) * 2},
        {z, B * cfg.d_inner * f4},
        {gdn_out, B * cfg.d_inner * f4},
        {attn_out, B * cfg.n_head * cfg.head_dim * f4},
        {gc_w, n_chunks * cfg.n_v_heads * 64 * 128 * 4},
        {gc_u, n_chunks * cfg.n_v_heads * 64 * 128 * 4},
    };
    std::size_t kk = 0;
    for (const auto& e : list) {
        if (kk == out.size()) break;
        if (e[0] == 0) continue;
        out[kk++] = e;
    }
    return kk;
}

}  // namespace whirl::qwen35
