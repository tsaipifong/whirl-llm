// WHIRL_DUMP_EXEC (arch025-S0a, see whirl/dump.h): the Model's numerics and execution flags after load + tune.
// SPDX-License-Identifier: Apache-2.0
//
// One "D exec: key=value" line per member, in declaration order of include/whirl/model.h. Listed: every scalar
// configuration member that load / loadOrTune sets and the forward reads. Not listed (per-call state that a forward
// sets and resets, the same after every load): row_n, row_pos, row_base, fwd_pos0, xq_n, x16_count, x16_fp8,
// cur_seq, kv_off, ctl_off, cap_rows, cap_n, n_inj, attn_win, rdelta, the device pointers and the kernel table (the
// launch log pins the kernels). tools/gates/exec_dump_check.py keeps the two lists complete against model.h.

#include "whirl/dump.h"
#include "whirl/model.h"

#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>

namespace whirl::qwen35 {

namespace {

std::string hexRow(const GemvR& r) {
    std::string s;
    for (std::uint8_t v : r) s += std::format("{:x}", v);
    return s;
}

std::uint64_t fnv(std::string_view s) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    return h;
}

}  // namespace

std::string execDump(const Model& m) {
    std::string s;
    auto add = [&s](std::string_view k, const std::string& v) { s += std::format("D exec: {}={}\n", k, v); };
    auto b = [](bool v) { return std::string(v ? "1" : "0"); };
    auto u = [](std::uint64_t v) { return std::to_string(v); };
    auto f = [](float v) { return std::format("{:.9g}", v); };
#define WD_B(x) add(#x, b(m.x))
#define WD_U(x) add(#x, u(m.x))
// arch025-N2b: the execution flags are the plan's (Model::nf()); same keys and values
#define WD_FB(x) add(#x, b(m.nf().x))
#define WD_FU(x) add(#x, u(m.nf().x))
    WD_U(max_ctx);
    WD_U(max_batch);
    add("arch", hip::archName(m.arch));
    add("allocations", u(m.allocations.size()));
    WD_U(alloc_total);
    for (std::size_t t = 0; t < m.type_bytes.size(); ++t)
        if (m.type_bytes[t] != 0) add(std::format("type_bytes[{}]", t), u(m.type_bytes[t]));
    add("decode_graph", b(m.decode_graph != nullptr));
    WD_FB(use_graph);
    WD_B(head_phase);
    WD_FB(no_fuse);
    WD_U(small_max);
    WD_FU(dbg_flags);
    add("tune_mask", std::format("{:#x}", m.tune_mask));
    WD_B(tune_cold);
    WD_FB(naive_attn);
    WD_FB(attn_wide);
    WD_FB(attn_dq4);
    WD_FB(dq4_dyn);
    WD_FB(attn_ws4);
    WD_FB(wide_verify);
    WD_FB(moe_wide);
    WD_FU(head_chunk);
    WD_FB(float_gemv);
    WD_FB(prec_dec);
    WD_FU(gemv_max);
    add("gemv_r", hexRow(m.gemv_r));
    for (std::size_t t = 0; t < m.gemv_w.size(); ++t) {
        bool any = false;
        for (std::uint8_t v : m.gemv_w[t]) any = any || v != 0;
        if (any) add(std::format("gemv_w[{}]", t), hexRow(m.gemv_w[t]));
    }
    add("gemv_w_head", hexRow(m.gemv_w_head));
    WD_FB(fp8_prefill);
    WD_FU(fp8_mask);
    WD_U(mm_class);
    WD_U(mx_fold_lossy);
    add("num_req.mode", numerics::modeName(m.num_req.mode));
    add("num_req.items", std::format("{:#x}", m.num_req.items));
    add("num_req.custom", b(m.num_req.custom));
    add("num_req.source", m.num_req.source);
    add("num_req.moved", std::format("{:#x}", m.num_req.moved));
    add("num_plan", numerics::logLine(m.num_plan));
    add("num_plan.q8dec", b(m.num_plan.q8dec));
    add("num_plan.q8dec_default", b(m.num_plan.q8dec_default));
    add("num_plan.lossy_override", b(m.num_plan.lossy_override));
    for (std::size_t i = 0; i < m.num_plan.overrides.size(); ++i) add(std::format("num_plan.override[{}]", i), m.num_plan.overrides[i]);
    add("relax.on", b(m.relax().on));
    add("relax.k", u(m.relax().k));
    add("relax.alpha", f(m.relax().alpha));
    add("relax.eps", f(m.relax().eps));
    add("relax.delta", f(m.relax().delta));
    add("kv_choice.kv", kvFormatName(m.kv_choice.kv));
    add("kv_choice.debug", b(m.kv_choice.debug));
    add("kv_choice.why", m.kv_choice.why);
    WD_U(gc_w_bytes);
    WD_U(gc_u_bytes);
    WD_U(gc_m_bytes);
    WD_FB(gdn_chunked);
    WD_FB(gdn_wmma);
    WD_FB(gdn_strix);
    WD_FU(gdn_strix_min);
    WD_FB(act_fuse);
    WD_FB(attn_kx_on);
    WD_FB(gdn_ba_on);
    WD_FB(gdn_in2_on);
    WD_U(x16_bytes);
    WD_FB(attn_dqf_on);
    WD_FU(attn_seg);
    WD_U(dq_rows);
    WD_FB(attn_kg_on);
    WD_FB(ffn_h16);
    WD_FB(h16_dq);
    WD_B(out_h16);
    WD_FB(g8t);
    WD_FB(gemmh_on);
    WD_FB(gemmhq_on);
    // arch025-N3: the four former KV bools, now properties of m.kv_fmt (same keys and values)
    add("kv_q8", b(kvQuant(m.kv_fmt)));
    add("kv_rot", b(kvRotated(m.kv_fmt)));
    add("kv_kf16", b(kvKf16(m.kv_fmt)));
    add("kv_q4", b(kvIs4bit(m.kv_fmt)));
    WD_U(fd_splits);
    add("kv_mode", m.kv_req ? kvFormatName(*m.kv_req) : "automatic");
    WD_U(pool_pages);
    WD_U(seq_pages);
    WD_U(snap_sets);
    add("seqs", u(m.seqs.size()));
    WD_B(gdn_replay);
    WD_B(mtp_batch);
    add("mtp", b(m.mtp.has_value()));
    WD_B(keep_hidden);
    WD_B(all_logits);
    WD_U(snap_rows);
    WD_U(draft_vocab);
    WD_FU(draft_window);
    WD_FU(draft_window_min);
    add("draft_p_min", f(m.draft_p_min));
    WD_U(draft_n_min);
    add("draft_head", b(m.draft_head.has_value()));
    add("draft_d2", b(m.draft_d2.has_value()));
    WD_U(draft_rows);
    add("draft_map", b(m.draft_map.has_value()));
    add("dflash", b(m.dflash != nullptr));
    WD_B(mtp_skipped);
    WD_FU(moe_bn_force);
    WD_U(ff_scratch);
    WD_FB(moe_fp8);
    WD_FB(moe_mxw);
    WD_FB(moe_rbf);
    WD_B(mrope_on);
    WD_B(rpos_rows);
#undef WD_B
#undef WD_U
#undef WD_FB
#undef WD_FU
    add("gv_group", b(m.nf().gv_group));
    add("gv_nmax", u(m.nf().gv_nmax));
    const std::string tune = m.writeTune();
    add("tune", std::format("{:016x} ({} bytes)", fnv(tune), tune.size()));
    return s;
}

void afterLoadTune(const Model& m) {
    if (dump::envOn("DUMP_EXEC")) {
        const std::string s = execDump(m);
        std::fwrite(s.data(), 1, s.size(), stderr);
        std::fflush(stderr);
    }
    if (dump::dumpOnly()) {
        std::fprintf(stderr, "D exec: --dump-only: exit after load + tune\n");
        std::fflush(stderr);
        std::fflush(stdout);
        std::exit(0);
    }
}

}  // namespace whirl::qwen35
