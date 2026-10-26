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

const char* kvModeName(KvMode k) {
    switch (k) {
        case KvMode::f16: return "f16";
        case KvMode::q8: return "q8";
        case KvMode::q8h: return "q8h";
        case KvMode::q8v: return "q8v";
        case KvMode::q4: return "q4";
        case KvMode::automatic: break;
    }
    return "automatic";
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
    WD_U(max_ctx);
    WD_U(max_batch);
    add("arch", hip::archName(m.arch));
    add("allocations", u(m.allocations.size()));
    WD_U(alloc_total);
    for (std::size_t t = 0; t < m.type_bytes.size(); ++t)
        if (m.type_bytes[t] != 0) add(std::format("type_bytes[{}]", t), u(m.type_bytes[t]));
    add("decode_graph", b(m.decode_graph != nullptr));
    WD_B(use_graph);
    WD_B(head_phase);
    WD_B(no_fuse);
    WD_U(small_max);
    WD_U(dbg_flags);
    add("tune_mask", std::format("{:#x}", m.tune_mask));
    WD_B(tune_cold);
    WD_B(naive_attn);
    WD_B(attn_wide);
    WD_B(attn_dq4);
    WD_B(dq4_dyn);
    WD_B(attn_ws4);
    WD_B(wide_verify);
    WD_B(moe_wide);
    WD_U(head_chunk);
    WD_B(float_gemv);
    WD_B(prec_dec);
    WD_U(gemv_max);
    add("gemv_r", hexRow(m.gemv_r));
    for (std::size_t t = 0; t < m.gemv_w.size(); ++t) {
        bool any = false;
        for (std::uint8_t v : m.gemv_w[t]) any = any || v != 0;
        if (any) add(std::format("gemv_w[{}]", t), hexRow(m.gemv_w[t]));
    }
    add("gemv_w_head", hexRow(m.gemv_w_head));
    WD_B(fp8_prefill);
    WD_U(fp8_mask);
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
    add("relax.on", b(m.relax.on));
    add("relax.k", u(m.relax.k));
    add("relax.alpha", f(m.relax.alpha));
    add("relax.eps", f(m.relax.eps));
    add("relax.delta", f(m.relax.delta));
    add("kv_choice.kv", numerics::kvKindName(m.kv_choice.kv));
    add("kv_choice.debug", b(m.kv_choice.debug));
    add("kv_choice.why", m.kv_choice.why);
    WD_U(gc_w_bytes);
    WD_U(gc_u_bytes);
    WD_U(gc_m_bytes);
    WD_B(gdn_chunked);
    WD_B(gdn_wmma);
    WD_B(gdn_strix);
    WD_U(gdn_strix_min);
    WD_B(act_fuse);
    WD_B(attn_kx_on);
    WD_B(gdn_ba_on);
    WD_B(gdn_in2_on);
    WD_U(x16_bytes);
    WD_B(attn_dqf_on);
    WD_U(attn_seg);
    WD_U(dq_rows);
    WD_B(attn_kg_on);
    WD_B(ffn_h16);
    WD_B(h16_dq);
    WD_B(out_h16);
    WD_B(g8t);
    if (m.g8t_v != 0) add("g8t_v", std::to_string(m.g8t_v));  // only when WHIRL_GEMM8T_V picked a variant (default dump unchanged)
    WD_B(gemmh_on);
    WD_B(gemmhq_on);
    WD_B(kv_q8);
    WD_B(kv_rot);
    WD_B(kv_kf16);
    WD_B(kv_q4);
    WD_U(fd_splits);
    add("kv_mode", kvModeName(m.kv_mode));
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
    WD_U(draft_window);
    WD_U(draft_window_min);
    add("draft_p_min", f(m.draft_p_min));
    WD_U(draft_n_min);
    add("draft_head", b(m.draft_head.has_value()));
    add("draft_d2", b(m.draft_d2.has_value()));
    WD_U(draft_rows);
    add("draft_map", b(m.draft_map.has_value()));
    add("dflash", b(m.dflash != nullptr));
    WD_B(mtp_skipped);
    WD_U(moe_bn_force);
    WD_U(ff_scratch);
    WD_B(moe_fp8);
    WD_B(moe_mxw);
    WD_B(moe_rbf);
    WD_B(mrope_on);
    WD_B(rpos_rows);
#undef WD_B
#undef WD_U
    add("gv_group", b(gv_group));
    add("gv_nmax", u(gv_nmax));
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
