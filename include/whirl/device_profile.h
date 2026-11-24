// Device profile (ARCH-1 D-4): what the host code does differently per GPU
// family, as data. Every host-side difference between the R9700 (gfx1201)
// and the Radeon 8060S (gfx1151) outside src/hip reads a field here instead
// of testing hip::Arch; a new family (an RDNA 3 dGPU, gfx1200) is a new row.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. No HIP. The rows are constexpr data, so a field read
// costs what the arch comparison it replaces cost. The table test is
// tests/device/device_tests.cpp (testProfiles): change a value and it fails.
// Kernel availability is not here: it comes from the code object (null
// hip::Function, kernels::Caps); this table holds the host's choices.

#pragma once

#include "whirl/hip.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace whirl::device {

// v0252-8sq5 (PLAN V1-109): one step of the q4 decode / verify split-count table: from `keys` keys (the longest row) on,
// `splits` splits, until the next step. splits 0 = an unused slot (the tail of a class)
struct Dq4SplitStep {
    std::uint32_t keys;
    std::uint16_t splits;
};
inline constexpr std::size_t kDq4SplitClasses = 4, kDq4SplitSteps = 8;
// the table per row class: [0] one row (plain decode), [1] two verify rows of one sequence at consecutive positions (1 MTP
// draft per cycle), [2] three such rows (2 drafts), [3] the MTP draft row with its attention window active. Steps ascending in
// keys; below the first step and for any other row set (4+ rows, rows of several sequences, a window with several rows) the
// table gives nothing and ops keeps today's rule (ops::dq4TableSplits)
struct Dq4SplitTable {
    bool on;
    std::array<std::array<Dq4SplitStep, kDq4SplitSteps>, kDq4SplitClasses> steps;
};
// v0252-8sws round 3 (PLAN V1-88 on gfx1151): the measured limits of the attn_ws4 / attn_ws4g split constants besides 64:
// max_keys[i][c] = the largest key count (the step's largest n_pos) at which split constant kernels::kWs4SplitCounts[i]
// (0: 8, 1: 16) is launched for row class c (0 = the MTP draft row: one row with the draft window argument, 1..9 = a step of
// that many rows of one sequence at consecutive positions); 0 = not measured. Above the limit, for other classes (10+ rows,
// several sequences, gapped rows, rows the host does not know) and when !on: 64 (today's kernels and bits). ops::ws4SplitForStep.
struct Ws4SplitLimits {
    bool on;
    std::uint32_t max_keys[2][10];
};

struct Profile {
    hip::Arch arch;
    // ---- memory
    // memory shared with the CPU (UMA iGPU): the load does not check the weights against the free VRAM,
    // which does not bound what it can allocate (loader.cpp; WHIRL_VRAM_LIMIT_MB still applies)
    bool uma_no_vram_fit_check;
    // ---- GEMV / GEMM dispatch (src/model/qwen35/forward/dispatch.cpp)
    // rows per block of the multi-row quantized GEMV per batch size 0..16 (Model::gemv_r default)
    std::array<std::uint8_t, 17> gemv_r;
    // gfx1151 MXFP4 gemvk variant (kernels/gfx1151/gemv_gk.hip, 1..9) per verify width 0..16, 0 = the gemvq path
    // (v0251-a6port; qwen35::defaultGemvK picks the row of the model class, WHIRL_GEMV_K replaces it): dense / MoE
    std::array<std::uint8_t, 17> gemvk_dense, gemvk_moe;
    // f16-in GEMV grid: kernels::gemvhGrid(nrows) blocks (else one block per 16 rows)
    bool gemvh_grid_by_rows;
    // v0252-8sdec (PLAN V1-142): precise decode / verify runs the same-input matrices of Context::matmulGroup as one
    // grouped gemvh launch (gemvhg_<T>, same bits) on dense models: numerics flag gemvh_grp's default (WHIRL_GEMVH_GRP
    // overrides; ops::chooseGemvhGroup also needs the kernel in the code object, gfx1151 only)
    bool gemvh_group_dense;
    // f16 GEMM outputs (h16) are decided from the tuner's candidate list, not the tuned choices (the code
    // object has no h16 fused / dequant GEMM; GEMVH-2)
    bool h16_from_candidates;
    // small-batch WMMA GEMM selection: ties between row blocks go to the larger BM (GEMVH-2)
    bool gemms_tie_larger_bm;
    // fused-dequant MXFP4 prefill GEMM (gemmfq, v0251-a3port): the default slot of ops::GemmFlags::fdq for an MXFP4
    // dequant choice at >= WHIRL_FDQ8S_MIN rows (1 c2, 2 t2 = kernels::kGemmFqNames[slot - 1]; 0 = the old
    // dequant_f16 + gemm_c<i>_f16); WHIRL_FDQ8S overrides it. Only the gfx1151 code object has the kernels, so a
    // slot here is also never selected on a family whose object lacks them (bitwise either way)
    std::uint8_t fdq_default;
    // v0251-a3fix: the slot from ops::GemmFlags::fdq_big_min (2305) rows (same encoding; WHIRL_FDQ8S overrides it too)
    std::uint8_t fdq_big;
    // v0252-8sbadopt (PLAN V1-53): the gemmfq slots that run their grouped-raster twin (kernels::MatmulKernels::gemmfq_grp,
    // same tile, grid and bits) per model class, bit (slot - 1) in the fdq_default encoding (1 = c2, 2 = t2);
    // Context::gemmFlags -> ops::GemmFlags::fq_grp. A family whose object lacks the twins runs its slots unchanged
    std::uint8_t fq_grp_dense, fq_grp_moe;
    // q4 KV decode through attn_dq4: the smallest split count with the same key partition (FAST-8S-1)
    bool attn_dq4_dynamic_splits;
    // v0252-ws4r97b (PLAN V1-87 phase 2): the 4-wave pipelined decode / verify attention attn_ws4r1 / attn_ws4r2
    // (kernels/attn_ws4r.hip, gfx1201 object only) in place of attn_wsplit1 / attn_wsplit2 after ops::chooseAttnDecode's
    // wsplit choice (same arguments, groups, grid, 128 threads and partial bits: kt ws4r), per KV format: q8 = q8 / q8h
    // (the _q8 kernels), f16 = f16. 0 = attn_wsplit (off), 1 = plain (LDS-scoped fence + s_barrier), 2 = b (inline-asm
    // s_wait_dscnt 0 + s_barrier_signal / wait). WHIRL_ATTN_WS4R=0 / 1 / 2 overrides both (numerics flag attn_ws4r).
    std::uint8_t attn_ws4r_q8, attn_ws4r_f16;
    // v0252-l2split (PLAN V1-88): the machine-fill block target of the f16 / q8 / q8h decode / verify attention grid
    // (attn_wsplit / attn_ws4r path; never q4, never the gfx1151 attn_ws4 path): split count = clamp(blocks / n_head_kv, 2,
    // today's) by ops::lsplitCount, the same for every row of a model (a row's key partition stays a function of its own
    // position: seqtest / MTP invariance). Balance / fast only (numerics flag attn_lsplit: precise 0); WHIRL_ATTN_LSPLIT =
    // auto / 0 / N overrides it. 0 = today's split count. gfx1201: 32 WGPs x 2 attn_ws4r blocks (LDS 47.8 / 58.4 KB of
    // 128 KB per WGP) = 64 = one machine wave (libr4d decode_splits / TRT-LLM MMHA law, progress/v0252-l2split.PRIOR_ART.md)
    std::uint16_t attn_lsplit_blocks;
    // v0252-8sws (PLAN V1-88 on gfx1151): the same machine-fill rule for the gfx1151 attn_ws4 / attn_ws4g decode / verify
    // attention with q8 / q8h KV: split count = ops::lsplitCount(64, n_head_kv, flag attn_lsplit, this field), launched only
    // where the code object has that split constant (kernels::kWs4SplitCounts: attn_ws4_q8_sp<S>, attn_ws4g<cg>_q8_sp<S>),
    // else today's 64 (ops::ws4SplitCount). A separate field, not attn_lsplit_blocks: on gfx1151 that one also drives the
    // attn_wsplit fallback (WHIRL_ATTN_WS4=0), which was not measured and keeps 64. Balance / fast only (flag attn_lsplit:
    // precise 0); f16 KV keeps 64 (no f16 _sp entries; precise's kernels never change). 0 = today's 64. gfx1151: 32 = the
    // measured best count x KV heads (kernel bench ws4s, 8060S 2026-11-24 19:10, q8h, B/A vs 64 +-0.15 %: Ornith 2 KV heads ->
    // 16 (0.63 .. 0.89 at 8192 .. 131072), Swift 4 -> 8 (0.71 .. 0.94; the one exception Swift 3 rows at 131072: S 8 1.017
    // [1.015, 1.019], 64 best there)); gfx1201 0 (no attn_ws4 path)
    std::uint16_t attn_ws4_split_blocks;
    // v0252-8sws round 3: where the split constant of attn_ws4_split_blocks is used (flag attn_lsplit auto only; a forced
    // WHIRL_ATTN_LSPLIT=N skips it): the largest MEASURED key count with a proven gain over 64 per row class (kernel bench ws4s
    // TABLE runs, 8060S 2026-11-25 01:24-01:43, B/A whole 95 % interval < 1 at that key count and every smaller measured one;
    // between that point and the next measured one nothing is known -> 64; above 262144 (the largest measured) -> 64)
    Ws4SplitLimits ws4_split_limits;
    // v0252-8sq4 (PLAN V1-140 / V1-109; fast only, opt-in until the main session's screen shows a gain beyond its error,
    // ADR-0019; a default flip is one line here plus the pinned CHECKs in tests/device, tests/iso/device.profile,
    // tests/ops testAttnDq4F8 and tests/unit testDq4F8Flags' shipped pin). v0252-8sflip2: attn_dq4_row_order is the gfx1151
    // default; dq4_min_keys and fast_kv_q8h_moe_ctx stay opt-in:
    //   attn_dq4_row_order (F8 order): q4 KV decode / verify runs attn_dq4o (gfx1151 object: attn_dq4 with the gfx1201 block
    //     order, verify row fastest, so the MTP verify rows share the q4 K / V tiles through L2; same partials). Numerics
    //     flag dq4_order (numerics::Target::dq4_order_profile); WHIRL_DQ4_ORDER = 0 off / any other value on. A family whose
    //     object has no attn_dq4o (gfx1201: its attn_dq4 has the order already) runs attn_dq4 whatever the row says
    //   dq4_min_keys (F8 split): the attn_dq4 split count from the longest row and this keys-per-split target
    //     (ops::dq4MinKeysSplits; 0 = today's rule, dq4Splits). Flag dq4_min_keys; WHIRL_DQ4_MINKEYS = 0 off / N keys.
    //     A host rule: the R9700 takes it later by its own row value
    //   fast_kv_q8h_moe_ctx (fallback): fast mode on a MoE model takes q8h KV (balance's format and decode / verify path,
    //     KvCaps::fast_q8h_moe; the 8060S dense rule RC6 extended to MoE) when the load's per-request context is at least
    //     this many tokens (1 = every load); 0 = off (q4). Fixed per KV pool at load (Model::load), never per session or
    //     per length. numerics::fastKvQ8hMoe; WHIRL_FAST_KVQ8H = 0 off / N tokens. Only where KvCaps::fast_q8h holds
    //     (gfx1151): the R9700 row is inert until its code path is chosen there
    bool attn_dq4_row_order;
    std::uint16_t dq4_min_keys;
    std::uint32_t fast_kv_q8h_moe_ctx;
    // v0252-8sq5 (PLAN V1-109; fast only: q4 KV): the attn_dq4 / attn_dq4o split count per row class and key count, measured
    // by the kernel bench whirl-kernel-test --bench dq4s (in-object ABBA pairs against today's ops::dq4Splits count). Read by
    // ops::dq4TableSplits when the numerics flag dq4_splits is auto (the fast default); WHIRL_DQ4_SPLITS=0 = today's rule,
    // =N = N splits. WHIRL_DQ4_MINKEYS (when set) still wins. on = false: today's rule
    Dq4SplitTable dq4_split_table;
    // ---- tuning (src/model/tune.cpp)
    bool tune_fast;                 // each kernel timed once per shape, slow candidates rejected early
    // tune cache file name suffix (table revision). A change to what the tuner times on this family (candidates, slots,
    // thresholds: qwen35::tuneCandidateKey) needs a new revision: tests/model testTuneCacheKey pins each one
    const char* tune_cache_suffix;
    const char* tune_time_hint;     // first-run message: how long the tune takes
    // f16-WMMA chunked DeltaNet prefill from strix-llama.cpp (GDN-8S; where gdn_wmma's kernels do not exist)
    bool gdn_strix;
    // the numerics plan's 8060S rows (numerics::Target::gfx1151 / KvCaps::fast_q8h): MXFP4-only relaxed
    // items, fast q8h KV on dense models, the 8060S reason texts
    bool numerics_8060s;
    // ---- MTP / prefill schedule
    // a verify row costs far more (UMA): the 8060S draft-count / cost-model defaults (qwen35::mtpDefaults)
    bool mtp_costly_verify_rows;
    // the embedded 64k MTP draft-head vocabulary subset is the default for MoE models (Ornith) on this family too
    // (spec.policy draftVocabDefaultFits moe_ok; dense qwen35 models have it on every family). Drafts only: the
    // verify is unchanged, so the output is the same (v0252-n2 / feat027-draft64k97 b334a52, DISCOVERY v0252 N1)
    bool draft64k_moe;
    // v0252-f2def2: relaxacc greedy uses F1's one-pass check (argmax_rank_rows_to ranks the MTP drafts in the verify's
    // argmax; no topk_rows launch) on this family. Both code objects have the kernel. On for the R9700 (ADR-0015 G3a) and,
    // since v0252-f1flip (ADR-0015 Revision 1, decided 11-10 05:09 by the per-cycle-cost rule), the 8060S: the pre-registered
    // speed test ended INCONCLUSIVE twice with no loss on any key (Ornith fast tg +1.65..+2.43 %, Swift +0.3..+0.7 %) and the
    // fast outputs were identical. Read by qwen35::numericsTarget (Target::k_relax_rank: the R9700's MoE fast arm needs it;
    // the 8060S arm k6a05 does not) and Context::setRelaxRank / relaxRankEnqueue (server and CLI decode)
    bool relax_rank_check;
    // v0252-f1arg3: the MTP / batched verify argmax runs the multi-block twins argmax_rows_to_mb / argmax_rank_rows_to_mb
    // (kernels/sample_argmax_mb.hip: AM_TILES(n_vocab) x rows blocks, bitwise the one-block argmax_rows(_to) /
    // argmax_rank_rows_to) when the code object has both; WHIRL_ARGMAX_MB (0 off, 1 on, 2 on + workspace tripwire)
    // overrides it, read once per load (Context::setArgmaxMb). The workspace is per Context (= per stream).
    // v0252-b5 round 3 (PLAN V1-117): one value per model class (dense / MoE), as e1_dense / e1_moe; the row's value is
    // the default in FAST mode only (Context::argmaxMbMode applies the mode: precise / balance were not measured for a gain)
    bool argmax_mb_dense, argmax_mb_moe;
    // v0252-strixc2 (PLAN V1-09): balance-mode prefill attention attn_sx<NP> (strix-llama.cpp fattn-rdna35, gfx1151 code
    // object only; not bitwise, so the numerics plan takes it in balance only: numerics::Target::attn_sx_profile)
    bool attn_sx_balance;
    // v0252-8sfp (PLAN V1-144): fast-mode prefill attention with int4 Q.K^T (attn_prefill_wmma_i4 / _s_i4: gfx1151 iu4 WMMA,
    // 2x its f16 rate; the gfx1201 object has no such kernel); not bitwise, so the numerics plan takes it in fast only
    // (numerics::Target::attn_i4qk_profile); WHIRL_ATTN_I4QK=0 turns it off
    bool attn_i4qk_fast;
    // v0252-8sattn3 (PLAN V1-148): the second form of that int4 kernel (attn_prefill_wmma_i4p / _s_i4p: int4 Q codes in LDS +
    // K / V register prefetch, bit-identical to attn_prefill_wmma_i4 / _s_i4; gfx1151 object only), taken only where
    // attn_i4qk is on (numerics::Target::attn_i4pf_profile, flag attn_i4pf); WHIRL_ATTN_I4PF: "0" / "" off, other on;
    // round 3: on in the gfx1151 row (fast default), off in the gfx1201 row (no such kernel there)
    bool attn_i4_qlds_pf;
    std::size_t prefill_exec_moe;  // default forward rows of a MoE solo prefill (prefillExecDefault)
    bool prefill_fill_batch;       // forwards fill whole chunks up to the batch (prefillExecFor fill, SRV-PP)
    // v0252-n10: whirl-server writes the TAIL-CK checkpoint inside the solo prefill forward that contains the tail point
    // (sched::tailInside, EngineOptions::tail_ck_in_forward) instead of ending a forward there; the CLI keeps its forward
    // end (same bits: docs/guide/en/kv-and-caching.md section 7, progress/v0252-n10.DESIGN.md). Both rows on since
    // v0252-tailck8s (the model still refuses where it cannot capture: Context::stateCaptureOk); WHIRL_TAIL_CK_IN_FWD
    // (settings::tailCkInForward) overrides the row in whirl-server
    bool tail_ck_in_forward;
    // v0252-e1def (PLAN V1-93): the WHIRL_E1 value whirl-server starts with when the variable is not set, per model class
    // (dense / MoE; qwen35::e1Default picks the row's value, server::chooseE1 applies it only where a cycle can use it;
    // WHIRL_E1 = 0 / 1 / 2 always wins). 0 = the host-input cycle, 2 = the E1 one-cycle lookahead (EngineOptions::e1).
    // The lookahead changes when a cycle is submitted, never what it computes: the reply is the same in the same mode
    std::uint8_t e1_dense, e1_moe;
    // v0252-pp2k2 (PLAN V1-139, F1): ops::GemmFlags::gemm_tail default - a gemmh / gemmhq f16 GEMM whose rows are just past a
    // 256 multiple runs its tail rows on the narrowest gemms tile (ops::planGemmF16; same bits). WHIRL_GEMM_TAIL=0 / 1
    // overrides it. gfx1201 only: the gfx1151 prefill GEMMs are gemmfq with other grids (physical-isolation rule)
    bool gemm_tail;
    // v0252-8smoe (A, PLAN V1-95): the MoE prefill combine (ops::MoeFlags::moep_comb) of a FAST plan that runs moeskip with
    // MXFP4 routed experts when WHIRL_MOEP_COMB is not set (numerics::Target::moep_comb_fast_skip_profile, read by
    // moepCombMoeSkip); 0 = the family's non-f16 combine (gfx1151 kMoepDefault.comb 1, gfx1201 0). 5 = the atomic scatter
    // fused into the down projection (gemm_moepda_*; f32 add order varies run to run, fast only). WHIRL_MOEP_COMB=1 = the
    // 0.2.5.1 arm. Fast without moeskip keeps its own rule (moepCombDefault)
    std::uint8_t moep_comb_fast_skip;
    // v0252-8smoe (B, PLAN V1-147): the routing-aware gate + up tile of the moep MoE prefill (llama.cpp mmq.cu:197-201: the
    // tile against the AVERAGE tokens per expert on RDNA3 / 3.5 / 4): kernels::kMoepGu index + 1 of the wide variant a
    // forward takes when its experts average at least that variant's token tile (n * K >= bn * R; ops::chooseMoe), else
    // the plan's moep_gu. 0 = off. Every moep gate + up variant stores the same bits (kt moemx), so it is bitwise in
    // every mode; the default applies in balance / fast (numerics::Target::moep_guw_profile; precise keeps its path until
    // its own screen arm). WHIRL_MOEP_GUW = a variant name / 0 overrides it
    std::uint8_t moep_guw;
    // v0252-8smoe2 (PLAN V1-147; opt-in until a screen shows a gain beyond its error; v0252-8sflip: C and E on in the
    // gfx1151 row, D still opt-in):
    //   moep_dhs (C): the coalesced f16 down epilogue gemm_moepdhs_* where the plan stores f16 expert outputs (balance /
    //     fast; same bits as gemm_moepdh); numerics::Target::moep_dhs_profile; WHIRL_MOEP_DHS=1 / 0 overrides it
    //   moep_dna (D): kernels::kMoepDn index + 1 of the atomic-scatter (COMB 5) down variant (fast; 4 = m128w8n128k64);
    //     numerics::Target::moep_dna_profile; WHIRL_MOEP_DNA = a variant name / 0 overrides it
    //   moep_fq (E): the fused-dequant moep twins gemm_moepfq*_* (every mode; same bits as the base kernels);
    //     numerics::Target::moep_fq_profile; WHIRL_MOEP_FQ=1 / 0 overrides it
    bool moep_dhs;
    std::uint8_t moep_dna;
    bool moep_fq;
    // ---- vision
    unsigned vision_attn_threads;  // kernels/vision/vision.hip vis_attn2 block size of this code object
};

inline constexpr Profile k_profile_gfx1201{
    .arch = hip::Arch::gfx1201,
    .uma_no_vram_fit_check = false,
    .gemv_r = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
    .gemvk_dense = {},  // the gfx1201 object has no gemvk
    .gemvk_moe = {},
    .gemvh_grid_by_rows = false,
    .gemvh_group_dense = false,  // v0252-8sdec: the gfx1201 object has no gemvhg (unchanged)
    .h16_from_candidates = false,
    .gemms_tie_larger_bm = false,
    .fdq_default = 0,
    .fdq_big = 0,
    .fq_grp_dense = 0,  // v0252-8sbadopt: the gfx1201 object has no gemmfq
    .fq_grp_moe = 0,
    // v0252-n2: FAST-8S-1 item 1 on the R9700 (same shared attn_dq4 / combine kernels; the dq4 rows are one query per
    // group here too, so the split rule gives the 256 grid's partitions: bitwise the same output). Measured R9700 Ornith
    // fast chat +9.68 % (v0252-n2 Part S, KB v0252-n2:s-speed-r9700-ornith)
    .attn_dq4_dynamic_splits = true,
    // v0252-ws4r97b: the R9700 kt verdict (q7 9525u, scratch/v0252_ws4r97/kt/20261111-1236/results.txt, 4 ABBA rounds in
    // one object, every cell bitwise): q8 B/A 0.716 (64k) / 0.715 (128k) one row, 0.63 at 4 rows -> b; f16 0.97 one row,
    // 0.82 at 4 rows -> plain (b 0.974 / 0.972 at 64k / 128k vs plain 0.969 / 0.965)
    .attn_ws4r_q8 = 2,
    .attn_ws4r_f16 = 1,
    // v0252-l2split: one machine wave (64 blocks: Swift 4 kv heads -> 16 splits, Ornith 2 -> 32; today 64 each); the R9700
    // kt lsplit sweep (scratch/v0252_l2split/proposed) and the e2e A/B decide whether it stays
    .attn_lsplit_blocks = 64,
    .attn_ws4_split_blocks = 0,  // v0252-8sws: no attn_ws4 path on gfx1201
    .ws4_split_limits = {},      // v0252-8sws round 3: off (no attn_ws4 path)
    .attn_dq4_row_order = false,  // v0252-8sq4: no attn_dq4o in the gfx1201 object (attn_dq4 has the order)
    .dq4_min_keys = 0,            // v0252-8sq4 F8 split: opt-in (R9700 fast tg 96k inversion: its own screen later)
    .fast_kv_q8h_moe_ctx = 0,     // v0252-8sq4 fallback: opt-in (KvCaps::fast_q8h is false here, so inert today)
    .dq4_split_table = {},        // v0252-8sq5: off (not measured on the R9700): today's rule ops::dq4Splits
    .tune_fast = false,
    .tune_cache_suffix = "",
    .tune_time_hint = "2 minutes (a few seconds for MXFP4 models)",
    .gdn_strix = false,
    .numerics_8060s = false,
    .mtp_costly_verify_rows = false,
    // v0252-n2 (N1): Ornith on the R9700, measured chat +2.89 % precise / +3.39 % balance / +3.85 % fast (v0252-n2
    // Part S, KB v0252-n2:s-speed-r9700-ornith; research draft_vocab 2026-10-26-draft64k-per-arch.md; feat027-draft64k97
    // b334a52); drafts only, output unchanged
    .draft64k_moe = true,
    .relax_rank_check = true,  // v0252-f1 G3a (ADR-0015)
    // v0252-f1arg3 (PLAN V1-63 / V1-64): the multi-block verify argmax twins (kt q7 9525g: B/A 0.06-0.26, bitwise; e2e rule
    // progress/v0252-f1arg2.HANDOFF.md section 5). WHIRL_ARGMAX_MB=0 turns it off
    // v0252-b5 (PLAN V1-119, round 2c): merged OFF. The branch shipped true; its e2e verdict was NO-GAIN (V1-63 / V1-64,
    // 2026-11-11), re-judged 2026-11-12 (V1-117: Ornith fast MTP 2k +0.65 % +-0.42, Swift balance -0.03 %, bitwise).
    // Round 3: on for the measured class and mode only = MoE models in fast (the mode is applied by
    // Context::argmaxMbMode); dense models stay off. WHIRL_ARGMAX_MB=0 turns it off, =1 / 2 on in every mode and class.
    .argmax_mb_dense = false,  // gfx1201 dense: off (Swift balance -0.03 %)
    .argmax_mb_moe = true,  // gfx1201 MoE: on in fast (PLAN V1-117)
    .attn_sx_balance = false,  // v0252-strixc2: no attn_sx in the gfx1201 code object
    .attn_i4qk_fast = false,   // v0252-8sfp: no attn_prefill_wmma_i4 in the gfx1201 code object (fp8 there, iu4 not used)
    .attn_i4_qlds_pf = false,  // v0252-8sattn3: no attn_prefill_wmma_i4p in the gfx1201 code object
    // v0252-ch4k (PLAN V1-75): MoE solo prefills run 4096-row forwards in the 8060S fill form (whole chunks up to the
    // batch 4096; no extra memory, the prefill buffers are sized for the batch). Timed e2e A/B on the R9700 (Ornith
    // MXFP4, 4 ABBA rounds, 4096-row forwards vs 2048, scratch/v0252_ch4k/gpu/20261111-0310/summary.md): precise pp 8k
    // +8.47 % / 32k +6.94 %, balance 8k +8.24 % / 32k +6.44 % / 128k +3.21 %, fast 8k +8.02 % / 32k +6.55 %, 2k +-0;
    // precise output bitwise the same at 2k / 8k / 32k / 129k. Dense (Swift) keeps 2048 (prefill_exec_default).
    .prefill_exec_moe = 4096,
    .prefill_fill_batch = true,
    .tail_ck_in_forward = true,  // v0252-n10 (R9700 precise pp 2k: one forward instead of 1024 + 1034)
    // v0252-e1def: R9700 screens of 2026-11-12 (exe = branch v0252-e1p1 032c817, WHIRL_E1=2 against 0, decode tok/s per
    // round): Ornith (MoE) balance +3.93 / +4.15 % (scratch/v0252_e1p1c/gpu/DONE_e1p1c_screen) and +3.35 / +3.97 %
    // (scratch/abscreen/e1p1c-orn/DONE_e1p1c-orn: KEEP), Ornith precise +1.62 / +1.30 % with identical output; Swift
    // (dense) balance -1.37 / +0.09 % (no gain: stays 0). Fast keeps 0 through server::chooseE1 (not run with the
    // lookahead; its relaxed acceptance never looks ahead)
    .e1_dense = 0,
    .e1_moe = 2,
    // v0252-pp2k2: the 2k cell's 2058-row forward ran a 9th 256-token tile for 10 rows (+10.4 % time for +0.8 % rows,
    // KB v0252-pp2k:r97-precise-pp2k-tile-tail); screen progress/pp2k-gemmtail.json decides it
    .gemm_tail = true,
    .moep_comb_fast_skip = 0,  // v0252-8smoe: the gfx1201 fast moeskip plan keeps the f32 moe_yd combine 0 (no moep kernels)
    .moep_guw = 0,             // v0252-8smoe: no moep kernels in the gfx1201 object
    .moep_dhs = false,         // v0252-8smoe2: no moep kernels in the gfx1201 object
    .moep_dna = 0,
    .moep_fq = false,
    .vision_attn_threads = 128,
};

inline constexpr Profile k_profile_gfx1151{
    .arch = hip::Arch::gfx1151,
    .uma_no_vram_fit_check = true,
    .gemv_r = {1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
    // v0251-a6port: Swift (dense) verify widths 5 / 6 on gemvk v7 (KW 8, PF 2): the A6 kernel bench pick (sum of the
    // 8 Swift shapes 0.900 / 0.897 of gemvq_nt*r2, scratch/feat026_ggemv/results_8060s_b2a.txt) and its e2e (balance
    // decode 2k +7.91 %, 128k +5.43 %, results_8060s_h2.txt). MoE (Ornith) stays on gemvq until its own e2e passes.
    // v0252-b5 round 3 (PLAN V1-91): verify width 4 on v7 as well (Swift balance chat decode +0.78 % +-0.48, KEEP; bitwise
    // the gemvq path). WHIRL_GEMV_K replaces the whole row: "5:7,6:7" = the row before this change, "off" = all gemvq.
    .gemvk_dense = {0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    .gemvk_moe = {},
    .gemvh_grid_by_rows = true,
    // v0252-8sdec (PLAN V1-142) / v0252-8sgrp: the default (precise dense MXFP4 only, numerics.cpp loadFlags). Screens
    // 8sdec-grp2 20261124-0314, 8sdec-screen2 20261123-0912, 8sdec-screen2-bands 20261123-1127, 8sdec-grp2-long
    // 20261124-0828 (Swift precise, WHIRL_GEMVH_GRP=1 vs the one-launch path, same exe): tg 8192 +2.36 % +- 2.00 (gain),
    // tg 2048 +1.44 % +- 2.52, tg 32768 +2.60 % +- 3.41, tg 98304 +0.36 % +- 6.46 (open, no loss), prefill flat; precise
    // bits SAME every round, kt gemvhg invariant. WHIRL_GEMVH_GRP=0 = one gemvh launch per matrix.
    .gemvh_group_dense = true,  // v0252-8sgrp: default on (8060S precise decode / verify, same bits)
    .h16_from_candidates = true,
    .gemms_tie_larger_bm = true,
    .fdq_default = 2,  // t2 (v0251-a3port; v0251-judge1 sec 1 screening)
    .fdq_big = 1,      // c2 from 2305 rows (v0251-a3fix; scratch/v0251_a3/gpu/8060s_20261104-0439/results.txt:106-124)
    // v0252-8sbadopt: dense (Swift, <= 2304-row forwards) t2 -> t2g8, MoE (Ornith, 4096-row forwards) c2 -> c2g8; one-object
    // A/B job 94 (scratch/v0252_8sbal/gpu/8060s_e2e2_20261109-1021, re-judged rejudge_*): balance pp 8k / 32k Swift +1.14 /
    // +2.17 %, Ornith +2.50 / +3.39 %, 6 of 6 rounds each, decode hash identical. Exactly the measured arms (Ornith's t2
    // tail and Swift's unused c2 stay plain)
    .fq_grp_dense = 2,
    .fq_grp_moe = 1,
    .attn_dq4_dynamic_splits = true,
    .attn_ws4r_q8 = 0,  // v0252-ws4r97b: the gfx1151 object has no attn_ws4r (attn_ws4 / attn_ws4g there)
    .attn_ws4r_f16 = 0,
    .attn_lsplit_blocks = 0,  // v0252-l2split: the gfx1151 decode path is attn_ws4 (constant 64 splits, DES-ATTN-8S); unchanged
    .attn_ws4_split_blocks = 32,  // v0252-8sws: attn_ws4 / attn_ws4g q8 / q8h: Ornith (2 KV heads) 16 splits, Swift (4) 8
    // v0252-8sws round 3: [0] S 8 (Swift shape), [1] S 16 (Ornith shape); class 0 draft row, 1..9 rows. First slower measured
    // key count above each limit (B/A): Swift rows 3 131072 1.0086, 4 131072 1.0017, 5 262144 1.033, 6 65536 1.0041, 7 98304
    // 1.006, 8 98304 0.9997 [0.9983, 1.0012] (flat, not a gain), 9 32768 1.011; Ornith rows 7 / 8 / 9 32768 1.027 / 1.019 / 1.008
    .ws4_split_limits = {.on = true,
                         .max_keys = {{262144, 262144, 262144, 98304, 98304, 196608, 49152, 65536, 65536, 16384},
                                      {262144, 262144, 262144, 262144, 262144, 262144, 262144, 16384, 16384, 16384}}},
    // v0252-8sq4: opt-in until the screens progress/8sq4-order.json / 8sq4-minkeys.json / 8sq4-kvq8h.json show a gain beyond
    // the error (ADR-0019). On: attn_dq4_row_order = true, dq4_min_keys = the screened target, fast_kv_q8h_moe_ctx = 1 (or
    // the context the screen supports)
    // v0252-8sflip2: F8 order ON by default (fast plans with q4 KV run attn_dq4o; same partials as attn_dq4: kt attndq4
    // exact 54/54, [kv q4 order] 27/27). Screens scratch/abscreen/8sq4-order and 8sq4-order-long (Ornith fast, same exe,
    // default vs WHIRL_DQ4_ORDER=1): tg 131072 +10.49 % +- 9.08 (gain, 4 rounds), tg 32768 +3.89 % +- 3.34 (gain),
    // tg 65536 +9.69 % +- 10.24 (open), tg 8192 -1.53 % +- 5.78 (open), prefill guards flat. WHIRL_DQ4_ORDER=0 = attn_dq4
    .attn_dq4_row_order = true,
    .dq4_min_keys = 0,            // WHIRL_DQ4_MINKEYS=1024 turns it on
    .fast_kv_q8h_moe_ctx = 0,     // WHIRL_FAST_KVQ8H=1 turns it on (every load)
    // v0252-8sq5: kernel bench dq4s on this card (2026-11-24 16:09, scratch/v0252_8sq5/gpu/20261124-1609/table.txt; Ornith 16 / 2
    // heads, attn_dq4o + attn_combine, 87 cells, 0 FAIL, B/A = time(S) / time(today's count), 95 % intervals ~+-0.15 %). Every
    // measured breakpoint is written out; between them the lower one holds. The pick per cell is the fastest S; ties (two S
    // within 1 %) go to the S that is also best at the neighbouring length: rows 2 @ 65536 8 (0.811) over 32 (0.818: 8 is also
    // best at 49152), rows 2 @ 98304 64 (0.859) over 32 (0.865: 64 is also best at 131072). Today's count is 32 / 64 / 128 at
    // 8192 / 16384 / >= 32768; the picks are 10-37 % faster there (e.g. rows 2 @ 32768: 270.9 -> 173.7 us). WHIRL_DQ4_SPLITS=0 =
    // today's rule
    .dq4_split_table = {.on = true,
                        .steps = {{
                            // one row: 8 (0.789) at 8192; 16 from 16384 (0.679 0.609 0.694 0.729 0.802 0.805)
                            {{{8192, 8}, {16384, 16}, {32768, 16}, {49152, 16}, {65536, 16}, {98304, 16}, {131072, 16}, {0, 0}}},
                            // two verify rows: 8 to 65536 (0.704 0.655 0.643 0.725 0.811), 64 from 98304 (0.859 0.897)
                            {{{8192, 8}, {16384, 8}, {32768, 8}, {49152, 8}, {65536, 8}, {98304, 64}, {131072, 64}, {0, 0}}},
                            // three verify rows: 8 to 49152 (0.720 0.688 0.643 0.711), 32 from 65536 (0.759 0.831 0.857)
                            {{{8192, 8}, {16384, 8}, {32768, 8}, {49152, 8}, {65536, 32}, {98304, 32}, {131072, 32}, {0, 0}}},
                            // the MTP draft row, window active (65536 ..): 16 (0.633 0.632 0.629)
                            {{{65536, 16}, {98304, 16}, {131072, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}},
                        }}},
    .tune_fast = true,
    .tune_cache_suffix = "-gfx1151r3",  // r3: gemmfq in the dequant family (v0251-a3key; tests/model testTuneCacheKey)
    .tune_time_hint = "8 minutes on the Radeon 8060S (less for MXFP4 models)",
    .gdn_strix = true,
    .numerics_8060s = true,
    .mtp_costly_verify_rows = true,
    // v0252-8sdraft (PLAN V1-58): on again for 0.2.5.2 (8060S balance / fast are in scope, ADR-0013); off since the 0.2.2
    // freeze (9a58324), FAST-8S-1 item 2 (b2a3a9b) measured chat +2.5..+3.7 % here in all three modes. Drafts only (MoE
    // relaxacc is off in every mode: exact acceptance), so the output is unchanged; kept only on the 8060S A/B
    // (scratch\v0252_8sdraft\s8draft_gpu.ps1 -Part s, rule in its header)
    .draft64k_moe = true,
    .relax_rank_check = true,  // v0252-f1flip (ADR-0015 Revision 1): F1 on here too; the 8060S speed test ended INCONCLUSIVE twice with no loss, outputs identical
    // v0252-f1arg3: the gfx1151 object has no twins; the one-block argmax_rows(_to) / argmax_rank_rows_to stay
    .argmax_mb_dense = false,  // gfx1151 dense: no twins
    .argmax_mb_moe = false,  // gfx1151 MoE: no twins
    .attn_sx_balance = true,  // v0252-strixc2: balance prefill attention (acceptance: progress/v0252-strixc2.HANDOFF.md)
    // v0252-8sfp: on (fast only); screen progress/8sfp-screen.json decides it (8060S Swift fast pp 8k / 32k)
    .attn_i4qk_fast = true,
    // v0252-8sattn3: on (same bits; kt attn 14 _i4p rows exact, whirl bench Ornith fast hash equal). Kernel time V6/V1 -1.8..-7.8 % at L 2k-96k on both models (attnpv);
    // end to end open, no loss: Ornith fast pp 32k +0.21 % +- 1.24, 96k +0.76 % +- 1.13; Swift pp 8k -0.22 % +- 0.97, 32k +0.60 % +- 0.70. WHIRL_ATTN_I4PF=0 = old kernel
    .attn_i4_qlds_pf = true,
    .prefill_exec_moe = 4096,
    .prefill_fill_batch = true,
    .tail_ck_in_forward = true,  // v0252-tailck8s (PLAN V1-138, ADR-0022 supplement 4; was false = the 0.2.5.1 forward end)
    // v0252-tailck8s: precise pp 2k one 2058-row forward instead of 1024 + 1034. Precise runs the f32 chunked DeltaNet scan
    // on this card (gdnwmma is a balance item), which the capture splits as on the R9700; balance / fast run the GDN-8S
    // strix chunks, which Context::stateCaptureOk refuses, so they keep the forward end (the 0.2.5.1 schedule).
    // WHIRL_TAIL_CK_IN_FWD=0 turns it off (A/B arm)
    // v0252-e1def: the 8060S verifies with DeltaNet snapshot sets (no E1 path: ServerModel::e1InputsOk), nothing measured
    .e1_dense = 0,
    .e1_moe = 0,
    .gemm_tail = false,  // v0252-pp2k2: the 8060S keeps the 0.2.5.1 GEMM plans (object compare only)
    // v0252-8smoe (A): COMB 5 for Ornith fast (moeskip on). Shipped exe ef5652c, WHIRL_MOEP_COMB=5 against the shipped fast
    // default (KB v0252-retest:comb5-8s-ship, 4 rounds): pp 8192 +1.08 % +-1.07, pp 32768 +1.31 % +-0.83; the long band
    // (98304) is spec progress/8smoe-comb5-long.json
    .moep_comb_fast_skip = 5,
    // v0252-8smoe (B): g64n128k64 (index 1 + 1) for forwards with >= 128 tokens per expert (Ornith 4096-row forwards: 8k /
    // 32k; the 2k forward keeps g64n64k64). v0252-8sopen: opt-in - screen progress/8smoe-screen.json (2026-11-23) was OPEN
    // (balance pp 8192 +0.45 % +-1.05, pp 32768 +0.45 % +-1.08; ADR-0019: not a default until a cell gains beyond its
    // error); WHIRL_MOEP_GUW=g64n128k64 (a kernels::kMoepGu name) turns it on (same bits)
    .moep_guw = 0,  // v0252-8sopen: opt-in (was 2 = g64n128k64)
    // v0252-8smoe2: opt-in until the screens progress/8smoe2-C.json / -D.json / -E.json show a gain beyond the error
    // (ADR-0019); a default flip is this line plus the pinned CHECK in tests/iso/device.profile + tests/device.
    // v0252-8sflip (PLAN V1-147): C and E are the default - Ornith prefill on this card, paired change +- 95 % (ADR-0019):
    // E (8smoe2-e, -ec-bands, -e-precise; all KEEP, output hash same): balance pp 2048 +11.07 % +-1.61, 8192 +7.45 % +-1.98,
    // 32768 +5.20 % +-1.56, 98304 +3.10 % +-2.34; fast pp 2048 +10.13 % +-1.60, 32768 +5.47 % +-1.36; precise pp 2048
    // +9.47 % +-1.49, 8192 +6.22 % +-1.59, 32768 +4.50 % +-1.35 (equal output). E + C (8smoe2-ec-bands KEEP): balance pp 8192
    // +11.62 % +-2.19, 32768 +8.68 % +-2.44, 98304 +5.09 % +-2.30. WHIRL_MOEP_FQ=0 / WHIRL_MOEP_DHS=0 = the previous kernels.
    .moep_dhs = true,  // C (balance / fast f16 epilogue; precise never stores f16): WHIRL_MOEP_DHS=0 turns it off
    .moep_dna = 0,     // D: opt-in (8smoe2-D OPEN); WHIRL_MOEP_DNA=m128w8n128k64 turns it on (index 3 + 1 = 4)
    .moep_fq = true,   // E (every mode): WHIRL_MOEP_FQ=0 turns it off
    .vision_attn_threads = 256,
};

// The profile of a code-object family (the one arch test outside src/hip). REVIEW-ARCH3 P3-3: a switch with no
// default, so a new hip::Arch enumerator without a row is a compile error (MSVC C4062 / clang -Wswitch raised to
// errors here) instead of silently getting the gfx1201 row; a value outside the enumerators throws.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(error : 4062)
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#endif
inline constexpr const Profile& profile(hip::Arch a) {
    switch (a) {
    case hip::Arch::gfx1201: return k_profile_gfx1201;
    case hip::Arch::gfx1151: return k_profile_gfx1151;
    }
    throw std::invalid_argument("device::profile: no profile row for this hip::Arch value");
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

// arch025-M6: the two host-side choices of a code-object family that are made before (or without) a device, named
// here so no host code outside the device layer names a hip::Arch value (tools/gates/pending_scan.py rule arch blocks).
// The family the host assumes while no device is known: a Model that never went through load (Model::arch), the
// TuneTable gemv_r default (qwen35::defaultGemvR; the load overwrites it with the device's row) and mtpDefaults'
// default argument. The same row hip::archFor gives every device that is not a gfx1151 (device-tests pin both).
inline constexpr hip::Arch k_default_arch = hip::Arch::gfx1201;
// The row the server's pre-load UMA estimate sizes the MTP verify snapshot sets for (server_main.cpp, UMA branch:
// the 8060S draft count): the Radeon 8060S, the one UMA family. A second UMA family is a decision here.
inline constexpr hip::Arch k_uma_estimate_arch = hip::Arch::gfx1151;
static_assert(profile(k_uma_estimate_arch).uma_no_vram_fit_check, "the UMA estimate row must be a UMA family");
static_assert(!profile(k_default_arch).uma_no_vram_fit_check, "the default family is the discrete reference row");

}  // namespace whirl::device
