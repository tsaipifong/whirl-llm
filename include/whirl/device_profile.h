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
    std::size_t prefill_exec_moe;  // default forward rows of a MoE solo prefill (prefillExecDefault)
    bool prefill_fill_batch;       // forwards fill whole chunks up to the batch (prefillExecFor fill, SRV-PP)
    // v0252-n10: whirl-server writes the TAIL-CK checkpoint inside the solo prefill forward that contains the tail point
    // (sched::tailInside, EngineOptions::tail_ck_in_forward) instead of ending a forward there; the CLI keeps its forward
    // end (same bits: docs/guide/en/kv-and-caching.md section 7, progress/v0252-n10.DESIGN.md). 0.2.5.2: gfx1201 only
    bool tail_ck_in_forward;
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
    .prefill_exec_moe = 2048,
    .prefill_fill_batch = false,
    .tail_ck_in_forward = true,  // v0252-n10 (R9700 precise pp 2k: one forward instead of 1024 + 1034)
    .vision_attn_threads = 128,
};

inline constexpr Profile k_profile_gfx1151{
    .arch = hip::Arch::gfx1151,
    .uma_no_vram_fit_check = true,
    .gemv_r = {1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
    // v0251-a6port: Swift (dense) verify widths 5 / 6 on gemvk v7 (KW 8, PF 2): the A6 kernel bench pick (sum of the
    // 8 Swift shapes 0.900 / 0.897 of gemvq_nt*r2, scratch/feat026_ggemv/results_8060s_b2a.txt) and its e2e (balance
    // decode 2k +7.91 %, 128k +5.43 %, results_8060s_h2.txt). MoE (Ornith) stays on gemvq until its own e2e passes.
    .gemvk_dense = {0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    .gemvk_moe = {},
    .gemvh_grid_by_rows = true,
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
    .prefill_exec_moe = 4096,
    .prefill_fill_batch = true,
    .tail_ck_in_forward = false,  // v0252-n10: 0.2.5.2 does not touch the Radeon 8060S (the 0.2.5.1 forward end stays)
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
