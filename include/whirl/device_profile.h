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

namespace whirl::device {

struct Profile {
    hip::Arch arch;
    // ---- memory
    // memory shared with the CPU (UMA iGPU): the load does not check the weights against the free VRAM,
    // which does not bound what it can allocate (loader.cpp; WHIRL_VRAM_LIMIT_MB still applies)
    bool uma_no_vram_fit_check;
    // ---- GEMV / GEMM dispatch (src/model/forward.cpp)
    // rows per block of the multi-row quantized GEMV per batch size 0..16 (Model::gemv_r default)
    std::array<std::uint8_t, 17> gemv_r;
    // f16-in GEMV grid: kernels::gemvhGrid(nrows) blocks (else one block per 16 rows)
    bool gemvh_grid_by_rows;
    // f16 GEMM outputs (h16) are decided from the tuner's candidate list, not the tuned choices (the code
    // object has no h16 fused / dequant GEMM; GEMVH-2)
    bool h16_from_candidates;
    // small-batch WMMA GEMM selection: ties between row blocks go to the larger BM (GEMVH-2)
    bool gemms_tie_larger_bm;
    // q4 KV decode through attn_dq4: the smallest split count with the same key partition (FAST-8S-1)
    bool attn_dq4_dynamic_splits;
    // ---- tuning (src/model/tune.cpp)
    bool tune_fast;                 // each kernel timed once per shape, slow candidates rejected early
    const char* tune_cache_suffix;  // tune cache file name suffix (table revision)
    const char* tune_time_hint;     // first-run message: how long the tune takes
    // f16-WMMA chunked DeltaNet prefill from strix-llama.cpp (GDN-8S; where gdn_wmma's kernels do not exist)
    bool gdn_strix;
    // the numerics plan's 8060S rows (numerics::Target::gfx1151 / KvCaps::fast_q8h): MXFP4-only relaxed
    // items, fast q8h KV on dense models, the 8060S reason texts
    bool numerics_8060s;
    // ---- MTP / prefill schedule
    // a verify row costs far more (UMA): the 8060S draft-count / cost-model defaults (qwen35::mtpDefaults)
    bool mtp_costly_verify_rows;
    std::size_t prefill_exec_moe;  // default forward rows of a MoE solo prefill (prefillExecDefault)
    bool prefill_fill_batch;       // forwards fill whole chunks up to the batch (prefillExecFor fill, SRV-PP)
    // ---- vision
    unsigned vision_attn_threads;  // kernels/vision/vision.hip vis_attn2 block size of this code object
};

inline constexpr Profile k_profile_gfx1201{
    .arch = hip::Arch::gfx1201,
    .uma_no_vram_fit_check = false,
    .gemv_r = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
    .gemvh_grid_by_rows = false,
    .h16_from_candidates = false,
    .gemms_tie_larger_bm = false,
    .attn_dq4_dynamic_splits = false,
    .tune_fast = false,
    .tune_cache_suffix = "",
    .tune_time_hint = "2 minutes (a few seconds for MXFP4 models)",
    .gdn_strix = false,
    .numerics_8060s = false,
    .mtp_costly_verify_rows = false,
    .prefill_exec_moe = 2048,
    .prefill_fill_batch = false,
    .vision_attn_threads = 128,
};

inline constexpr Profile k_profile_gfx1151{
    .arch = hip::Arch::gfx1151,
    .uma_no_vram_fit_check = true,
    .gemv_r = {1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
    .gemvh_grid_by_rows = true,
    .h16_from_candidates = true,
    .gemms_tie_larger_bm = true,
    .attn_dq4_dynamic_splits = true,
    .tune_fast = true,
    .tune_cache_suffix = "-gfx1151r2",
    .tune_time_hint = "8 minutes on the Radeon 8060S (less for MXFP4 models)",
    .gdn_strix = true,
    .numerics_8060s = true,
    .mtp_costly_verify_rows = true,
    .prefill_exec_moe = 4096,
    .prefill_fill_batch = true,
    .vision_attn_threads = 256,
};

// The profile of a code-object family (the one arch test outside src/hip).
inline constexpr const Profile& profile(hip::Arch a) { return a == hip::Arch::gfx1151 ? k_profile_gfx1151 : k_profile_gfx1201; }

}  // namespace whirl::device
