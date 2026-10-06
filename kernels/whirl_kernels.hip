// WHIRL device code: the single translation unit compiled once per GPU
// architecture into an embedded code object.
// SPDX-License-Identifier: Apache-2.0
//
// The inference kernels are split by family into the files below. They are
// fragments of one translation unit (later files use helpers, templates and
// macros of earlier ones), so the include order matters and matches the
// order of the original single-file source; kernel names, argument ABIs and
// generated code are unchanged by the split. The host-side view (argument
// structs, kernel names, launch tables) is include/whirl/kernels_abi.h.
//
// gfx1201 (RDNA 4) carries the full set. The gfx1151 (RDNA 3.5) set lives in
// kernels/gfx1151/ (same kernel names and ABIs, gfx11 WMMA fragments, no fp8).

#include <hip/hip_runtime.h>

#include "smoke.hip"

#if !defined(__gfx1151__)
#include "common.hip"              // helpers, quant block decoders
#include "gemv_f.hip"              // f32-activation matvec
#include "gemm_prefill.hip"        // prefill GEMM (WMMA f16, configurations)
#include "gemv_q.hip"              // int8 decode GEMV, multi-token, grouped ABI
#include "gemv_mr.hip"             // multi-row multi-token GEMV
#include "gemv_w.hip"              // mid-batch WMMA GEMV
#include "gemv_wp.hip"             // pipelined mid-batch WMMA GEMV
#include "gemv_wx.hip"             // wide mid-batch WMMA GEMV (17..32 tokens)
#include "fused_decode.hip"        // decode fusions (norm+quant, fused DeltaNet)
#include "misc_rows_norm.hip"      // get_rows, norms, rope, elementwise
#include "attn.hip"                // paged KV + decode / split / prefill attention
#include "gdn_prefill.hip"         // Gated DeltaNet per-token and chunked f32
#include "gdn_prefill_wmma.hip"    // Gated DeltaNet chunked prefill on WMMA
#include "prefill_fused.hip"       // prefill activation fusions
#include "attn_prefill_kp.hip"     // prefill attention (key-split pairs)
#include "gdn_seq.hip"             // DeltaNet conv / state / sequential scan
#include "misc_kv_argmax.hip"      // kv_store, argmax
#include "misc_probe_requant.hip"  // WMMA probes, Q6_K -> Q4_K requant
#include "draft_d2.hip"            // 2-bit MTP draft head
#include "sample_pick.hip"         // token tables, greedy / draft picks
#include "moe.hip"                 // mixture of experts
#include "sample_cand.hip"         // sampling candidates
#include "gemm_fp8.hip"            // MXFP4 x fp8 GEMM
#include "gemm_fp8t.hip"           // fragment-tiled fp8 / f16 GEMM
#include "gemm_small.hip"          // small-batch prefill GEMM
#include "moe_mxfp4.hip"           // MXFP4 routed experts (decode, fp8 grouped GEMM)
#endif

#if defined(__gfx1151__)
#include "gfx1151/whirl_kernels_gfx1151.hip"  // RDNA 3.5 kernel set
#endif

#include "spec_sample.hip"           // speculative sampling of MTP drafts (both kernel sets)
