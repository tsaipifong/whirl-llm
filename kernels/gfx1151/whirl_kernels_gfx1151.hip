// WHIRL device code for gfx1151 (RDNA 3.5, Radeon 8060S): the kernel set of
// the gfx1151 code object, included by kernels/whirl_kernels.hip when the
// device compiler targets gfx1151.
// SPDX-License-Identifier: Apache-2.0
//
// Like the gfx1201 set, the files are fragments of one translation unit and
// the include order matters. Kernel names and argument ABIs match the gfx1201
// kernels (include/whirl/kernels_abi.h); families gfx1151 cannot run (fp8 /
// int8-dot WMMA, the q8v KV format, the WMMA mid-batch GEMV, the small-batch
// GEMM, vision) are simply absent, and the host picks the remaining paths
// from what the code object contains (KernelTable / kernels::Caps).

#include "common.hip"              // helpers, packed int8 scale words
#include "quant.hip"               // quant block decoders
#include "gemv_f.hip"              // f32-activation matvec
#include "gemm_prefill.hip"        // prefill GEMM (WMMA f16, configurations)
#include "gemm_small.hip"          // small-batch prefill GEMM (raw quantized stages)
#include "gemv_q.hip"              // int8 decode GEMV, multi-token, grouped ABI
#include "gemv_mr.hip"             // multi-row multi-token GEMV
#include "gemv_wunit.hip"          // Q4_K / Q5_K / IQ4_XS / MXFP4 GEMV units
#include "gemv_q6.hip"             // Q6_K GEMV family
#include "fused_decode.hip"        // decode fusions (norm+quant, fused DeltaNet)
#include "misc_rows_norm.hip"      // norms, rope, elementwise
#include "attn.hip"                // paged KV + decode / split / prefill attention
#include "gdn.hip"                 // Gated DeltaNet, kv_store, argmax
#include "misc_probe_requant.hip"  // WMMA probes, Q6_K / Q8_0 -> Q4_K requant
#include "draft_pick.hip"          // 2-bit MTP draft head, token tables, picks
#include "moe.hip"                 // mixture of experts (incl. MXFP4 generic entries)
#include "sample_cand.hip"         // sampling candidates
#include "prefill_fused.hip"       // prefill activation fusions (f16)
#include "../moe_f.hip"            // precise MoE decode experts (f32 activations; shared with gfx1201)
