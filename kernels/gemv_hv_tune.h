// WHIRL device code (gfx1201 / RDNA 4): tune table of the precise gemvh (v0252-c2port, from feat026-gemvh).
// SPDX-License-Identifier: Apache-2.0
// WHIRL_GVH_<TYPE> = candidate id of kernels/gemv_hv.hip (0 = the 0.2.5.1 gemv_h.hip kernel).
// Preprocessor only. Included by gemv_h.hip before its GEMVH_TYPE lines (which skip a type whose id is non-zero) and by
// gemv_hv.hip (which then defines gemvh_<type> / gemvh2_<type> itself): both see the same value although gemv_hv.hip
// comes last in the gfx1201 block of whirl_kernels.hip. Adopting a candidate = changing a default here.
// Only the gfx1201 block reads this file (the gfx1151 object builds kernels/gfx1151/gemv_h.hip), like llama.cpp's
// per-arch constexpr mmvq parameter table (ggml-cuda/mmvq.cu get_device_table_id / calc_nwarps).
// v0252-c2acc (0.2.5.2, KB v0252-c2:screen): default = m1_q1, id 1 (ring depth 4, activation lookahead 1, no prefetch)
// for MXFP4 and Q6_K; bit-identical to gemm_c0 (kt gemvh), used only by precise decode / MTP-verify rows
// (ops::chooseGemmF16: prec_dec, n <= 32); balance / fast never select gemvh.
#pragma once

#ifndef WHIRL_GVH_MXFP4
#define WHIRL_GVH_MXFP4 1
#endif
#ifndef WHIRL_GVH_Q6_K
#define WHIRL_GVH_Q6_K 1
#endif
