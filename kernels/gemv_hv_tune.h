// WHIRL device code (gfx1201 / RDNA 4): tune table of the precise gemvh (v0252-c2port, from feat026-gemvh).
// SPDX-License-Identifier: Apache-2.0
// WHIRL_GVH_<TYPE> = candidate id of kernels/gemv_hv.hip (0 = the shipped gemv_h.hip kernel, the default).
// Preprocessor only. Included by gemv_h.hip before its GEMVH_TYPE lines (which skip a type whose id is non-zero) and by
// gemv_hv.hip (which then defines gemvh_<type> / gemvh2_<type> itself): both see the same value although gemv_hv.hip
// comes last in the gfx1201 block of whirl_kernels.hip. Adopting a candidate = changing a default here.
#pragma once

#ifndef WHIRL_GVH_MXFP4
#define WHIRL_GVH_MXFP4 0
#endif
#ifndef WHIRL_GVH_Q6_K
#define WHIRL_GVH_Q6_K 0
#endif
