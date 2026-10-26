// WHIRL device code: LDS-only workgroup barrier (feat026-barrierdrain), both kernel sets.
// SPDX-License-Identifier: Apache-2.0
//
// With ROCm 7.2's compiler every workgroup barrier form (__syncthreads, __builtin_amdgcn_s_barrier,
// fence(workgroup, "local") + barrier, and on gfx12 the s_barrier_signal / s_barrier_wait builtins) is
// preceded by a wait on ALL memory counters (gfx11 s_waitcnt vmcnt(0) [vscnt] lgkmcnt(0), gfx12
// s_wait_loadcnt_dscnt 0 [storecnt]): the subtargets lack FeatureBackOffBarrier in this compiler (LLVM
// 4b89e8adda; upstream re-enabled it for GFX11 / GFX12 in d6edc1a96f, PR #155370, with no extra wait in
// WGP mode). That wait drains register prefetches and in-flight global stores at every barrier.
//
// whirl_lds_barrier() is the barrier upstream LLVM emits in WGP mode for fence(local) + barrier: this
// wave's LDS operations complete (lgkmcnt / dscnt 0), then the barrier; global loads and stores stay in
// flight (the compiler still waits for each load at its first use). Valid ONLY where the waves of the
// workgroup exchange data through LDS across the barrier, never through global memory. The "memory"
// clobber keeps the compiler from moving memory accesses across it. Same form as hipfire AT_BARRIER /
// CK block_sync_lds_direct_load / gemm_fp8t g8t_lds_barrier. See progress/feat026-barrierdrain.PRIOR_ART.md.
//
// WHIRL_BARDRAIN (bit mask, default 0 = the shipped __syncthreads everywhere):
//   1 = gemmsd_impl (gfx1201 gemms_c5..c7 / gemmsh_c5..c7)
//   2 = gdn_sx_scan (gfx1151)
//   4 = gdn_step_norm (gfx1151)
//   8 = gdn_step_norm (gfx1201; feat026-barrierdrain2)
//  16 = gdn_wscan_impl (gfx1201 gdn_wscan4 / gdn_wscan8; feat026-barrierdrain2)
#pragma once
#ifndef WHIRL_BARDRAIN
#define WHIRL_BARDRAIN 0
#endif
#if WHIRL_BARDRAIN != 0
__device__ __forceinline__ void whirl_lds_barrier() {
#if defined(__GFX12__)
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
#else
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#endif
}
#endif
