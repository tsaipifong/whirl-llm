// v0251-a3port TEST-FIRST STATE (commit 1 only): the interface of the MXFP4 exponent-injection helpers that
// tests/unit/kernels_gemm_tests.cpp (testMxInject) checks, with placeholder bodies, so the test builds on 2421013 and
// FAILS on behaviour. Commit 2 replaces this file with the real helpers (feat026-fdq 90d5cf7 / c15f51f, verbatim).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

#if defined(__HIPCC__)
#define MXI_HD __host__ __device__ __forceinline__
#else
#define MXI_HD inline
#endif

MXI_HD uint32_t mxi_hib_lo(uint32_t) { return 0u; }
MXI_HD uint32_t mxi_hib_hi(uint32_t) { return 0u; }
constexpr uint32_t kMxiSpread0 = 0u, kMxiSpread1 = 0u;
MXI_HD uint32_t mxi_spread_ref(uint32_t, int) { return 0u; }
MXI_HD bool mxi_fast_ok(uint32_t) { return false; }
MXI_HD uint32_t mxi_scale_bits(uint32_t) { return 0u; }
