// feat026-fdq: integer part of the MXFP4 "exponent injection" decode used by the gfx1151 fused-dequant prefill GEMM
// (gemmfq_*_mxfp4, gemm_prefill.hip). Host + device: the CPU unit test (tests/unit, testMxInject) runs these same
// functions against the LUT reference, so the device kernel and the CPU check cannot drift apart.
// SPDX-License-Identifier: Apache-2.0
//
// MXFP4 code n = s e1 e0 m (E2M1). Placing e1 e0 m at f16 bits 11:9 and s at bit 15 gives the f16 whose value is
// exactly e2m1(n) * 2^-14 (code 1 = 0.5 lands on the subnormal 2^-15; idea: Modular decode_e2m1_to_f16,
// "exponent injection"). WHIRL's LUT path computes kvalues_mxfp4[n] * f16(2^(e - 128)) with kvalues = 2 * e2m1, i.e.
// the same real number e2m1(n) * 2^(e - 127). One f16 FMA by S = 2^(e - 113) with a +0 addend rounds that real
// once (exact for e in [104, 128]: every product is an f16 multiple of 2^-24 not above 12), and turns code 8 (-0)
// into +0 like the LUT (kvalues[8] = 0). Block exponents outside [104, 128] take the LUT math (bitwise by construction).
// Only the f16 byte HIGH halves are built here: the low byte of every injected f16 is 0.
#pragma once

#include <stdint.h>

#if defined(__HIPCC__)
#define MXI_HD __host__ __device__ __forceinline__
#else
#define MXI_HD inline
#endif

// per byte: high byte of the injected f16 of the LOW nibble (bit 3 -> bit 7, bits 2:0 -> bits 3:1)
MXI_HD uint32_t mxi_hib_lo(uint32_t q) { return ((q << 1) & 0x0E0E0E0Eu) | ((q << 4) & 0x80808080u); }
// per byte: high byte of the injected f16 of the HIGH nibble (bit 7 stays, bits 6:4 -> bits 3:1)
MXI_HD uint32_t mxi_hib_hi(uint32_t q) { return ((q >> 3) & 0x0E0E0E0Eu) | (q & 0x80808080u); }
// v_perm_b32 selectors: bytes {t.b(2j), t.b(2j+1)} of the S1 operand into the high bytes of two f16, low bytes 0
constexpr uint32_t kMxiSpread0 = 0x010C000Cu, kMxiSpread1 = 0x030C020Cu;
// portable form of __builtin_amdgcn_perm(0, t, kMxiSpread<j>) (the CPU check uses it; the device uses v_perm)
MXI_HD uint32_t mxi_spread_ref(uint32_t t, int j) {
    return (((t >> (16 * j)) & 0xFFu) << 8) | (((t >> (16 * j + 8)) & 0xFFu) << 24);
}
// block exponent range in which the injected decode is bit-identical to the LUT decode
MXI_HD bool mxi_fast_ok(uint32_t e) { return e - 104u <= 24u; }
// f16 bits of S = 2^(e - 113) (normal for e in [104, 128]: biased exponent e - 98 in [6, 30])
MXI_HD uint32_t mxi_scale_bits(uint32_t e) { return (e - 98u) << 10; }
