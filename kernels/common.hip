// WHIRL device code (gfx1201 / RDNA 4): shared device helpers (warp/block reductions) and GGUF quant block decoders (dec8).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// HIP kernels for RDNA 4 (gfx1201), Qwen3.5-family (qwen35 / qwen35moe).
//
// Compiled at build time to a gfx1201 code object, embedded in the host
// binary and loaded with hipModuleLoadData; every entry point is extern "C"
// so the host finds it by name (see include/whirl/kernels_abi.h). Weights
// stay in their GGUF block formats; each quant type gets a `dec8` that
// decodes 8 consecutive values, and all matvecs share one loop.
//
// Numerics follow the GGUF block formats and llama.cpp's reference math
// (dequantization, Gated DeltaNet recurrence, NEOX RoPE); the C++ CPU
// references in tests/kernels/ check them.

#include "shared/common_warp.h"  // also in the gfx1151 set (kernels/gfx1151/common.hip)

// Block-wide sum for blockDim.x <= 1024 (multiple of 32).
#include "shared/common_part_2.h"  // also in the gfx1151 set (kernels/gfx1151/common.hip)

// ---------------------------------------------------------------------------
// Quant formats. g = index of an 8-value group within the row.
// ---------------------------------------------------------------------------

#include "shared/common_part.h"  // also in the gfx1151 set (kernels/gfx1151/quant.hip)


// ---- shared kernel vocabulary (arch025 N14: kernels.common; was defined in the first family that used it) ----
#include "shared/common_vocab.h"
template <int T> constexpr int t2_sbb() { return 256 / Q<T>::blk * Q<T>::bytes; }
typedef unsigned int t2_u32x4 __attribute__((ext_vector_type(4)));

// Quantize the 32 values held by one wave (lane i holds v) into xq/xd at block b.
__device__ __forceinline__ void wave_q8(float v, int8_t *xq, float *xd, int b) {
    const int lane = threadIdx.x % WARP;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    xq[b * 32 + lane] = (int8_t)__float2int_rn(v * id);
    if (lane == 0) xd[b] = d;
}
