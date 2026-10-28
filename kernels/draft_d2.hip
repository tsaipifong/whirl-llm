// WHIRL device code (gfx1201 / RDNA 4): 2-bit MTP draft head GEMV (gemv_d2_nt*).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

#define D2_XD(p) (*(p))  // plain f32 activation scales
// ---------------------------------------------------------------------------
// 2-bit MTP draft head ("D2"). Only the draft argmax matters (a wrong draft
// costs one rejected verify row), and measured acceptance does not depend on
// the head's precision (Q6_K / Q4_K / Q3_K heads accept the same), so the
// draft head streams 2.5 bits per weight instead of Q4_K's 4.5.
// Row layout (K columns): codes [K/4 bytes] then f16 scales [K/32]:
//   unit u = 64 values = 16 code bytes at 16u; value at unit position
//   16k + 4kk + j is the 2-bit field (bits 8j + 2kk) of word k, so
//   (w_k >> 2kk) & 0x03030303 gives 4 codes in x order;
//   block b (32 values) has scale s_b: w = s_b * (2c - 3), c in 0..3.
// Arithmetic is free-form (drafts never have to match anything bitwise).
// ---------------------------------------------------------------------------
#include "shared/draft_d2_requant_q6k_d2.h"  // also in the gfx1151 set (kernels/gfx1151/draft_pick.hip)
