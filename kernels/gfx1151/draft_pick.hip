// WHIRL device code (gfx1151 / RDNA 3.5): 2-bit MTP draft head, token tables, greedy / draft picks.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

#define D2_XD(p) xdv(p)  // gfx1151 packed scale words
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
#include "shared/draft_d2_requant_q6k_d2.h"  // also in the gfx1201 set (kernels/draft_d2.hip)

// Token ids + positions for up to 16 rows without a host copy (a blocking
// hipMemcpy would drain the queue): rows [0, n_host) from the by-value table,
// the rest from dev_src (earlier on-device argmax results).
#include "shared/sample_pick_set_tokens.h"  // also in the gfx1201 set (kernels/sample_pick.hip)
