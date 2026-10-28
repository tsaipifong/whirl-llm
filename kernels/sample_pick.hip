// WHIRL device code (gfx1201 / RDNA 4): token tables, greedy picks for batched / verify rows, MTP draft pick with probability.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// Token ids + positions for up to 16 rows without a host copy (a blocking
// hipMemcpy would drain the queue): rows [0, n_host) from the by-value table,
// the rest from dev_src (earlier on-device argmax results).
#include "shared/sample_pick_set_tokens.h"  // also in the gfx1151 set (kernels/gfx1151/draft_pick.hip)

