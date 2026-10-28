// WHIRL device code (gfx1151 / RDNA 3.5): sampling candidates.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Sampling candidates (server sampling). For each of gridDim.x logits rows
// ([rows][n]): stats[row] = {max, sum_i exp((x_i - max) * inv_t)} (the full
// softmax normalizer at temperature 1/inv_t) and the K largest logits in
// ids/vals[row][0..K) (unordered). The K-th largest value is found with an
// 8-bit radix select over order-preserving float keys (4 passes, per-wave
// histograms to keep shared-atomic contention low).
#include "shared/sample_cand_topk_rows.h"  // also in the gfx1201 set (kernels/sample_cand.hip)
