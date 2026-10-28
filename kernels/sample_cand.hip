// WHIRL device code (gfx1201 / RDNA 4): sampling candidates (top-k logits per row).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Sampling candidates (server). For each of gridDim.x logits rows
// ([rows][n]): stats[row] = {max, sum_i exp((x_i - max) * inv_t)} (the full
// softmax normalizer at temperature 1/inv_t) and the K largest logits in
// ids/vals[row][0..K) (unordered). The K-th largest value is found with an
// 8-bit radix select over order-preserving float keys (4 passes, per-wave
// histograms to keep shared-atomic contention low).
#include "shared/sample_cand_topk_rows.h"  // also in the gfx1151 set (kernels/gfx1151/sample_cand.hip)

