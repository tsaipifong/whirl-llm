// WHIRL device code (gfx1201 / RDNA 4): KV cache row store (kv_store*), greedy argmax.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// Write n_tok new K/V rows into the (paged) cache: row t at logical position
// pos[0] + t (single sequence) or pos[t] (batched rows, kvbase). One block
// per row; row % 32 == 0 (whole waves per q8 scale group).
#include "shared/misc_kv_argmax_kv_store.h"  // also in the gfx1151 set (kernels/gfx1151/gdn.hip)

