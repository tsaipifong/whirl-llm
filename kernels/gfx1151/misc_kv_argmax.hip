// WHIRL device code (gfx1151 / RDNA 3.5): KV cache row store (kv_store*), greedy argmax.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included right after gdn.hip by kernels/gfx1151/whirl_kernels_gfx1151.hip (code object
// unchanged; arch025-sc11k, ADR-0005).

// Write n_tok new K/V rows into the (paged) cache: row t at logical position
// pos[0] + t (single sequence) or pos[t] (batched rows, kvbase). One block
// per row; row % 32 == 0 (whole waves per q8 scale group).
#include "shared/misc_kv_argmax_kv_store.h"  // also in the gfx1201 set (kernels/misc_kv_argmax.hip)
