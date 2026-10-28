// WHIRL device code (gfx1151 / RDNA 3.5): f32-activation matvec.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Matvec: y[r] (+)= sum_c W[r][c] * x[t][c], for NT tokens at once.
// One wave per row; 8 rows per 256-thread block.
// ---------------------------------------------------------------------------

#include "shared/gemv_f_gemv_entry.h"  // also in the gfx1201 set (kernels/gemv_f.hip)