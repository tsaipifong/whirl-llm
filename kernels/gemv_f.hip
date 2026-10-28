// WHIRL device code (gfx1201 / RDNA 4): f32-activation matvec gemv_<type>_<NT> (reference / fallback decode path).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Matvec: y[r] (+)= sum_c W[r][c] * x[t][c], for NT tokens at once.
// One wave per row; 8 rows per 256-thread block.
// ---------------------------------------------------------------------------

#include "shared/gemv_f_gemv_entry.h"  // also in the gfx1151 set (kernels/gfx1151/gemv_f.hip)