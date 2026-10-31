// WHIRL device code (gfx1151 / RDNA 3.5): GDN beta/alpha projection entries for the non-q8_0 quant types
// (gdn_ab_*, gdn_abconv_*; the q8_0 ones and gdn_ab_impl live in fused_decode.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included right before moe.hip so the code object is unchanged (arch025-sc11k, ADR-0005).
#include "shared/gdn_ab_entries.h"  // also in the other kernel set
