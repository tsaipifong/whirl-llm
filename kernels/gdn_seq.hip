// WHIRL device code (gfx1201 / RDNA 4): Gated DeltaNet token-parallel conv, conv state update, sequential scan, gated norm.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.


// Causal conv over a token block, parallel over (channel, token): the first
// d_conv-1 taps before the block come from the rolling state.
#include "shared/gdn_seq_gdn_conv_par.h"  // also in the gfx1151 set (kernels/gfx1151/gdn.hip)

