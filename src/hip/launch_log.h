// Launch log (ARCH-1 K-0): a record of every kernel launch of a process (kernel name, grid, block, dynamic shared
// memory), written at exit when WHIRL_LAUNCH_LOG names a folder. The kernel-abstraction steps K-1..K-5 move the
// kernel choice out of src/model/forward.cpp; this table pins what is chosen, grids included, so each step can show
// it launches exactly the same kernels with the same geometry (tools/impact/launch_table.py compares two runs).
// SPDX-License-Identifier: Apache-2.0
//
// Off (the default): one predictable branch per launch and per function lookup, nothing recorded. Decode graph
// replays do not pass through launchRaw (only their capture does), like the launch counter.

#pragma once

#include <cstdint>

namespace whirl::hip::launchlog {

// set once at static initialization from WHIRL_LAUNCH_LOG (a folder); never changes afterwards
extern const bool g_on;

void onFunction(const void* f, const char* name);
void onLaunch(const void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared);

}  // namespace whirl::hip::launchlog
