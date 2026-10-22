// Launch log (ARCH-1 K-0): a record of every kernel launch of a process (kernel name, grid, block, dynamic shared
// memory), written at exit when WHIRL_LAUNCH_LOG names a folder. The kernel-abstraction steps K-1..K-5 move the
// kernel choice out of src/model/forward.cpp; this table pins what is chosen, grids included, so each step can show
// it launches exactly the same kernels with the same geometry (tools/impact/launch_table.py compares two runs).
// SPDX-License-Identifier: Apache-2.0
//
// Off (the default): one predictable branch per launch, per function lookup and per allocation, nothing recorded.
// Decode graph replays do not pass through launchRaw (only their capture does), like the launch counter.
//
// REVIEW-ARCH4 P3-2: each launch also carries a digest of its arguments (hip::launch stages them): scalars as they
// are, and every 8-byte word that points into (or just below) a device allocation as (allocation ordinal, offset),
// so the digest is the same from run to run while it still pins head0 / r0 / r1 / copy offsets / n_split / group
// tables. The per-(kernel, geometry, digest) counts and an argument-sequence hash are written as extra lines
// ("# argseqhash", "@ ..."); a table without them (older builds) compares on kernels and geometry only.
// REVIEW-ARCH4 P3-6: the recorder is never destroyed; it is written once at exit (atexit / at_quick_exit) under its
// lock, and launches after that are ignored.
// Debug aid (arch6e): WHIRL_LAUNCH_LOG_RAW set as well writes launch_<pid>.raw.txt next to the table, one line per
// launch (index, kernel, geometry, digest, then per argument "|size" and each 8-byte word as in:ordinal:offset /
// below:ordinal:distance / v:value with "=raw value" after a normalised pointer, tail bytes as t:hex), so two runs can
// be diffed launch by launch.

#pragma once

#include <cstddef>
#include <cstdint>

namespace whirl::hip::launchlog {

// set once at static initialization from WHIRL_LAUNCH_LOG (a folder); never changes afterwards
extern const bool g_on;

void onFunction(const void* f, const char* name);
void onLaunch(const void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared);
// the arguments of the next onLaunch on this thread (hip::launch, include/whirl/hip.h)
void stageArgs(void* const* params, const std::uint32_t* sizes, std::size_t n);
// the digest stageArgs stages for these arguments, against the allocations registered so far (none while the log is
// off); also callable with the log off (unit tests)
std::uint64_t argDigest(void* const* params, const std::uint32_t* sizes, std::size_t n);
// device allocations (hip::malloc / free, module globals): the address space argument pointers are normalized against
void onAlloc(std::uint64_t base, std::uint64_t bytes);
void onFree(std::uint64_t base);
// unit tests (arch6f): register / clear allocations whether or not the log is on, to digest the same arguments
// against two address layouts
void testAlloc(std::uint64_t base, std::uint64_t bytes);
void testClearAllocs();

}  // namespace whirl::hip::launchlog
