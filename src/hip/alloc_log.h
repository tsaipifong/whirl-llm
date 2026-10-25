// Allocation log (arch025-S0a): the sequence of device / pinned-host allocations and releases of a process, by size
// and kind (no addresses), written at exit when WHIRL_ALLOC_LOG names a folder (alloc_<pid>_<start>[_<n>].txt, created exclusively, see log_name.h;
// older builds: alloc_<pid>.txt). It pins the
// invariant I2 of arch025.PLAN.md (the Model's device / pinned allocations keep their sizes and order) so a
// refactoring step can show it allocates exactly what the baseline (S0b) allocated.
// SPDX-License-Identifier: Apache-2.0
//
// Off (the default): one predictable branch per hip::malloc / free / hostMalloc / hostFree, nothing recorded, nothing
// printed (I4: no parsed log changes). The same shape as the launch log (launch_log.h): read once at static
// initialization, written once at exit (atexit / at_quick_exit), the recorder is never destroyed.
//
// File: "# whirl alloc log v1", "# cmd: ...", a summary ("# events N seqhash H", per kind count / bytes, peak live
// bytes) and then, when the event list is on, one line per event:
//   <index> dev <bytes> #<ordinal>          hip::malloc (bytes as allocated, tail pad included)
//   <index> host <bytes> #<ordinal>         hip::hostMalloc
//   <index> free #<ordinal> <bytes>         hip::free of that allocation ("#? 0" for a pointer it never saw)
//   <index> hostfree #<ordinal> <bytes>     hip::hostFree
// Ordinals count device and host allocations separately, from 0, in allocation order. The seqhash covers every event
// (kind, bytes, ordinal) whether or not the list is written.
// The event list is off in whirl-server unless WHIRL_ALLOC_LOG_RAW=server (serveMain calls serverProcess(); a long
// server allocates per request), and at most kMaxLines lines are kept (the rest counted in the last line).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace whirl::hip::alloclog {

// set once at static initialization from WHIRL_ALLOC_LOG (a folder); tests may turn it on (testEnable)
extern bool g_on;

void onDevAlloc(std::uint64_t base, std::uint64_t bytes);
void onDevFree(std::uint64_t base);
void onHostAlloc(std::uint64_t base, std::uint64_t bytes);
void onHostFree(std::uint64_t base);

// whirl-server: the event list off unless WHIRL_ALLOC_LOG_RAW=server (the summary is always written)
void serverProcess();

inline constexpr std::size_t kMaxLines = 200000;

// unit tests: reset the recorder (on = record; list = keep the event list; max_lines = its cap) and read the text
// the exit writer would write (without the "# cmd:" line)
void testEnable(bool on, bool list, std::size_t max_lines = kMaxLines);
std::string testText();
// unit tests (bug:log-pid-names): write the log file to dir now, as the exit writer would
void testWrite(const std::string& dir);

}  // namespace whirl::hip::alloclog
