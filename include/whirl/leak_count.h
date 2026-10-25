// GPU-side leak count (arch025-Q2, release condition X-13 (a) as rewritten in arch025.PLAN.md O8): live device
// allocations, pinned host allocations, streams and events made through whirl::hip (hip_runtime.cpp, the only caller
// of hipMalloc / hipHostMalloc / hipStreamCreate* / hipEventCreate*), counted when WHIRL_LEAK_CHECK is set.
// SPDX-License-Identifier: Apache-2.0
//
// Why counters: ASan sees no hipMalloc device memory and MSVC has no LeakSanitizer (progress\safe025.PRIOR_ART.md,
// progress\arch025-Q2.PRIOR_ART.md); the same balance-counter idea as llama.cpp's CUDA pool (pool_size, asserted 0 at
// destruction) and the CPU twin of device::NullBackend's ~NullBackend check (WHIRL_LEAKCHECK builds).
//
// Off (the default, WHIRL_LEAK_CHECK unset, empty or "0"): read once at static initialization; one predictable branch per
// hip::malloc / free / hostMalloc / hostFree / streamCreate / streamDestroy / eventCreate / eventDestroy, nothing
// recorded or printed, no HIP call added or moved (ADR-0003 work equivalence). On: a mutex-protected map of the live
// device / pinned blocks (by address, for their sizes) and four live counts; still no HIP call.
//
// Use: a test harness takes snapshot() before a unit (a kernel-test family, a whole `whirl selftest` / `seqtest` after the
// Model is gone) and calls check(unit, before) after it. A unit that leaves more alive than it found is a leak: check()
// prints "LEAK_CHECK: <unit>: ..." and returns true; the harness then fails. A clean run prints "leak check: ok (...)"
// once (report()). The variable is a test switch: it is in the SSD-tier fingerprint skip list (rule I6).

#pragma once

#include <cstdint>
#include <string>

namespace whirl::hip::leakcount {

// set once at static initialization from WHIRL_LEAK_CHECK; tests may turn it on (testEnable)
extern bool g_on;

struct Snapshot {
    std::int64_t dev = 0, host = 0, streams = 0, events = 0;  // live counts
    std::int64_t dev_bytes = 0, host_bytes = 0;                // live bytes (as allocated, tail pad included)
};

// the counters (hip_runtime.cpp calls these only when g_on)
void onDevAlloc(std::uint64_t base, std::uint64_t bytes);
void onDevFree(std::uint64_t base);
void onHostAlloc(std::uint64_t base, std::uint64_t bytes);
void onHostFree(std::uint64_t base);
void onStream(int delta);
void onEvent(int delta);

Snapshot snapshot();
// what is alive now beyond `before` ("" when nothing): "2 device allocations (8448 bytes), 1 stream"
std::string leaked(const Snapshot& before, const Snapshot& after);
// off: false, prints nothing. On: compares now with `before`; a leak prints "LEAK_CHECK: <unit>: <leaked>" to stdout,
// counts it and returns true
bool check(const char* unit, const Snapshot& before);
// on: prints "leak check: ok (<n> units)" or "leak check: FAIL (<k> of <n> units leaked)" and returns the leak count;
// off: prints nothing, returns 0
int report();

// unit tests: reset the counters and the unit tally, on = record
void testEnable(bool on);

}  // namespace whirl::hip::leakcount
