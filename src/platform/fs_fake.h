// FakeFs (arch025 PL2): test-only control of the file layer (platform/fs.h). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// A pass-through fake inside fs_win.cpp (the PL3 FakeNet pattern; prior art: RocksDB FaultInjectionTestFS, which
// wraps the real file system and injects errors; MSVC has no ld --wrap, ADR-0004 #5 has no virtual interface).
// Off unless a test calls begin(); then
//  - it records every fs::open (CreateFileW) with its path, access, share, disposition and flags, and its result;
//  - it injects faults, without touching the real call's arguments otherwise:
//      failWrite(n): the next n fs::writeAt calls return false and write nothing (lastError() unchanged);
//      shortRead(n, bytes): the next n fs::readAt / fs::readAsync calls ask Windows for at most `bytes` (a
//      sector multiple, so unbuffered handles stay legal), i.e. deliver a short read;
//      holdReadClose(): the fs::close of a handle opened with kOverlapped (the SSD tier's restore / prefetch read)
//      after this call waits until releaseClose() or end() (at most 30 s): the read job stops between storing
//      `failed` and `finished`, so a test can end the restore while the job still runs (arch025-tierfix).
// Off (the product), each fs::open / close / writeAt / readAt / readAsync costs one relaxed atomic load more.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace whirl::platform::fs::fake {

struct Open {
    std::wstring path;
    std::uint32_t access = 0, share = 0, disposition = 0, flags = 0;
    bool ok = false;  // the handle was valid
};

void begin();              // start recording / allow injections (clears everything recorded or injected before)
std::vector<Open> end();   // stop; the opens recorded since begin(), in order; pending injections are dropped
std::vector<Open> snapshot();
void failWrite(int n);
void shortRead(int n, std::uint32_t bytes);
int injectedLeft();        // write + read injections not used yet
void holdReadClose();
void releaseClose();
int closesHeld();          // closes that waited on holdReadClose since begin()

}  // namespace whirl::platform::fs::fake
