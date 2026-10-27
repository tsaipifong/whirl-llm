// safe025 / arch025-Q3: injected static-analysis faults. NOT part of any target: tools/safety/analyze_fault.bat compiles it with
// /analyze three times to prove both directions of the baseline gate (analyze_check.py) with the real compiler:
//   AF_MODE=0  clean:    only acceptedOld()'s warning, which tests/safety/analyze_fault_baseline.txt lists      -> gate PASS
//   AF_MODE=1  new:      + 3 planted defects in 3 new functions (3 new keys)                                    -> gate FAIL, 3 NEW
//   AF_MODE=2  same key: acceptedOld() gets a SECOND warning of the same rule (same file|rule|function key,
//                        other line), so the key occurs twice against a baseline count of 1 (review C3)        -> gate FAIL, NEW x1
// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#ifndef AF_MODE
#define AF_MODE 1
#endif

// The "already accepted" warning: C6001 (using uninitialized memory 'y'), listed in analyze_fault_baseline.txt.
int acceptedOld(bool c) {
    int y;
    if (c) y = 2;
#if AF_MODE == 2
    int z;            // the same rule again in the same function: C6001 on 'z'
    if (!c) z = 5;
    return y + z;
#else
    return y;
#endif
}

#if AF_MODE == 1
int usesUninitialized(bool c) {
    int x;            // C6001: using uninitialized memory
    if (c) x = 1;
    return x;
}
int derefsNull() {
    int* p = static_cast<int*>(std::malloc(sizeof(int)));
    *p = 3;           // C6011: dereferencing NULL pointer (malloc result unchecked)
    int v = *p;
    std::free(p);
    return v;
}
int overrun() {
    int a[4] = {};
    int s = 0;
    for (int i = 0; i <= 4; ++i) s += a[i];  // C6201: index 4 is out of the valid range 0..3 (MSVC 14.44 reports C6201 here)
    return s;
}
#endif
