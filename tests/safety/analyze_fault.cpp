// safe025: injected static-analysis faults. NOT part of any target: tools/safety/analyze_fault.bat compiles it with /analyze
// to prove that the baseline gate (analyze_check.py) fails on NEW warnings. Each function has one defect MSVC /analyze reports.
// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
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
