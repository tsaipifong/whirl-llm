// safe025: injected-fault driver for the C++ safety mechanisms (PLAN_0.2.1.md "Proof that the safety mechanisms work").
// Test-only. Each mode commits ONE fault on purpose; the mechanism that must catch it is named. mode "clean" is the control
// (no fault, exit 0). Built only with -DWHIRL_SAFETY_FAULTS=ON. SPDX-License-Identifier: Apache-2.0
//
//   missing_free   device malloc never freed        -> leak check (NullBackend destructor, WHIRL_LEAKCHECK), exit 97
//   missing_host   pinned hostMalloc never freed    -> leak check, exit 97
//   missing_stream stream + event never destroyed   -> leak check, exit 97
//   uaf            read of a freed device block     -> AddressSanitizer heap-use-after-free
//   uaf_vector     read through a dangling vector element pointer -> AddressSanitizer
//   overflow       write one past a device block    -> AddressSanitizer heap-buffer-overflow
//   double_free    free twice                       -> AddressSanitizer (or the backend's own error)
#include "device/null_backend.h"

#include <cstdio>
#include <cstring>
#include <vector>

using whirl::device::NullBackend;

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "clean";
    std::printf("safety_faults mode=%s\n", mode);
    std::fflush(stdout);
    NullBackend b;
    char out[16] = {};
    if (!std::strcmp(mode, "clean")) {
        auto p = b.malloc(4096);
        auto h = b.hostMalloc(4096);
        auto s = b.streamCreateNonBlocking();
        auto e = b.eventCreate();
        b.free(p);
        b.hostFree(h);
        b.eventDestroy(e);
        b.streamDestroy(s);
    } else if (!std::strcmp(mode, "missing_free")) {
        (void)b.malloc(4096);  // FAULT: never freed
    } else if (!std::strcmp(mode, "missing_host")) {
        (void)b.hostMalloc(4096);  // FAULT: never freed
    } else if (!std::strcmp(mode, "missing_stream")) {
        (void)b.streamCreateNonBlocking();  // FAULT
        (void)b.eventCreate();              // FAULT
    } else if (!std::strcmp(mode, "uaf")) {
        auto p = b.malloc(4096);
        b.free(p);
        b.download(out, p, sizeof out);  // FAULT: read after free
    } else if (!std::strcmp(mode, "uaf_vector")) {
        std::vector<int> v(8, 1);
        int* q = &v[0];
        v.resize(100000);  // reallocates
        std::printf("%d\n", *q);  // FAULT: dangling
    } else if (!std::strcmp(mode, "overflow")) {
        auto p = b.malloc(4096);
        b.download(out, p + 4096, 1);  // FAULT: one past the block
        b.free(p);
    } else if (!std::strcmp(mode, "double_free")) {
        auto p = b.malloc(4096);
        b.free(p);
        b.free(p);  // FAULT
    } else {
        std::fprintf(stderr, "unknown mode\n");
        return 2;
    }
    std::printf("no fault detected inline\n");
    return 0;
}
