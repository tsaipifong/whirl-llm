// arch025-n8b2 (PLAN N8 batch 2): the cases of forward.state (src/model/qwen35/context.cpp + the inline rules of context.h),
// moved verbatim out of tests/iso/forward/forward_tests.cpp, which includes this file at the same place (same cases, same order).
// Also included by the module's own isolation program tests/iso/forward.state/forward_state_tests.cpp (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "iso/iso_test.h"
#include "model/qwen35/context.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

namespace q = whirl::qwen35;

void testTuneBucketOfEachSize() {
    // every prefill GEMM tune size is its own bucket: w.tune[tuneBucket(n)] holds the choice tuned at tune_sizes[b]
    for (std::size_t b = 0; b < q::tune_sizes.size(); ++b) ISO_CHECK(q::tuneBucket(q::tune_sizes[b]) == b);
}

void testTuneBucketEdges() {
    // a batch goes to the smallest tuned size >= n; above tune_small_max to the 4096 bucket
    const std::uint32_t n[] = {1, 32, 33, 48, 49, 64, 65, 96, 97, 128, 129, 192, 193, 256, 257, 384, 385, 768, 769, 1024, 1025, 4096, 100000};
    const std::size_t want[] = {3, 3, 4, 4, 5, 5, 6, 6, 2, 2, 7, 7, 8, 8, 9, 9, 0, 0, 10, 10, 1, 1, 1};
    static_assert(std::size(n) == std::size(want));
    for (std::size_t i = 0; i < std::size(n); ++i) ISO_CHECK(q::tuneBucket(n[i]) == want[i]);
}

void testOpClassNames() {
    // the WHIRL_PROFILE table rows (cli decode, server /metrics): one name per class, in enum order
    const char* want[] = {"embed", "norm", "quant", "matmul", "attn", "gdn_prep", "gdn_rec", "gdn_norm", "misc", "head"};
    static_assert(std::size(want) == q::n_classes);
    for (std::size_t i = 0; i < q::n_classes; ++i) ISO_CHECK(std::strcmp(q::opClassName(static_cast<q::OpClass>(i)), want[i]) == 0);
}

void testProfileHostOnly() {
    // a Profile with nothing recorded: no timing event is made (made on first record), flush and reset touch no device
    q::Profile p;
    ISO_CHECK(p.events.size() == q::Profile::cap && p.classes.size() == q::Profile::cap && p.count == 0);
    p.ms[2] = 1.5;
    p.matmul_bytes = 7;
    p.flush();
    ISO_CHECK(p.ms[2] == 1.5);
    p.reset();
    ISO_CHECK(p.ms[2] == 0 && p.matmul_bytes == 0 && p.count == 0);
    // record needs a timing event: the device runtime (here the link double, which refuses)
    bool threw = false;
    try {
        p.record(nullptr, q::OpClass::matmul);
    } catch (const std::logic_error& e) {
        threw = std::string(e.what()).find("eventCreate") != std::string::npos;
    }
    ISO_CHECK(threw && p.count == 0);
}

void testDraftWindowArg() {
    ISO_CHECK(q::draftWindowArg(0, 1u << 20) == 0);                        // off
    ISO_CHECK(q::draftWindowArg(1, 0) == 1);                               // W / 64 rounded up
    ISO_CHECK(q::draftWindowArg(64, 0) == 1 && q::draftWindowArg(65, 0) == 2);
    ISO_CHECK(q::draftWindowArg(4096, 32768) == (64 | 32 << 16));          // threshold / 1024 in the high bits
    ISO_CHECK(q::draftWindowArg(64u * 70000u, 2048u * 0x7fffu) == static_cast<std::int32_t>(0xffffu | 0x7fffu << 16));  // clamped
}

}  // namespace
