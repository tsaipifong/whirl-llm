// whirl-iso-spec-engine-tests: src/spec/slot_drafts.cpp built with only spec.policy and its declared dependencies
// (ARCH-1 8.5 isolation, arch025-isoall).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "spec/slot_drafts.h"

#include <array>
#include <cstdint>

using namespace whirl;

namespace {

void testDraftCap() {
    ISO_CHECK(server::slotDraftCap(4, 4, true, true, 1, 100) == 4);
    ISO_CHECK(server::slotDraftCap(4, 2, true, true, 1, 100) == 2);   // n_draft caps
    ISO_CHECK(server::slotDraftCap(4, 4, false, true, 1, 100) == 1);  // not fused: one draft
    ISO_CHECK(server::slotDraftCap(4, 4, true, true, 2, 5) == 1);     // 2 * (nd + 1) <= 5 rows
    ISO_CHECK(server::slotDraftCap(4, 4, true, true, 3, 5) == 0);
}

void testSplitByGain() {
    qwen35::DraftAccept a(0.9f), b(0.9f);
    const std::array<const qwen35::DraftAccept*, 2> acc{&a, &b};
    std::array<std::uint32_t, 2> nd{3, 3};  // the caller starts every slot at nd
    server::splitDraftsByGain(acc, 3, 64, 3, nd);
    ISO_CHECK(nd[0] <= 3 && nd[1] <= 3);  // never above cap
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("spec.engine", {{"draft_cap", testDraftCap}, {"split_by_gain", testSplitByGain}}, argc, argv);
}
