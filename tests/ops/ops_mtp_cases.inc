// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.mtp/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/mtp.h"
#include "../test_check.h"

#include <cstdint>
#include <string>

namespace {

namespace ops = whirl::ops;

// the MTP rules (arch025-n8b1: the block of testGdnChoice that calls only src/ops/mtp.cpp / mtp.h, in a function of its own
// so ops.mtp's isolation program runs it; whirl-ops-tests calls it from testGdnChoice, in the same place)
void testMtpRules() {
    // MTP: draft head rows / map, the chained-draft gate, the batch limits
    {
        ops::DraftHeadProblem p;
        p.head_rows = 248320;
        p.draft_vocab = 32000;
        CHECK(ops::chooseDraftHead(p).nrows == 32000 && !ops::chooseDraftHead(p).map);
        p.draft_rows = 65536;  // no 2-bit head: the subset is not used
        CHECK(ops::chooseDraftHead(p).nrows == 32000);
        p.d2 = true;
        CHECK(ops::chooseDraftHead(p).nrows == 65536 && !ops::chooseDraftHead(p).map);
        p.map = true;
        CHECK(ops::chooseDraftHead(p).map);
        p.d2 = false;
        p.draft_rows = 0;
        p.draft_vocab = 300000;
        CHECK(ops::chooseDraftHead(p).nrows == 248320 && !ops::chooseDraftHead(p).map);  // the map needs the 2-bit subset
        CHECK(ops::draftGated(1, 0, true, 1, 0.1f) && !ops::draftGated(1, 0, true, 0, 0.1f) && !ops::draftGated(1, 1, true, 1, 0.1f) &&
              !ops::draftGated(0, 0, true, 1, 0.1f) && !ops::draftGated(1, 0, false, 1, 0.1f) && !ops::draftGated(1, 0, true, 1, 0.0f));
        CHECK(ops::verifySmallMax(16, 16, 32) == 16 && ops::verifySmallMax(17, 16, 32) == 32);
        CHECK(ops::unfusedVerifyOk(1, 1) && !ops::unfusedVerifyOk(1, 2) && !ops::unfusedVerifyOk(2, 0));
    }
}

}  // namespace
