// MTP drafting and speculative-verify choice (arch025-F6; moved from src/model/forward/mtp_verify.cpp
// Model::mtpEnqueue / mtpBatchStepEx). See whirl/ops/mtp.h.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/mtp.h"

namespace whirl::ops {

DraftHeadPlan chooseDraftHead(const DraftHeadProblem& p) {
    DraftHeadPlan pl;
    pl.nrows = p.head_rows;
    if (p.draft_rows > 0 && p.d2)
        pl.nrows = p.draft_rows;
    else if (p.draft_vocab > 0 && p.draft_vocab < pl.nrows)
        pl.nrows = p.draft_vocab;
    pl.map = p.draft_rows > 0 && p.d2 && p.map;
    return pl;
}

}  // namespace whirl::ops
