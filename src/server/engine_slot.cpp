// Server request slot: parkSlotCkpts (arch025-engine S1, moved verbatim out of engine_gov.cpp, where govFreeCache
// calls it). The declarations: engine_slot.h.
// SPDX-License-Identifier: Apache-2.0

#include "engine_slot.h"

namespace whirl::server {

std::uint64_t parkSlotCkpts(CkptStore& cs, Slot& sl) {
    // the spill to the RAM tier may still read the checkpoint buffers: its fence goes to park (F1)
    const std::uint64_t b = sl.ckpts.size() * cs.bytes(true);
    cs.park(sl.ckpts, {sl.spill_ev.get()});
    sl.ckpts.clear();
    sl.parked = true;
    return b;
}

}  // namespace whirl::server
