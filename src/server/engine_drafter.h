// Server engine, the DFlash2 drafter context of a slot (arch025-engine S4, module server.drafter_ctx; moved verbatim
// out of engine.h:223-227 and engine_decode.cpp:24-58 at 7ad196c; each definition line `Engine::f` -> `inline
// DrafterCtx::f`). DrafterCtx owns nothing: it holds the core.shared members its four functions use, under their
// Engine names (bst_: dflIngest counts dfl_gap; m_: the drafter's trim / ingest; opt_: read only). Engine holds it as
// dc_, declared after vis_ (destroyed first, trivially destructible: DESIGN v3 2.5). Header-only (cut header_inline):
// the decode cycle inlined them in engine_decode.cpp before the move. Includes are spelled from src/ (server/...), as
// in engine_common.h.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "server/backend.h"         // ServerModel (dflashMaxDrafts, captureN, dflashIngest, dflashTrim)
#include "server/engine_options.h"  // EngineOptions (use_mtp, use_dflash, n_draft, dfl_warm)
#include "server/engine_slot.h"     // Slot (id, pos, dfl_lo, dfl_hi)
#include "server/engine_state.h"    // BatchStats (dfl_gap)
#include "whirl/dflash_policy.h"    // dflash::Range, planIngest, ready

#include <cstdint>

namespace whirl::server {

struct DrafterCtx {
    // DFlash2 drafter context of a slot
    std::uint32_t draftRoom() const;
    void dflReset(Slot& sl, std::uint32_t at);
    void dflIngest(Slot& sl, std::uint32_t run_rows, std::uint32_t r0, std::uint32_t n, std::uint32_t pos0);
    bool dflReady(const Slot& sl) const;

    BatchStats& bst_;
    ServerModel& m_;
    const EngineOptions& opt_;
};

// ---------------------------------------------------------------------------
// DFlash2 drafter context (--draft dflash). Its rows come from the target features captured by
// each forward / verify; a slot's context is valid for the slot's own lineage only (same tokens at
// the same positions). Restores from a shared checkpoint or a host tier bring KV and state but no
// features: the context starts empty at the restore point and the slot drafts with DFlash2 again
// once dfl_warm rows were ingested after it (re-running the target over earlier positions would
// rewrite KV rows the restored state already holds).

inline std::uint32_t DrafterCtx::draftRoom() const {
    if (opt_.use_mtp) return opt_.n_draft;
    if (opt_.use_dflash) return m_.dflashMaxDrafts();
    return 0;
}

inline void DrafterCtx::dflReset(Slot& sl, std::uint32_t at) {
    if (!opt_.use_dflash) return;
    m_.dflashTrim(sl.id, 0);
    sl.dfl_lo = sl.dfl_hi = at;
}

// Rows r0 .. r0 + n - 1 of the latest forward / verify of run_rows rows are positions pos0 .. of
// slot sl: ingest those whose features were captured.
inline void DrafterCtx::dflIngest(Slot& sl, std::uint32_t run_rows, std::uint32_t r0, std::uint32_t n, std::uint32_t pos0) {
    if (!opt_.use_dflash || n == 0) return;
    ::whirl::dflash::Range rg{sl.dfl_lo, sl.dfl_hi};
    const ::whirl::dflash::Ingest g = ::whirl::dflash::planIngest(rg, m_.captureN(), run_rows, r0, n, pos0);
    bst_.dfl_gap += g.gap;
    if (g.any) m_.dflashIngest(sl.id, g.cap_row, g.n, g.pos);
    sl.dfl_lo = rg.lo;
    sl.dfl_hi = rg.hi;
}

inline bool DrafterCtx::dflReady(const Slot& sl) const {
    return opt_.use_dflash && ::whirl::dflash::ready(::whirl::dflash::Range{sl.dfl_lo, sl.dfl_hi}, sl.pos, opt_.dfl_warm);
}

}  // namespace whirl::server
