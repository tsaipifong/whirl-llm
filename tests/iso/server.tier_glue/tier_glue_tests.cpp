// whirl-iso-server-tier_glue-tests: module server.tier_glue (arch025-engine S15, progress/arch025-engine.DESIGN.md
// v3 2.3 L8; manifest arch025-regb2, PLAN N4-reg b2): class TierGlue (server/engine_tierglue.h), built with only its declared
// dependencies (tests/iso/server.tier_glue/isolation.txt) - not the modules above it (ARCH-1 8.5 isolation).
// References (DESIGN v3 3.1): ops_ opt_ pool_ ps_ slots_ tb_ + drafter_ctx, slot_end, slot_start, job_queue, slot_cache, governor.
// What this program pins: at compile time the shape 3.1 fixes (not copyable, not default constructible, constructible
// from exactly the reference list above in the constructor's order, and not from that list without its last entry);
// at run time that every public function of the class is defined in the objects the manifest links (the program is
// linked from those objects alone with /OPT:NOREF, so a body that reaches an undeclared module is LNK2019 even
// though nothing here calls it). The component's behaviour runs end to end in whirl-server-tests (MockModel /
// NullBackend) and whirl-pins: running it here needs an Engine-sized rig of every reference (tests/iso/
// server.vram_watch builds the one for its 12) - not done in this part (arch025-regb2 HANDOFF).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/engine_tierglue.h"

#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

namespace whirl::server {
namespace {

using X = TierGlue;
static_assert(!std::is_copy_constructible_v<X> && !std::is_copy_assignable_v<X>);
static_assert(!std::is_default_constructible_v<X>);
static_assert(std::is_constructible_v<X, device::Backend&, EngineOptions&, std::vector<Slot>&, PagePool&, PrefixStore&, TierBridge&, DrafterCtx&, SlotEnd&, SlotStart&, JobQueue&, SlotPages&, GovernorGlue&>);
static_assert(!std::is_constructible_v<X, device::Backend&, EngineOptions&, std::vector<Slot>&, PagePool&, PrefixStore&, TierBridge&, DrafterCtx&, SlotEnd&, SlotStart&, JobQueue&, SlotPages&>);

void checkOne(bool defined) { ISO_CHECK(defined); }
template <class... M>
void checkDefined(M... m) {
    (checkOne(m != nullptr), ...);
}

void testFunctionsLinked() { checkDefined(&X::attachTier, &X::waitsForTier, &X::prefetchFor, &X::tryRestore, &X::pollRestores, &X::tierTick, &X::restoreSteps, &X::flushTierOnStop); }

}  // namespace
}  // namespace whirl::server

int main(int argc, char** argv) {
    return iso::run("server.tier_glue", {{"functions_linked", whirl::server::testFunctionsLinked}}, argc, argv);
}
