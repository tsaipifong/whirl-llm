// whirl-iso-server-slot_end-tests: module server.slot_end (arch025-engine S11, progress/arch025-engine.DESIGN.md
// v3 2.3 L5; manifest arch025-regb2, PLAN N4-reg b2): class SlotEnd (server/engine_end.h), built with only its declared
// dependencies (tests/iso/server.slot_end/isolation.txt) - not the modules above it (ARCH-1 8.5 isolation).
// References (DESIGN v3 3.1): m_ ops_ opt_ sb_ slots_ tb_ vis_ + the placed state last_reap_ + the components below it: slot_cache (SlotPages), vram_watch (VramWatch), job_queue (JobQueue).
// What this program pins: at compile time the shape 3.1 fixes (not copyable, not default constructible, constructible
// from exactly the reference list above in the constructor's order, and not from that list without its last entry);
// at run time that every public function of the class is defined in the objects the manifest links (the program is
// linked from those objects alone with /OPT:NOREF, so a body that reaches an undeclared module is LNK2019 even
// though nothing here calls it). The component's behaviour runs end to end in whirl-server-tests (MockModel /
// NullBackend) and whirl-pins: running it here needs an Engine-sized rig of every reference (tests/iso/
// server.vram_watch builds the one for its 12) - not done in this part (arch025-regb2 HANDOFF).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/engine_end.h"

#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

namespace whirl::server {
namespace {

using X = SlotEnd;
static_assert(!std::is_copy_constructible_v<X> && !std::is_copy_assignable_v<X>);
static_assert(!std::is_default_constructible_v<X>);
static_assert(std::is_constructible_v<X, ServerModel&, device::Backend&, EngineOptions&, SampleBuffers&, SlotPages&, std::vector<Slot>&, TierBridge&, ImageEmbeddings&, VramWatch&, TimePoint&, JobQueue&>);
static_assert(!std::is_constructible_v<X, ServerModel&, device::Backend&, EngineOptions&, SampleBuffers&, SlotPages&, std::vector<Slot>&, TierBridge&, ImageEmbeddings&, VramWatch&, TimePoint&>);

void checkOne(bool defined) { ISO_CHECK(defined); }
template <class... M>
void checkDefined(M... m) {
    (checkOne(m != nullptr), ...);
}

void testFunctionsLinked() { checkDefined(&X::finishJob, &X::releaseSlot, &X::failSlot, &X::reapDisconnected); }

}  // namespace
}  // namespace whirl::server

int main(int argc, char** argv) {
    return iso::run("server.slot_end", {{"functions_linked", whirl::server::testFunctionsLinked}}, argc, argv);
}
