// whirl-iso-server-spec_cycle-tests: module server.spec_cycle (arch025-engine S13, progress/arch025-engine.DESIGN.md
// v3 2.3 L7; manifest arch025-regb2, PLAN N4-reg b2): class SpecCycle (server/engine_cycle.h), built with only its declared
// dependencies (tests/iso/server.spec_cycle/isolation.txt) - not the modules above it (ARCH-1 8.5 isolation).
// References (DESIGN v3 3.1): bst_ ct_ ctx_ m_ ops_ opt_ sb_ slots_ spec_ + the placed state spec_on_ + prefill's prefill_tok_cum_ + drafter_ctx, slot_end, slot_cache, vram_watch.
// What this program pins: at compile time the shape 3.1 fixes (not copyable, not default constructible, constructible
// from exactly the reference list above in the constructor's order, and not from that list without its last entry);
// at run time that every public function of the class is defined in the objects the manifest links (the program is
// linked from those objects alone with /OPT:NOREF, so a body that reaches an undeclared module is LNK2019 even
// though nothing here calls it). The component's behaviour runs end to end in whirl-server-tests (MockModel /
// NullBackend) and whirl-pins: running it here needs an Engine-sized rig of every reference (tests/iso/
// server.vram_watch builds the one for its 12) - not done in this part (arch025-regb2 HANDOFF).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/engine_cycle.h"

#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

namespace whirl::server {
namespace {

using X = SpecCycle;
static_assert(!std::is_copy_constructible_v<X> && !std::is_copy_assignable_v<X>);
static_assert(!std::is_default_constructible_v<X>);
static_assert(std::is_constructible_v<X, ServerModel&, device::Backend&, const chat::ChatTokens&, EngineOptions&, std::uint32_t&, SpecState&, SampleBuffers&, bool&, std::vector<Slot>&, BatchStats&, std::uint64_t&, DrafterCtx&, SlotEnd&, SlotPages&, VramWatch&>);
static_assert(!std::is_constructible_v<X, ServerModel&, device::Backend&, const chat::ChatTokens&, EngineOptions&, std::uint32_t&, SpecState&, SampleBuffers&, bool&, std::vector<Slot>&, BatchStats&, std::uint64_t&, DrafterCtx&, SlotEnd&, SlotPages&>);
// v0252-hostwait2 (REVIEW-v0252-hostwait D1): "References only" - the object is exactly its 15 references (a value member,
// e.g. a mode copy, a pinned buffer or an event handle owned here, makes it larger and fails this)
static_assert(sizeof(X) == 15 * sizeof(void*), "SpecCycle holds references only (module server.spec_cycle, DESIGN v3 3.1)");

void checkOne(bool defined) { ISO_CHECK(defined); }
template <class... M>
void checkDefined(M... m) {
    (checkOne(m != nullptr), ...);
}

void testFunctionsLinked() { checkDefined(&X::pickDrafts, &X::relaxRepeats, &X::decodeCycle, &X::relaxes, &X::verifyRoom, &X::dropOutOfRoom, &X::draftLimits, &X::ngramDrafts, &X::dflashDrafts, &X::mtpDrafts, &X::layoutBatch, &X::draftSampling, &X::enqueueVerify, &X::sampleRows, &X::trainCycle, &X::cycleStats, &X::acceptSlot, &X::finishCycle, &X::splitDrafts, &X::draftCap); }

}  // namespace
}  // namespace whirl::server

int main(int argc, char** argv) {
    return iso::run("server.spec_cycle", {{"functions_linked", whirl::server::testFunctionsLinked}}, argc, argv);
}
