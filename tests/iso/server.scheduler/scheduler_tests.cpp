// whirl-iso-server-scheduler-tests: module server.scheduler (arch025-engine S15b, progress/arch025-engine.DESIGN.md
// v3 2.3 L9; manifest arch025-regb2, PLAN N4-reg b2): class Scheduler (server/engine_sched.h), built with only its declared
// dependencies (tests/iso/server.scheduler/isolation.txt) - not the modules above it (ARCH-1 8.5 isolation).
// References (DESIGN v3 3.1): bst_ m_ opt_ owner_ pool_ ps_ slots_ spec_ tb_ vis_ vram_probe_ + the placed state last_tick_ + every component below it: slot_cache, governor, vram_watch, job_queue, slot_end, prefill, spec_cycle, slot_start, tier_glue.
// What this program pins: at compile time the shape 3.1 fixes (not copyable, not default constructible, constructible
// from exactly the reference list above in the constructor's order, and not from that list without its last entry);
// at run time that every public function of the class is defined in the objects the manifest links (the program is
// linked from those objects alone with /OPT:NOREF, so a body that reaches an undeclared module is LNK2019 even
// though nothing here calls it). The component's behaviour runs end to end in whirl-server-tests (MockModel /
// NullBackend) and whirl-pins: running it here needs an Engine-sized rig of every reference (tests/iso/
// server.vram_watch builds the one for its 12) - not done in this part (arch025-regb2 HANDOFF).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/engine_sched.h"

#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

namespace whirl::server {
namespace {

using X = Scheduler;
static_assert(!std::is_copy_constructible_v<X> && !std::is_copy_assignable_v<X>);
static_assert(!std::is_default_constructible_v<X>);
static_assert(std::is_constructible_v<X, ServerModel&, EngineOptions&, EngineThread&, SpecState&, std::vector<Slot>&, BatchStats&, std::function<wddmwatch::Sample()>&, PagePool&, TimePoint&, PrefixStore&, TierBridge&, ImageEmbeddings&, SlotPages&, GovernorGlue&, VramWatch&, JobQueue&, SlotEnd&, PrefillPhase&, SpecCycle&, SlotStart&, TierGlue&>);
static_assert(!std::is_constructible_v<X, ServerModel&, EngineOptions&, EngineThread&, SpecState&, std::vector<Slot>&, BatchStats&, std::function<wddmwatch::Sample()>&, PagePool&, TimePoint&, PrefixStore&, TierBridge&, ImageEmbeddings&, SlotPages&, GovernorGlue&, VramWatch&, JobQueue&, SlotEnd&, PrefillPhase&, SpecCycle&, SlotStart&>);

void checkOne(bool defined) { ISO_CHECK(defined); }
template <class... M>
void checkDefined(M... m) {
    (checkOne(m != nullptr), ...);
}

void testFunctionsLinked() { checkDefined(&X::runLoop, &X::logBatchStats); }

}  // namespace
}  // namespace whirl::server

int main(int argc, char** argv) {
    return iso::run("server.scheduler", {{"functions_linked", whirl::server::testFunctionsLinked}}, argc, argv);
}
