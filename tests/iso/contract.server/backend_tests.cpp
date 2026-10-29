// whirl-iso-contract-server-tests: the header-only module contract.server (src/server/backend.h, the ServerModel interface the engine runs on) built with its
// declared dependencies (the cache and device interfaces, the qwen35 KV / model types, the vision declarations) and the host-only implementation
// tests/iso/fake_server_model.h (ARCH-1 8.5 isolation, arch025-iso4). No test existed. The checks pin the interface's shape and the DEFAULTS of its optional
// hooks (what an implementation that does not override them answers: no pool resize, no sampled drafts, no M-RoPE, nothing lent, no profile, no
// DFlash), which every ServerModel that skips a hook relies on.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "iso/fake_server_model.h"

#include <array>
#include <cstring>
#include <type_traits>

namespace {

namespace srv = whirl::server;

void testInterfaceShape() {
    static_assert(std::is_abstract_v<srv::ServerModel>);
    static_assert(std::is_base_of_v<srv::CacheModel, srv::ServerModel>);  // the server's model contract includes the cache side
    static_assert(std::has_virtual_destructor_v<srv::ServerModel>);       // unlike CacheModel, a ServerModel is deleted through it
    static_assert(std::is_base_of_v<srv::ServerModel, whirl::test::FakeServerModel>);
    ISO_CHECK(true);
}

void testOptionalHookDefaults() {
    whirl::device::NullBackend ops;
    whirl::test::FakeServerModel fake(ops, 16, 8);
    srv::ServerModel& m = fake;
    ISO_CHECK(!m.resizePool(1024));
    ISO_CHECK(!m.draftSampleOk());
    ISO_CHECK(!m.hasMrope());
    std::array<std::array<std::uint64_t, 2>, 4> lend{};
    ISO_CHECK(m.lendScratch(lend) == 0);
    ISO_CHECK(!m.visRowsAllocated());
    ISO_CHECK(std::strcmp(m.profileClassName(0), "?") == 0);
    ISO_CHECK(m.verifyRows() == whirl::qwen35::max_small_batch);
    ISO_CHECK(m.dflashMaxDrafts() == 0 && m.captureN() == 0 && m.dflashRing() == 0);
    m.scratchClobbered();  // no-ops
    m.visRelease();
    m.profileStop();
    m.setVisMap(0, nullptr);
}

void testRequiredCallsThroughTheInterface() {
    whirl::device::NullBackend ops;
    whirl::test::FakeServerModel fake(ops, 16, 8);
    srv::ServerModel& m = fake;
    ISO_CHECK(m.cfg().n_vocab == 16 && m.cfg().n_embd == 8);
    ISO_CHECK(m.logits() != 0 && m.hn() != 0 && m.maxBatch() == whirl::qwen35::max_small_batch);
    ISO_CHECK(m.fusedDecode() == false && m.canSegment() == false && m.kvArrays().empty());
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("contract.server",
                    {{"interface_shape", testInterfaceShape},
                     {"optional_hook_defaults", testOptionalHookDefaults},
                     {"required_calls", testRequiredCallsThroughTheInterface}},
                    argc, argv);
}
