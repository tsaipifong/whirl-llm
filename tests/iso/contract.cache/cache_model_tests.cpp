// whirl-iso-contract-cache-tests: the header-only module contract.cache (src/cache/cache_model.h, the calls the cache side makes on the model) built with
// its declared dependencies (the device interface, the qwen35 KV / model types) and the host-only implementation tests/iso/fake_cache_model.h (ARCH-1 8.5
// isolation, arch025-iso4). No test existed. The checks pin the interface's shape (abstract, not destroyed through it) and that a complete implementation
// serves every call through a CacheModel&.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "iso/fake_cache_model.h"
#include <string>
#include <vector>

#include <type_traits>

namespace {

namespace srv = whirl::server;

void testInterfaceShape() {
    static_assert(std::is_abstract_v<srv::CacheModel>);
    static_assert(!std::is_destructible_v<srv::CacheModel>, "CacheModel has a protected destructor: it is not deleted through the interface");
    static_assert(std::is_base_of_v<srv::CacheModel, iso::FakeCacheModel>);
    ISO_CHECK(true);
}

void testCallsThroughTheInterface() {
    whirl::device::NullBackend ops;
    iso::FakeCacheModel fake(ops, {false, true, true, false}, 64, 128, 16, 32);
    srv::CacheModel& m = fake;
    ISO_CHECK(m.cfg().n_layer == 4 && m.cfg().n_embd == 16 && m.cfg().n_vocab == 32);
    ISO_CHECK(m.convState().size() == 4 && m.ssmState().size() == 4);
    ISO_CHECK(m.convState()[0] == 0 && m.convState()[1] != 0 && m.ssmState()[2] != 0);  // attention layers have no recurrent state
    ISO_CHECK(m.convBytes() == 64 && m.ssmBytes() == 128);
    fake.cur_ = 3;
    fake.seq_pages_ = 5;
    ISO_CHECK(m.curSeq() == 3 && m.seqPages() == 5);
    const std::int32_t phys[] = {7, 8};
    m.commitSeq(3);
    m.dropPending(2);
    m.mapPages(1, 4, phys);
    ISO_CHECK(fake.calls == (std::vector<std::string>{"commitSeq(3)", "dropPending(2)", "mapPages(1,4,7;8;)"}));
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("contract.cache", {{"interface_shape", testInterfaceShape}, {"calls_through_the_interface", testCallsThroughTheInterface}}, argc, argv);
}
