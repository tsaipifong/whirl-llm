// whirl-iso-page-pool-tests: src/cache/page_pool.cpp built alone (ARCH-1 8.5, arch025-lane2 leg 3) with its declared
// dependencies: the narrow model interface (cache/cache_model.h, contract.cache), the device contract with the
// NullBackend (host memory), and the owner check (threads: engine_thread.cpp + the server log). No ServerModel, no
// engine, no HIP layer: page refcounts, the free list, eviction, the page copy and copy-on-write run on host memory.
// SPDX-License-Identifier: Apache-2.0
#include "cache/page_pool.h"
#include "device/null_backend.h"
#include "iso/fake_cache_model.h"
#include "iso/iso_test.h"
#include "model/qwen35/kv.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

namespace sv = whirl::server;
namespace dv = whirl::device;
namespace q = whirl::qwen35;

std::uint8_t* at(dv::DevPtr p) { return reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(p)); }

struct NoEvict final : sv::PagePool::Evictor {
    int calls = 0;
    bool evictOne() override {
        ++calls;
        return false;
    }
};

// gives back one page it holds per call
struct Holder final : sv::PagePool::Evictor {
    sv::PagePool& pool;
    std::vector<std::int32_t> held;
    explicit Holder(sv::PagePool& p) : pool(p) {}
    bool evictOne() override {
        if (held.empty()) return false;
        const std::int32_t p = held.back();
        held.pop_back();
        pool.unref({&p, 1});
        return true;
    }
};

void testFreeListAndRefs() {
    dv::NullBackend ops;
    iso::FakeCacheModel m(ops, {}, 0, 0, 8, 10);
    sv::PagePool pool(ops, m);
    pool.init(3, {});
    ISO_CHECK(pool.pages() == 3 && pool.freeCount() == 3);
    ISO_CHECK(*pool.takeFree() == 0 && *pool.takeFree() == 1 && *pool.takeFree() == 2);  // increasing order
    ISO_CHECK(!pool.takeFree() && pool.refCount(1) == 1);
    pool.ref(1);
    const std::int32_t one = 1;
    pool.unref({&one, 1});
    ISO_CHECK(pool.refCount(1) == 1 && pool.freeCount() == 0);  // still held once
    pool.unref({&one, 1});
    ISO_CHECK(pool.refCount(1) == 0 && pool.freeCount() == 1 && *pool.takeFree() == 1);
}

void testTakeEvicts() {
    dv::NullBackend ops;
    iso::FakeCacheModel m(ops, {}, 0, 0, 8, 10);
    sv::PagePool pool(ops, m);
    pool.init(2, {});
    Holder h(pool);
    h.held = {*pool.takeFree(), *pool.takeFree()};
    const auto p = pool.take(h);  // nothing free: the evictor gives one back
    ISO_CHECK(p && *p == 1 && h.held.size() == 1);
    NoEvict none;
    pool.ref(h.held[0]);  // page 0 held twice: unref by an evictor would not free it, and this one frees nothing
    ISO_CHECK(!pool.take(none) && none.calls == 1);
}

void testSharedFromAndPagesFor() {
    dv::NullBackend ops;
    iso::FakeCacheModel m(ops, {}, 0, 0, 8, 10);
    sv::PagePool pool(ops, m);
    pool.init(4, {});
    std::vector<std::int32_t> pg = {*pool.takeFree(), *pool.takeFree(), *pool.takeFree()};
    ISO_CHECK(!pool.sharedFrom(pg, 0));
    pool.ref(pg[2]);
    ISO_CHECK(pool.sharedFrom(pg, 0) && pool.sharedFrom(pg, 2 * q::kv_page + 1));  // the shared page is kept, not reused
    ISO_CHECK(!pool.sharedFrom(pg, 3 * q::kv_page + 1));                             // reuse past every page
    ISO_CHECK(sv::PagePool::pagesFor(0) == 0 && sv::PagePool::pagesFor(1) == 1 && sv::PagePool::pagesFor(q::kv_page) == 1 &&
              sv::PagePool::pagesFor(q::kv_page + 1) == 2);
    ISO_CHECK(pool.soleOwned(pg) == 2);
}

void testCopyEveryArray() {
    // copy(dst, src): every KV array's page row block, kv_page * row bytes at base + page * that
    dv::NullBackend ops;
    iso::FakeCacheModel m(ops, {}, 0, 0, 8, 10);
    const std::uint64_t r0 = 4, r1 = 12, pages = 3;
    const dv::DevPtr a0 = ops.malloc(pages * q::kv_page * r0), a1 = ops.malloc(pages * q::kv_page * r1);
    for (std::uint64_t i = 0; i < pages * q::kv_page * r0; ++i) at(a0)[i] = static_cast<std::uint8_t>(i * 7);
    for (std::uint64_t i = 0; i < pages * q::kv_page * r1; ++i) at(a1)[i] = static_cast<std::uint8_t>(i * 13);
    sv::PagePool pool(ops, m);
    pool.init(static_cast<std::uint32_t>(pages), {{a0, r0}, {a1, r1}});
    pool.copy(2, 0);
    ISO_CHECK(std::memcmp(at(a0 + 2 * q::kv_page * r0), at(a0), q::kv_page * r0) == 0);
    ISO_CHECK(std::memcmp(at(a1 + 2 * q::kv_page * r1), at(a1), q::kv_page * r1) == 0);
    ISO_CHECK(at(a0 + q::kv_page * r0)[0] == static_cast<std::uint8_t>(q::kv_page * r0 * 7));  // page 1 untouched
    ops.free(a0);
    ops.free(a1);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("page_pool",
                    {{"free_list_and_refs", testFreeListAndRefs},
                     {"take_evicts", testTakeEvicts},
                     {"shared_from_and_pages_for", testSharedFromAndPagesFor},
                     {"copy_every_array", testCopyEveryArray}},
                    argc, argv);
}
