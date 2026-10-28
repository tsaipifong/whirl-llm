// whirl-iso-model-kv-tests: src/model/qwen35/kv.cpp built alone (ARCH-1 8.5 isolation, arch025-lane2 N9 S2c): the KV pool,
// the sequence table and the tier page layout over a host-side device stand-in. Model's KV / sequence functions
// (allocKvPool .. restoreSeqSnapshot, model_kv.cpp) stay in whirl-model-tests.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/kv.h"
#include "model/qwen35/model_error.h"
#include "whirl/kv_layout.h"
#include "whirl/model_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

namespace q = whirl::qwen35;

// 8 layers, attention every 4th (layers 3 and 7), 4 KV heads x 128
q::Config cfg8() {
    q::Config c;
    c.n_layer = 8;
    c.interval = 4;
    c.n_head_kv = 4;
    c.head_dim = 128;
    return c;
}

struct Ledger {
    std::vector<std::uint64_t> sizes;
    std::vector<q::DevPtr> freed;
    q::DevPtr next = 0x1000;
    q::KvPool::Device dev() {
        return {[this](std::uint64_t n) { sizes.push_back(n); return next += 0x100000; }, [this](q::DevPtr p) { freed.push_back(p); }};
    }
};

void prep(q::KvPool& p, const q::Config& c) {
    p.kcache.assign(c.n_layer, 0);
    p.vcache.assign(c.n_layer, 0);
    p.kscale.assign(c.n_layer, 0);
    p.vscale.assign(c.n_layer, 0);
}

std::string codeOf(const std::function<void()>& f) {
    try {
        f();
    } catch (const q::ModelError& e) {
        return e.code();
    }
    return "";
}

void testFixFormatOnce() {
    q::KvPool p;
    ISO_CHECK(p.kv_fmt == whirl::KvFormat::q8);  // the former initializer
    p.fixFormat(whirl::KvFormat::f16);
    ISO_CHECK(p.kv_fmt == whirl::KvFormat::f16);
    ISO_CHECK(codeOf([&] { p.fixFormat(whirl::KvFormat::q4); }) == "KvFormatFixed");
    ISO_CHECK(p.kv_fmt == whirl::KvFormat::f16);
}

void testAllocateOrderAndSizes() {
    // q8: K int8, V int8, K scales, V scales; MTP layer first, then layers 3 and 7
    const q::Config c = cfg8();
    q::KvPool p;
    prep(p, c);
    Ledger l;
    const std::uint32_t pages = p.allocate(c, true, 300, l.dev());
    ISO_CHECK(pages == 2 && p.pool_pages == 2);
    const std::uint64_t rows = 2ull * q::kv_page, e = 4ull * 128;
    const std::vector<std::uint64_t> one = {rows * e * 2 / 2, rows * e * 2 / 2, rows * e / 32 * 2, rows * e / 32 * 2};
    std::vector<std::uint64_t> want;
    for (int layer = 0; layer < 3; ++layer) want.insert(want.end(), one.begin(), one.end());
    ISO_CHECK(l.sizes == want);
    ISO_CHECK(p.mtp_kc != 0 && p.kcache[3] != 0 && p.kcache[7] != 0 && p.kcache[0] == 0 && p.kscale[3] != 0);
    ISO_CHECK(p.mtp_kc < p.kcache[3] && p.kcache[3] < p.kcache[7]);
    ISO_CHECK(codeOf([&] { p.allocate(c, true, 300, l.dev()); }) == "KvPoolAllocated");
    ISO_CHECK(codeOf([&] { p.fixFormat(whirl::KvFormat::f16); }) == "KvFormatFixed");  // allocate fixes the format
}

void testAllocateF16NoScales() {
    const q::Config c = cfg8();
    q::KvPool p;
    prep(p, c);
    p.fixFormat(whirl::KvFormat::f16);
    Ledger l;
    p.allocate(c, false, 256, l.dev());
    ISO_CHECK(l.sizes.size() == 4);  // K, V of layers 3 and 7; no MTP, no scales
    ISO_CHECK(l.sizes[0] == 256ull * 512 * 4 / 2);
    ISO_CHECK(p.mtp_kc == 0 && p.kscale[3] == 0 && p.vscale[7] == 0);
}

void testReleaseArrays() {
    const q::Config c = cfg8();
    q::KvPool p;
    prep(p, c);
    Ledger l;
    p.allocate(c, true, 256, l.dev());
    const std::vector<q::DevPtr> order = {p.mtp_kc, p.mtp_vc, p.mtp_ks, p.mtp_vs, p.kcache[3], p.vcache[3], p.kscale[3], p.vscale[3],
                                          p.kcache[7], p.vcache[7], p.kscale[7], p.vscale[7]};
    p.releaseArrays(c, l.dev());
    ISO_CHECK(l.freed == order);
    ISO_CHECK(p.mtp_kc == 0 && p.kcache[3] == 0 && p.vscale[7] == 0);
    ISO_CHECK(p.pool_pages == 1);  // the caller's (Model::freeKvPool)
}

void testKvArraysLayout() {
    const q::Config c = cfg8();
    q::KvPool p;
    prep(p, c);
    p.fixFormat(whirl::KvFormat::q8v);
    Ledger l;
    p.allocate(c, true, 256, l.dev());
    const std::vector<q::KvArray> a = q::kvArrays(p, c, true);
    const whirl::kv::LayerRows L = whirl::kv::layerRows(512, whirl::KvFormat::q8v);
    ISO_CHECK(L.n == 3 && a.size() == 3 * L.n);  // q8v: K f16, V int8, V scales; layers 3, 7, then MTP
    ISO_CHECK(a[0].base == p.kcache[3] && a[1].base == p.vcache[3] && a[2].base == p.vscale[3]);
    ISO_CHECK(a[6].base == p.mtp_kc && a[8].base == p.mtp_vs);
    for (std::size_t i = 0; i < a.size(); ++i) ISO_CHECK(a[i].row == L.a[i % L.n].bytes);
    ISO_CHECK(q::kvArrays(p, c, false).size() == 2 * L.n);
}

void testMapPages() {
    q::SeqTable t;
    t.seq_pages = 4;
    t.ptab = 0x10000;
    t.seqs.resize(2);
    t.seqs[1].kv_base = 4;
    std::vector<q::DevPtr> dst;
    std::vector<std::size_t> len;
    const auto up = [&](q::DevPtr d, const void*, std::size_t n) { dst.push_back(d); len.push_back(n); };
    const std::int32_t phys[] = {7, 8};
    t.mapPages(1, 1, phys, up);
    ISO_CHECK(dst.size() == 1 && dst[0] == 0x10000 + (4 + 1) * 4 && len[0] == 8);
    t.mapPages(0, 0, {}, up);  // nothing to write: no upload
    ISO_CHECK(dst.size() == 1);
    const std::int32_t three[] = {1, 2, 3};
    ISO_CHECK(codeOf([&] { t.mapPages(0, 2, three, up); }) == "ContextTooLong");
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("model.kv",
                    {{"fix_format_once", testFixFormatOnce},
                     {"allocate_order_and_sizes", testAllocateOrderAndSizes},
                     {"allocate_f16_no_scales", testAllocateF16NoScales},
                     {"release_arrays", testReleaseArrays},
                     {"kv_arrays_layout", testKvArraysLayout},
                     {"map_pages", testMapPages}},
                    argc, argv);
}
