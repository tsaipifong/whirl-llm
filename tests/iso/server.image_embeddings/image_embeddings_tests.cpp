// whirl-iso-server-image-embeddings-tests: the server engine's image input (src/server/engine_vision.cpp, class
// ImageEmbeddings) built with only its declared dependencies: the server log (src/server/log.cpp, its request lines),
// writeFile of src/common/util.cpp (WHIRL_VIS_DUMP) with the platform layer it calls, and the device / server / vision
// contracts it reaches through ServerModel&, device::Backend& and ServerVision* (headers; whirl/vision.h's
// lendRegions is inline). Test doubles: tests/iso/fake_server_model.h, device::NullBackend, a fake encoder below.
// No engine, no encoder (src/vision), no HIP runtime, no tokenizer (ARCH-1 8.5 isolation, arch025-lane1 leg 8). The
// checks are whirl-server-tests image_embeddings (tests/server/server_tests.cpp testImageEmbeddings).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "iso/fake_server_model.h"
#include "server/engine_vision.h"
#include "device/null_backend.h"
#include "whirl/vision.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

using namespace whirl;
using namespace whirl::server;
using whirl::device::NullBackend;
using whirl::test::FakeServerModel;

namespace {

// tests/server/server_tests.cpp MockVision without the image decoder: projDim 8, the embedding of a known hash
class FakeVision final : public ServerVision {
public:
    vision::Prepared prepare(std::span<const std::uint8_t>) const override { return {}; }
    std::uint32_t projDim() const override { return 8; }
    vision::EncodeStats encode(const vision::Prepared& p, std::span<const vision::Region>, float* out, std::size_t n) override {
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p.hash[i % 32]) + static_cast<float>(i);
        ++encodes;
        loaded_ = true;
        vision::EncodeStats st;
        st.n_patches = p.nTokens() * 4;
        return st;
    }
    bool loaded() const override { return loaded_; }
    bool idleTick() override { return false; }
    std::uint32_t idleS() const override { return 60; }
    int encodes = 0;

private:
    bool loaded_ = false;
};

vision::Prepared img(std::uint8_t h) {
    vision::Prepared p;
    p.nx = 2;
    p.ny = 2;
    p.hash[0] = h;
    return p;
}

void testLendRegions() {
    std::array<std::array<std::uint64_t, 2>, 16> raw{};
    raw[0] = {0x1000, 64};
    raw[1] = {0x2000, 128};
    raw[2] = {0x3000, 256};
    const std::vector<vision::Region> r = vision::lendRegions(raw, 2);
    ISO_CHECK(r.size() == 2u && r[0].ptr == DevPtr{0x1000} && r[0].len == 64u && r[1].ptr == DevPtr{0x2000} && r[1].len == 128u);
    ISO_CHECK(vision::lendRegions(raw, 0).empty());
}

void testCache() {
    NullBackend nb;
    FakeServerModel model(nb, 64);
    {
        ImageEmbeddings none(model, nb);
        none.attach(VisionConfig{});
        ISO_CHECK(!none.wakeNeeded());
        ISO_CHECK(none.config().vis == nullptr);
    }
    FakeVision fv;
    VisionConfig vc;
    vc.vis = &fv;
    vc.cache_bytes = 256;  // two 2x2-token images (4 tokens x projDim 8 x 4 bytes = 128 bytes each)
    ImageEmbeddings ie(model, nb);
    ie.attach(vc);
    ISO_CHECK(ie.config().vis == &fv);
    const vision::Prepared a = img(1), b = img(2), c = img(3);
    EmbEntry* ea = ie.get(a, 1, 0);
    ISO_CHECK(ea->data.size() == 32u && ea->pins == 1u && ie.encoded() == 1u && ie.bytes() == 128u);
    ISO_CHECK(ea->data[0] == 1.0f && ea->data[1] == 1.0f);  // the fake's embedding of hash {1, 0, ...}
    ISO_CHECK(ie.get(a, 2, 1) == ea);  // hit
    ISO_CHECK(ie.hits() == 1u && ie.encoded() == 1u && ea->pins == 2u && fv.encodes == 1);
    ISO_CHECK(ie.wakeNeeded());  // the fake encoder is loaded after its first encode
    ie.unpin(ea);
    ie.unpin(ea);
    ISO_CHECK(ea->pins == 0u);
    EmbEntry* eb = ie.get(b, 3, 0);
    EmbEntry* ec = ie.get(c, 4, 1);  // 384 > 256: the unpinned a (least recently used) goes
    ISO_CHECK(ie.bytes() == 256u && ie.encoded() == 3u);
    ISO_CHECK(ie.get(a, 5, 2) != nullptr);  // re-encoded; every entry pinned -> nothing evicted
    ISO_CHECK(ie.encoded() == 4u && ie.bytes() == 384u);
    // a request's spans: prepare pins the entry (hit) and points the span at it, done releases the pins
    JobVis jv;
    jv.preps.push_back(b);
    jv.spans.resize(1);
    jv.spans[0].start = 0;
    jv.spans[0].n = 4;
    jv.ents.resize(1, nullptr);
    const std::uint32_t pins_b = eb->pins;
    ie.prepare(0, &jv, 9, 0);
    ISO_CHECK(jv.ents[0] == eb && jv.spans[0].emb == eb->data.data() && eb->pins == pins_b + 1 && ie.hits() == 2u);
    ISO_CHECK(model.last_vismap == &jv.vmap);  // the slot's image map goes to the model
    ie.prepare(0, &jv, 9, 4);  // span already before `from`: no embeddings needed
    ISO_CHECK(jv.spans[0].emb == nullptr);
    ie.done(0, &jv);
    ISO_CHECK(jv.ents[0] == nullptr && eb->pins == pins_b);
    ie.prepare(0, nullptr, 9, 0);  // no images: the slot's map is cleared, nothing else
    ISO_CHECK(model.last_vismap == nullptr);
    ie.done(0, nullptr);
    ISO_CHECK(ie.encoded() == 4u);
    (void)ec;
    ie.idle();  // the fake's idleTick never releases: no state change
    ISO_CHECK(ie.bytes() == 384u);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.image_embeddings", {{"lend_regions", testLendRegions}, {"cache", testCache}}, argc, argv);
}
