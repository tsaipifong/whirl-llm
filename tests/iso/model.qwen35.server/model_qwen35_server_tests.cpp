// whirl-iso-model-qwen35-server-tests: the module model.qwen35.server (src/model/qwen35/server_model.cpp: the ServerModel /
// ServerVision adapters makeQwen35Backend / makeQwen35Vision over qwen35::Model and vision::Vision) built over the REAL model core
// (model.core and everything below it, the sources of tests/iso/model.core) and the server contract headers (contract.server:
// src/server/backend.h, vision_iface.h), WITHOUT the server, the engine, the server log or the KV host tier (ARCH-1 8.5
// isolation, arch025-n8b3, PLAN N8 batch 3: progress/arch025-n8b3.PRIOR_ART.md design 4). The adapter runs on an UNLOADED Model
// (no GPU): the cases are the host-only forwarding (sizes, cutoffs, capture rows, vision rows, profile class names) and the
// top-k launch through the launch recorder / fake kernel handles of tests/iso/fake_hip.cpp.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/backend.h"
#include "whirl/model.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// the embedded draft vocabulary (whirl-bin2c output in the product build; as tests/iso/model.core): three ids
extern const unsigned char whirl_draft_vocab_64k[] = {0x01, 0x00, 0x00, 0x00, 0x10, 0x27, 0x00, 0x00, 0x78, 0x56, 0x34, 0x12};
extern const unsigned long long whirl_draft_vocab_64k_size = sizeof(whirl_draft_vocab_64k);

// The launch recorder and fake kernel handles of the HIP link double (tests/iso/fake_hip.cpp).
namespace whirl::hip::isofake {
struct Launch {
    Function f;
    Dim3 grid, block;
};
void recordLaunches(bool on);
std::vector<Launch> recordedLaunches();
void fakeFunctions(bool on);
}  // namespace whirl::hip::isofake

namespace {

namespace q = whirl::qwen35;
namespace fake = whirl::hip::isofake;
namespace srv = whirl::server;

struct World {
    std::unique_ptr<q::Model> m = std::make_unique<q::Model>();
    std::unique_ptr<srv::ServerModel> b;
    World() {
        fake::fakeFunctions(true);  // the adapter looks up its top-k kernel on construction
        b = srv::makeQwen35Backend(*m);
        fake::fakeFunctions(false);
    }
};

void testSizes() {
    World w;
    w.m->pool_pages = 5;
    w.m->seq_pages = 7;
    w.m->max_batch = 9;
    w.m->snap_sets = 2;
    w.m->cur_seq = 1;
    ISO_CHECK(w.b->poolPages() == 5 && w.b->seqPages() == 7 && w.b->maxBatch() == 9 && w.b->snapSets() == 2 && w.b->curSeq() == 1);
    ISO_CHECK(!w.b->gdnReplay());
    w.m->gdn_replay = true;
    ISO_CHECK(w.b->gdnReplay());
    ISO_CHECK(&w.b->cfg() == &w.m->cfg);
}

void testDraftCutoffAndSample() {
    World w;
    w.b->setDraftCutoff(0.25f, 3);
    ISO_CHECK(w.m->draft_p_min == 0.25f && w.m->draft_n_min == 3);
    w.m->draft_samp.resize(2);
    whirl::spec::DraftSample ds;
    ds.on = true;
    w.b->setDraftSample(1, ds);
    ISO_CHECK(w.m->draft_samp[1].on && !w.m->draft_samp[0].on);
    w.b->setDraftSample(2, ds);  // beyond the table: ignored
    ISO_CHECK(w.m->draft_samp.size() == 2);
}

void testCaptureAndDflash() {
    World w;
    w.m->cap_n = 4;
    ISO_CHECK(w.b->captureN() == 0);  // no capture buffer: no captured rows
    w.m->cap_buf = 0x1000;
    ISO_CHECK(w.b->captureN() == 4);
    ISO_CHECK(w.b->dflashMaxDrafts() == 0 && w.b->dflashRing() == 0);  // no DFlash drafter
}

void testVision() {
    World w;
    ISO_CHECK(!w.b->hasMrope() && !w.b->visRowsAllocated());
    w.m->k.attn_prep_m = reinterpret_cast<whirl::hip::Function>(0x10);
    w.m->rpos_buf = 0x2000;
    ISO_CHECK(w.b->hasMrope() && w.b->visRowsAllocated());
    w.m->rpos_buf = 0;
    const q::VisMap vm{};
    w.b->setVisMap(1, &vm);
    ISO_CHECK(w.m->vis_seq[1] == &vm && w.m->vis_seq[0] == nullptr);
}

void testProfileNames() {
    World w;
    for (std::size_t i = 0; i < q::n_classes; ++i) ISO_CHECK(std::strcmp(w.b->profileClassName(i), q::opClassName(static_cast<q::OpClass>(i))) == 0);
}

void testTopkLaunch() {
    World w;
    w.m->cfg.n_vocab = 1000;
    fake::fakeFunctions(true);
    const whirl::hip::Function topk = w.m->module.getFunction("topk_rows");  // the handle the adapter looked up
    fake::fakeFunctions(false);
    fake::recordLaunches(true);
    w.b->topkRows(2, 3, 40, 1.0f, 0, 0, 0);
    const auto l = fake::recordedLaunches();
    fake::recordLaunches(false);
    ISO_CHECK(l.size() == 1 && l[0].f == topk && l[0].grid.x == 3 && l[0].block.x == 1024);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("model.qwen35.server",
                    {{"sizes", testSizes},
                     {"draft_cutoff_and_sample", testDraftCutoffAndSample},
                     {"capture_and_dflash", testCaptureAndDflash},
                     {"vision", testVision},
                     {"profile_names", testProfileNames},
                     {"topk_launch", testTopkLaunch}},
                    argc, argv);
}
