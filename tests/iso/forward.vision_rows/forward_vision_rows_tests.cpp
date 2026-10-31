// whirl-iso-forward-vision-rows-tests: the module forward.vision_rows (src/model/qwen35/forward/vision_rows.cpp: per-sequence
// image maps, multi-section RoPE row positions, image embedding runs) built with only forward.state (context.cpp) below it and
// no other forward source (ARCH-1 8.5 isolation, arch025-n8b3, PLAN N8 batch 3): a call from vision_rows.cpp into another
// forward file, class Model, the engine or the server log is LNK2019 here. Host-only cases on the HIP layer's link double
// tests/iso/fake_hip.cpp (its allocation and launch recorders).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/context.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Link doubles for this test program only (as tests/iso/forward.state): Context's TuneTable member defaults (model.core).
namespace whirl::qwen35 {
GemvR defaultGemvR() { return {}; }
GemvW defaultGemvW() { return {}; }
GemvR defaultGemvWHead() { return {}; }
}  // namespace whirl::qwen35

// The recorders of the HIP link double (tests/iso/fake_hip.cpp).
namespace whirl::hip::isofake {
struct Launch {
    Function f;
    Dim3 grid, block;
};
void recordAllocs(bool on);
std::vector<std::uint64_t> recordedSizes();
void recordLaunches(bool on);
std::vector<Launch> recordedLaunches();
}  // namespace whirl::hip::isofake

namespace {

namespace q = whirl::qwen35;
namespace fake = whirl::hip::isofake;

struct VisProbe : q::Context {
    using q::Context::injectRows;
    using q::Context::visAny;
    using q::Context::visBegin;
    using q::Context::visCommit;
    using q::Context::visEnsure;
    using q::Context::visOf;
    using q::Context::visRow;
    using q::Context::visRowsCur;
};

std::unique_ptr<VisProbe> probe() {
    auto c = std::make_unique<VisProbe>();
    c->cfg.n_embd = 4;
    c->max_batch = 32;
    c->rpos_host.assign(static_cast<std::size_t>(c->max_batch) * 3, -1);
    return c;
}

std::string codeOf(const std::function<void()>& f) {
    try {
        f();
    } catch (const q::ModelError& e) {
        return e.code();
    } catch (const std::logic_error&) {
        return "device";  // reached the HIP link double
    }
    return "";
}

void testVisAnyOf() {
    auto c = probe();
    ISO_CHECK(!c->visAny());
    ISO_CHECK(c->visOf(0) == nullptr);
    const q::VisSpan sp{2, 4, 2, 2, 0, nullptr};
    const q::VisMap active{std::span<const q::VisSpan>(&sp, 1)};
    const q::VisMap empty{};
    c->vis_single = &empty;
    ISO_CHECK(c->visAny());                // a map is set, even an inactive one
    ISO_CHECK(c->visOf(0) == nullptr);     // inactive -> none
    c->vis_single = &active;
    ISO_CHECK(c->visOf(0) == &active);     // no sequences: the single map
    c->vis_single = nullptr;
    c->vis_seq[1] = &active;
    ISO_CHECK(c->visAny());
    c->seqs.resize(2);
    ISO_CHECK(c->visOf(0) == nullptr && c->visOf(1) == &active);  // sequences: per-sequence maps
}

void testVisBegin() {
    auto c = probe();
    c->mrope_on = c->rpos_rows = true;
    c->rdelta = 5;
    c->n_inj = 3;
    c->visBegin();
    ISO_CHECK(!c->mrope_on && !c->rpos_rows && c->rdelta == 0 && c->n_inj == 0);
}

void testVisRowRuns() {
    auto c = probe();
    std::vector<float> emb(4 * 4);
    // positions 2..5 = one 2x2 image; 6.. text (delta after the image = 0 + 4 - 2 = 2)
    const q::VisSpan sp{2, 4, 2, 2, 0, emb.data()};
    const q::VisMap vm{std::span<const q::VisSpan>(&sp, 1)};
    c->visRow(nullptr, 0, 9);  // plain text row: all three sections = the position
    ISO_CHECK(c->rpos_host[0] == 9 && c->rpos_host[1] == 9 && c->rpos_host[2] == 9);
    ISO_CHECK(c->n_inj == 0);
    c->visRow(&vm, 1, 2);
    c->visRow(&vm, 2, 3);  // the next row of the same image continues the run
    c->visRow(&vm, 3, 4);
    ISO_CHECK(c->n_inj == 1 && c->inj[0].row == 1 && c->inj[0].n == 3 && c->inj[0].src == emb.data());
    ISO_CHECK(c->rpos_host[3 * 3 + 0] == 2 && c->rpos_host[3 * 3 + 1] == 3 && c->rpos_host[3 * 3 + 2] == 2);  // idx 2: {t, t+1, t+0}
    c->visRow(&vm, 5, 5);  // a gap in the rows (row 4 skipped): a new run
    ISO_CHECK(c->n_inj == 2 && c->inj[1].row == 5 && c->inj[1].n == 1 && c->inj[1].src == emb.data() + 3 * 4);
    c->visRow(&vm, 6, 6);  // text after the image: position 6 - delta 2, no run
    ISO_CHECK(c->n_inj == 2 && c->rpos_host[18] == 4 && c->rpos_host[19] == 4 && c->rpos_host[20] == 4);
    const q::VisSpan raw{0, 2, 1, 2, 0, nullptr};
    const q::VisMap unenc{std::span<const q::VisSpan>(&raw, 1)};
    ISO_CHECK(codeOf([&] { c->visRow(&unenc, 7, 0); }) == "ImageNotEncoded");
}

void testVisRowTooManyRuns() {
    auto c = probe();
    std::vector<float> emb(4 * 4);
    const q::VisSpan sp{0, 4, 2, 2, 0, emb.data()};
    const q::VisMap vm{std::span<const q::VisSpan>(&sp, 1)};
    c->rpos_host.assign(3 * 600, 0);
    c->n_inj = q::Context::max_inj - 1;
    c->visRow(&vm, 0, 0);
    ISO_CHECK(c->n_inj == q::Context::max_inj);
    ISO_CHECK(codeOf([&] { c->visRow(&vm, 2, 2); }) == "TooManyImageRuns");
}

void testVisCommit() {
    auto c = probe();
    ISO_CHECK(codeOf([&] { c->visCommit(1); }) == "MropeUnsupported");
    c->k.attn_prep_m = reinterpret_cast<whirl::hip::Function>(0x10);
    c->k.set_rpos = reinterpret_cast<whirl::hip::Function>(0x20);
    fake::recordLaunches(true);
    c->visCommit(q::max_small_batch);  // up to max_small_batch rows: positions as kernel arguments, one launch
    const auto l = fake::recordedLaunches();
    fake::recordLaunches(false);
    ISO_CHECK(l.size() == 1 && l[0].f == c->k.set_rpos && l[0].grid.x == 1 && l[0].block.x == 64);
    ISO_CHECK(c->mrope_on && c->rpos_rows);
    auto d = probe();
    d->k.attn_prep_m = c->k.attn_prep_m;
    d->k.set_rpos = c->k.set_rpos;
    ISO_CHECK(codeOf([&] { d->visCommit(q::max_small_batch + 1); }) == "device");  // more rows: the upload
    ISO_CHECK(!d->mrope_on);
}

void testVisEnsureRelease() {
    auto c = probe();
    c->rpos_host.clear();
    fake::recordAllocs(true);
    c->visEnsure();
    const auto sizes = fake::recordedSizes();
    ISO_CHECK(sizes.size() == 1 && sizes[0] == 32ull * 3 * 4);
    ISO_CHECK(c->rpos_buf != 0 && c->rpos_host.size() == 32u * 3);
    c->visEnsure();  // once
    ISO_CHECK(fake::recordedSizes().size() == 1);
    const q::VisSpan sp{0, 1, 1, 1, 0, nullptr};
    const q::VisMap vm{std::span<const q::VisSpan>(&sp, 1)};
    c->vis_single = &vm;
    c->visRelease();  // a map is still set: kept
    ISO_CHECK(c->rpos_buf != 0);
    c->vis_single = nullptr;
    c->visRelease();  // the sync reaches the link double (caught there), then freed
    ISO_CHECK(c->rpos_buf == 0);
    fake::recordAllocs(false);
}

void testVisRowsCurNoMap() {
    auto c = probe();
    c->mrope_on = true;
    c->visRowsCur(4, 0);  // no image map: only visBegin
    ISO_CHECK(!c->mrope_on && c->n_inj == 0 && c->rpos_buf == 0);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.vision_rows",
                    {{"vis_any_of", testVisAnyOf},
                     {"vis_begin", testVisBegin},
                     {"vis_row_runs", testVisRowRuns},
                     {"vis_row_too_many_runs", testVisRowTooManyRuns},
                     {"vis_commit", testVisCommit},
                     {"vis_ensure_release", testVisEnsureRelease},
                     {"vis_rows_cur_no_map", testVisRowsCurNoMap}},
                    argc, argv);
}
