// whirl-iso-model-tune-tests: src/model/qwen35/tune_table.cpp built alone (ARCH-1 8.5 isolation, arch025-lane2): the tune
// cache text as pure functions over the matrices. The writeTune == tuneText cross-check over a whole Model stays in
// whirl-model-tests (testTuneTable*); this program checks the module's own text rules.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/tune_table.h"

#include <array>
#include <string>
#include <vector>

namespace {

namespace q = whirl::qwen35;

q::Mat mat(q::GgmlType ty, std::uint32_t rows, std::uint32_t cols, std::uint8_t base) {
    q::Mat m;
    m.ty = ty;
    m.nrows = rows;
    m.ncols = cols;
    for (std::size_t b = 0; b < m.tune.size(); ++b) m.tune[b] = static_cast<std::uint8_t>(base + b);
    return m;
}

void testRoundTrip() {
    q::Mat a = mat(q::GgmlType::q4_k, 4096, 5120, 3), b = mat(q::GgmlType::q6_k, 248320, 5120, 40);
    std::vector<q::Mat*> mats{&a, &b};
    const std::string text = q::tuneText(mats);
    ISO_CHECK(!text.empty());
    q::Mat a2 = mat(q::GgmlType::q4_k, 4096, 5120, 0), b2 = mat(q::GgmlType::q6_k, 248320, 5120, 0);
    std::vector<q::Mat*> mats2{&a2, &b2};
    ISO_CHECK(q::applyTuneText(text, a.tune.size(), mats2));
    ISO_CHECK(a2.tune == a.tune && b2.tune == b.tune);
    ISO_CHECK(q::tuneText(mats2) == text);  // deterministic
}

void testSameShapeOneLine() {
    // one line per distinct (bucket, type, rows, cols): a repeated shape adds no text
    q::Mat a = mat(q::GgmlType::q4_k, 64, 128, 1), b = a;
    std::vector<q::Mat*> one{&a}, two{&a, &b};
    ISO_CHECK(q::tuneText(one) == q::tuneText(two));
}

void testStaleCacheRefused() {
    q::Mat a = mat(q::GgmlType::q4_k, 64, 128, 1);
    std::vector<q::Mat*> mats{&a};
    const std::string text = q::tuneText(mats);
    q::Mat other = mat(q::GgmlType::q4_k, 64, 256, 9);  // a shape the text does not have
    std::vector<q::Mat*> mats2{&other};
    ISO_CHECK(!q::applyTuneText(text, a.tune.size(), mats2));
    ISO_CHECK(!q::applyTuneText("", a.tune.size(), mats2));
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("model.tune",
                    {{"round_trip", testRoundTrip}, {"same_shape_one_line", testSameShapeOneLine}, {"stale_cache_refused", testStaleCacheRefused}},
                    argc, argv);
}
