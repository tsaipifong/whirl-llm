// whirl-iso-kv-format-tests: the header-only module kv_format (include/whirl/kv_format.h, kv_layout.h) compiled with nothing but its own headers
// and linked to nothing (ARCH-1 8.5 isolation, arch025-iso4). The checks pin the format names, the format predicates and the page-row sizes
// that the model's pool layout, the tiers and the numerics choice all read.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "whirl/kv_format.h"
#include "whirl/kv_layout.h"

#include <string>
#include <string_view>

namespace {

using whirl::KvFormat;
constexpr KvFormat k_all[] = {KvFormat::f16, KvFormat::q8, KvFormat::q8h, KvFormat::q8v, KvFormat::q4};

void testNames() {
    ISO_CHECK(std::string(whirl::kvFormatName(KvFormat::f16)) == "f16" && std::string(whirl::kvFormatName(KvFormat::q8)) == "q8");
    ISO_CHECK(std::string(whirl::kvFormatName(KvFormat::q8h)) == "q8h" && std::string(whirl::kvFormatName(KvFormat::q8v)) == "q8v");
    ISO_CHECK(std::string(whirl::kvFormatName(KvFormat::q4)) == "q4");
    for (KvFormat k : k_all) {
        const auto back = whirl::kvFormatFromName(whirl::kvFormatName(k));  // the inverse of the name
        ISO_CHECK(back && *back == k);
    }
    ISO_CHECK(!whirl::kvFormatFromName("auto") && !whirl::kvFormatFromName("") && !whirl::kvFormatFromName("Q8") && !whirl::kvFormatFromName("q8 "));
    static_assert(whirl::kvFormatFromName("q4").has_value() && std::string_view(whirl::kvFormatName(KvFormat::q8v)) == "q8v");
}

void testPredicates() {
    // (quantized, rotated, K f16, 4-bit) per format
    struct Row {
        KvFormat k;
        bool quant, rot, kf16, b4;
    };
    const Row rows[] = {{KvFormat::f16, false, false, false, false}, {KvFormat::q8, true, false, false, false},
                        {KvFormat::q8h, true, true, false, false},   {KvFormat::q8v, true, false, true, false},
                        {KvFormat::q4, true, true, false, true}};
    for (const Row& r : rows) {
        ISO_CHECK(whirl::kvQuant(r.k) == r.quant);
        ISO_CHECK(whirl::kvRotated(r.k) == r.rot);
        ISO_CHECK(whirl::kvKf16(r.k) == r.kf16);
        ISO_CHECK(whirl::kvIs4bit(r.k) == r.b4);
    }
}

void testLayerRows() {
    namespace kv = whirl::kv;
    const std::uint64_t e = 1024;  // n_head_kv * head_dim of one layer
    kv::LayerRows f = kv::layerRows(e, KvFormat::f16);
    ISO_CHECK(f.n == 2 && f.a[0].part == kv::Part::k && f.a[1].part == kv::Part::v && f.a[0].bytes == 2 * e && f.a[1].bytes == 2 * e);
    kv::LayerRows q = kv::layerRows(e, KvFormat::q8);
    ISO_CHECK(q.n == 4 && q.a[0].bytes == e && q.a[1].bytes == e && q.a[2].part == kv::Part::k_scale && q.a[2].bytes == e / 32 * 2 &&
              q.a[3].part == kv::Part::v_scale && q.a[3].bytes == e / 32 * 2);
    kv::LayerRows v = kv::layerRows(e, KvFormat::q8v);  // K stays f16: no K scale array
    ISO_CHECK(v.n == 3 && v.a[0].bytes == 2 * e && v.a[1].bytes == e && v.a[2].part == kv::Part::v_scale);
    kv::LayerRows h = kv::layerRows(e, KvFormat::q4);  // 4-bit K and V
    ISO_CHECK(h.n == 4 && h.a[0].bytes == e / 2 && h.a[1].bytes == e / 2);
    // layerBytes is the same sum as the rows (q4 counts its scales)
    for (KvFormat k : k_all) {
        ISO_CHECK(kv::layerBytes(e, k) == kv::layerRows(e, k).total());
    }
    ISO_CHECK(kv::layerBytes(e, KvFormat::q8) < kv::layerBytes(e, KvFormat::f16) && kv::layerBytes(e, KvFormat::q4) < kv::layerBytes(e, KvFormat::q8));
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("kv_format", {{"names", testNames}, {"predicates", testPredicates}, {"layer_rows", testLayerRows}}, argc, argv);
}
