// whirl-iso-model-types-tests: the header-only module model.types (include/whirl/model_types.h,
// src/model/qwen35/model_error.h) compiled with only its declared dependency headers (whirl/hip.h, whirl/kv_format.h)
// and linked to nothing (ARCH-1 8.5 isolation, arch025-lane2 leg 6, DESIGN section 6). The checks pin the module's
// inline rules the server core and the model's lower modules use without the Model class.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/model_error.h"
#include "whirl/model_types.h"

#include <string>

namespace {

using whirl::KvFormat;
namespace q = whirl::qwen35;

// a Qwen3.5-27B-shaped config (64 layers, attention every 4th, DeltaNet 16 k-heads / 48 v-heads, d_inner 6144)
q::Config cfg27() {
    q::Config c;
    c.n_layer = 64, c.interval = 4, c.d_conv = 4, c.d_state = 128, c.n_k_heads = 16, c.n_v_heads = 48, c.d_inner = 6144;
    return c;
}

void testConfigShape() {
    q::Config c = cfg27();
    ISO_CHECK(!c.isAttn(0) && !c.isAttn(2) && c.isAttn(3) && c.isAttn(63));  // (il + 1) % interval == 0
    ISO_CHECK(c.convCh() == 2u * 16 * 128 + 6144);
    ISO_CHECK(c.headV() == 128u);
    ISO_CHECK(c.gdnShapeSupported());
    c.d_conv = 3;
    ISO_CHECK(!c.gdnShapeSupported());  // the kernels are written for a 4-tap conv
    c = cfg27();
    c.d_inner = 48 * 48;  // headV 48: not a multiple of 32
    ISO_CHECK(!c.gdnShapeSupported());
    ISO_CHECK(std::string(c.archName()) == "qwen35");
    c.moe = true;
    ISO_CHECK(std::string(c.archName()) == "qwen35moe");
}

void testKvPerToken() {
    q::LoadEstimate e;
    e.kv_f16 = 4096, e.kv_q8v = 3072, e.kv_q8h = 2560, e.kv_q4 = 1536;
    ISO_CHECK(e.kvPerToken(KvFormat::f16) == 4096u);
    ISO_CHECK(e.kvPerToken(KvFormat::q8v) == 3072u);
    ISO_CHECK(e.kvPerToken(KvFormat::q8h) == 2560u);
    ISO_CHECK(e.kvPerToken(KvFormat::q8) == 2560u);  // q8 has q8h's rows
    ISO_CHECK(e.kvPerToken(KvFormat::q4) == 1536u);
}

void testConstants() {
    // values the server's sizing and the control area layout depend on
    ISO_CHECK(q::kv_page == 256u);
    ISO_CHECK(q::max_small_batch == 16u);
    ISO_CHECK(q::max_batch_default == 4096u && q::max_batch_limit == 16384u);
    ISO_CHECK(q::gdn_max_snap + 1 == q::gdn_max_seg && q::max_ng_drafts == q::gdn_max_snap);
    ISO_CHECK(q::ctl_drafts == q::ctl_rows + 16 && q::ctl_probs == q::ctl_drafts + 16 && q::ctl_stop < q::ctl_words);
}

void testModelError() {
    const q::ModelError a("UnsupportedArch", "general.architecture = 'x'");
    ISO_CHECK(a.code() == "UnsupportedArch");
    ISO_CHECK(std::string(a.what()) == "UnsupportedArch: general.architecture = 'x'");
    const q::ModelError b("MissingTensor");
    ISO_CHECK(std::string(b.what()) == "MissingTensor");
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("model.types",
                    {{"config_shape", testConfigShape},
                     {"kv_per_token", testKvPerToken},
                     {"constants", testConstants},
                     {"model_error", testModelError}},
                    argc, argv);
}
