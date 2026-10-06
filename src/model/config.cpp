// Config from GGUF metadata, constant tables, environment helpers.
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (Config, tables, tuneBucket).

#include "whirl/model.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

namespace whirl {

std::optional<std::string> envGet(std::string_view name) {
    const std::string key = std::string("WHIRL_") + std::string(name);
    // _dupenv_s keeps MSVC quiet about getenv
    char* buf = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&buf, &len, key.c_str()) == 0 && buf != nullptr) {
        std::string v(buf);
        std::free(buf);
        return v;
    }
    return std::nullopt;
}

bool envFlag(std::string_view name, bool def) {
    auto v = envGet(name);
    if (!v) return def;
    return *v != "0";
}

}  // namespace whirl

namespace whirl::qwen35 {

bool gv_group = true;
std::uint32_t gv_nmax = 0;

static_assert(n_choices == 2 * 24 + 8);

std::size_t tuneBucket(std::uint32_t n) {
    if (n <= 32) return 3;
    if (n <= 48) return 4;
    if (n <= 64) return 5;
    if (n <= 96) return 6;
    if (n <= 128) return 2;
    if (n <= 192) return 7;
    if (n <= 256) return 8;
    if (n <= 384) return 9;
    if (n <= 768) return 0;
    if (n <= tune_small_max) return 10;
    return 1;
}

GemvR defaultGemvR() { return {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}; }
GemvR gfx1151GemvR() { return {1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}; }

GemvW defaultGemvW() {
    GemvW t{};
    auto ti = [](GgmlType ty) { return static_cast<std::size_t>(ty); };
    for (std::uint32_t nt = 2; nt <= max_small_batch; ++nt) {
        t[ti(GgmlType::q4_k)][nt] = nt >= 8 ? 5 : nt >= 4 ? 6 : nt >= 3 ? 1 : 0;
        t[ti(GgmlType::q5_k)][nt] = nt >= 6 ? 5 : nt >= 3 ? 6 : 0;
        t[ti(GgmlType::iq4_xs)][nt] = nt >= 12 ? 8 : nt >= 6 ? 6 : nt >= 3 ? 1 : 0;
        t[ti(GgmlType::mxfp4)][nt] = nt >= 14 ? 8 : nt >= 4 ? 6 : nt >= 3 ? 1 : 0;
        t[ti(GgmlType::q3_k)][nt] = nt >= 5 ? 5 : 0;
        t[ti(GgmlType::iq3_s)][nt] = nt >= 12 ? 5 : 0;
        t[ti(GgmlType::q6_k)][nt] = nt >= 8 ? 7 : nt >= 6 ? 5 : 0;
        t[ti(GgmlType::iq4_nl)][nt] = nt >= 9 ? 4 : 0;
    }
    return t;
}

GemvR defaultGemvWHead() {
    GemvR t{};
    for (std::uint32_t nt = 6; nt <= max_small_batch; ++nt) t[nt] = 2;
    return t;
}

const char* opClassName(OpClass c) {
    static const char* names[] = {"embed", "norm", "quant", "matmul", "attn", "gdn_prep", "gdn_rec", "gdn_norm", "misc", "head"};
    return names[static_cast<int>(c)];
}

Config Config::fromGguf(const gguf::File& f) {
    const std::string arch(f.getStringOr("general.architecture", ""));
    const bool moe = arch == "qwen35moe";
    if (!moe && arch != "qwen35") throw ModelError("UnsupportedArch", "general.architecture = '" + arch + "' (WHIRL supports qwen35 and qwen35moe only)");
    auto K = [&](const char* s) { return arch + "." + s; };
    auto u32 = [&](const char* s) { return static_cast<std::uint32_t>(f.getUint(K(s))); };
    auto u32or = [&](const char* s, std::uint64_t d) { return static_cast<std::uint32_t>(f.getUintOr(K(s), d)); };
    Config c;
    const std::uint32_t n_all = u32("block_count");
    const std::uint32_t n_nextn = u32or("nextn_predict_layers", 0);
    const gguf::TensorInfo* emb = f.tensor("token_embd.weight");
    if (!emb) throw ModelError("MissingTensor", "token_embd.weight");
    const std::uint32_t n_ff_exp = moe ? u32("expert_feed_forward_length") : 0;
    c.n_layer = n_all - n_nextn;
    c.n_embd = u32("embedding_length");
    c.n_ff = moe ? u32or("expert_shared_feed_forward_length", n_ff_exp) : u32("feed_forward_length");
    c.n_head = u32("attention.head_count");
    c.n_head_kv = u32("attention.head_count_kv");
    c.head_dim = u32("attention.key_length");
    c.n_rot = u32("rope.dimension_count");
    c.rope_base = static_cast<float>(f.getFloat(K("rope.freq_base")));
    c.eps = static_cast<float>(f.getFloat(K("attention.layer_norm_rms_epsilon")));
    c.interval = u32or("full_attention_interval", 4);
    c.d_conv = u32("ssm.conv_kernel");
    c.d_state = u32("ssm.state_size");
    c.n_k_heads = u32("ssm.group_count");
    c.n_v_heads = u32("ssm.time_step_rank");
    c.d_inner = u32("ssm.inner_size");
    c.n_vocab = static_cast<std::uint32_t>(emb->ne[1]);
    c.n_nextn = n_nextn;
    c.moe = moe;
    c.n_expert = moe ? u32("expert_count") : 0;
    c.n_expert_used = moe ? u32("expert_used_count") : 0;
    c.n_ff_exp = n_ff_exp;
    // vision: multi-section RoPE sections (image prompts only)
    if (const gguf::Value* v = f.find(K("rope.dimension_sections")); v != nullptr && v->type == gguf::ValueType::array &&
        (v->arr.elem == gguf::ValueType::i32 || v->arr.elem == gguf::ValueType::u32) && v->arr.len >= 3) {
        for (std::size_t i = 0; i < std::min<std::size_t>(4, v->arr.len); ++i) std::memcpy(&c.rope_sections[i], v->arr.raw.data() + i * 4, 4);
    }
    return c;
}

namespace {

// "nt:v,nt:v" items with 2 <= nt <= max_small_batch accepted by `ok`.
template <class Fn>
void parseNtList(std::string_view list, std::uint32_t def, Fn apply) {
    std::size_t p = 0;
    while (p < list.size()) {
        std::size_t e = list.find(',', p);
        if (e == std::string_view::npos) e = list.size();
        const std::string item(list.substr(p, e - p));
        p = e + 1;
        if (item.empty()) continue;
        const std::size_t c = item.find(':');
        const std::uint32_t nt = static_cast<std::uint32_t>(std::stoul(item.substr(0, c)));
        const std::uint32_t v = c == std::string::npos ? def : static_cast<std::uint32_t>(std::stoul(item.substr(c + 1)));
        if (nt >= 2 && nt <= max_small_batch) apply(nt, v);
    }
}

}  // namespace

void parseGemvR(Model& m, std::string_view list) {
    parseNtList(list, 1, [&](std::uint32_t nt, std::uint32_t r) {
        if (r == 1 || r == 2 || r == 4) m.gemv_r[nt] = static_cast<std::uint8_t>(r);
    });
}

void parseGemvW(Model& m, std::string_view list) {
    parseNtList(list, 0, [&](std::uint32_t nt, std::uint32_t v) {
        if (v < n_gemvw)
            for (auto& row : m.gemv_w) row[nt] = static_cast<std::uint8_t>(v);
    });
}

void parseGemvWHead(Model& m, std::string_view list) {
    parseNtList(list, 0, [&](std::uint32_t nt, std::uint32_t v) {
        if (v < n_gemvw) m.gemv_w_head[nt] = static_cast<std::uint8_t>(v);
    });
}

void applyGemvEnv(Model& m) {
    if (auto v = envGet("GEMV_R")) parseGemvR(m, *v);
    if (auto v = envGet("GEMV_W")) parseGemvW(m, *v);
    if (auto v = envGet("GEMV_WH")) parseGemvWHead(m, *v);
}

MtpDefaults mtpDefaults(bool moe, hip::Arch arch) {
    if (arch == hip::Arch::gfx1151) {
        // Radeon 8060S (UMA, measured there): the draft counts, the draft-vocabulary default and
        // the n-gram minimum match measured the same as on the R9700 (within noise), but a verify
        // row costs far more: ~+11 ms per extra row on a ~95 ms 1-draft cycle at 9k context
        // (slope ~0.12, vs 0.015 on the R9700). With the R9700 prior the first n-gram proposal
        // of a request took 8 drafts and lost the cycle (agent tool-output replay: -6%).
        // qwen35moe (Ornith): 0.12 gave +1.6% on that replay but -2.2% on a file edit that
        // n-gram drafts best: it keeps the R9700 prior (and one MTP draft per cycle, also
        // fastest there: 2 fixed / 3 auto were 1-12% slower).
        if (moe) return {1, 0.0f, 0, false, 3, 0.015f};
        return {8, 0.0f, 0, true, 3, 0.12f};
    }
    // dense qwen35 (Qwen3.8-27B): chained drafts stay good out to 10 -> cost-model
    // auto, max 8. qwen35moe (Ornith-35B-A3B): one draft per cycle is fastest.
    if (moe) return {1, 0.0f, 0, false};
    return {8, 0.0f, 0, true};
}

}  // namespace whirl::qwen35
