// Config from GGUF metadata, constant tables, environment helpers.
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (Config, tables, tuneBucket).

#include "whirl/model.h"

#include <algorithm>
#include <array>
#include <cmath>
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

    // hyper-parameter sanity: these values size buffers and kernel grids
    auto bad = [&](const std::string& what) { throw ModelError("UnsupportedConfig", arch + ": " + what); };
    auto inRange = [&](const char* name, std::uint64_t v, std::uint64_t lo, std::uint64_t hi) {
        if (v < lo || v > hi) bad(std::string(name) + " = " + std::to_string(v) + " (supported " + std::to_string(lo) + ".." + std::to_string(hi) + ")");
    };
    // the keys were narrowed to 32 bits above: a value >= 2^32 must not pass as its low bits
    for (const char* key : {"block_count", "nextn_predict_layers", "embedding_length", "feed_forward_length", "attention.head_count",
                            "attention.head_count_kv", "attention.key_length", "rope.dimension_count", "full_attention_interval", "ssm.conv_kernel",
                            "ssm.state_size", "ssm.group_count", "ssm.time_step_rank", "ssm.inner_size", "expert_feed_forward_length",
                            "expert_shared_feed_forward_length", "expert_count", "expert_used_count"})
        if (f.has(K(key)) && f.getUint(K(key)) > 0xFFFFFFFFull) bad(std::string(key) + " does not fit in 32 bits");
    if (n_nextn >= n_all) bad("nextn_predict_layers (" + std::to_string(n_nextn) + ") must be smaller than block_count (" + std::to_string(n_all) + ")");
    inRange("block_count - nextn_predict_layers", c.n_layer, 1, 1024);
    inRange("nextn_predict_layers", c.n_nextn, 0, 8);
    inRange("embedding_length", c.n_embd, 256, 65536);
    if (c.n_embd % 256 != 0) bad("embedding_length " + std::to_string(c.n_embd) + " is not a multiple of 256");
    inRange(moe ? "expert_shared_feed_forward_length" : "feed_forward_length", c.n_ff, 1, 1u << 20);
    inRange("attention.head_count", c.n_head, 1, 1024);
    inRange("attention.head_count_kv", c.n_head_kv, 1, c.n_head);
    // flash decoding keeps up to 8 query heads per KV head in shared memory
    if (c.n_head % c.n_head_kv != 0 || c.n_head / c.n_head_kv > 8)
        bad("attention.head_count / head_count_kv = " + std::to_string(c.n_head) + " / " + std::to_string(c.n_head_kv) +
            " (must divide evenly, at most 8 query heads per KV head)");
    // attention kernels keep one head (<= 256 values) per shared-memory row
    inRange("attention.key_length", c.head_dim, 32, 256);
    if (c.head_dim % 32 != 0) bad("attention.key_length " + std::to_string(c.head_dim) + " is not a multiple of 32");
    if (const std::uint64_t vl = f.getUintOr(K("attention.value_length"), c.head_dim); vl != c.head_dim)
        bad("attention.value_length " + std::to_string(vl) + " differs from key_length " + std::to_string(c.head_dim));
    inRange("rope.dimension_count", c.n_rot, 2, c.head_dim);
    if (c.n_rot % 2 != 0) bad("rope.dimension_count " + std::to_string(c.n_rot) + " is odd");
    inRange("full_attention_interval", c.interval, 1, 1024);
    inRange("ssm.conv_kernel", c.d_conv, 2, 16);
    inRange("ssm.state_size", c.d_state, 1, 256);
    inRange("ssm.group_count", c.n_k_heads, 1, 1024);
    inRange("ssm.time_step_rank", c.n_v_heads, 1, 1024);
    inRange("ssm.inner_size", c.d_inner, 1, 1u << 20);
    if (c.n_v_heads % c.n_k_heads != 0) bad("ssm.time_step_rank must be a multiple of ssm.group_count");
    if (c.d_inner % c.n_v_heads != 0) bad("ssm.inner_size must be a multiple of ssm.time_step_rank");
    if (c.headV() > 256) bad("ssm.inner_size / time_step_rank = " + std::to_string(c.headV()) + " (supported up to 256)");
    if (emb->ne[1] < 1 || emb->ne[1] > (1u << 22))
        bad("vocabulary size " + std::to_string(emb->ne[1]) + " (token_embd.weight ne[1]; supported 1..4194304)");
    if (!(c.rope_base > 0.0f) || !(c.eps > 0.0f) || !std::isfinite(c.rope_base) || !std::isfinite(c.eps))
        bad("rope.freq_base and layer_norm_rms_epsilon must be positive");
    if (moe) {
        inRange("expert_count", c.n_expert, 1, 1024);
        inRange("expert_used_count", c.n_expert_used, 1, std::min<std::uint32_t>(64, c.n_expert));
        inRange("expert_feed_forward_length", c.n_ff_exp, 256, 65536);
        if (c.n_ff_exp % 256 != 0) bad("expert_feed_forward_length " + std::to_string(c.n_ff_exp) + " is not a multiple of 256");
    }
    return c;
}

void Config::validateTensors(const gguf::File& f) const {
    using u64 = std::uint64_t;
    using Shape = std::array<u64, 4>;
    const Config& c = *this;
    auto fmt = [](const Shape& s, std::uint32_t nd) {
        std::string r = "[";
        for (std::uint32_t d = 0; d < nd; ++d) r += (d ? ", " : "") + std::to_string(s[d]);
        return r + "]";
    };
    // ne[] padded with 1s, so [n] and [n, 1] are the same shape
    auto expect = [&](const std::string& name, Shape want) {
        const gguf::TensorInfo* t = f.tensor(name);
        if (!t) throw ModelError("MissingTensor", name);
        if (t->nbytes() == 0)
            throw ModelError("UnsupportedTensorType", name + ": type " + gguf::typeName(t->type) + " with row length " + std::to_string(t->ne[0]));
        const Shape got = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
        if (got != want) {
            std::uint32_t nd = 4;
            while (nd > 1 && want[nd - 1] == 1) --nd;
            throw ModelError("UnsupportedTensorShape", name + ": shape " + fmt(got, std::max<std::uint32_t>(t->n_dims, 1)) + ", expected " +
                                                           fmt(want, nd) + " from the model's hyper-parameters");
        }
    };
    auto opt = [&](const std::string& name, Shape want) {
        if (f.tensor(name)) expect(name, want);
    };
    const u64 E = c.n_embd, V = c.n_vocab, HD = c.head_dim;
    expect("token_embd.weight", {E, V, 1, 1});
    opt("output.weight", {E, V, 1, 1});
    expect("output_norm.weight", {E, 1, 1, 1});

    auto bn = [](std::uint32_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    auto ffn = [&](std::uint32_t il) {
        const u64 F = c.n_ff;
        if (c.moe) {
            const u64 FE = c.n_ff_exp, NE = c.n_expert;
            expect(bn(il, "ffn_gate_shexp.weight"), {E, F, 1, 1});
            expect(bn(il, "ffn_up_shexp.weight"), {E, F, 1, 1});
            expect(bn(il, "ffn_down_shexp.weight"), {F, E, 1, 1});
            expect(bn(il, "ffn_gate_inp.weight"), {E, NE, 1, 1});
            opt(bn(il, "ffn_gate_inp_shexp.weight"), {E, 1, 1, 1});
            expect(bn(il, "ffn_gate_exps.weight"), {E, FE, NE, 1});
            expect(bn(il, "ffn_up_exps.weight"), {E, FE, NE, 1});
            expect(bn(il, "ffn_down_exps.weight"), {FE, E, NE, 1});
        } else {
            expect(bn(il, "ffn_gate.weight"), {E, F, 1, 1});
            expect(bn(il, "ffn_up.weight"), {E, F, 1, 1});
            expect(bn(il, "ffn_down.weight"), {F, E, 1, 1});
        }
    };
    // gated attention: attn_q holds each head's query and its gate
    auto attn = [&](std::uint32_t il) {
        const u64 QH = static_cast<u64>(c.n_head) * HD, KH = static_cast<u64>(c.n_head_kv) * HD;
        expect(bn(il, "attn_q.weight"), {E, 2 * QH, 1, 1});
        expect(bn(il, "attn_k.weight"), {E, KH, 1, 1});
        expect(bn(il, "attn_v.weight"), {E, KH, 1, 1});
        expect(bn(il, "attn_output.weight"), {QH, E, 1, 1});
        expect(bn(il, "attn_q_norm.weight"), {HD, 1, 1, 1});
        expect(bn(il, "attn_k_norm.weight"), {HD, 1, 1, 1});
    };
    for (std::uint32_t il = 0; il < c.n_layer; ++il) {
        expect(bn(il, "attn_norm.weight"), {E, 1, 1, 1});
        expect(bn(il, "post_attention_norm.weight"), {E, 1, 1, 1});
        ffn(il);
        if (c.isAttn(il)) {
            attn(il);
        } else {
            const u64 CH = c.convCh(), DI = c.d_inner, NV = c.n_v_heads;
            expect(bn(il, "attn_qkv.weight"), {E, CH, 1, 1});
            expect(bn(il, "attn_gate.weight"), {E, DI, 1, 1});
            expect(bn(il, "ssm_beta.weight"), {E, NV, 1, 1});
            expect(bn(il, "ssm_alpha.weight"), {E, NV, 1, 1});
            expect(bn(il, "ssm_out.weight"), {DI, E, 1, 1});
            expect(bn(il, "ssm_conv1d.weight"), {c.d_conv, CH, 1, 1});
            expect(bn(il, "ssm_dt.bias"), {NV, 1, 1, 1});
            expect(bn(il, "ssm_a"), {NV, 1, 1, 1});
            expect(bn(il, "ssm_norm.weight"), {c.headV(), 1, 1, 1});
        }
    }
    // MTP block: loaded only when its eh_proj is present (as in Model::load)
    if (c.n_nextn > 0 && f.tensor(bn(c.n_layer, "nextn.eh_proj.weight")) != nullptr) {
        const std::uint32_t il = c.n_layer;
        attn(il);
        expect(bn(il, "attn_norm.weight"), {E, 1, 1, 1});
        expect(bn(il, "post_attention_norm.weight"), {E, 1, 1, 1});
        ffn(il);
        expect(bn(il, "nextn.eh_proj.weight"), {2 * E, E, 1, 1});
        expect(bn(il, "nextn.enorm.weight"), {E, 1, 1, 1});
        expect(bn(il, "nextn.hnorm.weight"), {E, 1, 1, 1});
        opt(bn(il, "nextn.shared_head_norm.weight"), {E, 1, 1, 1});
    }
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

MtpDefaults mtpDefaults(bool moe) {
    // dense qwen35 (Qwen3.8-27B): chained drafts stay good out to 10 -> cost-model
    // auto, max 8. qwen35moe (Ornith-35B-A3B): one draft per cycle is fastest.
    if (moe) return {1, 0.0f, 0, false};
    return {8, 0.0f, 0, true};
}

}  // namespace whirl::qwen35
