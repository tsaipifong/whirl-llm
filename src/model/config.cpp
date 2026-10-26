// Config from GGUF metadata, constant tables, environment helpers.
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (Config, tables, tuneBucket).

#include "whirl/model.h"
#include "config/env_snapshot.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace whirl {

// envGet / envFlag: src/config/env_snapshot.cpp (arch025-N1)

}  // namespace whirl

namespace whirl::qwen35 {

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

static_assert(std::tuple_size_v<GemvR> == std::tuple_size_v<decltype(device::Profile::gemv_r)>);
GemvR defaultGemvR() { return device::profile(hip::Arch::gfx1201).gemv_r; }

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
    // fail closed on shapes the DeltaNet kernels do not implement (only models with DeltaNet layers)
    if (c.interval > 1 && !c.gdnShapeSupported())
        bad("DeltaNet layers need ssm.conv_kernel = 4, ssm.state_size = 128 and ssm.inner_size / time_step_rank a multiple of 32; got " +
            std::to_string(c.d_conv) + ", " + std::to_string(c.d_state) + ", " + std::to_string(c.headV()));
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
        if (r == 1 || r == 2 || r == 4) m.tuneTable().gemv_r[nt] = static_cast<std::uint8_t>(r);
    });
}

void parseGemvW(Model& m, std::string_view list) {
    parseNtList(list, 0, [&](std::uint32_t nt, std::uint32_t v) {
        if (v < n_gemvw)
            for (auto& row : m.tuneTable().gemv_w) row[nt] = static_cast<std::uint8_t>(v);
    });
}

void parseGemvWHead(Model& m, std::string_view list) {
    parseNtList(list, 0, [&](std::uint32_t nt, std::uint32_t v) {
        if (v < n_gemvw) m.tuneTable().gemv_w_head[nt] = static_cast<std::uint8_t>(v);
    });
}

void applyGemvEnv(Model& m) {
    const auto env = whirl::config::snapshotOf(m);  // the load's (arch025-N1)
    if (const auto& v = env->gemv_r) parseGemvR(m, *v);
    if (const auto& v = env->gemv_w) parseGemvW(m, *v);
    if (const auto& v = env->gemv_wh) parseGemvWHead(m, *v);
}

// mtpDefaults: src/spec/policy.cpp (arch025-M4)

namespace {

// FIX-DFL (FR-DFL P2-1): the drafter forward (src/model/dflash.cpp) sizes its scratch, rings and kernel
// launches from the metadata (DFlashConfig), while WeightLoader::mat / vec upload whatever shape the file has
// and matmul writes w.nrows outputs per row. Any tensor that disagrees with the metadata would write past
// a scratch buffer or make a kernel read past a weight. So every tensor must have exactly the shape the
// metadata gives (like Config::validateTensors for the target), matrices a known type whose rows cut into
// whole blocks, vectors f32. Headers only: a mismatch is refused before anything is loaded and the caller
// keeps the MTP drafts (dflashIncompatible -> dflash::choose).
void validateDFlashTensors(const DFlashConfig& c, const gguf::File& f) {
    using u64 = std::uint64_t;
    using Shape = std::array<u64, 4>;
    auto bad = [](const std::string& what) { throw ModelError("DraftModelMismatch", what); };
    // block_count: at least one layer (dflashSetup reads layers[0]), at most kDFlashMaxLayers (the
    // official drafters have 5; a huge count would allocate rings for nothing)
    if (c.n_layer == 0 || c.n_layer > kDFlashMaxLayers)
        bad("block_count " + std::to_string(c.n_layer) + " (1.." + std::to_string(kDFlashMaxLayers) + " supported)");
    if (c.head_dim % 2 != 0) bad("head_dim " + std::to_string(c.head_dim) + " is odd (RoPE rotates pairs)");
    if (c.n_ff == 0) bad("feed_forward_length 0");
    auto fmt = [](const Shape& s) {
        std::uint32_t nd = 4;
        while (nd > 1 && s[nd - 1] == 1) --nd;
        std::string r = "[";
        for (std::uint32_t d = 0; d < nd; ++d) r += (d ? ", " : "") + std::to_string(s[d]);
        return r + "]";
    };
    auto shape = [&](const std::string& name, Shape want) -> const gguf::TensorInfo& {
        const gguf::TensorInfo* ti = f.tensor(name);
        if (!ti) bad("missing tensor " + name);
        const Shape got = {ti->ne[0], ti->ne[1], ti->ne[2], ti->ne[3]};
        if (got != want) bad(name + " is " + fmt(got) + ", expected " + fmt(want) + " from the drafter's metadata");
        return *ti;
    };
    auto mat = [&](const std::string& name, u64 ncols, u64 nrows) {
        const gguf::TensorInfo& ti = shape(name, {ncols, nrows, 1, 1});
        if (ti.nbytes() == 0) bad(name + ": type " + gguf::typeName(ti.type) + " with row length " + std::to_string(ncols));
    };
    auto vec = [&](const std::string& name, Shape want) {
        const gguf::TensorInfo& ti = shape(name, want);
        if (ti.type != GgmlType::f32) bad(name + ": type " + gguf::typeName(ti.type) + " (f32 expected)");
    };
    const u64 E = c.n_embd, HD = c.head_dim, QW = static_cast<u64>(c.n_head) * HD, KW = static_cast<u64>(c.n_head_kv) * HD;
    const u64 F = c.n_ff, K = c.conv_k, DS = 2 * K * (E / c.conv_group);
    // fc (shape checked by validate): the ingest splits it into n_target_layers column blocks of E values
    // (Model::dflashIngest: ptr + j * row_bytes / nl, ncols E), so each block must start on a whole quant
    // block; MXFP4 is refused: it is repacked per 256 values with one reference exponent per full row
    // (WeightLoader::uploadMx), which a column block does not have (FR-DFL P3-7; no MXFP4 drafter exists to test)
    if (const gguf::TensorInfo* fc = f.tensor("fc.weight")) {
        if (fc->type == GgmlType::mxfp4) bad("fc.weight is MXFP4 (the drafter splits fc into column blocks; use Q8_0 / f16 / f32)");
        const gguf::TypeTraits* tt = gguf::typeTraits(fc->type);
        if (!tt || fc->nbytes() == 0 || E % tt->block_size != 0)
            bad("fc.weight: type " + gguf::typeName(fc->type) + " cannot be split into column blocks of " + std::to_string(E));
    }
    vec("enc.output_norm.weight", {E, 1, 1, 1});
    vec("output_norm.weight", {E, 1, 1, 1});
    mat("selector_hidden.weight", E, c.sel_rank);
    for (std::uint32_t il = 0; il < c.n_layer; ++il) {
        auto n = [il](const char* s) { return "blk." + std::to_string(il) + "." + s; };
        vec(n("attn_norm.weight"), {E, 1, 1, 1});
        vec(n("ffn_norm.weight"), {E, 1, 1, 1});
        vec(n("attn_q_norm.weight"), {HD, 1, 1, 1});
        vec(n("attn_k_norm.weight"), {HD, 1, 1, 1});
        mat(n("attn_q.weight"), E, QW);
        mat(n("attn_k.weight"), E, KW);
        mat(n("attn_v.weight"), E, KW);
        mat(n("attn_output.weight"), QW, E);
        mat(n("ffn_gate.weight"), E, F);
        mat(n("ffn_up.weight"), E, F);
        mat(n("ffn_down.weight"), F, E);
        // dynamic convolution: projection rows = 2 sides x K taps x (E / group) channels (dflash_conv reads
        // dyn[t * ds + c / group + G (tap + K side)]; dflashSetup sizes dyn from layer 0), static kernels
        // [E][K][2] (base[(side K + tap) E + c])
        mat(n("attn_conv_proj.weight"), E, DS);
        mat(n("ffn_conv_proj.weight"), E, DS);
        vec(n("attn_conv_base"), {E, K, 2, 1});
        vec(n("ffn_conv_base"), {E, K, 2, 1});
    }
}

}  // namespace

DFlashConfig DFlashConfig::fromGguf(const gguf::File& f) {
    const std::string arch(f.getStringOr("general.architecture", ""));
    if (arch != "dflash") throw ModelError("UnsupportedArch", "draft model general.architecture = '" + arch + "' (expected dflash)");
    auto K = [&](const char* s) { return arch + "." + s; };
    auto u32 = [&](const char* s) { return static_cast<std::uint32_t>(f.getUint(K(s))); };
    auto u32or = [&](const char* s, std::uint64_t d) { return static_cast<std::uint32_t>(f.getUintOr(K(s), d)); };
    DFlashConfig c;
    c.n_layer = u32("block_count");
    c.n_embd = u32("embedding_length");
    c.n_head = u32("attention.head_count");
    c.n_head_kv = u32or("attention.head_count_kv", c.n_head);
    c.head_dim = u32or("attention.key_length", c.n_head ? c.n_embd / c.n_head : 0);
    c.n_ff = u32("feed_forward_length");
    c.block = u32or("block_size", 16);
    c.conv_k = u32or("conv_kernel_size", 0);
    c.conv_group = u32or("conv_group_size", 0);
    c.sel_rank = u32or("selector_rank", 0);
    c.sel_topk = u32or("selector_top_k", 0);
    c.window = u32or("attention.sliding_window", 0);
    c.rope_base = static_cast<float>(f.has(K("rope.freq_base")) ? f.getFloat(K("rope.freq_base")) : 10000.0);
    c.eps = static_cast<float>(f.getFloat(K("attention.layer_norm_rms_epsilon")));
    if (!f.has("tokenizer.ggml.mask_token_id")) throw ModelError("MissingKey", "tokenizer.ggml.mask_token_id");
    c.mask_id = static_cast<std::uint32_t>(f.getUint("tokenizer.ggml.mask_token_id"));
    if (!f.has(K("target_layers"))) throw ModelError("MissingKey", K("target_layers"));
    for (const std::int64_t v : f.intArray(K("target_layers"))) {
        if (v < 0) throw ModelError("BadKey", K("target_layers"));
        c.target_layers.push_back(static_cast<std::uint32_t>(v));
    }
    return c;
}

void DFlashConfig::validate(const Config& t, const gguf::File& f) const {
    auto bad = [](const std::string& what) { throw ModelError("DraftModelMismatch", what); };
    if (n_embd != t.n_embd) bad("hidden size " + std::to_string(n_embd) + " != target " + std::to_string(t.n_embd));
    if (target_layers.empty()) bad("no target layers");
    for (const std::uint32_t l : target_layers)
        if (l == 0 || l > t.n_layer) bad("target layer id " + std::to_string(l) + " outside 1.." + std::to_string(t.n_layer));
    if (sel_topk == 0 || sel_rank == 0) bad("not a DFlash2 drafter (no candidate selector)");
    if (block < 2 || block > 16) bad("block size " + std::to_string(block) + " (2..16 supported)");
    if (conv_k == 0 || conv_group == 0 || n_embd % conv_group != 0) bad("dynamic convolution parameters");
    if (n_head == 0 || n_head_kv == 0 || n_head % n_head_kv != 0 || head_dim == 0) bad("attention heads");
    if (mask_id >= t.n_vocab) bad("mask token " + std::to_string(mask_id) + " >= vocabulary " + std::to_string(t.n_vocab));
    // kernel limits (kernels/dflash.hip): attention keys = context ring (the sliding window, 2048 without
    // one) + block, one thread per head dimension, the selector lattice has 32 x top-k threads per block
    const std::uint64_t keys = static_cast<std::uint64_t>(window > 0 ? window : 2048) + block;
    if (keys > static_cast<std::uint64_t>(kernels::kDFlashMaxKeys))
        bad("attention window " + std::to_string(window) + " + block " + std::to_string(block) + " = " + std::to_string(keys) +
            " keys > the drafter attention limit " + std::to_string(kernels::kDFlashMaxKeys));
    if (head_dim > 256) bad("head_dim " + std::to_string(head_dim) + " > 256 (one thread per dimension)");
    if (sel_topk > 16) bad("selector top-k " + std::to_string(sel_topk) + " > 16 (dflash_lattice: 32 x top-k threads <= 512)");
    validateDFlashTensors(*this, f);  // FIX-DFL (FR-DFL P2-1): every tensor's shape / type
    const gguf::TensorInfo* fc = f.tensor("fc.weight");
    if (!fc || fc->ne[0] != static_cast<std::uint64_t>(target_layers.size()) * n_embd || fc->ne[1] != n_embd)
        bad("fc.weight is not [" + std::to_string(target_layers.size() * n_embd) + ", " + std::to_string(n_embd) + "]");
    for (const char* name : {"selector_predecessor.weight", "selector_successor.weight"}) {
        const gguf::TensorInfo* sel = f.tensor(name);
        if (!sel || sel->ne[0] != sel_rank || sel->ne[1] != t.n_vocab)
            bad(std::string(name) + " is not [rank, target vocabulary " + std::to_string(t.n_vocab) + "]");
    }
    // FIX-K97 (FR-K97 P2-1): dflash_lattice decodes the two codebooks with one type flag (Q8_0 or f32) and one
    // row size, so any other type (f16, k-quants) would be read as f32 -> garbage lattice and reads past the
    // tensor. Require both to be f32 or both Q8_0 (same type, same shape -> same row bytes).
    const gguf::TensorInfo* sp = f.tensor("selector_predecessor.weight");
    const gguf::TensorInfo* ss = f.tensor("selector_successor.weight");
    auto cb_ok = [](GgmlType ty) { return ty == GgmlType::f32 || ty == GgmlType::q8_0; };
    if (!cb_ok(sp->type) || !cb_ok(ss->type) || sp->type != ss->type)
        bad("selector codebooks are " + gguf::typeName(sp->type) + " / " + gguf::typeName(ss->type) +
            "; the drafter lattice needs both f32 or both Q8_0");
}

std::string dflashIncompatible(const gguf::File& target, const gguf::File& draft) {
    try {
        const Config t = Config::fromGguf(target);
        DFlashConfig::fromGguf(draft).validate(t, draft);
    } catch (const std::exception& ex) {
        return ex.what();
    }
    return {};
}

// FR-DFL P3-2: --draft-model / WHIRL_DRAFT_MODEL that cannot be opened or parsed (a typo, a stale
// variable, a truncated download) is one more "no fitting drafter" case: the reason, the caller warns
// and keeps the MTP drafts (docs/usage.md), instead of the exception stopping whirl-server / whirl chat.
std::string dflashIncompatibleFile(const gguf::File& target, const std::string& draft_path, std::optional<gguf::File>* opened) {
    std::optional<gguf::File> df;
    try {
        df.emplace(gguf::File::open(draft_path));
    } catch (const std::exception& ex) {
        return std::string("cannot open the drafter: ") + ex.what();
    }
    std::string why = dflashIncompatible(target, *df);
    if (why.empty() && opened) *opened = std::move(df);
    return why;
}

}  // namespace whirl::qwen35
