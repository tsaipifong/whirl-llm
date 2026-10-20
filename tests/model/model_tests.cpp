// Host-only unit tests of the model library (no GPU work): architecture
// whitelist, load-time hyper-parameter / tensor shape validation, tune buckets, GEMV tables, draft-count cost model, n-gram drafter, per-slot
// draft allocation (cycle cost, slot acceptance, allocDrafts).
// SPDX-License-Identifier: Apache-2.0

#include "../test_env.h"
#include "whirl/dflash_policy.h"
#include "whirl/gguf.h"
#include "whirl/kernels_abi.h"
#include "whirl/model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <fstream>
#include <functional>
#include <format>
#include <iterator>
#include <regex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace whirl;
namespace q = whirl::qwen35;

namespace {

int g_fail = 0, g_pass = 0;

void check(bool ok, const char* what) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", what);
    }
}
void check(bool ok, const std::string& what) { check(ok, what.c_str()); }

// Minimal GGUF v3 image builder.
struct GgufImage {
    std::vector<std::uint8_t> b;
    std::uint64_t n_kv = 0;
    std::vector<std::uint8_t> kv;
    void u32(std::vector<std::uint8_t>& v, std::uint32_t x) {
        for (int i = 0; i < 4; ++i) v.push_back(static_cast<std::uint8_t>(x >> (8 * i)));
    }
    void u64(std::vector<std::uint8_t>& v, std::uint64_t x) {
        for (int i = 0; i < 8; ++i) v.push_back(static_cast<std::uint8_t>(x >> (8 * i)));
    }
    void str(std::vector<std::uint8_t>& v, const std::string& s) {
        u64(v, s.size());
        v.insert(v.end(), s.begin(), s.end());
    }
    void kvString(const std::string& k, const std::string& val) {
        str(kv, k);
        u32(kv, 8);
        str(kv, val);
        ++n_kv;
    }
    void kvU32(const std::string& k, std::uint32_t val) {
        str(kv, k);
        u32(kv, 4);
        u32(kv, val);
        ++n_kv;
    }
    void kvF32(const std::string& k, float val) {
        str(kv, k);
        u32(kv, 6);
        std::uint32_t bits;
        std::memcpy(&bits, &val, 4);
        u32(kv, bits);
        ++n_kv;
    }
    void kvI32Array(const std::string& k, const std::vector<std::int32_t>& vals) {
        str(kv, k);
        u32(kv, 9);  // array
        u32(kv, 5);  // of i32
        u64(kv, vals.size());
        for (std::int32_t v : vals) u32(kv, static_cast<std::uint32_t>(v));
        ++n_kv;
    }
    // FIX-DFL: a tensor of up to 4 dimensions (type f32 = 0 by default, nbytes of payload)
    void tensorN(const std::string& name, const std::vector<std::uint64_t>& ne, std::uint32_t type = 0, std::uint64_t nbytes = 0) {
        str(ti, name);
        u32(ti, static_cast<std::uint32_t>(ne.size()));
        for (std::uint64_t x : ne) u64(ti, x);
        u32(ti, type);
        u64(ti, 0);
        ++n_t;
        data_bytes = std::max(data_bytes, nbytes);
    }
    // tensor infos only (an unknown type: no payload needed, shapes still readable)
    std::uint64_t n_t = 0;
    std::vector<std::uint8_t> ti;
    // a known type needs its payload inside the file: data_bytes zero bytes follow the header
    // (every tensor at offset 0; only the bounds are checked by File::parse)
    std::uint64_t data_bytes = 0;
    void tensor(const std::string& name, std::uint64_t ne0, std::uint64_t ne1, std::uint32_t type = 1000, std::uint64_t nbytes = 0) {
        str(ti, name);
        u32(ti, 2);
        u64(ti, ne0);
        u64(ti, ne1);
        u32(ti, type);  // 1000: unknown type
        u64(ti, 0);
        ++n_t;
        data_bytes = std::max(data_bytes, nbytes);
    }
    std::vector<std::uint8_t> build() {
        std::vector<std::uint8_t> out;
        u32(out, 0x46554747);  // "GGUF"
        u32(out, 3);
        u64(out, n_t);  // tensors
        u64(out, n_kv);
        out.insert(out.end(), kv.begin(), kv.end());
        out.insert(out.end(), ti.begin(), ti.end());
        while (out.size() % 32) out.push_back(0);
        out.insert(out.end(), data_bytes, 0);
        return out;
    }
};

void testArchWhitelist() {
    for (const char* arch : {"llama", "qwen3", "qwen2", "gemma3", "deepseek2"}) {
        GgufImage g;
        g.kvString("general.architecture", arch);
        g.kvU32(std::string(arch) + ".block_count", 4);
        const std::vector<std::uint8_t> img = g.build();
        const gguf::File f = gguf::File::parse(img);
        bool rejected = false;
        try {
            (void)q::Config::fromGguf(f);
        } catch (const q::ModelError& e) {
            rejected = e.code() == "UnsupportedArch";
        }
        check(rejected, (std::string("UnsupportedArch for ") + arch).c_str());
    }
    {
        // qwen35 passes the whitelist (then fails on the missing tensor)
        GgufImage g;
        g.kvString("general.architecture", "qwen35");
        g.kvU32("qwen35.block_count", 4);
        const std::vector<std::uint8_t> img = g.build();
        const gguf::File f = gguf::File::parse(img);
        std::string code;
        try {
            (void)q::Config::fromGguf(f);
        } catch (const q::ModelError& e) {
            code = e.code();
        }
        check(code == "MissingTensor", "qwen35 accepted by the whitelist");
    }
}

// Tiny synthetic qwen35 / qwen35moe model (all f32, zero payload) for the
// load-time validation (review SR-1 N1): hyper-parameters and tensor shapes.
struct TinyModel {
    bool moe = false;
    std::map<std::string, std::uint64_t> hp = {
        {"block_count", 3},  // 2 trunk layers (0: DeltaNet, 1: attention) + 1 MTP block
        {"nextn_predict_layers", 1},
        {"embedding_length", 256},
        {"feed_forward_length", 256},
        {"attention.head_count", 2},
        {"attention.head_count_kv", 1},
        {"attention.key_length", 32},
        {"attention.value_length", 32},
        {"rope.dimension_count", 16},
        {"full_attention_interval", 2},
        {"ssm.conv_kernel", 4},
        {"ssm.state_size", 128},  // the only state width the DeltaNet kernels implement
        {"ssm.group_count", 1},
        {"ssm.time_step_rank", 2},
        {"ssm.inner_size", 256},  // headV 128
    };
    std::uint64_t vocab = 16;
    // tensor name -> ne (overrides / removals applied by the tests)
    std::map<std::string, std::vector<std::uint64_t>> tensors;
    std::map<std::string, std::uint32_t> types;  // default f32

    std::uint64_t H(const char* k) const { return hp.at(k); }

    void makeTensors() {
        tensors.clear();
        const std::uint64_t E = H("embedding_length"), F = moe ? H("expert_shared_feed_forward_length") : H("feed_forward_length");
        const std::uint64_t HD = H("attention.key_length"), NH = H("attention.head_count"), NKV = H("attention.head_count_kv");
        const std::uint64_t DS = H("ssm.state_size"), NK = H("ssm.group_count"), NV = H("ssm.time_step_rank"), DI = H("ssm.inner_size");
        const std::uint64_t CH = 2 * NK * DS + DI;
        const std::uint64_t n_all = H("block_count"), n_layer = n_all - H("nextn_predict_layers");
        tensors["token_embd.weight"] = {E, vocab};
        tensors["output.weight"] = {E, vocab};
        tensors["output_norm.weight"] = {E};
        auto bn = [](std::uint64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
        auto ffn = [&](std::uint64_t l) {
            if (moe) {
                const std::uint64_t FE = H("expert_feed_forward_length"), NE = H("expert_count");
                tensors[bn(l, "ffn_gate_shexp.weight")] = {E, F};
                tensors[bn(l, "ffn_up_shexp.weight")] = {E, F};
                tensors[bn(l, "ffn_down_shexp.weight")] = {F, E};
                tensors[bn(l, "ffn_gate_inp.weight")] = {E, NE};
                tensors[bn(l, "ffn_gate_inp_shexp.weight")] = {E};
                tensors[bn(l, "ffn_gate_exps.weight")] = {E, FE, NE};
                tensors[bn(l, "ffn_up_exps.weight")] = {E, FE, NE};
                tensors[bn(l, "ffn_down_exps.weight")] = {FE, E, NE};
            } else {
                tensors[bn(l, "ffn_gate.weight")] = {E, F};
                tensors[bn(l, "ffn_up.weight")] = {E, F};
                tensors[bn(l, "ffn_down.weight")] = {F, E};
            }
        };
        auto attn = [&](std::uint64_t l) {
            tensors[bn(l, "attn_q.weight")] = {E, 2 * NH * HD};
            tensors[bn(l, "attn_k.weight")] = {E, NKV * HD};
            tensors[bn(l, "attn_v.weight")] = {E, NKV * HD};
            tensors[bn(l, "attn_output.weight")] = {NH * HD, E};
            tensors[bn(l, "attn_q_norm.weight")] = {HD};
            tensors[bn(l, "attn_k_norm.weight")] = {HD};
        };
        for (std::uint64_t l = 0; l < n_layer; ++l) {
            tensors[bn(l, "attn_norm.weight")] = {E};
            tensors[bn(l, "post_attention_norm.weight")] = {E};
            ffn(l);
            if ((l + 1) % H("full_attention_interval") == 0) {
                attn(l);
            } else {
                tensors[bn(l, "attn_qkv.weight")] = {E, CH};
                tensors[bn(l, "attn_gate.weight")] = {E, DI};
                tensors[bn(l, "ssm_beta.weight")] = {E, NV};
                tensors[bn(l, "ssm_alpha.weight")] = {E, NV};
                tensors[bn(l, "ssm_out.weight")] = {DI, E};
                tensors[bn(l, "ssm_conv1d.weight")] = {H("ssm.conv_kernel"), CH};
                tensors[bn(l, "ssm_dt.bias")] = {NV};
                tensors[bn(l, "ssm_a")] = {NV};
                tensors[bn(l, "ssm_norm.weight")] = {DI / NV};
            }
        }
        for (std::uint64_t l = n_layer; l < n_all; ++l) {
            attn(l);
            tensors[bn(l, "attn_norm.weight")] = {E};
            tensors[bn(l, "post_attention_norm.weight")] = {E};
            ffn(l);
            tensors[bn(l, "nextn.eh_proj.weight")] = {2 * E, E};
            tensors[bn(l, "nextn.enorm.weight")] = {E};
            tensors[bn(l, "nextn.hnorm.weight")] = {E};
            tensors[bn(l, "nextn.shared_head_norm.weight")] = {E};
        }
    }

    TinyModel(bool is_moe = false) : moe(is_moe) {
        if (moe) {
            hp["expert_count"] = 4;
            hp["expert_used_count"] = 2;
            hp["expert_feed_forward_length"] = 256;
            hp["expert_shared_feed_forward_length"] = 256;
            hp.erase("feed_forward_length");
        }
        makeTensors();
    }

    std::vector<std::uint8_t> build() const {
        GgufImage g;
        const std::string arch = moe ? "qwen35moe" : "qwen35";
        g.kvString("general.architecture", arch);
        for (const auto& [k, v] : hp) g.kvU32(arch + "." + k, static_cast<std::uint32_t>(v));
        auto kvF32 = [&](const std::string& k, float v) {
            g.str(g.kv, k);
            g.u32(g.kv, 6);
            std::uint32_t bits;
            std::memcpy(&bits, &v, 4);
            g.u32(g.kv, bits);
            ++g.n_kv;
        };
        kvF32(arch + ".rope.freq_base", 1e7f);
        kvF32(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        std::vector<std::uint8_t> out;
        g.u32(out, 0x46554747);
        g.u32(out, 3);
        g.u64(out, tensors.size());
        g.u64(out, g.n_kv);
        out.insert(out.end(), g.kv.begin(), g.kv.end());
        std::uint64_t off = 0;
        for (const auto& [name, ne] : tensors) {
            g.str(out, name);
            g.u32(out, static_cast<std::uint32_t>(ne.size()));
            std::uint64_t n = 1;
            for (std::uint64_t d : ne) g.u64(out, d), n *= d;
            const auto it = types.find(name);
            g.u32(out, it == types.end() ? 0 : it->second);
            g.u64(out, off);
            off += (n * 4 + 31) / 32 * 32;
        }
        while (out.size() % 32) out.push_back(0);
        out.resize(out.size() + off, 0);
        return out;
    }
};

// error code of Config::fromGguf + validateTensors on the image, "" if accepted
std::string loadCheck(const std::vector<std::uint8_t>& img, std::string* detail = nullptr) {
    try {
        const gguf::File f = gguf::File::parse(img);
        const q::Config cfg = q::Config::fromGguf(f);
        cfg.validateTensors(f);
    } catch (const q::ModelError& e) {
        if (detail) *detail = e.what();
        return e.code();
    } catch (const std::exception& e) {
        if (detail) *detail = e.what();
        return std::string("other: ") + e.what();
    }
    return "";
}

void testLoadValidation() {
    for (bool moe : {false, true}) {
        const char* tag = moe ? " (moe)" : " (dense)";
        auto expectCode = [&](const TinyModel& m, const char* code, const char* what) {
            std::string detail;
            const std::string got = loadCheck(m.build(), &detail);
            const bool ok = got == code;
            if (!ok) std::printf("  %s%s: got '%s' (%s)\n", what, tag, got.c_str(), detail.c_str());
            check(ok, (std::string(what) + tag).c_str());
        };
        {
            const TinyModel m(moe);
            expectCode(m, "", "tiny valid model accepted");
            const std::vector<std::uint8_t> img = m.build();
            const gguf::File f = gguf::File::parse(img);
            const q::Config c = q::Config::fromGguf(f);
            check(c.n_layer == 2 && c.n_nextn == 1 && c.n_vocab == 16 && c.head_dim == 32 && c.moe == moe, (std::string("tiny config values") + tag).c_str());
        }
        {
            TinyModel m(moe);  // no MTP block, no output.weight (tied embeddings)
            m.hp["nextn_predict_layers"] = 0;
            m.hp["block_count"] = 2;
            m.makeTensors();
            m.tensors.erase("output.weight");
            expectCode(m, "", "tiny model without MTP / output.weight accepted");
        }
        {
            TinyModel m(moe);
            m.hp["nextn_predict_layers"] = 3;  // == block_count: n_layer would be 0
            expectCode(m, "UnsupportedConfig", "nextn_predict_layers == block_count rejected");
            m.hp["nextn_predict_layers"] = 5;  // > block_count: n_layer would wrap to ~4e9
            expectCode(m, "UnsupportedConfig", "nextn_predict_layers > block_count rejected");
        }
        {
            TinyModel m(moe);
            m.hp["attention.key_length"] = 64;  // metadata head_dim no longer matches the tensors
            m.hp["attention.value_length"] = 64;
            expectCode(m, "UnsupportedTensorShape", "head_dim metadata vs attn tensors rejected");
        }
        {
            TinyModel m(moe);
            m.tensors["blk.1.attn_k.weight"] = {256, 64};  // one KV projection with the wrong head count / dim
            expectCode(m, "UnsupportedTensorShape", "mismatched attn_k shape rejected");
        }
        {
            TinyModel m(moe);
            m.hp["attention.key_length"] = 512;
            m.hp["attention.value_length"] = 512;
            m.makeTensors();  // consistent tensors, but head_dim beyond the kernels' 256
            expectCode(m, "UnsupportedConfig", "head_dim 512 rejected");
        }
        {
            TinyModel m(moe);
            m.hp["attention.value_length"] = 64;
            expectCode(m, "UnsupportedConfig", "value_length != key_length rejected");
        }
        {
            TinyModel m(moe);
            m.hp["attention.head_count_kv"] = 3;  // 2 % 3
            expectCode(m, "UnsupportedConfig", "head_count_kv not dividing head_count rejected");
        }
        {
            TinyModel m(moe);
            m.hp["full_attention_interval"] = 0;  // isAttn() would divide by zero
            expectCode(m, "UnsupportedConfig", "full_attention_interval 0 rejected");
        }
        {
            TinyModel m(moe);
            m.hp["ssm.conv_kernel"] = 0;  // d_conv - 1 would wrap
            expectCode(m, "UnsupportedConfig", "ssm.conv_kernel 0 rejected");
        }
        {
            // MAIN-002: shapes the DeltaNet kernels do not implement are rejected before any GPU work
            for (std::uint64_t conv : {2, 3, 5, 16}) {
                TinyModel m(moe);
                m.hp["ssm.conv_kernel"] = conv;
                m.makeTensors();
                expectCode(m, "UnsupportedConfig", ("ssm.conv_kernel " + std::to_string(conv) + " rejected").c_str());
            }
            for (std::uint64_t st : {1, 32, 64, 256}) {
                TinyModel m(moe);
                m.hp["ssm.state_size"] = st;
                m.makeTensors();
                expectCode(m, "UnsupportedConfig", ("ssm.state_size " + std::to_string(st) + " rejected").c_str());
            }
            for (std::uint64_t hv : {1, 31, 33, 255}) {
                TinyModel m(moe);
                m.hp["ssm.inner_size"] = 2 * hv;  // time_step_rank 2
                m.makeTensors();
                expectCode(m, "UnsupportedConfig", ("headV " + std::to_string(hv) + " rejected").c_str());
            }
            for (std::uint64_t hv : {32, 64, 128, 256}) {  // multiples of 32 (up to 256) are accepted
                TinyModel m(moe);
                m.hp["ssm.inner_size"] = 2 * hv;
                m.makeTensors();
                expectCode(m, "", ("headV " + std::to_string(hv) + " accepted").c_str());
            }
        }
        {
            TinyModel m(moe);
            m.hp["embedding_length"] = 100;
            m.makeTensors();
            expectCode(m, "UnsupportedConfig", "embedding_length not a multiple of 256 rejected");
        }
        {
            TinyModel m(moe);
            m.tensors["output.weight"] = {256, 17};  // vocab differs from token_embd
            expectCode(m, "UnsupportedTensorShape", "output.weight vocab mismatch rejected");
        }
        {
            TinyModel m(moe);
            m.tensors["blk.0.ssm_conv1d.weight"] = {4, 128, 2};  // extra dimension
            expectCode(m, "UnsupportedTensorShape", "extra tensor dimension rejected");
        }
        {
            TinyModel m(moe);
            m.tensors.erase("blk.0.ssm_out.weight");
            expectCode(m, "MissingTensor", "missing trunk tensor rejected");
        }
        {
            TinyModel m(moe);
            m.tensors.erase("blk.2.attn_q.weight");  // MTP block present (eh_proj) but incomplete
            expectCode(m, "MissingTensor", "missing MTP tensor rejected");
        }
        {
            TinyModel m(moe);
            m.types["blk.1.attn_q.weight"] = 12;  // q4_K with a 256-wide row is fine...
            expectCode(m, "", "q4_K tensor with matching shape accepted");
            m.types["blk.1.attn_q_norm.weight"] = 12;  // ...but a 32-wide q4_K row is not a whole block
            expectCode(m, "UnsupportedTensorType", "partial-block tensor rejected");
        }
        if (moe) {
            TinyModel m(moe);
            m.tensors["blk.0.ffn_up_exps.weight"] = {256, 256, 5};  // expert count differs
            expectCode(m, "UnsupportedTensorShape", "expert tensor shape mismatch rejected");
            TinyModel m2(moe);
            m2.hp["expert_used_count"] = 8;  // > expert_count
            expectCode(m2, "UnsupportedConfig", "expert_used_count > expert_count rejected");
        }
    }
}

// Optional: WHIRL_TEST_GGUF="a.gguf;b.gguf" runs the load-time validation on
// real model files (header + tensor directory only; no GPU, no weight reads).
void testRealFiles() {
    char* buf = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&buf, &len, "WHIRL_TEST_GGUF") != 0 || buf == nullptr) return;
    std::string list(buf);
    std::free(buf);
    std::size_t p = 0;
    while (p < list.size()) {
        std::size_t e = list.find(';', p);
        if (e == std::string::npos) e = list.size();
        const std::string path = list.substr(p, e - p);
        p = e + 1;
        if (path.empty()) continue;
        std::string err;
        try {
            const gguf::File f = gguf::File::open(path);
            const q::Config c = q::Config::fromGguf(f);
            c.validateTensors(f);
            std::printf("  real file OK: %s (%s, %u layers + %u MTP, embd %u, vocab %u, %zu tensors)\n", path.c_str(), c.archName(), c.n_layer, c.n_nextn,
                        c.n_embd, c.n_vocab, f.tensors().size());
            // pre-load VRAM estimate (Model::load fit check, server auto-shrink)
            const q::LoadEstimate est = q::estimateLoad(f, c, true);
            const double g = 1024.0 * 1024.0 * 1024.0;
            std::printf("    VRAM estimate: weights %.2f GiB (embedding %s), state %.1f MiB, buffers %.2f / %.2f / %.2f GiB at batch "
                        "4096 / 2048 / 1024; KV %.1f / %.1f / %.1f KiB per token (f16 / q8v / q8h)\n",
                        est.weights / g, est.embd_on_host ? "in host memory" : "in VRAM", est.state / 1048576.0,
                        q::bufferBytes(c, 4096, est.ffs, est.max_elems) / g, q::bufferBytes(c, 2048, est.ffs, est.max_elems) / g,
                        q::bufferBytes(c, 1024, est.ffs, est.max_elems) / g, est.kv_f16 / 1024.0, est.kv_q8v / 1024.0, est.kv_q8h / 1024.0);
        } catch (const std::exception& ex) {
            err = ex.what();
            std::printf("  real file REJECTED: %s: %s\n", path.c_str(), err.c_str());
        }
        check(err.empty(), ("real file passes load validation: " + path).c_str());
    }
}

void testTables() {
    check(q::tuneBucket(1) == 3 && q::tuneBucket(32) == 3 && q::tuneBucket(33) == 4 && q::tuneBucket(128) == 2 && q::tuneBucket(512) == 0 &&
              q::tuneBucket(1024) == 10 && q::tuneBucket(1025) == 1 && q::tuneBucket(4096) == 1,
          "tuneBucket boundaries");
    const q::GemvW w = q::defaultGemvW();
    const auto t = [](gguf::GgmlType ty) { return static_cast<std::size_t>(ty); };
    check(w[t(gguf::GgmlType::q4_k)][3] == 1 && w[t(gguf::GgmlType::q4_k)][4] == 6 && w[t(gguf::GgmlType::q4_k)][8] == 5, "GEMV-W Q4_K variants");
    check(w[t(gguf::GgmlType::mxfp4)][14] == 8 && w[t(gguf::GgmlType::q8_0)][16] == 0, "GEMV-W MXFP4 / Q8_0 variants");
    check(q::defaultGemvWHead()[5] == 0 && q::defaultGemvWHead()[6] == 2, "Q6_K head variants");
    check(q::n_choices == 56 && q::gemm_cfgs.size() == 24 && q::gemms_cfgs.size() == 8, "GEMM config counts");
    check(sizeof(q::GdnSeg) == 152 && sizeof(q::GdnSegs) == 152 * 16, "GdnSeg layout");
    check(sizeof(q::GvArgs) == 3 * 8 * 3 + 5 * 4 + 4, "GvArgs layout");
    check(sizeof(q::KvArgs) == 56 && sizeof(q::RowTab) == 384 && sizeof(q::AwGroups) == 256, "KvArgs / RowTab / AwGroups layout");
}

void testDraftModel() {
    q::DraftAccept a(0.5f);
    check(a.expected(0) == 1.0f && a.expected(2) == 1.75f, "DraftAccept::expected");
    a.update(3, 1);  // draft 0 accepted, draft 1 rejected
    check(a.alpha[0] > 0.5f && a.alpha[1] < 0.5f && a.alpha[2] == 0.5f, "DraftAccept::update");
    q::DraftTiming tm;
    check(!tm.estimate(3).has_value(), "DraftTiming empty");
    tm.update(2, 10.0f);
    check(std::fabs(*tm.estimate(4) - 10.0f * (1 + 0.12f * 2)) < 1e-4f, "DraftTiming one-point prior");
    {
        // MTP-01: a single measured point far from the asked draft count never extrapolates to <= 0
        q::DraftTiming far;
        far.update(15, 20.0f);
        const std::optional<float> e1 = far.estimate(1);
        check(e1.has_value() && *e1 >= 0.25f * 20.0f - 1e-4f && *e1 > 0, "DraftTiming one-point extrapolation floored");
        check(std::fabs(*far.estimate(14) - 20.0f * (1 - 0.12f)) < 1e-4f, "DraftTiming near point unchanged");
        check(std::fabs(*far.estimate(15) - 20.0f) < 1e-6f, "DraftTiming measured point exact");
        for (std::uint32_t nd = 0; nd <= q::max_ng_drafts; ++nd) check(*far.estimate(nd) > 0, "DraftTiming estimate positive for every draft count");
    }
    tm.update(6, 14.0f);
    const float e4 = *tm.estimate(4);
    check(e4 > 11.5f && e4 < 12.5f, "DraftTiming linear fit");
    q::DraftAccept hi(0.95f);
    const q::DraftAccept* accs[1] = {&hi};
    const std::uint32_t nd = q::pickDrafts(accs, tm, 8, 0, 0);
    check(nd >= 4, "pickDrafts prefers long chains at high acceptance");
}

void testDraftVocab() {
    std::vector<std::uint32_t> full(100);
    for (std::uint32_t i = 0; i < 100; ++i) full[i] = i;
    std::uint32_t added = 7;
    const std::vector<std::uint32_t> req = {99, 0, 50};
    check(q::draftVocabIds(full, 100, req, &added) == full && added == 0, "draft vocab: the full vocabulary is the identity map");
    const std::vector<std::uint32_t> sub = {3, 10, 42}, req2 = {99, 0, 10, 99, 150};
    const auto s = q::draftVocabIds(sub, 100, req2, &added);
    check(s == std::vector<std::uint32_t>{0, 3, 10, 42, 99} && added == 2, "draft vocab: missing special ids added (sorted, out of range ignored)");
    check(s[2] == 10 && s[4] == 99, "draft vocab: map[row] = token id");
    bool threw = false;
    try {
        q::draftVocabIds(std::vector<std::uint32_t>{5, 3}, 100, {});
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "draft vocab: ids not ascending are rejected");
    threw = false;
    try {
        q::draftVocabIds(std::vector<std::uint32_t>{1, 100}, 100, {});
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "draft vocab: an id >= rows is rejected");
    const std::string d48 = (std::filesystem::path("D") / "draft_vocab" / "subset_48k.bin").string();
    check(!q::draftVocabFile("off", "D") && !q::draftVocabFile("150000", "D") && q::draftVocabFile("48k", "D") == d48 &&
              q::draftVocabFile("x/y.bin", "D") == std::string("x/y.bin"),
          "draft vocab: WHIRL_DRAFT_VOCAB values (off, N, 48k, path)");
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "whirl_draft_vocab_test.bin";
    {
        std::ofstream f(tmp, std::ios::binary);
        for (std::uint32_t v : s) {
            const unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8),
                                        static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)};
            f.write(reinterpret_cast<const char*>(b), 4);
        }
    }
    check(q::readDraftVocab(tmp.string()) == s, "draft vocab: file round trip (uint32 little-endian)");
    std::filesystem::remove(tmp);
    using DK = q::DraftVocabChoice::Kind;
    const q::DraftVocabChoice c0 = q::draftVocabChoice(std::nullopt, "D");
    check(c0.kind == DK::embedded_64k && c0.by_default, "draft vocab: unset = embedded 64k by default");
    const q::DraftVocabChoice c64 = q::draftVocabChoice(std::string("64k"), "D");
    check(c64.kind == DK::embedded_64k && !c64.by_default, "draft vocab: 64k = embedded, explicit");
    check(q::draftVocabChoice(std::string("off"), "D").kind == DK::full && q::draftVocabChoice(std::string("0"), "D").kind == DK::full,
          "draft vocab: off / 0 = full head");
    const q::DraftVocabChoice c48 = q::draftVocabChoice(std::string("48k"), "D");
    const q::DraftVocabChoice cn = q::draftVocabChoice(std::string("150000"), "D");
    check(c48.kind == DK::file && c48.file == d48 && cn.kind == DK::first_n && cn.n == 150000,
          "draft vocab: 48k = file next to the exe, N = first N rows");
    check(q::draftVocabDefaultFits(false, 248320) && !q::draftVocabDefaultFits(true, 248320) &&
              !q::draftVocabDefaultFits(false, 151936),
          "draft vocab: default only for dense qwen35 with the 248320-token vocabulary");
    const std::vector<std::uint32_t> e64 = q::embeddedDraftVocab64k();
    bool asc = e64.size() == 65536;
    for (std::size_t i = 1; asc && i < e64.size(); ++i) asc = e64[i - 1] < e64[i];
    check(asc && e64.back() < 248320 && e64.front() == 0 && std::find(e64.begin(), e64.end(), 248044u) != e64.end(),
          "draft vocab: embedded 64k subset (65536 ascending ids < 248320, specials included)");
}

// FIX-DFL (FR-DFL P2-1): the drafter trunk of dflashImage (5 layers, 4 heads / 2 KV heads of 16, FFN 96,
// conv K 2 / group 16), every tensor f32 with its payload. g_trunk_edit (tests only) replaces one tensor's
// shape / type, or drops it, to build malformed drafters.
struct TrunkEdit {
    std::string name;
    std::vector<std::uint64_t> ne;  // empty: keep the shape
    std::uint32_t type = 0;
    bool drop = false;
};
const TrunkEdit* g_trunk_edit = nullptr;
void addDFlashTrunk(GgufImage& g, std::uint32_t embd, std::uint32_t rank) {
    const std::uint64_t E = embd, HD = 16, QW = 4 * HD, KW = 2 * HD, F = 96, K = 2, DS = 2 * K * (E / 16);
    auto add = [&](const std::string& name, std::vector<std::uint64_t> ne) {
        std::uint32_t ty = 0;
        if (g_trunk_edit && g_trunk_edit->name == name) {
            if (g_trunk_edit->drop) return;
            if (!g_trunk_edit->ne.empty()) ne = g_trunk_edit->ne;
            ty = g_trunk_edit->type;
        }
        std::uint64_t n = 1;
        for (std::uint64_t x : ne) n *= x;
        g.tensorN(name, ne, ty, ty == 0 ? 4 * n : ty == 1 ? 2 * n : ty == 8 ? n / 32 * 34 : 0);
    };
    add("enc.output_norm.weight", {E});
    add("output_norm.weight", {E});
    add("selector_hidden.weight", {E, rank});
    for (int il = 0; il < 5; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        add(b + "attn_norm.weight", {E});
        add(b + "ffn_norm.weight", {E});
        add(b + "attn_q_norm.weight", {HD});
        add(b + "attn_k_norm.weight", {HD});
        add(b + "attn_q.weight", {E, QW});
        add(b + "attn_k.weight", {E, KW});
        add(b + "attn_v.weight", {E, KW});
        add(b + "attn_output.weight", {QW, E});
        add(b + "ffn_gate.weight", {E, F});
        add(b + "ffn_up.weight", {E, F});
        add(b + "ffn_down.weight", {F, E});
        add(b + "attn_conv_proj.weight", {E, DS});
        add(b + "ffn_conv_proj.weight", {E, DS});
        add(b + "attn_conv_base", {E, K, 2});
        add(b + "ffn_conv_base", {E, K, 2});
    }
}

void testDraftWindowArg() {
    check(q::draftWindowArg(0, 65536) == 0, "draft window: W = 0 is off");
    check(q::draftWindowArg(8192, 65536) == (128 | 64 << 16), "draft window: W / 64 | threshold / 1024 << 16");
    check(q::draftWindowArg(100, 0) == 2, "draft window: W rounded up to 64");
    check((q::draftWindowArg(1u << 30, 1u << 30) & 0xffff) == 0xffff && (q::draftWindowArg(1u << 30, 1u << 30) >> 16) == 0x7fff,
          "draft window: fields clamped");
}

// DFlash2 drafter metadata (arch "dflash"): parsing and the checks against the target
// sel_ty: ggml type id of the two selector codebooks (0 f32, 1 f16, 8 Q8_0, 12 Q4_K, ...), sel_ty2 the successor's
// (-1: same as sel_ty); rank 8 for f32 / f16, 256 otherwise (one block row for every block size up to 256)
std::uint64_t selRowBytes(std::uint32_t ty, std::uint32_t rank) {
    switch (ty) {
        case 0: return rank * 4ull;
        case 1: return rank * 2ull;
        case 8: return rank / 32 * 34ull;
        case 12: return rank / 256 * 144ull;
        case 14: return rank / 256 * 210ull;
        default: return 0;
    }
}
GgufImage dflashImage(std::uint32_t embd, std::uint32_t vocab, bool mask = true, const char* arch = "dflash", std::uint32_t sel_ty = 0,
                      int sel_ty2 = -1) {
    GgufImage g;
    const std::string a = arch;
    g.kvString("general.architecture", a);
    g.kvU32(a + ".block_count", 5);
    g.kvU32(a + ".embedding_length", embd);
    g.kvU32(a + ".feed_forward_length", 96);
    g.kvU32(a + ".attention.head_count", 4);
    g.kvU32(a + ".attention.head_count_kv", 2);
    g.kvU32(a + ".attention.key_length", 16);
    g.kvF32(a + ".attention.layer_norm_rms_epsilon", 1e-6f);
    g.kvF32(a + ".rope.freq_base", 1e7f);
    g.kvU32(a + ".block_size", 8);
    g.kvU32(a + ".conv_kernel_size", 2);
    g.kvU32(a + ".conv_group_size", 16);
    const std::uint32_t ty2 = sel_ty2 < 0 ? sel_ty : static_cast<std::uint32_t>(sel_ty2);
    const std::uint32_t rank = (sel_ty <= 1 && ty2 <= 1) ? 8 : 256;
    g.kvU32(a + ".selector_rank", rank);
    g.kvU32(a + ".selector_top_k", 4);
    g.kvU32(a + ".attention.sliding_window", 2048);
    g.kvI32Array(a + ".target_layers", {2, 4, 6});
    if (mask) g.kvU32("tokenizer.ggml.mask_token_id", 95);
    if (g_trunk_edit && g_trunk_edit->name == "fc.weight") {  // FR-DFL P3-7: fc of another type (bytes by ggml type)
        const std::uint32_t ty = g_trunk_edit->type;
        const std::uint64_t n = 3ull * embd * embd;
        g.tensor("fc.weight", 3ull * embd, embd, ty, ty == 39 ? n / 32 * 17 : ty == 8 ? n / 32 * 34 : ty == 1 ? 2 * n : 0);
    } else {
        g.tensor("fc.weight", 3ull * embd, embd, 0, 3ull * embd * embd * 4);
    }
    addDFlashTrunk(g, embd, rank);
    g.tensor("selector_predecessor.weight", rank, vocab, sel_ty, selRowBytes(sel_ty, rank) * vocab);
    g.tensor("selector_successor.weight", rank, vocab, ty2, selRowBytes(ty2, rank) * vocab);
    return g;
}

// FIX-K97 (FR-K97 P2-3): gemmr_moe* step k by 64 with no tail; the host gate (Model::moeGuR) refuses other E / F
void testGemmrMoeShape() {
    namespace wk = whirl::kernels;
    check(wk::kGemmrMoeBk == 64, "gemmr_moe shape: k step 64");
    check(wk::gemmrMoeShapeOk(2048, 512), "gemmr_moe shape: Ornith 35B-A3B (E 2048, F 512) accepted");
    check(wk::gemmrMoeShapeOk(64, 64), "gemmr_moe shape: one k step accepted");
    check(!wk::gemmrMoeShapeOk(2048 + 32, 512), "gemmr_moe shape: E % 64 = 32 refused");
    check(!wk::gemmrMoeShapeOk(2048, 512 + 32), "gemmr_moe shape: F % 64 = 32 refused (down BK 64)");
    check(!wk::gemmrMoeShapeOk(2048, 520), "gemmr_moe shape: F % 64 = 8 refused");
    check(!wk::gemmrMoeShapeOk(0, 512) && !wk::gemmrMoeShapeOk(2048, 0), "gemmr_moe shape: empty refused");
    static_assert(wk::gemmrMoeShapeOk(4096, 1536), "constexpr");
}

void testDFlashConfig() {
    q::Config t;
    t.n_embd = 64;
    t.n_layer = 8;
    t.n_vocab = 100;
    const auto img = dflashImage(64, 100).build();
    const gguf::File f = gguf::File::parse(img);
    const q::DFlashConfig c = q::DFlashConfig::fromGguf(f);
    check(c.n_layer == 5 && c.n_embd == 64 && c.n_head == 4 && c.n_head_kv == 2 && c.head_dim == 16 && c.n_ff == 96 && c.block == 8 &&
              c.conv_k == 2 && c.conv_group == 16 && c.sel_rank == 8 && c.sel_topk == 4 && c.window == 2048 && c.mask_id == 95 &&
              c.target_layers == std::vector<std::uint32_t>{2, 4, 6} && std::fabs(c.rope_base - 1e7f) < 1.0f,
          "DFlash2 config: metadata");
    bool ok = true;
    try {
        c.validate(t, f);
    } catch (const q::ModelError&) {
        ok = false;
    }
    check(ok, "DFlash2 config: fits the target");
    auto code = [](const std::function<void()>& fn) {
        try {
            fn();
        } catch (const q::ModelError& e) {
            return e.code();
        }
        return std::string();
    };
    q::Config t2 = t;
    t2.n_embd = 128;
    check(code([&] { c.validate(t2, f); }) == "DraftModelMismatch", "DFlash2 config: hidden size mismatch rejected");
    q::Config t3 = t;
    t3.n_layer = 4;
    check(code([&] { c.validate(t3, f); }) == "DraftModelMismatch", "DFlash2 config: target layer beyond the target rejected");
    q::Config t4 = t;
    t4.n_vocab = 200;
    check(code([&] { c.validate(t4, f); }) == "DraftModelMismatch", "DFlash2 config: vocabulary mismatch rejected");
    const auto img_arch = dflashImage(64, 100, true, "qwen35").build();
    check(code([&] { (void)q::DFlashConfig::fromGguf(gguf::File::parse(img_arch)); }) == "UnsupportedArch", "DFlash2 config: arch whitelist");
    const auto img_mask = dflashImage(64, 100, false).build();
    check(code([&] { (void)q::DFlashConfig::fromGguf(gguf::File::parse(img_mask)); }) == "MissingKey", "DFlash2 config: mask token required");
    // kernel limits (C-52): attention keys = window (2048 without one) + block, head_dim, selector top-k
    auto limited = [&](auto edit) {
        q::DFlashConfig c2 = c;
        edit(c2);
        return code([&] { c2.validate(t, f); });
    };
    check(limited([](q::DFlashConfig& x) { x.window = 4096, x.block = 16; }).empty(), "DFlash2 config: Ornith's window 4096 + block 16 accepted");
    check(limited([](q::DFlashConfig& x) { x.window = 0; }).empty(), "DFlash2 config: no window (ring 2048) accepted");
    check(limited([](q::DFlashConfig& x) { x.window = whirl::kernels::kDFlashMaxKeys - 8; }).empty(),
          "DFlash2 config: window + block == the attention limit accepted");
    check(limited([](q::DFlashConfig& x) { x.window = whirl::kernels::kDFlashMaxKeys - 7; }) == "DraftModelMismatch",
          "DFlash2 config: window + block past the attention limit rejected");
    check(limited([](q::DFlashConfig& x) { x.head_dim = 512; }) == "DraftModelMismatch", "DFlash2 config: head_dim > 256 rejected");
    check(limited([](q::DFlashConfig& x) { x.sel_topk = 32; }) == "DraftModelMismatch", "DFlash2 config: selector top-k > 16 rejected");
    bool msg = false;
    try {
        q::DFlashConfig c2 = c;
        c2.window = 70000;
        c2.validate(t, f);
    } catch (const q::ModelError& e) {
        const std::string w = e.what();
        msg = w.find("70008") != std::string::npos && w.find(std::to_string(whirl::kernels::kDFlashMaxKeys)) != std::string::npos;
    }
    check(msg, "DFlash2 config: the refusal names the keys needed and the limit");
    // FIX-K97 (FR-K97 P2-1): dflash_lattice decodes f32 or Q8_0 codebooks with one type flag and one row size
    auto sel = [&](std::uint32_t ty, int ty2) {
        const auto im = dflashImage(64, 100, true, "dflash", ty, ty2).build();
        const gguf::File fs = gguf::File::parse(im);
        const q::DFlashConfig cs = q::DFlashConfig::fromGguf(fs);
        std::string what;
        const std::string cd = code([&] {
            try {
                cs.validate(t, fs);
            } catch (const q::ModelError& e) {
                what = e.what();
                throw;
            }
        });
        return std::make_pair(cd, what);
    };
    check(sel(0, -1).first.empty(), "DFlash2 config: f32 selector codebooks accepted");
    check(sel(8, -1).first.empty(), "DFlash2 config: Q8_0 selector codebooks accepted");
    const auto f16 = sel(1, -1);
    check(f16.first == "DraftModelMismatch" && f16.second.find("f32 or both Q8_0") != std::string::npos,
          "DFlash2 config: f16 selector codebooks rejected (" + f16.second + ")");
    check(sel(12, -1).first == "DraftModelMismatch", "DFlash2 config: Q4_K selector codebooks rejected");
    check(sel(14, -1).first == "DraftModelMismatch", "DFlash2 config: Q6_K selector codebooks rejected");
    check(sel(8, 0).first == "DraftModelMismatch", "DFlash2 config: Q8_0 predecessor + f32 successor rejected");
    check(sel(0, 8).first == "DraftModelMismatch", "DFlash2 config: f32 predecessor + Q8_0 successor rejected");
    check(sel(0, 1).first == "DraftModelMismatch", "DFlash2 config: f32 predecessor + f16 successor rejected");
}

// a target header (qwen35 / qwen35moe) for dflashIncompatible: metadata + token_embd only
GgufImage targetImage(const char* arch, std::uint32_t embd, std::uint32_t vocab, std::uint32_t layers) {
    GgufImage g;
    const std::string a = arch;
    const bool moe = a == "qwen35moe";
    g.kvString("general.architecture", a);
    g.kvU32(a + ".block_count", layers + 1);
    g.kvU32(a + ".nextn_predict_layers", 1);
    g.kvU32(a + ".embedding_length", embd);
    if (moe) {
        g.kvU32(a + ".expert_feed_forward_length", 256);  // kg3 load hardening: 256..65536
        g.kvU32(a + ".expert_count", 8);
        g.kvU32(a + ".expert_used_count", 2);
    } else {
        g.kvU32(a + ".feed_forward_length", 128);
    }
    g.kvU32(a + ".attention.head_count", 4);
    g.kvU32(a + ".attention.head_count_kv", 2);
    g.kvU32(a + ".attention.key_length", 32);  // kg3 load hardening: 32..256
    g.kvU32(a + ".rope.dimension_count", 32);
    g.kvF32(a + ".rope.freq_base", 1e7f);
    g.kvF32(a + ".attention.layer_norm_rms_epsilon", 1e-6f);
    g.kvU32(a + ".ssm.conv_kernel", 4);
    g.kvU32(a + ".ssm.state_size", 128);  // fix-review2 GDN-CFG: state 128 and headV (inner/rank) a multiple of 32
    g.kvU32(a + ".ssm.group_count", 2);
    g.kvU32(a + ".ssm.time_step_rank", 4);
    g.kvU32(a + ".ssm.inner_size", 512);
    g.tensor("token_embd.weight", embd, vocab);
    return g;
}

// FIX-DFL (FR-DFL P2-1): every drafter tensor must match the metadata (the forward sizes its scratch from
// the metadata and matmul writes w.nrows outputs: a larger tensor wrote past the scratch on the GPU)
void testDFlashTensors() {
    q::Config t;
    t.n_embd = 64;
    t.n_layer = 8;
    t.n_vocab = 100;
    auto why = [&](const TrunkEdit& e) {
        g_trunk_edit = &e;
        const auto img = dflashImage(64, 100).build();
        g_trunk_edit = nullptr;
        try {
            const gguf::File f = gguf::File::parse(img);
            q::DFlashConfig::fromGguf(f).validate(t, f);
        } catch (const q::ModelError& ex) {
            return ex.code() + ": " + ex.what();
        } catch (const std::exception& ex) {
            return std::string("other: ") + ex.what();
        }
        return std::string();
    };
    auto rejected = [&](const TrunkEdit& e, const char* what) {
        const std::string w = why(e);
        check(w.rfind("DraftModelMismatch", 0) == 0, std::string("DFlash2 tensors: ") + what + " rejected (" + w + ")");
    };
    check(why(TrunkEdit{}).empty(), "DFlash2 tensors: the well-formed drafter accepted (" + why(TrunkEdit{}) + ")");
    const std::uint64_t E = 64, DS = 2 * 2 * (64 / 16);
    rejected({"blk.0.attn_q.weight", {E, 80}}, "attn_q with more rows than n_head x head_dim");
    rejected({"blk.2.attn_k.weight", {E, 48}}, "attn_k with more rows than n_head_kv x head_dim");
    rejected({"blk.4.attn_v.weight", {E, 16}}, "attn_v with fewer rows");
    rejected({"blk.1.attn_output.weight", {E / 2, E}}, "attn_output with fewer columns than n_head x head_dim");
    rejected({"blk.0.ffn_up.weight", {E, 128}}, "ffn_up with more rows than feed_forward_length");
    rejected({"blk.3.ffn_gate.weight", {E, 128}}, "ffn_gate with more rows than feed_forward_length");
    rejected({"blk.2.ffn_down.weight", {128, E}}, "ffn_down with more columns than feed_forward_length");
    rejected({"blk.3.ffn_conv_proj.weight", {E, 2 * DS}}, "a later layer's ffn_conv_proj larger than layer 0's (dyn scratch)");
    rejected({"blk.1.attn_conv_proj.weight", {E, 2 * DS}}, "a later layer's attn_conv_proj larger than layer 0's");
    rejected({"blk.0.attn_conv_proj.weight", {E, DS / 2}}, "attn_conv_proj with fewer rows than 2 K (E / group)");
    rejected({"blk.0.attn_conv_base", {E, 2, 1}}, "attn_conv_base with one side");
    rejected({"blk.4.ffn_conv_base", {E / 2, 2, 2}}, "ffn_conv_base of half the channels");
    rejected({"selector_hidden.weight", {E, 16}}, "selector_hidden with rows != selector_rank");
    rejected({"enc.output_norm.weight", {E / 2}}, "enc.output_norm shorter than n_embd");
    rejected({"output_norm.weight", {E / 2}}, "output_norm shorter than n_embd");
    rejected({"blk.2.attn_norm.weight", {E / 2}}, "attn_norm shorter than n_embd");
    rejected({"blk.2.ffn_norm.weight", {E - 1}}, "ffn_norm shorter than n_embd");
    rejected({"blk.0.attn_q_norm.weight", {8}}, "attn_q_norm shorter than head_dim");
    rejected({"blk.0.attn_k_norm.weight", {8}}, "attn_k_norm shorter than head_dim");
    rejected({"blk.4.attn_v.weight", {}, 0, true}, "a missing layer tensor");
    rejected({"blk.1.attn_norm.weight", {}, 1}, "an f16 norm vector");
    rejected({"blk.1.attn_q.weight", {}, 1000}, "a matrix of an unknown type");
    // FR-DFL P3-7: fc is split into per-target-layer column blocks: MXFP4 (repacked, per-row reference) refused
    rejected({"fc.weight", {}, 39}, "an MXFP4 fc");
    check(why({"fc.weight", {}, 8}).empty(), "DFlash2 tensors: a Q8_0 fc accepted (" + why({"fc.weight", {}, 8}) + ")");
    check(why({"fc.weight", {}, 1}).empty(), "DFlash2 tensors: an f16 fc accepted");
    rejected({"fc.weight", {}, 1000}, "an fc of an unknown type");
    // block_count / head_dim limits (the metadata the shapes come from)
    const auto img = dflashImage(64, 100).build();
    const gguf::File f = gguf::File::parse(img);
    const q::DFlashConfig c = q::DFlashConfig::fromGguf(f);
    auto meta = [&](auto edit) {
        q::DFlashConfig c2 = c;
        edit(c2);
        try {
            c2.validate(t, f);
        } catch (const q::ModelError& ex) {
            return ex.code() + ": " + ex.what();
        }
        return std::string();
    };
    const std::string w0 = meta([](q::DFlashConfig& x) { x.n_layer = 0; });
    check(w0.find("block_count 0") != std::string::npos, "DFlash2 tensors: block_count 0 rejected (" + w0 + ")");
    const std::string wbig = meta([](q::DFlashConfig& x) { x.n_layer = q::kDFlashMaxLayers + 1; });
    check(wbig.find("block_count") != std::string::npos, "DFlash2 tensors: block_count past the limit rejected (" + wbig + ")");
    const std::string whd = meta([](q::DFlashConfig& x) { x.head_dim = 15; });
    check(whd.rfind("DraftModelMismatch", 0) == 0, "DFlash2 tensors: odd head_dim rejected (" + whd + ")");
    // the header-only fit check the server / CLI run before loading reports it (-> MTP drafts + warning)
    {
        TrunkEdit e{"blk.3.ffn_up.weight", {256, 192}};
        g_trunk_edit = &e;
        const auto draft = dflashImage(256, 100).build();
        g_trunk_edit = nullptr;
        const auto dense = targetImage("qwen35", 256, 100, 8).build();
        const std::string w = q::dflashIncompatible(gguf::File::parse(dense), gguf::File::parse(draft));
        check(w.find("blk.3.ffn_up.weight") != std::string::npos, "dflashIncompatible: a malformed drafter tensor named (" + w + ")");
    }
}

// FR-DFL P3-2: a --draft-model that cannot be opened / parsed is a reason (-> warning + MTP drafts), not
// an exception that stops whirl-server / whirl chat
void testDFlashDraftFile() {
    const auto dense = targetImage("qwen35", 256, 100, 8).build();
    const gguf::File tf = gguf::File::parse(dense);
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "whirl_fixdfl_p32";
    std::filesystem::create_directories(dir);
    std::string w;
    bool threw = false;
    try {
        w = q::dflashIncompatibleFile(tf, (dir / "no_such_drafter.gguf").string());
    } catch (...) {
        threw = true;
    }
    check(!threw && w.find("cannot open the drafter") != std::string::npos, "dflashIncompatibleFile: a missing file is a reason (" + w + ")");
    {
        std::ofstream(dir / "garbage.gguf", std::ios::binary) << "not a gguf file";
        threw = false;
        try {
            w = q::dflashIncompatibleFile(tf, (dir / "garbage.gguf").string());
        } catch (...) {
            threw = true;
        }
        check(!threw && w.find("cannot open the drafter") != std::string::npos, "dflashIncompatibleFile: an unparsable file is a reason (" + w + ")");
    }
    {
        const auto draft = dflashImage(256, 100).build();
        std::ofstream(dir / "ok.gguf", std::ios::binary).write(reinterpret_cast<const char*>(draft.data()), static_cast<std::streamsize>(draft.size()));
        std::optional<gguf::File> opened;
        w = q::dflashIncompatibleFile(tf, (dir / "ok.gguf").string(), &opened);
        check(w.empty() && opened.has_value(), "dflashIncompatibleFile: a fitting drafter file opens (" + w + ")");
    }
    // the caller turns the reason into the fall-back warning (whirl/dflash_policy.h choose)
    const auto ch = whirl::dflash::choose(whirl::dflash::Source::dflash, std::string("x.gguf"), "cannot open the drafter: x", true);
    check(ch.source == whirl::dflash::Source::mtp && ch.warning.find("using MTP drafts") != std::string::npos, "dflash choose: unopenable drafter -> MTP");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// --draft dflash: the header-only fit check before loading (a MoE target without a drafter of
// its size keeps MTP)
void testDFlashCompat() {
    // kg3 Config::fromGguf accepts embedding_length 256..65536 only (load hardening)
    const auto draft = dflashImage(256, 100).build();
    const gguf::File df = gguf::File::parse(draft);
    const auto dense = targetImage("qwen35", 256, 100, 8).build();
    const std::string why_dense = q::dflashIncompatible(gguf::File::parse(dense), df);
    check(why_dense.empty(), "dflashIncompatible: matching dense target (" + why_dense + ")");
    const auto moe = targetImage("qwen35moe", 512, 100, 8).build();
    const std::string why = q::dflashIncompatible(gguf::File::parse(moe), df);
    check(why.find("DraftModelMismatch") != std::string::npos, "dflashIncompatible: MoE target of another size rejected (" + why + ")");
    const auto shallow = targetImage("qwen35", 256, 100, 4).build();
    const std::string why_sh = q::dflashIncompatible(gguf::File::parse(shallow), df);
    check(why_sh.find("target layer id") != std::string::npos, "dflashIncompatible: target layers beyond the target rejected (" + why_sh + ")");
    const auto not_draft = dflashImage(256, 100, true, "qwen35").build();
    check(q::dflashIncompatible(gguf::File::parse(dense), gguf::File::parse(not_draft)).find("UnsupportedArch") != std::string::npos,
          "dflashIncompatible: a non-dflash draft file rejected");
}

void testNgram() {
    q::Ngram ng;
    // history: "1 2 3 4 5 6 7 8 9" then "... 1 2 3" -> drafts 4 5 6 ...
    std::vector<std::uint32_t> toks = {10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 1, 2, 3};
    std::uint32_t out[8];
    const q::Ngram::Match m = ng.lookup(toks, 3, out);
    check(m.n > 0 && out[0] == 4 && out[1] == 5 && out[2] == 6, "n-gram lookup continuation");
    check(m.mlen == 3, "n-gram matched length");
    toks.push_back(4);
    toks.push_back(5);
    const q::Ngram::Match m2 = ng.lookup(toks, 3, out);
    check(m2.n > 0 && out[0] == 6, "n-gram follows its source");
    q::Ngram ng2;
    std::vector<std::uint32_t> none = {1, 2, 3, 4, 5, 6, 7};
    check(ng2.lookup(none, 3, out).n == 0, "n-gram no match");
}


// ---- T4-1: cycle cost, per-slot acceptance, draft allocation

struct Lcg {
    std::uint64_t x = 0x9E3779B97F4A7C15ull;
    std::uint32_t next() {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint32_t>(x >> 33);
    }
    float uni() { return static_cast<float>(next()) / 2147483648.0f; }  // [0, 1)
    std::uint32_t range(std::uint32_t lo, std::uint32_t hi) { return lo + next() % (hi - lo + 1); }
};

// synthetic truth: T = 20 + 0.8 R + 2.5 S + 3 [R > 16] + 0.002 sum(rows_i ctx_i)
float trueCycle(const q::CycleFeat& x) {
    return 20.0f + 0.8f * x.rows + 2.5f * x.steps + (x.rows > 16 ? 3.0f : 0.0f) + 0.002f * x.row_ctx;
}

q::CycleFeat randomCycle(Lcg& g) {
    for (;;) {
        q::CycleFeat x;
        const std::uint32_t slots = g.range(1, 8);
        for (std::uint32_t i = 0; i < slots; ++i) {
            const std::uint32_t d = g.range(0, 7);
            const float ctx = static_cast<float>(g.range(1, 128));
            x.rows += static_cast<float>(d + 1);
            x.steps = std::max(x.steps, static_cast<float>(d));
            x.row_ctx += static_cast<float>(d + 1) * ctx;
        }
        if (x.rows <= 32) return x;
    }
}

q::CycleCost trainedCost(std::uint32_t n_obs) {
    q::CycleCost c;
    Lcg g;
    for (std::uint32_t i = 0; i < n_obs; ++i) {
        const q::CycleFeat x = randomCycle(g);
        c.observe(x, trueCycle(x) * (1.0f + 0.01f * (g.uni() - 0.5f)));  // +-0.5% noise
    }
    return c;
}

void testCycleCost() {
    const q::CycleCost c = trainedCost(300);
    Lcg g;
    g.x = 12345;
    float worst = 0;
    for (int i = 0; i < 200; ++i) {
        const q::CycleFeat x = randomCycle(g);
        worst = std::max(worst, std::fabs(c.predict(x) / trueCycle(x) - 1.0f));
    }
    check(worst < 0.02f, "CycleCost: synthetic fit within 2%");
    check(c.samples() == 300, "CycleCost: sample count");
    // prior from uniform timing points (4 slots, 1 and 3 drafts) reproduces those points
    q::DraftTiming tm;
    tm.update(1, 30.0f);
    tm.update(3, 40.0f);
    q::CycleCost p;
    p.prior(tm, 4);
    q::CycleFeat x1, x3;
    x1.rows = 8, x1.steps = 1;
    x3.rows = 16, x3.steps = 3;
    check(std::fabs(p.predict(x1) - 30.0f) < 1.5f && std::fabs(p.predict(x3) - 40.0f) < 1.5f, "CycleCost: prior from timing");
    check(p.samples() == 0, "CycleCost: a prior is not a sample");
}

void testSlotAccept() {
    q::SlotAccept s0(0.5f);
    s0.observe(1, 1);
    check(std::fabs(s0.a.alpha[0] - 0.6f) < 1e-6f, "SlotAccept: r = 0.2 in the first 16 cycles");
    // a broken chain pulls the never-observed positions to last x 0.95^k
    q::SlotAccept b(0.9f);
    for (int i = 0; i < 1500; ++i) b.observe(3, 2);  // positions 0, 1 hit, 2 missed
    check(b.a.alpha[1] > 0.95f && b.a.alpha[2] < 0.05f && b.a.alpha[5] < 0.05f, "SlotAccept: frozen positions follow a broken chain");
    // only position 0 is observed, half the chains complete (censored) and half break: the frozen
    // positions settle between alpha0 * 0.95^k and alpha0, non-increasing in k
    q::SlotAccept f(0.9f);
    std::array<float, 6> mid{};
    for (int i = 0; i < 2000; ++i) {
        f.observe(1, static_cast<std::uint32_t>(i % 2));
        if (i == 1799)
            for (std::size_t k = 0; k < 6; ++k) mid[k] = f.a.alpha[k];
    }
    const float a0 = f.a.alpha[0];
    bool conv = std::fabs(a0 - 0.5f) < 0.08f;
    for (std::uint32_t k = 1; k < 6; ++k) {
        conv = conv && f.a.alpha[k] <= f.a.alpha[k - 1] + 1e-6f && f.a.alpha[k] <= a0 + 0.03f &&
               f.a.alpha[k] >= a0 * std::pow(0.95f, static_cast<float>(k)) - 0.03f && std::fabs(f.a.alpha[k] - mid[k]) < 0.02f;
    }
    check(conv, "SlotAccept: unobserved positions settle in [alpha0 * 0.95^k, alpha0]");
    // whole chains accepted: censored, positions beyond are not pulled down
    q::SlotAccept c(0.5f);
    for (int i = 0; i < 500; ++i) c.observe(2, 2);
    check(c.a.alpha[1] > 0.95f && c.a.alpha[6] > 0.9f, "SlotAccept: full acceptance is censored, not a ceiling");
    // shrinkage towards the pool with few samples, count capped at 32
    q::SlotAccept few(0.75f);
    few.observe(1, 0);
    few.observe(1, 0);
    const q::DraftAccept pool(0.8f);
    const float raw = few.a.alpha[0];
    const float eff = few.effective(pool).alpha[0];
    check(few.n[0] == 2 && std::fabs(eff - (2 * raw + 4 * 0.8f) / 6) < 1e-6f && std::fabs(eff - 0.8f) < std::fabs(raw - 0.8f),
          "SlotAccept: few samples shrink towards the pool");
    for (int i = 0; i < 100; ++i) few.observe(1, 0);
    check(few.n[0] == 32 && few.n[1] == 0, "SlotAccept: count capped at 32, unobserved positions not counted");
}

// independent score / throughput of an allocation (same definitions as allocDrafts)
struct Score {
    float total = 0;
    std::vector<float> per;
};

Score scoreOf(std::span<const q::AllocSlot> slots, const q::CycleCost& c, const std::vector<std::uint32_t>& d) {
    q::CycleFeat x;
    std::vector<float> e(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const bool ng = slots[i].ng_rows > 0;
        const float rows = ng ? static_cast<float>(slots[i].ng_rows) : static_cast<float>(d[i] + 1);
        x.rows += rows;
        x.row_ctx += rows * slots[i].ctx_k;
        if (!ng) x.steps = std::max(x.steps, static_cast<float>(d[i]));
        e[i] = ng ? slots[i].ng_e : slots[i].acc->expected(d[i]);
    }
    const float t = c.predict(x);
    Score s;
    for (float v : e) {
        s.total += v / t;
        s.per.push_back(v / t);
    }
    return s;
}

void testAllocDrafts() {
    const q::CycleCost c = trainedCost(64);
    const q::DraftAccept hi(0.92f), lo(0.35f), mid(0.7f);
    std::vector<q::AllocSlot> slots(4);
    for (int i = 0; i < 4; ++i) {
        slots[i].acc = i < 2 ? &hi : &lo;
        slots[i].cap = 8;
        slots[i].ctx_k = 8;
    }
    const std::uint32_t U = 3;
    const std::vector<std::uint32_t> u(4, U);
    const q::AllocResult r = q::allocDrafts(slots, c, {32, 0xffffffffu}, U, {});
    std::uint32_t sum = 0;
    bool ge1 = true;
    for (std::uint32_t d : r.d) {
        sum += d;
        ge1 = ge1 && d >= 1 && d <= 8;
    }
    check(!r.uniform && r.d[0] > r.d[2] && r.d[1] > r.d[3], "allocDrafts: more drafts to the high-acceptance slots");
    check(sum <= 32 - 4 && ge1, "allocDrafts: sum(d) within the free rows, 1 <= d_i <= cap");
    const Score su = scoreOf(slots, c, u), sr = scoreOf(slots, c, r.d);
    bool floor = true;
    for (std::size_t i = 0; i < 4; ++i) floor = floor && sr.per[i] >= 0.97f * su.per[i] - 1e-6f;
    check(floor, "allocDrafts: every slot keeps >= 97% of its uniform throughput");
    check(sr.total > 1.01f * su.total && std::fabs(sr.total - r.score) < 1e-4f * sr.total, "allocDrafts: beats u by > 1%");
    check(r.rounds >= 1 && r.rounds <= 16, "allocDrafts: ends within 16 rounds");

    // tight budgets: 12 rows -> sum(d) <= 8; 9 snapshot sets -> sum(d) <= 9
    const q::AllocResult t1 = q::allocDrafts(slots, c, {12, 0xffffffffu}, 2, {});
    const q::AllocResult t2 = q::allocDrafts(slots, c, {32, 9}, 2, {});
    std::uint32_t s1 = 0, s2 = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        s1 += t1.d[i];
        s2 += t2.d[i];
    }
    check(s1 <= 8 && s2 <= 9, "allocDrafts: row and snapshot limits");

    // u is returned: one slot, few cost samples, near-equal acceptance
    const q::AllocResult one = q::allocDrafts(std::span<const q::AllocSlot>(slots.data(), 1), c, {32, 0xffffffffu}, 10, {});
    check(one.uniform && one.d.size() == 1 && one.d[0] == 8, "allocDrafts: A = 1 returns u (capped)");
    const q::CycleCost fresh;
    check(q::allocDrafts(slots, fresh, {32, 0xffffffffu}, U, {}).uniform, "allocDrafts: < 16 samples returns u");
    std::vector<q::AllocSlot> same = slots;
    for (q::AllocSlot& sl : same) sl.acc = &mid;
    const q::AllocResult rs = q::allocDrafts(same, c, {32, 0xffffffffu}, U, {});
    check(rs.uniform && rs.d == u, "allocDrafts: near-equal acceptance returns u");

    // hysteresis: an allowed previous allocation within 2% of the best is kept
    std::vector<std::uint32_t> prev = r.d;
    for (std::size_t i = 0; i < 2; ++i)
        if (prev[i] > 1) {
            prev[i] -= 1;
            prev[i + 2] += 1;
            break;
        }
    const q::AllocResult h = q::allocDrafts(slots, c, {32, 0xffffffffu}, U, prev);
    const Score sp = scoreOf(slots, c, prev);
    check(sp.total >= 0.98f * r.score ? h.d == prev : h.d == r.d, "allocDrafts: keeps prev within 2% of the best");
    check(q::allocDrafts(slots, c, {32, 0xffffffffu}, U, r.d).d == r.d, "allocDrafts: stable on its own result");

    // an n-gram slot keeps its fixed rows and gets no MTP drafts
    std::vector<q::AllocSlot> mix = slots;
    mix[3].ng_rows = 6;
    mix[3].ng_e = 4.0f;
    const q::AllocResult m = q::allocDrafts(mix, c, {32, 0xffffffffu}, U, {});
    std::uint32_t rows = 6;
    for (std::size_t i = 0; i < 3; ++i) rows += m.d[i] + 1;
    check(m.d[3] == 0 && rows <= 32, "allocDrafts: n-gram slot rows count against the budget");
}
// ---------------------------------------------------------------------------
// N8: the MXFP4 load-time repack has one implementation (kernels::repackMxfp4Row,
// called by the loader's uploadMx with a per-tensor scratch). These frozen copies
// of the two pre-v0.1.4 implementations pin its output bytes, reference exponent
// and lossy count on random rows.
std::uint8_t oldLoaderRepack(std::uint8_t* row, std::uint64_t ncols, std::vector<std::uint8_t>& tmp, std::uint64_t& lossy) {
    const std::size_t nsb = static_cast<std::size_t>(ncols / 256);
    std::uint8_t emax = 0;
    for (std::size_t sb = 0; sb < nsb; ++sb)
        for (std::size_t i = 0; i < 8; ++i) {
            const std::uint8_t* blk = row + (sb * 8 + i) * 17;
            tmp[sb * 136 + i] = blk[0];
            std::memcpy(&tmp[sb * 136 + 8 + i * 16], blk + 1, 16);
            emax = std::max(emax, blk[0]);
        }
    for (std::size_t bi = 0; bi < nsb * 8; ++bi) {
        const std::uint8_t e = row[bi * 17];
        if (static_cast<std::uint32_t>(emax) - e > 8) {
            bool nz = false;
            for (std::size_t q = 0; q < 16; ++q) nz = nz || (row[bi * 17 + 1 + q] & 0x77) != 0;
            if (nz) lossy += 1;
        }
    }
    std::memcpy(row, tmp.data(), nsb * 136);
    return emax;
}

std::uint8_t oldAbiRepack(const std::uint8_t* src, std::uint8_t* dst, int ncols, std::uint64_t* lossy) {
    const int nsb = ncols / 256;
    std::vector<std::uint8_t> tmp(static_cast<std::size_t>(nsb) * 136);
    std::uint8_t emax = 0;
    for (int sb = 0; sb < nsb; ++sb) {
        for (int i = 0; i < 8; ++i) {
            const std::uint8_t* blk = src + (static_cast<std::size_t>(sb) * 8 + i) * 17;
            tmp[static_cast<std::size_t>(sb) * 136 + i] = blk[0];
            std::memcpy(&tmp[static_cast<std::size_t>(sb) * 136 + 8 + i * 16], blk + 1, 16);
            emax = std::max(emax, blk[0]);
        }
    }
    if (lossy != nullptr) {
        for (int bi = 0; bi < nsb * 8; ++bi) {
            const std::uint8_t* blk = src + static_cast<std::size_t>(bi) * 17;
            if (static_cast<unsigned>(emax) - blk[0] > 8) {
                bool nz = false;
                for (int q = 1; q <= 16; ++q) nz = nz || (blk[q] & 0x77) != 0;
                if (nz) ++*lossy;
            }
        }
    }
    std::memcpy(dst, tmp.data(), tmp.size());
    return emax;
}

void testMxfp4Repack() {
    namespace wk = whirl::kernels;
    std::uint64_t seed = 0x9E3779B97F4A7C15ull;
    auto rnd = [&] {
        seed ^= seed << 13, seed ^= seed >> 7, seed ^= seed << 17;
        return seed;
    };
    bool bytes_ok = true, emax_ok = true, lossy_ok = true, inplace_ok = true;
    std::uint64_t lossy_seen = 0;
    for (int ncols : {256, 512, 2048, 5120}) {
        const std::size_t rb = static_cast<std::size_t>(ncols) / 32 * 17;
        check(wk::rowBytes(wk::QType::mxfp4, ncols) == rb, "mxfp4 repack: row size unchanged");
        std::vector<std::uint8_t> scratch(rb), tmp(rb);
        for (int r = 0; r < 64; ++r) {
            std::vector<std::uint8_t> src(rb);
            for (auto& b : src) b = static_cast<std::uint8_t>(rnd());
            // exponents near 127 with a wide spread (some > 8 steps below the max), some zero code blocks
            for (std::size_t bi = 0; bi < rb / 17; ++bi) {
                src[bi * 17] = static_cast<std::uint8_t>(110 + rnd() % 30);
                if (rnd() % 7 == 0) std::memset(&src[bi * 17 + 1], static_cast<int>(rnd() % 2) * 0x88, 16);
            }
            if (r == 0) src[0] = 255, src[17] = 0;  // extreme exponents
            std::vector<std::uint8_t> a = src, b(rb), c(rb), d = src;
            std::uint64_t la = 0, lb = 0, lc = 0, ld = 0;
            const std::uint8_t ea = oldLoaderRepack(a.data(), static_cast<std::uint64_t>(ncols), tmp, la);
            const std::uint8_t eb = oldAbiRepack(src.data(), b.data(), ncols, &lb);
            const std::uint8_t ec = wk::repackMxfp4Row(src.data(), c.data(), ncols, &lc, scratch.data());
            const std::uint8_t ed = wk::repackMxfp4Row(d.data(), d.data(), ncols, &ld, scratch.data());  // loader: in place
            std::vector<std::uint8_t> e(rb);
            const std::uint8_t ee = wk::repackMxfp4Row(src.data(), e.data(), ncols, nullptr);
            bytes_ok = bytes_ok && a == b && b == c && e == c;
            inplace_ok = inplace_ok && d == a;
            emax_ok = emax_ok && ea == eb && eb == ec && ec == ed && ed == ee;
            lossy_ok = lossy_ok && la == lb && lb == lc && lc == ld;
            lossy_seen += la;
        }
    }
    check(bytes_ok, "mxfp4 repack: bytes identical to both previous implementations");
    check(inplace_ok, "mxfp4 repack: in-place (loader) form identical");
    check(emax_ok, "mxfp4 repack: reference exponent identical");
    check(lossy_ok, "mxfp4 repack: lossy count identical");
    check(lossy_seen > 0, "mxfp4 repack: random rows exercise the lossy count");
    std::vector<std::uint8_t> buf(17 * 8 * 2), sc(136 * 2);
    bool threw = false;
    try {
        (void)wk::repackMxfp4Row(buf.data(), buf.data(), 128, nullptr, sc.data());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "mxfp4 repack: ncols not a multiple of 256 rejected");
}

}  // namespace

// pre-load VRAM estimate (fit check / server auto-shrink)
void testLoadEstimate() {
    for (bool moe : {false, true}) {
        const std::string tag = moe ? " (moe)" : " (dense)";
        const TinyModel m(moe);
        const std::vector<std::uint8_t> img = m.build();
        const gguf::File f = gguf::File::parse(img);
        const q::Config cfg = q::Config::fromGguf(f);
        std::uint64_t all = 0, emb = 0, max_elems = 0, ffs = cfg.n_ff;
        for (const gguf::TensorInfo& t : f.tensors()) {
            all += t.nbytes();
            if (t.name == "token_embd.weight") emb = t.nbytes();
            const std::string n(t.name);
            if (t.n_dims == 2 && (n.rfind("blk.0.", 0) == 0 || n.rfind("blk.1.", 0) == 0) && n.find("norm") == std::string::npos &&
                n.find("conv1d") == std::string::npos && n.find("ffn_gate_inp") == std::string::npos &&
                n.find("_exps") == std::string::npos && n.find("ssm_a") == std::string::npos && n.find("ssm_dt") == std::string::npos) {
                max_elems = std::max<std::uint64_t>(max_elems, t.ne[0] * t.ne[1]);
                ffs = std::max<std::uint64_t>(ffs, std::max(t.ne[0], t.ne[1]));
            }
        }
        const q::LoadEstimate ev = q::estimateLoad(f, cfg, false);
        const q::LoadEstimate eh = q::estimateLoad(f, cfg, true);
        check(!ev.embd_on_host && ev.weights == all, ("estimate: all tensor bytes in VRAM" + tag).c_str());
        check(eh.embd_on_host && eh.weights == all - emb && emb > 0, ("estimate: host embedding not counted" + tag).c_str());
        check(ev.has_mtp, ("estimate: nextn layer seen" + tag).c_str());
        check(ev.max_elems == max_elems && ev.ffs == ffs, ("estimate: scratch sizes from the layer matrices" + tag).c_str());
        // one DeltaNet layer (layer 0): conv + ssm state
        const std::uint64_t st = static_cast<std::uint64_t>(cfg.d_conv - 1) * cfg.convCh() * 4 +
                                 static_cast<std::uint64_t>(cfg.n_v_heads) * cfg.d_state * cfg.headV() * 4;
        check(ev.state == st, ("estimate: recurrent state of the DeltaNet layer" + tag).c_str());
        // KV per token: 1 attention layer + MTP; f16 K and V, q8v = f16 K + int8 V, q8h = int8 K and V
        const std::uint64_t e = static_cast<std::uint64_t>(cfg.n_head_kv) * cfg.head_dim, q8 = e + e / 32 * 2;
        check(ev.kv_f16 == 2 * (4 * e) && ev.kv_q8v == 2 * (2 * e + q8) && ev.kv_q8h == 2 * (2 * q8), ("estimate: KV bytes per token" + tag).c_str());
        check(q::kvBytesPerTokenCfg(cfg, false, false, false) == 4 * e, ("kvBytesPerTokenCfg without MTP" + tag).c_str());
        {
            // FR-DFL P3-1: --draft dflash skips the MTP block: no nextn weights, no MTP KV layer
            const q::LoadEstimate en = q::estimateLoad(f, cfg, false, true);
            const std::string pre = "blk." + std::to_string(cfg.n_layer) + ".";
            std::uint64_t nextn = 0;
            for (const gguf::TensorInfo& t : f.tensors())
                if (t.name.starts_with(pre)) nextn += t.nbytes();
            check(!en.has_mtp && nextn > 0 && en.weights == all - nextn && en.kv_f16 == 4 * e && en.state == ev.state,
                  ("estimate (no MTP block): nextn weights and the MTP KV layer not counted" + tag).c_str());
        }
        // buffers: linear in the batch for multiples of 64, growing with it
        const auto b = [&](std::uint32_t B) { return q::bufferBytes(cfg, B, ev.ffs, ev.max_elems); };
        check(b(4096) - b(2048) == 2 * (b(2048) - b(1024)) && b(1024) - b(512) == b(576) - b(64) && b(64) > 0,
              ("bufferBytes linear in the batch" + tag).c_str());
        check(b(4096) > b(2048) && b(2048) > b(1024), ("bufferBytes grows with the batch" + tag).c_str());
        // KV-FIT-ESTIMATE: q4 KV allocates 256 attention splits (part_ml + part_acc for max_verify_rows
        // rows), 192 more than the 64 the estimate used to assume
        {
            const std::uint64_t d = q::bufferBytes(cfg, 1024, ev.ffs, ev.max_elems, 256) - q::bufferBytes(cfg, 1024, ev.ffs, ev.max_elems, 64);
            const std::uint64_t want = static_cast<std::uint64_t>(q::max_verify_rows) * (256 - 64) * cfg.n_head * (2 + cfg.head_dim) * 4;
            check(d == want, ("bufferBytes: q4 KV attention splits counted" + tag).c_str());
            check(q::resolveFdSplits(true) >= 64 && q::resolveFdSplits(false) >= 64 && q::resolveFdSplits(true) % 64 == 0,
                  ("resolveFdSplits multiples of 64" + tag).c_str());
            // the default parameter is the f16 / q8 KV split count
            check(q::bufferBytes(cfg, 1024, ev.ffs, ev.max_elems) == q::bufferBytes(cfg, 1024, ev.ffs, ev.max_elems, 64),
                  ("bufferBytes default splits" + tag).c_str());
            // dq_st_ml (n_head x 2 f32 per row) and the identity page table are part of the estimate
            check(b(2048) - b(1024) >= 1024ull * cfg.n_head * 2 * 4, ("bufferBytes counts dq_st_ml" + tag).c_str());
        }
        // part of the per-row cost that is easy to see: x, h, hn (3 x E f32), mtp_cat (2 x E f32), ffn_g / ffn_u (2 x ffs f32)
        check((b(2048) - b(1024)) / 1024 >= (5ull * cfg.n_embd + 2ull * ev.ffs) * 4, ("bufferBytes per-row lower bound" + tag).c_str());
    }
    {
        // tied embeddings (no output.weight): the embedding stays in VRAM even if host was asked
        TinyModel m(false);
        m.makeTensors();
        m.tensors.erase("output.weight");
        const std::vector<std::uint8_t> img = m.build();
        const gguf::File f = gguf::File::parse(img);
        const q::Config cfg = q::Config::fromGguf(f);
        std::uint64_t all = 0;
        for (const gguf::TensorInfo& t : f.tensors()) all += t.nbytes();
        const q::LoadEstimate e = q::estimateLoad(f, cfg, true);
        check(!e.embd_on_host && e.weights == all, "estimate: tied embedding counted in VRAM");
    }
    {
        // MoE buffers exceed the dense ones of the same trunk
        const TinyModel md(false), mm(true);
        const std::vector<std::uint8_t> id = md.build(), im = mm.build();
        const gguf::File fd = gguf::File::parse(id), fm = gguf::File::parse(im);
        const q::Config cd = q::Config::fromGguf(fd), cm = q::Config::fromGguf(fm);
        check(q::bufferBytes(cm, 1024, cm.n_ff, 0) > q::bufferBytes(cd, 1024, cm.n_ff, 0), "bufferBytes: MoE buffers added");
    }
}

// RC-R1 / PREFILL-R2: f32 / f16 matrices (no int8 GEMV) at n = 2..gvMax() rows take the multi-token
// GEMV loop, bitwise == n = 1 per token: up to 16 rows, up to 32 inside a wide verify (small_max 32),
// and only the n = 1 kernel is required (without gemv_<T>_4 / _8 the loop runs _1 per token).
// Host-only: Model::matmulPath is the decision matmul() dispatches on (no GPU, no kernels loaded).
void testFloatGemvDispatch() {
    using MP = q::Model::MatmulPath;
    static int fake_kernel = 0;
    for (const gguf::GgmlType ty : {gguf::GgmlType::f16, gguf::GgmlType::f32}) {
        const std::string tag = ty == gguf::GgmlType::f16 ? " (f16)" : " (f32)";
        const std::size_t t = static_cast<std::size_t>(ty);
        q::Model m;
        m.k.gemv1[t] = &fake_kernel;  // only the n = 1 instance
        q::Mat w;
        w.ty = ty;
        w.nrows = 64;
        w.ncols = 64;
        check(m.matmulPath(w, 1) == MP::gemv1, "float dispatch: n = 1 -> gemv1" + tag);
        bool all = true;
        for (std::uint32_t n = 2; n <= 16; ++n) all = all && m.matmulPath(w, n) == MP::float_gemv;
        check(all, "float dispatch: n = 2..16 -> GEMV loop with only gemv_1" + tag);
        check(m.matmulPath(w, 17) == MP::gemm_f16, "float dispatch: n = 17 outside a wide verify -> f16 GEMM" + tag);
        m.small_max = 32;  // inside a wide verify / MTP step
        all = true;
        for (std::uint32_t n = 2; n <= 32; ++n) all = all && m.matmulPath(w, n) == MP::float_gemv;
        check(all, "float dispatch: wide verify n = 2..32 -> GEMV loop with only gemv_1 (RC-R1)" + tag);
        check(m.matmulPath(w, 33) == MP::gemm_f16, "float dispatch: n = 33 -> f16 GEMM" + tag);
        m.k.gemv4[t] = m.k.gemv8[t] = &fake_kernel;
        all = true;
        for (std::uint32_t n = 2; n <= 32; ++n) all = all && m.matmulPath(w, n) == MP::float_gemv;
        check(all, "float dispatch: wide verify n = 2..32 with gemv_4 / _8 -> GEMV loop" + tag);
        m.prec_dec = true;  // precise decode: every small batch on the f16 GEMM (unchanged by RC-R1)
        check(m.matmulPath(w, 1) == MP::gemm_f16 && m.matmulPath(w, 20) == MP::gemm_f16, "float dispatch: precise decode -> f16 GEMM" + tag);
        m.prec_dec = false;
        m.k.gemvq[t] = &fake_kernel;  // a type with the int8 GEMV keeps it
        check(m.matmulPath(w, 20) == MP::gemvq, "float dispatch: int8 GEMV present -> gemvq" + tag);
    }
    // the loop covers every token exactly once whatever instances the code object has
    bool cover = true;
    for (const bool h8 : {false, true})
        for (const bool h4 : {false, true})
            for (std::uint32_t n = 1; n <= 32; ++n) {
                std::uint32_t done = 0, launches = 0;
                while (done < n) {
                    const std::uint32_t st = q::Model::floatGemvStep(n - done, h8, h4);
                    cover = cover && st >= 1 && st <= n - done && (st == 1 || (st == 4 && h4) || (st == 8 && h8));
                    done += st;
                    ++launches;
                }
                cover = cover && done == n && (h8 || h4 || launches == n);
            }
    check(cover, "float dispatch: GEMV loop steps cover n tokens (8 / 4 / 1 by the instances present)");
}

// OVH-C1: the MTP draft window (Model::attn_win) is set only around the MTP draft attention and restored
// when that attention throws. (a) Model::draftAttn, through the attnBlock test hook (throws before any
// GPU work): the hook sees the draft window, and attn_win is back to its previous value after the throw.
// (b) both MTP call sites in forward.cpp go through draftAttn and nothing else assigns attn_win (a call
// site reverted to "attn_win = ...; attnBlock(); attn_win = 0;" fails here).
void testDraftWindowRestore() {
    struct Seen {
        std::int32_t win = -1;
        int calls = 0;
    } seen;
    q::Model m;
    m.draft_window = 16384;
    m.draft_window_min = 65536;
    m.attn_test_ctx = &seen;
    m.attn_test_hook = [](void* ctx, std::int32_t win) {
        auto* s = static_cast<Seen*>(ctx);
        s->win = win;
        ++s->calls;
        throw q::ModelError("TestThrow", "attnBlock failed");
    };
    for (const std::int32_t before : {0, 7}) {
        m.attn_win = before;
        bool threw = false;
        try {
            m.draftAttn(q::AttnW{}, 1);
        } catch (const q::ModelError&) {
            threw = true;
        }
        check(threw && seen.win == q::draftWindowArg(16384, 65536) && seen.win != 0,
              "OVH-C1: draftAttn runs attnBlock with the draft window");
        check(m.attn_win == before, "OVH-C1: attn_win restored after the draft attention threw (before = " + std::to_string(before) + ")");
    }
    check(seen.calls == 2, "OVH-C1: test hook reached");
#ifdef WHIRL_SOURCE_DIR
    std::ifstream in(std::string(WHIRL_SOURCE_DIR) + "/src/model/forward.cpp", std::ios::binary);
    const std::string fw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    check(!fw.empty(), "OVH-C1: src/model/forward.cpp readable");
    std::size_t sites = 0, raw_mtp_attn = 0, assigns = 0;
    for (std::size_t p = fw.find("draftAttn(mw.attn, n)"); p != std::string::npos; p = fw.find("draftAttn(mw.attn, n)", p + 1)) ++sites;
    for (std::size_t p = fw.find("attnBlock(mw.attn"); p != std::string::npos; p = fw.find("attnBlock(mw.attn", p + 1)) ++raw_mtp_attn;
    const std::regex asg(R"(\battn_win\s*=[^=])");
    for (auto it = std::sregex_iterator(fw.begin(), fw.end(), asg); it != std::sregex_iterator(); ++it) ++assigns;
    check(sites == 2 && raw_mtp_attn == 0 && assigns == 0,
          std::format("OVH-C1: both MTP call sites use draftAttn ({} sites, {} raw MTP attnBlock calls, {} attn_win assignments)", sites,
                      raw_mtp_attn, assigns));
#endif
}

int main() {
    whirl::test::clearWhirlEnv();  // FR-TEST P3-11
    testDraftWindowRestore();
    testFloatGemvDispatch();
    testArchWhitelist();
    testLoadEstimate();
    testLoadValidation();
    testRealFiles();
    testTables();
    testDraftModel();
    testNgram();
    testDraftVocab();
    testDraftWindowArg();
    testDFlashConfig();
    testGemmrMoeShape();
    testDFlashCompat();
    testDFlashTensors();
    testDFlashDraftFile();
    testCycleCost();
    testSlotAccept();
    testAllocDrafts();
    testMxfp4Repack();
    std::printf("model tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
