// Host-only unit tests of the model library (no GPU work): architecture
// whitelist, tune buckets, GEMV tables, draft-count cost model, n-gram drafter.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/gguf.h"
#include "whirl/model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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
    std::vector<std::uint8_t> build() {
        std::vector<std::uint8_t> out;
        u32(out, 0x46554747);  // "GGUF"
        u32(out, 3);
        u64(out, 0);  // tensors
        u64(out, n_kv);
        out.insert(out.end(), kv.begin(), kv.end());
        while (out.size() % 32) out.push_back(0);
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
    check(sizeof(q::KvArgs) == 56 && sizeof(q::RowTab) == 192 && sizeof(q::AwGroups) == 128, "KvArgs / RowTab / AwGroups layout");
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
    tm.update(6, 14.0f);
    const float e4 = *tm.estimate(4);
    check(e4 > 11.5f && e4 < 12.5f, "DraftTiming linear fit");
    q::DraftAccept hi(0.95f);
    const q::DraftAccept* accs[1] = {&hi};
    const std::uint32_t nd = q::pickDrafts(accs, tm, 8, 0, 0);
    check(nd >= 4, "pickDrafts prefers long chains at high acceptance");
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

}  // namespace

int main() {
    testArchWhitelist();
    testTables();
    testDraftModel();
    testNgram();
    std::printf("model tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
