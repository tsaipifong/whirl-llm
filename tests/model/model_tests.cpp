// Host-only unit tests of the model library (no GPU work): architecture
// whitelist, tune buckets, GEMV tables, draft-count cost model, n-gram drafter, per-slot
// draft allocation (cycle cost, slot acceptance, allocDrafts).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/gguf.h"
#include "whirl/model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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
}  // namespace

int main() {
    testArchWhitelist();
    testTables();
    testDraftModel();
    testNgram();
    testDraftVocab();
    testCycleCost();
    testSlotAccept();
    testAllocDrafts();
    std::printf("model tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
