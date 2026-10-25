// Unit tests of whirl-stage-bench's HIP-free part (tests/bench/stage_bench.{h,cpp}; measure-tools P4). CPU only.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Usage: whirl-stage-bench-tests [--emit FILE]  (--emit writes sample rows for the importer
// check, `db_record.py jsonl FILE --dry-run`). Exit 0 = all pass.

#include "../stage_bench.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace stage;

static int g_fail = 0, g_pass = 0;

static void check(const std::string& name, bool ok, const std::string& info = "") {
    std::printf("%s  %s", ok ? "PASS" : "FAIL", name.c_str());
    if (!ok && !info.empty()) std::printf("  (%s)", info.c_str());
    std::printf("\n");
    (ok ? g_pass : g_fail)++;
}

static ParseResult P(std::vector<std::string> a) { return parseArgs(a); }

class FakeClock : public kt::Clock {
public:
    double t = 100.0;
    double now() override { return t; }
    void sleep(double s) override { t += s; }
};

static std::string tmpPath(const std::string& name) {
    const auto p = std::filesystem::temp_directory_path() / ("whirl_stage_tests_" + name);
    std::error_code ec;
    std::filesystem::remove(p, ec);
    return p.string();
}

// ------------------------------------------------------------------ options
static void test_parse_depth() {
    check("depth 32768", parseDepth("32768") == 32768u);
    check("depth 32k", parseDepth("32k") == 32768u);
    check("depth 128K", parseDepth("128K") == 131072u);
    check("depth 0 rejected", !parseDepth("0").has_value());
    check("depth junk rejected", !parseDepth("12q").has_value() && !parseDepth("").has_value() && !parseDepth("k").has_value());
    check("depth > 1M rejected", !parseDepth("2000k").has_value());
    check("depth negative rejected", !parseDepth("-5").has_value());
}

static void test_parse_layers() {
    auto r = parseLayers("3-7");
    check("layers 3-7", r && r->first == 3 && r->second == 7);
    r = parseLayers("5");
    check("layers 5", r && r->first == 5 && r->second == 5);
    check("layers 7-3 rejected", !parseLayers("7-3").has_value());
    check("layers junk rejected", !parseLayers("a-b").has_value() && !parseLayers("1-").has_value() && !parseLayers("").has_value());
}

static void test_parse_args() {
    auto r = P({"m.gguf", "--stage", "decode_step", "--depth", "32k", "--jsonl", "o.jsonl"});
    check("args ok", r.opts.has_value() && r.error.empty(), r.error);
    if (r.opts) {
        check("args model", r.opts->model == "m.gguf");
        check("args stage", r.opts->stage == "decode_step");
        check("args depth", r.opts->depth == 32768);
        check("args jsonl", r.opts->jsonl == "o.jsonl");
        check("args defaults", r.opts->depth_source == "synthetic" && r.opts->prefix == 4096 && r.opts->tol == 0.02 && r.opts->state == "burst");
    }
    check("no model -> error", !P({"--stage", "decode_step", "--depth", "1k"}).error.empty());
    check("no depth -> error", !P({"m.gguf", "--stage", "decode_step"}).error.empty());
    check("unknown stage -> error", P({"m.gguf", "--stage", "foo", "--depth", "1k"}).error.find("unknown stage") != std::string::npos);
    const auto ni = P({"m.gguf", "--stage", "mtp_draft", "--depth", "1k"});
    check("design stage not implemented -> error with reason", ni.error.find("not implemented") != std::string::npos, ni.error);
    check("unknown option -> error", !P({"m.gguf", "--depth", "1k", "--bogus"}).error.empty());
    check("missing value -> error", !P({"m.gguf", "--depth"}).error.empty());
    check("bad depth-source", !P({"m.gguf", "--depth", "1k", "--depth-source", "x"}).error.empty());
    check("layers stage needs --layers", !P({"m.gguf", "--stage", "layers", "--depth", "1k"}).error.empty());
    check("--layers on prefill_chunk rejected", !P({"m.gguf", "--stage", "prefill_chunk", "--depth", "1k", "--layers", "1-2"}).error.empty());
    auto lr = P({"m.gguf", "--stage", "layers", "--depth", "1k", "--layers", "3-4"});
    check("layers stage ok", lr.opts && lr.opts->layers && lr.opts->layers->first == 3, lr.error);
    check("real needs --prompt-file", !P({"m.gguf", "--depth", "1k", "--depth-source", "real"}).error.empty());
    check("both ok with prompt", P({"m.gguf", "--depth", "1k", "--depth-source", "both", "--prompt-file", "p.txt"}).opts.has_value());
    check("verify rows 1 rejected", !P({"m.gguf", "--stage", "verify", "--depth", "1k", "--rows", "1"}).error.empty());
    check("verify rows 17 rejected", !P({"m.gguf", "--stage", "verify", "--depth", "1k", "--rows", "17"}).error.empty());
    check("decode rows 4 rejected", !P({"m.gguf", "--stage", "decode_step", "--depth", "1k", "--rows", "4"}).error.empty());
    auto pre = P({"m.gguf", "--depth", "1k", "--preload-s", "150"});
    check("preload -> sustained state", pre.opts && pre.opts->state == "sustained" && pre.opts->preload_s == 150);
    check("prefix 0 rejected", !P({"m.gguf", "--depth", "1k", "--prefix", "0"}).error.empty());
    check("help", P({"--help"}).help);
    check("mode precise", P({"m.gguf", "--depth", "1k", "--mode", "precise"}).opts->mode == "precise");
    check("mode bad", !P({"m.gguf", "--depth", "1k", "--mode", "turbo"}).error.empty());
    auto ms = P({"m.gguf", "--stage", "decode_step,prefill_chunk,layer.attn", "--depth", "1k"});
    check("multi-stage ok, prefill_chunk first", ms.opts && ms.opts->stages == std::vector<std::string>({"prefill_chunk", "decode_step", "layer.attn"}) &&
                                                     ms.opts->stage == "prefill_chunk",
          ms.error);
    check("multi-stage + --rows rejected", !P({"m.gguf", "--stage", "decode_step,prefill_chunk", "--depth", "1k", "--rows", "1"}).error.empty());
    check("multi-stage duplicate rejected", !P({"m.gguf", "--stage", "decode_step,decode_step", "--depth", "1k"}).error.empty());
    check("multi-stage empty name rejected", !P({"m.gguf", "--stage", "decode_step,", "--depth", "1k"}).error.empty());
    check("multi-stage one unknown rejected", !P({"m.gguf", "--stage", "decode_step,foo", "--depth", "1k"}).error.empty());
    check("--layers with two layer stages rejected", !P({"m.gguf", "--stage", "layer.attn,layer.gdn", "--depth", "1k", "--layers", "3"}).error.empty());
    check("--layers with one layer stage + others ok", P({"m.gguf", "--stage", "decode_step,layers", "--depth", "1k", "--layers", "3"}).opts.has_value());
    check("single stage rows kept", P({"m.gguf", "--stage", "verify", "--depth", "1k", "--rows", "8"}).opts->rows == 8);
    check("default rows 0 (per stage)", P({"m.gguf", "--depth", "1k"}).opts->rows == 0);
    check("tol bad", !P({"m.gguf", "--depth", "1k", "--tol", "0"}).error.empty());
}

static void test_stage_table() {
    const char* design[] = {"embed", "layer.attn", "layer.gdn", "layer.moe", "lm_head", "decode_step", "verify", "mtp_draft",
                            "prefill_chunk", "vision_encode"};
    bool all = true;
    for (const char* n : design) all = all && findStage(n) != nullptr;
    check("every DESIGN stage is in the table", all);
    for (const char* n : {"prefill_chunk", "decode_step", "verify", "head", "layers", "layer.attn", "layer.gdn"}) {
        const StageEntry* e = findStage(n);
        check(std::string("implemented: ") + n, e && e->implemented);
    }
    for (const char* n : {"mtp_draft", "vision_encode", "layer.moe", "embed", "lm_head"}) {
        const StageEntry* e = findStage(n);
        check(std::string("not implemented, with a note: ") + n, e && !e->implemented && std::strlen(e->info.note) > 10);
    }
    check("unknown -> null", findStage("nope") == nullptr);
}

// ------------------------------------------------------------------ tile plan
static void test_tile_plan() {
    for (auto [P0, D] : std::vector<std::pair<std::uint64_t, std::uint64_t>>{{4096, 131072}, {4096, 129024}, {4096, 32768}, {3000, 100000}, {256, 257}, {7, 7}, {10, 5}}) {
        const auto plan = tilePlan(P0, D);
        const auto rows = applyPlan(plan, P0, D);
        bool ok = rows.size() == std::max(D, P0);
        for (std::uint64_t r = 0; ok && r < D; ++r) ok = rows[r] == r % P0;
        bool no_overlap = true, reads_final = true;
        std::uint64_t filled = std::min(P0, D);
        for (const Copy& c : plan) {
            no_overlap = no_overlap && (c.src_row + c.rows <= c.dst_row);
            reads_final = reads_final && c.src_row + c.rows <= filled && c.dst_row == filled;
            filled += c.rows;
        }
        const std::string tag = std::to_string(P0) + "->" + std::to_string(D);
        check("tile " + tag + ": row r holds r % P", ok);
        check("tile " + tag + ": no copy overlaps its destination, reads final rows only", no_overlap && reads_final);
        check("tile " + tag + ": covers exactly [P, D)", filled == std::max(D, std::min(P0, D)));
        if (D > P0) check("tile " + tag + ": doubling (<= log2(D/P)+1 copies)", plan.size() <= static_cast<std::size_t>(std::ceil(std::log2(double(D) / double(P0)))) + 1, std::to_string(plan.size()));
    }
    check("tile depth <= prefix: no copies", tilePlan(4096, 2048).empty() && tilePlan(4096, 4096).empty());
    check("tile prefix 0: no copies", tilePlan(0, 100).empty());
}

static void test_subset_layers() {
    std::string err;
    const StageInfo head = findStage("head")->info;
    check("head: no layers", subsetLayers(head, std::nullopt, 64, err).empty() && err.empty());
    const StageInfo lay = findStage("layers")->info;
    auto v = subsetLayers(lay, std::make_pair(2u, 4u), 64, err);
    check("layers 2-4", v == std::vector<std::uint32_t>({2, 3, 4}) && err.empty());
    err.clear();
    v = subsetLayers(lay, std::make_pair(60u, 64u), 64, err);
    check("layers past n_layer -> error", v.empty() && !err.empty());
    err.clear();
    const StageInfo la = findStage("layer.attn")->info;
    v = subsetLayers(la, std::nullopt, 64, err);
    check("layer.attn default = layer 3 (interval 4)", v == std::vector<std::uint32_t>({3}), err);
    const StageInfo lg = findStage("layer.gdn")->info;
    v = subsetLayers(lg, std::nullopt, 64, err);
    check("layer.gdn default = layer 0", v == std::vector<std::uint32_t>({0}));
    err.clear();
    v = subsetLayers(findStage("prefill_chunk")->info, std::nullopt, 64, err);
    check("prefill_chunk: all layers", v.size() == 64 && v.front() == 0 && v.back() == 63);
}

// ------------------------------------------------------------------ golden
static void test_golden() {
    GoldenKey k;
    k.stage = "decode_step";
    k.model_sha = "abc";
    k.depth = 32768;
    k.rows = 1;
    k.mode = "balance";
    k.kv = "q8";
    k.arch = "gfx1201";
    k.depth_source = "synthetic";
    check("golden key text", k.str() == "decode_step|abc|d32768|n1|L-|balance|q8|gfx1201|synthetic|in-", k.str());
    const std::string path = tmpPath("golden.tsv");
    GoldenFile g = goldenLoad(path);
    check("golden missing file = empty", g.entries.empty() && g.error.empty());
    std::string err;
    check("golden absent, no record", goldenCheck(path, g, k, "h1", false, err) == GoldenStatus::absent && !std::filesystem::exists(path));
    check("golden recorded", goldenCheck(path, g, k, "h1", true, err) == GoldenStatus::recorded && err.empty(), err);
    g = goldenLoad(path);
    check("golden reload", g.entries.size() == 1 && g.entries[k.str()] == "h1");
    check("golden match", goldenCheck(path, g, k, "h1", true, err) == GoldenStatus::match);
    check("golden mismatch never overwritten", goldenCheck(path, g, k, "h2", true, err) == GoldenStatus::mismatch && goldenLoad(path).entries[k.str()] == "h1");
    GoldenKey k2 = k;
    k2.depth = 131072;
    check("golden second key appended", goldenCheck(path, g, k2, "h3", true, err) == GoldenStatus::recorded && goldenLoad(path).entries.size() == 2);
    GoldenFile bad = goldenParse("a\th1\na\th2\n");
    check("golden duplicate key with other hash = error", !bad.error.empty());
    GoldenFile same = goldenParse("# c\n\na\th1\na\th1\n");
    check("golden comment, blank, same duplicate ok", same.error.empty() && same.entries.size() == 1);
    check("golden malformed line = error", !goldenParse("no-tab-here\n").error.empty());
    check("golden status names", std::string(goldenStatusName(GoldenStatus::mismatch)) == "mismatch");
    GoldenKey k3 = k;
    k3.input = "0123456789abcdef";
    check("golden key input part", k3.str() != k.str() && k3.str().find("|in0123456789abcdef") != std::string::npos);
    std::filesystem::remove(path);
}

// ------------------------------------------------------------------ judge
static void test_judge() {
    check("judge 1.01 pass", judge(101, 100, 0.02).pass);
    check("judge 0.98 pass (edge)", judge(98, 100, 0.02).pass);
    check("judge 1.03 fail", !judge(103, 100, 0.02).pass);
    check("judge 0.97 fail", !judge(97, 100, 0.02).pass);
    check("judge ratio", std::fabs(judge(110, 100, 0.02).ratio - 1.1) < 1e-12);
    check("judge zero real fail", !judge(100, 0, 0.02).pass && !judge(NAN, 100, 0.02).pass);
}

// ------------------------------------------------------------------ rows
static RowInfo sampleInfo() {
    RowInfo ri;
    ri.stage = "prefill_chunk";
    ri.model_tag = "swift27b-mxfp4a";
    ri.mode = "balance";
    ri.kv = "q8";
    ri.depth_source = "synthetic";
    ri.gpu_arch = "gfx1201";
    ri.gpu_pci = "pci:71:0";
    ri.code_hash = "c0de";
    ri.model_sha = "m0de";
    ri.started = "2026-10-25T16:00:00";
    ri.tool_ver = "0.2.4+stagebench1";
    ri.state = "burst";
    ri.depth = 32768;
    ri.rows = 2048;
    ri.prefix = 4096;
    ri.out_hash = "feed";
    ri.golden_status = "match";
    ri.max_cell_s = 20;
    ri.rel = 0.02;
    return ri;
}

static void test_rows() {
    RowInfo ri = sampleInfo();
    check("cell key", cellKey(ri) == "swift27b-mxfp4a|d32768|n2048|balance", cellKey(ri));
    ri.layers = "3-3";
    check("cell key with layers", cellKey(ri) == "swift27b-mxfp4a|d32768|n2048|balance|L3-3");
    ri.layers.clear();
    const std::string sk = settingsKey(ri);
    check("settings key fields", sk.find("state=burst") != std::string::npos && sk.find("depth_source=synthetic") != std::string::npos &&
                                     sk.find("fill=tile") != std::string::npos && sk.find("prefix=4096") != std::string::npos &&
                                     sk.find("kv=q8") != std::string::npos && sk.find("max_cell_s=20") != std::string::npos,
          sk);
    RowInfo rr = ri;
    rr.depth_source = "real";
    check("settings key real: fill=prefill, no prefix", settingsKey(rr).find("fill=prefill") != std::string::npos && settingsKey(rr).find("prefix=") == std::string::npos, settingsKey(rr));
    StatsIn s;
    s.samples = {1000, 1010, 990, 1005, 995};
    s.st = whirl::bench::summarize(s.samples);
    s.stop_reason = "rel<=2%";
    s.rel_iqr = 0.01;
    const whirl::bench::Row r = stageRow(ri, s);
    const std::string j = whirl::bench::to_json(r);
    check("row metric", r.metric == "stage.prefill_chunk" && r.unit == "us" && r.tool == "whirl-stage-bench");
    check("row component", r.component == "stage");
    check("row value = median", std::fabs(r.value - 1000) < 1e-9);
    check("row json has depth/rows/out_hash/golden/tok_s", j.find("\"depth\": 32768") != std::string::npos && j.find("\"rows\": 2048") != std::string::npos &&
                                                       j.find("\"out_hash\": \"feed\"") != std::string::npos &&
                                                       j.find("\"golden\": \"match\"") != std::string::npos && j.find("\"tok_s\": ") != std::string::npos,
          j);
    check("row gpu + hashes", r.gpu_arch == "gfx1201" && r.gpu_pci == "pci:71:0" && r.code_hash == "c0de" && j.find("\"model_sha256\": \"m0de\"") != std::string::npos);
    StatsIn bad;
    bad.stop_reason = "error";
    const whirl::bench::Row rb = stageRow(ri, bad);
    check("error row: value null", whirl::bench::to_json(rb).find("\"value\": null") != std::string::npos);
    const whirl::bench::Row ir = inrunRow(rr, 1234.5);
    check("in-run row", ir.metric == "stage.prefill_chunk" && ir.n == 1 && ir.settings.find("depth_source=real_inrun") != std::string::npos && std::fabs(ir.value - 1234.5) < 1e-9,
          ir.settings);
    const whirl::bench::Row vr = verdictRow(ri, judge(1010, 1000, 0.02), "real_inrun");
    const std::string vj = whirl::bench::to_json(vr);
    check("verdict row", vr.metric == "stage.prefill_chunk.synth_over_real" && vr.unit == "ratio" && vj.find("\"pass\": true") != std::string::npos &&
                             vj.find("\"against\": \"real_inrun\"") != std::string::npos,
          vj);
    RowInfo li = ri;
    li.stage = "layer.attn";
    li.net_us = 123.0;
    check("layer row net_us", whirl::bench::to_json(stageRow(li, s)).find("\"net_us\": 123") != std::string::npos);
}

static void test_repeat_and_chunks() {
    check("repeatTo", repeatTo({1, 2, 3}, 7) == std::vector<std::uint32_t>({1, 2, 3, 1, 2, 3, 1}));
    check("repeatTo shorter", repeatTo({1, 2, 3}, 2) == std::vector<std::uint32_t>({1, 2}));
    check("repeatTo empty", repeatTo({}, 5).empty());
    check("chunkEnds", chunkEnds(5000, 2048) == std::vector<std::uint32_t>({2048, 4096, 5000}));
    check("chunkEnds exact", chunkEnds(4096, 2048) == std::vector<std::uint32_t>({2048, 4096}));
    check("chunkEnds 0", chunkEnds(0, 2048).empty());
    const auto e = chunkEnds(131072, 2048);
    check("chunkEnds 128k has a boundary at 129024", std::find(e.begin(), e.end(), 129024u) != e.end());
}

static void test_estimate() {
    EstIn e;
    e.depth = 131072;
    e.prefix = 4096;
    e.rows = 2048;
    e.prefill_tok_s = 300;  // 8060S Swift ~ 131072 / 450 s
    e.real = true;
    e.synthetic = true;
    e.max_cell_s = 20;
    const double s = estimateSeconds(e);
    check("estimate real 128k >= real prefill", s >= 131072.0 / 300 && s < 131072.0 / 300 + 400, std::to_string(s));
    e.real = false;
    const double s2 = estimateSeconds(e);
    check("estimate synthetic << real", s2 < 200 && s2 > 20, std::to_string(s2));
}

// ------------------------------------------------------------------ driver
// A fake stage: state = a counter the call mutates; restore resets it; out_hash reads it. A broken restore must be
// detected as inconsistent.
static void test_driver() {
    FakeClock clk;
    int state = 7, restores = 0, calls = 0;
    bool broken_restore = false;
    double timed_ms = 2.0;
    std::vector<int> state_at_timed;
    Driver d;
    d.restore = [&] {
        ++restores;
        if (!broken_restore) state = 7;
    };
    d.call = [&] {
        ++calls;
        state = state * 3 + 1;
        clk.t += 0.002;
    };
    d.sync = [] {};
    d.out_hash = [&] { return std::to_string(state); };
    d.timed = [&](int n, std::uint64_t) {
        state_at_timed.push_back(state);
        for (int j = 0; j < n; ++j) d.call();
        return timed_ms * n;
    };
    kt::BenchCfg cfg;
    cfg.min_samples = 5;
    cfg.max_cell_s = 5;
    cfg.rel_target = 0.02;
    cfg.warmup_ms = 1;
    kt::Bench b(cfg, &clk);
    StageResult r = runStage(d, b, 0.0, &clk);
    check("driver ok", r.cs.ok(), r.cs.error);
    check("driver consistent", r.consistent && r.hash_first == "22" && r.hash_last == "22", r.hash_first + " " + r.hash_last);
    check("driver median = 2000 us", std::fabs(r.cs.st.median - 2000) < 1e-6);
    bool fresh = true;
    for (int v : state_at_timed) fresh = fresh && v == 7;
    check("driver: every timed block starts from the restored state", fresh && !state_at_timed.empty());
    check("driver block 1", r.cs.block == 1, std::to_string(r.cs.block));
    // preload
    FakeClock clk2;
    kt::Bench b2(cfg, &clk2);
    Driver d2 = d;
    d2.call = [&] {
        state = state * 3 + 1;
        clk2.t += 0.5;
    };
    d2.timed = [&](int n, std::uint64_t) {
        for (int j = 0; j < n; ++j) d2.call();
        return 500.0 * n;
    };
    state = 7;
    StageResult r2 = runStage(d2, b2, 10.0, &clk2);
    check("driver preload >= 10 s", r2.preload_s >= 10.0 && r2.preload_calls >= 20, std::to_string(r2.preload_s) + " " + std::to_string(r2.preload_calls));
    // broken restore -> inconsistent
    broken_restore = true;
    state = 7;
    FakeClock clk3;
    kt::Bench b3(cfg, &clk3);
    StageResult r3 = runStage(d, b3, 0.0, &clk3);
    check("driver: broken restore detected (inconsistent)", !r3.consistent);
    broken_restore = false;
    // timer error -> not ok
    timed_ms = 0;
    FakeClock clk4;
    kt::Bench b4(cfg, &clk4);
    StageResult r4 = runStage(d, b4, 0.0, &clk4);
    check("driver: zero timer -> error", !r4.cs.ok());
}

int main(int argc, char** argv) {
    std::string emit;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--emit") == 0 && i + 1 < argc) emit = argv[++i];
    try {
        test_parse_depth();
        test_parse_layers();
        test_parse_args();
        test_stage_table();
        test_tile_plan();
        test_subset_layers();
        test_golden();
        test_judge();
        test_rows();
        test_repeat_and_chunks();
        test_estimate();
        test_driver();
    } catch (const std::exception& e) {
        check(std::string("exception: ") + e.what(), false);
    }
    if (!emit.empty()) {
        whirl::bench::JsonlWriter w(emit, false);
        RowInfo ri = sampleInfo();
        StatsIn s;
        s.samples = {1000, 1010, 990, 1005, 995};
        s.st = whirl::bench::summarize(s.samples);
        s.stop_reason = "rel<=2%";
        w.write(stageRow(ri, s));
        RowInfo rr = ri;
        rr.depth_source = "real";
        w.write(inrunRow(rr, 1003));
        w.write(verdictRow(ri, judge(1000, 1003, 0.02), "real_inrun"));
        std::printf("emitted %lld rows to %s\n", static_cast<long long>(w.lines()), emit.c_str());
    }
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
