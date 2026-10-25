// whirl-stage-bench, HIP-free part (see stage_bench.h). SPDX-License-Identifier: Apache-2.0
// Written for WHIRL (measure-tools P4).

#include "stage_bench.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace stage {

namespace wb = whirl::bench;

// ---------------------------------------------------------------- stages
const std::vector<StageEntry>& stageTable() {
    static const std::vector<StageEntry> t = {
        {{"prefill_chunk", Kind::prefill_chunk, false, 0, "forward of the server's chunk rows at pos0 = D"}, true},
        {{"decode_step", Kind::decode_step, false, 1, "one decode step at D (decodeStep, graph replay)"}, true},
        {{"verify", Kind::verify, false, 4, "Model::verify of rows 2..16 at D (all logits, argmax rows)"}, true},
        {{"head", Kind::head, true, 1, "embedding + final norm + output head (no trunk layers)"}, true},
        {{"layers", Kind::layers, true, 1, "embedding + the --layers a-b trunk layers + head"}, true},
        {{"layer.attn", Kind::layers, true, 1, "embedding + one full-attention layer (default: the first) + head; net_us = minus head"}, true},
        {{"layer.gdn", Kind::layers, true, 1, "embedding + one DeltaNet layer (default: layer 0) + head; net_us = minus head"}, true},
        {{"embed", Kind::head, true, 1,
          "not separable without a test seam in forward.cpp (run() always ends with the head): use stage head "
          "(embedding + final norm + head) instead"},
         false},
        {{"lm_head", Kind::head, true, 1, "not separable without a test seam (see embed): use stage head"}, false},
        {{"layer.moe", Kind::layers, true, 1,
          "router + experts alone need a seam into the private Model::moeBlock; layer.attn / layer.gdn / layers include "
          "the layer's FFN / MoE block"},
         false},
        {{"mtp_draft", Kind::layers, false, 1, "MTP block + draft head: deferred (needs the MTP KV filled too; follow-up)"}, false},
        {{"vision_encode", Kind::layers, false, 0, "vision encoder: deferred (no depth dependence; whirl-kernel-test vision covers it)"}, false},
    };
    return t;
}

const StageEntry* findStage(const std::string& name) {
    for (const StageEntry& e : stageTable())
        if (name == e.info.name) return &e;
    return nullptr;
}

// ---------------------------------------------------------------- options
std::string usage() {
    return "usage: whirl-stage-bench MODEL.gguf --stage S --depth D [--layers a-b] [--jsonl F]\n"
           "  --stage S          prefill_chunk (default) | decode_step | verify | head | layers | layer.attn | layer.gdn\n"
           "  --depth D          KV / state depth in tokens (32768, 32k, 128k)\n"
           "  --depth-source X   synthetic (default: tile a real prefix) | real (real prefill) | both (validation)\n"
           "  --prefix P         real tokens tiled into the synthetic depth (default 4096)\n"
           "  --rows N           rows of the stage (prefill_chunk: the server's chunk; verify 2..16; decode 1)\n"
           "  --layers a-b       trunk layers of the layers / layer.* stages (inclusive)\n"
           "  --prompt-file F    UTF-8 text for the real tokens (repeated to length); required for real / both\n"
           "  --golden F         per-stage golden file (key<TAB>sha256); --record-golden appends missing keys\n"
           "  --mode M           balance (default) | precise | fast\n"
           "  --device SPEC      index, gfx arch, r9700 or 8060s (the CLI's --device rule)\n"
           "  --model-sha H      the model file's sha256 for the golden key and rows (default: name:size)\n"
           "  --max-cell-s S     measuring time per stage cell (default 20)   --min-samples N (5)  --max-samples N (200)\n"
           "  --rel R            stop at IQR / median <= R (default 0.02)    --tol T  validation band (0.02)\n"
           "  --preload-s S      untimed restored calls for S seconds first (declares state=sustained)\n"
           "  --jsonl F          append one row per cell (tests/bench/bench_out.h contract)\n"
           "  --dry-run          print the plan and the estimate, no GPU\n"
           "exit: 0 ok, 1 output / golden / consistency failure, 2 usage, 3 validation out of band\n";
}

std::optional<std::uint32_t> parseDepth(const std::string& s0) {
    if (s0.empty()) return std::nullopt;
    std::string s = s0;
    std::uint64_t mul = 1;
    if (s.back() == 'k' || s.back() == 'K') {
        mul = 1024;
        s.pop_back();
    }
    if (s.empty() || s.size() > 9) return std::nullopt;
    for (char c : s)
        if (!std::isdigit(static_cast<unsigned char>(c))) return std::nullopt;
    const std::uint64_t v = std::stoull(s) * mul;
    if (v == 0 || v > (1u << 20)) return std::nullopt;
    return static_cast<std::uint32_t>(v);
}

namespace {
std::optional<std::uint32_t> parseU(const std::string& s) {
    if (s.empty() || s.size() > 9) return std::nullopt;
    for (char c : s)
        if (!std::isdigit(static_cast<unsigned char>(c))) return std::nullopt;
    return static_cast<std::uint32_t>(std::stoul(s));
}
std::optional<double> parseD(const std::string& s) {
    try {
        std::size_t used = 0;
        const double v = std::stod(s, &used);
        if (used != s.size() || !std::isfinite(v)) return std::nullopt;
        return v;
    } catch (...) {
        return std::nullopt;
    }
}
std::string withKv(std::string s, const std::string& kv) { return s.empty() ? kv : s + ";" + kv; }
std::string fmtNum(double v) {
    char b[48];
    std::snprintf(b, sizeof(b), "%g", v);
    return b;
}
}  // namespace

std::optional<std::pair<std::uint32_t, std::uint32_t>> parseLayers(const std::string& s) {
    const auto dash = s.find('-');
    if (dash == std::string::npos) {
        const auto a = parseU(s);
        if (!a) return std::nullopt;
        return std::make_pair(*a, *a);
    }
    const auto a = parseU(s.substr(0, dash)), b = parseU(s.substr(dash + 1));
    if (!a || !b || *a > *b) return std::nullopt;
    return std::make_pair(*a, *b);
}

ParseResult parseArgs(const std::vector<std::string>& args) {
    ParseResult pr;
    Opts o;
    bool have_depth = false, have_rows = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto next = [&](std::string& dst) {
            if (i + 1 >= args.size()) {
                pr.error = "missing value after " + a;
                return false;
            }
            dst = args[++i];
            return true;
        };
        std::string v;
        if (a == "-h" || a == "--help") {
            pr.help = true;
            return pr;
        } else if (a == "--stage") {
            if (!next(o.stage)) return pr;
        } else if (a == "--depth") {
            if (!next(v)) return pr;
            const auto d = parseDepth(v);
            if (!d) return pr.error = "bad --depth '" + v + "'", pr;
            o.depth = *d;
            have_depth = true;
        } else if (a == "--depth-source") {
            if (!next(o.depth_source)) return pr;
            if (o.depth_source != "synthetic" && o.depth_source != "real" && o.depth_source != "both")
                return pr.error = "bad --depth-source '" + o.depth_source + "'", pr;
        } else if (a == "--prefix") {
            if (!next(v)) return pr;
            const auto d = parseDepth(v);
            if (!d) return pr.error = "bad --prefix '" + v + "'", pr;
            o.prefix = *d;
        } else if (a == "--rows") {
            if (!next(v)) return pr;
            const auto d = parseU(v);
            if (!d || *d == 0) return pr.error = "bad --rows '" + v + "'", pr;
            o.rows = *d;
            have_rows = true;
        } else if (a == "--layers") {
            if (!next(v)) return pr;
            o.layers = parseLayers(v);
            if (!o.layers) return pr.error = "bad --layers '" + v + "'", pr;
        } else if (a == "--jsonl") {
            if (!next(o.jsonl)) return pr;
        } else if (a == "--golden") {
            if (!next(o.golden)) return pr;
        } else if (a == "--record-golden") {
            o.record_golden = true;
        } else if (a == "--prompt-file") {
            if (!next(o.prompt_file)) return pr;
        } else if (a == "--device") {
            if (!next(o.device)) return pr;
        } else if (a == "--model-sha") {
            if (!next(o.model_sha)) return pr;
        } else if (a == "--mode") {
            if (!next(o.mode)) return pr;
            if (o.mode != "balance" && o.mode != "precise" && o.mode != "fast") return pr.error = "bad --mode '" + o.mode + "'", pr;
        } else if (a == "--max-cell-s" || a == "--rel" || a == "--preload-s" || a == "--tol") {
            if (!next(v)) return pr;
            const auto d = parseD(v);
            const bool zero_ok = a == "--preload-s";
            if (!d || *d < 0 || (!zero_ok && *d == 0)) return pr.error = "bad " + a + " '" + v + "'", pr;
            (a == "--max-cell-s" ? o.max_cell_s : a == "--rel" ? o.rel : a == "--preload-s" ? o.preload_s : o.tol) = *d;
        } else if (a == "--min-samples" || a == "--max-samples") {
            if (!next(v)) return pr;
            const auto d = parseU(v);
            if (!d || *d == 0) return pr.error = "bad " + a + " '" + v + "'", pr;
            (a == "--min-samples" ? o.min_samples : o.max_samples) = static_cast<int>(*d);
        } else if (a == "--dry-run") {
            o.dry_run = true;
        } else if (!a.empty() && a[0] == '-') {
            return pr.error = "unknown option " + a, pr;
        } else if (o.model.empty()) {
            o.model = a;
        } else {
            return pr.error = "unexpected argument " + a, pr;
        }
    }
    if (o.model.empty()) return pr.error = "no MODEL.gguf", pr;
    if (!have_depth) return pr.error = "--depth is required", pr;
    // --stage a,b,c: several stages on ONE depth state (one real prefill for all of them); prefill_chunk first, so
    // that in a real run its first call is the in-run chunk right after the prefill
    o.stages.clear();
    {
        std::string cur;
        for (std::size_t i = 0; i <= o.stage.size(); ++i) {
            if (i == o.stage.size() || o.stage[i] == ',') {
                if (cur.empty()) return pr.error = "empty stage name in --stage '" + o.stage + "'", pr;
                if (std::find(o.stages.begin(), o.stages.end(), cur) != o.stages.end()) return pr.error = "stage " + cur + " twice", pr;
                o.stages.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(o.stage[i]);
            }
        }
    }
    std::stable_partition(o.stages.begin(), o.stages.end(), [](const std::string& s) { return s == "prefill_chunk"; });
    if (have_rows && o.stages.size() > 1) return pr.error = "--rows with several stages is ambiguous (each stage uses its default)", pr;
    bool any_layers = false;
    for (const std::string& sn : o.stages) {
        const StageEntry* e = findStage(sn);
        if (!e) return pr.error = "unknown stage '" + sn + "'", pr;
        if (!e->implemented) return pr.error = "stage " + sn + " not implemented: " + e->info.note, pr;
        const Kind k = e->info.kind;
        any_layers = any_layers || k == Kind::layers;
        if (sn == "layers" && !o.layers) return pr.error = "stage layers needs --layers a-b", pr;
        if (have_rows) {
            if (k == Kind::decode_step && o.rows != 1) return pr.error = "decode_step has 1 row", pr;
            if (k == Kind::verify && (o.rows < 2 || o.rows > 16)) return pr.error = "verify rows 2..16", pr;
        }
    }
    if (o.layers && !any_layers) return pr.error = "--layers only with the layers / layer.* stages", pr;
    if (o.layers && o.stages.size() > 1) {
        int n = 0;
        for (const std::string& sn : o.stages) n += findStage(sn)->info.kind == Kind::layers ? 1 : 0;
        if (n > 1) return pr.error = "--layers with several layer stages is ambiguous", pr;
    }
    o.stage = o.stages.front();
    if (o.prefix == 0) return pr.error = "bad --prefix", pr;
    if (o.depth_source != "synthetic" && o.prompt_file.empty()) return pr.error = "--depth-source " + o.depth_source + " needs --prompt-file", pr;
    if (o.preload_s > 0) o.state = "sustained";
    if (o.min_samples > o.max_samples) return pr.error = "--min-samples > --max-samples", pr;
    pr.opts = o;
    return pr;
}

// ---------------------------------------------------------------- fill
std::vector<Copy> tilePlan(std::uint64_t prefix, std::uint64_t depth) {
    std::vector<Copy> plan;
    if (prefix == 0 || depth <= prefix) return plan;
    std::uint64_t filled = prefix;
    while (filled < depth) {
        // copy rows [0, n) to [filled, filled + n): filled is a multiple of prefix, so row filled + j gets row j,
        // which holds row j % prefix -> (filled + j) % prefix
        const std::uint64_t n = std::min(filled, depth - filled);
        plan.push_back({0, filled, n});
        filled += n;
    }
    return plan;
}

std::vector<std::uint64_t> applyPlan(const std::vector<Copy>& plan, std::uint64_t prefix, std::uint64_t depth) {
    std::vector<std::uint64_t> rows(std::max(prefix, depth));
    for (std::uint64_t r = 0; r < rows.size(); ++r) rows[r] = r < prefix ? r : ~0ull;
    for (const Copy& c : plan)
        for (std::uint64_t j = 0; j < c.rows; ++j) rows[c.dst_row + j] = rows[c.src_row + j];
    return rows;
}

std::vector<std::uint32_t> subsetLayers(const StageInfo& st, std::optional<std::pair<std::uint32_t, std::uint32_t>> range,
                                        std::uint32_t n_layer, std::string& err, std::uint32_t interval) {
    std::vector<std::uint32_t> v;
    if (!st.layer_subset) {
        for (std::uint32_t i = 0; i < n_layer; ++i) v.push_back(i);
        return v;
    }
    if (st.kind == Kind::head) return v;
    const std::string name = st.name;
    if (!range) {
        if (name == "layer.attn") range = std::make_pair(interval - 1, interval - 1);
        else if (name == "layer.gdn") range = std::make_pair(0u, 0u);
        else {
            err = "no --layers";
            return v;
        }
    }
    if (range->second >= n_layer) {
        err = "--layers " + std::to_string(range->first) + "-" + std::to_string(range->second) + " past the model's " + std::to_string(n_layer) + " layers";
        return v;
    }
    for (std::uint32_t i = range->first; i <= range->second; ++i) v.push_back(i);
    return v;
}

// ---------------------------------------------------------------- golden
std::string GoldenKey::str() const {
    return stage + "|" + model_sha + "|d" + std::to_string(depth) + "|n" + std::to_string(rows) + "|L" + (layers.empty() ? "-" : layers) + "|" + mode +
           "|" + kv + "|" + arch + "|" + depth_source + "|in" + (input.empty() ? "-" : input);
}

const char* goldenStatusName(GoldenStatus s) {
    switch (s) {
        case GoldenStatus::match: return "match";
        case GoldenStatus::mismatch: return "mismatch";
        case GoldenStatus::absent: return "absent";
        case GoldenStatus::recorded: return "recorded";
    }
    return "?";
}

GoldenFile goldenParse(const std::string& text) {
    GoldenFile g;
    std::istringstream in(text);
    std::string line;
    int ln = 0;
    while (std::getline(in, line)) {
        ++ln;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos || tab == 0 || tab + 1 >= line.size()) {
            g.error = "golden line " + std::to_string(ln) + ": expected key<TAB>sha256";
            return g;
        }
        const std::string k = line.substr(0, tab), h = line.substr(tab + 1);
        auto it = g.entries.find(k);
        if (it != g.entries.end() && it->second != h) {
            g.error = "golden key twice with different hashes: " + k;
            return g;
        }
        g.entries[k] = h;
    }
    return g;
}

GoldenFile goldenLoad(const std::string& path) {
    std::ifstream f(std::filesystem::path(reinterpret_cast<const char8_t*>(path.c_str())), std::ios::binary);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return goldenParse(ss.str());
}

GoldenStatus goldenCheck(const std::string& path, GoldenFile& g, const GoldenKey& k, const std::string& hash, bool record, std::string& err) {
    const std::string key = k.str();
    auto it = g.entries.find(key);
    if (it != g.entries.end()) return it->second == hash ? GoldenStatus::match : GoldenStatus::mismatch;
    if (!record) return GoldenStatus::absent;
    std::ofstream f(std::filesystem::path(reinterpret_cast<const char8_t*>(path.c_str())), std::ios::binary | std::ios::app);
    if (!f) {
        err = "cannot append to golden file " + path;
        return GoldenStatus::absent;
    }
    f << key << '\t' << hash << '\n';
    f.flush();
    if (!f) {
        err = "write error on golden file " + path;
        return GoldenStatus::absent;
    }
    g.entries[key] = hash;
    return GoldenStatus::recorded;
}

// ---------------------------------------------------------------- judge
Verdict judge(double s, double r, double tol) {
    Verdict v;
    if (!(r > 0) || !std::isfinite(r) || !std::isfinite(s) || !(s > 0)) {
        v.text = "no valid reading";
        return v;
    }
    v.ratio = s / r;
    v.pass = std::fabs(v.ratio - 1.0) <= tol + 1e-12;
    char b[128];
    std::snprintf(b, sizeof(b), "synthetic / real = %.4f (%+.2f %%), band +-%.1f %% -> %s", v.ratio, (v.ratio - 1) * 100, tol * 100,
                  v.pass ? "IN" : "OUT");
    v.text = b;
    return v;
}

// ---------------------------------------------------------------- rows
std::string cellKey(const RowInfo& ri) {
    std::string c = ri.model_tag + "|d" + std::to_string(ri.depth) + "|n" + std::to_string(ri.rows) + "|" + ri.mode;
    if (!ri.layers.empty()) c += "|L" + ri.layers;
    return c;
}

std::string settingsKey(const RowInfo& ri) {
    std::string s = "state=" + ri.state + ";depth_source=" + ri.depth_source;
    if (ri.depth_source == "synthetic") s += ";fill=tile;prefix=" + std::to_string(ri.prefix);
    else s += ";fill=prefill";
    s += ";kv=" + ri.kv + ";max_cell_s=" + fmtNum(ri.max_cell_s) + ";rel=" + fmtNum(ri.rel) + ";preload_s=" + fmtNum(ri.preload_s) + ";anchor=off";
    return s;
}

namespace {
wb::Row baseRow(const RowInfo& ri) {
    wb::Row r;
    r.tool = "whirl-stage-bench";
    r.tool_ver = ri.tool_ver;
    r.component = "stage";
    r.cell = cellKey(ri);
    r.metric = "stage." + ri.stage;
    r.unit = "us";
    r.code_hash = ri.code_hash;
    r.shape = "d" + std::to_string(ri.depth) + "xn" + std::to_string(ri.rows);
    r.settings = settingsKey(ri);
    r.mode = ri.mode;
    r.gpu_arch = ri.gpu_arch;
    r.gpu_pci = ri.gpu_pci;
    r.started = ri.started;
    r.ended = wb::iso_now();
    r.extra.push_back({"depth", wb::json_number(static_cast<std::int64_t>(ri.depth))});
    r.extra.push_back({"rows", wb::json_number(static_cast<std::int64_t>(ri.rows))});
    if (!ri.layers.empty()) r.extra.push_back({"layers", wb::json_string(ri.layers)});
    if (!ri.model_sha.empty()) r.extra.push_back({"model_sha256", wb::json_string(ri.model_sha)});
    return r;
}
}  // namespace

wb::Row stageRow(const RowInfo& ri, const StatsIn& s) {
    wb::Row r = baseRow(ri);
    const bool ok = s.stop_reason != "error" && s.st.n > 0;
    if (ok) {
        r.value = s.st.median;
        r.set_stats(s.st);
    }
    r.stop_reason = s.stop_reason;
    wb::Work w;
    w.launches = 1;  // stage calls per sample (the launch count per call comes with P5's launch log)
    r.work = w;
    if (!ri.out_hash.empty()) r.extra.push_back({"out_hash", wb::json_string(ri.out_hash)});
    if (!ri.golden_status.empty()) r.extra.push_back({"golden", wb::json_string(ri.golden_status)});
    r.extra.push_back({"block", wb::json_number(static_cast<std::int64_t>(s.block))});
    r.extra.push_back({"invalid", wb::json_number(static_cast<std::int64_t>(s.invalid))});
    r.extra.push_back({"rel_iqr", wb::json_number(s.rel_iqr)});
    if (ok && s.st.median > 0) r.extra.push_back({"tok_s", wb::json_number(ri.rows / (s.st.median * 1e-6))});
    if (std::isfinite(ri.net_us)) r.extra.push_back({"net_us", wb::json_number(ri.net_us)});
    if (!s.samples.empty()) {
        std::string arr = "[";
        for (std::size_t i = 0; i < s.samples.size(); ++i) arr += (i ? "," : "") + wb::json_number(s.samples[i]);
        r.extra.push_back({"samples", arr + "]"});
    }
    return r;
}

wb::Row inrunRow(const RowInfo& ri0, double us) {
    RowInfo ri = ri0;
    ri.depth_source = "real_inrun";
    wb::Row r = baseRow(ri);
    r.value = us;
    wb::Stats st;
    st.n = 1;
    st.median = us;
    r.set_stats(st);
    r.stop_reason = "n";
    if (us > 0) r.extra.push_back({"tok_s", wb::json_number(ri.rows / (us * 1e-6))});
    r.extra.push_back({"note", wb::json_string("one in-run sample of a real prefill (the forward that starts at depth)")});
    return r;
}

wb::Row verdictRow(const RowInfo& ri, const Verdict& v, const std::string& against) {
    wb::Row r = baseRow(ri);
    r.metric += ".synth_over_real";
    r.unit = "ratio";
    if (v.ratio > 0) r.value = v.ratio;
    r.extra.push_back({"pass", v.pass ? "true" : "false"});
    r.extra.push_back({"against", wb::json_string(against)});
    r.extra.push_back({"verdict", wb::json_string(v.text)});
    return r;
}

std::vector<std::uint32_t> repeatTo(const std::vector<std::uint32_t>& t, std::size_t n) {
    std::vector<std::uint32_t> out;
    if (t.empty()) return out;
    out.reserve(n);
    while (out.size() < n) out.push_back(t[out.size() % t.size()]);
    return out;
}

std::vector<std::uint32_t> chunkEnds(std::uint32_t n, std::uint32_t chunk) {
    std::vector<std::uint32_t> e;
    if (chunk == 0) return e;
    for (std::uint32_t p = 0; p < n;) {
        p = std::min(n, p + chunk);
        e.push_back(p);
    }
    return e;
}

double estimateSeconds(const EstIn& e) {
    const double tps = e.prefill_tok_s > 0 ? e.prefill_tok_s : 300;
    // one stage cell: warm-up + calibration + up to max_cell_s of samples + 2 hashed calls; a chunk at depth ~
    // rows / tps x (1 + depth / 64k) [est: prefill throughput roughly halves from 0 to 64k on these models]
    const double call_s = e.rows / tps * (1.0 + e.depth / 65536.0);
    const double cell = e.max_cell_s * 1.2 + 4 * call_s + 2;
    double s = e.load_s;
    if (e.synthetic) s += std::min(e.prefix, e.depth) / tps + 2 + e.preload_s + cell;
    if (e.real) s += (static_cast<double>(e.depth) + e.rows) / tps + e.preload_s + cell;
    return s;
}

// ---------------------------------------------------------------- driver
StageResult runStage(const Driver& d, kt::Bench& b, double preload_s, kt::Clock* clock) {
    kt::Clock* clk = clock ? clock : &kt::realClock();
    StageResult r;
    d.restore();
    d.call();
    d.sync();
    r.hash_first = d.out_hash();
    if (preload_s > 0) {
        const double t0 = clk->now();
        while (clk->now() - t0 < preload_s) {
            d.restore();
            d.call();
            d.sync();
            ++r.preload_calls;
        }
        r.preload_s = clk->now() - t0;
    }
    const kt::BlockFn f = [&d](int n, std::uint64_t rot) {
        d.restore();
        return d.timed(n, rot);
    };
    r.cs = b.run(f);
    d.restore();
    d.call();
    d.sync();
    r.hash_last = d.out_hash();
    r.consistent = !r.hash_first.empty() && r.hash_first == r.hash_last;
    return r;
}

}  // namespace stage
