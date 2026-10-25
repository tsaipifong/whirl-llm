// whirl-stage-bench: forward stages at a synthetic KV / recurrent-state depth (measure-tools P4;
// progress\measure-tools.DESIGN.md A2, (d), (e) row P4; prior art progress\mt-p4.PRIOR_ART.md).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. No HIP in this header or in stage_bench.cpp: the option parser, the stage table, the KV tile
// plan, the per-stage golden file, the validation judge and the JSONL rows are unit-tested on the CPU
// (tests\bench\unit\stage_bench_tests.cpp, target whirl-stage-bench-tests). The GPU side is stage_bench_main.cpp.
//
// Synthetic depth D ("tile a real prefix", PRIOR_ART §3):
//   1. a real prefill of P = min(prefix, D) text tokens on the server's chunk size (every KV format, real values);
//   2. the K / V / K-scale / V-scale rows [0, P) of every attention layer are copied d2d to fill [P, D) (row r gets
//      row r mod P; doubling copies, tilePlan); the KV rows are opaque bytes (whirl/kv_layout.h layerRows), the
//      single-sequence page table is the identity, so a row is at row * row_bytes;
//   3. the GDN conv / ssm state after P tokens is snapshotted once; before every sample (outside the timed event
//      pair) it is restored, and the stage's own KV writes land on rows >= D (llama-bench -d / sweep-bench rule:
//      every repetition starts from the same state).
// Real depth (--depth-source real, the validation of DESIGN A2): a real prefill of D (+ rows) tokens; the forward
// that starts at D is timed in-run (one sample, the DESIGN comparison), then the stage is re-timed like the
// synthetic one on that real state. Judge: |synthetic / real - 1| <= tol (2 %).
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "bench_out.h"
#include "../kernels/kt_bench.h"

namespace stage {

// ---------------------------------------------------------------- stages
enum class Kind {
    prefill_chunk,  // forward of `rows` rows at pos0 = D (rows: the server's chunk, prefillExecDefault)
    decode_step,    // one decode step at position D (graph replay, as the server's plain decode)
    verify,         // Model::verify of `rows` (2..16) rows at pos0 = D (all logits + argmax rows)
    head,           // forward with NO trunk layers: embedding + final norm + output head (rows rows)
    layers,         // forward with only the layers of --layers a-b (embedding + those layers + head)
};
struct StageInfo {
    const char* name;        // CLI / metric name (stage.<name>)
    Kind kind;
    bool layer_subset;       // runs a subset of the trunk layers (head / layers / layer.*)
    std::uint32_t def_rows;  // 0 = the server's prefill chunk for the model (filled in by the GPU side)
    const char* note;
};
// every stage the DESIGN lists; implemented == false -> exit 2 with the reason (StageInfo::note)
struct StageEntry {
    StageInfo info;
    bool implemented;
};
const std::vector<StageEntry>& stageTable();
// lookup by name; nullptr = unknown
const StageEntry* findStage(const std::string& name);

// ---------------------------------------------------------------- options
struct Opts {
    std::string model, stage = "prefill_chunk", jsonl, golden, prompt_file, device, mode = "balance";
    std::string model_sha;                   // --model-sha (runner: progress\tools\sha_cache.tsv); "" -> name:size
    std::string depth_source = "synthetic";  // synthetic | real | both (both = the validation run)
    std::uint32_t depth = 0;
    std::uint32_t prefix = 4096;             // real tokens tiled into the synthetic depth
    std::vector<std::string> stages;         // --stage a,b (prefill_chunk first); stage = stages.front()
    std::uint32_t rows = 0;                  // 0 = each stage's default (StageInfo::def_rows; 0 there = the server chunk)
    std::optional<std::pair<std::uint32_t, std::uint32_t>> layers;  // --layers a-b (inclusive)
    bool record_golden = false;              // --record-golden: append missing keys (never overwrites)
    double max_cell_s = 20, rel = 0.02, preload_s = 0, tol = 0.02;
    int min_samples = 5, max_samples = 200;
    std::string state = "burst";             // declared power state: burst | sustained (preload_s > 0)
    bool dry_run = false;                    // --dry-run: print the plan, no GPU
};
struct ParseResult {
    std::optional<Opts> opts;
    std::string error;  // non-empty = usage error (exit 2)
    bool help = false;
};
ParseResult parseArgs(const std::vector<std::string>& args);
std::string usage();

// "32768", "32k" (x1024), "128K" -> tokens; nullopt on anything else or 0 / > 1M
std::optional<std::uint32_t> parseDepth(const std::string& s);
// "a-b" or "a" -> inclusive range, a <= b
std::optional<std::pair<std::uint32_t, std::uint32_t>> parseLayers(const std::string& s);

// ---------------------------------------------------------------- the synthetic fill
struct Copy {
    std::uint64_t src_row, dst_row, rows;
};
// d2d copies (in order) that fill rows [prefix, depth) from rows [0, prefix): row r ends up holding row r % prefix.
// Each copy reads only rows already final and never overlaps its destination. Empty when depth <= prefix.
std::vector<Copy> tilePlan(std::uint64_t prefix, std::uint64_t depth);
// row r of the result of applying `plan` to rows 0..prefix-1 (for tests): the source row it ends up holding
std::vector<std::uint64_t> applyPlan(const std::vector<Copy>& plan, std::uint64_t prefix, std::uint64_t depth);

// rows of the trunk in the subset run, given the layer count (layers stage: the --layers range; head: none)
std::vector<std::uint32_t> subsetLayers(const StageInfo& st, std::optional<std::pair<std::uint32_t, std::uint32_t>> range,
                                        std::uint32_t n_layer, std::string& err, std::uint32_t interval = 4);

// ---------------------------------------------------------------- per-stage golden
// key = stage | model sha256 | depth | rows | layers | mode | kv | arch | depth source | input (sha256 prefix of the
// token ids the stage saw: prefix / prompt and the stage rows). A golden is the sha256 of the
// stage's output bytes (logits + final hidden rows) after one restored call; recorded once with the same exe.
struct GoldenKey {
    std::string stage, model_sha, layers, mode, kv, arch, depth_source, input;
    std::uint32_t depth = 0, rows = 0;
    std::string str() const;
};
enum class GoldenStatus { match, mismatch, absent, recorded };
const char* goldenStatusName(GoldenStatus s);
// file: one "key<TAB>sha256" per line; '#' lines and blank lines ignored; a key twice with different hashes = error
struct GoldenFile {
    std::map<std::string, std::string> entries;
    std::string error;  // parse error (duplicate key with another hash, malformed line)
};
GoldenFile goldenParse(const std::string& text);
GoldenFile goldenLoad(const std::string& path);  // a missing file = empty, no error
// compare (and with record = true append an absent key to the file); a mismatch is never overwritten
GoldenStatus goldenCheck(const std::string& path, GoldenFile& g, const GoldenKey& k, const std::string& hash, bool record,
                         std::string& err);

// ---------------------------------------------------------------- validation (DESIGN A2 acceptance test)
struct Verdict {
    double ratio = 0;   // synthetic / real
    bool pass = false;  // |ratio - 1| <= tol
    std::string text;
};
Verdict judge(double synth_us, double real_us, double tol);

// ---------------------------------------------------------------- JSONL rows
struct RowInfo {
    std::string stage, model_tag, mode, kv, depth_source, gpu_arch, gpu_pci, code_hash, model_sha, started, tool_ver, state;
    std::uint32_t depth = 0, rows = 0, prefix = 0;
    std::string layers;          // "a-b" or ""
    std::string out_hash, golden_status;
    double max_cell_s = 0, rel = 0, preload_s = 0;
    int samples_n = 0;
    double net_us = std::numeric_limits<double>::quiet_NaN();  // layers: median(subset) - median(head)
};
// stats from kt::Bench (us per call); stop_reason / n etc. as there
struct StatsIn {
    whirl::bench::Stats st;
    std::string stop_reason;
    int invalid = 0, block = 1;
    double rel_iqr = 0;
    std::vector<double> samples;
};
std::string cellKey(const RowInfo& ri);       // model|d<D>|n<rows>|<mode>[|L<a-b>]
std::string settingsKey(const RowInfo& ri);   // state=..;depth_source=..;fill=..;prefix=..;max_cell_s=..;rel=..;preload_s=..;kv=..
whirl::bench::Row stageRow(const RowInfo& ri, const StatsIn& s);
// one in-run sample of a real prefill (the DESIGN comparison): metric stage.<name>, settings depth_source=real_inrun
whirl::bench::Row inrunRow(const RowInfo& ri, double us);
// the validation row: metric stage.<name>.synth_over_real, unit ratio
whirl::bench::Row verdictRow(const RowInfo& ri, const Verdict& v, const std::string& against);

// tokens of the real text repeated until n (the prompt file may be shorter than D); empty input -> empty
std::vector<std::uint32_t> repeatTo(const std::vector<std::uint32_t>& toks, std::size_t n);

// the server's fixed chunk schedule over [0, n): ends of forwards of `chunk` rows (the last one shorter)
std::vector<std::uint32_t> chunkEnds(std::uint32_t n, std::uint32_t chunk);

// hard estimate of this process (seconds, [est]) from the plan; the runner's cap = max(5 min, 2 x est + 2 min)
struct EstIn {
    std::uint32_t depth = 0, prefix = 0, rows = 0;
    bool real = false, synthetic = true;
    double load_s = 60, prefill_tok_s = 300, preload_s = 0, max_cell_s = 20;
};
double estimateSeconds(const EstIn& e);

// ---------------------------------------------------------------- the measuring driver (HIP-free, unit-tested)
// restore: GDN state snapshot -> live state (+ the decode inputs), enqueued before the timed event pair;
// call: the stage (async); sync: wait for the stream; out_hash: sha256 of the stage output (after sync);
// timed: kt::eventBlock around `call` (the GPU side builds it; tests pass a fake).
struct Driver {
    std::function<void()> restore, call, sync;
    std::function<std::string()> out_hash;
    kt::BlockFn timed;
};
struct StageResult {
    kt::CellStats cs;
    std::string hash_first, hash_last;  // output after one restored call before / after the measurement
    bool consistent = false;            // hash_first == hash_last (restore works, the stage is deterministic)
    int preload_calls = 0;
    double preload_s = 0;
};
// restore+call+hash, preload (restore+call+sync until preload_s of wall time), kt::Bench::run on
// [restore, timed] (one restore per block; stage calls are >= 1 ms so the block is 1), restore+call+hash again
StageResult runStage(const Driver& d, kt::Bench& b, double preload_s, kt::Clock* clock = nullptr);

}  // namespace stage
