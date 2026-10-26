// whirl-stage-bench: forward stages at a synthetic (or real) KV / recurrent-state depth (measure-tools P4).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Test executable: no product code is changed. It drives the public Model API (load, forward,
// verify, decodeStep) and the public members the server already uses (KV pool arrays, GDN state views, layers):
//   - synthetic depth: real prefill of the prefix, then d2d tiling of the KV rows (stage_bench.h);
//   - the layers / layer.* / head stages run Model::run with a SUBSET of the layer list (Model::LayerSubset): the
//     per-layer arrays (kcache / vcache / kscale / vscale / conv_state / ssm_state) are permuted to match and put
//     back afterwards (RAII). Everything else of the forward is unchanged.
// Usage: see stage::usage() (whirl-stage-bench --help).

#include "stage_bench.h"

#include "whirl/gguf.h"
#include "whirl/gpu_select.h"
#include "whirl/hip.h"
#include "whirl/kv_layout.h"
#include "whirl/model.h"
#include "whirl/numerics.h"
#include "whirl/prefill_schedule.h"
#include "whirl/tokenizer.h"
#include "whirl/device_profile.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace hip = whirl::hip;
namespace q = whirl::qwen35;
namespace wb = whirl::bench;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

namespace {

kt::EventApi hipEvents() {
    kt::EventApi a;
    a.create = [] { return static_cast<void*>(hip::eventCreate(true)); };
    a.destroy = [](void* e) { hip::eventDestroy(static_cast<hip::Event>(e)); };
    a.record = [](void* e, void* s) { hip::eventRecord(static_cast<hip::Event>(e), static_cast<hip::Stream>(s)); };
    a.sync = [](void* e) { hip::eventSync(static_cast<hip::Event>(e)); };
    a.stream_sync = [](void* s) { hip::streamSync(static_cast<hip::Stream>(s)); };
    a.elapsed_ms = [](void* e0, void* e1) { return static_cast<double>(hip::eventElapsedMs(static_cast<hip::Event>(e0), static_cast<hip::Event>(e1))); };
    return a;
}

// one timed call (fresh event pair, P2 fix 1): GPU ms, <= 0 invalid
double timeOnce(q::Model& m, const std::function<void()>& call) {
    hip::Event e0 = hip::eventCreate(true), e1 = hip::eventCreate(true);
    const auto h0 = std::chrono::steady_clock::now();
    hip::eventRecord(e0, m.stream);
    call();
    hip::eventRecord(e1, m.stream);
    hip::eventSync(e1);
    hip::streamSync(m.stream);
    const double ms = hip::eventElapsedMs(e0, e1);
    const double host = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
    hip::eventDestroy(e0);
    hip::eventDestroy(e1);
    return (ms > 0 && ms <= host + 0.5) ? ms : -1.0;
}

const char* kvShort(const q::Model& m) { return whirl::kvFormatName(m.kv_fmt); }

std::string readFile(const std::string& p) {
    std::ifstream f(std::filesystem::path(reinterpret_cast<const char8_t*>(p.c_str())), std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// the per-layer pool arrays of attention layer i, with their bytes per token row (kv_layout.h page order)
struct KvArr {
    hip::DevPtr base;
    u64 row_bytes;
};
std::vector<KvArr> kvArrays(const q::Model& m, std::size_t i) {
    const u64 elems = static_cast<u64>(m.cfg.n_head_kv) * m.cfg.head_dim;
    const whirl::kv::LayerRows lr = whirl::kv::layerRows(elems, m.kv_fmt);
    std::vector<KvArr> v;
    for (std::size_t j = 0; j < lr.n; ++j) {
        hip::DevPtr p = 0;
        switch (lr.a[j].part) {
            case whirl::kv::Part::k: p = m.kcache[i]; break;
            case whirl::kv::Part::v: p = m.vcache[i]; break;
            case whirl::kv::Part::k_scale: p = m.kscale[i]; break;
            case whirl::kv::Part::v_scale: p = m.vscale[i]; break;
        }
        if (p == 0) throw std::runtime_error("KV array missing for layer " + std::to_string(i));
        v.push_back({p, lr.a[j].bytes});
    }
    return v;
}

// tile rows [0, prefix) of every attention layer's KV arrays into [prefix, depth); returns bytes copied
u64 tileKv(q::Model& m, u32 prefix, u32 depth) {
    const std::vector<stage::Copy> plan = stage::tilePlan(prefix, depth);
    u64 bytes = 0;
    for (std::size_t i = 0; i < m.cfg.n_layer; ++i) {
        if (!m.cfg.isAttn(static_cast<u32>(i))) continue;
        for (const KvArr& a : kvArrays(m, i))
            for (const stage::Copy& c : plan) {
                hip::copyAsync(a.base + c.dst_row * a.row_bytes, a.base + c.src_row * a.row_bytes, c.rows * a.row_bytes, m.stream);
                bytes += c.rows * a.row_bytes;
            }
    }
    hip::streamSync(m.stream);
    return bytes;
}

// GDN conv / ssm state snapshot (taken once at depth) and its restore (enqueued before each timed sample)
struct StateSnap {
    // live = the state buffers at take() time (by address, not by layer index: LayerSubset permutes the
    // conv_state / ssm_state views while a subset stage runs, the buffers themselves stay)
    struct One {
        hip::DevPtr live_conv, live_ssm, conv, ssm;
    };
    std::vector<One> v;
    void take(q::Model& m) {
        if (v.empty())
            for (std::size_t i = 0; i < m.cfg.n_layer; ++i)
                if (m.ssm_state[i] != 0) v.push_back({0, 0, m.alloc(m.convBytes()), m.alloc(m.ssmBytes())});
        std::size_t j = 0;
        for (std::size_t i = 0; i < m.cfg.n_layer; ++i) {
            if (m.ssm_state[i] == 0) continue;
            One& o = v.at(j++);
            o.live_conv = m.conv_state[i];
            o.live_ssm = m.ssm_state[i];
            hip::copyAsync(o.conv, o.live_conv, m.convBytes(), m.stream);
            hip::copyAsync(o.ssm, o.live_ssm, m.ssmBytes(), m.stream);
        }
        hip::streamSync(m.stream);
    }
    void restore(q::Model& m) const {
        for (const One& o : v) {
            hip::copyAsync(o.live_conv, o.conv, m.convBytes(), m.stream);
            hip::copyAsync(o.live_ssm, o.ssm, m.ssmBytes(), m.stream);
        }
    }
};

// Model::run over a subset of the trunk layers: the model's own seam (Model::LayerSubset, src/model/qwen35/
// layer_subset.cpp; arch025-M3 C7: stage-bench no longer writes Model's members)
using LayerSubset = q::Model::LayerSubset;

struct Env {
    stage::Opts o;
    const stage::StageEntry* st = nullptr;
    q::Model* m = nullptr;
    std::vector<u32> toks;  // real text tokens, >= depth + rows
    u32 exec = 2048, rows = 1;
    std::string arch, pci, code_hash, model_sha, model_tag, tool_ver, started;
    wb::JsonlWriter* out = nullptr;
    stage::GoldenFile golden;
    int fails = 0;
    double preload_now = 0;  // preload of the next measured cell (consumed: only the first cell of a phase preloads)
    std::string input;       // golden key part: sha256 prefix of the token ids of this phase + stage rows
};

std::string outHash(q::Model& m, u32 rows, bool all_logits) {
    const u64 nl = static_cast<u64>(m.cfg.n_vocab) * (all_logits ? rows : 1);
    std::vector<float> buf(nl + static_cast<u64>(rows) * m.cfg.n_embd);
    hip::download(buf.data(), m.scratch().logits, nl * 4);
    hip::download(buf.data() + nl, m.scratch().x, static_cast<u64>(rows) * m.cfg.n_embd * 4);
    return kt::sha256Hex(buf.data(), buf.size() * 4);
}

// the stage call at depth D (async where the API allows)
std::function<void()> stageCall(Env& E, u32 D) {
    q::Model& m = *E.m;
    const stage::Kind k = E.st->info.kind;
    if (k == stage::Kind::decode_step) return [&m] { m.decodeStep(); };
    if (k == stage::Kind::verify) {
        auto out = std::make_shared<std::vector<u32>>(E.rows);
        std::vector<u32> t(E.toks.begin() + D, E.toks.begin() + D + E.rows);
        return [&m, t, out, D] { m.verify(t, D, *out); };
    }
    std::vector<u32> t(E.toks.begin() + D, E.toks.begin() + D + E.rows);
    return [&m, t, D] { m.forward(t, D); };
}

// decode inputs: token ids[0] and position pos_buf[0] = D (what beginDecode / the previous argmax leave)
std::function<void()> stageRestore(Env& E, const StateSnap& snap, u32 D) {
    q::Model& m = *E.m;
    const bool dec = E.st->info.kind == stage::Kind::decode_step;
    const std::int32_t tok = static_cast<std::int32_t>(E.toks[D]), pos = static_cast<std::int32_t>(D);
    return [&m, &snap, dec, tok, pos] {
        snap.restore(m);
        if (dec) {
            hip::streamSync(m.stream);
            hip::upload(m.scratch().ids, &tok, 4);
            hip::upload(m.scratch().pos_buf, &pos, 4);
        }
    };
}

kt::BenchCfg benchCfg(const stage::Opts& o) {
    kt::BenchCfg c;
    c.warmup_ms = 100;
    c.min_block_us = 1000;
    c.max_block = 64;
    c.min_samples = o.min_samples;
    c.max_samples = o.max_samples;
    c.rel_target = o.rel;
    c.max_cell_s = o.max_cell_s;
    return c;  // no anchor (PRIOR_ART §3.4): stage samples are long; sustained = --preload-s
}

stage::RowInfo rowInfo(const Env& E, const std::string& name, const std::string& src, u32 D, const std::string& layers) {
    stage::RowInfo ri;
    ri.stage = name;
    ri.model_tag = E.model_tag;
    ri.mode = E.o.mode;
    ri.kv = kvShort(*E.m);
    ri.depth_source = src;
    ri.gpu_arch = E.arch;
    ri.gpu_pci = E.pci;
    ri.code_hash = E.code_hash;
    ri.model_sha = E.model_sha;
    ri.started = wb::iso_now();
    ri.tool_ver = E.tool_ver;
    ri.state = E.o.state;
    ri.depth = D;
    ri.rows = E.rows;
    ri.prefix = std::min(E.o.prefix, D);
    ri.layers = layers;
    ri.max_cell_s = E.o.max_cell_s;
    ri.rel = E.o.rel;
    ri.preload_s = E.preload_now;
    return ri;
}

// measure one stage (name, layer subset) on the current state at depth D; writes the row, returns the median us
double measure(Env& E, const StateSnap& snap, u32 D, const std::string& src, const std::string& name, const std::vector<u32>* subset,
               const std::string& layers, double head_us = std::numeric_limits<double>::quiet_NaN()) {
    q::Model& m = *E.m;
    std::unique_ptr<LayerSubset> ls;
    if (subset) ls = std::make_unique<LayerSubset>(m, *subset);
    const bool all_logits = E.st->info.kind == stage::Kind::verify;
    stage::Driver d;
    d.restore = stageRestore(E, snap, D);
    d.call = stageCall(E, D);
    d.sync = [&m] { hip::streamSync(m.stream); };
    d.out_hash = [&m, &E, all_logits] { return outHash(m, E.rows, all_logits); };
    auto tstats = std::make_shared<kt::TimerStats>();
    const std::function<void()> call = d.call;
    d.timed = kt::eventBlock(hipEvents(), static_cast<void*>(m.stream),
                             [call, &m](std::uint64_t, void* s) {
                                 if (s != static_cast<void*>(m.stream)) throw std::runtime_error("timer stream != model stream");
                                 call();
                             },
                             nullptr, tstats);
    kt::Bench b(benchCfg(E.o));
    stage::RowInfo ri = rowInfo(E, name, src, D, layers);
    const double pl = E.preload_now;
    E.preload_now = 0;
    const stage::StageResult r = stage::runStage(d, b, pl);
    ri.out_hash = r.hash_last;
    ri.samples_n = static_cast<int>(r.cs.st.n);
    if (std::isfinite(head_us) && r.cs.ok()) ri.net_us = r.cs.st.median - head_us;
    // per-stage golden (lossless: bitwise)
    if (!E.o.golden.empty() && r.consistent) {
        stage::GoldenKey gk;
        gk.stage = name, gk.model_sha = E.model_sha, gk.layers = layers, gk.mode = E.o.mode, gk.kv = ri.kv, gk.arch = E.arch;
        gk.depth_source = src, gk.depth = D, gk.rows = E.rows, gk.input = E.input;
        std::string err;
        const stage::GoldenStatus gs = stage::goldenCheck(E.o.golden, E.golden, gk, r.hash_last, E.o.record_golden, err);
        ri.golden_status = stage::goldenStatusName(gs);
        if (!err.empty()) std::printf("  golden: %s\n", err.c_str());
        if (gs == stage::GoldenStatus::mismatch) {
            std::printf("  FAIL golden mismatch %s (stored %s, now %s)\n", gk.str().c_str(), E.golden.entries[gk.str()].c_str(), r.hash_last.c_str());
            ++E.fails;
        }
    }
    if (!r.consistent) {
        std::printf("  FAIL output not reproducible after restore: %s vs %s\n", r.hash_first.c_str(), r.hash_last.c_str());
        ++E.fails;
        ri.golden_status = "inconsistent";
    }
    stage::StatsIn si;
    si.st = r.cs.st;
    si.stop_reason = r.cs.stop_reason;
    si.invalid = r.cs.invalid;
    si.block = r.cs.block;
    si.rel_iqr = r.cs.rel();
    si.samples = r.cs.samples;
    if (!r.cs.ok()) {
        std::printf("  FAIL timing error: %s (timer blocks %lld, retries %lld, invalid %lld, last %s)\n", r.cs.error.c_str(),
                    static_cast<long long>(tstats->blocks), static_cast<long long>(tstats->retries), static_cast<long long>(tstats->invalid),
                    tstats->last_invalid.c_str());
        ++E.fails;
    }
    // a failed output check writes no timing row (DESIGN A1 / A2 rule); errors are rows with value null
    if (r.consistent && E.out && !E.out->write(stage::stageRow(ri, si))) {
        std::printf("  FAIL jsonl write\n");
        ++E.fails;
    }
    std::printf("  stage.%-14s %-9s d%-7u n%-5u %s%s median %.1f us (p10 %.1f p90 %.1f, n %lld, %s, block %d, preload %.0f s / %d calls) out %s%s\n",
                name.c_str(), src.c_str(), D, E.rows, layers.empty() ? "" : ("L" + layers + " ").c_str(), ri.kv.c_str(), r.cs.st.median,
                r.cs.st.p10, r.cs.st.p90, static_cast<long long>(r.cs.st.n), r.cs.stop_reason.c_str(), r.cs.block, r.preload_s, r.preload_calls,
                r.hash_last.substr(0, 16).c_str(), ri.golden_status.empty() ? "" : (" golden " + ri.golden_status).c_str());
    std::fflush(stdout);
    return r.cs.ok() && r.consistent ? r.cs.st.median : -1.0;
}

// all stage cells of the run on the current depth state
double measureAll(Env& E, const StateSnap& snap, u32 D, const std::string& src) {
    const stage::StageInfo& si = E.st->info;
    if (!si.layer_subset) return measure(E, snap, D, src, si.name, nullptr, "");
    const std::vector<u32> none;
    const double head = measure(E, snap, D, src, "head", &none, "");
    if (si.kind == stage::Kind::head) return head;
    std::string err;
    const std::vector<u32> sel = stage::subsetLayers(si, E.o.layers, E.m->cfg.n_layer, err, E.m->cfg.interval);
    if (!err.empty()) throw std::runtime_error(err);
    if (std::string(si.name) == "layer.attn" && !E.m->cfg.isAttn(sel[0])) throw std::runtime_error("layer " + std::to_string(sel[0]) + " is not full attention");
    if (std::string(si.name) == "layer.gdn" && E.m->cfg.isAttn(sel[0])) throw std::runtime_error("layer " + std::to_string(sel[0]) + " is not DeltaNet");
    const std::string lr = std::to_string(sel.front()) + "-" + std::to_string(sel.back());
    return measure(E, snap, D, src, si.name, &sel, lr, head);
}

// real prefill of [0, n) on the server's chunk size (fresh event pair per forward); returns per-forward ms
std::vector<double> prefillTimed(Env& E, u32 n, u32 chunk, bool print) {
    q::Model& m = *E.m;
    std::vector<double> ms;
    u32 p0 = 0;
    for (u32 e : stage::chunkEnds(n, chunk)) {
        std::vector<u32> t(E.toks.begin() + p0, E.toks.begin() + e);
        const u32 pos = p0;
        ms.push_back(timeOnce(m, [&] { m.forward(t, pos); }));
        if (print && (e == n || ms.size() % 16 == 0)) std::printf("  prefill %u..%u: %.1f ms\n", p0, e, ms.back());
        p0 = e;
    }
    return ms;
}

int run(stage::Opts o) {
    Env E;
    E.o = o;
    E.st = stage::findStage(o.stage);
    E.tool_ver = std::string(WHIRL_SB_VERSION) + "+stagebench1";
    E.started = wb::iso_now();
    // device: the CLI's rule (s8-719 root fix: restrict the HIP runtime to the chosen GPU before the first HIP call)
    const whirl::gpusel::Restriction& rs = q::restrictDevice(o.device);
    if (!rs.note.empty()) std::fprintf(stderr, "whirl-stage-bench: %s\n", rs.note.c_str());
    const int dev = q::pickDevice(o.device);
    const hip::DeviceInfo info = hip::describeDevice(dev);
    hip::setDevice(dev);
    E.arch = info.gcn_arch;
    E.pci = "pci:" + std::to_string(info.pci_bus) + ":" + std::to_string(info.pci_device);
    std::printf("device %d: %s (%s) %s\n", dev, info.name.c_str(), info.gcn_arch.c_str(), E.pci.c_str());
    if (const hip::EmbeddedObject* eo = hip::embeddedObjectFor(hip::archFor(info.gcn_arch))) E.code_hash = kt::sha256Hex(eo->data, static_cast<std::size_t>(*eo->size));
    const std::filesystem::path mp(reinterpret_cast<const char8_t*>(o.model.c_str()));
    E.model_tag = mp.stem().string();
    E.model_sha = !o.model_sha.empty() ? o.model_sha : E.model_tag + ":" + std::to_string(std::filesystem::file_size(mp));

    const bool want_syn = o.depth_source != "real", want_real = o.depth_source != "synthetic";
    whirl::gguf::File f = whirl::gguf::File::open(o.model);
    const whirl::Tokenizer tok = whirl::Tokenizer::fromGguf(f);
    const whirl::device::Profile& prof = whirl::device::profile(hip::archFor(info.gcn_arch));
    q::LoadOptions lo;
    lo.numerics = whirl::numerics::resolve(o.mode, std::nullopt);
    lo.ctx_explicit = true;
    lo.kv_req = q::kvRequestFromEnv();
    // rows are known after load (the chunk depends on max_batch); reserve the largest a stage can use
    const u32 max_rows = std::max<u32>(o.rows, static_cast<u32>(q::max_batch_default));
    const u32 ctx = o.depth + max_rows + 16;
    q::LoadStats ls;
    const auto t_load = std::chrono::steady_clock::now();
    std::unique_ptr<q::Model> mp_ = q::Model::load(f, ctx, ls, lo);
    q::Model& m = *mp_;
    std::string log;
    q::loadOrTune(m, o.model, log);
    E.m = &m;
    E.exec = static_cast<u32>(whirl::sched::prefillExecFor(whirl::sched::prefillExecDefault(prof, m.cfg.moe), m.max_batch, prof.prefill_fill_batch));
    // rows per stage: --rows (one stage), else the stage default (0 = the server's chunk)
    auto rowsFor = [&](const std::string& sn) -> u32 {
        if (o.rows != 0) return o.rows;
        const u32 d = stage::findStage(sn)->info.def_rows;
        return d != 0 ? d : E.exec;
    };
    u32 rows_max = 1;
    for (const std::string& sn : o.stages) rows_max = std::max(rows_max, rowsFor(sn));
    std::printf("model %s: %u layers, %s, KV %s, max_ctx %u, chunk %u, max rows %u, loaded in %.1f s\n", E.model_tag.c_str(), m.cfg.n_layer,
                m.cfg.archName(), kvShort(m), m.max_ctx, E.exec, rows_max,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count());
    if (o.depth + rows_max > m.max_ctx) throw std::runtime_error("depth + rows > max_ctx " + std::to_string(m.max_ctx));
    if (rows_max > m.max_batch) throw std::runtime_error("rows > max_batch " + std::to_string(m.max_batch));

    // real tokens: the prompt file (repeated to length), else a fixed id sequence over the vocabulary (synthetic only)
    const u32 need = o.depth + rows_max + 1;
    std::vector<u32> base;
    if (!o.prompt_file.empty()) {
        const std::string text = readFile(o.prompt_file);
        for (auto id : tok.encode(text, false)) base.push_back(static_cast<u32>(id));
    } else {
        // 104729 (prime) stride over the vocabulary: deterministic, no special-token runs
        for (u32 i = 0; i < 8192; ++i) base.push_back(static_cast<u32>((1000ull + 104729ull * i) % (m.cfg.n_vocab - 1024)));
    }
    if (base.empty()) throw std::runtime_error("no tokens in " + o.prompt_file);
    E.toks = stage::repeatTo(base, need);
    std::printf("tokens: %zu real (%s), repeated to %u\n", base.size(), o.prompt_file.empty() ? "fixed id sequence" : o.prompt_file.c_str(), need);

    wb::JsonlWriter w(o.jsonl);
    if (!w.ok()) throw std::runtime_error("cannot open --jsonl " + o.jsonl);
    E.out = &w;
    if (!o.golden.empty()) {
        E.golden = stage::goldenLoad(o.golden);
        if (!E.golden.error.empty()) throw std::runtime_error(E.golden.error);
    }
    // golden input part: token ids [0, n) of the phase + the stage rows at D
    auto inputHash = [&](u32 n_prefix, u32 rows) {
        std::vector<u32> v(E.toks.begin(), E.toks.begin() + n_prefix);
        v.insert(v.end(), E.toks.begin() + o.depth, E.toks.begin() + o.depth + rows);
        return kt::sha256Hex(v.data(), v.size() * 4).substr(0, 16);
    };

    StateSnap snap;
    std::map<std::string, double> syn_us, real_us, inrun_us;
    const u32 D = o.depth;
    if (want_syn) {
        const u32 P = std::min(o.prefix, D);
        m.reset();
        const auto t0 = std::chrono::steady_clock::now();
        prefillTimed(E, P, E.exec, false);
        const u64 bytes = tileKv(m, P, D);
        std::printf("synthetic depth %u: real prefix %u tokens + %.2f GB of KV tiled in %.1f s\n", D, P, bytes / 1e9,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        snap.take(m);
        E.preload_now = o.preload_s;  // DESIGN (d) sustained: once, before the first cell of the phase
        for (const std::string& sn : o.stages) {
            E.st = stage::findStage(sn);
            E.rows = rowsFor(sn);
            E.input = inputHash(P, E.rows);
            syn_us[sn] = measureAll(E, snap, D, "synthetic");
        }
    }
    if (want_real) {
        m.reset();
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<double> ms = prefillTimed(E, D, E.exec, true);
        double tot = 0;
        for (double v : ms) tot += v;
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("real depth %u: %zu forwards, GPU %.1f s, wall %.1f s\n", D, ms.size(), tot / 1000, wall);
        snap.take(m);
        // the real prefill is load already: only the rest of the sustained pre-load is added
        E.preload_now = std::max(0.0, o.preload_s - wall);
        for (const std::string& sn : o.stages) {
            E.st = stage::findStage(sn);
            E.rows = rowsFor(sn);
            E.input = inputHash(D, E.rows);
            if (E.st->info.kind == stage::Kind::prefill_chunk) {
                // the DESIGN A2 comparison: the forward that starts at D inside the real prompt, first call right after
                // the prefill (prefill_chunk is always the first stage), no warm-up
                std::vector<u32> t(E.toks.begin() + D, E.toks.begin() + D + E.rows);
                const double us = timeOnce(m, [&] { m.forward(t, D); }) * 1000.0;
                inrun_us[sn] = us;
                stage::RowInfo ri = rowInfo(E, sn, "real", D, "");
                std::printf("  stage.%s in-run (real prompt, chunk at %u): %.1f us\n", sn.c_str(), D, us);
                if (us > 0) w.write(stage::inrunRow(ri, us));
                snap.restore(m);
                hip::streamSync(m.stream);
            }
            real_us[sn] = measureAll(E, snap, D, "real");
        }
    }
    int rc = E.fails > 0 ? 1 : 0;
    if (want_syn && want_real) {
        bool pass = true;
        for (const std::string& sn : o.stages) {
            E.st = stage::findStage(sn);
            E.rows = rowsFor(sn);
            stage::RowInfo ri = rowInfo(E, sn, "synthetic", D, "");
            const stage::Verdict v = stage::judge(syn_us[sn], real_us[sn], o.tol);
            std::printf("VALIDATION stage.%s d%u vs real (same protocol): %s\n", sn.c_str(), D, v.text.c_str());
            w.write(stage::verdictRow(ri, v, "real"));
            pass = pass && v.pass;
            if (inrun_us.count(sn) && inrun_us[sn] > 0) {
                const stage::Verdict v2 = stage::judge(syn_us[sn], inrun_us[sn], o.tol);
                std::printf("VALIDATION stage.%s d%u vs real in-run chunk: %s\n", sn.c_str(), D, v2.text.c_str());
                w.write(stage::verdictRow(ri, v2, "real_inrun"));
                pass = pass && v2.pass;
            }
        }
        if (rc == 0 && !pass) rc = 3;
    }
    std::printf("whirl-stage-bench: %s (%d failures), %lld rows%s\n", rc == 0 ? "OK" : rc == 3 ? "VALIDATION OUT" : "FAIL", E.fails,
                static_cast<long long>(w.lines()), o.jsonl.empty() ? "" : (" -> " + o.jsonl).c_str());
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    // s8-719 root fix: the GPU list helper runs as a child of this program (gpu_select.h)
    if (argc == 2 && std::string(argv[1]) == whirl::gpusel::k_probe_arg) return whirl::gpusel::probeMain();
    std::vector<std::string> args(argv + 1, argv + argc);
    const stage::ParseResult pr = stage::parseArgs(args);
    if (pr.help) {
        std::fputs(stage::usage().c_str(), stdout);
        return 0;
    }
    if (!pr.error.empty()) {
        std::fprintf(stderr, "whirl-stage-bench: %s\n%s", pr.error.c_str(), stage::usage().c_str());
        return 2;
    }
    const stage::Opts& o = *pr.opts;
    if (o.dry_run) {
        stage::EstIn e;
        e.depth = o.depth, e.prefix = o.prefix, e.rows = o.rows ? o.rows : 2048;
        e.real = o.depth_source != "synthetic", e.synthetic = o.depth_source != "real";
        e.preload_s = o.preload_s, e.max_cell_s = o.max_cell_s;
        std::printf("plan: stage %s depth %u source %s prefix %u rows %u mode %s state %s; estimate %.0f s [est, 300 tok/s prefill]\n", o.stage.c_str(),
                    o.depth, o.depth_source.c_str(), o.prefix, o.rows, o.mode.c_str(), o.state.c_str(), stage::estimateSeconds(e));
        return 0;
    }
    try {
        return run(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "whirl-stage-bench: error: %s\n", e.what());
        return 1;
    }
}
