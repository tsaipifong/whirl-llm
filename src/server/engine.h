// Server engine: continuous batching over request slots, prefix cache with
// checkpoints, shared prefix checkpoints, host tiers (RAM / SSD), MTP and
// n-gram speculative decoding, sampling, and the streamed / one-shot
// OpenAI-style responses. The main thread owns every model / device call;
// connection threads only build jobs (parse, render, tokenize) and wait.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "backend.h"
#include "cache/ckpt_store.h"
#include "cache/page_pool.h"
#include "cache/prefix_cache.h"
#include "decode_floor.h"
#include "engine_api.h"
#include "engine_thread.h"
#include "prefix_store.h"
#include "tier_bridge.h"
#include "protocol.h"
#include "device/backend.h"
#include "device/handles.h"
#include "stream.h"
#include "tier/idle_flush.h"
#include "tier/kv_tier.h"
#include "tokens.h"
#include "vision_iface.h"
#include "whirl/chat.h"
#include "whirl/json.h"
#include "whirl/mem_governor.h"
#include "whirl/model.h"
#include "whirl/prefill_schedule.h"
#include "whirl/tokenizer.h"
#include "whirl/wddm_watch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace whirl::server {

inline double msSince(TimePoint t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

constexpr std::uint32_t kv_page = qwen35::kv_page;
constexpr std::uint32_t gdn_max_seg = qwen35::gdn_max_seg;
constexpr std::uint32_t max_small_batch = qwen35::max_small_batch;
constexpr std::uint32_t k_small = 256;
constexpr std::uint32_t k_big = 16384;
constexpr std::uint32_t max_rows = qwen35::max_verify_rows;
// the cold prefill schedule (shared with whirl seqtest's gate: include/whirl/prefill_schedule.h)
using sched::prefill_chunk;
using sched::prefill_merge;
// TAIL-CK: per-slot checkpoints needed for the tail checkpoint (tail + prompt-end + gen-end); with
// fewer, a tail would evict what a regenerate or the next chat turn resumes from, so none is taken
constexpr std::size_t tail_ck_slots = cache::ckpt_keep_prefix_min;
using sched::prefillChunkLen;
using sched::prefillExecFor;
using sched::prefill_exec_default;
using sched::prefillExecDefault;

struct Cand {
    std::uint32_t id;
    float logit;
    double p;
};

struct Sampler {
    double temp = 0;
    float inv_t = 1;
    std::uint32_t top_k = 0;
    double top_p = 1;
    double min_p = 0;
    std::uint64_t seed = 0;
    std::uint64_t pos_idx = 0;
    std::vector<Cand>* cands = nullptr;
    std::size_t n = 0;
    bool closed = false;
    float m = 0;
    float sum = 1;
    std::uint32_t row = 0;
    std::uint32_t row_base = 0;
    std::uint32_t full_used = 0;
    std::uint32_t big_used = 0;
    bool race = false;  // specsample: race draw coupled to the MTP drafts (whirl/spec_sample.h)

    // Uniform in [0, 1) keyed by (seed, output token index).
    double uniform() const {
        std::uint64_t z = seed + (pos_idx + 1) * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    }
};

// Candidates [0, count) -> sorted, probabilities, top-k / top-p / min-p.
// false when the kept set may extend past the candidates.
bool filterCands(Sampler& s, std::size_t count, bool complete);

enum class Phase { idle, prefill, decode, restore };

// cache_tokens, last_used: cache::SlotCache; pages, checkpoints and checkpoint points: SlotPrefix (arch024 M4);
// the tier entry and the spill fence: SlotTier (M5). The members; Slot adds the F1 wait (below).
struct SlotData : SlotPrefix, SlotTier {
    std::uint32_t id = 0;
    Phase phase = Phase::idle;
    Job* job = nullptr;
    Sampler sm;
    bool greedy = true;
    bool ignore_eos = false;
    std::uint32_t max_tokens = 0;
    std::uint32_t n_prompt = 0;
    std::size_t pf_off = 0;
    TimePoint tp0{};
    TimePoint td0{};
    double prompt_ms = 0;
    Out out;
    StreamState st;
    std::uint32_t n_gen = 0;
    std::string finish = "length";
    bool done = false;
    bool stopped_str = false;
    bool pool_cut = false;  // FR-SRV P3-8: ended because the KV pool had no page left (decode)
    std::uint32_t n_after_stop = 0;  // FR-SRV P3-7: generated tokens after the stop string (same cycle)
    std::uint32_t cycles = 0;
    std::uint32_t drafted = 0;
    std::uint32_t accepted = 0;
    std::uint32_t relaxed = 0;  // relaxacc: drafts kept by the relaxed rule (not the exact one)
    double relax_gap = 0;       // relaxacc greedy: sum of ln p(argmax) - ln p(kept draft)
    std::uint32_t pos = 0;
    std::uint32_t next = 0;
    // host tiers (tier_id, tier_pages, spill_ev, spill_busy, spill_upto, spill_wanted, restore, rs_map, rs_tail,
    // rs_shared: SlotTier)
    std::optional<TimePoint> tp0_pre;
    std::array<std::uint32_t, max_small_batch> pend{};
    std::size_t n_pend = 0;
    std::uint32_t pend_pos = 0;
    std::uint32_t last_acc = 0;
    std::uint32_t last_row = 0;
    bool logits_ok = false;
    float acc_ema = 0;
    qwen35::DraftAccept dacc;
    qwen35::SlotAccept sacc;      // WHIRL_SLOT_DRAFTS=2: per-slot acceptance with pool shrinkage
    std::uint32_t d_prev = 0;     // MTP drafts of this slot in its last MTP cycle
    qwen35::NgramPolicy ng;
    std::uint32_t ng_cycles = 0;
    // DFlash2 (--draft dflash): the drafter context of this slot holds positions dfl_lo .. dfl_hi - 1
    // as one unbroken run (rows below dfl_lo may exist but follow a gap); acceptance by draft
    // position and the mean draft count of its blocks
    std::uint32_t dfl_lo = 0, dfl_hi = 0;
    qwen35::DraftAccept dfl_acc;
    float dfl_nd = 0;
    std::uint32_t dfl_cycles = 0;
    std::array<std::uint32_t, max_small_batch + 1> batch{};
    std::size_t n_batch = 0;
    std::uint32_t last_log_n = 0;
    TimePoint last_log_t{};
    // WDDM-LOG: sharp decode slowdown within the request (checked at each "gen:" report)
    wddmwatch::DecodeWatch dec_watch;
    // v0.2.2.2 memory governor: the checkpoint buffers were freed (VRAM back to Windows); the slot takes no new
    // request until they are restored
    bool parked = false;

    std::uint64_t reqId() const { return job ? job->id : 0; }
};

// F1 (arch024 M1 / M2): a slot that goes away or is overwritten (vector clear / resize, exception unwinding,
// move-assign) first waits for its last spill, which may still read the checkpoint buffers; only then are the
// members (Ckpt's DevBuf / PinnedBuf) destroyed. All five special members are written out: a destructor alone would
// suppress the implicit moves, and slots_.resize needs the noexcept move constructor.
struct Slot : SlotData {
    Slot() = default;
    Slot(Slot&&) noexcept = default;  // resources change hands, nothing is freed: no wait
    Slot& operator=(Slot&& o) noexcept {
        if (this != &o) {
            waitSpill();
            SlotData::operator=(std::move(o));
        }
        return *this;
    }
    ~Slot() { waitSpill(); }
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;

private:
    // a destructor cannot throw: an event error (device lost) leaves nothing to wait for
    void waitSpill() noexcept {
        try {
            spill_ev.waitIfPending();
        } catch (...) {
        }
    }
};

// govFreeCache's checkpoint part for one idle slot: F1 passes the slot's spill fence to CkptStore::park, then the
// slot holds no checkpoint buffer (parked). Returns the VRAM freed. A free function so the F1 tests drive this call.
std::uint64_t parkSlotCkpts(CkptStore& cs, Slot& sl);

struct CycleProf;

// One decoding slot as the per-slot draft planner (WHIRL_SLOT_DRAFTS=2) sees it.
struct DraftPlanSlot {
    const qwen35::DraftAccept* acc = nullptr;  // MTP slot: effective acceptance; unused for n-gram slots
    std::uint32_t pos = 0;                     // context length
    std::uint32_t ng_rows = 0;                 // > 0: n-gram slot with this many verify rows
    float ng_e = 0;                            // n-gram slot: expected tokens
    std::uint32_t d_prev = 0;                  // MTP drafts in its last MTP cycle (0: none)
};

// MTP draft counts of the MTP slots (in order, n-gram slots skipped) from qwen35::allocDrafts:
// `uniform` drafts each unless the cost model finds a better split within `rows` verify rows and
// `snaps` snapshot sets; every MTP slot gets 1..cap. Pure (no engine state), for tests.
std::vector<std::uint32_t> planSlotDrafts(std::span<const DraftPlanSlot> slots, const qwen35::CycleCost& cost,
                                          std::uint32_t uniform, std::uint32_t cap, std::uint32_t rows, std::uint32_t snaps);

struct EngineOptions {
    std::string model_name;
    std::string model_file;  // GGUF file name without directory (GET /props model_path)
    std::uint32_t ctx = 131072;  // context per slot (cap)
    std::string numerics_json;   // GET /props "numerics" (numerics::propsJson; empty: not reported)
    std::string startup_json;    // GET /props and /health "startup": effective values and startup adjustments (v0.2.2.1)
    std::string ctx_note;        // why the context per request is what it is when startup lowered it (logged on a too-long prompt)
    std::string kv_ssd_mode = "off";  // --kv-ssd-mode as in effect (GET /props kv_ssd)
    bool kv_supersede = true;         // CACHE-1 superseded-session cut (WHIRL_KV_SUPERSEDE=0: off)
    // TIER-IDLE: --kv-ssd-mode shutdown writes reused entries to SSD after this long idle (tier/idle_flush.h);
    // no CLI flag, WHIRL_KV_IDLE_FLUSH_MS overrides it for tests (0: off)
    std::uint64_t kv_idle_flush_ms = tier::idle_flush_ms_default;
    std::uint32_t parallel = 4;
    chat::TemplateKind tmpl = chat::TemplateKind::a;
    bool use_mtp = false;
    // --draft dflash: DFlash2 drafts instead of MTP (the backend has the drafter; use_mtp is false)
    bool use_dflash = false;
    float dfl_p_min = 0;
    std::uint32_t dfl_n_min = 0;
    // a slot drafts with DFlash2 once its drafter context has an unbroken run of at least
    // min(pos, dfl_warm) rows ending at pos (after a restore whose features are gone)
    std::uint32_t dfl_warm = 64;
    std::uint32_t n_draft = 3;
    float p_min = 0;
    std::uint32_t n_min = 0;
    std::uint32_t mtp_adapt = 0;
    bool mtp_auto = false;
    std::array<std::uint32_t, gdn_max_seg + 1> batch_drafts{};
    // without DeltaNet replay: snapshot sets the n-gram / DFlash2 drafts may use (0 = all; CONC-8S
    // keeps them at the budget they had before the MTP caps under load were raised)
    std::uint32_t ng_snap_sets = 0;
    bool no_prefix_cache = false;
    bool verify_cache = false;
    // COLD-RESUME (--precise): resume only from checkpoints whose state is bitwise the cold prefill's: on the
    // new prompt's chunk schedule and produced by prefill (never 'gen-end', whose history came from the decode /
    // verify kernels). Off (balance / fast): the longest checkpoint, as before. WHIRL_COLD_RESUME=0/1 overrides.
    bool cold_resume = false;
    bool seg_prefill = true;
    bool timing_reset = true;
    bool ngram = true;
    std::uint32_t ngram_min = 3;
    float ngram_slope = 0.015f;
    std::uint32_t ngram_max = 0;
    bool ngram_force = false;
    // per-slot draft counts: 0 uniform, 1 marginal-gain swaps (splitDrafts), 2 cost model (allocDrafts)
    std::uint32_t slot_drafts = 0;
    bool trace_nd = false;
    bool loop_log = false;
    bool tier_verify = false;
    std::uint32_t tier_min_gain = 512;
    double gather_ms = 30;
    // decode floor (--decode-min-tps): while slots decode, each keeps >= this many tok/s
    // (prefill forwards limited, decode cycles interleaved); 0 = off
    double decode_min_tps = 0;
    int profile = 0;  // 0 off, 1 events, 2 cycle stats only
    std::uint32_t sys_min = 2048;
    bool lcp_on = true;
    std::size_t n_ck = 2;
    // TAIL-CK: a prompt of >= tail_ck_min tokens keeps a slot checkpoint (kind 'prefix') at the start
    // of its last schedule chunk, so the same long document with another question resumes there on
    // the second request (not only the third). Needs n_ck >= tail_ck_slots, so the tail never
    // pushes out the prompt-end / gen-end checkpoints a regenerate or the next chat turn resumes from.
    // WHIRL_TAIL_CK=0 turns it off (debug).
    bool tail_ck = true;
    std::uint32_t tail_ck_min = sched::tail_ck_min_default;
    std::size_t n_spe = 2;
    bool ckpt_host = false;
    std::size_t prefill_exec = prefill_exec_default;
    std::uint64_t seed = 0;
    // numerics item specsample (balance / fast): temperature > 0 requests sample by a race keyed by
    // (seed, output index, token) and draw MTP drafts from the draft head's filtered distribution q
    // by the same race (whirl/spec_sample.h); off (precise): inverse-CDF sampling, greedy drafts.
    // Both accept a draft iff it equals the sampled token: the reply is a function of the seed only
    // (MTP == plain bit for bit within the mode).
    bool spec_sample = false;
    // numerics item relaxacc (fast): relaxed draft acceptance (whirl/relax_accept.h); greedy
    // requests keep near-argmax drafts, sampling requests use typical acceptance (and greedy
    // drafts: specsample is not used with it)
    whirl::relax::Params relax;
};

class Engine final : public EngineApi {
public:
    Engine(ServerModel& model, device::Backend& ops, const Tokenizer& tok, const EngineOptions& opt);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Startup phases (the caller logs memory around them):
    // checkpoint / shared checkpoint / sampling buffers ...
    void allocState();
    // ... and, once the model's KV pool exists, the page bookkeeping.
    void initPool();
    // Host tiers (optional; the Tier was created / started by the caller).
    void attachTier(tier::Tier* t);
    // Vision encoder (optional; after the KV pool exists): image_url parts become image tokens.
    void attachVision(const VisionConfig& vc);
    const VisionConfig& visionConfig() const { return vis_cfg_; }

    // Main loop (returns after stop()). On stop: queued requests get 503, the
    // running ones finish, idle slots spill to the RAM tier and pending RAM -> SSD
    // tier writes are flushed (bounded, shutdown_flush_ms), then it returns.
    void runLoop();
    void stop();
    // Tests (REVIEW-ARCH3 P2-1): run fn on the engine thread at the start of the next loop iteration, i.e. after the
    // job whose response the caller already received has finished, including its spill when that spill ran at once
    // (a deferred spill, spill_wanted, runs later in tierTick: not covered). Blocks until fn ran. Engine state (tier
    // entries, statistics) read inside fn is synchronized with the engine. Throws when the loop does not claim fn
    // within timeout (not started / stopped); fn is then cancelled and never runs, so it may capture the caller's
    // locals by reference (REVIEW-ARCH4 P3-5). Not used by the product.
    void runOnLoop(std::function<void()> fn, std::chrono::milliseconds timeout = std::chrono::milliseconds(60000));
    // arch024 M8: the thread inside runLoop (none before / after it); the modules check their entry points against it
    const EngineThread& owner() const { return owner_; }
    // tests: every module checks against owner()
    bool modulesOwned() const {
        return ck_.ownerThread() == &owner_ && pool_.ownerThread() == &owner_ && ps_.ownerThread() == &owner_ &&
               tb_.ownerThread() == &owner_;
    }
    // flush bound: --kv-ssd-mode always (a few entries are pending); shutdown (everything in RAM is
    // written at stop, so it may be gigabytes)
    static constexpr std::uint64_t shutdown_flush_ms = 10000;
    static constexpr std::uint64_t shutdown_flush_long_ms = 60000;

    // Connection threads (EngineApi)
    std::uint64_t nextJobId() override { return next_id_.fetch_add(1); }
    void submitAndWait(Job& job) override;
    void beginBuilding() override { n_building_.fetch_add(1); }
    void endBuilding() override;
    Health health() override;

    // the HTTP layer's facts (EngineApi; taken once by HttpServer, after the setters below)
    EngineFacts facts() const override;
    const EngineOptions& options() const { return opt_; }
    // before initPool: precise f16 KV lowered the context per request; numerics mode for /props
    void setCtx(std::uint32_t c) {
        ctx_ = c;
        opt_.ctx = c;
        ctx_pub_.store(c, std::memory_order_relaxed);
    }
    void setNumerics(std::string j) { opt_.numerics_json = std::move(j); }
    void setStartup(std::string j, std::string ctx_note) {
        opt_.startup_json = std::move(j);
        opt_.ctx_note = std::move(ctx_note);
    }
    void setKvSsdMode(std::string m) { opt_.kv_ssd_mode = std::move(m); }
    // WDDM-LOG (issues #5 / #6): before runLoop, after everything sized at startup is allocated. The
    // probe (hip::wddmMemInfo on Windows) is sampled for each "batch |" status line, at each request
    // end and once a minute while idle, never per decode cycle. base: the baseline for the non-local
    // growth (the server passes the read the KV pool was sized from, so a spill already there at startup
    // is not hidden); nullopt = the first sample. warn = false (UMA iGPU, where "shared" is the same RAM)
    // logs the numbers without the warning. startup_warned: the startup line already printed the spill
    // warning (the watch counts it as its first). others: read only when a spill warning is printed.
    void setVramProbe(std::function<wddmwatch::Sample()> probe, bool warn, std::optional<wddmwatch::Sample> base = std::nullopt,
                      wddmwatch::Config cfg = {}, bool startup_warned = false,
                      std::function<std::optional<wddmwatch::Others>()> others = nullptr);
    // v0.2.2.2 memory governor (after setVramProbe, before runLoop): acts on the samples of the vram probe when the WDDM
    // budget drops while the server runs (whirl/mem_governor.h). margin: what must stay free under a given budget
    // (poolbudget::defaultMargin); ram_tier_mb: the RAM tier in use (0: none), for its log line.
    // vram_total: the GPU's dedicated VRAM (0: unknown); a grow is capped so WHIRL + what other programs use now (vram others
    // probe) + the margin fit it
    void setGovernor(const memgov::Config& c, std::function<std::uint64_t(std::uint64_t)> margin, std::uint64_t ram_tier_mb,
                     std::uint64_t vram_total = 0);
    std::string governorJson() override;
    std::uint32_t ctxPublished() const override { return ctx_pub_.load(std::memory_order_relaxed); }
    const Tokenizer& tokenizer() const { return tok_; }
    ServerModel& model() { return m_; }

    // statistics for tests (Stats: EngineApi)
    Stats stats() const;
    Stats statsShared() const override;

private:
    // checkpoints
    void saveCkpt(Slot& sl, std::uint32_t pos, const char* kind, std::optional<DevPtr> hid, std::optional<DevPtr> logits);
    void dropCache(Slot& sl);
    void freePages(Slot& sl);
    // pages (arch024 M3: the bookkeeping is pool_'s; these pass the slot's pages and the evictor)
    struct SlotEvictor;  // PagePool::Evictor over the idle slots and shared checkpoints (prefix cache's choice)
    bool evictFor(Slot* sl);
    std::optional<std::int32_t> takePage(Slot* sl);
    void cowPages(Slot& sl, std::uint32_t reuse, bool spill);
    bool ensurePages(Slot& sl, std::uint64_t n_tok);
    // shared prefix checkpoints (the table, the lookups and the schedule: ps_)
    void dropSpe(Spe& s);
    void createSpe(Slot& sl, std::span<const std::uint32_t> toks_all, std::uint32_t pos, const char* kind,
                   std::optional<DevPtr> hid_row, const Ckpt* src, std::uint64_t tier_id);
    void attachSpe(Slot& sl, Spe& s);
    bool waitsForSys(Job& job);
    bool waitsForTier(Job& job);
    void prefetchFor(Job& job);
    // sampling
    void launchTopk(std::uint32_t row0, std::uint32_t rows, std::uint32_t k, float inv_t, DevPtr dst,
                    std::uint32_t dst_rows, std::uint32_t out0);
    static std::size_t sampBytes();
    void fetchCands(Sampler& s, std::uint32_t rows);
    double probOf(Sampler& s, std::uint32_t t);  // relaxacc typical acceptance
    void prepareRow(Sampler& s, std::uint32_t r);
    void loadFullRow(std::uint32_t gr);
    std::uint32_t sampleRow(Sampler& s, std::optional<std::uint32_t> ex, double p_ex);
    std::uint32_t sampleRace(Sampler& s);
    DevPtr logitsRow(std::uint32_t r) const;
    // request lifecycle
    bool emit(Slot& sl, std::uint32_t t);
    void flushOut(Slot& sl);
    bool startJob(Slot& sl, Job& job);
    std::size_t chunkLen(const Slot& sl) const;
    std::vector<DecodeFloor::SlotTok> floorProtected() const;
    void maybeSplitCkpt(Slot& sl, DevPtr hid_row);
    void prefillStep(Slot& sl);
    std::size_t prefillGroup(Slot& first, std::size_t row_budget);
    void finishPrefill(Slot& sl);
    std::uint32_t pickDrafts(std::span<Slot* const> act);
    bool relaxRepeats(const Slot& sl, std::span<const std::uint32_t> dr, std::uint32_t acc) const;
    void decodeCycle(std::span<Slot* const> act_in);
    void splitDrafts(std::span<Slot* const> act, std::uint32_t nd, std::uint32_t rows, std::span<std::uint32_t> nd_m);
    std::uint32_t draftCap(std::uint32_t A);
    // DFlash2 drafter context of a slot
    std::uint32_t draftRoom() const;
    void dflReset(Slot& sl, std::uint32_t at);
    void dflIngest(Slot& sl, std::uint32_t run_rows, std::uint32_t r0, std::uint32_t n, std::uint32_t pos0);
    bool dflReady(const Slot& sl) const;
    void finishJob(Slot& sl);
    // host tiers (the spill half, the span lists and the idle / shutdown flush: tb_)
    // the scheduler's part of a restore (whether the request fits, the slot's pages and phase, startJob); the tier
    // part is TierBridge's (arch024-S7)
    bool tryRestore(Slot& sl, Job& job);
    void pollRestores();
    bool tierTick();
    TierBridge::RestoreSteps restoreSteps(Slot& sl);
    // shutdown: tb_.flushOnStop over tierTick, then the tier line
    void flushTierOnStop();
    void releaseSlot(Slot& sl);
    void failSlot(Slot& sl, const std::string& err);
    // FR-SRV P2-2: queued / running requests whose client disconnected end (slot freed)
    void reapDisconnected();
    // FR-SRV P2-3: pool pages a prompt of n_tok tokens needs; pages free or evictable now minus
    // what the running prefills still need
    std::size_t pagesFor(std::uint64_t n_tok) const;
    std::size_t poolAvailable() const;
    void writeError(Job& job, int status, const std::string& msg);
    void logBatchStats(bool force);
    // samples the WDDM probe (warns through vram_watch_); "" without a probe
    std::string vramNote();
    // idle: one sample a minute so /health and the spill-over line stay current; returns the next due time
    std::optional<TimePoint> vramIdleTick();
    void abortQueued(std::unique_lock<std::mutex>& lk);
    // v0.2.2.2 memory governor
    void govTick(const wddmwatch::Sample& s);   // after each sample: decide (and log the decisions)
    void govApply();                            // at a safe point of runLoop (no forward in flight): act
    bool govAllIdle() const;
    std::uint64_t govFreeable() const;
    void govFreeCache(bool park);
    void govRestoreCache();
    bool govResize(std::uint32_t tokens, bool shrink);
    void govDrain();
    void govPublish();
    std::uint32_t ctxForNew() const { return std::min(ctx_, ctx_new_); }
    std::uint64_t kvTokBytes() const {
        std::uint64_t b = 0;
        for (const KvArr& a : kv_arrays_) b += a.row;
        return b;
    }
    // vision: image embeddings (cache by content hash), per-slot image maps, idle release
    EmbEntry* embGet(const vision::Prepared& p, std::uint64_t req, std::uint32_t slot);
    void embUnpin(EmbEntry* ent);
    void embEvict();
    void visPrepare(Slot& sl, Job& job, std::uint32_t from);
    void visDone(Slot& sl, Job& job);
    void visIdle();
    bool visWakeNeeded() const;

    ServerModel& m_;
    device::Backend& ops_;
    const Tokenizer& tok_;
    ChatTokens ct_;
    EngineOptions opt_;
    std::uint32_t ctx_;
    EngineThread owner_;  // arch024 M8: set by runLoop; ck_, pool_, ps_, tb_ and the governor glue check against it
    CkptStore ck_;  // checkpoint buffers (arch024 M2)
    std::array<qwen35::DraftTiming, gdn_max_seg + 1> timing_{};
    // WHIRL_SLOT_DRAFTS=2: cycle time per number of decoding slots, global acceptance pool
    std::array<qwen35::CycleCost, gdn_max_seg + 1> ccost_{};
    std::array<bool, gdn_max_seg + 1> ccost_primed_{};
    qwen35::DraftAccept dpool_{};
    // DFlash2 cycle times by decoding slots (drafting included), keyed by the largest draft count
    std::array<qwen35::DraftTiming, gdn_max_seg + 1> dtiming_{};
    std::unique_ptr<CycleProf> prof_;
    std::uint32_t n_cycles_ = 0;
    // most rows of one batched verify (ServerModel::verifyRows: 16, or 32 with the wide GEMV path)
    std::uint32_t vrows_ = max_small_batch;
    std::uint32_t prev_nd_ = 0;
    std::uint32_t policy_cycles_ = 0;  // cycles since the draft-policy state was last reset
    // decode floor bookkeeping (main thread)
    DecodeFloor floor_;
    TimePoint floor_epoch_{};
    TimePoint floor_log_t_{};
    std::uint64_t stat_floor_waits_ = 0, stat_floor_periods_ = 0;
    // sampling buffers
    device::DevBuf samp_dev_, big_dev_;
    device::DevBuf relax_dev_;                // relaxacc: top-K of the greedy verify rows [ids | vals | stats]
    std::vector<std::uint8_t> relax_host_;
    std::vector<std::uint8_t> samp_host_, big_host_;
    std::vector<float> row_host_;
    std::vector<std::uint8_t> in_cands_;
    std::vector<Cand> cands_scratch_;
    std::vector<std::int32_t> ctl_host_;
    bool spec_on_ = false;
    std::uint64_t conv_bytes_ = 0, ssm_bytes_ = 0;
    std::vector<Slot> slots_;
    // queue
    std::mutex q_mutex_;
    std::condition_variable q_cond_;
    std::deque<Job*> queue_;
    std::uint32_t n_active_ = 0;
    std::atomic<std::uint32_t> n_building_{0};
    std::atomic<std::uint64_t> next_id_{1};
    IdGen ids_;         // response / tool-call ids and default sampling seeds (one stream)
    ResponseCtx rcx_;   // what each slot's StreamState reads (model id, tokenizer, ids_)
    bool stop_ = false;
    std::vector<std::function<void()>> loop_calls_;  // under q_mutex_: runOnLoop (tests)
    void runLoopCalls(std::unique_lock<std::mutex>& lk);
    // batch statistics
    TimePoint stat_t0_{};
    std::uint64_t stat_tokens_ = 0, stat_cycles_ = 0, stat_rows_ = 0, stat_slots_ = 0, stat_drafts_ = 0,
                  stat_lit_drafts_ = 0,  // FR-SRV P2-5: DFlash2 / n-gram drafts verified (all slots)
                  stat_prefill_tok_ = 0, stat_seg_batches_ = 0, stat_seg_chunks_ = 0;
    std::uint64_t stat_dfl_cycles_ = 0, stat_dfl_drafted_ = 0, stat_dfl_accepted_ = 0, stat_dfl_cold_ = 0, stat_dfl_gap_ = 0;
    // WDDM-LOG: runtime video memory watch (setVramProbe)
    std::function<wddmwatch::Sample()> vram_probe_;
    std::function<std::optional<wddmwatch::Others>()> vram_others_;
    wddmwatch::Watch vram_watch_;
    TimePoint vram_idle_next_{};    // next idle sample (once a minute while no request runs)
    double vram_idle_s_ = 60.0;
    std::uint64_t prefill_tok_cum_ = 0;  // all slots, since start (DecodeWatch: was another request prefilling)
    bool vram_warn_ = false;
    std::mutex vram_mu_;  // vram_last_ / vram_spill_ for health() (HTTP threads)
    wddmwatch::Sample vram_last_;
    bool vram_spill_ = false;
    std::vector<std::uint32_t> crlf_norm_;
    TimePoint vram_last_t_{};  // last probe sample
    // v0.2.2.2 memory governor (main thread; gov_json_ under vram_mu_)
    std::optional<memgov::Governor> gov_;
    std::function<std::uint64_t(std::uint64_t)> gov_margin_;
    memgov::Plan gov_plan_;
    bool gov_pending_ = false;
    bool gov_hold_ = false;       // new requests queue (a smaller pool waits for the running ones)
    bool spe_parked_ = false;     // shared prefix checkpoint buffers freed
    std::uint64_t gov_parked_ = 0;  // checkpoint VRAM freed (to restore)
    std::uint64_t gov_ram_mb_ = 0;
    std::uint64_t gov_vram_total_ = 0;
    bool gov_grow_denied_logged_ = false;
    std::uint64_t gov_restored_ = 0;  // checkpoint bytes the last govRestoreCache allocated
    std::uint64_t gov_resize_before_ = 0;  // local usage read right before the last re-allocation
    std::int64_t gov_resize_delta_ = 0;    // bytes it added (> 0) or freed (< 0)
    std::uint32_t ctx_new_ = 0xffffffffu;  // context per request for new requests (governor); running requests keep ctx_
    std::string gov_warn_;        // the spill warning, logged after the governor's actions
    std::string gov_last_;
    std::string gov_json_;
    std::atomic<std::uint32_t> ctx_pub_{0};
    // host tiers (the tier itself: tb_.tier())
    std::vector<KvArr> kv_arrays_;
    bool tier_wake_ = false;
    std::uint64_t wake_seq_ = 0;  // under q_mutex_: bumped by a queued request, a finished build, the tier IO thread
    PagePool pool_;  // KV pool pages (arch024 M3); here, where its quarantine was, so ~Engine destroys its events in
                     // the same order
    TimePoint last_tick_{};
    TimePoint last_reap_{};
    // shared prefix checkpoints, prefix cache decisions, chunk schedule (arch024 M4); here, where spes_ was, so
    // ~Engine destroys the shared checkpoints (and their events) at the same point
    PrefixStore ps_;
    // host tier bridge, spill half (arch024 M5): after ps_ and kv_arrays_ (it keeps references to both)
    TierBridge tb_;
    std::uint64_t stat_spe_new_ = 0, stat_spe_hits_ = 0, stat_spe_tok_ = 0, stat_spe_evict_ = 0, stat_unshare_ = 0;
    std::uint64_t stat_evictions_ = 0;
    // vision
    VisionConfig vis_cfg_;
    std::vector<std::unique_ptr<EmbEntry>> emb_cache_;
    std::uint64_t emb_bytes_ = 0;
    std::uint64_t emb_seq_ = 0;
    std::uint64_t stat_vis_enc_ = 0, stat_vis_hit_ = 0;
    std::array<std::array<std::uint64_t, 2>, 16> vis_lend_raw_{};
    std::vector<vision::Region> vis_lend_;
    hip::LocalMemSnap vis_snap0_;  // device memory counters right after the mmproj was loaded (no image yet)
    bool vis_rel_pending_ = false;
};

}  // namespace whirl::server
