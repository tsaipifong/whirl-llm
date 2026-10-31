// Server engine: continuous batching over request slots, prefix cache with
// checkpoints, shared prefix checkpoints, host tiers (RAM / SSD), MTP and
// n-gram speculative decoding, sampling, and the streamed / one-shot
// OpenAI-style responses. The main thread owns every model / device call;
// connection threads only build jobs (parse, render, tokenize) and wait.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "backend.h"
#include "whirl/small_helpers.h"  // satSub / alignUp / unixNow / msSince / nowMs (arch025-N6)
#include "cache/ckpt_store.h"
#include "cache/page_pool.h"
#include "cache/prefix_cache.h"
#include "decode_floor.h"
#include "engine_sampling.h"  // SampleBuffers, k_small / k_big / max_rows (arch025-N4a)
#include "engine_slot.h"  // Phase, SlotData, Slot, parkSlotCkpts, max_small_batch (arch025-engine S1)
#include "engine_vision.h"  // ImageEmbeddings (arch025-N4b)
#include "engine_api.h"
#include "engine_thread.h"
#include "prefix_store.h"
#include "tier_bridge.h"
#include "protocol.h"
#include "sampling/acceptor.h"
#include "sampling/sampler.h"
#include "device/backend.h"
#include "device/handles.h"
#include "stream.h"
#include "tier/idle_flush.h"
#include "tier/kv_tier.h"
#include "spec/policy.h"  // DraftAccept, SlotAccept, NgramPolicy, DraftTiming, CycleCost (arch025-M4)
#include "spec/slot_drafts.h"  // DraftPlanSlot, planSlotDrafts, splitDraftsByGain, slotDraftCap (arch025-E1b)
#include "whirl/chat_tokens.h"  // chat::ChatTokens (arch025-N15c, shared with the CLI)
#include "vision_iface.h"
#include "whirl/vision.h"  // vision::Vision / Prepared / Region (vision_iface.h only names them: arch025-lane2 leg 5 S6)
#include "whirl/chat.h"
#include "whirl/json.h"
#include "whirl/mem_governor.h"
#include "whirl/model_types.h"
#include "whirl/prefill_schedule.h"
#include "whirl/relax_accept.h"  // relax::Params (was reached through model.h)
#include "whirl/thread_annotations.h"
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

// arch025-engine S3 (module server.engine_types): last, where their text stood (after every other header)
#include "server/engine_common.h"   // aliases (kv_page, gdn_max_seg, prefill schedule, Cand, Sampler), KvPoolFull
#include "server/engine_options.h"  // EngineOptions
#include "server/engine_state.h"    // BatchStats, SpecState, FloorState
#include "server/engine_drafter.h"   // DrafterCtx (arch025-engine S4, module server.drafter_ctx)
#include "server/engine_output.h"    // emit, flushOut, writeError (arch025-engine S5, module server.slot_output)
#include "server/engine_pages.h"     // SlotPagesCounters, SlotPages (arch025-engine S6 / S7, module server.slot_cache)
#include "server/engine_gov.h"       // GovState, GovernorGlue (arch025-engine S8, module server.governor)

namespace whirl::server {

class Engine final : public EngineApi {
public:
    Engine(ServerModel& model, device::Backend& ops, const Tokenizer& tok, const EngineOptions& opt);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Startup phases (the caller logs memory around them):
    // checkpoint / shared checkpoint / sampling buffers ...
    void allocState();
    // ... the same between two memInfo reads of the engine's Backend (arch024-S11c, F-7): the startup log's
    // "checkpoints" VRAM is the difference, so the reads must sit right around this allocation (S11b mutant C1 moved
    // allocState before the first read; a CPU test checks the call order on NullBackend).
    struct StateMem {
        hip::MemInfo before, after;
    };
    StateMem allocStateMeasured();
    // ... and, once the model's KV pool exists, the page bookkeeping.
    void initPool();
    // Host tiers (optional; the Tier was created / started by the caller).
    void attachTier(tier::Tier* t);
    // Vision encoder (optional; after the KV pool exists): image_url parts become image tokens.
    void attachVision(const VisionConfig& vc);
    const VisionConfig& visionConfig() const { return vis_.config(); }

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
        gv_.govPublishCtx(c);
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
    std::uint32_t ctxPublished() const override { return gv_.govCtxPublished(); }
    const Tokenizer& tokenizer() const { return tok_; }
    ServerModel& model() { return m_; }

    // statistics for tests (Stats: EngineApi)
    Stats stats() const;
    Stats statsShared() const override;

private:
    // the KV pages and checkpoints of the slots: sc_ (arch025-engine S7, module server.slot_cache)
    bool waitsForTier(Job& job);
    void prefetchFor(Job& job);
    // request lifecycle
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
    // decodeCycle's phases in call order (arch025-E2: extracted verbatim). Cycle = the values one cycle passes from
    // phase to phase (engine.cpp). sampleRows holds the cycle's one host sync (readCtl).
    struct Cycle;
    bool relaxes(const Cycle& cy, std::size_t k) const;
    std::uint32_t verifyRoom(const Slot& sl, std::uint32_t lim) const;
    void dropOutOfRoom(Cycle& cy, std::span<Slot* const> act_in);
    void draftLimits(Cycle& cy);
    void ngramDrafts(Cycle& cy);
    void dflashDrafts(Cycle& cy);
    void mtpDrafts(Cycle& cy);
    void layoutBatch(Cycle& cy);
    void draftSampling(Cycle& cy);
    void enqueueVerify(Cycle& cy);
    void sampleRows(Cycle& cy);
    void trainCycle(Cycle& cy);
    void cycleStats(Cycle& cy);
    void acceptSlot(Cycle& cy, std::size_t k, const sampling::Acceptor& acceptor);
    void finishCycle(Cycle& cy);
    // runLoop's phases in call order (arch025-E2: extracted verbatim)
    bool idleWait(whirl::UniqueLock& lk, bool& was_busy) WH_REQUIRES(q_mutex_);
    std::vector<Job*> admitQueued(std::uint32_t n_busy) WH_REQUIRES(q_mutex_);  // q_mutex_ held by the caller
    void startAdmitted(const std::vector<Job*>& admit);
    void runPrefill(Slot* pf, bool& idle_wait, std::size_t& pf_rows, bool& floor_pf);
    void noteFloor(TimePoint tl2, TimePoint tl3, std::size_t pf_rows, bool floor_pf);
    void splitDrafts(std::span<Slot* const> act, std::uint32_t nd, std::uint32_t rows, std::span<std::uint32_t> nd_m);
    std::uint32_t draftCap(std::uint32_t A);
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
    void logBatchStats(bool force);
    // samples the WDDM probe (warns through vram_watch_); "" without a probe
    std::string vramNote();
    // idle: one sample a minute so /health and the spill-over line stay current; returns the next due time
    std::optional<TimePoint> vramIdleTick();
    void abortQueued(whirl::UniqueLock& lk) WH_REQUIRES(q_mutex_);
    // v0.2.2.2 memory governor: gst_ / gv_ (arch025-engine S8, module server.governor)

    ServerModel& m_;
    device::Backend& ops_;
    const Tokenizer& tok_;
    chat::ChatTokens ct_;
    EngineOptions opt_;
    std::uint32_t ctx_;
    EngineThread owner_;  // arch024 M8: set by runLoop; ck_, pool_, ps_, tb_ and the governor glue check against it
    CkptStore ck_;  // checkpoint buffers (arch024 M2)
    SpecState spec_;  // arch025-N4f: the former members timing_ .. policy_cycles_, here and in that order
    FloorState fl_;   // arch025-N4f: the former members floor_ .. stat_floor_periods_, here and in that order
    // sampling buffers and sampling::RowSource over this engine's verify rows (the draws and the Acceptor read the
    // device through it); arch025-N4a: the former members samp_dev_ .. ctl_host_, here and in that order
    SampleBuffers sb_{m_, ops_};
    bool spec_on_ = false;
    std::uint64_t conv_bytes_ = 0, ssm_bytes_ = 0;
    std::vector<Slot> slots_;
    // queue
    whirl::Mutex q_mutex_;
    std::condition_variable q_cond_;
    std::deque<Job*> queue_ WH_GUARDED_BY(q_mutex_);
    std::uint32_t n_active_ WH_GUARDED_BY(q_mutex_) = 0;
    std::atomic<std::uint32_t> n_building_{0};
    std::atomic<std::uint64_t> next_id_{1};
    IdGen ids_;         // response / tool-call ids and default sampling seeds (one stream)
    ResponseCtx rcx_;   // what each slot's StreamState reads (model id, tokenizer, ids_)
    bool stop_ WH_GUARDED_BY(q_mutex_) = false;
    std::vector<std::function<void()>> loop_calls_ WH_GUARDED_BY(q_mutex_);  // runOnLoop (tests)
    void runLoopCalls(whirl::UniqueLock& lk) WH_REQUIRES(q_mutex_);
    // batch statistics (arch025-N4c: the former stat_t0_ .. stat_dfl_gap_)
    BatchStats bst_;
    // WDDM-LOG: runtime video memory watch (setVramProbe)
    std::function<wddmwatch::Sample()> vram_probe_;
    std::function<std::optional<wddmwatch::Others>()> vram_others_;
    wddmwatch::Watch vram_watch_;
    TimePoint vram_idle_next_{};    // next idle sample (once a minute while no request runs)
    double vram_idle_s_ = 60.0;
    std::uint64_t prefill_tok_cum_ = 0;  // all slots, since start (DecodeWatch: was another request prefilling)
    bool vram_warn_ = false;
    whirl::Mutex vram_mu_;  // vram_last_ / vram_spill_ for health() (HTTP threads)
    wddmwatch::Sample vram_last_ WH_GUARDED_BY(vram_mu_);
    bool vram_spill_ WH_GUARDED_BY(vram_mu_) = false;
    std::vector<std::uint32_t> crlf_norm_;
    TimePoint vram_last_t_{};  // last probe sample
    // v0.2.2.2 memory governor (arch025-engine S8, module server.governor): the former members gov_ .. gov_json_, here
    // (where gov_ was) and in that order, plus gov_mu_ (F14: gov_json_ under its own lock, no longer vram_mu_); private,
    // reached only through gv_ (DESIGN v3 2.5)
    GovState gst_;
    std::atomic<std::uint32_t> ctx_pub_{0};
    // host tiers (the tier itself: tb_.tier())
    std::vector<KvArr> kv_arrays_;
    bool tier_wake_ WH_GUARDED_BY(q_mutex_) = false;
    std::uint64_t wake_seq_ WH_GUARDED_BY(q_mutex_) = 0;  // bumped by a queued request, a finished build, the tier IO thread
    PagePool pool_;  // KV pool pages (arch024 M3); here, where its quarantine was, so ~Engine destroys its events in
                     // the same order
    TimePoint last_tick_{};
    TimePoint last_reap_{};
    // shared prefix checkpoints, prefix cache decisions, chunk schedule (arch024 M4); here, where spes_ was, so
    // ~Engine destroys the shared checkpoints (and their events) at the same point
    PrefixStore ps_;
    // host tier bridge, spill half (arch024 M5): after ps_ and kv_arrays_ (it keeps references to both)
    TierBridge tb_;
    // shared prefix checkpoint / KV page counters (arch025-engine S6, module server.slot_cache): the former members
    // stat_spe_new_ .. stat_evictions_, here (where they were) and in that order (DESIGN v3 2.5)
    SlotPagesCounters cnt_;
    // vision: image embeddings (cache by content hash), per-slot image maps, idle release; arch025-N4b: the former
    // members vis_cfg_ .. vis_rel_pending_, here (the last members) and in that order
    ImageEmbeddings vis_{m_, ops_};
    // DFlash2 drafter context of a slot (arch025-engine S4, module server.drafter_ctx): references to bst_, m_, opt_
    // only; the last member, destroyed first, trivial destructor (DESIGN v3 2.5)
    DrafterCtx dc_{bst_, m_, opt_};
    // KV pages and checkpoints of the slots (arch025-engine S7, module server.slot_cache): references only - core.shared
    // ck_ .. tb_, slot_cache's placed state kv_arrays_ / conv_bytes_ / ssm_bytes_ / cnt_ (here, where they were) and
    // dc_; after dc_, the last member, destroyed first, trivial destructor (DESIGN v3 2.5)
    SlotPages sc_{ck_, m_, ops_, opt_, pool_, ps_, slots_, tb_, kv_arrays_, conv_bytes_, ssm_bytes_, cnt_, dc_};
    // the memory governor's logic (arch025-engine S8, module server.governor): references only - core.shared ck_ .. vram_probe_,
    // the governor's placed state vram_others_ / ctx_pub_ (here, where they were), sc_ and gst_'s members; after sc_, the
    // last member, destroyed first, trivial destructor (DESIGN v3 2.5)
    GovernorGlue gv_{ck_, ctx_, m_, ops_, opt_, owner_, pool_, ps_, slots_, tb_, vram_probe_, vram_others_, ctx_pub_, sc_, gst_};
};

}  // namespace whirl::server
