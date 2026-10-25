// What the HTTP front end sees of the engine (arch024 M6): the job it builds, the connection writer the engine
// answers through, the facts that do not change once the server accepts requests, and the calls a connection
// thread makes. http.h includes only this header; Engine (engine.h) implements EngineApi, tests use a fake.
// Prior art: llama.cpp server_context_meta / server_routes::update_meta (taken once before is_ready), vLLM's
// EngineClient (progress\arch024-S8.PRIOR_ART.md).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "protocol.h"
#include "vision_iface.h"
#include "whirl/chat.h"
#include "whirl/json.h"
#include "whirl/tokenizer.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::server {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

// Connection writer (buffered; errors make it "gone").
class Conn {
public:
    virtual ~Conn() = default;
    virtual bool write(std::string_view data) = 0;
    virtual bool flush() = 0;
    // false once the client has closed the connection (FR-SRV P2-2: polled by the engine for
    // requests that write nothing for a long time: queued, prefilling, non-streaming)
    virtual bool alive() { return true; }
};

enum class JobKind { chat, completion };

struct Job {
    std::uint64_t id = 0;
    JobKind kind = JobKind::chat;
    std::string endpoint;
    Conn* w = nullptr;
    std::vector<std::uint32_t> tokens;
    Params params;
    bool think = false;
    json::Array tools;
    bool tools_on = false;
    TimePoint t_arrive{};
    bool sys_wait_logged = false;
    bool tier_wait_logged = false;
    bool pool_wait_logged = false;  // FR-SRV P2-3
    bool pf_checked = false;
    // vision: the request's images (preprocessed), their spans and embeddings
    std::shared_ptr<JobVis> vis;
    bool done = false;  // guarded by Engine::q_mutex
};

// Read-only facts for the HTTP layer, taken once when the HttpServer is built (after every startup setter:
// setCtx / setNumerics / setKvSsdMode / attachVision / setStartup) and never changed afterwards. What changes while
// the server runs (the published context, health, counters, governor) stays an EngineApi call.
struct EngineFacts {
    std::string model_name;    // the model id (/v1/models, /props, every response)
    std::string model_file;    // GGUF file name without directory (GET /props model_path; "" = model_name)
    chat::TemplateKind tmpl = chat::TemplateKind::a;
    const Tokenizer* tok = nullptr;  // encodes prompts on the connection threads (const, thread-safe)
    bool use_mtp = false;
    std::string numerics_json;  // GET /props "numerics" ("" = not reported)
    std::string startup_json;   // GET /props and /health "startup" ("" = not reported)
    std::string kv_ssd_mode;    // GET /props kv_ssd mode
    VisionConfig vision;        // vision.vis == nullptr: no --mmproj
    // model metadata (/v1/models meta; completion token ids are checked against n_vocab)
    std::string arch;
    std::uint32_t n_vocab = 0, n_embd = 0, n_layer = 0;
};

class EngineApi {
public:
    struct Health {
        std::uint32_t active = 0;
        std::size_t queued = 0;
        std::size_t slots = 0;
        // WDDM-LOG: the last video memory sample (vram_ok false: no probe / no sample yet)
        bool vram_ok = false, vram_spill = false;
        std::uint64_t vram_local = 0, vram_budget = 0, vram_shared = 0;
    };
    struct Stats {
        std::uint64_t spe_new = 0, spe_hits = 0, spe_tok = 0, spe_evict = 0, unshare = 0, evictions = 0;
        std::uint64_t cow_pages = 0, q_waits = 0, spill_waits = 0;
        std::uint64_t vis_enc = 0, vis_hit = 0;  // vision: images encoded / embedding cache hits
        // SSD tier: bytes written since start, bytes not written (shared
        // blocks), bytes in use, superseded sessions trimmed
        std::uint64_t ssd_written = 0, ssd_dedup = 0, ssd_used = 0, ssd_entries = 0, superseded = 0;
        // DFlash2: cycles that drafted with it, its drafts / accepted, cycles of a slot whose drafter
        // context was not ready (no DFlash2 drafts), positions without features (forward rows
        // outside the capture, restores)
        std::uint64_t dfl_cycles = 0, dfl_drafted = 0, dfl_accepted = 0, dfl_cold = 0, dfl_gap = 0;
        // DFlash2 width auto (WHIRL_DFLASH_WIDTH=auto): cycles whose cost model skipped the drafter; full-width probes
        std::uint64_t dfl_skipped = 0, dfl_probes = 0;
    };

    virtual ~EngineApi() = default;
    virtual std::uint64_t nextJobId() = 0;
    // Queue a built job and wait until the main thread has answered it. Ends the connection's beginBuilding (the
    // caller does not call endBuilding after it).
    virtual void submitAndWait(Job& job) = 0;
    // A connection is reading / building a request (the engine waits a moment before a lone prefill while one is);
    // ended by endBuilding (request refused) or by submitAndWait.
    virtual void beginBuilding() = 0;
    virtual void endBuilding() = 0;
    virtual Health health() = 0;
    // FR-SRV P3-5: the counters other threads may read (HTTP /props): atomics only
    virtual Stats statsShared() const = 0;
    // "governor" object of /health and /props ("" when off)
    virtual std::string governorJson() = 0;
    // context per request for new requests (/props n_ctx; the memory governor lowers it while the WDDM budget is low)
    virtual std::uint32_t ctxPublished() const = 0;
    virtual EngineFacts facts() const = 0;
};

}  // namespace whirl::server
