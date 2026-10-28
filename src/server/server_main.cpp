// `whirl-server MODEL.gguf [options]`: loads a qwen35 / qwen35moe GGUF on
// the GPU and serves the OpenAI-compatible API (continuous batching over
// --parallel slots, prefix cache, shared prefix checkpoints, host tiers).
// Reimplements the WHIRL Zig research prototype's server.zig (main, buildKvArrays,
// tierFingerprint; defaultBatchDrafts is spec.policy's since arch025-N10).
// SPDX-License-Identifier: Apache-2.0

#include "server_main.h"

#include "backend.h"
#include "cache/fingerprint.h"
#include "draft_caps.h"
#include "engine.h"
#include "http.h"
#include "log.h"
#include "server_args.h"  // Options, parseServerArgs, resolveServerConfig, kHelpBody (arch025-C1a)
#include "startup_steps.h"
#include "device/backend.h"
#include "platform/env.h"
#include "platform/paths.h"
#include "platform/time.h"
#include "hip/alloc_log.h"
#include "hip/launch_log.h"
#include "tier/kv_tier.h"
#include "tier/ram_size.h"
#include "whirl/dump.h"
#include "whirl/common.h"
#include "whirl/small_helpers.h"  // satSub / alignUp / unixNow / msSince / nowMs (arch025-N6)
#include "whirl/device_mem.h"
#include "whirl/dflash_policy.h"
#include "whirl/gguf.h"
#include "whirl/gpu_select.h"
#include "whirl/hip.h"
#include "whirl/hip_runtime_loader.h"
#include "whirl/model.h"
#include "model/arch_open.h"  // the architecture module: config + pre-load sizing (arch025-M5; openArch: N15d)
#include "whirl/pool_budget.h"
#include "whirl/startup_fit.h"
#include "whirl/mem_governor.h"
#include "whirl/tokenizer.h"
#include "whirl/uma_mem.h"
#include "config/run_settings.h"  // the run settings the server shares with the CLI (arch025-N15)
#include "config/device_choice.h"  // the GPU choice shared with the CLI (arch025-N15b)
#include "whirl/vram_limit.h"
#include "whirl/wddm_watch.h"
#include "release/release.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>  // offsetof (arch025-surv member-order guard)
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "platform/proc.h"  // shutdown event, console control handler, working set (arch025 PL4)
#include "platform/sys.h"   // physical RAM (arch025 PL4)

namespace whirl::server {

// planFit's smallest batch must hold the largest forward the scheduler builds (VRAM-01)
static_assert(vram::min_prefill_batch >= prefill_chunk + prefill_merge);

namespace {

// arch024-S0a (PLAN v3 N3, P-size): the inputs of the startup sizing's pure functions (startfit::*, vram::planFit,
// numerics::serverPoolFit, tier::ramTierSize), gathered while the server sizes itself and logged as one line
// "startup inputs: key=value ..." so tests/pins/startup_replay can recompute the decisions from a real log. Log only:
// nothing here feeds a decision. Values are percent-encoded (space %20, % %25, = %3D, newline %0A).
std::string inEnc(std::string_view s) {
    std::string o;
    for (const char c : s) {
        if (c == ' ') o += "%20";
        else if (c == '%') o += "%25";
        else if (c == '=') o += "%3D";
        else if (c == '\n') o += "%0A";
        else o += c;
    }
    return o.empty() ? std::string("%") : o;  // "%" alone: the empty string
}
struct StartupInputs {
    std::string line;
    template <class T>
    void put(std::string_view k, const T& v) {
        if constexpr (std::is_same_v<T, bool>) line += std::format(" {}={}", k, v ? 1 : 0);
        else if constexpr (std::is_convertible_v<const T&, std::string_view>) line += std::format(" {}={}", k, inEnc(std::string_view(v)));
        else line += std::format(" {}={}", k, v);
    }
    void putFit(std::string_view p, const vram::FitInput& fi) {
        const std::string s(p);
        put(s + ".avail", fi.avail);
        put(s + ".weights", fi.weights);
        put(s + ".fixed", fi.fixed);
        put(s + ".buf_per_row", fi.buf_per_row);
        put(s + ".buf_fixed", fi.buf_fixed);
        put(s + ".ckpt_bytes", fi.ckpt_bytes);
        put(s + ".parallel", fi.parallel);
        put(s + ".kv_per_token", fi.kv_per_token);
        put(s + ".pool_min_tokens", fi.pool_min_tokens);
        put(s + ".pool_hard_min", fi.pool_hard_min);
        put(s + ".batch", fi.batch);
        put(s + ".n_ck", fi.n_ck);
        put(s + ".n_spe", fi.n_spe);
        put(s + ".batch_fixed", fi.batch_fixed);
        put(s + ".ck_fixed", fi.ck_fixed);
    }
};

// Graceful shutdown: Ctrl+C / Ctrl+Break / closing the console window (and logoff /
// system shutdown) stop the engine, which answers queued requests 503, lets the
// running ones finish, flushes the prefix-cache tiers (pending RAM -> SSD writes,
// bounded by Engine::shutdown_flush_ms) and returns from runLoop; main then exits.
// The handler thread waits for that (the OS ends the process about 5 s after a
// close / logoff / shutdown event regardless). A second Ctrl+C terminates at once.
std::atomic<Engine*> g_engine{nullptr};
std::atomic<bool> g_stopping{false};
platform::proc::Handle g_stopped = nullptr;  // set when runLoop has returned

// platform::proc::addCtrlHandler's trampoline returns true as TRUE, false as FALSE
bool onConsoleCtrl(int type) {
    namespace pproc = platform::proc;
    Engine* e = g_engine.load();
    if (e == nullptr || g_stopping.exchange(true)) return false;  // default handling: exit now
    const char* what = type == pproc::kCtrlC ? "Ctrl+C" : type == pproc::kCtrlBreak ? "Ctrl+Break" : type == pproc::kCtrlClose ? "console closed" : "logoff / shutdown";
    logI("shutdown requested ({}): no new requests; finishing running ones and flushing the prefix-cache tiers", what);
    e->stop();
    const bool interactive = type == pproc::kCtrlC || type == pproc::kCtrlBreak;
    const std::uint32_t wait_ms = interactive ? static_cast<std::uint32_t>(Engine::shutdown_flush_long_ms + 60000) : 4500;
    if (g_stopped != nullptr && pproc::wait(g_stopped, wait_ms) == pproc::kWaitObject0) {
        // main returns from serveMain and the process exits normally; for close /
        // logoff / shutdown events the OS ends the process when the handler returns
        if (!interactive) pproc::sleepMs(500);
        return true;
    }
    logW("shutdown: the engine did not stop within {} s; exiting", wait_ms / 1000);
    return false;
}

// Windows timer resolution: 1 ms while serving. At the default tick (10-16 ms) every
// short wait of the main loop (the 1 ms polls of a burst gather and of host-tier
// restores, the restore tail's 200 us poll) sleeps a whole tick. Restored on exit.
struct TimerRes {
    bool on = platform::time::beginPeriod1ms();  // arch025 PL1: timeBeginPeriod(1) / timeEndPeriod(1)
    ~TimerRes() {
        if (on) platform::time::endPeriod1ms();
    }
};

// WHIRL_TIMER_PROBE: mean actual duration of a short sleep (diagnostics)
double sleepMs(std::chrono::microseconds d, int n) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < n; ++k) std::this_thread::sleep_for(d);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
}

constexpr std::uint32_t ctx_per_slot_default = 131072;
constexpr std::uint32_t floor_second_ctx = 65536;
constexpr std::uint32_t serve_prefill_batch = 4096;
constexpr std::size_t n_ckpt_default = 4;
// -np 2..4: 3 per slot (TAIL-CK: tail checkpoint before the question + prompt-end + gen-end; with 2
// the tail would push out what a regenerate or the next turn resumes from), about 150.6 MiB each
// for Qwen3.8-27B, 63.8 MiB for Ornith-1.5-35B-A3B; the KV-fit estimate below counts them and
// lowers them to 2 (tail off) when VRAM is tight
constexpr std::size_t n_ckpt_default_par = 3;
constexpr std::size_t n_spe_default = 2;
// KV pool (tokens, smallest format) below which startup lowers the prefill batch / checkpoints
constexpr std::uint32_t vram_tight_pool = 16384;
constexpr std::uint32_t sys_min_default = sched::sys_min_default;  // shared with the CLI (CLI-SCHED)
constexpr std::uint64_t kv_ssd_gb_default = 64;
constexpr std::uint32_t decode_min_tps_default = 20;

std::string g_program = "whirl-server";

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "%s: error: %s\n  (run '%s --help' for the options)\n", g_program.c_str(), msg.c_str(), g_program.c_str());
    std::exit(app::exit_usage);
}

// the server's settings::Warn (a bad WHIRL_* value: a warning line, never a startup failure; FR-SRV P3-10)
void settingsWarn(const std::string& msg) { logW("{}", msg); }

// WHIRL_DRAFT_VOCAB (default: the embedded 64k subset on dense qwen35 models with the 248320-token
// vocabulary): keep only that vocabulary subset of the 2-bit draft head (special and byte tokens are
// always added); outputs are unchanged, only the acceptance can drop.
void applyDraftVocab(qwen35::Model& model, const Tokenizer& tok, const qwen35::DraftVocabChoice& dv) {
    // the decision and the subset are spec.policy's (shared with the CLI, arch025-N10 leg 6); the messages are the server's
    switch (qwen35::draftVocabGate(dv, model.cfg.moe, model.cfg.n_vocab, model.draft_d2.has_value())) {
        case qwen35::DraftVocabGate::none: return;
        case qwen35::DraftVocabGate::not_fit:
            if (dv.by_default)
                logI("draft head: full vocabulary (the embedded 64k subset is for dense qwen35 models with a {}-token vocabulary)",
                     qwen35::draft_vocab_embedded_n_vocab);
            else
                logW("WHIRL_DRAFT_VOCAB=64k ignored: the embedded subset needs a {}-token vocabulary (model: {})",
                     qwen35::draft_vocab_embedded_n_vocab, model.cfg.n_vocab);
            return;
        case qwen35::DraftVocabGate::no_d2:
            if (dv.by_default) logI("draft head: full vocabulary (the vocabulary subset needs the 2-bit draft head)");
            else logW("WHIRL_DRAFT_VOCAB ignored: it needs the 2-bit draft head (Q6_K output head, WHIRL_DRAFT_HEAD not q4)");
            return;
        case qwen35::DraftVocabGate::subset: break;
    }
    const qwen35::DraftVocabSubset sub =
        qwen35::draftVocabSubset(dv, &qwen35::embeddedDraftVocab64k, model.cfg.n_vocab, qwen35::draftVocabRequired(tok));
    if (model.setDraftVocab(sub.ids))
        logI("draft head: vocabulary subset {} ({} of {} rows; {} special / byte tokens added)", sub.label, sub.ids.size(),
             model.cfg.n_vocab, sub.added);
    else if (dv.by_default)  // the default only: no warning for a setting the user did not make
        logI("draft head: full vocabulary (no vocabulary subset with this draft head / kernel set)");
    else
        logW("WHIRL_DRAFT_VOCAB ignored: it needs the 2-bit draft head (Q6_K output head, WHIRL_DRAFT_HEAD not q4)");
}

double gib(std::size_t a, std::size_t b) { return static_cast<double>(a > b ? a - b : 0) / (1024.0 * 1024.0 * 1024.0); }

// Identity of everything that decides the bits of a tier entry.
std::uint64_t tierFingerprint(const qwen35::Model& m, const std::string& path, const qwen35::LoadStats& stats, bool use_mtp,
                              std::uint64_t n_gdn, const tier::Layout& lay) {
    // assembly in cache::serverFingerprintInputs (tested in whirl-cache-tests; FIX-TEST-A, FR-TEST P2-3)
    cache::ServerIdentity id;
    id.exe_id = tier::exeIdentity();
    id.model_path = path;
    id.tensors = stats.tensors;
    id.bytes = stats.bytes;
    id.kv_name = m.kvName();
    id.use_mtp = use_mtp;
    id.n_gdn = n_gdn;
    id.ck_bytes = lay.ck_bytes;
    id.page_bytes = lay.page_bytes;
    id.max_batch = m.max_batch;
    id.numerics = whirl::numerics::logLine(m.num_plan);
    const hip::DeviceInfo dev = hip::describeDevice(hip::getDevice());
    id.gpu_arch = dev.gcn_arch;
    id.gpu_name = dev.name;
    const hip::EmbeddedObject* co = hip::embeddedObjectFor(m.arch);
    id.kernel_arch = hip::archName(m.arch);
    id.kernel_co_bytes = co != nullptr ? *co->size : 0ull;
    id.env = cache::fingerprintEnv();
    return cache::tierFingerprint(cache::serverFingerprintInputs(id));
}

// arch025-C1a: what buildModel leaves for the rest of serveMain (planPool / startTier / runServer in C1b). The members
// are serveMain's old locals in their old declaration order (so they are destroyed in the old order, after the
// engine and the tiers); buildModel's body still names them through references of the old names.
struct LoadedModel {
    TimePoint t_load0;
    std::optional<gguf::File> f;
    const model::ArchModule* am = nullptr;
    qwen35::Config cfg;
    int dev = 0;
    hip::DeviceInfo info;
    std::unique_ptr<device::Backend> ops;
    bool is_uma = false;
    hip::UmaMem uma;
    std::optional<std::uint32_t> pool_req;
    std::uint32_t cap_max = 0;
    std::uint32_t slot_ctx = 0;
    vram::CardDefaults card;
    hip::GpuOthers others0;
    hip::WddmMemInfo wddm_start;
    std::vector<std::string> adjusted;
    StartupInputs sin;
    bool margin_explicit = false;
    std::optional<Tokenizer> tok;
    chat::TemplateKind tmpl{};
    qwen35::LoadStats stats;
    qwen35::LoadOptions lo;
    bool no_pc = false;
    std::size_t n_ck = 0;
    std::uint32_t n_spe_plan = 0;
    std::optional<std::string> dm_resolved;
    std::optional<gguf::File> dfile;
    bool use_dflash = false;
    hip::MemInfo mem0;
    std::unique_ptr<qwen35::Model> mp;
    std::size_t prefill_exec = 0;
    hip::MemInfo mem;
    std::uint64_t embd_pinned = 0;
    std::vector<std::string> dfl_warn;
    whirl::dflash::Params dfl_par;
    qwen35::DraftVocabChoice draft_vocab;
    bool use_mtp = false;
    qwen35::MtpDefaults def;
    std::uint32_t drafts = 0;
    float p_min = 0;
    std::uint32_t n_min = 0;
    std::optional<std::string> adapt_env;
    std::uint32_t adapt = 0;
    bool auto_mode = false;
    std::array<std::uint32_t, gdn_max_seg + 1> batch_drafts{};
    std::optional<std::string> bd_env;

    // an explicit setting lowered to what fits: one "adjusted:" line each (the startup summary lists them)
    void adjust(const std::string& what, const std::string& why) {
        logW("adjusted: {} ({})", what, why);
        adjusted.push_back(what);
    }
    // the margin kept free under the WDDM local budget (pool sizing, the re-check after allocation)
    poolbudget::Margin budgetMargin(std::uint64_t budget) const {
        const hip::GpuOthers& o = others0;
        poolbudget::MarginIn mi;
        if (margin_explicit) mi.room_override = static_cast<std::uint64_t>(settings::envU32("POOL_BUDGET_MARGIN_MB", 0, settingsWarn)) << 20;
        mi.local_budget = budget;
        mi.budget_start = wddm_start.ok ? wddm_start.local_budget : 0;
        mi.moe = cfg.moe;
        mi.others_known = o.ok;
        mi.others = o.others;
        mi.display = o.display_known && o.display;
        return poolbudget::defaultMargin(mi);
    }
};

// arch025-C1a: serveMain from the GGUF open to the MTP setup (arch module via model::findArch, GPU choice, the
// pre-load VRAM fit, Model::load + tune, the DFlash2 drafter, numerics::applyFront, the MTP draft head), verbatim; its
// results go to `b`. Same calls, log lines and env reads in the same order as before.
void buildModel(Options& opt, const numerics::Request& nreq, const char* lad, LoadedModel& b) {
    b.t_load0 = Clock::now();
    const gguf::File& f = b.f.emplace(gguf::File::open(opt.path));
    // the architecture module (src/model/arch_registry.h, arch025-M5): the config and the pre-load sizing below
    // (model::openArch, the CLI's too: arch025-N15d)
    {
        model::OpenedArch oa = model::openArch(f);
        b.am = oa.am;
        b.cfg = std::move(oa.cfg);
    }
    const model::ArchModule* const& am = b.am;
    const qwen35::Config& cfg = b.cfg;
    int& dev = b.dev = 0;
    // the GPU choice shared with the CLI (src/config/device_choice.h, arch025-N15b): WHIRL_HIP_DEVICE, else the s8-719
    // restriction (the runtime sees only the chosen GPU; its note logged before pickDevice) + pickDevice
    const settings::DeviceChoice dch = settings::chooseDevice(opt.device, settingsWarn, [](const std::string& s) { logI("{}", s); });
    dev = dch.dev;
    const int dev_shown = dch.shown;  // the index as --list-devices prints it (before the restriction)
    const hip::DeviceInfo& info = b.info = hip::describeDevice(dev);
    hip::setDevice(dev);
    // the device backend: memory queries of the plan below, then the engine's / tiers' device operations (ARCH-1 D-5)
    const std::unique_ptr<device::Backend>& ops = b.ops = device::makeHipBackend();
    // integrated GPU (Radeon 8060S): the "VRAM" is system memory shared with the CPU
    const bool& is_uma = b.is_uma = info.integrated;
    // its real dedicated memory is the BIOS carve-out (DXGI), not info.total_mem (UMA-1)
    const hip::UmaMem& uma = b.uma = is_uma ? ops->umaMemory(dev) : hip::UmaMem{};
    std::optional<std::uint32_t>& pool_req = b.pool_req = opt.ctx ? opt.ctx : (is_uma ? std::optional<std::uint32_t>(262144) : std::nullopt);
    b.cap_max = 262144;
    const std::uint32_t& slot_ctx = b.slot_ctx = static_cast<std::uint32_t>(alignUp(
        opt.ctx_per_slot ? *opt.ctx_per_slot : (pool_req ? std::min(*pool_req, ctx_per_slot_default) : ctx_per_slot_default),
        kv_page));
    if (slot_ctx < 1024) die(std::format("context per slot must be >= 1024 (got {})", slot_ctx));
    logI("model: {}", opt.path);
    logI("arch {}, layers {} (attn every {}), embd {}, ff {}, vocab {}{}", cfg.archName(), cfg.n_layer, cfg.interval,
         cfg.n_embd, cfg.n_ff, cfg.n_vocab, cfg.n_nextn > 0 ? ", MTP (nextn) layer" : "");
    if (cfg.moe)
        logI("experts {} (top {}, ff {}) + shared expert ff {}", cfg.n_expert, cfg.n_expert_used, cfg.n_ff_exp, cfg.n_ff);
    if (uma.known)
        logI("device {}: {} ({}), {} GPU memory (UMA carve-out from the BIOS / Armoury Crate; the runtime reports {:.1f} GiB, "
             "which includes system RAM){}",
             dev_shown, info.name, info.gcn_arch, hip::umaSizeText(uma), static_cast<double>(info.total_mem) / (1024.0 * 1024.0 * 1024.0),
             vram::limitBytes() ? " (simulated limit)" : "");
    else
        logI("device {}: {} ({}), {:.1f} GiB VRAM{}", dev_shown, info.name, info.gcn_arch,
             static_cast<double>(info.total_mem) / (1024.0 * 1024.0 * 1024.0), vram::limitBytes() ? " (simulated limit)" : "");
    logI("HIP runtime: {}", hip::runtime::describe(hip::runtime::load()));  // the exact DLL loaded (ARCH-1 D-1)
    if (vram::limitBytes()) logI("{}: memory sizes, the WDDM budget and allocations are capped as on a card of that size", vram::limitNote());
    // small cards (< 20 GiB, e.g. a 16 GB RX 9070 XT that also drives the desktop): 1 slot,
    // VRAM headroom for other apps (the KV format is the mode's, as on every card)
    vram::CardDefaults& card = b.card;
    {
        vram::CardInput ci;
        ci.total = info.total_mem;
        ci.uma = is_uma;
        if (opt.parallel_given) ci.parallel_arg = opt.parallel;
        ci.parallel_default = opt.parallel;
        ci.headroom_mb_env = settings::vramHeadroomMb(settingsWarn);
        ci.reserve_explicit = settings::poolReserveGiven();
        card = vram::cardDefaults(ci);
        opt.parallel = card.parallel;
        if (!card.note.empty()) logI("{}", card.note);
    }
    // v0.2.2.1 (owner 2026-10-21): what other processes already use on this GPU, read once now, synchronously,
    // before WHIRL loads or sizes anything (the first PDH query of a process takes ~0.2-0.3 s; owner: 「查記憶體整個
    // 才不到1秒，不慢」). This process's own usage is read in the same collection and subtracted. The WDDM budget
    // margin grows with it and with a monitor on the card (poolbudget::defaultMargin).
    const hip::GpuOthers& others0 = b.others0 = ops->gpuOthers(dev);
    const hip::WddmMemInfo& wddm_start = b.wddm_start = ops->wddmMemInfo();  // the budget before WHIRL allocates (MarginIn::budget_start)
    {
        const char* const disp = others0.display_known ? (others0.display ? "yes" : "no") : "unknown";
        if (others0.ok)
            logI("other processes use {:.2f} GiB of VRAM on this GPU (display attached: {}) (adapter {:.2f} GiB, this process "
                 "{:.2f} GiB; read in {:.0f} ms)",
                 others0.others / 1073741824.0, disp, others0.adapter_dedicated / 1073741824.0,
                 others0.own_dedicated / 1073741824.0, others0.cost_us / 1000.0);
        else
            logI("other processes' VRAM on this GPU: unknown (GPU performance counters not readable; display attached: {}); "
                 "the WDDM budget margin uses the budget-only rule ({:.0f} ms)",
                 disp, others0.cost_us / 1000.0);
    }
    // owner 2026-10-21 (「我們系統先查而調整反而是好事，再將調整結果寫在log」): an explicit setting that does not fit
    // the measured resources is lowered to what fits, never silently: one "adjusted:" line each, a summary at the end
    // of startup, and the effective values in /props and /health ("startup"). Only when even one slot with a minimal
    // pool does not fit does the server refuse to start. Order in which they run (each step only if still needed):
    //   1. --parallel: each slot's recurrent state and checkpoints are fixed VRAM; lowered when one full request
    //      (min(ctx-per-slot, 16384)) no longer fits next to them: in the pre-load estimate, then again with the VRAM
    //      measured after the weights are loaded, before any slot is allocated;
    //   2. prefill batch / checkpoint counts set by the user (the VRAM-tight plan; only if still short);
    //   3. RAM tier (--kv-ram-mb): system RAM, not VRAM (17 GiB pinned did not move the R9700's WDDM budget), so it is
    //      never lowered to make room in VRAM. RAM-side rules only, in this order (0.2.2.2): every RAM tier capped at
    //      20480 MiB while a crash with larger pinned RAM tiers is investigated (task_063b1573; the automatic size gets
    //      an info line, an explicit one an "adjusted:" line), then a size above half of the available RAM -> half;
    //   4. -c / --ctx (the shared pool) down to what fits under the budget with the margin;
    //   5. --ctx-per-slot down to the pool. The per-request context reserves no memory by itself (the pool is
    //      shared), so lowering it first frees nothing: it is capped after the pool is known.
    StartupInputs& sin = b.sin;  // arch024-S0a: logged as "startup inputs:" at the end of the sizing (log only)
    sin.put("others.ok", others0.ok);
    sin.put("others.mib", others0.others >> 20);
    sin.put("others.display", others0.display_known ? (others0.display ? 1 : 0) : -1);
    sin.put("wddm_start.ok", wddm_start.ok);
    sin.put("wddm_start.budget", wddm_start.ok ? wddm_start.local_budget : 0);
    const auto adjust = [&b](const std::string& what, const std::string& why) { b.adjust(what, why); };
    // the margin kept free under the WDDM local budget (pool sizing, the re-check after allocation)
    b.margin_explicit = settings::envGet("POOL_BUDGET_MARGIN_MB").has_value();
    const auto budgetMargin = [&b](std::uint64_t budget) { return b.budgetMargin(budget); };

    const Tokenizer& tok = b.tok.emplace(Tokenizer::fromGguf(f));
    const std::string_view tmpl_text = f.getStringOr("tokenizer.chat_template", "");
    const chat::TemplateKind& tmpl = b.tmpl = chat::detectTemplate(tmpl_text);
    logI("chat template: GGUF tokenizer.chat_template ({} B) -> built-in Qwen3.x renderer variant '{}' ({})", tmpl_text.size(),
         chat::templateName(tmpl),
         tmpl == chat::TemplateKind::a ? "reasoning effort, merged system, preserve_thinking"
         : tmpl == chat::TemplateKind::b ? "system + tools, <think> in history"
                                         : "system + tools, <think> in history, scalar tool arguments as True / False / None");

    qwen35::LoadStats& stats = b.stats;
    qwen35::LoadOptions& lo = b.lo;
    lo.numerics = nreq;
    lo.max_batch = serve_prefill_batch;
    lo.embd_on_host = settings::envFlag("EMBD_HOST", true);  // WHIRL_EMBD_HOST=0: embedding in VRAM
    if (auto v = settings::envGet("PREFILL_BATCH")) {
        // (arch025-N15: the shared number rule; a malformed value keeps the default batch)
        if (auto x = settings::parseU32(*v)) {
            // the scheduler's largest forward is a 1024-row chunk + a 256-row tail: a smaller batch would
            // fail every long prompt with BatchTooLarge (VRAM-01), so it is raised to that floor
            // (arch025-N15c: vram::prefillBatchAtLoad, the CLI's rule too)
            if (*x < vram::min_prefill_batch)
                logW("WHIRL_PREFILL_BATCH={} is below the smallest prefill batch the scheduler needs ({} rows: one {}-row chunk + "
                     "a {}-row tail); using {}",
                     *v, vram::min_prefill_batch, prefill_chunk, prefill_merge, vram::min_prefill_batch);
            lo.max_batch = vram::prefillBatchAtLoad(*x, qwen35::max_batch_limit);
        }
    }
    // prefix checkpoints (per slot / shared); the auto-shrink below may lower the defaults
    const bool& no_pc = b.no_pc = settings::envFlag("NO_PREFIX_CACHE", false);
    const std::size_t ck_def = opt.parallel == 1 ? n_ckpt_default : (opt.parallel <= 4 ? n_ckpt_default_par : 1);
    std::size_t& n_ck = b.n_ck = no_pc ? 0 : settings::envU32("SERVE_CKPTS", static_cast<std::uint32_t>(ck_def), settingsWarn);
    if (n_ck > tier::max_ck) {  // FR-CACHE P2-3: tier entries / restore maps hold at most max_ck checkpoints
        logW("WHIRL_SERVE_CKPTS={} is above the maximum ({}); using {}", n_ck, tier::max_ck, tier::max_ck);
        n_ck = tier::max_ck;
    }
    std::uint32_t& n_spe_plan = b.n_spe_plan = (n_ck == 0 || no_pc) ? 0 : settings::envU32("SYS_CKPTS", static_cast<std::uint32_t>(n_spe_default), settingsWarn);
    // --draft dflash: check the drafter against the target before loading (headers only, before the
    // VRAM fit check, which counts the drafter instead of the MTP block); a
    // model without a matching drafter keeps the MTP drafts (whirl/dflash_policy.h)
    // (use_dflash is this block's result: the VRAM fit below takes it, so the choice cannot move after the fit; arch025-C0)
    std::optional<std::string>& dm_resolved = b.dm_resolved;
    std::optional<gguf::File>& dfile = b.dfile;
    bool& use_dflash = b.use_dflash = [&] {
        const std::optional<std::string> dm_cli = opt.draft_model.empty() ? std::nullopt : std::optional<std::string>(opt.draft_model);
        const auto dm = dm_resolved = whirl::dflash::resolveDraftModel(dm_cli, settings::draftModel());
        const auto src = *whirl::dflash::parseSource(opt.draft);
        if (src == whirl::dflash::Source::dflash)
            if (const std::string note = whirl::dflash::draftModelNote(dm_cli, settings::draftModel()); !note.empty()) logI("{}", note);
        std::string why;
        if (src == whirl::dflash::Source::dflash && dm) why = qwen35::dflashIncompatibleFile(f, *dm, &dfile);
        const whirl::dflash::Choice ch = whirl::dflash::choose(src, dm, why, true);
        if (!ch.warning.empty()) logW("{}", ch.warning);
        const bool on = ch.source == whirl::dflash::Source::dflash;
        if (on) lo.no_mtp = true;
        else dfile.reset();
        return on;
    }();
    // VRAM estimate before loading: when the KV pool left after weights, prefill buffers and
    // checkpoints would be small, lower the prefill batch and the checkpoint counts first
    // (only values the user did not set; nothing changes when the pool has room). Not on the
    // UMA iGPU, whose VRAM numbers do not bound what it can allocate.
    if (!is_uma) {
        const hip::MemInfo mf = ops->memInfo();
        const hip::WddmMemInfo w = ops->wddmMemInfo();
        poolbudget::RoomIn ri;
        ri.reserve_explicit = settings::poolReserveGiven();
        ri.reserve = static_cast<std::uint64_t>(settings::poolReserveMb(poolbudget::defaultReserveMb(cfg.moe, w.ok), settingsWarn)) << 20;
        const poolbudget::Margin mg = budgetMargin(w.local_budget);
        const std::uint64_t margin = mg.bytes;
        ri.margin = margin;
        ri.free = mf.free;
        ri.wddm_ok = w.ok;
        ri.local_budget = w.local_budget;
        ri.local_usage = w.local_usage;
        ri.headroom = card.headroom;
        const std::uint64_t av = poolbudget::kvRoom(ri).bytes;
        // the module's load estimate, the fit input (the DFlash2 drafter instead of the MTP block, FR-DFL P3-1)
        // and the first plan (startup::dgpuFit, arch025-C0: the server tests run the same code)
        startup::DgpuFitIn fin;
        fin.avail = av;
        fin.parallel = opt.parallel;
        fin.embd_on_host = lo.embd_on_host;
        if (use_dflash) {
            startup::DflashFitIn d;
            std::vector<std::string> pw;
            d.params = whirl::dflash::parseParams([](const char* n) { return settings::envGet(n); }, pw);
            d.dc = qwen35::DFlashConfig::fromGguf(*dfile);
            d.file = &*dfile;
            if (auto v = settings::envGet("SNAP_SETS")) {
                try {
                    d.snap_env = static_cast<std::uint32_t>(std::stoul(*v));
                } catch (const std::exception&) {
                }
            }
            d.gdn_replay = settings::envFlag("GDN_REPLAY", true);
            d.ngram = settings::envFlag("NGRAM", true);
            fin.dflash = std::move(d);
        }
        fin.nreq = nreq;
        fin.kv_req = numerics::kvEstimateRequest(settings::envGet("KV"));
        fin.ckpt_host = settings::envFlag("CKPT_HOST", false);
        fin.pool_min_tokens = pool_req ? *pool_req : std::min<std::uint32_t>(slot_ctx, vram_tight_pool);
        fin.max_batch = lo.max_batch;
        fin.n_ck = static_cast<std::uint32_t>(n_ck);
        fin.n_spe = n_spe_plan;
        fin.batch_fixed = settings::envGet("PREFILL_BATCH").has_value();
        fin.ck_fixed = settings::envGet("SERVE_CKPTS").has_value() || settings::envGet("SYS_CKPTS").has_value();
        const startup::DgpuFit fit = startup::dgpuFit(am->sizing, f, cfg, fin);
        // the pre-load decisions and their texts (startfit::preloadAdjust, module server.kv_fit; arch025-N10 leg 6: the -c
        // re-plan, fewer --parallel slots, the user's prefill batch / checkpoint counts, the tight / refusal / short texts)
        startfit::PreloadIn pin;
        pin.fi = fit.fi;
        pin.plan = fit.plan;
        pin.st1 = fit.ssb.st1;
        pin.pend1 = fit.ssb.pend1;
        pin.parallel = opt.parallel;
        pin.parallel_given = opt.parallel_given;
        pin.ctx_given = opt.ctx.has_value();
        pin.slot_ctx = slot_ctx;
        pin.tight_pool = vram_tight_pool;
        pin.avail = av;
        pin.margin = margin;
        pin.pending = mg.pending;
        pin.room = mg.room;
        pin.weights = fit.est.weights;
        pin.total = mf.total;
        const startfit::PreloadAdjust pa = startfit::preloadAdjust(pin);
        sin.putFit("fitp", pa.fitp);
        sin.put("fitp.per_slot", pa.per_slot);
        sin.put("fitp.given", opt.parallel_given);
        sin.putFit("fitf", pa.fitf);
        sin.put("fitf.plan.batch", pa.fitf_plan.batch);
        sin.put("fitf.plan.n_ck", pa.fitf_plan.n_ck);
        sin.put("fitf.plan.n_spe", pa.fitf_plan.n_spe);
        sin.put("fitf.plan.pool_tokens", pa.fitf_plan.pool_tokens);
        opt.parallel = pa.parallel;
        for (const startfit::Adjusted& a : pa.adjusted) adjust(a.what, a.why);
        const vram::FitPlan& plan = pa.plan;
        if (plan.reduced) {
            lo.max_batch = plan.batch;
            n_ck = plan.n_ck;
            n_spe_plan = plan.n_spe;
            logW("{}", pa.tight);
        }
        // WHIRL_FIT_CHECK=0 loads anyway; borderline cases only warn
        if (pa.refuse && settings::envFlag("FIT_CHECK", true)) throw qwen35::ModelError("VramLimit", pa.refusal);
        if (!plan.fits) logW("{}", pa.short_warn);
    } else if (uma.known && hip::umaCheckOn()) {
        // integrated GPU (UMA-1): weights, buffers, checkpoints and the KV cache must fit the BIOS
        // carve-out. Too small for one request at the context asked for: refuse now, before the
        // first device-memory call (a tiny carve-out fails in hipMemGetInfo with hipError 719,
        // issue #7), naming the setting. Room for less than the default pool: the pool shrinks to
        // what the carve-out holds instead of spilling into slower system memory.
        const qwen35::LoadEstimate est = am->sizing.estimateLoad(f, cfg, lo.embd_on_host, use_dflash);
        // per-slot recurrent state + replay rows (model::slotStateBytes: the same formula for the dGPU fit and the UMA check)
        const model::SlotStateBytes ssb = model::slotStateBytes(am->sizing, cfg);
        const std::uint64_t st1 = ssb.st1;
        const std::uint64_t pend1 = ssb.pend1;
        // (WHIRL_DUMP_KV: one "D kv: site=server.est_uma" line)
        const KvFormat kv_est = numerics::kvEstimate(numerics::KvEstSite::server_uma, nreq, cfg.moe, numerics::kvEstimateRequest(settings::envGet("KV")));
        hip::UmaNeed need;
        need.weights = est.weights + (use_dflash && dfile ? am->sizing.dflashWeightBytes(*dfile) : 0);
        const std::uint64_t bufs =
            am->sizing.bufferBytes(cfg, lo.max_batch, est.ffs, est.max_elems, am->sizing.resolveFdSplits(kv_est));
        // the fixed part (startfit::umaFixed, module server.kv_fit; arch025-N10 leg 6): slots with their snapshot sets
        // (one recurrent state per MTP draft), checkpoints, the MTP block, 256 MiB, +8%
        startfit::UmaFixedIn ui;
        ui.weights = need.weights;
        ui.state = est.state;
        ui.bufs = bufs;
        ui.st1 = st1;
        ui.pend1 = pend1;
        ui.drafts = am->sizing.mtpDefaults(cfg.moe, device::k_uma_estimate_arch).drafts;
        ui.parallel = opt.parallel;
        ui.n_ck = n_ck;
        ui.n_spe = n_spe_plan;
        ui.has_mtp = est.has_mtp;
        ui.n_embd = cfg.n_embd;
        ui.n_vocab = cfg.n_vocab;
        const startfit::UmaFixed uf = startfit::umaFixed(ui);
        const std::uint64_t slots_b = uf.slots;
        const std::uint64_t ck_b = uf.ck;
        need.fixed = uf.fixed;
        need.kv_per_token = est.kvPerToken(kv_est);
        need.min_tokens = opt.ctx ? *opt.ctx : slot_ctx;
        need.want_tokens = pool_req ? *pool_req : slot_ctx;
        const double g = 1024.0 * 1024.0 * 1024.0;
        if (std::string why = hip::umaRefusal(uma, need, info.name); !why.empty()) {
            if (why.back() == '\n') why.pop_back();
            throw app::UserError(app::exit_vram, why);
        }
        logI("UMA carve-out check (estimate): weights {:.2f} + state and buffers {:.2f} + slots {:.2f} + checkpoints {:.2f} GiB "
             "(+8% slack) = {:.2f} GiB, KV {:.1f} KiB/token: {:.1f} GiB for {} tokens, {:.1f} GiB with the {:.1f} GiB reserve, "
             "of {:.1f} GiB dedicated",
             need.weights / g, (est.state + bufs) / g, slots_b / g, ck_b / g, need.fixed / g, need.kv_per_token / 1024.0,
             need.at(need.want_tokens) / g - hip::uma_reserve / g, need.want_tokens, need.at(need.want_tokens) / g,
             hip::uma_reserve / g, uma.dedicated / g);
        if (const std::optional<std::uint32_t> cap = hip::umaPoolCap(uma, need, kv_page); cap && !opt.ctx) {
            logW("the {:.1f} GiB GPU carve-out holds a KV pool of about {} tokens, not the default {}: pool reduced (a {} GB "
                 "carve-out holds the full pool; {})",
                 uma.dedicated / g, *cap, need.want_tokens, hip::umaGbFor(need.at(need.want_tokens)),
                 "raise the GPU memory / UMA frame buffer size in Armoury Crate or the BIOS");
            pool_req = *cap;
        }
    }
    b.mem0 = ops->memInfo();
    std::optional<std::size_t> exec_req;  // unset = the device default (prefillExecDefault, after the load)
    if (settings::envGet("PREFILL_CHUNK")) {
        exec_req = settings::prefillChunk(nullptr);  // (arch025-N15: the shared number rule; malformed = unset)
        logI("prefill forward up to {} rows requested (WHIRL_PREFILL_CHUNK)", exec_req.value_or(0));
    }
    std::unique_ptr<qwen35::Model>& mp = b.mp = qwen35::Model::load(f, 0, stats, lo);
    qwen35::Model& model = *mp;
    // whole schedule chunks that fit the prefill batch (a forward never exceeds the batch:
    // Engine::chunkLen caps it); default 2048 rows, gfx1151 MoE 4096 (SRV-PP; dense 2048, SRV-PP-DENSE)
    const device::Profile& prof = device::profile(model.arch);
    b.prefill_exec = sched::prefillExec(exec_req, prof, model.cfg.moe, lo.max_batch);  // (the CLI's formula too, arch025-N15c)
    {
        std::string tl;
        qwen35::loadOrTune(model, opt.path, tl);
        std::size_t p = 0;
        while (p < tl.size()) {
            std::size_t e = tl.find('\n', p);
            if (e == std::string::npos) e = tl.size();
            const std::string_view l = trimWs(std::string_view(tl).substr(p, e - p));
            if (!l.empty()) logI("{}", l);
            p = e + 1;
        }
    }
    b.mem = ops->memInfo();
    logI("loaded {} tensors, {:.2f} GiB in {:.1f} s ({:.2f} GB/s)", stats.tensors,
         static_cast<double>(stats.bytes) / (1024.0 * 1024.0 * 1024.0), stats.ms / 1000.0,
         static_cast<double>(stats.bytes) / (stats.ms * 1e6));

    // pinned host memory shows up as this process's GPU "Shared Usage": declare it so monitoring
    // does not read the embedding table as VRAM spill (the RAM tier / vision add theirs below)
    const std::uint64_t& embd_pinned = b.embd_pinned =
        model.embd_host ? static_cast<std::uint64_t>(model.tok_embd.row_bytes) * model.tok_embd.nrows : 0;
    if (embd_pinned > 0) {
        tier::declarePinned(lad, embd_pinned);
        logI("token embedding: {:.0f} MiB in pinned host memory (counted as this process's GPU 'Shared Usage', declared; "
             "WHIRL_EMBD_HOST=0 keeps it in VRAM)",
             static_cast<double>(embd_pinned) / 1048576.0);
    }

    // --draft dflash: the DFlash2 drafter (WHIRL_DRAFT_MODEL) and the hidden capture of its target
    // layers (WHIRL_DFLASH_CAPTURE_ROWS rows, default the drafter's window or 2048)
    if (use_dflash && !model.mtp_skipped) {
        // the code object has no DFlash2 kernels: Model::load kept the MTP block
        logW("{} (missing: {})", whirl::dflash::choose(whirl::dflash::Source::dflash, dm_resolved, "", false).warning,
             model.dflashMissingKernel());
        use_dflash = false;
        dfile.reset();
    }
    std::vector<std::string>& dfl_warn = b.dfl_warn;
    const whirl::dflash::Params& dfl_par = b.dfl_par = whirl::dflash::parseParams([](const char* n) { return settings::envGet(n); }, dfl_warn);
    if (use_dflash)
        for (const std::string& w : dfl_warn) logW("{}", w);
    if (use_dflash) {
        const std::string dm = *dm_resolved;
        const hip::MemInfo d0 = ops->memInfo();
        const gguf::File& df = *dfile;
        qwen35::LoadStats ds;
        const std::uint64_t wb = model.loadDrafter(df, ds);
        const auto& dc = model.dflash->cfg;
        model.enableCapture(whirl::dflash::captureRows(dfl_par, dc.window, model.max_batch));
        const hip::MemInfo d1 = ops->memInfo();
        std::string tl;
        for (const std::uint32_t l : dc.target_layers) tl += (tl.empty() ? "" : ",") + std::to_string(l);
        logI("draft model {}: DFlash2, {} layers, block {}, window {}, target layers {}, weights {:.2f} GiB, capture {} rows ({:.0f} MiB), "
             "VRAM +{:.2f} GiB; MTP block not loaded",
             dm, dc.n_layer, dc.block, dc.window, tl, static_cast<double>(wb) / (1024.0 * 1024.0 * 1024.0), model.cap_rows,
             static_cast<double>(dc.target_layers.size()) * model.cap_rows * cfg.n_embd * 4 / (1024.0 * 1024.0),
             static_cast<double>(d0.free > d1.free ? d0.free - d1.free : 0) / (1024.0 * 1024.0 * 1024.0));
    }

    // same knobs as the CLI. arch025-N2b (R-5): numerics::applyFront(Front::server) on the model's plan, with the
    // server's parse rules (server_main.cpp:1040-1120 before N2b, moved in arch025-N2a): NO_FUSE, FLOAT_GEMV,
    // NAIVE_ATTN, GDN_SEQ, MOE_BN, DRAFT_WINDOW(_MIN), use_graph off, WIDE_VERIFY, HEAD_CHUNK, GDN_V0, ATTN_WIDE,
    // ATTN_DQ4, DQ4_DYN, ATTN_WS4. All before their first reader (wideCapable below); the CLI-only names are not sent.
    {
        namespace nu = whirl::numerics;
        nu::RuntimeValues rv;
        for (const auto& e : nu::RuntimeValues::names()) {
            const std::string_view n = e.name;
            if (n == "DBG" || n == "GEMV_MAX" || n == "NO_GRAPH" || n == "GV_GROUP" || n == "GV_NMAX") continue;  // CLI only
            rv.*e.field = settings::envGet(e.name);
        }
        hip::Function fv0 = nullptr;
        if (rv.gdn_v0) fv0 = model.module.getFunctionOpt("gdn_step_norm_v0");
        nu::applyFront(model.num_plan, nu::Front::server, rv, fv0 != nullptr);
        if (model.nf().gdn_v0) model.k.gdn_step_norm = fv0;
    }
    if (settings::envGet("PREFILL_BATCH"))  // (same floor as above: never below vram::min_prefill_batch; the CLI's rule too)
        model.max_batch = vram::prefillBatchCap(model.max_batch, settings::envU32("PREFILL_BATCH", model.max_batch, settingsWarn));
    // WHIRL_DRAFT_VOCAB: unset = embedded 64k subset (dense qwen35, 248320 vocab), 64k = embedded,
    // 48k / <file> = frequency subset file, N = the first N rows (old experiment), off = full head
    const qwen35::DraftVocabChoice& draft_vocab = b.draft_vocab = qwen35::draftVocabChoice(settings::draftVocab(), qwen35::exeDirectory());
    if (draft_vocab.kind == qwen35::DraftVocabChoice::Kind::first_n) model.draft_vocab = draft_vocab.n;
    // (applyFront set DRAFT_WINDOW / _MIN; envU32 here only logs a bad value at its old place, with the same default)
    (void)settings::envU32("DRAFT_WINDOW", model.nf().draft_window, settingsWarn);
    (void)settings::envU32("DRAFT_WINDOW_MIN", model.nf().draft_window_min, settingsWarn);
    if (settings::envGet("GEMV_R") || settings::envGet("GEMV_W") || settings::envGet("GEMV_WH"))
        logW("WHIRL_GEMV_R / WHIRL_GEMV_W / WHIRL_GEMV_WH are not supported by whirl-server (ignored)");
    const bool& use_mtp = b.use_mtp = model.mtp.has_value() && settings::mtp();
    const qwen35::MtpDefaults& def = b.def = qwen35::mtpDefaults(model.cfg.moe, model.arch);
    std::optional<std::uint32_t> drafts_env;
    if (auto v = settings::envGet("MTP_DRAFTS")) {
        std::uint32_t d = def.drafts;
        try {
            d = static_cast<std::uint32_t>(std::stoul(*v));
        } catch (const std::exception&) {
        }
        drafts_env = std::clamp(d, 1u, qwen35::max_drafts);
    }
    const std::optional<std::uint32_t> drafts_user = opt.mtp_drafts ? opt.mtp_drafts : drafts_env;
    const std::uint32_t& drafts = b.drafts = drafts_user.value_or(def.drafts);
    float& p_min = b.p_min = def.p_min;
    if (auto v = settings::envGet("MTP_PMIN")) {
        try {
            p_min = std::stof(*v);
        } catch (const std::exception&) {
        }
    }
    const std::uint32_t& n_min = b.n_min = settings::envU32("MTP_NMIN", 0, settingsWarn);
    const auto& adapt_env = b.adapt_env = settings::envGet("MTP_ADAPT");
    std::uint32_t& adapt = b.adapt = def.adapt;
    if (adapt_env) adapt = settings::parseMtpAdapt(adapt_env).adapt;  // (the rule shared with the CLI, arch025-N15)
    // an explicit draft count is used as is (fixed) unless WHIRL_MTP_ADAPT asks otherwise
    const bool& auto_mode = b.auto_mode = adapt_env ? (*adapt_env == "auto") : (def.automatic && !drafts_user);
    if (settings::envGet("HEAD_CHUNK")) (void)settings::envU32("HEAD_CHUNK", model.nf().head_chunk, settingsWarn);  // (bad-value warning only, as above)
    // the per-slot draft caps: spec.policy's default table and WHIRL_MTP_BATCH_DRAFTS rule (arch025-N10 leg 6)
    auto& batch_drafts = b.batch_drafts = qwen35::defaultBatchDrafts(model.cfg.moe);
    batch_drafts[1] = drafts;
    const auto& bd_env = b.bd_env = settings::envGet("MTP_BATCH_DRAFTS");
    if (bd_env) batch_drafts = qwen35::batchDraftsFrom(batch_drafts, *bd_env);
    // MTP block as Q4_K from Q6_K or Q8_0 (drafts only; WHIRL_MTP_Q4=0 keeps the file's types)
    if (use_mtp && settings::mtpQ4()) model.requantMtpQ4();
    if (use_mtp && !settings::envGet("MTP_FULLHEAD")) {
        const bool q4 = settings::draftHeadQ4();
        model.buildDraftHeadEx(q4 ? qwen35::Model::DraftHeadKind::q4 : qwen35::Model::DraftHeadKind::d2);
        applyDraftVocab(model, tok, draft_vocab);
    }
    if (use_mtp) {
        logI("MTP speculative decoding: on, up to {} draft(s) per cycle{}, p-min {:.2f}, n-min {}, draft count {}", drafts,
             (!model.fusedDecode() && drafts > 1) ? " (fused decode off: 1 used)" : "", p_min, n_min,
             auto_mode ? "auto (cost model)" : (adapt > 0 ? "adaptive margin" : "fixed"));
        std::string tbl;
        for (std::uint32_t k = 1; k <= opt.parallel; ++k) tbl += (k > 1 ? ", " : "") + std::to_string(batch_drafts[k]);
        logI("MTP under load: max drafts per cycle with 1..{} decoding slots: {{ {} }}", opt.parallel, tbl);
    } else if (use_dflash) {
        logI("MTP speculative decoding: off (--draft dflash)");
    } else {
        logI("MTP speculative decoding: off{}", !model.mtp ? " (no nextn layer)" : " (WHIRL_MTP=0)");
    }
}

// local struct of serveMain before arch025-C1b (the tiers' IO thread is joined before the engine is destroyed)
struct TierIoStop {
    std::unique_ptr<tier::Tier>& a;
    std::unique_ptr<tier::Tier>& b;
    ~TierIoStop() {
        if (a) a->stopIo();
        if (b) b->stopIo();
    }
};

// arch025-C1b: serveMain after buildModel, as the phases of one object (method object; llama.cpp keeps its server
// state in server_context_impl the same way). serveMain calls the phases in the old order; each phase is the old
// text, so every call, log line, env read and VRAM / pinned measurement (mem_ck0 / mem_ck1, the pinned room,
// mem_t1, wddm0 / pinned0) happens at the same moment relative to every allocation as before. The members are the
// old locals that more than one phase uses, in their old declaration order (destroyed in the old order: the
// vision encoder, then the tier IO stop, the engine, the tiers, the backend, then buildModel's results); a local
// used by one phase stays in it.
class ServerRun {
public:
    ServerRun(Options& opt_, const numerics::Request& nreq_, const char* lad_, LoadedModel& lm_, bool hip_cache_user_,
              bool hip_cache_set_)
        : opt(opt_), nreq(nreq_), lad(lad_), lm(lm_), model(*lm_.mp), tok(*lm_.tok), hip_cache_user(hip_cache_user_),
          hip_cache_set(hip_cache_set_) {
        // arch025-surv (mutation survivor c1b-tierio-order): members are destroyed in reverse declaration order, so
        // tier_io_stop (declared after the engine) joins the tier IO thread before ~Engine runs (its wake callback calls
        // into the engine, FR-CACHE P3-5), and the tiers outlive the engine (bug-server-exit-av). Only a graceful
        // shutdown with a busy tier would show a wrong order, so the order is checked here at compile time.
        static_assert(offsetof(ServerRun, tier_io_stop) > offsetof(ServerRun, engine), "tier_io_stop must be declared after engine");
        static_assert(offsetof(ServerRun, engine) > offsetof(ServerRun, tier_owned) && offsetof(ServerRun, engine) > offsetof(ServerRun, tier_pre), "the tiers (tier_pre, tier_owned) must be declared before engine");
    }
    void planSlots();  // the slot re-check after loading (--parallel lowered to what fits), then startup::setupSlots
    void makeEngine();  // the engine options, the backend, the Engine, and the checkpoint VRAM around allocState (mem_ck0 / mem_ck1)
    void reserveRamTier();  // the RAM tier's size (pinned room measured now) and its pinned allocation, before the KV pool is sized
    void loadVisionEarly();  // discrete GPU with --mmproj: the vision encoder read before the KV pool (its reserve)
    void planPool();  // mem_t1, the WDDM budget / pinned0 read, the KV pool size, startup::allocKvPool and the budget re-check
    void reportPool();  // the pool / slot / VRAM lines, the runtime watch baseline, initPool and the WDDM memory line
    void startTier();  // the host tiers (RAM tier start, SSD tier, fingerprint)
    void attachVision();  // the vision encoder attached to the engine (integrated GPU: loaded here, after the KV pool)
    int runServer();  // model ready, timer resolution, the VRAM probe, the memory governor, the startup summary, HTTP, the loop

private:
    Options& opt;
    const numerics::Request& nreq;
    const char* lad;
    LoadedModel& lm;  // buildModel's results
    qwen35::Model& model;
    const Tokenizer& tok;
    const bool hip_cache_user;  // GPU_RESOURCE_CACHE_SIZE set by the user
    const bool hip_cache_set;   // WHIRL set GPU_RESOURCE_CACHE_SIZE=256

    hip::MemInfo mem_s0;
    std::optional<startup::SlotSetup> slots;  // startup::SlotSetup has a private default constructor (only setupSlots makes one)
    std::uint32_t lit_snap_sets = 0;          // engine: snapshot sets for n-gram / DFlash2 drafts (0 = all)
    std::uint64_t conv_bytes = 0;
    std::uint64_t ssm_bytes = 0;
    std::uint64_t n_gdn = 0;
    std::uint32_t V = 0;
    std::uint64_t seed0 = 0;
    std::string name;
    EngineOptions eo;
    std::unique_ptr<ServerModel> backend;
    // bug-server-exit-av: the host tiers are declared before the engine, so the engine is destroyed first: its
    // slot / shared-prefix spill fences are recorded on the tier's stream, and HIP reads that stream through the
    // event when ~Slot / ~Spe query it (hipEventQuery -> checkEventCaptureRestrictions; a destroyed stream there
    // was the access violation at amdhip64_7.dll +0x8f725 after "server stopped"). llama.cpp's order: events
    // before streams (~ggml_backend_cuda_context). tier_io_stop (declared after the engine, so destroyed before
    // it) joins the tier IO thread first, whose wake callback calls into the engine (why the tier used to go first,
    // FR-CACHE P3-5); it runs on every exit path, also exception unwinding.
    std::unique_ptr<tier::Tier> tier_pre;
    std::unique_ptr<tier::Tier> tier_owned;
    std::optional<Engine> engine;  // constructed in makeEngine (where serveMain constructed it)
    TierIoStop tier_io_stop{tier_pre, tier_owned};
    hip::MemInfo mem_ck0;
    hip::MemInfo mem_ck1;
    std::uint64_t kv_ram_mb = 0;
    double tier_alloc_s = 0.0;
    std::optional<std::string> mmp_path;
    std::unique_ptr<vision::Vision> vis_store;
    std::optional<hip::LocalMemTaken> vis_load_taken;
    double vis_load_s = 0;
    std::uint64_t vis_reserve = 0;
    std::string vis_reserve_note;
    hip::MemInfo mem_t1;
    bool reserve_explicit = false;
    hip::WddmMemInfo wddm0;
    std::uint64_t pinned0 = 0;  // pinned host memory inside wddm0's non-local usage
    std::uint64_t budget_margin = 0;
    std::string kv_note;
    std::uint32_t slot_ctx_eff = 0;
    std::uint64_t per_tok = 0;
    wddmwatch::Sample vram_base;
    wddmwatch::Config vram_cfg;
    bool gov_on = false;
    bool startup_spill = false;
    std::unique_ptr<ServerVision> vis_srv;
};

void ServerRun::planSlots() {
    // slots: KV-cache regions, recurrent state, MTP hidden rows
    model.gdn_replay = settings::envFlag("GDN_REPLAY", true);
    // REVIEW-0221 C3: the pre-load estimate can be off (it lowered --parallel 16 -> 15 for a 16384-token request and the
    // real pool held 13312). With the weights and buffers loaded, measure the VRAM usable now (same rule as the pool
    // sizing below) and lower --parallel again, one slot at a time, until one request of min(ctx-per-slot, 16384)
    // tokens fits next to every slot's state, replay rows and checkpoints (+4% WDDM overhead, + shared checkpoints,
    // snapshot sets and 256 MiB slack). Each step is logged.
    if (!lm.is_uma && opt.parallel_given && opt.parallel > 1 && !model.kv_choice.debug) {
        const hip::WddmMemInfo w = lm.ops->wddmMemInfo();
        const hip::MemInfo mf = lm.ops->memInfo();
        poolbudget::RoomIn ri;
        ri.reserve_explicit = settings::poolReserveGiven();
        ri.reserve = static_cast<std::uint64_t>(settings::poolReserveMb(poolbudget::defaultReserveMb(lm.cfg.moe, w.ok), settingsWarn)) << 20;
        ri.margin = lm.budgetMargin(w.local_budget).bytes;
        ri.free = mf.free;
        ri.wddm_ok = w.ok;
        ri.local_budget = w.local_budget;
        ri.local_usage = w.local_usage;
        ri.headroom = lm.card.headroom;
        const std::uint64_t av = poolbudget::kvRoom(ri).bytes;
        std::uint64_t ng = 0;
        for (std::uint32_t li = 0; li < lm.cfg.n_layer; ++li)
            if (!lm.cfg.isAttn(li)) ++ng;
        const std::uint64_t st = ng * (model.convBytes() + model.ssmBytes());
        startfit::SlotFixedIn sfi;
        sfi.state = st;
        sfi.replay = whirl::model::slotStateBytes(lm.am->sizing, lm.cfg).pend1;
        sfi.n_ck = lm.n_ck;
        sfi.ck = settings::envFlag("CKPT_HOST", false) ? 0 : st + static_cast<std::uint64_t>(lm.cfg.n_vocab) * 4 + static_cast<std::uint64_t>(lm.cfg.n_embd) * 4;
        // v0.2.2.2: + the DFlash2 context ring per slot (dflashSetup runs after setupSeqs, below; REVIEW-0221-2)
        sfi.dflash_slot = lm.use_dflash ? model.dflashSlotBytes() : 0;
        sfi.n_spe = lm.n_spe_plan;
        sfi.snap_sets = model.snap_sets;
        const startfit::SlotFixed sf = startfit::slotFixed(sfi);
        const std::uint64_t per_slot = sf.per_slot;
        const std::uint64_t shared = sf.shared;
        const std::uint64_t target = std::min<std::uint32_t>(lm.slot_ctx, vram_tight_pool);
        const std::uint64_t per_tok_est = model.kvBytesPerToken();
        const auto poolFor = [&](std::uint32_t n) { return startfit::poolTokensWith(n, av, per_slot, shared, per_tok_est); };
        for (std::uint32_t p = opt.parallel; p > 1 && poolFor(p) < target; --p)
            lm.adjust(std::format("--parallel {} -> {}", p, p - 1),
                   std::format("measured after loading the weights: with {} slots the KV pool would hold about {} tokens, less than one "
                               "{}-token request ({:.2f} GiB usable, {:.0f} MiB per slot)",
                               p, poolFor(p), target, av / 1073741824.0, per_slot / 1048576.0));
        lm.sin.put("slots.parallel", opt.parallel);
        lm.sin.put("slots.avail", av);
        lm.sin.put("slots.per_slot", per_slot);
        lm.sin.put("slots.shared", shared);
        lm.sin.put("slots.per_token", per_tok_est);
        lm.sin.put("slots.target", target);
        opt.parallel = startfit::slotsThatFit(opt.parallel, av, per_slot, shared, per_tok_est, target);
    }
    mem_s0 = lm.ops->memInfo();
    // the slots, the DFlash2 rings / scratch, then the DeltaNet snapshot sets, before the KV pool is sized from the
    // VRAM left (startup::setupSlots, arch025-C0: the server tests run the same code; allocKvPool takes its result)
    startup::SlotSetupIn ssi;
    ssi.parallel = opt.parallel;
    ssi.slot_ctx = lm.slot_ctx;
    ssi.n_layer = lm.cfg.n_layer;
    ssi.use_mtp = lm.use_mtp;
    ssi.use_dflash = lm.use_dflash;
    ssi.drafts = lm.drafts;
    ssi.user_batch_drafts = lm.bd_env.has_value();
    ssi.dfl_par = lm.dfl_par;
    ssi.ngram = settings::envFlag("NGRAM", true);
    ssi.snap_sets_env = settings::envGet("SNAP_SETS");
    slots.emplace(startup::setupSlots(model, *lm.ops, ssi, lm.batch_drafts));
    lm.batch_drafts = slots->batch_drafts;
    lit_snap_sets = slots->lit_snap_sets;  // engine: snapshot sets for n-gram / DFlash2 drafts (0 = all)
}

void ServerRun::makeEngine() {
    // engine (checkpoint buffers, shared checkpoints, sampling buffers)
    conv_bytes = model.convBytes();
    ssm_bytes = model.ssmBytes();
    n_gdn = 0;
    for (std::uint32_t li = 0; li < lm.cfg.n_layer; ++li)
        if (model.ssm_state[li] != 0) ++n_gdn;
    V = lm.cfg.n_vocab;
    std::random_device rd;
    seed0 = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();

    const std::string base = narrow(std::filesystem::path(widen(opt.path)).filename().wstring());
    name = opt.alias ? *opt.alias : base;
    if (!opt.alias && name.size() > 5) {
        std::string tail = name.substr(name.size() - 5);
        for (char& c : tail) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (tail == ".gguf") name = name.substr(0, name.size() - 5);
    }
    eo.model_name = name;
    eo.model_file = base;
    eo.ctx = lm.slot_ctx;
    eo.parallel = opt.parallel;
    eo.tmpl = lm.tmpl;
    eo.use_mtp = lm.use_mtp;
    eo.spec_sample = model.num_plan.on(numerics::Item::specsample);
    eo.relax = model.relax();
    eo.use_dflash = lm.use_dflash;
    eo.dfl_p_min = lm.dfl_par.p_min;
    eo.dfl_n_min = lm.dfl_par.n_min;
    eo.dfl_warm = lm.dfl_par.warm;
    if (lm.use_dflash && eo.relax.on) {  // whirl::dflash::relaxApplies: MTP drafts only
        // relaxacc relaxes MTP drafts only; DFlash2 and n-gram drafts keep exact acceptance, so
        // the fast mode's other items (4-bit KV, ...) stay and the decode equals greedy of them
        logI("relaxacc: not used with --draft dflash (DFlash2 / n-gram drafts keep exact acceptance)");
        eo.relax.on = false;
        // /props and the tier fingerprint must not report relaxacc as active
        numerics::skipItem(model.num_plan, numerics::Item::relaxacc, "--draft dflash: DFlash2 / n-gram drafts keep exact acceptance");
    }
    eo.n_draft = lm.drafts;
    eo.p_min = lm.p_min;
    eo.n_min = lm.n_min;
    eo.mtp_adapt = lm.adapt;
    eo.mtp_auto = lm.auto_mode;
    eo.batch_drafts = lm.batch_drafts;
    eo.ng_snap_sets = lit_snap_sets;
    eo.no_prefix_cache = lm.no_pc;
    eo.verify_cache = settings::envGet("PREFIX_CACHE_VERIFY").has_value();
    // COLD-RESUME: --precise resumes only from cold-exact checkpoints (output bitwise equal to a cold prefill)
    eo.cold_resume = settings::envFlag("COLD_RESUME", nreq.mode == numerics::Mode::precise);
    eo.seg_prefill = settings::envFlag("SEG_PREFILL", true);
    eo.timing_reset = settings::envFlag("TIMING_RESET", true);
    eo.ngram = settings::envFlag("NGRAM", true);
    eo.ngram_min = settings::envU32("NGRAM_MIN", lm.def.ngram_min, settingsWarn);
    eo.ngram_slope = lm.def.ngram_slope;
    if (auto v = settings::envGet("NGRAM_SLOPE")) eo.ngram_slope = std::strtof(v->c_str(), nullptr);
    eo.ngram_max = settings::envU32("NGRAM_MAX", 0, settingsWarn);
    eo.ngram_force = settings::envFlag("NGRAM_FORCE", false);
    // WHIRL_SLOT_DRAFTS: 0 / unset uniform, 1 marginal-gain swaps, 2 cost-model allocation
    if (auto v = settings::envGet("SLOT_DRAFTS")) eo.slot_drafts = *v == "2" ? 2u : (*v != "0" ? 1u : 0u);
    if (lm.use_dflash) {
        // MTP-only settings: the per-slot MTP draft allocation, the draft-head vocabulary subset and
        // the MTP draft attention window have nothing to act on without the MTP block
        if (eo.slot_drafts != 0) logW("WHIRL_SLOT_DRAFTS ignored with --draft dflash (it allocates MTP drafts)");
        eo.slot_drafts = 0;
        if (settings::envGet("DRAFT_VOCAB") || settings::envGet("DRAFT_WINDOW") || settings::envGet("DRAFT_WINDOW_MIN"))
            logI("WHIRL_DRAFT_VOCAB / WHIRL_DRAFT_WINDOW not used with --draft dflash (MTP draft head only)");
    }
    eo.trace_nd = settings::envGet("TRACE_ND").has_value();
    eo.loop_log = settings::envGet("LOOP_LOG").has_value();
    eo.tier_verify = settings::envGet("TIER_VERIFY").has_value();
    eo.tier_min_gain = settings::envU32("TIER_MIN_GAIN", 512, settingsWarn);
    if (auto v = settings::envGet("GATHER_MS")) {
        try {
            eo.gather_ms = std::stod(*v);
        } catch (const std::exception&) {
        }
    }
    eo.decode_min_tps = opt.decode_min_tps ? *opt.decode_min_tps : settings::envU32("DECODE_MIN_TPS", decode_min_tps_default, settingsWarn);
    if (eo.decode_min_tps > 0)
        logI("decode floor: >= {:.0f} tok/s per decoding request while others prefill (--decode-min-tps)", eo.decode_min_tps);
    else
        logI("decode floor: off (--decode-min-tps 0: prefill forwards are not limited)");
    if (auto v = settings::envGet("PROFILE")) eo.profile = *v == "2" ? 2 : 1;
    eo.sys_min = settings::sysMin(sys_min_default, settingsWarn);
    eo.kv_supersede = settings::envU32("KV_SUPERSEDE", 1, settingsWarn) != 0;
    // TIER-IDLE: hidden test override of the 300 s idle write delay (no CLI flag; 0: off)
    if (settings::envGet("KV_IDLE_FLUSH_MS")) eo.kv_idle_flush_ms = settings::envU32("KV_IDLE_FLUSH_MS", 0, settingsWarn);
    eo.lcp_on = settings::envFlag("SYS_LCP", true);
    eo.n_ck = lm.n_ck;
    eo.tail_ck = settings::tailCk();  // debug knob: WHIRL_TAIL_CK=0 turns the tail checkpoint off
    eo.n_spe = (lm.n_ck == 0 || lm.no_pc) ? 0 : lm.n_spe_plan;
    eo.ckpt_host = settings::envFlag("CKPT_HOST", false);
    eo.prefill_exec = lm.prefill_exec;
    eo.seed = seed0;

    backend = makeQwen35Backend(model);
    engine.emplace(*backend, *lm.ops, tok, eo);
    // the checkpoint VRAM is measured around allocState only (F-7: Engine::allocStateMeasured, CPU-tested order)
    const Engine::StateMem mem_ck = engine->allocStateMeasured();
    mem_ck0 = mem_ck.before;
    mem_ck1 = mem_ck.after;
}

void ServerRun::reserveRamTier() {
    // host tiers, VRAM side first (tier stream, pinned arena mapped for the
    // GPU): the KV pool below is sized after it
    std::uint64_t entry_mb = 0;  // one max-context entry, whole GiB (floor of the integrated-GPU rule)
    const std::uint64_t ram_def = [&] {
        const std::uint64_t ck_stride = alignUp(n_gdn * (conv_bytes + ssm_bytes) + static_cast<std::uint64_t>(lm.cfg.n_embd) * 4 +
                                                    static_cast<std::uint64_t>(V) * 4,
                                                tier::block_bytes);
        const std::uint64_t full = static_cast<std::uint64_t>(lm.slot_ctx) * model.kvBytesPerTokenFmt(KvFormat::f16) + lm.n_ck * ck_stride;
        entry_mb = alignUp(full, 1ull << 30) >> 20;
        return std::max(tier::ram_auto_min_mb, entry_mb);
    }();
    // size: explicit (--kv-ram-mb / WHIRL_KV_RAM_MB), else 1/4 of physical RAM
    // within [ram_def, 32 GiB] and at most half of the RAM available now;
    // integrated GPU: OS-visible RAM / 8 within [one max-context entry, 16 GiB], at most
    // 1/4 of the RAM available now, off under 32 GB. The KV pool there is the BIOS
    // carve-out (separate from the OS RAM on Windows); the tier is extra pinned OS RAM,
    // and a restore from it (and from the SSD tier behind it) replaces a prefill
    tier::RamTierInput rin;
    rin.floor_mb = ram_def;
    rin.entry_mb = entry_mb;
    rin.integrated = lm.info.integrated;
    if (opt.kv_ram_mb) {
        rin.explicit_mb = *opt.kv_ram_mb;
    } else if (auto v = settings::envGet("KV_RAM_MB")) {
        try {
            rin.explicit_mb = std::stoull(*v);
            rin.explicit_src = "WHIRL_KV_RAM_MB";
        } catch (const std::exception&) {
            logW("kv tier: WHIRL_KV_RAM_MB=\"{}\" is not a number; using the automatic size", *v);
        }
    }
    {
        std::uint64_t total = 0, avail = 0;
        if (platform::sys::memoryStatus(&total, &avail)) {
            rin.total_phys = total;
            rin.avail_phys = avail;
        }
    }
    const tier::RamTierSize rsz = tier::ramTierSize(rin);
    kv_ram_mb = rsz.mb;
    // 1. RAM tier: capped at 20480 MiB (a larger one crashed on the first request on the R9700 test box, cause not known
    // yet), then an explicit size above half of the RAM available now is lowered to half (v0.2.2.2: cap first, so asking
    // for more never gives less). The automatic size the user never chose is capped with an info line, not an adjustment.
    const std::uint64_t ram_auto_mb = [&] {
        tier::RamTierInput a = rin;
        a.explicit_mb.reset();
        return tier::ramTierSize(a).mb;
    }();
    // s8-719 / kvram719: pinned host memory the HIP GPUs can keep resident (startfit::pinnedRoomMb), measured now (the
    // embedding is already pinned)
    std::uint64_t pinned_room = startfit::pinned_room_unknown;
    std::string pinned_why;
    if (kv_ram_mb > 0 && lm.n_ck > 0 && !lm.no_pc) {
        const hip::GpuPinned gp = hip::gpuPinnedNow();
        startfit::PinnedState pst;
        pst.total_phys = rin.total_phys;
        pst.pinned_max = gp.ok ? gp.max_shared : 0;
        std::string per;
        for (const hip::GpuPinnedDev& g : gp.devs) {
            const std::uint64_t own = std::max(g.nonlocal_usage, g.own_shared);  // iGPU: DXGI says 0 (REVIEW C4)
            pst.gpus.push_back({g.nonlocal_budget, own, g.shared_system, g.adapter_shared});
            per += std::format("{}{} MiB {} (this process {} MiB, all programs {})", per.empty() ? "" : ", ",
                               (g.nonlocal_budget > 0 ? g.nonlocal_budget : g.shared_system) >> 20,
                               g.nonlocal_budget > 0 ? "non-local budget" : "shared GPU memory", own >> 20,
                               g.adapter_shared ? std::format("{} MiB", g.adapter_shared >> 20) : std::string("unknown"));
        }
        // REVIEW-S8-719-3 N3-4: the vision encoder's weights are pinned after this (Vision::load, discrete GPU below, the
        // integrated GPU later): count them now, from the mmproj file size plus its padding (startfit::mmprojPinnedBytes;
        // the file size alone is below what is pinned, REVIEW-B0 N1)
        if (const std::optional<std::string> mm = opt.mmproj ? opt.mmproj : settings::envGet("MMPROJ")) {
            std::error_code ec;
            const std::uintmax_t sz = std::filesystem::file_size(*mm, ec);
            if (!ec) pst.later = startfit::mmprojPinnedBytes(static_cast<std::uint64_t>(sz));
        }
        pinned_room = startfit::pinnedRoomMb(pst);
        lm.sin.put("pin.measured", true);
        lm.sin.put("pin.total_phys", pst.total_phys);
        lm.sin.put("pin.pinned_max", pst.pinned_max);
        lm.sin.put("pin.later", pst.later);
        lm.sin.put("pin.gpus", pst.gpus.size());
        for (std::size_t gi = 0; gi < pst.gpus.size(); ++gi) {
            const std::string g = std::format("pin.g{}", gi);
            lm.sin.put(g + ".nonlocal_budget", pst.gpus[gi].nonlocal_budget);
            lm.sin.put(g + ".own_nonlocal", pst.gpus[gi].own_nonlocal);
            lm.sin.put(g + ".shared_system", pst.gpus[gi].shared_system);
            lm.sin.put(g + ".adapter_shared", pst.gpus[gi].adapter_shared);
        }
        logI("kv tier: pinned memory now: {} HIP GPU(s) [{}]; all programs {} MiB (largest adapter) of half the RAM ({} MiB){} -> room for the RAM tier {} "
             "(reserves {} / {} MiB)",
             pst.gpus.size(), per, pst.pinned_max >> 20, pst.total_phys >> 21,
             pst.later ? std::format("; the vision encoder pins {} MiB more later", pst.later >> 20) : std::string(),
             pinned_room == startfit::pinned_room_unknown ? std::string("unknown") : std::format("{} MiB", pinned_room),
             startfit::pinned_reserve_mb, startfit::pinned_machine_reserve_mb);
        pinned_why = startfit::pinnedWhy(pinned_room == startfit::pinned_room_unknown ? 0 : pinned_room, pst.gpus.size());
    }
    lm.sin.put("ram.explicit_mb", rin.explicit_mb ? std::to_string(*rin.explicit_mb) : std::string("-"));
    lm.sin.put("ram.explicit_src", std::string_view(rin.explicit_src));
    lm.sin.put("ram.total_phys", rin.total_phys);
    lm.sin.put("ram.avail_phys", rin.avail_phys);
    lm.sin.put("ram.floor_mb", rin.floor_mb);
    lm.sin.put("ram.entry_mb", rin.entry_mb);
    lm.sin.put("ram.integrated", rin.integrated);
    lm.sin.put("ram.n_ck", lm.n_ck);
    lm.sin.put("ram.no_pc", lm.no_pc);
    std::optional<startfit::RamFit> rf =
        startfit::ramTierFit(kv_ram_mb, rin.avail_phys, ram_auto_mb, rin.explicit_mb.has_value(), pinned_room, entry_mb);
    if (lm.n_ck == 0 || lm.no_pc) rf.reset();
    // the line texts (startfit::ramTierText, unit-tested): REVIEW-S8-719 N3, REVIEW-S8-719-3 N3-2 / N3-6
    const startfit::RamTierText ram_text = startfit::ramTierText(rf, kv_ram_mb, rin.explicit_mb.has_value(), rin.explicit_src,
                                                                 rsz.reason, pinned_why, entry_mb, rin.avail_phys >> 20);
    const std::string& ram_off_reason = ram_text.off_reason;
    if (rf) {
        if (!rin.explicit_mb) {
            if (rf->pinned)
                logW("kv tier: RAM tier {} MiB (the automatic size would be {} MiB): {}", rf->mb, kv_ram_mb, ram_text.pinned_why);
            else
                logI("kv tier: RAM tier capped at {} MiB (the automatic size would be {} MiB) while a crash with larger RAM "
                     "tiers is investigated",
                     rf->mb, kv_ram_mb);
        } else {
            lm.adjust(std::format("{} {} -> {} MiB", std::string(rin.explicit_src), kv_ram_mb, rf->mb), ram_text.adjusted_why);
        }
        kv_ram_mb = rf->mb;
    }
    tier_alloc_s = 0.0;
    if (kv_ram_mb > 0 && lm.n_ck > 0 && !lm.no_pc) {
        if (rsz.avail_limited) logW("kv tier: RAM tier size {} MiB ({})", kv_ram_mb, ram_text.size_reason);
        else logI("kv tier: RAM tier size {} MiB ({})", kv_ram_mb, ram_text.size_reason);
        const TimePoint t_alloc0 = Clock::now();
        try {
            tier_pre = tier::Tier::create(*lm.ops, kv_ram_mb << 20, seed0 ^ 0x5bd1e995ull, lm.dev);
        } catch (const std::exception& ex) {
            logE("kv tier: cannot allocate {} MiB of pinned host memory ({}); running without host tiers", kv_ram_mb, ex.what());
        }
        tier_alloc_s = msSince(t_alloc0) / 1000.0;
    } else if (kv_ram_mb == 0 && lm.n_ck > 0 && !lm.no_pc) {
        logI("kv tier: RAM tier off ({})", ram_off_reason);
    }
}

void ServerRun::loadVisionEarly() {
    // ---- vision encoder, discrete GPU: read the mmproj now (pinned host memory only, no VRAM) so the KV pool below
    // leaves room for what the encoder adds after startup (REVIEW-GOV-3 question 4: on the R9700 the first image added
    // 873 MiB on Ornith and a 1920x1080 image 134 MiB more, leaving 30 MiB before the governor's trigger). Text-only
    // runs and the integrated GPU (where a pinned allocation lowers hipMemGetInfo free) keep the old order and pool.
    mmp_path = opt.mmproj ? opt.mmproj : settings::envGet("MMPROJ");
    vis_load_s = 0;
    vis_reserve = 0;
    if (mmp_path && !lm.is_uma) {
        const TimePoint tl0 = Clock::now();
        const hip::LocalMemSnap ml0 = lm.ops->localMemSnap();
        try {
            vis_store = vision::Vision::load(*mmp_path);
        } catch (const std::exception& ex) {
            logE("vision: cannot load mmproj {}: {}", *mmp_path, ex.what());
            throw;
        }
        vis_load_taken = hip::localMemTaken(ml0, lm.ops->localMemSnap());
        vis_load_s = msSince(tl0) / 1000.0;
        vision::Mode vm = *vision::parseMode(opt.vis_mode);
        if (auto x = settings::envGet("VIS_MODE")) vm = vision::parseMode(*x).value_or(vm);
        const bool resident = vm != vision::Mode::stream;  // auto: resident when it fits, which the reserve makes room for
        std::array<std::array<std::uint64_t, 2>, 16> raw{};
        const std::size_t n_lend = backend->lendScratch(raw);
        std::array<std::uint64_t, 16> lens{};
        for (std::size_t k = 0; k < n_lend; ++k) lens[k] = raw[k][1];
        poolbudget::VisionVram vv;
        vv.weights = resident ? vis_store->totalBytes() : 0;
        vv.act_extra = vis_store->extraBytes(vis_store->hp.max_tokens, resident, std::span<const std::uint64_t>(lens.data(), n_lend));
        vis_reserve = poolbudget::visionReserve(vv);
        vis_reserve_note = std::format("{} MiB for the vision encoder (--mmproj: weights {} MiB{}, activations of a {}-token image "
                                       "beyond the prefill scratch {} MiB, fixed {} MiB)",
                                       vis_reserve >> 20, vv.weights >> 20, resident ? "" : " (stream mode: none)",
                                       vis_store->hp.max_tokens, vv.act_extra >> 20, poolbudget::vision_fixed_mb);
    }
}

void ServerRun::planPool() {
    mem_t1 = lm.ops->memInfo();
    // shared KV pool: --ctx, or (R9700) the VRAM left now minus a reserve
    // The reserve is kept below both the free VRAM and this process's WDDM budget: going over the
    // budget makes Windows demote allocations to system memory (seen as GPU 'Shared Usage'), and a
    // demoted KV array is read across PCIe on every step (FIX-1: ~128k agent decode 29 -> 68 tok/s).
    // WHIRL_POOL_RESERVE_MB keeps the old free-VRAM-only rule with that reserve.
    reserve_explicit = settings::poolReserveGiven();
    wddm0 = lm.ops->wddmMemInfo();
    pinned0 = lm.ops->pinnedHostBytes();  // pinned host memory inside wddm0's non-local usage
    poolbudget::RoomIn ri;
    ri.reserve_explicit = reserve_explicit;
    ri.reserve = static_cast<std::uint64_t>(settings::poolReserveMb(poolbudget::defaultReserveMb(lm.cfg.moe, wddm0.ok), settingsWarn)) << 20;
    const poolbudget::Margin margin0 = lm.budgetMargin(wddm0.local_budget);
    budget_margin = margin0.bytes;
    ri.margin = budget_margin;
    ri.free = lm.ops->memInfo().free;
    ri.wddm_ok = wddm0.ok;
    ri.local_budget = wddm0.local_budget;
    ri.local_usage = wddm0.local_usage;
    ri.headroom = lm.card.headroom;
    // --mmproj (discrete GPU, not WHIRL_POOL_RESERVE_MB): room for the vision encoder; text only: vis_reserve 0, unchanged
    ri.vis_reserve = vis_reserve;
    const poolbudget::Room room0 = poolbudget::kvRoom(ri);
    const std::uint64_t avail = room0.bytes;
    const std::string budget_note = poolbudget::kvRoomNote(ri, room0, margin0.why, vis_reserve_note);
    if (!lm.pool_req) logI("kv pool sizing: {}", budget_note);
    // KV format: fixed at load by the mode and model type (numerics::chooseKv, as the CLI).
    // A pool / context that does not fit in that format shrinks when not given explicitly,
    // else the server refuses to start; never a lower-precision format. WHIRL_KV (debug
    // override): the pool is sized from memory as before.
    kv_note = std::format(" ({})", model.kv_choice.why);
    const bool kv_fixed = !model.kv_choice.debug;
    slot_ctx_eff = lm.slot_ctx;
    std::optional<std::uint32_t> fit_pool;
    if (kv_fixed) {
        // v0.2.2.1: explicit -c / --ctx-per-slot that do not fit are lowered (adjusted: lines), not refused; the UMA
        // iGPU keeps its carve-out rule (pool_req there is the 262144 default or -c, checked by umaRefusal above)
        const bool fit_explicit = !lm.is_uma;
        {
            const numerics::KvFitLabel lb = numerics::kvFitLabel(nreq, lm.cfg.moe, model.kv_choice.kv);
            lm.sin.put("pool.avail", avail);
            lm.sin.put("pool.per_token", model.kvBytesPerToken());
            lm.sin.put("pool.req", lm.pool_req ? std::to_string(*lm.pool_req) : std::string("-"));
            lm.sin.put("pool.pool_explicit", opt.ctx.has_value() && !fit_explicit);
            lm.sin.put("pool.slot_ctx", lm.slot_ctx);
            lm.sin.put("pool.slot_explicit", opt.ctx_per_slot.has_value() && !fit_explicit);
            lm.sin.put("pool.page", kv_page);
            lm.sin.put("pool.cap", 8 * lm.cap_max);
            lm.sin.put("pool.mode", std::string_view(lb.mode));
            lm.sin.put("pool.fmt", std::string_view(lb.fmt));
            lm.sin.put("pool.alt", std::string_view(lb.alt));
        }
        const numerics::PoolFit pf = numerics::serverPoolFit(avail, model.kvBytesPerToken(), lm.pool_req,
                                                             opt.ctx.has_value() && !fit_explicit, lm.slot_ctx,
                                                             opt.ctx_per_slot.has_value() && !fit_explicit, kv_page, 8 * lm.cap_max,
                                                             numerics::kvFitLabel(nreq, lm.cfg.moe, model.kv_choice.kv));
        if (pf.refuse) throw app::UserError(app::exit_vram, pf.msg);
        const std::string fit_why = std::format("{:.2f} GiB usable for the KV cache ({}), {:.1f} KiB per token in {}", avail / 1073741824.0,
                                                budget_note.empty() ? "free VRAM" : "WDDM budget minus the margin",
                                                model.kvBytesPerToken() / 1024.0, model.kvName());
        const bool adj_c = fit_explicit && opt.ctx && pf.pool < *opt.ctx;
        const bool adj_s = fit_explicit && opt.ctx_per_slot && pf.slot_ctx < lm.slot_ctx;
        if (adj_c) lm.adjust(std::format("-c {} -> {}", *opt.ctx, pf.pool), fit_why);
        if (adj_s) lm.adjust(std::format("--ctx-per-slot {} -> {}", lm.slot_ctx, pf.slot_ctx), std::format("the KV pool holds {} tokens", pf.pool));
        if (!pf.msg.empty() && !(adj_c || adj_s)) logW("{}", pf.msg);
        fit_pool = pf.pool;
        slot_ctx_eff = pf.slot_ctx;
    }
    per_tok = model.kvBytesPerToken();
    const std::uint32_t pool_tokens = fit_pool ? *fit_pool : lm.pool_req ? *lm.pool_req : [&] {
        const std::uint64_t t = std::min(avail / per_tok, static_cast<std::uint64_t>(8) * lm.cap_max);
        return static_cast<std::uint32_t>(t / kv_page * kv_page);
    }();
    if (pool_tokens < 4096)
        throw app::UserError(app::exit_vram,
                             std::format("not enough GPU memory left for the KV cache: a pool of {} tokens (at least 4096 needed).\n"
                                         "  Use a smaller model or quantization, give --ctx explicitly,\n"
                                         "  or close other programs that use the GPU.",
                                         pool_tokens));
    if (slot_ctx_eff != lm.slot_ctx) engine->setCtx(slot_ctx_eff);
    try {
        startup::allocKvPool(model, pool_tokens, *slots);
    } catch (const std::exception& ex) {
        logE("cannot allocate a KV pool of {} tokens ({:.2f} GiB): {}", pool_tokens,
             static_cast<double>(per_tok * pool_tokens) / (1024.0 * 1024.0 * 1024.0), ex.what());
        throw;
    }
    // issue #5: the WDDM budget read when the pool was sized can be lower once the pool exists (31.02 ->
    // 29.55 GiB on a user's R9700, ~220 ms later; why it fell is not known). Read it again now and shrink the
    // pool (whole pages, before anything uses it) until local usage + margin fits; up to 3 rounds. A round
    // only follows one whose free really lowered the local usage (WDDM counters can lag a free: re-polled until
    // two reads 50 ms apart agree, at most 1 s), so stale numbers cannot shrink it twice. WDDM numbers available,
    // not the UMA iGPU, no WHIRL_POOL_RESERVE_MB (never resized: a warning says how many tokens would fit). An
    // explicit -c or --ctx-per-slot that no longer fits is lowered with an "adjusted:" line (owner 2026-10-21).
    // Output is unchanged (only the pool's size changes).
    if (wddm0.ok && !lm.is_uma) {
        const auto poolGib = [&](std::uint64_t tokens) { return static_cast<double>(per_tok * tokens) / 1073741824.0; };
        if (!reserve_explicit) {
            std::uint64_t prev_usage = 0, prev_freed = 0;
            for (int round = 0; round < 3; ++round) {
                hip::WddmMemInfo wa = lm.ops->wddmMemInfo();
                if (!wa.ok) break;
                if (round > 0) {
                    // the WDDM usage follows a free over ~0.1-0.3 s (R9700: 2.67 GiB freed, 1.78 GiB seen at once, the
                    // rest later): wait until it has fallen and two reads 50 ms apart agree (at most 1 s), so the next
                    // round is computed from settled numbers and does not shrink the pool further than needed
                    std::uint64_t last = wa.local_usage;
                    for (int i = 0; i < 20 && wa.ok; ++i) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        wa = lm.ops->wddmMemInfo();
                        if (wa.ok && wa.local_usage == last && poolbudget::usageFell(prev_usage, wa.local_usage, prev_freed)) break;
                        last = wa.local_usage;
                    }
                    if (!wa.ok) break;
                    if (!poolbudget::usageFell(prev_usage, wa.local_usage, prev_freed)) {
                        logW("kv pool check: local usage {:.2f} -> {:.2f} GiB after freeing {:.2f} GiB of KV pool (the WDDM "
                             "numbers did not follow within 1 s); not shrinking further",
                             prev_usage / 1073741824.0, wa.local_usage / 1073741824.0, prev_freed / 1073741824.0);
                        break;
                    }
                }
                const std::uint32_t have = model.pool_pages * kv_page;
                // the margin for the budget now: other processes the budget now shows are no longer pending
                const poolbudget::Margin mk = lm.budgetMargin(wa.local_budget);
                const std::uint64_t budget_margin = mk.bytes;
                const poolbudget::Trim tr =
                    poolbudget::fitPool(have, wa.local_usage, wa.local_budget, budget_margin, per_tok, kv_page, 4096);
                if (tr.over == 0) {
                    if (round > 0 || wa.local_budget != wddm0.local_budget)
                        logI("kv pool check: WDDM budget {:.2f} GiB now ({:.2f} GiB when the pool was sized), local {:.2f} GiB + "
                             "margin {} MiB fits (KV pool {} tokens)",
                             wa.local_budget / 1073741824.0, wddm0.local_budget / 1073741824.0, wa.local_usage / 1073741824.0,
                             budget_margin >> 20, have);
                    break;
                }
                if (!tr.shrink) break;  // already at the smallest pool; the WDDM memory line below warns
                logW("kv pool check: WDDM budget {:.2f} GiB now ({:.2f} GiB when the pool was sized), local {:.2f} GiB + margin "
                     "{} MiB is {:.2f} GiB over -> shrinking the KV pool from {} to {} tokens ({:.2f} -> {:.2f} GiB) so WHIRL "
                     "stays in VRAM{}",
                     wa.local_budget / 1073741824.0, wddm0.local_budget / 1073741824.0, wa.local_usage / 1073741824.0,
                     budget_margin >> 20, tr.over / 1073741824.0, have, tr.pool_tokens, poolGib(have), poolGib(tr.pool_tokens),
                     tr.at_floor ? " (the smallest pool, 4096 tokens: it may still not fit)" : "");
                prev_usage = wa.local_usage;
                prev_freed = per_tok * (have - tr.pool_tokens);
                model.freeKvPool();
                try {
                    startup::allocKvPool(model, tr.pool_tokens, *slots);
                } catch (const std::exception& ex) {
                    logE("cannot allocate the smaller KV pool of {} tokens ({:.2f} GiB): {}", tr.pool_tokens, poolGib(tr.pool_tokens),
                         ex.what());
                    throw;
                }
                if (opt.ctx)
                    lm.adjust(std::format("-c {} -> {}", have, tr.pool_tokens),
                           std::format("the WDDM budget is {:.2f} GiB now ({:.2f} GiB when the pool was sized)", wa.local_budget / 1073741824.0,
                                       wddm0.local_budget / 1073741824.0));
                if (slot_ctx_eff > tr.pool_tokens) {
                    if (opt.ctx_per_slot)
                        lm.adjust(std::format("--ctx-per-slot {} -> {}", slot_ctx_eff, tr.pool_tokens),
                               std::format("the KV pool that fits the WDDM budget holds {} tokens", tr.pool_tokens));
                    else
                        logW("kv pool check: context per request lowered to {} tokens (wanted {}): the pool that fits the WDDM "
                             "budget is smaller",
                             tr.pool_tokens, slot_ctx_eff);
                    slot_ctx_eff = tr.pool_tokens;
                    engine->setCtx(slot_ctx_eff);
                }
            }
        } else {
            // WHIRL_POOL_RESERVE_MB (the old free-VRAM rule, no WDDM budget): only say whether it fits the budget now
            const hip::WddmMemInfo wa = lm.ops->wddmMemInfo();
            const std::uint32_t have = model.pool_pages * kv_page;
            if (wa.ok) {
                const poolbudget::Trim tr = poolbudget::fitPool(have, wa.local_usage, wa.local_budget, budget_margin, per_tok, kv_page, kv_page);
                if (tr.over > 0)
                    logW("kv pool check: local {:.2f} GiB + margin {} MiB is {:.2f} GiB over the WDDM budget of {:.2f} GiB with "
                         "the KV pool of {} tokens ({:.2f} GiB), which is not resized because {}. About {} tokens fit: {}",
                         wa.local_usage / 1073741824.0, budget_margin >> 20, tr.over / 1073741824.0, wa.local_budget / 1073741824.0,
                         have, poolGib(have),
                         lm.pool_req ? "-c / --ctx was given" : "WHIRL_POOL_RESERVE_MB is set (free-VRAM rule, no WDDM budget)",
                         tr.at_floor ? 0u : tr.pool_tokens,
                         lm.pool_req ? "restart with that -c or a smaller one, or close other programs that use the GPU"
                                  : "unset WHIRL_POOL_RESERVE_MB (WHIRL_POOL_BUDGET_MARGIN_MB sets the margin), or close other "
                                    "programs that use the GPU");
            }
        }
    }
    {
        const std::uint32_t pool_now = model.pool_pages * kv_page;
        const std::uint64_t floor = static_cast<std::uint64_t>(slot_ctx_eff) + (opt.parallel > 1 ? std::min(floor_second_ctx, slot_ctx_eff) : 0);
        if (pool_now < floor)
            logW("kv pool {} tokens < {} (one {}-token request + a second one reaching {}): concurrent long requests wait in "
                 "the queue, or end early while generating (finish_reason length, finish_cause kv_pool_exhausted){}",
                 pool_now, floor, slot_ctx_eff, opt.parallel > 1 ? std::min(floor_second_ctx, slot_ctx_eff) : 0,
                 model.kv_choice.kv == KvFormat::f16 && !lm.cfg.moe && nreq.mode == numerics::Mode::precise
                     ? "; --balance allows int8 KV (a longer pool)"
                     : "");
    }
}

void ServerRun::reportPool() {
    const double ck_mb = static_cast<double>(n_gdn * (conv_bytes + ssm_bytes) + static_cast<std::uint64_t>(V) * 4 +
                                             static_cast<std::uint64_t>(lm.cfg.n_embd) * 4) /
                         (1024.0 * 1024.0);
    if (lm.no_pc) logI("prefix cache: disabled (WHIRL_NO_PREFIX_CACHE)");
    else logI("prefix cache: {} checkpoint(s) per slot x {:.1f} MiB ({} DeltaNet layers of recurrent state + logits row)", lm.n_ck, ck_mb, n_gdn);
    if (!lm.no_pc && eo.cold_resume)
        logI("prefix cache: cold-exact resume (precise): a request resumes only at a checkpoint on its own chunk schedule "
             "that prefill produced (never the end of a generated reply), so its output is bitwise the cold run's; a next "
             "turn re-prefills from the previous prompt's last schedule point (WHIRL_COLD_RESUME=0: longest checkpoint)");
    if (!lm.no_pc) {
        if (!eo.tail_ck)
            logW("prefix cache: tail checkpoint off (WHIRL_TAIL_CK=0, debug override): a second question about the same long "
                 "document re-prefills it");
        else if (lm.n_ck >= tail_ck_slots)
            logI("prefix cache: tail checkpoint on: prompts >= {} tokens keep one of the {} per-slot checkpoints ({:.1f} MiB) at the "
                 "start of their last chunk, so another question about the same document resumes there",
                 eo.tail_ck_min, lm.n_ck, ck_mb);
        else
            logI("prefix cache: tail checkpoint off ({} checkpoint(s) per slot, needs {}; WHIRL_SERVE_CKPTS={} turns it on at "
                 "+{:.1f} MiB VRAM per slot each)",
                 lm.n_ck, tail_ck_slots, tail_ck_slots, ck_mb);
    }
    {
        const double st_mib = static_cast<double>(n_gdn * (conv_bytes + ssm_bytes)) / (1024.0 * 1024.0);
        const double snap_mib = st_mib * model.snap_sets;
        logI("kv pool: {} tokens ({} pages of {}), {} KV{}, {:.1f} KiB/token, {:.2f} GiB, shared by {} slots; max {} tokens per "
             "request{}",
             model.pool_pages * kv_page, model.pool_pages, kv_page, model.kvName(), kv_note, static_cast<double>(per_tok) / 1024.0,
             static_cast<double>(per_tok * model.pool_pages * kv_page) / (1024.0 * 1024.0 * 1024.0), opt.parallel,
             std::min<std::uint64_t>(slot_ctx_eff, static_cast<std::uint64_t>(model.pool_pages) * kv_page),
             lm.use_dflash ? " (--draft dflash: no MTP KV layer; the drafter's weights and rings came first)" : "");
        logI("slots: {}; per slot: recurrent state {:.1f} MiB, prefix checkpoints {:.1f} MiB; shared: {} snapshot sets {:.1f} MiB",
             opt.parallel, st_mib, ck_mb * static_cast<double>(lm.n_ck), model.snap_sets, snap_mib);
    }
    const hip::MemInfo mem2 = lm.ops->memInfo();
    if (tier_pre)
        logI("kv tier: VRAM taken by the tier stream / pinned arena mapping: {:.1f} MiB (before the pool was sized)",
             static_cast<double>(mem_ck1.free > mem_t1.free ? mem_ck1.free - mem_t1.free : 0) / 1048576.0);
    logI("VRAM use: weights + buffers {:.2f} GiB (prefill batch {}, embedding {}), slots {:.2f} GiB, other {:.2f} GiB, checkpoints "
         "{:.2f} GiB{}, KV pool {:.2f} GiB, left {:.2f} GiB",
         gib(lm.mem0.free, lm.mem.free), model.max_batch, model.embd_host ? "in pinned host memory" : "in VRAM",
         gib(mem_s0.free, mem_ck0.free), gib(lm.mem.free, mem_s0.free), gib(mem_ck0.free, mem_ck1.free),
         eo.ckpt_host ? " (in pinned host memory)" : "", gib(mem_ck1.free, mem2.free),
         static_cast<double>(mem2.free) / (1024.0 * 1024.0 * 1024.0));
    logI("context {} tokens per request, pool {} tokens; VRAM free {:.2f} / {:.2f} GiB (after weights {:.2f}, at start {:.2f})",
         std::min<std::uint64_t>(slot_ctx_eff, static_cast<std::uint64_t>(model.pool_pages) * kv_page), model.pool_pages * kv_page, static_cast<double>(mem2.free) / (1024.0 * 1024.0 * 1024.0),
         static_cast<double>(mem2.total) / (1024.0 * 1024.0 * 1024.0), static_cast<double>(lm.mem.free) / (1024.0 * 1024.0 * 1024.0),
         static_cast<double>(lm.mem0.free) / (1024.0 * 1024.0 * 1024.0));
    if (lm.use_mtp)
        logI("prompt-lookup (n-gram) drafting: {} (WHIRL_NGRAM{}), min match {}, up to {} drafts", eo.ngram ? "on" : "off",
             eo.ngram ? "=0 turns it off" : "=1 turns it on", eo.ngram_min,
             eo.ngram_max > 0 ? std::min(eo.ngram_max, qwen35::max_ng_drafts) : qwen35::max_ng_drafts);
    // runtime watch baseline (P3-4): the sizing read, so a spill already present at startup is not hidden;
    // pinned host memory WHIRL allocates after it is subtracted (wddmwatch::Sample::pinned)
    const auto vramSampleOf = [](const hip::WddmMemInfo& m, std::uint64_t pinned) {
        wddmwatch::Sample s;
        s.ok = m.ok;
        s.local_budget = m.local_budget;
        s.local_usage = m.local_usage;
        s.nonlocal_usage = m.nonlocal_usage;
        s.pinned = pinned;
        return s;
    };
    vram_base = vramSampleOf(wddm0, pinned0);
    vram_cfg.pool_tokens = model.pool_pages * kv_page;
    // v0.2.2.2 memory governor: discrete GPU with WDDM numbers, the budget rule in use (not WHIRL_POOL_RESERVE_MB, whose
    // pool is never resized), WHIRL_GOVERNOR=0 turns it off
    gov_on = wddm0.ok && !lm.is_uma && !reserve_explicit && settings::envFlag("GOVERNOR", true);
    vram_cfg.governor = gov_on;
    startup_spill = false;
    engine->setNumerics(numerics::propsJson(model.num_plan, kvFormatName(model.kv_fmt)));
    engine->initPool();
    {
        // after everything sized at startup is allocated: any growth of the non-local segment
        // since the pool was sized is VRAM demoted to system memory (pinned host buffers made
        // before that point, such as the embedding table and the RAM tier, are in the baseline)
        const hip::WddmMemInfo w = lm.ops->wddmMemInfo();
        if (w.ok && wddm0.ok) {
            const std::uint64_t pinned_now = lm.ops->pinnedHostBytes();
            const std::uint64_t grow = w.nonlocal_usage > wddm0.nonlocal_usage ? w.nonlocal_usage - wddm0.nonlocal_usage : 0;
            const std::uint64_t pinned_since = pinned_now > pinned0 ? pinned_now - pinned0 : 0;
            const std::string msg = std::format(
                "WDDM memory: local {:.2f} / budget {:.2f} GiB; non-local (GPU 'Shared Usage') {:.0f} MiB, of which {:.0f} MiB was there "
                "before the KV pool (declared pinned host memory: embedding {:.0f} MiB{}); growth {:.0f} MiB ({:.0f} MiB of it pinned "
                "host memory WHIRL allocated since)",
                w.local_usage / 1073741824.0, w.local_budget / 1073741824.0, w.nonlocal_usage / 1048576.0,
                wddm0.nonlocal_usage / 1048576.0, lm.embd_pinned / 1048576.0, tier_pre ? ", RAM tier" : "", grow / 1048576.0,
                pinned_since / 1048576.0);
            // same test and wording as the runtime watch (wddm_watch.h), against the same baseline
            const wddmwatch::Sample now_s = vramSampleOf(w, pinned_now);
            wddmwatch::Watch startup(vram_base, vram_cfg);
            const auto ev = !lm.is_uma ? startup.check(now_s, 0.0) : std::nullopt;
            startup_spill = ev.has_value() && ev->warn;
            if (startup_spill)
                logW("{} | at startup: {}", msg, ev->text);
            else if (w.local_usage > w.local_budget)  // UMA iGPU only (no warning text there)
                logW("{} -- over the WDDM budget, part of the GPU buffers may live in system memory and decode can be several "
                     "times slower; give a smaller --ctx",
                     msg);
            else
                logI("{}", msg);
        }
        // integrated GPU: what this process holds of the carve-out now (its GPU "Dedicated Usage"), against
        // the pre-load estimate (UMA-1)
        if (lm.uma.known) {
            const hip::LocalMemSnap s = lm.ops->localMemSnap();
            if (s.pdh_ok)
                logI("UMA carve-out use after load: {:.2f} GiB of {:.1f} GiB dedicated (GPU Dedicated Usage of this process); "
                     "GPU Shared Usage {:.2f} GiB",
                     s.dedicated / 1073741824.0, lm.uma.dedicated / 1073741824.0, s.shared / 1073741824.0);
        }
    }
    if (eo.n_spe > 0)
        logI("shared prefix checkpoints: {} x {:.1f} MiB in VRAM; system messages >= {} tokens are prefilled as their own chunk "
             "run and kept (WHIRL_SYS_MIN, 0 = no split), plus checkpoints at prefixes common to sessions ({})",
             eo.n_spe, ck_mb, eo.sys_min, eo.lcp_on ? "on; WHIRL_SYS_LCP=0 turns it off" : "off");
    else
        logI("shared prefix checkpoints: off (system message split at >= {} tokens{})", eo.sys_min, eo.sys_min == 0 ? ": off" : "");
}

void ServerRun::startTier() {
    // host prefix-cache tiers (pinned RAM, SSD; tier_owned is declared before the engine)
    {
        std::uint64_t ssd_gb = kv_ssd_gb_default;
        if (opt.kv_ssd_gb) ssd_gb = *opt.kv_ssd_gb;
        else if (auto v = settings::envGet("KV_SSD_GB")) {
            try {
                ssd_gb = std::stoull(*v);
            } catch (const std::exception&) {
            }
        }
        std::uint64_t page_row = 0;
        for (const KvArr& a : backend->kvArrays()) page_row += a.row;
        if (tier_pre && page_row == per_tok) {
            std::optional<std::string> ssd_dir;
            std::string ssd_mode = opt.kv_ssd_mode.value_or(settings::envGet("KV_SSD_MODE").value_or("shutdown"));
            if (ssd_mode != "always" && ssd_mode != "off") ssd_mode = "shutdown";
            if (ssd_gb > 0 && ssd_mode != "off") {
                if (opt.kv_ssd_dir) ssd_dir = opt.kv_ssd_dir;
                else if (auto v = settings::envGet("KV_SSD_DIR")) ssd_dir = *v;
                else if (lad) ssd_dir = platform::paths::kvCacheDir(lad);
            }
            tier::Layout lay;
            lay.ck_bytes = n_gdn * (conv_bytes + ssm_bytes) + static_cast<std::uint64_t>(lm.cfg.n_embd) * 4 + static_cast<std::uint64_t>(V) * 4;
            lay.page_bytes = per_tok * kv_page;
            lay.page_tokens = kv_page;
            lay.tok_cap = slot_ctx_eff;
            const TimePoint t_tier0 = Clock::now();
            tier::Config tc;
            tc.ram_bytes = kv_ram_mb << 20;
            tc.ssd_dir = ssd_dir;
            tc.ssd_cap = ssd_gb << 30;
            tc.min_tokens = settings::envU32("KV_TIER_MIN", 2048, settingsWarn);
            if (opt.kv_ssd_delay) tc.ssd_delay_ms = *opt.kv_ssd_delay * 1000;
            else tc.ssd_delay_ms = settings::envU32("KV_SSD_DELAY_MS", 300000, settingsWarn);
            tc.expire_s = settings::envU32("KV_SSD_EXPIRE_S", 7 * 24 * 3600, settingsWarn);  // test override only
            tc.ssd_shutdown_only = ssd_mode == "shutdown";
            tc.dedup = settings::envU32("KV_SSD_DEDUP", 1, settingsWarn) != 0;
            // the wake hook must be in place before the IO thread starts
            engine->attachTier(tier_pre.get());
            try {
                // throws when the GPU identity is missing: an error, never the old shared format
                const std::uint64_t fp = tierFingerprint(model, opt.path, lm.stats, lm.use_mtp, n_gdn, lay);
                tier_pre->start(tc, lay, fp);
                tier_owned = std::move(tier_pre);
            } catch (const std::exception& ex) {
                engine->attachTier(nullptr);
                logE("kv tier: cannot set up the host tiers ({}); running without them", ex.what());
            }
            if (tier_owned) {
                tier::Tier& tr = *tier_owned;
                engine->setKvSsdMode(tr.hasSsd() ? ssd_mode : std::string("off"));
                if (tr.ssdDirBusy())
                    logW("kv tier: {} is in use by another WHIRL server; SSD tier off for this one (give each server its own --kv-ssd-dir)",
                         *ssd_dir);
                tier::declarePinned(lad, lm.embd_pinned + tr.pinnedBytes());
                logI("kv tier: RAM {:.2f} GiB of pinned host memory (counted as this process's GPU 'Shared Usage'; pinned in {:.1f} s, "
                     "SSD index {:.1f} s), entries >= {} tokens; SSD {}{} (cap {} GiB, {} entries / {:.2f} GiB indexed); entry: {:.1f} "
                     "MiB per checkpoint, {:.1f} MiB per 256-token page",
                     static_cast<double>(tr.pinnedBytes()) / 1073741824.0, tier_alloc_s, msSince(t_tier0) / 1000.0, tr.cfg.min_tokens,
                     ssd_dir ? *ssd_dir : std::string("off"), (ssd_dir && !tr.hasSsd()) ? " (unavailable)" : "", ssd_gb,
                     tr.ssdEntries(), static_cast<double>(tr.ssd_bytes) / 1073741824.0,
                     static_cast<double>(lay.ck_bytes) / 1048576.0, static_cast<double>(lay.page_bytes) / 1048576.0);
            }
        } else if (kv_ram_mb > 0 && tier_pre && page_row != per_tok) {
            logW("kv tier: KV layout not understood ({} vs {} bytes per token); host tiers off", page_row, per_tok);
        } else {
            logI("kv tier: off");
        }
    }
}

void ServerRun::attachVision() {
    // ---- vision encoder (discrete GPU: read above, before the KV pool, which leaves room for it; integrated GPU: here,
    // after the KV pool, whose size does not depend on it)
    if (const std::optional<std::string> mmp = mmp_path) {
        const TimePoint tv0 = Clock::now();
        const hip::LocalMemSnap mem_v0 = lm.ops->localMemSnap();
        if (!vis_store) {
            try {
                vis_store = vision::Vision::load(*mmp);
            } catch (const std::exception& ex) {
                logE("vision: cannot load mmproj {}: {}", *mmp, ex.what());
                throw;
            }
        }
        vision::Vision& v = *vis_store;
        v.idle_s = opt.vis_idle_s;
        if (auto x = settings::envGet("VIS_IDLE_S")) {
            try {
                v.idle_s = static_cast<std::uint32_t>(std::stoul(*x));
            } catch (const std::exception&) {
            }
        }
        v.mode = *vision::parseMode(opt.vis_mode);
        if (auto x = settings::envGet("VIS_MODE")) v.mode = vision::parseMode(*x).value_or(v.mode);
        VisionConfig vc;
        {
            const TokenId ve = tok.find("<|vision_end|>");
            vc.vision_end_id = ve < 0 ? 0 : static_cast<std::uint32_t>(ve);
        }
        if (auto x = settings::envGet("VIS_CKPT_MIN")) {
            try {
                vc.ck_min = static_cast<std::uint32_t>(std::stoul(*x));
            } catch (const std::exception&) {
            }
        }
        const TokenId pad = tok.find("<|image_pad|>");
        if (pad < 0) {
            logE("vision: the model's vocabulary has no <|image_pad|> token");
            throw std::runtime_error("NoImagePadToken");
        }
        vc.image_pad_id = static_cast<std::uint32_t>(pad);
        if (!backend->hasMrope()) {
            logE("vision: this GPU's kernels have no multi-section RoPE (attn_prep_m)");
            throw std::runtime_error("MropeUnsupported");
        }
        std::uint64_t cache_mb = opt.vis_cache_mb;
        if (auto x = settings::envGet("VIS_CACHE_MB")) {
            try {
                cache_mb = std::stoull(*x);
            } catch (const std::exception&) {
            }
        }
        vc.cache_bytes = cache_mb << 20;
        vc.allow_files = opt.allow_local_images;
        if (auto x = settings::envGet("VIS_DUMP")) vc.dump_prefix = *x;
        vis_srv = makeQwen35Vision(v, model);
        vc.vis = vis_srv.get();
        std::array<std::array<std::uint64_t, 2>, 16> raw{};
        const std::size_t n_lend = backend->lendScratch(raw);
        std::uint64_t lent = 0;
        for (std::size_t k = 0; k < n_lend; ++k) lent += raw[k][1];
        tier::declarePinned(lad, lm.embd_pinned + (tier_owned ? tier_owned->pinnedBytes() : 0) + v.pinnedBytes());
        hip::LocalMemTaken vis_taken = hip::localMemTaken(mem_v0, lm.ops->localMemSnap());
        if (vis_load_taken) {
            // read before the KV pool: the load's change plus the attach's change here
            vis_taken.mib += vis_load_taken->mib;
            vis_taken.hip_free_mib += vis_load_taken->hip_free_mib;
            vis_taken.shared_mib += vis_load_taken->shared_mib;
        }
        engine->attachVision(vc);
        logI("vision: {}: {} layers, {} pinned MiB in host RAM ({:.1f} s; VRAM change {:.1f} MiB [{}; hipMemGetInfo free {:+.1f} MiB, GPU Shared Usage {:+.1f} MiB]); encoder loaded on demand, released "
             "after {} s idle; weights {}; activations in the prefill scratch ({} buffers, {:.0f} MiB; one 4096-token image needs {:.0f} "
             "MiB); embedding cache {} MiB; local image files {}",
             *mmp, v.hp.n_layer, v.pinnedBytes() >> 20, msSince(tv0) / 1000.0 + vis_load_s,
             vis_taken.mib, vis_taken.how, -vis_taken.hip_free_mib, vis_taken.shared_mib, v.idle_s, vision::modeName(v.mode), n_lend,
             static_cast<double>(lent) / 1048576.0, static_cast<double>(v.arenaBytes(v.hp.max_tokens)) / 1048576.0, cache_mb,
             vc.allow_files ? "allowed" : "off");
    }
}

int ServerRun::runServer() {
    logI("model ready in {:.1f} s (weights {:.1f} s)", msSince(lm.t_load0) / 1000.0, lm.stats.ms / 1000.0);
    const bool timer_probe = settings::envGet("TIMER_PROBE").has_value();
    const double probe_1ms = timer_probe ? sleepMs(std::chrono::milliseconds(1), 50) : 0.0;
    const double probe_200us = timer_probe ? sleepMs(std::chrono::microseconds(200), 50) : 0.0;
    const TimerRes timer_res;
    if (!timer_res.on) logW("timer: 1 ms resolution not available; short waits of the main loop take one system tick");
    if (timer_probe)
        logI("timer probe | sleep 1 ms: {:.2f} ms before, {:.2f} ms after timeBeginPeriod(1) | sleep 200 us: {:.2f} / {:.2f} ms",
             probe_1ms, sleepMs(std::chrono::milliseconds(1), 50), probe_200us, sleepMs(std::chrono::microseconds(200), 50));
    // WDDM-LOG (issues #5 / #6): video memory and RAM in the periodic status line and at request end, against a
    // baseline taken now that the KV pool, RAM tier and vision buffers exist
    engine->setVramProbe(
        [be = lm.ops.get()] {
            const hip::WddmMemInfo w = be->wddmMemInfo();
            wddmwatch::Sample s;
            s.ok = w.ok;
            s.local_budget = w.local_budget;
            s.local_usage = w.local_usage;
            s.nonlocal_usage = w.nonlocal_usage;
            s.pinned = be->pinnedHostBytes();
            std::uint64_t ws = 0;
            if (platform::proc::workingSetBytes(platform::proc::currentProcess(), &ws)) s.rss = ws;
            return s;
        },
        !lm.is_uma, wddm0.ok ? std::optional<wddmwatch::Sample>(vram_base) : std::nullopt, vram_cfg, startup_spill,
        // what other processes use on the GPU, read only when a spill warning is printed (one PDH read)
        [be = lm.ops.get(), dev = lm.dev, o0 = lm.others0]() -> std::optional<wddmwatch::Others> {
            const hip::GpuOthers o = be->gpuOthers(dev);
            if (!o.ok) return std::nullopt;
            return wddmwatch::Others{o.others, o0.ok ? std::optional<std::uint64_t>(o0.others) : std::nullopt};
        });
    if (gov_on) {
        memgov::Config gc;
        gc.per_tok = per_tok;
        gc.page = kv_page;
        gc.pool_start = model.pool_pages * kv_page;
        gc.pool_min = std::min<std::uint32_t>(gc.pool_start, 4096);
        gc.ctx = slot_ctx_eff;  // the engine's context per request (the governor lowers it only below the startup pool)
        // test overrides (no CLI flag): recovery time, hold after a shrink
        if (auto v = settings::envGet("GOV_RECOVER_S")) gc.recover_s = std::strtod(v->c_str(), nullptr);
        if (auto v = settings::envGet("GOV_HOLD_S")) gc.hold_s = std::strtod(v->c_str(), nullptr);
        engine->setGovernor(gc, [&](std::uint64_t budget) { return lm.budgetMargin(budget).bytes; }, tier_owned ? kv_ram_mb : 0,
                           lm.ops->memInfo().total);
        const poolbudget::Margin mg = lm.budgetMargin(wddm0.local_budget);
        logI("memory governor: on: when less than {} MiB (half the {} MiB margin) is free under the WDDM budget while the server "
             "runs, it frees cache, queues new requests and re-allocates the KV pool smaller when every slot is idle; it grows "
             "back when idle after {:.0f} s with room (more than {} MiB free), straight to {} tokens (the startup pool, startup "
             "context again) once that fits with {} MiB free (WHIRL_GOVERNOR=0 turns it off)",
             mg.bytes / 2 >> 20, mg.bytes >> 20, gc.recover_s, (mg.bytes + mg.bytes / 2) >> 20, gc.pool_start,
             memgov::endMargin(mg.bytes) >> 20);
        logI("memory governor: HIP resource cache {} (freed GPU memory larger than that goes back to Windows at once)",
             hip_cache_set ? std::string("limited to 256 MiB (GPU_RESOURCE_CACHE_SIZE; WHIRL_HIP_CACHE_LIMIT=0 keeps the runtime default)")
             : hip_cache_user ? std::format("set by GPU_RESOURCE_CACHE_SIZE={} MiB", platform::env::get("GPU_RESOURCE_CACHE_SIZE").value_or(""))
                              : std::string("at the runtime default (WHIRL_HIP_CACHE_LIMIT=0: part of freed GPU memory is kept, 0.87 GiB of a 3.91 GiB shrink in our R9700 test)"));
    } else if (wddm0.ok && !lm.is_uma) {
        logI("memory governor: off ({})", reserve_explicit ? "WHIRL_POOL_RESERVE_MB is set: the pool is never resized" : "WHIRL_GOVERNOR=0");
    }
    {
        // v0.2.2.1: one line with every startup adjustment, and the effective values for /props and /health, set before
        // the HTTP server starts (REVIEW-0221 C5: /health readers on other threads)
        const std::uint32_t ctx_eff = std::min<std::uint32_t>(slot_ctx_eff, model.pool_pages * kv_page);
        const std::uint64_t target_ctx = std::min<std::uint32_t>(lm.slot_ctx, vram_tight_pool);
        if (opt.parallel_given && ctx_eff < target_ctx)
            lm.adjusted.push_back(std::format("target not reached: context {} tokens per request with {} slot(s), less than {}", ctx_eff,
                                           opt.parallel, target_ctx));
        // arch024-S0a (PLAN v3 N3): every input of the sizing's pure functions, for tests/pins startup_replay (log only)
        logI("startup inputs:{}", lm.sin.line);
        if (lm.adjusted.empty()) {
            logI("startup adjustments: none (the requested settings fit)");
        } else {
            std::string all;
            for (const std::string& a : lm.adjusted) all += (all.empty() ? "" : "; ") + a;
            logW("startup adjustments ({}): {}. Effective: context {} tokens per request, KV pool {} tokens, {} slot(s), RAM tier {} "
                 "MiB; a longer prompt is rejected as too long for the context, and generation stops at the context limit",
                 lm.adjusted.size(), all, ctx_eff, model.pool_pages * kv_page, opt.parallel, tier_owned ? kv_ram_mb : 0);
        }
        startfit::StartupInfo si;
        si.ctx_per_request = ctx_eff;
        si.pool_tokens = model.pool_pages * kv_page;
        si.parallel = opt.parallel;
        si.kv_ram_mb = tier_owned ? kv_ram_mb : 0;  // REVIEW-0221 A1: tier_pre is empty once it moved into tier_owned
        si.wddm_margin_mib = wddm0.ok && !lm.is_uma ? budget_margin >> 20 : 0;
        si.other_processes_mib = lm.others0.ok ? lm.others0.others >> 20 : 0;
        si.display_attached = lm.others0.display_known && lm.others0.display;
        si.adjusted = lm.adjusted;
        std::string note;
        for (const std::string& a : lm.adjusted)
            if (a.find("--ctx-per-slot") != std::string::npos || a.rfind("-c ", 0) == 0) note += (note.empty() ? "" : "; ") + a;
        engine->setStartup(startfit::startupJson(si), note.empty() ? std::string() : "adjusted at startup to fit the GPU memory: " + note);
    }
    HttpOptions ho;
    ho.cors_origins = opt.cors_origins;
    ho.allowed_hosts = opt.allowed_hosts;
    HttpServer http(*engine, ho);
    http.start(opt.host, opt.port);
    logI("server listening on http://{}:{} (model id \"{}\", {} slots); endpoints: GET /health, GET /v1/models, POST "
         "/v1/chat/completions, POST /v1/completions, GET /props, GET /version",
         opt.host, http.port(), name, opt.parallel);
    // main thread: continuous batching over the slots (until a console control
    // event stops the engine: graceful shutdown, see onConsoleCtrl)
    g_stopped = platform::proc::createEvent();
    g_engine.store(&*engine);
    platform::proc::restoreCtrlC();  // an "ignore Ctrl+C" attribute inherited from the parent would hide Ctrl+C
    platform::proc::addCtrlHandler(onConsoleCtrl);
    engine->runLoop();
    http.stop();
    logI("server stopped");
    g_engine.store(nullptr);
    if (g_stopped != nullptr) platform::proc::setEvent(g_stopped);
    return 0;
}

}  // namespace

std::string serveHelp(const char* program) {
    return std::format("{} {}\n\nusage: {} MODEL.gguf [options]\n\n", program, app::version(), program) + kHelpBody +
           app::envHelp(app::sc_serve) + "\n" + app::exitCodeHelp();
}

int serveMain(int argc, char** argv, const char* program) {
    // argv: UTF-8, argv[0] = the program (or the `serve` subcommand), options from argv[1]
    g_program = program;
    // launch log (WHIRL_LAUNCH_LOG): no per-launch RAW dump in a server unless WHIRL_LAUNCH_LOG_RAW=server (C-2)
    hip::launchlog::serverProcess();
    hip::alloclog::serverProcess();  // arch025-S0a: the allocation event list off unless WHIRL_ALLOC_LOG_RAW=server
    std::vector<std::string> args;
    for (int k = 1; k < argc; ++k) {
        // arch025-S0a: --dump-only = exit 0 right after load + tune (baseline dump matrix, whirl/dump.h)
        if (std::string_view(argv[k]) == "--dump-only") {
            dump::setDumpOnly(true);
            continue;
        }
        args.push_back(argv[k]);
    }
    for (const std::string& a : args) {
        // help / version first: they need neither a model nor a GPU
        if (a == "-h" || a == "--help" || a == "/?") {
            const std::string h = serveHelp(program);
            std::fwrite(h.data(), 1, h.size(), stdout);
            return app::exit_ok;
        }
        if (a == "-V" || a == "--version") {
            const std::string v = app::versionText(program);
            std::fwrite(v.data(), 1, v.size(), stdout);
            return app::exit_ok;
        }
        if (a == "--list-devices") {
            // every GPU (never restricted here); [default] = the GPU used without --device
            try {
                const std::vector<gpusel::GpuRecord> devs = gpusel::enumerate();
                if (const auto hv = platform::env::get("HIP_VISIBLE_DEVICES"))
                    std::printf("  (HIP_VISIBLE_DEVICES=%s is set: only those GPUs are listed, numbered from 0)\n", hv->c_str());
                if (devs.empty()) std::printf("  (%s)\n", gpusel::noGpuTextHere().c_str());  // arch025-c2fu: names a hiding variable
                else std::fputs(gpusel::listingText(devs, gpusel::defaultGpu(devs).index).c_str(), stdout);
            } catch (const std::exception& ex) {
                std::fprintf(stderr, "%s: error: %s\n", program, ex.what());
                return app::exit_gpu;
            }
            return app::exit_ok;
        }
    }
    // v0.2.2.2 memory governor: ROCclr keeps freed device allocations smaller than its resource cache limit (its default size
    // was not checked in the ROCclr source; REVIEW-GOV-2 A-5) for reuse instead of returning them to Windows, and reuses a cached block for any
    // request between half its size and its size. A KV pool re-allocated smaller then kept most of the old pool's VRAM
    // (R9700 S2c: 1.7 GiB of a 4 GiB shrink still counted against the WDDM budget, decode stayed 57 instead of 259 tok/s).
    // Set before the first HIP API call: gpuPreflight has already loaded amdhip64_7.dll by now, but ROCclr reads its flags
    // when the runtime initialises at the first API call, not at DLL load (measured: setting it here == setting it in the
    // parent environment, cache_test V1 == V2). A value the user set wins; WHIRL_HIP_CACHE_LIMIT=0 leaves the runtime
    // default. Allocation only: kernels and output are not affected.
    const bool hip_cache_user = platform::env::get("GPU_RESOURCE_CACHE_SIZE").has_value();
    const bool hip_cache_set = !hip_cache_user && settings::envFlag("HIP_CACHE_LIMIT", true);
    if (hip_cache_set) platform::env::set("GPU_RESOURCE_CACHE_SIZE", "256");  // arch025 PL1: _putenv_s
    // arch025-C1a: the argument loop, then the numerics request and the --parallel range (server_args.cpp, run by the
    // argument table test); WHIRL_MODE is read here, after the loop, as before. A bad command line: die(message).
    Options opt0 = [&] {
        try {
            return parseServerArgs(args, [](std::string_view s) { return vision::parseMode(s).has_value(); });
        } catch (const ArgError& ex) {
            die(ex.what());
        }
    }();
    std::optional<std::string> mode_env = settings::mode();
    ResolvedConfig rc = [&] {
        try {
            return resolveServerConfig(std::move(opt0), std::move(mode_env));
        } catch (const ArgError& ex) {
            die(ex.what());
        }
    }();
    Options& opt = rc.opt;
    numerics::Request& nreq = rc.nreq;
    // bug-emptyrootpaths: WHIRL_KV_SSD_DIR = --kv-ssd-dir (docs/usage.md), so an empty value is refused the same way,
    // before the model load (read again at the tier setup below; --kv-ssd-dir wins there as before). The check is
    // server_args.cpp checkKvSsdDirEnv (server-tests server_kv_ssd_dir_env); WHIRL_KV_SSD_DIR is read here only when
    // no --kv-ssd-dir was given, as before.
    try {
        if (!opt.kv_ssd_dir) checkKvSsdDirEnv(opt, settings::envGet("KV_SSD_DIR"));
    } catch (const ArgError& ex) {
        die(ex.what());
    }
    // the model file and the port are checked before the (long) model load
    try {
        app::requireModelFile(opt.path);
        if (opt.mmproj) app::requireModelFile(*opt.mmproj, "vision encoder (--mmproj) file");
        app::requirePortFree(opt.host, opt.port);
    } catch (const std::exception& ex) {
        return app::explain(ex, g_program);
    }

    // arch025 PL1: LOCALAPPDATA (the CRT narrow value, as std::getenv) and the default paths are platform::paths
    // (bug-emptyrootpaths: set but empty is nullopt = unset: log in the working folder, no default SSD folder)
    const std::optional<std::string> lad_v = platform::paths::localAppData();
    const char* lad = lad_v ? lad_v->c_str() : nullptr;
    const std::string log_path = opt.log_file ? *opt.log_file : platform::paths::serverLogPath(lad_v);
    try {
        Log::open(log_path);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "warning: %s\n", ex.what());
    }

    try {
        logI("WHIRL serve starting, log file {}", log_path);
        LoadedModel lm;
        buildModel(opt, nreq, lad, lm);
        // the rest of the startup and the serving loop (arch025-C1b), in the old order
        ServerRun run(opt, nreq, lad, lm, hip_cache_user, hip_cache_set);
        run.planSlots();
        run.makeEngine();
        run.reserveRamTier();
        run.loadVisionEarly();
        run.planPool();
        run.reportPool();
        run.startTier();
        run.attachVision();
        return run.runServer();
    } catch (const std::exception& ex) {
        logE("fatal: {}", ex.what());
        int code = app::exit_error;
        const std::string msg = app::explainText(ex, &code);
        std::fprintf(stderr, "%s: error: %s", g_program.c_str(), msg.c_str());
        std::fflush(stderr);
        return code;
    }
}

}  // namespace whirl::server
