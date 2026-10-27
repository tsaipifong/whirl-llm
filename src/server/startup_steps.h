// Two ordered steps of the server's startup, moved verbatim out of serveMain (arch025-C0) so the server tests run the
// same code and record its calls (before, tests/server/server_tests.cpp compared positions in server_main.cpp's text):
//  - dgpuFit: the dGPU pre-load VRAM fit (FR-DFL P3-1): the module's load estimate (no MTP block with --draft dflash),
//    the fit input with the DFlash2 drafter counted before the 4% slack and before vram::planFit plans the KV pool;
//  - setupSlots: the slots, the DFlash2 rings / scratch, then the DeltaNet snapshot sets (FR-DFL P3-4; they need the
//    drafter's block and must exist before the KV pool is sized from the VRAM left). allocKvPool takes setupSlots'
//    result, so the pool cannot be allocated before them.
// No environment reads here (serveMain reads WHIRL_* and passes the values); the model type is a template parameter
// (qwen35::Model in whirl-server.exe, a recording fake in the tests). Header-only, like draft_caps.h: it needs
// whirl/model.h (DFlashConfig), which the server core library must not reach (tools/gates/include_check.py).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL.

#pragma once

#include "device/backend.h"
#include "draft_caps.h"
#include "log.h"
#include "model/arch_registry.h"
#include "platform/proc.h"  // armConsoleCtrl (arch025-surv)
#include "whirl/dflash_policy.h"
#include "whirl/gguf.h"
#include "whirl/model.h"
#include "whirl/numerics.h"
#include "whirl/vram_limit.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>

namespace whirl::server::startup {

// --- dGPU pre-load fit ----------------------------------------------------------------------------------------------

// --draft dflash: the drafter the fit counts (serveMain reads these before the fit)
struct DflashFitIn {
    qwen35::DFlashConfig dc;                 // DFlashConfig::fromGguf(drafter)
    const gguf::File* file = nullptr;        // the drafter (sizing.dflashWeightBytes)
    ::whirl::dflash::Params params;          // parseParams(WHIRL_DFLASH_*)
    std::optional<std::uint32_t> snap_env;   // WHIRL_SNAP_SETS
    bool gdn_replay = true;                  // WHIRL_GDN_REPLAY
    bool ngram = true;                       // WHIRL_NGRAM
};

struct DgpuFitIn {
    std::uint64_t avail = 0;                 // VRAM usable (free minus reserve, WDDM room, card headroom)
    std::uint32_t parallel = 1;
    bool embd_on_host = false;
    std::optional<DflashFitIn> dflash;       // engaged: --draft dflash was chosen (use_dflash)
    numerics::Request nreq;
    std::optional<KvFormat> kv_req;          // numerics::kvEstimateRequest(WHIRL_KV)
    bool ckpt_host = false;                  // WHIRL_CKPT_HOST
    std::uint32_t pool_min_tokens = 0;
    std::uint32_t max_batch = 0;
    std::uint32_t n_ck = 0;
    std::uint32_t n_spe = 0;
    bool batch_fixed = false;                // WHIRL_PREFILL_BATCH set
    bool ck_fixed = false;                   // WHIRL_SERVE_CKPTS / WHIRL_SYS_CKPTS set
};

struct DgpuFit {
    qwen35::LoadEstimate est;
    model::SlotStateBytes ssb;
    std::optional<::whirl::dflash::Mem> dflash_mem;  // what the drafter added to fi.fixed
    vram::FitInput fi;
    vram::FitPlan plan;                      // vram::planFit(fi)
};

inline DgpuFit dgpuFit(const model::ArchSizing& sizing, const gguf::File& f, const qwen35::Config& cfg, const DgpuFitIn& in) {
    DgpuFit r;
    const bool use_dflash = in.dflash.has_value();
    r.est = sizing.estimateLoad(f, cfg, in.embd_on_host, use_dflash);  // dflash: no MTP block
    const qwen35::LoadEstimate& est = r.est;
    // per-slot recurrent state + replay rows (model::slotStateBytes: the same formula for the dGPU fit and the UMA check)
    r.ssb = model::slotStateBytes(sizing, cfg);
    const std::uint64_t st1 = r.ssb.st1;
    const std::uint64_t pend1 = r.ssb.pend1;
    vram::FitInput& fi = r.fi;
    fi.avail = in.avail;
    fi.weights = est.weights;
    // load-time state, per-slot state + replay rows, draft head (64k rows, 2 bit), slack; and the
    // ~4% more that WDDM sees for this process than the sum of its allocations
    fi.fixed = est.state + in.parallel * (st1 + pend1) + (est.has_mtp ? 65536ull * cfg.n_embd * 5 / 16 : 0) + (256ull << 20);
    if (use_dflash) {
        // FR-DFL P3-1: the DFlash2 drafter (weights, capture, rings per slot, scratch, replay-less
        // snapshot sets) instead of the MTP block / draft head / MTP KV layer (est above)
        const DflashFitIn& d = *in.dflash;
        const std::uint32_t sets = dflashSnapSets(false, true, d.gdn_replay, d.dc.block, d.ngram, d.snap_env);
        const ::whirl::dflash::Mem dm = dflashFitBytes(d.dc, sizing.dflashWeightBytes(*d.file), cfg,
                                                       ::whirl::dflash::captureRows(d.params, d.dc.window, in.max_batch), in.parallel,
                                                       sets, st1);
        fi.fixed += dm.total();
        r.dflash_mem = dm;
        logI("VRAM estimate: DFlash2 drafter {:.2f} GiB (weights {:.2f}, capture {:.0f} MiB, rings {:.0f} MiB, scratch {:.0f} MiB, "
             "snapshot sets {:.0f} MiB); no MTP block",
             dm.total() / 1073741824.0, dm.weights / 1073741824.0, dm.capture / 1048576.0, dm.rings / 1048576.0,
             dm.scratch / 1048576.0, dm.snapshots / 1048576.0);
    }
    fi.fixed += (est.weights + fi.fixed) / 25;
    // the mode's KV format (numerics::chooseKv; the code object is not loaded yet: only dGPUs
    // reach here, and the R9700 code object has q4 KV kernels since FC-1c)
    // (WHIRL_DUMP_KV: one "D kv: site=server.est_dgpu" line, arch025-S0a)
    const KvFormat kv_est = numerics::kvEstimate(numerics::KvEstSite::server_dgpu, in.nreq, cfg.moe, in.kv_req);
    // (q4 KV uses 256 attention splits: more split-K partial buffers)
    const std::uint32_t fd_splits_est = sizing.resolveFdSplits(kv_est);
    const std::uint64_t b2 = sizing.bufferBytes(cfg, 2048, est.ffs, est.max_elems, fd_splits_est);
    const std::uint64_t b4 = sizing.bufferBytes(cfg, 4096, est.ffs, est.max_elems, fd_splits_est);
    fi.buf_per_row = (b4 - b2) / 2048;
    fi.buf_fixed = b2 - fi.buf_per_row * 2048;
    // (WHIRL_CKPT_HOST=1: checkpoints in pinned host memory, no VRAM)
    fi.ckpt_bytes = in.ckpt_host ? 0 : st1 + static_cast<std::uint64_t>(cfg.n_embd) * 4 + static_cast<std::uint64_t>(cfg.n_vocab) * 4;
    fi.parallel = in.parallel;
    fi.kv_per_token = est.kvPerToken(kv_est);
    fi.pool_min_tokens = in.pool_min_tokens;
    fi.batch = in.max_batch;
    fi.n_ck = in.n_ck;
    fi.n_spe = in.n_spe;
    fi.batch_fixed = in.batch_fixed;
    fi.ck_fixed = in.ck_fixed;
    r.plan = vram::planFit(fi);
    return r;
}

// --- slots, DFlash2 setup, snapshot sets, then the KV pool ---------------------------------------------------------

using BatchDrafts = std::array<std::uint32_t, qwen35::gdn_max_seg + 1>;

struct SlotSetupIn {
    std::uint32_t parallel = 1;
    std::uint32_t slot_ctx = 0;
    std::uint32_t n_layer = 0;               // cfg.n_layer
    bool use_mtp = false;
    bool use_dflash = false;
    std::uint32_t drafts = 0;                // MTP drafts per cycle of one slot
    bool user_batch_drafts = false;          // WHIRL_MTP_BATCH_DRAFTS set
    ::whirl::dflash::Params dfl_par;
    bool ngram = true;                       // WHIRL_NGRAM
    std::optional<std::string> snap_sets_env;  // WHIRL_SNAP_SETS (raw)
};

// setupSlots' result (the engine's draft caps and literal snapshot sets); only setupSlots makes one, and allocKvPool
// needs it
class SlotSetup {
public:
    BatchDrafts batch_drafts{};
    std::uint32_t lit_snap_sets = 0;  // engine: snapshot sets for n-gram / DFlash2 drafts (0 = all)

private:
    SlotSetup() = default;
    template <class M>
    friend SlotSetup setupSlots(M& model, device::Backend& ops, const SlotSetupIn& in, const BatchDrafts& batch_drafts_in);
};

template <class M>
SlotSetup setupSlots(M& model, device::Backend& ops, const SlotSetupIn& in, const BatchDrafts& batch_drafts_in) {
    SlotSetup out;
    out.batch_drafts = batch_drafts_in;
    BatchDrafts& batch_drafts = out.batch_drafts;
    model.setupSeqs(in.parallel, in.slot_ctx);
    if (in.use_dflash) {
        // per-slot drafter context rings and the draft scratch (before the KV pool: it is sized
        // from the VRAM left after them)
        const hip::MemInfo r0 = ops.memInfo();
        model.dflashSetup(in.parallel, model.cap_rows);
        const hip::MemInfo r1 = ops.memInfo();
        logI("DFlash2 speculative decoding: on, block {} (up to {} drafts per cycle), context ring {} rows x {} slots, VRAM +{:.0f} MiB; "
             "p-min {:.2f}, n-min {}; n-gram drafts {}",
             model.dfr->block, model.dfr->block - 1, model.dfr->ring, in.parallel,
             static_cast<double>(r0.free > r1.free ? r0.free - r1.free : 0) / (1024.0 * 1024.0),
             static_cast<double>(in.dfl_par.p_min), in.dfl_par.n_min,
             in.ngram ? "on (chosen per cycle by the cost model)" : "off");
    }
    if (model.gdn_replay) {
        logI("DeltaNet verify: replay of kept rows ({:.1f} MiB per slot), no snapshot sets",
             static_cast<double>(model.pendLayerBytes() * model.gdn_ord[in.n_layer - 1]) / (1024.0 * 1024.0));
    }
    if (in.use_mtp && !model.gdn_replay) {
        std::optional<std::uint32_t> snap_env;
        if (auto v = in.snap_sets_env) {
            std::uint32_t x = 0;
            try {
                x = static_cast<std::uint32_t>(std::stoul(*v));
            } catch (const std::exception&) {
            }
            snap_env = x;
        }
        const SnapDraftCaps sc = snapshotDraftCaps(batch_drafts, in.drafts, in.parallel, in.user_batch_drafts, snap_env, model.relax().on);
        batch_drafts = sc.caps;
        out.lit_snap_sets = sc.lit_sets;
        model.ensureSnapshots(sc.sets);
        std::string tbl;
        for (std::uint32_t k = 1; k <= in.parallel; ++k) tbl += (k > 1 ? ", " : "") + std::to_string(std::min(batch_drafts[k], in.drafts));
        logI("MTP under load, DeltaNet snapshot sets: max drafts per cycle with 1..{} decoding slots: {{ {} }}; {} snapshot sets ({} for n-gram drafts)",
             in.parallel, tbl, model.snap_sets, out.lit_snap_sets > 0 ? std::to_string(out.lit_snap_sets) : std::string("all"));
    } else if (in.use_dflash && !model.gdn_replay) {
        // FR-SRV P2-5: without replay the literal drafts need snapshot sets (before the KV pool is sized)
        std::optional<std::uint32_t> snap_env;
        if (auto v = in.snap_sets_env) {
            try {
                snap_env = static_cast<std::uint32_t>(std::stoul(*v));
            } catch (const std::exception&) {
                logW("WHIRL_SNAP_SETS={} is not a number (ignored)", *v);
            }
        }
        model.ensureSnapshots(dflashSnapSets(in.use_mtp, in.use_dflash, model.gdn_replay, model.dfr->block, in.ngram, snap_env));
        std::uint64_t gdn_layers = 0;
        for (std::uint32_t li = 0; li < in.n_layer; ++li)
            if (model.ssm_state[li] != 0) ++gdn_layers;
        logI("DeltaNet verify: no replay kernels: {} snapshot sets of {:.1f} MiB for DFlash2 / n-gram drafts (up to {} drafts "
             "per cycle over all decoding slots)",
             model.snap_sets,
             static_cast<double>(gdn_layers * (model.convBytes() + model.ssmBytes())) / (1024.0 * 1024.0), model.snap_sets);
    }
    return out;
}

// The KV pool (sized from the VRAM left after setupSlots' rings, scratch and snapshot sets).
template <class M>
void allocKvPool(M& model, std::uint32_t tokens, const SlotSetup&) {
    model.allocKvPool(tokens);
}

// --- console control (the last startup step before runLoop) --------------------------------------------------------

// arch025-surv: the two calls of runServer moved here verbatim (mutation survivor pl4-ctrl-restore), so the server
// tests run them in a child with its own console and an inherited "ignore Ctrl+C" attribute
// (tests/server/server_tests.cpp testConsoleCtrlArm).
inline void armConsoleCtrl(platform::proc::CtrlHandler on_ctrl) {
    platform::proc::restoreCtrlC();  // an "ignore Ctrl+C" attribute inherited from the parent would hide Ctrl+C
    platform::proc::addCtrlHandler(on_ctrl);
}

}  // namespace whirl::server::startup
