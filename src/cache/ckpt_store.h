// Checkpoint buffers of the server's prefix cache (arch024 M2): the recurrent state (conv / ssm per DeltaNet layer),
// the MTP hidden row and the logits row saved at a prefix position, so a later request with the same prefix resumes
// there. CkptStore makes, fills, reads and frees them; the buffers belong to the Ckpt (device::DevBuf /
// device::PinnedBuf), so a Ckpt that goes away (vector clear / resize, exception unwinding, move-assign) frees them.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Moved out of Engine (initCkpt / freeCkpt / saveCkpt / loadCkpt / ckptVram) without a change of
// allocation size, order or timing. Prior art: progress/arch024-S3.PRIOR_ART.md (llama.cpp's container-owned
// checkpoints; vLLM / PyTorch: the holder of the fence waits before the buffer is given back).
//
// F1 (never free what the GPU may still read): a spill to the RAM tier reads the checkpoint buffers on the tier
// stream. The fence of that spill belongs to the slot (one event per slot spill, not per checkpoint), so CkptStore
// does not know it: whoever frees checkpoints alone passes the fences to park / release, which wait for them
// (device::waitIfPending) before anything is freed. Slot / Spe wait in their own destructor and move-assign.

#pragma once

#include "cache/prefix_cache.h"
#include "device/backend.h"
#include "device/handles.h"
#include "cache/cache_model.h"  // not server/backend.h: the narrow model interface (arch025-lane2 D2.2)
#include "server/engine_thread.h"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <vector>

namespace whirl::server {

// index fields (valid, pos, seq, kind, has_logits): cache::CkptKey
struct Ckpt : cache::CkptKey {
    std::vector<DevPtr> conv, ssm;  // per layer (0: no DeltaNet state in that layer); slices of dev or host
    DevPtr hid = 0;                 // MTP hidden row: a slice of dev, or hid_buf with a pinned checkpoint
    DevPtr logits = 0;              // slice of dev or host
    // owned buffers (declaration order: destruction frees host before hid_buf, as freeCkpt did)
    device::DevBuf dev;      // the one VRAM allocation that conv / ssm / hid / logits are slices of
    device::DevBuf hid_buf;  // pinned checkpoint (WHIRL_CKPT_HOST): the hidden row stays in VRAM
    device::PinnedBuf host;  // pinned checkpoint: conv / ssm / logits
};

class CkptStore {
public:
    struct Layout {
        std::uint64_t conv_bytes = 0, ssm_bytes = 0;  // of one DeltaNet layer
        bool ckpt_host = false;                       // WHIRL_CKPT_HOST: per-slot checkpoints in pinned host memory
    };
    CkptStore(device::Backend& ops, CacheModel& m, Layout lay) : ops_(ops), m_(m), lay_(lay) {}
    // arch024 M8: the engine thread that may call the entry points below (nullptr: unchecked; docs/dev/threads.md)
    void setOwner(const EngineThread* o) { owner_ = o; }
    const EngineThread* ownerThread() const { return owner_; }

    // Allocates c's buffers (c holds none). host_ok: a per-slot checkpoint (pinned with WHIRL_CKPT_HOST); shared
    // prefix checkpoints are always in VRAM. On an allocation error the buffers already made stay in c.
    void init(Ckpt& c, bool host_ok);
    // the current sequence's recurrent state (after commitSeq) and the given rows into c, on the model stream
    void save(Ckpt& c, std::optional<DevPtr> hid, std::optional<DevPtr> logits);
    // c's recurrent state into the current sequence (its pending rows dropped), on the model stream
    void load(const Ckpt& c);
    // VRAM of one checkpoint (slot_ckpt: a per-slot one)
    std::uint64_t bytes(bool slot_ckpt) const;
    // F1: waits for every fence in wait_for (device::waitIfPending), then frees the buffers of cks / c (left empty,
    // index fields untouched). The fences are the caller's (the slot's spill_ev, a shared checkpoint's spill_ev /
    // ready_ev); {} only for buffers no spill has read (just allocated).
    void park(std::span<Ckpt> cks, std::initializer_list<device::Event> wait_for);
    void release(Ckpt& c, std::initializer_list<device::Event> wait_for);
    // the buffers again, in order (init each; an error leaves the ones made so far in place for release)
    void unpark(std::span<Ckpt> cks, bool host_ok);

private:
    void wait(std::initializer_list<device::Event> wait_for);
    static void free(Ckpt& c);
    device::Backend& ops_;
    CacheModel& m_;
    Layout lay_;
    const EngineThread* owner_ = nullptr;
};

}  // namespace whirl::server
