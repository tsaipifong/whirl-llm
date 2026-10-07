// The model operations the server engine needs, behind an interface so the
// engine (scheduling, prefix cache, shared prefix checkpoints, host tiers,
// sampling) can run against a host-memory mock in tests. The production
// implementation (backend_qwen35.cpp) forwards to whirl::qwen35::Model
// (include/whirl/model.h, owned by the model area); the semantics of every
// call are those of the Model member of the same name.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tier/device_ops.h"
#include "vision_iface.h"
#include "whirl/model.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace whirl::server {

using tier::DevPtr;
using qwen35::MSeg;
using qwen35::PSeg;
using qwen35::VSeg;

// One KV pool array (every attention layer, then the MTP layer): base
// address and bytes per token row. Page p of the array is at
// base + p * kv_page * row.
struct KvArr {
    DevPtr base = 0;
    std::uint64_t row = 0;
};

class ServerModel {
public:
    virtual ~ServerModel() = default;

    virtual const qwen35::Config& cfg() const = 0;
    virtual tier::Stream stream() const = 0;
    virtual bool hasMtp() const = 0;

    // ---- device buffers (see Model members of the same names)
    virtual DevPtr logits() const = 0;   // row r at logits + r * n_vocab * 4
    virtual DevPtr hn() const = 0;       // normed trunk hidden rows (n_embd f32 each)
    virtual DevPtr mtpH() const = 0;     // MTP hidden rows of the current sequence
    virtual DevPtr seqHid(std::uint32_t s) const = 0;
    virtual DevPtr ctlArea() const { return 0; }  // control words of every sequence (debug reads)
    // per-layer recurrent state of the current sequence (0 = attention layer)
    virtual std::span<const DevPtr> convState() const = 0;
    virtual std::span<const DevPtr> ssmState() const = 0;
    virtual std::uint64_t convBytes() const = 0;
    virtual std::uint64_t ssmBytes() const = 0;
    virtual std::uint32_t curSeq() const = 0;

    // ---- sizes and switches
    virtual std::uint32_t poolPages() const = 0;
    virtual std::uint32_t seqPages() const = 0;
    virtual std::uint32_t maxBatch() const = 0;
    virtual std::uint32_t snapSets() const = 0;
    virtual bool gdnReplay() const = 0;
    virtual bool fusedDecode() const = 0;
    virtual bool canSegment() const = 0;
    virtual std::vector<KvArr> kvArrays() const = 0;

    // ---- sequences
    virtual void selectSeq(std::uint32_t s) = 0;
    virtual void commitSeq(std::uint32_t s) = 0;
    virtual void dropPending(std::uint32_t s) = 0;
    virtual void setPending(std::uint32_t s, std::uint32_t rows) = 0;
    virtual void restoreSeqSnapshot(std::uint32_t s, std::uint32_t set) = 0;
    virtual void mapPages(std::uint32_t s, std::uint32_t first, std::span<const std::int32_t> phys) = 0;
    virtual void reset() = 0;

    // ---- forward
    virtual void forward(std::span<const std::uint32_t> tokens, std::uint32_t pos0) = 0;
    virtual void prefillMtpChunk(std::span<const std::uint32_t> tokens, std::uint32_t pos0, std::size_t off,
                                 std::size_t n, std::optional<DevPtr> prev_hidden) = 0;
    virtual void prefillSegs(std::span<const PSeg> segs, bool with_mtp) = 0;
    virtual void segLogitsToFront(std::uint32_t k) = 0;
    virtual std::uint32_t argmax() = 0;

    // ---- speculative decoding
    virtual void setDraftCutoff(float p_min, std::uint32_t n_min) = 0;
    virtual void mtpBatchStepEx(std::span<const MSeg> segs, std::uint32_t r, bool draft) = 0;
    virtual std::uint32_t verifyBatchEnqueue(std::span<const VSeg> segs) = 0;
    virtual void readCtl(std::span<std::int32_t> dst) = 0;
    virtual void mtpEnqueue(DevPtr hidden, std::span<const std::uint32_t> tokens, std::uint32_t pos0) = 0;

    // ---- sampling: topk_rows over logits rows row0 .. row0 + rows - 1 into
    // ids [rows][k] i32, vals [rows][k] f32, stats [rows][2] {max, sum exp((x - max) * inv_t)}
    virtual void topkRows(std::uint32_t row0, std::uint32_t rows, std::uint32_t k, float inv_t, DevPtr ids,
                          DevPtr vals, DevPtr stats) = 0;

    // ---- vision (image prompts; see whirl/vismap.h). Defaults: no image support.
    // image map of sequence s for the following forwards (nullptr: text only)
    virtual void setVisMap(std::uint32_t s, const qwen35::VisMap* vm) { (void)s, (void)vm; }
    // the model has the multi-section RoPE attention prep (attn_prep_m)
    virtual bool hasMrope() const { return false; }
    // prefill scratch the encoder may borrow between forwards: {ptr, bytes}
    virtual std::size_t lendScratch(std::span<std::array<std::uint64_t, 2>> out) const { (void)out; return 0; }
    // the lent scratch was overwritten (drop cached activation copies)
    virtual void scratchClobbered() {}
    // free the RoPE row buffer when no sequence has an image map
    virtual void visRelease() {}
    // the RoPE row buffer is allocated
    virtual bool visRowsAllocated() const { return false; }

    // ---- optional profiling (WHIRL_PROFILE): per-op-class GPU times of the
    // MTP steps (which = 0) and the verify (which = 1) of decode cycles with
    // `slots` decoding slots
    static constexpr std::size_t n_prof_classes = 10;
    virtual void profileStart(int which, std::size_t slots) { (void)which, (void)slots; }
    virtual void profileStop() {}
    virtual void profileFlush(std::size_t slots) { (void)slots; }
    // accumulated ms per op class (and reset)
    virtual void profileTake(std::size_t slots, std::array<double, n_prof_classes>& mtp_ms,
                             std::array<double, n_prof_classes>& ver_ms) {
        (void)slots, (void)mtp_ms, (void)ver_ms;
    }
    virtual const char* profileClassName(std::size_t i) const { (void)i; return "?"; }
    // most rows of one batched verify / MTP step (qwen35::max_verify_rows with the wide GEMV path)
    virtual std::uint32_t verifyRows() const { return qwen35::max_small_batch; }
};

// Production backend over a loaded qwen35::Model (setupSeqs / allocKvPool done).
std::unique_ptr<ServerModel> makeQwen35Backend(qwen35::Model& m);

// Production image encoder for the server (vision::Vision on the model stream).
std::unique_ptr<ServerVision> makeQwen35Vision(vision::Vision& v, qwen35::Model& m);

}  // namespace whirl::server
