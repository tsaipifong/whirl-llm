// Server engine, sampling buffers (arch025-N4a, from class Engine): the device / host buffers the token draws and
// the draft acceptance read the verify rows through (top-k of each row, the larger top-k of an open row, the full row,
// the relaxacc top-K, the per-slot control words), and sampling::RowSource over them. Main thread only.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "backend.h"
#include "device/backend.h"
#include "device/handles.h"
#include "sampling/sampler.h"
#include "whirl/model_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace whirl::server {

constexpr std::uint32_t k_small = 256;
constexpr std::uint32_t k_big = 16384;
constexpr std::uint32_t max_rows = qwen35::max_verify_rows;

// The members in their former Engine order (samp_dev_ .. ctl_host_): ~Engine destroys them at the same point and in
// the same order (the former SampleRows member, which held no resource, is this class's RowSource base).
class SampleBuffers final : public sampling::RowSource {
public:
    SampleBuffers(ServerModel& m, device::Backend& ops) : m_(m), ops_(ops) {}
    SampleBuffers(const SampleBuffers&) = delete;
    SampleBuffers& operator=(const SampleBuffers&) = delete;

    // Engine::allocState's sampling-buffer block (same allocations, same order): n_vocab from the model, ctl words
    // for `parallel` slots, the relaxacc top-relax_topk rows when relax (relaxacc on).
    void alloc(std::uint32_t parallel, bool relax, std::uint32_t relax_topk);
    // bytes of the top-k_small buffer: [max_rows * k_small] ids, [max_rows * k_small] logits, [max_rows] (max, sum)
    static std::size_t sampBytes();
    // device top-k of rows row0 .. row0+rows into dst (ids | vals | stats layout of dst_rows rows), at row out0
    void launchTopk(std::uint32_t row0, std::uint32_t rows, std::uint32_t k, float inv_t, DevPtr dst,
                    std::uint32_t dst_rows, std::uint32_t out0);
    // the first `rows` rows' top-k_small on the host (s.row_base = 0)
    void fetchCands(sampling::Sampler& s, std::uint32_t rows);
    DevPtr logitsRow(std::uint32_t r) const;

    // sampling::RowSource
    void prepareRow(sampling::Sampler& s, std::uint32_t r) override;
    const float* loadFullRow(std::uint32_t gr) override;
    float logitAt(std::uint32_t gr, std::uint32_t t) override;
    std::uint32_t nVocab() const override;
    std::uint8_t* inCands() override { return in_cands_.data(); }

    // the buffers the engine's decode cycle reads / launches into
    DevPtr sampDev() const { return samp_dev_.get(); }
    DevPtr relaxDev() const { return relax_dev_.get(); }
    std::vector<std::uint8_t>& sampHost() { return samp_host_; }
    std::vector<std::uint8_t>& relaxHost() { return relax_host_; }
    std::vector<float>& rowHost() { return row_host_; }
    std::vector<sampling::Cand>* cands() { return &cands_scratch_; }
    std::vector<std::int32_t>& ctlHost() { return ctl_host_; }
    // v0252-hostwait (WHIRL_CYCLE_SYNC bit 0; REVIEW-v0252-hostwait D1: the buffers belong here, SpecCycle holds
    // references only): pinned mirrors of ctlHost / sampHost / relaxHost (relaxacc on only) that the decode cycle's async
    // readbacks land in, and the event the cycle waits on once. allocPinned: once, after alloc (SpecCycle::initSync);
    // cycle_sync 0 never calls it (nothing allocated).
    void allocPinned();
    device::PinnedBuf& ctlPin() { return ctl_pin_; }
    device::PinnedBuf& sampPin() { return samp_pin_; }
    device::PinnedBuf& relaxPin() { return relax_pin_; }
    device::Event readbackEvent() const { return rb_ev_.get(); }

private:
    void loadRow(std::uint32_t gr);

    ServerModel& m_;
    device::Backend& ops_;
    device::DevBuf samp_dev_, big_dev_;
    device::DevBuf relax_dev_;  // relaxacc: top-K of the greedy verify rows [ids | vals | stats]
    std::vector<std::uint8_t> relax_host_;
    std::vector<std::uint8_t> samp_host_, big_host_;
    std::vector<float> row_host_;
    std::vector<std::uint8_t> in_cands_;
    std::vector<sampling::Cand> cands_scratch_;
    std::vector<std::int32_t> ctl_host_;
    // v0252-hostwait: after the former Engine members (new resources, destroyed first; the backend outlives them)
    device::PinnedBuf ctl_pin_, samp_pin_, relax_pin_;
    device::EventH rb_ev_;
};

}  // namespace whirl::server
