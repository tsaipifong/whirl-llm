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

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace whirl::server {

constexpr std::uint32_t k_small = 256;
constexpr std::uint32_t k_big = 16384;
constexpr std::uint32_t max_rows = qwen35::max_verify_rows;

// v0252-e1p1 (feat026-acc port, WHIRL_SPEC_ACC): slot cycles decided on the device; device results out of range in gpu /
// gpu-zc (the host decided instead: a broken result, never expected); compared with the host path and disagreeing (check
// only); next-cycle row tables compared / disagreeing (check only). Engine totals since start (Engine::stats, the
// per-request "spec-acc stats" line). Main thread only.
struct SpecAccStats {
    std::uint64_t dev = 0, fallback = 0, checked = 0, mismatch = 0, tab_checked = 0, tab_mismatch = 0;
    std::uint64_t e1 = 0;  // v0252-e1p1b (WHIRL_E1): slot cycles that took their inputs from the device (VSeg::dev)
    // v0252-e1p1c (WHIRL_E1=2, the lookahead): cycles read through the ring; of those, the ones submitted before the cycle
    // before them was read; submitted cycles dropped because the request ended at the cycle before them
    std::uint64_t ring = 0, ahead = 0, discard = 0;
};

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

    // ---- v0252-e1p1 (feat026-acc port, WHIRL_SPEC_ACC; EngineOptions::spec_acc) ----
    // Engine::allocState, after alloc: for mode >= 1 the control words of `parallel` slots in pinned host memory; a device
    // mode (2 gpu, 3 gpu-zc, 4 check) without the model's accept kernel falls back to 1 (pinned read + host accept).
    // Returns the mode in effect (0 for 0 or a value above 4).
    std::uint32_t allocSpecAcc(std::uint32_t parallel, std::uint32_t mode);
    // THE read of a decode cycle's control words (SpecCycle::sampleRows, after the verify and its argmax are enqueued); the
    // host decides from what it returns (`words` = slots * ctl_words):
    //   0 host     the pageable readCtl into ctlHost()                              (as before feat026-acc)
    //   1 pinned   readCtlPinned into the pinned buffer (async copy + stream sync)
    //   2 gpu / 4 check   aa != null: the accept kernel (specAccept, host = null), then the pinned read
    //   3 gpu-zc   aa != null: the accept kernel writes the verify's control areas into the pinned buffer itself; the host
    //              only syncs the model stream - no D2H copy command (E1 rule R-c)
    // aa == null in a device mode: the pinned read only. The pointer stays valid until the next call.
    const std::int32_t* readCycleCtl(std::uint32_t mode, std::size_t words, const SpecAccArgs* aa);
    // the counters: accStats() to count (the decode cycle), accCounts() to read (Engine::stats, the per-request line; a
    // const-only name, so the engine split census sees a read)
    SpecAccStats& accStats() { return acc_stats_; }
    const SpecAccStats& accCounts() const { return acc_stats_; }

    // ---- v0252-e1p1c (E1 one-cycle lookahead, WHIRL_E1=2; SpecCycle::aheadCycle / aheadBuild) ----
    // The ring: two entries, each `parallel` control areas in pinned host memory (readCtl's layout), one event and the
    // hidden rows of one cycle (`rows` rows) in device memory. Cycle n writes entry n % 2, so the host can read cycle n
    // after cycle n + 1 was submitted - and after it finished.
    // allocAhead (Engine::allocState, after allocSpecAcc): false without a device accept mode (nothing allocated).
    bool allocAhead(std::uint32_t parallel, std::uint32_t rows);
    bool aheadOk() const { return static_cast<bool>(ring_pin_); }
    // The submit side of entry e, right after the cycle's verify was enqueued: the accept kernel decides and writes the
    // control areas into the entry (zero copy: no D2H copy command, E1 rule R-c); the cycle's `rows` hidden rows from
    // verify row row0 go to the entry's keep and to sequence seq's MTP input (the next cycle's step 0 reads them); ONE
    // event is recorded behind all of it (R-b).
    void aheadSubmit(std::uint32_t e, const SpecAccArgs& aa, std::uint32_t seq, std::uint32_t row0, std::uint32_t rows);
    // The host's one wait of the cycle in entry e: its event only (a cycle submitted after it goes on running; no event
    // query, no stream sync, no download: R-a / R-e), then the entry's control words (valid until the entry is submitted
    // again).
    const std::int32_t* aheadWait(std::uint32_t e);
    // entry e's kept hidden rows (row r at + r * n_embd * 4)
    DevPtr aheadHid(std::uint32_t e) const;

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
    // v0252-e1p1 (feat026-acc port): the pinned control words (WHIRL_SPEC_ACC >= 1) and the device-accept counters; last, so
    // every member before them keeps its place (tests/pins/engine_member_order.txt audit row)
    device::PinnedBuf ctl_pin_;
    SpecAccStats acc_stats_;
    // v0252-e1p1c (the lookahead ring; last, as above): the two control-word areas, the two hidden-row keeps (one device
    // buffer), the two events, the sizes of one entry and the accept mode allocSpecAcc returned
    device::PinnedBuf ring_pin_;
    device::DevBuf ring_hid_;
    std::array<device::EventH, 2> ring_ev_;
    std::uint64_t ring_words_ = 0, ring_hid_bytes_ = 0;
    std::uint32_t acc_mode_ = 0;
};

}  // namespace whirl::server
