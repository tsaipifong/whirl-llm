// Server engine, sampling buffers (arch025-N4a: moved verbatim from engine.cpp; Engine:: -> SampleBuffers::).
// SPDX-License-Identifier: Apache-2.0

#include "engine_sampling.h"

#include <algorithm>
#include <cstring>

namespace whirl::server {

void SampleBuffers::alloc(std::uint32_t parallel, bool relax, std::uint32_t relax_topk) {
    const std::uint32_t V = m_.cfg().n_vocab;
    const std::uint64_t samp_bytes = sampBytes();
    const std::uint64_t big_bytes = static_cast<std::uint64_t>(k_big) * 8 + 8;
    samp_dev_ = device::DevBuf::alloc(ops_, samp_bytes);
    big_dev_ = device::DevBuf::alloc(ops_, big_bytes);
    samp_host_.assign(samp_bytes, 0);
    big_host_.assign(big_bytes, 0);
    row_host_.assign(V, 0.0f);
    in_cands_.assign(V, 0);
    cands_scratch_.resize(std::max<std::size_t>(V, k_big));
    ctl_host_.assign(static_cast<std::size_t>(parallel) * qwen35::ctl_words, 0);
    if (relax) {
        const std::uint64_t rb = static_cast<std::uint64_t>(max_rows) * relax_topk * 8 + max_rows * 8;
        relax_dev_ = device::DevBuf::alloc(ops_, rb);
        relax_host_.assign(rb, 0);
    }
}

DevPtr SampleBuffers::logitsRow(std::uint32_t r) const {
    return m_.logits() + static_cast<std::uint64_t>(r) * m_.cfg().n_vocab * 4;
}

std::size_t SampleBuffers::sampBytes() {
    return static_cast<std::size_t>(max_rows) * k_small * 8 + max_rows * 8;
}

void SampleBuffers::launchTopk(std::uint32_t row0, std::uint32_t rows, std::uint32_t k, float inv_t, DevPtr dst,
                               std::uint32_t dst_rows, std::uint32_t out0) {
    const DevPtr ids = dst + static_cast<std::uint64_t>(out0) * k * 4;
    const DevPtr vals = dst + static_cast<std::uint64_t>(dst_rows) * k * 4 + static_cast<std::uint64_t>(out0) * k * 4;
    const DevPtr stats = dst + static_cast<std::uint64_t>(dst_rows) * k * 8 + static_cast<std::uint64_t>(out0) * 8;
    m_.topkRows(row0, rows, k, inv_t, ids, vals, stats);
}

void SampleBuffers::fetchCands(sampling::Sampler& s, std::uint32_t rows) {
    s.row_base = 0;
    launchTopk(0, rows, k_small, s.inv_t, samp_dev_.get(), max_rows, 0);
    ops_.download(samp_host_.data(), samp_dev_.get(), sampBytes());
}

void SampleBuffers::prepareRow(sampling::Sampler& s, std::uint32_t r) {
    const std::uint32_t gr = s.row_base + r;
    s.row = gr;
    const auto* host = samp_host_.data();
    const auto* ids = reinterpret_cast<const std::int32_t*>(host);
    const auto* vals = reinterpret_cast<const float*>(host + static_cast<std::size_t>(max_rows) * k_small * 4);
    const auto* stats = reinterpret_cast<const float*>(host + static_cast<std::size_t>(max_rows) * k_small * 8);
    s.m = stats[2 * gr];
    s.sum = stats[2 * gr + 1];
    std::vector<sampling::Cand>& c = *s.cands;
    for (std::uint32_t j = 0; j < k_small; ++j) {
        const std::size_t idx = static_cast<std::size_t>(gr) * k_small + j;
        c[j] = {static_cast<std::uint32_t>(ids[idx]), vals[idx], 0};
    }
    if (sampling::filterCands(s, k_small, false)) return;
    // escalate: a larger device top-k for this row, then the full row
    s.big_used += 1;
    launchTopk(gr, 1, k_big, s.inv_t, big_dev_.get(), 1, 0);
    ops_.download(big_host_.data(), big_dev_.get(), big_host_.size());
    {
        const auto* bids = reinterpret_cast<const std::int32_t*>(big_host_.data());
        const auto* bvals = reinterpret_cast<const float*>(big_host_.data() + static_cast<std::size_t>(k_big) * 4);
        const auto* bst = reinterpret_cast<const float*>(big_host_.data() + static_cast<std::size_t>(k_big) * 8);
        s.m = bst[0];
        s.sum = bst[1];
        for (std::uint32_t j = 0; j < k_big; ++j) c[j] = {static_cast<std::uint32_t>(bids[j]), bvals[j], 0};
    }
    if (sampling::filterCands(s, k_big, false)) return;
    s.full_used += 1;
    loadRow(gr);
    const std::uint32_t V = m_.cfg().n_vocab;
    for (std::uint32_t j = 0; j < V; ++j) c[j] = {j, row_host_[j], 0};
    sampling::filterCands(s, V, true);
}

void SampleBuffers::loadRow(std::uint32_t gr) {
    const std::uint32_t V = m_.cfg().n_vocab;
    ops_.download(row_host_.data(), logitsRow(gr), static_cast<std::uint64_t>(V) * 4);
}

const float* SampleBuffers::loadFullRow(std::uint32_t gr) {
    loadRow(gr);
    return row_host_.data();
}

float SampleBuffers::logitAt(std::uint32_t gr, std::uint32_t t) {
    float l = 0;
    ops_.download(&l, logitsRow(gr) + static_cast<std::uint64_t>(t) * 4, 4);
    return l;
}

std::uint32_t SampleBuffers::nVocab() const { return m_.cfg().n_vocab; }

// v0252-e1p1 (feat026-acc port: Engine::allocState's pinned block and Engine::readCycleCtl of feat026-acc-p25 4ac9da6,
// here with the control words they read; the engine logs the mode)
std::uint32_t SampleBuffers::allocSpecAcc(std::uint32_t parallel, std::uint32_t mode) {
    if (mode == 0 || mode > 4) return 0;
    const std::uint64_t cb = static_cast<std::uint64_t>(parallel) * qwen35::ctl_words * 4;
    ctl_pin_ = device::PinnedBuf::alloc(ops_, cb);
    std::memset(ctl_pin_.get(), 0, cb);
    // the device accept needs the kernel (else the pinned read + the host accept)
    if (mode >= 2 && !m_.specAcceptOk()) mode = 1;
    acc_mode_ = mode;  // v0252-e1p1c: allocAhead needs a device mode
    return mode;
}

// v0252-e1p1c (E1 one-cycle lookahead): the ring of two entries (engine_sampling.h)
bool SampleBuffers::allocAhead(std::uint32_t parallel, std::uint32_t rows) {
    if (acc_mode_ < 2 || parallel == 0 || rows == 0) return false;
    ring_words_ = static_cast<std::uint64_t>(parallel) * qwen35::ctl_words;
    ring_pin_ = device::PinnedBuf::alloc(ops_, 2 * ring_words_ * 4);
    std::memset(ring_pin_.get(), 0, 2 * ring_words_ * 4);
    ring_hid_bytes_ = static_cast<std::uint64_t>(rows) * m_.cfg().n_embd * 4;
    ring_hid_ = device::DevBuf::alloc(ops_, 2 * ring_hid_bytes_);
    for (device::EventH& ev : ring_ev_) ev = device::EventH::create(ops_);
    return true;
}

DevPtr SampleBuffers::aheadHid(std::uint32_t e) const { return ring_hid_.get() + static_cast<std::uint64_t>(e & 1u) * ring_hid_bytes_; }

void SampleBuffers::aheadSubmit(std::uint32_t e, const SpecAccArgs& aa, std::uint32_t seq, std::uint32_t row0, std::uint32_t rows) {
    e &= 1u;
    m_.specAccept(aa, static_cast<std::int32_t*>(ring_pin_.get()) + e * ring_words_);
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(m_.cfg().n_embd) * 4;
    const std::uint64_t bytes = std::min(static_cast<std::uint64_t>(rows) * row_bytes, ring_hid_bytes_);
    const DevPtr src = m_.hn() + static_cast<std::uint64_t>(row0) * row_bytes;
    // (the keep first: it must hold the verify's rows whatever the second copy's destination is)
    ops_.copyAsync(aheadHid(e), src, bytes, m_.stream());
    ops_.copyAsync(m_.seqHid(seq), src, bytes, m_.stream());
    ops_.eventRecord(ring_ev_[e].get(), m_.stream());
}

const std::int32_t* SampleBuffers::aheadWait(std::uint32_t e) {
    e &= 1u;
    ops_.eventSync(ring_ev_[e].get());
    return static_cast<const std::int32_t*>(ring_pin_.get()) + e * ring_words_;
}

const std::int32_t* SampleBuffers::readCycleCtl(std::uint32_t mode, std::size_t words, const SpecAccArgs* aa) {
    if (mode == 0 || !ctl_pin_) {
        m_.readCtl(std::span<std::int32_t>(ctl_host_.data(), std::min(words, ctl_host_.size())));
        return ctl_host_.data();
    }
    auto* pin = static_cast<std::int32_t*>(ctl_pin_.get());
    if (aa != nullptr && mode == 3) {
        // zero copy: the kernel writes the verify's control words into the pinned buffer; the host only waits for it
        m_.specAccept(*aa, pin);
        ops_.streamSync(m_.stream());
        return pin;
    }
    if (aa != nullptr) m_.specAccept(*aa, nullptr);
    m_.readCtlPinned(std::span<std::int32_t>(pin, std::min<std::size_t>(words, ctl_pin_.bytes() / 4)));
    return pin;
}

}  // namespace whirl::server
