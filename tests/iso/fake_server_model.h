// A minimal server::ServerModel for the isolation test programs of the engine components that hold a ServerModel&
// (server.sample_buffers, server.image_embeddings; arch025-lane1 leg 8; server.cycle_prof, server.drafter_ctx, server.slot_cache). Header only, no tokenizer, no GGUF: the
// server tests' MockModel needs a real Tokenizer (lib whirl), which these programs must not link. The config has
// n_vocab / n_embd as given; logits / hidden rows live in device::NullBackend memory (host addresses); topkRows is the
// host reference of tests/server/mock_backend.cpp (MockModel::topkRows). Every other call does nothing.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "device/null_backend.h"
#include "server/backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace whirl::test {

class FakeServerModel final : public server::ServerModel {
public:
    FakeServerModel(device::Backend& ops, std::uint32_t n_vocab, std::uint32_t n_embd = 8) : ops_(ops) {
        cfg_.n_vocab = n_vocab;
        cfg_.n_embd = n_embd;
        logits_ = ops_.malloc(static_cast<std::uint64_t>(n_vocab) * 4 * qwen35::max_small_batch);
        hn_ = ops_.malloc(static_cast<std::uint64_t>(qwen35::max_small_batch) * n_embd * 4);
    }
    ~FakeServerModel() override {
        ops_.free(logits_);
        ops_.free(hn_);
    }
    FakeServerModel(const FakeServerModel&) = delete;
    FakeServerModel& operator=(const FakeServerModel&) = delete;

    const qwen35::Config& cfg() const override { return cfg_; }
    device::Stream stream() const override { return {}; }
    server::DevPtr logits() const override { return logits_; }
    server::DevPtr hn() const override { return hn_; }
    server::DevPtr mtpH() const override { return hn_; }
    server::DevPtr seqHid(std::uint32_t) const override { return hn_; }
    std::span<const server::DevPtr> convState() const override { return {}; }
    std::span<const server::DevPtr> ssmState() const override { return {}; }
    std::uint64_t convBytes() const override { return 0; }
    std::uint64_t ssmBytes() const override { return 0; }
    std::uint32_t curSeq() const override { return 0; }
    std::uint32_t poolPages() const override { return pool_pages; }
    std::uint32_t seqPages() const override { return seq_pages; }
    std::uint32_t maxBatch() const override { return qwen35::max_small_batch; }
    std::uint32_t snapSets() const override { return 0; }
    void ensureSnapshots(std::uint32_t) override {}
    bool gdnReplay() const override { return false; }
    bool fusedDecode() const override { return false; }
    bool canSegment() const override { return false; }
    std::vector<server::KvArr> kvArrays() const override { return kv_arrs; }
    void selectSeq(std::uint32_t) override {}
    void commitSeq(std::uint32_t) override {}
    void dropPending(std::uint32_t) override {}
    void setPending(std::uint32_t, std::uint32_t) override {}
    void restoreSeqSnapshot(std::uint32_t, std::uint32_t) override {}
    void mapPages(std::uint32_t, std::uint32_t, std::span<const std::int32_t>) override {}
    void reset() override {}
    void forward(std::span<const std::uint32_t>, std::uint32_t) override {}
    void prefillMtpChunk(std::span<const std::uint32_t>, std::uint32_t, std::size_t, std::size_t, std::optional<server::DevPtr>) override {}
    void prefillSegs(std::span<const server::PSeg>, bool) override {}
    void segLogitsToFront(std::uint32_t) override {}
    std::uint32_t argmax() override { return 0; }
    void setDraftCutoff(float, std::uint32_t) override {}
    void mtpBatchStepEx(std::span<const server::MSeg>, std::uint32_t, bool) override {}
    std::uint32_t verifyBatchEnqueue(std::span<const server::VSeg>) override { return 0; }
    void readCtl(std::span<std::int32_t>) override {}
    void mtpEnqueue(server::DevPtr, std::span<const std::uint32_t>, std::uint32_t) override {}
    // (tests/server/mock_backend.cpp MockModel::topkRows)
    void topkRows(std::uint32_t row0, std::uint32_t rows, std::uint32_t k, float inv_t, server::DevPtr ids, server::DevPtr vals,
                  server::DevPtr stats) override {
        const std::uint32_t V = cfg_.n_vocab;
        std::vector<std::uint32_t> idx(V);
        for (std::uint32_t r = 0; r < rows; ++r) {
            const float* x = at<const float>(logits_) + static_cast<std::uint64_t>(row0 + r) * V;
            float m = -INFINITY;
            for (std::uint32_t i = 0; i < V; ++i) m = std::max(m, x[i]);
            double sum = 0;
            for (std::uint32_t i = 0; i < V; ++i) sum += std::exp(static_cast<double>(x[i] - m) * inv_t);
            at<float>(stats)[2 * r] = m;
            at<float>(stats)[2 * r + 1] = static_cast<float>(sum);
            std::iota(idx.begin(), idx.end(), 0u);
            const std::uint32_t kk = std::min(k, V);
            std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](std::uint32_t a, std::uint32_t b) {
                if (x[a] != x[b]) return x[a] > x[b];
                return a < b;
            });
            for (std::uint32_t j = 0; j < kk; ++j) {
                at<std::int32_t>(ids)[static_cast<std::uint64_t>(r) * k + j] = static_cast<std::int32_t>(idx[j]);
                at<float>(vals)[static_cast<std::uint64_t>(r) * k + j] = x[idx[j]];
            }
        }
    }

    // vision calls ImageEmbeddings makes (counted)
    void setVisMap(std::uint32_t, const qwen35::VisMap* vm) override { last_vismap = vm; }
    void visRelease() override { ++vis_releases; }
    const qwen35::VisMap* last_vismap = nullptr;
    int vis_releases = 0;

    // WHIRL_PROFILE (arch025-engine S2, server.cycle_prof test): the per-op-class ms profileTake hands out (and resets),
    // and the class names; all 0 / "?" by default, as ServerModel's own defaults
    std::array<double, n_prof_classes> prof_mtp{}, prof_ver{};
    std::vector<const char*> prof_names;
    void profileTake(std::size_t, std::array<double, n_prof_classes>& mtp_ms, std::array<double, n_prof_classes>& ver_ms) override {
        mtp_ms = prof_mtp;
        ver_ms = prof_ver;
        prof_mtp = {};
        prof_ver = {};
    }
    const char* profileClassName(std::size_t i) const override { return i < prof_names.size() ? prof_names[i] : "?"; }

    // DFlash2 drafter (arch025-engine S4, server.drafter_ctx test): settable dflashMaxDrafts / captureN (0 by default,
    // as ServerModel's own defaults); dflashTrim / dflashIngest are recorded in call order
    std::uint32_t dfl_max_drafts = 0, capture_n = 0;
    struct DflCall {
        char kind;  // 't' dflashTrim(s, keep), 'i' dflashIngest(s, row0, n, pos0)
        std::uint32_t s, a, b, c;
    };
    std::vector<DflCall> dfl_calls;
    std::uint32_t dflashMaxDrafts() const override { return dfl_max_drafts; }
    std::uint32_t captureN() const override { return capture_n; }
    void dflashIngest(std::uint32_t s, std::uint32_t row0, std::uint32_t n, std::uint32_t pos0) override {
        dfl_calls.push_back({'i', s, row0, n, pos0});
    }
    void dflashTrim(std::uint32_t s, std::uint32_t keep) override { dfl_calls.push_back({'t', s, keep, 0, 0}); }

    // KV pool (arch025-engine S7, server.slot_cache test: initPool, ensurePages): settable poolPages / seqPages / kvArrays (0 / none by default,
    // as before)
    std::uint32_t pool_pages = 0, seq_pages = 0;
    std::vector<server::KvArr> kv_arrs;
    // KV pool re-allocation (arch025-engine S8, server.governor test: govResize): off by default = the interface default
    // (false, nothing changes); on: the pool becomes tokens / kv_page pages (the arrays stay, sized by the test for the most)
    bool resize_ok = false;
    bool resizePool(std::uint32_t tokens) override {
        if (!resize_ok) return false;
        pool_pages = tokens / qwen35::kv_page;
        return true;
    }

private:
    template <class T>
    static T* at(server::DevPtr p) {
        return reinterpret_cast<T*>(static_cast<std::uintptr_t>(p));
    }
    device::Backend& ops_;
    qwen35::Config cfg_;
    server::DevPtr logits_ = 0, hn_ = 0;
};

}  // namespace whirl::test
