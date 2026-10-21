// Host-memory mock of the server's device and model interfaces, for testing
// the engine (scheduling, prefix cache, shared prefix checkpoints, host
// tiers, MTP / n-gram acceptance, sampling) without a GPU.
// SPDX-License-Identifier: Apache-2.0
//
// The mock model is a deterministic toy language model whose next-token
// distribution depends on the *whole* history as the engine hands it over:
//   - a recurrent hash H kept in every DeltaNet layer's conv / ssm state
//     buffers (each layer stores a layer-salted copy; inconsistent copies
//     "poison" the state),
//   - the KV rows of positions 0 .. p read back through the sequence's page
//     table (every attention layer and the MTP layer store (token, pos, H);
//     a row whose position field does not match "poisons" the output),
// so a wrong checkpoint, page table, page copy or tier restore changes the
// generated text, and every such inconsistency is counted (poison()).
// Verify follows the replay model (pending rows reach the state at the next
// verify / commitSeq). MTP drafts are the true continuation, corrupted with
// a deterministic probability, so acceptance varies but greedy output never
// depends on the drafts.

#pragma once

#include "server/backend.h"
#include "device/null_backend.h"
#include "whirl/tokenizer.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace whirl::test {

struct MockConfig {
    std::uint32_t n_layer = 8;     // attention every 4th layer
    std::uint32_t n_embd = 16;
    std::uint32_t parallel = 4;
    std::uint32_t slot_ctx = 16384;
    std::uint32_t pool_tokens = 65536;
    std::uint32_t max_batch = 4096;
    bool mtp = true;
    double draft_miss = 0.3;       // probability that a draft is wrong
    double eos_rate = 0.0;         // per-token probability that <|im_end|> is the argmax
    // simulated device time (busy wait; 0 = none): per prefill row, per verify cycle
    double prefill_us_per_row = 0;
    double cycle_us = 0;
    // > 0: the next token depends on the last token only, drawn from the first `markov` words
    // (greedy output cycles: n-gram drafts hit)
    std::uint32_t markov = 0;
    // > 0: chunk-schedule sensitivity (TAIL-CK bitwise guard): a prefill forward that starts at a
    // position that is not a multiple of sched_grid perturbs the recurrent state, like a GPU forward
    // split off the cold chunk schedule rounds differently; the greedy reference never splits, so
    // any off-grid prefill split shows up as a changed output
    std::uint32_t sched_grid = 0;
    // DFlash2 drafter (--draft dflash): block size, capture rows, probability that a draft is wrong
    bool dflash = false;
    std::uint32_t dfl_block = 8;
    std::uint32_t dfl_cap_rows = 2048;
    double dfl_miss = 0.3;
    // FR-DFL P2-3: a wrong draft is the target row's runner-up (logit 8.65 vs 10: p / p_max 0.26, inside
    // relaxacc's greedy top-k / alpha 0.1 window) instead of a random text token
    bool dfl_miss_runner_up = false;
    // FR-DFL P3-3: context ring rows (0 = unlimited map): an ingest of position p drops p - ring, p - 2 ring, ...
    // like the real ring; dfl_need_ctx > 0: a draft whose context lacks one of the min(pos, need) rows
    // before pos counts in dflHoles() (the engine called the slot ready with rows the ring no longer has)
    std::uint32_t dfl_ring = 0;
    std::uint32_t dfl_need_ctx = 0;
    // FR-SRV P2-5: false = the DeltaNet verify keeps snapshot sets instead of replaying kept rows
    // (gfx1151 / Radeon 8060S, WHIRL_GDN_REPLAY=0): the state after a verify holds every row, and
    // restoreSeqSnapshot(s, j) rolls it back to snapshot set j. Sets are allocated lazily by a verify
    // that needs them (like Model::verifyBatchEnqueue) or up front by ensureSnapshots.
    bool replay = true;
};

class MockModel final : public server::ServerModel {
public:
    MockModel(device::Backend& ops, const Tokenizer& tok, const MockConfig& mc);
    ~MockModel() override;

    const qwen35::Config& cfg() const override { return cfg_; }
    device::Stream stream() const override { return stream_; }
    bool hasMtp() const override { return mc_.mtp; }
    server::DevPtr logits() const override { return logits_; }
    server::DevPtr hn() const override { return hn_; }
    server::DevPtr mtpH() const override { return seqs_[cur_].hid; }
    server::DevPtr seqHid(std::uint32_t s) const override { return seqs_[s].hid; }
    std::span<const server::DevPtr> convState() const override { return seqs_[cur_].conv; }
    std::span<const server::DevPtr> ssmState() const override { return seqs_[cur_].ssm; }
    std::uint64_t convBytes() const override { return 32; }
    std::uint64_t ssmBytes() const override { return 64; }
    std::uint32_t curSeq() const override { return cur_; }
    std::uint32_t poolPages() const override { return pool_pages_; }
    std::uint32_t seqPages() const override { return seq_pages_; }
    std::uint32_t maxBatch() const override { return mc_.max_batch; }
    std::uint32_t snapSets() const override { return mc_.replay ? 0 : snap_sets_; }
    void ensureSnapshots(std::uint32_t n) override;
    bool gdnReplay() const override { return mc_.replay; }
    bool fusedDecode() const override { return true; }
    bool canSegment() const override { return true; }
    std::vector<server::KvArr> kvArrays() const override;
    // v0.2.2.2 memory governor: new pool arrays (the old rows are gone: a stale page table or a restore into the old
    // arrays poisons the output); fail_resize: the next resize fails (keeps the size)
    bool resizePool(std::uint32_t tokens) override;
    std::uint32_t resizes() const { return resizes_.load(); }
    std::atomic<bool> fail_resize{false};

    void selectSeq(std::uint32_t s) override { cur_ = s; }
    void commitSeq(std::uint32_t s) override;
    void dropPending(std::uint32_t s) override { seqs_[s].npend = 0; }
    void setPending(std::uint32_t s, std::uint32_t rows) override { seqs_[s].npend = rows; }
    void restoreSeqSnapshot(std::uint32_t, std::uint32_t) override;
    void mapPages(std::uint32_t s, std::uint32_t first, std::span<const std::int32_t> phys) override;
    void reset() override;

    void forward(std::span<const std::uint32_t> tokens, std::uint32_t pos0) override;
    void prefillMtpChunk(std::span<const std::uint32_t> tokens, std::uint32_t pos0, std::size_t off, std::size_t n,
                         std::optional<server::DevPtr> prev_hidden) override;
    void prefillSegs(std::span<const server::PSeg> segs, bool with_mtp) override;
    void segLogitsToFront(std::uint32_t k) override;
    std::uint32_t argmax() override;

    void setDraftCutoff(float, std::uint32_t) override {}
    void mtpBatchStepEx(std::span<const server::MSeg> segs, std::uint32_t r, bool draft) override;
    std::uint32_t verifyBatchEnqueue(std::span<const server::VSeg> segs) override;
    void readCtl(std::span<std::int32_t> dst) override;
    void mtpEnqueue(server::DevPtr hidden, std::span<const std::uint32_t> tokens, std::uint32_t pos0) override;
    void topkRows(std::uint32_t row0, std::uint32_t rows, std::uint32_t k, float inv_t, server::DevPtr ids,
                  server::DevPtr vals, server::DevPtr stats) override;

    bool hasDFlash() const override { return mc_.dflash; }
    std::uint32_t dflashMaxDrafts() const override { return mc_.dflash ? mc_.dfl_block - 1 : 0; }
    std::uint32_t captureN() const override;
    void dflashIngest(std::uint32_t s, std::uint32_t row0, std::uint32_t n, std::uint32_t pos0) override;
    std::vector<std::uint32_t> dflashDraft(std::uint32_t s, std::uint32_t anchor, std::uint32_t pos, float p_min,
                                           std::uint32_t n_min) override;
    void dflashTrim(std::uint32_t s, std::uint32_t keep) override;
    std::uint32_t dflashRing() const override { return mc_.dfl_ring; }
    std::uint64_t dflHoles() const { return dfl_holes_; }
    // DFlash2: ingested rows that did not match the capture (wrong row / sequence / position),
    // context rows from another lineage seen by a draft, draft blocks run
    std::uint64_t dflBad() const { return dfl_bad_; }
    std::uint64_t dflStale() const { return dfl_stale_; }
    std::uint64_t dflBlocks() const { return dfl_blocks_; }

    // Reference: greedy continuation of `prompt` computed from scratch
    // (independent of the engine), stopping at <|im_end|> / <|endoftext|>
    // unless ignore_eos, or after max_tokens.
    std::vector<std::uint32_t> greedyReference(std::span<const std::uint32_t> prompt, std::uint32_t max_tokens,
                                               bool ignore_eos) const;

    // Consistency violations seen (wrong state copies, page tables, KV rows).
    std::uint64_t poison() const { return poison_.load(); }
    // Hidden rows consumed with the wrong position (MTP bookkeeping).
    std::uint64_t hidMismatch() const { return hid_mismatch_.load(); }
    std::uint64_t rowsRun() const { return rows_run_; }
    // most rows of one prefill forward so far (like Model::forward, more than max_batch throws BatchTooLarge)
    std::uint32_t maxPrefillRows() const { return max_pf_rows_.load(); }

private:
    void prefillRows(std::size_t n);
    struct State {
        std::uint64_t h = 0;   // recurrent hash
        std::uint64_t kv = 0;  // fold of the KV rows 0 .. pos
        std::uint32_t pos = 0; // position of the last token folded in (+1), for hidden rows
    };
    struct PendRow {
        std::uint32_t tok, pos;
    };
    struct Seq {
        std::vector<server::DevPtr> conv, ssm;
        server::DevPtr hid = 0;
        std::vector<std::int32_t> ptab;
        std::vector<PendRow> pend;  // rows of the last verify
        std::uint32_t npend = 0;
        std::array<std::int32_t, qwen35::ctl_words> ctl{};
        State chain{};              // MTP draft chain
        std::uint32_t chain_pos = 0;
    };
    static std::uint64_t mix(std::uint64_t h, std::uint64_t a, std::uint64_t b);
    std::uint64_t readH(const Seq& s) const;
    void writeH(Seq& s, std::uint64_t h);
    void writeKv(Seq& s, std::uint32_t pos, std::uint32_t tok, std::uint64_t h);
    // fold of KV rows [0, pos) of sequence s (validated)
    std::uint64_t kvPrefix(const Seq& s, std::uint32_t pos) const;
    static std::uint64_t kvRow(std::uint32_t tok, std::uint32_t pos, std::uint64_t h);
    std::uint32_t nextToken(std::uint64_t h, std::uint64_t kv) const;
    std::uint32_t runnerUp(std::uint64_t h, std::uint64_t kv) const;
    void writeLogits(std::uint32_t row, std::uint64_t h, std::uint64_t kv);
    void writeHidden(server::DevPtr dst, const State& st) const;
    State readHidden(server::DevPtr src) const;
    // runs tokens through sequence s from its committed state, writing KV,
    // hidden rows (hn rows r0 ..), returns the state after the last token
    State runRows(Seq& s, std::span<const std::uint32_t> toks, std::uint32_t pos0, std::uint32_t r0, bool commit);
    void prefillOne(std::uint32_t s, std::span<const std::uint32_t> toks, std::uint32_t pos0, std::uint32_t r0,
                    std::uint32_t logits_row, std::optional<server::DevPtr> prev_hidden, bool mtp);
    std::uint32_t corruptDraft(std::uint32_t truth, std::uint64_t key) const;
    std::uint64_t stepH(std::uint64_t h, std::uint32_t tok, std::uint32_t pos) const;
    struct CapRow {
        std::uint32_t seq, tok;
        State st;
    };
    void capRow(std::uint32_t seq, std::uint32_t tok, const State& st);
    std::vector<CapRow> cap_;
    std::vector<std::map<std::uint32_t, State>> dfl_ctx_;
    std::uint64_t dfl_bad_ = 0, dfl_stale_ = 0, dfl_blocks_ = 0, dfl_holes_ = 0;

    device::Backend& ops_;
    const Tokenizer& tok_;
    MockConfig mc_;
    qwen35::Config cfg_;
    device::Stream stream_ = nullptr;
    server::DevPtr logits_ = 0, hn_ = 0;
    std::vector<Seq> seqs_;
    std::uint32_t cur_ = 0;
    std::uint32_t pool_pages_ = 0, seq_pages_ = 0;
    std::vector<server::DevPtr> kc_, vc_;  // per attention layer + MTP (last)
    std::vector<std::uint32_t> attn_layers_;
    std::vector<std::uint32_t> text_ids_;
    std::uint32_t im_end_ = 0, eot_ = 0;
    mutable std::atomic<std::uint64_t> poison_{0};
    std::atomic<std::uint32_t> resizes_{0};
    std::atomic<std::uint32_t> max_pf_rows_{0};
    mutable std::atomic<std::uint64_t> hid_mismatch_{0};
    std::uint64_t rows_run_ = 0;
    std::vector<std::uint32_t> seg_logits_rows_;
    // replay off: snapshot sets (sequence, recurrent hash after that verify row)
    struct Snap {
        std::uint32_t seq = ~0u;
        std::uint64_t h = 0;
    };
    std::vector<Snap> snaps_;
    std::uint32_t snap_sets_ = 0;
};

}  // namespace whirl::test
