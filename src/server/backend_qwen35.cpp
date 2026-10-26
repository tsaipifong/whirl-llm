// ServerModel over whirl::qwen35::Model.
// SPDX-License-Identifier: Apache-2.0

#include "backend.h"
#include "vision_iface.h"

#include "whirl/hip.h"
#include "whirl/model.h"  // the qwen35::Model this backend forwards to (backend.h sees model_types.h only, arch025-M3)

namespace whirl::server {

namespace {

class Qwen35Backend final : public ServerModel {
public:
    explicit Qwen35Backend(qwen35::Model& m) : m_(m) { topk_ = m_.module.getFunction("topk_rows"); }

    const qwen35::Config& cfg() const override { return m_.cfg; }
    device::Stream stream() const override { return m_.stream; }
    bool hasMtp() const override { return m_.mtp.has_value(); }

    DevPtr logits() const override { return m_.scratch().logits; }
    DevPtr hn() const override { return m_.scratch().hn; }
    DevPtr mtpH() const override { return m_.scratch().mtp_h; }
    DevPtr seqHid(std::uint32_t s) const override { return m_.seqs[s].hid; }
    std::span<const DevPtr> convState() const override { return {m_.conv_state.data(), m_.conv_state.size()}; }
    std::span<const DevPtr> ssmState() const override { return {m_.ssm_state.data(), m_.ssm_state.size()}; }
    std::uint64_t convBytes() const override { return m_.convBytes(); }
    std::uint64_t ssmBytes() const override { return m_.ssmBytes(); }
    std::uint32_t curSeq() const override { return m_.cur_seq; }

    std::uint32_t poolPages() const override { return m_.pool_pages; }
    std::uint32_t seqPages() const override { return m_.seq_pages; }
    std::uint32_t maxBatch() const override { return m_.max_batch; }
    std::uint32_t snapSets() const override { return m_.snap_sets; }
    void ensureSnapshots(std::uint32_t n) override { m_.ensureSnapshots(n); }
    bool gdnReplay() const override { return m_.gdn_replay; }
    bool fusedDecode() const override { return m_.fusedDecode(); }
    bool canSegment() const override { return m_.canSegment(); }

    // KV pool arrays of every attention layer then the MTP layer, in the order of a tier entry's page layout
    // (qwen35::kvArrays, src/model/qwen35/kv.h: pure, tested on the CPU against whirl/kv_layout.h; arch025-M2)
    std::vector<KvArr> kvArrays() const override { return qwen35::kvArrays(m_, m_.cfg, m_.mtp.has_value()); }

    bool resizePool(std::uint32_t tokens) override {
        const std::uint32_t old = m_.pool_pages * qwen35::kv_page;
        hip::sync();
        m_.freeKvPool();
        // (R9700: the WDDM usage showed part of a freed pool only after a later synchronize; sync after the free so the
        // driver can release it before the new arrays are allocated, and after the allocation)
        hip::sync();
        try {
            m_.allocKvPool(tokens);
            hip::sync();
        } catch (const std::exception&) {
            // free what the failed allocation got (freeKvPool frees every non-null array once pool_pages != 0), then
            // the previous size again (throws if even that fails)
            m_.pool_pages = 1;
            m_.freeKvPool();
            m_.allocKvPool(old);
            return false;
        }
        return true;
    }

    void selectSeq(std::uint32_t s) override { m_.selectSeq(s); }
    void commitSeq(std::uint32_t s) override { m_.commitSeq(s); }
    void dropPending(std::uint32_t s) override { m_.dropPending(s); }
    void setPending(std::uint32_t s, std::uint32_t rows) override { m_.setPending(s, rows); }
    void restoreSeqSnapshot(std::uint32_t s, std::uint32_t set) override { m_.restoreSeqSnapshot(s, set); }
    void mapPages(std::uint32_t s, std::uint32_t first, std::span<const std::int32_t> phys) override {
        m_.mapPages(s, first, phys);
    }
    void reset() override { m_.reset(); }

    void forward(std::span<const std::uint32_t> tokens, std::uint32_t pos0) override { m_.forward(tokens, pos0); }
    void prefillMtpChunk(std::span<const std::uint32_t> tokens, std::uint32_t pos0, std::size_t off, std::size_t n,
                         std::optional<DevPtr> prev_hidden) override {
        m_.prefillMtpChunk(tokens, pos0, off, n, prev_hidden);
    }
    void prefillSegs(std::span<const PSeg> segs, bool with_mtp) override { m_.prefillSegs(segs, with_mtp); }
    void segLogitsToFront(std::uint32_t k) override { m_.segLogitsToFront(k); }
    std::uint32_t argmax() override { return m_.argmax(); }

    void setDraftCutoff(float p_min, std::uint32_t n_min) override {
        m_.draft_p_min = p_min;
        m_.draft_n_min = n_min;
    }
    void mtpBatchStepEx(std::span<const MSeg> segs, std::uint32_t r, bool draft) override {
        m_.mtpBatchStepEx(segs, r, draft);
    }
    std::uint32_t verifyBatchEnqueue(std::span<const VSeg> segs) override { return m_.verifyBatchEnqueue(segs); }
    void readCtl(std::span<std::int32_t> dst) override { m_.readCtl(dst); }
    bool draftSampleOk() const override { return m_.draftSampleOk(); }
    void setDraftSample(std::uint32_t s, const spec::DraftSample& ds) override {
        if (s < m_.draft_samp.size()) m_.draft_samp[s] = ds;
    }
    void mtpEnqueue(DevPtr hidden, std::span<const std::uint32_t> tokens, std::uint32_t pos0) override {
        m_.mtpEnqueue(hidden, tokens, 0, pos0, std::nullopt);
    }

    void topkRows(std::uint32_t row0, std::uint32_t rows, std::uint32_t k, float inv_t, DevPtr ids, DevPtr vals,
                  DevPtr stats) override {
        const std::uint32_t V = m_.cfg.n_vocab;
        const DevPtr x = m_.scratch().logits + static_cast<std::uint64_t>(row0) * V * 4;
        hip::launch(topk_, hip::Dim3{rows, 1, 1}, hip::Dim3{1024, 1, 1}, 0, m_.stream, x, static_cast<std::int32_t>(V),
                    static_cast<std::int32_t>(k), inv_t, ids, vals, stats);
    }

    void profileStart(int which, std::size_t slots) override {
        auto& set = which == 0 ? prof_mtp_ : prof_ver_;
        if (!set[slots]) set[slots] = std::make_unique<qwen35::Profile>();
        qwen35::Profile& p = *set[slots];
        p.count = 0;
        m_.prof = &p;
        p.record(m_.stream, qwen35::OpClass::misc);
    }
    void profileStop() override { m_.prof = nullptr; }
    std::uint32_t verifyRows() const override { return m_.verifyRows(); }
    bool hasDFlash() const override { return m_.dfr != nullptr; }
    std::uint32_t dflashMaxDrafts() const override { return m_.dfr ? m_.dfr->block - 1 : 0; }
    std::uint32_t captureN() const override { return m_.cap_buf ? m_.cap_n : 0; }
    void dflashIngest(std::uint32_t s, std::uint32_t row0, std::uint32_t n, std::uint32_t pos0) override {
        if (m_.cap_buf) m_.dflashIngest(s, m_.cap_buf, m_.cap_rows, row0, n, pos0);
    }
    std::vector<std::uint32_t> dflashDraft(std::uint32_t s, std::uint32_t anchor, std::uint32_t pos, float p_min,
                                           std::uint32_t n_min) override {
        return m_.dflashDraft(s, anchor, pos, p_min, n_min);
    }
    void dflashTrim(std::uint32_t s, std::uint32_t keep) override { m_.dflashTrim(s, keep); }
    std::uint32_t dflashRing() const override { return m_.dfr ? m_.dfr->ring : 0; }
    void profileFlush(std::size_t slots) override {
        if (prof_mtp_[slots]) prof_mtp_[slots]->flush();
        if (prof_ver_[slots]) prof_ver_[slots]->flush();
    }
    void profileTake(std::size_t slots, std::array<double, n_prof_classes>& mtp_ms,
                     std::array<double, n_prof_classes>& ver_ms) override {
        mtp_ms.fill(0);
        ver_ms.fill(0);
        for (std::size_t i = 0; i < n_prof_classes && i < qwen35::n_classes; ++i) {
            if (prof_mtp_[slots]) mtp_ms[i] = prof_mtp_[slots]->ms[i];
            if (prof_ver_[slots]) ver_ms[i] = prof_ver_[slots]->ms[i];
        }
        if (prof_mtp_[slots]) prof_mtp_[slots]->reset();
        if (prof_ver_[slots]) prof_ver_[slots]->reset();
    }
    const char* profileClassName(std::size_t i) const override {
        return qwen35::opClassName(static_cast<qwen35::OpClass>(i));
    }

    // vision (image prompts)
    void setVisMap(std::uint32_t s, const qwen35::VisMap* vm) override { m_.vis_seq[s] = vm; }
    bool hasMrope() const override { return m_.k.attn_prep_m != nullptr; }
    std::size_t lendScratch(std::span<std::array<std::uint64_t, 2>> out) const override { return m_.lendScratch(out); }
    void scratchClobbered() override { m_.scratchClobbered(); }
    void visRelease() override { m_.visRelease(); }
    bool visRowsAllocated() const override { return m_.rpos_buf != 0; }

private:
    qwen35::Model& m_;
    hip::Function topk_ = nullptr;
    std::array<std::unique_ptr<qwen35::Profile>, qwen35::gdn_max_seg + 1> prof_mtp_, prof_ver_;
};

}  // namespace

std::unique_ptr<ServerModel> makeQwen35Backend(qwen35::Model& m) { return std::make_unique<Qwen35Backend>(m); }

namespace {

// The production encoder: whirl::vision::Vision on the model stream.
class Qwen35Vision final : public ServerVision {
public:
    Qwen35Vision(vision::Vision& v, hip::Stream stream) : v_(v), stream_(stream) {}
    vision::Prepared prepare(std::span<const std::uint8_t> bytes) const override { return v_.prepare(bytes); }
    std::uint32_t projDim() const override { return v_.hp.proj_dim; }
    vision::EncodeStats encode(const vision::Prepared& p, std::span<const vision::Region> lend, float* out, std::size_t n) override {
        return v_.encode(p, lend, stream_, out, n);
    }
    bool loaded() const override { return v_.loaded(); }
    bool idleTick() override { return v_.idleTick(); }
    std::uint32_t idleS() const override { return v_.idle_s; }

private:
    vision::Vision& v_;
    hip::Stream stream_;
};

}  // namespace

std::unique_ptr<ServerVision> makeQwen35Vision(vision::Vision& v, qwen35::Model& m) { return std::make_unique<Qwen35Vision>(v, m.stream); }

}  // namespace whirl::server
