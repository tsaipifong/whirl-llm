// SPDX-License-Identifier: Apache-2.0

#include "mock_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <numeric>
#include <stdexcept>

namespace whirl::test {

namespace {

// simulated device time: a busy wait (sleep granularity on Windows is ~1-16 ms)
void spinUs(double us) {
    if (us <= 0) return;
    const auto t_end = std::chrono::steady_clock::now() + std::chrono::duration<double, std::micro>(us);
    while (std::chrono::steady_clock::now() < t_end) {
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// MockModel

namespace {
constexpr std::uint64_t kH0 = 0x5EED5EED12345678ull;
constexpr std::uint64_t kSaltC = 0x9E3779B97F4A7C15ull;
constexpr std::uint64_t kSaltS = 0xC2B2AE3D27D4EB4Full;
constexpr std::uint32_t kKRow = 24;  // u32 tok, u32 pos, u64 h, u64 salt
constexpr std::uint32_t kVRow = 8;   // u64 h ^ ~layer
constexpr std::uint32_t kPage = qwen35::kv_page;

template <class T>
T* at(server::DevPtr p) {
    return reinterpret_cast<T*>(p);
}
}  // namespace

std::uint64_t MockModel::mix(std::uint64_t h, std::uint64_t a, std::uint64_t b) {
    std::uint64_t z = h ^ (a * 0xBF58476D1CE4E5B9ull) ^ (b * 0x94D049BB133111EBull + 0x2545F4914F6CDD1Dull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

std::uint64_t MockModel::kvRow(std::uint32_t tok, std::uint32_t pos, std::uint64_t h) { return mix(h, tok, pos); }

std::uint64_t MockModel::stepH(std::uint64_t h, std::uint32_t tok, std::uint32_t pos) const {
    // markov: the state is the last token only (greedy output cycles, n-gram drafts hit)
    return mc_.markov > 0 ? mix(0x5EEDull, tok, 0) : mix(h, tok, pos);
}

MockModel::MockModel(device::Backend& ops, const Tokenizer& tok, const MockConfig& mc) : ops_(ops), tok_(tok), mc_(mc) {
    cfg_.n_layer = mc.n_layer;
    cfg_.n_embd = mc.n_embd;
    cfg_.n_ff = 64;
    cfg_.n_head = 2;
    cfg_.n_head_kv = 1;
    cfg_.head_dim = 8;
    cfg_.interval = 4;
    cfg_.d_conv = 4;
    cfg_.d_state = 8;
    cfg_.n_k_heads = 1;
    cfg_.n_v_heads = 1;
    cfg_.d_inner = 8;
    cfg_.n_vocab = static_cast<std::uint32_t>(tok.size());
    cfg_.n_nextn = mc.mtp ? 1 : 0;
    stream_ = ops_.streamCreateNonBlocking();
    for (std::uint32_t i = 0; i < cfg_.n_layer; ++i)
        if (cfg_.isAttn(i)) attn_layers_.push_back(i);
    // clean ASCII word tokens as the toy model's vocabulary
    for (std::size_t i = 0; i < tok.size() && text_ids_.size() < 3000; ++i) {
        if (tok.type(static_cast<TokenId>(i)) != TokenType::normal) continue;
        const std::string p = tok.piece(static_cast<TokenId>(i), false);
        if (p.size() < 2 || p.size() > 8) continue;
        bool ok = true;
        for (std::size_t k = 0; k < p.size(); ++k) {
            const char c = p[k];
            if (!((c >= 'a' && c <= 'z') || (k == 0 && c == ' '))) ok = false;
        }
        if (ok) text_ids_.push_back(static_cast<std::uint32_t>(i));
    }
    im_end_ = static_cast<std::uint32_t>(tok.find("<|im_end|>"));
    eot_ = static_cast<std::uint32_t>(tok.find("<|endoftext|>"));
    const std::uint64_t V = cfg_.n_vocab;
    logits_ = ops_.malloc(V * 4 * qwen35::max_small_batch);
    hn_ = ops_.malloc(static_cast<std::uint64_t>(mc.max_batch + qwen35::max_small_batch) * cfg_.n_embd * 4);
    seq_pages_ = (mc.slot_ctx + kPage - 1) / kPage;
    seqs_.resize(mc.parallel);
    dfl_ctx_.resize(mc.parallel);
    for (Seq& s : seqs_) {
        s.conv.assign(cfg_.n_layer, 0);
        s.ssm.assign(cfg_.n_layer, 0);
        for (std::uint32_t i = 0; i < cfg_.n_layer; ++i) {
            if (cfg_.isAttn(i)) continue;
            s.conv[i] = ops_.malloc(convBytes());
            s.ssm[i] = ops_.malloc(ssmBytes());
        }
        s.hid = ops_.malloc(static_cast<std::uint64_t>(qwen35::max_small_batch) * cfg_.n_embd * 4);
        s.ptab.assign(seq_pages_, -1);
        writeH(s, kH0);
    }
    // KV pool
    pool_pages_ = mc.pool_tokens / kPage;
    const std::size_t n_arr = attn_layers_.size() + (mc.mtp ? 1 : 0);
    xrow_ = mc.kv_extra_rows;
    for (std::size_t a = 0; a < n_arr; ++a) {
        kc_.push_back(ops_.malloc(static_cast<std::uint64_t>(pool_pages_) * kPage * kKRow));
        vc_.push_back(ops_.malloc(static_cast<std::uint64_t>(pool_pages_) * kPage * kVRow));
        for (std::uint32_t r : xrow_) xc_.push_back(ops_.malloc(static_cast<std::uint64_t>(pool_pages_) * kPage * r));
    }
}

MockModel::~MockModel() {
    for (Seq& s : seqs_) {
        for (auto p : s.conv)
            if (p) ops_.free(p);
        for (auto p : s.ssm)
            if (p) ops_.free(p);
        ops_.free(s.hid);
    }
    for (auto p : kc_) ops_.free(p);
    for (auto p : vc_) ops_.free(p);
    for (auto p : xc_) ops_.free(p);
    ops_.free(logits_);
    ops_.free(hn_);
    ops_.streamDestroy(stream_);
}

bool MockModel::resizePool(std::uint32_t tokens) {
    if (fail_resize.exchange(false)) return false;
    for (auto p : kc_) ops_.free(p);
    for (auto p : vc_) ops_.free(p);
    for (auto p : xc_) ops_.free(p);
    const std::size_t n_arr = kc_.size();
    kc_.clear();
    vc_.clear();
    xc_.clear();
    pool_pages_ = (tokens + kPage - 1) / kPage;
    for (std::size_t a = 0; a < n_arr; ++a) {
        kc_.push_back(ops_.malloc(static_cast<std::uint64_t>(pool_pages_) * kPage * kKRow));
        vc_.push_back(ops_.malloc(static_cast<std::uint64_t>(pool_pages_) * kPage * kVRow));
        for (std::uint32_t r : xrow_) xc_.push_back(ops_.malloc(static_cast<std::uint64_t>(pool_pages_) * kPage * r));
    }
    for (Seq& sq : seqs_) sq.ptab.assign(seq_pages_, -1);
    resizes_.fetch_add(1);
    return true;
}

std::vector<server::KvArr> MockModel::kvArrays() const {
    std::vector<server::KvArr> out;
    for (std::size_t a = 0; a < kc_.size(); ++a) {
        out.push_back({kc_[a], kKRow});
        out.push_back({vc_[a], kVRow});
        for (std::size_t j = 0; j < xrow_.size(); ++j) out.push_back({xc_[a * xrow_.size() + j], xrow_[j]});
    }
    return out;
}

std::uint64_t MockModel::readH(const Seq& s) const {
    std::uint64_t h = 0;
    bool first = true, bad = false;
    for (std::uint32_t i = 0; i < cfg_.n_layer; ++i) {
        if (s.ssm[i] == 0) continue;
        const std::uint64_t* ss = at<std::uint64_t>(s.ssm[i]);
        const std::uint64_t* cc = at<std::uint64_t>(s.conv[i]);
        if (first) {
            h = ss[0];
            first = false;
        }
        if (ss[0] != h || ss[1] != (h ^ (i * kSaltS)) || cc[0] != (h ^ (i * kSaltC))) bad = true;
    }
    if (bad) {
        poison_ += 1;
        return h ^ 0xBADBADBADBADull;
    }
    return h;
}

void MockModel::writeH(Seq& s, std::uint64_t h) {
    for (std::uint32_t i = 0; i < cfg_.n_layer; ++i) {
        if (s.ssm[i] == 0) continue;
        std::uint64_t* ss = at<std::uint64_t>(s.ssm[i]);
        std::uint64_t* cc = at<std::uint64_t>(s.conv[i]);
        ss[0] = h;
        ss[1] = h ^ (i * kSaltS);
        cc[0] = h ^ (i * kSaltC);
    }
}

void MockModel::writeKv(Seq& s, std::uint32_t pos, std::uint32_t tok, std::uint64_t h) {
    const std::uint32_t pg = pos / kPage;
    if (pg >= s.ptab.size() || s.ptab[pg] < 0 || static_cast<std::uint32_t>(s.ptab[pg]) >= pool_pages_) {
        poison_ += 1;  // write to an unmapped page
        return;
    }
    const std::uint64_t row = static_cast<std::uint64_t>(s.ptab[pg]) * kPage + pos % kPage;
    for (std::size_t a = 0; a < kc_.size(); ++a) {
        if (a == attn_layers_.size()) continue;  // MTP layer: written by the MTP steps
        std::uint8_t* k = at<std::uint8_t>(kc_[a]) + row * kKRow;
        std::memcpy(k, &tok, 4);
        std::memcpy(k + 4, &pos, 4);
        std::memcpy(k + 8, &h, 8);
        const std::uint64_t salt = h ^ (a * kSaltC);
        std::memcpy(k + 16, &salt, 8);
        const std::uint64_t v = h ^ ~static_cast<std::uint64_t>(a);
        std::memcpy(at<std::uint8_t>(vc_[a]) + row * kVRow, &v, 8);
    }
}

std::uint64_t MockModel::kvPrefix(const Seq& s, std::uint32_t pos) const {
    std::uint64_t kv = 0;
    bool bad = false;
    for (std::uint32_t p = 0; p < pos; ++p) {
        const std::uint32_t pg = p / kPage;
        if (pg >= s.ptab.size() || s.ptab[pg] < 0 || static_cast<std::uint32_t>(s.ptab[pg]) >= pool_pages_) {
            bad = true;
            break;
        }
        const std::uint64_t row = static_cast<std::uint64_t>(s.ptab[pg]) * kPage + p % kPage;
        std::uint32_t tok0 = 0, pos0 = 0;
        std::uint64_t h0 = 0;
        for (std::size_t a = 0; a < attn_layers_.size(); ++a) {
            const std::uint8_t* k = at<const std::uint8_t>(kc_[a]) + row * kKRow;
            std::uint32_t t, pp;
            std::uint64_t h, salt, v;
            std::memcpy(&t, k, 4);
            std::memcpy(&pp, k + 4, 4);
            std::memcpy(&h, k + 8, 8);
            std::memcpy(&salt, k + 16, 8);
            std::memcpy(&v, at<const std::uint8_t>(vc_[a]) + row * kVRow, 8);
            if (pp != p || salt != (h ^ (a * kSaltC)) || v != (h ^ ~static_cast<std::uint64_t>(a))) bad = true;
            if (a == 0) {
                tok0 = t;
                pos0 = pp;
                h0 = h;
            } else if (t != tok0 || pp != pos0 || h != h0) {
                bad = true;
            }
        }
        kv = mix(kv, kvRow(tok0, p, h0), p);
    }
    if (bad) {
        poison_ += 1;
        kv ^= 0xBADC0FFEEull;
    }
    return kv;
}

std::uint32_t MockModel::nextToken(std::uint64_t h, std::uint64_t kv) const {
    const std::uint64_t seed = mc_.markov > 0 ? mix(h, 0, 77) : mix(h, kv, 77);
    if (mc_.markov > 0) return text_ids_[(seed >> 7) % std::min<std::size_t>(mc_.markov, text_ids_.size())];
    if (mc_.eos_rate > 0) {
        const double u = static_cast<double>(seed >> 11) * (1.0 / 9007199254740992.0);
        if (u < mc_.eos_rate) return im_end_;
    }
    return text_ids_[(seed >> 7) % text_ids_.size()];
}

void MockModel::writeLogits(std::uint32_t row, std::uint64_t h, std::uint64_t kv) {
    const std::uint32_t V = cfg_.n_vocab;
    float* x = at<float>(logits_) + static_cast<std::uint64_t>(row) * V;
    std::fill(x, x + V, -10.0f);
    const std::uint32_t top = nextToken(h, kv);
    const std::uint64_t seed = mix(h, kv, 99);
    for (std::uint32_t j = 1; j < 16; ++j) {
        const std::uint32_t c = text_ids_[mix(seed, j, 5) % text_ids_.size()];
        if (c == top) continue;
        x[c] = 9.0f - 0.35f * static_cast<float>(j);
    }
    x[top] = 10.0f;
}

// the second-highest logit of writeLogits(h, kv): the first candidate j = 1, 2, ... that is not the top
std::uint32_t MockModel::runnerUp(std::uint64_t h, std::uint64_t kv) const {
    const std::uint32_t top = nextToken(h, kv);
    const std::uint64_t seed = mix(h, kv, 99);
    for (std::uint32_t j = 1; j < 16; ++j) {
        const std::uint32_t c = text_ids_[mix(seed, j, 5) % text_ids_.size()];
        if (c != top) return c;
    }
    return top;
}

void MockModel::writeHidden(server::DevPtr dst, const State& st) const {
    std::uint64_t* p = at<std::uint64_t>(dst);
    p[0] = st.h;
    p[1] = st.kv;
    p[2] = st.pos;
}

MockModel::State MockModel::readHidden(server::DevPtr src) const {
    const std::uint64_t* p = at<const std::uint64_t>(src);
    return {p[0], p[1], static_cast<std::uint32_t>(p[2])};
}

MockModel::State MockModel::runRows(Seq& s, std::span<const std::uint32_t> toks, std::uint32_t pos0, std::uint32_t r0,
                                    bool commit) {
    State st{readH(s), kvPrefix(s, pos0), pos0};
    const std::uint64_t E = cfg_.n_embd;
    for (std::size_t i = 0; i < toks.size(); ++i) {
        const std::uint32_t p = pos0 + static_cast<std::uint32_t>(i);
        st.h = stepH(st.h, toks[i], p);
        writeKv(s, p, toks[i], st.h);
        st.kv = mix(st.kv, kvRow(toks[i], p, st.h), p);
        st.pos = p;
        writeHidden(hn_ + (r0 + i) * E * 4, st);
        capRow(static_cast<std::uint32_t>(&s - seqs_.data()), toks[i], st);
        rows_run_ += 1;
    }
    if (commit) writeH(s, st.h);
    return st;
}

void MockModel::prefillOne(std::uint32_t si, std::span<const std::uint32_t> toks, std::uint32_t pos0, std::uint32_t r0,
                           std::uint32_t logits_row, std::optional<server::DevPtr> prev_hidden, bool mtp) {
    spinUs(mc_.prefill_us_per_row * static_cast<double>(toks.size()));
    Seq& s = seqs_[si];
    commitSeq(si);
    if (mtp && pos0 > 0 && prev_hidden) {
        if (readHidden(*prev_hidden).pos != pos0 - 1) hid_mismatch_ += 1;
    }
    if (mc_.sched_grid > 0 && pos0 % mc_.sched_grid != 0 && !toks.empty()) writeH(s, mix(readH(s), 0x7a11c0ffeeull, pos0));
    const State st = runRows(s, toks, pos0, r0, true);
    writeLogits(logits_row, st.h, st.kv);
    if (mtp && !toks.empty()) {
        const std::uint64_t E = cfg_.n_embd;
        ops_.copyAsync(s.hid, hn_ + (r0 + toks.size() - 1) * E * 4, E * 4, stream_);
    }
}

void MockModel::commitSeq(std::uint32_t si) {
    Seq& s = seqs_[si];
    if (s.npend == 0) return;
    std::uint64_t h = readH(s);
    for (std::uint32_t i = 0; i < s.npend && i < s.pend.size(); ++i) h = stepH(h, s.pend[i].tok, s.pend[i].pos);
    writeH(s, h);
    s.npend = 0;
}

void MockModel::ensureSnapshots(std::uint32_t n) {
    snap_sets_ = std::max(snap_sets_, std::min(n, qwen35::gdn_max_snap));
    if (snaps_.size() < snap_sets_) snaps_.resize(snap_sets_);
}

void MockModel::restoreSeqSnapshot(std::uint32_t s, std::uint32_t set) {
    // replay keeps no snapshots: a restore there is a bookkeeping error
    if (mc_.replay || set >= snaps_.size() || snaps_[set].seq != s) {
        poison_ += 1;
        return;
    }
    writeH(seqs_[s], snaps_[set].h);
    seqs_[s].npend = 0;
}

void MockModel::mapPages(std::uint32_t s, std::uint32_t first, std::span<const std::int32_t> phys) {
    Seq& q = seqs_[s];
    for (std::size_t i = 0; i < phys.size(); ++i) {
        if (first + i >= q.ptab.size()) throw std::runtime_error("mock: mapPages past the page table");
        q.ptab[first + i] = phys[i];
    }
}

void MockModel::reset() {
    Seq& s = seqs_[cur_];
    writeH(s, kH0);
    s.npend = 0;
    s.pend.clear();
}

// one prefill forward of n rows: the real model's limit (forward.cpp: n > max_batch -> BatchTooLarge)
void MockModel::prefillRows(std::size_t n) {
    if (n == 0 || n > mc_.max_batch) throw std::runtime_error("BatchTooLarge");
    const auto r = static_cast<std::uint32_t>(n);
    if (r > max_pf_rows_.load()) max_pf_rows_.store(r);
}

void MockModel::forward(std::span<const std::uint32_t> tokens, std::uint32_t pos0) {
    prefillRows(tokens.size());
    cap_.clear();
    prefillOne(cur_, tokens, pos0, 0, 0, std::nullopt, false);
}

void MockModel::prefillMtpChunk(std::span<const std::uint32_t> tokens, std::uint32_t pos0, std::size_t off, std::size_t n,
                                std::optional<server::DevPtr> prev_hidden) {
    prefillRows(n);
    cap_.clear();
    prefillOne(cur_, tokens.subspan(off, n), pos0 + static_cast<std::uint32_t>(off), 0, 0,
               off == 0 ? prev_hidden : std::nullopt, true);
}

void MockModel::prefillSegs(std::span<const server::PSeg> segs, bool with_mtp) {
    std::size_t rows = 0;
    for (const server::PSeg& sg : segs) rows += sg.n;
    prefillRows(rows);
    cap_.clear();
    std::uint32_t r0 = 0;
    for (std::size_t k = 0; k < segs.size(); ++k) {
        const server::PSeg& sg = segs[k];
        prefillOne(sg.seq, sg.tokens.subspan(sg.off, sg.n), sg.pos0 + static_cast<std::uint32_t>(sg.off), r0,
                   static_cast<std::uint32_t>(k), sg.off == 0 ? sg.prev_hidden : std::nullopt, with_mtp);
        r0 += static_cast<std::uint32_t>(sg.n);
    }
}

void MockModel::segLogitsToFront(std::uint32_t k) {
    if (k == 0) return;
    const std::uint64_t V = cfg_.n_vocab;
    std::memcpy(at<float>(logits_), at<float>(logits_) + k * V, V * 4);
}

std::uint32_t MockModel::argmax() {
    const std::uint32_t V = cfg_.n_vocab;
    const float* x = at<float>(logits_);
    return static_cast<std::uint32_t>(std::max_element(x, x + V) - x);
}

std::uint32_t MockModel::corruptDraft(std::uint32_t truth, std::uint64_t key) const {
    const double u = static_cast<double>(mix(key, 3, 4) >> 11) * (1.0 / 9007199254740992.0);
    if (u >= mc_.draft_miss) return truth;
    return text_ids_[mix(key, 8, 9) % text_ids_.size()];
}

void MockModel::mtpBatchStepEx(std::span<const server::MSeg> segs, std::uint32_t r, bool draft) {
    if (!mc_.mtp) throw std::runtime_error("NoMtp");
    const std::uint64_t E = cfg_.n_embd;
    for (const server::MSeg& sg : segs) {
        Seq& s = seqs_[sg.seq];
        if (r == 0) {
            if (sg.pend.empty()) throw std::runtime_error("mock: MTP step without pend rows");
            for (std::size_t j = 0; j < sg.pend.size(); ++j) {
                const State hs = readHidden(s.hid + j * E * 4);
                if (hs.pos + 1 != sg.pend_pos + j) hid_mismatch_ += 1;
            }
            const State last = readHidden(s.hid + (sg.pend.size() - 1) * E * 4);
            const std::uint32_t p = sg.pend_pos + static_cast<std::uint32_t>(sg.pend.size()) - 1;
            State ch{stepH(last.h, sg.pend.back(), p), 0, p};
            ch.kv = mix(last.kv, kvRow(sg.pend.back(), p, ch.h), p);
            s.chain = ch;
            s.chain_pos = p;
            s.ctl[qwen35::ctl_nd] = 0;
            if (!draft) continue;
        }
        const std::uint32_t truth = nextToken(s.chain.h, s.chain.kv);
        const std::uint32_t d = corruptDraft(truth, mix(s.chain.h, r, s.chain_pos));
        s.ctl[qwen35::ctl_drafts + r] = static_cast<std::int32_t>(d);
        s.ctl[qwen35::ctl_nd] = static_cast<std::int32_t>(r + 1);
        const std::uint32_t p = s.chain_pos + 1;
        s.chain.h = stepH(s.chain.h, d, p);
        s.chain.kv = mix(s.chain.kv, kvRow(d, p, s.chain.h), p);
        s.chain_pos = p;
    }
}

std::uint32_t MockModel::verifyBatchEnqueue(std::span<const server::VSeg> segs) {
    std::uint32_t row = 0;
    cap_.clear();
    spinUs(mc_.cycle_us);
    if (!mc_.replay) {
        // snapshot sets for every draft row of the batch (lazily allocated, like the model)
        std::uint32_t need = 0;
        for (const server::VSeg& sg : segs) need += sg.nd;
        if (need > qwen35::gdn_max_snap) throw std::runtime_error("BatchTooLarge");
        ensureSnapshots(need);
        for (Snap& x : snaps_) x = Snap{};
    }
    std::uint32_t snap = 0;
    for (const server::VSeg& sg : segs) {
        Seq& s = seqs_[sg.seq];
        commitSeq(sg.seq);  // the kept rows of the previous verify reach the state
        std::vector<std::uint32_t> toks{sg.next};
        for (std::uint32_t r = 0; r < sg.nd; ++r)
            toks.push_back(!sg.drafts.empty() ? sg.drafts[r]
                                              : static_cast<std::uint32_t>(s.ctl[qwen35::ctl_drafts + r]));
        if (row + toks.size() > qwen35::max_small_batch) throw std::runtime_error("BatchTooLarge");
        if (sg.pos + toks.size() > mc_.slot_ctx) throw std::runtime_error("ContextTooLong");
        const State st0{readH(s), kvPrefix(s, sg.pos), sg.pos};
        State st = st0;
        const std::uint64_t E = cfg_.n_embd;
        s.pend.clear();
        for (std::size_t i = 0; i < toks.size(); ++i) {
            const std::uint32_t p = sg.pos + static_cast<std::uint32_t>(i);
            st.h = stepH(st.h, toks[i], p);
            writeKv(s, p, toks[i], st.h);
            st.kv = mix(st.kv, kvRow(toks[i], p, st.h), p);
            st.pos = p;
            writeHidden(hn_ + (row + i) * E * 4, st);
            writeLogits(row + static_cast<std::uint32_t>(i), st.h, st.kv);
            capRow(sg.seq, toks[i], st);
            s.ctl[qwen35::ctl_rows + i] = static_cast<std::int32_t>(nextToken(st.h, st.kv));
            s.pend.push_back({toks[i], p});
            rows_run_ += 1;
            // replay off: set snap + i holds the state after row i (next + i drafts kept)
            if (!mc_.replay && i < sg.nd) snaps_[snap + i] = Snap{sg.seq, st.h};
        }
        s.npend = 0;
        if (!mc_.replay) {
            // the state advances over every row; the engine restores a snapshot when drafts are rejected
            writeH(s, st.h);
            s.pend.clear();
        }
        snap += sg.nd;
        row += static_cast<std::uint32_t>(toks.size());
    }
    return row;
}

void MockModel::readCtl(std::span<std::int32_t> dst) {
    for (std::size_t s = 0; s < seqs_.size() && (s + 1) * qwen35::ctl_words <= dst.size(); ++s)
        std::memcpy(dst.data() + s * qwen35::ctl_words, seqs_[s].ctl.data(), qwen35::ctl_words * 4);
}

void MockModel::mtpEnqueue(server::DevPtr, std::span<const std::uint32_t>, std::uint32_t) {}

// feat026-acc: what spec_accept_rows does on the device, with the same header code
void MockModel::specAccept(const SpecAccArgs& a_in, std::int32_t* host) {
    SpecAccArgs a = a_in;
    if (a.n <= 0 || a.n > WHIRL_SA_MAX_SEG) throw std::runtime_error("BadBatch");
    a.n_vocab = static_cast<int>(cfg_.n_vocab);
    const std::size_t W = qwen35::ctl_words;
    std::vector<std::int32_t> all(seqs_.size() * W);
    for (std::size_t s = 0; s < seqs_.size(); ++s) std::memcpy(all.data() + s * W, seqs_[s].ctl.data(), W * 4);
    SpecAccResult res[WHIRL_SA_MAX_SEG];
    for (int k = 0; k < a.n; ++k) {
        if (a.seq[k] < 0 || static_cast<std::size_t>(a.seq[k]) >= seqs_.size()) throw std::runtime_error("BadBatch");
        a.kvbase[k] = 0;
        std::int32_t* ctl = all.data() + static_cast<std::size_t>(a.seq[k]) * W;
        SpecAccResult r{};
        if (a.dev[k]) r = spec_accept_one(a, k, ctl);
        else r.chosen = ctl[WHIRL_SA_CTL_ROWS];
        spec_store(a, k, r, ctl);
        res[k] = r;
    }
    spec_next_rows(a, all.data(), res, &spec_tab_);
    for (int k = 0; k < a.n; ++k) {
        const std::size_t s = static_cast<std::size_t>(a.seq[k]);
        std::memcpy(seqs_[s].ctl.data(), all.data() + s * W, W * 4);
        if (host != nullptr) std::memcpy(host + s * W, all.data() + s * W, W * 4);
    }
    spec_accepts_ += 1;
}

void MockModel::topkRows(std::uint32_t row0, std::uint32_t rows, std::uint32_t k, float inv_t, server::DevPtr ids,
                         server::DevPtr vals, server::DevPtr stats) {
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

// DFlash2 mock: the capture keeps (sequence, token, state) of the last dfl_cap_rows rows of the
// latest forward / verify; a draft block continues the true chain from the context row at pos - 1,
// so drafts are right only with that row present and from the sequence's own lineage.
void MockModel::capRow(std::uint32_t seq, std::uint32_t tok, const State& st) {
    if (!mc_.dflash) return;
    cap_.push_back({seq, tok, st});
    if (cap_.size() > 2 * static_cast<std::size_t>(mc_.dfl_cap_rows))
        cap_.erase(cap_.begin(), cap_.end() - mc_.dfl_cap_rows);
}

std::uint32_t MockModel::captureN() const {
    return static_cast<std::uint32_t>(std::min<std::size_t>(cap_.size(), mc_.dfl_cap_rows));
}

void MockModel::dflashIngest(std::uint32_t s, std::uint32_t row0, std::uint32_t n, std::uint32_t pos0) {
    const std::size_t base = cap_.size() - captureN();
    for (std::uint32_t i = 0; i < n; ++i) {
        if (row0 + i >= captureN()) {
            dfl_bad_ += 1;
            continue;
        }
        const CapRow& c = cap_[base + row0 + i];
        if (c.seq != s || c.st.pos != pos0 + i) dfl_bad_ += 1;
        dfl_ctx_[s][pos0 + i] = c.st;
        if (mc_.dfl_ring > 0)
            for (std::uint32_t q = pos0 + i; q >= mc_.dfl_ring;) {
                q -= mc_.dfl_ring;
                dfl_ctx_[s].erase(q);
            }
    }
}

void MockModel::dflashTrim(std::uint32_t s, std::uint32_t keep) {
    auto& m = dfl_ctx_[s];
    m.erase(m.lower_bound(keep), m.end());
}

std::vector<std::uint32_t> MockModel::dflashDraft(std::uint32_t s, std::uint32_t anchor, std::uint32_t pos, float, std::uint32_t) {
    dfl_blocks_ += 1;
    std::vector<std::uint32_t> out;
    const auto& m = dfl_ctx_[s];
    if (mc_.dfl_need_ctx > 0)
        for (std::uint32_t p = pos - std::min(pos, mc_.dfl_need_ctx); p < pos; ++p)
            if (!m.count(p)) {
                dfl_holes_ += 1;
                break;
            }
    auto it = pos > 0 ? m.find(pos - 1) : m.end();
    if (it == m.end()) {
        // no context row before the anchor: guesses
        for (std::uint32_t r = 0; r + 1 < mc_.dfl_block; ++r) out.push_back(text_ids_[mix(anchor, pos, r) % text_ids_.size()]);
        return out;
    }
    // every context row must be this sequence's own (its KV rows hold the same state)
    const Seq& q = seqs_[s];
    for (auto jt = m.begin(); jt != m.end(); ++jt) {
        const std::uint32_t p = jt->first;
        if (p >= pos) break;
        const std::uint32_t pg = p / kPage;
        if (pg >= q.ptab.size() || q.ptab[pg] < 0) {
            dfl_stale_ += 1;
            continue;
        }
        const std::uint64_t row = static_cast<std::uint64_t>(q.ptab[pg]) * kPage + p % kPage;
        std::uint64_t h = 0;
        std::memcpy(&h, at<const std::uint8_t>(kc_[0]) + row * kKRow + 8, 8);
        if (h != jt->second.h) dfl_stale_ += 1;
    }
    State ch = it->second;
    ch.h = stepH(ch.h, anchor, pos);
    ch.kv = mix(ch.kv, kvRow(anchor, pos, ch.h), pos);
    std::uint32_t p = pos;
    for (std::uint32_t r = 0; r + 1 < mc_.dfl_block; ++r) {
        const std::uint32_t truth = nextToken(ch.h, ch.kv);
        const double u = static_cast<double>(mix(ch.h, 0xDF, p) >> 11) * (1.0 / 9007199254740992.0);
        std::uint32_t d = u < mc_.dfl_miss ? text_ids_[mix(ch.h, 0xD2, p) % text_ids_.size()] : truth;
        if (u < mc_.dfl_miss && mc_.dfl_miss_runner_up) d = runnerUp(ch.h, ch.kv);
        out.push_back(d);
        p += 1;
        ch.h = stepH(ch.h, d, p);
        ch.kv = mix(ch.kv, kvRow(d, p, ch.h), p);
    }
    return out;
}

std::vector<std::uint32_t> MockModel::greedyReference(std::span<const std::uint32_t> prompt, std::uint32_t max_tokens,
                                                      bool ignore_eos) const {
    std::uint64_t h = kH0, kv = 0;
    std::uint32_t p = 0;
    for (std::uint32_t t : prompt) {
        h = stepH(h, t, p);
        kv = mix(kv, kvRow(t, p, h), p);
        ++p;
    }
    std::vector<std::uint32_t> out;
    while (out.size() < max_tokens) {
        const std::uint32_t t = nextToken(h, kv);
        if (!ignore_eos && (t == im_end_ || t == eot_)) break;
        out.push_back(t);
        h = stepH(h, t, p);
        kv = mix(kv, kvRow(t, p, h), p);
        ++p;
    }
    return out;
}

}  // namespace whirl::test
