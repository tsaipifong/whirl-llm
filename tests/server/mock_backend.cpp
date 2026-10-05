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
// MockDeviceOps

MockDeviceOps::~MockDeviceOps() = default;

void MockDeviceOps::copyAsync(tier::DevPtr dst, tier::DevPtr src, std::uint64_t bytes, tier::Stream) {
    if (bytes == 0) return;
    std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), bytes);
    copied_ += bytes;
}

void MockDeviceOps::download(void* dst, tier::DevPtr src, std::uint64_t bytes) {
    std::memcpy(dst, reinterpret_cast<const void*>(src), bytes);
}

tier::Stream MockDeviceOps::streamCreateNonBlocking() {
    return reinterpret_cast<tier::Stream>(next_handle_.fetch_add(16));
}

void MockDeviceOps::streamDestroy(tier::Stream) {}

tier::Event MockDeviceOps::eventCreate() { return reinterpret_cast<tier::Event>(next_handle_.fetch_add(16)); }

void MockDeviceOps::eventDestroy(tier::Event) {}

void* MockDeviceOps::hostMalloc(std::uint64_t bytes) {
    void* p = _aligned_malloc(bytes ? bytes : 1, 4096);
    if (!p) throw std::bad_alloc();
    std::memset(p, 0, bytes);
    std::lock_guard<std::mutex> lk(mu_);
    live_.insert(p);
    return p;
}

void MockDeviceOps::hostFree(void* p) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        live_.erase(p);
    }
    _aligned_free(p);
}

tier::DevPtr MockDeviceOps::malloc(std::uint64_t bytes) {
    void* p = _aligned_malloc(bytes ? bytes : 1, 256);
    if (!p) throw std::bad_alloc();
    std::memset(p, 0, bytes);
    std::lock_guard<std::mutex> lk(mu_);
    live_.insert(p);
    return reinterpret_cast<tier::DevPtr>(p);
}

void MockDeviceOps::free(tier::DevPtr p) {
    void* q = reinterpret_cast<void*>(p);
    {
        std::lock_guard<std::mutex> lk(mu_);
        live_.erase(q);
    }
    _aligned_free(q);
}

std::size_t MockDeviceOps::liveAllocations() const {
    std::lock_guard<std::mutex> lk(mu_);
    return live_.size();
}

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

MockModel::MockModel(tier::DeviceOps& ops, const Tokenizer& tok, const MockConfig& mc) : ops_(ops), tok_(tok), mc_(mc) {
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
    for (std::size_t a = 0; a < n_arr; ++a) {
        kc_.push_back(ops_.malloc(static_cast<std::uint64_t>(pool_pages_) * kPage * kKRow));
        vc_.push_back(ops_.malloc(static_cast<std::uint64_t>(pool_pages_) * kPage * kVRow));
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
    ops_.free(logits_);
    ops_.free(hn_);
    ops_.streamDestroy(stream_);
}

std::vector<server::KvArr> MockModel::kvArrays() const {
    std::vector<server::KvArr> out;
    for (std::size_t a = 0; a < kc_.size(); ++a) {
        out.push_back({kc_[a], kKRow});
        out.push_back({vc_[a], kVRow});
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
    const std::uint64_t seed = mix(h, kv, 77);
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
        st.h = mix(st.h, toks[i], p);
        writeKv(s, p, toks[i], st.h);
        st.kv = mix(st.kv, kvRow(toks[i], p, st.h), p);
        st.pos = p;
        writeHidden(hn_ + (r0 + i) * E * 4, st);
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
    for (std::uint32_t i = 0; i < s.npend && i < s.pend.size(); ++i) h = mix(h, s.pend[i].tok, s.pend[i].pos);
    writeH(s, h);
    s.npend = 0;
}

void MockModel::restoreSeqSnapshot(std::uint32_t, std::uint32_t) { poison_ += 1; }

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

void MockModel::forward(std::span<const std::uint32_t> tokens, std::uint32_t pos0) {
    prefillOne(cur_, tokens, pos0, 0, 0, std::nullopt, false);
}

void MockModel::prefillMtpChunk(std::span<const std::uint32_t> tokens, std::uint32_t pos0, std::size_t off, std::size_t n,
                                std::optional<server::DevPtr> prev_hidden) {
    prefillOne(cur_, tokens.subspan(off, n), pos0 + static_cast<std::uint32_t>(off), 0, 0,
               off == 0 ? prev_hidden : std::nullopt, true);
}

void MockModel::prefillSegs(std::span<const server::PSeg> segs, bool with_mtp) {
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
            State ch{mix(last.h, sg.pend.back(), p), 0, p};
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
        s.chain.h = mix(s.chain.h, d, p);
        s.chain.kv = mix(s.chain.kv, kvRow(d, p, s.chain.h), p);
        s.chain_pos = p;
    }
}

std::uint32_t MockModel::verifyBatchEnqueue(std::span<const server::VSeg> segs) {
    std::uint32_t row = 0;
    spinUs(mc_.cycle_us);
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
            st.h = mix(st.h, toks[i], p);
            writeKv(s, p, toks[i], st.h);
            st.kv = mix(st.kv, kvRow(toks[i], p, st.h), p);
            st.pos = p;
            writeHidden(hn_ + (row + i) * E * 4, st);
            writeLogits(row + static_cast<std::uint32_t>(i), st.h, st.kv);
            s.ctl[qwen35::ctl_rows + i] = static_cast<std::int32_t>(nextToken(st.h, st.kv));
            s.pend.push_back({toks[i], p});
            rows_run_ += 1;
        }
        s.npend = 0;
        row += static_cast<std::uint32_t>(toks.size());
    }
    return row;
}

void MockModel::readCtl(std::span<std::int32_t> dst) {
    for (std::size_t s = 0; s < seqs_.size() && (s + 1) * qwen35::ctl_words <= dst.size(); ++s)
        std::memcpy(dst.data() + s * qwen35::ctl_words, seqs_[s].ctl.data(), qwen35::ctl_words * 4);
}

void MockModel::mtpEnqueue(server::DevPtr, std::span<const std::uint32_t>, std::uint32_t) {}

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

std::vector<std::uint32_t> MockModel::greedyReference(std::span<const std::uint32_t> prompt, std::uint32_t max_tokens,
                                                      bool ignore_eos) const {
    std::uint64_t h = kH0, kv = 0;
    std::uint32_t p = 0;
    for (std::uint32_t t : prompt) {
        h = mix(h, t, p);
        kv = mix(kv, kvRow(t, p, h), p);
        ++p;
    }
    std::vector<std::uint32_t> out;
    while (out.size() < max_tokens) {
        const std::uint32_t t = nextToken(h, kv);
        if (!ignore_eos && (t == im_end_ || t == eot_)) break;
        out.push_back(t);
        h = mix(h, t, p);
        kv = mix(kv, kvRow(t, p, h), p);
        ++p;
    }
    return out;
}

}  // namespace whirl::test
