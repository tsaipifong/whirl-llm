// Prefix store of the server (arch024 M4); see prefix_store.h.
// SPDX-License-Identifier: Apache-2.0

#include "server/prefix_store.h"

#include "cache/tier_adapter.h"
#include "whirl/vismap.h"

#include <algorithm>

namespace whirl::server {

std::uint8_t ckKindCode(std::string_view k) {
    if (k == "prompt-end") return 1;
    if (k == "think-open") return 2;
    if (k == "gen-end") return 3;
    if (k == "system") return 4;
    if (k == "prefix") return 5;
    return 0;
}

const char* ckKindStr(std::uint8_t c) {
    switch (c) {
        case 1: return "prompt-end";
        case 2: return "think-open";
        case 3: return "gen-end";
        case 4: return "system";
        case 5: return "prefix";
        default: return "restored";
    }
}

PrefixStore::PrefixStore(const ChatTokens& ct, const Config& cfg) : ct_(ct), cfg_(cfg) {
    cache::Config pc;
    pc.enabled = cfg_.enabled;
    pc.lcp_on = cfg_.lcp_on;
    pc.sys_min = cfg_.sys_min;
    pc.kv_page = cfg_.kv_page;
    pc_ = cache::makePrefixCache(pc);
}

void PrefixStore::attachTier(tier::Tier* t) {
    ownerCheck(owner_, "PrefixStore::attachTier");
    tier_view_ = t ? cache::makeTierAdapter(*t) : nullptr;
    pc_->attachTier(tier_view_.get());
}

std::uint32_t PrefixStore::sysSplit(std::span<const std::uint32_t> toks) const {
    if (cfg_.sys_min == 0) return 0;
    return static_cast<std::uint32_t>(sched::sysSplit(ct_.sysBoundary(toks), toks.size(), cfg_.sys_min));
}

std::vector<std::uint32_t> PrefixStore::imgSplits(std::span<const std::uint32_t> toks) const {
    if (!vis_.on) return {};
    // shared with the CLI (include/whirl/prefill_schedule.h, CLI-SCHED)
    return sched::imageSplits(toks, qwen35::image_id_base, vis_.vision_end_id, vis_.ck_min);
}

std::size_t PrefixStore::schedFloor(std::span<const std::uint32_t> toks, std::size_t lim) const {
    const std::vector<std::uint32_t> imgs = imgSplits(toks);
    return sched::schedFloor(toks.size(), sysSplit(toks), ct_.thinkOpenSplit(toks), imgs, lim);
}

bool PrefixStore::onSchedule(std::span<const std::uint32_t> toks, std::size_t a) const {
    const std::vector<std::uint32_t> imgs = imgSplits(toks);
    return sched::onSchedule(toks.size(), sysSplit(toks), ct_.thinkOpenSplit(toks), imgs, a);
}

bool PrefixStore::resumeOk(std::span<const std::uint32_t> toks, std::uint32_t pos, std::uint8_t kind) const {
    return !cfg_.cold_resume || (kind != ckKindCode("gen-end") && onSchedule(toks, pos));
}

std::uint32_t PrefixStore::tailCk(const SlotPrefix& sl, std::span<const std::uint32_t> toks) const {
    const std::size_t N = toks.size();
    // COLD-RESUME (precise): the tail checkpoint is what a conversation's next turn resumes from (gen-end is not
    // cold-exact), so every prompt keeps one, and when the tail point is a forced split of this prompt only (the
    // think-open split moves with the conversation) it goes one schedule point earlier
    if (!(cfg_.tail_ck && sl.on_sched && sl.ckpts.size() >= cache::ckpt_keep_prefix_min &&
          (cfg_.cold_resume || N >= cfg_.tail_ck_min)))
        return 0;
    const std::vector<std::uint32_t> imgs = imgSplits(toks);
    std::uint32_t t = static_cast<std::uint32_t>(sched::tailPoint(N, sysSplit(toks), ct_.thinkOpenSplit(toks), imgs));
    if (cfg_.cold_resume && t > 0 && t == sl.pf_split)
        t = static_cast<std::uint32_t>(sched::schedFloor(N, sysSplit(toks), ct_.thinkOpenSplit(toks), imgs, t - 1));
    if (t > sl.reuse && t != sl.sys_split && t != sl.lcp_ck && t != sl.img_ck && t != sl.pf_split) return t;
    return 0;
}

void PrefixStore::waitSpeFences() {
    ownerCheck(owner_, "PrefixStore::waitSpeFences");
    for (Spe& sp : spes_) {
        sp.spill_ev.waitIfPending();
        sp.ready_ev.waitIfPending();
    }
}

std::uint64_t PrefixStore::parkSpes(CkptStore& cs) {
    ownerCheck(owner_, "PrefixStore::parkSpes");
    std::uint64_t b = 0;
    for (Spe& sp : spes_) {
        b += cs.bytes(false);
        cs.release(sp.ck, {sp.spill_ev.get(), sp.ready_ev.get()});
    }
    spes_.clear();  // destroys the events (device::EventH)
    return b;
}

std::uint64_t PrefixStore::unparkSpes(CkptStore& cs, std::size_t n) {
    ownerCheck(owner_, "PrefixStore::unparkSpes");
    try {
        spes_.clear();
        spes_.resize(n);
        for (Spe& sp : spes_) cs.init(sp.ck, false);
    } catch (const std::exception&) {
        // REVIEW-GOV-1 P2-2: free what the unfinished table got, so it stays parked holding nothing
        for (Spe& sp : spes_) cs.release(sp.ck, {});
        spes_.clear();
        throw;
    }
    return spes_.size() * cs.bytes(false);
}

Spe* PrefixStore::speMatch(std::span<const std::uint32_t> toks) {
    ownerCheck(owner_, "PrefixStore::speMatch");
    const int i = pc_->matchShared(spes_, toks, [&](std::size_t a) { return onSchedule(toks, a); });
    return i < 0 ? nullptr : &spes_[static_cast<std::size_t>(i)];
}

PrefixStore::Match PrefixStore::cacheMatch(SlotPrefix& sl, std::span<const std::uint32_t> toks) const {
    if (cfg_.cold_resume) {
        // COLD-RESUME: the checkpoints a cold-exact resume may use (same indices, the others marked invalid)
        std::vector<cache::CkptKey> ks(sl.ckpts.begin(), sl.ckpts.end());
        for (cache::CkptKey& k : ks)
            if (k.valid && !resumeOk(toks, k.pos, ckKindCode(k.kind))) k.valid = false;
        const cache::SlotMatch m = pc_->matchSlot(sl.cache_tokens, ks, toks);
        return {m.reuse, m.ck < 0 ? nullptr : &sl.ckpts[static_cast<std::size_t>(m.ck)], m.lcp};
    }
    const cache::SlotMatch m = pc_->matchSlot(sl.cache_tokens, sl.ckpts, toks);
    return {m.reuse, m.ck < 0 ? nullptr : &sl.ckpts[static_cast<std::size_t>(m.ck)], m.lcp};
}

std::uint32_t PrefixStore::lcpTarget(cache::View<const cache::SlotCache> slots, std::span<const std::uint32_t> toks,
                                     std::uint32_t reuse, std::uint32_t sys) const {
    return pc_->lcpTarget(slots, spes_, toks, reuse, sys, [&](std::size_t lim) { return schedFloor(toks, lim); });
}

}  // namespace whirl::server
