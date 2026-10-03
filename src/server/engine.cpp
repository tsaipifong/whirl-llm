// Reimplements the WHIRL Zig research prototype's server.zig (Engine, Slot, Spe, StreamState,
// CycleProf; the decode / prefill / prefix-cache / host-tier logic).
// SPDX-License-Identifier: Apache-2.0

#include "engine.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <thread>

namespace whirl::server {

namespace {

constexpr std::uint8_t kNoTag = 0xff;

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

std::size_t lcpLen(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b) {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

bool prefixEq(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

std::uint64_t prngNext(void* ctx) { return (*static_cast<std::mt19937_64*>(ctx))(); }

std::int64_t unixNow() { return static_cast<std::int64_t>(std::time(nullptr)); }

struct KvPoolFull : std::runtime_error {
    KvPoolFull() : std::runtime_error("KvPoolFull") {}
};

}  // namespace

// ---------------------------------------------------------------------------
// WHIRL_PROFILE decode-cycle profile (index = decoding slots)

struct CycleProf {
    static constexpr std::size_t n = gdn_max_seg + 1;
    std::array<std::uint64_t, n> cycles{}, rows{}, drafts{}, tokens{};
    std::array<double, n> wall_ms{}, host_ms{}, gap_ms{}, post_ms{}, enq_ms{};
    std::optional<TimePoint> last_end;
    std::optional<TimePoint> t_ready;
    bool events = true;

    void print(ServerModel& m) {
        for (std::size_t a = 1; a < n; ++a) {
            if (cycles[a] == 0) continue;
            const double c = static_cast<double>(cycles[a]);
            std::array<double, ServerModel::n_prof_classes> pm{}, pv{};
            m.profileTake(a, pm, pv);
            double tm = 0, tv = 0;
            for (double v : pm) tm += v;
            for (double v : pv) tv += v;
            logI("profile | slots {} | {} cycles, {:.2f} rows, {:.2f} drafts/slot, {:.2f} tok/cycle | wall {:.2f} ms/cycle "
                 "(pre-launch host {:.2f}, gap before cycle {:.2f}) | GPU mtp {:.2f} + verify {:.2f} ms | {:.1f} tok/s in "
                 "cycles",
                 a, cycles[a], static_cast<double>(rows[a]) / c, static_cast<double>(drafts[a]) / c / static_cast<double>(a),
                 static_cast<double>(tokens[a]) / c, wall_ms[a] / c, host_ms[a] / c, gap_ms[a] / c, tm / c, tv / c,
                 static_cast<double>(tokens[a]) * 1000.0 / (wall_ms[a] + gap_ms[a]));
            for (std::size_t i = 0; i < ServerModel::n_prof_classes; ++i) {
                if (pm[i] > 0 || pv[i] > 0)
                    logI("profile | slots {} |   {:<9} mtp {:>6.3f}  verify {:>6.3f} ms/cycle", a, m.profileClassName(i),
                         pm[i] / c, pv[i] / c);
            }
            logI("profile | slots {} |   host: enqueue {:.3f} ms, after readCtl {:.3f} ms per cycle", a, enq_ms[a] / c,
                 post_ms[a] / c);
            cycles[a] = rows[a] = drafts[a] = tokens[a] = 0;
            wall_ms[a] = host_ms[a] = gap_ms[a] = post_ms[a] = enq_ms[a] = 0;
        }
        last_end.reset();
    }
};

// ---------------------------------------------------------------------------
// construction

Engine::Engine(ServerModel& model, tier::DeviceOps& ops, const Tokenizer& tok, const EngineOptions& opt)
    : m_(model), ops_(ops), tok_(tok), ct_(tok), opt_(opt), ctx_(opt.ctx), id_prng_(opt.seed) {
    if (opt_.parallel < 1 || opt_.parallel > gdn_max_seg) throw std::runtime_error("parallel must be 1..16");
    if (opt_.n_ck == 0 || opt_.no_prefix_cache) opt_.n_spe = 0;
    if (opt_.no_prefix_cache) opt_.n_ck = 0;
    if (opt_.profile > 0) {
        prof_ = std::make_unique<CycleProf>();
        prof_->events = opt_.profile != 2;
    }
    conv_bytes_ = m_.convBytes();
    ssm_bytes_ = m_.ssmBytes();
    vrows_ = std::clamp(m_.verifyRows(), max_small_batch, qwen35::max_verify_rows);
    if (opt_.ngram && opt_.use_mtp) crlf_norm_ = ChatTokens::crlfToLf(tok_);
    floor_.configure(opt_.decode_min_tps);
    floor_epoch_ = Clock::now();
}

Engine::~Engine() {
    for (Slot& sl : slots_) {
        for (Ckpt& c : sl.ckpts) freeCkpt(c);
        if (sl.spill_ev) ops_.eventDestroy(sl.spill_ev);
    }
    for (Spe& s : spes_) {
        freeCkpt(s.ck);
        if (s.spill_ev) ops_.eventDestroy(s.spill_ev);
        if (s.ready_ev) ops_.eventDestroy(s.ready_ev);
    }
    for (QBatch& q : quarantine_)
        if (q.ev) ops_.eventDestroy(q.ev);
    if (samp_dev_) ops_.free(samp_dev_);
    if (big_dev_) ops_.free(big_dev_);
}

void Engine::initCkpt(Ckpt& c, bool host_ok) {
    const auto& cfg = m_.cfg();
    const std::uint32_t L = cfg.n_layer;
    const std::uint64_t V = cfg.n_vocab;
    c.conv.assign(L, 0);
    c.ssm.assign(L, 0);
    const auto ssm0 = m_.ssmState();
    if (host_ok && opt_.ckpt_host) {
        // one pinned host buffer per checkpoint; save / load copy over PCIe
        std::uint64_t n_gdn = 0;
        for (std::uint32_t li = 0; li < L; ++li)
            if (ssm0[li] != 0) ++n_gdn;
        auto al = [](std::uint64_t x) { return (x + 255) / 256 * 256; };
        const std::uint64_t total = n_gdn * (al(conv_bytes_) + al(ssm_bytes_)) + V * 4;
        c.host = ops_.hostMalloc(total);
        std::uint64_t p = reinterpret_cast<std::uint64_t>(c.host);
        for (std::uint32_t li = 0; li < L; ++li) {
            if (ssm0[li] == 0) continue;
            c.conv[li] = p;
            p += al(conv_bytes_);
            c.ssm[li] = p;
            p += al(ssm_bytes_);
        }
        c.logits = p;
        // the MTP hidden row stays in VRAM (prefill reads it as a device pointer)
        c.hid = ops_.malloc(static_cast<std::uint64_t>(cfg.n_embd) * 4);
        return;
    }
    for (std::uint32_t li = 0; li < L; ++li) {
        if (ssm0[li] == 0) continue;
        c.conv[li] = ops_.malloc(conv_bytes_);
        c.ssm[li] = ops_.malloc(ssm_bytes_);
    }
    c.hid = ops_.malloc(static_cast<std::uint64_t>(cfg.n_embd) * 4);
    c.logits = ops_.malloc(V * 4);
}

void Engine::freeCkpt(Ckpt& c) {
    if (c.host) {
        ops_.hostFree(c.host);
        c.host = nullptr;
        if (c.hid) ops_.free(c.hid);
    } else {
        for (DevPtr p : c.conv)
            if (p) ops_.free(p);
        for (DevPtr p : c.ssm)
            if (p) ops_.free(p);
        if (c.hid) ops_.free(c.hid);
        if (c.logits) ops_.free(c.logits);
    }
    c.conv.clear();
    c.ssm.clear();
    c.hid = c.logits = 0;
}

void Engine::allocState() {
    const auto& cfg = m_.cfg();
    const std::uint32_t V = cfg.n_vocab;
    slots_.resize(opt_.parallel);
    m_.selectSeq(0);
    for (std::uint32_t si = 0; si < opt_.parallel; ++si) {
        Slot& sl = slots_[si];
        sl.id = si;
        sl.rs_map.fill(-1);
        sl.ckpts.resize(opt_.n_ck);
        for (Ckpt& c : sl.ckpts) initCkpt(c, true);
    }
    spes_.resize(opt_.n_spe);
    for (Spe& s : spes_) initCkpt(s.ck, false);
    // sampling buffers
    const std::uint64_t samp_bytes = sampBytes();
    const std::uint64_t big_bytes = static_cast<std::uint64_t>(k_big) * 8 + 8;
    samp_dev_ = ops_.malloc(samp_bytes);
    big_dev_ = ops_.malloc(big_bytes);
    samp_host_.assign(samp_bytes, 0);
    big_host_.assign(big_bytes, 0);
    row_host_.assign(V, 0.0f);
    in_cands_.assign(V, 0);
    cands_scratch_.resize(std::max<std::size_t>(V, k_big));
    ctl_host_.assign(static_cast<std::size_t>(opt_.parallel) * qwen35::ctl_words, 0);
}

void Engine::initPool() {
    pool_pages_ = m_.poolPages();
    page_ref_.assign(pool_pages_, 0);
    free_pages_.clear();
    free_pages_.reserve(pool_pages_);
    // every page free (popped in increasing order)
    for (std::uint32_t pg = pool_pages_; pg > 0;) {
        --pg;
        free_pages_.push_back(static_cast<std::int32_t>(pg));
    }
    kv_arrays_ = m_.kvArrays();
}

void Engine::attachTier(tier::Tier* t) {
    tier_ = t;
    if (t) {
        t->setWake([this] {
            std::lock_guard<std::mutex> lk(q_mutex_);
            tier_wake_ = true;
            q_cond_.notify_all();
        });
        t->log_fn = [](std::string_view msg) { logI("{}", msg); };
    }
}

void Engine::stop() {
    std::lock_guard<std::mutex> lk(q_mutex_);
    stop_ = true;
    q_cond_.notify_all();
}

std::string Engine::randId(std::string_view prefix, std::size_t n) {
    std::lock_guard<std::mutex> lk(rng_mutex_);
    return randomId(prefix, n, prngNext, &id_prng_);
}

Engine::Health Engine::health() {
    std::lock_guard<std::mutex> lk(q_mutex_);
    return {n_active_, queue_.size(), slots_.size()};
}

Engine::Stats Engine::stats() const {
    Stats s;
    s.spe_new = stat_spe_new_;
    s.spe_hits = stat_spe_hits_;
    s.spe_tok = stat_spe_tok_;
    s.spe_evict = stat_spe_evict_;
    s.unshare = stat_unshare_;
    s.evictions = stat_evictions_;
    s.cow_pages = stat_cow_pages_;
    s.q_waits = stat_q_waits_;
    s.spill_waits = stat_spill_waits_;
    s.vis_enc = stat_vis_enc_;
    s.vis_hit = stat_vis_hit_;
    return s;
}

DevPtr Engine::logitsRow(std::uint32_t r) const {
    return m_.logits() + static_cast<std::uint64_t>(r) * m_.cfg().n_vocab * 4;
}

// ---------------------------------------------------------------------------
// checkpoints (of the currently selected slot)

void Engine::saveCkpt(Slot& sl, std::uint32_t pos, const char* kind, std::optional<DevPtr> hid,
                      std::optional<DevPtr> logits) {
    if (opt_.no_prefix_cache || sl.ckpts.empty()) return;
    // item T3: a restore tail still arriving: done -> its checkpoints are
    // valid; else it stops where it is and the main stream waits for the
    // enqueued copies
    if (sl.restore != nullptr && sl.phase != Phase::restore) cutTail(sl);
    // a spill to the host tier may still be reading the checkpoint buffers
    waitSpill(sl);
    Ckpt* slot = &sl.ckpts[0];
    for (Ckpt& c : sl.ckpts) {
        if (c.valid && c.pos == pos) {
            slot = &c;
            break;
        }
        if (!c.valid) {
            if (slot->valid) slot = &c;
        } else if (slot->valid && c.seq < slot->seq) {
            slot = &c;
        }
    }
    m_.commitSeq(m_.curSeq());
    const auto conv = m_.convState();
    const auto ssm = m_.ssmState();
    const auto& cfg = m_.cfg();
    for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
        if (ssm[i] == 0) continue;
        ops_.copyAsync(slot->conv[i], conv[i], conv_bytes_, m_.stream());
        ops_.copyAsync(slot->ssm[i], ssm[i], ssm_bytes_, m_.stream());
    }
    if (hid) ops_.copyAsync(slot->hid, *hid, static_cast<std::uint64_t>(cfg.n_embd) * 4, m_.stream());
    if (logits) ops_.copyAsync(slot->logits, *logits, static_cast<std::uint64_t>(cfg.n_vocab) * 4, m_.stream());
    slot->has_logits = logits.has_value();
    sl.ckpt_seq += 1;
    slot->valid = true;
    slot->pos = pos;
    slot->seq = sl.ckpt_seq;
    slot->kind = kind;
}

void Engine::loadCkpt(const Ckpt& c) {
    m_.dropPending(m_.curSeq());
    const auto conv = m_.convState();
    const auto ssm = m_.ssmState();
    for (std::uint32_t i = 0; i < m_.cfg().n_layer; ++i) {
        if (ssm[i] == 0) continue;
        ops_.copyAsync(conv[i], c.conv[i], conv_bytes_, m_.stream());
        ops_.copyAsync(ssm[i], c.ssm[i], ssm_bytes_, m_.stream());
    }
}

void Engine::dropCache(Slot& sl) {
    for (Ckpt& c : sl.ckpts) c.valid = false;
    sl.cache_tokens.clear();
}

void Engine::freePages(Slot& sl) {
    const bool busy = sl.spill_busy && !ops_.eventDone(sl.spill_ev);
    sl.spill_busy = false;
    if (sl.spill_wanted) logW("slot {} | kv tier: deferred spill dropped (pages evicted)", sl.id);
    sl.spill_wanted = false;
    sl.tier_id = 0;
    sl.tier_pages = 0;
    dropCache(sl);
    if (busy) {
        try {
            quarantinePages(sl.pages);
        } catch (const std::exception&) {
            // cannot track them: order the main stream after the spill instead
            stat_spill_waits_ += 1;
            try {
                ops_.streamWaitEvent(m_.stream(), sl.spill_ev);
            } catch (const std::exception&) {
            }
            unrefPages(sl.pages);
        }
    } else {
        unrefPages(sl.pages);
    }
    sl.pages.clear();
}

// ---------------------------------------------------------------------------
// pool pages (item S: reference counts)

std::optional<std::int32_t> Engine::takeFreePage() {
    if (free_pages_.empty()) return std::nullopt;
    const std::int32_t p = free_pages_.back();
    free_pages_.pop_back();
    page_ref_[static_cast<std::size_t>(p)] = 1;
    return p;
}

void Engine::unrefPages(std::span<const std::int32_t> pages) {
    for (std::int32_t p : pages) {
        std::uint16_t& r = page_ref_[static_cast<std::size_t>(p)];
        r -= 1;
        if (r == 0) free_pages_.push_back(p);
    }
}

bool Engine::sharedFrom(const Slot& sl, std::uint32_t reuse) const {
    const std::size_t keep = reuse == 0 ? 0 : (reuse - 1) / kv_page;
    if (keep >= sl.pages.size()) return false;
    for (std::size_t i = keep; i < sl.pages.size(); ++i)
        if (page_ref_[static_cast<std::size_t>(sl.pages[i])] > 1) return true;
    return false;
}

void Engine::copyPage(std::int32_t dst_p, std::int32_t src_p) {
    const std::uint64_t src = static_cast<std::uint64_t>(src_p);
    const std::uint64_t dst = static_cast<std::uint64_t>(dst_p);
    for (const KvArr& a : kv_arrays_) {
        const std::uint64_t len = static_cast<std::uint64_t>(kv_page) * a.row;
        ops_.copyAsync(a.base + dst * len, a.base + src * len, len, m_.stream());
    }
}

std::optional<std::int32_t> Engine::takePage(Slot* sl) {
    for (;;) {
        if (free_pages_.empty() && !quarantine_.empty()) reclaimPages(false);
        if (auto p = takeFreePage()) return p;
        Slot* victim = nullptr;
        Spe* vspe = nullptr;
        std::uint64_t lu = std::numeric_limits<std::uint64_t>::max();
        for (Slot& o : slots_) {
            if ((sl != nullptr && &o == sl) || o.phase != Phase::idle || o.pages.empty()) continue;
            if (o.last_used < lu) {
                victim = &o;
                lu = o.last_used;
            }
        }
        for (Spe& s : spes_) {
            if (!s.valid || s.pin) continue;
            if (s.last_used < lu) {
                vspe = &s;
                victim = nullptr;
                lu = s.last_used;
            }
        }
        if (vspe) {
            logI("kv pool full: dropping shared {} checkpoint @{} ({} pages)", vspe->kind, vspe->n, vspe->pages.size());
            dropSpe(*vspe);
            continue;
        }
        if (!victim) {
            // only pages a spill still reads: wait for it on the GPU
            if (!quarantine_.empty()) {
                reclaimPages(true);
                continue;
            }
            return std::nullopt;
        }
        if (sl) {
            logI("kv pool full: evicting idle slot {} ({} pages, {} cached tokens) for slot {}", victim->id,
                 victim->pages.size(), victim->cache_tokens.size(), sl->id);
        } else {
            logI("kv pool full: evicting idle slot {} ({} pages, {} cached tokens)", victim->id, victim->pages.size(),
                 victim->cache_tokens.size());
        }
        stat_evictions_ += 1;
        freePages(*victim);
    }
}

void Engine::quarantinePages(std::span<const std::int32_t> pages) {
    if (pages.empty()) return;
    if (!tier_) throw std::runtime_error("NoTier");
    tier::Event ev = ops_.eventCreate();
    try {
        ops_.eventRecord(ev, tier_->stream);
    } catch (...) {
        ops_.eventDestroy(ev);
        throw;
    }
    quarantine_.push_back({std::vector<std::int32_t>(pages.begin(), pages.end()), ev});
}

void Engine::reclaimPages(bool force) {
    std::size_t i = 0;
    while (i < quarantine_.size()) {
        QBatch& q = quarantine_[i];
        bool done = ops_.eventDone(q.ev);
        if (!done && force) {
            stat_q_waits_ += 1;
            try {
                ops_.streamWaitEvent(m_.stream(), q.ev);
            } catch (const std::exception&) {
            }
            done = true;
        }
        if (!done) {
            ++i;
            continue;
        }
        unrefPages(q.pages);
        ops_.eventDestroy(q.ev);
        quarantine_.erase(quarantine_.begin() + static_cast<std::ptrdiff_t>(i));
    }
}

void Engine::cowPages(Slot& sl, std::uint32_t reuse, bool spill) {
    const std::size_t keep = reuse == 0 ? 0 : (reuse - 1) / kv_page;
    if (keep >= sl.pages.size()) return;
    std::optional<std::int32_t> newp;
    if (reuse > 0) {
        newp = takePage(&sl);
        if (!newp) {
            if (!spill) throw KvPoolFull();
            waitSpill(sl);
            return;
        }
        copyPage(*newp, sl.pages[keep]);
    }
    const std::span<const std::int32_t> old(sl.pages.data() + keep, sl.pages.size() - keep);
    if (spill) quarantinePages(old);
    else unrefPages(old);
    if (spill) stat_cow_pages_ += sl.pages.size() - keep;
    else stat_unshare_ += 1;
    sl.pages.resize(keep);
    if (newp) {
        sl.pages.push_back(*newp);
        m_.mapPages(sl.id, static_cast<std::uint32_t>(keep), std::span<const std::int32_t>(sl.pages.data() + keep, 1));
    }
    sl.tier_pages = std::min(sl.tier_pages, static_cast<std::uint32_t>(keep));
}

bool Engine::ensurePages(Slot& sl, std::uint64_t n_tok) {
    const std::size_t need = static_cast<std::size_t>((n_tok + kv_page - 1) / kv_page);
    if (need <= sl.pages.size()) return true;
    if (need > m_.seqPages()) return false;
    const std::size_t first = sl.pages.size();
    sl.pages.reserve(need);
    while (sl.pages.size() < need) {
        auto p = takePage(&sl);
        if (!p) break;
        sl.pages.push_back(*p);
    }
    m_.mapPages(sl.id, static_cast<std::uint32_t>(first),
                std::span<const std::int32_t>(sl.pages.data() + first, sl.pages.size() - first));
    return sl.pages.size() >= need;
}

// ---------------------------------------------------------------------------
// item S: shared prefix checkpoints

std::uint32_t Engine::sysSplit(std::span<const std::uint32_t> toks) const {
    if (opt_.sys_min == 0) return 0;
    const std::size_t b = ct_.sysBoundary(toks);
    if (b < opt_.sys_min || b >= toks.size()) return 0;
    return static_cast<std::uint32_t>(b);
}

std::size_t Engine::schedNext(std::size_t n, std::size_t sys, std::size_t think, std::size_t pos) {
    if (sys > pos) return sys;
    if (think > pos) return think;
    return n;
}

// vision: schedNext with the prompt's image-end splits (imgs sorted; empty for text)
std::size_t Engine::schedNextV(std::size_t n, std::size_t sys, std::size_t think, std::span<const std::uint32_t> imgs,
                               std::size_t pos) {
    std::size_t nx = schedNext(n, sys, think, pos);
    for (std::uint32_t b : imgs)
        if (b > pos) {
            nx = std::min<std::size_t>(nx, b);
            break;
        }
    return nx;
}

// vision: the image-end splits of toks (position after the <|vision_end|> that follows a
// run of image ids), those >= ck_min; a function of the tokens only, so every prompt
// with the same prefix gets the same splits there (at most 32)
std::vector<std::uint32_t> Engine::imgSplits(std::span<const std::uint32_t> toks) const {
    std::vector<std::uint32_t> out;
    if (vis_cfg_.vis == nullptr || vis_cfg_.ck_min == 0) return out;
    std::size_t i = 0;
    while (i < toks.size() && out.size() < 32) {
        if (toks[i] < qwen35::image_id_base) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < toks.size() && toks[j] >= qwen35::image_id_base) ++j;
        const std::size_t b = (j < toks.size() && toks[j] == vis_cfg_.vision_end_id) ? j + 1 : j;
        if (b >= vis_cfg_.ck_min && b < toks.size()) out.push_back(static_cast<std::uint32_t>(b));
        i = j + 1;
    }
    return out;
}

std::size_t Engine::schedFloor(std::span<const std::uint32_t> toks, std::size_t lim) const {
    const std::size_t sys = sysSplit(toks);
    const std::size_t th = ct_.thinkOpenSplit(toks);
    const std::vector<std::uint32_t> imgs = imgSplits(toks);
    std::size_t pos = 0, last = 0;
    while (pos < toks.size() && pos <= lim) {
        last = pos;
        pos += prefillChunkLen(schedNextV(toks.size(), sys, th, imgs, pos) - pos);
    }
    if (pos <= lim) last = pos;
    return last;
}

bool Engine::onSchedule(std::span<const std::uint32_t> toks, std::size_t a) const {
    return a == 0 || schedFloor(toks, a) == a;
}

Spe* Engine::speMatch(std::span<const std::uint32_t> toks) {
    if (opt_.no_prefix_cache) return nullptr;
    Spe* best = nullptr;
    for (Spe& s : spes_) {
        if (!s.valid || s.n >= toks.size()) continue;
        if (best && s.n <= best->n) continue;
        if (!prefixEq(s.tokens, toks.subspan(0, s.n))) continue;
        if (!onSchedule(toks, s.n)) continue;
        best = &s;
    }
    return best;
}

void Engine::dropSpe(Spe& s) {
    if (!s.valid) return;
    const bool busy = s.spill_busy && !ops_.eventDone(s.spill_ev);
    s.spill_busy = false;
    if (busy) {
        try {
            quarantinePages(s.pages);
        } catch (const std::exception&) {
            stat_spill_waits_ += 1;
            try {
                ops_.streamWaitEvent(m_.stream(), s.spill_ev);
            } catch (const std::exception&) {
            }
            unrefPages(s.pages);
        }
    } else {
        unrefPages(s.pages);
    }
    s.pages.clear();
    s.tokens.clear();
    s.valid = false;
    s.n = 0;
    s.tier_id = 0;
    stat_spe_evict_ += 1;
}

void Engine::createSpe(Slot& sl, std::span<const std::uint32_t> toks_all, std::uint32_t pos, const char* kind,
                       std::optional<DevPtr> hid_row, const Ckpt* src, std::uint64_t tier_id) {
    if (spes_.empty() || opt_.no_prefix_cache || pos == 0 || pos > toks_all.size()) return;
    const auto toks = toks_all.subspan(0, pos);
    use_seq_ += 1;
    for (Spe& s : spes_) {
        if (s.valid && s.n == pos && prefixEq(s.tokens, toks)) {
            s.last_used = use_seq_;
            return;
        }
    }
    const std::size_t last = (pos - 1) / kv_page;
    if (sl.pages.size() <= last) return;
    Spe* v = nullptr;
    for (Spe& s : spes_) {
        if (s.pin) continue;
        if (!s.valid) {
            v = &s;
            break;
        }
        if (!v || s.last_used < v->last_used) v = &s;
    }
    if (!v) return;
    Spe& s = *v;
    if (s.valid) logI("shared prefix: replacing {} checkpoint @{} (used {}x)", s.kind, s.n, s.uses);
    dropSpe(s);
    s.pin = true;
    struct Unpin {
        Spe& s;
        ~Unpin() { s.pin = false; }
    } unpin{s};
    s.tokens.reserve(pos);
    s.pages.reserve(last + 1);
    auto np = takePage(&sl);
    if (!np) return;
    copyPage(*np, sl.pages[last]);
    for (std::size_t i = 0; i < last; ++i) {
        refPage(sl.pages[i]);
        s.pages.push_back(sl.pages[i]);
    }
    s.pages.push_back(*np);
    // a spill of the previous content may still read the state buffers
    if (s.spill_ev != nullptr && !ops_.eventDone(s.spill_ev)) {
        stat_spill_waits_ += 1;
        ops_.streamWaitEvent(m_.stream(), s.spill_ev);
    }
    Ckpt& c = s.ck;
    const auto& cfg = m_.cfg();
    if (src) {
        for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
            if (c.conv[i] == 0) continue;
            ops_.copyAsync(c.conv[i], src->conv[i], conv_bytes_, m_.stream());
            ops_.copyAsync(c.ssm[i], src->ssm[i], ssm_bytes_, m_.stream());
        }
        ops_.copyAsync(c.hid, src->hid, static_cast<std::uint64_t>(cfg.n_embd) * 4, m_.stream());
    } else {
        m_.selectSeq(sl.id);
        m_.commitSeq(m_.curSeq());
        const auto conv = m_.convState();
        const auto ssm = m_.ssmState();
        for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
            if (ssm[i] == 0) continue;
            ops_.copyAsync(c.conv[i], conv[i], conv_bytes_, m_.stream());
            ops_.copyAsync(c.ssm[i], ssm[i], ssm_bytes_, m_.stream());
        }
        if (hid_row) ops_.copyAsync(c.hid, *hid_row, static_cast<std::uint64_t>(cfg.n_embd) * 4, m_.stream());
    }
    c.valid = true;
    c.pos = pos;
    c.seq = 1;
    c.kind = kind;
    c.has_logits = false;
    s.tokens.assign(toks.begin(), toks.end());
    s.n = pos;
    s.kind = kind;
    s.valid = true;
    s.last_used = use_seq_;
    s.uses = 0;
    s.tier_id = tier_id;
    stat_spe_new_ += 1;
    logI("req {} | slot {} | shared prefix: kept {} checkpoint @{} ({} pages shared, 1 copied){}", sl.reqId(), sl.id,
         kind, pos, last, src ? " from the restored entry" : "");
    if (tier_id == 0) {
        try {
            spillSpe(s);
        } catch (const std::exception& ex) {
            logW("shared prefix: tier spill failed: {} ({})", ex.what(), ops_.lastErrorString());
        }
    }
}

void Engine::attachSpe(Slot& sl, Spe& s) {
    s.pin = true;
    struct Unpin {
        Spe& s;
        ~Unpin() { s.pin = false; }
    } unpin{s};
    freePages(sl);
    const std::size_t last = s.pages.size() - 1;
    sl.pages.reserve(last + 1);
    sl.cache_tokens.reserve(s.n);
    auto np = takePage(&sl);
    if (!np) throw KvPoolFull();
    for (std::size_t i = 0; i < last; ++i) {
        refPage(s.pages[i]);
        sl.pages.push_back(s.pages[i]);
    }
    sl.pages.push_back(*np);
    copyPage(*np, s.pages[last]);
    m_.mapPages(sl.id, 0, sl.pages);
    sl.cache_tokens.insert(sl.cache_tokens.end(), s.tokens.begin(), s.tokens.end());
    use_seq_ += 1;
    s.last_used = use_seq_;
    s.uses += 1;
    stat_spe_hits_ += 1;
    stat_spe_tok_ += s.n;
}

void Engine::spillSpe(Spe& s) {
    tier::Tier* t = tier_;
    if (!t) return;
    if (s.n < t->cfg.min_tokens) return;
    // already there (restored from it, or written before a restart)
    for (tier::Entry* x : t->entries) {
        if (x->nck != 1 || x->ck[0].valid == 0 || x->ck[0].pos != s.n || x->n_tok != s.n) continue;
        if (x->tokens.size() < s.n || !std::equal(s.tokens.begin(), s.tokens.end(), x->tokens.begin())) continue;
        s.tier_id = x->id;
        return;
    }
    const tier::Layout lay = t->lay;
    tier::Entry* x = t->newEntry(1);
    const std::uint32_t npages = static_cast<std::uint32_t>(s.pages.size());
    const std::uint64_t kv_off = x->kvOff(lay);
    bool ok = true;
    {
        const auto r = tier::Tier::lbRange(0, lay.ck_bytes);
        if (!t->ensureBlocks(x, r[0], r[1])) ok = false;
    }
    if (ok) {
        const auto r = tier::Tier::lbRange(kv_off, static_cast<std::uint64_t>(npages) * lay.page_bytes);
        if (!t->ensureBlocks(x, r[0], r[1])) ok = false;
    }
    if (!ok) {
        logW("shared prefix: no room for {} tokens in the RAM tier; not kept there", s.n);
        t->removeEntry(x);
        return;
    }
    // the state / page copies into the checkpoint run on the main stream
    // (mid-prefill): the tier stream starts after them
    if (s.ready_ev == nullptr) s.ready_ev = ops_.eventCreate();
    ops_.eventRecord(s.ready_ev, m_.stream());
    ops_.streamWaitEvent(t->stream, s.ready_ev);
    tier_spans_.clear();
    ckSpans(s.ck, 0);
    for (std::size_t j = 0; j < s.pages.size(); ++j) pageSpans(s.pages[j], kv_off + j * lay.page_bytes);
    const std::uint64_t bytes = t->copyOut(x, tier_spans_);
    x->tokens.insert(x->tokens.end(), s.tokens.begin(), s.tokens.end());
    x->n_tok = s.n;
    x->pages = npages;
    x->ck[0] = tier::CkMeta{s.n, 1, 0, ckKindCode(s.kind), 0, 1};
    t->trim(x);
    x->t_mod = Clock::now();
    x->owner.reset();
    if (s.spill_ev == nullptr) s.spill_ev = ops_.eventCreate();
    t->fence(x, s.spill_ev);
    s.spill_busy = true;
    s.tier_id = x->id;
    t->stats.spills += 1;
    t->stats.spill_bytes += bytes;
    logI("shared prefix: spill {} checkpoint @{} to RAM ({:.1f} MiB)", s.kind, s.n, static_cast<double>(bytes) / 1048576.0);
}

std::uint32_t Engine::lcpTarget(std::span<const std::uint32_t> toks, std::uint32_t reuse, std::uint32_t sys) {
    if (!opt_.lcp_on || spes_.empty() || opt_.no_prefix_cache) return 0;
    std::size_t L = 0;
    for (Slot& o : slots_) L = std::max(L, lcpLen(o.cache_tokens, toks));
    for (Spe& s : spes_)
        if (s.valid) L = std::max(L, lcpLen(s.tokens, toks));
    if (tier_)
        for (tier::Entry* x : tier_->entries) L = std::max(L, lcpLen(x->tokens, toks));
    const std::size_t mn = opt_.sys_min > 0 ? opt_.sys_min : 2048;
    if (L < mn || L < 2) return 0;
    const std::size_t a = schedFloor(toks, std::min(L, toks.size() - 1));
    if (a < mn || a == sys || a < reuse + static_cast<std::size_t>(kv_page) * 4) return 0;
    return static_cast<std::uint32_t>(a);
}

bool Engine::waitsForSys(Job& job) {
    if (spes_.empty() || opt_.no_prefix_cache || !job.params.cache_prompt) return false;
    const std::span<const std::uint32_t> toks = job.tokens;
    const std::uint32_t b = sysSplit(toks);
    if (b == 0) return false;
    if (Spe* s = speMatch(toks); s && s->n >= b) return false;
    for (Slot& sl : slots_) {
        if (sl.phase != Phase::prefill || !sl.on_sched || sl.sys_split != b) continue;
        if (sl.reuse + sl.pf_off >= b) continue;
        const auto& jt = sl.job->tokens;
        if (jt.size() < b || !std::equal(jt.begin(), jt.begin() + b, toks.begin())) continue;
        if (!job.sys_wait_logged) {
            logI("req {} | waiting for the system-message checkpoint @{} that slot {} is prefilling", job.id, b, sl.id);
            job.sys_wait_logged = true;
        }
        return true;
    }
    return false;
}

bool Engine::waitsForTier(Job& job) {
    tier::Tier* t = tier_;
    if (!t) return false;
    if (!job.params.cache_prompt || opt_.no_prefix_cache) return false;
    const std::span<const std::uint32_t> toks = job.tokens;
    for (Slot& sl : slots_) {
        if (sl.phase != Phase::restore) continue;
        tier::Restore* r = sl.restore;
        if (!r) continue;
        tier::Entry* x = t->find(r->entry_id);
        if (!x) continue;
        const std::span<const std::uint32_t> et(x->tokens.data(), std::min<std::size_t>(x->n_tok, x->tokens.size()));
        const std::size_t lcp = lcpLen(et, toks);
        std::uint32_t reuse = 0;
        std::uint8_t kind = 0;
        for (std::uint32_t k = 0; k < x->nck; ++k) {
            const tier::CkMeta& c = x->ck[k];
            if (c.valid == 0 || c.pos == 0 || c.pos > lcp || c.pos > toks.size()) continue;
            if (c.pos == toks.size() && c.has_logits == 0) continue;
            if (c.pos > reuse) {
                reuse = c.pos;
                kind = c.kind;
            }
        }
        if (reuse < t->cfg.min_tokens) continue;
        if (kind >= 4 && !onSchedule(toks, reuse)) continue;
        std::uint32_t vr = 0;
        if (Spe* s = speMatch(toks)) vr = s->n;
        for (Slot& o : slots_)
            if (o.phase == Phase::idle) vr = std::max(vr, cacheMatch(o, toks).reuse);
        if (reuse <= vr || reuse - vr < 512) continue;
        if (!job.tier_wait_logged) {
            logI("req {} | waiting for the restore of its prefix ({} tok) into slot {}", job.id, reuse, sl.id);
            job.tier_wait_logged = true;
        }
        return true;
    }
    return false;
}

void Engine::prefetchFor(Job& job) {
    tier::Tier* t = tier_;
    if (!t) return;
    if (job.pf_checked) return;
    job.pf_checked = true;
    if (!job.params.cache_prompt || opt_.no_prefix_cache) return;
    const std::span<const std::uint32_t> toks = job.tokens;
    const auto mt = t->lookup(toks);
    if (!mt) return;
    tier::Entry* x = mt->e;
    if (x->in_ram || x->loading || x->io_busy) return;
    if (x->ck[mt->ck].kind >= 4 && !onSchedule(toks, mt->reuse)) return;
    std::uint32_t vr = 0;
    if (Spe* s = speMatch(toks)) vr = s->n;
    for (Slot& o : slots_) vr = std::max(vr, cacheMatch(o, toks).reuse);
    if (mt->reuse <= vr || mt->reuse - vr < 512) return;
    bool ok = false;
    try {
        ok = t->prefetch(x);
    } catch (const std::exception&) {
        ok = false;
    }
    if (ok) logI("req {} | kv tier: prefetching {} tok from SSD into RAM while queued", job.id, x->n_tok);
}

Engine::CacheMatch Engine::cacheMatch(Slot& sl, std::span<const std::uint32_t> toks) {
    if (opt_.no_prefix_cache) return {};
    const std::size_t lcp = lcpLen(sl.cache_tokens, toks);
    Ckpt* ck = nullptr;
    for (Ckpt& c : sl.ckpts) {
        if (!c.valid || c.pos > lcp || c.pos > toks.size() || c.pos == 0) continue;
        if (c.pos == toks.size() && !c.has_logits) continue;
        if (!ck || c.pos > ck->pos) ck = &c;
    }
    return {ck ? ck->pos : 0, ck, lcp};
}

Slot* Engine::pickSlot(std::span<const std::uint32_t> toks) {
    Slot* best = nullptr;
    std::uint32_t best_reuse = 0;
    const std::uint32_t spe_n = [&] {
        Spe* s = speMatch(toks);
        return s ? s->n : 0u;
    }();
    for (Slot& sl : slots_) {
        if (sl.phase != Phase::idle) continue;
        const std::uint32_t r = std::max(cacheMatch(sl, toks).reuse, spe_n);
        if (!best || r > best_reuse || (r == best_reuse && sl.last_used < best->last_used)) {
            best = &sl;
            best_reuse = r;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// sampling

std::size_t Engine::sampBytes() {
    return static_cast<std::size_t>(max_rows) * k_small * 8 + max_rows * 8;
}

void Engine::launchTopk(std::uint32_t row0, std::uint32_t rows, std::uint32_t k, float inv_t, DevPtr dst,
                        std::uint32_t dst_rows, std::uint32_t out0) {
    const DevPtr ids = dst + static_cast<std::uint64_t>(out0) * k * 4;
    const DevPtr vals = dst + static_cast<std::uint64_t>(dst_rows) * k * 4 + static_cast<std::uint64_t>(out0) * k * 4;
    const DevPtr stats = dst + static_cast<std::uint64_t>(dst_rows) * k * 8 + static_cast<std::uint64_t>(out0) * 8;
    m_.topkRows(row0, rows, k, inv_t, ids, vals, stats);
}

void Engine::fetchCands(Sampler& s, std::uint32_t rows) {
    s.row_base = 0;
    launchTopk(0, rows, k_small, s.inv_t, samp_dev_, max_rows, 0);
    ops_.download(samp_host_.data(), samp_dev_, sampBytes());
}

bool filterCands(Sampler& s, std::size_t count, bool complete) {
    std::vector<Cand>& cv = *s.cands;
    Cand* c = cv.data();
    std::sort(c, c + count, [](const Cand& x, const Cand& y) {
        if (x.logit != y.logit) return x.logit > y.logit;
        return x.id < y.id;
    });
    if (complete) {
        // exact host normalizer for the full row
        float mx = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < count; ++i) mx = std::max(mx, c[i].logit);
        double z = 0;
        for (std::size_t i = 0; i < count; ++i) z += std::exp(static_cast<double>(c[i].logit - mx) * s.inv_t);
        s.m = mx;
        s.sum = static_cast<float>(z);
    }
    for (std::size_t i = 0; i < count; ++i)
        c[i].p = std::exp(static_cast<double>(c[i].logit - s.m) * s.inv_t) / static_cast<double>(s.sum);
    std::size_t n = count;
    bool closed = complete;
    if (s.top_k > 0) {
        if (s.top_k <= count) {
            n = s.top_k;
            closed = true;
        } else if (!complete) {
            return false;
        }
    }
    if (s.top_p < 1.0) {
        double z = 1.0;
        if (closed) {
            z = 0;
            for (std::size_t i = 0; i < n; ++i) z += c[i].p;
        }
        double cum = 0;
        bool found = false;
        for (std::size_t j = 0; j < n; ++j) {
            cum += c[j].p / z;
            if (cum >= s.top_p) {
                n = j + 1;
                found = true;
                break;
            }
        }
        if (found) closed = true;
        else if (!closed) return false;
    }
    if (s.min_p > 0) {
        const double thr = s.min_p * c[0].p;
        std::size_t j = 0;
        while (j < n && c[j].p >= thr) ++j;
        if (j == n && !closed) return false;
        n = std::max<std::size_t>(j, 1);
        closed = true;
    }
    if (closed) {
        double z = 0;
        for (std::size_t i = 0; i < n; ++i) z += c[i].p;
        for (std::size_t i = 0; i < n; ++i) c[i].p /= z;
    }
    s.n = n;
    s.closed = closed;
    return true;
}

void Engine::prepareRow(Sampler& s, std::uint32_t r) {
    const std::uint32_t gr = s.row_base + r;
    s.row = gr;
    const auto* host = samp_host_.data();
    const auto* ids = reinterpret_cast<const std::int32_t*>(host);
    const auto* vals = reinterpret_cast<const float*>(host + static_cast<std::size_t>(max_rows) * k_small * 4);
    const auto* stats = reinterpret_cast<const float*>(host + static_cast<std::size_t>(max_rows) * k_small * 8);
    s.m = stats[2 * gr];
    s.sum = stats[2 * gr + 1];
    std::vector<Cand>& c = *s.cands;
    for (std::uint32_t j = 0; j < k_small; ++j) {
        const std::size_t idx = static_cast<std::size_t>(gr) * k_small + j;
        c[j] = {static_cast<std::uint32_t>(ids[idx]), vals[idx], 0};
    }
    if (filterCands(s, k_small, false)) return;
    // escalate: a larger device top-k for this row, then the full row
    s.big_used += 1;
    launchTopk(gr, 1, k_big, s.inv_t, big_dev_, 1, 0);
    ops_.download(big_host_.data(), big_dev_, big_host_.size());
    {
        const auto* bids = reinterpret_cast<const std::int32_t*>(big_host_.data());
        const auto* bvals = reinterpret_cast<const float*>(big_host_.data() + static_cast<std::size_t>(k_big) * 4);
        const auto* bst = reinterpret_cast<const float*>(big_host_.data() + static_cast<std::size_t>(k_big) * 8);
        s.m = bst[0];
        s.sum = bst[1];
        for (std::uint32_t j = 0; j < k_big; ++j) c[j] = {static_cast<std::uint32_t>(bids[j]), bvals[j], 0};
    }
    if (filterCands(s, k_big, false)) return;
    s.full_used += 1;
    loadFullRow(gr);
    const std::uint32_t V = m_.cfg().n_vocab;
    for (std::uint32_t j = 0; j < V; ++j) c[j] = {j, row_host_[j], 0};
    filterCands(s, V, true);
}

void Engine::loadFullRow(std::uint32_t gr) {
    const std::uint32_t V = m_.cfg().n_vocab;
    ops_.download(row_host_.data(), logitsRow(gr), static_cast<std::uint64_t>(V) * 4);
}

double Engine::probOf(Sampler& s, std::uint32_t t) {
    const std::vector<Cand>& c = *s.cands;
    for (std::size_t i = 0; i < s.n; ++i)
        if (c[i].id == t) return c[i].p;
    if (s.closed) return 0;
    float l = 0;
    ops_.download(&l, logitsRow(s.row) + static_cast<std::uint64_t>(t) * 4, 4);
    return std::exp(static_cast<double>(l - s.m) * s.inv_t) / static_cast<double>(s.sum);
}

std::uint32_t Engine::sampleRow(Sampler& s, std::optional<std::uint32_t> ex, double p_ex) {
    const double u = s.uniform();
    const double target = u * (1.0 - p_ex);
    double cum = 0;
    std::optional<std::uint32_t> last;
    const std::vector<Cand>& c = *s.cands;
    for (std::size_t i = 0; i < s.n; ++i) {
        if (ex && c[i].id == *ex) continue;
        cum += c[i].p;
        last = c[i].id;
        if (cum > target) return c[i].id;
    }
    if (s.closed) return last ? *last : c[0].id;
    // open distribution and the draw fell in the tail: walk the rest of the vocab
    s.full_used += 1;
    loadFullRow(s.row);
    const std::uint32_t V = m_.cfg().n_vocab;
    std::fill(in_cands_.begin(), in_cands_.begin() + V, std::uint8_t{0});
    for (std::size_t i = 0; i < s.n; ++i) in_cands_[c[i].id] = 1;
    for (std::uint32_t j = 0; j < V; ++j) {
        if (in_cands_[j]) continue;
        if (ex && j == *ex) continue;
        cum += std::exp(static_cast<double>(row_host_[j] - s.m) * s.inv_t) / static_cast<double>(s.sum);
        last = j;
        if (cum > target) return j;
    }
    return last ? *last : c[0].id;
}

// ---------------------------------------------------------------------------
// request lifecycle

bool Engine::emit(Slot& sl, std::uint32_t t) {
    if (!sl.ignore_eos && (t == ct_.im_end || t == ct_.endoftext)) {
        sl.finish = "stop";
        return true;
    }
    sl.batch[sl.n_batch] = t;
    sl.n_batch += 1;
    sl.n_gen += 1;
    if (sl.n_gen >= sl.max_tokens) {
        sl.finish = "length";
        return true;
    }
    return false;
}

void Engine::flushOut(Slot& sl) {
    if (sl.st.push(sl.out, std::span<const std::uint32_t>(sl.batch.data(), sl.n_batch))) {
        sl.done = true;
        sl.stopped_str = true;
    }
    sl.n_batch = 0;
}

bool Engine::startJob(Slot& sl, Job& job) {
    const Params& p = job.params;
    const std::span<const std::uint32_t> toks = job.tokens;
    const std::uint32_t N = static_cast<std::uint32_t>(toks.size());
    const std::uint32_t nd_max = opt_.use_mtp ? opt_.n_draft : 0;
    const auto& cfg = m_.cfg();
    sl.job = &job;
    use_seq_ += 1;
    sl.last_used = use_seq_;
    if (N + nd_max + 2 >= ctx_ || static_cast<std::uint64_t>(N) + nd_max + 2 >= static_cast<std::uint64_t>(pool_pages_) * kv_page) {
        writeError(job, 400, "prompt is too long for the context size");
        return false;
    }
    const std::uint32_t room = ctx_ - N - nd_max - 2;
    sl.max_tokens = std::min(p.max_tokens.value_or(room), room);
    sl.greedy = p.temperature <= 0;
    sl.ignore_eos = p.ignore_eos;
    sl.n_prompt = N;
    std::uint64_t seed = p.seed;
    if (!p.seed_given) {
        std::lock_guard<std::mutex> lk(rng_mutex_);
        seed = id_prng_();
    }
    sl.sm = Sampler{};
    sl.sm.temp = p.temperature;
    sl.sm.inv_t = sl.greedy ? 1.0f : 1.0f / p.temperature;
    sl.sm.top_k = p.top_k;
    sl.sm.top_p = p.top_p;
    sl.sm.min_p = p.min_p;
    sl.sm.seed = seed;
    sl.sm.cands = &cands_scratch_;
    if (sl.greedy) {
        logI("req {} | slot {} | sampling: greedy (temperature 0), max_tokens {}{}, stop strings {}{}", job.id, sl.id,
             sl.max_tokens, p.max_tokens ? "" : " (ctx limit)", p.stop.size(), p.ignore_eos ? ", ignore_eos" : "");
    } else {
        logI("req {} | slot {} | sampling: temperature {:.2f}, top_k {}, top_p {:.3f}, min_p {:.3f}, seed {}{}, max_tokens "
             "{}{}, stop strings {}{}",
             job.id, sl.id, p.temperature, p.top_k, p.top_p, p.min_p, seed, p.seed_given ? "" : " (random)", sl.max_tokens,
             p.max_tokens ? "" : " (ctx limit)", p.stop.size(), p.ignore_eos ? ", ignore_eos" : "");
    }
    if (p.presence_penalty != 0 || p.frequency_penalty != 0) {
        logW("req {} | slot {} | presence_penalty {:.2f} / frequency_penalty {:.2f} are accepted but not applied", job.id,
             sl.id, p.presence_penalty, p.frequency_penalty);
    }

    // ---- prefix cache lookup (this slot's KV region and checkpoints)
    CacheMatch mt = p.cache_prompt ? cacheMatch(sl, toks) : CacheMatch{};
    // item S: a shared prefix checkpoint longer than the slot's own reusable prefix
    Spe* spe_hit = nullptr;
    if (p.cache_prompt) {
        if (Spe* s = speMatch(toks); s && s->n > mt.reuse) spe_hit = s;
    }
    const std::uint32_t sys = sysSplit(toks);
    const std::uint32_t lcp_t = lcpTarget(toks, spe_hit ? spe_hit->n : mt.reuse, sys);
    if (spe_hit) {
        attachSpe(sl, *spe_hit);
        mt = CacheMatch{spe_hit->n, &spe_hit->ck, std::max<std::size_t>(mt.lcp, spe_hit->n)};
    }
    sl.reuse = mt.reuse;
    // the state at reuse is the cold one (a shared checkpoint is cold-exact)
    sl.on_sched = mt.reuse == 0 || spe_hit != nullptr ||
                  (mt.ck != nullptr && ckKindCode(mt.ck->kind) >= 4 && onSchedule(toks, mt.reuse));
    // positions >= reuse are rewritten: pages a spill still reads, or that
    // are shared, are replaced (copy on write), never waited for
    bool spill = false;
    if (sl.spill_busy && mt.reuse < sl.spill_upto) {
        if (ops_.eventDone(sl.spill_ev)) sl.spill_busy = false;
        else spill = true;
    }
    if (spill || sharedFrom(sl, mt.reuse)) cowPages(sl, mt.reuse, spill);
    sl.tier_pages = std::min(sl.tier_pages, mt.reuse / kv_page);
    // a request that reuses nothing of the slot starts a new lineage
    if (mt.reuse == 0) {
        sl.tier_id = 0;
        sl.tier_pages = 0;
    }
    for (Ckpt& c : sl.ckpts)
        if (c.valid && c.pos > mt.reuse) c.valid = false;
    sl.cache_tokens.resize(std::min<std::size_t>(sl.cache_tokens.size(), mt.reuse));
    {
        const std::size_t need_p =
            static_cast<std::size_t>((static_cast<std::uint64_t>(N) + nd_max + 2 + kv_page - 1) / kv_page);
        if (sl.pages.size() > need_p) {
            const std::span<const std::int32_t> extra(sl.pages.data() + need_p, sl.pages.size() - need_p);
            if (sl.spill_busy && !ops_.eventDone(sl.spill_ev)) quarantinePages(extra);
            else unrefPages(extra);
            sl.pages.resize(need_p);
            sl.tier_pages = std::min(sl.tier_pages, static_cast<std::uint32_t>(need_p));
        }
    }
    if (opt_.no_prefix_cache) {
        logI("req {} | slot {} | prompt {} tok, prefix cache disabled, prefill {} new", job.id, sl.id, N, N);
    } else if (mt.ck) {
        logI("req {} | slot {} | prompt {} tok, common prefix {}, reused {} tok (checkpoint '{}' @{}), prefill {} new",
             job.id, sl.id, N, mt.lcp, mt.reuse, mt.ck->kind, mt.ck->pos, N - mt.reuse);
    } else {
        logI("req {} | slot {} | prompt {} tok, common prefix {}, reused 0 tok (no usable checkpoint), prefill {} new",
             job.id, sl.id, N, mt.lcp, N);
    }
    sl.tp0 = sl.tp0_pre ? *sl.tp0_pre : Clock::now();
    sl.tp0_pre.reset();
    // vision: encode (or reuse) the images whose tokens are prefilled
    visPrepare(sl, job, opt_.verify_cache ? 0 : mt.reuse);
    m_.selectSeq(sl.id);
    sl.ck_hid.reset();
    if (mt.ck) {
        const Ckpt& c = *mt.ck;
        loadCkpt(c);
        sl.ck_hid = c.hid;
        if (spe_hit != nullptr && opt_.use_mtp) {
            // own copy of the MTP hidden row: the shared checkpoint may be
            // replaced before this slot's first chunk runs
            const DevPtr own = sl.ckpts[0].hid;
            waitSpill(sl);
            ops_.copyAsync(own, c.hid, static_cast<std::uint64_t>(cfg.n_embd) * 4, m_.stream());
            sl.ck_hid = own;
        }
        if (mt.reuse == N) {
            ops_.copyAsync(m_.logits(), c.logits, static_cast<std::uint64_t>(cfg.n_vocab) * 4, m_.stream());
            if (opt_.use_mtp) ops_.copyAsync(m_.mtpH(), c.hid, static_cast<std::uint64_t>(cfg.n_embd) * 4, m_.stream());
        }
    } else {
        m_.reset();
    }
    sl.cache_tokens.insert(sl.cache_tokens.end(), toks.begin() + mt.reuse, toks.end());
    sl.pf_off = 0;
    {
        const std::uint32_t sp = static_cast<std::uint32_t>(ct_.thinkOpenSplit(toks));
        sl.pf_split = sp > mt.reuse ? sp : 0;
        sl.sys_split = sys > mt.reuse ? sys : 0;
        sl.lcp_ck = (sl.on_sched && lcp_t > mt.reuse && lcp_t != sl.sys_split) ? lcp_t : 0;
        if (sl.lcp_ck > 0)
            logI("req {} | slot {} | shared prefix: common prefix with an earlier session; keeping a 'prefix' checkpoint @{}",
                 job.id, sl.id, sl.lcp_ck);
        // vision: a shared checkpoint at the end of the last image (follow-up questions
        // about the same image in new conversations start there)
        sl.img_ck = 0;
        if (sl.on_sched && job.vis) {
            const std::vector<std::uint32_t> imgs = imgSplits(toks);
            if (!imgs.empty()) {
                const std::uint32_t b = imgs.back();
                if (b > mt.reuse && b != sl.sys_split && b != sl.lcp_ck) {
                    sl.img_ck = b;
                    logI("req {} | slot {} | shared prefix: keeping a 'prefix' checkpoint at the end of the image @{}", job.id, sl.id, b);
                }
            }
        }
    }
    sl.phase = Phase::prefill;
    // reset per-request counters
    sl.n_gen = 0;
    sl.finish = "length";
    sl.done = false;
    sl.stopped_str = false;
    sl.cycles = 0;
    sl.drafted = 0;
    sl.accepted = 0;
    sl.n_batch = 0;
    sl.acc_ema = static_cast<float>(opt_.n_draft);
    sl.dacc = qwen35::DraftAccept{};
    sl.ng.reset();
    sl.ng.ng.norm = crlf_norm_;
    sl.ng_cycles = 0;
    if (opt_.timing_reset && opt_.use_mtp) {
        bool others = false;
        for (Slot& o : slots_) others = others || (&o != &sl && o.phase == Phase::decode);
        if (!others) {
            for (auto& t : timing_) t = qwen35::DraftTiming{};
            prev_nd_ = 0;
        }
    }
    if (mt.reuse == N) finishPrefill(sl);
    return true;
}

std::size_t Engine::chunkLen(const Slot& sl) const {
    const std::size_t cur = sl.reuse + sl.pf_off;
    const std::size_t lim = schedNextV(sl.job->tokens.size(), sl.sys_split, sl.pf_split, imgSplits(sl.job->tokens), cur);
    std::size_t n = prefillChunkLen(lim - cur);
    if (opt_.prefill_exec <= prefill_chunk) return n;
    for (const Slot& o : slots_)
        if (&o != &sl && o.phase == Phase::decode) return n;
    while (cur + n < lim) {
        if (sl.lcp_ck > 0 && cur + n == sl.lcp_ck) break;  // a shared-prefix checkpoint is kept there
        const std::size_t nx = prefillChunkLen(lim - cur - n);
        if (n + nx > opt_.prefill_exec + prefill_merge) break;
        n += nx;
    }
    return n;
}

void Engine::maybeSplitCkpt(Slot& sl, DevPtr hid_row) {
    const std::size_t cur = sl.reuse + sl.pf_off;
    // item S: shared prefix checkpoints (a failure only loses the checkpoint)
    if (sl.on_sched && ((sl.sys_split > 0 && cur == sl.sys_split) || (sl.lcp_ck > 0 && cur == sl.lcp_ck) || (sl.img_ck > 0 && cur == sl.img_ck))) {
        const char* kind = cur == sl.sys_split ? "system" : "prefix";
        try {
            createSpe(sl, sl.job->tokens, static_cast<std::uint32_t>(cur), kind,
                      opt_.use_mtp ? std::optional<DevPtr>(hid_row) : std::nullopt, nullptr, 0);
        } catch (const std::exception& ex) {
            logW("req {} | slot {} | shared prefix: checkpoint @{} not kept: {} ({})", sl.reqId(), sl.id, cur, ex.what(),
                 ops_.lastErrorString());
        }
    }
    if (sl.pf_split == 0 || cur != sl.pf_split) return;
    m_.selectSeq(sl.id);
    saveCkpt(sl, sl.pf_split, "think-open", opt_.use_mtp ? std::optional<DevPtr>(hid_row) : std::nullopt, std::nullopt);
}

void Engine::prefillStep(Slot& sl) {
    Job& job = *sl.job;
    const std::span<const std::uint32_t> toks = std::span<const std::uint32_t>(job.tokens).subspan(sl.reuse);
    const std::size_t n = chunkLen(sl);
    if (!ensurePages(sl, sl.reuse + sl.pf_off + n)) throw KvPoolFull();
    m_.selectSeq(sl.id);
    if (opt_.use_mtp) {
        m_.prefillMtpChunk(toks, sl.reuse, sl.pf_off, n, sl.ck_hid);
    } else {
        m_.forward(toks.subspan(sl.pf_off, n), sl.reuse + static_cast<std::uint32_t>(sl.pf_off));
    }
    sl.pf_off += n;
    stat_prefill_tok_ += n;
    maybeSplitCkpt(sl, m_.hn() + static_cast<std::uint64_t>(n - 1) * m_.cfg().n_embd * 4);
    if (sl.pf_off == toks.size()) finishPrefill(sl);
}

// Returns the rows run (0 when nothing ran). row_budget caps the rows of the
// whole forward; the oldest request's chunk always runs (decode floor).
std::size_t Engine::prefillGroup(Slot& first, std::size_t row_budget) {
    const std::size_t n0 = chunkLen(first);
    const std::size_t small_max = max_small_batch;
    auto solo = [&] {
        try {
            prefillStep(first);
        } catch (const std::exception& ex) {
            failSlot(first, ex.what());
        }
    };
    if (!opt_.seg_prefill || n0 <= small_max || !m_.canSegment()) {
        solo();
        return n0;
    }
    const std::size_t cap = std::min<std::size_t>(m_.maxBatch(), std::max(row_budget, n0));
    // the other prefilling slots, oldest first
    std::vector<Slot*> cand;
    for (Slot& sl : slots_) {
        if (sl.phase != Phase::prefill || &sl == &first) continue;
        std::size_t at = cand.size();
        cand.push_back(&sl);
        while (at > 0 && cand[at - 1]->job->id > sl.job->id) {
            cand[at] = cand[at - 1];
            --at;
        }
        cand[at] = &sl;
    }
    std::vector<Slot*> grp{&first};
    std::size_t total = n0;
    for (Slot* sl : cand) {
        if (grp.size() == max_small_batch) break;
        const std::size_t n = chunkLen(*sl);
        // a chunk that cannot join waits for its own turn (it is then the oldest)
        if (n <= small_max || total + n > cap) continue;
        grp.push_back(sl);
        total += n;
    }
    if (grp.size() == 1) {
        solo();
        return n0;
    }
    // pool pages for every chunk of the group (a slot that cannot get them fails alone)
    {
        std::vector<Slot*> kept;
        for (Slot* sl : grp) {
            bool ok = false;
            try {
                ok = ensurePages(*sl, sl->reuse + sl->pf_off + chunkLen(*sl));
            } catch (const std::exception&) {
                ok = false;
            }
            if (!ok) {
                failSlot(*sl, "KvPoolFull");
                continue;
            }
            kept.push_back(sl);
        }
        grp = std::move(kept);
        if (grp.empty()) return 0;
    }
    std::vector<PSeg> segs(grp.size());
    for (std::size_t k = 0; k < grp.size(); ++k) {
        Slot& sl = *grp[k];
        segs[k].seq = sl.id;
        segs[k].tokens = std::span<const std::uint32_t>(sl.job->tokens).subspan(sl.reuse);
        segs[k].pos0 = sl.reuse;
        segs[k].off = sl.pf_off;
        segs[k].n = chunkLen(sl);
        if (opt_.use_mtp) segs[k].prev_hidden = sl.ck_hid;
    }
    try {
        m_.prefillSegs(segs, opt_.use_mtp);
    } catch (const std::exception& ex) {
        for (Slot* sl : grp) failSlot(*sl, ex.what());
        return 0;
    }
    stat_prefill_tok_ += total;
    stat_seg_batches_ += 1;
    stat_seg_chunks_ += grp.size();
    std::uint64_t r0 = 0;
    for (std::size_t k = 0; k < grp.size(); ++k) {
        Slot& sl = *grp[k];
        sl.pf_off += segs[k].n;
        r0 += segs[k].n;
        try {
            maybeSplitCkpt(sl, m_.hn() + (r0 - 1) * m_.cfg().n_embd * 4);
        } catch (const std::exception& ex) {
            failSlot(sl, ex.what());
            continue;
        }
        if (sl.pf_off != segs[k].tokens.size()) continue;
        try {
            m_.segLogitsToFront(static_cast<std::uint32_t>(k));
        } catch (const std::exception& ex) {
            failSlot(sl, ex.what());
            continue;
        }
        try {
            finishPrefill(sl);
        } catch (const std::exception& ex) {
            failSlot(sl, ex.what());
        }
    }
    return static_cast<std::size_t>(r0);
}

void Engine::finishPrefill(Slot& sl) {
    Job& job = *sl.job;
    const std::span<const std::uint32_t> toks = job.tokens;
    const std::uint32_t N = sl.n_prompt;
    const auto& cfg = m_.cfg();
    const std::uint32_t V = cfg.n_vocab;
    m_.selectSeq(sl.id);
    std::uint32_t first;
    if (sl.greedy) {
        first = m_.argmax();
    } else {
        fetchCands(sl.sm, 1);
        prepareRow(sl.sm, 0);
        sl.sm.pos_idx = 0;
        first = sampleRow(sl.sm, std::nullopt, 0);
    }
    if (opt_.verify_cache && sl.reuse > 0) {
        // diagnostic: compare the cached-prefix logits with a from-scratch prefill
        std::vector<float> cached(V);
        ops_.download(cached.data(), m_.logits(), static_cast<std::uint64_t>(V) * 4);
        {
            const bool busy = sl.spill_busy && !ops_.eventDone(sl.spill_ev);
            if (busy || sharedFrom(sl, 0)) {
                cowPages(sl, 0, busy);
                if (!ensurePages(sl, N)) throw KvPoolFull();
            }
        }
        m_.selectSeq(sl.id);
        m_.reset();
        const std::size_t v_sys = sysSplit(toks);
        const std::size_t v_th = ct_.thinkOpenSplit(toks);
        const std::vector<std::uint32_t> v_imgs = imgSplits(toks);
        std::size_t off = 0;
        while (off < toks.size()) {
            const std::size_t n = prefillChunkLen(schedNextV(toks.size(), v_sys, v_th, v_imgs, off) - off);
            if (opt_.use_mtp) m_.prefillMtpChunk(toks, 0, off, n, std::nullopt);
            else m_.forward(toks.subspan(off, n), static_cast<std::uint32_t>(off));
            off += n;
        }
        std::vector<float>& fresh = row_host_;
        ops_.download(fresh.data(), m_.logits(), static_cast<std::uint64_t>(V) * 4);
        float mc = -std::numeric_limits<float>::infinity(), mf = mc;
        std::size_t ac = 0, af = 0;
        float maxd = 0;
        for (std::size_t j = 0; j < V; ++j) {
            if (cached[j] > mc) {
                mc = cached[j];
                ac = j;
            }
            if (fresh[j] > mf) {
                mf = fresh[j];
                af = j;
            }
            maxd = std::max(maxd, std::fabs(cached[j] - fresh[j]));
        }
        double zc = 0, zf = 0;
        for (std::size_t j = 0; j < V; ++j) {
            zc += std::exp(static_cast<double>(cached[j] - mc));
            zf += std::exp(static_cast<double>(fresh[j] - mf));
        }
        double kl = 0;
        for (std::size_t j = 0; j < V; ++j) {
            const double pf = std::exp(static_cast<double>(fresh[j] - mf)) / zf;
            const double pc = std::exp(static_cast<double>(cached[j] - mc)) / zc;
            if (pf > 0) kl += pf * (std::log(pf) - std::log(std::max(pc, 1e-300)));
        }
        logI("req {} | slot {} | prefix-cache verify: cached vs from-scratch prefill logits: max abs diff {:.4f}, KL {:.2e}, "
             "top1 {} ({} vs {})",
             job.id, sl.id, maxd, kl, ac == af ? "same" : "DIFFERENT", ac, af);
        // continue from the from-scratch state (the prompt-end checkpoint below is saved from it)
        if (sl.greedy) {
            first = m_.argmax();
        } else {
            fetchCands(sl.sm, 1);
            prepareRow(sl.sm, 0);
            first = sampleRow(sl.sm, std::nullopt, 0);
        }
    }
    sl.prompt_ms = msSince(sl.tp0);
    const std::uint32_t n_new = N - sl.reuse;
    // (thinking-mode prompts keep the think-open checkpoint at N - 1 instead)
    if (sl.reuse < N && ct_.thinkOpenSplit(toks) == 0)
        saveCkpt(sl, N, "prompt-end", opt_.use_mtp ? std::optional<DevPtr>(m_.mtpH()) : std::nullopt, m_.logits());
    logI("req {} | slot {} | prefill: {} tok in {:.1f} ms ({:.1f} tok/s)", job.id, sl.id, n_new, sl.prompt_ms,
         sl.prompt_ms > 0 ? static_cast<double>(n_new) * 1000.0 / sl.prompt_ms : 0.0);

    sl.out = Out{};
    sl.out.chat = job.kind == JobKind::chat;
    sl.out.think_open = job.kind == JobKind::chat && job.think;
    sl.out.tools_on = job.tools_on;
    sl.out.stops = job.params.stop;
    sl.st = StreamState{};
    sl.st.job = &job;
    sl.st.e = this;
    sl.st.begin();
    sl.pos = N;
    sl.next = first;
    sl.pend[0] = first;
    sl.n_pend = 1;
    sl.pend_pos = N;
    sl.last_acc = 0;
    sl.last_row = 0;
    sl.logits_ok = true;
    sl.td0 = Clock::now();
    sl.last_log_t = sl.td0;
    sl.last_log_n = 0;
    sl.done = emit(sl, first);
    flushOut(sl);
    sl.phase = Phase::decode;
    if (sl.done || sl.st.gone) finishJob(sl);
}

std::uint32_t Engine::pickDrafts(std::span<Slot* const> act) {
    if (!opt_.use_mtp) return 0;
    const std::uint32_t A = static_cast<std::uint32_t>(act.size());
    std::uint32_t nd = std::min(opt_.batch_drafts[std::min(A, gdn_max_seg)], opt_.n_draft);
    if (!m_.fusedDecode()) nd = std::min(nd, 1u);
    while (nd > 0 && (A * (nd + 1) > vrows_ || (!m_.gdnReplay() && A * nd > qwen35::gdn_max_snap))) nd -= 1;
    if (nd == 0) return 0;
    if (opt_.mtp_auto) {
        std::vector<const qwen35::DraftAccept*> accs;
        for (Slot* sl : act) accs.push_back(&sl->dacc);
        const std::uint32_t pick = qwen35::pickDrafts(accs, timing_[std::min(A, gdn_max_seg)], nd, n_cycles_, prev_nd_);
        prev_nd_ = pick;
        return pick;
    }
    if (opt_.mtp_adapt > 0) {
        std::uint32_t want = 1;
        for (Slot* sl : act)
            want = std::max(want, static_cast<std::uint32_t>(std::ceil(sl->acc_ema + static_cast<float>(opt_.mtp_adapt))));
        nd = std::min(nd, want);
    }
    return nd;
}

void Engine::decodeCycle(std::span<Slot* const> act_in) {
    const auto& cfg = m_.cfg();
    const std::uint64_t E = cfg.n_embd;
    const TimePoint t_cycle0 = Clock::now();
    // slots out of context room end here
    std::vector<Slot*> act;
    const std::uint32_t nd_room = opt_.use_mtp ? opt_.n_draft : 0;
    for (Slot* sl : act_in) {
        const bool room_ok = sl->pos + nd_room + 1 < ctx_ && ensurePages(*sl, sl->pos + nd_room + 2);
        if (!room_ok) {
            if (sl->pos + nd_room + 1 < ctx_)
                logW("req {} | slot {} | kv pool exhausted at {} tokens (every page belongs to a running request)",
                     sl->reqId(), sl->id, sl->pos);
            sl->finish = "length";
            sl->logits_ok = false;  // other work ran since its last cycle
            finishJob(*sl);
            continue;
        }
        act.push_back(sl);
    }
    const std::size_t A = act.size();
    if (A == 0) return;
    // n-gram drafts per slot (history = cache_tokens + next) when they promise
    // more tokens per ms than MTP drafts
    std::array<std::uint32_t, gdn_max_seg> ng_n{};
    std::array<std::array<std::uint32_t, qwen35::max_ng_drafts>, gdn_max_seg> ng_d{};
    std::size_t n_ng = 0;
    // replay keeps no snapshot sets: the verify row budget is the only shared limit
    const std::uint32_t snap_cap = m_.gdnReplay() ? vrows_ : std::min(qwen35::gdn_max_snap, m_.snapSets());
    if (opt_.ngram && opt_.use_mtp) {
        const std::uint32_t Au = static_cast<std::uint32_t>(A);
        std::uint32_t lim = std::min({qwen35::max_ng_drafts, snap_cap / Au, vrows_ / Au > 0 ? vrows_ / Au - 1 : 0u});
        if (opt_.ngram_max > 0) lim = std::min(lim, opt_.ngram_max);
        if (!m_.fusedDecode()) lim = std::min(lim, 1u);
        const qwen35::DraftTiming& tm = timing_[std::min<std::size_t>(A, gdn_max_seg)];
        const std::uint32_t nd0 = std::max(1u, std::min(prev_nd_ > 0 ? prev_nd_ : 3u, draftCap(Au)));
        for (std::size_t k = 0; k < A; ++k) {
            if (lim == 0) break;
            Slot& sl = *act[k];
            sl.cache_tokens.push_back(sl.next);
            struct Pop {
                std::vector<std::uint32_t>& v;
                ~Pop() { v.pop_back(); }
            } pop{sl.cache_tokens};
            // the verify writes KV rows pos .. pos + drafts: stay inside the
            // pages mapped for this slot and the context
            const std::uint64_t mapped = static_cast<std::uint64_t>(sl.pages.size()) * kv_page;
            const std::uint64_t a1 = mapped > static_cast<std::uint64_t>(sl.pos) + 1 ? mapped - (sl.pos + 1ull) : 0;
            const std::uint64_t a2 = ctx_ > sl.pos + 2ull ? ctx_ - (sl.pos + 2ull) : 0;
            const std::uint32_t lim_k = static_cast<std::uint32_t>(std::min({static_cast<std::uint64_t>(lim), a1, a2}));
            if (lim_k == 0) continue;
            const auto mt = sl.ng.propose(sl.cache_tokens, opt_.ngram_min, std::span<std::uint32_t>(ng_d[k].data(), lim_k));
            if (mt.n == 0) continue;
            float mtp_score = 0;
            if (auto t = tm.estimate(nd0)) mtp_score = sl.dacc.expected(nd0) / *t;
            ng_n[k] = opt_.ngram_force ? static_cast<std::uint32_t>(mt.n) : sl.ng.choose(mt.n, mt.mlen, mtp_score, tm);
            if (ng_n[k] > 0) n_ng += 1;
        }
    }
    // MTP drafts for the other slots
    std::vector<Slot*> act_mtp;
    for (std::size_t k = 0; k < A; ++k)
        if (ng_n[k] == 0) act_mtp.push_back(act[k]);
    const std::size_t n_mtp = act_mtp.size();
    std::uint32_t nd = n_mtp > 0 ? pickDrafts(act_mtp) : 0;
    {
        // keep the verify within vrows_ rows (and the snapshot sets without replay)
        std::uint32_t ng_tot = 0;
        for (std::size_t k = 0; k < A; ++k) ng_tot += ng_n[k];
        while (nd > 0 && (static_cast<std::uint32_t>(A) + ng_tot + static_cast<std::uint32_t>(n_mtp) * nd > vrows_ ||
                          ng_tot + static_cast<std::uint32_t>(n_mtp) * nd > snap_cap))
            nd -= 1;
    }
    // per-slot counts (MTP slots, in act_mtp order), MTP steps = the largest
    std::array<std::uint32_t, gdn_max_seg> nd_m{};
    std::uint32_t nd_max = nd;
    if (n_mtp > 0) {
        std::uint32_t ng_tot = 0;
        for (std::size_t k = 0; k < A; ++k) ng_tot += ng_n[k];
        const std::uint32_t used = static_cast<std::uint32_t>(A) + ng_tot;
        const std::uint32_t rows_free = vrows_ > used ? vrows_ - used : 0;
        const std::uint32_t snap_free = snap_cap > ng_tot ? snap_cap - ng_tot : 0;
        splitDrafts(act_mtp, nd, std::min(rows_free, snap_free), std::span<std::uint32_t>(nd_m.data(), n_mtp));
        nd_max = 0;
        for (std::size_t i = 0; i < n_mtp; ++i) nd_max = std::max(nd_max, nd_m[i]);
    }
    const TimePoint tc0 = Clock::now();
    const std::size_t pa = std::min<std::size_t>(A, gdn_max_seg);
    CycleProf* p = prof_.get();
    if (p && p->events) m_.profileStart(0, pa);
    std::array<MSeg, gdn_max_seg> msegs{};
    std::array<VSeg, gdn_max_seg> vsegs{};
    std::array<std::uint32_t, gdn_max_seg> row_of{}, snap_of{}, nd_of{};
    // MTP steps r >= 1 run on the MTP slots with more than r drafts: MTP slots
    // sorted by draft count (descending) at the front of the batch
    std::array<std::size_t, qwen35::max_drafts + 1> n_step{};
    {
        std::array<std::size_t, gdn_max_seg> ord{};
        for (std::size_t i = 0; i < n_mtp; ++i) ord[i] = i;
        for (std::size_t i = 1; i < n_mtp; ++i) {
            std::size_t j = i;
            while (j > 0 && nd_m[ord[j - 1]] < nd_m[ord[j]]) {
                std::swap(ord[j - 1], ord[j]);
                --j;
            }
        }
        for (std::size_t q = 0; q < n_mtp; ++q) {
            Slot& s = *act_mtp[ord[q]];
            msegs[q] = MSeg{s.id, std::span<const std::uint32_t>(s.pend.data(), s.n_pend), s.pend_pos, s.pos};
        }
        for (std::size_t r = 0; r <= qwen35::max_drafts; ++r) {
            std::size_t c = 0;
            for (std::size_t i = 0; i < n_mtp; ++i)
                if (nd_m[i] > r) ++c;
            n_step[r] = c;
        }
    }
    {
        // n-gram slots after the MTP slots in the MTP batch (their pend rows only)
        std::size_t ig = n_mtp, im = 0;
        std::uint32_t row = 0, snap = 0;
        for (std::size_t k = 0; k < A; ++k) {
            Slot& sl = *act[k];
            const MSeg ms{sl.id, std::span<const std::uint32_t>(sl.pend.data(), sl.n_pend), sl.pend_pos, sl.pos};
            if (ng_n[k] == 0) {
                nd_of[k] = nd_m[im];
                ++im;
            } else {
                msegs[ig] = ms;
                ++ig;
                nd_of[k] = ng_n[k];
            }
            vsegs[k] = VSeg{sl.id, sl.next, nd_of[k], sl.pos,
                            ng_n[k] > 0 ? std::span<const std::uint32_t>(ng_d[k].data(), ng_n[k])
                                        : std::span<const std::uint32_t>()};
            row_of[k] = row;
            snap_of[k] = snap;
            row += nd_of[k] + 1;
            snap += nd_of[k];
        }
    }
    if (nd_max > 0) {
        m_.setDraftCutoff(opt_.p_min, std::min(opt_.n_min, nd_max));
        struct Reset {
            ServerModel& m;
            ~Reset() { m.setDraftCutoff(0, 0); }
        } reset{m_};
        m_.mtpBatchStepEx(std::span<const MSeg>(msegs.data(), A), 0, true);
        for (std::uint32_t r = 1; r < nd_max; ++r)
            m_.mtpBatchStepEx(std::span<const MSeg>(msegs.data(), n_step[r]), r, true);
    } else if (opt_.use_mtp) {
        // no MTP drafts this cycle: still feed the pend rows to the MTP block so
        // its KV cache stays complete for later drafting cycles
        m_.mtpBatchStepEx(std::span<const MSeg>(msegs.data(), A), 0, false);
    }
    if (p && p->events) m_.profileStart(1, pa);
    const std::uint32_t rows = m_.verifyBatchEnqueue(std::span<const VSeg>(vsegs.data(), A));
    if (p) m_.profileStop();
    bool any_sampling = false;
    for (std::size_t k = 0; k < A; ++k) {
        Slot& sl = *act[k];
        if (sl.greedy) continue;
        any_sampling = true;
        launchTopk(row_of[k], nd_of[k] + 1, k_small, sl.sm.inv_t, samp_dev_, max_rows, row_of[k]);
    }
    const TimePoint t_enq = Clock::now();
    m_.readCtl(std::span<std::int32_t>(ctl_host_.data(), slots_.size() * qwen35::ctl_words));
    if (p) {
        p->t_ready = Clock::now();
        p->enq_ms[pa] += std::chrono::duration<double, std::milli>(t_enq - tc0).count();
    }
    if (any_sampling) ops_.download(samp_host_.data(), samp_dev_, sampBytes());
    if (nd > 0 && n_ng == 0) {
        // swaps keep the rows of the common count (an extra step at most)
        std::uint32_t tot = 0;
        for (std::size_t i = 0; i < n_mtp; ++i) tot += nd_m[i];
        if (tot == nd * static_cast<std::uint32_t>(n_mtp))
            timing_[std::min<std::size_t>(A, gdn_max_seg)].update(nd, static_cast<float>(msSince(tc0)));
    }
    if (n_ng == A) {
        // n-gram cycle times (every slot drafted from its history): per slot
        std::uint32_t ng_max_n = 0;
        for (std::size_t k = 0; k < A; ++k) ng_max_n = std::max(ng_max_n, ng_n[k]);
        const float ms = static_cast<float>(msSince(tc0));
        for (Slot* sl : act) sl->ng.timing.update(ng_max_n, ms);
    }
    if (p) {
        if (p->events) m_.profileFlush(pa);
        p->cycles[pa] += 1;
        p->rows[pa] += rows;
        p->drafts[pa] += rows - A;
        p->wall_ms[pa] += msSince(t_cycle0);
        p->host_ms[pa] += msSince(t_cycle0) - msSince(tc0);
        if (p->last_end) p->gap_ms[pa] += std::chrono::duration<double, std::milli>(t_cycle0 - *p->last_end).count();
    }
    if (opt_.trace_nd) logI("trace | cycle {} | slots {} | nd {} | {:.2f} ms", n_cycles_, A, nd, msSince(tc0));
    n_cycles_ += 1;
    stat_cycles_ += 1;
    stat_rows_ += rows;
    stat_slots_ += A;
    stat_drafts_ += nd;

    for (std::size_t k = 0; k < A; ++k) {
        Slot& sl = *act[k];
        const std::int32_t* ctl = ctl_host_.data() + static_cast<std::size_t>(sl.id) * qwen35::ctl_words;
        const std::uint32_t row0 = row_of[k];
        const std::uint32_t nd_k = nd_of[k];
        const bool is_ng = ng_n[k] > 0;
        // MTP: drafts (and the device p-min cutoff) from the control area; n-gram: literal
        const std::uint32_t nd_dev =
            is_ng ? nd_k : (nd_k > 0 ? std::min(nd_k, static_cast<std::uint32_t>(std::max(0, ctl[qwen35::ctl_nd]))) : 0);
        std::array<std::uint32_t, qwen35::max_ng_drafts> dr{};
        for (std::uint32_t r = 0; r < nd_k; ++r)
            dr[r] = is_ng ? ng_d[k][r] : static_cast<std::uint32_t>(ctl[qwen35::ctl_drafts + r]);
        if (is_ng) sl.ng_cycles += 1;
        sl.cycles += 1;
        sl.drafted += nd_dev;
        std::uint32_t acc = 0;
        std::uint32_t chosen;
        if (sl.greedy) {
            while (acc < nd_dev && static_cast<std::uint32_t>(ctl[qwen35::ctl_rows + acc]) == dr[acc]) ++acc;
            chosen = static_cast<std::uint32_t>(ctl[qwen35::ctl_rows + acc]);
        } else {
            // speculative sampling with a deterministic draft: the draft is
            // accepted iff it is the sampled token
            sl.sm.row_base = row0;
            for (;;) {
                prepareRow(sl.sm, acc);
                sl.sm.pos_idx = sl.n_gen + acc;
                const std::uint32_t t = sampleRow(sl.sm, std::nullopt, 0);
                if (acc < nd_dev && t == dr[acc]) {
                    ++acc;
                    continue;
                }
                chosen = t;
                break;
            }
        }
        sl.accepted += acc;
        if (!is_ng) {
            // n-gram cycles stay out of the MTP acceptance models
            sl.acc_ema = 0.7f * sl.acc_ema + 0.3f * static_cast<float>(acc);
            sl.dacc.update(nd_dev, acc);
        }
        // an accepted end-of-turn draft ends the reply: the trunk keeps only the
        // rows before it and the stop token becomes `chosen`
        if (!sl.ignore_eos) {
            for (std::uint32_t r = 0; r < acc; ++r) {
                if (dr[r] == ct_.im_end || dr[r] == ct_.endoftext) {
                    chosen = dr[r];
                    acc = r;
                    break;
                }
            }
        }
        // same for max_tokens: the last emitted token is not fed to the trunk
        {
            const std::uint32_t budget = sl.max_tokens > sl.n_gen ? sl.max_tokens - sl.n_gen : 0;
            if (budget >= 1 && acc + 1 > budget) {
                chosen = dr[budget - 1];
                acc = budget - 1;
            }
        }
        // the verify ran nd + 1 rows of this slot: keep acc + 1 of them
        if (m_.gdnReplay()) m_.setPending(sl.id, acc + 1);
        else if (acc < nd_k) m_.restoreSeqSnapshot(sl.id, snap_of[k] + acc);
        // trunk now holds next, dr[0..acc) at pos .. pos + acc
        sl.cache_tokens.push_back(sl.next);
        sl.cache_tokens.insert(sl.cache_tokens.end(), dr.begin(), dr.begin() + acc);
        const std::uint32_t n_before = sl.n_gen;
        for (std::uint32_t r = 0; r < acc; ++r)
            if (!sl.done) sl.done = emit(sl, dr[r]);
        if (!sl.done) sl.done = emit(sl, chosen);
        stat_tokens_ += sl.n_gen - n_before;
        if (opt_.use_mtp)
            ops_.copyAsync(m_.seqHid(sl.id), m_.hn() + static_cast<std::uint64_t>(row0) * E * 4, (acc + 1) * E * 4,
                           m_.stream());
        for (std::uint32_t r = 0; r < acc; ++r) sl.pend[r] = dr[r];
        sl.pend[acc] = chosen;
        sl.n_pend = acc + 1;
        sl.pend_pos = sl.pos + 1;
        sl.next = chosen;
        sl.pos += acc + 1;
        sl.last_acc = acc;
        sl.last_row = row0 + acc;
        sl.logits_ok = true;
        flushOut(sl);
        if (sl.n_gen >= sl.last_log_n + 100 || msSince(sl.last_log_t) >= 5000) {
            const double el = msSince(sl.td0);
            if (opt_.use_mtp) {
                logI("req {} | slot {} | gen: n_gen {}, tg {:.2f} t/s, draft acceptance {:.1f}% ({:.2f} tok/cycle)",
                     sl.reqId(), sl.id, sl.n_gen, static_cast<double>(sl.n_gen) * 1000.0 / el,
                     sl.drafted > 0 ? 100.0 * sl.accepted / sl.drafted : 0.0,
                     static_cast<double>(sl.accepted + sl.cycles) / sl.cycles);
            } else {
                logI("req {} | slot {} | gen: n_gen {}, tg {:.2f} t/s", sl.reqId(), sl.id, sl.n_gen,
                     static_cast<double>(sl.n_gen) * 1000.0 / el);
            }
            sl.last_log_n = sl.n_gen;
            sl.last_log_t = Clock::now();
        }
    }
    if (p) {
        std::uint64_t tk = 0;
        for (Slot* sl : act) tk += sl->last_acc + 1;
        p->tokens[pa] += tk;
    }
    // finish after every slot of the cycle is processed (their logits rows are
    // still intact for the gen-end checkpoints)
    bool any_done = false;
    for (Slot* sl : act) {
        if (sl->done || sl->st.gone) {
            any_done = true;
            finishJob(*sl);
        }
    }
    if (p) {
        if (p->t_ready) p->post_ms[pa] += msSince(*p->t_ready);
        if (any_done) p->last_end.reset();
        else p->last_end = Clock::now();
    }
}

void Engine::splitDrafts(std::span<Slot* const> act, std::uint32_t nd, std::uint32_t rows, std::span<std::uint32_t> nd_m) {
    for (auto& x : nd_m) x = nd;
    if (!opt_.slot_drafts || !opt_.mtp_auto || act.size() < 2 || nd == 0) return;
    const std::uint32_t cap = std::min(opt_.n_draft, qwen35::max_drafts);
    std::uint32_t used = nd * static_cast<std::uint32_t>(act.size());
    float e_tot = 0;
    for (Slot* sl : act) e_tot += sl->dacc.expected(nd);
    const float step_frac = 0.03f;
    for (std::uint32_t it = 0; it < 16; ++it) {
        std::uint32_t mx = 0;
        for (auto x : nd_m) mx = std::max(mx, x);
        // best receiver: largest marginal of its next draft
        std::optional<std::size_t> r_best;
        float g_best = 0;
        for (std::size_t k = 0; k < act.size(); ++k) {
            if (nd_m[k] >= cap) continue;
            const float g = act[k]->dacc.expected(nd_m[k] + 1) - act[k]->dacc.expected(nd_m[k]);
            if (!r_best || g > g_best) {
                r_best = k;
                g_best = g;
            }
        }
        if (!r_best) break;
        const std::size_t r = *r_best;
        const bool needs_step = nd_m[r] + 1 > mx;
        const float bar = needs_step ? step_frac * e_tot : 0;
        if (used < rows) {
            // a spare row: worth it when the draft is likely enough
            if (g_best > 0.35f && g_best > bar) {
                nd_m[r] += 1;
                used += 1;
                e_tot += g_best;
                continue;
            }
        }
        // best donor: smallest marginal of its last draft (not the receiver)
        std::optional<std::size_t> d_best;
        float l_best = 0;
        for (std::size_t k = 0; k < act.size(); ++k) {
            if (k == r || nd_m[k] == 0) continue;
            const float l = act[k]->dacc.expected(nd_m[k]) - act[k]->dacc.expected(nd_m[k] - 1);
            if (!d_best || l < l_best) {
                d_best = k;
                l_best = l;
            }
        }
        if (!d_best) break;
        const std::size_t d = *d_best;
        if (g_best - l_best <= bar || g_best - l_best < 0.02f) break;
        nd_m[r] += 1;
        nd_m[d] -= 1;
        e_tot += g_best - l_best;
    }
}

std::uint32_t Engine::draftCap(std::uint32_t A) {
    std::uint32_t nd = std::min(opt_.batch_drafts[std::min(A, gdn_max_seg)], opt_.n_draft);
    if (!m_.fusedDecode()) nd = std::min(nd, 1u);
    while (nd > 0 && (A * (nd + 1) > vrows_ || (!m_.gdnReplay() && A * nd > qwen35::gdn_max_snap))) nd -= 1;
    return nd;
}

void Engine::finishJob(Slot& sl) {
    Job& job = *sl.job;
    const std::uint64_t E = m_.cfg().n_embd;
    const std::uint32_t N = sl.n_prompt;
    m_.selectSeq(sl.id);
    ops_.streamSync(m_.stream());
    const double gen_ms = msSince(sl.td0);
    if (sl.stopped_str) sl.finish = "stop";
    // ---- checkpoint after the last token the trunk has seen
    if (sl.pos > N && sl.pos == sl.cache_tokens.size() && sl.logits_ok) {
        const std::optional<DevPtr> last_hid =
            opt_.use_mtp ? std::optional<DevPtr>(m_.mtpH() + static_cast<std::uint64_t>(sl.last_acc) * E * 4) : std::nullopt;
        saveCkpt(sl, sl.pos, "gen-end", last_hid, logitsRow(sl.last_row));
        // MTP KV rows for the accepted drafts of the last cycle (normally
        // written at the start of the next cycle)
        if (opt_.use_mtp && sl.n_pend > 1)
            m_.mtpEnqueue(m_.mtpH(), std::span<const std::uint32_t>(sl.pend.data(), sl.n_pend - 1), sl.pend_pos);
    } else if (sl.pos == sl.cache_tokens.size()) {
        // no gen-end checkpoint; the MTP KV rows of the last cycle are still needed
        if (opt_.use_mtp && sl.n_pend > 1)
            m_.mtpEnqueue(m_.mtpH(), std::span<const std::uint32_t>(sl.pend.data(), sl.n_pend - 1), sl.pend_pos);
    } else {
        logW("req {} | slot {} | cache bookkeeping mismatch (pos {}, tokens {}); dropping prefix cache", job.id, sl.id, sl.pos,
             sl.cache_tokens.size());
        dropCache(sl);
    }
    ops_.streamSync(m_.stream());

    // ---- final output
    const Final fin = sl.st.finish(sl.out, sl.finish);
    const double tot_ms = msSince(sl.tp0);
    const std::uint32_t n_new = N - sl.reuse;
    Timings tim;
    tim.cache_n = sl.reuse;
    tim.prompt_n = n_new;
    tim.prompt_ms = sl.prompt_ms;
    tim.predicted_n = sl.n_gen;
    tim.predicted_ms = gen_ms;
    tim.draft_n = sl.drafted;
    tim.draft_n_accepted = sl.accepted;
    sl.st.end(sl.out, fin, tim, N);

    // ---- summary (llama.cpp style)
    const double pn = n_new, gn = sl.n_gen;
    logI("req {} | slot {} | prompt eval time = {:>10.2f} ms / {:>5} tokens ({:>8.2f} ms per token, {:>8.2f} tokens per second)",
         job.id, sl.id, sl.prompt_ms, n_new, n_new > 0 ? sl.prompt_ms / pn : 0.0,
         sl.prompt_ms > 0 ? pn * 1000.0 / sl.prompt_ms : 0.0);
    logI("req {} | slot {} |        eval time = {:>10.2f} ms / {:>5} tokens ({:>8.2f} ms per token, {:>8.2f} tokens per second)",
         job.id, sl.id, gen_ms, sl.n_gen, sl.n_gen > 0 ? gen_ms / gn : 0.0, gen_ms > 0 ? gn * 1000.0 / gen_ms : 0.0);
    logI("req {} | slot {} |       total time = {:>10.2f} ms / {:>5} tokens", job.id, sl.id, tot_ms, n_new + sl.n_gen);
    if (opt_.use_mtp) {
        logI("req {} | slot {} | MTP: drafted {}, accepted {}, acceptance rate {:.1f}%, {:.2f} tokens per cycle ({} verify "
             "cycles)",
             job.id, sl.id, sl.drafted, sl.accepted, sl.drafted > 0 ? 100.0 * sl.accepted / sl.drafted : 0.0,
             sl.cycles > 0 ? static_cast<double>(sl.accepted + sl.cycles) / sl.cycles : 0.0, sl.cycles);
        if (opt_.ngram) logI("req {} | slot {} | n-gram drafts in {} of {} cycles", job.id, sl.id, sl.ng_cycles, sl.cycles);
    }
    if (!sl.greedy && (sl.sm.big_used > 0 || sl.sm.full_used > 0))
        logI("req {} | slot {} | sampler fallbacks: top-{} {}x, full row {}x", job.id, sl.id, k_big, sl.sm.big_used,
             sl.sm.full_used);
    logI("req {} | slot {} | finish_reason {}{}, completion {} tok, reasoning {} B, content {} B, tool_calls {}, cache now {} "
         "tok",
         job.id, sl.id, fin.reason, sl.st.gone ? " (client disconnected)" : "", sl.n_gen, fin.reason_len, fin.content_len,
         fin.n_tools, sl.cache_tokens.size());
    releaseSlot(sl);
    // item G: copy the idle slot's new content to the host tier (async)
    try {
        spillSlot(sl);
    } catch (const std::exception& ex) {
        logW("slot {} | kv tier: spill failed: {} ({})", sl.id, ex.what(), ops_.lastErrorString());
    }
}

// ---------------------------------------------------------------------------
// item G: host tiers

void Engine::waitSpill(Slot& sl) {
    // (also after an eviction cleared spill_busy: the spill may still read the
    // checkpoint buffers; item S)
    if (sl.spill_ev == nullptr) return;
    sl.spill_busy = false;
    if (ops_.eventDone(sl.spill_ev)) return;
    stat_spill_waits_ += 1;
    ops_.streamWaitEvent(m_.stream(), sl.spill_ev);
}

void Engine::ckSpans(const Ckpt& c, std::uint64_t base, std::uint8_t tag) {
    const auto& cfg = m_.cfg();
    std::uint64_t off = base;
    for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
        if (c.conv[i] == 0) continue;
        tier_spans_.push_back({c.conv[i], off, conv_bytes_, tag});
        off += conv_bytes_;
        tier_spans_.push_back({c.ssm[i], off, ssm_bytes_, tag});
        off += ssm_bytes_;
    }
    const std::uint64_t hb = static_cast<std::uint64_t>(cfg.n_embd) * 4;
    tier_spans_.push_back({c.hid, off, hb, tag});
    off += hb;
    tier_spans_.push_back({c.logits, off, static_cast<std::uint64_t>(cfg.n_vocab) * 4, tag});
}

void Engine::pageSpans(std::int32_t phys, std::uint64_t base) {
    std::uint64_t off = base;
    for (const KvArr& a : kv_arrays_) {
        const std::uint64_t len = static_cast<std::uint64_t>(kv_page) * a.row;
        tier_spans_.push_back({a.base + static_cast<std::uint64_t>(phys) * len, off, len, kNoTag});
        off += len;
    }
}

void Engine::spillSlot(Slot& sl) {
    tier::Tier* t = tier_;
    if (!t) return;
    if (sl.phase != Phase::idle || sl.pages.empty()) return;
    std::uint32_t P = 0;
    for (const Ckpt& c : sl.ckpts)
        if (c.valid && c.pos > P) P = c.pos;
    if (P < t->cfg.min_tokens || P > sl.cache_tokens.size()) return;
    const std::uint32_t npages = (P + kv_page - 1) / kv_page;
    if (npages > sl.pages.size()) return;
    tier::Entry* x = nullptr;
    std::uint32_t vp = 0;
    bool fresh = true;
    if (sl.tier_id != 0) {
        if (tier::Entry* old = t->find(sl.tier_id)) {
            if (old->in_ram && old->owner && *old->owner == sl.id && old->pin == 0 && !old->loading &&
                old->nck == sl.ckpts.size()) {
                if (old->io_busy) {
                    // an SSD write is reading it: retry when that is done
                    sl.spill_wanted = true;
                    return;
                }
                x = old;
                fresh = false;
                vp = std::min(sl.tier_pages, old->n_tok / kv_page);
            }
        }
    }
    if (fresh) x = t->newEntry(static_cast<std::uint32_t>(sl.ckpts.size()));
    const tier::Layout lay = t->lay;
    const std::uint64_t stride = lay.ckStride();
    const std::uint64_t kv_off = x->kvOff(lay);
    // RAM blocks: changed checkpoints, pages vp ..
    bool ok = true;
    for (std::size_t j = 0; j < sl.ckpts.size(); ++j) {
        const Ckpt& c = sl.ckpts[j];
        if (!c.valid || c.pos > P) continue;
        const auto r = tier::Tier::lbRange(j * stride, lay.ck_bytes);
        if (!t->ensureBlocks(x, r[0], r[1])) ok = false;
    }
    if (ok && npages > vp) {
        const auto r = tier::Tier::lbRange(kv_off + static_cast<std::uint64_t>(vp) * lay.page_bytes,
                                           static_cast<std::uint64_t>(npages - vp) * lay.page_bytes);
        if (!t->ensureBlocks(x, r[0], r[1])) ok = false;
    }
    if (!ok) {
        logW("slot {} | kv tier: no room for {} tokens in the RAM tier ({} MiB); not kept", sl.id, P, t->cfg.ram_bytes >> 20);
        if (fresh) t->removeEntry(x);
        else t->trim(x);
        if (!fresh) sl.tier_id = x->id;
        return;
    }
    // spans: older checkpoints first (the next request overwrites them first), then pages
    tier_spans_.clear();
    const std::size_t nck = std::min(sl.ckpts.size(), tier::max_ck);
    std::array<std::size_t, tier::max_ck> order{};
    for (std::size_t j = 0; j < nck; ++j) order[j] = j;
    std::sort(order.begin(), order.begin() + nck,
              [&](std::size_t a, std::size_t b) { return sl.ckpts[a].seq < sl.ckpts[b].seq; });
    std::uint64_t ck_bytes = 0;
    for (std::size_t oi = 0; oi < nck; ++oi) {
        const std::size_t j = order[oi];
        const Ckpt& c = sl.ckpts[j];
        if (!c.valid || c.pos > P) {
            if (x->ck[j].valid != 0) t->dropCkBlocks(x, static_cast<std::uint32_t>(j));
            x->ck[j] = tier::CkMeta{};
            continue;
        }
        const bool same = x->ck[j].valid != 0 && x->ck[j].pos == c.pos && x->ck[j].seq == c.seq;
        if (!same) {
            ckSpans(c, j * stride);
            ck_bytes += lay.ck_bytes;
        }
        x->ck[j] = tier::CkMeta{c.pos, 1, static_cast<std::uint8_t>(c.has_logits ? 1 : 0), ckKindCode(c.kind), 0, c.seq};
    }
    for (std::uint32_t pg = vp; pg < npages; ++pg) pageSpans(sl.pages[pg], kv_off + static_cast<std::uint64_t>(pg) * lay.page_bytes);
    const std::uint64_t bytes = t->copyOut(x, tier_spans_);
    x->tokens.assign(sl.cache_tokens.begin(), sl.cache_tokens.begin() + P);
    x->n_tok = P;
    x->pages = npages;
    t->trim(x);
    x->t_mod = Clock::now();
    x->owner = sl.id;
    if (sl.spill_ev == nullptr) sl.spill_ev = ops_.eventCreate();
    t->fence(x, sl.spill_ev);
    if (opt_.tier_verify) {
        try {
            ops_.eventSync(sl.spill_ev);
        } catch (const std::exception&) {
        }
        try {
            const auto v = t->verifySpans(x, tier_spans_);
            if (v.bad > 0) logE("slot {} | kv tier verify: spill of {} tok: {} pieces differ", sl.id, P, v.bad);
            else
                logI("slot {} | kv tier verify: spill of {} tok byte-identical ({:.1f} MiB)", sl.id, P,
                     static_cast<double>(v.bytes) / 1048576.0);
        } catch (const std::exception&) {
        }
    }
    sl.spill_busy = true;
    sl.spill_upto = P;
    sl.spill_wanted = false;
    sl.tier_id = x->id;
    sl.tier_pages = P / kv_page;
    t->stats.spills += 1;
    t->stats.spill_bytes += bytes;
    logI("slot {} | kv tier: spill {} tok to RAM ({}, pages {}..{}, checkpoints {:.0f} MiB, {:.1f} MiB in all; RAM tier {:.2f} "
         "/ {:.2f} GiB)",
         sl.id, P, fresh ? "new entry" : "in place", vp, npages, static_cast<double>(ck_bytes) / 1048576.0,
         static_cast<double>(bytes) / 1048576.0, static_cast<double>(t->ramUsedBytes()) / 1073741824.0,
         static_cast<double>(t->cfg.ram_bytes) / 1073741824.0);
}

bool Engine::tryRestore(Slot& sl, Job& job) {
    tier::Tier* t = tier_;
    if (!t) return false;
    if (!job.params.cache_prompt || opt_.no_prefix_cache) return false;
    const std::span<const std::uint32_t> toks = job.tokens;
    const std::uint32_t nd_max = opt_.use_mtp ? opt_.n_draft : 0;
    if (toks.size() + nd_max + 2 >= ctx_ || toks.size() + nd_max + 2 >= static_cast<std::uint64_t>(pool_pages_) * kv_page)
        return false;
    const auto mt = t->lookup(toks);
    if (!mt) return false;
    // item S: a shared prefix entry off this prompt's schedule would not give
    // the cold result; one already in VRAM needs no restore
    if (mt->e->ck[mt->ck].kind >= 4 && !onSchedule(toks, mt->reuse)) return false;
    std::uint32_t vr = cacheMatch(sl, toks).reuse;
    if (Spe* s = speMatch(toks)) vr = std::max(vr, s->n);
    if (mt->reuse <= vr) return false;
    const std::uint32_t gain = mt->reuse - vr;
    if (gain < opt_.tier_min_gain || (opt_.tier_min_gain >= 512 && gain < mt->reuse / 40)) return false;
    try {
        return beginRestore(sl, job, *mt);
    } catch (const std::exception& ex) {
        logW("req {} | slot {} | kv tier: restore failed to start: {} ({})", job.id, sl.id, ex.what(), ops_.lastErrorString());
        return false;
    }
}

bool Engine::beginRestore(Slot& sl, Job& job, const tier::Match& mt) {
    tier::Tier* t = tier_;
    tier::Entry* x = mt.e;
    const tier::Layout lay = t->lay;
    const TimePoint t0 = Clock::now();
    // the slot's own VRAM content goes (it is in the tier if it was worth keeping)
    freePages(sl);
    sl.job = &job;
    sl.phase = Phase::restore;
    if (!ensurePages(sl, x->n_tok)) {
        logW("req {} | slot {} | kv tier: not restoring {} tok: the kv pool has no room ({} of {} pages for this slot, {} "
             "free, {} quarantined batches)",
             job.id, sl.id, x->n_tok, sl.pages.size(), (x->n_tok + kv_page - 1) / kv_page, free_pages_.size(),
             quarantine_.size());
        sl.phase = Phase::prefill;
        return false;
    }
    // checkpoints: the matched one first, then the others by position, up to the slot's count
    std::array<bool, tier::max_ck> chosen{};
    sl.rs_map.fill(-1);
    chosen[mt.ck] = true;
    for (std::size_t n_sel = 1; n_sel < sl.ckpts.size(); ++n_sel) {
        std::optional<std::size_t> best;
        for (std::size_t k = 0; k < x->nck; ++k) {
            if (chosen[k] || x->ck[k].valid == 0) continue;
            if (!best || x->ck[k].pos > x->ck[*best].pos) best = k;
        }
        if (!best) break;
        chosen[*best] = true;
    }
    std::int32_t nj = 0;
    for (std::size_t k = 0; k < x->nck; ++k) {
        if (!chosen[k]) continue;
        sl.rs_map[k] = nj;
        ++nj;
    }
    // item T3: the request's part first (the matched checkpoint and the KV
    // pages); the slot's other checkpoints follow on the tier stream (tail)
    tier_spans_.clear();
    ckSpans(sl.ckpts[static_cast<std::size_t>(sl.rs_map[mt.ck])], static_cast<std::uint64_t>(mt.ck) * lay.ckStride());
    const std::uint64_t kv_off = x->kvOff(lay);
    for (std::uint32_t pg = 0; pg < x->pages; ++pg)
        pageSpans(sl.pages[pg], kv_off + static_cast<std::uint64_t>(pg) * lay.page_bytes);
    const std::size_t n_main = tier_spans_.size();
    sl.rs_tail.fill(false);
    for (std::size_t k = 0; k < x->nck; ++k) {
        if (!chosen[k] || k == mt.ck) continue;
        ckSpans(sl.ckpts[static_cast<std::size_t>(sl.rs_map[k])], static_cast<std::uint64_t>(k) * lay.ckStride(),
                static_cast<std::uint8_t>(k));
        sl.rs_tail[k] = true;
    }
    tier::Restore* r = t->beginRestore(x, tier_spans_, n_main);
    if (!r) {
        sl.phase = Phase::prefill;
        return false;
    }
    r->t0 = t0;
    sl.restore = r;
    sl.tp0_pre = t0;
    // item S: a shared prefix entry is never updated in place by a slot
    sl.rs_shared = x->ck[mt.ck].kind >= 4;
    if (!sl.rs_shared) x->owner = sl.id;
    logI("req {} | slot {} | kv tier: restoring {} tok from {} (checkpoint '{}' @{}, {:.1f} MiB; VRAM prefix was {})", job.id,
         sl.id, x->n_tok, r->from_ssd ? "SSD" : "RAM", ckKindStr(x->ck[mt.ck].kind), mt.reuse,
         static_cast<double>(r->bytes) / 1048576.0, cacheMatch(sl, job.tokens).reuse);
    return true;
}

void Engine::finishTail(Slot& sl, bool wait) {
    tier::Tier* t = tier_;
    if (!t) return;
    tier::Restore* r = sl.restore;
    if (!r) return;
    for (;;) {
        bool done = true;
        try {
            done = t->pumpTail(r);
        } catch (const std::exception&) {
            done = true;
        }
        if (done) break;
        if (!wait) return;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    sl.restore = nullptr;
    tier::Entry* x = t->find(r->entry_id);
    if (opt_.tier_verify && !r->failed && x != nullptr && r->cursor > r->n_main) {
        // (pieces of a checkpoint saveCkpt has overwritten since differ by design)
        std::size_t n_bad = 0, n = 0;
        for (std::size_t i = r->n_main; i < r->cursor; ++i) {
            const tier::Piece& pc = r->pieces[i];
            if (pc.ck == kNoTag || !sl.rs_tail[pc.ck]) continue;
            const std::int32_t j = sl.rs_map[pc.ck];
            if (j >= 0 && sl.ckpts[static_cast<std::size_t>(j)].valid) continue;
            ++n;
            try {
                n_bad += t->verifyPieces(std::span<const tier::Piece>(&pc, 1)).bad;
            } catch (const std::exception&) {
            }
        }
        if (n_bad > 0) logE("slot {} | kv tier verify: restore tail: {} of {} pieces differ", sl.id, n_bad, n);
        else logI("slot {} | kv tier verify: restore tail byte-identical ({} pieces)", sl.id, n);
    }
    if (r->failed || x == nullptr) {
        // copies may still be in flight into the tail buffers: order the main stream after them
        try {
            ops_.eventRecord(r->ev, t->stream);
            ops_.streamWaitEvent(m_.stream(), r->ev);
        } catch (const std::exception&) {
        }
    } else if (!sl.cache_tokens.empty()) {
        for (std::size_t k = 0; k < x->nck; ++k) {
            if (!sl.rs_tail[k]) continue;
            const std::int32_t j = sl.rs_map[k];
            if (j < 0) continue;
            Ckpt& c = sl.ckpts[static_cast<std::size_t>(j)];
            const tier::CkMeta cm = x->ck[k];
            if (c.valid || cm.valid == 0 || cm.pos > sl.reuse || cm.pos > sl.cache_tokens.size()) continue;
            c.valid = true;
            c.pos = cm.pos;
            c.seq = cm.seq;
            c.kind = ckKindStr(cm.kind);
            c.has_logits = cm.has_logits != 0;
        }
    }
    sl.rs_tail.fill(false);
    t->endRestore(r);
}

void Engine::cutTail(Slot& sl) {
    finishTail(sl, false);
    tier::Restore* r = sl.restore;
    if (!r) return;
    tier::Tier* t = tier_;
    try {
        t->pumpRestore(r);
    } catch (const std::exception&) {
    }
    for (std::size_t i = r->cursor; i < r->pieces.size(); ++i)
        if (r->pieces[i].ck != kNoTag) sl.rs_tail[r->pieces[i].ck] = false;
    tier::Event ev = nullptr;
    try {
        ev = t->cutTail(r);
    } catch (const std::exception&) {
        finishTail(sl, true);
        return;
    }
    try {
        ops_.streamWaitEvent(m_.stream(), ev);
    } catch (const std::exception&) {
        finishTail(sl, true);
    }
}

void Engine::pollRestores() {
    tier::Tier* t = tier_;
    if (!t) return;
    for (Slot& sl : slots_) {
        tier::Restore* r = sl.restore;
        if (!r) continue;
        if (sl.phase != Phase::restore) {
            finishTail(sl, false);
            continue;
        }
        bool done = true;
        try {
            done = t->pumpRestore(r);
        } catch (const std::exception&) {
            done = true;
        }
        if (!done) continue;
        Job& job = *sl.job;
        tier::Entry* x = t->find(r->entry_id);
        if (r->failed || x == nullptr) {
            logW("req {} | slot {} | kv tier: restore failed; prefilling from scratch", job.id, sl.id);
            // copies already enqueued may still write the slot's pages / checkpoints
            try {
                ops_.eventRecord(r->ev, t->stream);
                ops_.streamWaitEvent(m_.stream(), r->ev);
            } catch (const std::exception&) {
            }
            sl.restore = nullptr;
            sl.rs_tail.fill(false);
            dropCache(sl);
            t->endRestore(r);
        } else {
            tier::Entry& en = *x;
            sl.cache_tokens.assign(en.tokens.begin(), en.tokens.begin() + en.n_tok);
            for (Ckpt& c : sl.ckpts) c.valid = false;
            std::uint64_t max_seq = 0;
            for (std::size_t k = 0; k < en.nck; ++k) {
                const std::int32_t j = sl.rs_map[k];
                if (j < 0) continue;
                max_seq = std::max(max_seq, en.ck[k].seq);
                if (sl.rs_tail[k]) continue;  // item T3: valid once the tail is in
                Ckpt& c = sl.ckpts[static_cast<std::size_t>(j)];
                const tier::CkMeta cm = en.ck[k];
                c.valid = true;
                c.pos = cm.pos;
                c.seq = cm.seq;
                c.kind = ckKindStr(cm.kind);
                c.has_logits = cm.has_logits != 0;
                max_seq = std::max(max_seq, cm.seq);
            }
            sl.ckpt_seq = std::max(sl.ckpt_seq, max_seq);
            if (sl.rs_shared) {
                sl.tier_id = 0;
                sl.tier_pages = 0;
            } else {
                sl.tier_id = en.id;
                sl.tier_pages = en.n_tok / kv_page;
                en.owner = sl.id;
            }
            const double ms = msSince(r->t0);
            if (r->from_ssd) t->stats.restores_ssd += 1;
            else t->stats.restores_ram += 1;
            t->stats.restore_bytes += r->bytes;
            std::uint64_t main_bytes = 0;
            for (std::size_t i = 0; i < r->n_main; ++i) main_bytes += r->pieces[i].len;
            if (opt_.tier_verify) {
                try {
                    const auto v = t->verifyPieces(std::span<const tier::Piece>(r->pieces.data(), r->n_main));
                    if (v.bad > 0)
                        logE("req {} | slot {} | kv tier verify: restore of {} tok: {} of {} pieces differ from the entry", job.id,
                             sl.id, en.n_tok, v.bad, r->n_main);
                    else
                        logI("req {} | slot {} | kv tier verify: restore of {} tok byte-identical ({:.1f} MiB)", job.id, sl.id,
                             en.n_tok, static_cast<double>(v.bytes) / 1048576.0);
                } catch (const std::exception&) {
                }
            }
            logI("req {} | slot {} | kv tier: restored {} tok from {} in {:.1f} ms ({:.2f} GB/s){}", job.id, sl.id, en.n_tok,
                 r->from_ssd ? "SSD" : "RAM", ms, static_cast<double>(main_bytes) / (ms * 1e6),
                 r->n_main < r->pieces.size() ? "; other checkpoints follow" : "");
            // item S: a restored shared prefix entry becomes a VRAM shared checkpoint again
            if (sl.rs_shared && en.nck >= 1) {
                for (Ckpt& c : sl.ckpts) {
                    if (!c.valid || ckKindCode(c.kind) < 4) continue;
                    try {
                        createSpe(sl, sl.cache_tokens, c.pos, c.kind, std::nullopt, &c, en.id);
                    } catch (const std::exception& ex) {
                        logW("req {} | slot {} | shared prefix: not kept from the restored entry: {}", job.id, sl.id, ex.what());
                    }
                    break;
                }
            }
        }
        sl.phase = Phase::prefill;
        bool ok = true;
        try {
            ok = startJob(sl, job);
        } catch (const std::exception& ex) {
            failSlot(sl, ex.what());
            ok = true;
        }
        if (!ok) {
            sl.job = &job;
            releaseSlot(sl);
        }
    }
}

bool Engine::tierTick() {
    tier::Tier* t = tier_;
    if (!t) return false;
    bool busy = t->tick();
    if (!quarantine_.empty()) {
        reclaimPages(false);
        busy = busy || !quarantine_.empty();
    }
    for (Slot& sl : slots_) {
        if (sl.phase == Phase::idle && sl.spill_wanted) {
            try {
                spillSlot(sl);
            } catch (const std::exception&) {
            }
            // the spill (or its retry) leaves background work: the SSD write after it
            busy = true;
        }
    }
    return busy;
}

void Engine::logTier() {
    tier::Tier* t = tier_;
    if (!t) return;
    const tier::Stats& s = t->stats;
    logI("kv tier | RAM {:.2f} / {:.2f} GiB in {} entries, SSD {:.2f} GiB in {} entries | spills {} ({:.0f} MiB), restores RAM "
         "{} / SSD {} ({:.0f} MiB), SSD writes {} ({:.0f} MiB), RAM evictions {}, dropped {}, spill waits {}, pages copied on "
         "write {}, quarantine waits {}",
         static_cast<double>(t->ramUsedBytes()) / 1073741824.0, static_cast<double>(t->cfg.ram_bytes) / 1073741824.0,
         t->countRam(), static_cast<double>(t->ssd_bytes) / 1073741824.0, t->ssdEntries(), s.spills,
         static_cast<double>(s.spill_bytes) / 1048576.0, s.restores_ram, s.restores_ssd,
         static_cast<double>(s.restore_bytes) / 1048576.0, s.ssd_writes, static_cast<double>(s.ssd_write_bytes) / 1048576.0,
         s.ram_evictions, s.dropped, stat_spill_waits_, stat_cow_pages_, stat_q_waits_);
}

// Shutdown (stop()): finished sessions still waiting for their spill go to the RAM
// tier, and every RAM entry not yet on the SSD tier (new or updated in place) is
// written now instead of after the usual delay, so a graceful stop (Ctrl+C,
// Ctrl+Break, console close) does not lose the last turns. Bounded by
// shutdown_flush_ms; runs on the engine thread like every other tier step.
void Engine::flushTierOnStop() {
    tier::Tier* t = tier_;
    if (!t) return;
    const std::uint64_t saved_delay = t->cfg.ssd_delay_ms;
    t->cfg.ssd_delay_ms = 0;
    const auto t0 = Clock::now();
    const std::uint64_t w0 = t->stats.ssd_writes;
    bool busy = tierTick();
    if (busy) logI("shutdown: writing pending prefix-cache entries to the tiers (up to {} s)", shutdown_flush_ms / 1000);
    while (busy && msSince(t0) < static_cast<double>(shutdown_flush_ms)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        busy = tierTick();
    }
    t->cfg.ssd_delay_ms = saved_delay;
    if (busy)
        logW("shutdown: tier writes still pending after {} s; exiting without them", shutdown_flush_ms / 1000);
    else if (t->stats.ssd_writes != w0 || msSince(t0) > 1.0)
        logI("shutdown: tier writes done in {:.0f} ms ({} SSD writes)", msSince(t0), t->stats.ssd_writes - w0);
    logTier();
}

// ---------------------------------------------------------------------------

void Engine::releaseSlot(Slot& sl) {
    if (sl.restore != nullptr && sl.phase != Phase::restore) finishTail(sl, true);
    Job* job = sl.job;
    if (!job) return;
    visDone(sl, *job);
    {
        std::lock_guard<std::mutex> lk(q_mutex_);
        job->done = true;
        n_active_ -= 1;
        q_cond_.notify_all();
    }
    sl.job = nullptr;
    sl.phase = Phase::idle;
}

void Engine::failSlot(Slot& sl, const std::string& err) {
    Job* job = sl.job;
    if (!job) return;
    if (sl.restore != nullptr && sl.phase != Phase::restore) finishTail(sl, true);
    logE("req {} | slot {} | failed: {} ({})", job->id, sl.id, err, ops_.lastErrorString());
    if (sl.phase == Phase::decode && job->params.stream) {
        sl.st.fail();
    } else {
        const std::string r = errorResponse(500, err);
        if (job->w) {
            job->w->write(r);
            job->w->flush();
        }
    }
    dropCache(sl);
    releaseSlot(sl);
}

void Engine::failJobNoSlot(Job& job) {
    const std::string r = errorResponse(500, "OutOfMemory");
    if (job.w) {
        job.w->write(r);
        job.w->flush();
    }
    std::lock_guard<std::mutex> lk(q_mutex_);
    job.done = true;
    n_active_ -= 1;
    q_cond_.notify_all();
}

void Engine::writeError(Job& job, int status, const std::string& msg) {
    logW("req {} | {} {}", job.id, status, msg);
    const std::string r = errorResponse(status, msg);
    if (job.w) {
        job.w->write(r);
        job.w->flush();
    }
}

void Engine::logBatchStats(bool force) {
    const double el = msSince(stat_t0_);
    if (!force && el < 5000) return;
    if (stat_cycles_ > 0 || stat_prefill_tok_ > 0) {
        std::uint32_t n_dec = 0, n_pf = 0;
        for (const Slot& sl : slots_) {
            if (sl.phase == Phase::decode) ++n_dec;
            else if (sl.phase == Phase::prefill || sl.phase == Phase::restore) ++n_pf;
        }
        const double cyc = static_cast<double>(std::max<std::uint64_t>(stat_cycles_, 1));
        logI("batch | slots busy {}/{} (decode {}, prefill {}) | {} cycles, {:.2f} slots/cycle, {:.2f} rows/cycle, {:.2f} "
             "drafts/cycle | aggregate tg {:.1f} tok/s, prefill {:.0f} tok/s over {:.1f} s",
             n_dec + n_pf, slots_.size(), n_dec, n_pf, stat_cycles_, static_cast<double>(stat_slots_) / cyc,
             static_cast<double>(stat_rows_) / cyc, static_cast<double>(stat_drafts_) / cyc,
             static_cast<double>(stat_tokens_) * 1000.0 / el, static_cast<double>(stat_prefill_tok_) * 1000.0 / el, el / 1000.0);
    }
    stat_t0_ = Clock::now();
    stat_tokens_ = stat_cycles_ = stat_rows_ = stat_slots_ = stat_drafts_ = 0;
    if (stat_seg_batches_ > 0)
        logI("batch | segmented prefill: {} forwards, {} chunks ({:.2f} chunks/forward)", stat_seg_batches_, stat_seg_chunks_,
             static_cast<double>(stat_seg_chunks_) / static_cast<double>(stat_seg_batches_));
    stat_prefill_tok_ = stat_seg_batches_ = stat_seg_chunks_ = 0;
    if (stat_floor_periods_ > 0)
        logI("batch | decode floor: {} prefill forwards while decoding, {} decode cycles held prefill back", stat_floor_periods_,
             stat_floor_waits_);
    stat_floor_periods_ = stat_floor_waits_ = 0;
}

void Engine::submitAndWait(Job& job) {
    std::unique_lock<std::mutex> lk(q_mutex_);
    const std::size_t free = slots_.size() - std::min<std::size_t>(slots_.size(), n_active_);
    const std::size_t waiting = queue_.size();
    queue_.push_back(&job);
    n_building_.fetch_sub(1);
    q_cond_.notify_all();
    lk.unlock();
    if (waiting >= free)
        logI("req {} | queued ({} request(s) waiting ahead, all {} slots busy)", job.id, waiting - free, slots_.size());
    lk.lock();
    q_cond_.wait(lk, [&] { return job.done; });
}

void Engine::abortQueued(std::unique_lock<std::mutex>& lk) {
    // shutting down: queued requests are answered 503
    for (Job* job : queue_) {
        const std::string r = errorResponse(503, "server is shutting down");
        if (job->w) {
            job->w->write(r);
            job->w->flush();
        }
        job->done = true;
    }
    queue_.clear();
    q_cond_.notify_all();
    (void)lk;
}

void Engine::runLoop() {
    stat_t0_ = Clock::now();
    bool was_busy = false;
    for (;;) {
        const TimePoint tl0 = Clock::now();
        // ---- admit
        std::uint32_t n_busy = 0;
        for (const Slot& sl : slots_)
            if (sl.phase != Phase::idle) ++n_busy;
        std::unique_lock<std::mutex> lk(q_mutex_);
        if (stop_ && n_busy == 0) {
            abortQueued(lk);
            lk.unlock();
            flushTierOnStop();
            return;
        }
        // shutting down: no new requests are admitted while the running ones finish
        if (stop_ && !queue_.empty()) abortQueued(lk);
        if (n_busy == 0 && queue_.empty()) {
            if (was_busy) {
                lk.unlock();
                logBatchStats(true);
                if (prof_) prof_->print(m_);
                was_busy = false;
                lk.lock();
            }
            // idle: tier background work is polled; otherwise sleep until a
            // request arrives or the tier IO thread wakes us
            bool logged = false;
            while (queue_.empty()) {
                if (stop_) {
                    abortQueued(lk);
                    lk.unlock();
                    flushTierOnStop();
                    return;
                }
                if (tier_ != nullptr) {
                    lk.unlock();
                    const bool busy = tierTick();
                    if (!busy && !logged) {
                        logTier();
                        logged = true;
                    }
                    if (busy) std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    lk.lock();
                    if (busy) continue;
                }
                // vision: while the encoder is loaded, wake every 100 ms for its idle timeout
                // (a timed wait on the queue condition, not a sleep: a request that arrives
                // meanwhile starts at once; the prototype slept 100 ms here)
                if (visWakeNeeded()) {
                    lk.unlock();
                    visIdle();
                    lk.lock();
                    tier_wake_ = false;
                    if (queue_.empty() && !stop_)
                        q_cond_.wait_for(lk, std::chrono::milliseconds(100), [&] { return !queue_.empty() || tier_wake_ || stop_; });
                    continue;
                }
                tier_wake_ = false;
                q_cond_.wait(lk, [&] { return !queue_.empty() || tier_wake_ || stop_; });
            }
            stat_t0_ = Clock::now();
        }
        std::vector<Job*> admit;
        std::size_t qi = 0;
        while (qi < queue_.size() && n_busy + admit.size() < slots_.size()) {
            // item S: a new session whose system message another request is
            // prefilling right now waits for that checkpoint
            if (waitsForSys(*queue_[qi]) || waitsForTier(*queue_[qi])) {
                ++qi;
                continue;
            }
            admit.push_back(queue_[qi]);
            queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(qi));
            n_active_ += 1;
        }
        // item T4: requests left in the queue (all slots busy, or waiting)
        std::vector<Job*> pf_jobs;
        if (tier_ != nullptr) {
            for (Job* qj : queue_) {
                if (pf_jobs.size() == 8) break;
                if (qj->pf_checked) continue;
                pf_jobs.push_back(qj);
            }
        }
        lk.unlock();
        was_busy = true;
        for (Job* qj : pf_jobs) prefetchFor(*qj);
        std::size_t n_requeued = 0;
        for (Job* job : admit) {
            // item T3: admitted together with a request whose restore just
            // started on the same prefix: back to the queue, wait for it
            if (waitsForTier(*job)) {
                std::lock_guard<std::mutex> g(q_mutex_);
                queue_.insert(queue_.begin() + static_cast<std::ptrdiff_t>(n_requeued), job);
                ++n_requeued;
                n_active_ -= 1;
                continue;
            }
            Slot& sl = *pickSlot(job->tokens);
            const double wait_ms = msSince(job->t_arrive);
            if (wait_ms > 50) logI("req {} | slot {} | starting after {:.0f} ms in queue", job->id, sl.id, wait_ms);
            else logI("req {} | slot {} | assigned", job->id, sl.id);
            if (tryRestore(sl, *job)) continue;
            sl.phase = Phase::prefill;  // reserve before the next pick
            bool ok = true;
            try {
                ok = startJob(sl, *job);
            } catch (const std::exception& ex) {
                failSlot(sl, ex.what());
                ok = true;
            }
            if (!ok) {
                sl.job = job;
                releaseSlot(sl);
            }
        }
        const TimePoint tl1 = Clock::now();
        // ---- host-tier restores in progress / tier housekeeping (<= every 2 ms)
        if (tier_ != nullptr) {
            pollRestores();
            if (msSince(last_tick_) >= 2) {
                tierTick();
                last_tick_ = Clock::now();
            }
        }
        // ---- one prefill chunk (oldest request first), batched with other
        // prefilling requests' chunks when that is bit-identical
        const TimePoint tl2 = Clock::now();
        Slot* pf = nullptr;
        for (Slot& sl : slots_) {
            if (sl.phase != Phase::prefill) continue;
            if (!pf || sl.job->id < pf->job->id) pf = &sl;
        }
        bool idle_wait = false;
        // decode floor: only protects slots that were ALREADY decoding before the
        // current batch of prefill requests arrived (i.e. an interactive stream
        // the user is reading). Requests arriving in the same burst (within
        // gather window) or after this prefill batch are not protected against it,
        // so simultaneous arrivals can merge prefill forwards at full throughput.
        std::vector<DecodeFloor::SlotTok> dec_toks;
        if (floor_.on()) {
            TimePoint pf_latest{};
            TimePoint pf_earliest{};
            bool has_pf = false;
            for (const Slot& sl : slots_) {
                if (sl.phase != Phase::prefill || !sl.job) continue;
                const TimePoint t_arr = sl.job->t_arrive;
                if (!has_pf) {
                    pf_latest = pf_earliest = t_arr;
                    has_pf = true;
                } else {
                    if (t_arr > pf_latest) pf_latest = t_arr;
                    if (t_arr < pf_earliest) pf_earliest = t_arr;
                }
            }
            if (has_pf) {
                const double gather_ms = opt_.gather_ms > 0 ? opt_.gather_ms : 30.0;
                const auto gather_dur =
                    std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(gather_ms));
                for (const Slot& sl : slots_) {
                    if (sl.phase != Phase::decode || !sl.job) continue;
                    const bool separate_arrival = (sl.job->t_arrive + gather_dur < pf_earliest);
                    const bool decode_before_pf = (sl.td0 < pf_latest);
                    if (separate_arrival && decode_before_pf) {
                        dec_toks.push_back({sl.id, sl.reqId(), sl.n_gen});
                    }
                }
            }
            if (dec_toks.empty()) floor_.idle();
        }
        auto floorNow = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - floor_epoch_).count(); };
        std::size_t pf_rows = 0;
        bool floor_pf = false;
        if (pf) {
            // a burst in flight: hold the first chunk briefly so the other
            // requests join its segmented forward
            const bool gather = opt_.seg_prefill && pf->pf_off == 0 && opt_.gather_ms > 0 && n_building_.load() > 0 &&
                                msSince(pf->tp0) < opt_.gather_ms;
            if (gather) {
                idle_wait = true;
            } else if (!dec_toks.empty() && !floor_.prefillAllowed(floorNow(), dec_toks)) {
                // decoding slots are below the floor in this period: a decode cycle alone
                ++stat_floor_waits_;
            } else {
                std::size_t budget = static_cast<std::size_t>(-1);
                if (!dec_toks.empty()) {
                    floor_.startPeriod(floorNow(), dec_toks);
                    const DecodeFloor::Plan pl = floor_.plan(chunkLen(*pf), m_.maxBatch());
                    budget = pl.rows;
                    floor_pf = true;
                    ++stat_floor_periods_;
                    if (msSince(floor_log_t_) >= 10000) {
                        floor_log_t_ = Clock::now();
                        if (pl.cycles > 0)
                            logI("decode floor | {:.0f} tok/s per decoding slot ({} decoding): prefill forward <= {} rows, ~{} decode "
                                 "cycles between forwards{} (cycle {:.1f} ms, prefill {:.0f} rows/s)",
                                 floor_.minTps(), dec_toks.size(), pl.rows, pl.cycles,
                                 pl.reachable ? "" : " (floor out of reach: prefill keeps 20% of the time)", floor_.cycleMs(),
                                 floor_.rowsPerMs() * 1000.0);
                        else
                            logI("decode floor | {:.0f} tok/s per decoding slot ({} decoding): prefill forward <= {} rows "
                                 "(not measured yet)",
                                 floor_.minTps(), dec_toks.size(), pl.rows);
                    }
                }
                pf_rows = prefillGroup(*pf, budget);
                if (prof_) prof_->last_end.reset();
            }
        }
        // ---- one decode cycle over every decoding slot
        std::vector<Slot*> act;
        for (Slot& sl : slots_)
            if (sl.phase == Phase::decode) act.push_back(&sl);
        bool restoring = false;
        for (const Slot& sl : slots_) restoring = restoring || sl.phase == Phase::restore;
        const TimePoint tl3 = Clock::now();
        if (!act.empty()) {
            try {
                decodeCycle(act);
            } catch (const std::exception& ex) {
                for (Slot* sl : act)
                    if (sl->phase == Phase::decode) failSlot(*sl, ex.what());
            }
            if (floor_.on()) {
                // measurements: the cycle synchronizes the stream, so [tl2, now] covers
                // this iteration's prefill forward too
                const TimePoint te = Clock::now();
                const double it_ms = std::chrono::duration<double, std::milli>(te - tl2).count();
                const double cyc_ms = std::chrono::duration<double, std::milli>(te - tl3).count();
                std::vector<DecodeFloor::SlotTok> now_dec;
                for (const Slot& sl : slots_)
                    if (sl.phase == Phase::decode) now_dec.push_back({sl.id, sl.reqId(), sl.n_gen});
                floor_.noteCycle(cyc_ms, pf_rows > 0, now_dec);
                if (floor_pf && pf_rows > 0) floor_.notePrefill(pf_rows, it_ms - std::min(floor_.cycleMs(), 0.5 * it_ms));
            }
        } else if (idle_wait || (restoring && pf == nullptr)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (opt_.loop_log && (!act.empty() || restoring || pf != nullptr)) {
            const TimePoint tl4 = Clock::now();
            auto d = [](TimePoint a, TimePoint b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            logI("loop | admit {:.2f} | tier {:.2f} | pf {:.2f} | dec {:.2f} ms | A {} | restoring {} | prefill {}", d(tl0, tl1),
                 d(tl1, tl2), d(tl2, tl3), d(tl3, tl4), act.size(), restoring, pf != nullptr);
        }
        logBatchStats(false);
        visIdle();
    }
}

// ---------------------------------------------------------------------------
// response writing (stream: SSE chunks as tokens arrive; else one JSON body)

void StreamState::send(const std::string& b) {
    if (gone) return;
    if (!job->w || !job->w->write(b)) fail();
}

void StreamState::begin() {
    const char* prefix = job->kind == JobKind::chat ? "chatcmpl-" : "cmpl-";
    id = e->randId(prefix, 24);
    created = unixNow();
    if (!job->params.stream) return;
    std::string b =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    if (job->kind == JobKind::chat) {
        chunkBegin(b);
        b += "{\"role\":\"assistant\",\"content\":null}";
        chunkEnd(b, nullptr);
    }
    send(b);
    if (!gone && !job->w->flush()) fail();
}

void StreamState::fail() {
    if (!gone) logW("req {} | client connection lost; stopping generation", job->id);
    gone = true;
}

void StreamState::chunkBegin(std::string& b) const {
    b += "data: {\"id\":";
    appendJsonStr(b, id);
    if (job->kind == JobKind::chat) {
        b += std::format(",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":", created);
        appendJsonStr(b, e->opt_.model_name);
        b += ",\"system_fingerprint\":\"whirl\",\"choices\":[{\"index\":0,\"delta\":";
    } else {
        b += std::format(",\"object\":\"text_completion\",\"created\":{},\"model\":", created);
        appendJsonStr(b, e->opt_.model_name);
        b += ",\"system_fingerprint\":\"whirl\",\"choices\":[{\"index\":0,\"text\":";
    }
}

void StreamState::chunkEnd(std::string& b, const char* fin_reason) {
    b += ",\"logprobs\":null,\"finish_reason\":";
    if (fin_reason) appendJsonStr(b, fin_reason);
    else b += "null";
    b += "}]}\n\n";
}

bool StreamState::push(Out& out, std::span<const std::uint32_t> toks) {
    for (std::uint32_t t : toks) out.text += e->tok_.piece(static_cast<TokenId>(t), true);
    out.valid = validUtf8Prefix(out.text);
    const Out::View v = out.view(false);
    if (job->params.stream && !gone) {
        sendDelta(out, v);
        if (!gone && !job->w->flush()) fail();
    }
    return v.stopped;
}

void StreamState::sendDelta(Out& out, const Out::View& v) {
    std::string b;
    if (job->kind == JobKind::chat) {
        if (v.reason.size() > out.sent_reason) {
            chunkBegin(b);
            b += "{\"reasoning_content\":";
            appendJsonStr(b, v.reason.substr(out.sent_reason));
            b += "}";
            chunkEnd(b, nullptr);
            out.sent_reason = v.reason.size();
        }
        if (v.content.size() > out.sent_content) {
            chunkBegin(b);
            b += "{\"content\":";
            appendJsonStr(b, v.content.substr(out.sent_content));
            b += "}";
            chunkEnd(b, nullptr);
            out.sent_content = v.content.size();
        }
    } else if (v.content.size() > out.sent_content) {
        chunkBegin(b);
        appendJsonStr(b, v.content.substr(out.sent_content));
        chunkEnd(b, nullptr);
        out.sent_content = v.content.size();
    }
    if (!b.empty()) send(b);
}

Final StreamState::finish(Out& out, const std::string& reason_in) {
    out.text = sanitizeUtf8(out.text);
    out.valid = out.text.size();
    Out::View v = out.view(true);
    Final f;
    f.reason = reason_in;
    if (v.stopped) f.reason = "stop";
    std::string content(v.content);
    if (v.tool) {
        if (auto calls = parseToolCalls(*v.tool, job->tools)) {
            f.n_tools = calls->size();
            f.calls = std::move(calls);
            f.reason = "tool_calls";
        } else {
            // not a parsable call: keep it as text
            std::string joined(v.content);
            if (!v.content.empty()) joined += "\n\n";
            joined += *v.tool;
            content = std::string(trimRightWs(joined));
        }
    }
    f.content = content;
    f.reasoning = std::string(v.reason);
    f.reason_len = v.reason.size();
    f.content_len = content.size();
    if (job->params.stream && !gone) {
        // remaining text (e.g. an unparsable tool block) then the tool calls
        if (content.size() >= out.sent_content) {
            Out::View v2 = v;
            v2.content = f.content;
            sendDelta(out, v2);
        }
        if (f.calls) {
            std::string b;
            chunkBegin(b);
            b += "{\"tool_calls\":[";
            for (std::size_t i = 0; i < f.calls->size(); ++i) {
                const ToolCall& c = (*f.calls)[i];
                const std::string cid = e->randId("call_", 24);
                if (i > 0) b += ",";
                b += std::format("{{\"index\":{},\"id\":", i);
                appendJsonStr(b, cid);
                b += ",\"type\":\"function\",\"function\":{\"name\":";
                appendJsonStr(b, c.name);
                b += ",\"arguments\":";
                appendJsonStr(b, c.args);
                b += "}}";
            }
            b += "]}";
            chunkEnd(b, nullptr);
            send(b);
        }
    }
    return f;
}

void StreamState::end(Out& out, const Final& f, const Timings& tim, std::uint32_t n_prompt) {
    (void)out;
    if (job->params.stream) {
        if (gone) return;
        // final chunk: empty delta + finish_reason (+ usage / timings as llama.cpp)
        std::string b;
        chunkBegin(b);
        b += job->kind == JobKind::chat ? "{}" : "\"\"";
        b += ",\"logprobs\":null,\"finish_reason\":";
        appendJsonStr(b, f.reason);
        b += "}],\"usage\":";
        writeUsage(b, n_prompt, tim);
        b += ",\"timings\":";
        tim.write(b);
        b += "}\n\n";
        if (job->params.include_usage) {
            b += "data: {\"id\":";
            appendJsonStr(b, id);
            b += std::format(",\"object\":\"{}\",\"created\":{},\"model\":",
                             job->kind == JobKind::chat ? "chat.completion.chunk" : "text_completion", created);
            appendJsonStr(b, e->opt_.model_name);
            b += ",\"choices\":[],\"usage\":";
            writeUsage(b, n_prompt, tim);
            b += "}\n\n";
        }
        b += "data: [DONE]\n\n";
        send(b);
        if (!gone && !job->w->flush()) fail();
        return;
    }
    std::string b = "{\"id\":";
    appendJsonStr(b, id);
    if (job->kind == JobKind::chat) {
        b += std::format(",\"object\":\"chat.completion\",\"created\":{},\"model\":", created);
        appendJsonStr(b, e->opt_.model_name);
        b += ",\"system_fingerprint\":\"whirl\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":";
        if (f.calls && f.content.empty()) b += "null";
        else appendJsonStr(b, f.content);
        if (!f.reasoning.empty()) {
            b += ",\"reasoning_content\":";
            appendJsonStr(b, f.reasoning);
        }
        if (f.calls) {
            b += ",\"tool_calls\":[";
            for (std::size_t i = 0; i < f.calls->size(); ++i) {
                const ToolCall& c = (*f.calls)[i];
                const std::string cid = e->randId("call_", 24);
                if (i > 0) b += ",";
                b += "{\"id\":";
                appendJsonStr(b, cid);
                b += ",\"type\":\"function\",\"function\":{\"name\":";
                appendJsonStr(b, c.name);
                b += ",\"arguments\":";
                appendJsonStr(b, c.args);
                b += "}}";
            }
            b += "]";
        }
        b += "},\"logprobs\":null,\"finish_reason\":";
    } else {
        b += std::format(",\"object\":\"text_completion\",\"created\":{},\"model\":", created);
        appendJsonStr(b, e->opt_.model_name);
        b += ",\"system_fingerprint\":\"whirl\",\"choices\":[{\"index\":0,\"text\":";
        appendJsonStr(b, f.content);
        b += ",\"logprobs\":null,\"finish_reason\":";
    }
    appendJsonStr(b, f.reason);
    b += "}],\"usage\":";
    writeUsage(b, n_prompt, tim);
    b += ",\"timings\":";
    tim.write(b);
    b += "}";
    const std::string r = httpResponse(200, "application/json; charset=utf-8", b);
    if (!job->w || !job->w->write(r) || !job->w->flush()) fail();
}

}  // namespace whirl::server
