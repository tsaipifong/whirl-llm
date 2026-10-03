// Server engine, image input: embedding cache by content hash, per-slot image
// maps for the model, encoder idle release; base64 for data: URLs.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from the research prototype (server.zig, item VIS).

#include "engine.h"

#include "log.h"

#include <algorithm>

namespace whirl::server {

void Engine::attachVision(const VisionConfig& vc) {
    vis_cfg_ = vc;
    vis_lend_.clear();
    if (vc.vis == nullptr) return;
    const std::size_t n = m_.lendScratch(vis_lend_raw_);
    for (std::size_t k = 0; k < n; ++k) vis_lend_.push_back({vis_lend_raw_[k][0], vis_lend_raw_[k][1]});
    vis_free0_ = vc.vis->vramFree();
}

// The embeddings of image p (cache hit, or encode now on the model stream,
// borrowing the prefill scratch); pinned until embUnpin.
EmbEntry* Engine::embGet(const vision::Prepared& p, std::uint64_t req, std::uint32_t slot) {
    emb_seq_ += 1;
    for (const auto& ent : emb_cache_) {
        if (ent->hash == p.hash) {
            ent->pins += 1;
            ent->last = emb_seq_;
            stat_vis_hit_ += 1;
            logI("req {} | slot {} | image {} tok ({}x{}): embeddings cached", req, slot, p.nTokens(), p.nx, p.ny);
            return ent.get();
        }
    }
    ServerVision& v = *vis_cfg_.vis;
    auto ent = std::make_unique<EmbEntry>();
    ent->data.resize(static_cast<std::size_t>(p.nTokens()) * v.projDim());
    const vision::EncodeStats st = v.encode(p, vis_lend_, ent->data.data(), ent->data.size());
    m_.scratchClobbered();
    // diagnostics: WHIRL_VIS_DUMP=prefix writes each encoded image's embeddings (prefix.<req>.f32)
    if (!vis_cfg_.dump_prefix.empty()) {
        try {
            writeFile(vis_cfg_.dump_prefix + "." + std::to_string(req) + ".f32",
                      std::string_view(reinterpret_cast<const char*>(ent->data.data()), ent->data.size() * 4));
        } catch (...) {
        }
    }
    ent->hash = p.hash;
    ent->last = emb_seq_;
    ent->pins = 1;
    emb_bytes_ += ent->data.size() * 4;
    stat_vis_enc_ += 1;
    logI("req {} | slot {} | image {} tok ({}x{}, {} patches): encoded in {:.1f} ms ({}{}{})", req, slot, p.nTokens(), p.nx, p.ny,
         st.n_patches, st.ms_total, st.resident ? "resident weights" : "weights streamed", st.ms_upload > 0 ? ", first upload" : "",
         st.borrowed_extra > 0 ? ", extra VRAM" : "");
    EmbEntry* raw = ent.get();
    emb_cache_.push_back(std::move(ent));
    embEvict();
    return raw;
}

void Engine::embUnpin(EmbEntry* ent) {
    if (ent->pins > 0) ent->pins -= 1;
    embEvict();
}

// drop unpinned entries, least recently used first, down to the cap
void Engine::embEvict() {
    while (emb_bytes_ > vis_cfg_.cache_bytes) {
        std::optional<std::size_t> best;
        for (std::size_t k = 0; k < emb_cache_.size(); ++k) {
            if (emb_cache_[k]->pins > 0) continue;
            if (!best || emb_cache_[k]->last < emb_cache_[*best]->last) best = k;
        }
        if (!best) return;
        emb_bytes_ -= emb_cache_[*best]->data.size() * 4;
        // (swap-remove, as the prototype's ArrayList.swapRemove)
        std::swap(emb_cache_[*best], emb_cache_.back());
        emb_cache_.pop_back();
    }
}

// Embeddings for the spans that the prefill will run (from `from` on).
void Engine::visPrepare(Slot& sl, Job& job, std::uint32_t from) {
    JobVis* jv = job.vis.get();
    if (jv == nullptr) {
        m_.setVisMap(sl.id, nullptr);
        return;
    }
    for (std::size_t k = 0; k < jv->spans.size(); ++k) {
        qwen35::VisSpan& sp = jv->spans[k];
        if (sp.start + sp.n <= from) {
            sp.emb = nullptr;
            continue;
        }
        if (jv->ents[k] == nullptr) jv->ents[k] = embGet(jv->preps[k], job.id, sl.id);
        sp.emb = jv->ents[k]->data.data();
    }
    m_.setVisMap(sl.id, &jv->vmap);
}

// release the job's embedding pins (the job ends)
void Engine::visDone(Slot& sl, Job& job) {
    if (vis_cfg_.vis == nullptr) return;
    m_.setVisMap(sl.id, nullptr);
    JobVis* jv = job.vis.get();
    if (jv == nullptr) return;
    for (EmbEntry*& ent : jv->ents) {
        if (ent != nullptr) {
            embUnpin(ent);
            ent = nullptr;
        }
    }
}

bool Engine::visWakeNeeded() const {
    return vis_cfg_.vis != nullptr && (vis_cfg_.vis->loaded() || m_.visRowsAllocated());
}

// Vision idle: unload the encoder (kernels, resident weights, copy stream) after
// its idle timeout, and the RoPE row buffer once no slot needs it.
void Engine::visIdle() {
    ServerVision* v = vis_cfg_.vis;
    if (v == nullptr) return;
    if (v->idleTick()) vis_rel_pending_ = true;
    if (!v->loaded()) m_.visRelease();
    if (vis_rel_pending_ && !v->loaded() && !m_.visRowsAllocated()) {
        vis_rel_pending_ = false;
        const double fr = static_cast<double>(v->vramFree());
        logI("vision: idle for {} s; encoder released; VRAM free {:.1f} MiB (before the first image {:.1f} MiB, delta {:.1f} MiB)",
             v->idleS(), fr / 1048576.0, static_cast<double>(vis_free0_) / 1048576.0, (static_cast<double>(vis_free0_) - fr) / 1048576.0);
    }
}

// Base64, standard alphabet; padded input must be a multiple of 4 long, unpadded
// input must not contain '='; unused trailing bits must be zero (as Zig's
// std.base64 standard / standard_no_pad decoders the prototype used).
bool base64Decode(std::string_view in, std::vector<std::uint8_t>& out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    out.clear();
    std::size_t n = in.size();
    if (n % 4 == 0) {
        // padded: up to two '=' at the end
        std::size_t pad = 0;
        while (pad < 2 && n > 0 && in[n - 1] == '=') {
            --n;
            ++pad;
        }
    }
    if (n % 4 == 1) return false;
    out.reserve(n / 4 * 3 + 2);
    std::uint32_t acc = 0;
    int bits = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int v = val(in[i]);
        if (v < 0) return false;
        acc = (acc << 6) | static_cast<std::uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>(acc >> bits));
            acc &= (1u << bits) - 1;
        }
    }
    if (acc != 0) return false;  // non-zero leftover bits
    return true;
}

}  // namespace whirl::server
