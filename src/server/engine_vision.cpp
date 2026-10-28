// Server engine, image input: embedding cache by content hash, per-slot image
// maps for the model, encoder idle release; base64 for data: URLs.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from the research prototype (server.zig, item VIS).

#include "engine.h"
#include "engine_vision.h"

#include "log.h"
#include "whirl/common.h"  // writeFile (was reached through model.h)

#include <algorithm>

namespace whirl::server {

void Engine::attachVision(const VisionConfig& vc) {
    ps_.setVision({vc.vis != nullptr, vc.vision_end_id, vc.ck_min});
    vis_.attach(vc);
}

void ImageEmbeddings::attach(const VisionConfig& vc) {
    cfg_ = vc;
    lend_.clear();
    if (vc.vis == nullptr) return;
    const std::size_t n = m_.lendScratch(lend_raw_);
    lend_ = vision::lendRegions(lend_raw_, n);
    snap0_ = ops_.localMemSnap();
}

// The embeddings of image p (cache hit, or encode now on the model stream,
// borrowing the prefill scratch); pinned until unpin.
EmbEntry* ImageEmbeddings::get(const vision::Prepared& p, std::uint64_t req, std::uint32_t slot) {
    seq_ += 1;
    for (const auto& ent : cache_) {
        if (ent->hash == p.hash) {
            ent->pins += 1;
            ent->last = seq_;
            n_hit_ += 1;
            logI("req {} | slot {} | image {} tok ({}x{}): embeddings cached", req, slot, p.nTokens(), p.nx, p.ny);
            return ent.get();
        }
    }
    ServerVision& v = *cfg_.vis;
    auto ent = std::make_unique<EmbEntry>();
    ent->data.resize(static_cast<std::size_t>(p.nTokens()) * v.projDim());
    const vision::EncodeStats st = v.encode(p, lend_, ent->data.data(), ent->data.size());
    m_.scratchClobbered();
    // diagnostics: WHIRL_VIS_DUMP=prefix writes each encoded image's embeddings (prefix.<req>.f32)
    if (!cfg_.dump_prefix.empty()) {
        try {
            writeFile(cfg_.dump_prefix + "." + std::to_string(req) + ".f32",
                      std::string_view(reinterpret_cast<const char*>(ent->data.data()), ent->data.size() * 4));
        } catch (...) {
        }
    }
    ent->hash = p.hash;
    ent->last = seq_;
    ent->pins = 1;
    bytes_ += ent->data.size() * 4;
    n_enc_ += 1;
    logI("req {} | slot {} | image {} tok ({}x{}, {} patches): encoded in {:.1f} ms ({}{}{})", req, slot, p.nTokens(), p.nx, p.ny,
         st.n_patches, st.ms_total, st.resident ? "resident weights" : "weights streamed", st.ms_upload > 0 ? ", first upload" : "",
         st.borrowed_extra > 0 ? ", extra VRAM" : "");
    EmbEntry* raw = ent.get();
    cache_.push_back(std::move(ent));
    evict();
    return raw;
}

void ImageEmbeddings::unpin(EmbEntry* ent) {
    if (ent->pins > 0) ent->pins -= 1;
    evict();
}

// drop unpinned entries, least recently used first, down to the cap
void ImageEmbeddings::evict() {
    while (bytes_ > cfg_.cache_bytes) {
        std::optional<std::size_t> best;
        for (std::size_t k = 0; k < cache_.size(); ++k) {
            if (cache_[k]->pins > 0) continue;
            if (!best || cache_[k]->last < cache_[*best]->last) best = k;
        }
        if (!best) return;
        bytes_ -= cache_[*best]->data.size() * 4;
        // (swap-remove, as the prototype's ArrayList.swapRemove)
        std::swap(cache_[*best], cache_.back());
        cache_.pop_back();
    }
}

// Embeddings for the spans that the prefill will run (from `from` on).
void ImageEmbeddings::prepare(std::uint32_t slot, JobVis* jv, std::uint64_t req, std::uint32_t from) {
    if (jv == nullptr) {
        m_.setVisMap(slot, nullptr);
        return;
    }
    for (std::size_t k = 0; k < jv->spans.size(); ++k) {
        qwen35::VisSpan& sp = jv->spans[k];
        if (sp.start + sp.n <= from) {
            sp.emb = nullptr;
            continue;
        }
        if (jv->ents[k] == nullptr) jv->ents[k] = get(jv->preps[k], req, slot);
        sp.emb = jv->ents[k]->data.data();
    }
    m_.setVisMap(slot, &jv->vmap);
}

// release the job's embedding pins (the job ends)
void ImageEmbeddings::done(std::uint32_t slot, JobVis* jv) {
    if (cfg_.vis == nullptr) return;
    m_.setVisMap(slot, nullptr);
    if (jv == nullptr) return;
    for (EmbEntry*& ent : jv->ents) {
        if (ent != nullptr) {
            unpin(ent);
            ent = nullptr;
        }
    }
}

bool ImageEmbeddings::wakeNeeded() const {
    return cfg_.vis != nullptr && (cfg_.vis->loaded() || m_.visRowsAllocated());
}

// Vision idle: unload the encoder (kernels, resident weights, copy stream) after
// its idle timeout, and the RoPE row buffer once no slot needs it.
void ImageEmbeddings::idle() {
    ServerVision* v = cfg_.vis;
    if (v == nullptr) return;
    if (v->idleTick()) rel_pending_ = true;
    if (!v->loaded()) m_.visRelease();
    if (rel_pending_ && !v->loaded() && !m_.visRowsAllocated()) {
        rel_pending_ = false;
        // delta: hipMemGetInfo free, this process's view on both device classes (it returns exactly on
        // hipFree; on an integrated GPU it also drops by 2x any pinned host memory allocated since,
        // which the GPU Shared Usage growth shows). Dedicated / Shared Usage: whirl/device_mem.h.
        const hip::LocalMemSnap now = ops_.localMemSnap();
        const hip::LocalMemTaken t = hip::localMemTaken(snap0_, now);
        logI("vision: idle for {} s; encoder released; VRAM free {:.1f} MiB (before the first image {:.1f} MiB, delta {:.1f} MiB) "
             "[since the first image: GPU Dedicated Usage {:+.1f} MiB, Shared Usage {:+.1f} MiB{}]",
             v->idleS(), static_cast<double>(now.hip_free) / 1048576.0, static_cast<double>(snap0_.hip_free) / 1048576.0, t.hip_free_mib,
             (static_cast<double>(now.dedicated) - static_cast<double>(snap0_.dedicated)) / 1048576.0, t.shared_mib,
             now.pdh_ok && snap0_.pdh_ok ? "" : ", GPU process counters unreadable");
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
