// Checkpoint buffers (arch024 M2); see ckpt_store.h.
// SPDX-License-Identifier: Apache-2.0

#include "cache/ckpt_store.h"

namespace whirl::server {

namespace {
std::uint64_t al(std::uint64_t x) { return (x + 255) / 256 * 256; }
}  // namespace

void CkptStore::init(Ckpt& c, bool host_ok) {
    ownerCheck(owner_, "CkptStore::init");
    const auto& cfg = m_.cfg();
    const std::uint32_t L = cfg.n_layer;
    const std::uint64_t V = cfg.n_vocab;
    c.conv.assign(L, 0);
    c.ssm.assign(L, 0);
    const auto ssm0 = m_.ssmState();
    std::uint64_t n_gdn = 0;
    for (std::uint32_t li = 0; li < L; ++li)
        if (ssm0[li] != 0) ++n_gdn;
    if (host_ok && lay_.ckpt_host) {
        // one pinned host buffer per checkpoint; save / load copy over PCIe
        const std::uint64_t total = n_gdn * (al(lay_.conv_bytes) + al(lay_.ssm_bytes)) + V * 4;
        c.host = device::PinnedBuf::alloc(ops_, total);
        std::uint64_t p = c.host.addr();
        for (std::uint32_t li = 0; li < L; ++li) {
            if (ssm0[li] == 0) continue;
            c.conv[li] = p;
            p += al(lay_.conv_bytes);
            c.ssm[li] = p;
            p += al(lay_.ssm_bytes);
        }
        c.logits = p;
        // the MTP hidden row stays in VRAM (prefill reads it as a device pointer)
        c.hid_buf = device::DevBuf::alloc(ops_, static_cast<std::uint64_t>(cfg.n_embd) * 4);
        c.hid = c.hid_buf.get();
        return;
    }
    // one VRAM allocation per checkpoint, sliced like the pinned layout (256-byte aligned parts);
    // separate allocations per GDN layer (~98 on the 27B models) cost allocation granularity
    const std::uint64_t hid_bytes = al(static_cast<std::uint64_t>(cfg.n_embd) * 4);
    c.dev = device::DevBuf::alloc(ops_, n_gdn * (al(lay_.conv_bytes) + al(lay_.ssm_bytes)) + hid_bytes + V * 4);
    std::uint64_t p = c.dev.get();
    for (std::uint32_t li = 0; li < L; ++li) {
        if (ssm0[li] == 0) continue;
        c.conv[li] = p;
        p += al(lay_.conv_bytes);
        c.ssm[li] = p;
        p += al(lay_.ssm_bytes);
    }
    c.hid = p;
    p += hid_bytes;
    c.logits = p;
}

void CkptStore::save(Ckpt& c, std::optional<DevPtr> hid, std::optional<DevPtr> logits) {
    ownerCheck(owner_, "CkptStore::save");
    m_.commitSeq(m_.curSeq());
    const auto conv = m_.convState();
    const auto ssm = m_.ssmState();
    const auto& cfg = m_.cfg();
    for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
        if (ssm[i] == 0) continue;
        ops_.copyAsync(c.conv[i], conv[i], lay_.conv_bytes, m_.stream());
        ops_.copyAsync(c.ssm[i], ssm[i], lay_.ssm_bytes, m_.stream());
    }
    if (hid) ops_.copyAsync(c.hid, *hid, static_cast<std::uint64_t>(cfg.n_embd) * 4, m_.stream());
    if (logits) ops_.copyAsync(c.logits, *logits, static_cast<std::uint64_t>(cfg.n_vocab) * 4, m_.stream());
    c.has_logits = logits.has_value();
}

void CkptStore::load(const Ckpt& c) {
    ownerCheck(owner_, "CkptStore::load");
    m_.dropPending(m_.curSeq());
    const auto conv = m_.convState();
    const auto ssm = m_.ssmState();
    for (std::uint32_t i = 0; i < m_.cfg().n_layer; ++i) {
        if (ssm[i] == 0) continue;
        ops_.copyAsync(conv[i], c.conv[i], lay_.conv_bytes, m_.stream());
        ops_.copyAsync(ssm[i], c.ssm[i], lay_.ssm_bytes, m_.stream());
    }
}

std::uint64_t CkptStore::bytes(bool slot_ckpt) const {
    const auto& cfg = m_.cfg();
    const auto ssm0 = m_.ssmState();
    std::uint64_t n_gdn = 0;
    for (std::uint32_t li = 0; li < cfg.n_layer; ++li)
        if (ssm0[li] != 0) ++n_gdn;
    const std::uint64_t hid = al(static_cast<std::uint64_t>(cfg.n_embd) * 4);
    if (slot_ckpt && lay_.ckpt_host) return hid;  // WHIRL_CKPT_HOST: only the hidden row is in VRAM
    return n_gdn * (al(lay_.conv_bytes) + al(lay_.ssm_bytes)) + hid + static_cast<std::uint64_t>(cfg.n_vocab) * 4;
}

void CkptStore::wait(std::initializer_list<device::Event> wait_for) {
    for (const device::Event e : wait_for) device::waitIfPending(ops_, e);
}

void CkptStore::free(Ckpt& c) {
    c.host.reset();  // pinned checkpoint: host buffer, then the hidden row (freeCkpt's order)
    c.hid_buf.reset();
    c.dev.reset();   // conv / ssm / hid / logits are slices of it
    c.conv.clear();
    c.ssm.clear();
    c.hid = c.logits = 0;
}

void CkptStore::park(std::span<Ckpt> cks, std::initializer_list<device::Event> wait_for) {
    ownerCheck(owner_, "CkptStore::park");
    wait(wait_for);
    for (Ckpt& c : cks) free(c);
}

void CkptStore::release(Ckpt& c, std::initializer_list<device::Event> wait_for) {
    ownerCheck(owner_, "CkptStore::release");
    wait(wait_for);
    free(c);
}

void CkptStore::unpark(std::span<Ckpt> cks, bool host_ok) {
    ownerCheck(owner_, "CkptStore::unpark");
    for (Ckpt& c : cks) init(c, host_ok);
}

}  // namespace whirl::server
