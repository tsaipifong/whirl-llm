// Model forward, vision rows (arch025-F4, moved from src/model/forward.cpp): per-sequence image maps, the
// multi-section RoPE row positions (visRow / visCommit / visRowsCur) and the image embedding rows (injectRows).
// SPDX-License-Identifier: Apache-2.0

#include "../forward_internal.h"

namespace whirl::qwen35 {

bool Context::visAny() const {
    if (vis_single != nullptr) return true;
    for (const VisMap* v : vis_seq)
        if (v != nullptr) return true;
    return false;
}

const VisMap* Context::visOf(u32 seq) const {
    const VisMap* vm = seqs.empty() ? vis_single : vis_seq[seq];
    return vm != nullptr && vm->active() ? vm : nullptr;
}

void Context::visBegin() {
    mrope_on = false;
    rpos_rows = false;
    rdelta = 0;
    n_inj = 0;
}

void Context::visEnsure() {
    if (rpos_buf != 0) return;
    if (rpos_host.empty()) rpos_host.assign(static_cast<std::size_t>(max_batch) * 3, 0);
    // (not in `allocations`: visRelease frees it again once no sequence has images)
    rpos_own = device::DevBuf::alloc(device::hipBackend(), static_cast<std::uint64_t>(max_batch) * 3 * 4);
    rpos_buf = rpos_own.get();
}

void Context::visRelease() {
    if (rpos_buf == 0 || visAny()) return;
    try {
        hip::sync();
    } catch (...) {
    }
    rpos_own.reset();
    rpos_buf = 0;
}

// row r of the next forward is cache position pos of a sequence with image map vm
void Context::visRow(const VisMap* vm, u32 r, u32 pos) {
    const std::array<i32, 3> rp = vm != nullptr ? vm->rope(pos) : std::array<i32, 3>{static_cast<i32>(pos), static_cast<i32>(pos), static_cast<i32>(pos)};
    std::copy(rp.begin(), rp.end(), rpos_host.begin() + 3 * static_cast<std::size_t>(r));
    if (vm == nullptr) return;
    const VisSpan* sp = vm->spanAt(pos);
    if (sp == nullptr) return;
    const std::size_t E = cfg.n_embd;
    if (sp->emb == nullptr) throw ModelError("ImageNotEncoded");
    const float* src = sp->emb + static_cast<std::size_t>(pos - sp->start) * E;
    if (n_inj > 0) {
        Inj& last = inj[n_inj - 1];
        if (last.row + last.n == r && last.src + last.n * E == src) {
            last.n += 1;
            return;
        }
    }
    if (n_inj == max_inj) throw ModelError("TooManyImageRuns");
    inj[n_inj++] = Inj{r, 1, src};
}

// upload the rows' RoPE positions; mrope_on for the next forward
void Context::visCommit(u32 n) {
    if (k.attn_prep_m == nullptr) throw ModelError("MropeUnsupported");
    if (n <= max_small_batch && k.set_rpos != nullptr) {
        struct Rpos16 {
            i32 v[48];
            static bool launchPtrWord(std::uint32_t) { return false; }  // positions only (launch-log digest)
        } tab{};
        std::copy(rpos_host.begin(), rpos_host.begin() + 3 * static_cast<std::size_t>(n), tab.v);
        hip::launch(k.set_rpos, D(1), D(64), 0, stream, rpos_buf, tab, I(n));
    } else {
        hip::upload(rpos_buf, rpos_host.data(), static_cast<std::size_t>(n) * 3 * 4);
    }
    mrope_on = true;
    rpos_rows = true;
}

// rows 0 .. n-1 = positions pos0 .. of the current sequence
void Context::visRowsCur(u32 n, u32 pos0) {
    visBegin();
    if (!visAny()) return;
    const VisMap* vm = visOf(cur_seq);
    if (vm == nullptr) return;
    visEnsure();
    for (u32 r = 0; r < n; ++r) visRow(vm, r, pos0 + r);
    visCommit(n);
}

// copy the image rows recorded by visRow into x (after the token lookup)
void Context::injectRows() {
    const u64 E = cfg.n_embd;
    // the sources are pageable host memory: such a copy is not reliably ordered after
    // the work already queued on the stream (which may still read x), so drain first
    hip::streamSync(stream);
    for (u32 j = 0; j < n_inj; ++j)
        hip::copyAnyAsync(x + static_cast<u64>(inj[j].row) * E * 4, reinterpret_cast<DevPtr>(inj[j].src), static_cast<u64>(inj[j].n) * E * 4, stream);
    n_inj = 0;
}

}  // namespace whirl::qwen35
