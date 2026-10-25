// SPDX-License-Identifier: Apache-2.0
// Default MTP draft-head vocabulary per (GPU arch, dense / MoE): whether the embedded 64k frequency
// subset (data/draft_vocab/subset_64k.bin, draft_vocab_embed.cpp) replaces the full 248,320-row 2-bit
// draft head when WHIRL_DRAFT_VOCAB is unset. Header-only and GPU-free (tests/unit covers the table).
//
// Output: unchanged in every numerics mode. The draft head only proposes tokens; the target verify is
// the exact forward of the active mode, so the subset can only change the acceptance, never the text
// (FAST-8S-1: Ornith precise / balance / fast text identical 64k vs off; P-2: Swift hashes identical).
//
// Cost model (feat027-draft64k97, research\draft_vocab\2026-10-26-draft64k-per-arch.md):
//   saved bytes per draft step   S = (n_vocab - subset_rows) x (E/4 + E/16)      (2-bit codes + f16 scale / 32)
//   saved time per draft step    s = S / BW_gemv(arch)
//   cycle gain (fraction)        g = d x s x R
//     d = MTP draft steps per emitted token in chat, calibrated on the one measured cell
//         (8060S Ornith precise chat: cycle -3.0 % with s = 0.510 ms at R = 92.9 tok/s -> d = 0.633)
//     R = reference chat tok/s of the class with the full draft head (0.2.4 ledger, see k_cells)
//   use the subset when the vocabulary fits and g >= min_gain (1 %, the noise floor of 3 interleaved rounds).
//   Acceptance: a property of (model, subset, prompt), not of the GPU (same gemv_d2 + draft_pick code on both
//   archs): Ornith 8060S tok/s gain ~= cycle gain (acceptance ~0); Swift R9700 zh code tok/cycle 0 %, en code -4..-5 %.
#pragma once

#include "whirl/hip.h"

#include <cstdint>

namespace whirl::draft_vocab {

inline constexpr std::uint32_t embedded_n_vocab = 248320;  // qwen35 tokenizer the embedded subset was built for
inline constexpr std::uint32_t subset_rows = 65536;        // embedded subset size (before special / byte tokens)
inline constexpr double drafts_per_token = 0.633;          // d, calibrated (FAST-8S-1 item 2, 8060S Ornith chat)
inline constexpr double min_gain = 0.01;                   // enable threshold (fraction of a cycle)

enum class Basis { measured, estimated };

// one row per (arch, dense / MoE) class; the reference model of the class is named in `model`
struct Cell {
    hip::Arch arch;
    bool moe;
    const char* model;   // reference product model
    double gemv_gbs;     // 2-bit draft-head GEMV bandwidth on this arch (GB/s)
    double ref_tok_s;    // chat tok/s with the full draft head (reference for R)
    Basis basis;         // is the subset gain on this cell measured or estimated?
    const char* source;  // where the numbers come from
};

// gemv_gbs: 8060S 230 GB/s measured (PROF-8S: gemv_d2_nt1 159 MB in 690 us, 97 %); R9700 600 GB/s estimated
// from the measured int8 decode GEMVs (KGAP-1: 573-626 GB/s), no per-kernel gemv_d2 timing on gfx1201.
// ref_tok_s: 0.2.4 ledger balance decode_coding_chat (Ornith: full head on both cards in 0.2.4; Swift: 0.2.4 ran
// the subset, so the full-head figure is the measured value / 1.03 [E]).
inline constexpr Cell k_cells[] = {
    {hip::Arch::gfx1151, true, "Ornith-1.5-35B-A3B MXFP4", 230.0, 92.9, Basis::measured,
     "FAST-8S-1 item 2: chat +2.5..+3.7 % all modes, cycle -3.0..-3.6 %, text identical"},
    {hip::Arch::gfx1201, true, "Ornith-1.5-35B-A3B MXFP4", 600.0, 246.0, Basis::estimated,
     "cost model: s = 0.195 ms, R = 246 tok/s (0.2.4) -> ~+3.0 %; GPU A/B feat027_draft64k97_gpu.ps1 pending"},
    {hip::Arch::gfx1201, false, "Swift-1.5 27B MXFP4-A", 600.0, 104.0, Basis::measured,
     "P-2 (C-31, R9700 Swift MXFP4-A): zh code +5.0/+3.4/+3.4 % (2k/32k/128k), en code +2.2/+2.3/-1.2 %; DEF-1 default"},
    {hip::Arch::gfx1151, false, "Swift-1.5 27B MXFP4-A", 230.0, 34.0, Basis::estimated,
     "cost model: s = 1.27 ms, R = 34 tok/s -> ~+2.7 %; on since DEF-1 (0.1.3), never A/B'd on the 8060S"},
};

inline constexpr std::uint64_t d2RowBytes(std::uint32_t n_embd) {
    return static_cast<std::uint64_t>(n_embd) / 4 + static_cast<std::uint64_t>(n_embd) / 16;
}
// bytes one draft step no longer reads with the subset (0 when the subset is not smaller)
inline constexpr std::uint64_t savedBytesPerDraft(std::uint32_t n_vocab, std::uint32_t n_embd, std::uint32_t rows = subset_rows) {
    return n_vocab > rows ? static_cast<std::uint64_t>(n_vocab - rows) * d2RowBytes(n_embd) : 0;
}

inline constexpr const Cell* findCell(hip::Arch arch, bool moe) {
    for (const Cell& c : k_cells)
        if (c.arch == arch && c.moe == moe) return &c;
    return nullptr;
}

struct Plan {
    bool use = false;           // apply the embedded subset by default
    bool fits = false;          // vocabulary matches the embedded subset
    double saved_mb = 0;        // MB per draft step (also the VRAM the subset frees)
    double saved_ms = 0;        // per draft step
    double gain = 0;            // estimated cycle-time fraction saved
    const Cell* cell = nullptr; // null: no cell for this (arch, moe)
};

// The default decision. n_embd = the draft head's input width (model.cfg.n_embd).
inline constexpr Plan plan(hip::Arch arch, bool moe, std::uint32_t n_vocab, std::uint32_t n_embd) {
    Plan p;
    p.fits = n_vocab == embedded_n_vocab;
    p.cell = findCell(arch, moe);
    if (!p.fits || p.cell == nullptr || n_embd == 0) return p;
    const double bytes = static_cast<double>(savedBytesPerDraft(n_vocab, n_embd));
    p.saved_mb = bytes / 1e6;
    p.saved_ms = bytes / (p.cell->gemv_gbs * 1e9) * 1e3;
    p.gain = drafts_per_token * (p.saved_ms / 1e3) * p.cell->ref_tok_s;
    p.use = p.gain >= min_gain;
    return p;
}

}  // namespace whirl::draft_vocab
