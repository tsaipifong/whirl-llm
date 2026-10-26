// SPDX-License-Identifier: Apache-2.0
// Default MTP draft-head vocabulary per (GPU arch, dense / MoE): whether the embedded 64k frequency
// subset (data/draft_vocab/subset_64k.bin, draft_vocab_embed.cpp) replaces the full 248,320-row 2-bit
// draft head when WHIRL_DRAFT_VOCAB is unset. Header-only and GPU-free (tests/unit covers the table).
//
// Output: the draft head only proposes tokens. Where acceptance is exact - precise, balance (greedy exact
// match; temperature > 0: specsample, whose text for a seed does not depend on drafts) and fast with
// WHIRL_RELAX=0 - the subset changes only the acceptance rate, never the text (FAST-8S-1: Ornith 64k vs off
// identical in precise / balance / fast-without-relaxacc; P-2: Swift hashes identical). With relaxed acceptance
// (relaxacc: fast on dense models by default, MoE with WHIRL_RELAX=1) a draft is kept when it is close to the
// target, so which token the draft head proposes decides what is kept: fast text can differ between the subset
// and the full head. fast is judged by task accuracy (>= 95 % of precise), not by identical text.
//
// Decision (feat027-draft64k97-r1, research\draft_vocab\2026-10-26-draft64k-per-arch.md, PRIOR_ART
// progress\feat027-draft64k97-r1.PRIOR_ART.md):
//   saved bytes per draft step   S = (n_vocab - subset_rows) x (E/4 + E/16)      (2-bit codes + f16 scale / 32)
//   saved time per draft step    s = S / BW_gemv(arch)
//   cycle-time fraction saved    g = d x s x R      d = MTP draft steps per emitted token (per model class)
//                                                   R = chat tok/s with the full head on this cell
//   net tok/s gain               net = (1 - a) / (1 - g) - 1    a = tok/cycle loss of the subset (per model class)
//   use the subset iff
//     measured cell : the measured primary-workload (zh coding chat) tok/s gain >= min_gain, the measured worst
//                     workload >= -max_loss, and the model net >= min_gain (scales the cell to other n_embd)
//     shipped cell  : default since an earlier release, never A/B'd on this card: kept while the model's
//                     lower bound net - err >= min_gain (err = the class's held-out prediction error)
//     estimated cell: never by default (no measurement on this card yet); the model value is the expectation
//                     the GPU A/B (progress\tools\feat027_draft64k97_gpu.ps1) has to confirm.
//   d / a / err are fitted on one set of cells and err is checked on cells not used for the fit (tests/unit).
#pragma once

#include "whirl/hip.h"

#include <cstdint>

namespace whirl::draft_vocab {

inline constexpr std::uint32_t embedded_n_vocab = 248320;  // qwen35 tokenizer the embedded subset was built for
inline constexpr std::uint32_t subset_rows = 65536;        // embedded subset size (before special / byte tokens)
inline constexpr double min_gain = 0.01;                   // enable threshold: 1 %, the round spread of 3 interleaved rounds
inline constexpr double max_loss = 0.02;                   // a measured workload may lose at most 2 % tok/s

// per model class: draft steps per token, acceptance loss of the 64k subset, held-out prediction error
struct ModelClass {
    bool moe;
    const char* model;   // reference product model
    double d;            // MTP draft steps per emitted token in chat (fitted)
    double a_primary;    // tok/cycle loss with the subset, primary workload (zh question + English code)
    double a_worst;      // tok/cycle loss with the subset, worst measured workload
    double err;          // max |predicted - measured| tok/s gain on cells not used for the fit
    const char* source;
};

inline constexpr ModelClass k_classes[] = {
    // dense: d fitted on P-2 zh code 2k / 32k / 128k (R9700, 64k vs off, mean of 0.995 / 0.801 / 1.255);
    // a: zh code tok/cycle 0 / 0 / 0 %, en code -4.0 / -5.4 / -5.4 %; err: 15 held-out P-2 cells (zh text, en code,
    // the 48k subset; max 0.032 on en code 32k, the model under-predicts there)
    {false, "Swift-1.5 27B MXFP4-A", 1.017, 0.0, 0.054, 0.035,
     "P-2 (C-31, R9700 Swift MXFP4-A, 2 rounds; 128k en / zh text 1 round)"},
    // MoE: d fitted on FAST-8S-1 item 2 precise chat (8060S Ornith, cycle -3.0 % at R = 92.9 tok/s); a: same chat
    // tok/s +2.5 % with cycle -3.0 % -> tok/cycle -0.6 %, the only same-prompt pair (qa runs used other prompts);
    // err: cycle on held-out qa 2k / 8k / 32k / 128k (max 0.0065)
    {true, "Ornith-1.5-35B-A3B MXFP4", 0.633, 0.006, 0.006, 0.008,
     "FAST-8S-1 item 2 (8060S Ornith, 3 rounds)"},
};

enum class Basis {
    measured,   // A/B of the subset on this card
    shipped,    // default since an earlier release, not A/B'd on this card
    estimated,  // model only
};

// one row per (arch, dense / MoE) class
struct Cell {
    hip::Arch arch;
    bool moe;
    double gemv_gbs;        // 2-bit draft-head GEMV bandwidth on this arch (GB/s)
    double ref_tok_s;       // chat tok/s with the full draft head (R)
    Basis basis;
    double meas_primary;    // measured: primary-workload tok/s gain of the subset (fraction); else 0
    double meas_worst;      // measured: worst workload tok/s gain; else 0
    const char* source;
};

// gemv_gbs: 8060S 230 GB/s measured (PROF-8S: gemv_d2_nt1 159 MB in 690 us, 97 %); R9700 600 GB/s estimated
// from the measured int8 decode GEMVs (KGAP-1: 573-626 GB/s), no per-kernel gemv_d2 timing on gfx1201.
// ref_tok_s: FAST-8S-1 precise chat for the 8060S Ornith fit cell; else the 0.2.4 ledger balance decode_coding_chat
// (Ornith R9700 246; Swift ran the subset in 0.2.4, so the full-head figure is 107.7 / 1.03 and 35 / 1.03 [E]).
inline constexpr Cell k_cells[] = {
    {hip::Arch::gfx1151, true, 230.0, 92.9, Basis::measured, 0.025, -0.003,
     "FAST-8S-1 item 2: chat +2.5 / +3.0 / +3.7 % (precise / balance / fast), worst qa 128k precise -0.3 % "
     "(other prompt, cycle -1.6 %), text identical"},
    {hip::Arch::gfx1201, true, 600.0, 246.0, Basis::estimated, 0.0, 0.0,
     "model only: s = 0.195 ms, R = 246 tok/s -> net ~+2.4 %; off until feat027_draft64k97_gpu.ps1 measures it"},
    {hip::Arch::gfx1201, false, 600.0, 104.0, Basis::measured, 0.034, -0.012,
     "P-2: zh code +5.0 / +3.4 / +3.4 % (2k / 32k / 128k), en code +2.2 / +2.3 / -1.2 %; DEF-1 default since 0.1.3"},
    {hip::Arch::gfx1151, false, 230.0, 34.0, Basis::shipped, 0.0, 0.0,
     "DEF-1 default since 0.1.3, never A/B'd on the 8060S: model s = 1.27 ms, R = 34 tok/s -> net ~+4.6 %"},
};

inline constexpr std::uint64_t d2RowBytes(std::uint32_t n_embd) {
    return static_cast<std::uint64_t>(n_embd) / 4 + static_cast<std::uint64_t>(n_embd) / 16;
}
// bytes one draft step no longer reads with the subset (0 when the subset is not smaller)
inline constexpr std::uint64_t savedBytesPerDraft(std::uint32_t n_vocab, std::uint32_t n_embd, std::uint32_t rows = subset_rows) {
    return n_vocab > rows ? static_cast<std::uint64_t>(n_vocab - rows) * d2RowBytes(n_embd) : 0;
}
// cycle-time fraction saved: g = d x s x R (s in seconds)
inline constexpr double cycleGain(double d, double saved_s, double ref_tok_s) { return d * saved_s * ref_tok_s; }
// tok/s gain when the cycle is a fraction g shorter and tok/cycle a fraction a lower
inline constexpr double netGain(double g, double a) { return g < 1.0 ? (1.0 - a) / (1.0 - g) - 1.0 : 0.0; }

inline constexpr const Cell* findCell(hip::Arch arch, bool moe) {
    for (const Cell& c : k_cells)
        if (c.arch == arch && c.moe == moe) return &c;
    return nullptr;
}
inline constexpr const ModelClass* findClass(bool moe) {
    for (const ModelClass& m : k_classes)
        if (m.moe == moe) return &m;
    return nullptr;
}

enum class Why {
    no_fit,           // vocabulary is not the embedded subset's
    no_cell,          // no row for (arch, moe)
    model_low,        // model net (or its lower bound) below min_gain
    measured_low,     // measured primary gain below min_gain
    measured_loss,    // a measured workload loses more than max_loss
    not_measured,     // estimated cell: off until the GPU A/B
    use_measured,
    use_shipped,
};

struct Plan {
    bool use = false;           // apply the embedded subset by default
    bool fits = false;          // vocabulary matches the embedded subset
    Why why = Why::no_fit;
    double saved_mb = 0;        // MB per draft step (also the VRAM the subset frees)
    double saved_ms = 0;        // per draft step
    double gain = 0;            // model: cycle-time fraction saved (g)
    double net = 0;             // model: primary-workload tok/s gain
    double net_worst = 0;       // model: worst-workload tok/s gain
    double net_lo = 0;          // model: net - err
    const Cell* cell = nullptr; // null: no cell for this (arch, moe)
};

// The decision for one (cell, class); exposed so tests can feed synthetic cells and classes.
inline constexpr Plan evaluate(const Cell& c, const ModelClass& m, std::uint32_t n_vocab, std::uint32_t n_embd) {
    Plan p;
    p.fits = n_vocab == embedded_n_vocab;
    p.cell = &c;
    if (!p.fits) return p;
    const double bytes = static_cast<double>(savedBytesPerDraft(n_vocab, n_embd));
    p.saved_mb = bytes / 1e6;
    p.saved_ms = bytes / (c.gemv_gbs * 1e9) * 1e3;
    p.gain = cycleGain(m.d, p.saved_ms / 1e3, c.ref_tok_s);
    p.net = netGain(p.gain, m.a_primary);
    p.net_worst = netGain(p.gain, m.a_worst);
    p.net_lo = p.net - m.err;
    switch (c.basis) {
        case Basis::measured:
            if (c.meas_primary < min_gain) p.why = Why::measured_low;
            else if (c.meas_worst < -max_loss) p.why = Why::measured_loss;
            else if (p.net < min_gain) p.why = Why::model_low;
            else p.why = Why::use_measured;
            break;
        case Basis::shipped:
            p.why = p.net_lo >= min_gain ? Why::use_shipped : Why::model_low;
            break;
        case Basis::estimated:
            p.why = Why::not_measured;
            break;
    }
    p.use = p.why == Why::use_measured || p.why == Why::use_shipped;
    return p;
}

// The default decision. n_embd = the draft head's input width (model.cfg.n_embd).
inline constexpr Plan plan(hip::Arch arch, bool moe, std::uint32_t n_vocab, std::uint32_t n_embd) {
    const Cell* c = findCell(arch, moe);
    const ModelClass* m = findClass(moe);
    if (c == nullptr || m == nullptr || n_embd == 0) {
        Plan p;
        p.fits = n_vocab == embedded_n_vocab;
        p.why = p.fits ? Why::no_cell : Why::no_fit;
        p.cell = c;
        return p;
    }
    return evaluate(*c, *m, n_vocab, n_embd);
}

}  // namespace whirl::draft_vocab
