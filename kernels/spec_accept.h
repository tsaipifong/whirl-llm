// WHIRL feat026-acc: accept decision of a speculative verify and the next cycle's rows, shared by the
// device kernel spec_accept_rows (kernels/spec_accept.hip) and the host (engine reference + unit tests).
// SPDX-License-Identifier: Apache-2.0
//
// One verify of a sequence ran nd + 1 rows: the token `next` and drafts d[0..nd). The trunk's greedy
// token of row r is ctl[ROWS + r]. Exact (greedy) acceptance keeps the longest prefix of drafts that
// equal the trunk's tokens; the token after them is the trunk's token of the first row not kept. Then
//  - an accepted end-of-turn draft ends the reply: only the rows before it stay, it becomes `chosen`;
//  - max_tokens: at most `budget` tokens are emitted (acc + 1 <= budget), the last one is `chosen`.
// This is the host loop of Engine::decodeCycle (engine.cpp, base 1a5a547) for greedy slots without
// relaxacc, moved here unchanged so the device can make the same decision.
//
// The next cycle (same sequence) then feeds the MTP block the pend rows [d[0..acc), chosen] at
// positions pos + 1 .. pos + acc + 1, and verifies `chosen` at position pos + acc + 1. spec_next_rows
// builds those rows (the RowTab the model's set_rows would get) so phase 2 can enqueue the next
// cycle without the host (see progress\feat026-acc.HANDOFF.md).
#pragma once

#include <stdint.h>

#if defined(__HIP__) || defined(__HIPCC__)
#define WHIRL_SA_HD __host__ __device__
#else
#define WHIRL_SA_HD
#endif

// control-area words of a sequence (kernels_abi.h kCtl*; host static_asserts check them)
#define WHIRL_SA_CTL_ROWS 0      // [0, 16) trunk argmax per verify row
#define WHIRL_SA_CTL_DRAFTS 16   // [16, 32) MTP drafts
#define WHIRL_SA_CTL_ND 48       // MTP drafts kept by the device p-min cutoff
// results of spec_accept_rows (words 50 .. 63 were unused)
#define WHIRL_SA_CTL_ACC 50      // accepted drafts (after the EOS / budget cuts)
#define WHIRL_SA_CTL_CHOSEN 51   // token after them
#define WHIRL_SA_CTL_NDDEV 52    // drafts compared (MTP: min(nd, kept by p-min); literal: nd)
#define WHIRL_SA_CTL_FLAGS 53    // WHIRL_SA_F_*
#define WHIRL_SA_CTL_NPOS 54     // next cycle's verify position (pos + acc + 1)
#define WHIRL_SA_CTL_NROW 55     // last kept verify row (row0 + acc): its hidden feeds the MTP head
#define WHIRL_SA_CTL_ACC0 56     // accepted drafts before the EOS / budget cuts (what the acceptance models see)
#define WHIRL_SA_CTL_WORDS 64

#define WHIRL_SA_F_DEV 1         // decided on the device (greedy exact)
#define WHIRL_SA_F_EOS 2         // cut at an accepted end-of-turn draft
#define WHIRL_SA_F_BUDGET 4      // cut by max_tokens

#define WHIRL_SA_MAX_SEG 16      // sequences of one verify (gdn_max_seg)
#define WHIRL_SA_MAX_ND 15       // drafts of one sequence (ctl_rows holds nd + 1 <= 16 rows)
#define WHIRL_SA_MAX_ROWS 32     // RowTab rows (kMaxVerifyRows)

// kernel argument (by value, about 1.7 KB): one entry per sequence of the verify, in verify order
struct SpecAccArgs {
    int n;                 // sequences
    int n_vocab;           // lookupId: t < n_vocab ? t : 0
    unsigned stop0, stop1; // end-of-turn tokens (0xffffffff: none)
    int seq[WHIRL_SA_MAX_SEG];     // control area = ctl_all + seq * 64
    int nd[WHIRL_SA_MAX_SEG];      // drafts verified (nd_k)
    int lit[WHIRL_SA_MAX_SEG];     // 1: literal drafts (n-gram / DFlash2) in lit_d; 0: MTP drafts in the control area
    int dev[WHIRL_SA_MAX_SEG];     // 1: greedy exact (decide); 0: the host decides (sampling, relaxacc): copy only
    int eos[WHIRL_SA_MAX_SEG];     // 1: cut at end-of-turn drafts (request without ignore_eos)
    int budget[WHIRL_SA_MAX_SEG];  // max_tokens - n_gen (0: no cut, as the host)
    int pos[WHIRL_SA_MAX_SEG];     // verify position of `next`
    int row0[WHIRL_SA_MAX_SEG];    // first verify row of the sequence
    int kvbase[WHIRL_SA_MAX_SEG];  // KV region base (RowTab.base)
    int lit_d[WHIRL_SA_MAX_SEG][WHIRL_SA_MAX_ND];
};

struct SpecAccResult {
    int acc0;  // before the cuts
    int acc;
    int chosen;
    int nd_dev;
    int flags;
};

// the next cycle's rows, as Model::mtpBatchStepEx (step 0) and verifyBatchEnqueue (row 0) build them
struct SpecNextTab {
    int n_mtp;                           // MTP step-0 rows of all sequences (sum of acc + 1)
    int mtp_tok[WHIRL_SA_MAX_ROWS];      // lookupId(pend[j])
    int mtp_pos[WHIRL_SA_MAX_ROWS];      // pos + 1 + j
    int mtp_base[WHIRL_SA_MAX_ROWS];
    int mtp_last[WHIRL_SA_MAX_SEG];      // last MTP row of each sequence (RowIdx `last`)
    int ver_tok[WHIRL_SA_MAX_SEG];       // verify row 0: lookupId(chosen)
    int ver_pos[WHIRL_SA_MAX_SEG];       // pos + acc + 1
    int ver_base[WHIRL_SA_MAX_SEG];
};

// draft r of sequence k (MTP: control area; literal: argument table)
WHIRL_SA_HD inline unsigned spec_draft(const SpecAccArgs& a, int k, const int* ctl, int r) {
    return a.lit[k] ? (unsigned)a.lit_d[k][r] : (unsigned)ctl[WHIRL_SA_CTL_DRAFTS + r];
}

// greedy exact accept of sequence k (ctl: its 64 control words after the verify argmax)
WHIRL_SA_HD inline SpecAccResult spec_accept_one(const SpecAccArgs& a, int k, const int* ctl) {
    SpecAccResult o;
    const int nd_k = a.nd[k];
    int nd_dev = nd_k;
    if (!a.lit[k]) {
        const int kept = ctl[WHIRL_SA_CTL_ND] > 0 ? ctl[WHIRL_SA_CTL_ND] : 0;
        nd_dev = nd_k > 0 ? (nd_k < kept ? nd_k : kept) : 0;
    }
    int acc = 0;
    while (acc < nd_dev && (unsigned)ctl[WHIRL_SA_CTL_ROWS + acc] == spec_draft(a, k, ctl, acc)) ++acc;
    o.acc0 = acc;
    unsigned chosen = (unsigned)ctl[WHIRL_SA_CTL_ROWS + acc];
    int flags = WHIRL_SA_F_DEV;
    if (a.eos[k]) {
        for (int r = 0; r < acc; ++r) {
            const unsigned d = spec_draft(a, k, ctl, r);
            if (d == a.stop0 || d == a.stop1) {
                chosen = d;
                acc = r;
                flags |= WHIRL_SA_F_EOS;
                break;
            }
        }
    }
    const int budget = a.budget[k];
    if (budget >= 1 && acc + 1 > budget) {
        chosen = spec_draft(a, k, ctl, budget - 1);
        acc = budget - 1;
        flags |= WHIRL_SA_F_BUDGET;
    }
    o.acc = acc;
    o.chosen = (int)chosen;
    o.nd_dev = nd_dev;
    o.flags = flags;
    return o;
}

WHIRL_SA_HD inline int spec_lookup(const SpecAccArgs& a, unsigned t) { return t < (unsigned)a.n_vocab ? (int)t : 0; }

// next-cycle rows from the per-sequence results (res[k] valid for every k < a.n)
WHIRL_SA_HD inline void spec_next_rows(const SpecAccArgs& a, const int* ctl_all, const SpecAccResult* res, SpecNextTab* t) {
    int n = 0;
    for (int k = 0; k < a.n; ++k) {
        const int* ctl = ctl_all + a.seq[k] * WHIRL_SA_CTL_WORDS;
        const int acc = res[k].acc;
        for (int j = 0; j <= acc && n < WHIRL_SA_MAX_ROWS; ++j) {
            const unsigned tok = j < acc ? spec_draft(a, k, ctl, j) : (unsigned)res[k].chosen;
            t->mtp_tok[n] = spec_lookup(a, tok);
            t->mtp_pos[n] = a.pos[k] + 1 + j;
            t->mtp_base[n] = a.kvbase[k];
            ++n;
        }
        t->mtp_last[k] = n - 1;
        t->ver_tok[k] = spec_lookup(a, (unsigned)res[k].chosen);
        t->ver_pos[k] = a.pos[k] + acc + 1;
        t->ver_base[k] = a.kvbase[k];
    }
    t->n_mtp = n;
}

// result words of sequence k into its control area
WHIRL_SA_HD inline void spec_store(const SpecAccArgs& a, int k, const SpecAccResult& r, int* ctl) {
    ctl[WHIRL_SA_CTL_ACC] = r.acc;
    ctl[WHIRL_SA_CTL_CHOSEN] = r.chosen;
    ctl[WHIRL_SA_CTL_NDDEV] = r.nd_dev;
    ctl[WHIRL_SA_CTL_FLAGS] = r.flags;
    ctl[WHIRL_SA_CTL_NPOS] = a.pos[k] + r.acc + 1;
    ctl[WHIRL_SA_CTL_NROW] = a.row0[k] + r.acc;
    ctl[WHIRL_SA_CTL_ACC0] = r.acc0;
}
