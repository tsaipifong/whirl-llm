// WHIRL device code (both kernel sets): the by-value kernel argument structs, defined once (arch025 N13).
// SPDX-License-Identifier: Apache-2.0
// The host twins are in include/whirl/kernels_abi.h; both sides static_assert their layout against the one table in
// include/whirl/kernels_abi_layout.h (tools/gates/kernel_abi_check.py). Included by whirl_kernels.hip before every
// fragment, and by each fragment that uses a struct (declared dependency).
#pragma once
#include "../include/whirl/kernels_abi_layout.h"

// Paged KV cache arguments of one attention layer (see kernels/attn.hip, kernels/gfx1151/attn.hip).
struct KvArgs {
    void *k;
    void *v;
    _Float16 *ks;
    _Float16 *vs;
    const int *ptab;
    const int *kvbase;
    int tab0;
    int win;  // attn_wsplit*: MTP draft window, W / 64 | (threshold / 1024) << 16, 0 = off; others ignore it
};

// Grouped one-token GEMV: one launch of a gemvq_<type> covers up to 3 matrices that share the input: blocks
// [0, b1) run segment 0, [b1, b2) segment 1, the rest segment 2 (see gv_sel in gemv_q.hip of each set).
struct GvArgs {
    const uint8_t *W0, *W1, *W2;
    float *y0, *y1, *y2;
    uint64_t rb0, rb1, rb2;
    int n0, n1, n2;
    int b1, b2;
};

// Token ids + positions for up to 16 rows without a host copy (a blocking
// hipMemcpy would drain the queue): rows [0, n_host) from the by-value table,
// the rest from dev_src (earlier on-device argmax results).
struct Tok16 {
    int t[16];
};

// Batched decode rows over several sequences (kMaxVerifyRows = 32 rows): token (>= 0 from the table,
// < 0: dev_src[-tok - 1], e.g. a sequence's draft slot), position and KV region base per row, all from kernel args.
struct RowTab {
    int tok[32];
    int pos[32];
    int base[32];
};

struct RowIdx {
    int v[32];
};

// attn_wsplit query groups (32): queries first[z] .. first[z] + count[z] - 1.
struct AwGroups {
    int first[32];
    int count[32];
};

// draft_sample_rows per-row parameters (speculative sampling of MTP drafts, kernels/spec_sample.hip)
struct DSampArgs {
    int on[32];
    int top_k[32];
    int qoff[32];  // i32 word offset of this row's q entry in qbuf
    float inv_t[32];
    float top_p[32];
    float min_p[32];
    unsigned key_lo[32];  // race key base (whirl::spec::raceBase)
    unsigned key_hi[32];
};

// Recurrent-state segments for the fused DeltaNet decode kernels: rows
// [row0, row0 + nrows) of the batch belong to one sequence whose state lives
// at `state`; snap[t] (if non-null) receives the state right after local row t
// (speculative verify rollback). Passed by value (kernel args, no host copy).
#define GDN_MAX_SEG 16
#define GDN_MAX_SNAP 15
struct GdnSeg {
    float *state;
    float *snap[GDN_MAX_SNAP];
    int row0, nrows;
    // replay mode (pend != null): the inputs of this segment's rows are kept in
    // pend (conv: raw input rows [row][ch]; delta rule: [row][head][k 128 | v 128
    // | decay | beta]); the first npend rows kept by the previous verify are
    // applied to the state first and the state is written once (the new base),
    // then the new rows run without writing state or snapshots
    float *pend;
    int npend, pad_;
};
struct GdnSegs {
    GdnSeg s[GDN_MAX_SEG];
};

// ---- layout against include/whirl/kernels_abi_layout.h (the host checks the same table) ----
namespace whirl_kernel_args_check {
namespace L = whirl::kernels::abi;
#define WHIRL_KA_OFF(T, f) __builtin_offsetof(T, f)
static_assert(GDN_MAX_SEG == L::kGdnMaxSeg && GDN_MAX_SNAP == L::kGdnMaxSnap, "GDN segment constants");
static_assert(sizeof(KvArgs) == L::kKvArgsSize, "KvArgs size");
static_assert(WHIRL_KA_OFF(KvArgs, k) == L::kKvArgsK && WHIRL_KA_OFF(KvArgs, v) == L::kKvArgsV &&
                  WHIRL_KA_OFF(KvArgs, ks) == L::kKvArgsKs && WHIRL_KA_OFF(KvArgs, vs) == L::kKvArgsVs &&
                  WHIRL_KA_OFF(KvArgs, ptab) == L::kKvArgsPtab && WHIRL_KA_OFF(KvArgs, kvbase) == L::kKvArgsKvbase &&
                  WHIRL_KA_OFF(KvArgs, tab0) == L::kKvArgsTab0 && WHIRL_KA_OFF(KvArgs, win) == L::kKvArgsWin,
              "KvArgs fields");
static_assert(sizeof(GvArgs) == L::kGvArgsSize, "GvArgs size");
static_assert(WHIRL_KA_OFF(GvArgs, W0) == L::kGvArgsW && WHIRL_KA_OFF(GvArgs, W2) == L::kGvArgsW + 16 &&
                  WHIRL_KA_OFF(GvArgs, y0) == L::kGvArgsY && WHIRL_KA_OFF(GvArgs, y2) == L::kGvArgsY + 16 &&
                  WHIRL_KA_OFF(GvArgs, rb0) == L::kGvArgsRb && WHIRL_KA_OFF(GvArgs, rb2) == L::kGvArgsRb + 16 &&
                  WHIRL_KA_OFF(GvArgs, n0) == L::kGvArgsN && WHIRL_KA_OFF(GvArgs, n2) == L::kGvArgsN + 8 &&
                  WHIRL_KA_OFF(GvArgs, b1) == L::kGvArgsB1 && WHIRL_KA_OFF(GvArgs, b2) == L::kGvArgsB2 &&
                  WHIRL_KA_OFF(GvArgs, b2) + sizeof(int) == L::kGvArgsTail,
              "GvArgs fields");
static_assert(sizeof(Tok16) == L::kTok16Size, "Tok16 size");
static_assert(sizeof(RowTab) == L::kRowTabSize && WHIRL_KA_OFF(RowTab, tok) == L::kRowTabTok &&
                  WHIRL_KA_OFF(RowTab, pos) == L::kRowTabPos && WHIRL_KA_OFF(RowTab, base) == L::kRowTabBase,
              "RowTab layout");
static_assert(sizeof(RowIdx) == L::kRowIdxSize, "RowIdx size");
static_assert(sizeof(AwGroups) == L::kAwGroupsSize && WHIRL_KA_OFF(AwGroups, first) == L::kAwGroupsFirst &&
                  WHIRL_KA_OFF(AwGroups, count) == L::kAwGroupsCount,
              "AwGroups layout");
static_assert(sizeof(DSampArgs) == L::kDSampArgsSize && WHIRL_KA_OFF(DSampArgs, on) == L::kDSampOn &&
                  WHIRL_KA_OFF(DSampArgs, top_k) == L::kDSampTopK && WHIRL_KA_OFF(DSampArgs, qoff) == L::kDSampQoff &&
                  WHIRL_KA_OFF(DSampArgs, inv_t) == L::kDSampInvT && WHIRL_KA_OFF(DSampArgs, top_p) == L::kDSampTopP &&
                  WHIRL_KA_OFF(DSampArgs, min_p) == L::kDSampMinP && WHIRL_KA_OFF(DSampArgs, key_lo) == L::kDSampKeyLo &&
                  WHIRL_KA_OFF(DSampArgs, key_hi) == L::kDSampKeyHi,
              "DSampArgs layout");
static_assert(sizeof(GdnSeg) == L::kGdnSegSize && WHIRL_KA_OFF(GdnSeg, state) == L::kGdnSegState &&
                  WHIRL_KA_OFF(GdnSeg, snap) == L::kGdnSegSnap && WHIRL_KA_OFF(GdnSeg, row0) == L::kGdnSegRow0 &&
                  WHIRL_KA_OFF(GdnSeg, nrows) == L::kGdnSegNrows && WHIRL_KA_OFF(GdnSeg, pend) == L::kGdnSegPend &&
                  WHIRL_KA_OFF(GdnSeg, npend) == L::kGdnSegNpend && WHIRL_KA_OFF(GdnSeg, pad_) == L::kGdnSegPad,
              "GdnSeg layout");
static_assert(sizeof(GdnSegs) == L::kGdnSegsSize, "GdnSegs size");
#undef WHIRL_KA_OFF
}  // namespace whirl_kernel_args_check
