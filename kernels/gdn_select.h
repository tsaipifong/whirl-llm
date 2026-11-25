// WHIRL v0252-e1s8 (PLAN V1-93 P2a): the DeltaNet state SELECT mode of the E1 one-cycle lookahead in snapshot mode (the
// gfx1151 / Radeon 8060S verify keeps snapshot sets, it has no replay). Shared by the gfx1151 fused DeltaNet kernels
// (kernels/gfx1151/fused_decode.hip), the model (src/model/qwen35: the segment table and the host bookkeeping) and the
// server tests' mock model, so the device rule and the host rule are one definition.
// SPDX-License-Identifier: Apache-2.0
//
// A snapshot-mode verify of nd drafts reads its input state, writes snapshot r = the state after local row r for r < nd,
// and writes the final state (after row nd). After `acc` accepted drafts (after the end-of-turn cut) the sequence's state
// is the state after row acc: snapshot acc if acc < nd, else the final state. Ordinarily the host swaps that snapshot in.
// With the lookahead the next verify is submitted before the host knows acc, so that verify SELECTS its input on the
// device: its candidates are the outputs of the verify before it, in the order P[0 .. m) = snapshot sets 0 .. m - 1,
// P[m] = the live state (its final output), m = that verify's nd. The device reads c = acc from the sequence's NPEND word
// (acc + 1, written by spec_accept_rows), takes P[c] as input, writes its own snapshot r into P[gdn_sel_out(r, c)] (the
// candidates it did not take, in order) and its final state into the SPARE set. Nothing it writes is P[c]: the state after
// the previous cycle survives the in-flight cycle, so a request that ends there keeps its gen-end checkpoint state
// (where vLLM's fused_recurrent_gated_delta_rule writes in place; progress/v0252-e1s8.PRIOR_ART.md section 3).
// Once the host knows c it renames the buffers (gdn_sel_settle): snapshot set r <- P[gdn_sel_out(r, c)], live <- spare,
// spare <- P[c]; the next select verify finds its candidates in the same places again. Requires nd <= m.
#pragma once

#if defined(__HIP__) || defined(__HIPCC__)
#define WHIRL_GS_HD __host__ __device__
#else
#define WHIRL_GS_HD
#endif

// the candidate the device takes: c = NPEND word - 1, clamped to [0, m]
WHIRL_GS_HD inline int gdn_sel_pick(int npend_word, int m) {
    const int c = npend_word - 1;
    return c < 0 ? 0 : (c > m ? m : c);
}

// the candidate that receives the selecting verify's snapshot r (r < nd <= m): the r-th candidate other than c
WHIRL_GS_HD inline int gdn_sel_out(int r, int c) { return r + (r >= c ? 1 : 0); }

#if !defined(__HIP_DEVICE_COMPILE__)
// host: rename the per-layer buffer tables once c is known. sets[0 .. m) = the snapshot set tables, live / spare the live
// and spare tables (any swappable table type: std::vector<DevPtr>, std::span views are not - pass the owning objects)
template <class T>
void gdn_sel_settle(T* sets, int m, T& live, T& spare, int c) {
    // P = [sets[0 .. m), live]; new sets[r] = P[r < c ? r : r + 1], new live = spare, new spare = P[c]
    if (c < m) {
        T taken = static_cast<T&&>(sets[c]);
        for (int r = c; r + 1 < m; ++r) sets[r] = static_cast<T&&>(sets[r + 1]);
        sets[m - 1] = static_cast<T&&>(live);
        live = static_cast<T&&>(spare);
        spare = static_cast<T&&>(taken);
    } else {
        // c == m: the live state was taken; the snapshot sets stay
        T taken = static_cast<T&&>(live);
        live = static_cast<T&&>(spare);
        spare = static_cast<T&&>(taken);
    }
}
#endif
