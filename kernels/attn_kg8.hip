// WHIRL device code (gfx1201 / RDNA 4 only): prefill attention attn_kg with NP 8 (v0252-c11, candidate C11 of 0.2.5.2).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included LAST in the `#if !defined(__gfx1151__)` block of whirl_kernels.hip (after attn_kgv.hip),
// so no existing kernel moves and the gfx1151 code object is unchanged.
//
// GQA 8 (Ornith-1.5-35B-A3B: 16 query heads / 2 KV heads) ran attn_kg4: two blocks per KV head, each reading the
// head's K / V fragments once, i.e. every K / V row twice per query tile. attn_kg8 is the same template, attn_kg_impl
// (kernels/attn_prefill_kp.hip), instantiated with NP = 8: one block = the 8 query heads of one KV head x 16 queries,
// two waves per head (16 waves, 512 threads), the 8 heads reading the same rows at the same time (one K / V pass per
// query tile, as llama.cpp's RDNA ncols2 = 8 packing, fattn.cu:247-251, and FlashInfer's head-group fusion).
// Per-thread code is attn_kg4's: each head's S^T WMMA chain (16 dimension steps), the online softmax of each 32-key
// tile and the P.V order per output tile are the same instructions on the same operands, only the head / wave index
// and the S^T exchange slots (Sx[2][16][32], 32 KB of LDS) differ, so the output is bit-identical to attn_kg4 and
// attn_prefill_wmma (kt attn rows). f16 KV only (no _q8 / _q8v entry): the plain f16 path and Q8P's f16 rows.
// grid (ceil(n_tok / 16), heads / 8), block 512; 8 | n_head / n_kv and 8 | the launch's head range (host: ops).
KG_ENTRY(attn_kg8, 8, 0)
